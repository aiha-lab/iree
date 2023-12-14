// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PIM/KernelConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
/// Lowers an hal.executable.variant operation to scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code
/// - then convert to NVVM/ROCDL dialect.
/// This should be merged with the equivalent pass in LinalgToLLVM. Fo
/// simplicity it is currently a separate pass.
class PIMLowerExecutableTargetPass
    : public PIMLowerExecutableTargetBase<
          PIMLowerExecutableTargetPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<IREE::Codegen::IREECodegenDialect,
                IREE::HAL::HALDialect,
                IREE::LinalgExt::IREELinalgExtDialect,
                linalg::LinalgDialect,
                linalg::transform::LinalgTransformDialect,
                gpu::GPUDialect,
                nvgpu::NVGPUDialect,
                pdl::PDLDialect,
                pdl_interp::PDLInterpDialect,
                scf::SCFDialect,
                tensor::TensorDialect,
                transform::TransformDialect,
                vector::VectorDialect>();
    // clang-format on
  }

  PIMLowerExecutableTargetPass() = default;
  PIMLowerExecutableTargetPass(
      const PIMLowerExecutableTargetPass &pass){};

  void runOnOperation() override;

 private:
  Option<bool> testLoweringConfiguration{
      *this, "test-lowering-configuration",
      llvm::cl::desc(
          "Flag used for lit-testing the default configuration set for root "
          "ops in hal.executable.variants. Defaults to false and is set to "
          "true "
          "for lit tests. Not for general usage"),
      llvm::cl::init(false)};
};
}  // namespace

/// Verify that valid configuration is set for all ops within the compiled
/// module.
template <typename F>
static LogicalResult verifyLoweringConfiguration(
    ModuleOp module, IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize, F verificationFn) {
  auto walkResult = module.walk([&](Operation *op) -> WalkResult {
    IREE::Codegen::LoweringConfigAttr loweringConfig = getLoweringConfig(op);
    if (!loweringConfig) return WalkResult::advance();
    return verificationFn(op, loweringConfig, translationInfo, workgroupSize);
  });
  return failure(walkResult.wasInterrupted());
}

static LogicalResult verifyEntryPoint(
    ModuleOp moduleOp, IREE::Codegen::TranslationInfoAttr translationInfo,
    IREE::HAL::ExecutableExportOp exportOp) {
  Optional<mlir::ArrayAttr> workgroupSizeAttr = exportOp.getWorkgroupSize();

  if (workgroupSizeAttr.has_value()) {
    std::array<int64_t, 3> workgroupSizes;
    for (auto [index, attr] : llvm::enumerate(workgroupSizeAttr.value())) {
      workgroupSizes[index] = attr.cast<IntegerAttr>().getInt();
    }

    switch (translationInfo.getDispatchLoweringPassPipeline()) {
      case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt:
        return verifyLoweringConfiguration(moduleOp, translationInfo,
                                           workgroupSizes,
                                           verifyGPUMatmulSimtPassPipeline);
        break;
      case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulTensorCore:
        return verifyLoweringConfiguration(moduleOp, translationInfo,
                                           workgroupSizes,
                                           verifyGPUMatmulTensorCorePipeline);
        break;
      default:;
    }
  }
  return success();
}

void PIMLowerExecutableTargetPass::runOnOperation() {
  // DBG
  llvm::outs() << "PIMLowerExecutableTargetPass::runOnOperation\n";
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();
  // DBG
  moduleOp.print(llvm::outs());
  llvm::outs() << "\n";
  
  /*
  std::size_t found;
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    std::string input = funcOp.getName().str();
    found = input.find("generic");

  }
  if (found == std::string::npos)*/
  
  OpPassManager executableLoweringPipeline(
      IREE::HAL::ExecutableVariantOp::getOperationName());

  
  if (failed(initPIMLaunchConfig(moduleOp))) {
    return signalPassFailure();
  }
  // DBG
  // moduleOp.print(llvm::outs());
  // llvm::outs() << "\n";

  // There might be multiple entry points in the module. Currently, all of
  // them need to have the same pipeline.
  // TODO(ravishankarm): This is strange that this is not enforced
  // structurally, but something to address later on. For now this restriction
  // is fine.
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  Optional<IREE::Codegen::TranslationInfoAttr> translationInfo;
  for (auto &it : exportOps) {
    auto exportOp = it.second;
    if (IREE::Codegen::TranslationInfoAttr currTranslationInfo =
            getTranslationInfo(exportOp)) {
      if (translationInfo) {
        if (currTranslationInfo != translationInfo.value()) {
          moduleOp.emitOpError(
              "unhandled compilation of entry point functions with different "
              "translation info");
        }
      } else {
        translationInfo = currTranslationInfo;
      }

      // Verify the properties of each entry point based on the target
      // pipeline.
      if (failed(verifyEntryPoint(moduleOp, currTranslationInfo, exportOp))) {
        return signalPassFailure();
      }
    }
  }

  if (!testLoweringConfiguration && translationInfo.has_value()) {
    switch (translationInfo.value().getDispatchLoweringPassPipeline()) {
      case IREE::Codegen::DispatchLoweringPassPipeline::PIMMatmul:
        addPIMMatmulPassPipeline(executableLoweringPipeline);
        break;
      default:
        variantOp.emitOpError("Unsupported pipeline on GPU target.");
        return signalPassFailure();
    }
  }

  if (failed(runPipeline(executableLoweringPipeline, variantOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createPIMLowerExecutableTargetPass() {
  return std::make_unique<PIMLowerExecutableTargetPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
