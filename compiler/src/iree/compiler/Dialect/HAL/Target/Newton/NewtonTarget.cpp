// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/Newton/NewtonTarget.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/CUDA/LLVMPasses.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "iree/schemas/newton_executable_def_builder.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {


class NewtonTargetBackend final : public TargetBackend {
 public:
  NewtonTargetBackend() = default;

  std::string name() const override { return "newton"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
                    IREE::Codegen::IREECodegenDialect>();
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    // Indicates that the runtime HAL driver operates only in the legacy
    // synchronous mode.
    configItems.emplace_back(b.getStringAttr("legacy_sync"), b.getUnitAttr());

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }
  
  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    // For now we disable translation if the variant has external object files.
    // We could instead perform linking with those objects (if they're bitcode
    // ala libdevice.bc, etc).
    if (variantOp.isExternal()) return;
    return;

    // buildLLVMGPUTransformPassPipeline(passManager, false);
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {

    if (!variantOp.getObjects().has_value()) {
      return variantOp.emitOpError()
             << "no objects defined for external variant";
    } else if (variantOp.getObjects()->getValue().size() != 1) {
      // For now we assume there will be exactly one object file.
      // In the future we will want to perform a linking step here and ideally
      // support _also_ linking in the codegen results.
      return variantOp.emitOpError() << "only one object reference is "
                                        "supported for external variants";
    }

    // Get entryPointNames 
    SmallVector<StringRef, 8> entryPointNames;
    for (auto exportOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
      entryPointNames.emplace_back(exportOp.getSymName());
    }

    std::string isrCmd; // Binary code for Newton ISR command stream
   	
    isrCmd = "FFFF"; // TODO: Replace to real ISR CMD

    FlatbufferBuilder builder;
    iree_NewtonExecutableDef_start_as_root(builder);

    auto isrCmdRef = flatbuffers_uint32_vec_create(
        builder, reinterpret_cast<const uint32_t *>(isrCmd.data()),
        isrCmd.size() / sizeof(uint32_t));

    auto entryPointsRef = builder.createStringVec(entryPointNames);

    iree_NewtonExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_NewtonExecutableDef_isr_cmd_add(builder, isrCmdRef);
    iree_NewtonExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();

  }

 private:
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    targetAttrs.push_back(getExecutableTarget(context));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    // Add some configurations to the `hal.executable.target` attribute.
    /*
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(StringAttr::get(context, name), value);
    };
    */
    // Set target arch
    // addConfig("target_arch", StringAttr::get(context, clTargetChip));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("newton"), b.getStringAttr("newton-isr-fb"),
        configAttr);
  }
};

/*
void registerNewtonTargetBackends() {
  // #hal.device.target<"newton", ...
  // #hal.executable.target<"newton", ...
  static TargetBackendRegistration registration("newton", [=]() {
    return std::make_shared<NewtonTargetBackend>();
  });
}*/

/*
void registerNewtonTargetBackends() {
  auto backendFactory = [=]() {
    return std::make_shared<NewtonTargetBackend>();
  };
  // #hal.device.target<"newton", ...
  static TargetBackendRegistration registration("newton", backendFactory);
}*/

void registerNewtonTargetBackends() {
  // #hal.device.target<"newton", ...
  // #hal.executable.target<"newton", ...
  static TargetBackendRegistration registration(
      "newton", [=]() { return std::make_shared<NewtonTargetBackend>(); });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
