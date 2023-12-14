// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <random>
#include <sstream>
#include "iree/compiler/Dialect/HAL/Target/PIM/PIMTarget.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/CUDA/LLVMPasses.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/PIM/IR/PIMDialect.h"
#include "iree/compiler/Dialect/PIM/IR/PIMOps.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "iree/schemas/pim_executable_def_builder.h"
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "json/json.h"


namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Convert PIM operations to INT64 code
// LayerNorm1: 1, QKVGen: 2, QKMatmul: 3, Softmax: 4,
// SVMatmul: 5, OutProj: 6, LayerNorm2: 7, FFN1: 8, FFN2: 9
void GenOpCommand(mlir::Operation *op, std::vector<uint64_t> &code){
  if (isa<IREE::PIM::LayerNorm1Op>(op)) {
    uint64_t cmd = 1;
    code.push_back(cmd);
  }
  else if (isa<IREE::PIM::QKVGenOp>(op)) {
    uint64_t cmd = 2;
    code.push_back(cmd);
  }
  else if (isa<IREE::PIM::QKMatmulOp>(op)) {
    uint64_t cmd = 3;
    code.push_back(cmd);
  }
  else if (isa<IREE::PIM::SoftmaxOp>(op)) {
    uint64_t cmd = 4;
    code.push_back(cmd);
  }
  else if (isa<IREE::PIM::SVMatmulOp>(op)) {
    uint64_t cmd = 5;
    code.push_back(cmd);
  }
  else if (isa<IREE::PIM::OutProjOp>(op)) {
    uint64_t cmd = 6;
    code.push_back(cmd);
  }
  else if (isa<IREE::PIM::LayerNorm2Op>(op)) {
    uint64_t cmd = 7;
    code.push_back(cmd);
  }
  else if (isa<IREE::PIM::FFN1Op>(op)) {
    uint64_t cmd = 8;
    code.push_back(cmd);
  }
  else if (isa<IREE::PIM::FFN2Op>(op)) {
    uint64_t cmd = 9;
    code.push_back(cmd);
  }
}

class PIMTargetBackend final : public TargetBackend {
 public:
  PIMTargetBackend() = default;

  std::string name() const override { return "pim"; }

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
    // std::cout << "TranslationPassPipeline\n";
    
    buildPIMCodegenPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {

    /*	  
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
    */
    
    // AiMDMACmdCreator cmd_creator;
    std::vector<uint64_t> code;
    std::vector<uint64_t> out_dim;
    // std::vector<uint64_t> comm_flag;

    auto innerModule = variantOp.getInnerModule();
    // for (auto &op : innerModule.getBody()->without_terminator()) {
    for (auto &op : *innerModule.getBody()) {
      if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
        // std::cout << "Get access to func op" << std::endl;
        funcOp.walk([/*&cmd_creator,*/ &code, &out_dim/*, &comm_flag*/](mlir::Operation *op) {
          /*
          if (isa<IREE::PIM::AllReduceOp>(op)) {
            comm_flag.push_back(0);
            comm_flag.push_back(1);
          }
          else if (isa<IREE::PIM::AllGatherOp>(op)) {
            comm_flag.push_back(1);
            comm_flag.push_back(1);
          }*/
          GenOpCommand(op, /*cmd_creator,*/ code);
        });
      }
    }
    /*
    if (comm_flag.size() == 0) {
      comm_flag.push_back(2);
      comm_flag.push_back(2);
    }
    llvm::outs() << "PIMTarget.cpp: comm_flag size: " << comm_flag.size() << "\n";
    for (uint64_t i: comm_flag)
      llvm::outs() << "PIMTarget.cpp: comm_flag data: " << i << "\n";
    */
    llvm::outs() <<"PIMTarget.cpp: Generated from op command stream (bit) : \n";
    for (uint64_t i: code)
      std::cout << std::bitset<64>(i) << '\n';
    llvm::outs() << "\n";
    llvm::outs() << "PIMTarget.cpp: code size: " << code.size() << "\n";

    // Get entryPointNames
    SmallVector<StringRef, 8> entryPointNames;
    for (auto exportOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
      entryPointNames.emplace_back(exportOp.getSymName());
    }
    
    FlatbufferBuilder builder;
    iree_PIMExecutableDef_start_as_root(builder);

    auto codeRef = flatbuffers_uint64_vec_create(
        builder, code.data(),
        code.size());

    /*
    auto commFlagRef = flatbuffers_uint64_vec_create(
        builder, comm_flag.data(),
        comm_flag.size());*/

    auto entryPointsRef = builder.createStringVec(entryPointNames);

    iree_PIMExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_PIMExecutableDef_code_add(builder, codeRef);
    // iree_PIMExecutableDef_comm_flag_add(builder, commFlagRef);
    iree_PIMExecutableDef_end_as_root(builder);

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
        context, b.getStringAttr("pim"), b.getStringAttr("pim-isr-fb"),
        configAttr);
  }
};

void registerPIMTargetBackends() {
  // #hal.device.target<"pim", ...
  // #hal.executable.target<"pim", ...
  static TargetBackendRegistration registration(
      "pim", [=]() { return std::make_shared<PIMTargetBackend>(); });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
