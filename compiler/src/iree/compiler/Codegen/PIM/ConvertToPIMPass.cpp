// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- CovertToPIMPass.cpp - Performs the final PIM conversion -------===//
//
// This file implements a pass to perform the final conversion to PIM.
//
//===----------------------------------------------------------------------===//

#include <tuple>
#include <iostream>

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/PIM/IR/PIMOps.h" // include PIM
#include "iree/compiler/Dialect/PIM/IR/PIMDialect.h"
#include "iree/compiler/Dialect/PIM/Conversion/LinalgToPIM/Patterns.h"
#include "iree/compiler/Dialect/PIM/Conversion/ConversionTarget.h"

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "json/json.h"
#include <fstream>

using namespace std;

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// Resource utilities
//===----------------------------------------------------------------------===//

} // namespace
//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace{

//===----------------------------------------------------------------------===//
// Conversion pass
//===----------------------------------------------------------------------===//

/// A pass to perform the PIM conversion.
///
/// This pass converts loop/standard ops into
/// corresponding PIM ops.

class ConvertToPIMPass : public ConvertToPIMBase<ConvertToPIMPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::PIM::PIMDialect, 
	    arith::ArithDialect, IREE::Flow::FlowDialect>();
  }

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) return failure();
    return success();
  }

  void runOnOperation() override;

 private:
  const std::vector<std::string> fused_matmul_pattern = {"matmul", "generic"};  // Softmax pattern

  bool doesModuleContainMatmulGenericPattern(mlir::ModuleOp moduleOp) {
    std::vector<std::string> operationSequence;
    moduleOp.walk([&](Operation *op) {
      if ( isa<mlir::linalg::MatmulOp>(op) ||  isa<mlir::linalg::BatchMatmulOp>(op)) {
        operationSequence.push_back("matmul");
      }
      else if (isa<mlir::linalg::GenericOp>(op)) {
        operationSequence.push_back("generic");
      }
    });
    if (operationSequence == fused_matmul_pattern)
      return true; // Pattern found
    return false; // Pattern not found
  }

};

} // namespace


void ConvertToPIMPass::runOnOperation() {
  // std::cout << "\nConvertToPIMPass\n";
  MLIRContext *context = &getContext();
  // context->getOrLoadDialect<IREE::PIM::PIMDialect>();
  RewritePatternSet patterns(&getContext());
  ModuleOp moduleOp = getOperation();

  std::string disp_num;
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    std::string input = funcOp.getName().str();
    std::string key = "dispatch_";
    size_t startPos = input.find(key);
    if (startPos != std::string::npos) {
        startPos += key.length(); // Start after "dispatch_"
        size_t endPos = input.find('_', startPos);
        disp_num = input.substr(startPos, endPos - startPos);
    }
  }

  Json::Value root;
  Json::Reader reader;
  // llvm::outs() << "Open executable map json\n";
  std::ifstream json("exe_map.json", ifstream::binary);
  reader.parse(json, root);
  string layer, sync;
  int num_device;
  std::vector<int> config;
  
  num_device = std::stoi(root["num_device"].asString());
  if (root["workload"]=="decoder") {
    layer = root[disp_num]["layer"].asString();
    sync = root[disp_num]["sync"].asString();
    config.push_back(root["config"]["d_model"].asInt()); // d_model
    config.push_back(root["config"]["n_head"].asInt()); // n_head
    config.push_back(root["config"]["d_head"].asInt()); // d_head
    config.push_back(root["config"]["token"].asInt()); // token
  }

  else {
    layer = "normal";
    sync = root[disp_num].asString();
  }
  
  llvm::outs() << "Layer: " << layer << "\n";
  llvm::outs() << "Communication type: " << sync << "\n";

  bool is_fused_matmul = doesModuleContainMatmulGenericPattern(moduleOp);
  
  // std::cout<< "Is fused matmul : "<< is_fused_matmul << std::endl;
  // Declare a vector to store pairs of int values.
  std::vector<std::pair<int, int>> hal_buffer_info;

  /*
  moduleOp.walk([&](Operation *op) {
  if (auto subspanOp = dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(op)) {
    mlir::Value byte_offset = subspanOp.getByteOffset();
    int offset_val = 0;
    if (auto constantOp = byte_offset.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
      offset_val = constantOp.value();
      // llvm::outs() << "Offset Value: " << offset_val << "\n";
    }
    int buffer_type = 2;
    if(auto DescriptorFlagsAttr = subspanOp.getDescriptorFlagsAttr()) {
      int32_t intValue = DescriptorFlagsAttr.getInt();
      switch (intValue) {
        case 0x0000:
          // llvm::outs() << "The descriptor flag is None.\n";
          buffer_type = 0;
          break;
        case 0x0001:
          // llvm::outs() << "The descriptor flag is ReadOnly.\n";
          buffer_type = 1;
          break;
        
        // default:
          // llvm::outs() << "Unknown descriptor flag with value: " << intValue << ".\n";
          // break;
      }
    }
    else {
      // llvm::outs() << "Is Write Only.\n\n";
    }
    // Add the pair to the vector.
    hal_buffer_info.push_back({buffer_type, offset_val});
    // Print out the pairs.
    for (const auto& info : hal_buffer_info) {
      llvm::outs() << "(buffer_type: " << info.first << ", offset: " << info.second << ")\n";
    }
  }
  });*/
  

  // populateArithToPIMPatterns(context, patterns);
  populateLinalgToPIMPatterns(context, patterns, is_fused_matmul, hal_buffer_info, layer, sync, num_device, config);
  // populateMemRefToPIMPatterns(context, patterns);
  // populateFlowToPIMPatterns(context, patterns);

  ConversionTarget target(getContext());
  // target.addIllegalOp<linalg::MatmulOp>();
  target.addLegalDialect<IREE::PIM::PIMDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect>();

  // Apply the conversion patterns.
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }

}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createConvertToPIMPass(
	) {
  return std::make_unique<ConvertToPIMPass>();
}

} // namespace iree_compiler
} // namespace mlir
