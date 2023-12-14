// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/PIM/Conversion/LinalgToPIM/Patterns.h"

#include "iree/compiler/Dialect/PIM/IR/PIMDialect.h"
#include "iree/compiler/Dialect/PIM/IR/PIMOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Value.h"
#include <iostream>
#include <bitset>

namespace mlir {
namespace iree_compiler {

namespace {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Patterns to generate a PIM command stack for Softmax
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int doesGenericOpMatchPattern(linalg::GenericOp genericOp) {
  std::vector<std::string> opsInGeneric;
  int nl_flag = 0;
  const std::vector<std::string> sm_pattern 
    = {"arith.divf", "linalg.yield", "linalg.generic"};  // Softmax pattern
  
  const std::vector<std::string> ln_pattern 
    = {"arith.divf", "arith.addf", "math.rsqrt", "arith.mulf", "linalg.yield", "linalg.generic"};  // LayerNorm pattern

  const std::vector<std::string> sc_pattern 
    = {"arith.addf", "arith.addf", "linalg.yield", "linalg.generic"};  // Shortcut pattern
  
  genericOp->walk([&](Operation *nestedOp) {
    opsInGeneric.push_back(nestedOp->getName().getStringRef().str());
  });

  // Check if opsInGeneric matches the desired pattern.
  if (opsInGeneric == sm_pattern) {
    llvm::outs() << "ConvertGeneric.cpp: Softmax pattern\n";
    nl_flag = 1;
    return nl_flag;
  }
  else if (opsInGeneric == ln_pattern) {
    llvm::outs() << "ConvertGeneric.cpp: LayerNorm pattern\n";
    nl_flag = 2;
    return nl_flag;
  }
  else if (opsInGeneric == sc_pattern) {
    llvm::outs() << "ConvertGeneric.cpp: Shortcut pattern\n";
    nl_flag = 3;
    return nl_flag;
  }
  return nl_flag;
}

// Pattern to convert a linalg.matmul operation into PIM command operations.

struct NonlinearPattern : public mlir::OpRewritePattern<linalg::GenericOp> {
  NonlinearPattern(MLIRContext *context, bool isFusedMatmul, string layerType, string commType, int numDevice, std::vector<int> decoderConfig)
      : OpRewritePattern<linalg::GenericOp>(context), is_fused_matmul(std::move(isFusedMatmul)), 
        layer(std::move(layerType)), sync(std::move(commType)), num_device(std::move(numDevice)), decoder_config(std::move(decoderConfig)) {}

/*
struct NonlinearPattern : public mlir::OpRewritePattern<linalg::GenericOp> {
  using mlir::OpRewritePattern<linalg::GenericOp>::OpRewritePattern;*/

  mlir::LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    
    llvm::outs() << "ConvertGeneric.cpp: enter function\n";

    // Access the first input tensor operand of the linalg::MatmulOp.
    mlir::Value inputTensor_l = op.getDpsInputOperands()[0]->get();
    // Access the type of the tensor, which will be a ShapedType.
    mlir::ShapedType inputType_l = inputTensor_l.getType().cast<mlir::ShapedType>();
    // The shape of the tensor contains the dimension values.
    mlir::ArrayRef<int64_t> shape_l = inputType_l.getShape();
    /*
    // Access the second input tensor operand of the linalg::MatmulOp.
    mlir::Value inputTensor_r = op.getDpsInputOperands()[1]->get();
    // Access the type of the tensor, which will be a ShapedType.
    mlir::ShapedType inputType_r = inputTensor_r.getType().cast<mlir::ShapedType>();
    // The shape of the tensor contains the dimension values.
    mlir::ArrayRef<int64_t> shape_r = inputType_r.getShape();*/
    
    // Check if the operations inside the genericOp match the desired pattern.
    int op_type = doesGenericOpMatchPattern(op);
    if (layer == "softmax" && op_type == 1) {
      llvm::outs() << "ConvertGeneric.cpp: Softmax cmd gen\n";
      int d_model = decoder_config[0];
      int n_head = shape_l[0]/num_device;
      int token = shape_l[1];
      mlir::Value length = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), token, 32);
      rewriter.create<IREE::PIM::SoftmaxOp>(op.getLoc(), length);
      return mlir::success();
    }

    else if (layer == "layernorm1") {
      llvm::outs() << "ConvertGeneric.cpp: Layernorm cmd gen\n";
      int d_model = shape_l[1];
      mlir::Value length = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), d_model, 32);
      rewriter.create<IREE::PIM::LayerNorm1Op>(op.getLoc(), length);
      return mlir::success();
    }

    else if (layer == "layernorm2") {
      llvm::outs() << "ConvertGeneric.cpp: Layernorm cmd gen\n";
      int d_model = shape_l[1];
      mlir::Value length = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), d_model, 32);
      rewriter.create<IREE::PIM::LayerNorm2Op>(op.getLoc(), length);
      return mlir::success();
    }

    /*
    else if ((layer=="c_proj+residual")||(layer=="fc2+residual")) {
      llvm::outs() << "ConvertGeneric.cpp: Shorcut cmd gen - " << layer << "\n";
      int d_model = shape_l[1];
      mlir::Value length = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), d_model, 32);
      rewriter.create<IREE::PIM::AddOp>(op.getLoc(), length);
      
      return mlir::success();
    }*/
    
    else
      // std::cout << "SoftmaxPattern End" << std::endl;
      return mlir::success();
  }

  private:
    std::vector<std::pair<int, int>> hal_buffer_info;
    bool is_fused_matmul;
    string layer;
    string sync;
    int num_device;
    std::vector<int> decoder_config;
};

} // namespace
 

void populateLinalgGenericToPIMPatterns(MLIRContext* context, RewritePatternSet &patterns, bool is_fused_matmul, 
  string layer, string sync, int num_device, std::vector<int> decoder_config) {
  // llvm::outs() << "\npopulateLinalgGenericToPIMPatterns\n";
  context->getOrLoadDialect<IREE::PIM::PIMDialect>();
  context->getOrLoadDialect<mlir::math::MathDialect>();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  
  patterns.add<NonlinearPattern>(context, is_fused_matmul, layer, sync, num_device, decoder_config);

}

}  // namespace iree_compiler
}  // namespace mlir