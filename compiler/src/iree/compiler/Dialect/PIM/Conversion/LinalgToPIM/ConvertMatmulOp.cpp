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
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Value.h"
#include <iostream>
#include <bitset>
#include <cmath>

namespace mlir {
namespace iree_compiler {

namespace {

///////////////////////////////////////////////////////////////////////////////////
// Pattern to convert a linalg.matmul operation into PIM command operations.

struct MatmulOpPattern : public mlir::OpRewritePattern<linalg::MatmulOp> {
  MatmulOpPattern(MLIRContext *context, std::vector<std::pair<int, int>> bufferInfo, bool isFusedMatmul, string layerType, string commType, int numDevice, std::vector<int> decoderConfig)
      : OpRewritePattern<linalg::MatmulOp>(context), hal_buffer_info(std::move(bufferInfo)), 
      is_fused_matmul(std::move(isFusedMatmul)), layer(std::move(layerType)), sync(std::move(commType)), num_device(std::move(numDevice)), decoder_config(std::move(decoderConfig)) {}
  
  mlir::LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
    
    llvm::outs() << "ConvertMatmulOp.cpp: enter function\n";
    if (is_fused_matmul)
      llvm::outs() << "ConvertMatmulOp.cpp: Fused matmul (bias)\n";
    // std::string op_name = "matmul";

    SmallVector<int64_t> tileSizes;
    auto loweringConfig = op->getAttrOfType<IREE::Codegen::LoweringConfigAttr>("lowering_config");
    // SmallVector<int64_t> tileSizes = loweringConfig.getTileSizeVals(0);
    
    // Get tile config attribute
    if(loweringConfig){
      llvm::outs() << "ConvertMatmulOp.cpp: Get tileSizesAttr\n";
      // loweringConfig.dump();
      tileSizes.assign(loweringConfig.getTileSizeVals(0));
      llvm::outs() << "ConvertMatmulOp.cpp: Tile sizes: ";
      for (size_t i = 0; i < tileSizes.size(); ++i) {
        std::cout << tileSizes[i];
        if (i < tileSizes.size() - 1) {
            std::cout << ", ";
        }
      }
      std::cout << "\n";
    }
    else {
      llvm::outs() << "ConvertMatmulOp.cpp: There is no lowering_config attribute\n";
    }

    // Access the first input tensor operand of the linalg::MatmulOp.
    mlir::Value inputTensor_l = op.getDpsInputOperands()[0]->get();
    // Access the type of the tensor, which will be a ShapedType.
    mlir::ShapedType inputType_l = inputTensor_l.getType().cast<mlir::ShapedType>();
    // The shape of the tensor contains the dimension values.
    mlir::ArrayRef<int64_t> shape_l = inputType_l.getShape();
    // Access the second input tensor operand of the linalg::MatmulOp.
    mlir::Value inputTensor_r = op.getDpsInputOperands()[1]->get();
    // Access the type of the tensor, which will be a ShapedType.
    mlir::ShapedType inputType_r = inputTensor_r.getType().cast<mlir::ShapedType>();
    // The shape of the tensor contains the dimension values.
    mlir::ArrayRef<int64_t> shape_r = inputType_r.getShape();

    
    // Print out the shape
    llvm::outs() << "ConvertMatmulOp.cpp: 1st argument shape: ";
    for (int64_t dim : shape_l) {
      llvm::outs() << dim << " ";
    }
    llvm::outs() << "\n";
    
    llvm::outs() << "ConvertMatmulOp.cpp: 2nd argument shape: ";
    for (int64_t dim : shape_r) {
      llvm::outs() << dim << " ";
    }
    llvm::outs() << "\n";

    int dim_m;
    int dim_n;
    int dim_k;

    if (loweringConfig) {
      dim_m = tileSizes[0];
      dim_n = tileSizes[1];
      dim_k = tileSizes[2];
    }
    else {  
      dim_m = shape_l[0];
      dim_n = shape_r[1];
      dim_k = shape_r[0]; // = shape_l[1]
    }

    int split_type = 2;

    if (!(dim_k == shape_r[0])) { // col-wise, all-reduce
      llvm::outs() << "ConvertMatmulOp.cpp: column-wise\n";
      split_type = 0;
    }
    
    else if (!(dim_n == shape_r[1])) { // row-wise, all-gather
      // llvm::outs() << "ConvertMatmulOp.cpp: row-wise\n";
      split_type = 1;
    }
    
    // llvm::outs() << "ConvertMatmulOp.cpp: MNK: " << dim_m << " " << dim_n << " " << dim_k << "\n";
    int act = 0;
    bool is_multi_bias = false;
    if (layer == "fc1") {
      // llvm::outs() << "ConvertMatmulOp.cpp: Matmul with the activation function\n";
      act = 1;
    }
    if ((layer == "c_proj+residual")||(layer=="fc2+residual")) {
      is_multi_bias = true;
    }
    
    int d_model = decoder_config[0];
    
    mlir::Value lhs = op.getInputs()[0];
    mlir::Value rhs = op.getInputs()[1];
    mlir::Value result = op.getOutputs()[0];

    mlir::Value dim_m_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), dim_m, 32);
    mlir::Value dim_n_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), dim_n, 32);
    mlir::Value dim_k_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), dim_k, 32);
    // mlir::Value act_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), act, 32);
    
    if (layer=="c_attn")
      rewriter.create<IREE::PIM::QKVGenOp>(op.getLoc(), dim_m_val, dim_n_val, dim_k_val);

    else if (layer == "c_proj+residual")
      rewriter.create<IREE::PIM::OutProjOp>(op.getLoc(), dim_m_val, dim_n_val, dim_k_val);

    else if (layer=="fc1")
      rewriter.create<IREE::PIM::FFN1Op>(op.getLoc(), dim_m_val, dim_n_val, dim_k_val);

    else if (layer=="fc2+residual")
      rewriter.create<IREE::PIM::FFN2Op>(op.getLoc(), dim_m_val, dim_n_val, dim_k_val);
    
    /*
    if ((sync == "reduce") && (num_device!=1)) {
      rewriter.create<IREE::PIM::AllReduceOp>(op.getLoc());
    }  
    
    else if ((sync == "gather") && (num_device!=1))
      rewriter.create<IREE::PIM::AllGatherOp>(op.getLoc());
    */

    llvm::outs() << "ConvertMatmulOp.cpp: MatmulOpPattern End" << "\n";
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

// Pattern to erase a flow.tensor.store operation.
struct EraseFlowDispatchTensorStoreOpPattern : public mlir::OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
  using mlir::OpRewritePattern<IREE::Flow::DispatchTensorStoreOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorStoreOp op, PatternRewriter &rewriter) const override {
    
    llvm::outs() << "\nEraseFlowDispatchTensorStoreOpPattern\n";
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace
 
void populateLinalgMatmulToPIMPatterns(MLIRContext* context, RewritePatternSet &patterns, bool is_fused_matmul, 
  std::vector<std::pair<int, int>> hal_buffer_info, string layer, string sync, int num_device, std::vector<int> decoder_config) {

  // llvm::outs() << "ConvertMatmulOp: populateLinalgToPIMPatterns\n";
  context->getOrLoadDialect<IREE::PIM::PIMDialect>();
  context->getOrLoadDialect<IREE::Flow::FlowDialect>();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  patterns.add<MatmulOpPattern>(context, hal_buffer_info, is_fused_matmul, layer, sync, num_device, decoder_config);
}

}  // namespace iree_compiler
}  // namespace mlir
