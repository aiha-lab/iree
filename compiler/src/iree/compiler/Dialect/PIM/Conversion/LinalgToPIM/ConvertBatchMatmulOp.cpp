// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/PIM/Conversion/LinalgToPIM/Patterns.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Dialect/PIM/IR/PIMDialect.h"
#include "iree/compiler/Dialect/PIM/IR/PIMOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Value.h"
#include <iostream>
#include <bitset>
#include <cmath>
#include <algorithm>

using namespace std;

namespace mlir {
namespace iree_compiler {

namespace {

// Pattern to convert a linalg.batch_matmul operation into PIM command operations.
struct BatchMatmulOpPattern : public mlir::OpRewritePattern<linalg::BatchMatmulOp> {
  BatchMatmulOpPattern(MLIRContext *context, std::vector<std::pair<int, int>> bufferInfo, string layerType, string commType, int numDevice, std::vector<int> decoderConfig)
      : OpRewritePattern<linalg::BatchMatmulOp>(context), hal_buffer_info(std::move(bufferInfo)), layer(std::move(layerType)), 
        sync(std::move(commType)), num_device(std::move(numDevice)), decoder_config(std::move(decoderConfig)) {}

  mlir::LogicalResult matchAndRewrite(linalg::BatchMatmulOp op, PatternRewriter &rewriter) const override {
    
    std::cout << "ConvertBatchMatmulOp.cpp: enter function\n";

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

    // Get tile sizes
    SmallVector<int64_t> tileSizes;
    auto loweringConfig = op->getAttrOfType<IREE::Codegen::LoweringConfigAttr>("lowering_config");
    if(loweringConfig){
      llvm::outs() << "ConvertBatchMatmulOp.cpp: Get tileSizesAttr\n";
      // loweringConfig.dump();
      tileSizes.assign(loweringConfig.getTileSizeVals(0));
      llvm::outs() << "ConvertBatchMatmulOp.cpp: Tile sizes: ";
      for (size_t i = 0; i < tileSizes.size(); ++i) {
        std::cout << tileSizes[i];
        if (i < tileSizes.size() - 1) {
            std::cout << ", ";
        }
      }
      std::cout << "\n";
    }
    else {
      llvm::outs() << "ConvertBatchMatmulOp.cpp: There is no lowering_config attribute\n";
    }

    int dyn = 1; // dynamic iteration mode
    int n_head;
    if(loweringConfig) {
      n_head = tileSizes[0];
    }
    else
      n_head = shape_r[0];

    ////////////////////////////////////////
    // Q*K
    ////////////////////////////////////////
    // n_head, d_head, token
    if (layer == "qk") {
      llvm::outs() << "ConvertBatchMatmulOp: Is Query*Key\n";
      llvm::outs() << "ConvertBatchMatmulOp: n_head: " << n_head << "\n";
      int d_model = decoder_config[0];
      int d_head = decoder_config[2];
      int token = decoder_config[3];

      // Query*Key instruction gen
      mlir::Value dim_b_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), n_head, 32);
      mlir::Value dim_m_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 1, 32);
      mlir::Value dim_n_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), token, 32);
      mlir::Value dim_k_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), d_head, 32);
      rewriter.create<IREE::PIM::QKMatmulOp>(op.getLoc(), dim_b_val, dim_m_val, dim_n_val, dim_k_val);
    }

    else if (layer == "sv") {
      int d_model = decoder_config[0];
      int d_head = decoder_config[2];
      int token = decoder_config[3];

      llvm::outs() << "ConvertBatchMatmulOp: Is Score*Value\n";
      llvm::outs() << "ConvertBatchMatmulOp: n_head: " << n_head << "\n";
      
      // Score*Value instruction gen
      mlir::Value dim_b_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), n_head, 32);
      mlir::Value dim_m_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 1, 32);
      mlir::Value dim_n_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), d_head, 32);
      mlir::Value dim_k_val = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), token, 32);
      rewriter.create<IREE::PIM::SVMatmulOp>(op.getLoc(), dim_b_val, dim_m_val, dim_n_val, dim_k_val);

    } // Score*Value codegen

    llvm::outs() << "ConvertBatchMatmulOp.cpp: BatchMatmulOpPattern End\n";
    return mlir::success();
  }

  private:
    std::vector<std::pair<int, int>> hal_buffer_info;
    string layer;
    string sync;
    int num_device;
    std::vector<int> decoder_config;
};

} // namespace
 

void populateLinalgBatchMatmulToPIMPatterns(MLIRContext* context, RewritePatternSet &patterns, bool is_fused_matmul, std::vector<std::pair<int, int>> hal_buffer_info, 
  string layer, string sync, int num_device, std::vector<int> decoder_config) {
  // std::cout<< "ConvertBatchMatmulOp: populateLinalgBatchMatmulToPIMPatterns\n";
  context->getOrLoadDialect<IREE::PIM::PIMDialect>();
  context->getOrLoadDialect<IREE::Flow::FlowDialect>();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  
  patterns.add<BatchMatmulOpPattern>(context, hal_buffer_info, layer, sync, num_device, decoder_config);
}

}  // namespace iree_compiler
}  // namespace mlir