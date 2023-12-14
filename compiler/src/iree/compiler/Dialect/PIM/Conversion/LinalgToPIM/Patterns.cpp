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

extern void populateLinalgMatmulToPIMPatterns(MLIRContext* context, RewritePatternSet &patterns, bool is_fused_matmul, 
  std::vector<std::pair<int, int>> hal_buffer_info, string layer, string sync, int num_device, std::vector<int> decoder_config);
extern void populateLinalgBatchMatmulToPIMPatterns(MLIRContext* context, RewritePatternSet &patterns, bool is_fused_matmul, 
  std::vector<std::pair<int, int>> hal_buffer_info, string layer, string sync, int num_device, std::vector<int> decoder_config);
extern void populateLinalgGenericToPIMPatterns(MLIRContext* context, RewritePatternSet &patterns, bool is_fused_matmulm, string layer, string sync, int num_device, std::vector<int> decoder_config);

void populateLinalgToPIMPatterns(MLIRContext* context, RewritePatternSet &patterns, bool is_fused_matmul, 
  std::vector<std::pair<int, int>> hal_buffer_info, string layer, string sync, int num_device, std::vector<int> decoder_config) {

  std::cout<< "\npopulateLinalgToPIMPatterns\n";

  populateLinalgMatmulToPIMPatterns(context, patterns, is_fused_matmul, hal_buffer_info, layer, sync, num_device, decoder_config);
  populateLinalgBatchMatmulToPIMPatterns(context, patterns, is_fused_matmul, hal_buffer_info, layer, sync, num_device, decoder_config);
  populateLinalgGenericToPIMPatterns(context, patterns, is_fused_matmul, layer, sync, num_device, decoder_config);

}

}  // namespace iree_compiler
}  // namespace mlir
