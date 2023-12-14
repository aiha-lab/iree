// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_PIM_CONVERSION_LINALGTOPIM_PATTERNS_H_
#define IREE_COMPILER_DIALECT_PIM_CONVERSION_LINALGTOPIM_PATTERNS_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <regex>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <cmath>

using namespace std;

namespace mlir {
namespace iree_compiler {

// Populates conversion patterns for linalg->PIM.
void populateLinalgToPIMPatterns(MLIRContext* context, RewritePatternSet &patterns, bool is_fused_matmul, 
    std::vector<std::pair<int, int>> hal_buffer_info, string layer, string sync, int num_device, std::vector<int> decoder_config);


}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_PIM_CONVERSION_LINALGTOPIM_PATTERNS_H_
