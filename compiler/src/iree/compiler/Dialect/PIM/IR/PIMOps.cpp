// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/PIM/IR/PIMOps.h"

#include "iree/compiler/Dialect/PIM/IR/PIMTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace PIM {

/*void MACABKOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                            int32_t op_size, int32_t thread_idx, int32_t ch_mask, int32_t row_addr, int32_t col_addr) {
  state.addAttribute("op_size", builder.getI32IntegerAttr(op_size));
  state.addAttribute("thread_idx", builder.getI32IntegerAttr(thread_idx));
  state.addAttribute("ch_mask", builder.getI32IntegerAttr(ch_mask));
  state.addAttribute("row_addr", builder.getI32IntegerAttr(row_addr));
  state.addAttribute("col_addr", builder.getI32IntegerAttr(col_addr));
  // If the operation has results, you can also add them here.
  // For example:
  // state.addTypes({result_type});
}*/

}  // namespace PIM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/PIM/IR/PIMOps.cpp.inc"
