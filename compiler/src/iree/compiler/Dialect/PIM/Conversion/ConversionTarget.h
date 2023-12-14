// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_PIM_CONVERSION_CONVERSIONTARGET_H_
#define IREE_COMPILER_DIALECT_PIM_CONVERSION_CONVERSIONTARGET_H_

#include "iree/compiler/Dialect/PIM/IR/PIMDialect.h"
#include "iree/compiler/Dialect/PIM/IR/PIMOps.h"
#include "iree/compiler/Dialect/PIM/IR/PIMTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// A conversion target for the HAL dialect that ensures that tensor types are
// fully removed. Conversions targeting the HAL dialect should always use this.
class PIMConversionTarget : public ConversionTarget {
 public:
  PIMConversionTarget(MLIRContext *context, TypeConverter &typeConverter);

  // Attempts to rewrite an op that may use tensor values into an op using HAL
  // buffers. See HALOpConversion for more information.
  static LogicalResult applyDefaultBufferRewrite(
      Operation *srcOp, ValueRange operands, StringRef dstOpName,
      TypeConverter &typeConverter, ConversionPatternRewriter &rewriter);
};


}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_PIM_CONVERSION_CONVERSIONTARGET_H_
