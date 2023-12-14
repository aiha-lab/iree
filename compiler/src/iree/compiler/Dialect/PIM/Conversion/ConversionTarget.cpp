// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/PIM/Conversion/ConversionTarget.h"

// #include "iree/compiler/Dialect/PIM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/PIM/IR/PIMOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {

PIMConversionTarget::PIMConversionTarget(MLIRContext *context,
                                         TypeConverter &typeConverter)
    : ConversionTarget(*context) {
  // The PIM dialect allows PIM ops as input as we may be running on partially
  // processed files or may have already lowered some constructs (like constant
  // pools).
  addLegalDialect("pim");

  // Setup the fallback handler such that all ops without explicitly
  // registered patterns will be checked to ensure that they don't use any
  // illegal types.
  markUnknownOpDynamicallyLegal([&](Operation *op) {
    // Short-circuit test that bails on the first illegal type.
    const auto isTypeIllegal = [&](Type type) {
      return !typeConverter.isLegal(type);
    };
    return !(llvm::any_of(op->getOperandTypes(), isTypeIllegal) ||
             llvm::any_of(op->getResultTypes(), isTypeIllegal));
  });
}

}  // namespace iree_compiler
}  // namespace mlir
