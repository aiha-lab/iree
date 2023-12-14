// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_PIM_IR_PIMATTRIBUTES_H_
#define IREE_COMPILER_DIALECT_PIM_IR_PIMATTRIBUTES_H_

#include "iree/compiler/Dialect/PIM/IR/PIMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/PIM/IR/PIMAttributes.h.inc"  // IWYU pragma: export

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace PIM {


}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_PIM_IR_PIMATTRIBUTES_H_
