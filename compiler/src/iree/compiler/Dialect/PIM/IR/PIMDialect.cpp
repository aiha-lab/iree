// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/PIM/IR/PIMDialect.h"
#include "iree/compiler/Dialect/PIM/IR/PIMOps.h"
#include "iree/compiler/Dialect/PIM/IR/PIMAttributes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace PIM {

PIMDialect::PIMDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<PIMDialect>()) {
  registerAttributes();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/PIM/IR/PIMOps.cpp.inc"
	  >();
}

}  // namespace PIM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
