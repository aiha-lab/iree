// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/PIM/IR/PIMAttributes.h"

#include "iree/compiler/Dialect/PIM/IR/PIMDialect.h"
#include "iree/compiler/Dialect/PIM/IR/PIMTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/PIM/IR/PIMAttributes.cpp.inc"  
namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace PIM {

//===----------------------------------------------------------------------===//
// TargetEnv
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Attribute Parsing
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Attribute Printing
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void PIMDialect::registerAttributes() {
}

}  // namespace PIM 
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
