// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_PIM_BASE
#define IREE_DIALECT_PIM_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// PIM dialect definition
//===----------------------------------------------------------------------===//

def PIM_Dialect : Dialect {
  let name = "pim";
  let cppNamespace = "::mlir::iree_compiler::IREE::PIM";
  let summary = "The PIM dialect in IREE";
  let description = [{
    "Represent the AiM instruction-level operations"
  }];
}

//===----------------------------------------------------------------------===//
// Base PIM op classes
//===----------------------------------------------------------------------===//

/*
class PIM_Op<string mnemonic, list<Trait> traits = []> :
    Op<PIM_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}
*/

//===----------------------------------------------------------------------===//
// Utility definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target environment
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target triple
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Subgroup features
//===----------------------------------------------------------------------===//

#endif  // IREE_DIALECT_PIM_BASE
