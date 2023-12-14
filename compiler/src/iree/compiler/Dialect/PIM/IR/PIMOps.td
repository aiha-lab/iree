// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_PIM_OPS
#define IREE_DIALECT_PIM_OPS

include "mlir/IR/OpBase.td"
include "iree/compiler/Dialect/PIM/IR/PIMBase.td"
include "iree/compiler/Dialect/Util/IR/UtilAttrs.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"


class PIM_Op<string mnemonic, list<Trait> traits = []> :
    Op<PIM_Dialect, mnemonic, traits>;

/*
def PIM_MatmulOp : PIM_Op<"matmul"> {
  let summary = "matrix multiplication operation";
  let description = [{
    Matrix multiplication in PIM device.
  }];

  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  let assemblyFormat = [{
    attr-dict $lhs `,` $rhs `:` type($lhs) `x` type($rhs) `->` type(results)
  }];

}*/

def PIM_LayerNorm1Op : PIM_Op<"layernorm1"> {
  let summary = "layernorm operation";
  let description = [{
    Compute layer normalization in PIM device.
  }];

  let arguments = (ins
    I32:$length
  );

  let assemblyFormat = "operands attr-dict";
}

def PIM_QKVGenOp : PIM_Op<"qkv_gen"> {
  let summary = "QKV generation";
  let description = [{
    QKV gen matrix multiplication in PIM device.
  }];

  let arguments = (ins
    I32:$dim_m,
    I32:$dim_n,
    I32:$dim_k
  );

  let assemblyFormat = "operands attr-dict";
}

def PIM_QKMatmulOp : PIM_Op<"qk_matmul"> {
  let summary = "Query-key multiplication operation";
  let description = [{
    Batched matrix multiplication in PIM device.
  }];

  let arguments = (ins
    I32:$dim_b,
    I32:$dim_m,
    I32:$dim_n,
    I32:$dim_k
  );

  let assemblyFormat = "operands attr-dict";
}

def PIM_SoftmaxOp : PIM_Op<"softmax"> {
  let summary = "softmax operation";
  let description = [{
    Compute softmax in PIM device.
  }];

  let arguments = (ins
    I32:$length
  );

  let assemblyFormat = "operands attr-dict";
}

def PIM_SVMatmulOp : PIM_Op<"sv_matmul"> {
  let summary = "Score-value multiplication operation";
  let description = [{
    Batched matrix multiplication in PIM device.
  }];

  let arguments = (ins
    I32:$dim_b,
    I32:$dim_m,
    I32:$dim_n,
    I32:$dim_k
  );

  let assemblyFormat = "operands attr-dict";
}

def PIM_OutProjOp : PIM_Op<"out_proj"> {
  let summary = "Output projection";
  let description = [{
    Output projection matrix multiplication in PIM device.
  }];

  let arguments = (ins
    I32:$dim_m,
    I32:$dim_n,
    I32:$dim_k
  );

  let assemblyFormat = "operands attr-dict";
}

def PIM_LayerNorm2Op : PIM_Op<"layernorm2"> {
  let summary = "layernorm operation";
  let description = [{
    Compute layer normalization in PIM device.
  }];

  let arguments = (ins
    I32:$length
  );

  let assemblyFormat = "operands attr-dict";
}

def PIM_FFN1Op : PIM_Op<"ffn1"> {
  let summary = "FFN1";
  let description = [{
    FFN1 matrix multiplication in PIM device.
  }];

  let arguments = (ins
    I32:$dim_m,
    I32:$dim_n,
    I32:$dim_k
  );

  let assemblyFormat = "operands attr-dict";
}

def PIM_FFN2Op : PIM_Op<"ffn2"> {
  let summary = "FFN2";
  let description = [{
    FFN2 matrix multiplication in PIM device.
  }];

  let arguments = (ins
    I32:$dim_m,
    I32:$dim_n,
    I32:$dim_k
  );

  let assemblyFormat = "operands attr-dict";
}

/*
def PIM_AllReduceOp : PIM_Op<"all_reduce"> {
  let summary = "all-reduce operation";
  let description = [{
    Reduction across multi PIM devices.
  }];

  // let arguments = ();

  let assemblyFormat = "operands attr-dict";
}

def PIM_AllGatherOp : PIM_Op<"all_gather"> {
  let summary = "all-gather operation";
  let description = [{
    Gathering across multi PIM devices.
  }];

  // let arguments = ();

  let assemblyFormat = "operands attr-dict";
}*/

#endif  // IREE_DIALECT_PIM_OPS
