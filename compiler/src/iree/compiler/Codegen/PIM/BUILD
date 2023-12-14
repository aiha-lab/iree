# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_compiler_cc_library(
    name = "PIM",
    srcs = [
	    "Passes.cpp",
            ],
    hdrs = [
            ],
    deps = [
        "//compiler/src/iree/compiler/Codegen:PassHeaders",
        "//compiler/src/iree/compiler/Codegen/Common",
        "//compiler/src/iree/compiler/Codegen/Dialect:IREECodegenDialect",
        "//compiler/src/iree/compiler/Codegen/Transforms",
        "//compiler/src/iree/compiler/Codegen/Utils",
        "//compiler/src/iree/compiler/Dialect/Flow/IR",
        "//compiler/src/iree/compiler/Dialect/HAL/IR",
        "//compiler/src/iree/compiler/Dialect/HAL/IR:HALDialect",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "//llvm-external-projects/iree-dialects:IREELinalgExtPasses",
        "//llvm-external-projects/iree-dialects:IREELinalgExtTransforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:LinalgUtils",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToGPU",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorToSPIRV",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TosaToArith",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorInterfaces",
        "@llvm-project//mlir:VectorToGPU",
        "@llvm-project//mlir:VectorTransforms",
     ],
)
