# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_test")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_compiler_cc_test(
    name = "api-test-binary",
    testonly = True,
    srcs = ["api-test-main.c"],
    deps = [
        "//compiler/src/iree/compiler/API2:Headers",
        "//compiler/src/iree/compiler/API2:Impl",
        "//runtime/src/iree/base",
    ],
)
