# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_check_single_backend_test_suite(
    name = "check_cuda",
    srcs = enforce_glob(
        # keep sorted
        [
            "attention.mlir",
            "reverse.mlir",
            "scan.mlir",
            "scatter.mlir",
            "softmax.mlir",
            "sort.mlir",
            "top-k.mlir",
        ],
        include = ["*.mlir"],
        exclude = [
            "pack.mlir",
            "unpack.mlir",
            "winograd_input.mlir",
            "winograd_output.mlir",
        ],
    ),
    driver = "cuda",
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backend = "cuda",
)

iree_check_single_backend_test_suite(
    name = "check_cuda-topk-split-reduction",
    srcs = [
        "top-k.mlir",
    ],
    compiler_flags = ["--iree-flow-topk-split-reduction=2"],
    driver = "cuda",
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backend = "cuda",
)

iree_check_single_backend_test_suite(
    name = "check_cuda-topk-split-reduction-double",
    srcs = [
        "top-k.mlir",
    ],
    compiler_flags = ["--iree-flow-topk-split-reduction=3,2"],
    driver = "cuda",
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backend = "cuda",
)

iree_cmake_extra_content(
    content = """
# Failing on Emscripten: https://github.com/iree-org/iree/issues/12129
if(NOT EMSCRIPTEN)
""",
    inline = True,
)

iree_check_single_backend_test_suite(
    name = "check_llvm-cpu_local-task",
    srcs = enforce_glob(
        # keep sorted
        [
            "attention.mlir",
            "pack.mlir",
            "reverse.mlir",
            "scan.mlir",
            "scatter.mlir",
            "softmax.mlir",
            "sort.mlir",
            "top-k.mlir",
            "unpack.mlir",
            "winograd_input.mlir",
            "winograd_output.mlir",
        ],
        include = ["*.mlir"],
    ),
    driver = "local-task",
    target_backend = "llvm-cpu",
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)

iree_check_single_backend_test_suite(
    name = "check_vmvx_ukernel_local-task",
    srcs = [
        "pack.mlir",
        "unpack.mlir",
    ],
    compiler_flags = [
        "--iree-vmvx-enable-microkernels",
        # Some testcases have linalg.generic ops with multiple ops in the body.
        # If we don't opt out from it, DecomposeLinalgGenericPass splits those
        # into smaller linalg.generic ops with only one op in the body. This
        # results in the creation of temporary buffers between these split
        # linalg.generic ops, causing:
        # > error: failed to legalize operation 'memref.alloca' that was explicitly marked illegal
        "--iree-vmvx-enable-microkernels-decompose-linalg-generic=false",
    ],
    driver = "local-task",
    target_backend = "vmvx",
)

iree_check_single_backend_test_suite(
    name = "check_vmvx_local-task",
    srcs = enforce_glob(
        # keep sorted
        [
            "pack.mlir",
            "reverse.mlir",
            "scan.mlir",
            "scatter.mlir",
            "sort.mlir",
            "top-k.mlir",
            "unpack.mlir",
        ],
        include = ["*.mlir"],
        exclude = [
            "attention.mlir",
            "softmax.mlir",
            "winograd_input.mlir",
            "winograd_output.mlir",
        ],
    ),
    driver = "local-task",
    target_backend = "vmvx",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = enforce_glob(
        # keep sorted
        [
            "reverse.mlir",
            "scatter.mlir",
            # Top-k test disabled due to miscompile on vulkan.
            #    "top-k.mlir",
            "sort.mlir",
            "winograd_input.mlir",
            "winograd_output.mlir",
        ],
        include = ["*.mlir"],
        exclude = [
            "pack.mlir",
            "unpack.mlir",
            # TODO(antiagainst): scan fails on Adreno GPUs due to driver bug.
            # Re-enable this once we have new devices with up-to-date drivers.
            "top-k.mlir",
            "scan.mlir",
            "softmax.mlir",
            "attention.mlir",
        ],
    ),
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

test_suite(
    name = "check",
    tests = [
        ":check_cuda",
        ":check_llvm-cpu_local-task",
        ":check_vmvx_local-task",
        ":check_vulkan-spirv_vulkan",
    ],
)
