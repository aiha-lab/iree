// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Passes.cpp - Pipelines from Linalg ops to PIM -------------------===//
//
// This file contains various pipelines to lower IREE HAL executables containing
// Linalg ops to PIM.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Passes.h"

#include "iree/compiler/Dialect/PIM/IR/PIMOps.h" // include PIM
#include "iree/compiler/Dialect/PIM/IR/PIMDialect.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>

#define DEBUG_TYPE "iree-pim-lowering-pass-pipeline"

namespace mlir {
namespace iree_compiler {


//===----------------------------------------------------------------------===//
// Common Pass Recipes
//===----------------------------------------------------------------------===//

static void addTileAndDistributeToWorkgroupsPasses(
    OpPassManager &passManager, bool useFuseTensorPadWithConsumerPass = false,
    bool useWARForCooperativeMatrixCodegen = false) {
  passManager.addPass(createTileAndDistributeToWorkgroupsPass());
  auto &nestedModulePM = passManager.nest<ModuleOp>();
  if (useFuseTensorPadWithConsumerPass) {
    nestedModulePM.addNestedPass<func::FuncOp>(
        createFuseTensorPadWithConsumerPass());
  }
  nestedModulePM.addNestedPass<func::FuncOp>(
      createConvertToDestinationPassingStylePass(
          useWARForCooperativeMatrixCodegen));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

static void tileAndDistributeToWorkgroup(
    OpPassManager &pm, bool useWARForCooperativeMatrixCodegen = false) {
  pm.addPass(createTileAndDistributeToWorkgroupsPass());

  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createConvertToDestinationPassingStylePass(
          useWARForCooperativeMatrixCodegen));
  nestedModulePM.addNestedPass<func::FuncOp>(
      IREE::LinalgExt::createTileAndDecomposeAttentionPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

static void addPIMLoweringPasses(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerAffinePass());

  pm.addPass(createConvertToPIMPass());

  // TODO: additional optimization passes for PIM operations

}

//===---------------------------------------------------------------------===//
// Codegen pipelines.
//===---------------------------------------------------------------------===//

void addPIMMatmulPassPipeline(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createWorkgroupSpecializationPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

void buildPIMCodegenPassPipeline(OpPassManager &pm) {
  std::cout << "PIMLowerExecutableTarget.cpp: PIMCodegenPassPipeline\n";
  pm.nest<ModuleOp>().nest<func::FuncOp>().addPass(createTypePropagationPass());
  pm.nest<ModuleOp>().addPass(createBufferizeCopyOnlyDispatchesPass());
  pm.nest<ModuleOp>().addNestedPass<func::FuncOp>(
      IREE::LinalgExt::createDecomposeSoftmaxPass());
  // Temporary solution to avoid large allocations due to softmax lowering.
  pm.nest<ModuleOp>().addNestedPass<func::FuncOp>(
      createRematerializeParallelOpsPass());
  
  pm.addPass(createPIMLowerExecutableTargetPass());

  addTileAndDistributeToWorkgroupsPasses(
       pm, /*useFuseTensorPadWithConsumerPass=*/true);

  addPIMLoweringPasses(pm.nest<ModuleOp>());

  LLVM_DEBUG({
    llvm::dbgs() << "Using PIM pass pipeline:\n";
    pm.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

}  // namespace iree_compiler
}  // namespace mlir
