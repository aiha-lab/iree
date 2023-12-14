// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PIM/KernelConfig.h"

#include <numeric>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/TransposeUtils.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "json/json.h"
#include <vector>
#include <fstream>

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace std;

static constexpr unsigned cudaWarpSize = 32;
static constexpr StringLiteral kCudaTarget = "cuda";
namespace mlir {
namespace iree_compiler {

}  // namespace iree_compiler
}  // namespace mlir

namespace {

/// Structure to represent target features.
struct TargetInfo {
  int device_num = 1;
  string workload = "normal";
  std::vector<string> layer_info;
};

struct TileWorkgroupSizePair {
  // How many scalar elements each workgroup should handle along each dimension.
  std::array<int64_t, 3> tileSize;
  std::array<int64_t, 3> workgroupSize;
};

string extractNumberAfterDispatch(const std::string& input) {
    std::string key = "dispatch_";
    size_t startPos = input.find(key);
    if (startPos != std::string::npos) {
        startPos += key.length(); // Start after "dispatch_"
        size_t endPos = input.find('_', startPos);
        string numberStr = input.substr(startPos, endPos - startPos);
        return numberStr; // Convert to integer
    }
    return "none"; // Return -1 or some error value if "dispatch_" is not found
}

// Get partining and sync info from json file
std::vector<string> getDecoderLayerInfo(Json::Value root, string disp_num) {
    std::vector<string> layer_info;
    string layer = root[disp_num]["layer"].asString();
    string partitioning = root[disp_num]["partioning"].asString();
    string sync = root[disp_num]["sync"].asString();
    layer_info.push_back(layer);
    layer_info.push_back(partitioning);
    layer_info.push_back(sync);
    return layer_info;
}

// Software pipeline depths
constexpr unsigned softwarePipelineDepthTensorCore = 4;
// Simt codegen does not do software pipelining.
constexpr unsigned softwarePipelineDepthSimt = 0;
}  // namespace

/// Return the best combination of tile size and wg size. It will then used to
/// pick the best size aligned with the shape dimension.
static void getMatmulConfig(SmallVectorImpl<TileWorkgroupSizePair> &tileSizes) {
  // Pick tile size so that M*K and K*N dividible by wgSize * \*vecSize=*\4.
  // This way workgroup memory copy don't need to be masked. Once we support
  // masked load we can get performance out of more configuration.
  tileSizes.push_back(TileWorkgroupSizePair({{32, 128, 32}, {32, 8, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{128, 64, 8}, {16, 8, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{16, 256, 32}, {64, 2, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{8, 32, 32}, {8, 8, 1}}));

  tileSizes.push_back(TileWorkgroupSizePair({{32, 128, 4}, {32, 8, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{8, 128, 4}, {32, 1, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{16, 64, 4}, {16, 2, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{1, 128, 8}, {32, 1, 1}}));
}

static TargetInfo getTargetInfo(func::FuncOp entryPoint, string workload, std::vector<string> layer_info) {
  TargetInfo info;
  int num_device;

  Json::Value root;
  Json::Reader reader;
  // llvm::outs() << "Open executable map json\n";
  std::ifstream json("exe_map.json", ifstream::binary);
  reader.parse(json, root);
  num_device = std::stoi(root["num_device"].asString());

  info.device_num = num_device;
  info.workload = workload;
  info.layer_info = layer_info;
  return info;
}

static LogicalResult setMatmulTileConfig(func::FuncOp entryPoint,
                                       linalg::LinalgOp op,
                                       const TargetInfo &targetInfo) {
  if (!isa<linalg::MatmulOp>(op))
    return failure();
  // Don't consider operations that don't have a broadcast, those should go
  // through reductions.
  if (llvm::any_of(op.getIndexingMapsArray(),
                   [](AffineMap m) { return m.isPermutation(); }))
    return failure();

  auto setMatmulConfig =
      [&entryPoint, &op](int64_t tileX, int64_t tileY, int64_t tileK,
                         llvm::ArrayRef<int64_t> workgroupSize,
                         unsigned softwarePipelineDepth,
                         IREE::Codegen::DispatchLoweringPassPipeline pipeline) {
        TileSizesListType tileSizes;
        unsigned numParallelLoops = op.getNumParallelLoops();
        SmallVector<int64_t> workgroupTileSizes(numParallelLoops - 2, 1);
        workgroupTileSizes.append({tileX, tileY});
        workgroupTileSizes.append(op.getNumReductionLoops(), tileK);

        SmallVector<unsigned> partitionedLoops =
            cast<PartitionableLoopsInterface>(op.getOperation())
                .getPartitionableLoops(kNumMaxParallelDims);
        llvm::SmallDenseSet<unsigned, 4> partitionedLoopsSet;
        partitionedLoopsSet.insert(partitionedLoops.begin(),
                                   partitionedLoops.end());
        for (auto loopID : llvm::seq<unsigned>(0, numParallelLoops)) {
          if (!partitionedLoopsSet.count(loopID)) {
            workgroupTileSizes[loopID] = 0;
          }
        }

        tileSizes.emplace_back(
            std::move(workgroupTileSizes));  // Workgroup level.
        return setOpConfigAndEntryPointFnTranslation(
            entryPoint, op, tileSizes, pipeline, workgroupSize,
            /*subgroupSize=*/std::nullopt, softwarePipelineDepth);
      };
  // Infer the MxN size of the matmul based on operands and indexing maps.
  auto lhsShape =
      op.getDpsInputOperand(0)->get().getType().cast<ShapedType>().getShape();
  auto rhsShape =
      op.getDpsInputOperand(1)->get().getType().cast<ShapedType>().getShape();
  int64_t sizeM = ShapedType::kDynamic;
  int64_t sizeN = ShapedType::kDynamic;
  int64_t sizeK = ShapedType::kDynamic;
  auto outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  for (unsigned i = 0; i < lhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getDpsInputOperand(0)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 2)) {
      sizeM = lhsShape[i];
      break;
    }
  }
  for (unsigned i = 0; i < rhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getDpsInputOperand(1)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 1)) {
      sizeN = rhsShape[i];
      break;
    }
  }
  SmallVector<unsigned> exprs;
  op.getReductionDims(exprs);
  if (exprs.size() == 1) {
    for (unsigned i = 0; i < lhsShape.size(); i++) {
      if (op.getMatchingIndexingMap(op.getDpsInputOperand(0))
              .getDimPosition(i) == exprs[0]) {
        sizeK = lhsShape[i];
        break;
      }
    }
  }
  bool isStaticSize = sizeM != ShapedType::kDynamic &&
                      sizeN != ShapedType::kDynamic &&
                      sizeK != ShapedType::kDynamic;

  int dev_num = targetInfo.device_num;
  // If we haven't found any config, use the best tile size hoping that
  // the workgroup specialization handles the main tile path efficiently.
  SmallVector<TileWorkgroupSizePair> tileSizeConfig;
  // Query the best configuration.
  getMatmulConfig(tileSizeConfig);
  constexpr size_t configIndex = 0;
  const TileWorkgroupSizePair &config = tileSizeConfig[configIndex];
  
  /*
  const int64_t tileX = config.tileSize[0];
  const int64_t tileY = config.tileSize[1];
  int64_t tileK = config.tileSize[2];*/

  // Default spliting: row-wise (all-gather)
  int64_t tileX = sizeM;
  int64_t tileY = sizeN/dev_num;
  int64_t tileK = sizeK;
  
  if (targetInfo.workload == "decoder") {
    llvm::outs() << "is decoder\n";
    string split = targetInfo.layer_info[1];
    string sync = targetInfo.layer_info[2];
    llvm::outs() << split << sync << "\n";
    if (split == "row-wise") {
      llvm::outs() << "KernelConfig.cpp: MatmulTileConfig: row-wise partitioning\n";
      tileX = sizeM;
      tileY = sizeN/dev_num;
      tileK = sizeK;
    }
    else if (split == "col-wise") {
      llvm::outs() << "KernelConfig.cpp: MatmulTileConfig: col-wise partitioning\n";
      tileX = sizeM;
      tileY = sizeN;
      tileK = sizeK/dev_num;
    }
  }

  else if (targetInfo.workload == "normal") {
    llvm::outs() << "is decoder\n";
    string split = targetInfo.layer_info[1];
    string sync = targetInfo.layer_info[2];
    llvm::outs() << split << sync << "\n";
    if (split == "row-wise") {
      llvm::outs() << "KernelConfig.cpp: MatmulTileConfig: row-wise partitioning\n";
      tileX = sizeM;
      tileY = sizeN/dev_num;
      tileK = sizeK;
    }
    else if (split == "col-wise") {
      llvm::outs() << "KernelConfig.cpp: MatmulTileConfig: col-wise partitioning\n";
      tileX = sizeM;
      tileY = sizeN;
      tileK = sizeK/dev_num;
    }
  }

  /*
  if ((sizeK == sizeN * 4)) {
    tileY = sizeN;
    tileK = sizeK/dev_num;
    llvm::outs() << "Second matmul of the FFN - col-wise partitioning\n";
  }*/

  // Since specialization doesn't work for K loop and peeling is not enabled yet
  // we pick a tileK size that is aligned on the K size.
  if (ShapedType::isDynamic(sizeK)) tileK = 1;
  while (sizeK % tileK != 0) {
    tileK >>= 1;
  }
  
  llvm::outs() << "KernelConfig.cpp: MatmulTileConfig: Tile size MNK : ";
  llvm::outs() << tileX << " ";
  llvm::outs() << tileY << " ";
  llvm::outs() << tileK << "\n";

  const std::array<int64_t, 3> workgroupSize{config.workgroupSize[0],
                                             config.workgroupSize[1],
                                             config.workgroupSize[2]};
  return setMatmulConfig(
      tileX, tileY, tileK, workgroupSize, softwarePipelineDepthSimt,
      IREE::Codegen::DispatchLoweringPassPipeline::PIMMatmul);
}

static LogicalResult setBatchMatmulTileConfig(func::FuncOp entryPoint,
                                       linalg::LinalgOp op,
                                       const TargetInfo &targetInfo) {
  if (!isa<linalg::BatchMatmulOp>(op))
    return failure();
  // Don't consider operations that don't have a broadcast, those should go
  // through reductions.
  if (llvm::any_of(op.getIndexingMapsArray(),
                   [](AffineMap m) { return m.isPermutation(); }))
    return failure();

  auto setBatchMatmulConfig =
      [&entryPoint, &op](int64_t tileB, int64_t tileX, int64_t tileY, int64_t tileK,
                         llvm::ArrayRef<int64_t> workgroupSize,
                         unsigned softwarePipelineDepth,
                         IREE::Codegen::DispatchLoweringPassPipeline pipeline) {
        TileSizesListType tileSizes;
        unsigned numParallelLoops = op.getNumParallelLoops();
        llvm::outs() << "numParallelLoops: " << numParallelLoops << "\n";
        SmallVector<int64_t> workgroupTileSizes(numParallelLoops - 2, tileB);
        
        /*
        llvm::outs() << "workgroupTileSizes: ";
        for (const auto& element : workgroupTileSizes) {
          llvm::outs() << element << " ";
        }
        llvm::outs() << "\n";
        */

        workgroupTileSizes.append({tileX, tileY});
        workgroupTileSizes.append(op.getNumReductionLoops(), tileK);

        SmallVector<unsigned> partitionedLoops =
            cast<PartitionableLoopsInterface>(op.getOperation())
                .getPartitionableLoops(kNumMaxParallelDims);
        llvm::SmallDenseSet<unsigned, 4> partitionedLoopsSet;
        partitionedLoopsSet.insert(partitionedLoops.begin(),
                                   partitionedLoops.end());
        for (auto loopID : llvm::seq<unsigned>(0, numParallelLoops)) {
          if (!partitionedLoopsSet.count(loopID)) {
            workgroupTileSizes[loopID] = 0;
          }
        }

        tileSizes.emplace_back(
            std::move(workgroupTileSizes));  // Workgroup level.
        return setOpConfigAndEntryPointFnTranslation(
            entryPoint, op, tileSizes, pipeline, workgroupSize,
            /*subgroupSize=*/std::nullopt, softwarePipelineDepth);
      };
  // Infer the MxN size of the matmul based on operands and indexing maps.
  auto lhsShape =
      op.getDpsInputOperand(0)->get().getType().cast<ShapedType>().getShape();
  auto rhsShape =
      op.getDpsInputOperand(1)->get().getType().cast<ShapedType>().getShape();
  
  int64_t sizeB = ShapedType::kDynamic;
  int64_t sizeM = ShapedType::kDynamic;
  int64_t sizeN = ShapedType::kDynamic;
  int64_t sizeK = ShapedType::kDynamic;
  auto outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  sizeB = lhsShape[0];
  llvm::outs() << "KernelConfig.cpp: setBatchMatmulTileConfig: sizeB: " << sizeB << "\n";
  for (unsigned i = 0; i < lhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getDpsInputOperand(0)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 2)) {
      sizeM = lhsShape[i];
      llvm::outs() << "KernelConfig.cpp: setBatchMatmulTileConfig: sizeM: " << sizeM << "\n";
      break;
    }
  }
  for (unsigned i = 0; i < rhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getDpsInputOperand(1)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 1)) {
      sizeN = rhsShape[i];
      llvm::outs() << "KernelConfig.cpp: setBatchMatmulTileConfig: sizeN: " << sizeN << "\n";
      break;
    }
  }
  SmallVector<unsigned> exprs;
  op.getReductionDims(exprs);
  if (exprs.size() == 1) {
    for (unsigned i = 0; i < lhsShape.size(); i++) {
      if (op.getMatchingIndexingMap(op.getDpsInputOperand(0))
              .getDimPosition(i) == exprs[0]) {
        sizeK = lhsShape[i];
        llvm::outs() << "KernelConfig.cpp: setBatchMatmulTileConfig: sizeK: " << sizeK << "\n";
        break;
      }
    }
  }
  bool isStaticSize = sizeM != ShapedType::kDynamic &&
                      sizeN != ShapedType::kDynamic &&
                      sizeK != ShapedType::kDynamic;

  int dev_num = targetInfo.device_num;
  // If we haven't found any config, use the best tile size hoping that
  // the workgroup specialization handles the main tile path efficiently.
  SmallVector<TileWorkgroupSizePair> tileSizeConfig;
  // Query the best configuration.
  getMatmulConfig(tileSizeConfig);
  constexpr size_t configIndex = 0;
  const TileWorkgroupSizePair &config = tileSizeConfig[configIndex];
  
  /*
  const int64_t tileX = config.tileSize[0];
  const int64_t tileY = config.tileSize[1];
  int64_t tileK = config.tileSize[2];*/
  int64_t tileB = sizeB/dev_num;
  int64_t tileX = sizeM;
  int64_t tileY = sizeN;
  int64_t tileK = sizeK;

  if (targetInfo.workload == "decoder") {
    string split = targetInfo.layer_info[1];
    string sync = targetInfo.layer_info[2];
    if (split == "head-wise") {
      llvm::outs() << "KernelConfig.cpp: setBatchMatmulTileConfig: head-wise partitioning\n";
      tileB = sizeB/dev_num;
      tileX = sizeM;
      tileY = sizeN;
      tileK = sizeK;
    }
  }

  // Since specialization doesn't work for K loop and peeling is not enabled yet
  // we pick a tileK size that is aligned on the K size.
  if (ShapedType::isDynamic(sizeK)) tileK = 1;
  while (sizeK % tileK != 0) {
    tileK >>= 1;
  }
  
  llvm::outs() << "KernelConfig.cpp: setBatchMatmulTileConfig: Tile size BMNK : ";
  llvm::outs() << tileB << " ";
  llvm::outs() << tileX << " ";
  llvm::outs() << tileY << " ";
  llvm::outs() << tileK << "\n";

  const std::array<int64_t, 3> workgroupSize{config.workgroupSize[0],
                                             config.workgroupSize[1],
                                             config.workgroupSize[2]};
  return setBatchMatmulConfig(
      tileB, tileX, tileY, tileK, workgroupSize, softwarePipelineDepthSimt,
      IREE::Codegen::DispatchLoweringPassPipeline::PIMMatmul);
}

static LogicalResult setGenericConfig(func::FuncOp entryPoint,
                                       linalg::LinalgOp op,
                                       const TargetInfo &targetInfo) {
  if (!isa<linalg::GenericOp>(op))
    return failure();

  return success();
}

// Basic default properties for linalg ops that haven't been tuned.
static LogicalResult setRootDefaultConfig(func::FuncOp entryPoint,
                                          Operation *op) {
  IREE::Codegen::DispatchLoweringPassPipeline passPipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute;
  TileSizesListType tileSizes;
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops = interfaceOp.getPartitionableLoops(std::nullopt);
  if (partitionedLoops.empty()) {
    tileSizes.push_back({});
    return setOpConfigAndEntryPointFnTranslation(entryPoint, op, tileSizes,
                                                 passPipeline, {1, 1, 1});
  }

  size_t numLoops = partitionedLoops.back() + 1;
  // To get peak occupancy we need a workgroup size of at least two warps
  std::array<int64_t, 3> workgroupSize = {2 * cudaWarpSize, 1, 1};
  unsigned vectorSize = 4;
  SmallVector<int64_t, 4> workgroupTileSizes(numLoops, 1);
  // Set all non-parallel loops to zero tile size.
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto depth : llvm::seq<int64_t>(0, numLoops)) {
    if (!partitionedLoopsSet.count(depth)) {
      workgroupTileSizes[depth] = 0;
    }
  }

  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    for (auto [index, outputOperand] :
         llvm::enumerate(genericOp.getDpsInitOperands())) {
      if (!genericOp.getMatchingIndexingMap(outputOperand)
               .isProjectedPermutation()) {
        vectorSize = 1;
        break;
      }
      ArrayRef<int64_t> shape = cast<linalg::LinalgOp>(op)
                                    .getDpsInitOperand(index)
                                    ->get()
                                    .getType()
                                    .cast<ShapedType>()
                                    .getShape();
      if (llvm::any_of(shape, ShapedType::isDynamic)) {
        vectorSize = 1;
        break;
      }
      // Since we vectorize along the most inner dimension, make sure if can be
      // dividied by number of threads * vectorSize.
      while (vectorSize > 1 &&
             shape.back() % (workgroupSize[0] * vectorSize) != 0) {
        vectorSize /= 2;
      }
      if (vectorSize == 1)  // assume there is fastpath + slowpath
        vectorSize = 4;
      int64_t problemSize = std::accumulate(
          shape.begin(), shape.end(), 1,
          [](const int64_t &a, const int64_t &b) { return a * b; });
      if ((problemSize / (cudaWarpSize * vectorSize)) < 64) {
        vectorSize = 1;
        break;
      }
    }
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  // Pick a vectorSize of 1 for op that we know won't get vectorizedd.
  // Also skip vectorization for linalg on memref (no result) as the pipeline
  // relies on tensor level tiling.
  // TODO(thomasraoux): This could be improved by checking if the linalg op
  // would fail vectorization.
  if (!linalgOp || op->getNumResults() != 1 ||
      llvm::any_of(linalgOp.getIndexingMapsArray(),
                   [](AffineMap m) { return !m.isProjectedPermutation(); })) {
    vectorSize = 1;
  } else {
    passPipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUVectorize;
  }

  // Set the inner most parallel loop to `lowerTs`.
  for (int64_t depth = numLoops; depth > 0; depth--) {
    if (partitionedLoopsSet.count(depth - 1)) {
      workgroupTileSizes[depth - 1] = workgroupSize[0] * vectorSize;
      break;
    }
  }
  if (linalgOp) {
    // Tile reduction dimension to 4 to allow doing load4 if the reduction size
    // is the most inner dimension.
    workgroupTileSizes.append(linalgOp.getNumReductionLoops(), 4);
  }
  tileSizes.emplace_back(std::move(workgroupTileSizes));  // Workgroup level
  return setOpConfigAndEntryPointFnTranslation(entryPoint, op, tileSizes,
                                               passPipeline, workgroupSize);
}

/// Return the size of the given dimension in the linalg op.
// TODO: this should be part of LinalgOp interface, the equivalent member
// function currently only support the case where all the dimensions are static
// while we want to support dynamic shapes.
static Optional<int64_t> getLinalgDimSize(linalg::LinalgOp op, int64_t d) {
  for (auto [mapIdx, map] : llvm::enumerate(op.getIndexingMapsArray())) {
    for (auto [dimIdx, dim] : llvm::enumerate(map.getResults())) {
      auto expr = dim.dyn_cast<AffineDimExpr>();
      if (expr && expr.getPosition() == d) {
        auto type = op->getOperand(mapIdx).getType().cast<ShapedType>();
        if (type.isDynamicDim(dimIdx)) return std::nullopt;
        return type.getDimSize(dimIdx);
      }
    }
  }
  return std::nullopt;
}

static bool hasTwoOrThreeLoopsInfo(linalg::LinalgOp linalgOp) {
  return linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   Operation *computeOp) {

  llvm::outs() << "KernelConfig.cpp: setRootConfig\n";

  ////////////////////////////////////////////////////////////////////////////
  llvm::outs() << "KernelConfig.cpp: setRootConfig: funcOp name: " << entryPointFn.getName() << "\n";
  string dispatch_num = extractNumberAfterDispatch(entryPointFn.getName().str());
  llvm::outs() << "KernelConfig.cpp: setRootConfig: dispatch_num: " << dispatch_num << "\n";

  Json::Value root;
  Json::Reader reader;
  ifstream json("exe_map.json", ifstream::binary);
  reader.parse(json, root);
  string workload;
  std::vector<string> layer_info;
  if (root["workload"].asString() == "decoder") {
    // llvm::outs() << "is decoder\n";
    workload = "decoder";
    layer_info = getDecoderLayerInfo(root, dispatch_num);
  }
  ////////////////////////////////////////////////////////////////////////////

  TargetInfo targetInfo = getTargetInfo(entryPointFn, workload, layer_info);
  if (IREE::Codegen::CompilationInfoAttr compilationInfo =
          getCompilationInfo(computeOp)) {
    // If the op already has a lowering config coming from the IR use this and
    // bypass the heuristic.
    // llvm::outs() << "compilationInfo\n";
    return setUserConfig(entryPointFn, computeOp, compilationInfo);
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {

    if (succeeded(setMatmulTileConfig(entryPointFn, linalgOp, targetInfo))) {
      llvm::outs() << "KernelConfig.cpp: setMatmulTileConfig\n";
      return success();
    }
    if (succeeded(setBatchMatmulTileConfig(entryPointFn, linalgOp, targetInfo))) {
      llvm::outs() << "KernelConfig.cpp: setBatchMatmulTileConfig\n";
      return success();
    }
    if (succeeded(setGenericConfig(entryPointFn, linalgOp, targetInfo))) {
      llvm::outs() << "KernelConfig.cpp: setGenericConfig\n";
      return success();
    }
    /*
    auto genericOp = dyn_cast<linalg::GenericOp>(computeOp);
    if (genericOp && succeeded(setTransposeConfig(entryPointFn, genericOp))) {
      llvm::outs() << "setTransposeConfig\n";
      return success();
    }*/
  }
  llvm::outs() << "KernelConfig.cpp: setRootDefaultConfig\n";
  return setRootDefaultConfig(entryPointFn, computeOp);
}

namespace mlir {
namespace iree_compiler {

LogicalResult initPIMLaunchConfig(ModuleOp moduleOp) {
  llvm::outs() << "initPIMLaunchConfig\n";
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);

  int serial_mm_cnt = 0;
  bool is_matmul = false;

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp) continue;
    if (getTranslationInfo(exportOp)) continue;
    SmallVector<Operation *> computeOps;
    if (failed(getComputeOps(funcOp, computeOps))) {
      return funcOp.emitOpError("failed to get compute ops");
    }

    Operation *rootOperation = nullptr;
    // Find the root operation. linalg.generic and linalg.fill are not root
    // operations if there are other compute operations present.
    for (Operation *op : llvm::reverse(computeOps)) {
      if (!isa<linalg::GenericOp, linalg::FillOp>(op)) {
        rootOperation = op;
        break;
      }
      if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        // linalg.generic with `reduction` iterator types are roots as well.
        if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
          rootOperation = op;
          break;
        }
      }
    }

    if (!rootOperation) {
      for (Operation *op : llvm::reverse(computeOps)) {
        if (isa<linalg::GenericOp, linalg::FillOp>(op)) {
          rootOperation = op;
          break;
        }
      }
    }

    if (!rootOperation) {
      // No root operation found. Allow it to pass through without a config.
      continue;
    }

    if (failed(setRootConfig(funcOp, rootOperation))) continue;

    // Propogate the configuration to the other ops.
    // TODO(ravishankarm, thomasraoux): This is a very specific use (and
    // fragile). In general, this should not be needed. Things are already tiled
    // and distributed. The rest of the compilation must be structured to either
    // use `TileAndFuse` or they are independent configurations that are
    // determined based on the op.
    if (IREE::Codegen::LoweringConfigAttr config =
            getLoweringConfig(rootOperation)) {
      for (auto op : computeOps) {
        if (op == rootOperation) continue;
        setLoweringConfig(op, config);
      }
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
