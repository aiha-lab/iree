#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "json/json.h"

#include <fstream>
#include <iostream>
#include <string>

namespace mlir {

namespace iree_compiler {
namespace IREE {
namespace Flow {
  
void GenerateMetaData(FunctionOpInterface funcOp)
{
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> op_pattern;

    const std::vector<std::string> gpt_decoder_pattern
        = {"arith.addf", "linalg.generic", "flow.dispatch.region", // layernorm
        "arith.divf", "arith.subf", "linalg.generic", "flow.dispatch.region", // layernorm
        "arith.mulf", "arith.addf", "linalg.generic", "flow.dispatch.region", // layernorm
        "arith.divf", "arith.addf", "math.rsqrt", "arith.mulf", "linalg.generic", "flow.dispatch.region", // layernorm
        "arith.mulf", "arith.addf", "linalg.matmul", "arith.addf", "linalg.generic", "flow.dispatch.region", // c_attn
        "arith.mulf", "arith.addf", "linalg.batch_matmul", "flow.dispatch.region", // QK
        "arith.maxf", "linalg.generic", "arith.subf", "linalg.generic", "flow.dispatch.region", // softmax
        "arith.addf", "linalg.generic", "arith.divf", "linalg.generic", "flow.dispatch.region", // softmax
        "arith.mulf", "arith.addf", "linalg.batch_matmul", "flow.dispatch.region", // SV
        "arith.mulf", "arith.addf", "linalg.matmul", "arith.addf", "arith.addf", "linalg.generic", "flow.dispatch.region", // c_proj+shortcut
        "arith.addf", "linalg.generic", "flow.dispatch.region", // layernorm
        "arith.divf", "arith.subf", "linalg.generic", "flow.dispatch.region", // layernorm
        "arith.mulf", "arith.addf", "linalg.generic", "flow.dispatch.region", // layernorm
        "arith.divf", "arith.addf", "math.rsqrt", "arith.mulf", "linalg.generic", "flow.dispatch.region", // layernorm
        "arith.mulf", "arith.addf", "linalg.matmul", "arith.addf", "linalg.generic", "flow.dispatch.region", // fc1
        "arith.mulf", "arith.addf", "linalg.matmul", "arith.addf", "arith.addf", "linalg.generic", "flow.dispatch.region" // fc2+shortcut
    };

    const std::vector<std::string> layernorm_pattern
        = {"arith.addf", "linalg.generic", "flow.dispatch.region",
        "arith.divf", "arith.subf", "linalg.generic", "flow.dispatch.region",
        "arith.mulf", "arith.addf", "linalg.generic", "flow.dispatch.region",
        "arith.divf", "arith.addf", "math.rsqrt", "arith.mulf", "linalg.generic", "flow.dispatch.region"
    };

    const std::vector<std::string> matmul_pattern
        = {"arith.mulf", "arith.addf", "linalg.matmul", "arith.addf", "linalg.generic", "flow.dispatch.region"
    };

    const std::vector<std::string> qk_pattern
        = {"arith.mulf", "arith.addf", "linalg.batch_matmul", "flow.dispatch.region"};
    
    const std::vector<std::string> softmax_pattern
        = {"arith.maxf", "linalg.generic", "arith.subf", "linalg.generic", "flow.dispatch.region",
        "arith.addf", "linalg.generic", "arith.divf", "linalg.generic", "flow.dispatch.region"};

    const std::vector<std::string> sv_pattern
        = {"arith.mulf", "arith.addf", "linalg.batch_matmul", "flow.dispatch.region"};
    
    const std::vector<std::string> matmul_shortcut_pattern
        = {"arith.mulf", "arith.addf", "linalg.matmul", "arith.addf", "arith.addf", "linalg.generic", "flow.dispatch.region"};


    // std::string file_path = "pattern.txt";
    // std::ofstream file(file_path/*, std::ios::app*/);

    // funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());

    funcOp.walk([&](Operation *op) {
        // Check if the file is open
        // if (file.is_open()) {
        // Write the string to the file
        if(isa<linalg::LinalgOp>(op) && !isa<linalg::FillOp/*, linalg::GenericOp*/>(op))
            // file << op->getName().getStringRef().str() <<std::endl;
            op_pattern.push_back(op->getName().getStringRef().str());
        else if(isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
                arith::DivFOp, arith::DivSIOp, arith::DivUIOp, arith::NegFOp,
                arith::TruncFOp, arith::TruncIOp, arith::ExtFOp, arith::ExtSIOp,
                arith::ExtUIOp, arith::FPToSIOp, arith::FPToUIOp, arith::SIToFPOp,
                arith::UIToFPOp, arith::MulFOp, arith::MaxFOp, math::RsqrtOp>(op))
            // file << op->getName().getStringRef().str() <<std::endl;
            op_pattern.push_back(op->getName().getStringRef().str());
        else if(isa<IREE::Flow::DispatchRegionOp>(op))
            // file << op->getName().getStringRef().str() <<std::endl;
            op_pattern.push_back(op->getName().getStringRef().str());

        // std::cout << "File saved successfully." << std::endl;
        // }
    });

    int matmul_cnt = 0;
    int batch_matmul_cnt = 0;
    int d_model, n_head, d_head, token;
    if (op_pattern == gpt_decoder_pattern) {
        Json::Value root;
        Json::Value input, weight, weight_bm, bias;
        root["workload"] = "decoder";
        root["n_layer"] = "12";
        llvm::outs() << "Decoder pattern matched!\n";
        funcOp.walk([&](Operation *op_) {
        // if(isa<linalg::MatmulOp>(op_)) {
        if(auto op = dyn_cast<linalg::MatmulOp>(op_)) {
            // Get tensor shape
            mlir::Value inputTensor_l = op.getDpsInputOperands()[0]->get();
            mlir::ShapedType inputType_l = inputTensor_l.getType().cast<mlir::ShapedType>();
            mlir::ArrayRef<int64_t> shape_l = inputType_l.getShape();
            mlir::Value inputTensor_r = op.getDpsInputOperands()[1]->get();
            mlir::ShapedType inputType_r = inputTensor_r.getType().cast<mlir::ShapedType>();
            mlir::ArrayRef<int64_t> shape_r = inputType_r.getShape();
            /*
            for (int64_t dim : shape_l) {
            llvm::outs() << dim << " ";
            }
            for (int64_t dim : shape_r) {
            llvm::outs() << dim << " ";
            }*/ 
            switch (matmul_cnt) {
            case 0:
                input["shape"][0] = shape_l[0];
                input["shape"][1] = shape_l[1];
                input["rank"] = 2;
                root["1"] = (input);
                d_model = shape_l[1];

                weight["shape"][0] = shape_r[0];
                weight["shape"][1] = shape_r[1];
                weight["rank"] = 2;
                root["2"] = (weight);

                bias["shape"][0] = 1;
                bias["shape"][1] = shape_r[1];
                bias["rank"] = 2;
                root["3"] = (bias);
                break;
            
            case 1:
                weight["shape"][0] = shape_r[0];
                weight["shape"][1] = shape_r[1];
                weight["rank"] = 2;
                root["6"] = (weight);

                bias["shape"][0] = 1;
                bias["shape"][1] = shape_r[1];
                bias["rank"] = 2;
                root["7"] = (bias);
                break;

            case 2:
                weight["shape"][0] = shape_r[0];
                weight["shape"][1] = shape_r[1];
                weight["rank"] = 2;
                root["8"] = (weight);

                bias["shape"][0] = 1;
                bias["shape"][1] = shape_r[1];
                bias["rank"] = 2;
                root["9"] = (bias);
                break;
            case 3:
                weight["shape"][0] = shape_r[0];
                weight["shape"][1] = shape_r[1];
                weight["rank"] = 2;
                root["10"] = (weight);

                bias["shape"][0] = 1;
                bias["shape"][1] = shape_r[1];
                bias["rank"] = 2;
                root["11"] = (bias);
                break;
            }
            matmul_cnt++;
        }

        if(auto op = dyn_cast<linalg::BatchMatmulOp>(op_)) {
            // Get tensor shape
            mlir::Value inputTensor_l = op.getDpsInputOperands()[0]->get();
            mlir::ShapedType inputType_l = inputTensor_l.getType().cast<mlir::ShapedType>();
            mlir::ArrayRef<int64_t> shape_l = inputType_l.getShape();
            mlir::Value inputTensor_r = op.getDpsInputOperands()[1]->get();
            mlir::ShapedType inputType_r = inputTensor_r.getType().cast<mlir::ShapedType>();
            mlir::ArrayRef<int64_t> shape_r = inputType_r.getShape();
            /*
            for (int64_t dim : shape_l) {
            llvm::outs() << dim << " ";
            }
            for (int64_t dim : shape_r) {
            llvm::outs() << dim << " ";
            }*/ 
            switch (batch_matmul_cnt) {
            case 0:
                weight_bm["shape"][0] = shape_r[0];
                weight_bm["shape"][1] = shape_r[1];
                weight_bm["shape"][2] = shape_r[2];
                weight_bm["rank"] = 3;
                root["4"] = (weight_bm);
                n_head = shape_r[0];
                d_head = shape_r[1];
                token = shape_r[2];
                break;
            
            case 1:
                weight_bm["shape"][0] = shape_r[0];
                weight_bm["shape"][1] = shape_r[1];
                weight_bm["shape"][2] = shape_r[2];
                weight_bm["rank"] = 3;
                root["5"] = (weight_bm);
                break;
            }
            batch_matmul_cnt++;
        }

        }); // funcOp.walk
        std::ofstream outFile("meta_data.json", std::ios::out);
        outFile << root;
        outFile.close();

        // Executable json
        Json::Value root_exe;
        Json::Value exe;
        Json::Value config;
        root_exe["workload"] = "decoder";
        root_exe["num_device"] = "1";
        config["d_model"] = d_model;
        config["n_head"] = n_head;
        config["d_head"] = d_head;
        config["token"] = token;
        root_exe["config"] = config;

        int num_exe = 16;
        for (int i=0; i<num_exe; i++) {
        if (i>=0 && i<3) {
            exe["layer"] = "layernorm1-empty";
            exe["partioning"] = "copy";
            exe["sync"] = "none";
        }
        if (i==3) {
            exe["layer"] = "layernorm1";
            exe["partioning"] = "copy";
            exe["sync"] = "none";
        }
        if (i>9 && i<13) {
            exe["layer"] = "layernorm2-empty";
            exe["partioning"] = "copy";
            exe["sync"] = "none";
        }
        if (i==13) {
            exe["layer"] = "layernorm2";
            exe["partioning"] = "copy";
            exe["sync"] = "none";
        }
        else if (i==4) {
            exe["layer"] = "c_attn";
            exe["partioning"] = "row-wise";
            exe["sync"] = "none";
        }
        else if (i==5) {
            exe["layer"] = "qk";
            exe["partioning"] = "head-wise";
            exe["sync"] = "none";
        }
        else if ((i==6)) {
            exe["layer"] = "softmax-empty";
            exe["partioning"] = "head-wise";
            exe["sync"] = "none";
        }
        else if ((i==7)) {
            exe["layer"] = "softmax";
            exe["partioning"] = "head-wise";
            exe["sync"] = "none";
        }
        else if (i==8) {
            exe["layer"] = "sv";
            exe["partioning"] = "head-wise";
            exe["sync"] = "none";
        }
        else if (i==9) {
            exe["layer"] = "c_proj+residual";
            exe["partioning"] = "col-wise";
            exe["sync"] = "reduce";
        }
        else if (i==14) {
            exe["layer"] = "fc1";
            exe["partioning"] = "row-wise";
            exe["sync"] = "none";
        }
        else if (i==15) {
            exe["layer"] = "fc2+residual";
            exe["partioning"] = "col-wise";
            exe["sync"] = "reduce";
        }
        root_exe[std::to_string(i)] = (exe);
        }
        std::ofstream outFile2("exe_map.json", std::ios::out);
        outFile2 << root_exe;
        outFile2.close();

    } // if decoder pattern

    else {
        Json::Value root;
        root["workload"] = "normal";
        llvm::outs() << "Normal graph pattern\n";
        funcOp.walk([&](Operation *op_) {
        // if(isa<linalg::MatmulOp>(op_)) {
        if(auto op = dyn_cast<linalg::MatmulOp>(op_)) {
            // Get tensor shape
            mlir::Value inputTensor_l = op.getDpsInputOperands()[0]->get();
            mlir::ShapedType inputType_l = inputTensor_l.getType().cast<mlir::ShapedType>();
            mlir::ArrayRef<int64_t> shape_l = inputType_l.getShape();
            mlir::Value inputTensor_r = op.getDpsInputOperands()[1]->get();
            mlir::ShapedType inputType_r = inputTensor_r.getType().cast<mlir::ShapedType>();
            mlir::ArrayRef<int64_t> shape_r = inputType_r.getShape();
            switch (matmul_cnt) {
            case 0:
                root["0"] = "gather";
                break;
            case 1:
                root["1"] = "reduce";
                break;
            }
            matmul_cnt++;
        }
        }); // funcOp.walk
        std::ofstream outFile("exe_map.json", std::ios::out);
        outFile << root;
        outFile.close();
    } // else (decoder pattern)

    // Close the file
    // file.close();
    }

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir