/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/matmul_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class MatMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of MatMulOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of MatMulOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MatMulOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");
    auto dim_y = ctx->GetInputDim("Y");
    bool transpose_x = ctx->Attrs().Get<bool>("transposeX");
    bool transpose_y = ctx->Attrs().Get<bool>("transposeY");

    PADDLE_ENFORCE_GE(dim_x.size(), 1,
                      "Input tensor X must be at least 1-dimensional.");
    PADDLE_ENFORCE_GE(dim_y.size(), 1,
                      "Input tensor Y must be at least 1-dimensional.");
    PADDLE_ENFORCE_LE(dim_x.size(), 3,
                      "Input tensor X must be at most 3-dimensional.");
    PADDLE_ENFORCE_LE(dim_y.size(), 3,
                      "Input tensor Y must be at most 3-dimensional.");

    int M = 0, N = 0, KX = 0, KY = 0, batchCountX = 0, batchCountY = 0;
    bool remove_initial_dim = false, remove_final_dim = false;

    switch (dim_x.size()) {
      case 1:
        if (transpose_x) {
          M = dim_x[0];
          KX = 1;
        } else {
          M = 1;
          KX = dim_x[0];
          remove_initial_dim = true;
        }
        break;
      case 2:
        M = transpose_x ? dim_x[1] : dim_x[0];
        KX = transpose_x ? dim_x[0] : dim_x[1];
        break;
      case 3:
        batchCountX = dim_x[0];
        M = transpose_x ? dim_x[2] : dim_x[1];
        KX = transpose_x ? dim_x[1] : dim_x[2];
        break;
      default:
        assert(false);
    }

    switch (dim_y.size()) {
      case 1:
        if (transpose_y) {
          N = dim_y[0];
          KY = 1;
        } else {
          N = 1;
          KY = dim_y[0];
          remove_final_dim = true;
        }
        break;
      case 2:
        KY = transpose_y ? dim_y[1] : dim_y[0];
        N = transpose_y ? dim_y[0] : dim_y[1];
        break;
      case 3:
        batchCountY = dim_y[0];
        KY = transpose_y ? dim_y[2] : dim_y[1];
        N = transpose_y ? dim_y[1] : dim_y[2];
        break;
      default:
        assert(false);
    }

    PADDLE_ENFORCE_EQ(
        KX, KY,
        "First matrix's width must be equal with second matrix's height.");
    if (batchCountX && batchCountY) {
      PADDLE_ENFORCE_EQ(
          batchCountX, batchCountY,
          "Input(X) and Input(Y) must have same batch dimension.");
    }
    int batchCount = std::max(batchCountX, batchCountY);

    std::vector<int64_t> dim_out;
    if (batchCount) {
      dim_out.push_back(batchCount);
    }
    if (!remove_initial_dim) {
      dim_out.push_back(M);
    }
    if (!remove_final_dim) {
      dim_out.push_back(N);
    }
    ctx->SetOutputDim("Out", framework::make_ddim(dim_out));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class MatMulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MatMulOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of MatMul op");
    AddInput("Y", "The second input of MatMul op");
    AddOutput("Out", "The output of MatMul op");
    AddAttr<bool>("transposeX",
                  R"DOC(If true, use the transpose of X.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>("transposeY",
                  R"DOC(If true, use the transpose of Y.
        )DOC")
        .SetDefault(false);
    AddComment(R"DOC(
The MatMul operator is used to perform (batched) matrix multiplication for
input tensors X and Y. The behavior is similar to the `numpy.matmul` function.

Both the input `X` and `Y` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD with input `X`.
)DOC");
  }
};

class MatMulOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    auto x_mat_dims =
        framework::flatten_to_2d(x_dims, Attr<int>("x_num_col_dims"));
    auto y_mat_dims =
        framework::flatten_to_2d(y_dims, Attr<int>("y_num_col_dims"));

    PADDLE_ENFORCE_EQ(
        x_mat_dims[0], out_dims[0],
        "The first dimension of Out@GRAD must equal to the first dimension of "
        "the first operand.");
    PADDLE_ENFORCE_EQ(
        y_mat_dims[1], out_dims[1],
        "The second dimension of Out@GRAD must equal to the second "
        "dimension of the second operand.");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(matmul, ops::MatMulOp, ops::MatMulOpMaker, matmul_grad,
            ops::MatMulOpGrad);
REGISTER_OP_CPU_KERNEL(matmul,
                       ops::MatMulKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    matmul_grad, ops::MatMulGradKernel<paddle::platform::CPUPlace, float>);
