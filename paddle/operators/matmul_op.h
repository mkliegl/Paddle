/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   You may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace matmul_detail {

using Tensor = framework::Tensor;
using DDim = framework::DDim;
using framework::make_ddim;
using framework::vectorize;

template <typename Place, typename T>
class MatMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor& x = *context.Input<Tensor>("X");
    const Tensor& y = *context.Input<Tensor>("Y");
    Tensor* out = context.Output<Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());
    bool transpose_x = context.Attr<bool>("transposeX");
    bool transpose_y = context.Attr<bool>("transposeY");

    math::MatMulFunctor<Place, T>()(context.device_context(), x, transpose_x, y,
                                    transpose_y, T(1), out, T(0));
  }
};

template <typename T>
inline Tensor Reshape(const Tensor& input, const DDim& dims) {
  Tensor output;
  output.ShareDataWith<T>(input);
  output.Resize(dims);
  return output;
}

// Using dimensional constraints on matrix multiplication, it is
// straight-forward to check the following table for when X and Y
// are both matrices.
//
// transposeX | False    | True     | False    | True
// transposeY | False    | False    | True     | True
// -----------+----------+----------+----------+-----------
//       dX = | dOut Y^T | Y dOut^T | dOut Y   | Y^T dOut^T
//       dY = | X^T dOut | X dOut   | dOut^T X | dOut^T X^T
//
// When X is a vector of size K, we treat it instead as a matrix of shape
// (1, K). Similarly, when Y is a vector of size K, we treat it instead as
// a matrix of shape (K, 1).
//
// When X and Y are both 3-dimensional tensors, then the first dimension
// the batch dimension can be ignored and the exact same formulas apply
// as for two matrices.
//
// Finally, when X ia 3-dimensional tensor but Y is a matrix, we end up
// with formulas like
//
//   dY_{ij} = \sum_{p, m} X_{pmi} dOut_{pmj}
//
// To handle this sort of scenario, we reshape X : P x M x K, dOut: P x M x N
// to X: (P * M) x K, dOut: (P * M) x N.
template <typename Place, typename T>
class MatMulGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor& x = *context.Input<Tensor>("X");
    const Tensor& y = *context.Input<Tensor>("Y");
    const Tensor& dout = *context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* dx = context.Output<Tensor>(framework::GradVarName("X"));
    Tensor* dy = context.Output<Tensor>(framework::GradVarName("Y"));
    bool transpose_x = context.Attr<bool>("transposeX");
    bool transpose_y = context.Attr<bool>("transposeY");

    std::vector<int64_t> x_dims = vectorize(x.dims());
    std::vector<int64_t> y_dims = vectorize(y.dims());

    // If X is a vector, reshape it (and possibly dOut) to a matrix.
    if (x_dims.size() == 1) {
      x_dims.insert(x_dims.begin(), 1);
    }

    // If Y is a vector, reshape it (and possibly dOut) to a matrix.
    if (y_dims.size() == 1) {
      y_dims.push_back(1);
    }

    // Fix the dOut dimensions.
    int M = 0, N = 0, batchCountX = 0, batchCountY = 0;

    switch (x_dims.size()) {
      case 2:
        M = transpose_x ? x_dims[1] : x_dims[0];
        break;
      case 3:
        batchCountX = x_dims[0];
        M = transpose_x ? x_dims[2] : x_dims[1];
        break;
      default:
        assert(false);
    }

    switch (y_dims.size()) {
      case 2:
        N = transpose_y ? y_dims[0] : y_dims[1];
        break;
      case 3:
        batchCountY = y_dims[0];
        N = transpose_y ? y_dims[1] : y_dims[2];
        break;
      default:
        assert(false);
    }
    if (batchCountX && batchCountY) {
      PADDLE_ENFORCE_EQ(
          batchCountX, batchCountY,
          "When Input(X) and Input(Y) are both three dimensional, they "
          "must have the same batch dimension.");
    }
    int batchCount = std::max(batchCountX, batchCountY);
    std::vector<int64_t> dout_dims = {M, N};
    if (batchCount) {
      dout_dims.insert(dout_dims.begin(), batchCount);
    }
    const Tensor& X = Reshape<T>(x, make_ddim(x_dims));
    const Tensor& Y = Reshape<T>(y, make_ddim(y_dims));
    const Tensor& dOut = Reshape<T>(dout, make_ddim(dout_dims));

    if (dx) {
      dx->mutable_data<T>(context.GetPlace());
      if (transpose_x) {
        math::MatMulFunctor<Place, T>()(context.device_context(), Y,
                                        transpose_y, dOut, transpose_x, T(1),
                                        dx, T(0));
      } else {
        math::MatMulFunctor<Place, T>()(context.device_context(), dOut,
                                        transpose_x, Y, !transpose_y, T(1), dx,
                                        T(0));
      }
    }

    if (dy) {
      dy->mutable_data<T>(context.GetPlace());
      if (transpose_y) {
        math::MatMulFunctor<Place, T>()(context.device_context(), dOut,
                                        transpose_y, X, transpose_x, T(1), dy,
                                        T(0));
      } else {
        math::MatMulFunctor<Place, T>()(context.device_context(), X,
                                        !transpose_x, dOut, transpose_y, T(1),
                                        dy, T(0));
      }
    }
  }
};
}  // namespace matmul_detail

using matmul_detail::MatMulKernel;
using matmul_detail::MatMulGradKernel;

}  // namespace operators
}  // namespace paddle
