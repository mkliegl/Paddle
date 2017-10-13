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

template <typename Place, typename T>
class MatMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* y = context.Input<Tensor>("Y");
    Tensor* out = context.Output<Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());
    bool transpose_x = context.Attr<bool>("transposeX");
    bool transpose_y = context.Attr<bool>("transposeY");

    math::MatMulFunctor<Place, T>()(context.device_context(), *x, transpose_x,
                                    *y, transpose_y, T(1), out, T(0));
  }
};

// Based on dimensional compatibility for matrix multiplication, it is
// straight-forward to check the following table:
//
// transposeX | False    | True     | False    | True
// transposeY | False    | False    | True     | True
// -----------+----------+----------+----------+-----------
//       dX = | dOut Y^T | Y dOut^T | dOut Y   | Y^T dOut^T
//       dY = | X^T dOut | X dOut   | dOut^T X | dOut^T X^T
template <typename Place, typename T>
class MatMulGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* y = context.Input<Tensor>("Y");
    const Tensor* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* dx = context.Output<Tensor>(framework::GradVarName("X"));
    Tensor* dy = context.Output<Tensor>(framework::GradVarName("Y"));
    bool transpose_x = context.Attr<bool>("transposeX");
    bool transpose_y = context.Attr<bool>("transposeY");

    if (dx) {
      dx->mutable_data<T>(context.GetPlace());
      if (transpose_x) {
        math::MatMulFunctor<Place, T>()(context.device_context(), *y,
                                        transpose_y, *dout, transpose_x, T(1),
                                        dx, T(0));
      } else {
        math::MatMulFunctor<Place, T>()(context.device_context(), *dout,
                                        transpose_x, *y, !transpose_y, T(1), dx,
                                        T(0));
      }
    }

    if (dy) {
      dy->mutable_data<T>(context.GetPlace());
      if (transpose_y) {
        math::MatMulFunctor<Place, T>()(context.device_context(), *dout,
                                        transpose_y, *x, transpose_x, T(1), dy,
                                        T(0));
      } else {
        math::MatMulFunctor<Place, T>()(context.device_context(), *x,
                                        !transpose_x, *dout, transpose_y, T(1),
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
