/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/math_unary_elementwise_func.h"

namespace oneflow {

template<template<typename> class UnaryFunctor, typename T>
class MathUnaryElementwiseCpuKernel final : public user_op::OpKernel {
 public:
  MathUnaryElementwiseCpuKernel() = default;
  ~MathUnaryElementwiseCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* x = tensor_x->dptr<T>();
    T* y = tensor_y->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    // compute is_contiguous and construct input/output stride params
    const int32_t ndim = tensor_x->shape().NumAxes();
    const StrideVector& in_stride_vec = tensor_x->stride().StrideVec();
    const StrideVector& out_stride_vec = tensor_y->stride().StrideVec();
    DimVector in_shape_vec;
    tensor_x->shape().ToDimVector(&in_shape_vec);
    bool is_contiguous = oneflow::one::IsContiguous(in_shape_vec, in_stride_vec);
    StrideParam param_in_stride(in_stride_vec.data(), ndim),
        param_out_stride(out_stride_vec.data(), ndim);
    if (is_contiguous) {
      for (int32_t i = 0; i < n; ++i) { y[i] = UnaryFunctor<T>::Forward(x[i]); }
    } else {
      for (int32_t i = 0; i < n; ++i) {
        int32_t src_idx = compute_index(i, param_in_stride, param_out_stride);
        y[i] = UnaryFunctor<T>::Forward(x[src_idx]);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class UnaryFunctor, typename T>
class MathUnaryElementwiseGradCpuKernel final : public user_op::OpKernel {
 public:
  MathUnaryElementwiseGradCpuKernel() = default;
  ~MathUnaryElementwiseGradCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const T* x = tensor_x->dptr<T>();
    const T* dy = tensor_dy->dptr<T>();
    T* dx = tensor_dx->mut_dptr<T>();
    int64_t n = tensor_x->shape().elem_cnt();
    CHECK_LE(n, GetMaxVal<int32_t>() / 2);
    for (int32_t i = 0; i < n; ++i) { dx[i] = UnaryFunctor<T>::Backward(x[i], dy[i]); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MATH_UNARY_ELEMENTWISE_CPU_KERNEL_AND_GRAD(math_type_pair, data_type_pair)        \
  REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(math_type_pair))                                           \
      .SetCreateFn<                                                                                \
          MathUnaryElementwiseCpuKernel<OF_PP_CAT(OF_PP_PAIR_SECOND(math_type_pair), Functor),     \
                                        OF_PP_PAIR_FIRST(data_type_pair)>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                              \
                       && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(data_type_pair))      \
                       && (user_op::HobDataType("y", 0) == OF_PP_PAIR_SECOND(data_type_pair)));    \
                                                                                                   \
  REGISTER_USER_KERNEL((std::string("") + OF_PP_PAIR_FIRST(math_type_pair) + "_grad"))             \
      .SetCreateFn<                                                                                \
          MathUnaryElementwiseGradCpuKernel<OF_PP_CAT(OF_PP_PAIR_SECOND(math_type_pair), Functor), \
                                            OF_PP_PAIR_FIRST(data_type_pair)>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                              \
                       && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(data_type_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_CPU_KERNEL_AND_GRAD,
                                 MATH_UNARY_ELEMENTWISE_FUNC_SEQ, FLOATING_DATA_TYPE_SEQ)

// For some special dtype kernel register.
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_CPU_KERNEL_AND_GRAD,
                                 OF_PP_MAKE_TUPLE_SEQ("abs", Abs), UNSIGNED_INT_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_UNARY_ELEMENTWISE_CPU_KERNEL_AND_GRAD,
                                 OF_PP_MAKE_TUPLE_SEQ("abs", Abs), INT_DATA_TYPE_SEQ)

}  // namespace oneflow
