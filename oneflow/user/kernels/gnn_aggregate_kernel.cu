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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GnnAggregateForwardGpu(const T* w_self, const T* h_k, const T* b, T* y, int64_t N) {
  // Compute each thread's global row and column index
  int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if((row >= N) || (col >= N)){}
  else{
      // Iterate over row, and down column
    y[row * N + col] = 0;
    for (int k = 0; k < N; k++) {
      // Accumulate results for a single element
      y[row * N + col] += w_self[row * N + k] * h_k[k * N + col];
    }
    y[row * N + col] += b[col]; 
  }
}

template<typename T>
__global__ void GnnAggregateForwardGpuSin(const T* w_self, const T* h_k, const T* b, T* y, int64_t N) {
  // Compute each thread's global row and column index
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  int64_t row = thread_id / N;
  int64_t col = thread_id % N;

  // Iterate over row, and down column
  y[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    // Accumulate results for a single element
    y[row * N + col] += w_self[row * N + k] * h_k[k * N + col];
  }
  y[row * N + col] += b[col]; 
  
}

}  // namespace

template<typename T>
class GnnAggregateKernel final : public user_op::OpKernel {
 public:
  GnnAggregateKernel() = default;
  ~GnnAggregateKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* w_self = ctx->Tensor4ArgNameAndIndex("w_self", 0);
    CHECK_EQ(w_self->shape_view().NumAxes(), 2) << "w_self Numdims should be equal to 2. ";
    const DataType data_type = w_self->data_type();

    const user_op::Tensor* h_k = ctx->Tensor4ArgNameAndIndex("h_k", 0);
    CHECK_EQ(h_k->shape_view().NumAxes(), 2) << "h_k Numdims should be equal to 2. ";
    CHECK_EQ(h_k->data_type(), data_type) << "Matrix w_self Datatype should be equal to Matrix h_k";

    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    CHECK_EQ(b->shape_view().NumAxes(), 1) << "B Numdims should be equal to 1. ";
    CHECK_EQ(b->data_type(), data_type) << "Matrix w_self Datatype should be equal to Vector b";

    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    CHECK_EQ(y->shape_view().NumAxes(), 2) << "y Numdims should be equal to 2. ";
    CHECK_EQ(y->data_type(), data_type) << "y Datatype should be equal to input's. ";

    //std::cout << "int32_t elem_cnt = (w_self->shape_view().At(0)) * (w_self->shape_view().At(1)) = " << elem_cnt << std::endl;
    
    int64_t N = w_self->shape_view().At(0);

    std::cout << "int N = w_self->shape_view().At(0) = " << N << std::endl;

    //使用二维索引
    /* int THREADS = 32; 

    int BLOCKS = (N + THREADS - 1) / THREADS;
    
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);
   
    GnnAggregateForwardGpu<T><<<blocks, threads>>>(w_self->dptr<T>(), h_k->dptr<T>(), b->dptr<T>(), y->mut_dptr<T>(), N); */

    //RUN_CUDA_KERNEL((GnnAggregateForwardGpu<T>), ctx->stream(), elem_cnt,
    //w_self->dptr<T>(), h_k->dptr<T>(), b->dptr<T>(), y->mut_dptr<T>(), N);

    //使用一维索引
    const int32_t elem_cnt = (w_self->shape_view().At(0)) * (w_self->shape_view().At(1));
    int threads = 256;
    int blocks = (elem_cnt + threads - 1) / threads;
    GnnAggregateForwardGpuSin<T><<<blocks, threads>>>(w_self->dptr<T>(), h_k->dptr<T>(), b->dptr<T>(), y->mut_dptr<T>(), N);

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GNN_AGGREGATE_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("gnn_aggregate")                               \
      .SetCreateFn<GnnAggregateKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GNN_AGGREGATE_KERNEL(float)
REGISTER_GNN_AGGREGATE_KERNEL(double)
REGISTER_GNN_AGGREGATE_KERNEL(uint8_t)
REGISTER_GNN_AGGREGATE_KERNEL(int8_t)
REGISTER_GNN_AGGREGATE_KERNEL(int32_t)
REGISTER_GNN_AGGREGATE_KERNEL(int64_t)

}  // namespace oneflow
