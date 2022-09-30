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
#ifndef ONEFLOW_CORE_PRIMITIVE_COMMON_CONSTANT_PAD_H_
#define ONEFLOW_CORE_PRIMITIVE_COMMON_CONSTANT_PAD_H_

#include "oneflow/core/ep/include/primitive/primitive.h"
// #include "oneflow/core/ep/include/primitive/offset_to_index_calculator.h"
#include "oneflow/core/ep/include/primitive/fast_div.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace ep {

namespace primitive {

namespace {

constexpr int32_t kMaxNumDims = 8;

constexpr int32_t Min(int32_t a, int32_t b) { return a < b ? a : b; }
constexpr int32_t kMaxPackBytes = 128 / 8;

template<typename T>
constexpr int32_t GetMaxPackSize() {
  return Min(kMaxPackBytes / sizeof(T), 8);
}

template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template<typename T, size_t pack_size>
union Pack {
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
  explicit OF_DEVICE_FUNC Pack(T value) {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < pack_size; i++) { elem[i] = value; }
  }
  T elem[pack_size];
  PackType<T, pack_size> storage;
};

template<typename T>
T GetValue(Scalar value) {
  return value.Value<T>();
}

template<size_t num_dims, typename IndexType>
struct ConstantPadParams {
  IndexType src_strides[num_dims];
  IndexType dst_strides[num_dims];
  FastDivide<IndexType> fast_dividers[num_dims];
  IndexType valid_start[num_dims];
  IndexType valid_end[num_dims];
  IndexType elem_cnt{};
  const void* src{};
  void* dst{};
};

template<size_t max_pack_size>
size_t GetLaunchPackSize(size_t num_dims, void* dst, const int64_t* dst_dims, const void* src,
                         const int64_t* src_dims, const int64_t* padding_before,
                         const int64_t* padding_after) {
  static_assert(max_pack_size > 0 && (max_pack_size & (max_pack_size - 1)) == 0, "");
  const int64_t last_dst_dim_size = dst_dims[num_dims - 1];
  const int64_t last_src_dim_size = src_dims[num_dims - 1];
  const int64_t last_padding_before_size = padding_before[num_dims - 1];
  const int64_t last_padding_after_size = padding_after[num_dims - 1];
  auto src_ptr = reinterpret_cast<std::uintptr_t>(src);
  auto dst_ptr = reinterpret_cast<std::uintptr_t>(dst);
  for (size_t size = max_pack_size; size > 1; size /= 2) {
    if (last_dst_dim_size % size == 0 && last_src_dim_size % size == 0
        && last_padding_before_size % size == 0 && last_padding_after_size % size == 0
        && src_ptr % size == 0 && dst_ptr % size == 0) {
      return size;
    }
  }
  return 1;
}

void SimplifyPadDims(size_t num_dims, const int64_t* src_dims, const int64_t* padding_before,
                     const int64_t* padding_after, size_t* simplified_num_dims,
                     int64_t* simplified_dst_dims, int64_t* simplified_src_dims,
                     int64_t* simplified_padding_before, int64_t* simplified_padding_after) {
  CHECK_NE(num_dims, 0);
  size_t valid_num_dims = 0;
  FOR_RANGE(size_t, i, 0, num_dims) {
    const int64_t dst_dim = src_dims[i] + padding_before[i] + padding_after[i];
    if ((i != 0) && (padding_before[i] == 0 && padding_after[i] == 0)) {
      simplified_dst_dims[valid_num_dims - 1] *= dst_dim;
      simplified_src_dims[valid_num_dims - 1] *= src_dims[i];
      simplified_padding_before[valid_num_dims - 1] *= src_dims[i];
      simplified_padding_after[valid_num_dims - 1] *= src_dims[i];
    } else {
      simplified_dst_dims[valid_num_dims] = dst_dim;
      simplified_src_dims[valid_num_dims] = src_dims[i];
      simplified_padding_before[valid_num_dims] = padding_before[i];
      simplified_padding_after[valid_num_dims] = padding_after[i];
      valid_num_dims += 1;
    }
  }
  *simplified_num_dims = valid_num_dims;
}

}  // namespace

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_CONSTANT_PAD_H_
