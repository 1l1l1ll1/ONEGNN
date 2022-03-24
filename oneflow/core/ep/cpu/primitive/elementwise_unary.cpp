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
#include "oneflow/core/ep/common/primitive/elementwise_unary.h"
#include "oneflow/core/ep/cpu/primitive/unary_functor.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"

namespace oneflow {

namespace ep {
namespace primitive {
namespace {

template<UnaryOp unary_op, typename Src, typename Dst>
class ElementwiseUnaryImpl : public ElementwiseUnary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryImpl);
  ElementwiseUnaryImpl() = default;
  ~ElementwiseUnaryImpl() override = default;

  void Launch(Stream* stream, const void* src_ptr, void* dst_ptr, size_t count) override {
    CpuStream* cpu_stream = stream->As<CpuStream>();

    Dst* dst = reinterpret_cast<Dst*>(dst_ptr);
    const Src* src = reinterpret_cast<const Src*>(src_ptr);
    cpu_stream->ParallelFor(0, count, [src, dst](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; i++) {
        dst[i] = UnaryFunctor<DeviceType::kCPU, unary_op, Dst, Src>()(src[i]);
      }
    });
  }

  void LaunchWithStride(Stream* stream, const void* src_ptr, void* dst_ptr, size_t count,
                        const StrideParam& in_stride, const StrideParam& out_stride) override {
    CpuStream* cpu_stream = stream->As<CpuStream>();

    Dst* dst = reinterpret_cast<Dst*>(dst_ptr);
    const Src* src = reinterpret_cast<const Src*>(src_ptr);
    cpu_stream->ParallelFor(
        0, count, [src, dst, in_stride, out_stride](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; i++) {
            int64_t src_idx = compute_index(i, in_stride, out_stride);
            dst[i] = UnaryFunctor<DeviceType::kCPU, unary_op, Dst, Src>()(src[src_idx]);
          }
        });
  }
};

template<UnaryOp unary_op, typename Src, typename Dst>
std::unique_ptr<ElementwiseUnary> NewElementwiseUnary() {
  return std::unique_ptr<ElementwiseUnary>(new ElementwiseUnaryImpl<unary_op, Src, Dst>());
}

class ElementwiseUnaryFactoryImpl : public ElementwiseUnaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryFactoryImpl);
  ElementwiseUnaryFactoryImpl() = default;
  ~ElementwiseUnaryFactoryImpl() override = default;

  std::unique_ptr<ElementwiseUnary> New(UnaryOp unary_op, DataType src_type,
                                        DataType dst_dtype) override {
#define MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY(unary_op, dtype_pair)                   \
  {std::make_tuple(unary_op, OF_PP_PAIR_SECOND(dtype_pair), OF_PP_PAIR_SECOND(dtype_pair)), \
   NewElementwiseUnary<unary_op, OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(dtype_pair)>},

#define MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY(unary_op, src_type_pair, dst_dtype_pair)  \
  {std::make_tuple(unary_op, OF_PP_PAIR_SECOND(src_type_pair), OF_PP_PAIR_SECOND(dst_dtype_pair)), \
   NewElementwiseUnary<unary_op, OF_PP_PAIR_FIRST(src_type_pair),                                  \
                       OF_PP_PAIR_FIRST(dst_dtype_pair)>},

    static const std::map<std::tuple<UnaryOp, DataType, DataType>,
                          std::function<std::unique_ptr<ElementwiseUnary>()>>
        new_elementwise_unary_handle{
            // For All Type OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                             UNARY_MATH_OP_SEQ, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)
            // For Float Type OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                             UNARY_FLOATING_MATH_OP_SEQ,
                                             CPU_PRIMITIVE_FLOATING_TYPE_SEQ)
            // For Logical OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                             UNARY_LOGICAL_OP_SEQ, CPU_PRIMITIVE_NATIVE_TYPE_SEQ,
                                             CPU_PRIMITIVE_INT8_TYPE_SEQ)};

#undef MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY

#undef MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY

    const auto it =
        new_elementwise_unary_handle.find(std::make_tuple(unary_op, src_type, dst_dtype));
    if (it != new_elementwise_unary_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, ElementwiseUnaryFactory, ElementwiseUnaryFactoryImpl);

}  // namespace
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
