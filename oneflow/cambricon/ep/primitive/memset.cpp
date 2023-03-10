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
#ifdef WITH_MLU

#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/combricon/ep/mlu_stream.h"


namespace oneflow {

namespace ep {
namespace primitive {

namespace {

class MemsetImpl : public Memset {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemsetImpl);
  MemsetImpl() = default;
  ~MemsetImpl() override = default;

  void Launch(Stream* stream, void* ptr, int value, size_t count) override {
    auto* mlu_stream = stream->As<MluStream>();
    OF_MLU_CHECK(cnrtMemsetAsync(ptr, value, count, mlu_stream->mlu_stream()));
  }
};

class MemsetFactoryImpl : public MemsetFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemsetFactoryImpl);
  MemsetFactoryImpl() = default;
  ~MemsetFactoryImpl() override = default;

  std::unique_ptr<Memset> New() override { return std::unique_ptr<Memset>(new MemsetImpl()); }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kMLU, MemsetFactory, MemsetFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif // WITH_MLU
