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

#include <algorithm>

#include "oneflow/core/common/env_var/dtr.h"
#include "oneflow/core/eager/dtr_util.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/vm/dtr_ep_allocator.h"
#include "oneflow/core/vm/ep_backend_allocator.h"
#include "oneflow/core/vm/op_call_instruction_policy.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/user/kernels/stateful_opkernel.h"

namespace oneflow {

namespace dtr {

bool is_enabled() { return true; }

size_t memory_threshold() { return EnvInteger<ONEFLOW_DTR_BUDGET_MB>() * 1024 * 1024; }

bool is_enabled_and_debug() { return is_enabled() && debug_level() > 0; }

int debug_level() {
  if (!is_enabled()) { return 0; }
  return EnvInteger<ONEFLOW_DTR_DEBUG_LEVEL>();
}

bool is_check_enabled() {
  if (!is_enabled()) { return false; }
  return EnvBool<ONEFLOW_DTR_CHECK>();
}

double append_memory_frag_info_and_get(size_t free_mem, size_t threshold) {
  static size_t num = 0;
  // maintain a summation of memory frag rate
  static double memory_frag_rate_sum = 0;
  if (threshold > 0) {
    memory_frag_rate_sum += (1. * free_mem / threshold);
    num++;
  }
  return memory_frag_rate_sum / num;
}

vm::DtrEpAllocator* AllocatorManager::CreateOrGetAllocator(DeviceType device_type,
                                                           size_t device_index) {
  auto key = std::make_pair(device_type, device_index);
  auto it = allocators_.find(key);
  if (it == allocators_.end()) {
    auto ep_device =
        Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(device_type, device_index);
    auto ep_backend_allocator =
        std::make_unique<vm::EpBackendAllocator>(ep_device, ep::AllocationOptions{});
    auto allocator = std::make_unique<vm::DtrEpAllocator>(ep::kMaxAlignmentRequirement,
                                                          std::move(ep_backend_allocator));
    allocators_.emplace(key, std::move(allocator));
    return allocators_.at(key).get();
  } else {
    return it->second.get();
  }
}

}  // namespace dtr

#ifdef ENABLE

namespace vm {

namespace {

auto ConvertToDTRVector(const std::vector<std::shared_ptr<EagerBlobObject>>& base_class_vector) {
  std::vector<std::shared_ptr<DTREagerBlobObject>> sub_class_vector;
  std::transform(base_class_vector.begin(), base_class_vector.end(),
                 std::back_inserter(sub_class_vector),
                 [](const std::shared_ptr<EagerBlobObject>& x) {
                   return CHECK_NOTNULL(std::dynamic_pointer_cast<DTREagerBlobObject>(x));
                 });
  return sub_class_vector;
};

}  // namespace

std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTRInputs(
    const std::shared_ptr<const LocalCallOpKernelPhyInstrOperand>& operand) {
  return GetDTRInputs(operand.get());
}

std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTROutputs(
    const std::shared_ptr<const LocalCallOpKernelPhyInstrOperand>& operand) {
  return GetDTROutputs(operand.get());
}

std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTRInputs(
    const LocalCallOpKernelPhyInstrOperand* operand) {
  return ConvertToDTRVector(*operand->inputs());
}

std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTROutputs(
    const LocalCallOpKernelPhyInstrOperand* operand) {
  return ConvertToDTRVector(*operand->outputs());
}

std::shared_ptr<LocalCallOpKernelPhyInstrOperand> DTROp2LocalCallOp(DTRInstrOperand* operand) {
  const auto& inputs = operand->inputs();
  const auto& outputs = operand->outputs();

  std::shared_ptr<one::EagerBlobObjectList> input_shared_ptr =
      std::make_shared<one::EagerBlobObjectList>(inputs.size());
  std::shared_ptr<one::EagerBlobObjectList> output_shared_ptr =
      std::make_shared<one::EagerBlobObjectList>(outputs.size());

  for (int i = 0; i < inputs.size(); ++i) {
    if (auto input = inputs[i].lock()) {
      input_shared_ptr->at(i) = input;
    } else {
      // CHECK_JUST(Global<dtr::TensorPool>::Get()->display2());
      LOG(FATAL) << "null at input " << i << " of op "
                 << operand->shared_opkernel()->op_type_name();
    }
  }

  for (int i = 0; i < outputs.size(); ++i) {
    if (auto output = outputs[i].lock()) {
      output_shared_ptr->at(i) = output;
    } else {
      // CHECK_JUST(Global<dtr::TensorPool>::Get()->display2());
      LOG(FATAL) << "null at output " << i << " of op "
                 << operand->shared_opkernel()->op_type_name();
    }
  }

  auto phy_instr_operand = CHECK_JUST(LocalCallOpKernelPhyInstrOperand::New(
      operand->shared_opkernel(), input_shared_ptr, output_shared_ptr,
      operand->consistent_tensor_infer_result(), operand->op_interp_ctx(),
      operand->dev_vm_dep_object_consume_mode()));

  return phy_instr_operand;
}

namespace {
Maybe<void> CheckInMemory(const std::vector<std::shared_ptr<DTREagerBlobObject>>& vec) {
  int i = 0;
  for (auto& dtr_blob_object : vec) {
    if (dtr_blob_object->shape().elem_cnt() > 0) {
      CHECK_OR_RETURN(dtr_blob_object->is_in_memory());
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object->dptr());
    }
    i++;
  }
  return Maybe<void>::Ok();
}
}  // namespace

Maybe<void> CheckInputInMemory(LocalCallOpKernelPhyInstrOperand* operand) {
  return CheckInMemory(GetDTRInputs(operand));
}

Maybe<void> CheckOutputInMemory(LocalCallOpKernelPhyInstrOperand* operand) {
  return CheckInMemory(GetDTROutputs(operand));
}

}  // namespace vm

#endif

}  // namespace oneflow