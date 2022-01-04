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
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/eager/opkernel_object.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/eager/opkernel_instruction.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/object.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/job/parallel_signature.cfg.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/operator/op_node_signature_desc.h"
#include "oneflow/core/operator/op_conf_symbol.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/framework/tensor_pool.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {
namespace vm {

class InitOpKernelObjectInstructionType final : public vm::InstructionType {
 public:
  InitOpKernelObjectInstructionType() = default;
  ~InitOpKernelObjectInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  void Infer(vm::Instruction* instruction) const override {
    FlatMsgView<NewOpKernelObjectInstrOperand> view;
    CHECK(view.Match(instruction->instr_msg().operand()));
    CHECK_EQ(view->op_conf_size(), view->op_size());
    const auto* operand_job_desc = instruction->operand_type(view->job_desc());
    CHECK_NOTNULL(operand_job_desc);
    const auto& job_desc_object = CHECK_JUST(operand_job_desc->Get<vm::ObjectWrapper<JobDesc>>());
    for (int i = 0; i < view->op_size(); ++i) {
      CHECK_GT(view->op(i).logical_object_id(), 0);
      const auto* operand_op_conf = instruction->operand_type(view->op_conf(i));
      CHECK_NOTNULL(operand_op_conf);
      const auto& op_conf_object =
          CHECK_JUST(operand_op_conf->Get<vm::ObjectWrapper<OperatorConfSymbol>>());
      CHECK(op_conf_object.Get().op_conf().has_user_conf());
      vm::RwMutexedObject* rw_mutexed_object = instruction->mut_operand_type(view->op(i));
      const auto& parallel_desc = instruction->parallel_desc();
      CHECK(static_cast<bool>(parallel_desc));
      DeviceType device_type = parallel_desc->device_type();
      rw_mutexed_object->Init<OpKernelObject>(op_conf_object.Get().op_conf(),
                                              job_desc_object.GetPtr(), device_type);
    }
  }
  void Compute(vm::Instruction* instruction) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<InitOpKernelObjectInstructionType>("InitOpKernelObject"));

class DeleteOpKernelObjectInstructionType final : public vm::InstructionType {
 public:
  DeleteOpKernelObjectInstructionType() = default;
  ~DeleteOpKernelObjectInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  void Infer(vm::Instruction* instruction) const override {
    FlatMsgView<DeleteOpKernelObjectInstrOperand> view;
    CHECK(view.Match(instruction->instr_msg().operand()));
    for (int i = 0; i < view->op_size(); ++i) {
      auto* type_rw_mutexed_object = instruction->mut_operand_type(view->op(i));
      CHECK(type_rw_mutexed_object->Has<OpKernelObject>());
      type_rw_mutexed_object->reset_object();
    }
  }
  void Compute(vm::Instruction* instruction) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<DeleteOpKernelObjectInstructionType>("DeleteOpKernelObject"));

namespace {

template<typename T, typename CallbackT>
Maybe<void> ForEachConstInputBnAndBlobObject(vm::Instruction* instruction, const T& args,
                                             const CallbackT& Callback) {
  CHECK_EQ_OR_RETURN(args.const_ibn_size(), args.const_input_blob_size());
  FOR_RANGE(int, i, 0, args.const_ibn_size()) {
    const auto* operand_ibn = instruction->operand_type(args.const_ibn(i));
    CHECK_NOTNULL_OR_RETURN(operand_ibn);
    const std::string& bn_in_op = JUST(operand_ibn->template Get<vm::StringObject>()).str();
    const auto* operand_input_blob = instruction->operand_type(args.const_input_blob(i));
    CHECK_NOTNULL_OR_RETURN(operand_input_blob)
        << "bn_in_op: " << bn_in_op
        << ", object_id: " << args.const_input_blob(i).logical_object_id();
    const auto& blob_object = JUST(operand_input_blob->template Get<BlobObject>());
    JUST(Callback(bn_in_op, blob_object));
  }
  return Maybe<void>::Ok();
}

template<typename T, typename CallbackT>
Maybe<void> ForEachMutInputBnAndBlobObject(vm::Instruction* instruction, const T& args,
                                           const CallbackT& Callback) {
  CHECK_EQ_OR_RETURN(args.mut_ibn_size(), args.mut_input_blob_size());
  FOR_RANGE(int, i, 0, args.mut_ibn_size()) {
    const auto* operand_ibn = instruction->operand_type(args.mut_ibn(i));
    CHECK_NOTNULL_OR_RETURN(operand_ibn);
    const std::string& bn_in_op = JUST(operand_ibn->template Get<vm::StringObject>()).str();
    auto* operand_input_blob = instruction->mut_operand_type(args.mut_input_blob(i));
    CHECK_NOTNULL_OR_RETURN(operand_input_blob)
        << "bn_in_op: " << bn_in_op
        << ", object_id: " << args.mut_input_blob(i).logical_object_id();
    auto* blob_object = JUST(operand_input_blob->template Mut<BlobObject>());
    JUST(Callback(bn_in_op, blob_object));
  }
  return Maybe<void>::Ok();
}

template<typename T, typename CallbackT>
Maybe<void> ForEachOutputBnAndBlobObject(vm::Instruction* instruction, const T& args,
                                         const CallbackT& Callback) {
  CHECK_EQ_OR_RETURN(args.obn_size(), args.output_blob_size());
  FOR_RANGE(int, i, 0, args.obn_size()) {
    const auto* operand_obn = instruction->operand_type(args.obn(i));
    CHECK_NOTNULL_OR_RETURN(operand_obn);
    const std::string& bn_in_op = JUST(operand_obn->template Get<vm::StringObject>()).str();
    auto* operand_output_blob = instruction->mut_operand_type(args.output_blob(i));
    CHECK_NOTNULL_OR_RETURN(operand_output_blob) << "obn: " << bn_in_op;
    auto* blob_object = JUST(operand_output_blob->template Mut<BlobObject>());
    JUST(Callback(bn_in_op, blob_object));
  }
  CHECK_EQ_OR_RETURN(args.mut2_obn_size(), args.mut2_output_blob_size());
  FOR_RANGE(int, i, 0, args.mut2_obn_size()) {
    const auto* operand_obn = instruction->operand_type(args.mut2_obn(i));
    CHECK_NOTNULL_OR_RETURN(operand_obn);
    const std::string& bn_in_op = JUST(operand_obn->template Get<vm::StringObject>()).str();
    auto* operand_output_blob = instruction->mut_operand_type(args.mut2_output_blob(i));
    CHECK_NOTNULL_OR_RETURN(operand_output_blob) << "obn: " << bn_in_op;
    auto* blob_object = JUST(operand_output_blob->template Mut<BlobObject>());
    JUST(Callback(bn_in_op, blob_object));
  }
  return Maybe<void>::Ok();
}

template<typename DoEachT>
Maybe<void> ForEachDTROutputTensor(std::shared_ptr<LocalCallOpKernelPhyInstrOperand>& operand,
                                   const DoEachT& DoEach) {
  auto* ptr = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand.get());
  CHECK_NOTNULL_OR_RETURN(ptr);
  for (const auto& output : *ptr->outputs()) {
    CHECK_OR_RETURN(static_cast<bool>(output.get()));
    auto shared_dtr_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(output);
    CHECK_NOTNULL_OR_RETURN(shared_dtr_blob_object);
    JUST(DoEach(shared_dtr_blob_object));
  }
  return Maybe<void>::Ok();
}

template<typename DoEachT>
Maybe<void> ForEachDTRInputTensor(std::shared_ptr<LocalCallOpKernelPhyInstrOperand>& operand,
                                  const DoEachT& DoEach) {
  auto* ptr = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand.get());
  CHECK_NOTNULL_OR_RETURN(ptr);
  for (const auto& input : *ptr->inputs()) {
    CHECK_OR_RETURN(static_cast<bool>(input.get()));
    auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(input.get());
    CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
    JUST(DoEach(dtr_blob_object));
  }
  return Maybe<void>::Ok();
}

template<typename DoEachT>
Maybe<void> ForEachDTRInputTensor(LocalCallOpKernelPhyInstrOperand* operand,
                                  const DoEachT& DoEach) {
  auto* ptr = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand);
  CHECK_NOTNULL_OR_RETURN(ptr);
  for (const auto& input : *ptr->inputs()) {
    CHECK_OR_RETURN(static_cast<bool>(input.get()));
    auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(input.get());
    CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
    JUST(DoEach(dtr_blob_object));
  }
  return Maybe<void>::Ok();
}

std::unique_ptr<LocalCallOpKernelPhyInstrOperand> DTROp2LocalCallOp(DTRInstrOperand* operand) {
  const auto& inputs = operand->inputs();
  const auto& outputs = operand->outputs();

  std::shared_ptr<one::EagerBlobObjectList> input_shared_ptr =
      std::make_shared<one::EagerBlobObjectList>(inputs.size());
  std::shared_ptr<one::EagerBlobObjectList> output_shared_ptr =
      std::make_shared<one::EagerBlobObjectList>(outputs.size());

  for (int i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].lock()) {
      std::cout << "null at input " << i << std::endl;
      CHECK_JUST(Global<one::DTRTensorPool>::Get()->display2());
    }
    input_shared_ptr->at(i) = CHECK_NOTNULL(inputs[i].lock());
  }

  for (int i = 0; i < outputs.size(); ++i) {
    output_shared_ptr->at(i) = CHECK_NOTNULL(outputs[i].lock());
  }

  auto phy_instr_operand = std::make_unique<LocalCallOpKernelPhyInstrOperand>(
      operand->shared_opkernel(), input_shared_ptr, output_shared_ptr,
      operand->consistent_tensor_infer_result(), operand->op_interp_ctx(),
      operand->dev_vm_dep_object_consume_mode());

  return phy_instr_operand;
}

template<typename T>
Maybe<void> MakeBlobDesc4BnInOp(vm::Instruction* instruction, const T& args,
                                std::function<BlobDesc*(const std::string&)>* BlobDesc4BnInOp) {
  const auto& obn2blob_desc = std::make_shared<HashMap<std::string, BlobDesc*>>();
  {
    HashSet<const BlobDesc*> out_blob_descs;
    JUST(ForEachOutputBnAndBlobObject(
        instruction, args,
        [&](const std::string& bn_in_op, BlobObject* blob_object) -> Maybe<void> {
          auto* blob_desc = blob_object->mut_blob_desc();
          CHECK_OR_RETURN(out_blob_descs.insert(blob_desc).second);
          CHECK_OR_RETURN(obn2blob_desc->emplace(bn_in_op, blob_desc).second);
          return Maybe<void>::Ok();
        }));
  }
  const auto& ibn2blob_desc = std::make_shared<HashMap<std::string, const BlobDesc*>>();
  JUST(ForEachConstInputBnAndBlobObject(
      instruction, args,
      [&](const std::string& bn_in_op, const BlobObject& blob_object) -> Maybe<void> {
        CHECK_OR_RETURN(ibn2blob_desc->emplace(bn_in_op, &blob_object.blob_desc()).second);
        return Maybe<void>::Ok();
      }));
  JUST(ForEachMutInputBnAndBlobObject(
      instruction, args, [&](const std::string& bn_in_op, BlobObject* blob_object) -> Maybe<void> {
        CHECK_OR_RETURN(ibn2blob_desc->emplace(bn_in_op, &blob_object->blob_desc()).second);
        return Maybe<void>::Ok();
      }));
  *BlobDesc4BnInOp = [obn2blob_desc, ibn2blob_desc](const std::string& bn_in_op) -> BlobDesc* {
    const auto& output_iter = obn2blob_desc->find(bn_in_op);
    if (output_iter != obn2blob_desc->end()) { return output_iter->second; }
    const auto& input_iter = ibn2blob_desc->find(bn_in_op);
    if (input_iter != ibn2blob_desc->end()) { return const_cast<BlobDesc*>(input_iter->second); }
    return nullptr;
  };
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> MakeBlob4BnInOp(
    vm::Instruction* instruction, const T& args,
    std::function<Blob*(const std::string&)>* Blob4BnInOp,
    const std::function<bool(const std::string&, const BlobObject&)>& FilterOutBlob) {
  const auto& obn2blob = std::make_shared<HashMap<std::string, Blob*>>();
  JUST(ForEachOutputBnAndBlobObject(
      instruction, args, [&](const std::string& bn_in_op, BlobObject* blob_object) -> Maybe<void> {
        if (!FilterOutBlob(bn_in_op, *blob_object)) { return Maybe<void>::Ok(); }
        CHECK_OR_RETURN(obn2blob->emplace(bn_in_op, blob_object->mut_blob()).second);
        return Maybe<void>::Ok();
      }));
  const auto& ibn2blob = std::make_shared<HashMap<std::string, const Blob*>>();
  JUST(ForEachConstInputBnAndBlobObject(
      instruction, args,
      [&](const std::string& bn_in_op, const BlobObject& blob_object) -> Maybe<void> {
        CHECK_OR_RETURN(ibn2blob->emplace(bn_in_op, &blob_object.blob()).second);
        return Maybe<void>::Ok();
      }));
  JUST(ForEachMutInputBnAndBlobObject(
      instruction, args, [&](const std::string& bn_in_op, BlobObject* blob_object) -> Maybe<void> {
        CHECK_OR_RETURN(ibn2blob->emplace(bn_in_op, blob_object->mut_blob()).second);
        return Maybe<void>::Ok();
      }));
  *Blob4BnInOp = [obn2blob, ibn2blob](const std::string& bn_in_op) -> Blob* {
    const auto& output_iter = obn2blob->find(bn_in_op);
    if (output_iter != obn2blob->end()) { return output_iter->second; }
    const auto& input_iter = ibn2blob->find(bn_in_op);
    if (input_iter != ibn2blob->end()) { return const_cast<Blob*>(input_iter->second); }
    return nullptr;
  };
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> MakeBlob4BnInOp(vm::Instruction* instruction, const T& args,
                            std::function<Blob*(const std::string&)>* Blob4BnInOp) {
  return MakeBlob4BnInOp(instruction, args, Blob4BnInOp,
                         [](const std::string&, const BlobObject&) { return true; });
}

template<typename T>
void InitOutputBlobObjects(vm::Instruction* instruction, const T& args,
                           const std::shared_ptr<MemoryCase>& mem_case, DataType data_type) {
  const auto& InitRwMutexedObject = [&](vm::RwMutexedObject* rw_mutexed_object) {
    const auto& parallel_desc = instruction->parallel_desc();
    CHECK(static_cast<bool>(parallel_desc));
    if (rw_mutexed_object->has_object()) {
      // mutable input
      CHECK(rw_mutexed_object->Has<BlobObject>());
    } else {
      rw_mutexed_object->Init<EagerBlobObject>(mem_case, std::make_shared<Shape>(), data_type,
                                               std::make_shared<TensorBuffer>());
    }
  };
  FOR_RANGE(int, i, 0, args.output_blob_size()) {
    InitRwMutexedObject(instruction->mut_operand_type(args.output_blob(i)));
  }
  FOR_RANGE(int, i, 0, args.mut2_output_blob_size()) {
    InitRwMutexedObject(instruction->mut_operand_type(args.mut2_output_blob(i)));
  }
}

template<typename T>
Maybe<void> CheckBlobParallel(vm::Instruction* instruction, const T& args,
                              const OpNodeSignatureDesc* op_node_signature) {
  const auto& bn_in_op2parallel_desc_symbol_id =
      op_node_signature->parallel_signature().bn_in_op2parallel_desc_symbol_id();

  const auto& ParallelDesc4BnInOp = [&](const std::string& bn_in_op) -> Maybe<const ParallelDesc*> {
    const auto& iter = bn_in_op2parallel_desc_symbol_id.find(bn_in_op);
    // TODO(Liang Depeng): should not tolerate nullptr.
    if (iter == bn_in_op2parallel_desc_symbol_id.end()) { return nullptr; }
    int64_t symbol_id = iter->second;
    const symbol::Storage<ParallelDesc>* symbol_storage_ptr =
        Global<symbol::Storage<ParallelDesc>>::Get();
    CHECK_OR_RETURN(symbol_storage_ptr->Has(symbol_id));
    return symbol_storage_ptr->GetPtr(symbol_id).get();
  };

  JUST(ForEachOutputBnAndBlobObject(
      instruction, args, [&](const std::string& bn_in_op, BlobObject* blob_object) -> Maybe<void> {
        const auto* parallel_desc = JUST(ParallelDesc4BnInOp(bn_in_op));
        if (parallel_desc == nullptr) { return Maybe<void>::Ok(); }
        JUST(blob_object->CheckMemCase(*parallel_desc, instruction->stream().machine_id()));
        return Maybe<void>::Ok();
      }));

  JUST(ForEachConstInputBnAndBlobObject(
      instruction, args,
      [&](const std::string& bn_in_op, const BlobObject& blob_object) -> Maybe<void> {
        const auto* parallel_desc = JUST(ParallelDesc4BnInOp(bn_in_op));
        if (parallel_desc == nullptr) { return Maybe<void>::Ok(); }
        JUST(blob_object.CheckMemCase(*parallel_desc, instruction->stream().machine_id()));
        return Maybe<void>::Ok();
      }));
  JUST(ForEachMutInputBnAndBlobObject(
      instruction, args, [&](const std::string& bn_in_op, BlobObject* blob_object) -> Maybe<void> {
        const auto* parallel_desc = JUST(ParallelDesc4BnInOp(bn_in_op));
        if (parallel_desc == nullptr) { return Maybe<void>::Ok(); }
        JUST(blob_object->CheckMemCase(*parallel_desc, instruction->stream().machine_id()));
        return Maybe<void>::Ok();
      }));
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> OpKernelInfer(OpKernelObject* opkernel_obj, vm::Instruction* instruction, const T& args,
                          const std::shared_ptr<MemoryCase>& mem_case) {
  {
    DataType default_data_type = opkernel_obj->job_desc().DefaultDataType();
    CHECK_NE_OR_RETURN(default_data_type, DataType::kInvalidDataType);
    InitOutputBlobObjects(instruction, args, mem_case, default_data_type);
  }
  std::function<BlobDesc*(const std::string&)> BlobDesc4BnInOp;
  JUST(MakeBlobDesc4BnInOp(instruction, args, &BlobDesc4BnInOp));
  const OpNodeSignatureDesc* op_node_signature = nullptr;
  {
    const auto* operand = instruction->operand_type(args.op_node_signature());
    const auto& op_node_signature_object =
        JUST(operand->template Get<vm::ObjectWrapper<OpNodeSignatureDesc>>());
    op_node_signature = &op_node_signature_object.Get();
  }
  ParallelContext parallel_ctx;
  JUST(instruction->parallel_desc()->GetParallelContext(
      &parallel_ctx, instruction->stream().machine_id(), instruction->stream().device_id()));
  JUST(opkernel_obj->ResetOpAndKernel(*op_node_signature, &parallel_ctx, BlobDesc4BnInOp,
                                      instruction->parallel_desc().get()));
  JUST(CheckBlobParallel(instruction, args, op_node_signature));
  JUST(ForEachOutputBnAndBlobObject(
      instruction, args, [](const std::string& obn, BlobObject* blob_object) -> Maybe<void> {
        return blob_object->TryInitBlob();
      }));
  std::function<Blob*(const std::string&)> Blob4BnInOp;
  Shape empty_shape{};
  const auto& FilterOutBlob = [&](const std::string& bn_in_op, const BlobObject& blob_object) {
    return !(bn_in_op == "tmp_buffer_0" && blob_object.blob_desc().shape() == empty_shape);
  };
  JUST(MakeBlob4BnInOp(instruction, args, &Blob4BnInOp, FilterOutBlob));
  opkernel_obj->kernel().Infer(Blob4BnInOp);
  return Maybe<void>::Ok();
}

Maybe<void> OpKernelInfer(SystemOpKernelObject* opkernel_obj, vm::Instruction* instruction,
                          const StatelessCallOpKernelInstrOperand& args,
                          const std::shared_ptr<MemoryCase>& mem_case) {
  {
    DataType default_data_type = opkernel_obj->job_desc().DefaultDataType();
    CHECK_NE_OR_RETURN(default_data_type, DataType::kInvalidDataType);
    InitOutputBlobObjects(instruction, args, mem_case, default_data_type);
  }
  std::function<BlobDesc*(const std::string&)> BlobDesc4BnInOp;
  JUST(MakeBlobDesc4BnInOp(instruction, args, &BlobDesc4BnInOp));
  const OpNodeSignatureDesc* op_node_signature = nullptr;
  {
    const auto* operand = instruction->operand_type(args.op_node_signature());
    const auto& op_node_signature_object =
        JUST(operand->template Get<vm::ObjectWrapper<OpNodeSignatureDesc>>());
    op_node_signature = &op_node_signature_object.Get();
  }
  ParallelContext parallel_ctx;
  JUST(instruction->parallel_desc()->GetParallelContext(
      &parallel_ctx, instruction->stream().machine_id(), instruction->stream().device_id()));
  JUST(opkernel_obj->ResetKernel(*op_node_signature, &parallel_ctx, BlobDesc4BnInOp,
                                 instruction->parallel_desc().get()));
  JUST(CheckBlobParallel(instruction, args, op_node_signature));
  JUST(ForEachOutputBnAndBlobObject(
      instruction, args, [](const std::string& obn, BlobObject* blob_object) -> Maybe<void> {
        return blob_object->TryInitBlob();
      }));
  std::function<Blob*(const std::string&)> Blob4BnInOp;
  JUST(MakeBlob4BnInOp(instruction, args, &Blob4BnInOp));
  opkernel_obj->kernel_ctx()->UpdateBnInOp2BlobFn(Blob4BnInOp);
  opkernel_obj->kernel().SystemForwardHeader(opkernel_obj->kernel_ctx());
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> OpKernelCompute(OpKernelObject* opkernel_obj, vm::Instruction* instruction,
                            const T& args) {
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
  JUST(ForEachOutputBnAndBlobObject(
      instruction, args, [&](const std::string&, BlobObject* blob_object) -> Maybe<void> {
        JUST(blob_object->TryAllocateBlobBodyMemory(device_ctx));
        return Maybe<void>::Ok();
      }));
  std::shared_ptr<user_op::OpKernelState> new_state;
  {
    std::function<Blob*(const std::string&)> Blob4BnInOp;
    Shape empty_shape{};
    const auto& FilterOutBlob = [&](const std::string& bn_in_op, const BlobObject& blob_object) {
      return !(bn_in_op == "tmp_buffer_0" && blob_object.blob_desc().shape() == empty_shape);
    };
    JUST(MakeBlob4BnInOp(instruction, args, &Blob4BnInOp, FilterOutBlob));
    EagerKernel* eager_kernel = opkernel_obj->mut_kernel();
    const auto& old_state = opkernel_obj->opkernel_state();
    new_state = eager_kernel->EagerForward(old_state, device_ctx, Blob4BnInOp);
  }
  opkernel_obj->reset_opkernel_state(new_state);
  return Maybe<void>::Ok();
}

Maybe<void> OpKernelCompute(SystemOpKernelObject* opkernel_obj, vm::Instruction* instruction,
                            const StatelessCallOpKernelInstrOperand& args) {
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
  JUST(ForEachOutputBnAndBlobObject(
      instruction, args, [&](const std::string&, BlobObject* blob_object) -> Maybe<void> {
        JUST(blob_object->TryAllocateBlobBodyMemory(device_ctx));
        return Maybe<void>::Ok();
      }));
  std::function<Blob*(const std::string&)> Blob4BnInOp;
  JUST(MakeBlob4BnInOp(instruction, args, &Blob4BnInOp));
  opkernel_obj->kernel_ctx()->UpdateBnInOp2BlobFn(Blob4BnInOp);
  opkernel_obj->kernel_ctx()->set_device_ctx(device_ctx);
  opkernel_obj->kernel().SystemForwardDataContent(opkernel_obj->kernel_ctx());
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<T*> GetSharedOpKernel(vm::Instruction* instruction, DeviceType device_type,
                            const StatelessCallOpKernelInstrOperand& args) {
  const auto* operand_job_desc = instruction->operand_type(args.job_desc());
  CHECK_NOTNULL_OR_RETURN(operand_job_desc);
  const auto& job_desc_ptr = JUST(operand_job_desc->Get<vm::ObjectWrapper<JobDesc>>()).GetPtr();
  const auto* operand_op_conf = instruction->mut_operand_type(args.op_conf());
  const auto& op_conf =
      JUST(operand_op_conf->Get<vm::ObjectWrapper<OperatorConfSymbol>>()).Get().op_conf();
  vm::RwMutexedObject* rw_mutexed_object = instruction->mut_operand_type(args.shared_opkernel());
  CHECK_OR_RETURN(!rw_mutexed_object->has_object() || rw_mutexed_object->Has<OpKernelObject>()
                  || rw_mutexed_object->Has<SystemOpKernelObject>());
  const auto& parallel_desc = instruction->parallel_desc();
  CHECK_OR_RETURN(static_cast<bool>(parallel_desc));
  CHECK_EQ_OR_RETURN(device_type, parallel_desc->device_type());
  rw_mutexed_object->reset_object();
  return rw_mutexed_object->Init<T>(op_conf, job_desc_ptr, device_type);
}

}  // namespace

struct LocalCallOpKernelUtil {
  static inline Maybe<void> DoInfer(LocalCallOpKernelPhyInstrOperand* operand,
                                    const vm::Stream& stream) {
    operand->mut_opkernel()->composed_attrs_for_scheduler_thread()->ResetPrior(operand->attrs());
    operand->set_user_opkernel(JUST(operand->mut_opkernel()->ChooseOpKernel(
        operand->inputs(), operand->outputs(), operand->consistent_tensor_infer_result())));
    JUST(CheckOutputBlobObjectsMemCase(operand, stream));
    JUST(InitOutputBlobs(operand));
    return Maybe<void>::Ok();
  }
  static inline Maybe<void> Infer(vm::Instruction* instruction) {
    auto* operand = JUST(GetLocalCallOpKernelPhyInstrOperand(instruction));
    auto& stream = instruction->stream();
    JUST(DoInfer(operand, stream));
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> FullCompute(LocalCallOpKernelPhyInstrOperand* operand,
                                        DeviceCtx* device_ctx) {
    JUST(InferTempStorageBlobDesc(operand));
    JUST(ResetTempStorageBlob(operand));
    JUST(AllocateOutputBlobsMemory(operand, device_ctx));
    JUST(TryAllocateTempStorageBlobMemory(operand, device_ctx));
    user_op::OpKernelState* state;
    TryInitOpKernelState(operand, device_ctx, &state);
    JUST(OpKernelCompute(operand, device_ctx, state));
    
    JUST(DeallocateTempStorageBlobMemory(operand, device_ctx));
    operand->set_user_opkernel(nullptr);
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> Compute(vm::Instruction* instruction) {
    auto* operand = JUST(GetLocalCallOpKernelPhyInstrOperand(instruction));
    DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
    JUST(FullCompute(operand, device_ctx));
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> Prepare(vm::Instruction* instruction);
  static inline Maybe<void> InitOutputBlobAttrs(vm::Instruction* instruction) {
    return Maybe<void>::Ok();
  }
  static inline Maybe<void> UpdateTensorInfo(vm::Instruction* instruction, double compute_time) {
    return Maybe<void>::Ok();
  }
  // static inline Maybe<void> evict(vm::DTREagerBlobObject* blob_object) {
  //   return Maybe<void>::Ok();
  // }

  static inline Maybe<LocalCallOpKernelPhyInstrOperand*> GetLocalCallOpKernelPhyInstrOperand(
      vm::Instruction* instruction) {
    const auto& operand = instruction->instr_msg().phy_instr_operand();
    CHECK_OR_RETURN(static_cast<bool>(operand));
    auto* ptr = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand.get());
    CHECK_NOTNULL_OR_RETURN(ptr);
    return ptr;
  }

 public:
  static inline Maybe<LocalCallOpKernelPhyInstrOperand> GetSharedLocalCallOpKernelPhyInstrOperand(
      vm::Instruction* instruction) {
    const auto& operand = instruction->instr_msg().phy_instr_operand();
    CHECK_OR_RETURN(static_cast<bool>(operand));
    auto local_operand = std::dynamic_pointer_cast<LocalCallOpKernelPhyInstrOperand>(operand);
    CHECK_NOTNULL_OR_RETURN(local_operand);
    return local_operand;
  }

 private:
  static inline Maybe<const MemoryCase&> GetMemCase(LocalCallOpKernelPhyInstrOperand* operand) {
    const auto& mem_case = operand->opkernel().mem_case();
    CHECK_OR_RETURN(static_cast<bool>(mem_case));
    return *mem_case;
  }

  static inline Maybe<void> CheckMemCase(const MemoryCase& mem_case, DeviceType device_type,
                                         int64_t device_id) {
    if (mem_case.has_host_mem()) {
      CHECK_EQ_OR_RETURN(device_type, DeviceType::kCPU);
    } else if (mem_case.has_device_cuda_mem()) {
      CHECK_EQ_OR_RETURN(mem_case.device_cuda_mem().device_id(), device_id);
    } else {
      OF_UNIMPLEMENTED();
    }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> CheckOutputBlobObjectsMemCase(LocalCallOpKernelPhyInstrOperand* operand,
                                                          const vm::Stream& stream) {
    DeviceType device_type = JUST(DeviceType4DeviceTag(stream.stream_type().device_tag()));
    const auto& mem_case = JUST(GetMemCase(operand));
    JUST(CheckMemCase(mem_case, device_type, stream.device_id()));
    JUST(operand->ForEachOutputTensor([&](vm::EagerBlobObject* blob_object) -> Maybe<void> {
      CHECK_OR_RETURN(static_cast<bool>(blob_object));
      if (operand->opkernel().need_check_mem_case()) {
        JUST(CheckMemCase(blob_object->mem_case(), device_type, stream.device_id()));
      }
      return Maybe<void>::Ok();
    }));
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> InitOutputBlobs(LocalCallOpKernelPhyInstrOperand* operand) {
    JUST(operand->ForEachOutputTensor([&](vm::EagerBlobObject* blob_object) -> Maybe<void> {
      CHECK_OR_RETURN(static_cast<bool>(blob_object));
      JUST(blob_object->TryInitBlob());
      return Maybe<void>::Ok();
    }));
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> InferTempStorageBlobDesc(LocalCallOpKernelPhyInstrOperand* operand) {
    const auto& InferTmpSizeFn = operand->opkernel().GetInferTmpSizeFn(operand->user_opkernel());
    auto* temp_blob_desc = operand->mut_opkernel()->mut_temp_blob_object()->mut_blob_desc();
    CHECK_OR_RETURN(temp_blob_desc->data_type() == DataType::kChar);
    one::LocalUserOpInferContext* op_infer_ctx =
        operand->opkernel().op_infer_ctx_for_scheduler_thread();
    op_infer_ctx->Update(operand->inputs(), operand->outputs(),
                         operand->consistent_tensor_infer_result());
    size_t temp_size = InferTmpSizeFn(op_infer_ctx);
    temp_blob_desc->mut_shape() = Shape({static_cast<int64_t>(temp_size)});
    temp_blob_desc->set_is_dynamic(true);
    op_infer_ctx->Update(nullptr, nullptr, nullptr);
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> ResetTempStorageBlob(LocalCallOpKernelPhyInstrOperand* operand) {
    JUST(operand->mut_opkernel()->mut_temp_blob_object()->InitBlob());
    return Maybe<void>::Ok();
  }

  template<typename CallbackT>
  static inline Maybe<void> WithComputeContext(LocalCallOpKernelPhyInstrOperand* operand,
                                               DeviceCtx* device_ctx, const CallbackT& Callback) {
    auto* opkernel = operand->mut_opkernel();
    JUST(Callback(opkernel->UpdateComputeContext(operand->inputs(), operand->outputs(),
                                                 operand->consistent_tensor_infer_result(),
                                                 device_ctx)));
    // tensor tuples are not allowed to be hold by StatefulLocalOpKernel
    opkernel->UpdateComputeContext(nullptr, nullptr, nullptr, nullptr);
    return Maybe<void>::Ok();
  }

  static inline void TryInitOpKernelState(LocalCallOpKernelPhyInstrOperand* operand,
                                          DeviceCtx* device_ctx, user_op::OpKernelState** state) {
    if (operand->op_interp_ctx().state) {
      *state = operand->op_interp_ctx().state.get();
      return;
    }
    operand->mut_opkernel()->TryInitOpKernelState(operand->user_opkernel(), device_ctx,
                                                  operand->inputs(), operand->outputs(),
                                                  operand->consistent_tensor_infer_result(), state);
  }

  static inline Maybe<void> AllocateOutputBlobsMemory(LocalCallOpKernelPhyInstrOperand* operand,
                                                      DeviceCtx* device_ctx) {
    JUST(operand->ForEachOutputTensor([&](vm::EagerBlobObject* blob_object) -> Maybe<void> {
      JUST(blob_object->TryAllocateBlobBodyMemory(device_ctx));
      return Maybe<void>::Ok();
    }));
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> TryAllocateTempStorageBlobMemory(
      LocalCallOpKernelPhyInstrOperand* operand, DeviceCtx* device_ctx) {
    JUST(operand->mut_opkernel()->mut_temp_blob_object()->TryAllocateBlobBodyMemory(device_ctx));
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> OpKernelCompute(LocalCallOpKernelPhyInstrOperand* operand,
                                            DeviceCtx* device_ctx, user_op::OpKernelState* state) {
    JUST(WithComputeContext(operand, device_ctx,
                            [&](user_op::KernelComputeContext* compute_ctx) -> Maybe<void> {
                              operand->user_opkernel()->Compute(compute_ctx, state);
                              return Maybe<void>::Ok();
                            }));

    if (oneflow::DTRDebugLevel() >= 3) {
      for (int i : operand->opkernel().input_tuple_indexes4mut_ibns()) {
        const std::string& op_type_name = operand->opkernel().op_type_name();
        std::cout << "mutable! op: " << op_type_name << ", input " << i;
        const auto& mut_input = operand->inputs()->at(i);
        if (mut_input->mem_case().has_device_cuda_mem()) {
          size_t bytes = mut_input->blob_desc().ByteSizeOfBlobBody();
          std::vector<float> tmp(bytes / 4);
          cudaMemcpy(tmp.data(), mut_input->blob().dptr(), bytes,
                     cudaMemcpyKind::cudaMemcpyDeviceToHost);
          float x = 0;
          for (float f : tmp) { x += f; }
          mut_input->hash_ = x;
          mut_input->backup_data_.resize(bytes / 4);
          memcpy(mut_input->backup_data_.data(), tmp.data(), bytes);
          std::cout << ", gpu memory." << std::endl;
        } else {
          std::cout << ", non gpu memory." << std::endl;
        }
      }

      // compare_input_hash flag
      bool compare_input_hash = false;
      for (const auto& base_class_output : *operand->outputs()) {
        if (base_class_output->mem_case().has_device_cuda_mem()) {
          size_t bytes = base_class_output->blob_desc().ByteSizeOfBlobBody();
          CHECK_EQ_OR_RETURN(bytes % 4, 0);
          std::vector<float> tmp(bytes / 4);
          cudaMemcpy(tmp.data(), base_class_output->blob().dptr(), bytes,
                     cudaMemcpyKind::cudaMemcpyDeviceToHost);
          float x = 0;
          for (float f : tmp) { x += f; }
          if (const auto output =
                  std::dynamic_pointer_cast<DTREagerBlobObject>(base_class_output)) {
            if (output->hash_ != -1) {
              if (output->hash_ != x) {
                std::cout << "wrong!!!!"
                          << " compute op: "
                          << output->compute_op()->shared_opkernel()->user_op_conf_->op_type_name()
                          << ", old hash: " << output->hash_ << ", new hash: " << x
                          << ", old data[0]: " << output->backup_data_[0]
                          << ", new data[0]: " << tmp[0]
                          << ", shape: " << output->blob_desc().shape() << std::endl;

                // compare hash of inputs
                compare_input_hash = true;
              } else {
                std::cout << "correct :)"
                          << " compute op: "
                          << output->compute_op()->shared_opkernel()->user_op_conf_->op_type_name()
                          << ", old hash: " << output->hash_ << ", new hash: " << x << std::endl;
              }
            } else {
              std::cout << "first! set hash to " << x << std::endl;
            }
          }
          base_class_output->hash_ = x;
          base_class_output->backup_data_.resize(bytes / 4);
          memcpy(base_class_output->backup_data_.data(), tmp.data(), bytes);
        } else {
          std::cout << "compute non gpu memory, op is: " << operand->opkernel().op_type_name()
                    << std::endl;
        }
      }
      if (compare_input_hash) {
        for (const auto& base_class_input : *operand->inputs()) {
          if (const auto input = std::dynamic_pointer_cast<DTREagerBlobObject>(base_class_input)) {
            if (input->mem_case().has_device_cuda_mem()) {
              size_t bytes = input->blob_desc().ByteSizeOfBlobBody();
              CHECK_EQ_OR_RETURN(bytes % 4, 0);
              std::vector<float> tmp(bytes / 4);
              cudaMemcpy(tmp.data(), input->blob().dptr(), bytes,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost);
              float x = 0;
              for (float f : tmp) { x += f; }
              if (input->hash_ != -1) {
                if (input->hash_ != x) {
                  std::cout << "input hash wrong!!!!"
                            << ", old hash: " << input->hash_ << ", new hash: " << x
                            << ", old data[0]: " << input->backup_data_[0]
                            << ", new data[0]: " << tmp[0]
                            << ", shape: " << input->blob_desc().shape() << std::endl;
                } else {
                  std::cout << "input hash correct :)"
                            << ", shape: " << input->blob_desc().shape() << std::endl;
                }
              } else {
                std::cout << "input not initialized!!!!!" << x << std::endl;
              }
            } else {
              std::cout << "input non gpu memory, op is: " << operand->opkernel().op_type_name()
                        << std::endl;
            }
          }
        }
      }
    }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> DeallocateTempStorageBlobMemory(
      LocalCallOpKernelPhyInstrOperand* operand, DeviceCtx* device_ctx) {
    JUST(operand->mut_opkernel()->mut_temp_blob_object()->DeallocateBlobDataPtr());
    return Maybe<void>::Ok();
  }
};

struct EagerLocalCallOpKernelUtil final : public LocalCallOpKernelUtil {
  static inline Maybe<void> Prepare(vm::Instruction* instruction) { return Maybe<void>::Ok(); }

  static inline Maybe<void> DisplayCount(vm::Instruction* instruction) {
    auto operand = JUST(GetSharedLocalCallOpKernelPhyInstrOperand(instruction));
    std::cout << "======================== Display input blobobject shared_ptr's count "
                 "========================"
              << std::endl;
    size_t input_id = 0;
    for (const auto& input : *operand->inputs()) {
      std::cout << input_id++ << "th input shared_ptr's count: " << input.use_count() << std::endl;
    }
    std::cout << "======================== Display output blobobject shared_ptr's count "
                 "========================"
              << std::endl;
    size_t output_id = 0;
    for (const auto& output : *operand->outputs()) {
      std::cout << output_id++ << "th output shared_ptr's count: " << output.use_count()
                << std::endl;
    }
    return Maybe<void>::Ok();
  }
};

one::EagerBlobObjectListPtr global_pinned_ebos = nullptr;

struct PinGuard {
  OF_DISALLOW_COPY_AND_MOVE(PinGuard);
  explicit PinGuard(const one::EagerBlobObjectListPtr& ebos)
      : ebos_(ebos), old_ebos_(global_pinned_ebos) {
    if (old_ebos_ != nullptr) {
      for (auto& ebo : *old_ebos_) {
        if (auto dtr_ebo = std::dynamic_pointer_cast<DTREagerBlobObject>(ebo)) {
          dtr_ebo->unpin();
        } else {
          CHECK(false);
        }
      }
    }
    for (auto& ebo : *ebos_) {
      if (auto dtr_ebo = std::dynamic_pointer_cast<DTREagerBlobObject>(ebo)) {
        dtr_ebo->pin();
      } else {
        CHECK(false);
      }
    }
    global_pinned_ebos = ebos_;
  }
  ~PinGuard() {
    if (old_ebos_ != nullptr) {
      for (auto& ebo : *old_ebos_) {
        if (auto dtr_ebo = std::dynamic_pointer_cast<DTREagerBlobObject>(ebo)) {
          dtr_ebo->pin();
        } else {
          CHECK(false);
        }
      }
    }
    for (auto& ebo : *ebos_) {
      if (auto dtr_ebo = std::dynamic_pointer_cast<DTREagerBlobObject>(ebo)) {
        dtr_ebo->unpin();
      } else {
        CHECK(false);
      }
    }
    global_pinned_ebos = old_ebos_;
  }

 private:
  one::EagerBlobObjectListPtr ebos_;
  one::EagerBlobObjectListPtr old_ebos_;
};

struct DTRLocalCallOpKernelUtil final : public LocalCallOpKernelUtil {
  static inline Maybe<void> Prepare(vm::Instruction* instruction) {
    auto operand = JUST(GetSharedLocalCallOpKernelPhyInstrOperand(instruction));
    if (oneflow::DTRDebugEnabled()) {
      std::cout << "prepare start for " << operand->opkernel().op_type_name() << std::endl;
    }
    auto& stream = instruction->stream();
    JUST(
        ForEachDTRInputTensor(operand, [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          // pin inputs
          dtr_blob_object->pin();
          return Maybe<void>::Ok();
        }));
    // PinGuard guard(operand->inputs());
    JUST(
        ForEachDTRInputTensor(operand, [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          if (!dtr_blob_object->is_in_memory()) {
            CHECK_GT_OR_RETURN(dtr_blob_object->input_size(), 0);
            // TODO: recursive recompute the inputs
            JUST(recompute(dtr_blob_object, stream));
          }
          dtr_blob_object->update_access_time();
          dtr_blob_object->update_user_ops(operand);
          return Maybe<void>::Ok();
        }));

    if (oneflow::DTRDebugEnabled()) {
      std::cout << "prepare ok for " << operand->opkernel().op_type_name() << std::endl;
      std::cout << "===============================" << std::endl;
    }

    for (int i : operand->opkernel().input_tuple_indexes4mut_ibns()) {
      const auto& mut_input = operand->inputs()->at(i);
      if (const auto dtr_input = std::dynamic_pointer_cast<DTREagerBlobObject>(mut_input)) {
        dtr_input->set_evict_attr(false);
      }
    }

    return Maybe<void>::Ok();
  }

  static inline Maybe<void> InitOutputBlobAttrs(vm::Instruction* instruction) {
    auto operand = JUST(GetSharedLocalCallOpKernelPhyInstrOperand(instruction));
    JUST(ForEachDTROutputTensor(
        operand,
        [&](const std::shared_ptr<vm::DTREagerBlobObject>& dtr_blob_object) -> Maybe<void> {
          JUST(dtr_blob_object->InitBlobAttrs(operand));
          return Maybe<void>::Ok();
        }));
    return Maybe<void>::Ok();
  }

  static Maybe<double> GetEstimatedComputeTime(std::shared_ptr<oneflow::vm::LocalCallOpKernelPhyInstrOperand> operand) {
    size_t estimated_compute_time = 0;
    JUST(
        ForEachDTRInputTensor(operand, [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          estimated_compute_time += dtr_blob_object->BlobBodyBytes();
          return Maybe<void>::Ok();
        }));
    JUST(
        ForEachDTROutputTensor(operand, [&](const std::shared_ptr<vm::DTREagerBlobObject>& dtr_blob_object) -> Maybe<void> {
          estimated_compute_time += dtr_blob_object->BlobBodyBytes();
          return Maybe<void>::Ok();
        }));

    return estimated_compute_time;
  }

  static inline Maybe<void> UpdateTensorInfo(vm::Instruction* instruction,
                                             double compute_time = -1.0) {
    auto operand = JUST(GetSharedLocalCallOpKernelPhyInstrOperand(instruction));

    const double estimated_compute_time = JUST(GetEstimatedComputeTime(operand));
    JUST(
        ForEachDTRInputTensor(operand, [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          // unpin inputs
          dtr_blob_object->unpin();
          return Maybe<void>::Ok();
        }));
    Global<one::DTRTensorPool>::Get()->time_flies(estimated_compute_time);

    // find in_place op and do sth
    bool is_in_place = false;
    auto* op = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand.get());
    CHECK_NOTNULL_OR_RETURN(op);
    for (const auto& input : *op->inputs()) {
      CHECK_OR_RETURN(static_cast<bool>(input.get()));
      auto in_dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(input.get());
      CHECK_NOTNULL_OR_RETURN(in_dtr_blob_object);
      for (const auto& output : *op->outputs()) {
        CHECK_OR_RETURN(static_cast<bool>(output.get()));
        auto out_dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(output.get());
        CHECK_NOTNULL_OR_RETURN(out_dtr_blob_object);
        if (in_dtr_blob_object->object_dptr() == out_dtr_blob_object->object_dptr()) {
          is_in_place = true;
          break;
        }
      }
      if (is_in_place) {
        break;
      }
    }

    JUST(ForEachDTROutputTensor(
        operand,
        [&](const std::shared_ptr<vm::DTREagerBlobObject>& dtr_blob_object) -> Maybe<void> {
          // dtr_blob_object->set_compute_time(estimated_compute_time);
          dtr_blob_object->set_compute_time(dtr_blob_object->blob_body_bytes_double());
          // Condition - insert current blob into candidates only when blob memory > threshold (with
          // default 0)
          dtr_blob_object->reset_node(estimated_compute_time);
          if (is_in_place) {
            dtr_blob_object->set_evict_attr(false);
          } else {
            JUST(Global<one::DTRTensorPool>::Get()->insert(dtr_blob_object));
          }
          return Maybe<void>::Ok();
        }));

    if (oneflow::DTRDebugLevel() >= 3) {
      JUST(Global<one::DTRTensorPool>::Get()->display2());
    }

    // Display info of current tensor pool
    // if (oneflow::DTRDebugEnabled()) { JUST(Global<one::DTRTensorPool>::Get()->display()); }

    // // Display output shared_ptr's count
    // std::cout << "======================== Display output dtrblobobject shared_ptr's count
    // ========================" << std::endl; size_t output_id = 0; for (const auto& output :
    // *operand->outputs()) {
    //   std::cout << output_id++ << "th output shared_ptr's count: " << output.use_count() <<
    //   std::endl;
    // }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> recompute(vm::DTREagerBlobObject* object, const vm::Stream& stream) {
    if (oneflow::DTRDebugEnabled()) {
      std::cout << "going to recompute "
                << object->compute_op()->shared_opkernel()->user_op_conf_->op_type_name() << " for "
                << object << ", whose dptr is " << object->blob().dptr()
                << ", is in memory: " << object->is_in_memory() << std::endl;
    }
    auto unique_op = DTROp2LocalCallOp(object->compute_op());
    CHECK_NOTNULL_OR_RETURN(unique_op);
    DeviceCtx* device_ctx = stream.device_ctx().get();

    // pin inputs
    // TODO for each ptr rather than shared_ptr
    auto* operand = unique_op.get();
    // PinGuard guard(operand->inputs());
    JUST(
        ForEachDTRInputTensor(operand, [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          dtr_blob_object->pin();
          return Maybe<void>::Ok();
        }));

    // recompute inputs not in memory
    JUST(
        ForEachDTRInputTensor(operand, [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          if (!dtr_blob_object->is_in_memory()) {
            if (dtr_blob_object->input_size() == 0) {
              std::cout << dtr_blob_object << std::endl;
            }
            CHECK_GT_OR_RETURN(dtr_blob_object->input_size(), 0);
            JUST(recompute(dtr_blob_object, stream));
          }
          dtr_blob_object->update_access_time();
          return Maybe<void>::Ok();
        }));

    // TODO: execute function, update outputs, if execute failure (OOM), evict()
    // auto* ptr = dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand.get());
    CHECK_NOTNULL_OR_RETURN(operand);
    JUST(DoInfer(operand, stream));
    JUST(FullCompute(operand, device_ctx));

    CHECK_GT_OR_RETURN(object->compute_time(), 0);
    Global<one::DTRTensorPool>::Get()->time_flies(object->compute_time());
    // JUST(Global<one::DTRTensorPool>::Get()->time_flies(GetEstimatedComputeTime(instruction)));
    if (oneflow::DTRDebugEnabled()) {
      // if (oneflow::DTRDebugEnabled() || !object->is_in_memory()) {
      CHECK_OR_RETURN(object->is_in_memory());
    }

    // unpin inputs
    JUST(
        ForEachDTRInputTensor(operand, [&](vm::DTREagerBlobObject* dtr_blob_object) -> Maybe<void> {
          dtr_blob_object->unpin();
          return Maybe<void>::Ok();
        }));

    Global<one::DTRTensorPool>::Get()->add_recompute_times();
    Global<one::DTRTensorPool>::Get()->update_after_recompute(object);
    return Maybe<void>::Ok();
  }

  // static inline Maybe<void> evict(vm::DTREagerBlobObject* object) {
  //   object->evict();
  //   return Maybe<void>::Ok();
  // }
};

Maybe<void> DTRUtil::recompute(vm::DTREagerBlobObject* object, const vm::Stream& stream) {
  if (object->is_in_memory()) { return Maybe<void>::Ok(); }
  return DTRLocalCallOpKernelUtil::recompute(object, stream);
}

void LocalCallOpKernelInstructionType::Infer(vm::Instruction* instruction) const {
  UNIMPLEMENTED();
}

void LocalCallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  if (oneflow::DTREnabled()) {
    CHECK_OK(DTRLocalCallOpKernelUtil::Prepare(instruction));
    CHECK_OK(DTRLocalCallOpKernelUtil::Infer(instruction));
    CHECK_OK(DTRLocalCallOpKernelUtil::InitOutputBlobAttrs(instruction));
    CHECK_OK(DTRLocalCallOpKernelUtil::Compute(instruction));
    CHECK_OK(DTRLocalCallOpKernelUtil::UpdateTensorInfo(instruction));
    auto operand =
        CHECK_JUST(LocalCallOpKernelUtil::GetSharedLocalCallOpKernelPhyInstrOperand(instruction));
    if (oneflow::DTRDebugLevel() >= 1) {
      std::cout << "all compute ok for " << operand->opkernel().op_type_name() << std::endl;
    }
  } else {
    CHECK_OK(EagerLocalCallOpKernelUtil::Prepare(instruction));
    CHECK_OK(EagerLocalCallOpKernelUtil::Infer(instruction));
    CHECK_OK(EagerLocalCallOpKernelUtil::Compute(instruction));
    // CHECK_OK(EagerLocalCallOpKernelUtil::DisplayCount(instruction));
  }
}

const std::string& LocalCallOpKernelInstructionType::DebugOpTypeName(
    vm::Instruction* instruction) const {
  auto* operand =
      CHECK_JUST(LocalCallOpKernelUtil::GetLocalCallOpKernelPhyInstrOperand(instruction));
  return operand->opkernel().op_type_name();
}

Maybe<void> CallOpKernelInstructionType::MaybeInfer(vm::Instruction* instruction,
                                                    const CallOpKernelInstrOperand& args) const {
  auto* opkernel_obj = JUST(instruction->mut_operand_type(args.opkernel())->Mut<OpKernelObject>());
  DeviceType device_type = JUST(DeviceType4DeviceTag(this->device_tag()));
  int64_t device_id = instruction->stream().device_id();
  const auto& mem_case = MemoryCaseUtil::MakeMemCase(device_type, device_id);
  JUST(OpKernelInfer(opkernel_obj, instruction, args, mem_case));
  return Maybe<void>::Ok();
}

void CallOpKernelInstructionType::Infer(vm::Instruction* instruction) const {
  FlatMsgView<CallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  CHECK_OK(MaybeInfer(instruction, args.Get()))
      << "\ndevice_tag: " << device_tag() << "\nmachine_id: " << instruction->stream().machine_id()
      << "\ndevice_id: " << instruction->stream().device_id()
      << "\n============ parallel_conf ============\n"
      << instruction->parallel_desc()->parallel_conf().DebugString();
}

Maybe<void> CallOpKernelInstructionType::MaybeCompute(vm::Instruction* instruction,
                                                      const CallOpKernelInstrOperand& args) const {
  auto* opkernel_obj = JUST(instruction->mut_operand_type(args.opkernel())->Mut<OpKernelObject>());
  JUST(OpKernelCompute(opkernel_obj, instruction, args));
  return Maybe<void>::Ok();
}

void CallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  FlatMsgView<CallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  CHECK_OK(MaybeCompute(instruction, args.Get()))
      << "\ndevice_tag: " << device_tag() << "\nmachine_id: " << instruction->stream().machine_id()
      << "\ndevice_id: " << instruction->stream().device_id()
      << "\n============ parallel_conf ============\n"
      << instruction->parallel_desc()->parallel_conf().DebugString();
}

Maybe<const OperatorConf&> GetOpConf(vm::Instruction* instruction,
                                     const StatelessCallOpKernelInstrOperand& args) {
  const auto* operand_op_conf = instruction->operand_type(args.op_conf());
  CHECK_NOTNULL_OR_RETURN(operand_op_conf);
  return JUST(operand_op_conf->Get<vm::ObjectWrapper<OperatorConfSymbol>>()).Get().op_conf();
}

Maybe<void> UserStatelessCallOpKernelInstructionType::Infer(
    vm::Instruction* instruction, const StatelessCallOpKernelInstrOperand& args) const {
  DeviceType device_type = JUST(DeviceType4DeviceTag(this->device_tag()));
  int64_t device_id = instruction->stream().device_id();
  auto* opkernel = JUST(GetSharedOpKernel<OpKernelObject>(instruction, device_type, args));
  const auto& mem_case = MemoryCaseUtil::MakeMemCase(device_type, device_id);
  JUST(OpKernelInfer(opkernel, instruction, args, mem_case));
  return Maybe<void>::Ok();
}

void UserStatelessCallOpKernelInstructionType::Infer(vm::Instruction* instruction) const {
  FlatMsgView<StatelessCallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  CHECK_OK(Infer(instruction, args.Get()))
      << "\nmachine_id: " << instruction->stream().machine_id()
      << "\ndevice_id: " << instruction->stream().device_id()
      << "\n============ parallel_conf ============\n"
      << instruction->parallel_desc()->parallel_conf().DebugString()
      << "\n============ op_conf ============\n"
      << CHECK_JUST(GetOpConf(instruction, args.Get())).DebugString();
}

Maybe<void> UserStatelessCallOpKernelInstructionType::Compute(
    vm::Instruction* instruction, const StatelessCallOpKernelInstrOperand& args) const {
  auto* opkernel_obj =
      JUST(instruction->mut_operand_type(args.shared_opkernel())->Mut<OpKernelObject>());
  JUST(OpKernelCompute(opkernel_obj, instruction, args));
  return Maybe<void>::Ok();
}

void UserStatelessCallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  FlatMsgView<StatelessCallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  CHECK_OK(Compute(instruction, args.Get()))
      << "\nmachine_id: " << instruction->stream().machine_id()
      << "\ndevice_id: " << instruction->stream().device_id()
      << "\n============ parallel_conf ============\n"
      << instruction->parallel_desc()->parallel_conf().DebugString()
      << "\n============ op_conf ============\n"
      << CHECK_JUST(GetOpConf(instruction, args.Get())).DebugString();
}

std::shared_ptr<MemoryCase> SystemStatelessCallOpKernelInstructionType::GetOutBlobMemCase(
    const DeviceType device_type, const int64_t device_id) const {
  return MemoryCaseUtil::MakeMemCase(device_type, device_id);
}

Maybe<void> SystemStatelessCallOpKernelInstructionType::Infer(
    vm::Instruction* instruction, const StatelessCallOpKernelInstrOperand& args) const {
  DeviceType device_type = JUST(DeviceType4DeviceTag(this->device_tag()));
  int64_t device_id = instruction->stream().device_id();
  auto* opkernel = JUST(GetSharedOpKernel<SystemOpKernelObject>(instruction, device_type, args));
  const auto& mem_case = GetOutBlobMemCase(device_type, device_id);
  JUST(OpKernelInfer(opkernel, instruction, args, mem_case));
  return Maybe<void>::Ok();
}

void SystemStatelessCallOpKernelInstructionType::Infer(vm::Instruction* instruction) const {
  FlatMsgView<StatelessCallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  CHECK_OK(Infer(instruction, args.Get()))
      << "\nmachine_id: " << instruction->stream().machine_id()
      << "\ndevice_id: " << instruction->stream().device_id()
      << "\n============ parallel_conf ============\n"
      << instruction->parallel_desc()->parallel_conf().DebugString()
      << "\n============ op_conf ============\n"
      << CHECK_JUST(GetOpConf(instruction, args.Get())).DebugString();
}

Maybe<void> SystemStatelessCallOpKernelInstructionType::Compute(
    vm::Instruction* instruction, const StatelessCallOpKernelInstrOperand& args) const {
  auto* opkernel_obj =
      JUST(instruction->mut_operand_type(args.shared_opkernel())->Mut<SystemOpKernelObject>());
  JUST(OpKernelCompute(opkernel_obj, instruction, args));
  return Maybe<void>::Ok();
}

void SystemStatelessCallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  FlatMsgView<StatelessCallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  CHECK_OK(Compute(instruction, args.Get()))
      << "\nmachine_id: " << instruction->stream().machine_id()
      << "\ndevice_id: " << instruction->stream().device_id()
      << "\n============ parallel_conf ============\n"
      << instruction->parallel_desc()->parallel_conf().DebugString()
      << "\n============ op_conf ============\n"
      << CHECK_JUST(GetOpConf(instruction, args.Get())).DebugString();
}

template<typename T>
void FeedOrFetchBlob(vm::Instruction* instruction) {
  FlatMsgView<T> args(instruction->instr_msg().operand());
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
  auto* rw_mutext_blob = instruction->mut_operand_type(args->blob());
  auto* blob_object = CHECK_JUST(rw_mutext_blob->template Mut<BlobObject>());
  OfBlob of_blob(device_ctx, blob_object->mut_blob());
  int64_t of_blob_ptr = reinterpret_cast<int64_t>(&of_blob);
  (*Global<std::shared_ptr<ForeignCallback>>::Get())
      ->OfBlobCall(args->unique_callback_id(), of_blob_ptr);
}

void FetchBlobHeaderInstructionType::Infer(vm::Instruction* instruction) const {
  FeedOrFetchBlob<FetchBlobInstrOperand>(instruction);
}

void FetchBlobBodyInstructionType::Compute(vm::Instruction* instruction) const {
  FeedOrFetchBlob<FetchBlobInstrOperand>(instruction);
}

void FeedBlobInstructionType::Compute(vm::Instruction* instruction) const {
  FeedOrFetchBlob<FeedBlobInstrOperand>(instruction);
}

}  // namespace vm
}  // namespace oneflow
