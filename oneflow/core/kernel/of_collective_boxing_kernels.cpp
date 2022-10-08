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
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/job/of_collective_boxing/collective_manager.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/graph/boxing/of_collective_boxing_util.h"
#include "oneflow/core/lazy/actor/of_collective_boxing_actor_context.h"

namespace oneflow {

using namespace boxing::of_collective;

namespace {

OfCollectiveBoxingActorContext* GetOfCollectiveBoxingActorContext(KernelContext* kernel_ctx) {
  auto* actor_context_provider = CHECK_NOTNULL(dynamic_cast<ActorContextProvider*>(kernel_ctx));
  return CHECK_NOTNULL(
      dynamic_cast<OfCollectiveBoxingActorContext*>(actor_context_provider->GetActorContext()));
}

class OfCollectiveBoxingKernelState final : public KernelState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingKernelState);
  explicit OfCollectiveBoxingKernelState(const RankDesc& rank_desc)
      : coll_id_(Singleton<CollectiveMgr>::Get()->KernelGetCollId(rank_desc)),
        ofccl_rank_ctx_(Singleton<CollectiveMgr>::Get()->KernelGetOfcclRankCtx(rank_desc.rank())) {}
  ~OfCollectiveBoxingKernelState() = default;

  int coll_id() { return coll_id_; }
  ofcclRankCtx_t ofccl_rank_ctx() { return ofccl_rank_ctx_; }

 private:
  int coll_id_;
  ofcclRankCtx_t ofccl_rank_ctx_;
};

class OfCollectiveBoxingGenericKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingGenericKernel);
  OfCollectiveBoxingGenericKernel() = default;
  ~OfCollectiveBoxingGenericKernel() override = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override;
  //   bool IsKernelLaunchSynchronized() const override { return false; }
  void ForwardDataContent(KernelContext* ctx) const override;
};

void OfCollectiveBoxingGenericKernel::VirtualKernelInit(KernelContext* ctx) {
  const RankDesc& rank_desc = this->op_conf().of_collective_boxing_generic_conf().rank_desc();
  ctx->set_state(std::make_shared<OfCollectiveBoxingKernelState>(rank_desc));
}

void OfCollectiveBoxingGenericKernel::ForwardDataContent(KernelContext* ctx) const {
  VLOG(1) << "Enter OfCollectiveBoxingGenericKernel::ForwardDataContent";
  // Blob* in = ctx->BnInOp2Blob("in");
  // Blob* out = ctx->BnInOp2Blob("out");
  // AutoMemcpy(ctx->stream(), out, in);

  const void* send_buff = nullptr;
  void* recv_buff = nullptr;
  const RankDesc& rank_desc = this->op_conf().of_collective_boxing_generic_conf().rank_desc();
  const DataType data_type = rank_desc.op_desc().data_type();
  if (GenericOpHasInput(rank_desc)) {
    const Blob* in = ctx->BnInOp2Blob("in");
    CHECK_EQ(in->data_type(), data_type);
    CHECK(in->shape() == ShapeView(GenericOpGetInputShape(rank_desc)));
    send_buff = in->dptr();
  }
  if (GenericOpHasOutput(rank_desc)) {
    Blob* out = ctx->BnInOp2Blob("out");
    CHECK_EQ(out->data_type(), data_type);
    CHECK(out->shape() == ShapeView(GenericOpGetOutputShape(rank_desc)));
    recv_buff = out->mut_dptr();
  }

  VLOG(1) << "OfCollectiveBoxingGenericKernel::ForwardDataContent Done" << send_buff << recv_buff;
}

REGISTER_KERNEL(OperatorConf::kOfCollectiveBoxingGenericConf, OfCollectiveBoxingGenericKernel);

}  // namespace

}  // namespace oneflow
