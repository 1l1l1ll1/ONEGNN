#include "oneflow/core/actor/accumulate_actor.h"

namespace oneflow {

void AccumulateActor::Init(const TaskProto& task_proto,
                           const ThreadCtx& thread_ctx, int32_t max_acc_cnt) {
  if (JobDesc::Singleton()->GetDeviceType() == DeviceType::kCPU) {
    MemsetFunc = &Memset<DeviceType::kCPU>;
    mut_device_ctx().reset(new CpuDeviceCtx);
  } else {
    MemsetFunc = &Memset<DeviceType::kGPU>;
    mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
  }
  OF_SET_MSG_HANDLER(&AccumulateActor::HandlerNormal);
  acc_cnt_ = max_acc_cnt;
  max_acc_cnt_ = max_acc_cnt;
  next_acc_piece_id_ = 0;
}

int AccumulateActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    // CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessOneEord();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      waiting_in_regst_.push(regst);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int AccumulateActor::HandlerUntilReadAlwaysUnReady(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  ActUntilFail();
  if (waiting_in_regst_.empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&AccumulateActor::HandlerZombie);
  }
  return 0;
}

void AccumulateActor::Act() {
  Regst* in_regst = waiting_in_regst_.front();
  KernelCtx ctx = GenDefaultKernelCtx();
  // ForEachCurWriteableRegst([&](Regst* regst) {
  //  if (acc_cnt_ != max_acc_cnt_) { return; }
  //  Blob* packed_blob = regst->GetBlobPtrFromLbn(kPackedBlobName);
  //  MemsetFunc(ctx.device_ctx, packed_blob->mut_dptr(), 0,
  //             packed_blob->TotalByteSize());
  //  acc_cnt_ = 0;
  //});
  AsyncLaunchKernel(ctx, [this](uint64_t regst_desc_id) -> Regst* {
    Regst* regst = GetCurWriteableRegst(regst_desc_id);
    if (regst == nullptr) {
      CHECK_EQ(regst_desc_id, waiting_in_regst_.front()->regst_desc_id());
      return waiting_in_regst_.front();
    } else {
      return regst;
    }
  });
  acc_cnt_ += 1;
  if (acc_cnt_ == max_acc_cnt_) {
    AsyncSendRegstMsgToConsumer([&](Regst* acc_regst) {
      acc_regst->set_piece_id(next_acc_piece_id_++);
    });
  }
  AsyncSendRegstMsgToProducer(in_regst);
  waiting_in_regst_.pop();
}

}  // namespace oneflow
