#include "oneflow/core/actor/copy_comm_net_actor.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

class CopyCommNetActor::CommNetDeviceCtx final : public DeviceCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(CommNetDeviceCtx);
  CommNetDeviceCtx() = delete;
  ~CommNetDeviceCtx() = default;

  CommNetDeviceCtx(void* actor_read_id)
      : DeviceCtx(), actor_read_id_(actor_read_id), read_id_(nullptr) {}

  void AddCallBack(std::function<void()> callback) const override {
    CommNet::Singleton()->AddReadCallBack(actor_read_id_, read_id_, callback);
  }

  void set_read_id(void* val) { read_id_ = val; }

 private:
  void* actor_read_id_;
  void* read_id_;
};

CopyCommNetActor::~CopyCommNetActor() {
  CommNet::Singleton()->DeleteActorReadId(actor_read_id_);
}

void CopyCommNetActor::VirtualActorInit(const TaskProto& task_proto) {
  actor_read_id_ = CommNet::Singleton()->NewActorReadId();
  comm_net_device_ctx_ = new CommNetDeviceCtx(actor_read_id_);
  mut_device_ctx().reset(comm_net_device_ctx_);
  OF_SET_MSG_HANDLER(&CopyCommNetActor::HandlerNormal);
}

int CopyCommNetActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    // CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessOneEord();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (msg.SrcMachineId() == RuntimeCtx::Singleton()->this_machine_id()) {
      CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
    } else {
      RegstCtx regst_ctx;
      regst_ctx.comm_net_token = msg.comm_net_token();
      regst_ctx.regst_raw_ptr = msg.regst();
      regst_ctx.producer = msg.src_actor_id();
      CHECK(piece_id2regst_ctx.emplace(msg.piece_id(), regst_ctx).second);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int CopyCommNetActor::HandlerUntilReadAlwaysUnReady(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  ActUntilFail();
  if (piece_id2regst_ctx.empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&CopyCommNetActor::HandlerZombie);
  }
  return 0;
}

void CopyCommNetActor::Act() {
  int64_t cur_piece_id = -1;  // expected_piece_id();
  // readable
  auto readable_it = piece_id2regst_ctx.find(cur_piece_id);
  const void* readable_token = readable_it->second.comm_net_token;
  Regst* readable_regst = readable_it->second.regst_raw_ptr;
  int64_t src_machine_id =
      IDMgr::Singleton()->MachineId4ActorId(readable_it->second.producer);
  // writeable
  Regst* writeable_regst = GetCurSoleWriteableRegst();
  const void* writeable_token =
      writeable_regst->packed_blob()->comm_net_token();
  // Async
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  void* read_id = nullptr;
  auto other_val = std::make_tuple(actor_read_id_, &read_id, src_machine_id,
                                   readable_token, writeable_token);
  kernel_ctx.other = &other_val;
  // AsyncLaunchKernel(kernel_ctx);
  comm_net_device_ctx_->set_read_id(read_id);
  AsyncSendRegstMsgToConsumer(
      [&](Regst* regst) { regst->set_piece_id(cur_piece_id); });
  AsyncSendRegstMsgToProducer(readable_regst, readable_it->second.producer);
  comm_net_device_ctx_->set_read_id(nullptr);
  CommNet::Singleton()->AddReadCallBackDone(actor_read_id_, read_id);
  // Finish
  piece_id2regst_ctx.erase(readable_it);
}

}  // namespace oneflow
