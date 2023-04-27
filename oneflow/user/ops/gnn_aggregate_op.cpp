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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

Maybe<void> InferTensorDesc4(user_op::InferContext* ctx) {
  //检查w_self和h_k的维度是否相同,w_self和h_k必须均为m*m维度
  const user_op::TensorDesc& w_self = ctx->InputTensorDesc("w_self", 0);
  const user_op::TensorDesc& h_k = ctx->InputTensorDesc("h_k", 0);
  CHECK_EQ_OR_RETURN(w_self.shape().NumAxes(), h_k.shape().NumAxes());
  
  //检查向量b的维度
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  int64_t m = w_self.shape().At(0);
  int64_t k = w_self.shape().At(1);
  CHECK_EQ_OR_RETURN(k, b.shape().At(0)) << "Dim K should be equal to vector b's dim0. ";

  //推导输出张量形状
  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  Shape output = ctx->InputShape("w_self", 0);
  // ctx->SetOutputIsDynamic("y", 0, ctx->InputIsDynamic("w_self", 0));
  output.Set(0, m);
  output.Set(1, m);
  y->set_shape(output);

  return Maybe<void>::Ok();
}

// 验证输入张量形状，推导输出张量形状
/* static */Maybe<void> GnnAggregateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4(ctx);
}

Maybe<void> GnnAggregateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

// Split, Broadcast, 
Maybe<void> GnnAggregateOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("w_self", 0), 0)
      .Broadcast(user_op::OpArg("h_k", 0))
      .Broadcast(user_op::OpArg("b", 0))
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> GnnAggregateOp::InferDataType(user_op::InferContext* ctx) {

  DataType dtype = ctx->InputDType("w_self", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), dtype)
      << "InferDataType Failed. Expected " << DataType_Name(dtype) << ", but got "
      << DataType_Name(ctx->InputDType("b", 0));
  ctx->SetOutputDType("y", 0, dtype);
  return Maybe<void>::Ok();
}

}  // namespace oneflow