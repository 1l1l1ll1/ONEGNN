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

/* static */ Maybe<void> LogSoftmaxOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("prob", 0, ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LogSoftmaxOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, axis, 0, in_tensor.shape().NumAxes() - 1) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), axis)
        .Split(user_op::OpArg("prob", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LogSoftmaxOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("prob", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LogSoftmaxGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& y_shape = ctx->InputShape("prob", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_OR_RETURN(dy_shape == y_shape);
  ctx->SetOutputShape("dx", 0, dy_shape);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LogSoftmaxGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& y_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("prob", 0);
  FOR_RANGE(int64_t, axis, 0, y_tensor.shape().NumAxes() - 1) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("prob", 0), axis)
        .Split(user_op::OpArg("dy", 0), axis)
        .Split(user_op::OpArg("dx", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> LogSoftmaxGradOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("prob", 0), ctx->InputDType("dy", 0))
      << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("dy", 0))
      << ", but got " << DataType_Name(ctx->InputDType("prob", 0));
  ctx->SetOutputDType("dx", 0, ctx->InputDType("prob", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
