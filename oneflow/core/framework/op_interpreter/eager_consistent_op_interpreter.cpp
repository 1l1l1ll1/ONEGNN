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

#include <sstream>
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/consistent_tensor_infer_cache.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"
#include "oneflow/core/framework/consistency_check.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/tensor_consistent_id.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/boxing/eager_boxing_logger.h"
#include "oneflow/core/common/cpp_attribute.h"

namespace oneflow {
namespace one {

namespace {

Maybe<Symbol<ParallelDesc>> GetParallelDesc(const TensorTuple& inputs,
                                            const OpExprInterpContext& ctx) {
  if (!inputs.empty()) { return inputs.at(0)->parallel_desc(); }
  return JUST(ctx.parallel_desc);
}

std::string GetDynamicOpConsistentFailedDebugString(const UserOpExpr& user_op_expr,
                                                    const StatefulLocalOpKernel& kernel) {
  CHECK(!kernel.output_tuple_indexes4mut2_obns().empty());
  std::string plentysuffix = kernel.output_tuple_indexes4mut2_obns().size() == 1 ? "s" : "";
  std::stringstream ss;
  ss << "operator `" << user_op_expr.op_type_name() << "`"
     << " does not support consistent mode because the shape" << plentysuffix << " of output tensor"
     << plentysuffix << " ";
  int i = 0;
  for (const auto& out_index : kernel.output_tuple_indexes4mut2_obns()) {
    if (i++ > 0) { ss << ", "; }
    ss << out_index;
  }
  ss << " are not infered before op computation.";
  return ss.str();
}

Maybe<bool> IsAllZeroSizeTensorMeta(const std::vector<Symbol<ConsistentTensorMeta>>& tensor_metas) {
  if (tensor_metas.empty()) { return false; }
  for (const auto& tensor_meta : tensor_metas) {
    if (tensor_meta->shape().elem_cnt() != 0) { return false; }
  }
  return true;
}

constexpr auto* CachedIsAllZeroSizeTensorMeta =
    DECORATE(&IsAllZeroSizeTensorMeta, ThreadLocalCopiable);

Maybe<Tensor> CalcBoxingOutput(const std::shared_ptr<Tensor>& input, Symbol<NdSbp> out_nd_sbp,
                               Symbol<ParallelDesc> out_parallel_desc,
                               bool current_rank_local_is_valid) {
  const auto& logical_shape = input->shape();
  const auto& logical_stride = JUST(input->stride());
  // If the input is a tensor of size 0, construct the output directly.
  if (unlikely(logical_shape->elem_cnt() == 0)) {
    ConsistentTensorMeta tensor_meta(logical_shape, logical_stride, input->dtype()->data_type(),
                                     out_nd_sbp, out_parallel_desc);
    const auto& tensor_impl =
        JUST(EagerConsistentTensorImpl::New(SymbolOf(tensor_meta), input->requires_grad(), false));
    std::shared_ptr<Tensor> output = std::make_shared<ConsistentTensor>(tensor_impl);
    return output;
  }
  const auto* mgr = Global<EagerBoxingInterpreterManager>::Get();
  // Eager boxing
  const auto& in_nd_sbp = JUST(input->nd_sbp());
  const auto& in_parallel_desc = JUST(input->parallel_desc());
  const auto& boxing_interpreter = JUST(mgr->GetEagerBoxingInterpreter(
      in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc, *logical_shape));
  Global<const EagerBoxingLogger>::Get()->Log(
      *JUST(boxing_interpreter->boxing_interpreter_status()), /* prefix */ "");
  if (!current_rank_local_is_valid) { return input; }
  const auto& output = JUST(boxing_interpreter->Interpret(input, in_nd_sbp, out_nd_sbp,
                                                          in_parallel_desc, out_parallel_desc));
  return output;
}

auto* GetBoxingOutput =
    DECORATE(DECORATE(&CalcBoxingOutput, CheckConsistentTensorMeta), DisableRecusiveBoxingCall);

Maybe<void> Interpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                      TensorTuple* outputs, const OpExprInterpContext& ctx) {
  CHECK_EQ_OR_RETURN(outputs->size(), user_op_expr.output_size());
  const auto& parallel_desc = JUST(GetParallelDesc(inputs, ctx));
  std::shared_ptr<const ConsistentTensorInferResult> result;
  NonRecursiveMetaInfoConsistencyCheckScope scope;
  if (inputs.empty()) {
    // check consistency placement and nd_sbp, do not check in non-src op because it is assumed that
    // InferSbp in op is a deterministic algorithm
    JUST(MetaInfoConsistencyCheck(parallel_desc, ctx.nd_sbp, 1));
    const auto& infer_args =
        JUST(SrcOpConsistentTensorMetaInferArgs::New(ctx.attrs, parallel_desc, JUST(ctx.nd_sbp)));
    result = JUST(user_op_expr.mut_consistent_tensor_infer_cache()->GetOrInfer(*infer_args));
  } else {
    const auto& infer_args = JUST(ConsistentTensorMetaInferArgs::New(ctx.attrs, inputs));
    result = JUST(user_op_expr.mut_consistent_tensor_infer_cache()->GetOrInfer(*infer_args));
  }
  const auto& output_tensor_metas = result->output_tensor_metas();
  Optional<int64_t> parallel_id;
  const auto& tensor_device = JUST(GetTensorDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
  for (int i = 0; i < outputs->size(); ++i) {
    if (!outputs->at(i)) {
      const auto& tensor_impl = JUST(EagerConsistentTensorImpl::New(
          output_tensor_metas.at(i), tensor_device, parallel_id, false, false));
      // tensor_impl->mut_tensor_meta()->set_stride(std::make_shared<Stride>(*tensor_impl->shape()));
      outputs->at(i).reset(new ConsistentTensor(tensor_impl));
    }
  }
  // Do nothing if output_tensors has 0-size shape. Since the input of some ops is 0-size but the
  // output is not 0-size, it cannot be judged based on the input, such as flow.cat
  if (unlikely(JUST(CachedIsAllZeroSizeTensorMeta(output_tensor_metas)))) {
    return Maybe<void>::Ok();
  }
  // Run instruction LocalCallOpKernel
  const auto& kernel = JUST(user_op_expr.MutKernel4Stream(result->stream()));
  CHECK_EQ_OR_RETURN(kernel->output_tuple_indexes4mut2_obns().size(), 0)
      << Error::UnimplementedError()
      << GetDynamicOpConsistentFailedDebugString(user_op_expr, *kernel);
  std::shared_ptr<EagerBlobObjectList> input_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(inputs.size());
  // expand lifetime of boxing outputs to the end of this function
  TensorTuple boxing_outputs;
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor> input = inputs.at(i);
    const auto& infered_input_meta = result->input_tensor_metas().at(i);
    const auto& input_parallel_desc = JUST(input->parallel_desc());
    CHECK_OR_RETURN(input_parallel_desc == infered_input_meta->parallel_desc());
    if (input_parallel_desc->parallel_num() != 1
        && infered_input_meta->nd_sbp() != JUST(input->nd_sbp())) {
      input = JUST(GetBoxingOutput(input, infered_input_meta->nd_sbp(),
                                   infered_input_meta->parallel_desc(), parallel_id.has_value()));
      boxing_outputs.emplace_back(input);
    }
    const auto& local_tensor = JUST(input->cur_rank_phy_tensor());
    input_eager_blob_objects->at(i) = JUST(local_tensor->eager_blob_object());
  }
  // Do nothing if the `parallel_desc` doesn't cover current ProcessCtx.
  if (!parallel_id.has_value()) { return Maybe<void>::Ok(); }
  std::shared_ptr<EagerBlobObjectList> output_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(outputs->size());
  for (int i = 0; i < outputs->size(); ++i) {
    const auto& local_tensor = JUST(outputs->at(i)->cur_rank_phy_tensor());
    output_eager_blob_objects->at(i) = JUST(local_tensor->eager_blob_object());
  }
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->LocalCallOpKernel(kernel, input_eager_blob_objects, output_eager_blob_objects,
                                      result, ctx, result->stream());
  }));
  return Maybe<void>::Ok();
}

auto* InterpretThenInitConsistentId = DECORATE(&Interpret, NonRecursiveInitConsistentId);

}  // namespace

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  return InterpretThenInitConsistentId(op_expr, inputs, outputs, ctx);
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

namespace {

static constexpr auto* RecursiveGetBoxingOutput =
    DECORATE(&CalcBoxingOutput, CheckConsistentTensorMeta);

Maybe<void> RawConsistentToConsistent(const ConsistentToConsistentOpExpr& op_expr,
                                      const TensorTuple& inputs, TensorTuple* outputs,
                                      const OpExprInterpContext& ctx) {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  const auto& input = inputs.at(0);
  CHECK_OR_RETURN(input->is_consistent());
  CHECK_OR_RETURN(ctx.parallel_desc.has_value());
  CHECK_OR_RETURN(ctx.nd_sbp.has_value());
  const auto& in_parallel_desc = JUST(input->parallel_desc());
  const auto& out_nd_sbp = JUST(ctx.nd_sbp);
  const auto& out_parallel_desc = JUST(ctx.parallel_desc);
  const auto& in_parallel_id = JUST(GetParallelId4CurrentProcessCtx(in_parallel_desc));
  const auto& out_parallel_id = JUST(GetParallelId4CurrentProcessCtx(out_parallel_desc));
  const auto& tensor =
      JUST(RecursiveGetBoxingOutput(input, out_nd_sbp, out_parallel_desc,
                                    in_parallel_id->has_value() || out_parallel_id->has_value()));
  CHECK_OR_RETURN(tensor);
  if (out_parallel_id->has_value()) {
    const auto& nd_sbp = JUST(tensor->nd_sbp());
    const auto& parallel_desc = JUST(tensor->parallel_desc());
    CHECK_OR_RETURN(nd_sbp == out_nd_sbp)
        << ". nd_sbp: " << NdSbpToString(nd_sbp) << ", out_nd_sbp" << NdSbpToString(out_nd_sbp);
    CHECK_OR_RETURN(parallel_desc == out_parallel_desc);
    outputs->at(0) = tensor;
  } else {
    ConsistentTensorMeta tensor_meta(tensor->shape(), JUST(tensor->stride()),
                                     tensor->dtype()->data_type(), out_nd_sbp, out_parallel_desc);
    const auto& tensor_impl =
        JUST(EagerConsistentTensorImpl::New(SymbolOf(tensor_meta), tensor->requires_grad(), false));
    outputs->at(0).reset(new ConsistentTensor(tensor_impl));
  }
  CHECK_OR_RETURN(outputs->at(0));
  return Maybe<void>::Ok();
}

static constexpr auto* ConsistentToConsistent =
    DECORATE(&RawConsistentToConsistent, NonRecursiveInitConsistentId);

}  // namespace

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const ConsistentToConsistentOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  JUST(ConsistentToConsistent(op_expr, inputs, outputs, ctx));
  return Maybe<void>::Ok();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastToConsistentOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastFromConsistentOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  const auto& input_tensor = inputs.at(0);
  const auto& mirrored_tensor = JUST(JUST(input_tensor->cur_rank_phy_tensor())->detach());
  bool requires_grad = autograd::GradMode::is_enabled() && input_tensor->requires_grad();
  JUST(mirrored_tensor->set_requires_grad(requires_grad));
  mirrored_tensor->set_is_leaf(!requires_grad);
  outputs->at(0) = mirrored_tensor;
  return Maybe<void>::Ok();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastToMirroredOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastFromMirroredOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeSplitOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeConcatOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const SelectTopNOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

}  // namespace one
}  // namespace oneflow
