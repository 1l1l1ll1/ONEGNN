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

#include "oneflow/core/auto_parallel/sbp_constructor.h"
#include "oneflow/core/auto_parallel/sbp_node.h"
#include "oneflow/core/auto_parallel/sbp_util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/job.pb.h"
#include "sbp_collector.h"

namespace oneflow {

namespace auto_parallel {

Maybe<void> SbpConstructor::Init(const OpGraph& op_graph, Job* job /*Maybe not use*/,
                                 bool take_curr_sbp) {
  JUST(InitSbpGraph(op_graph, *job, take_curr_sbp));
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::InitSbpGraph(const OpGraph& op_graph, const Job& job,
                                         bool take_curr_sbp) {
  // TODO: process mirrored node
  JUST(GenerateNodeAndEdge(op_graph, job));
  JUST(FillSbpSignatureForOpNode(op_graph, job, take_curr_sbp));
  JUST(InitComputationCost(op_graph));
  if (enable_mainstem_algo_) { JUST(ApplyMainstemAlgo()); }
  if (use_sbp_collector_) {
    // Load logical blobs on all sbp edges.
    LoadLbi2SbpEdge(op_graph);
    // Use sbp collector to create sbp proxy for nodes with multiple downstream operators.
    SbpCollector sbp_collector;
    sbp_collector.CollectUniverse(sbp_graph_);
    sbp_collector.ProxySbpCandidate(op_graph, op_name2sbp_node_, sbp_graph_);
  }

  JUST(InitCopyCost(op_graph));
  // TODO:  Set all the sbp signature id to be 0 for initialization.
  //        Could revert it back to
  // sbp_graph_.RandomSbpSignature(use_sbp_collector_);
  //        after settling down the synchronization of sbp strategy.
  sbp_graph_.Set0SbpSignature();
  double ori_cost = sbp_graph_.ComputeCost();
  LOG(INFO) << "Initial cost: " << ori_cost;
  JUST(StealSbpSignatureFromOpNode(op_graph, job));
  ori_cost = sbp_graph_.ComputeCost();
  LOG(INFO) << "OpGraph cost: " << ori_cost;
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::FindBestSbpSignature() {
  double ori_cost = sbp_graph_.ComputeCost();
  LOG(INFO) << "Initial cost: " << ori_cost;
  int elimination_num = sbp_graph_.NodeAndEdgeEliminations();
  LOG(INFO) << "Elimination number: " << elimination_num;
  if (ori_cost > cut_cost) {
    JUST(sbp_graph_.Find1Strategy4Greedy());
    ori_cost = sbp_graph_.ComputeCost();
    LOG(INFO) << "Greedy cost: " << ori_cost;
  }
  sbp_graph_.GreedyStrategy(4);
  sbp_graph_.FinalizeSbp();

  double final_cost = sbp_graph_.ComputeCost();
  LOG(INFO) << "Final cost: " << final_cost;
  if (ori_cost + 1.0 < final_cost) { LOG(WARNING) << "ori_cost less than final_cost!!!"; }
  // TODO: Restart searching with another original random strategy
  CHECK_LT_OR_RETURN(final_cost, cut_cost)
      << "Failed! Auto parallel can't find a strategy with reasonable cost!";
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::DumpNdSbpSignatureForJob(const OpGraph& op_graph, Job* job) {
  op_graph.ForEachNode([&](const OpNode* node) -> void {
    SbpNode<NdSbpSignature>* sbp_node = op_name2sbp_node_[node->op().op_name()];
    // Update NdSbpSignature
    // sbp_node->FinalSbpSignature()->ToProto(
    //     &(*job->mutable_job_parallel_view_conf()
    //            ->mutable_op_name2nd_sbp_signature_conf())[node->op().op_name()]);
    (*job->mutable_job_parallel_view_conf()
          ->mutable_op_name2nd_sbp_signature_conf())[node->op().op_name()]
        .CopyFrom(*sbp_node->FinalSbpSignature());
    // If we have 1D SbpSignature Conf
    if (node->parallel_desc().hierarchy()->NumAxes() == 1) {
      // Update SbpSignature
      SbpSignature sbp_signature;
      NdSbpSignatureToSbpSignature(*sbp_node->FinalSbpSignature(), &sbp_signature);
      (*job->mutable_job_parallel_view_conf()
            ->mutable_op_name2sbp_signature_conf())[node->op().op_name()]
          .CopyFrom(sbp_signature);
    }
    // TODO: Specially update sbp conf by using polymorphism function
    // Update sbp for variable op
    if (node->op().op_conf().has_variable_conf()) {
      for (auto& op : *job->mutable_net()->mutable_op()) {
        if (op.name() == node->op().op_name()) {
          op.mutable_variable_conf()->clear_nd_sbp();
          const auto nd_sbp = sbp_node->FinalSbpSignature()->bn_in_op2nd_sbp().at("out");
          for (const auto& sbp_parallel : nd_sbp.sbp_parallel()) {
            op.mutable_variable_conf()->mutable_nd_sbp()->Add(SbpParallelToString(sbp_parallel));
          }
        }
      }
    }
  });
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::GenerateNodeAndEdge(const OpGraph& op_graph, const Job& job) {
  JobParallelViewConf job_parallel_view_conf(job.job_parallel_view_conf());

  // Collect op_node
  std::vector<OpNode*> OpNodeList;
  op_graph.ForEachNode([&](OpNode* op_node) {
    // TODO: support mirror op
    bool is_mirrored_conf = false;
    {
      const auto& op_name2is_mirrored = job_parallel_view_conf.op_name2is_mirrored_parallel_view();
      const auto& iter = op_name2is_mirrored.find(op_node->op().op_name());
      if (iter != op_name2is_mirrored.end()) { is_mirrored_conf = iter->second; }
    }
    CHECK(is_mirrored_conf == false) << "Haven't deal with mirror operators.";
    OpNodeList.push_back(op_node);
  });

  // Decide the order to visit the op
  std::vector<int32_t> order;
  auto comp_op_name = [&](OpNode* a, OpNode* b) {
    return a->op().op_name().compare(b->op().op_name()) > 0;
  };
  auto_parallel::DecideOrder(OpNodeList, order, comp_op_name);
  std::vector<int32_t> output_order;

  // Create sbp nodes
  for (int32_t i = 0; i < OpNodeList.size(); i++) {
    OpNode* op_node = OpNodeList[order[i]];
    // Generate sbp node in cost model and link it with corresponding op node
    SbpNode<NdSbpSignature>* sbp_node = sbp_graph_.GenerateNode();
    // Mapping from sbp_node to op_node
    sbp_node->op_node = op_node;  // TODO: SetOpNode()
    op_name2sbp_node_[op_node->op().op_name()] = sbp_node;
  }
  // Create sbp edges
  for (int32_t i = 0; i < OpNodeList.size(); i++) {
    OpNode* op_node = OpNodeList[order[i]];
    // Get corresponding sbp node
    SbpNode<NdSbpSignature>* sbp_node = op_name2sbp_node_[op_node->op().op_name()];
    std::vector<OpNode*> OutputNodeList;
    for (const auto op_edge : op_node->out_edges()) {
      OutputNodeList.push_back(op_edge->dst_node());
    }
    auto_parallel::DecideOrder(OutputNodeList, output_order, comp_op_name);
    for (int32_t j : output_order) {
      const auto& end_node_name = OutputNodeList[j]->op().op_name();
      // Generate sbp edge in cost model
      sbp_node->PointTo(op_name2sbp_node_[end_node_name]);
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::FillSbpSignatureForOpNode(const OpGraph& op_graph, const Job& job,
                                                      bool take_curr_sbp) {
  // take_curr_sbp means only taking current sbp signature
  // TODO: use user sbp signature in JobParallelViewConf
  // const JobParallelViewConf& job_parallel_view_conf(job.job_parallel_view_conf());
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    SbpNode<NdSbpSignature>* sbp_node = op_name2sbp_node_[op_node->op().op_name()];
    if (take_curr_sbp) {
      // Get current sbp_signatures
      sbp_node->SbpSignatureObjList.push_back(op_node->nd_sbp_signature());
    } else {
      // Get all valid sbp_signatures
      HashMap<std::string, const BlobDesc*> ibn2blob_desc;
      auto FindShape4Blobs = [&](const PbRpf<std::string>& bns) -> Maybe<void> {
        for (const std::string& ibn : bns) {
          const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
          const BlobDesc* logical_blob_desc = &op_node->LogicalBlobDesc4Lbi(lbi);
          ibn2blob_desc.emplace(ibn, logical_blob_desc);
        }
        return Maybe<void>::Ok();
      };
      JUST(FindShape4Blobs(op_node->op().input_bns()));
      JUST(FindShape4Blobs(op_node->op().output_bns()));
      // Get logical blob description
      auto LogicalBlobDesc4Ibn = [&](const std::string& ibn) -> Maybe<const BlobDesc&> {
        auto it = ibn2blob_desc.find(ibn);
        if (it == ibn2blob_desc.end()) {
          return Error::InvalidValueError(
              "Cannot find corresponding blob description for input_blob_name : " + ibn + " in "
              + op_node->op().op_name());
        }
        return *(it->second);
      };
      // Get all valid sbp_signatures from op node
      JUST(op_node->op().GetValidNdSbpSignatureList(LogicalBlobDesc4Ibn, op_node->parallel_desc(),
                                                    &sbp_node->SbpSignatureObjList));
    }
    sbp_node->InitializeSbp();
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::StealSbpSignatureFromOpNode(const OpGraph& op_graph, const Job& job) {
  // Steal some strategy from original op graph
  for (auto* sbp_node : sbp_graph_.NodeList) {
    // sbp_collectors do not have op_node
    if (sbp_node->op_node) {
      for (int32_t sbp_id = 0; sbp_id < sbp_node->SbpSignatureObjList.size(); sbp_id++) {
        if (*JUST(sbp_node->op_node->op().nd_sbp_signature())
            == sbp_node->SbpSignatureObjList[sbp_id]) {
          sbp_node->FinalSbpSignatureId = sbp_id;
          break;
        }
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::InitComputationCost(const OpGraph& op_graph) {
  // Compute computation cost for sbp nodes
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    // get corresponding sbp node producer
    SbpNode<NdSbpSignature>* sbp_node = op_name2sbp_node_[op_node->op().op_name()];
    // get parallel description. Number of devices.
    const ParallelDesc& parallel_desc = op_node->parallel_desc();

    CHECK_EQ_OR_RETURN(sbp_node->Cost.size(), sbp_node->SbpSignatureList.size());
    auto logical_blob_desc4bn = [&](const std::string& bn) -> const BlobDesc& {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(bn);
      return op_node->LogicalBlobDesc4Lbi(lbi);
    };
    for (int32_t sbp_id = 0; sbp_id < sbp_node->SbpSignatureList.size(); sbp_id++) {
      double comp_cost = JUST(op_node->op().GetComputeComplexity(
          sbp_node->SbpSignatureList[sbp_id], logical_blob_desc4bn, parallel_desc));
      if (comp_cost > cut_cost) {
        sbp_node->Cost.at(sbp_id) = comp_cost;
      } else {
        sbp_node->Cost.at(sbp_id) = cost_ratio_ * comp_cost;
      }
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::InitCopyCost(const OpGraph& op_graph) {
  // Compute copy cost for sbp edges
  op_graph.ForEachNode([&](OpNode* op_node) {
    // get corresponding sbp node consumer
    SbpNode<NdSbpSignature>* sbp_node_consumer = op_name2sbp_node_[op_node->op().op_name()];
    // Initialize copy cost between two nodes
    for (auto* sbp_edge : sbp_node_consumer->EdgesIn) {
      // producer sbp node
      const auto* sbp_node_producer = sbp_edge->StartNode;
      // skip it if proxy
      if (!sbp_node_producer->op_node) { continue; }
      sbp_edge->Cost.resize(sbp_node_producer->SbpSignatureList.size());
      int32_t consumer_sbp_size = sbp_node_consumer->SbpSignatureList.size();
      // look through sbp signature in producer
      for (int32_t i = 0; i < sbp_node_producer->SbpSignatureList.size(); ++i) {
        sbp_edge->Cost[i].resize(consumer_sbp_size, 0);
      }
    }
    // Find all those cases with wait time
    // Do not skip edges carrying no lbi
    sbp_node_consumer->InitializeCopyCost(false, use_sbp_collector_);
    for (auto* sbp_edge : sbp_node_consumer->EdgesIn) {
      // skip it if proxy
      if (!sbp_edge->StartNode->op_node) { continue; }
      // Reset Wait time
      for (int32_t i = 0; i < sbp_edge->Cost.size(); ++i) {
        for (int32_t j = 0; j < sbp_edge->Cost[i].size(); ++j) {
          // If transferring between devices, we need to add wait time.
          if (sbp_edge->Cost[i][j] > 0.0) { sbp_edge->Cost[i][j] = sbp_edge->WaitTime; }
        }
      }
    }

    // Re-compute the costs, skip edges carrying no lbi
    sbp_node_consumer->InitializeCopyCost(true, use_sbp_collector_);
  });
  return Maybe<void>::Ok();
}

Maybe<void> SbpConstructor::ApplyMainstemAlgo() {
  // Compute layer number for each node
  int32_t max_MinLayer = sbp_graph_.ComputeLayer(op_name2sbp_node_);
  // Accumulate cost on the mainstem after initializing computation cost
  sbp_graph_.FindMainstem(max_MinLayer, op_name2sbp_node_);
  return Maybe<void>::Ok();
}

// Load logical blob ids onto sbp edges
void SbpConstructor::LoadLbi2SbpEdge(const OpGraph& op_graph) {
  // Load logical blobs onto sbp edges

  for (auto* sbp_node_consumer : sbp_graph_.NodeList) {
    auto* op_node = sbp_node_consumer->op_node;

    // Loading logical blobs between two nodes
    // look through input blobs
    for (const std::string& ibn : op_node->op().input_bns()) {
      // Each input blob has one source op node.
      OpNode* producer = op_node->MutSrcNode4Ibn(ibn);
      // producer sbp node
      const auto* sbp_node_producer = op_name2sbp_node_[producer->op().op_name()];
      // TODO: recode this
      auto* edge_found = auto_parallel::FindEdgeBetweenNodes(sbp_node_producer, sbp_node_consumer);

      CHECK(edge_found != NULL) << "SbpEdge not found while loading!" << std::endl;

      // Add copy cost for each blob
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
      edge_found->LoadLbi(lbi);
    }
  };
}

Maybe<void> SbpConstructor::CheckSbpAgreement(const Job& job) {
  Job new_job;
  new_job.CopyFrom(job);
  OpGraph op_graph(new_job);
  // Compare sbp in job
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    const std::string& op_name = op_node->op().op_name();
    const NdSbpSignature& auto_parallel_sbp =
        NdSbpSignature(job.job_parallel_view_conf().op_name2nd_sbp_signature_conf().at(op_name));
    const NdSbpSignature& new_sbp = op_node->nd_sbp_signature();
    CHECK_EQ_OR_RETURN(auto_parallel_sbp.bn_in_op2nd_sbp_size(), new_sbp.bn_in_op2nd_sbp_size());
    for (const auto& iter : auto_parallel_sbp.bn_in_op2nd_sbp()) {
      const NdSbp& new_sbp_parallel = new_sbp.bn_in_op2nd_sbp().at(iter.first);
      const NdSbp& auto_parallel_sbp = iter.second;
      // According error message, we can find op_type in op_conf.proto with type_id and locate
      // the error op type.
      const std::string& error_mgs =
          "Op: `" + op_name + "`(type_id: " + std::to_string(op_node->op().op_conf().op_type_case())
          + ") changed sbp from " + NdSbpToString(auto_parallel_sbp) + "(AutoParallel) to "
          + NdSbpToString(new_sbp_parallel) + "(OpGraph) with blob_name: `" + iter.first + "`.";
      CHECK_OR_RETURN(new_sbp_parallel == auto_parallel_sbp) << error_mgs;
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

// Print the graph with SBP in order
void SbpConstructor::PrintSBPGraphDebugInfo() {
  // sbp constructor information
  std::cout << "cost_ratio_:" << cost_ratio_ << std::endl;
  std::cout << "transfer_cost_:" << sbp_graph_.transfer_cost << std::endl;
  std::cout << "wait_time_:" << sbp_graph_.wait_time << std::endl;
  std::cout << "use_sbp_collector_" << use_sbp_collector_ << std::endl;
  // test debug
  std::cout << "Get Into Print Op Graph" << std::endl;
  // Collect op_node
  std::vector<OpNode*> NodeList;
  for (const auto& op_name_sbp_node : op_name2sbp_node_) {
    auto* op_node_ = op_name_sbp_node.second->op_node;
    if (op_node_) { NodeList.push_back(op_node_); }
  }

  // test debug
  std::cout << "Deciding order" << std::endl;
  // Decide the order to visit the op
  std::vector<int32_t> order;
  auto_parallel::DecideOrder(NodeList, order, [&](OpNode* a, OpNode* b) {
    return a->op().op_name().compare(b->op().op_name()) > 0;
  });
  std::vector<int32_t> str_order;

  // test debug
  std::cout << "Finish deciding order" << std::endl;

  for (int32_t i = 0; i < NodeList.size(); i++) {
    OpNode* op_node = NodeList[order[i]];
    std::cout << op_node->op().op_name() << " (^_^):" << std::endl;
    // get corresponding sbp node
    auto it = op_name2sbp_node_.find(op_node->op().op_name());
    // Print debug information for sbp graph
    CHECK(it != op_name2sbp_node_.end());
    const SbpNode<NdSbpSignature>* sbp_node = it->second;
    std::cout << "Computation Cost: " << sbp_node->Cost[sbp_node->FinalSbpSignatureId];
    std::cout << ", Min Layer: " << sbp_node->MinLayer << ", Max Layer: " << sbp_node->MaxLayer
              << ", Tributary Layer: " << sbp_node->TributaryLayer
              << ", in mainstem: " << sbp_node->IfMainstem
              << ", Remain Cost: " << sbp_node->AccMainstemCost << std::endl;
    // Sort before printing
    const auto& op_input_bns = op_node->op().input_bns();
    auto comp = [](const std::string& a, const std::string& b) { return a.compare(b) > 0; };
    auto_parallel::DecideOrder(op_input_bns, str_order, comp);
    const NdSbpSignature& sbp_signature = *sbp_node->FinalSbpSignature();
    // Print out SBP information for input operator
    for (int32_t j : str_order) {
      const auto& ibn = op_input_bns[j];
      const auto& producer_node = op_node->SrcNode4Ibn(ibn);
      std::cout << "Pre Op:" << producer_node.op().op_name() << ": " << ibn;
      const auto& this_sbp_parallel = sbp_signature.bn_in_op2nd_sbp().at(ibn);
      std::cout << ", " << NdSbpToString(this_sbp_parallel);
      if (IsSameSbp(op_node, ibn)) { std::cout << ", same SBP"; }
      std::cout << ", "
                << op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(ibn)).shape().elem_cnt();
      std::cout << std::endl;
    }
    // Sort before printing
    const auto& op_output_bns = op_node->op().output_bns();
    auto_parallel::DecideOrder(op_output_bns, str_order, comp);
    // Print out SBP information for output blobs
    for (int32_t j : str_order) {
      const auto& obn = op_output_bns[j];
      std::cout << "Out Op:" << obn;
      const auto& this_sbp_parallel = sbp_signature.bn_in_op2nd_sbp().at(obn);
      std::cout << ", " << NdSbpToString(this_sbp_parallel);
      std::cout << ", "
                << op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(obn)).shape().elem_cnt();
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

// Explicitly show the control edges
void SbpConstructor::ExposeCtrlEdges() {
  for (auto* sbp_node : sbp_graph_.NodeList) { sbp_node->ExposeCtrlEdges(op_name2sbp_node_); }
}

// Algorithms for straightening
Maybe<void> SbpConstructor::StraightenNodes(
    const std::function<Maybe<void>(OpNode*, OpNode*)>& add_control_edge) {
  // Set the counter to be the number of producer
  for (auto* sbp_node : sbp_graph_.NodeList) { sbp_node->counter = sbp_node->EdgesIn.size(); }
  // The time passed before finishing the current transfer
  double acc_comp_time = 0.0;
  // The transfer happening at this moment
  std::queue<std::pair<SbpNode<NdSbpSignature>*, double>> waiting_transfer;
  // Decide which node should run first
  struct comp {
    bool operator()(const SbpNode<NdSbpSignature>* a, const SbpNode<NdSbpSignature>* b) const {
      if (a->TributaryLayer == b->TributaryLayer) {
        if (a->MinLayer == b->MinLayer) {
          // the order does not matter right now
          // return a->Cost[0] < b->Cost[0];
          // we need a strict order
          return a->NodeListId < b->NodeListId;
        } else {
          // the node that shows up first has higher priority
          return a->MinLayer < b->MinLayer;
        }
      } else {
        // the urgent node has the higher priority
        return a->TributaryLayer < b->TributaryLayer;
      }
    }
  };
  // The computation ready for execution
  std::set<SbpNode<NdSbpSignature>*, comp> waiting_computation;
  // Finish the transfer of one producer
  auto finish_one_transfer = [&](SbpNode<NdSbpSignature>* sbp_node) {
    sbp_node->counter--;
    if (sbp_node->counter == 0) { waiting_computation.insert(sbp_node); }
  };
  // Finish the front of the waiting transfer list
  auto pop_waiting_transfer = [&]() {
    SbpNode<NdSbpSignature>* sbp_node = waiting_transfer.front().first;
    for (auto* edge_out : sbp_node->EdgesOut) {
      if (!edge_out->Cost.empty() && edge_out->Cost[0][0] > 0.0) {
        finish_one_transfer(edge_out->EndNode);
      }
    }
    waiting_transfer.pop();
  };
  // The previous node that just finished the execution
  SbpNode<NdSbpSignature>* previous_node = nullptr;

  // Node execution
  auto execute = [&](SbpNode<NdSbpSignature>* sbp_node) -> Maybe<void> {
    // NOTE: I am not sure whether the source ops have execution time
    // Assume the source operators have no execution time
    // No overlaps for the source operators
    if (!sbp_node->EdgesIn.empty()) {
      // Add a control edge from the previous node to this node
      // The aim of the straightening algorithm
      JUST(add_control_edge(previous_node->op_node, sbp_node->op_node));
      // Delete it from the waiting list if exists
      auto it = waiting_computation.find(sbp_node);
      if (it != waiting_computation.end()) { waiting_computation.erase(it); }
      // Transfer happens during computation
      acc_comp_time += sbp_node->Cost[0];
      while (!waiting_transfer.empty() && acc_comp_time > waiting_transfer.front().second) {
        // Finish the front of the waiting transfer list as the time pass
        acc_comp_time -= waiting_transfer.front().second;
        pop_waiting_transfer();
      }
      // Waist the computation time because no transfer is waiting
      if (waiting_transfer.empty()) { acc_comp_time = 0.0; }
    }
    // Finish execution of this node
    previous_node = sbp_node;
    // Add the consumer into waiting list, either transfer or computation
    for (auto* edge_out : sbp_node->EdgesOut) {
      double total_copy_cost = 0.0;
      if (edge_out->Cost.empty() || edge_out->Cost[0][0] == 0.0) {
        // No transfer, wait for computation immediately
        finish_one_transfer(edge_out->EndNode);
      } else {
        total_copy_cost += edge_out->Cost[0][0];
      }
      // wait for transfer
      if (total_copy_cost > 0.0) { waiting_transfer.push({sbp_node, total_copy_cost}); }
    }
    return Maybe<void>::Ok();
  };
  // Execute all the source op
  for (auto* sbp_node : sbp_graph_.NodeList) {
    if (sbp_node->EdgesIn.size() == 0) { execute(sbp_node); }
  }
  // Execute nodes or transfer as time pass
  while (true) {
    if (waiting_computation.empty()) {
      // if we have no nodes waiting in the list
      if (waiting_transfer.empty()) {
        // No nodes waiting, no transfer waiting, done
        break;
      } else {
        // Take some time for transfer. At this moment, no overlap occurs
        acc_comp_time = 0.0;
        pop_waiting_transfer();
      }
    } else {
      // if we have nodes waiting in the list, execute the first one
      execute(*waiting_computation.begin());
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace auto_parallel
}  // namespace oneflow
