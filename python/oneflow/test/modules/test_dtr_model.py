"""
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
"""
import os
import sys
import re
import unittest
import time
import random

import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest

import flowvision


def sync():
    flow.comm.barrier()
    # sync_tensor.numpy()


def allocated_memory(device):
    return flow._oneflow_internal.dtr.allocated_memory(device)


class TestDTRCorrectness(flow.unittest.TestCase):

    def test_dtr_correctness(test_case):
        seed = 20
        flow.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        device = 'cuda'

        bs = 115
        model = flowvision.models.resnet50()

        criterion = nn.CrossEntropyLoss()

        model.to(device)
        criterion.to(device)
        model.train()

        learning_rate = 1e-3
        optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

        train_data = flow.tensor(
            np.random.uniform(size=(bs, 3, 224, 224)).astype(np.float32), device=device
        )
        train_label = flow.tensor(
            (np.random.uniform(size=(bs,)) * 1000).astype(np.int32),
            dtype=flow.int32,
            device=device,
        )

        WARMUP_ITERS = 3
        ALL_ITERS = 10
        total_time = 0
        for x in model.parameters():
            x.grad = flow.zeros_like(x).to(device)
        initial_cpu_mem = allocated_memory('cpu')
        initial_cuda_mem = allocated_memory('cuda')
        for iter in range(ALL_ITERS):
            print(f'iter {iter}')
            # flow._oneflow_internal.dtr.display(device)
            cpu_mem = allocated_memory('cpu')
            cuda_mem = allocated_memory('cuda')
            test_case.assertEqual(initial_cpu_mem, cpu_mem, iter)
            test_case.assertEqual(initial_cuda_mem, cuda_mem, iter)
            if iter >= WARMUP_ITERS:
                start_time = time.time()
            logits = model(train_data)
            loss = criterion(logits, train_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"loss: {loss}")
            # if iter == ALL_ITERS - 1:
            #     test_case.assertGreater(loss, 5.98)
            #     test_case.assertLess(loss, 6.01)
            del logits
            del loss
            sync()
            if iter >= WARMUP_ITERS:
                end_time = time.time()
                this_time = end_time - start_time
                total_time += this_time

        end_time = time.time()
        print(
            f"{ALL_ITERS - WARMUP_ITERS} iters: avg {(total_time) / (ALL_ITERS - WARMUP_ITERS)}s"
        )


if __name__ == "__main__":
    unittest.main()
