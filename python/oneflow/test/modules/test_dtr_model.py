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


class TestDTRCorrectness(flow.unittest.TestCase):

    def test_dtr_correctness(test_case):
        seed = 20
        flow.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        bs = 115
        model = flowvision.models.resnet50()

        criterion = nn.CrossEntropyLoss()

        model.to('cuda')
        criterion.to('cuda')

        learning_rate = 1e-3
        optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

        train_data = flow.tensor(
            np.random.uniform(size=(bs, 3, 224, 224)).astype(np.float32), device='cuda'
        )
        train_label = flow.tensor(
            (np.random.uniform(size=(bs,)) * 1000).astype(np.int32),
            dtype=flow.int32,
            device='cuda',
        )

        WARMUP_ITERS = 5
        ALL_ITERS = 40
        total_time = 0
        for iter in range(ALL_ITERS):
            for x in model.parameters():
                x.grad = flow.zeros_like(x).to('cuda')

            if iter >= WARMUP_ITERS:
                start_time = time.time()
            logits = model(train_data)
            loss = criterion(logits, train_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"loss: {loss}")
            if iter == ALL_ITERS - 1:
                test_case.assertGreater(loss, 5.98)
                test_case.assertLess(loss, 6.01)
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
