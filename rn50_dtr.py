# coding=utf-8

import time
import os

import numpy as np
from numpy import random
import oneflow as flow
import oneflow.nn as nn

import resnet50_model


# resnet50 bs 32, use_disjoint_set=False: threshold ~800MB

# memory policy:
# 1: only reuse the memory block with exactly the same size
# 2: reuse the memory block with the same size or larger

dtr_enabled = os.getenv("OF_DTR", None) is not None

if dtr_enabled:
    dtr_ee_enabled = os.getenv("OF_DTR_NO_EE", None) is None
    threshold = os.environ["OF_DTR_THRESHOLD"]
    debug_level = int(os.getenv("OF_DTR_DEBUG", "0"))
    left_right = os.getenv("OF_DTR_LR", None) is not None
else:
    if any([os.getenv(x) is None for x in ["OF_DTR_NO_EE", "OF_DTR_THRESHOLD", "OF_DTR_DEBUG", "OF_DTR_LR"]]):
        print("warning! dtr is not enabled but dtr related env var is set")
    dtr_ee_enabled = False
    threshold = "NaN"
    debug_level = "NaN"
    left_right = "invalid"

# no bn:
# full: 1700MB
# lr=1 no_ee=1 high add_n cost 600mb success
# lr=1 no_ee=0 high add_n cost 600mb fail

# has bn:
# full: 2850mb
# lr=1 no_ee=0 high add_n cost 850mb 0.352s
# lr=1 no_ee=1 high add_n cost 850mb 0.477s
# lr=1 no_ee=1 high add_n cost 750mb 0.528s

memory_policy = 1
heuristic = "eq_compute_time_and_last_access"

if dtr_enabled:
    print(f'dtr_enabled: {dtr_enabled}, threshold: {threshold}, eager eviction: {dtr_ee_enabled}, left and right: {left_right}, debug_level: {debug_level}, memory_policy: {memory_policy}, heuristic: {heuristic}')
else:
    print(f'dtr_enabled: {dtr_enabled}')

if dtr_enabled:
    flow.enable_dtr(dtr_enabled, threshold, debug_level, memory_policy, heuristic)

seed = 20
flow.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def sync():
    flow._oneflow_internal.eager.multi_client.Sync()


def display():
    flow._oneflow_internal.dtr.display()


# init model
# model = resnet50_model.resnet50(norm_layer=nn.Identity)
model = resnet50_model.resnet50()
model.load_state_dict(flow.load('/tmp/abcde'))
# flow.save(model.state_dict(), '/tmp/abcde')

criterion = nn.CrossEntropyLoss()

cuda0 = flow.device('cuda:0')

# enable module to use cuda
model.to(cuda0)

criterion.to(cuda0)

learning_rate = 1e-3
# optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

batch_size = 32

# generate random data and label
train_data = flow.tensor(
    np.random.uniform(size=(batch_size, 3, 224, 224)).astype(np.float32), device=cuda0
)
train_label = flow.tensor(
    (np.random.uniform(size=(batch_size,)) * 1000).astype(np.int32), dtype=flow.int32, device=cuda0
)

def temp():
    sync()
    print('----------allocator start')
    flow._oneflow_internal.eager.multi_client.Temp()
    sync()
    print('----------allocator end')

# run forward, backward and update parameters
WARMUP_ITERS = 2
ALL_ITERS = 52
for iter in range(ALL_ITERS):
    if dtr_enabled:
        for x in model.parameters():
            x.grad = flow.zeros_like(x).to(cuda0)

        temp()
    if iter == WARMUP_ITERS:
        start_time = time.time()
    logits = model(train_data)
    loss = criterion(logits, train_label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(True)
    # sync()
    # exit(2)
    if debug_level > 0:
        sync()
        display()
    # if (epoch + 1) % 1 == 0:
        # print('loss: ', loss.numpy())
    del logits
    del loss
    sync()
    print(f'iter {iter} end')

end_time = time.time()
print(f'{ALL_ITERS - WARMUP_ITERS} iters: avg {(end_time - start_time) / (ALL_ITERS - WARMUP_ITERS)}s')
