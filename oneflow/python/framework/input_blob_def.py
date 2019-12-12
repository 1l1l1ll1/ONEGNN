from __future__ import absolute_import

import sys

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as lbi_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.distribute as distribute_util
from oneflow.python.oneflow_export import oneflow_export
from functools import reduce
import numpy as np
import oneflow

@oneflow_export('input_blob_def')
class input_blob_def(blob_desc.BlobDesc):
    def __init__(self, shape,
                 dtype = data_type_util.kFloat,
                 is_dynamic = False,
                 is_tensor_list = False,
                 batch_axis = 0,
                 name = None,
                 **kw):
        lbi = lbi_util.LogicalBlobId()
        if name is None: name = id_util.UniqueStr("Input_")
        lbi.op_name = name
        lbi.blob_name = "out"
        blob_desc.BlobDesc.__init__(self, lbi, **kw)
        assert type(shape) is tuple
        for dim in shape: assert type(dim) is int
        self.shape_ = shape
        self.dtype_ = dtype
        self.is_dynamic_ = is_dynamic
        self.is_tensor_list_ = is_tensor_list
        self.batch_axis_ = batch_axis

    @property
    def static_shape(self): return self.shape_

    @property
    def shape(self): return self.shape_

    @property
    def dtype(self): return self.dtype_

    @property
    def batch_axis(self): return self.batch_axis_

    @property
    def is_dynamic(self): return self.is_dynamic_

    @property
    def is_tensor_list(self): return self.is_tensor_list_

    def parallel_conf(self):
        TODO()

    def parallel_conf(self):
        TODO()

    def with_distribute(self, distribute):
        return input_blob_def(shape = self.shape_, dtype = self.dtype_,               \
                        is_dynamic = self.is_dynamic_, batch_axis = self.batch_axis_, \
                        distribute = distribute, name = self.lbi.op_name)

    def CheckInputNdarray(self, ndarray):
        if self.is_tensor_list:
            self._CheckNdarrayList(ndarray)
        else:
            self._CheckDenseNdarray(ndarray)

    def ToInterfaceBlobConf(self):
        interface_blob_conf = op_conf_util.InterfaceBlobConf()
        interface_blob_conf.shape.dim.extend(self.shape_)
        interface_blob_conf.data_type = self.dtype_
        interface_blob_conf.is_dynamic = self.is_dynamic_
        interface_blob_conf.is_tensor_list = self.is_tensor_list
        if type(self.batch_axis_) is int:
            assert self.batch_axis_ >= 0
            interface_blob_conf.batch_axis.value = self.batch_axis_
        else:
            assert self.batch_axis_ is None or self.batch_axis_ is False
            interface_blob_conf.batch_axis.ClearField("value")
        if type(self.distribute_) is distribute_util.SplitDistribute:
            interface_blob_conf.split_axis.value = self.distribute_.axis
        elif type(self.distribute_) is distribute_util.BroadcastDistribute:
            interface_blob_conf.split_axis.ClearField("value")
        else:
            # do nothing
            pass
        return interface_blob_conf

    def __add__(self, rhs):
        return oneflow.math.add(self, rhs)

    def __radd__(self, lhs):
        return oneflow.math.add(lhs, self)

    def __sub__(self, rhs):
        return oneflow.math.subtract(self, rhs)

    def __rsub__(self, lhs):
        return oneflow.math.subtract(lhs, self)

    def __mul__(self, rhs):
        return oneflow.math.multiply(self, rhs)

    def __rmul__(self, lhs):
        return oneflow.math.multiply(lhs, self)

    def __mul__(self, rhs):
        return oneflow.math.multiply(self, rhs)

    def __rmul__(self, lhs):
        return oneflow.math.multiply(lhs, self)

    def __truediv__(self, rhs):
        return oneflow.math.divide(self, rhs)

    def __div__(self, rhs):
        return oneflow.math.divide(self, rhs)

    def __eq__(self, rhs):
        return oneflow.math.equal(self, rhs)

    def __ne__(self, rhs):
        return oneflow.math.not_equal(self, rhs)

    def __lt__(self, rhs):
        return oneflow.math.less(self, rhs)

    def __le__(self, rhs):
        return oneflow.math.less_equal(self, rhs)

    def __gt__(self, rhs):
        return oneflow.math.greater(self, rhs)

    def __ge__(self, rhs):
        return oneflow.math.greater_equal(self, rhs)

    def _CheckDenseNdarray(self, ndarray):
        assert isinstance(ndarray, np.ndarray)
        def GetElemCnt(shape): return reduce(lambda x, y: x * y, shape, 1)
        assert len(ndarray.shape) == len(self.shape)
        if self.is_dynamic:
            assert GetElemCnt(ndarray.shape) <= GetElemCnt(self.shape)
        else:
            assert ndarray.shape == self.shape

    def _CheckNdarrayList(self, ndarray_nested_list):
        raise NotImplementedError
