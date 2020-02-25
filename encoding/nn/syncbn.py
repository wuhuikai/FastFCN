##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Synchronized Cross-GPU Batch Normalization Module"""
import collections
import torch

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.functional import batch_norm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

from ..functions import normalization
from .comm import SyncMaster


__all__ = ['SyncBatchNorm', 'BatchNorm']


class SyncBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not self.training:
            return batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input_shape[0], self.num_features, -1)

        # sum(x) and sum(x^2)
        N = input.size(0) * input.size(2)
        xsum = input.sum((0, 2))
        xsqsum = input.pow(2).sum((0, 2))

        # all-reduce for global sum(x) and sum(x^2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(xsum, xsqsum, N))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(xsum, xsqsum, N))
        # forward
        return normalization(input, mean, inv_std, self.weight, self.bias).view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, (bias_var + self.eps) ** -0.5

    @classmethod
    def convert_sync_batchnorm(cls, module, skip_classes=()):
        for skip_class in skip_classes:
            if isinstance(module, skip_class):
                return module

        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SyncBatchNorm(module.num_features,
                                          module.eps,
                                          module.momentum,
                                          module.affine)
            if module.affine:
                with torch.no_grad():
                    module_output.weight.copy_(module.weight)
                    module_output.bias.copy_(module.bias)
                # keep requires_grad unchanged
                module_output.weight.requires_grad = module.weight.requires_grad
                module_output.bias.requires_grad = module.bias.requires_grad
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, skip_classes))
        del module
        return module_output


# API adapted from https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
_ChildMessage = collections.namedtuple('Message', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class BatchNorm(_BatchNorm):
    def _check_input_dim(self, input):
        pass
