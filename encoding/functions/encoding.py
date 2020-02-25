##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Functions for Encoding Layer"""
import torch

from torch.autograd import Function

__all__ = ['aggregate', 'scaled_l2']

class Aggregate(Function):
    @staticmethod
    def forward(ctx, A, X, C):
        ctx.save_for_backward(A, X, C)

        return (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) -
             C.unsqueeze(0).unsqueeze(0)).mul_(A.unsqueeze(3)).sum(1)

    @staticmethod
    def backward(ctx, GE):
        A, X, C = ctx.saved_variables

        gradA = (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) -
                 C.unsqueeze(0).unsqueeze(0)).mul_(GE.unsqueeze(1)).sum(3)
        gradX = torch.bmm(A, GE)
        gradC = A.sum(1).unsqueeze(2).mul(GE).mul_(-1).sum(0)

        return gradA, gradX, gradC

def aggregate(A, X, C):
    r""" Aggregate operation, aggregate the residuals of inputs (:math:`X`) with repect
    to the codewords (:math:`C`) with assignment weights (:math:`A`).

    .. math::

        e_{k} = \sum_{i=1}^{N} a_{ik} (x_i - d_k)

    Shape:
        - Input: :math:`A\in\mathcal{R}^{B\times N\times K}`
          :math:`X\in\mathcal{R}^{B\times N\times D}` :math:`C\in\mathcal{R}^{K\times D}`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`

    Examples:
        >>> B,N,K,D = 2,3,4,5
        >>> A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), requires_grad=True)
        >>> X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> func = encoding.aggregate()
        >>> E = func(A, X, C)
    """
    return Aggregate.apply(A, X, C)

class ScaledL2(Function):
    @staticmethod
    def forward(ctx, X, C, S):
        SL = (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) -
              C.unsqueeze(0).unsqueeze(0)).pow_(2).sum(3).mul_(S.view(1, 1, C.size(0)))
        ctx.save_for_backward(X, C, S, SL)
        return SL

    @staticmethod
    def backward(ctx, GSL):
        X, C, S, SL = ctx.saved_variables

        tmp = (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) - C.unsqueeze(0).unsqueeze(0)).mul_(
            (2 * GSL).mul_(S.view(1, 1, C.size(0))).unsqueeze(3)
        )

        GX = tmp.sum(2)
        GC = tmp.sum((0, 1)).mul_(-1)
        GS = SL.div(S.view(1, 1, C.size(0))).mul_(GSL).sum((0, 1))

        return GX, GC, GS

def scaled_l2(X, C, S):
    r""" scaled_l2 distance

    .. math::
        sl_{ik} = s_k \|x_i-c_k\|^2

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}`
          :math:`C\in\mathcal{R}^{K\times D}` :math:`S\in \mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times N\times K}`
    """
    return ScaledL2.apply(X, C, S)

if __name__ == '__main__':
    B, N, D, K = 3, 4, 5, 6
    X = torch.randn((B, N, D), dtype=torch.double,requires_grad=True).cuda()
    C = torch.randn((K, D), dtype=torch.double,requires_grad=True).cuda()
    S = torch.randn((K,), dtype=torch.double,requires_grad=True).cuda()
    assert torch.autograd.gradcheck(scaled_l2, (X, C, S))

    A = torch.randn((B, N, K), dtype=torch.double, requires_grad=True).cuda()
    X = torch.randn((B, N, D), dtype=torch.double, requires_grad=True).cuda()
    C = torch.randn((K, D), dtype=torch.double, requires_grad=True).cuda()
    assert torch.autograd.gradcheck(aggregate, (A, X, C))
