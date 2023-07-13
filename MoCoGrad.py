import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from LibMTL.weighting.abstract_weighting import AbsWeighting


class MoCoGrad(AbsWeighting):
    r"""

    .. warning::
            MoCoGrad is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """

    def __init__(self):
        super(MoCoGrad, self).__init__()

    def _momt2vec(self, momentum):
        grad_momt = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count + 1)])
                grad_momt[beg:end] = momentum[param].data.view(-1)
                # grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad_momt

    def _compute_momt(self):
        momentum = defaultdict()
        for group in self.opt.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.opt.state[p]
                if len(state) == 0:
                    momentum[p] = torch.zeros_like(p)
                    continue
                    # return None
                ori_exp_avg, ori_exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                momentum[p] = ori_exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                # momentum = ori_exp_avg_sq.mul(beta2).addcmul(grad, grad, alpha=1 - beta2)
        return momentum

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        we rewrite the _compute_grad function of AbsWeighting class
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            momentums = {}
            grad_momts = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    momentums[tn] = self._compute_momt()
                    grads[tn] = self._grad2vec()
                    grad_momts[tn] = self._momt2vec(momentums[tn])
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            grad_momts = {}
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
                    grad_momts[tn] = None

        return grads, grad_momts

    def backward(self, losses, **kwargs):
        batch_weight = np.ones(len(losses))
        lmd = kwargs['lambda']

        # consider the momentums is None, namely the optimizer step = 0
        # the MoCoGrad only works when the parameters have momentums

        if 'optimizer' in kwargs.keys():
            self.opt = kwargs['optimizer']
        else:
            raise ValueError('The Optimizer needs to be passed in')

        if self.rep_grad:
            raise ValueError('No support method MoCoGrad with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads, grad_momts = self._compute_grad(losses, mode='backward')  # [task_num, grad_dim]

        mc_grads = grads.clone()
        mc_momts = grad_momts.clone()
        for tn_i in range(self.task_num):
            if torch.equal(grad_momts[tn_i], torch.zeros_like(grad_momts[tn_i])):
                continue
            task_index = list(range(self.task_num))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                phi_ij = torch.dot(mc_grads[tn_i], grads[tn_j]) / (mc_grads[tn_i].norm()*grads[tn_j].norm())
                g_ij = torch.dot(mc_grads[tn_i], grads[tn_j])
                if g_ij < 0:
                    w_ij = lmd * grads[tn_j].norm() / mc_momts[tn_j].norm()
                    mc_grads[tn_i] += w_ij * mc_momts[tn_j]
                    batch_weight[tn_j] = w_ij.item()
        new_grads = mc_grads.sum(0)
        self._reset_grad(new_grads)
        return batch_weight
