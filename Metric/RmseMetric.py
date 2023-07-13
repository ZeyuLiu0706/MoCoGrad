import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metric import AbsMetric

class RmseMetric(AbsMetric):
    def __init__(self, metric_name: list = ['RMSE']):
        all_metric_info={'RMSE':0}
        super(RmseMetric, self).__init__(metric_name, all_metric_info)

    def update_fun(self, pred, gt):
        pred = pred.flatten()
        squared_err = torch.sum((pred-gt)**2)
        self.record.append(squared_err)
        self.bs.append(pred.size()[0])

    def score_fun(self):
        records = torch.tensor(self.record,dtype=float)
        batch_size = torch.tensor(self.bs,dtype=float)
        # print(records)
        tmp = torch.sum(records) / torch.sum(batch_size)
        rmse = torch.sqrt(tmp)
        return {'RMSE': rmse.item()}
