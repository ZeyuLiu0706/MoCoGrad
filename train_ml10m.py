from LibMTL.core.trainer import Trainer
import argparse
from LibMTL.core import Trainer
from typing import Optional, Type
import torch
import torch.nn as nn
from LibMTL.utils.timer import TimeRecorder
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Configuration for LibMTL')
    parser.add_argument('--dataset', default='ml10m', type=str, help='dataset_name')
    parser.add_argument('--method', default='EW', type=str, help='the weighting methods: Nash, PCGrad')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    yaml_config = parse_args()
    yaml_path = './' + yaml_config.dataset + '_' + yaml_config.method + '.yaml'
    model = Trainer(yaml_path)
    model.train()
