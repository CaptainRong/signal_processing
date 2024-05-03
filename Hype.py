"""
超参数设置
"""
import torch

BATCHSIZE = 128
EPOCHS = 200
lr = 2e-3
CLASSES = 21
device = torch.device('cuda')
