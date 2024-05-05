import torch

BATCHSIZE = 128
EPOCHS = 201
lr = 2e-3
CLASSES = 21
device = torch.device('cuda')


class global_value():
    def __init__(self, value):
        self.value = value

    def update(self, new):
        self.value = new

    def get(self):
        return self.value
