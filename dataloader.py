import torch
from torch.utils.data import Dataset
import numpy as np


class Transform:
    def __init__(self):
        pass

    def __call__(self, sample):
        input_data = sample['input']
        label = sample['label']

        # Apply your transformers methods here
        input_data = torch.FloatTensor(input_data)
        label = torch.FloatTensor(label)

        return {'input': input_data, 'label': label}


class CustomDataset(Dataset):
    def __init__(self, data, labels, classes, transform):
        self.data = data
        self.labels = np.eye(classes)[labels]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'input': self.data[idx],
            'label': self.labels[idx]
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':

    data = np.array([[1.3, 1.3], [2.3, 2.3], [3.4, 3.5]])
    N = len(data)
    labels = [1, 0, 1]

    # 创建自定义数据集
    custom_dataset = CustomDataset(data, labels, 2, transform=Transform())

    # 创建数据加载器
    data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=2, shuffle=True)

    for i, sample in enumerate(data_loader):
        input_data = sample['input']
        label = sample['label']
        print(f"Sample {i + 1}:")
        print("Input data:", input_data)
        print("Label:", label)
