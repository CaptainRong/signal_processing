import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from data_process import get_wav_mfcc, librosa_get_wav_mfcc
from dataloader import CustomDataset, Transform
from Hype import *


def create_datasets(root_path):
    wavs = []
    labels = []  # labels 和 testlabels 这里面存的值都是对应标签的下标，下标对应的名字在labsInd中
    testwavs = []
    testlabels = []

    labsInd = []  ## 训练集标签的名字   0：seven   1：stop
    testlabsInd = []  ## 测试集标签的名字   0：seven   1：stop

    root_train = root_path + "train/"
    root_test = root_path + "test/"
    classes = os.listdir(root_train)

    print("Loading train_datasets...")
    for cls in classes:
        print(f"{cls} loading...")
        path = root_train + f"{cls}/"
        files = os.listdir(path)
        for file in files:
            # print(file)
            waveData = librosa_get_wav_mfcc(path + file)
            # print(waveData)
            wavs.append(waveData)
            if not (cls in labsInd):
                labsInd.append(cls)
                # print(cls)
            labels.append(labsInd.index(cls))
        print(f"{cls} load finished")
        print(labsInd.index(cls))
    # 现在为了测试方便和快速直接写死，后面需要改成自动扫描文件夹和标签的形式

    print("Loading test_datasets...")
    for cls in classes:
        path = root_test + f"{cls}/"
        print(f"{cls} loading...")
        try:
            files = os.listdir(path)
            for file in files:
                # print(file)
                waveData = librosa_get_wav_mfcc(path + file)
                testwavs.append(waveData)
                if not (cls in testlabsInd):
                    # print(cls)
                    testlabsInd.append(cls)
                testlabels.append(testlabsInd.index(cls))

            print(f"{cls} load finished")
            # print(testlabsInd.index(cls))
        except:
            raise Exception(f"Test dataset {path} not found!")

    # print(labels, testlabels, labsInd, testlabsInd)
    wavs = np.array(wavs)
    labels = np.array(labels)
    testwavs = np.array(testwavs)
    testlabels = np.array(testlabels)

    TrainData = CustomDataset(wavs, labels, len(labsInd), transform=Transform())
    TestData = CustomDataset(testwavs, testlabels, len(testlabsInd), transform=Transform())
    # print(TrainData.data, TrainData.labels)
    # print(TestData.data, TestData.labels)
    TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=BATCHSIZE, shuffle=True)
    TestDataLoader = torch.utils.data.DataLoader(TestData, batch_size=BATCHSIZE, shuffle=True)

    return TrainDataLoader, TestDataLoader


if __name__ == '__main__':
    TrainDataLoader, TestDataLoader = create_datasets('wav/')
    print(TrainDataLoader.batch_size)
    print(TestDataLoader.dataset)

