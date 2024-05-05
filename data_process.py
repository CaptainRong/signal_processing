import os
import numpy as np
import wave
import shutil
import random
import librosa
import librosa.feature


def dataset_chk(root_path, rate: float = 0):
    # 获取训练目录中的类别列表
    classes = os.listdir(root_path + 'train/')
    print('All classes are as follows:\n', classes)

    # 获取测试目录中的类别列表
    test_cls = os.listdir(root_path + "test/")

    # 对比检查测试集目录中是否有缺少的类别
    for cls in classes:
        if cls not in test_cls:
            # 如果指定了rate则生成
            if rate:
                print(f"{root_path + 'test/' + cls} not found! Auto generating test data at rate {rate} from train data.")
            # 如果未指定rate且缺少测试类别，则引发异常
            else:
                raise Exception(f"{root_path + 'test/' + cls} not found!")

            # 根据设置比率随机抽取并移动数据至测试集
            os.mkdir(root_path + 'test/' + cls)
            train_data = os.listdir(root_path + 'train/' + cls)
            files = random.sample(train_data, int(len(train_data) * rate))
            for file in files:
                shutil.move(root_path + 'train/' + cls + f"/{file}", root_path + 'test/' + cls + f"/{file}")


def librosa_get_wav_mfcc(wav_path):
    # 使用librosa.load加载音频文件，返回音频数据和采样率
    x, sr = librosa.load(wav_path)

    # 提取音频数据的MFCC特征，n_mfcc参数指定提取的MFCC特征数量
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=32)

    # 由于需要统一输入特征的维度大小，故需要对部分样本的特征进行填充
    padding = [(0, 0),
               (0, 44 - mfccs.shape[1])]

    # 使用np.pad函数对MFCC特征矩阵进行填充，以满足所需的时间长度
    mfccs = np.pad(mfccs, padding, mode='constant', constant_values=0)
    return mfccs


if __name__ == "__main__":
    dataset_chk(root_path='wav/', rate=0.3)
    # # print(data1.shape)
    # get_wav_mfcc(wav_path='wav/test/two/1000.wav')