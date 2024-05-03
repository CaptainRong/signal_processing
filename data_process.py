import os
import numpy as np
import wave
import shutil
import random
import librosa
import librosa.feature


def dataset_chk(root_path, rate: float = 0):
    # for dir in os.listdir(root_path):
    #     path = os.path.join(root_path, dir)
    #     # print(1, path)
    #     for cls in os.listdir(path):
    #         # print(2, cls)
    #         pth = os.path.join(path, cls)
    #         for i, data in enumerate(os.listdir(pth)):
    #             ext = data.split(".")[1]
    #             # 新文件名
    #             name = f'{i:04d}.{ext}'
    #             # 重命名文件
    #             while os.path.exists(os.path.join(pth, name)):
    #                 i += 1
    #                 name = f'{i:04d}.{ext}'
    #
    #             os.rename(os.path.join(pth, data), os.path.join(pth, name))
    #             # print(3, name)
    #         print(f'cls {cls} chk finished.')
    classes = os.listdir(root_path + 'train/')
    print('All classes are as follows:\n', classes)
    test_cls = os.listdir(root_path + "test/")
    for cls in classes:
        if cls not in test_cls:
            if rate:
                print(f"{root_path + 'test/' + cls} not found! Auto generating test data at rate {rate} from train data.")
            else:
                raise Exception(f"{root_path + 'test/' + cls} not found!")
            os.mkdir(root_path + 'test/' + cls)
            train_data = os.listdir(root_path + 'train/' + cls)
            files = random.sample(train_data, int(len(train_data) * rate))
            for file in files:
                shutil.move(root_path + 'train/' + cls +f"/{file}", root_path + 'test/' + cls+f"/{file}")


def get_wav_mfcc(wav_path):
    f = wave.open(wav_path, 'rb')
    params = f.getparams()
    # print("params:",params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
    waveData = np.reshape(waveData, [nframes, nchannels]).T
    f.close()
    ### 对音频数据进行长度大小的切割，保证每一个的长度都是一样的【因为训练文件全部是1秒钟长度，16000帧的，所以这里需要把每个语音文件的长度处理成一样的】
    data = list(np.array(waveData[0]))
    # print(len(data))
    while len(data) > 16000:
        del data[len(waveData[0]) - 1]
        del data[0]
    # print(len(data))
    while len(data) < 16000:
        data.append(0)
    # print(len(data))

    data = np.array(data)

    # 平方之后，开平方，取正数，值的范围在  0-1  之间
    data = data ** 2
    data = data ** 0.5
    data = np.reshape(data, (1, 16000))

    return data


def librosa_get_wav_mfcc(wav_path):
    x, sr = librosa.load(wav_path)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=32)
    norm = np.linalg.norm(mfccs)
    # mfccs /= norm # 归一
    padding = [(0, 0),  # 在第一维度上扩张
               (0, 44 - mfccs.shape[1])]  # 在第二维度上扩张

    # 对矩阵进行扩张
    mfccs = np.pad(mfccs, padding, mode='constant', constant_values=0)

    # mfccs = np.abs(mfccs)
    # print(mfccs.shape) (32->features, 44->time_length)
    return mfccs
    print(wav_path, mfccs, mfccs.shape)


if __name__ == "__main__":
    dataset_chk(root_path='wav/', rate=0.3)
    # # print(data1.shape)
    # get_wav_mfcc(wav_path='wav/test/two/1000.wav')