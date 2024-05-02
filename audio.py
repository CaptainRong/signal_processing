import os

import pyaudio
import numpy as np
import librosa
import librosa.display
import time
from model import FCNModel
from Hype import *


# 定义常量
CHUNK_SIZE = 22050  # 每个缓冲区的大小
FORMAT = pyaudio.paFloat32  # 输入音频格式
CHANNELS = 1  # 声道数
RATE = 22050  # 采样率
DURATION = 1  # 每次读取音频的持续时间（秒）


def get_mfcc(audio_data):
    # 计算MFCC
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=32)
    norm = np.linalg.norm(mfccs)
    # mfccs /= norm # 归一
    padding = [(0, 0),  # 在第一维度上扩张
               (0, 44 - mfccs.shape[1])]  # 在第二维度上扩张

    # 对矩阵进行扩张
    mfccs = np.pad(mfccs, padding, mode='constant', constant_values=0)
    mfccs = torch.FloatTensor(mfccs)
    mfccs = torch.unsqueeze(mfccs, 0).to(device)
    return mfccs


if __name__ == '__main__':
    pa = pyaudio.PyAudio()
    model = FCNModel(input_size=32 * 44, classes=CLASSES).to(device)
    model.load_state_dict(torch.load('model/complete_FCNModel-180-acc82.7710.pth'))
    label_lst = os.listdir("wav/train/")
    # 打开音频流
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK_SIZE)

    try:
        while True:
            # 读取音频数据
            audio_data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.float32)
            mfccs = get_mfcc(audio_data)

            with torch.no_grad():
                _, predicted = model(mfccs).max(1)
                print(label_lst[predicted.item()])

            # 等待1秒
            time.sleep(DURATION)

    except KeyboardInterrupt:
        # 处理键盘中断
        print("停止程序")

    finally:
        # 关闭音频流
        stream.stop_stream()
        stream.close()
        pa.terminate()
