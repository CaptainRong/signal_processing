import os
import wave

import librosa
import numpy as np
import pyaudio
import soundfile
import soundfile as sf

from Hype import *
from model import FCNModel


def get_mfcc(audio_data):
    print(audio_data.shape)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=32)
    print(mfccs.shape)
    norm = np.linalg.norm(mfccs)
    # mfccs /= norm # 归一
    padding = [(0, 0),  # 在第一维度上扩张
               (0, 44 - mfccs.shape[1])]  # 在第二维度上扩张

    # 对矩阵进行扩张
    mfccs = np.pad(mfccs, padding, mode='constant', constant_values=0)
    mfccs = torch.FloatTensor(mfccs)
    mfccs = torch.unsqueeze(mfccs, 0).to(device)
    return mfccs


def record_audio(duration, sample_rate):
    chunk = 1024
    format = pyaudio.paFloat32  # 设置数据类型为浮点数
    channels = 1
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk)

    print("Recording...")

    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.float32))  # 将数据转换为浮点数

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    audio_data = np.concatenate(frames)
    return audio_data


def resize_audio(audio, target_length):
    current_length = len(audio)
    scale = current_length / target_length
    y_resized = librosa.effects.time_stretch(audio, rate=scale)
    print('resized as:', y_resized.shape)
    save_path = '../resized_audio.wav'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存调整大小后的音频
    soundfile.write(save_path, y_resized, 22050)
    print('DEBUG: OUTPUT SAVE SUCCESSFULLY.')
    return y_resized


if __name__ == '__main__':
    # 录制1秒钟的音频
    output_filename = 'recorded_audio.wav'
    duration = 2 # 录制的时长（秒）
    sample_rate = 22050  # 采样率

    audio_data = record_audio(duration, sample_rate)
    # 以下是你之后的处理步骤，不需要保存到本地文件
    print(f"audio_data:{audio_data.shape}")
    target_length = sample_rate  # 1秒钟的采样率
    y_resized = resize_audio(audio_data, target_length)
    print(f"y_resized:{y_resized.shape}")
    mfccs = get_mfcc(y_resized)

    model = FCNModel(input_size=32 * 44, classes=CLASSES).to(device)
    model.load_state_dict(torch.load('model/complete_FCNModel-180-acc82.7710.pth'))
    label_lst = os.listdir("wav/train/")
    with torch.no_grad():
        _, predicted = model(mfccs).max(1)
        print(label_lst[predicted.item()])
