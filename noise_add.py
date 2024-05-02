import os
import random
import numpy as np
import scipy.io.wavfile as wav

# 设置数据集路径和背景噪声路径
dataset_path = "./"
background_noise_path = os.path.join(dataset_path, "_background_noise_")

# 加载背景噪声文件列表
background_noise_files = os.listdir(background_noise_path)


# 定义函数：从背景噪声中随机选择并添加到语音信号中
def add_background_noise(audio_signal, noise_path, noise_level=0.1):
    # 随机选择背景噪声文件
    noise_file = random.choice(os.listdir(noise_path))
    noise_file_path = os.path.join(noise_path, noise_file)

    # 读取背景噪声音频文件
    _, noise = wav.read(noise_file_path)

    # 根据噪声级别调整噪声信号的幅度
    noise = noise * noise_level

    # 将噪声信号添加到语音信号中
    augmented_audio = audio_signal + noise

    return augmented_audio


# 示例：读取语音信号文件并添加背景噪声
def main():
    # 读取语音信号文件
    audio_file = "path_to_audio_file.wav"
    sample_rate, audio_signal = wav.read(audio_file)

    # 添加背景噪声
    augmented_audio = add_background_noise(audio_signal, background_noise_path)

    # 可选：保存增强后的语音信号
    # wav.write("augmented_audio.wav", sample_rate, np.int16(augmented_audio))


if __name__ == "__main__":
    main()
