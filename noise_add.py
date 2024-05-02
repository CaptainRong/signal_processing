import tkinter as tk
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa

class AudioRecorderApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Recorder")

        self.record_button = tk.Button(master, text="Start Recording", command=self.start_recording)
        self.record_button.pack()

        self.stop_button = tk.Button(master, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack()

    def start_recording(self):
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.frames = []

        def callback(indata, frames, time, status):
            self.frames.append(indata.copy())

        # 开始录音
        with sd.InputStream(callback=callback):
            self.master.mainloop()

    def stop_recording(self):
        self.stop_button.config(state=tk.DISABLED)
        self.record_button.config(state=tk.NORMAL)

        # 将录音数据保存为WAV文件
        audio_data = np.concatenate(self.frames, axis=0)
        sf.write('recorded_audio.wav', audio_data, 44100)

        # 缩放音频为1秒
        y_resized = librosa.effects.time_stretch(audio_data, rate=1.0 / len(audio_data))

        # 保存缩放后的音频
        sf.write('resized_audio.wav', y_resized, 44100)

root = tk.Tk()
app = AudioRecorderApp(root)
root.mainloop()