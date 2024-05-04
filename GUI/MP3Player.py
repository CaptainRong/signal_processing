from PyQt5.QtWidgets import (QWidget, QDesktopWidget,
                             QMessageBox, QHBoxLayout, QVBoxLayout, QSlider, QListWidget,
                             QPushButton, QLabel, QComboBox, QFileDialog)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import os, time
import configparser
from random import random
from Hype import *
import numpy as np
import pyaudio
import threading
from model import FCNModel
from resize_test import resize_audio, get_mfcc
from refer_dict import refer_dict
from utils import speak


class MP3Player(QWidget):
    def __init__(self):
        super().__init__()

        # some initial values
        self.frames = []
        self.stream = ...
        self.recording = False
        self.sr = self.sample_rate = 22050
        self.model = FCNModel(input_size=32 * 44, classes=CLASSES).to(device)
        self.model.load_state_dict(torch.load('../model/complete_FCNModel-180-acc82.7710.pth'))
        torch.no_grad()
        self.label_lst = os.listdir("../wav/train/")

        self.startTimeLabel = QLabel('00:00')
        self.endTimeLabel = QLabel('00:00')
        self.slider = QSlider(Qt.Horizontal, self)
        self.PlayModeBtn = QPushButton(self)
        self.playBtn = QPushButton(self)
        self.prevBtn = QPushButton(self)
        self.nextBtn = QPushButton(self)
        self.openBtn = QPushButton(self)
        self.voiceBeginBtn = QPushButton('开始', self)
        self.voiceEndBtn = QPushButton('结束', self)
        self.musicList = QListWidget()
        self.song_formats = ['mp3', 'm4a', 'flac', 'wav', 'ogg']
        self.songs_list = []
        self.cur_playing_song = ''
        self.is_pause = True
        self.player = QMediaPlayer()
        self.is_switching = False
        self.playMode = 0
        self.settingfilename = 'config.ini'
        self.textLable = QLabel('111信号处理')
        self.infoLabel = QLabel('v1.0.0')

        self.playBtn.setStyleSheet("QPushButton{border-image: url(resource/image/play.png)}")
        self.playBtn.setFixedSize(48, 48)
        self.nextBtn.setStyleSheet("QPushButton{border-image: url(resource/image/next.png)}")
        self.nextBtn.setFixedSize(48, 48)
        self.prevBtn.setStyleSheet("QPushButton{border-image: url(resource/image/prev.png)}")
        self.prevBtn.setFixedSize(48, 48)
        self.openBtn.setStyleSheet("QPushButton{border-image: url(resource/image/open.png)}")
        self.openBtn.setFixedSize(24, 24)
        self.PlayModeBtn.setStyleSheet("QPushButton{border-image: url(resource/image/sequential.png)}")
        self.PlayModeBtn.setFixedSize(24, 24)

        self.timer = QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.playByMode)

        self.hBoxSlider = QHBoxLayout()
        self.hBoxSlider.addWidget(self.startTimeLabel)
        self.hBoxSlider.addWidget(self.slider)
        self.hBoxSlider.addWidget(self.endTimeLabel)

        self.hBoxButton = QHBoxLayout()
        self.hBoxButton.addWidget(self.PlayModeBtn)
        self.hBoxButton.addStretch(1)
        self.hBoxButton.addWidget(self.prevBtn)
        self.hBoxButton.addWidget(self.playBtn)
        self.hBoxButton.addWidget(self.nextBtn)

        self.hBoxButton.addStretch(1)
        self.hBoxButton.addWidget(self.voiceBeginBtn)
        self.hBoxButton.addWidget(self.voiceEndBtn)
        self.hBoxButton.addWidget(self.openBtn)

        self.vBoxControl = QVBoxLayout()
        self.vBoxControl.addLayout(self.hBoxSlider)
        self.vBoxControl.addLayout(self.hBoxButton)

        self.hBoxAbout = QHBoxLayout()
        self.hBoxAbout.addWidget(self.textLable)
        self.hBoxAbout.addStretch(1)
        self.hBoxAbout.addWidget(self.infoLabel)

        self.vboxMain = QVBoxLayout()
        self.vboxMain.addWidget(self.musicList)
        self.vboxMain.addLayout(self.vBoxControl)
        self.vboxMain.addLayout(self.hBoxAbout)

        self.setLayout(self.vboxMain)

        self.openBtn.clicked.connect(self.openMusicFloder)
        self.playBtn.clicked.connect(self.playMusic)
        self.prevBtn.clicked.connect(self.prevMusic)
        self.nextBtn.clicked.connect(self.nextMusic)
        self.musicList.itemDoubleClicked.connect(self.doubleClicked)
        self.slider.sliderMoved[int].connect(lambda: self.player.setPosition(self.slider.value()))
        self.PlayModeBtn.clicked.connect(self.playModeSet)

        self.voiceBeginBtn.clicked.connect(self.voiceRcognitionThreadOn)
        self.voiceEndBtn.clicked.connect(self.voice_recognition_end)

        self.loadingSetting()

        self.initUI()

    # 初始化界面
    def initUI(self):
        self.resize(600, 400)
        self.center()
        self.setWindowTitle('音乐播放器')
        self.setWindowIcon(QIcon('resource/image/favicon.ico'))
        self.show()

    # 窗口显示居中
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 打开文件夹
    def openMusicFloder(self):
        self.cur_path = QFileDialog.getExistingDirectory(self, "选取音乐文件夹", './')
        if self.cur_path:
            self.showMusicList()
            self.cur_playing_song = ''
            self.startTimeLabel.setText('00:00')
            self.endTimeLabel.setText('00:00')
            self.slider.setSliderPosition(0)
            self.updateSetting()
            self.is_pause = True
            self.playBtn.setStyleSheet("QPushButton{border-image: url(resource/image/play.png)}")

    # 显示音乐列表
    def showMusicList(self):
        self.musicList.clear()
        for song in os.listdir(self.cur_path):
            if song.split('.')[-1] in self.song_formats:
                self.songs_list.append([song, os.path.join(self.cur_path, song).replace('\\', '/')])
                self.musicList.addItem(song)
        self.musicList.setCurrentRow(0)
        if self.songs_list:
            self.cur_playing_song = self.songs_list[self.musicList.currentRow()][-1]

    # 提示
    def Tips(self, message):
        QMessageBox.about(self, "提示", message)

    # 设置当前播放的音乐
    def setCurPlaying(self):
        self.cur_playing_song = self.songs_list[self.musicList.currentRow()][-1]
        self.player.setMedia(QMediaContent(QUrl(self.cur_playing_song)))

    # 播放/暂停播放
    def playMusic(self):
        if self.musicList.count() == 0:
            self.Tips('当前路径内无可播放的音乐文件')
            return
        if not self.player.isAudioAvailable():
            self.setCurPlaying()
        if self.is_pause or self.is_switching:
            self.player.play()
            self.is_pause = False
            self.playBtn.setStyleSheet("QPushButton{border-image: url(resource/image/pause.png)}")
        elif (not self.is_pause) and (not self.is_switching):
            self.player.pause()
            self.is_pause = True
            self.playBtn.setStyleSheet("QPushButton{border-image: url(resource/image/play.png)}")

    # 上一曲
    def prevMusic(self):
        self.slider.setValue(0)
        if self.musicList.count() == 0:
            self.Tips('当前路径内无可播放的音乐文件')
            return
        pre_row = self.musicList.currentRow() - 1 if self.musicList.currentRow() != 0 else self.musicList.count() - 1
        self.musicList.setCurrentRow(pre_row)
        self.is_switching = True
        self.setCurPlaying()
        self.playMusic()
        self.is_switching = False

    # 下一曲
    def nextMusic(self):
        self.slider.setValue(0)
        if self.musicList.count() == 0:
            self.Tips('当前路径内无可播放的音乐文件')
            return
        next_row = self.musicList.currentRow() + 1 if self.musicList.currentRow() != self.musicList.count() - 1 else 0
        self.musicList.setCurrentRow(next_row)
        self.is_switching = True
        self.setCurPlaying()
        self.playMusic()
        self.is_switching = False

        # 双击歌曲名称播放音乐

    def doubleClicked(self):
        self.slider.setValue(0)
        self.is_switching = True
        self.setCurPlaying()
        self.playMusic()
        self.is_switching = False

    # 根据播放模式自动播放，并刷新进度条
    def playByMode(self):
        # 刷新进度条
        if (not self.is_pause) and (not self.is_switching):
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.player.duration())
            self.slider.setValue(self.slider.value() + 1000)
        self.startTimeLabel.setText(time.strftime('%M:%S', time.localtime(self.player.position() / 1000)))
        self.endTimeLabel.setText(time.strftime('%M:%S', time.localtime(self.player.duration() / 1000)))
        # 顺序播放
        if (self.playMode == 0) and (not self.is_pause) and (not self.is_switching):
            if self.musicList.count() == 0:
                return
            if self.player.position() == self.player.duration():
                self.nextMusic()
        # 单曲循环
        elif (self.playMode == 1) and (not self.is_pause) and (not self.is_switching):
            if self.musicList.count() == 0:
                return
            if self.player.position() == self.player.duration():
                self.is_switching = True
                self.setCurPlaying()
                self.slider.setValue(0)
                self.playMusic()
                self.is_switching = False
        # 随机播放
        elif (self.playMode == 2) and (not self.is_pause) and (not self.is_switching):
            if self.musicList.count() == 0:
                return
            if self.player.position() == self.player.duration():
                self.is_switching = True
                self.musicList.setCurrentRow(random.randint(0, self.musicList.count() - 1))
                self.setCurPlaying()
                self.slider.setValue(0)
                self.playMusic()
                self.is_switching = False

    # 更新配置文件
    def updateSetting(self):
        config = configparser.ConfigParser()
        config.read(self.settingfilename)
        if not os.path.isfile(self.settingfilename):
            config.add_section('MP3Player')
        config.set('MP3Player', 'PATH', self.cur_path)
        config.write(open(self.settingfilename, 'w'))

    # 加载配置文件
    def loadingSetting(self):
        config = configparser.ConfigParser()
        config.read(self.settingfilename)
        if not os.path.isfile(self.settingfilename):
            return
        self.cur_path = config.get('MP3Player', 'PATH')
        self.showMusicList()

    # 播放模式设置
    def playModeSet(self):
        # 设置为单曲循环模式
        if self.playMode == 0:
            self.playMode = 1
            self.PlayModeBtn.setStyleSheet("QPushButton{border-image: url(resource/image/circulation.png)}")
        # 设置为随机播放模式
        elif self.playMode == 1:
            self.playMode = 2
            self.PlayModeBtn.setStyleSheet("QPushButton{border-image: url(resource/image/random.png)}")
        # 设置为顺序播放模式
        elif self.playMode == 2:
            self.playMode = 0
            self.PlayModeBtn.setStyleSheet("QPushButton{border-image: url(resource/image/sequential.png)}")

    # 确认用户是否要真正退出
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "确定要退出吗？", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def command_activate(self, label):
        print(f"正在执行{label}\n目前的指令有{refer_dict.keys()}")
        if label in refer_dict.keys():
            print(f"执行成功")
            """有点丑陋的逻辑判断，只能说能用就行"""
            if label == "stop" or label == "on":
                self.playMusic()
            elif label == "up":
                self.nextMusic()
            elif label == "off":
                self.prevMusic()
            """语音合成"""
            speak(label)



        else:
            print(f"无法识别指令，请重新告诉我指令")
        pass

    def voice_recognition_end(self):
        self.recording = False
        print('Loading stream and audio...')
        (stream, audio) = self.stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print('audio terminated.')

        print(f'frames all get.')
        audio_data = np.concatenate(self.frames)

        y_resized = resize_audio(audio_data, self.sample_rate * 1)
        mfcc = get_mfcc(y_resized)
        _, predicted = self.model(mfcc).max(1)
        result = self.label_lst[predicted.item()]
        print(f"1111:{result},type:{type(result)}")
        """执行识别到的指令"""
        self.command_activate(result)

    def voiceRcognitionThreadOn(self):
        # set default values
        print("Recording...")
        threading._start_new_thread(self.voiceRcognitionBegin, ())

    def voiceRcognitionBegin(self):
        chunk = 1024
        format = pyaudio.paFloat32  # 设置数据类型为浮点数
        channels = 1
        audio = pyaudio.PyAudio()
        self.recording = True
        stream = audio.open(format=format, channels=channels,
                            rate=self.sample_rate, input=True,
                            frames_per_buffer=chunk)
        self.stream = (stream, audio)

        frames = []
        while self.recording:
            data = stream.read(chunk)
            frames.append(np.frombuffer(data, dtype=np.float32))  # 将数据转换为浮点数
            if not self.recording:
                self.frames = frames
                print(f"frames uploaded to class successfully,{self.playMode}")
        pass
