import pyttsx3

context = {
    'on': "开始播放",
    'stop': "已暂停",
    'up': "下一曲",
    'off': "上一曲",

}


def speak(label):
    engine = pyttsx3.init()
    engine.say(context[label])
    engine.runAndWait()


if __name__ == '__main__':
    speak('on')
