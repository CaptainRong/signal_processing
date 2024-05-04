import pyttsx3

context = {
    'on': "开始播放",
    'off': "暂停音乐",
    'alrdy_off': "您已暂停",
    'alrdy_on': "您已播放",
    'up': "已为您播放上一取",
    'down': "已为您播放下一取",
}


def speak(label):
    engine = pyttsx3.init()
    engine.say(context[label])
    engine.runAndWait()


if __name__ == '__main__':
    speak('on')
