"""
refer to https://github.com/Mculover666/MP3Player
"""
import sys
from MP3Player import MP3Player
from PyQt5.QtWidgets import (QApplication)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MP3Player()
    sys.exit(app.exec_())