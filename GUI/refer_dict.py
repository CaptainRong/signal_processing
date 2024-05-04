"""
列出所有的指令：
目前：
1. 暂停
2. 播放
3. 下一曲
4. 上一曲
5. 播放模式（随机，顺序，单曲循环）
"""

refer_dict = {
    'stop': '暂停',
    'on': '播放',
    'up': '下一曲',
    'off': '上一曲',
}
refer_command = {
    'stop': 'playMusic',
    'on': 'playMusic',
    'off': 'prevMusic',
    'up': 'nextMusic',

}
