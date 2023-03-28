'''
ref:https://www.kaggle.com/code/robikscube/working-with-audio-in-python/notebook
'''
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

audio_files = glob('archive/*/*.wav')
print(audio_files)

# 不使用多频道音频文件
# data_path = f'E:/QQFile/20230327/20230103/2023-01-03_02_19_38.wav'

# play audio file
ipd.display(audio_files[0])
'''
librosa.load()将音频文件加载为浮点时间序列。音频将自动重新采样到给定的速率
返回音频时间序列，采样率
'''
y,sr = librosa.load(audio_files[0],sr=None) # 保留文件原始采样率
print(f'y: {y[:10]}')
print(f'shape y: {y.shape}')
print(f'sr: {sr}')
# 可视化将y转换为series
pd.Series(y).plot(figsize=(10,5),lw=1,title='Raw Audio img')
plt.show()

# 删除前后为静音的片段，只保留有声音片段
'''
librosa.effects.trim()修剪音频信号中的前导和尾随静音
返回微调信号，和非静音区对应的音程
'''
y_trimmed ,_ = librosa.effects.trim(y,top_db=20)
pd.Series(y_trimmed).plot(figsize=(10,5),lw=1,title='Raw Audio Trimmed img')
plt.show()
# 可以通过调节切片索引来更近一步的查看音频信息
pd.Series(y_trimmed[30000:35000]).plot(figsize=(10,5),lw=1,title='Raw Audio Trimmed img')
plt.show()

# 绘制頻譜圖
'''
librosa.stft() 短时傅里叶变换，通过在重叠窗口上计算离散傅里叶变换表示1时频域中的信号
返回复值矩阵
'''
D = librosa.stft(y)
'''
librosa.amplitude_to_db()将振幅谱图转换为db标度谱图
返回一个矩阵
'''
S_db = librosa.amplitude_to_db(np.abs(D),ref=np.max)
print(S_db.shape)
# 可视化
fig,ax = plt.subplots(figsize=(10,5))
'''
librosa.display.specshow()显示频谱图/色谱图/cqt/等。
'''
img = librosa.display.specshow(S_db,x_axis='time',y_axis='log',ax=ax)
ax.set_title('Spectogram Example')# 频谱图
fig.colorbar(img,ax=ax,format=f'%0.2f')
plt.show()

# Mel 频谱图
'''
librosa.feature.melspectrogram()计算梅尔标度的频谱图。
    如果提供了频谱图输入S，则它会通过 直接映射到 mel 基础上mel_f.dot(S)。
    如果提供 时间序列输入，则首先计算其幅度谱图，然后通过 映射到梅尔刻度 。y, srSmel_f.dot(S**power)
返回梅尔谱图矩阵
'''
S_mel = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128*2)
S_db_mel = librosa.amplitude_to_db(S_mel,ref=np.max)
# 可视化
fig, ax = plt.subplots(figsize=(10, 5))
img = librosa.display.specshow(S_db_mel,x_axis='time',y_axis='log',ax=ax)
ax.set_title('MEL Spectogram Example')# MEl频谱图
fig.colorbar(img,ax=ax,format=f'%0.2f')
plt.show()
