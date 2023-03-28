

# 第三方库

liborsa 

ipython.display 

# 基本参数

频率

音调

音幅



## Sample Rate

Sample rate is specific to how the computer reads in the audio file 

Think of its as the 'resolution' of the image 

## spectrogram 频谱图
> https://vibrationresearch.com/blog/what-is-a-spectrogram/
> 
> https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505

### 为什么使用频谱图
1. 时域分析可以指出DUT中的缺陷 但没有具体说明缺陷的位置或性质。作为时频分析的集合，频谱图可用于识别非平稳或非线性信号的特征。因此，频谱图是分析具有各种频率分量和/或机械和电气噪声的真实世界数据的有用工具。

    **频谱图最有助于在不断变化的环境中进行振动分析。它说明了在 FFT 或PSD中可能看不到的能量变化模式**,与 FFT 相比，频谱图可以更好地了解振动随时间的变化情况。 
     在频谱图中，有许多损坏指标，而且它们可能很复杂。尽管如此，非典型条带仍然可以指示有关潜在损坏的非常有用的信息。

2. Deep learning models rarely take this raw audio directly as input. As we learned in Part 1, the common practice is to convert the audio into a spectrogram. The spectrogram is a concise ‘snapshot’ of an audio wave and since it is an image, it is well suited to being input to CNN-based architectures developed for handling images.
    深度學習模型很少直接將這種原始音頻作為輸入。 正如我們在第 1 部分中了解到的，通常的做法是將音頻轉換為頻譜圖。 頻譜圖是音頻波的簡明“快照”，並且由於它是圖像，因此非常適合輸入到為處理圖像而開發的基於 CNN 的架構中。

#### 和傅里叶变换之间的关系
<u>频谱图是使用傅里叶变换从声音信号生成的</u>。傅里叶变换将信号分解为其组成频率，并显示信号中存在的每个频率的幅度。

频谱图将声音信号的持续时间分成更小的时间段，然后对每个段应用傅立叶变换，以确定该段中包含的频率。然后它将所有这些片段的傅立叶变换组合成一个图。

它绘制了频率（y 轴）与时间（x 轴）的关系图，并使用不同的颜色来指示每个频率的幅度。颜色越亮，信号的能量就越高。


## MEl 频谱图 
> https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505

相对于绘制频率与时间关系的常规频谱图，梅尔频谱图有两个重要变化。

- 它在 y 轴上使用梅尔标度而不是频率。
- 它使用 Decibel Scale 而不是 Amplitude 来指示颜色。

对于深度学习模型，我们通常使用这个而不是简单的 Spectrogram。

示例代码
```python 
# use the mel-scale instead of raw frequency
sgram_mag, _ = librosa.magphase(sgram)
mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
librosa.display.specshow(mel_scale_sgram)
```
# 读取音频

## Ipython.display()

```python
path = ""
ipython.display(path)
```



## librosa.read()

```python
y,sample_rate = librosa.load(path,)
print(f"y={y} y.shape = {y.shape}, sr={sample_rate}")
```

https://youtu.be/ZqpSb5p1xQo?t=573 <br>
code ref:https://www.kaggle.com/code/robikscube/working-with-audio-in-python/notebook
> audio_data_process_1.py
> 


# 语音识别全流程
https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition


# [Hands On Signal Processing with Python](https://towardsdatascience.com/hands-on-signal-processing-with-python-9bda8aad39de)
basic operations of signal processing, namely the frequency analysis, the noise filtering and the amplitude spectrum extraction techniques.
信號處理的基本操作，即頻率分析、噪聲過濾和幅度譜提取技術。
