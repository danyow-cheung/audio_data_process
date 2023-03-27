https://www.youtube.com/watch?v=ZqpSb5p1xQo

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

https://youtu.be/ZqpSb5p1xQo?t=573