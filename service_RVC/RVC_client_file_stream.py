import librosa
import numpy as np
import sys
import pyaudio
import numpy as np
import base64
import json
import time
import webrtcvad
import socketio
from tqdm.auto import tqdm


def read_audio_in_chunks(file_path, sample_rate=16000, chunk_duration=0.25):
    # 计算每个片段的样本数
    chunk_size = int(sample_rate * chunk_duration)
    # 使用 librosa 以指定采样率读取音频文件
    audio, _ = librosa.load(file_path, sr=sample_rate)
    # 计算音频的总样本数
    total_samples = len(audio)
    # 循环读取音频片段
    for start in range(0, total_samples, chunk_size):
        end = start + chunk_size
        # 截取当前片段
        chunk = audio[start:end]
        # 如果当前片段长度不足 chunk_size，进行填充
        if len(chunk) < chunk_size:
            padding = np.zeros(chunk_size - len(chunk))
            chunk = np.concatenate((chunk, padding))
        yield chunk


server_url = 'https://u212392-acba-ac1c14ab.bjb1.seetacloud.com:8443/'  # 替换为实际的服务端地址
file_path = '/Users/zhou/Downloads/tmp/作为原声驱动/董宇辉带货_16k_mono.wav'
model_name = "wuyusen_manual_clear.pth"
model_name = "kikiv2.pth"

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=False,  # 不作为输入流
                output=True)


# 创建 Socket.IO 客户端实例
sio = socketio.Client()
sio.connect(server_url)


@sio.on('load_model')
def on_load_model(info):
    global loaded
    print(info)
    loaded = True


@sio.on('process_audio')
def on_process_audio(info):
    info = json.loads(info)
    if info['status'] != 0:
        print(f"process failed, info:'{info}'")
        return
    # 解析音频数据
    audio_bytes = base64.b64decode(info["message"]["audio"])
    # 转换为音频数组（int16）
    audio_arr_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    # 播放音频（创建输出流）
    stream.write(audio_arr_int16.tobytes(), exception_on_underflow=False)  # 写入音频数据


loaded = False
sio.emit('load_model', {"model_name": model_name})
while not loaded:
    time.sleep(1)
    print(">>> 模型还未加载...")


sr = 16000
# 调用函数按 250ms 片段读取音频
for chunk in tqdm(read_audio_in_chunks(file_path, sr)):
    chunk = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
    audio_buffer = chunk.tobytes()
    # 构造请求数据
    audio_b64 = base64.b64encode(audio_buffer).decode()
    b = time.time()
    # 发送音频数据到服务端
    sio.emit('audio_data', json.dumps({"audio": audio_b64}))
    time.sleep(0.3)


while True:
    time.sleep(1)
