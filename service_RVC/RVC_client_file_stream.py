import librosa
import soundfile
import numpy as np
import sys
import pyaudio
import base64
import json
import time

import soundfile
import webrtcvad
import socketio
from tqdm.auto import tqdm
from queue import Queue
import threading


def read_audio_in_chunks(file_path, sample_rate=16000, chunk_duration=0.25):
    # 计算每个片段的样本数
    chunk_size = int(sample_rate * chunk_duration)
    # 使用 librosa 以指定采样率读取音频文件
    audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    # 计算音频的总样本数
    total_samples = len(audio)
    # 循环读取音频片段
    for start in range(0, total_samples, chunk_size):
        # 截取当前片段
        chunk = audio[start:start + chunk_size]
        # 如果当前片段长度不足 chunk_size，进行填充
        if len(chunk) < chunk_size:
            padding = np.zeros(chunk_size - len(chunk))
            chunk = np.concatenate((chunk, padding))
        yield chunk


server_url = 'https://u212392-acba-ac1c14ab.bjb1.seetacloud.com:8443/'  # 替换为实际的服务端地址
server_url = 'https://u212392-a13f-30455eaf.bjb1.seetacloud.com:8443/'  # 替换为实际的服务端地址
file_path = '/Users/zhou/Downloads/tmp/作为原声驱动/董宇辉带货_16k_mono.wav'
# file_path = '/Users/zhou/Downloads/下载 (3).wav'
# file_path = '/Users/zhou/Downloads/下载 (2).wav'
# file_path = '/Users/zhou/0-Codes/VoiceSamples/doctorwho_sliced/2月22日(3).wav'
"""
yujiesangsang.pth
yujie4.pth
yueliang40k.pth
xuexue.pth
xiumeng.pth
XIAOYUTING02.pth
xiaoru.pth
xianbao.pth
wuyusen.pth
wuyusen_manual.pth
wuyusen_manual_clear.pth
tt.pth
tianer15.pth
SHILIU.pth
Shaoyu.pth
qingju40k.pth
nikou48k.pth
mi-test.pth
manbo.pth
luoli.pth
lipuplusnew11.pth
lipuplusnew11 (1).pth
kikiv2.pth
kelala-v1_e200_s6200.pth
doubaoxin (1).pth
chuxueronghe.pth
chaojiwudijia.pth
bailu2.pth
acu.pth
"""
# 男变女，声音升调
model_name, f0 = "xuexue.pth", +12
model_name, f0 = "luoli.pth", +12
model_name, f0 = "kikiv2.pth", +12
# model_name, f0 = "manbo.pth", +12
# 男变男
# model_name, f0 = "wuyusen_manual_clear.pth", +0
BLOCK_TIME = 0.25
SR = 16000

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=False,  # 不作为输入流
                output=True)

# 创建 Socket.IO 客户端实例
sio = socketio.Client()
sio.connect(server_url, wait_timeout=5)

# 用于存储音频数据的队列
audio_queue = Queue()

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
    # 将音频数据放入队列
    audio_queue.put(audio_arr_int16)


# 子线程函数，用于从队列中取出音频数据并写入音频流
def play_audio():
    while True:
        if not audio_queue.empty():
            audio_arr_int16 = audio_queue.get()
            # 写入音频数据
            stream.write(audio_arr_int16.tobytes(), exception_on_underflow=False)
        else:
            time.sleep(0.01)


# 启动子线程
audio_thread = threading.Thread(target=play_audio)
audio_thread.start()


# loaded = False
# sio.emit('load_model', json.dumps({"model_name": model_name, "f0": f0, "block_time": BLOCK_TIME}))
# while not loaded:
#     time.sleep(1)
#     print(">>> 模型还未加载...")


# 调用函数按 250ms 片段读取音频
for idx, chunk in enumerate(read_audio_in_chunks(file_path, SR, BLOCK_TIME)):
    chunk = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
    # soundfile.write(f"/Users/zhou/Downloads/{idx}.wav", chunk, SR)
    audio_buffer = chunk.tobytes()
    # 构造请求数据
    audio_b64 = base64.b64encode(audio_buffer).decode()
    print(idx, chunk.shape, len(audio_buffer), len(audio_b64))
    # 发送音频数据到服务端
    sio.emit(event='audio_data',
             data=json.dumps({"audio": audio_b64, "traceId": f"{time.time()*1000:.0f}_123"}))
    time.sleep(0.1)

print("idle...")
while True:
    time.sleep(1)
