import sys
import pyaudio
import numpy as np
import base64
import json
import time
import webrtcvad
import socketio

# 配置音频参数
FORMAT = pyaudio.paInt16
OUTPUT_FORMAT, OUTPUT_RATE = pyaudio.paInt16, 1600  # 服务端返回的是 int16 16khz 格式
CHANNELS = 1
RATE = 16000
# 修改为 30ms 一帧，满足 webrtcvad 要求
CHUNK_TIME = 30
CHUNK = int(CHUNK_TIME * RATE / 1000)

server_url = 'https://u212392-a13f-30455eaf.bjb1.seetacloud.com:8443/'  # 替换为实际的服务端地址
model_name, f0 = "wuyusen_manual_clear.pth", 0
model_name, f0 = "chaojiwudijia.pth", 12
inp_device, opt_device = "桐的AirPods Pro #4", "MacBook Pro扬声器"
# inp_device, opt_device = "MacBook Pro麦克风", "桐的AirPods Pro #4"
# inp_device, opt_device = "MacBook Pro麦克风", "MacBook Pro扬声器"
audio_frames = []
silent_frames_count = 0
VOLUME_THRESHOLD = 50
SILENT_FRAMES_THRESHOLD = 5  # 连续 5 个无声帧视为停顿
SRC_LANG, TGT_LANG = "zh", "en"

p = pyaudio.PyAudio()

# 获取所有可用的音频设备
inp_device_idx, opt_device_idx = None, None

# 查找输入设备（例如 "Tong的AirPods"）
print([(p.get_device_info_by_index(i)["name"],
        p.get_device_info_by_index(i)["maxInputChannels"],
        p.get_device_info_by_index(i)["maxOutputChannels"])
       for i in range(p.get_device_count())])
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if device_info["name"] == inp_device and device_info["maxInputChannels"] > 0:
        inp_device_idx = i
    if device_info["name"] == opt_device and device_info["maxOutputChannels"] > 0:
        opt_device_idx = i

# sys.exit(0)
print(f">>> 准备输入流 输入设备将使用 {inp_device_idx} {p.get_device_info_by_index(inp_device_idx)['name']}")
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=inp_device_idx,
                output=True,
                output_device_index=opt_device_idx,
                frames_per_buffer=CHUNK)
print(f">>> 准备输出流 输出设备将使用 {opt_device_idx} {p.get_device_info_by_index(opt_device_idx)['name']}")

print(">> 初始化 VAD")
vad = webrtcvad.Vad(3)


def calculate_volume(audio_data):
    # 检查并处理 NaN 和 Inf 值
    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sqrt(np.mean(np.square(audio_data)))


# 标志位，用于判断是否已经开始说话
is_speaking = False

# 创建 Socket.IO 客户端实例
sio = socketio.Client()
sio.connect(server_url)

loaded=False
sio.emit('load_model', {"model_name": model_name})
@sio.on('load_model')
def on_load_model(info):
    global loaded
    print(info)
    loaded = True


while not loaded:
    time.sleep(1)
    print(">>> 模型还未加载...")

print("* 开始录音")


@sio.on('process_audio')
def on_process_audio(info):
    info = json.loads(info)
    if info['status'] != 0:
        print(info)
        return
    global audio_frames, silent_frames_count, is_speaking
    # 解析音频数据
    audio_bytes = base64.b64decode(info["message"]["audio"])
    # 转换为音频数组（int16）
    audio_arr_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    e = time.time()
    # 播放音频（创建输出流）
    stream.write(audio_arr_int16.tobytes())  # 写入音频数据
    print(f"返回结果: elapse={e - b:.2f}s")
    # 清空音频帧列表
    audio_frames = []
    silent_frames_count = 0
    is_speaking = False


while True:
    data = stream.read(CHUNK, exception_on_overflow=False)

    # 进行 VAD 检测
    is_speech = vad.is_speech(data, RATE)

    if is_speech:
        if not is_speaking:
            print("检测到开始说话")
        # 如果检测到语音，开始记录音频帧
        is_speaking = True
        audio_frames.append(data)
        silent_frames_count = 0
    else:
        if is_speaking:
            # 如果已经开始说话，遇到无声帧则增加计数
            silent_frames_count += 1
            audio_frames.append(data)
        else:
            # 如果还未开始说话，清空音频帧列表
            audio_frames = []

    if is_speaking and silent_frames_count >= SILENT_FRAMES_THRESHOLD:
        print("检测到停顿，发送音频数据到服务端")
        audio_buffer = b''.join(audio_frames)
        # 构造请求数据
        audio_b64 = base64.b64encode(audio_buffer).decode()
        b = time.time()
        # 发送音频数据到服务端
        sio.emit('audio_data', json.dumps({"audio": audio_b64, "f0": f0}))

        # 清空音频帧列表
        audio_frames = []
        silent_frames_count = 0
        is_speaking = False


print("* 录音结束")

stream.stop_stream()
stream.close()
p.terminate()
sio.disconnect()
