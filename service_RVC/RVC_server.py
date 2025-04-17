import config
from flask import Flask
from flask_socketio import SocketIO, send, emit
import os
import sys
from dotenv import load_dotenv
from RVC_Model import RVCModel
from glob import glob
import numpy as np
import base64

PROJ_DIR = "/root/Retrieval-based-Voice-Conversion-WebUI_v2"

app = Flask(__name__)
socketio = SocketIO(app)
M: RVCModel = None


@socketio.on("get_model_list")
def get_model_list():
    fplist = glob(os.path.join(PROJ_DIR, "assets", "weights", "*.pth"))
    names = ",".join([os.path.basename(i) for i in fplist])
    emit('get_model_list', {'status': 'success', 'message': names})


@socketio.on('load_model')
def load_model(data):
    global M
    model_name = data.get('model_name')  # e.g. 'wuyusen_manual_clear.pth'
    if model_name:
        try:
            pth_file = os.path.join(PROJ_DIR, "assets", "weights", model_name)
            M = RVCModel(pth_file)
            # 发送成功消息给客户端
            emit('load_model', {'status': 'success', 'message': 'Model loaded successfully'})
        except Exception as e:
            # 发送错误消息给客户端
            emit('load_model', {'status': 'error', 'message': str(e)})
    else:
        # 发送参数缺失消息给客户端
        emit('load_model', {'status': 'error', 'message': 'Missing model_name parameter'})


@socketio.on('audio_data')
def process_audio(data):
    global M
    # 将接收到的语音数据放入输入队列
    audio_buffer = base64.b64decode(data)
    audio_arr_int16_16khz = np.frombuffer(audio_buffer, dtype=np.int16)
    audio_arr_float32_16khz = audio_arr_int16_16khz.astype(np.float32) / 32768.0
    sr, audio = M.predict(audio_arr_float32_16khz, f0_up_key=0)
    # 将处理后的语音数据发送回客户端
    emit('process_audio', base64.b64encode(audio.tobytes()).decode())


# 处理客户端连接
@socketio.on('connect')
def handle_connect():
    print('Client connected')


# 处理客户端断开连接
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    # socketio.run(app, debug=True, host='0.0.0.0', port=6006)
    socketio.run(app, host='0.0.0.0', port=6006)

