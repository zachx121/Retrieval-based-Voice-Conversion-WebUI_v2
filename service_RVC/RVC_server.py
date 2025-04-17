from flask import Flask
from flask_socketio import SocketIO, send, emit
import os
from RVC_Model import RVCModel
from glob import glob
import numpy as np
import base64
import json

PROJ_DIR = "/root/Retrieval-based-Voice-Conversion-WebUI_v2"

app = Flask(__name__)
socketio = SocketIO(app)
M: RVCModel = RVCModel("/root/Retrieval-based-Voice-Conversion-WebUI_v2/assets/weights/chaojiwudijia.pth")


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
            print(f"model loaded '{model_name}'")
            emit('load_model', {'status': 'success', 'message': 'Model loaded successfully'})
        except Exception as e:
            # 发送错误消息给客户端
            print(f"ERROR model loaded '{str(e)}'")
            emit('load_model', {'status': 'error', 'message': str(e)})
    else:
        # 发送参数缺失消息给客户端
        print(f"ERROR model loaded no 'model_name'. {data}")
        emit('load_model', {'status': 'error', 'message': 'Missing model_name parameter'})
    print("new loaded M.pth_file: ", M.pth_file)


@socketio.on('audio_data')
def process_audio(data):
    global M
    # 将接收到的语音数据放入输入队列
    print("M.pth_file", M.pth_file)
    if M:
        try:
            data = json.loads(data)
            audio_buffer = base64.b64decode(data["audio"])
            audio_arr_int16_16khz = np.frombuffer(audio_buffer, dtype=np.int16)
            audio_arr_float32_16khz = audio_arr_int16_16khz.astype(np.float32) / 32768.0
            sr, audio = M.predict(audio_arr_float32_16khz, f0_up_key=int(data.get("f0", 0)))
            # >>>
            import scipy
            import time
            tag = int(time.time())
            scipy.io.wavfile.write(f"./{tag}_inp.wav", 16000, audio_arr_int16_16khz)
            scipy.io.wavfile.write(f"./{tag}_opt.wav", sr, audio)
            # <<<
            # 将处理后的语音数据发送回客户端
            audio_b64 = base64.b64encode(audio.tobytes()).decode()
            emit('process_audio', json.dumps({'status': 0, 'message': {"audio": audio_b64}}))
        except Exception as e:
            # 发送错误消息给客户端
            emit('process_audio', json.dumps({'status': 1, 'message': str(e)}))
    else:
        emit('process_audio', json.dumps({'status': 1, 'message': "Model not loaded"}))


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

