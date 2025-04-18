from flask import Flask
from flask_socketio import SocketIO, send, emit
import os
# from RVC_Model import RVCModel
from RVC_RTModel import RTRVCModel
from glob import glob
import numpy as np
import base64
import json
import librosa
import logging
logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)

PROJ_DIR = "/root/Retrieval-based-Voice-Conversion-WebUI_v2"

app = Flask(__name__)
socketio = SocketIO(app)

pth_file = "/root/Retrieval-based-Voice-Conversion-WebUI_v2/assets/weights/chaojiwudijia.pth"
M: RTRVCModel = RTRVCModel(pth_file)


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
            M = RTRVCModel(pth_file)
            M.warmup()
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
    print(f"newly loaded M.pth_file: {M.pth_file}")


@socketio.on('audio_data')
def process_audio(data):
    global M
    logging.debug(f"process_audio...")
    if M:
        try:
            data = json.loads(data)
            logging.debug(f"process_audio {data.get('trace_id','default_id')} with M.pth_file: {M.pth_file}")
            audio_buffer = base64.b64decode(data["audio"])
            audio_arr_int16_16khz = np.frombuffer(audio_buffer, dtype=np.int16)
            audio_arr_float32_16khz = audio_arr_int16_16khz.astype(np.float32) / 32768.0
            sr, audio_opt = M.audio_callback(audio_arr_float32_16khz)  # float32 16khz
            audio_opt = (np.clip(audio_opt[:, 0], -1.0, 1.0) * 32767).astype(np.int16)
            # # >>>
            # print(f"result is: sr={sr} audio.shape={audio_opt.shape} audio.dtype={audio_opt.dtype} audio={audio_opt[:5]}")
            # import scipy
            # import time
            # tag = int(time.time())
            # scipy.io.wavfile.write(f"./{tag}_inp.wav", 16000, audio_arr_int16_16khz)
            # scipy.io.wavfile.write(f"./{tag}_opt.wav", sr, audio_opt)
            # # <<<
            # 将处理后的语音数据发送回客户端
            audio_b64 = base64.b64encode(audio_opt.tobytes()).decode()
            emit('process_audio', json.dumps({'status': 0, 'message': {"audio": audio_b64}}))
        except Exception as e:
            # 发送错误消息给客户端
            raise e
            print(e)
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


def play_all_wav():
    inp, opt = [], []
    for fp_inp in glob("/root/Retrieval-based-Voice-Conversion-WebUI_v2/*_inp.wav"):
        fp_opt = fp_inp.replace("_inp.wav", "_opt.wav")
        inp.append(librosa.load(fp_inp, sr=16000, mono=True)[0])
        opt.append(librosa.load(fp_opt, sr=16000, mono=True)[0])
    # Audio(np.hstack(inp), rate=16000)
    # Audio(np.hstack(opt), rate=16000)


if __name__ == '__main__':
    # socketio.run(app, debug=True, host='0.0.0.0', port=6006)
    socketio.run(app, host='0.0.0.0', port=6006)

