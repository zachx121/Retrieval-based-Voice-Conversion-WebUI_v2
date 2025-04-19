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
import scipy
import time
import logging
logger = logging.getLogger("RVC_server")
logger.setLevel(logging.DEBUG)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("fairseq.tasks.hubert_pretraining").setLevel(logging.ERROR)
logging.getLogger("configs.config").setLevel(logging.ERROR)

PROJ_DIR = "/root/Retrieval-based-Voice-Conversion-WebUI_v2"

app = Flask(__name__)
socketio = SocketIO(app)

logger.info(">>> load model")
pth_file = "/root/Retrieval-based-Voice-Conversion-WebUI_v2/assets/weights/chaojiwudijia.pth"
M: RTRVCModel = RTRVCModel(pth_file, block_time=0.25)
M.warmup()
logger.info(">>> load model(done.)")


@socketio.on("get_model_list")
def get_model_list():
    fplist = glob(os.path.join(PROJ_DIR, "assets", "weights", "*.pth"))
    names = ",".join([os.path.basename(i) for i in fplist])
    emit('get_model_list', {'status': 'success', 'message': names})


@socketio.on('load_model')
def load_model(data):
    logger.info(f">>> receive request on 'load_model', data: {data}")
    data = json.loads(data)
    global M
    model_name = data.get('model_name')  # e.g. 'wuyusen_manual_clear.pth'
    f0 = data.get('f0', 0)
    index_rate = data.get("index_rate", 0.0)
    block_time = data.get("block_time", 0.25)
    if model_name:
        try:
            pth_file = os.path.join(PROJ_DIR, "assets", "weights", model_name)
            M = RTRVCModel(pth_file, pitch=f0, index_rate=index_rate, block_time=block_time)
            M.warmup()
            # 发送成功消息给客户端
            logger.debug(f"model loaded '{model_name}'")
            emit('load_model', {'status': 'success', 'message': 'Model loaded successfully'})
        except Exception as e:
            # 发送错误消息给客户端
            logger.error(f"ERROR model loaded '{str(e)}'")
            emit('load_model', {'status': 'error', 'message': str(e)})
    else:
        # 发送参数缺失消息给客户端
        logger.error(f"ERROR model loaded no 'model_name'. {data}")
        emit('load_model', {'status': 'error', 'message': 'Missing model_name parameter'})
    logger.info(f"newly loaded M.pth_file: {M.pth_file}")


@socketio.on('audio_data')
def process_audio(data):
    global M
    logger.debug(f"\n>>> receive request on 'audio_data', data: {data}")
    if M:
        try:
            data = json.loads(data)
            tid = data.get("traceId", int(time.time()*1000))
            logger.debug(f"tid='{tid}' M.pth='{os.path.basename(M.pth_file)}' M.block_time={M.block_time}")
            audio_buffer = base64.b64decode(data["audio"])
            audio_arr_int16_16khz = np.frombuffer(audio_buffer, dtype=np.int16)
            audio_arr_float32_16khz = audio_arr_int16_16khz.astype(np.float32) / 32768.0
            logger.debug(f"tid='{tid}' receive: audio.shape={audio_arr_float32_16khz.shape} audio.dtype={audio_arr_float32_16khz.dtype} audio={audio_arr_float32_16khz[:5]} {np.min(audio_arr_float32_16khz)} {np.max(audio_arr_float32_16khz)}")
            sr, audio_opt = M.audio_callback_int16(audio_arr_float32_16khz)  # float32 16khz
            # >>>
            logger.debug(f"tid='{tid}' result: sr={sr} audio.shape={audio_opt.shape} audio.dtype={audio_opt.dtype} audio={audio_opt[:5]} {np.min(audio_opt)} {np.max(audio_opt)}")
            with open(f"./{tid}_inp.pcm", "wb") as fw:
                fw.write(audio_buffer)
            with open(f"./{tid}_opt.pcm", "wb") as fw:
                fw.write(audio_opt.tobytes())
            # <<<
            emit('process_audio', json.dumps({'status': 0, 'message': {"audio": base64.b64encode(audio_opt.tobytes()).decode()}}))
        except Exception as e:
            # 发送错误消息给客户端
            # raise e
            logger.error(e)
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


def play_all_pcm():
    pcm_list = []
    for fp in sorted(glob("*inp.pcm")):
        with open(fp, "rb") as frb:
            line = b"".join(frb.readlines())
            # len(line),fp
            pcm_list.append(line)
    audio_list = [np.frombuffer(i, dtype=np.int16) for i in pcm_list]
    print(len(audio_list), ", ".join([str(i.shape) for i in audio_list]))
    audio = np.hstack(audio_list)
    print(f"min: {audio.min()}, max: {audio.max()}")
    x = np.arange(audio.shape[0])
    # _ = plt.plot(x, audio)
    # plt.show()
    # Audio(audio, rate=16000)


def manual_infer_pcm():
    pth_file = "/root/Retrieval-based-Voice-Conversion-WebUI_v2/assets/weights/wuyusen_manual_clear.pth"
    M: RTRVCModel = RTRVCModel(pth_file, pitch=0, block_time=0.25)
    M.warmup()

    opt_list = []
    for fp in sorted(glob("*inp.pcm")):
        with open(fp, "rb") as frb:
            audio_buffer = b"".join(frb.readlines())

        audio_arr_int16_16khz = np.frombuffer(audio_buffer, dtype=np.int16)
        audio_arr_float32_16khz = audio_arr_int16_16khz.astype(np.float32) / 32768.0
        # logger.debug(f"tid='{tid}' receive: audio.shape={audio_arr_float32_16khz.shape} audio.dtype={audio_arr_float32_16khz.dtype} audio={audio_arr_float32_16khz[:5]} {np.min(audio_arr_float32_16khz)} {np.max(audio_arr_float32_16khz)}")
        sr, audio_opt = M.audio_callback(audio_arr_float32_16khz)  # float32 16khz
        audio_opt = audio_opt[:, 0]
        # logger.debug(f"tid='{tid}' audio_opt:{audio_opt}")
        audio_opt = (np.clip(audio_opt, -1.0, 1.0) * 32767).astype(np.int16)
        audio_b64 = base64.b64encode(audio_opt.tobytes()).decode()

        opt_list.append(audio_opt)

    # Audio(np.hstack(opt_list), rate=16000)


if __name__ == '__main__':
    # socketio.run(app, debug=True, host='0.0.0.0', port=6006)
    socketio.run(app, host='0.0.0.0', port=6006, allow_unsafe_werkzeug=True)

