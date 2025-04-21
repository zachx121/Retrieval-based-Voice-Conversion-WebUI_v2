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
from queue import Queue
import threading
import pika
import signal
import torch
import sys
from logging.handlers import TimedRotatingFileHandler
import logging
logger = logging.getLogger("RVC_server")
logger.setLevel(logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("fairseq.tasks.hubert_pretraining").setLevel(logging.ERROR)
logging.getLogger("configs.config").setLevel(logging.ERROR)

PROJ_DIR = "/root/Retrieval-based-Voice-Conversion-WebUI_v2"
# msg properties for rabbmitmq
PROPERTIES = pika.BasicProperties(content_type='application/json')  # 设置 content_type 为 JSON
app = Flask(__name__, static_folder="./static_folder", static_url_path="")

logger.info(">>> load model")
pth_file = "/root/Retrieval-based-Voice-Conversion-WebUI_v2/assets/weights/chaojiwudijia.pth"
M: RTRVCModel = RTRVCModel(pth_file, block_time=0.25)
M.warmup()
logger.info(">>> load model(done.)")

request_queue_name = "queue_rvc_request"
result_queue_name = "queue_rvc_result"


def connect_to_rabbitmq():
    # RabbitMQ 连接信息
    rabbitmq_config = {
        "address": "120.24.144.127",
        "ports": [5672, 5673, 5674],
        "username": "admin",
        "password": "aibeeo",
        "virtual_host": "test-0208"
    }

    # 连接到 RabbitMQ
    credentials = pika.PlainCredentials(rabbitmq_config["username"], rabbitmq_config["password"])
    parameters = pika.ConnectionParameters(
        host=rabbitmq_config["address"],
        port=rabbitmq_config["ports"][0],  # 默认使用第一个端口
        virtual_host=rabbitmq_config["virtual_host"],
        credentials=credentials
    )

    try:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        logger.info("Connected to RabbitMQ successfully.")
        return connection, channel
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {repr(e)}")
        return None, None


def config_log(log_dir="./log", log_file="server.log"):
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 配置日志
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别为 INFO

    # 创建按天分隔的文件处理器
    log_path = os.path.join(log_dir, log_file)
    file_handler = TimedRotatingFileHandler(
        filename=log_path,  # 日志文件路径
        when="midnight",    # 按天分隔（午夜生成新日志文件）
        interval=1,         # 每 1 天分隔一次
        backupCount=7,      # 最多保留最近 7 天的日志文件
        encoding="utf-8"    # 设置编码，避免中文日志乱码
    )
    file_handler.suffix = "%Y-%m-%d"  # 设置日志文件后缀格式，例如 server.log.2025-01-09
    file_handler.setFormatter(logging.Formatter(
        fmt='[%(asctime)s-%(levelname)s]: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # 将文件处理器添加到日志记录器中
    logger.addHandler(file_handler)
    return logger


"""
GSV里是在load_model的同时，启动多个进程执行model_process消费mq
    q_inp = mp.Queue()
    process_list = []
    _load_events = []
    for _ in range(sid_num):
        event = mp.Event()
        p = mp.Process(target=model_process, args=(sid, event))  # 移除q_out q_inp

        process_list.append(p)
        _load_events.append(event)
        p.start()
"""
def model_process(sid: str, event):
    global M
    connection, channel = connect_to_rabbitmq()

    def signal_handler(sig, frame):
        # 关闭通道和连接
        if channel is not None:
            channel.stop_consuming()
            channel.close()
        if connection is not None:
            connection.close()
        logger.info(f"Close process of sid={sid}: Channel and connection closed.")
        if M is not None:
            del M
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        logger.info(f"Close process of sid={sid}: Model deleted.")
        sys.exit(0)

    # 注册信号处理器 处理外部主进程触发的 p.terminate()
    signal.signal(signal.SIGTERM, signal_handler)

    def call_back_func(ch, method, properties, body):
        try:
            params = json.loads(body.decode('utf-8'))
            tid = params['traceId']
            audio_buffer = base64.b64decode(params["audio"])
            audio_arr_int16_16khz = np.frombuffer(audio_buffer, dtype=np.int16)
            audio_arr_float32_16khz = audio_arr_int16_16khz.astype(np.float32) / 32768.0
            logger.debug(f"tid='{tid}' receive: audio.shape={audio_arr_float32_16khz.shape} audio.dtype={audio_arr_float32_16khz.dtype} audio={audio_arr_float32_16khz[:5]} {np.min(audio_arr_float32_16khz)} {np.max(audio_arr_float32_16khz)}")
            # Predict
            sr, audio_opt = M.audio_callback_int16(audio_arr_float32_16khz)  # int16 16khz
            # SendBack
            rsp = {"trace_id": tid,
                   "audio": base64.b64encode(audio_opt.tobytes()).decode(),
                   "sample_rate": sr}
            rsp = json.dumps({"code": 0,
                              "msg": "",
                              "result": rsp})
            channel.basic_publish(exchange='', routing_key=result_queue_name, body=rsp, properties=PROPERTIES)
        except Exception as e:
            logger.error(f"推理错误: {e}")
            raise e

    channel.basic_consume(queue=request_queue_name, auto_ack=True, on_message_callback=call_back_func)

    try:
        channel.start_consuming()
    except Exception as e:
        logger.error(f"Error during consuming in model_process. sid={sid}, error: {e}")
    finally:
        # 关闭通道和连接
        logger.warning(f"close channel/connection and del M in try-catch...")
        channel.close()
        connection.close()
        del M
        import gc
        gc.collect()
        torch.cuda.empty_cache()


@app.route("/load_model", methods=['POST'])
def load_model(data):
    logger.info(f">>> receive request on 'load_model', data: {data}")
    data = json.loads(data)
    global M
    model_name = data['model_name']  # e.g. 'wuyusen_manual_clear.pth'
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


if __name__ == '__main__':
    # socketio.run(app, debug=True, host='0.0.0.0', port=6006)
    # socketio.run(app, host='0.0.0.0', port=6006, allow_unsafe_werkzeug=True)
    logger = config_log()

