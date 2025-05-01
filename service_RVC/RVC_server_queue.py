from flask import Flask, request
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
from pika import BlockingConnection
from pika.adapters.blocking_connection import BlockingChannel

import signal
import torch
import sys
from subprocess import getstatusoutput, check_output
import multiprocessing as mp
from logging.handlers import TimedRotatingFileHandler
import logging
import random

from typing import Optional

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

M_dict = {}
request_queue_name_default = "queue_rvc_request"
result_queue_name_default = "queue_rvc_result"


class MQConnectionManager:
    """增强版连接管理器（支持多端口重试）"""
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_connection()
        return cls._instance

    def _init_connection(self):
        """带端口轮询的连接初始化"""
        rabbitmq_config = {
            "address": "120.24.144.127",
            "ports": [5672, 5673, 5674],
            "username": "admin",
            "password": "aibeeo",
            "virtual_host": "test-0208",
            "heartbeat": 600  # 增加心跳配置
        }

        # 随机打乱端口顺序实现负载均衡
        shuffled_ports = random.sample(rabbitmq_config["ports"], len(rabbitmq_config["ports"]))

        for port in shuffled_ports:
            try:
                credentials = pika.PlainCredentials(
                    rabbitmq_config["username"],
                    rabbitmq_config["password"]
                )
                parameters = pika.ConnectionParameters(
                    host=rabbitmq_config["address"],
                    port=port,
                    virtual_host=rabbitmq_config["virtual_host"],
                    credentials=credentials,
                    heartbeat=rabbitmq_config["heartbeat"],
                    blocked_connection_timeout=5  # 增加阻塞超时检测
                )
                self.connection = pika.BlockingConnection(parameters)
                logger.info(f"Connected via port {port}")
                return
            except Exception as e:
                logger.warning(f"Port {port} connection failed: {str(e)}")

        raise ConnectionError("All ports connection failed")

    def get_channel(self) -> BlockingChannel:
        """获取新通道（带自动重连）"""
        if not self.connection or self.connection.is_closed:
            self._init_connection()
        return self.connection.channel()


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
        when="midnight",  # 按天分隔（午夜生成新日志文件）
        interval=1,  # 每 1 天分隔一次
        backupCount=7,  # 最多保留最近 7 天的日志文件
        encoding="utf-8"  # 设置编码，避免中文日志乱码
    )
    file_handler.suffix = "%Y-%m-%d"  # 设置日志文件后缀格式，例如 server.log.2025-01-09
    file_handler.setFormatter(logging.Formatter(
        fmt='[%(asctime)s-%(levelname)s]: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # 将文件处理器添加到日志记录器中
    logger.addHandler(file_handler)
    return logger


# 常量定义
REQUEST_QUEUE_PREFIX = "queue_rvc_request"
RESPONSE_QUEUE_PREFIX = "queue_rvc_response"
END_STREAM_MARKER = b"END_STREAM"
HEARTBEAT_INTERVAL = 600  # 心跳间隔（秒）

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
    model = None  # 统一变量命名

    def cleanup_resources():
        nonlocal model
        if model is not None:
            del model
            torch.cuda.empty_cache()
            logger.info(f"Model {sid} resources released")

        if main_channel.is_open:
            main_channel.close()

    try:
        pth_file = f"/root/Retrieval-based-Voice-Conversion-WebUI_v2/assets/weights/{sid}.pth"
        model = RTRVCModel(pth_file, block_time=0.25)  # 变量名统一为model
        model.warmup()
        event.set()

        mq_manager = MQConnectionManager()
        main_channel = mq_manager.get_channel()

        # 声明主队列
        main_queue = f"{REQUEST_QUEUE_PREFIX}_{sid}"

        def signal_handler(sig, frame):
            logger.info(f"Received termination signal for sid={sid}")
            cleanup_resources()
            sys.exit(0)

        # 注册信号处理器 处理外部主进程触发的 p.terminate()
        signal.signal(signal.SIGTERM, signal_handler)

        def process_audio_stream(trace_id: str):
            """处理单个音频流任务"""
            channel = mq_manager.get_channel()
            sub_queue = f"{main_queue}_{trace_id}"
            response_queue = f"{RESPONSE_QUEUE_PREFIX}_{sid}_{trace_id}"

            while True:
                method, _, body = channel.basic_get(sub_queue, auto_ack=True)
                if not body:
                    continue

                params = json.loads(body.decode('utf-8'))

                # 检查params是否为字典
                if not isinstance(params, dict):
                    logger.error("params不是字典类型")
                    continue

                try:
                    if (params["type"] == 2):
                        logger.debug("收到结束标记")
                        # 删除sub_queue
                        channel.queue_delete(queue=sub_queue)
                        logger.debug(f"队列 {sub_queue} 已删除")
                        # 发送结束标记到response队列做删除
                        rsp = {"trace_id": trace_id,
                               "type": 2,
                               "sample_rate": sr}
                        rsp = json.dumps({"code": 0,
                                          "msg": "",
                                          "result": rsp})
                        channel.basic_publish(exchange='', routing_key=response_queue, body=rsp, properties=PROPERTIES)

                        break

                    audio_buffer = base64.b64decode(params["audio"])
                    audio_arr_int16_16khz = np.frombuffer(audio_buffer, dtype=np.int16)
                    audio_arr_float32_16khz = audio_arr_int16_16khz.astype(np.float32) / 32768.0
                    logger.debug(
                        f"trace_id='{trace_id}' receive: audio.shape={audio_arr_float32_16khz.shape} audio.dtype={audio_arr_float32_16khz.dtype} audio={audio_arr_float32_16khz[:5]} {np.min(audio_arr_float32_16khz)} {np.max(audio_arr_float32_16khz)}")
                    # Predict
                    sr, audio_opt = model.audio_callback_int16(audio_arr_float32_16khz)  # int16 16khz
                    # SendBack
                    rsp = {"trace_id": trace_id,
                           "type": 1,
                           "audio_buffer_int16": base64.b64encode(audio_opt.tobytes()).decode(),
                           "sample_rate": sr}
                    rsp = json.dumps({"code": 0,
                                      "msg": "",
                                      "result": rsp})

                    channel.basic_publish(exchange='', routing_key=response_queue, body=rsp, properties=PROPERTIES)
                except Exception as e:
                    logger.error(f"Processing error for {trace_id}: {str(e)}")
                    logger.error("Stack trace:", exc_info=True)
                    channel.basic_publish(
                        exchange='',
                        routing_key=response_queue,
                        body=json.dumps({"error": str(e)})
                    )
                    break
            channel.close()

            # 主任务循环

        while True:
            method, _, body = main_channel.basic_get(main_queue, auto_ack=True)
            if not body:
                time.sleep(1)
                continue

            try:
                logger.info(f"get task body: {body}")
                task = json.loads(body)
                trace_id = task['traceId']
                logger.info(f"Processing new task: {trace_id}")
                process_audio_stream(trace_id)
            except Exception as e:
                logger.error(f"Task processing failed: {str(e)}")
            finally:
                torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Critical error in model_process: {str(e)}")
    finally:
        cleanup_resources()


# 返回所有GPU的内存空余量，是一个list
def get_free_gpu_mem():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


@app.route("/load_model", methods=['POST'])
def load_model():
    """
    http params:
    - speaker:str
    - speaker_num:int
    """
    res = {"code": 0, "msg": "", "result": ""}
    info = request.get_json()
    sid = info['speaker']
    sid_num = info['speaker_num']
    #   download_overwrite = info.get("download_overwrite", "0")
    logger.info(f"load_model: {info}")

    # todo add 显存管理、模型卸载

    # 开启N个子进程加载模型
    # 保留inp用于发布关闭信息
    process_list = []
    _load_events = []
    for _ in range(sid_num):
        event = mp.Event()
        p = mp.Process(target=model_process, args=(sid, event))  # 移除q_out q_inp

        process_list.append(p)
        _load_events.append(event)
        p.start()

    # # 阻塞直到所有模型加载完毕
    # while not all([event.is_set() for event in _load_events]):
    #     pass

    M_dict[sid] = {"process_list": process_list}
    res['msg'] = "Init Success"
    return json.dumps(res)


if __name__ == '__main__':
    # socketio.run(app, debug=True, host='0.0.0.0', port=6006)
    # socketio.run(app, host='0.0.0.0', port=6006, allow_unsafe_werkzeug=True)
    logger = config_log()
    # 启动 Flask 服务
    logger.info("Starting Flask server...")

    app.run(host="0.0.0.0", port=8002)

