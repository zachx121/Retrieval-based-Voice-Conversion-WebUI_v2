import os
PROJ_DIR = os.path.abspath(os.path.join(__file__, "../../"))
print("PROJ_DIR", PROJ_DIR)
import sys
sys.path.append(PROJ_DIR)
print(sys.path)

import torch
import torchaudio.transforms as tat
import torch.nn.functional as F
from infer.modules.gui import TorchGate
from gui import phase_vocoder
import numpy as np
from configs import Config
from infer.lib import rtrvc
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import load_hubert
from rvc.synthesizer import load_synthesizer
from dotenv import load_dotenv
import time
import librosa

load_dotenv(os.path.join(PROJ_DIR, ".env"))
load_dotenv(os.path.join(PROJ_DIR, "sha256.env"))


class GUIConfig:
    def __init__(self, n_cpu=8) -> None:
        self.pth_path: str = ""
        self.index_path: str = ""
        self.pitch: int = 0
        self.formant: float = 0.0
        self.sr_type: str = "sr_model"
        self.block_time: float = 0.25  # s
        self.threhold: int = -60
        self.crossfade_time: float = 0.05
        self.extra_time: float = 2.5
        self.I_noise_reduce: bool = False
        self.O_noise_reduce: bool = False
        self.use_pv: bool = False
        self.rms_mix_rate: float = 0.0
        self.index_rate: float = 0.0
        self.n_cpu: int = min(n_cpu, 4)
        self.f0method: str = "rmvpe"
        self.sg_hostapi: str = ""
        self.wasapi_exclusive: bool = False
        self.sg_input_device: str = ""
        self.sg_output_device: str = ""


class RTRVCModel:
    def __init__(self, pth_file, idx_file="", block_time=0.25, sr=None):
        self.gui_config = GUIConfig()
        self.config = Config()
        self.gui_config.pth_path = self.pth_file = pth_file
        self.gui_config.index_path = self.idx_file = idx_file
        self.gui_config.block_time = self.block_time = block_time
        self.function = "vc"

        # >>> start_vc:
        torch.cuda.empty_cache()
        self.rvc: rtrvc.RVC = rtrvc.RVC(
            self.gui_config.pitch,
            self.gui_config.formant,
            self.gui_config.pth_path,
            self.gui_config.index_path,
            self.gui_config.index_rate,
            self.gui_config.n_cpu,
            self.config.device,
            self.config.use_jit,
            self.config.is_half,
            self.config.dml,
        )

        self.gui_config.samplerate = self.sr = self.rvc.tgt_sr if sr is None else sr
        self.gui_config.channels = 1
        self.zc = self.gui_config.samplerate // 100
        self.block_frame = (
                int(
                    np.round(
                        self.gui_config.block_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
                int(
                    np.round(
                        self.gui_config.crossfade_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
                int(
                    np.round(
                        self.gui_config.extra_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
        )
        self.input_wav: torch.Tensor = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
        self.input_wav_res: torch.Tensor = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer: torch.Tensor = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
        self.output_buffer: torch.Tensor = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
                                     self.block_frame + self.sola_buffer_frame + self.sola_search_frame
                             ) // self.zc
        self.fade_in_window: torch.Tensor = (
                torch.sin(
                    0.5
                    * np.pi
                    * torch.linspace(
                        0.0,
                        1.0,
                        steps=self.sola_buffer_frame,
                        device=self.config.device,
                        dtype=torch.float32,
                    )
                )
                ** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.gui_config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        if self.rvc.tgt_sr != self.gui_config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.gui_config.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
        else:
            self.resampler2 = None
        self.tg = TorchGate(
            sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.config.device)

    @NotImplementedError
    def _predict(self,
                audio_16khz_float32,
                f0_up_key=0,
                f0_method="rmvpe",
                index_rate=0.85,
                filter_radius=3,
                rms_mix_rate=0.25,
                protect=0.33,
                resample_sr=16000):
        """

        Args:
            audio_16khz_float32: input
            f0_up_key: 12是一个八度，女变男降八度(-12)、男变女升八度(+12)
            f0_method: "pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"
            index_rate: 0~1, 模拟得有多相似
            filter_radius: 0~7, If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.
            rms_mix_rate: 0~1, 音量模拟 0 mimicks volume of original vocals, 1 stands for consistently loud volume
            protect: 0~0.5 Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy
            resample_sr: resample_sr
        Returns:

        """
        infer_wav = self.rvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            self.gui_config.f0method,
        )

    def audio_callback(self, indata: np.ndarray):
        """
        音频处理
        indata: 长度为10000的float32数组（40khz音频的250ms片段）
        """
        global flag_vc
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        if self.gui_config.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc:]
            indata = indata[2 * self.zc - self.zc // 2:]
            db_threhold = (
                    librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc: (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2:]
        self.input_wav[: -self.block_frame] = self.input_wav[
                                              self.block_frame:
                                              ].clone()
        self.input_wav[-indata.shape[0]:] = torch.from_numpy(indata).to(
            self.config.device
        )
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                                                      self.block_frame_16k:
                                                      ].clone()
        # input noise reduction and resampling
        if self.gui_config.I_noise_reduce:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                                                          self.block_frame:
                                                          ].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame:]
            input_wav = self.tg(
                input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
            ).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += (
                    self.nr_buffer * self.fade_out_window
            )
            self.input_wav_denoise[-self.block_frame:] = input_wav[
                                                         : self.block_frame
                                                         ]
            self.nr_buffer[:] = input_wav[self.block_frame:]
            self.input_wav_res[-self.block_frame_16k - 160:] = self.resampler(
                self.input_wav_denoise[-self.block_frame - 2 * self.zc:]
            )[160:]
        else:
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1):] = (
                self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc:])[
                160:
                ]
            )
        # infer
        if self.function == "vc":
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.gui_config.f0method,
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
        elif self.gui_config.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame:].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame:].clone()
        # output noise reduction
        if self.gui_config.O_noise_reduce and self.function == "vc":
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                                                      self.block_frame:
                                                      ].clone()
            self.output_buffer[-self.block_frame:] = infer_wav[-self.block_frame:]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        # volume envelop mixing
        if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
            if self.gui_config.I_noise_reduce:
                input_wav = self.input_wav_denoise[self.extra_frame:]
            else:
                input_wav = self.input_wav[self.extra_frame:]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(
                rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate)
            )
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[
                     None, None, : self.sola_buffer_frame + self.sola_search_frame
                     ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input ** 2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        # printt("sola_offset = %d", int(sola_offset))
        infer_wav = infer_wav[sola_offset:]
        if "privateuseone" in str(self.config.device) or not self.gui_config.use_pv:
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                    self.sola_buffer * self.fade_out_window
            )
        else:
            infer_wav[: self.sola_buffer_frame] = phase_vocoder(
                self.sola_buffer,
                infer_wav[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        self.sola_buffer[:] = infer_wav[
                              self.block_frame: self.block_frame + self.sola_buffer_frame
                              ]
        res = (
            infer_wav[: self.block_frame]
            .repeat(self.gui_config.channels, 1)
            .t()
            .cpu()
            .numpy()
        )
        total_time = time.perf_counter() - start_time
        print("Infer time: %.2f", total_time)
        return res


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


if __name__ == '__main__':
    import librosa

    pth_file = "assets/weights/wuyusen_manual_clear.pth"
    idx_file = "assets/indices/wuyusen_manual_IVF3201_Flat_nprobe_1_wuyusen_manual_v2.index"
    M = RTRVCModel(pth_file, sr=16000)


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


    file_path = '/root/autodl-fs/audio_samples/董宇辉带货_40k_mono.wav'
    audio_inp = []
    audio_opt = []
    # 调用函数按 250ms 片段读取音频
    sr = 16000
    for chunk in read_audio_in_chunks(file_path, sr, 0.25):
        audio_inp.append(chunk)
        res = M.audio_callback(chunk)[:, 0]
        audio_opt.append(res)

    # Audio(np.hstack(audio_inp), rate=sr)
    # Audio(np.hstack(audio_opt), rate=M.gui_config.samplerate)
