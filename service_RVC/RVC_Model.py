
import numpy as np
from configs import Config
from infer.modules.vc import VC
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import load_hubert
from rvc.synthesizer import load_synthesizer


class RVCModel:
    def __init__(self, pth_file, idx_file="", is_half=False):
        self.config = Config()
        self.config.is_half = is_half

        self.hubert_model = load_hubert(self.config.device, self.config.is_half)
        self.file_index = idx_file

        self.net_g, self.cpt = load_synthesizer(pth_file, self.config.device)
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")
        self.pipeline = Pipeline(self.tgt_sr, self.config)


    """
            sid,
            input_audio_path,
            f0_up_key = vc_transform0,
            f0_file = f0_file,
            f0_method = f0method0,
            file_index = file_index1,
            file_index2 = file_index2,
            index_rate = index_rate1,
            filter_radius = filter_radius0,
            resample_sr = resample_sr0,
            rms_mix_rate = rms_mix_rate0,
            protect=protect0,
    """
    def predict(self,
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
        sid = 0
        audio_max = np.abs(audio_16khz_float32).max() / 0.95
        if audio_max > 1:
            np.divide(audio_16khz_float32, audio_max, audio_16khz_float32)
        times = [0, 0, 0]

        audio_opt = self.pipeline.pipeline(
            self.hubert_model,
            self.net_g,
            sid,
            audio_16khz_float32,
            times,
            f0_up_key,
            f0_method,
            self.file_index,
            index_rate,
            self.if_f0,
            filter_radius,
            self.tgt_sr,
            resample_sr,
            rms_mix_rate,
            self.version,
            protect
        ).astype(np.int16)
        if self.tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        else:
            tgt_sr = self.tgt_sr
        return tgt_sr, audio_opt


if __name__ == '__main__':
    import librosa

    pth_file = "assets/weights/wuyusen_manual_clear.pth"
    idx_file = "assets/indices/wuyusen_manual_IVF3201_Flat_nprobe_1_wuyusen_manual_v2.index"
    M = RVCModel(pth_file)

    audio_16khz_float32, sr = librosa.load("/root/autodl-fs/audio_samples/董宇辉带货.m4a", sr=16000, mono=True)
    sr, audio = M.predict(audio_16khz_float32)
    # Audio(audio, rate=sr)

    audio_16khz_float32, sr = librosa.load("/root/autodl-fs/audio_samples/小Lin说.m4a", sr=16000, mono=True)
    sr, audio = M.predict(audio_16khz_float32, f0_up_key=-6)
    # Audio(audio, rate=sr)
