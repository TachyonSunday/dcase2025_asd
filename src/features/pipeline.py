"""
特征流水线编排模块 —— 将去噪、频谱提取、帧切分串联为端到端的处理流程。
"""

import os
import glob
from typing import Optional, Dict, Any, List

import yaml
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from src.features.denoiser import AudioDenoiser
from src.features.spectrogram import LogMelExtractor


class FeaturePipeline:
    """
    端到端的音频特征提取流水线。

    使用方式::

        pipeline = FeaturePipeline("config.yaml")
        pipeline.process_file("audio.wav", "output.pt")
        # 或批量处理整个目录
        pipeline.process_directory("data/raw/", "data/processed/")

    参数
    ----
    config_path : str
        YAML 配置文件的路径。
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # 初始化各个子模块
        audio_cfg = self.config["audio"]
        mel_cfg = self.config["mel"]

        self.sample_rate: int = audio_cfg["sample_rate"]
        self.target_duration: float = audio_cfg["duration"]
        self.denoise_enabled: bool = audio_cfg.get("denoise_enabled", True)

        # 归一化统计量 (在训练集上计算)
        self.norm_mean: Optional[float] = None
        self.norm_std: Optional[float] = None

        self.denoiser = AudioDenoiser(
            sample_rate=self.sample_rate,
            highpass_cutoff=audio_cfg.get("highpass_cutoff", 80.0),
        )
        self.extractor = LogMelExtractor(
            sample_rate=self.sample_rate,
            n_fft=mel_cfg["n_fft"],
            hop_length=mel_cfg["hop_length"],
            n_mels=mel_cfg["n_mels"],
            f_min=mel_cfg.get("f_min", 50.0),
            f_max=mel_cfg.get("f_max", 8000.0),
            power=mel_cfg.get("power", 2.0),
            top_db=mel_cfg.get("top_db", 80.0),
        )

    def load_audio(self, file_path: str) -> np.ndarray:
        """
        加载并标准化音频文件。

        参数
        ----
        file_path : str
            音频文件路径 (支持 wav/mp3/flac 等 soundfile 兼容格式)。

        返回
        ----
        np.ndarray
            标准化后的浮点音频波形, shape=(samples,), dtype=float32。
        """
        waveform, sr = sf.read(file_path, dtype="float32")
        # 若是多声道, 取平均转为单声道
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1).astype(np.float32)
        # 重采样到目标采样率 (若不一致)
        if sr != self.sample_rate:
            import librosa
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=self.sample_rate
            ).astype(np.float32)
        # 截断或填充到目标时长
        target_samples = int(self.sample_rate * self.target_duration)
        if len(waveform) > target_samples:
            waveform = waveform[:target_samples]
        elif len(waveform) < target_samples:
            waveform = np.pad(waveform, (0, target_samples - len(waveform)))
        return waveform.astype(np.float32)

    def process_file(self, input_path: str, output_path: Optional[str] = None) -> torch.Tensor:
        """
        处理单个音频文件: 加载 → 去噪 → Log-Mel 提取 → 保存/返回张量。

        参数
        ----
        input_path : str
            输入音频文件路径。
        output_path : str, 可选
            输出 .pt 文件路径, 若为 None 则不保存。

        返回
        ----
        torch.Tensor
            Log-Mel 频谱张量, shape=(1, n_mels, T)。
        """
        # 步骤1: 加载并标准化
        waveform = self.load_audio(input_path)
        # 步骤2: 去噪 (高通滤波 + 谱减法)
        waveform = self.denoiser.denoise(waveform, enabled=self.denoise_enabled)
        # 步骤3: Log-Mel 频谱提取
        log_mel_tensor = self.extractor.extract_tensor(waveform)
        # 步骤4: 归一化 (若已计算统计量)
        if self.norm_mean is not None and self.norm_std is not None:
            log_mel_tensor = (log_mel_tensor - self.norm_mean) / (self.norm_std + 1e-8)
        # 步骤5: 保存
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.save(log_mel_tensor, output_path)
        return log_mel_tensor

    def compute_norm_stats(self, data_dir: str, file_pattern: str = "*.wav",
                           max_files: int = 500) -> Tuple[float, float]:
        """
        从训练数据计算 Log-Mel 频谱的全局均值与标准差, 用于 Z-score 归一化。

        返回 (mean, std) 并存储到 self.norm_mean / self.norm_std。
        """
        import glob as _glob
        files = sorted(_glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True))
        if max_files and len(files) > max_files:
            files = files[:max_files]

        all_vals = []
        for fpath in tqdm(files, desc="计算归一化统计量"):
            wav = self.load_audio(fpath)
            wav = self.denoiser.denoise(wav, enabled=self.denoise_enabled)
            mel = self.extractor.extract(wav)
            all_vals.append(mel.flatten())

        if all_vals:
            all_concat = np.concatenate(all_vals)
            self.norm_mean = float(np.mean(all_concat))
            self.norm_std = float(np.std(all_concat))
            print(f"[Pipeline] 归一化统计量 — mean={self.norm_mean:.4f}, std={self.norm_std:.4f}")
        return self.norm_mean, self.norm_std

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.wav",
    ) -> List[str]:
        """
        批量处理目录下所有音频文件。

        参数
        ----
        input_dir : str
            输入音频目录。
        output_dir : str
            输出 .pt 文件保存目录。
        file_pattern : str
            匹配音频文件的 glob 模式, 默认 ``"*.wav"``。

        返回
        ----
        list[str]
            保存的 .pt 文件路径列表。
        """
        os.makedirs(output_dir, exist_ok=True)
        audio_files = sorted(
            glob.glob(os.path.join(input_dir, "**", file_pattern), recursive=True)
        )
        if not audio_files:
            print(f"[FeaturePipeline] 警告: 在 {input_dir} 中未找到匹配 {file_pattern} 的文件")
            return []

        saved_paths: List[str] = []
        for audio_path in tqdm(audio_files, desc="提取特征"):
            # 保持子目录结构
            rel_path = os.path.relpath(audio_path, input_dir)
            out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".pt")
            try:
                self.process_file(audio_path, out_path)
                saved_paths.append(out_path)
            except Exception as e:
                print(f"[FeaturePipeline] 处理失败: {audio_path} — {e}")

        print(f"[FeaturePipeline] 完成: {len(saved_paths)}/{len(audio_files)} 个文件处理成功")
        return saved_paths

    def extract_for_inference(self, file_path: str) -> torch.Tensor:
        """
        为推理准备特征: 等同于 process_file 但不保存文件。

        参数
        ----
        file_path : str
            音频文件路径。

        返回
        ----
        torch.Tensor
            Log-Mel 频谱张量, shape=(1, n_mels, T)。
        """
        return self.process_file(file_path, output_path=None)
