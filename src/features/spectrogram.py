"""
Log-Mel 频谱提取模块 —— STFT → Mel 滤波器组 → 对数压缩, 输出可用于神经网络的二维特征。
"""

from typing import Optional, Tuple

import numpy as np
import librosa
import torch


class LogMelExtractor:
    """
    Log-Mel 频谱图提取器, 将原始音频波形转换为对数尺度的梅尔频谱图。

    参数
    ----
    sample_rate : int
        目标采样率 (Hz)。
    n_fft : int
        FFT 窗口大小。
    hop_length : int
        帧移 (控制时间轴分辨率)。
    n_mels : int
        Mel 滤波器组中滤波器的数量 (频谱图高度)。
    f_min : float
        最低分析频率 (Hz)。
    f_max : float
        最高分析频率 (Hz)。
    power : float
        功率谱指数 (1.0=幅度谱, 2.0=功率谱)。
    top_db : float
        动态范围上限 (dB), 低于 ``max_db - top_db`` 的值将被截断。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        power: float = 2.0,
        top_db: float = 80.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.top_db = top_db

        # 预创建 Mel 滤波器组矩阵 (复用, 避免重复分配)
        self._mel_basis: Optional[np.ndarray] = None

    def _get_mel_basis(self) -> np.ndarray:
        """懒加载 Mel 滤波器组矩阵。"""
        if self._mel_basis is None:
            self._mel_basis = librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
            )
        return self._mel_basis

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """
        从音频波形提取 Log-Mel 频谱图。

        参数
        ----
        waveform : np.ndarray
            输入的一维音频波形, shape=(samples,), dtype=float32。

        返回
        ----
        np.ndarray
            Log-Mel 频谱图, shape=(n_mels, T), 其中 T 为时间帧数。
        """
        waveform = np.asarray(waveform, dtype=np.float32)
        # STFT → 功率谱 (线性刻度)
        stft: np.ndarray = librosa.stft(
            y=waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window="hann",
            center=True,
        )
        magnitude = np.abs(stft) ** self.power

        # 线性功率谱 → Mel 刻度
        mel_basis = self._get_mel_basis()
        mel_spec: np.ndarray = np.dot(mel_basis, magnitude)

        # 对数压缩 (dB 尺度)
        log_mel: np.ndarray = librosa.power_to_db(
            mel_spec, ref=np.max, top_db=self.top_db
        )
        return log_mel.astype(np.float32)

    def extract_tensor(self, waveform: np.ndarray) -> torch.Tensor:
        """
        提取 Log-Mel 频谱图并以 PyTorch 张量返回。

        参数
        ----
        waveform : np.ndarray
            输入的一维音频波形。

        返回
        ----
        torch.Tensor
            Log-Mel 频谱图, shape=(1, n_mels, T), 已添加通道维度。
        """
        log_mel = self.extract(waveform)
        tensor = torch.from_numpy(log_mel).unsqueeze(0)  # (1, n_mels, T)
        return tensor

    def time_to_frames(self, duration_seconds: float) -> int:
        """将时间 (秒) 转换为对应的梅尔频谱帧数。"""
        # 使用 librosa 的 time_to_frames 进行精确转换
        return librosa.time_to_frames(
            duration_seconds, sr=self.sample_rate, hop_length=self.hop_length
        )
