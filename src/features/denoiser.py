"""
音频去噪模块 —— 高通滤波 + 谱减法降噪, 提升异常声音检测的信噪比。
"""

from typing import Optional

import numpy as np
import scipy.signal as signal
import noisereduce as nr


class AudioDenoiser:
    """
    音频降噪器, 组合高通滤波与谱减法。

    参数
    ----
    sample_rate : int
        目标采样率 (Hz)。
    highpass_cutoff : float
        高通滤波截止频率 (Hz), 用于去除低频机械噪声。
    """

    def __init__(self, sample_rate: int = 16000, highpass_cutoff: float = 80.0) -> None:
        self.sample_rate = sample_rate
        self.highpass_cutoff = highpass_cutoff
        # 设计高通滤波器系数 (二阶 Butterworth)
        nyquist = 0.5 * sample_rate
        normal_cutoff = highpass_cutoff / nyquist
        self._b, self._a = signal.butter(2, normal_cutoff, btype="high", analog=False)

    def apply_highpass(self, waveform: np.ndarray) -> np.ndarray:
        """对音频波形施加高通滤波, 滤除低频底噪。"""
        if self.highpass_cutoff <= 0:
            return waveform
        return signal.lfilter(self._b, self._a, waveform).astype(np.float32)

    def apply_spectral_gating(self, waveform: np.ndarray) -> np.ndarray:
        """
        使用谱减法 (spectral gating) 去除稳态背景噪声。
        依赖 noisereduce 库实现非平稳噪声抑制。
        """
        if len(waveform) == 0:
            return waveform
        reduced: np.ndarray = nr.reduce_noise(
            y=waveform,
            sr=self.sample_rate,
            prop_decrease=0.9,  # 噪声衰减比例
            n_fft=1024,
            win_length=1024,
            hop_length=512,
        )
        return reduced.astype(np.float32)

    def denoise(self, waveform: np.ndarray, enabled: bool = True) -> np.ndarray:
        """
        执行完整的去噪流水线: 高通滤波 → 谱减法。

        参数
        ----
        waveform : np.ndarray
            输入的一维音频波形, shape=(samples,)。
        enabled : bool
            是否启用谱减法 (若为 False 则仅做高通滤波)。

        返回
        ----
        np.ndarray
            去噪后的音频波形, shape=(samples,), dtype=float32。
        """
        waveform = np.asarray(waveform, dtype=np.float32)
        # 步骤1: 高通滤波
        waveform = self.apply_highpass(waveform)
        # 步骤2: 谱减法降噪
        if enabled:
            waveform = self.apply_spectral_gating(waveform)
        return waveform
