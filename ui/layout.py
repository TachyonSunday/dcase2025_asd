"""
页面布局模块 —— Streamlit 侧边栏配置、三列主区域、状态管理。
"""

from typing import Optional, Dict, Any, Tuple
import os

import streamlit as st
import numpy as np


def configure_page() -> None:
    """配置 Streamlit 全局页面设置。"""
    st.set_page_config(
        page_title="DCASE 2025 Task 2 — 异常声音检测",
        page_icon="🔊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar() -> Dict[str, Any]:
    """
    渲染侧边栏: 文件上传、模型选择、超参数面板。

    返回
    ----
    dict
        包含用户所有输入的配置字典:
        - ``"uploaded_file"``: 上传的文件对象或 None
        - ``"model_type"``: "conv_ae" 或 "dann"
        - ``"checkpoint_path"``: 模型检查点路径
        - ``"threshold"``: 手动阈值或 None
    """
    with st.sidebar:
        st.title("🎵 异常声音检测")
        st.markdown("---")

        # ---- 音频上传 ----
        st.header("📁 上传音频")
        uploaded_file = st.file_uploader(
            "选择音频文件",
            type=["wav", "mp3", "flac", "ogg"],
            help="支持 WAV / MP3 / FLAC / OGG 格式",
        )

        st.markdown("---")

        # ---- 模型选择 ----
        st.header("🧠 模型配置")
        model_type = st.selectbox(
            "模型类型",
            options=["mlp", "conv_ae", "dann"],
            format_func=lambda x: {"mlp": "MLP-AE (官方基线)", "conv_ae": "ConvAE (卷积自编码器)", "dann": "DANN (域对抗网络)"}[x],
            help="选择用于异常检测的模型",
        )

        default_ckpt = {
            "mlp": "results/baseline_v5/ToyCar/checkpoint.pt",
            "conv_ae": "checkpoints/best_model.pt",
            "dann": "checkpoints/best_dann_model.pt",
        }
        checkpoint_path = st.text_input(
            "模型检查点路径",
            value=default_ckpt[model_type],
            help=".pt 检查点文件的路径",
        )

        st.markdown("---")

        # ---- 高级参数 ----
        st.header("⚙️ 检测参数")
        threshold = st.number_input(
            "异常判定阈值",
            value=0.0,
            min_value=0.0,
            step=0.1,
            format="%.4f",
            help="异常分数阈值 (0 = 自动)",
        )
        threshold = threshold if threshold > 0 else None

        show_advanced = st.checkbox("显示高级参数", value=False)
        if show_advanced:
            st.markdown("**频谱参数** (仅展示, 修改请编辑 config.yaml)")
            import yaml
            try:
                with open("config.yaml") as f:
                    cfg = yaml.safe_load(f)
                st.caption(f"n_fft: {cfg['mel']['n_fft']}")
                st.caption(f"n_mels: {cfg['mel']['n_mels']}")
                st.caption(f"hop_length: {cfg['mel']['hop_length']}")
                st.caption(f"window_size: {cfg['frame']['window_size']}")
            except Exception:
                pass

        st.markdown("---")
        st.caption("DCASE 2025 Task 2 — First-Shot ASD")

        return {
            "uploaded_file": uploaded_file,
            "model_type": model_type,
            "checkpoint_path": checkpoint_path,
            "threshold": threshold,
        }


def render_results(
    waveform: np.ndarray,
    waveform_denoised: np.ndarray,
    sample_rate: int,
    log_mel: np.ndarray,
    frame_scores: np.ndarray,
    file_score: float,
    topk_score: float,
    max_score: float,
    is_anomaly: bool,
    threshold: Optional[float],
    recon_error_map: Optional[np.ndarray],
    hop_length: int = 512,
    window_size: int = 64,
) -> None:
    """
    渲染主区域: 三列布局 (波形+频谱+仪表盘) + 底部全宽面板。

    参数
    ----
    (各参数由 InferenceEngine.predict() 返回)
    """
    from ui.components import (
        waveform_plot,
        mel_spectrogram_plot,
        anomaly_gauge,
        frame_scores_plot,
        recon_comparison_plot,
    )

    # ---- 顶部判定横幅 ----
    if is_anomaly:
        st.error(f"⚠️ 检测到异常声音! 均值={file_score:.2f} | 峰值段均值={topk_score:.2f}")
    else:
        st.success(f"✅ 声音正常。均值={file_score:.2f} | 峰值段均值={topk_score:.2f} | 最大值={max_score:.2f}")

    # ---- 第一行: 三列 (波形 / 频谱 / 仪表盘) ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.plotly_chart(
            waveform_plot(waveform_denoised, sample_rate, title="去噪后波形"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            mel_spectrogram_plot(log_mel, sample_rate, hop_length, title="梅尔瀑布图"),
            use_container_width=True,
        )

    with col3:
        if threshold is not None:
            gauge_fig = anomaly_gauge(file_score, threshold, title="异常分数仪表盘")
        else:
            # 阈值未设置时用自适应显示
            auto_thresh = float(np.mean(frame_scores) + 2 * np.std(frame_scores))
            gauge_fig = anomaly_gauge(file_score, auto_thresh, title="异常分数 (自适应阈值)")
        st.plotly_chart(gauge_fig, use_container_width=True)

    # ---- 第二行: 两列 (逐帧分数 / 重建误差) ----
    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        st.plotly_chart(
            frame_scores_plot(
                frame_scores, hop_length, sample_rate,
                threshold=threshold, window_size=window_size,
            ),
            use_container_width=True,
        )

    with col5:
        if recon_error_map is not None:
            st.pyplot(
                recon_comparison_plot(
                    log_mel, recon_error_map, sample_rate, hop_length,
                ),
            )
        else:
            st.info("重建误差热力图不可用 (模型未返回误差数据)")

    # ---- 第三行: 文件信息 ----
    st.markdown("---")
    st.caption(
        f"采样率: {sample_rate} Hz | "
        f"频谱帧数: {log_mel.shape[1]} | "
        f"Mel 频带: {log_mel.shape[0]} | "
        f"异常分数聚合: 逐帧均值 | "
        f"判定结果: {'异常' if is_anomaly else '正常'}"
    )
