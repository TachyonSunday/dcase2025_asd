"""
可视化组件模块 —— 封装 Plotly 图表, 为 Streamlit 提供开箱即用的音频分析视图。
"""

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def waveform_plot(
    waveform: np.ndarray,
    sample_rate: int,
    title: str = "音频波形",
    anomaly_segments: Optional[List[tuple]] = None,
) -> go.Figure:
    """
    绘制音频波形图, 可选标注异常片段。

    参数
    ----
    waveform : np.ndarray
        一维音频波形, shape=(samples,)。
    sample_rate : int
        采样率 (Hz)。
    title : str
        图表标题。
    anomaly_segments : list[tuple], 可选
        异常起止时间列表, 格式 ``[(start_sec, end_sec), ...]``。

    返回
    ----
    go.Figure
        Plotly 波形图对象。
    """
    t = np.arange(len(waveform)) / sample_rate
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t,
        y=waveform,
        mode="lines",
        name="波形",
        line=dict(color="#1f77b4", width=0.8),
        hovertemplate="时间: %{x:.3f}s<br>幅度: %{y:.3f}<extra></extra>",
    ))

    # 标注异常片段
    if anomaly_segments:
        for i, (start, end) in enumerate(anomaly_segments):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
                name=f"异常片段 {i+1}",
            )

    fig.update_layout(
        title=title,
        xaxis_title="时间 (秒)",
        yaxis_title="幅度",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )
    return fig


def mel_spectrogram_plot(
    log_mel: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    title: str = "Log-Mel 频谱图",
) -> go.Figure:
    """
    绘制交互式 Log-Mel 频谱热力图 (梅尔瀑布图)。

    参数
    ----
    log_mel : np.ndarray
        Log-Mel 频谱图, shape=(n_mels, T)。
    sample_rate : int
        采样率 (Hz)。
    hop_length : int
        帧移 (用于时间轴换算)。
    title : str
        图表标题。

    返回
    ----
    go.Figure
        Plotly 热力图对象。
    """
    n_mels, n_frames = log_mel.shape
    # 时间轴 (秒)
    times = np.arange(n_frames) * hop_length / sample_rate
    # 频率轴 (Mel 编号)
    mel_bins = np.arange(n_mels)

    fig = go.Figure(data=go.Heatmap(
        z=log_mel,
        x=times,
        y=mel_bins,
        colorscale="Viridis",
        zmin=-80,
        zmax=0,
        colorbar=dict(title="dB"),
        hovertemplate="时间: %{x:.3f}s<br>Mel: %{y}<br>dB: %{z:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="时间 (秒)",
        yaxis_title="Mel 频带",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def anomaly_gauge(
    score: float,
    threshold: float,
    title: str = "异常分数",
) -> go.Figure:
    """
    绘制异常分数仪表盘 (Gauge Chart)。

    参数
    ----
    score : float
        当前文件的异常分数。
    threshold : float
        判定阈值。
    title : str
        图表标题。

    返回
    ----
    go.Figure
        Plotly 仪表盘对象。
    """
    # 归一化分数到 [0, 1] 区间 (以阈值为参考)
    max_display = max(threshold * 2, score * 1.2)
    normalized = min(score / max_display * 100, 100)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": title},
        delta={"reference": threshold, "increasing": {"color": "red"}},
        number={"suffix": " MSE", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, max_display], "tickwidth": 1},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, threshold * 0.8], "color": "lightgreen"},
                {"range": [threshold * 0.8, threshold], "color": "yellow"},
                {"range": [threshold, max_display], "color": "lightcoral"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": threshold,
            },
        },
    ))

    fig.update_layout(
        template="plotly_white",
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


def frame_scores_plot(
    frame_scores: np.ndarray,
    hop_length: int,
    sample_rate: int,
    threshold: Optional[float] = None,
    window_size: int = 64,
    title: str = "逐帧异常分数",
) -> go.Figure:
    """
    绘制逐帧异常分数折线图, 标记超过阈值的异常区域。

    参数
    ----
    frame_scores : np.ndarray
        逐帧 MSE 分数, shape=(N,)。
    hop_length : int
        帧移。
    sample_rate : int
        采样率。
    threshold : float, 可选
        判定阈值 (虚线)。
    window_size : int
        窗口大小 (用于时间对齐)。
    title : str
        图表标题。

    返回
    ----
    go.Figure
        Plotly 折线图对象。
    """
    # 每帧对应的时间 (窗口中心)
    times = np.arange(len(frame_scores)) * hop_length / sample_rate + \
            (window_size * hop_length) / (2 * sample_rate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=frame_scores,
        mode="lines",
        name="异常分数",
        line=dict(color="#d62728", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(214, 39, 40, 0.1)",
        hovertemplate="时间: %{x:.3f}s<br>分数: %{y:.4f}<extra></extra>",
    ))

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"阈值 = {threshold:.4f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=title,
        xaxis_title="时间 (秒)",
        yaxis_title="MSE 异常分数",
        template="plotly_white",
        height=250,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )
    return fig


def recon_comparison_plot(
    log_mel: np.ndarray,
    recon_error: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    title: str = "重建误差热力图",
) -> go.Figure:
    """
    绘制重建误差热力图, 展示频谱哪些区域未被模型良好重建。

    参数
    ----
    log_mel : np.ndarray
        原始 Log-Mel 频谱图, shape=(n_mels, T)。
    recon_error : np.ndarray
        逐元素重建平方误差, shape 应与 log_mel 相容 (或按时间裁剪)。
    sample_rate : int
        采样率。
    hop_length : int
        帧移。
    title : str
        图表标题。

    返回
    ----
    go.Figure
        双面板对比图 (原频谱 + 重建误差)。
    """
    n_mels, n_frames = log_mel.shape
    times = np.arange(n_frames) * hop_length / sample_rate
    mel_bins = np.arange(n_mels)

    has_error = recon_error is not None and recon_error.size > 0 and np.max(np.abs(recon_error)) > 1e-8

    if has_error:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("原始 Log-Mel 频谱", "重建误差 (MSE)"),
            vertical_spacing=0.12,
        )
        # 上: 原始频谱
        fig.add_trace(
            go.Heatmap(z=log_mel, x=times, y=mel_bins,
                       colorscale="Viridis", zmin=-80, zmax=0,
                       colorbar=dict(title="dB", x=0.46, len=0.4, y=0.82)),
            row=1, col=1,
        )
        # 下: 重建误差
        err_frames = min(recon_error.shape[-1], n_frames)
        fig.add_trace(
            go.Heatmap(z=recon_error[:, :err_frames], x=times[:err_frames],
                       y=np.arange(recon_error.shape[0]),
                       colorscale="Reds",
                       colorbar=dict(title="MSE", x=0.46, len=0.4, y=0.32)),
            row=2, col=1,
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=log_mel, x=times, y=mel_bins,
            colorscale="Viridis", zmin=-80, zmax=0,
            colorbar=dict(title="dB"),
        ))

    fig.update_xaxes(title_text="时间 (秒)", row=2, col=1)
    fig.update_yaxes(title_text="Mel 频带", row=1, col=1)
    fig.update_yaxes(title_text="Mel 频带", row=2, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=550,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
