"""
DCASE 2025 Task 2 — 无监督异常声音检测 Web 前端 (Streamlit)

启动方式::

    streamlit run ui/app.py
"""

import os
import sys
import tempfile
import traceback

import streamlit as st
import numpy as np

# 将项目根目录加入 Python 路径 (确保 src 可导入)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.layout import configure_page, render_sidebar, render_results
from ui.inference import InferenceEngine


def main() -> None:
    """Streamlit 应用主入口。"""
    configure_page()

    # ---- 页面标题 ----
    st.title("🔊 DCASE 2025 Task 2 — 无监督异常声音检测")
    st.markdown(
        """
        上传机器运行音频 (WAV / MP3 / FLAC)，使用卷积自编码器 (ConvAE)
        或域对抗网络 (DANN) 实时检测异常声音。
        """
    )
    st.markdown("---")

    # ---- 渲染侧边栏并获取用户配置 ----
    config = render_sidebar()

    # ---- 初始化 Session State (缓存引擎, 避免重复加载) ----
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "engine_model_type" not in st.session_state:
        st.session_state.engine_model_type = None
    if "engine_checkpoint" not in st.session_state:
        st.session_state.engine_checkpoint = None

    uploaded_file = config["uploaded_file"]

    # ---- 无文件时显示引导提示 ----
    if uploaded_file is None:
        st.info("👈 请从左侧边栏上传音频文件开始分析。")
        _render_demo_info()
        return

    # ---- 加载/切换模型 ----
    model_type = config["model_type"]
    checkpoint_path = config["checkpoint_path"]

    if not os.path.exists(checkpoint_path):
        st.warning(
            f"⚠️ 检查点文件不存在: `{checkpoint_path}`\n\n"
            "请先训练模型 (参考 README.md) 或修改侧边栏中的检查点路径。\n\n"
            "当前将使用**未训练的随机权重模型**进行演示。"
        )

    engine_needs_reload = (
        st.session_state.engine is None
        or st.session_state.engine_model_type != model_type
        or st.session_state.engine_checkpoint != checkpoint_path
    )

    if engine_needs_reload:
        with st.spinner(f"加载 {model_type.upper()} 模型..."):
            try:
                engine = InferenceEngine(
                    config_path=os.path.join(PROJECT_ROOT, "config.yaml")
                )
                if os.path.exists(checkpoint_path):
                    engine.load_model(
                        checkpoint_path,
                        model_type=model_type,
                        threshold=config["threshold"],
                    )
                else:
                    # 即使没有检查点, 也可以加载随机权重模型演示界面
                    engine.model_type = model_type
                    import torch, yaml
                    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
                        cfg = yaml.safe_load(f)
                    if model_type == "mlp":
                        from ui.inference import MLPAE
                        engine.model = MLPAE().to(engine.device)
                    elif model_type == "conv_ae":
                        from src.models.conv_ae import ConvAE
                        engine.model = ConvAE.from_config(os.path.join(PROJECT_ROOT, "config.yaml"))
                        engine.model.bind(torch.randn(1, 1, cfg["mel"]["n_mels"], cfg["frame"]["window_size"]))
                        engine.model.to(engine.device)
                    else:
                        from src.models.conv_ae import ConvAE
                        from src.models.dann import DANNAutoEncoder
                        conv_ae = ConvAE.from_config(os.path.join(PROJECT_ROOT, "config.yaml"))
                        engine.model = DANNAutoEncoder(conv_ae=conv_ae, num_domains=cfg["dann"]["num_domains"])
                        engine.model.bind(torch.randn(1, 1, cfg["mel"]["n_mels"], cfg["frame"]["window_size"]))
                        engine.model.to(engine.device)
                    engine.threshold = config["threshold"]

                st.session_state.engine = engine
                st.session_state.engine_model_type = model_type
                st.session_state.engine_checkpoint = checkpoint_path
                st.success(f"✅ {model_type.upper()} 模型已加载")
            except Exception as e:
                st.error(f"模型加载失败: {e}")
                st.code(traceback.format_exc())
                return

    # ---- 执行推理 ----
    engine = st.session_state.engine

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        with st.spinner("正在分析音频..."):
            result = engine.predict(tmp_path)

        # 渲染结果
        render_results(
            waveform=result["waveform"],
            waveform_denoised=result["waveform_denoised"],
            sample_rate=result["sample_rate"],
            log_mel=result["log_mel"],
            frame_scores=result["frame_scores"],
            file_score=result["file_score"],
            is_anomaly=result["is_anomaly"],
            threshold=engine.threshold,
            recon_error_map=result["recon_error_map"],
            hop_length=engine.pipeline.extractor.hop_length,
            window_size=engine.window_size,
        )

    except Exception as e:
        st.error(f"推理失败: {e}")
        st.code(traceback.format_exc())
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _render_demo_info() -> None:
    """在无文件时展示系统能力说明。"""
    st.markdown("### 🧪 系统能力")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**音频预处理**")
        st.markdown("- 高通滤波 (去除低频噪声)")
        st.markdown("- 谱减法降噪")
        st.markdown("- 自动重采样到 16kHz")
    with col_b:
        st.markdown("**特征提取**")
        st.markdown("- Log-Mel 频谱图 (128 频带)")
        st.markdown("- 滑动窗口帧切分")
        st.markdown("- 实时瀑布图可视化")
    with col_c:
        st.markdown("**异常检测**")
        st.markdown("- ConvAE 重构误差")
        st.markdown("- DANN 域泛化")
        st.markdown("- 逐帧 + 文件级分数")


if __name__ == "__main__":
    main()
