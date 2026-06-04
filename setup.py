"""
DCASE 2025 Task 2 — 无监督异常声音检测系统
项目安装脚本
"""
from setuptools import setup, find_packages

setup(
    name="dcase2025_asd",
    version="0.1.0",
    description="First-Shot Unsupervised Anomalous Sound Detection (DCASE 2025 Task 2)",
    author="ASD Engineer",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "streamlit>=1.30.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "noisereduce>=3.0.0",
    ],
)
