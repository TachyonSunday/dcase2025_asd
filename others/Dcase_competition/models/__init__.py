"""
models/__init__.py
模型模块初始化文件
"""

from .feature_extractor import BEATsFeatureExtractor, get_feature_extractor

__all__ = ["BEATsFeatureExtractor", "get_feature_extractor"]
