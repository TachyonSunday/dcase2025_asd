"""
梯度反转层 (Gradient Reversal Layer, GRL) —— DANN 的核心机制。
在前向传播中作为恒等映射, 在反向传播中将梯度乘以 -λ,
迫使特征提取器学习域不变 (domain-invariant) 的表示。

参考: Ganin & Lempitsky (2015), "Unsupervised Domain Adaptation by Backpropagation"
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    自定义 autograd 函数: 前向恒等, 反向梯度取反并缩放。

    静态方法
    --------
    forward(ctx, x, lambda_)
        保存 λ 并原样返回输入。
    backward(ctx, grad_output)
        返回 -λ * grad_output。
    """

    @staticmethod
    def forward(ctx: "Function", x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx: "Function", grad_output: torch.Tensor):
        lambda_ = ctx.lambda_
        # 用 grad_output.new_tensor 保证设备兼容
        lambda_tensor = grad_output.new_tensor(lambda_)
        grad_input = -lambda_tensor * grad_output
        return grad_input, None  # None 对应 lambda_ 参数 (不需要梯度)


class GradientReversalLayer(nn.Module):
    """
    梯度反转层 (GRL) — 作为 nn.Module 使用, 支持动态调整 λ。

    参数
    ----
    lambda_init : float
        初始梯度缩放系数, 默认 0.0 (无对抗)。
    """

    def __init__(self, lambda_init: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("lambda_", torch.tensor(lambda_init, dtype=torch.float32))

    def set_lambda(self, value: float) -> None:
        """动态设置 λ 值。"""
        self.lambda_.fill_(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 — 恒等映射, 反向时梯度反转。

        参数
        ----
        x : torch.Tensor
            输入张量 (通常是隐向量 z)。

        返回
        ----
        torch.Tensor
            与输入相同的张量 (但梯度路径已反转)。
        """
        return GradientReversalFunction.apply(x, float(self.lambda_.item()))
