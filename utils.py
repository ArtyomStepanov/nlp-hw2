"""Вспомогательные функции для ноутбуков и обучения."""

import torch


def get_device() -> torch.device:
    """Выбрать доступное устройство в порядке CUDA, MPS, CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_compile_precision() -> None:
    """Work around the pinned torch2.9 / HF5.10 TF32 API mismatch.

    HF sets the new global precision override, while Inductor and Accelerate
    still read/write allow_tf32. Clear new overrides before using that API.
    Call after Trainer/Accelerator initialization, before the first compile.
    This changes FP32 matmul settings, not BF16/FP16 autocast or parameter dtype.
    """
    if torch.__version__.split('+')[0].split('.')[:2] != ['2', '9']:
        return
    torch.backends.fp32_precision = 'none'
    torch.backends.cuda.matmul.fp32_precision = 'none'
    torch.backends.cudnn.fp32_precision = 'none'
    torch.backends.cudnn.conv.fp32_precision = 'none'
    torch.backends.cudnn.rnn.fp32_precision = 'none'
    enable = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)
    torch.backends.cuda.matmul.allow_tf32 = enable
    torch.backends.cudnn.allow_tf32 = enable
