"""Вспомогательные функции для ноутбуков и обучения."""

import torch


def get_device() -> torch.device:
    """Выбрать доступное устройство в порядке CUDA, MPS, CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
