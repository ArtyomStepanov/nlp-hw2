"""Общий интерфейс моделей: обучение, генерация и сохранение весов."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import Seq2SeqLMOutput

from .config import TransformerConfig


class BaseSeq2SeqModel(nn.Module, ABC):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        """Вернуть logits и, если переданы labels, loss."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        max_new_tokens: int = 256,
        num_beams: int = 1,
    ) -> Tensor:
        """Вернуть сгенерированные ID токенов для всего батча."""
        raise NotImplementedError

    def prepare_labels(self, target_ids: Tensor) -> Tensor:
        """Из [BOS, код, EOS, PAD] получить labels без BOS и с PAD=-100.

        Ожидается двумерный батч результатов нашего токенизатора с BOS в начале.
        Сдвиг decoder inputs вправо выполняет сама модель при forward(labels=...).
        """
        if target_ids.ndim != 2 or target_ids.shape[1] < 2:
            raise ValueError("target_ids должен иметь форму [batch, length >= 2].")
        labels = target_ids[:, 1:].clone()
        return labels.masked_fill(labels == self.config.pad_token_id, -100)

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Сохранить конфигурацию и веса модели; состояние оптимизатора не входит."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_class": type(self).__name__,
                "config": self.config.to_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "BaseSeq2SeqModel":
        """Восстановить конкретный класс модели из своего checkpoint."""
        checkpoint = torch.load(Path(path).expanduser(), map_location=map_location, weights_only=True)
        if checkpoint["model_class"] != cls.__name__:
            raise ValueError(f"Checkpoint принадлежит модели {checkpoint['model_class']}, а не {cls.__name__}.")
        model = cls(TransformerConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(map_location)
