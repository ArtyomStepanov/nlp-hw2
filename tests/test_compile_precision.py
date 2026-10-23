"""Reproduce the real HF setter / Inductor getter mismatch without a CUDA GPU."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_compiled_t5_trainer_normalizes_tf32_and_updates_weights(tmp_path):
    # Removing normalization from _TokenLossTrainer must make the legacy getter
    # raise exactly the exception observed on H100, even in this CPU subprocess.
    code = textwrap.dedent('''
        import math, sys
        import torch
        from datasets import Dataset
        from transformers import Seq2SeqTrainingArguments, default_data_collator
        from transformers.utils.import_utils import enable_tf32
        from models import T5Transformer, TransformerConfig
        from training import _TokenLossTrainer, _token_normalized_loss

        torch.set_num_threads(1)
        enable_tf32(True)
        try:
            torch.backends.cuda.matmul.allow_tf32
        except RuntimeError as exc:
            assert 'legacy and new APIs' in str(exc)
        else:
            raise AssertionError('Expected the pinned PyTorch2.9 conflict')

        model = T5Transformer(TransformerConfig(vocab_size=24, d_model=16,
            encoder_layers=1, decoder_layers=1, attention_heads=2, ffn_dim=32,
            max_position_embeddings=32, dropout=0)).network
        data = Dataset.from_dict({'input_ids': [[2,7,3],[2,8,3]],
            'attention_mask': [[1,1,1],[1,1,1]],
            'labels': [[7,3],[8,3]], 'decoder_input_ids': [[2,7],[2,8]]})
        before = model.shared.weight.detach().clone()
        args = Seq2SeqTrainingArguments(output_dir=sys.argv[1], use_cpu=True,
            max_steps=2, per_device_train_batch_size=1, learning_rate=0.001,
            torch_compile=True, torch_compile_backend='eager', report_to='none',
            save_strategy='no', logging_steps=1)
        trainer = _TokenLossTrainer(model=model, args=args, train_dataset=data,
            data_collator=default_data_collator, compute_loss_func=_token_normalized_loss)
        trainer.model_accepts_loss_kwargs=False
        trainer._loss_shifts_labels=False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        result = trainer.train()
        assert trainer.state.global_step == 2
        assert math.isfinite(result.training_loss)
        assert not torch.equal(before, model.shared.weight.detach())
    ''')
    env = os.environ | {'ACCELERATE_USE_CPU': 'true', 'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True,
        text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
