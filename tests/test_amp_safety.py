"""Exercise the runner's AMP guard with real scalar CPU tensors, without training."""
import json
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from transformers import TrainerControl, TrainerState

import kaggle_experiments as k


def scalar_run(tmp_path, *, fused=False):
    model = torch.nn.Module()
    model.register_parameter('weight', torch.nn.Parameter(torch.tensor(1.)))
    base = (torch.optim.AdamW(model.parameters(), lr=.1, fused=True)
        if fused else torch.optim.SGD(model.parameters(), lr=.1))
    scaler = torch.amp.GradScaler('cpu')
    Accelerator(cpu=True)
    optimizer = AcceleratedOptimizer(base, device_placement=False, scaler=scaler)
    optimizer.gradient_state._set_sync_gradients(True)
    callback = k._safety_and_artifacts_callback(tmp_path, {}, lambda: scaler)
    state = TrainerState(global_step=0, is_world_process_zero=True)
    return model, base, optimizer, scaler, callback, state


def overflow_backward(model, base, scaler):
    # Loss is finite, but its FP16 backward cast cannot represent scale=65536.
    loss = model.weight.to(torch.float16).float()
    assert loss.item() == 1.
    scaler.scale(loss).backward()
    scaler.unscale_(base)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).item()


@pytest.mark.parametrize('fused', [False, True], ids=['sgd', 'fused-adamw'])
def test_detected_amp_overflow_reaches_native_skip_and_backoff(tmp_path, fused):
    model, base, optimizer, scaler, callback, state = scalar_run(tmp_path, fused=fused)
    norm = overflow_backward(model, base, scaler)
    assert norm == float('inf')
    assert torch.isnan(model.weight.grad)

    # The old unconditional guard raises here, blocking normal scaler recovery.
    callback.on_pre_optimizer_step(None, state, TrainerControl(), model=model, optimizer=optimizer)
    optimizer.step()
    assert model.weight.item() == 1.
    assert scaler.get_scale() == 32768.
    if fused:
        assert base.state[model.weight]['step'].item() == 0.
        # Fused kernels skip internally; Accelerate's method-call flag misses it.
        assert optimizer.step_was_skipped is False
    else:
        assert optimizer.step_was_skipped is True

    optimizer.zero_grad()
    state.global_step = 1
    loss = model.weight.to(torch.float16).float()
    scaler.scale(loss).backward()
    scaler.unscale_(base)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    callback.on_pre_optimizer_step(None, state, TrainerControl(), model=model, optimizer=optimizer)
    optimizer.step()
    assert model.weight.item() < 1.
    assert torch.isfinite(model.weight)
    assert scaler.get_scale() == 32768.


def log_like_trainer(callback, state, logs):
    # Trainer.log copies the row before invoking on_log, so both copies matter.
    state.log_history.append({**logs, 'step': state.global_step})
    callback.on_log(None, state, TrainerControl(), logs=logs)


def test_known_overflow_grad_norm_is_null_in_live_and_saved_history(tmp_path):
    model, base, optimizer, scaler, callback, state = scalar_run(tmp_path, fused=True)
    norm = overflow_backward(model, base, scaler)
    callback.on_pre_optimizer_step(None, state, TrainerControl(), model=model, optimizer=optimizer)
    optimizer.step()
    state.global_step = 1
    logs = {'loss': 1., 'grad_norm': norm, 'learning_rate': .1}
    log_like_trainer(callback, state, logs)
    assert logs['grad_norm'] is None
    assert state.log_history[-1]['grad_norm'] is None
    assert state.log_history[-1]['amp_overflow_count'] == 1
    assert state.log_history[-1]['amp_loss_scale'] == 32768.
    saved = json.loads((tmp_path/'log_history.json').read_text())
    assert saved == state.log_history
    json.dumps(saved, allow_nan=False)
    assert callback.amp_diagnostics() == {'overflow_count': 1, 'final_loss_scale': 32768.,
        'scope': 'current_process_session'}


@pytest.mark.parametrize('metric', ['loss', 'learning_rate', 'eval_loss', 'eval_token_loss'])
def test_known_overflow_never_exempts_nonfinite_loss_lr_or_eval(tmp_path, metric):
    model, base, optimizer, scaler, callback, state = scalar_run(tmp_path)
    overflow_backward(model, base, scaler)
    callback.on_pre_optimizer_step(None, state, TrainerControl(), model=model, optimizer=optimizer)
    optimizer.step()
    state.global_step = 1
    with pytest.raises(FloatingPointError, match='Nonfinite metric'):
        log_like_trainer(callback, state, {metric: float('nan')})
    assert not (tmp_path/'log_history.json').exists()


def test_unrelated_later_grad_norm_is_still_fatal(tmp_path):
    model, base, optimizer, scaler, callback, state = scalar_run(tmp_path)
    overflow_backward(model, base, scaler)
    callback.on_pre_optimizer_step(None, state, TrainerControl(), model=model, optimizer=optimizer)
    optimizer.step()
    state.global_step = 2
    with pytest.raises(FloatingPointError, match='grad_norm'):
        log_like_trainer(callback, state, {'grad_norm': float('inf')})


def test_grad_norm_without_completed_scaler_backoff_is_fatal(tmp_path):
    model, base, optimizer, scaler, callback, state = scalar_run(tmp_path)
    overflow_backward(model, base, scaler)
    callback.on_pre_optimizer_step(None, state, TrainerControl(), model=model, optimizer=optimizer)
    state.global_step = 1
    with pytest.raises(FloatingPointError, match='grad_norm'):
        log_like_trainer(callback, state, {'grad_norm': float('inf')})


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16], ids=['fp32', 'bf16'])
@pytest.mark.parametrize('scaling', ['none', 'disabled', 'unrecorded'])
def test_nonfinite_gradients_without_detected_scaler_overflow_stay_fatal(tmp_path, dtype, scaling):
    model = torch.nn.Module()
    model.register_parameter('weight', torch.nn.Parameter(torch.tensor(1., dtype=dtype)))
    model.weight.grad = torch.full_like(model.weight, float('nan'))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = None if scaling == 'none' else torch.amp.GradScaler('cpu', enabled=scaling != 'disabled')
    callback = k._safety_and_artifacts_callback(tmp_path, {}, lambda: scaler)
    with pytest.raises(FloatingPointError, match='Nonfinite gradients'):
        callback.on_pre_optimizer_step(None, TrainerState(), TrainerControl(), model=model, optimizer=optimizer)
    assert model.weight.item() == 1.


def test_another_optimizer_overflow_does_not_exempt_bad_gradients(tmp_path):
    model, base, optimizer, scaler, callback, state = scalar_run(tmp_path)
    overflow_backward(model, base, scaler)
    other = torch.optim.SGD(model.parameters(), lr=.1)
    with pytest.raises(FloatingPointError, match='Nonfinite gradients'):
        callback.on_pre_optimizer_step(None, state, TrainerControl(), model=model, optimizer=other)


def test_missing_gradients_stay_fatal_with_or_without_scaler_overflow(tmp_path):
    model, base, optimizer, scaler, callback, state = scalar_run(tmp_path)
    with pytest.raises(FloatingPointError, match='gradients'):
        callback.on_pre_optimizer_step(None, state, TrainerControl(), model=model, optimizer=optimizer)
    overflow_backward(model, base, scaler)
    model.weight.grad = None
    with pytest.raises(FloatingPointError, match='gradients'):
        callback.on_pre_optimizer_step(None, state, TrainerControl(), model=model, optimizer=optimizer)


def test_real_nan_forward_loss_still_stops_before_backward():
    logits = torch.full((1, 1, 2), float('nan'), requires_grad=True)
    with pytest.raises(FloatingPointError, match='Nonfinite training loss'):
        k.checked_token_loss(SimpleNamespace(logits=logits), torch.tensor([[1]]))
    assert logits.grad is None
