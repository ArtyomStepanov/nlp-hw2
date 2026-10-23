import json
from pathlib import Path

import numpy as np
import pytest

from train_sweep import build_trials, prepare_data, select_indices, trial_action


def test_grid_is_four_independent_equal_budget_trials(tmp_path):
    trials = build_trials(tmp_path, micro_batch=128)
    assert {(t['encoder_layers'], t['decoder_layers'], t['learning_rate']) for t in trials} == {
        (10, 10, 1e-4), (10, 10, 2e-4), (12, 8, 1e-4), (12, 8, 2e-4)}
    assert len({t['output_dir'] for t in trials}) == 4
    assert all(t['train_batch_size'] * t['gradient_accumulation_steps'] == 256 for t in trials)
    assert all(t['num_train_epochs'] == 1 and t['seed'] == 42 for t in trials)
    larger = build_trials(tmp_path, micro_batch=256)
    assert all(t['gradient_accumulation_steps'] == 1 for t in larger)
    with pytest.raises(ValueError):
        build_trials(tmp_path, micro_batch=192)


def test_fixed_subset_is_reproducible_without_replacement():
    a = select_indices(1000, 100, 42)
    np.testing.assert_array_equal(a, select_indices(1000, 100, 42))
    assert not np.array_equal(a, select_indices(1000, 100, 43))
    assert len(set(a)) == 100 and a.min() >= 0 and a.max() < 1000
    for count in (0, -1, 1001):
        with pytest.raises(ValueError):
            select_indices(1000, count, 42)


def test_guard_never_overwrites_another_experiment(tmp_path):
    trial = build_trials(tmp_path, micro_batch=128)[0]
    output = Path(trial['output_dir'])
    assert trial_action(output, trial) == 'new'
    output.mkdir(parents=True)
    (output/'unknown.txt').write_text('unrelated result')
    with pytest.raises(ValueError):
        trial_action(output, trial)
    (output/'trial.json').write_text(json.dumps(trial))
    with pytest.raises(ValueError):
        trial_action(output, trial)  # interrupted before any checkpoint
    (output/'complete.json').write_text('{}')
    assert trial_action(output, trial) == 'skip'
    with pytest.raises(ValueError):
        trial_action(output, trial | {'learning_rate': 9e-4})


def test_custom_trials_keep_requested_architectures_and_distinct_learning_rates(tmp_path):
    specs = [
        {'encoder_layers': 8, 'decoder_layers': 12, 'learning_rate': 1.1e-4},
        {'encoder_layers': 8, 'decoder_layers': 12, 'learning_rate': 1.2e-4},
        {'encoder_layers': 8, 'decoder_layers': 8, 'learning_rate': 2e-4},
    ]
    trials = build_trials(tmp_path, specs=specs)
    assert [(t['encoder_layers'], t['decoder_layers'], t['learning_rate']) for t in trials] == [
        (8, 12, 1.1e-4), (8, 12, 1.2e-4), (8, 8, 2e-4)]
    assert len({t['output_dir'] for t in trials}) == 3
    assert all(t['train_batch_size'] * t['gradient_accumulation_steps'] == 256 for t in trials)
    assert all(t['seed'] == 42 and t['num_train_epochs'] == 1 for t in trials)


def test_invalid_or_duplicate_trial_specs_are_rejected(tmp_path):
    valid = {'encoder_layers': 8, 'decoder_layers': 12, 'learning_rate': 2e-4}
    bad = [[], [valid, valid], [valid | {'encoder_layers': 0}],
           [valid | {'decoder_layers': True}], [valid | {'learning_rate': float('nan')}],
           [valid | {'learning_rate': -1}], [valid | {'d_model': 768}]]
    for specs in bad:
        with pytest.raises(ValueError):
            build_trials(tmp_path, specs=specs)


def test_extension_reuses_identical_prepared_data_without_writing_new_subset(tmp_path):
    import datasets as ds

    source = tmp_path / 'source'
    rows = ds.Dataset.from_dict({'input_ids': [[i] for i in range(12)],
        'attention_mask': [[1]] * 12, 'labels': [[2]] * 12, 'length': [1] * 12})
    ds.DatasetDict(train=rows, val=rows).save_to_disk(str(source))
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('{}')
    original = tmp_path / 'original'
    config = dict(output_dir=str(original), tokenized_dataset=str(source),
                  tokenizer=str(tokenizer), train_rows=4, validation_rows=2, seed=42)
    manifest = prepare_data(config)
    extension = tmp_path / 'extension'
    reused = config | {'output_dir': str(extension), 'prepared_data_dir': str(original)}
    assert prepare_data(reused) == manifest
    assert not (extension / 'data').exists()
    assert not (extension / 'data_manifest.json').exists()
    with pytest.raises(ValueError):
        prepare_data(reused | {'seed': 43})
    with pytest.raises(ValueError):
        prepare_data(reused | {'prepared_data_dir': str(tmp_path / 'missing')})


def test_t5_trials_are_distinct_and_keep_equal_budget(tmp_path):
    spec = dict(encoder_layers=8, decoder_layers=8, learning_rate=2e-4)
    trials = build_trials(tmp_path, specs=[spec, spec | {'architecture': 't5'}])
    assert trials[0]['name'] == 'e8_d8_lr2e-04'
    assert trials[1]['name'] == 't5_e8_d8_lr2e-04'
    assert trials[1]['architecture'] == 't5'
    assert trials[1]['d_model'] == 1024 and trials[1]['ffn_dim'] == 2816
    assert trials[1]['train_batch_size'] * trials[1]['gradient_accumulation_steps'] == 256
    with pytest.raises(ValueError):
        build_trials(tmp_path, specs=[spec | {'architecture': 'unknown'}])


def test_sweep_factory_uses_t5_and_actual_tokenizer_ids(tmp_path):
    import train_sweep
    from tokenizers import Tokenizer, models as tm
    assert hasattr(train_sweep, 'build_trial_model'), 'Architecture factory missing'
    tok = Tokenizer(tm.WordLevel({'word': 0, '<unk>': 1, '<eos>': 2, '<pad>': 3, '<bos>': 4}))
    spec = dict(architecture='t5', encoder_layers=8, decoder_layers=8, learning_rate=1e-4)
    trial = build_trials(tmp_path, specs=[spec])[0]
    model = train_sweep.build_trial_model(trial, tok, smoke=True)
    assert model.network.config.model_type == 't5'
    assert model.network.config.vocab_size == 5
    assert model.network.config.decoder_start_token_id == 4
    assert model.network.config.pad_token_id == 3
    assert model.network.config.eos_token_id == 2
