"""Stage2 gates and restart safety, without launching training or a GPU job."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


GOLDEN = Path(__file__).resolve().parents[1]/'reports/a100_sequential_20261022/reference_identity.json'


def api():
    assert importlib.util.find_spec('tools.run_a100_stage2'), 'Stage2 runner is missing'
    from tools import run_a100_stage2
    return run_a100_stage2


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage1_fixture(root, *, copy_score=.255, winner='bart'):
    references = json.loads(GOLDEN.read_text())
    models = {}
    for architecture, score in (('bart', .25), ('copy', copy_score)):
        metrics = dict(architecture=architecture, **{'pass@1': score},
            num_evaluated=19886, num_beams=4, max_new_tokens=256,
            length_penalty=1., validation_sha256=references['validation_sha256'])
        models[architecture] = metrics
        destination = root/'evaluations'/architecture
        put(destination/'metrics.json', metrics)
        (destination/'evaluation').mkdir()
        (destination/'evaluation/evaluation.jsonl').write_text(''.join(
            json.dumps(row)+'\n' for row in references['rows']))
    put(root/'stage1_complete.json', dict(winner=winner, models=models))
    return root


@pytest.mark.parametrize(('score', 'winner'), [(.25499, 'bart'), (.255, 'bart'), (.25501, 'copy')])
def test_stage1_winner_rechecks_saved_results_and_strict_tie_rule(tmp_path, score, winner):
    runner = api()
    source = stage1_fixture(tmp_path/'stage1', copy_score=score, winner=winner)
    assert runner.verify_stage1(source, GOLDEN)['winner'] == winner


@pytest.mark.parametrize('change', ['winner', 'reference', 'validation', 'eligible', 'nan', 'summary'])
def test_stage1_rejects_wrong_winner_or_invalid_evaluation(tmp_path, change):
    runner = api()
    source = stage1_fixture(tmp_path/'stage1')
    marker = json.loads((source/'stage1_complete.json').read_text())
    metrics_path = source/'evaluations/bart/metrics.json'
    metrics = json.loads(metrics_path.read_text())
    if change == 'winner':
        marker['winner'] = 'copy'
    elif change == 'reference':
        path = source/'evaluations/bart/evaluation/evaluation.jsonl'
        rows = path.read_text().splitlines()
        row = json.loads(rows[0])
        row['reference_result_json'] = 'different answer'
        rows[0] = json.dumps(row)
        path.write_text('\n'.join(rows)+'\n')
    elif change == 'validation':
        metrics['validation_sha256'] = '0'*64
    elif change == 'eligible':
        metrics['num_evaluated'] = 19885
    elif change == 'nan':
        metrics['pass@1'] = float('nan')
    else:
        marker['models']['bart']['pass@1'] = .1
    put(metrics_path, metrics)
    if change not in ('summary', 'winner', 'reference'):
        marker['models']['bart'] = metrics
    put(source/'stage1_complete.json', marker)
    with pytest.raises((ValueError, FloatingPointError)):
        runner.verify_stage1(source, GOLDEN)


def test_stage2_training_configuration_is_full_budget_and_refuses_batch_drift(tmp_path):
    runner = api()
    config = runner.make_training_config('copy', tmp_path/'trial', 'f'*64)
    assert (config.architecture, config.precision, config.expected_rows) == ('copy', 'fp16', (4000000, 20000))
    assert (config.micro_batch, config.global_batch, config.learning_rate, config.num_train_epochs) == (128, 256, .0002, 1.)
    assert (config.seed, config.eval_steps, config.max_steps) == (42, 1000, -1)
    assert not any((config.gradient_checkpointing, config.torch_compile, config.use_cpu,
                    config.tiny_model, config.preflight))
    for architecture, batch in (('t5', 128), ('bart', 64), ('copy', 256)):
        with pytest.raises(ValueError):
            runner.make_training_config(architecture, tmp_path/'trial', 'f'*64, micro_batch=batch)


def training_fixture(root, *, architecture='bart'):
    runner = api()
    config = runner.make_training_config(architecture, root, 'a'*64)
    settings = asdict(config)
    settings.pop('output_dir')
    settings['expected_rows'] = [4000000, 20000]
    model = dict(d_model=1024, encoder_layers=8, decoder_layers=8,
                 attention_heads=8, ffn_dim=4096, vocab_size=8000)
    identity = dict(bundle_sha256='a'*64, selection_sha256='b'*64)
    put(root/'run_manifest.json', dict(config=settings, precision='fp16', world_size=1,
        gradient_accumulation_steps=2, **identity, model_config=model,
        train_rows=4000000, validation_rows=20000))
    put(root/'trainer_args.json', dict(num_train_epochs=1., max_steps=-1,
        per_device_train_batch_size=128, per_device_eval_batch_size=128,
        gradient_accumulation_steps=2, learning_rate=.0002, warmup_steps=.05,
        lr_scheduler_type='cosine', fp16=True, bf16=False, use_cpu=False,
        gradient_checkpointing=False, torch_compile=False, seed=42, data_seed=42,
        eval_steps=1000, save_steps=1000, logging_nan_inf_filter=False))
    put(root/'complete.json', dict(architecture=architecture, global_step=15625,
        metrics={'eval_token_loss': 1.2}, precision='fp16', world_size=1,
        effective_batch=256))
    put(root/'trainer_state.json', dict(global_step=15625))
    put(root/'log_history.json', [dict(step=1, loss=3.1), dict(step=15625, eval_token_loss=1.2)])
    put(root/'final/model_config.json', model)
    put(root/'final/config.json', dict(model_type='bart' if architecture == 'bart' else 'copy_bart',
        d_model=1024, encoder_layers=8, decoder_layers=8, encoder_attention_heads=8,
        decoder_attention_heads=8, encoder_ffn_dim=4096, decoder_ffn_dim=4096, vocab_size=8000))
    (root/'final/model.safetensors').write_bytes(b'fixture weights')
    (root/'source_tokenizer.json').write_text('fixture tokenizer')
    identity['tokenizer_sha256'] = digest(root/'source_tokenizer.json')
    return config, identity


@pytest.mark.parametrize('change', [None, 'steps', 'nonfinite', 'hidden_nonfinite', 'schedule', 'layers', 'batch', 'weights'])
def test_training_completion_requires_exact_config_finite_loss_and_checkpoint(tmp_path, change):
    runner = api()
    root = tmp_path/'trial'
    config, identity = training_fixture(root)
    if change in ('steps', 'nonfinite'):
        path = root/'complete.json'
        value = json.loads(path.read_text())
        if change == 'steps':
            value['global_step'] = 15624
        else:
            value['metrics']['eval_token_loss'] = float('inf')
        put(path, value)
    elif change == 'hidden_nonfinite':
        put(root/'log_history.json', [dict(step=1, loss=float('nan'))])
    elif change in ('schedule', 'batch'):
        path = root/'trainer_args.json'
        value = json.loads(path.read_text())
        value['warmup_steps' if change == 'schedule' else 'gradient_accumulation_steps'] = .1 if change == 'schedule' else 1
        put(path, value)
    elif change == 'layers':
        path = root/'final/config.json'
        value = json.loads(path.read_text())
        value['decoder_layers'] = 6
        put(path, value)
    elif change == 'weights':
        (root/'final/model.safetensors').unlink()
    if change:
        with pytest.raises((ValueError, FloatingPointError)):
            runner.verify_training(root, config, identity)
    else:
        assert runner.verify_training(root, config, identity)['global_step'] == 15625


def test_completed_phase_is_sealed_and_corruption_cannot_be_skipped(tmp_path):
    runner = api()
    phase, receipt = tmp_path/'trial', tmp_path/'training_verified.json'
    launches = []
    def launch():
        launches.append('launched')
        phase.mkdir()
        (phase/'complete.json').write_text('{"status":"complete"}')
        (phase/'weights').write_bytes(b'unchanged')
    def verify():
        return json.loads((phase/'complete.json').read_text())
    result = runner.ensure_phase(phase, receipt, request_sha='c'*64, launch=launch, verify=verify)
    assert result['status'] == 'complete'
    runner.ensure_phase(phase, receipt, request_sha='c'*64, launch=launch, verify=verify)
    assert len(launches) == 1
    (phase/'weights').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='integrity'):
        runner.ensure_phase(phase, receipt, request_sha='c'*64, launch=launch, verify=verify)
    assert len(launches) == 1


def test_failure_retains_partial_outputs_and_refuses_duplicate_attempt(tmp_path):
    runner = api()
    phase, receipt = tmp_path/'trial', tmp_path/'verified.json'
    def launch():
        phase.mkdir()
        (phase/'partial.bin').write_bytes(b'inspect me')
        raise RuntimeError('synthetic OOM')
    with pytest.raises(RuntimeError, match='OOM'):
        runner.ensure_phase(phase, receipt, request_sha='c'*64, launch=launch, verify=lambda: {})
    assert (phase/'partial.bin').read_bytes() == b'inspect me'
    assert not receipt.exists()
    with pytest.raises(ValueError, match='Incomplete|unverified'):
        runner.ensure_phase(phase, receipt, request_sha='c'*64,
            launch=lambda: pytest.fail('partial output must not launch again'), verify=lambda: {})


def test_queue_lock_excludes_a_second_driver(tmp_path):
    runner = api()
    with runner.queue_lock(tmp_path):
        with pytest.raises(RuntimeError, match='already running'):
            with runner.queue_lock(tmp_path):
                pytest.fail('duplicate driver entered')


def test_failed_dependency_records_clear_state_without_launching(tmp_path):
    runner = api()
    output = tmp_path/'out'
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.run_stage2(tmp_path/'missing-stage1', tmp_path/'prepared', output,
            reference_identity=GOLDEN)
    state = json.loads((output/'queue_state.json').read_text())
    assert state['status'] == 'failed'
    assert state['phase'] == 'dependency_verification'
    assert state['child_pid'] is None
    assert state['error']
    assert not (output/'trials').exists()


@pytest.mark.parametrize('change', [None, 'beams', 'batch', 'rows', 'reference', 'checkpoint', 'predictions'])
def test_evaluation_gate_rejects_decoding_or_reference_drift(tmp_path, change):
    runner = api()
    root = stage1_fixture(tmp_path/'source')/'evaluations/bart'
    identity = dict(bundle_sha256='a'*64, selection_sha256='b'*64)
    checkpoint = tmp_path/'trial/final'
    config = dict(architecture='bart', checkpoint=str(checkpoint), checkpoint_step=15625,
        validation_rows=20000, validation_sha256=json.loads(GOLDEN.read_text())['validation_sha256'],
        **identity, batch_size=64, num_beams=4, max_new_tokens=256, length_penalty=1.,
        device='cuda:0', do_sample=False, use_cache=True, official_test_used=False)
    if change in ('beams', 'batch', 'rows', 'checkpoint'):
        key, value = {'beams': ('num_beams', 2), 'batch': ('batch_size', 32),
                      'rows': ('validation_rows', 2000), 'checkpoint': ('checkpoint_step', 3907)}[change]
        config[key] = value
    elif change == 'reference':
        path = root/'evaluation/evaluation.jsonl'
        path.write_text(path.read_text().replace('"reference_status": "ok"',
                                                '"reference_status": "changed"', 1))
    put(root/'config.json', config)
    put(root/'progress.json', {'phase':'complete'})
    (root/'predictions.jsonl').write_text(''.join(json.dumps(dict(sample_row=index))+'\n'
        for index in range(1 if change == 'predictions' else 20000)))
    if change:
        with pytest.raises(ValueError):
            runner.verify_evaluation(root, architecture='bart', checkpoint=checkpoint,
                identity=identity, reference_identity=GOLDEN)
    else:
        assert runner.verify_evaluation(root, architecture='bart', checkpoint=checkpoint,
            identity=identity, reference_identity=GOLDEN)['num_evaluated'] == 19886


def test_prepared_pair_uses_actual_checksums_and_same_tiny_validation(tmp_path):
    """Exercise the real preparer and Arrow loader, with eight rows instead of4M."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_a100_diversity import make_sources
    from tools.prepare_a100_diversity import prepare_diversity
    runner = api()
    original, full = make_sources(tmp_path)
    prepared = tmp_path/'prepared'
    prepare_diversity(original, full, prepared, base_rows=(4, 2), full_rows=(12, 6),
        repeat_factor=2, batch_size=2)
    from kaggle_experiments import load_bundle
    data, _, _ = load_bundle(prepared/'repeat4m', expected_rows=(8, 2))
    val_digest = hashlib.sha256()
    for row in data['val']:
        val_digest.update((json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n').encode())
    kwargs = dict(expected_rows=(8, 2), base_rows=(4, 2), repeat_factor=2,
                  full_rows=(12, 6), expected_original_manifest_sha=digest(original/'bundle_manifest.json'),
                  expected_validation_sha=val_digest.hexdigest())
    proof = runner.verify_prepared(prepared, **kwargs)
    for variant in ('repeat4m', 'distinct4m'):
        assert proof['bundles'][variant]['bundle_sha256'] == digest(prepared/variant/'bundle_manifest.json')
    with pytest.raises(ValueError, match='ordered rows'):
        runner.verify_prepared(prepared, **(kwargs | {'expected_validation_sha':'0'*64}))
    # An otherwise valid copied bundle cannot replace the other strategy.
    marker = json.loads((prepared/'complete.json').read_text())
    marker['bundles']['repeat4m']['bundle_manifest_sha256'] = '0'*64
    put(prepared/'complete.json', marker)
    with pytest.raises(ValueError, match='manifest SHA'):
        runner.verify_prepared(prepared, **kwargs)


def test_subprocess_failure_is_reported_without_reusing_its_log(tmp_path):
    import sys
    runner = api()
    state = dict(variant='repeat4m', phase='training')
    with runner.queue_lock(tmp_path) as lock:
        with pytest.raises(RuntimeError, match='exited 7'):
            runner._invoke([sys.executable, '-c', 'print("synthetic failure"); raise SystemExit(7)'],
                log=tmp_path/'logs/training.log', state=state,
                state_path=tmp_path/'queue_state.json', lock=lock)
    assert 'synthetic failure' in (tmp_path/'logs/training.log').read_text()
    assert state['child_pid'] is None
    with runner.queue_lock(tmp_path) as lock:
        with pytest.raises(FileExistsError):
            runner._invoke([sys.executable, '-c', 'raise SystemExit(0)'],
                log=tmp_path/'logs/training.log', state=state,
                state_path=tmp_path/'queue_state.json', lock=lock)


def test_evaluation_worker_passes_exact_stage2_bundle_and_decoding_contract(monkeypatch, tmp_path):
    from argparse import Namespace
    from tools import evaluate_fixed_bundle
    runner = api()
    def evaluate(bundle, checkpoint, output, **kwargs):
        assert bundle == str(tmp_path/'bundle')
        assert checkpoint == str(tmp_path/'final')
        assert output == str(tmp_path/'evaluation')
        assert kwargs == dict(architecture='copy', expected_rows=(4000000, 20000),
            expected_manifest_sha='d'*64, device='cuda:0', batch_size=64,
            max_new_tokens=256, num_beams=4, length_penalty=1.)
        return {'evaluated':True}
    monkeypatch.setattr(evaluate_fixed_bundle, 'run', evaluate)
    args = Namespace(phase='evaluation', bundle_root=str(tmp_path/'bundle'),
        checkpoint=str(tmp_path/'final'), output_dir=str(tmp_path/'evaluation'),
        architecture='copy', manifest_sha='d'*64, micro_batch=128)
    assert runner.worker(args) == {'evaluated':True}
    args.micro_batch = 64
    with pytest.raises(ValueError, match='batch layout'):
        runner.worker(args)


def queue_fixture(tmp_path, monkeypatch):
    """Replace only the4M/GPU boundaries; use real saved-result validation."""
    runner = api()
    stage1 = stage1_fixture(tmp_path/'stage1')
    prepared = tmp_path/'prepared'
    prepared.mkdir()
    identity = dict(bundle_sha256='a'*64, selection_sha256='b'*64,
        tokenizer_sha256=hashlib.sha256(b'fixture tokenizer').hexdigest())
    monkeypatch.setattr(runner, 'verify_prepared', lambda path: dict(
        completion_sha256='f'*64, bundles={name:identity for name in ('repeat4m', 'distinct4m')}))
    return runner, stage1, prepared, identity


def test_queue_runs_sequentially_then_verifies_every_phase_without_relaunch(monkeypatch, tmp_path):
    runner, stage1, prepared, identity = queue_fixture(tmp_path, monkeypatch)
    launches = []
    def invoke(command, **kwargs):
        args = dict(zip(command[3::2], command[4::2]))
        phase, output = args['--phase'], Path(args['--output-dir'])
        launches.append((output.name, phase))
        assert args['--manifest-sha'] == 'a'*64
        if phase == 'training':
            training_fixture(output)
        else:
            references = json.loads(GOLDEN.read_text())
            score = .26 if output.name == 'repeat4m' else .27
            put(output/'metrics.json', dict(architecture='bart', **{'pass@1':score},
                num_evaluated=19886, num_beams=4, max_new_tokens=256,
                length_penalty=1., validation_sha256=references['validation_sha256']))
            put(output/'config.json', dict(architecture='bart', checkpoint=args['--checkpoint'],
                checkpoint_step=15625, validation_rows=20000,
                validation_sha256=references['validation_sha256'], bundle_sha256='a'*64,
                selection_sha256='b'*64, device='cuda:0', batch_size=64, num_beams=4,
                max_new_tokens=256, length_penalty=1., do_sample=False,
                use_cache=True, official_test_used=False))
            put(output/'progress.json', {'phase':'complete'})
            (output/'evaluation').mkdir()
            (output/'evaluation/evaluation.jsonl').write_text(''.join(
                json.dumps(row)+'\n' for row in references['rows']))
            (output/'predictions.jsonl').write_text(''.join(json.dumps(dict(sample_row=i))+'\n'
                for i in range(20000)))
    monkeypatch.setattr(runner, '_invoke', invoke)
    out = tmp_path/'out'
    first = runner.run_stage2(stage1, prepared, out, reference_identity=GOLDEN)
    assert launches == [('repeat4m','training'), ('repeat4m','evaluation'),
                        ('distinct4m','training'), ('distinct4m','evaluation')]
    assert first['ranking'] == ['distinct4m', 'repeat4m']
    assert runner.run_stage2(stage1, prepared, out, reference_identity=GOLDEN) == first
    assert len(launches) == 4
    (out/'trials/repeat4m/final/model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='integrity'):
        runner.run_stage2(stage1, prepared, out, reference_identity=GOLDEN)
    assert len(launches) == 4
    assert json.loads((out/'queue_state.json').read_text())['status'] == 'failed'


def test_queue_stops_before_launch_if_source_snapshot_changes(monkeypatch, tmp_path):
    runner, stage1, prepared, _ = queue_fixture(tmp_path, monkeypatch)
    snapshots = iter([{'runner.py':'original'}, {'runner.py':'changed'}])
    monkeypatch.setattr(runner, '_sources', lambda: next(snapshots))
    monkeypatch.setattr(runner, '_invoke', lambda *args, **kwargs: pytest.fail('changed source launched'))
    with pytest.raises(ValueError, match='source'):
        runner.run_stage2(stage1, prepared, tmp_path/'out', reference_identity=GOLDEN)


def test_queue_detects_actual_code_runner_change_after_saving_source_snapshot(monkeypatch, tmp_path):
    import shutil
    runner, stage1, prepared, _ = queue_fixture(tmp_path, monkeypatch)
    temporary_root = tmp_path/'source-copy'
    # Keep source hashing real; only a temporary copy may be mutated by this test.
    for name in set(runner._sources()) | {'_code_runner.py'}:
        destination = temporary_root/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runner.ROOT/name, destination)
    code_runner = temporary_root/'_code_runner.py'
    original_sha = digest(code_runner)
    monkeypatch.setattr(runner, 'ROOT', temporary_root)
    write_json = runner.write_json
    def write_and_change_source(path, value):
        write_json(path, value)
        if Path(path).name == 'run_request.json':
            code_runner.write_text(code_runner.read_text()+'\n# Changed after source snapshot.\n')
    monkeypatch.setattr(runner, 'write_json', write_and_change_source)
    monkeypatch.setattr(runner, '_invoke', lambda *args, **kwargs: pytest.fail('changed execution runner launched'))
    out = tmp_path/'out'
    with pytest.raises(ValueError, match='source'):
        runner.run_stage2(stage1, prepared, out, reference_identity=GOLDEN)
    request = json.loads((out/'run_request.json').read_text())
    assert request['sources']['_code_runner.py'] == original_sha
    assert digest(code_runner) != original_sha
    assert not (out/'trials').exists()
    assert json.loads((out/'queue_state.json').read_text())['status'] == 'failed'


def test_queue_failure_records_phase_and_never_restarts_partial_gpu_work(monkeypatch, tmp_path):
    runner, stage1, prepared, _ = queue_fixture(tmp_path, monkeypatch)
    launches = []
    def invoke(command, **kwargs):
        launches.append(command)
        args = dict(zip(command[3::2], command[4::2]))
        partial = Path(args['--output-dir'])
        partial.mkdir(parents=True)
        (partial/'partial.bin').write_bytes(b'failed GPU work')
        raise RuntimeError('synthetic OOM')
    monkeypatch.setattr(runner, '_invoke', invoke)
    out = tmp_path/'out'
    with pytest.raises(RuntimeError, match='OOM'):
        runner.run_stage2(stage1, prepared, out, reference_identity=GOLDEN)
    state = json.loads((out/'queue_state.json').read_text())
    assert (state['status'], state['phase'], state['variant'], state['child_pid']) == ('failed', 'training', 'repeat4m', None)
    assert not (out/'stage2_complete.json').exists()
    with pytest.raises(ValueError, match='Incomplete|unverified'):
        runner.run_stage2(stage1, prepared, out, reference_identity=GOLDEN)
    assert len(launches) == 1
