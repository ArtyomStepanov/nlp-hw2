"""Create a separate verified benchmark overlay; preserve the uploaded bundle."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil


def patch_source(source):
    old = "            require(getattr(generation, key), expected, f'Loaded generation {key}')"
    new = "\n".join((
        "            actual = getattr(generation, key)",
        "            if actual is None:",
        "                actual = generation._get_default_generation_params().get(key)",
        "            require(actual, expected, f'Loaded generation {key}')"))
    if source.count(old) != 1:
        raise ValueError('Expected exactly one original generation-validation statement')
    return source.replace(old, new)


def repair_bundle(bundle_root, output_root):
    root, output = Path(bundle_root).resolve(), Path(output_root).resolve()
    partial = output.with_name('.'+output.name+'.partial')
    receipt_path = output.with_name(output.name+'.repair.json')
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Overlay must be separate from the original bundle')
    if output.exists() or partial.exists() or receipt_path.exists():
        raise FileExistsError('Fresh overlay and receipt paths are required')
    api = runpy.run_path(str(root/'benchmark_full_t4.py'))
    manifest, _ = api['verify_bundle'](root)
    digest = api['digest']
    original_manifest_sha = digest(root/'manifest.json')
    source = (root/'benchmark_full_t4.py').read_text()
    patched = patch_source(source)
    original_source_sha = digest(root/'benchmark_full_t4.py')
    partial.mkdir(parents=True, exist_ok=False)
    for name in manifest['files']:
        source_path, target = root/name, partial/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == 'checkpoint/model.safetensors':
            os.link(source_path, target)
        else:
            shutil.copy2(source_path, target)
    (partial/'benchmark_full_t4.py').write_text(patched)
    repair = dict(kind='generation_effective_default_validation',
        original_bundle_manifest_sha256=original_manifest_sha,
        original_benchmark_sha256=original_source_sha,
        repaired_benchmark_sha256=digest(partial/'benchmark_full_t4.py'),
        generation_behavior_changed=False, weights_changed=False,
        explanation='Validate None with the same pinned Transformers defaults used by generate; '
                    'retain rejection of explicit nondefault values. Generation configuration is unchanged.')
    updated = json.loads(json.dumps(manifest))
    updated['validation_repair'] = repair
    path = partial/'benchmark_full_t4.py'
    updated['files']['benchmark_full_t4.py'] = dict(bytes=path.stat().st_size, sha256=digest(path))
    (partial/'manifest.json').write_text(json.dumps(updated, ensure_ascii=False, indent=2)+'\n')
    api['verify_bundle'](partial)
    if digest(root/'manifest.json') != original_manifest_sha or digest(root/'benchmark_full_t4.py') != original_source_sha:
        raise ValueError('Original frozen bundle changed during overlay creation')
    new_manifest_sha = digest(partial/'manifest.json')
    partial.rename(output)
    receipt = dict(status='verified', original_bundle_root=str(root), overlay_root=str(output),
        overlay_manifest_sha256=new_manifest_sha, repair=repair,
        patcher_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            if '__file__' in globals() else None)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(receipt, receipt_path=str(receipt_path)), indent=2), flush=True)
    return receipt
