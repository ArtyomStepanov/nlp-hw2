"""Exercise the actual benchmark validation loop, without loading model weights."""
import ast
import json
from pathlib import Path

import pytest
from transformers import GenerationConfig

from tools import benchmark_full_t4 as benchmark


def actual_generation():
    path = Path(__file__).resolve().parents[1]/'reports/a100_full_20261023/final_report/training/trial/final/generation_config.json'
    return GenerationConfig.from_dict(json.loads(path.read_text()))


def validate(source, generation):
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'benchmark')
    loop = next(node for node in ast.walk(function) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [item.id for item in node.target.elts] == ['key', 'expected'])
    module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    exec(compile(module, '<actual benchmark validation>', 'exec'),
         {'generation': generation, 'require': benchmark.require})


def test_actual_saved_generation_leaves_standard_defaults_unset():
    generation = actual_generation()
    assert [getattr(generation, key) for key in ('length_penalty', 'early_stopping', 'do_sample', 'num_return_sequences')] == [None]*4
    before = generation.to_dict()
    validate(Path(benchmark.__file__).read_text(), generation)
    assert generation.to_dict() == before


def test_repaired_validation_accepts_effective_defaults_without_mutating_generation():
    from tools.repair_full_t4_defaults import patch_source
    generation = actual_generation()
    before = generation.to_dict()
    original = Path(benchmark.__file__).read_text().replace(
        "            actual = getattr(generation, key)\n            if actual is None:\n                actual = generation._get_default_generation_params().get(key)\n            require(actual, expected, f'Loaded generation {key}')",
        "            require(getattr(generation, key), expected, f'Loaded generation {key}')")
    validate(patch_source(original), generation)
    assert generation.to_dict() == before


@pytest.mark.parametrize('key,value', [('length_penalty', 1.2), ('early_stopping', True),
                                     ('do_sample', True), ('num_return_sequences', 2)])
def test_repaired_validation_still_rejects_nondefault_decoding(key, value):
    generation = actual_generation()
    setattr(generation, key, value)
    with pytest.raises(ValueError, match=f'Loaded generation {key}'):
        validate(Path(benchmark.__file__).read_text(), generation)
