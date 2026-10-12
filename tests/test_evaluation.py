import json

import datasets as ds
import pytest


def test_pass_at_one_separates_broken_code_from_wrong_answers(tmp_path):
    from evaluation import EvaluationConfig, evaluate_code

    predictions = ds.Dataset.from_dict({
        "question": ["correct", "wrong", "syntax", "missing", "exception", "empty"],
        "generated_code": ["def solve(): return 2", "def solve(): return 3", "def solve(:", "x = 2", "def solve(): return 1 / 0", ""],
        "code": ["def solve(): return 2"] * 6,
    })
    result = evaluate_code(predictions, EvaluationConfig(workers=2, output_dir=tmp_path / "evaluation"))
    assert result.metrics["pass@1"] == pytest.approx(1 / 6)
    assert result.metrics["bad_code_rate"] == pytest.approx(4 / 6)
    assert result.metrics["wrong_answer_rate"] == pytest.approx(1 / 6)
    assert result.predictions["status"] == ["passed", "wrong_answer", "syntax_error", "missing_solve", "runtime_error", "empty_code"]
    assert result.metrics["num_evaluated"] == 6
    assert (tmp_path / "evaluation" / "evaluation.jsonl").is_file()
    assert json.loads((tmp_path / "evaluation" / "metrics.json").read_text())["pass@1"] == pytest.approx(1 / 6)


def test_reference_errors_are_visible_and_do_not_change_bad_code_denominator():
    from evaluation import evaluate_code

    result = evaluate_code(ds.Dataset.from_dict({"generated_code": ["def solve(): return 1", "def solve(:"], "code": ["def solve(): return 1", "def solve(:"]}))
    assert result.metrics["pass@1"] == 1
    assert result.metrics["num_evaluated"] == 1
    assert result.metrics["reference_error_rate"] == 0.5
    assert result.metrics["bad_code_rate"] == 0.5
    assert result.predictions["reference_status"] == ["ok", "syntax_error"]


def test_question_only_test_gets_code_health_but_no_invented_pass_at_one():
    from evaluation import evaluate_code

    result = evaluate_code(ds.Dataset.from_dict({"generated_code": ["def solve(): return 42"]}))
    assert result.metrics["pass@1"] is None
    assert result.metrics["num_evaluated"] == 0
    assert result.metrics["bad_code_rate"] == 0
    assert result.predictions["status"] == ["ok"]


@pytest.mark.parametrize("code,status", [
    ("def solve():\n    while True: pass", "timeout"),
    ("import os\ndef solve(): return os.getcwd()", "unsupported_code"),
    ("def solve(): return ().__class__.__bases__", "unsupported_code"),
    ("def solve(): return float('nan')", "invalid_output"),
    ("def solve(): return None", "invalid_output"),
    ("def solve():\n    x: int = 3\n    return x", "ok"),
    ("import math\ndef solve(): return math.sqrt(4)", "ok"),
    ("from math import sqrt\ndef solve(): return sqrt(4)", "ok"),
    ('g = (g.gi_frame.f_back.f_back.f_globals["os"].getcwd() for item in [0])\ndef solve():\n    for value in g:\n        return value', "unsupported_code"),
    ('g = ("{0.gi_frame.f_back.f_back.f_globals[os]}".format(g) for item in [0])\ndef solve(): return list(g)', "unsupported_code"),
    ('def solve():\n    xs = [1, 2]\n    xs.append(3)\n    return sum(x for x in xs)', "ok"),
])
def test_execution_is_bounded_and_classifies_failures(code, status):
    from evaluation import EvaluationConfig, evaluate_code

    result = evaluate_code(ds.Dataset.from_dict({"generated_code": [code]}), EvaluationConfig(timeout_seconds=0.5, workers=1))
    assert result.predictions["status"] == [status]


def test_numeric_tolerance_nested_results_and_bad_config():
    from evaluation import EvaluationConfig, evaluate_code

    result = evaluate_code(ds.Dataset.from_dict({
        "generated_code": ["def solve(): return 0.1 + 0.2", "def solve(): return [1, {'x': 2.0}]", "def solve(): return True"],
        "code": ["def solve(): return 0.3", "def solve(): return [1.0, {'x': 2}]", "def solve(): return 1"],
    }))
    assert result.predictions["status"] == ["passed", "passed", "wrong_answer"]
    with pytest.raises(ValueError):
        EvaluationConfig(timeout_seconds=0)
    with pytest.raises(ValueError):
        evaluate_code(ds.Dataset.from_dict({"generated_code": []}))
