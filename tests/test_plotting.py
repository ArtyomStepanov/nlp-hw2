import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_training_plot_handles_sparse_trainer_logs_and_writes_figure(tmp_path):
    from plotting import plot_training_history

    history = [{"step": 1, "loss": 2.0, "learning_rate": 0.001}, {"step": 2, "eval_loss": 1.8, "eval_token_loss": 1.7}, {"step": 3, "loss": 1.5}]
    (tmp_path / "log_history.json").write_text(json.dumps(history))
    fig = plot_training_history(tmp_path, save_path=tmp_path / "training.png")
    assert (tmp_path / "training.png").stat().st_size > 1000
    assert list(fig.axes[0].lines[0].get_ydata()) == [2.0, 1.5]
    assert any(list(line.get_ydata()) == [1.7] for line in fig.axes[0].lines)
    plt.close(fig)


def test_evaluation_and_comparison_plots_work_without_labels(tmp_path):
    from evaluation import evaluate_code
    from plotting import plot_evaluation, plot_experiment_comparison
    import datasets as ds

    result = evaluate_code(ds.Dataset.from_dict({"generated_code": ["def solve(): return 1", ""]}))
    fig = plot_evaluation(result, save_path=tmp_path / "quality.png")
    assert (tmp_path / "quality.png").stat().st_size > 1000
    assert fig.axes
    comparison = plot_experiment_comparison({"first": result.metrics, "second": {"pass@1": 0.7, "bad_code_rate": 0.1}}, save_path=tmp_path / "comparison.png")
    assert (tmp_path / "comparison.png").stat().st_size > 1000
    assert comparison.axes
    plt.close(fig)
    plt.close(comparison)


def test_evaluation_plot_uses_configured_prediction_column(tmp_path):
    from evaluation import EvaluationConfig, evaluate_code
    from plotting import plot_evaluation
    import datasets as ds

    result = evaluate_code(ds.Dataset.from_dict({"candidate": ["def solve(): return 1"]}), EvaluationConfig(prediction_column="candidate"))
    fig = plot_evaluation(result, save_path=tmp_path / "custom-column.png")
    assert (tmp_path / "custom-column.png").stat().st_size > 1000
    plt.close(fig)
