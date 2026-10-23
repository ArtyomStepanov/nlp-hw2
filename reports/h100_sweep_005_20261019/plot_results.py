"""Build the final comparison from saved sweep summaries and Trainer histories."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOTS = [HERE.parent / 'h100_sweep_20261019', HERE]
rows = []
histories = {}
for root in ROOTS:
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['status'] == 'complete' and not summary['failures']
    assert len(summary['completed']) == 4
    for result in summary['completed']:
        assert result['steps'] == 3907 and result['train_rows'] == 1_000_000
        assert result['validation_rows'] == 20_000
        assert result['metrics']['eval_target_tokens'] == 1_344_127
        trial = result['trial']
        row = dict(name=trial['name'], encoder=trial['encoder_layers'],
                   decoder=trial['decoder_layers'], lr=trial['learning_rate'],
                   loss=result['metrics']['eval_token_loss'], seconds=result['wall_seconds'],
                   parameters=result['parameters'], peak_allocated_gib=result['peak_allocated_gib'],
                   checkpoint=result['checkpoint'], data_sha256=trial['data_manifest_sha256'])
        rows.append(row)
        history = json.loads((root / f"{trial['name']}_log_history.json").read_text())
        histories[trial['name']] = {item['step']: item['eval_token_loss']
                                   for item in history if 'eval_token_loss' in item}

assert len({r['name'] for r in rows}) == 8
assert len({r['data_sha256'] for r in rows}) == 1
rows.sort(key=lambda r: r['loss'])
combined = dict(train_rows=1_000_000, validation_rows=20_000, seed=42, epochs=1,
                global_batch=256, metric='eval_token_loss',
                total_trial_wall_seconds=sum(r['seconds'] for r in rows), results=rows)
(HERE / 'combined_summary.json').write_text(json.dumps(combined, indent=2) + '\n')

colors = {(8, 8): '#168079', (8, 12): '#3273B9', (10, 10): '#C67C2E', (12, 8): '#AF4F63'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, (ax, bars) = plt.subplots(1, 2, figsize=(13.5, 5.7), gridspec_kw={'width_ratios': [1, 1.15]})
for row in rows:
    if row['lr'] != 2e-4:
        continue
    points = sorted(histories[row['name']].items())
    ax.plot([x for x, _ in points], [y for _, y in points], marker='o', markersize=4,
            color=colors[row['encoder'], row['decoder']], label=f"{row['encoder']}+{row['decoder']}")
ax.set(title='Архитектуры при LR 2e-4', xlabel='Шаг оптимизатора', ylabel='Validation token loss ↓')
ax.set_xticks([1000, 2000, 3000, 3907])
ax.grid(alpha=.18)
ax.legend(title='Encoder + decoder', frameon=False)
labels = [f"{r['encoder']}+{r['decoder']} · LR {r['lr']:.0e}" for r in rows]
bars.barh(labels, [r['loss'] for r in rows], height=.65,
          color=[colors[r['encoder'], r['decoder']] for r in rows])
bars.invert_yaxis()
for i, row in enumerate(rows):
    bars.text(row['loss']+.012, i, f"{row['loss']:.4f}", va='center', fontsize=10)
bars.set(title='Все восемь конфигураций', xlabel='Финальный validation token loss ↓', xlim=(0, 1.0))
bars.grid(axis='x', alpha=.18)
bars.set_axisbelow(True)
for panel in (ax, bars):
    panel.spines[['top', 'right']].set_visible(False)
fig.suptitle('Поиск архитектуры и LR на H100: одна эпоха', fontsize=15, y=.98)
fig.text(.5, .925, '1 млн train · 20 тыс. val · seed 42 · d_model 1024 · общий batch 256',
         ha='center', color='#555555', fontsize=10)
fig.text(.5, .015, 'Один seed и короткий бюджет: рейтинг предварительный; качество генерации и время на T4 не измерены.',
         ha='center', color='#555555', fontsize=9)
fig.tight_layout(rect=(0, .045, 1, .89), w_pad=3)
fig.savefig(HERE / 'comparison.png', dpi=180)
fig.savefig(HERE / 'comparison.pdf')
plt.close(fig)
print(json.dumps(combined, indent=2))
