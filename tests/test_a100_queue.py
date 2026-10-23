import json

import pytest


@pytest.mark.parametrize('change', ['evaluable', 'reference_result_json', None])
def test_reference_comparison_detects_changed_eligibility_or_answer(tmp_path, change):
    from tools.run_a100_stage1 import verify_reference_results
    rows = [dict(sample_row=0, evaluable=True, reference_status='ok', reference_result_json='2'),
        dict(sample_row=1, evaluable=False, reference_status='runtime_error', reference_result_json=None)]
    actual = [dict(row) for row in rows]
    if change == 'evaluable':
        # The total eligible count stays one, but a different row is scored.
        actual[0]['evaluable'], actual[1]['evaluable'] = False, True
    elif change:
        actual[0][change] = '3'
    p = tmp_path/'evaluation.jsonl'
    p.write_text(''.join(json.dumps(row)+'\n' for row in actual))
    if change:
        with pytest.raises(ValueError, match='reference mismatch'):
            verify_reference_results(p, rows)
    else:
        verify_reference_results(p, rows)
