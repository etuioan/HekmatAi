import pytest

from hekmatai.own_benchmark.run import main


def test_eval_run_refuses_until_decision_rules_are_written(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--seeds", "eval", "--out", str(tmp_path)])
    assert "decision_rules" in str(exc.value)
