"""A realtime clock jump expires cached and uncached optimizer closures."""

import time

from piginot.training.lbfgs_stage import train_lbfgs_stage
from tests.training.test_lbfgs_stage import Rosenbrock, config, objective


def test_expired_utc_deadline_reserves_no_evaluation(tmp_path):
    result = train_lbfgs_stage(
        tmp_path / "stage",
        model=Rosenbrock(),
        objective=objective([]),
        binding="a" * 64,
        config=config(),
        deadline_utc=time.time() - 1,
    )
    assert result.status == "TIME_LIMIT" and result.evaluations == 0


def test_forward_clock_jump_rolls_back_and_keeps_spent_evaluation(
    tmp_path, monkeypatch
):
    clock = {"now": 100.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    model = Rosenbrock()
    before = model.point.detach().clone()
    base = objective([])

    def evaluate(model):
        value = base(model)
        clock["now"] = 102.0
        return value

    result = train_lbfgs_stage(
        tmp_path / "stage",
        model=model,
        objective=evaluate,
        binding="a" * 64,
        config=config(),
        deadline_utc=101.0,
    )
    assert result.status == "TIME_LIMIT" and result.evaluations == 1
    assert (model.point == before).all()
