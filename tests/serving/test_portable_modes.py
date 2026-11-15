"""New candidates cannot acquire a historical serving claim."""

from types import SimpleNamespace
from typing import Any

import pytest

from piginot.bounds import CalibrationError
from piginot.serving import hybrid


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"mode": "certified"}, ValueError, "only fast"),
        ({"mode": "exact"}, ValueError, "only fast"),
        ({"recipe": object()}, ValueError, "recipe"),
        ({"calibration": object()}, CalibrationError, "calibration"),
        ({"confidence": True}, CalibrationError, "calibration"),
    ],
)
def test_portable_claims_refuse_before_prediction_or_fem(
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("unsupported mode reached prediction or FEM")

    monkeypatch.setattr(hybrid, "predict", forbidden)
    monkeypatch.setattr(hybrid, "_fem", forbidden)
    bundle = SimpleNamespace(model=SimpleNamespace(problem=object()))
    with pytest.raises(error, match=message):
        hybrid.serve(bundle, {}, **options)


@pytest.mark.parametrize("confidence", [None, False])
def test_portable_fast_returns_uncalibrated_prediction(
    monkeypatch: pytest.MonkeyPatch, confidence: bool | None
) -> None:
    prediction = SimpleNamespace(
        document=lambda: {"confidence_status": "NOT_CALIBRATED"}
    )
    monkeypatch.setattr(hybrid, "predict", lambda *args, **kwargs: prediction)
    bundle = SimpleNamespace(model=SimpleNamespace(problem=object()))
    served = hybrid.serve(bundle, {}, confidence=confidence)
    assert served.path is hybrid.ServingPath.OPERATOR
    assert served.bounds is None
    assert served.document() == {"confidence_status": "NOT_CALIBRATED"}
