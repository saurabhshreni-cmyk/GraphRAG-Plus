"""Calibration tests."""

from pathlib import Path

from graphrag_plus.app.calibration.module import CalibrationModule


def test_calibration_outputs_expected_fields(tmp_path: Path) -> None:
    module = CalibrationModule(tmp_path / "calibration.json")
    module.update_reliability([0.2, 0.8], [0, 1])
    output = module.calibrate(0.7)
    assert 0.0 <= output.calibrated_confidence <= 1.0
    assert output.calibration_error >= 0.0

