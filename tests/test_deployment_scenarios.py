from pathlib import Path

import yaml

CONFIG = Path(__file__).parent.parent / "configs" / "deployment_scenarios.yaml"


def test_scenarios_are_complete() -> None:
    data = yaml.safe_load(CONFIG.read_text())
    assert set(data["scenarios"]) == {"A", "B", "C"}
    for scenario in data["scenarios"].values():
        for key in ("hardware", "inference", "api", "image", "training"):
            assert key in scenario


def test_gpu_forbidden_at_inference_outside_scenario_c() -> None:
    data = yaml.safe_load(CONFIG.read_text())
    assert data["scenarios"]["A"]["inference"]["gpu_allowed"] is False
    assert data["scenarios"]["B"]["inference"]["gpu_allowed"] is False
