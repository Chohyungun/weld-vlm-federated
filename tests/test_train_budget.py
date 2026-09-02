"""R·E·N 확정값 고정 — 2026-09-03 총괄 확정 (의사결정로그 "본실험 착수 승인 + R·E 확정").

이 시험이 깨지는 유일한 정당한 경로는 **총괄이 값을 다시 확정하는 것**이다.
그 외의 diff 는 전부 사고다 — 학습량 등가 R × E = N 이 다섯 칸 공정성 주장의
골격이라 값 하나가 슬쩍 바뀌면 비교 자체가 무효가 된다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _budget() -> dict:
    cfg = yaml.safe_load((REPO_ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    return cfg["fixed_before_main_runs"]["train_budget"]


def test_확정값이_그대로_있다():
    assert _budget() == {"num_rounds": 50, "local_epochs": 2, "total_epochs": 100}


def test_등가식이_성립한다():
    """R × E = N. 세 값은 등식으로 묶여 있어 하나만 바꿀 수 없다."""
    b = _budget()
    assert b["num_rounds"] * b["local_epochs"] == b["total_epochs"]


def test_전부_정수다():
    """YAML 이 "50" 같은 문자열을 실어도 조용히 통과하지 않게. bool 도 int 의
    하위형이라 명시적으로 배제한다."""
    for k, v in _budget().items():
        assert type(v) is int, f"{k} 가 int 가 아니다: {type(v).__name__}"
