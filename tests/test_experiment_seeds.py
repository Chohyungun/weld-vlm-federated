"""시드 3세트 고정 — 2026-09-03 총괄 확정 (의사결정로그 "시드 3세트 확정").

`experiment.seeds` 는 `scripts/main_det.py`(C) 가 읽는다. 시드 2·3 이 등록 전에는 C 가
값을 지어내지 않고 거부하도록 짜여 있었다 — 그 거부를 여기 등록이 푼다. 그러니 이 값이
바뀌면 시드 2·3 의 모든 결과가 무엇 위에서 나온 건지 흔들린다.

파생 규칙(base_seed + {0,1,2})도 시험한다. 규칙이 시험에 없으면 "값 3개가 맞다"만
남고 "왜 이 값인가"가 사라진다 — 자의성 제거가 확정의 근거였다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

SEED1 = 20260828
"""파일럿·초기 가중치·LoRA 초기화·§4-6 게이트가 전부 이 상수 위에 있다."""


def _cfg() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs/base.yaml").read_text(encoding="utf-8"))


def test_확정값이_그대로_있다():
    assert _cfg()["experiment"]["seeds"] == [20260828, 20260829, 20260830]


def test_기계적_파생_규칙이_성립한다():
    """시드 i = 시드 1 + (i − 1). 시드 1 은 기존 상수라 옮길 수 없다."""
    seeds = _cfg()["experiment"]["seeds"]
    assert seeds[0] == SEED1
    assert seeds == [SEED1 + i for i in range(3)]


def test_세_개고_전부_정수다():
    seeds = _cfg()["experiment"]["seeds"]
    assert len(seeds) == 3
    for v in seeds:
        assert type(v) is int, f"{v!r} 가 int 가 아니다"


def test_데이터_시드와_다르다():
    """전역 `seed`(분할·타일·샘플링)와 학습 시드는 다른 것이다. 같아지면 데이터 분할이
    학습 시드에 끌려간 것이므로 여기서 잡는다."""
    cfg = _cfg()
    assert cfg["seed"] not in cfg["experiment"]["seeds"]


def test_R_E_N_별칭이_train_budget_과_같은_값이다():
    """`experiment.rounds` 등은 `train_budget` 앵커의 별칭이다. 리터럴이 두 곳이 되면
    정본이 갈라지므로, 파싱 결과가 같은지를 본다."""
    cfg = _cfg()
    exp, tb = cfg["experiment"], cfg["fixed_before_main_runs"]["train_budget"]
    assert exp["rounds"] == tb["num_rounds"] == 50
    assert exp["local_epochs"] == tb["local_epochs"] == 2
    assert exp["total_epochs"] == tb["total_epochs"] == 100


def test_리터럴이_한_곳뿐이다():
    """앵커가 풀려 리터럴이 복제되면 파싱 결과는 같아도 정본은 둘이 된다. 원문에서 잡는다."""
    text = (REPO_ROOT / "configs/base.yaml").read_text(encoding="utf-8")
    exp_block = text[text.index("\nexperiment:"):]
    for key in ("rounds", "local_epochs", "total_epochs"):
        line = next(ln for ln in exp_block.splitlines() if ln.strip().startswith(f"{key}:"))
        assert "*" in line, f"experiment.{key} 가 별칭이 아니라 리터럴이다: {line.strip()}"
