"""conf 스윕 격자 고정 — 2026-09-08 총괄 판정 22번 (의사결정로그 "회복률은 곡선으로 낸다").

단일 임계를 사전등록하지 않고 격자를 등록한다. 여기서 고정하는 것은 넷이다.

1. **값 14개 그대로** — 시드 2·3 이 시드 1 과 같은 격자를 쓴다. 격자가 바뀌면 시드 사이
   채점 기준이 갈린다.
2. **격자의 구조** — 오름차순 · (0, 1] · D 의 재추론 하한 아래로 내려가지 않는다.
3. **단일 임계 키의 부재** — D 의 `resolve_conf` 가 configs 에서 어떤 값도 찾지 못해야
   "단일 임계 사전등록 금지" 가 실제로 성립한다.
4. **채점기가 격자를 파일에서 읽는다** — 구현은 D 소관(22번 §3, dispatch_D 과제 3).
   착지 전까지 `xfail(strict=True)` 로 둔다. D 가 구현하면 XPASS 가 실패로 보고되므로
   A 에 통지해 표식을 걷는다.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_YAML = REPO_ROOT / "configs/base.yaml"
KEY = "evaluation.detection.conf_sweep_grid"

GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
"""22번 §1-2-2 가 지정한 14점 — D 가 시드 1 채점에 쓴 격자와 동일하다."""

SEED1_SWEEP = REPO_ROOT / "outputs/main_d/seed1/sweep_detection_conf_v1.json"
"""시드 1 스윕 산출물. 추적되지 않는 실물이라 없으면 그 시험만 건너뛴다."""


def _cfg() -> dict:
    return yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))


def _dig(cfg: object, dotted: str) -> object | None:
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _grid() -> list:
    g = _dig(_cfg(), KEY)
    assert isinstance(g, list), f"{KEY} 가 리스트가 아니다: {type(g).__name__}"
    return g


# --------------------------------------------------------------------------------------
# 1 · 2 — 값과 구조
# --------------------------------------------------------------------------------------


def test_확정값이_그대로_있다():
    assert _grid() == GRID


def test_오름차순이고_0_과_1_사이다():
    g = _grid()
    assert len(g) == 14
    for v in g:
        assert type(v) is float, f"{v!r} 가 float 가 아니다"
        assert 0.0 < v <= 1.0
    assert all(a < b for a, b in pairwise(g)), "격자가 오름차순이 아니다"


def test_하한_아래로_내려가지_않는다():
    """첫 점은 D 의 재추론 하한과 같다. 하한 아래 박스는 저장되지 않으므로 격자를 그 밑으로
    내리면 스윕이 조용히 빈 점을 낸다."""
    from evaluation.params import CONF_FLOOR

    assert _grid()[0] == CONF_FLOOR


def test_운용점_예시가_격자_위에_있다():
    """0.25 는 확증 기준이 아니라 운용점 예시다. 표에 남기려면 곡선 위의 한 점이어야 한다."""
    from evaluation.params import CONF_FALLBACK

    g = _grid()
    assert 0.25 in g
    assert CONF_FALLBACK in g


def test_시드1_채점이_쓴_격자와_같다():
    """시드 2·3 은 시드 1 과 같은 격자여야 한다. 시드 1 산출물이 기록한 격자와 대조한다.
    D 의 폴백 상수가 남아 있다면 그것도 같아야 한다 — 정본이 둘로 갈리지 않게."""
    from evaluation import params

    g = _grid()
    fallback = getattr(params, "CONF_SWEEP", None)
    if fallback is not None:
        assert [float(v) for v in fallback] == g, "D 폴백 상수가 configs 격자와 갈렸다"

    if not SEED1_SWEEP.exists():
        pytest.skip(f"시드 1 스윕 산출물 없음: {SEED1_SWEEP}")
    recorded = json.loads(SEED1_SWEEP.read_text(encoding="utf-8"))["params"]["conf_sweep"]
    assert [float(v) for v in recorded] == g, "시드 1 이 쓴 격자와 등록값이 다르다"


# --------------------------------------------------------------------------------------
# 3 — 단일 임계는 configs 에 없다
# --------------------------------------------------------------------------------------


def test_단일_임계_키가_configs_에_없다():
    """22번: 단일 임계 사전등록 금지. D 의 `resolve_conf` 후보 경로 전부가 비어 있어야
    채점기가 configs 에서 단일 임계를 받아 가는 일이 없다."""
    from evaluation.params import CONF_KEYS

    cfg = _cfg()
    for k in CONF_KEYS:
        assert _dig(cfg, k) is None, f"단일 임계가 configs 에 등록돼 있다: {k}"
    assert "conf" not in cfg["evaluation"]["detection"]


# --------------------------------------------------------------------------------------
# 4 — 채점기가 격자를 파일에서 읽는다 (D 구현 대기)
# --------------------------------------------------------------------------------------


# xfail(strict) 표식은 D 구현 착지(XPASS 통지)로 걷었다 — 이제부터 이 시험이 실질 검증이다.
def test_채점기가_격자를_configs_에서_읽는다(monkeypatch):
    """`load_base_config` 를 등록값과 다른 격자로 바꿔치기하면 채점기의 `conf_sweep` 이 그것을
    따라야 한다. 상수를 베껴 둔 것과 파일을 읽는 것은 이 시험만이 구별한다."""
    from evaluation import params

    probe = [0.01, 0.20, 0.50]
    monkeypatch.setattr(
        params,
        "load_base_config",
        lambda *a, **k: {"evaluation": {"detection": {"conf_sweep_grid": probe}}},
    )
    p = params.ScoringParams(snapshot=Path("s"), pilot=Path("p"), out=Path("o"))
    assert [float(v) for v in p.conf_sweep] == probe
