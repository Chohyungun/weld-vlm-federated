"""vlm/coords.py 단위·속성 테스트 (지정 함정 구간 #4).

세 층으로 나눈다.
1. 계약 층 — 의존성 제로 리프 원칙, 해시 입력 셋. 깨지면 채점 쪽이 import를 못 한다.
2. 산식 층 — 골든 픽스처(손계산 기대값) + 무작위 왕복 속성.
3. 정책 층 — 판정은 하되 고치지 않는다는 규칙이 코드로 지켜지는가.
"""

from __future__ import annotations

import ast
import json
import math
import random
from pathlib import Path

import pytest

from vlm.coords import (
    COORD_SPACES,
    CoordCfg,
    CoordError,
    ImageGeom,
    coord_cfg_hash,
    coords_source_sha256,
    is_degenerate,
    quantize,
    roundtrip_budget_px,
    roundtrip_error_px,
    snap_to_bounds,
    to_model,
    to_px,
    validate_box_px,
)

REPO = Path(__file__).resolve().parent.parent
COORDS_PY = REPO / "vlm" / "coords.py"
FIXTURES_JSON = REPO / "vlm" / "coords_fixtures" / "golden_fixtures.json"

#: coords.py 가 import해도 되는 것 — 전부 표준 라이브러리다.
ALLOWED_IMPORTS = {"__future__", "hashlib", "json", "math", "dataclasses", "typing"}


# --------------------------------------------------------------------------
# 1. 계약 층
# --------------------------------------------------------------------------

def test_coords_는_표준라이브러리만_import한다():
    """의존성 제로 리프 원칙. torch·transformers·numpy 가 들어오면 D가 import할 수 없다."""
    tree = ast.parse(COORDS_PY.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
            elif node.level:
                pytest.fail(f"상대 import 금지 (리프 모듈이어야 한다): {ast.dump(node)}")
    extra = imported - ALLOWED_IMPORTS
    assert not extra, f"리프 원칙 위반 — 허용되지 않은 import: {sorted(extra)}"


def test_vlm_패키지_초기화가_비어있다():
    """vlm/__init__.py 에 import가 생기면 `from vlm.coords import ...` 에 딸려 들어간다."""
    tree = ast.parse((REPO / "vlm" / "__init__.py").read_text(encoding="utf-8"))
    non_doc = [
        n for n in tree.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    assert not non_doc, "vlm/__init__.py 는 docstring 외에 아무것도 두지 않는다"


def test_coord_cfg_hash_는_세_입력을_모두_반영한다():
    """CoordCfg 직렬화 + coords.py 소스 + transformers 버전 (계약 #4 §3-2, 게이트 #7 결정 H)."""
    base = CoordCfg(coord_space="NORM_1000", max_pixels=1_000_000, transformers_version="5.15.0")

    # (1) 설정값이 바뀌면 해시가 바뀐다
    assert coord_cfg_hash(base) != coord_cfg_hash(
        CoordCfg(coord_space="ABS_RESIZED", max_pixels=1_000_000, transformers_version="5.15.0")
    )
    assert coord_cfg_hash(base) != coord_cfg_hash(
        CoordCfg(coord_space="NORM_1000", max_pixels=999_999, transformers_version="5.15.0")
    )

    # (2) 라이브러리 버전이 바뀌면 해시가 바뀐다 — 깨지는 게 기능이다
    assert coord_cfg_hash(base) != coord_cfg_hash(
        CoordCfg(coord_space="NORM_1000", max_pixels=1_000_000, transformers_version="5.16.0")
    )

    # (3) 소스 해시가 입력에 실제로 들어가 있다
    assert coords_source_sha256() in _hash_payload_probe()

    # (4) 같은 입력이면 같은 값 (C와 D가 같은 값을 계산해야 한다)
    assert coord_cfg_hash(base) == coord_cfg_hash(base)


def _hash_payload_probe() -> str:
    """해시 입력 문자열을 재조립해 소스 해시 포함 여부를 확인한다."""
    cfg = CoordCfg(coord_space="NORM_1000", transformers_version="x")
    return "\n".join((cfg.canonical_json(), coords_source_sha256(), cfg.transformers_version))


def test_알수없는_coord_space는_즉시_실패한다():
    """기본값 추측 금지 — 선언 누락이 조용히 통과하면 그것이 좌표계 사고다."""
    with pytest.raises(CoordError):
        CoordCfg(coord_space="NORMALIZED")
    with pytest.raises(CoordError):
        CoordCfg(coord_space="")


def test_abs_resized는_리사이즈_치수를_추정하지_않는다():
    """smart_resize 재계산 금지. 값이 없으면 변환하지 않고 실패한다."""
    cfg = CoordCfg(coord_space="ABS_RESIZED")
    geom = ImageGeom(orig_w=1234, orig_h=707)  # resized 미제공
    with pytest.raises(CoordError):
        to_model([10, 10, 20, 20], geom, cfg)
    with pytest.raises(CoordError):
        to_px([10, 10, 20, 20], geom, cfg)


# --------------------------------------------------------------------------
# 2. 산식 층
# --------------------------------------------------------------------------

def _load_fixtures() -> list[dict]:
    return json.loads(FIXTURES_JSON.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _load_fixtures(), ids=lambda c: c["id"])
def test_골든_픽스처(case: dict):
    """손계산 기대값과 대조한다. D가 같은 픽스처를 독립 재실행한다(계약 #4 G9)."""
    tol = 1e-9
    cfg = CoordCfg(**case["cfg"])
    geom = ImageGeom(**case["geom"])
    exp = case["expect"]

    model = to_model(case["bbox_px"], geom, cfg)
    assert model == pytest.approx(exp["model_float"], abs=tol)

    q = quantize(model)
    assert list(q) == list(exp["model_quantized"])

    back = to_px(q, geom, cfg)
    assert back == pytest.approx(exp["back_px"], abs=tol)

    assert roundtrip_error_px(case["bbox_px"], geom, cfg) == pytest.approx(
        exp["max_roundtrip_err_px"], abs=tol
    )
    assert is_degenerate(q) is exp["degenerate_after_quantize"]


@pytest.mark.parametrize("space", COORD_SPACES)
def test_왕복_속성_1000케이스(space: str):
    """무작위 케이스에서 왕복 오차가 축별 예산 안에 있는가.

    예산은 축별 max(0.5, dim/2000). NORM_1000 은 분모 1000 양자화 때문에 원본 한 변이
    2000px 를 넘으면 0.5px 를 넘을 수 있는데, 이는 규약의 성질이지 구현 오차가 아니다.
    """
    rng = random.Random(20260817)
    checked = 0
    for _ in range(1000):
        w = rng.randint(64, 4096)
        h = rng.randint(64, 4096)
        x1 = rng.uniform(0, w - 2)
        y1 = rng.uniform(0, h - 2)
        x2 = rng.uniform(x1 + 1, w)
        y2 = rng.uniform(y1 + 1, h)

        if space == "ABS_RESIZED":
            geom = ImageGeom(w, h, resized_w=rng.randint(28, 2048), resized_h=rng.randint(28, 2048))
        else:
            geom = ImageGeom(w, h)
        cfg = CoordCfg(coord_space=space)

        q = quantize(to_model([x1, y1, x2, y2], geom, cfg))
        if is_degenerate(q):
            continue  # 뭉개진 박스는 예산 개념이 성립하지 않는다 — 폐기 대상이다
        checked += 1
        err = roundtrip_error_px([x1, y1, x2, y2], geom, cfg)
        budget = roundtrip_budget_px(geom, cfg)
        assert err <= budget + 1e-9, f"{space} {w}x{h} 왕복 오차 {err} > 예산 {budget}"
    assert checked > 900, f"유효 케이스가 너무 적다: {checked}"


def test_축을_뒤바꾼_구현은_통과할_수_없다():
    """1280x720 과 720x1280 은 다른 결과를 내야 한다. 단일 scale 구현을 잡는 회귀 테스트."""
    cfg = CoordCfg(coord_space="NORM_1000")
    box = [100, 200, 300, 400]
    assert to_model(box, ImageGeom(1280, 720), cfg) != to_model(box, ImageGeom(720, 1280), cfg)


def test_float_왕복은_양자화_없으면_정확하다():
    """양자화를 빼면 왕복이 부동소수 오차 수준으로 정확해야 한다 — 산식 자체의 검증."""
    for space, geom in [
        ("ABS_ORIG", ImageGeom(1234, 707)),
        ("NORM_1000", ImageGeom(1234, 707)),
        ("ABS_RESIZED", ImageGeom(1234, 707, resized_w=1232, resized_h=700)),
    ]:
        cfg = CoordCfg(coord_space=space)
        assert roundtrip_error_px([13.7, 41.2, 998.3, 611.9], geom, cfg, quantize_model=False) < 1e-9


def test_quantize는_banker_rounding이_아니다():
    """내장 round 는 0.5 를 짝수로 보낸다(round(0.5)==0, round(2.5)==2). 우리는 floor(v+0.5)."""
    assert quantize([0.5, 1.5, 2.5, 3.5]) == (1, 2, 3, 4)
    assert quantize([-0.5, -1.5, 0.4999, 0.5001]) == (0, -1, 0, 1)


# --------------------------------------------------------------------------
# 3. 정책 층 — 판정하되 고치지 않는다
# --------------------------------------------------------------------------

def test_퇴화_판정():
    assert is_degenerate([10, 10, 10, 20]) is True   # x 폭 0
    assert is_degenerate([10, 10, 20, 10]) is True   # y 높이 0
    assert is_degenerate([20, 10, 10, 20]) is True   # 순서 뒤바뀜
    assert is_degenerate([10, 10, 11, 11]) is False


@pytest.mark.parametrize(
    "box,expect_prefix",
    [
        ([10, 10, 20, 20], None),
        ([20, 10, 10, 20], "bbox_invalid"),          # 순서 뒤바뀜 — 자동 스왑하지 않는다
        ([10, 10, 10, 20], "bbox_invalid"),          # 면적 0
        ([-500, -500, -100, -100], "bbox_invalid"),  # 이미지와 교집합 없음
        ([-5, 10, 20, 20], "out_of_bounds"),         # 부분 이탈 — 보고만 한다
        ([10, 10, 20], "bbox_invalid"),              # 원소 3개
        ([10, 10, float("nan"), 20], "bbox_invalid"),
        ([10, 10, float("inf"), 20], "bbox_invalid"),
    ],
)
def test_validate_box_px(box, expect_prefix):
    geom = ImageGeom(100, 100)
    result = validate_box_px(box, geom)
    if expect_prefix is None:
        assert result is None
    else:
        assert result is not None and result.startswith(expect_prefix)


def test_validate는_입력을_바꾸지_않는다():
    box = [-5, 10, 20, 20]
    before = list(box)
    validate_box_px(box, ImageGeom(100, 100))
    assert box == before


def test_snap_은_허용치_이내에서만_동작한다():
    geom = ImageGeom(100, 100)

    snapped, did = snap_to_bounds([-0.4, 10, 20, 100.6], geom, tol=1.0)
    assert did is True
    assert snapped == (0.0, 10.0, 20.0, 100.0)

    # 허용치를 넘는 이탈은 손대지 않는다 — 클램프는 IoU 를 올리는 방향으로만 작동한다
    untouched, did2 = snap_to_bounds([-5.0, 10, 20, 120.0], geom, tol=1.0)
    assert did2 is False
    assert untouched == (-5.0, 10.0, 20.0, 120.0)


def test_snap_후_퇴화는_호출자가_잡을_수_있다():
    """스냅이 박스를 뭉갤 수 있다. 모듈은 살리지 않고, 호출자가 재검사하도록 신호만 준다."""
    geom = ImageGeom(100, 100)
    snapped, did = snap_to_bounds([-0.5, 10, 0.0, 20], geom, tol=1.0)
    assert did is True
    assert is_degenerate(snapped) is True


def test_bool은_좌표로_받지_않는다():
    """파이썬에서 bool 은 int 의 하위형이라 조용히 0/1 로 섞여 들어간다."""
    with pytest.raises(CoordError):
        quantize([True, 10, 20, 20])


def test_이미지_크기는_양수여야_한다():
    with pytest.raises(CoordError):
        ImageGeom(0, 100)
    with pytest.raises(CoordError):
        ImageGeom(100, 100, resized_w=0, resized_h=10)
