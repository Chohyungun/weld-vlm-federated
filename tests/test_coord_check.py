"""평가 쪽 독립 역변환 · 골든 픽스처 지문 · 통합형 좌표 설정 기록 — 07번 §31-6 · 진입점 미니스펙 3판 8-5.

지키는 것.

1. 역변환은 규약 셋을 따로 짠다 — **평가 모듈은 `vlm.coords` 를 가져오지 않는다**(구문 트리로 본다).
2. 두 구현(평가 쪽 `inverse_px` · 쓰는 쪽 `vlm.coords.to_px`)이 골든 픽스처 열두 사례에서 같은 값을 낸다 — 이 시험만 쓰는 쪽을 가져온다.
3. 지문은 손계산 기대값과 맞은 뒤에만 나온다. 기대값 하나가 틀리면 그 사례 id 를 들고 멈춘다.
4. 같은 바이트면 같은 지문, 사례 id 나 규약이 바뀌면 다른 지문.
5. 통합형 좌표 설정 기록은 좌표 코드의 바이트가 이력의 마지막 값과 같을 때만 선다.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.coord_check import FixtureMismatch, coord_fixture_digest, inverse_px, unified_coord_cfg_record

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "vlm/coords_fixtures/golden_fixtures.json"
HASH_RECORD = REPO / "vlm/coords_fixtures/HASH_RECORD.yaml"
COORDS = REPO / "vlm/coords.py"


def test_규약_셋의_역변환() -> None:
    assert inverse_px([10, 20, 30, 40], "ABS_ORIG", orig_wh=(1280, 720)) == (10, 20, 30, 40)
    assert inverse_px([500, 500, 1000, 1000], "NORM_1000", orig_wh=(1280, 720)) == (640.0, 360.0, 1280.0, 720.0)
    assert inverse_px([0, 0, 1280, 704], "ABS_RESIZED", orig_wh=(1280, 720), model_wh=(1280, 704)) == \
        (0.0, 0.0, 1280.0, 720.0)


@pytest.mark.parametrize(("box", "space", "kw"), [
    ([1, 2, 3], "ABS_ORIG", {}), ([1, 2, 3, float("nan")], "ABS_ORIG", {}), ([1, 2, 3, True], "ABS_ORIG", {}),
    ([1, 2, 3, 4], "NORM_100", {}), ([1, 2, 3, 4], "ABS_RESIZED", {}),
])
def test_꼴이_틀린_입력은_받지_않는다(box, space: str, kw: dict) -> None:
    with pytest.raises(ValueError):
        inverse_px(box, space, orig_wh=(1280, 720), **kw)


def test_평가_모듈은_쓰는_쪽_좌표_코드를_가져오지_않는다() -> None:
    tree = ast.parse((REPO / "evaluation/coord_check.py").read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
           {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in mods if m and m.startswith("vlm")}


def test_두_구현이_골든_픽스처에서_같은_값을_낸다() -> None:
    coords = pytest.importorskip("vlm.coords")
    doc = json.loads(FIXTURE.read_bytes())
    assert len(doc["cases"]) == 12
    for case in doc["cases"]:
        space, geom = case["cfg"]["coord_space"], case["geom"]
        model_wh = (geom["resized_w"], geom["resized_h"]) if space == "ABS_RESIZED" else None
        mine = inverse_px(case["expect"]["model_quantized"], space,
                          orig_wh=(geom["orig_w"], geom["orig_h"]), model_wh=model_wh)
        theirs = coords.to_px(case["expect"]["model_quantized"],
                              coords.ImageGeom(**geom), coords.CoordCfg(coord_space=space))
        assert mine == pytest.approx(theirs, abs=1e-9), case["id"]


def test_골든_픽스처가_지문을_낸다() -> None:
    raw = FIXTURE.read_bytes()
    d = coord_fixture_digest(raw)
    assert len(d) == 64 and d == coord_fixture_digest(raw)


def _mutated(edit) -> bytes:
    doc = json.loads(FIXTURE.read_bytes())
    edit(doc)
    return json.dumps(doc, ensure_ascii=False).encode("utf-8")


def test_기대값이_틀리면_그_사례를_들고_멈춘다() -> None:
    def edit(doc):
        doc["cases"][2]["expect"]["back_px"][0] += 0.01
    with pytest.raises(FixtureMismatch) as exc:
        coord_fixture_digest(_mutated(edit))
    assert exc.value.case_ids == [json.loads(FIXTURE.read_bytes())["cases"][2]["id"]]


def test_사례_id_가_바뀌면_지문이_바뀐다() -> None:
    def edit(doc):
        doc["cases"][0]["id"] = "renamed"
    assert coord_fixture_digest(_mutated(edit)) != coord_fixture_digest(FIXTURE.read_bytes())


def test_허용치_안의_끝자리_흔들림은_지문을_가르지_않는다() -> None:
    def edit(doc):
        doc["cases"][0]["expect"]["back_px"][0] += 1e-12
    assert coord_fixture_digest(_mutated(edit)) == coord_fixture_digest(FIXTURE.read_bytes())


def test_통합형_좌표_설정_기록은_첫_등록이다() -> None:
    rec = unified_coord_cfg_record("a" * 64, fixture_digest="f" * 64, registered_on="2026-10-01",
                                   coords_source=COORDS.read_bytes(), hash_record=HASH_RECORD.read_bytes())
    assert rec.accepted == ("a" * 64,) and rec.frozen_value == rec.current_value == "a" * 64
    assert rec.change_scope == "first_registration"
    assert any("첫 등록" in e for e in rec.equivalence_evidence)


def test_좌표_코드가_이력과_다르면_기록을_만들지_않는다() -> None:
    with pytest.raises(ValueError, match="HASH_RECORD"):
        unified_coord_cfg_record("a" * 64, fixture_digest="f" * 64, registered_on="2026-10-01",
                                 coords_source=COORDS.read_bytes() + b"# x\n", hash_record=HASH_RECORD.read_bytes())


def test_이력의_마지막_값은_지금_좌표_코드의_바이트다() -> None:
    """이력을 읽는 규칙이 지금 파일에서 서는지 — 값은 파일에서 센다."""
    now = hashlib.sha256(COORDS.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert now in HASH_RECORD.read_text(encoding="utf-8")
