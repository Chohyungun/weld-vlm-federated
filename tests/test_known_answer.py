"""M1 기지답(`evaluation/known_answer.py`) — 커밋한 합성 박스 픽스처의 손 계수가 공통 채점기와 같고, 계수표 sha 가 픽스처 한 상자에도 갈린다."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import known_answer as KA

RAW = (ROOT / KA.FIXTURE_REL).read_bytes()


def test_손으로_센_기대가_공통_채점기의_계수표와_같다() -> None:
    fx = KA.read_fixture(RAW)
    table = KA.count_table(fx)
    assert KA.expected_mismatches(fx, table) == []
    assert int(table.boundary.sum()) == 1 and int(table.tp.sum()) == 5            # 경계 재판정 한 쌍 · 교차의 최대 짝 수 둘


def test_계수표_바이트는_정해진_꼴이고_sha_는_그_바이트다() -> None:
    raw = KA.table_bytes(KA.count_table(KA.read_fixture(RAW)))
    obj = json.loads(raw)
    assert set(obj) == {"kind", "classes", "image_ids", "tp", "fp", "fn", "boundary", "pred_per_image"}
    assert obj["kind"] == KA.TABLE_KIND and obj["image_ids"] == sorted(obj["image_ids"]) and not raw.endswith(b"\n")
    assert obj["classes"] == KA.read_fixture(RAW)["classes"] and all(len(r) == 4 for r in obj["tp"])
    assert raw == json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert KA.known_answer_sha256(RAW) == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(("where", "edit"), [
    ("ka-02", lambda im: im["pred"][0][1].__setitem__(3, 21)),                     # 경계 맞춤 → 1/2 미만
    ("ka-08", lambda im: im["pred"].pop()),                                        # 교차의 예측 하나를 뺀다
    ("ka-06", lambda im: im["pred"][0].__setitem__(0, "301")),                     # 오탐의 클래스
])
def test_픽스처의_상자_하나를_바꾸면_sha_가_갈린다(where, edit) -> None:
    fx = json.loads(RAW)
    other = copy.deepcopy(fx)
    edit(next(im for im in other["images"] if im["image_id"] == where))
    assert KA.known_answer_sha256(json.dumps(other).encode("utf-8")) != KA.known_answer_sha256(RAW)


@pytest.mark.parametrize("edit", [
    lambda o: o.update(kind="other"),
    lambda o: o["images"].append(dict(o["images"][0])),
    lambda o: o.update(images=[]),
])
def test_꼴이_틀린_픽스처는_거부한다(edit) -> None:
    o = json.loads(RAW)
    edit(o)
    with pytest.raises(KA.KnownAnswerError):
        KA.read_fixture(json.dumps(o).encode("utf-8"))
