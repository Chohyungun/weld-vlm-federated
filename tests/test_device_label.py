"""장비 문자열의 꼴 — `evaluation/bundle_v14.py` 의 `DEVICE_LABEL_NONCANONICAL`(2026-10-18).

정본 꼴 `"{이름}:{인덱스}"`(§28-3 나)은 앞 판 그대로 받는다. 정본 꼴이 아니어도 비지 않은 문자열이고 도장 · 모든 시작 줄 · 곁 파일에서
한 문자열이면 거부하지 않고 표지를 싣는다. 자리마다 어긋나거나 · 정본 꼴과 섞이거나 · 비었으면 거부한다. 묶음 사이에서도 한 문자열이어야 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.bundle_v14 import (
    DEVICE_LABEL_CANONICAL,
    DEVICE_LABEL_NONCANONICAL,
    EVENT_START,
    REASON_RESUME,
    check_env_consistency,
)
from evaluation.reject_v14 import RejectCode
from tests.test_bundle_v14 import DEVICE, make, nine

LABEL = "A100-80GB-PCIe-single"
OTHER = "A100-80GB-PCIe-other"
TWO = [(0, None, 4), (4, REASON_RESUME, 5)]


def _rows(*labels):
    """시작 줄의 장비를 시도 순서대로 바꾼다 — 곁 파일은 마지막 시작 줄에서 옮겨 적는다(`make`)."""
    def edit(rows):
        starts = [r for r in rows if r["event"] == EVENT_START]
        for r, label in zip(starts, labels * len(starts) if len(labels) == 1 else labels, strict=False):
            r["device"] = label
        return rows
    return edit


def _same(tmp: Path, label, plan=((0, None, 9),), **kw):
    """도장 · 모든 시작 줄 · 곁 파일이 한 문자열인 묶음."""
    return nine(tmp, list(plan), start_over={"device": label}, attempts_edit=_rows(label), **kw)


def test_정본_꼴은_앞_판처럼_받고_표지가_없다(tmp_path: Path) -> None:
    got = nine(tmp_path, TWO).verify()
    assert "device_label" not in got.notes
    assert list(got.notes["devices"]) == [DEVICE]


@pytest.mark.parametrize("label", [LABEL, "0", "cuda"])
@pytest.mark.parametrize("plan", [[(0, None, 9)], TWO])
def test_정본_꼴이_아니어도_한_문자열이면_받고_표지를_싣는다(tmp_path: Path, label, plan) -> None:
    got = _same(tmp_path, label, plan).verify()
    assert got.notes["device_label"] == {"status": DEVICE_LABEL_NONCANONICAL, "value": label}
    assert list(got.notes["devices"]) == [label] and got.notes["devices_spanned"] is False


def test_에코_묶음도_같은_규칙이다(tmp_path: Path) -> None:
    got = make(tmp_path, echo=True, start_over={"device": LABEL}, attempts_edit=_rows(LABEL)).verify()
    assert got.notes["device_label"] == {"status": DEVICE_LABEL_NONCANONICAL, "value": LABEL}


@pytest.mark.parametrize(("kw", "want"), [
    ({"start_over": {"device": OTHER}, "attempts_edit": _rows(LABEL)}, {RejectCode.START_STAMP_SCHEMA}),
    ({"start_over": {"device": LABEL}, "attempts_edit": _rows(LABEL), "meta_over": {"device": OTHER}},
     {RejectCode.START_STAMP_SCHEMA, RejectCode.ATTEMPTS_SCHEMA}),
    ({"start_over": {"device": LABEL}, "attempts_edit": _rows(LABEL, OTHER)},
     {RejectCode.START_STAMP_SCHEMA, RejectCode.ATTEMPTS_SCHEMA}),
    ({"start_over": {"device": DEVICE}, "attempts_edit": _rows(DEVICE, LABEL)},            # 정본 꼴 → 아닌 것
     {RejectCode.START_STAMP_SCHEMA, RejectCode.ATTEMPTS_SCHEMA}),
    ({"start_over": {"device": LABEL}, "attempts_edit": _rows(LABEL, DEVICE)},             # 아닌 것 → 정본 꼴
     {RejectCode.START_STAMP_SCHEMA, RejectCode.ATTEMPTS_SCHEMA}),
])
def test_정본_꼴이_아닌_문자열이_자리마다_다르거나_정본_꼴과_섞이면_거부한다(tmp_path: Path, kw, want) -> None:
    got = nine(tmp_path, TWO, **kw).rejects()
    assert want <= got


@pytest.mark.parametrize("bad", ["", "   ", None, 0])
def test_빈_값과_문자열이_아닌_값은_모든_자리가_같아도_거부한다(tmp_path: Path, bad) -> None:
    got = _same(tmp_path, bad, TWO).rejects()
    assert {RejectCode.START_STAMP_SCHEMA, RejectCode.ATTEMPTS_SCHEMA} <= got


def test_곁_파일만_비면_마지막_시작_줄과의_대조가_거부한다(tmp_path: Path) -> None:
    f = nine(tmp_path, TWO, meta_over={"device": ""})
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()
    assert f.detail()["item"] == "device"


def _env_device(bundles) -> list[dict]:
    return [r.detail for r in check_env_consistency(bundles) if r.detail.get("item") == "device"]


def test_묶음_사이에서도_정본_꼴이_아닌_문자열은_같아야_한다(tmp_path: Path) -> None:
    a = _same(tmp_path / "a", LABEL, tag="uni_local_C1").verify()
    b = _same(tmp_path / "b", LABEL, tag="uni_local_C3").verify()
    assert check_env_consistency([a, b]) == []
    c = _same(tmp_path / "c", OTHER, tag="uni_local_C3").verify()
    d = nine(tmp_path / "d", TWO, tag="uni_local_C3").verify()                          # 정본 꼴과 섞임
    for other in (c, d):
        (det,) = _env_device([a, other])
        assert det["n_distinct"] == 2
    assert RejectCode.ENV_MISMATCH in {r.code for r in check_env_consistency([a, c])}


def test_정본_꼴_묶음끼리는_장비가_달라도_앞_판처럼_거부하지_않는다(tmp_path: Path) -> None:
    a = nine(tmp_path / "a", TWO, tag="uni_local_C1").verify()
    b = _same(tmp_path / "b", "NVIDIA A100-SXM4-40GB:0", TWO, tag="uni_local_C3").verify()
    assert "device_label" not in b.notes
    assert check_env_consistency([a, b]) == []


def test_엄격_산출의_장비_표지는_문자열과_묶음을_싣는다() -> None:
    from scripts.probe.score_unified import _device_label

    def vb(stem, label=None):
        notes = {} if label is None else {"device_label": {"status": DEVICE_LABEL_NONCANONICAL, "value": label}}
        return SimpleNamespace(files=SimpleNamespace(stem=stem), notes=notes)

    assert _device_label([vb("uni_central_s1"), vb("uni_central_s1.echo")]) == {"status": DEVICE_LABEL_CANONICAL}
    got = _device_label([vb("uni_local_C3_s1", LABEL), vb("uni_central_s1", LABEL), vb("uni_central_s1.echo", LABEL)])
    assert got["status"] == DEVICE_LABEL_NONCANONICAL == "DEVICE_LABEL_NONCANONICAL"
    assert got["values"] == [LABEL] and got["bundles"] == ["uni_central_s1", "uni_central_s1.echo", "uni_local_C3_s1"]
