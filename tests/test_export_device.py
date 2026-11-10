"""export 의 장비 문자열 꼴(2026-10-18 다섯 칸 평가 사고 — 설계 결정 기록 2026-10-18 22:2x). 모델 · GPU 없이.

본다: 장비 꼴 판정이 채점기(`evaluation.bundle_v14._device_ok`)와 같은지, 새 시작은 그 꼴만 받는지(`export_device_form` — 측정 · 사전
점검 전에, 도장 · 시작 줄을 쓰기 전에, `--abort-mismatched` 면 옛 파일을 옮기기 전에), 재개는 남은 도장(= 첫 시작 줄)을 따르는지
(정본 도장이면 정본만 — 다른 장비로 이어 가기는 채점기가 적는다, 비정본 도장이면 그 문자열을 그대로 잇는다).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import bundle_v14 as BV
from tests import uni_export_world as W
from tests.test_uni_export import _elist, _stamp_fields
from vlm import export_run as XR
from vlm import export_writer as XW
from vlm.export_writer import ExportRefused, ExportWriter

#: 2026-10-18 서버 운영 wrapper 가 넘긴 값 — 채점기의 꼴({이름}:{인덱스})이 아니다.
NONCANON = "A100-80GB-PCIe-single"


@pytest.fixture
def w(tmp_path):
    return W.build(tmp_path)


def _folder(w) -> Path:
    return w.root / "export"


def _starts(w) -> list[dict]:
    rows = [json.loads(ln) for ln in (_folder(w) / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r["event"] == "start"]


def _stamp(w) -> dict:
    return json.loads((_folder(w) / "uni_local_C1_s1.export_start.json").read_text(encoding="utf-8"))


def _code(fn) -> str:
    with pytest.raises(ExportRefused) as exc:
        fn()
    return exc.value.code


def test_장비_꼴_판정은_채점기와_같다():
    for v in (W.DEVICE, "cuda:0", "a:b:3", "NVIDIA A100 80GB PCIe:12", "cpu", "cuda", NONCANON, ":0", "x:", "x:y",
              "x:0:", "", None, 0, ["x:0"]):
        assert XW.device_form_ok(v) == BV._device_ok(v), v
    assert XW.device_form_ok(W.DEVICE) and not XW.device_form_ok(NONCANON)


def test_정본_장비는_새_시작부터_봉인까지_채점기를_지난다(w):
    res = W.run(w)
    assert res.sealed and _stamp(w)["device"] == W.DEVICE
    assert list(W.verify(w).notes["devices"]) == [W.DEVICE]


def test_비정본_장비의_새_시작은_측정_전에_거부되고_아무것도_쓰지_않는다(w):
    calls = []

    def spy(spec):
        calls.append(spec)
        return W.measure(spec)

    assert _code(lambda: W.run(w, device=NONCANON, measure_fn=spy)) == "export_device_form"
    assert calls == [] and not _folder(w).exists()
    # 쓰기 객체 단독 — 실행기의 앞 검사를 거치지 않아도 도장을 쓰기 전에 멈춘다
    wr = ExportWriter(w.root / "solo", tag="uni_local_C1", seed_index=1, mode="model", eval_list=_elist(),
                      stamp_fields=_stamp_fields(), device=NONCANON, clock=W.Clock())
    assert _code(wr.begin) == "export_device_form" and not (w.root / "solo").exists()


def test_비정본으로_시작한_묶음의_재개는_도장의_문자열을_잇는다(w, monkeypatch):
    clock = W.Clock()
    with monkeypatch.context() as m:                             # 꼴 검사 전 코드로 시작한 묶음(2026-10-18 서버)을 만든다
        m.setattr(XW, "check_new_device", lambda device: device)
        m.setattr(XR, "check_new_device", lambda device: device)
        assert not W.run(w, device=NONCANON, clock=clock, stop_after_lines=1).sealed
    assert _stamp(w)["device"] == NONCANON
    assert not W.run(w, device=NONCANON, clock=clock, stop_after_lines=1).sealed           # 같은 문자열
    res = W.run(w, device="NVIDIA A100 80GB PCIe:1", clock=clock)                             # 다른 값을 줘도 도장을 잇는다
    assert res.sealed and [r["reason"] for r in _starts(w)] == ["first_pass"] + ["resume_after_interruption"] * 2
    assert {r["device"] for r in _starts(w)} == {NONCANON}
    assert res.meta["device"] == NONCANON                                                     # 곁 파일 = 마지막 시작 줄


def test_정본으로_시작한_묶음의_재개는_정본만_받고_다른_장비는_채점기가_적는다(w):
    clock = W.Clock()
    assert not W.run(w, clock=clock, stop_after_lines=2).sealed
    n = len(_starts(w))
    assert _code(lambda: W.run(w, device=NONCANON, clock=clock)) == "export_device_form"
    assert len(_starts(w)) == n                                                               # 시작 줄을 쓰지 않았다
    other = "NVIDIA GeForce RTX 5060 Ti:1"
    assert W.run(w, device=other, clock=clock).sealed
    vb = W.verify(w)
    assert list(vb.notes["devices"]) == sorted({W.DEVICE, other}) and vb.notes["devices_spanned"] is True


def test_도장이_어긋나_새로_시작하는_명시_인자도_비정본이면_옛_파일을_옮기기_전에_멈춘다(tmp_path):
    kw = {"tag": "uni_local_C1", "seed_index": 1, "mode": "model", "eval_list": _elist(), "clock": W.Clock()}
    ExportWriter(tmp_path / "w", stamp_fields=_stamp_fields(), device=W.DEVICE, **kw).begin()
    before = sorted(p.name for p in (tmp_path / "w").iterdir())
    changed = {**_stamp_fields(), "export_commit": "d" * 40}
    wr = ExportWriter(tmp_path / "w", stamp_fields=changed, device=NONCANON, abort_mismatched=True, **kw)
    assert _code(wr.begin) == "export_device_form"
    assert sorted(p.name for p in (tmp_path / "w").iterdir()) == before                       # .aborted 개명 0


@pytest.mark.parametrize("stamp_device", [None, "", "  ", 3])
def test_남은_도장의_장비가_비었거나_문자열이_아니면_이어_쓰지_않는다(stamp_device):
    assert _code(lambda: XW.resume_device(W.DEVICE, stamp_device)) == "export_device_form"
