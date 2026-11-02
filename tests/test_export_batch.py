"""묶음 생성 1 판(`_workspace/2026-10-29-batchgen/MINISPEC_batch_generation.md` §5) — export 의 묶음 경로 · 모델 생성기의 묶음 ·
곁 파일의 지연 정의 · 내용 해시. **합성 대역**으로 본다. 1 장 대 묶음의 실물 동치는 `scripts/probe/batch_equivalence.py` 가 GPU 에서 잰다.

지키는 것.

1. 배치 1 의 모델 생성기는 묶음 생성 전 모듈(96f9a699)과 **같은 입력에서 같은 값**을 낸다(R-B1).
2. 묶음 경로의 줄은 `batch_n` · `latency_ms` 를 빼면 단일 경로의 줄과 같은 바이트고, 내용 해시가 같다(R-B2).
3. 묶음의 경계는 목록의 절대 위치다 — 계획된 멈춤 · 재개 · 찢긴 줄 뒤에도 경계로 돌아온다(R-B3 ~ R-B5).
4. 곁 파일의 `latency_basis` 는 모델 · 배치 > 1 에서만 `per_call`, 에코는 단일 경로다(R-B6). 거부는 사유 코드로(R-B7).
5. 묶음에서 먼저 끝난 행은 첫 종료 토큰에서 끊고 pad 를 세지 않는다(R-B8).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.content_key import generations_content_sha256
from tests import uni_export_world as W

BEFORE = "96f9a699"
"""묶음 생성 전 — 판 6 연결 묶음 2 병합 커밋. 모델 생성기(`vlm/export_model.py`)의 옛 바이트를 여기서 꺼낸다."""
SEVEN = [f"aihub00000{i}" for i in range(1, 8)]
STEM = "uni_local_C1_s1"


class BatchGen:
    """세계의 대역 생성기에 묶음 경로를 더한다 — 호출마다 묶음 크기를 기록한다. 묶음 줄의 지연은 호출 하나의 값(같은 값)이다."""

    def __init__(self, text_of=None):
        self.single = W.gen(text_of)
        self.calls: list[int] = []

    def __call__(self, img, image_id):
        self.calls.append(1)
        return self.single(img, image_id)

    def generate_batch(self, items):
        items = list(items)
        self.calls.append(len(items))
        return [replace(self.single(img, iid), batch_n=len(items), latency_ms=100.0 * len(items)) for img, iid in items]


def _rows(w) -> list[dict]:
    return [json.loads(x) for x in (w.root / "export" / f"{STEM}.generations.jsonl").read_text(encoding="utf-8").splitlines()]


def _raw(w) -> bytes:
    return (w.root / "export" / f"{STEM}.generations.jsonl").read_bytes()


def _reasons(vb) -> list[str]:
    return list(vb.notes["attempts_reasons"])


# ------------------------------------------------------------------ R-B2 · R-B6
def test_batch_rows_equal_single_rows_on_fakes(tmp_path):
    w1 = W.build(tmp_path / "one", ids=SEVEN)
    w3 = W.build(tmp_path / "three", ids=SEVEN, batch_size=3)
    g1, g3 = BatchGen(), BatchGen()
    r1 = W.run(w1, loader=W.Loader(g1))
    r3 = W.run(w3, loader=W.Loader(g3))
    assert g1.calls == [1] * 7 and g3.calls == [3, 3, 1]                     # 배치 1 은 단일 경로(generate_batch 를 부르지 않는다)
    def strip(r):
        return {k: v for k, v in r.items() if k not in ("batch_n", "latency_ms")}

    assert [strip(r) for r in _rows(w1)] == [strip(r) for r in _rows(w3)]
    assert generations_content_sha256(_raw(w1)) == generations_content_sha256(_raw(w3))   # 내용 해시 같음(판정 한 번 규칙)
    assert [r["batch_n"] for r in _rows(w3)] == [3, 3, 3, 3, 3, 3, 1] and all("batch_n" not in r for r in _rows(w1))
    assert [r["latency_ms"] for r in _rows(w3)] == [300.0] * 6 + [100.0]
    assert r3.meta["latency_basis"] == "per_call" and r1.meta["latency_basis"] == "per_image"
    assert r3.meta["batch_size"] == 3 and r1.meta["batch_size"] == 1          # 등록값 옮겨 싣기 — 평가 쪽 대조 칸
    assert _reasons(W.verify(w3)) == ["first_pass"] and _reasons(W.verify(w1)) == ["first_pass"]   # 평가 쪽 검증기가 받는다


def test_batch_echo_mode_stays_single_path(tmp_path):
    w3 = W.build(tmp_path, ids=SEVEN, batch_size=3)
    g = BatchGen()
    res = W.run(w3, mode="echo", loader=W.Loader(g))
    assert g.calls == [1] * 7 and res.meta["latency_basis"] == "per_image" and res.meta["batch_size"] == 3
    rows = [json.loads(x) for x in (w3.root / "export_echo" / f"{STEM}.echo.generations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("batch_n" not in r for r in rows)
    assert _reasons(W.verify(w3, mode="echo")) == ["first_pass"]


# ------------------------------------------------------------------ R-B3 · R-B4
def test_batch_boundaries_absolute_and_resume_partial(tmp_path):
    w = W.build(tmp_path, ids=SEVEN, batch_size=3)
    g, clock = BatchGen(), W.Clock()
    first = W.run(w, loader=W.Loader(g), clock=clock, stop_after_lines=2)   # 계획된 멈춤 — 묶음을 2 로 자른다(남는 생성 없음)
    assert not first.sealed and first.n_written == 2 and g.calls == [2]
    second = W.run(w, loader=W.Loader(g), clock=clock)
    assert second.sealed and second.plan.reason == "resume_after_interruption" and second.plan.first_id == SEVEN[2]
    assert g.calls == [2, 1, 3, 1]                                          # 재개 첫 묶음은 경계(3)까지 1 장, 그 뒤 3 · 마지막 1
    assert [r["batch_n"] for r in _rows(w)] == [2, 2, 1, 3, 3, 3, 1]
    assert _reasons(W.verify(w)) == ["first_pass", "resume_after_interruption"]


def test_batch_stop_after_lines_no_surplus(tmp_path):
    w = W.build(tmp_path, ids=SEVEN, batch_size=3)
    g = BatchGen()
    res = W.run(w, loader=W.Loader(g), stop_after_lines=4)
    assert res.n_written == 4 and g.calls == [3, 1]                          # 4 줄 — 3 + 1, 다섯째 장을 생성하지 않는다
    assert [r["batch_n"] for r in _rows(w)] == [3, 3, 3, 1]


# ------------------------------------------------------------------ R-B5
class _Exited(BaseException):
    pass


@pytest.fixture
def no_exit(monkeypatch):
    from vlm import fault as F

    def die(code):
        raise _Exited(code)

    monkeypatch.setattr(F.os, "_exit", die)


def test_batch_torn_line_inside_batch_returns_to_boundary(tmp_path, no_exit):
    from vlm.fault import write_planned

    w = W.build(tmp_path, ids=SEVEN, batch_size=3)
    g, clock = BatchGen(), W.Clock()
    fault, nonce = "export:torn_line:uni_local_C1:n=2", "1" * 32
    write_planned(w.root, nonce, fault, stage_no=3)
    with pytest.raises(_Exited):
        W.run(w, loader=W.Loader(g), clock=clock, env={"WELD_REHEARSAL_FAULT": fault, "WELD_REHEARSAL_NONCE": nonce})
    raw = _raw(w)
    assert raw.count(b"\n") == 1 and raw[raw.rfind(b"\n") + 1:] and g.calls == [3]   # 첫 줄 완결 · 둘째 줄 앞 절반 · 셋째 줄 없음
    res = W.run(w, loader=W.Loader(g), clock=clock)
    assert res.plan.reason == "retry_torn_tail" and res.plan.truncated_ids == (SEVEN[1],)
    assert g.calls == [3, 2, 3, 1]                                          # 둘째 id 부터 경계까지 2 장 — 그 뒤 3 · 1
    assert [r["batch_n"] for r in _rows(w)] == [3, 2, 2, 3, 3, 3, 1]
    assert _reasons(W.verify(w)) == ["first_pass", "retry_torn_tail"]


# ------------------------------------------------------------------ R-B7
def test_batch_export_refuses_generator_without_batch_path(tmp_path):
    from vlm.export_run import ExportRefused

    w = W.build(tmp_path, ids=SEVEN, batch_size=3)
    with pytest.raises(ExportRefused) as exc:
        W.run(w, loader=W.Loader(W.gen()))                                  # 단일 생성기만 있다
    assert exc.value.code == "batch_unsupported"
    assert not (w.root / "export" / f"{STEM}.generations.jsonl").exists()   # 생성 파일을 열기 전에 멈춘다


# ------------------------------------------------------------------ R-B1 · R-B8 — 모델 생성기(대역 모델)
def _old_export_model(tmp: Path):
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BEFORE}:vlm/export_model.py"], capture_output=True, check=False)
    if src.returncode != 0:
        pytest.skip(f"이 체크아웃에 {BEFORE} 가 없다")
    p = tmp / "export_model_before.py"
    p.write_bytes(src.stdout)
    spec = importlib.util.spec_from_file_location("export_model_before", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_model_before"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("export_model_before", None)
    return mod


def _model_world():
    from tests import test_uni_export_model as M

    return M


@pytest.fixture
def img():
    from PIL import Image

    return Image.new("L", (1280, 720), 40)


def test_batch_single_call_output_identical_to_legacy(tmp_path, img):
    from vlm.export_model import load_model_generator

    M = _model_world()
    M.adapter_npz(tmp_path / "adapter_last.npz")
    old = _old_export_model(tmp_path)
    for over in ({}, {"max_new_tokens": 8}, {"max_new_tokens": len(M.TARGET)}):
        a = old.load_model_generator(M.cfg(tmp_path, **over), model_loader=M.Loader())(img, "aihub000001")
        b = load_model_generator(M.cfg(tmp_path, **over), model_loader=M.Loader())(img, "aihub000001")
        for f in ("text", "gen_stop", "n_new_tokens", "model_input_wh", "grid_thw", "token_ids", "token_logprobs"):
            assert getattr(a, f) == getattr(b, f), (over, f)
        assert a.batch_n is None and b.batch_n == 1
    short_old = old.load_model_generator(M.cfg(tmp_path, max_new_tokens=40), model_loader=M.Loader(short=True, eos=False))(img, "a")
    short_new = load_model_generator(M.cfg(tmp_path, max_new_tokens=40), model_loader=M.Loader(short=True, eos=False))(img, "a")
    assert (short_old.gen_stop, short_old.n_new_tokens) == (short_new.gen_stop, short_new.n_new_tokens) == ("stop_undetermined", 20)


def test_batch_eos_cut_and_pad_rows(tmp_path, img):
    from vlm.export_model import load_model_generator

    M = _model_world()
    M.adapter_npz(tmp_path / "adapter_last.npz")
    loader = M.Loader(scripts=["ab", M.TARGET, "xyzw"])
    g = load_model_generator(M.cfg(tmp_path, batch_size=3, max_new_tokens=8), model_loader=loader)
    assert g.batch_size == 3 and callable(g.generate_batch)
    outs = g.generate_batch([(img, "a"), (img, "b"), (img, "c")])
    assert [o.gen_stop for o in outs] == ["eos", "length", "eos"]
    assert [o.n_new_tokens for o in outs] == [3, 8, 5]                      # 'ab'+eos · 한도 8 · 'xyzw'+eos — pad 는 세지 않는다
    assert [o.text for o in outs] == ["ab", M.TARGET[:8], "xyzw"]
    assert all(len(o.token_ids) == len(o.token_logprobs) == o.n_new_tokens for o in outs)
    assert all(o.batch_n == 3 for o in outs) and len({o.latency_ms for o in outs}) == 1   # 호출 하나의 지연을 같이 쓴다
    assert all(o.grid_thw == (1, 45, 80) and o.model_input_wh == (1280, 720) for o in outs)
    call, = loader.model.calls
    assert call["batch"] == 3 and call["do_sample"] is False and call["num_beams"] == 1 and call["max_new_tokens"] == 8
    assert loader.model.tok.padding_side == "left"
    with pytest.raises(ValueError, match="묶음은 1 ~ 3"):
        g.generate_batch([(img, str(i)) for i in range(4)])
    # 단일 호출은 묶음 경로의 1 장과 같다
    one = g(img, "a")
    assert one.batch_n == 1 and one.gen_stop == "eos" and one.n_new_tokens == 3
