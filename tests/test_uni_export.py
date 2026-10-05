"""export(C8) — 파싱과 역변환의 분리, 쓰기 객체의 상태 기계, 시작 절차, **평가 쪽 실제 함수로 하는 사전 점검**, 평가 쪽 검증기
(리허설 2판 §2-1 ~ §2-4 · §3 ③).

사전 점검은 평가 쪽 진입점 조각(등록 로더 · 영수증 종류 · 닻 · 프로세서 인자 · 목록 · 분할 · 실측값 대조 · 원장 판독기)을 그대로 부른다.
합성 세계(`tests/uni_export_world.py`)는 git 저장소 · 동결 스냅샷 · 평가 쪽으로 만든 등록 · 영수증 · 노출 원장 · 학습 산출물이다.
이음새 둘(실측 · 생성기 적재)만 대역이다 — 리허설 목적에서만 받는다. 모델을 올리지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import uni_export_world as W
from vlm import export_run as XR
from vlm import export_writer as XW
from vlm.coords import CoordCfg, ImageGeom
from vlm.export_run import parser_sha256
from vlm.export_writer import ExportRefused, ExportStateError, ExportWriter
from vlm.gen_parse import conformance_tuple, line_fields, pair_ok, parse_generation

KST = W.KST
DEVICE = W.DEVICE
IDS = W.IDS
STEM = "uni_local_C1_s1"


def hx(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture
def w(tmp_path):
    return W.build(tmp_path)


def _folder(w, mode="model") -> Path:
    return w.root / ("export_echo" if mode == "echo" else "export")


def _reasons(vb) -> list[str]:
    return list(vb.notes["attempts_reasons"])


def _rows(w, name: str) -> list[dict]:
    return [json.loads(x) for x in (_folder(w) / name).read_text(encoding="utf-8").splitlines()]


def _snapshot(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in d.iterdir()} if d.exists() else {}


# ================================================================ 1. 파싱과 역변환의 분리
def test_적합성_벡터_판_2_전량을_쓰는_쪽_파서가_지난다():
    from evaluation.conformance_v14 import (
        KNOWN_CODES,
        SCORING_CODES,
        VECTOR_VERSION,
        all_cases,
        check_case,
    )

    assert VECTOR_VERSION == "2"
    bad = [c.name for c in all_cases()
           if not check_case(c, parse=lambda t: conformance_tuple(t, known_iso_codes=KNOWN_CODES,
                                                                 scoring_iso_codes=SCORING_CODES)).passed]
    assert bad == [] and XR.conformance_problems() == []


def test_모든_벡터_사례에서_사유와_결과의_짝이_선다():
    from evaluation.conformance_v14 import all_cases

    for c in all_cases():
        lf = line_fields(parse_generation(c.text), ImageGeom(orig_w=1280, orig_h=720), CoordCfg(coord_space="ABS_ORIG"))
        assert pair_ok(lf["bbox_px_parsed"], lf["parse_error"]), c.name


def test_파싱은_역변환_전_값을_내고_줄은_to_px_한_번으로_돌린다():
    text = '{"defects": [{"iso_code": "2011", "bbox_2d": [100, 200, 300, 400]}], "verdict": "합격"}'
    p = parse_generation(text)
    assert p.items == (("2011", 100.0, 200.0, 300.0, 400.0),) and p.cited is None     # 인용 키가 없으면 None
    geom = ImageGeom(orig_w=1280, orig_h=720)
    abs_ = line_fields(p, geom, CoordCfg(coord_space="ABS_ORIG"))
    assert abs_["bbox_px_parsed"]["defects"][0]["bbox_px"] == [100.0, 200.0, 300.0, 400.0]
    assert abs_["bbox_px_parsed"]["cited_clauses"] is None                           # 메우지 않는다
    norm = line_fields(p, geom, CoordCfg(coord_space="NORM_1000"))
    assert norm["bbox_px_parsed"]["defects"][0]["bbox_px"] == [128.0, 144.0, 384.0, 288.0]
    assert line_fields(parse_generation("산문만"), geom, CoordCfg(coord_space="ABS_ORIG")) == \
        {"bbox_px_parsed": None, "parse_error": "no_json", "n_bad_items_dropped": 0}


def test_파서는_모델_출력의_어떤_꼴에도_예외를_내지_않는다():
    for text in ["[" * 100000, "{" + '"a":' * 50000, '{"defects": [' + "9" * 5000 + "]}",
                 '{"defects": [{"bbox_2d": [1e999, 0, 2, 2]}]}', '{"defects": 3}', "", "{" * 3 + "x"]:
        p = parse_generation(text)
        assert p.parse_error in (None, "no_json", "json_decode", "truncated", "schema_violation")


# ================================================================ 4. 평가 쪽 검증기 — 평가 쪽 투영 · 원장 판독기로
def test_한_번에_끝낸_모델_묶음을_평가_쪽_검증기가_받는다(w):
    res = W.run(w, with_tokens=True)
    assert res.sealed and res.plan.reason == "first_pass" and res.generator_loaded
    vb = W.verify(w)
    assert _reasons(vb) == ["first_pass"] and vb.mode == "model"
    assert vb.notes["stand_in"] is True and list(vb.notes["fault_events"]) == []   # 리허설의 대역 실행으로 적힌다
    meta = res.meta
    assert meta["image_grid_thw_observed"] == {"[1, 44, 80]": len(IDS)} and meta["fault_events"] == []
    assert meta["rehearsal_id"] == W.REH and meta["aborted_stamps"] == 0
    assert set(meta["impl_ids"]) == {"export_measure", "export_generator"}
    assert meta["generation_sha256"] == w.reg.generation_sha256()


def test_에코_묶음을_평가_쪽_검증기가_받는다(w):
    res = W.run(w, mode="echo", loader=W.Loader(W.gen(latency_ms=10_000.0)))
    assert res.sealed
    d = _folder(w, "echo")
    stamp = json.loads((d / f"{STEM}.echo.export_start.json").read_text(encoding="utf-8"))
    assert not {"train_run_id", "train_ledger_sha256", "scored_adapter_sha256"} & set(stamp)
    rows = [json.loads(x) for x in (d / f"{STEM}.echo.generations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["latency_ms"] < 10_000.0 for r in rows)                 # 에코는 파싱 경로만의 시간
    assert W.verify(w, mode="echo").mode == "echo"


def _mixed(image_id):
    return {IDS[0]: '{"defects": [{"iso_code": "2011", "bbox_2d": [10, 20, 40, 55]}], "verdict": "합격", '
                    '"cited_clauses": []}',
            IDS[1]: "산문만 있고 JSON 이 없다",
            IDS[2]: '```json\n{"defects": [{"iso_code": "2011", "bbox_2d": [5, 5, 5, 9]}, '
                    '{"iso_code": "2011", "bbox_2d": [1.5, 2, 30, 40]}], "verdict": "불합격"}\n```',
            IDS[3]: '{"defects": [{"iso_code": "2011", "bbox_2d": [1, 2'}[image_id]


def test_레코드_실패가_섞인_묶음을_검증기와_어댑터가_받는다(w):
    from evaluation.adapters_v14 import adapt_unified_main
    from evaluation.conformance_v14 import KNOWN_CODES, SCORING_CODES

    W.run(w, loader=W.Loader(W.gen(text_of=_mixed)))
    W.verify(w)
    lines = (_folder(w) / f"{STEM}.generations.jsonl").read_bytes().decode("utf-8")[:-1].split("\n")
    rows = [json.loads(x) for x in lines]
    assert [r["parse_error"] for r in rows] == [None, "no_json", None, "truncated"]
    assert rows[2]["n_bad_items_dropped"] == 1 and rows[2]["bbox_px_parsed"]["cited_clauses"] is None
    rep = adapt_unified_main(lines, tag="uni_local_C1", seed=7, coupling_rule="decoupled_v2",
                             known_iso_codes=KNOWN_CODES, scoring_iso_codes=SCORING_CODES)
    assert rep.n_lines == len(IDS) and [r.parse_error for r in rep.records] == [None, "no_json", None, "truncated"]
    assert [[list(d.bbox_px) for d in r.defects] for r in rep.records if r.parse_ok] == \
        [[[10.0, 20.0, 40.0, 55.0]], [[1.5, 2.0, 30.0, 40.0]]]


def test_계획된_멈춤_뒤_이어_쓴_묶음을_받는다(w):
    clock = W.Clock()
    first = W.run(w, clock=clock, stop_after_lines=2)
    assert not first.sealed and first.n_written == 2
    second = W.run(w, clock=clock)
    assert second.plan.reason == "resume_after_interruption" and second.plan.first_id == IDS[2]
    assert second.meta["image_grid_thw_observed"] == {"[1, 44, 80]": len(IDS)}   # 앞 시도의 줄도 센다
    assert _reasons(W.verify(w)) == ["first_pass", "resume_after_interruption"]


def test_찢긴_꼬리는_시작_줄을_먼저_쓰고_바이트를_남긴_뒤_자른다(w):
    clock = W.Clock()
    W.run(w, clock=clock, stop_after_lines=2)
    gen = _folder(w) / f"{STEM}.generations.jsonl"
    whole = gen.read_bytes()
    torn = b'{"image_id": "aihub000003", "text": "\xec\x95\x88'          # UTF-8 다바이트 문자의 가운데에서 찢겼다
    gen.write_bytes(whole + torn)
    res = W.run(w, clock=clock)
    assert res.plan.reason == "retry_torn_tail" and res.plan.truncated_ids == (IDS[2],)
    assert (_folder(w) / f"{STEM}.truncated-2.bin").read_bytes() == torn
    rows = _rows(w, "attempts.jsonl")
    assert rows[2]["event"] == "start" and rows[2]["truncated_ids"] == [IDS[2]] and rows[2]["first_id"] == IDS[2]
    assert gen.read_bytes().startswith(whole)                                  # 완결 줄은 한 바이트도 바뀌지 않았다
    assert _reasons(W.verify(w)) == ["first_pass", "retry_torn_tail"]


def test_자르기_전에_죽으면_다음_시도가_같은_절차를_밟는다(w, monkeypatch):
    clock = W.Clock()
    W.run(w, clock=clock, stop_after_lines=1)
    gen = _folder(w) / f"{STEM}.generations.jsonl"
    gen.write_bytes(gen.read_bytes() + b'{"image_id": "aihub0')
    real = XW._write_all

    def die_on_bin(path, data, mode):
        if str(path).endswith(".bin"):
            raise RuntimeError("잘라 낸 바이트를 남기기 전에 죽었다")
        return real(path, data, mode)

    monkeypatch.setattr(XW, "_write_all", die_on_bin)
    with pytest.raises(RuntimeError):
        W.run(w, clock=clock)
    monkeypatch.setattr(XW, "_write_all", real)
    res = W.run(w, clock=clock)
    assert res.plan.reason == "retry_torn_tail" and res.plan.attempt_no == 3
    assert _reasons(W.verify(w)) == ["first_pass", "retry_torn_tail", "retry_torn_tail"]


def test_잘라_낸_뒤_죽어도_잘라_낸_id_의_기록이_남는다(w, monkeypatch):
    clock = W.Clock()
    W.run(w, clock=clock, stop_after_lines=2)
    gen = _folder(w) / f"{STEM}.generations.jsonl"
    gen.write_bytes(gen.read_bytes() + b'{"image_id": "aihub000003", "te')
    real = ExportWriter._truncate

    def cut_then_die(path, size):
        real(path, size)
        raise RuntimeError("잘라 낸 직후 죽었다")

    monkeypatch.setattr(ExportWriter, "_truncate", staticmethod(cut_then_die))
    with pytest.raises(RuntimeError):
        W.run(w, clock=clock)
    monkeypatch.setattr(ExportWriter, "_truncate", staticmethod(real))
    rows = _rows(w, "attempts.jsonl")
    assert rows[-1]["reason"] == "retry_torn_tail" and rows[-1]["truncated_ids"] == [IDS[2]]
    assert (_folder(w) / f"{STEM}.truncated-2.bin").exists()
    res = W.run(w, clock=clock)
    assert res.plan.reason == "resume_after_interruption"
    assert _reasons(W.verify(w)) == ["first_pass", "retry_torn_tail", "resume_after_interruption"]


def test_생성_줄을_얻지_못한_토큰_한_줄은_잘라_내고_재개로_적는다(w):
    clock = W.Clock()
    W.run(w, clock=clock, stop_after_lines=2, with_tokens=True)
    tok = _folder(w) / f"{STEM}.tokens.jsonl"
    extra = json.dumps({"image_id": IDS[2], "token_ids": [9], "token_logprobs": [0.0]}).encode() + b"\n"
    tok.write_bytes(tok.read_bytes() + extra)
    res = W.run(w, clock=clock, with_tokens=True)
    assert res.plan.reason == "resume_after_interruption" and res.plan.truncated_ids == ()
    assert (_folder(w) / f"{STEM}.tokens.truncated-2.bin").read_bytes() == extra
    assert _reasons(W.verify(w)) == ["first_pass", "resume_after_interruption"]


def test_마지막_줄_뒤_끝_줄_전에_죽은_묶음은_생성기를_올리지_않고_빈_구간으로_닫는다(w, monkeypatch):
    clock = W.Clock()
    real_end = ExportWriter.end

    def die(self):
        raise RuntimeError("끝 줄 전에 죽었다")

    monkeypatch.setattr(ExportWriter, "end", die)
    with pytest.raises(RuntimeError):
        W.run(w, clock=clock)
    monkeypatch.setattr(ExportWriter, "end", real_end)
    loader = W.Loader()
    res = W.run(w, clock=clock, loader=loader)
    assert res.plan.first_id is None and res.n_written == 0 and res.sealed
    assert loader.calls == 0 and not res.generator_loaded
    assert _reasons(W.verify(w)) == ["first_pass", "resume_after_interruption"]


# ================================================================ 2. 시작 절차의 거부
def test_완결_묶음이면_시작_줄을_쓰지_않고_멈춘다(w):
    W.run(w)
    att = _folder(w) / "attempts.jsonl"
    before = att.read_bytes()
    with pytest.raises(ExportRefused) as exc:
        W.run(w)
    assert exc.value.code == "bundle_complete" and att.read_bytes() == before


@pytest.mark.parametrize("broken", [b'{"event": "start"}', b'not json\n', b'{"a": 1}\r\n'])
def test_시도_기록이_깨졌으면_덧붙이지_않고_멈춘다(w, broken):
    att = _folder(w) / "attempts.jsonl"
    att.parent.mkdir(parents=True)
    att.write_bytes(broken)
    with pytest.raises(ExportRefused) as exc:
        W.run(w)
    assert exc.value.code == "attempts_broken" and att.read_bytes() == broken


def _other_commit(w) -> None:
    (w.code / "m.py").write_text("x = 2\n", encoding="utf-8")
    W.git(w.code, "commit", "-q", "-am", "다른 코드")


def _other_registration(w) -> Path:
    def edit(r):
        r.generation.padding_side = "right"          # 실측과 무관한 생성 조건 하나 — 생성 지문만 바뀐다

    w.reg = W.registration(w, edit=edit)
    return W.write_registration(w, w.reg, name="rehearsal-20261001-2")


@pytest.mark.parametrize("how", ["export_commit", "parser_sha256", "generation_sha256"])
def test_지문이_다르면_이어_쓰지_않고_명시_인자로만_개명한다(w, monkeypatch, how):
    clock = W.Clock()
    W.run(w, clock=clock, stop_after_lines=1)
    d = _folder(w)
    before = _snapshot(d)
    over = {}
    if how == "export_commit":
        _other_commit(w)
    elif how == "parser_sha256":
        monkeypatch.setattr(XR, "parser_sha256", lambda: "e" * 64)
    else:
        over = {"receipt_path": _other_registration(w)}
    with pytest.raises(ExportRefused) as exc:
        W.run(w, clock=clock, **over)
    assert exc.value.code == "stamp_mismatch" and _snapshot(d) == before
    res = W.run(w, clock=clock, abort_mismatched=True, **over)
    assert res.plan.attempt_no == 1 and res.plan.reason == "first_pass" and res.meta["aborted_stamps"] == 1
    for name in (f"{STEM}.generations.jsonl", f"{STEM}.export_start.json"):
        assert (d / f"{name}.aborted-1").read_bytes() == before[name]
    if how != "parser_sha256":
        assert _reasons(W.verify(w)) == ["first_pass"]


def test_개명이_도중에_멈추면_같은_번호로_마저_옮긴다(w, monkeypatch):
    clock = W.Clock()
    W.run(w, clock=clock, stop_after_lines=1)
    d = _folder(w)
    real = XW._rename_retry

    def die_on_stamp(src, dst):
        if src.name.endswith(".export_start.json"):
            raise RuntimeError("도장을 옮기기 전에 죽었다")
        return real(src, dst)

    _other_commit(w)
    monkeypatch.setattr(XW, "_rename_retry", die_on_stamp)
    with pytest.raises(RuntimeError):
        W.run(w, clock=clock, abort_mismatched=True)
    monkeypatch.setattr(XW, "_rename_retry", real)
    assert (d / f"{STEM}.generations.jsonl.aborted-1").exists() and (d / f"{STEM}.export_start.json").exists()
    with pytest.raises(ExportRefused) as exc:                          # 그냥 이어 쓰지 않는다
        W.run(w, clock=clock)
    assert exc.value.code == "stamp_mismatch" and "개명" in str(exc.value)
    res = W.run(w, clock=clock, abort_mismatched=True)
    assert (d / f"{STEM}.export_start.json.aborted-1").exists()
    assert not (d / f"{STEM}.generations.jsonl.aborted-2").exists()
    assert res.plan.reason == "first_pass" and res.meta["aborted_stamps"] == 1
    assert _reasons(W.verify(w)) == ["first_pass"]


def test_도장만_있고_시작_줄이_없으면_이어_쓰지_않는다(w, monkeypatch):
    clock = W.Clock()
    monkeypatch.setattr(ExportWriter, "_write_start", lambda self, plan: (_ for _ in ()).throw(RuntimeError("죽음")))
    with pytest.raises(RuntimeError):
        W.run(w, clock=clock)
    monkeypatch.undo()
    with pytest.raises(ExportRefused) as exc:
        W.run(w, clock=clock)
    assert exc.value.code == "stamp_mismatch" and "시작 줄이 없다" in str(exc.value)


def test_도장_없이_생성_파일만_있으면_쓰지_않는다(w):
    d = _folder(w)
    d.mkdir(parents=True)
    (d / f"{STEM}.generations.jsonl").write_bytes(b'{"image_id": "aihub000001"}\n')
    with pytest.raises(ExportRefused) as exc:
        W.run(w)
    assert exc.value.code == "orphan_generations"
    assert sorted(p.name for p in d.iterdir()) == [f"{STEM}.generations.jsonl"]


# ================================================================ 1. 상태 기계 · 되읽기 · 시각
def _elist(ids=IDS):
    from evaluation.eval_list import canonical_bytes, validate_eval_list

    return validate_eval_list(canonical_bytes(list(ids)))


def _stamp_fields():
    return {"generation_sha256": hx("gen_fp"), "export_commit": "c" * 40, "parser_sha256": parser_sha256(),
            "train_run_id": W.RUN_ID, "train_ledger_sha256": hx("ledger"), "scored_adapter_sha256": hx("adapter"),
            "purpose": "rehearsal"}


def _writer(tmp: Path, **kw) -> ExportWriter:
    return ExportWriter(tmp / "w", tag="uni_local_C1", seed_index=1, mode="model", eval_list=_elist(),
                        stamp_fields=_stamp_fields(), device=DEVICE, clock=kw.pop("clock", W.Clock()), **kw)


def test_상태_기계를_건너뛰는_호출은_예외다(tmp_path):
    w = _writer(tmp_path)
    with pytest.raises(ExportStateError):
        w.open_generations()                            # 시작 줄 전
    with pytest.raises(ExportStateError):
        w.write({"image_id": IDS[0]})
    w.begin()
    with pytest.raises(ExportStateError):
        w.write({"image_id": IDS[0]})                   # open 전
    w.open_generations()
    with pytest.raises(ExportStateError):
        w.write({"image_id": IDS[1]})                   # 목록 순서가 아니다
    with pytest.raises(ExportStateError):
        w.seal({})                                      # 끝 줄 전


def test_시작_줄을_되읽어_다르면_생성_파일을_열지_않는다(tmp_path):
    w = _writer(tmp_path)
    w.begin()
    att = w.p_att
    att.write_bytes(att.read_bytes().replace(b'"attempt_no": 1', b'"attempt_no": 2'))
    with pytest.raises(ExportRefused) as exc:
        w.open_generations()
    assert exc.value.code == "start_line_reread" and not w.p_gen.exists()


def test_끝_줄을_덧붙이기_전에도_시도_기록을_검사한다(tmp_path):
    w = _writer(tmp_path)
    w.begin()
    w.p_att.write_bytes(w.p_att.read_bytes() + b"\n")                  # 빈 줄이 끼었다
    before = w.p_att.read_bytes()
    with pytest.raises(ExportRefused) as exc:
        w.end()
    assert exc.value.code == "attempts_broken" and w.p_att.read_bytes() == before


def test_시계가_뒤로_가면_쓰지_않는다(tmp_path):
    clock = W.Clock()
    w = _writer(tmp_path, clock=clock)
    w.begin()
    w.open_generations()
    clock.t -= timedelta(seconds=5)
    with pytest.raises(ExportRefused) as exc:
        w.write({"image_id": IDS[0]})
    assert exc.value.code == "clock_back" and w.p_gen.read_bytes() == b""


def test_새_프로세스는_이_도장의_마지막_기록보다_이른_시각에_시작하지_않는다(w):
    clock = W.Clock()
    W.run(w, clock=clock, stop_after_lines=1)
    before = _snapshot(_folder(w))
    with pytest.raises(ExportRefused) as exc:
        W.run(w, clock=W.Clock(datetime(2026, 10, 1, 11, 0, tzinfo=KST)))
    assert exc.value.code == "clock_back" and _snapshot(_folder(w)) == before


def test_모든_파일이_LF_UTF8_BOM_없음이다(w):
    W.run(w, with_tokens=True)
    for p in _folder(w).iterdir():
        raw = p.read_bytes()
        assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf"), p.name
        raw.decode("utf-8")


def test_도장은_정해진_키만_받고_목록은_검사를_지난_것만_받는다(tmp_path):
    with pytest.raises(ExportRefused) as exc:
        ExportWriter(tmp_path, tag="uni_local_C1", seed_index=1, mode="model", eval_list=_elist(),
                     stamp_fields={**_stamp_fields(), "start_stamp_sha256": "0" * 64}, device=DEVICE)
    assert exc.value.code == "stamp_unknown_key"
    with pytest.raises(ExportRefused) as exc:
        ExportWriter(tmp_path, tag="uni_local_C1", seed_index=1, mode="model", eval_list=list(IDS),
                     stamp_fields=_stamp_fields(), device=DEVICE)
    assert exc.value.code == "eval_list_unchecked"


def test_곁_파일의_같은_키에_다른_값이_오면_쓰지_않는다(tmp_path):
    w = _writer(tmp_path)
    w.begin()
    w.open_generations()
    for iid in IDS:
        w.write({"image_id": iid})
    w.end()
    with pytest.raises(ExportRefused) as exc:
        w.seal({"n_lines": 99})
    assert exc.value.code == "sidecar_clash" and not w.p_side.exists()


# ================================================================ 가드의 순서
def test_본실험_목적은_대역_실측과_생성기를_부르기_전에_거부한다(w):
    from vlm.seams import SeamRejected

    calls: list[str] = []
    loader = W.Loader()
    with pytest.raises(SeamRejected):
        W.run(w, purpose="main", loader=loader, measure_fn=lambda spec: calls.append("m"),
              folder=w.tmp / "main_out")
    assert calls == [] and loader.calls == 0 and not (w.tmp / "main_out").exists()


@pytest.mark.parametrize("purpose", ["main", "frame_diag"])
def test_본실험과_진단은_고의_중단_변수가_보이면_export_를_시작하지_않는다(w, purpose):
    from vlm.fault import FaultRefused, write_planned

    nonce = "0" * 32
    write_planned(w.root, nonce, "export:after_lines:uni_local_C1:n=1", stage_no=1)
    env = {"WELD_REHEARSAL_FAULT": "export:after_lines:uni_local_C1:n=1", "WELD_REHEARSAL_NONCE": nonce}
    with pytest.raises(FaultRefused) as exc:
        W.run(w, purpose=purpose, env=env)
    assert exc.value.code == "fault_env_forbidden" and not _folder(w).exists()


# ================================================================ export 의 고의 중단(2판 §1-3)
class _Exited(BaseException):
    pass


@pytest.fixture
def no_exit(monkeypatch):
    """`os._exit(75)` 를 예외로 바꿔 끼운다 — 프로세스 안 시험이 pytest 를 죽이지 않게. 진짜 죽음은 자식 시험이 본다."""
    from vlm import fault as F

    def die(code):
        raise _Exited(code)

    monkeypatch.setattr(F.os, "_exit", die)


def _fault_env(w, fault: str, nonce: str = "1" * 32) -> dict:
    from vlm.fault import write_planned

    write_planned(w.root, nonce, fault, stage_no=3)
    return {"WELD_REHEARSAL_FAULT": fault, "WELD_REHEARSAL_NONCE": nonce}


def _faults(w) -> tuple[list, list]:
    d = w.root / "faults"
    return sorted(d.glob("*.activated.json")), sorted(d.glob("*.fired.json"))


def test_줄_뒤의_중단은_그_줄까지_쓰고_죽고_다음_시도가_이어_간다(w, no_exit):
    clock = W.Clock()
    with pytest.raises(_Exited) as exc:
        W.run(w, clock=clock, env=_fault_env(w, "export:after_lines:uni_local_C1:n=2"))
    assert exc.value.args == (75,)
    act, fired = _faults(w)
    assert len(act) == len(fired) == 1
    assert len(_rows(w, f"{STEM}.generations.jsonl")) == 2 and [r["event"] for r in _rows(w, "attempts.jsonl")] == ["start"]
    res = W.run(w, clock=clock)
    assert res.sealed and res.plan.reason == "resume_after_interruption" and res.plan.first_id == IDS[2]
    ev, = res.meta["fault_events"]
    assert ev["activated"] == act[0].name and ev["fired"] == fired[0].name
    assert _reasons(W.verify(w)) == ["first_pass", "resume_after_interruption"]


def test_찢긴_줄의_중단은_앞_절반을_남기고_다음_시도가_다시_만든다(w, no_exit):
    clock = W.Clock()
    with pytest.raises(_Exited):
        W.run(w, clock=clock, env=_fault_env(w, "export:torn_line:uni_local_C1:n=3"))
    raw = (_folder(w) / f"{STEM}.generations.jsonl").read_bytes()
    torn = raw[raw.rfind(b"\n") + 1:]
    assert raw.count(b"\n") == 2 and torn                         # 두 줄과 셋째 줄의 앞 절반 — 개행이 없다
    res = W.run(w, clock=clock)
    assert res.plan.reason == "retry_torn_tail" and res.plan.truncated_ids == (IDS[2],)
    assert (_folder(w) / f"{STEM}.truncated-2.bin").read_bytes() == torn
    assert _reasons(W.verify(w)) == ["first_pass", "retry_torn_tail"]


def test_목록을_다_쓴_뒤의_중단은_다음_시도가_모델_없이_닫는다(w, no_exit):
    clock = W.Clock()
    with pytest.raises(_Exited):
        W.run(w, clock=clock, env=_fault_env(w, "export:after_all_lines:uni_local_C1:-"))
    loader = W.Loader()
    res = W.run(w, clock=clock, loader=loader)
    assert res.sealed and res.plan.first_id is None and loader.calls == 0
    assert _reasons(W.verify(w)) == ["first_pass", "resume_after_interruption"]


def test_도장_뒤의_중단은_시작_줄이_없어_명시_인자로만_새로_시작한다(w, no_exit):
    clock = W.Clock()
    with pytest.raises(_Exited):
        W.run(w, clock=clock, env=_fault_env(w, "export:after_stamp:uni_local_C1:-"))
    assert not (_folder(w) / "attempts.jsonl").exists()
    with pytest.raises(ExportRefused) as exc:
        W.run(w, clock=clock)
    assert exc.value.code == "stamp_mismatch" and "시작 줄이 없다" in str(exc.value)
    res = W.run(w, clock=clock, abort_mismatched=True)
    assert res.sealed and res.plan.reason == "first_pass"


def test_같은_열쇠의_활성화_기록이_있으면_다시_켜지_않고_끝까지_쓴다(w, no_exit, capsys):
    # 찢긴 줄은 다음 시도가 잘라 내고 **같은 줄 번호를 다시 쓴다** — 같은 지점에 다시 닿는다.
    clock = W.Clock()
    env = _fault_env(w, "export:torn_line:uni_local_C1:n=3")
    with pytest.raises(_Exited):
        W.run(w, clock=clock, env=env)
    res = W.run(w, clock=clock, env=env)                          # 같은 변수로 다시 — 켜지지 않는다
    assert res.sealed and res.plan.reason == "retry_torn_tail" and len(_faults(w)[0]) == 1
    assert "다시 켜지 않는다" in capsys.readouterr().err


def test_줄_번호가_목록_밖이면_쓰기_전에_거부한다(w, no_exit):
    from vlm.fault import FaultRefused

    with pytest.raises(FaultRefused) as exc:
        W.run(w, env=_fault_env(w, "export:after_lines:uni_local_C1:n=9"))
    assert exc.value.code == "fault_range" and not _folder(w).exists()


CHILD = r'''
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, sys.argv[1])
from datetime import timedelta
from pathlib import Path
from tests import uni_export_world as W
w = W.attach(Path(sys.argv[2]))
clock = W.Clock(W.Clock().t + timedelta(minutes=int(sys.argv[3])))      # 다시 띄운 프로세스는 나중에 시작한다
env = {"WELD_REHEARSAL_FAULT": sys.argv[4], "WELD_REHEARSAL_NONCE": sys.argv[5]} if len(sys.argv) > 4 else {}
res = W.run(w, env=env, clock=clock)
print("SEALED" if res.sealed else "OPEN")
'''


def test_자식_프로세스가_줄_뒤에_75_로_죽고_다시_띄운_자식이_끝낸다(w):
    import subprocess

    from vlm.fault import write_planned

    fault, nonce = "export:after_lines:uni_local_C1:n=1", "2" * 32
    write_planned(w.root, nonce, fault, stage_no=3)
    py = [sys.executable, "-X", "utf8", "-c", CHILD, str(ROOT), str(w.tmp)]
    first = subprocess.run(py + ["0", fault, nonce], capture_output=True, timeout=600, check=False)
    assert first.returncode == 75, first.stderr.decode("utf-8", "replace")[-800:]
    assert len(_rows(w, f"{STEM}.generations.jsonl")) == 1
    second = subprocess.run(py + ["5"], capture_output=True, timeout=600, check=False)
    assert second.returncode == 0 and b"SEALED" in second.stdout, second.stderr.decode("utf-8", "replace")[-800:]


def test_리허설_출력이나_입력이_루트_밖이면_쓰기_전에_거부한다(w):
    with pytest.raises(ExportRefused) as exc:
        W.run(w, folder=w.tmp / "elsewhere" / "export")
    assert exc.value.code == "run_root" and not (w.tmp / "elsewhere").exists()
    outside = w.tmp / "gen.txt"
    outside.write_bytes(w.lists["model"].read_bytes())
    with pytest.raises(ExportRefused) as exc:
        W.run(w, list_path=outside)
    assert exc.value.code == "run_root" and not _folder(w).exists()


def test_본실험_출력이_비본실험_부모_아래면_거부한다(w, monkeypatch):
    # 승인된 생성기가 아직 없어 순서 3 을 지나게 하려고 시험이 승인 목록을 바꿔 끼운다 — 순서 4 만 본다.
    loader = W.Loader()
    monkeypatch.setattr(XR, "APPROVED_MEASURES", (W.measure,))
    monkeypatch.setattr(XR, "APPROVED_GENERATOR_LOADERS", {"model": (loader,), "echo": (loader,)})
    with pytest.raises(ExportRefused) as exc:
        W.run(w, purpose="main", loader=loader, plan_path=None)
    assert exc.value.code == "run_root" and not _folder(w).exists()


def test_적합성_벡터에서_어긋나면_쓰기_전에_거부한다(w, monkeypatch):
    import vlm.gen_parse as gp

    real = gp.conformance_tuple
    monkeypatch.setattr(gp, "conformance_tuple", lambda t, **kw: ("ok",) + real(t, **kw)[1:])
    with pytest.raises(ExportRefused) as exc:
        W.run(w)
    assert exc.value.code == "conformance" and not _folder(w).exists()


def test_짝이_어긋난_줄도_멈추지_않고_그대로_쓴다(w, monkeypatch):
    import vlm.gen_parse as gp

    monkeypatch.setattr(XR, "conformance_problems", list)        # 사전의 짝 점검을 지나게 한다
    monkeypatch.setattr(gp, "line_fields", lambda p, g, c: {"bbox_px_parsed": None, "parse_error": None,
                                                             "n_bad_items_dropped": 0})
    res = W.run(w)
    assert res.sealed and res.n_written == len(IDS)
    rows = _rows(w, f"{STEM}.generations.jsonl")
    assert all(r["parse_error"] is None and r["bbox_px_parsed"] is None for r in rows)
    # 그 줄이 배관 고장의 표시다 — 평가 쪽 어댑터가 레코드 실패로 세지 않고 거부한다.
    from evaluation.adapters_v14 import UpstreamContractError, adapt_unified_main
    from evaluation.conformance_v14 import KNOWN_CODES, SCORING_CODES

    lines = (_folder(w) / f"{STEM}.generations.jsonl").read_bytes().decode("utf-8")[:-1].split("\n")
    with pytest.raises(UpstreamContractError):
        adapt_unified_main(lines, tag="uni_local_C1", seed=7, coupling_rule="decoupled_v2",
                           known_iso_codes=KNOWN_CODES, scoring_iso_codes=SCORING_CODES)


# ================================================================ 사전 점검 — 평가 쪽 실제 함수(2판 §2-4 의 1)
def _bytes_edit(p: Path, fn) -> None:
    p.write_bytes(fn(p.read_bytes()))


def _receipt_edit(w, **over) -> None:
    rc = json.loads(w.receipt.read_text(encoding="utf-8"))
    rc.update(over)
    w.receipt.write_text(json.dumps(rc), encoding="utf-8")


def _commit_config(w, text: str) -> None:
    (w.repo / "configs/base.yaml").write_text(text, encoding="utf-8")
    W.git(w.repo, "commit", "-q", "-am", "설정을 바꿨다")


CASES = {
    # 영수증 · 등록
    "receipt_kind": (lambda w: _receipt_edit(w, kind="frame_diag"), "entry_RECEIPT_KIND"),
    "registration_bytes": (lambda w: _bytes_edit(w.root / "registration/rehearsal-20261001-1.json",
                                                 lambda b: b.replace(b'"left"', b'"lefT"')), "entry_REGISTRATION_FILE_HASH"),
    "receipt_fingerprint": (lambda w: _receipt_edit(w, generation_sha256="0" * 64), "entry_RECEIPT"),
    # 닻 · 프로세서 인자(본줄기의 설정 바이트)
    "anchor": (lambda w: _commit_config(w, (w.repo / "configs/base.yaml").read_text(encoding="utf-8")
                                        .replace(w.digest, "f" * 64)), "entry_SNAPSHOT_NOT_ANCHOR"),
    "processor": (lambda w: _commit_config(w, (w.repo / "configs/base.yaml").read_text(encoding="utf-8")
                                           .replace(f"max_pixels: {W.MAX_PIXELS}", "max_pixels: 1")),
                  "entry_PROCESSOR_MISMATCH"),
    # 목록 · 분할 · 노출 기록
    "list_binding": (lambda w: w.lists["model"].write_bytes(
        __import__("evaluation.eval_list", fromlist=["x"]).canonical_bytes(IDS[:3])), "list_binding"),
    "exposure": (lambda w: w.exposure.write_bytes(b""), "exposure_not_planned"),
    # 계획 · 실측값 · 학습 출처 · 청결
    "plan": (lambda w: w.plan.write_text(w.plan.read_text(encoding="utf-8") + "# 바꿨다\n", encoding="utf-8"),
             "plan_hash"),
    "adapter_sha": (lambda w: _bytes_edit(w.adapter / "adapter_last.npz", lambda b: b + b"\0"), "adapter_sha"),
    "adapter_purpose": (lambda w: W.write_adapter(w, purpose="main"), "adapter_identity"),
    "ledger_open": (lambda w: W.write_adapter(w, close=False), "ledger_LEDGER_NOT_CLOSED"),
    "adapter_processor": (lambda w: W.write_adapter(w, meta_over={"processor_config_sha256": "0" * 64}),
                          "adapter_processor"),
    "train_rows": (lambda w: W.write_adapter(w, meta_over={"train_rows_digest": "0" * 64}), "train_rows"),
    "tree_dirty": (lambda w: (w.code / "m.py").write_text("x = 3\n", encoding="utf-8"), "tree_dirty"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_사전_점검이_거부하면_아무것도_쓰지_않고_생성기를_올리지_않는다(w, case):
    tamper, code = CASES[case]
    tamper(w)
    loader = W.Loader()
    with pytest.raises(ExportRefused) as exc:
        W.run(w, loader=loader)
    assert exc.value.code == code, str(exc.value)
    assert loader.calls == 0 and not _folder(w).exists()


def test_목록의_id_가_목적의_분할이_아니면_쓰기_전에_거부한다(tmp_path):
    w = W.build(tmp_path, ids=IDS)
    from evaluation.eval_list import canonical_bytes, validate_eval_list

    raw = canonical_bytes(IDS[:3] + [W.EVAL_ID])           # 평가 id 가 섞인 목록 — 등록도 그 목록에 맞춘다
    w.lists["model"].write_bytes(raw)
    el = validate_eval_list(raw)

    def edit(r):
        r.generation.eval_list_file_sha256, r.generation.eval_list_set_sha256 = el.file_sha256, el.set_sha256

    w.reg = W.registration(w, edit=edit)
    receipt = W.write_registration(w, w.reg, name="rehearsal-20261001-2")
    with pytest.raises(ExportRefused) as exc:
        W.run(w, receipt_path=receipt)
    assert exc.value.code == "entry_LIST_SPLIT" and not _folder(w).exists()


def test_등록의_동결본이_닻과_다르면_등록_내용을_쓰기_전에_거부한다(w):
    """설정과 스냅샷은 맞고 **등록만** 다른 동결본을 적었다 — 스냅샷 검증은 지나고 닻 대조가 잡는다(실측값 대조보다 먼저)."""
    def edit(r):
        r.generation.snapshot_digest = "f" * 64

    w.reg = W.registration(w, edit=edit)
    receipt = W.write_registration(w, w.reg, name="rehearsal-20261001-2")
    with pytest.raises(ExportRefused) as exc:
        W.run(w, receipt_path=receipt)
    assert exc.value.code == "entry_SNAPSHOT_NOT_ANCHOR" and not _folder(w).exists()


def test_실측값이_등록과_다르면_항목_이름을_들고_거부한다(w):
    def other(spec):
        return {**W.MEASURED, "gen_prefix_sha256": "0" * 64}

    loader = W.Loader()
    with pytest.raises(ExportRefused) as exc:
        W.run(w, loader=loader, measure_fn=other)
    assert exc.value.code == "actuals" and "gen_prefix_sha256" in str(exc.value)
    assert loader.calls == 0


def test_실측은_설정의_모델과_프로세서_인자와_등록_프롬프트로_연다(w):
    seen = []

    def spy(spec):
        seen.append(spec)
        return dict(W.MEASURED)

    loader = W.Loader()
    W.run(w, measure_fn=spy, loader=loader)
    spec, = seen
    assert (spec.model_id, spec.model_revision) == ("Qwen/Qwen3.5-4B", "a" * 40)
    assert dict(spec.processor_kwargs) == {"max_pixels": W.MAX_PIXELS}
    assert spec.prompt_text == (ROOT / W.PROMPT_REL).read_text(encoding="utf-8")
    cfg = loader.cfg
    assert (cfg.max_new_tokens, cfg.batch_size, cfg.coord_space) == (512, 1, "ABS_ORIG")
    assert cfg.adapter_path == w.adapter / "adapter_last.npz"


def test_리허설_구현_식별자가_등록과_다르면_대역_실행으로_적고_지나간다(w):
    def edit(r):
        r.generation.impl_ids = ("export_generator=x:y@z.py", "export_measure=x:y@z.py")

    w.reg = W.registration(w, edit=edit)
    receipt = W.write_registration(w, w.reg, name="rehearsal-20261001-2")
    res = W.run(w, receipt_path=receipt)
    assert res.sealed and res.stand_in is True


def test_끝난_묶음은_노출_원장에_생성_줄을_남기고_중간_시도는_남기지_않는다(w):
    from vlm.exposure import read_rows

    clock = W.Clock()
    n0 = len(read_rows(w.exposure))
    first = W.run(w, clock=clock, stop_after_lines=2)
    assert first.exposure_line is None and len(read_rows(w.exposure)) == n0
    second = W.run(w, clock=clock)
    rows = read_rows(w.exposure)
    assert len(rows) == n0 + 1 and rows[-1]["event"] == "generated"
    assert (rows[-1]["run_id"], rows[-1]["tag"], rows[-1]["mode"], rows[-1]["n_written"]) == \
        (W.REH, "uni_local_C1", "model", len(IDS))
    assert rows[-1]["list_file_sha256"] == w.elist.file_sha256 and second.exposure_line.endswith(b"\n")


def test_실측의_실제_구현은_캐시의_프로세서에서_일곱_값을_낸다():
    """실제 프로세서(캐시의 `Qwen/Qwen3.5-4B`, CPU · 망 없음)를 열어 잰다 — 모델 가중치는 올리지 않는다."""
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache("Qwen/Qwen3.5-4B", "preprocessor_config.json"), str):
        pytest.skip("프로세서 캐시가 없다")
    from transformers import AutoProcessor

    from vlm.export_measure import MEASURED_KEYS, measure_processor

    proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True, max_pixels=W.MAX_PIXELS)
    got = measure_processor(proc, prompt_text="좌표를 적어라", chat_template_kwargs={"enable_thinking": False})
    assert set(got) == set(MEASURED_KEYS)
    assert (got["processor_max_pixels"], got["patch_size"], got["merge_size"]) == (W.MAX_PIXELS, 16, 2)
    assert type(got["processor_min_pixels"]) is int and len(got["gen_prefix_sha256"]) == 64


# ================================================================ 에코 생성기의 실제 구현(캐시의 프로세서)
def _cached_processor():
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache("Qwen/Qwen3.5-4B", "preprocessor_config.json"), str):
        pytest.skip("프로세서 캐시가 없다")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True, max_pixels=W.MAX_PIXELS)


def _val_pairs(w) -> tuple[Path, list[dict]]:
    rows = [{"image_id": i, "image_path": str(w.images[i]), "client": "C1", "split": "val",
             "skeleton": {"defects": [{"type": "2011", "bbox_px": [10 + k, 20, 300, 400.5]}], "verdict": "불합격",
                          "clauses": []}} for k, i in enumerate(IDS)]
    p = w.tmp / "pairs.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p, rows


def test_에코_생성기는_학습_타깃_문자열을_넣고_실제_프로세서의_격자를_적는다(w):
    from dataclasses import replace

    from PIL import Image

    from vlm.export_echo import EchoUnavailable, load_echo_generator
    from vlm.export_preflight import GenerationConfig
    from vlm.pilot_vlm import build_target

    proc = _cached_processor()
    pairs, rows = _val_pairs(w)
    cfg = GenerationConfig(mode="echo", tag="uni_local_C1", model_id="Qwen/Qwen3.5-4B", model_revision=None,
                           processor_kwargs={}, prompt_text="", chat_template_kwargs={}, max_new_tokens=512,
                           decoding={}, batch_size=1, padding_side="left", coord_space="ABS_ORIG", pairs_path=str(pairs))
    g = load_echo_generator(cfg, processor=proc)
    img = Image.open(w.images[IDS[1]])
    out = g(img, IDS[1])
    assert out.text == build_target(rows[1], ImageGeom(orig_w=1280, orig_h=720), CoordCfg(coord_space="ABS_ORIG"))
    assert (out.grid_thw, out.model_input_wh, out.gen_stop) == ((1, 44, 80), (1280, 704), "eos")
    assert out.n_new_tokens > 1
    lf = line_fields(parse_generation(out.text), ImageGeom(orig_w=1280, orig_h=720), CoordCfg(coord_space="ABS_ORIG"))
    from vlm.coords import quantize, to_model, to_px

    geom, cc = ImageGeom(orig_w=1280, orig_h=720), CoordCfg(coord_space="ABS_ORIG")
    want = [float(v) for v in to_px(quantize(to_model(rows[1]["skeleton"]["defects"][0]["bbox_px"], geom, cc)), geom, cc)]
    assert lf["bbox_px_parsed"]["defects"][0]["bbox_px"] == want                  # 학습 타깃의 양자화 한 번뿐이다
    with pytest.raises(EchoUnavailable):
        g(img, "aihub999999")
    for bad in (replace(cfg, mode="model"), replace(cfg, pairs_path=None)):
        with pytest.raises(EchoUnavailable):
            load_echo_generator(bad, processor=proc)


def test_에코_생성기의_실제_구현으로_쓴_에코_묶음을_평가_쪽_검증기가_받는다(w):
    from dataclasses import replace

    from vlm.export_echo import load_echo_generator

    proc = _cached_processor()
    pairs, _ = _val_pairs(w)
    seen = []

    def load(cfg):                     # 시험이 연 프로세서와 페어 경로를 넣는다 — 리허설에서만 받는 대역 적재
        seen.append(cfg)
        return load_echo_generator(replace(cfg, pairs_path=str(pairs)), processor=proc)

    res = W.run(w, mode="echo", loader=load)
    assert res.sealed and len(seen) == 1 and res.meta["image_grid_thw_observed"] == {"[1, 44, 80]": len(IDS)}
    rows = [json.loads(x) for x in (_folder(w, "echo") / f"{STEM}.echo.generations.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [r["bbox_px_parsed"]["defects"][0]["bbox_px"][0] for r in rows] == [10.0, 11.0, 12.0, 13.0]
    assert W.verify(w, mode="echo").mode == "echo"


def test_두_모드의_승인_목록이_같은_적재_함수다():
    from vlm.export_generator import load_generator, model_generator_available

    assert XR.APPROVED_GENERATOR_LOADERS["echo"] == (load_generator,)           # 두 모드가 같은 적재 함수를 지난다
    assert model_generator_available() and XR.APPROVED_GENERATOR_LOADERS["model"] == (load_generator,)


def test_커밋된_진단_영수증이면_대역을_한_번도_부르기_전에_거부한다(w):
    """진짜 진단 등록(본줄기의 조상 커밋을 적은 진단 영수증)은 대역 이음새를 받지 않는다 — 대역 실측 · 생성기의 호출이 0 이다(외부 검토 8)."""
    import subprocess

    root_commit = subprocess.run(["git", "-C", str(ROOT), "rev-list", "--max-parents=0", "refs/heads/main"],
                                 capture_output=True, check=True).stdout.decode().split()[-1]
    rc = json.loads(w.receipt.read_text(encoding="utf-8"))
    committed = w.root / "registration" / "fdiag-committed.receipt.json"
    committed.write_text(json.dumps(rc | {"kind": "frame_diag", "main_commit": root_commit}), encoding="utf-8")
    calls = []
    loader = W.Loader()
    with pytest.raises(ExportRefused) as exc:
        W.run(w, purpose="frame_diag", receipt_path=committed, measure_fn=lambda spec: calls.append(spec),
              loader=loader, standin_allowed=True)
    assert exc.value.code == "seam_on_committed_receipt"
    assert calls == [] and loader.calls == 0 and not _folder(w).exists()
