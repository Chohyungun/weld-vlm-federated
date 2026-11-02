"""두 구현 대조 틀(`evaluation.narrative_conformance`) — 실행기 · 세 겹 대조 · 실패를 종료 코드로 돌려준다(03 2판 §5b · 195 §4 · M3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import narrative_conformance as NC
from evaluation import narrative_extract as NE

REPO = Path(__file__).resolve().parents[1]
VECTORS = NC.load_vectors()


def _always_missing(text, *args):
    """일부러 틀린 구현 — 모든 벡터를 미기재로 낸다."""
    return {"state": "missing", "narrative": "", "front_text": False, "stop_kept": "eos", "upstream_reason": None}


def test_D_의_실행은_기대와_전부_같다() -> None:
    res = NC.compare(VECTORS, d_rows=NC.run(NE.extract, VECTORS))
    assert res["ok"] and res["counts"]["기대 = D"] == {"same": len(VECTORS), "n": len(VECTORS)}


def test_칸_없음_벡터는_인자를_빼고_부르고_null_은_None_을_넘긴다() -> None:
    seen: list[tuple] = []

    def rec(text, *args):
        seen.append(args)
        return NE.extract(text, *args)

    NC.run(rec, VECTORS)                                    # 벡터 순서대로 부른다 — 같은 글을 쓰는 벡터(B4 · B5a · B5b · F5)가 있다
    by_id = dict(zip((v["id"] for v in VECTORS), seen, strict=True))
    assert by_id["B5b"] == () and by_id["E2"] == (None,) and by_id["F5"] == (None,) and by_id["A1"] == ("eos",)


def test_한_칸이라도_어긋나면_실패이고_어느_칸인지_싣는다() -> None:
    def wrong(text, *args):
        r = NE.extract(text, *args)
        if "A1 의 첫" in text:
            return NE.NarrativeExtract(r.state, r.narrative + " ", r.front_text, r.stop_kept, r.upstream_reason)
        return r

    res = NC.compare(VECTORS, d_rows=NC.run(wrong, VECTORS))
    assert not res["ok"]
    assert {m["field"] for m in res["mismatches"] if m["id"] == "A1"} == {"narrative", "narrative_sha256"}


def test_두_구현이_같은_오답을_내도_기대와의_대조가_잡는다() -> None:
    rows = NC.run(_always_missing, VECTORS)
    res = NC.compare(VECTORS, d_rows=rows, c_rows=rows)
    assert res["counts"]["C = D"]["same"] == len(VECTORS)
    assert res["counts"]["기대 = D"]["same"] < len(VECTORS) and not res["ok"]


def test_빠지거나_겹치거나_모르는_사례는_실패다() -> None:
    rows = NC.run(NE.extract, VECTORS)
    for bad, word in ((rows[:-1], "빠진"), (rows + rows[:1], "겹친"),
                      (rows + [{**rows[0], "id": "ZZ"}], "벡터에 없는")):
        res = NC.compare(VECTORS, d_rows=bad)
        assert not res["ok"] and any(word in p for p in res["problems"])


def test_입력이_다르면_전달_사고로_표시한다() -> None:
    rows = NC.run(NE.extract, VECTORS)
    rows[0] = {**rows[0], "input_sha256": "0" * 64}
    res = NC.compare(VECTORS, d_rows=rows)
    assert any(m["field"] == "input_sha256" and m["cause_hint"] for m in res["mismatches"])


def test_명령줄은_실패를_종료_코드로_돌려준다(tmp_path: Path) -> None:
    out = tmp_path / "d.jsonl"
    assert NC.main(["run", "--out", str(out)]) == 0
    assert NC.main(["compare", "--d", str(out), "--out", str(tmp_path / "s.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["ok"] is True
    assert NC.main(["run", "--out", str(out)]) == 2                         # 새 파일로만 쓴다

    lines = out.read_text(encoding="utf-8").split("\n")
    row = json.loads(lines[1])
    del row["stop_kept"]
    broken = tmp_path / "broken.jsonl"
    broken.write_bytes("\n".join([lines[0], json.dumps(row, ensure_ascii=False), *lines[2:]]).encode("utf-8"))
    assert NC.main(["compare", "--d", str(broken)]) == 1                     # 칸이 빠졌다

    head = json.loads(lines[0])
    head["vector_sha256"] = "f" * 64
    other = tmp_path / "other.jsonl"
    other.write_bytes("\n".join([json.dumps(head, ensure_ascii=False), *lines[1:]]).encode("utf-8"))
    assert NC.main(["compare", "--d", str(other)]) == 1                      # 다른 벡터로 낸 산출

    assert NC.main(["run", "--impl", f"{__name__}:_always_missing", "--out", str(tmp_path / "w.jsonl")]) == 1


def test_호출_지문은_종료_사유의_있음과_값까지_묶는다() -> None:
    """201 I-04 — 같은 글에 다른 종료 사유(값 · null · 칸 없음)를 주면 입력 sha256 은 같고 호출 지문은 갈린다."""
    text = VECTORS[0]["text"]
    forms = [{"text": text, "gen_stop": "eos"}, {"text": text, "gen_stop": "length"}, {"text": text, "gen_stop": None},
             {"text": text}]
    assert len({NC.text_sha256(v["text"]) for v in forms}) == 1
    assert len({NC.call_sha256(v) for v in forms}) == len(forms)
    assert NC.call_sha256({"id": "x", "text": text, "gen_stop": "eos", "expect": {}}) == NC.call_sha256(forms[0])  # 다른 칸은 안 묶는다


def test_종료_사유만_다른_입력을_넘기면_전달_사고로_잡는다() -> None:
    """C 쪽이 같은 글에 다른 종료 사유로 불렀다 — 다섯 값이 같아도 호출 지문이 갈려 전달 사고로 표시한다."""
    d_rows = NC.run(NE.extract, VECTORS)
    swapped = [{**v, "gen_stop": "length"} if v["id"] == "A1" else v for v in VECTORS]
    c_rows = [{**r, "call_sha256": NC.call_sha256(v)} for r, v in zip(d_rows, swapped, strict=True)]
    res = NC.compare(VECTORS, d_rows=d_rows, c_rows=c_rows, require_both=True)
    bad = [m for m in res["mismatches"] if m["field"] == "call_sha256"]
    assert not res["ok"] and {m["id"] for m in bad} == {"A1"} and all(m["cause_hint"].startswith("전달") for m in bad)


def test_값은_자료형까지_맞댄다() -> None:
    """04 §7-10 — `0` 과 `False` 는 다르다(C 의 `_same` 과 같은 잣대)."""
    rows = NC.run(NE.extract, VECTORS)
    i = next(k for k, r in enumerate(rows) if r["front_text"] is False)
    rows[i] = {**rows[i], "front_text": 0}
    res = NC.compare(VECTORS, d_rows=rows)
    assert not res["ok"] and [m["field"] for m in res["mismatches"]] == ["front_text"]
    assert NC.same(False, False) and not NC.same(0, False) and not NC.same(1, True) and NC.same(None, None)


def test_세_겹_모드는_두_쪽이_다_있고_호출_지문을_실어야_통과다() -> None:
    """201 I-03 — 한쪽만 준 대조는 단독 점검이고, `require_both` 는 세 줄 모두 전부 같을 때만 통과다."""
    rows = NC.run(NE.extract, VECTORS)
    alone = NC.compare(VECTORS, d_rows=rows)
    assert alone["ok"] and alone["mode"] == "단독 점검"
    strict_alone = NC.compare(VECTORS, d_rows=rows, require_both=True)
    assert not strict_alone["ok"] and any("둘 다" in p for p in strict_alone["problems"])
    both = NC.compare(VECTORS, d_rows=rows, c_rows=[dict(r) for r in rows], require_both=True)
    assert both["ok"] and both["mode"] == "세 겹(A2)"
    assert {k: v["same"] for k, v in both["counts"].items()} == dict.fromkeys(("기대 = D", "기대 = C", "C = D"), len(VECTORS))
    old = [{k: v for k, v in r.items() if k != "call_sha256"} for r in rows]
    no_call = NC.compare(VECTORS, d_rows=rows, c_rows=old, require_both=True)
    assert not no_call["ok"] and any("호출 지문" in p for p in no_call["problems"])


def test_옛_꼴_영수증_03k_를_그대로_읽는다() -> None:
    """호출 지문 칸이 없는 10-22 의 03k 도 읽고 대조한다 — 기존 영수증은 바꾸지 않는다(201 I-04)."""
    head, rows = NC.read_jsonl(REPO / "docs/dev_log/2026-10-20-설계복구/03k_추출대조_D.jsonl")
    assert rows and all(set(r) == set(NC.ROW_KEYS_V1) for r in rows)
    res = NC.compare(VECTORS, d_rows=rows)
    assert res["ok"] and res["mode"] == "단독 점검" and head["vector_sha256"] == NC.VECTOR_SHA256


def test_명령줄의_세_겹_인자(tmp_path: Path) -> None:
    out = tmp_path / "d.jsonl"
    assert NC.main(["run", "--out", str(out)]) == 0
    assert NC.main(["compare", "--d", str(out), "--require-both"]) == 1                 # C 가 없다
    assert NC.main(["compare", "--d", str(out), "--c", str(out), "--require-both", "--out", str(tmp_path / "s.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["mode"] == "세 겹(A2)"


def test_벡터_파일의_바이트가_다르면_읽지_않는다(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    p.write_bytes((REPO / NC.VECTOR_REL).read_bytes() + b" ")
    with pytest.raises(NC.ConformanceError, match="sha256"):
        NC.load_vectors(p)


def test_산출_파일에_CR_이_있으면_읽지_않는다(tmp_path: Path) -> None:
    out = tmp_path / "d.jsonl"
    NC.write_jsonl(out, NC.head_for("x:y", "0" * 64, NC.VECTOR_REL, NC.VECTOR_SHA256), NC.run(NE.extract, VECTORS))
    crlf = tmp_path / "crlf.jsonl"
    crlf.write_bytes(out.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(NC.ConformanceError, match="CR"):
        NC.read_jsonl(crlf)
