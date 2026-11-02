"""서술 추출 적합성 — 벡터 실행기와 세 겹 대조(03 2판 §5b · 04 §1-6 · 195 M3).

학습 쪽 점검(C)과 채점 쪽 추출기(D)는 **같은 벡터 파일 하나**(`VECTOR_REL`, sha256 `VECTOR_SHA256`)를 바이트로 읽어 벡터마다
같은 다섯 값을 내야 한다. 이 모듈은 둘을 한 꼴로 맞댄다.

- `run` — 추출 함수(`모듈:이름`)를 벡터마다 불러 JSONL 하나를 쓴다. 머리 줄에 벡터 경로 · sha256 · 구현 식별자 · 구현 LF sha256 ·
  파이썬 판, 벡터마다 한 줄에 사례 id · 입력 sha256 · 호출 지문 · 다섯 값 · 추출문 sha256(UTF-8). `gen_stop` 열쇠가 없는 벡터는
  인자를 빼고 부른다. 입력 sha256(`input_sha256`)은 `text` 만 묶는다. 호출 지문(`call_sha256`)은 `text` 와 종료 사유 칸의 있음 ·
  값을 함께 묶는다(201 I-04) — 같은 글에 다른 종료 사유를 넘기면 입력 sha256 은 같고 호출 지문은 갈린다.
- `compare` — 기대(벡터 파일) · D · C 의 JSONL 을 벡터마다 다섯 값 + 추출문 sha256 으로 맞대 `기대 = D` · `기대 = C` · `C = D` 를
  센다. 값은 **자료형까지** 맞댄다(`0` · `1` 과 `False` · `True` 는 다르다 — 04 §7-10). 입력 sha256 · 호출 지문이 다르면 전달 사고
  (다른 입력을 넘겼다)로 표시한다. 호출 지문이 없는 옛 꼴 산출(10-22 의 03k)도 읽고, 그때는 호출 지문을 맞대지 않는다.
  `--require-both`(세 겹 — A2 입구, 201 I-03)는 C · D 가 둘 다 있고 둘 다 호출 지문을 싣고 세 줄이 모두 전부 같을 때만 통과다.
  한쪽만 준 대조는 `단독 점검` 이다 — 세 겹 통과로 쓰지 않는다.

**실패를 돌려준다.** 필수 칸이 없거나 · 벡터 sha 가 다르거나 · 사례가 빠지거나 겹치거나 · 하나라도 어긋나면 종료 코드 1 이다
(03g 는 문제를 적고도 0 을 돌려주었다 — 195 §4). 어긋남의 원인(전달 · 계산 · 계약 미정)을 가리는 것은 사람이다 — 기대를 구현에 맞춰
고치지 않는다(03 2판 §5b 의 5).

    python -X utf8 -B -m evaluation.narrative_conformance run --impl evaluation.narrative_extract:extract --out d.jsonl
    python -X utf8 -B -m evaluation.narrative_conformance compare --d d.jsonl [--c c.jsonl] [--out summary.json]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import platform
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VECTOR_REL = "evaluation/vectors/narrative_extract.v1.json"
VECTOR_SHA256 = "e7c401fcb81b6fea85e80ff4c300fa6de1b0aa20b9e1267a1eeaf5a440b33964"
"""공통 벡터 정본의 바이트 sha256 — dev_log 의 `03f-2_추출벡터_D.json` 과 같은 바이트다(04 §7-5 ⑤). C · D 시험 · 등록이 이 값 하나를 핀한다."""

KIND = "narrative_extract_conformance"
FIELDS = ("state", "narrative", "front_text", "stop_kept", "upstream_reason")
ROW_KEYS = ("id", "input_sha256", "call_sha256", *FIELDS, "narrative_sha256")
ROW_KEYS_V1 = ("id", "input_sha256", *FIELDS, "narrative_sha256")
"""호출 지문 칸이 없는 옛 꼴(10-22 의 03k) — 읽되 호출 지문은 맞대지 않는다."""
HEAD_KEYS = ("kind", "vector_path", "vector_sha256", "impl_id", "impl_lf_sha256", "python")


class ConformanceError(ValueError):
    """벡터 · 산출 파일이 계약의 꼴이 아니다."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def text_sha256(s: str | None) -> str | None:
    return None if s is None else sha256_bytes(s.encode("utf-8"))


def call_sha256(vector: dict) -> str:
    """호출 입력 지문 — 생성문 `text` 와 종료 사유 칸의 **있음 · 값**을 정본 JSON(키 정렬 · 여백 없음 · UTF-8)으로 묶은 sha256(201 I-04)."""
    body = {"text": vector["text"], "gen_stop_present": "gen_stop" in vector, "gen_stop": vector.get("gen_stop")}
    return sha256_bytes(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def same(a, b) -> bool:
    """값과 자료형이 함께 같은가 — `0` · `1` 과 `False` · `True` 를 다르게 본다(04 §7-10)."""
    return type(a) is type(b) and a == b


def load_vectors(path: str | Path | None = None, *, expect_sha256: str = VECTOR_SHA256) -> list[dict]:
    """벡터 파일을 **바이트로** 읽는다 — 글 모드는 줄끝을 바꿀 수 있다. sha256 이 다르면 멈춘다."""
    p = Path(path) if path is not None else REPO / VECTOR_REL
    raw = p.read_bytes()
    got = sha256_bytes(raw)
    if got != expect_sha256:
        raise ConformanceError(f"벡터 파일의 sha256 이 {got} 다 — 정본은 {expect_sha256}")
    doc = json.loads(raw.decode("utf-8"))
    vectors = doc.get("vectors")
    if not isinstance(vectors, list) or not vectors:
        raise ConformanceError("vectors 가 비었다")
    ids = [v.get("id") for v in vectors]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise ConformanceError("사례 id 가 비었거나 겹친다")
    return vectors


def call(extract: Callable, vector: dict):
    """벡터 하나를 부른다 — `gen_stop` 열쇠가 없으면 인자를 뺀다(칸 없음), 있으면 그 값(null 이면 None)을 넘긴다."""
    if "gen_stop" in vector:
        return extract(vector["text"], vector["gen_stop"])
    return extract(vector["text"])


def _as_fields(result) -> dict:
    """구현의 반환을 다섯 값으로 — `as_dict()` 가 있는 객체 · 사전 · 다섯 칸 튜플을 받는다."""
    if hasattr(result, "as_dict"):
        d = result.as_dict()
    elif isinstance(result, dict):
        d = result
    elif isinstance(result, (tuple, list)) and len(result) == len(FIELDS):
        d = dict(zip(FIELDS, result, strict=True))
    else:
        raise ConformanceError(f"반환 꼴을 모른다: {type(result).__name__}")
    missing = [k for k in FIELDS if k not in d]
    if missing:
        raise ConformanceError(f"반환에 칸이 없다: {missing}")
    return {k: d[k] for k in FIELDS}


def row_of(vector: dict, fields: dict) -> dict:
    return {"id": vector["id"], "input_sha256": text_sha256(vector["text"]), "call_sha256": call_sha256(vector), **fields,
            "narrative_sha256": text_sha256(fields["narrative"])}


def expected_rows(vectors: Sequence[dict]) -> list[dict]:
    return [row_of(v, {k: v["expect"][k] for k in FIELDS}) for v in vectors]


def run(extract: Callable, vectors: Sequence[dict]) -> list[dict]:
    return [row_of(v, _as_fields(call(extract, v))) for v in vectors]


def resolve(spec: str) -> tuple[Callable, str, str]:
    """`모듈:이름` → (함수, 식별자, 구현 파일의 LF sha256)."""
    mod_name, _, attr = spec.partition(":")
    if not mod_name or not attr:
        raise ConformanceError(f"구현은 '모듈:이름' 꼴이다 — {spec!r}")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr)
    src = Path(inspect.getsourcefile(mod))
    return fn, f"{mod_name}:{attr}", sha256_bytes(src.read_bytes().replace(b"\r\n", b"\n"))


def head_for(impl_id: str, impl_sha: str, vector_path: str, vector_sha: str) -> dict:
    return {"kind": KIND, "vector_path": vector_path, "vector_sha256": vector_sha, "impl_id": impl_id,
            "impl_lf_sha256": impl_sha, "python": platform.python_version()}


def write_jsonl(path: str | Path, head: dict, rows: Sequence[dict]) -> None:
    lines = [json.dumps(head, ensure_ascii=False, sort_keys=True)]
    lines += [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]
    Path(path).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def read_jsonl(path: str | Path) -> tuple[dict, list[dict]]:
    """산출 JSONL — 머리 줄 + 벡터마다 한 줄. 칸이 다르면 멈춘다(빠진 칸을 빈 값으로 메우지 않는다)."""
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise ConformanceError(f"{path}: BOM 이나 CR 이 있다")
    lines = raw.decode("utf-8").split("\n")
    if lines[-1] != "":
        raise ConformanceError(f"{path}: 끝 줄바꿈이 없다")
    head, *rows = (json.loads(ln) for ln in lines[:-1])
    if set(head) != set(HEAD_KEYS) or head["kind"] != KIND:
        raise ConformanceError(f"{path}: 머리 줄의 칸이 {sorted(head)} 다")
    for r in rows:
        if set(r) not in (set(ROW_KEYS), set(ROW_KEYS_V1)):
            raise ConformanceError(f"{path}: 사례 줄의 칸이 다르다 — {sorted(r)}")
    return head, rows


def compare(vectors: Sequence[dict], d_rows: Sequence[dict] | None = None,
            c_rows: Sequence[dict] | None = None, *, require_both: bool = False) -> dict:
    """세 겹 대조. 각 쪽은 같은 사례를 한 번씩 가져야 한다 — 빠지거나 겹치면 어긋남으로 센다.

    `require_both` 이면(A2 입구 — 201 I-03) C · D 가 둘 다 있고, 둘 다 호출 지문을 싣고, `기대 = D` · `기대 = C` · `C = D` 셋이 모두
    벡터 전부에서 같아야 `ok` 다. 아니면 한쪽만 준 대조는 `단독 점검` 이라 적는다.
    """
    want = {r["id"]: r for r in expected_rows(vectors)}
    sides = {"D": d_rows, "C": c_rows}
    by_side: dict[str, dict[str, dict]] = {}
    problems: list[str] = []
    if require_both and (d_rows is None or c_rows is None):
        problems.append("세 겹 대조는 C · D 산출이 둘 다 있어야 한다")
    for name, rows in sides.items():
        if rows is None:
            continue
        ids = [r["id"] for r in rows]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        extra = sorted(set(ids) - set(want))
        lost = sorted(set(want) - set(ids))
        problems += [f"{name}: 겹친 사례 {dup}"] if dup else []
        problems += [f"{name}: 벡터에 없는 사례 {extra}"] if extra else []
        problems += [f"{name}: 빠진 사례 {lost}"] if lost else []
        if require_both and any("call_sha256" not in r for r in rows):
            problems.append(f"{name}: 호출 지문(call_sha256)이 없는 줄이 있다 — 세 겹 대조는 호출 입력까지 맞댄다")
        by_side[name] = {r["id"]: r for r in rows}
    mismatches: list[dict] = []

    def diff(a_name: str, a: dict, b_name: str, b: dict, vid: str) -> int:
        bad = 0
        for k in ("input_sha256", "call_sha256", *FIELDS, "narrative_sha256"):
            if k == "call_sha256" and (k not in a or k not in b):
                continue                                   # 옛 꼴 산출 — 호출 지문을 맞대지 않는다
            if not same(a.get(k), b.get(k)):
                bad = 1
                mismatches.append({"id": vid, "pair": f"{a_name}={b_name}", "field": k, a_name: a.get(k), b_name: b.get(k),
                                   "cause_hint": "전달(입력이 다르다)" if k in ("input_sha256", "call_sha256") else None})
        return bad

    counts: dict[str, list[int]] = {}
    for vid, w in want.items():
        for name in ("D", "C"):
            if name in by_side and vid in by_side[name]:
                ok = not diff("기대", w, name, by_side[name][vid], vid)
                counts.setdefault(f"기대 = {name}", [0, 0])
                counts[f"기대 = {name}"][0] += ok
                counts[f"기대 = {name}"][1] += 1
        if "D" in by_side and "C" in by_side and vid in by_side["D"] and vid in by_side["C"]:
            ok = not diff("C", by_side["C"][vid], "D", by_side["D"][vid], vid)
            counts.setdefault("C = D", [0, 0])
            counts["C = D"][0] += ok
            counts["C = D"][1] += 1
    summary = " · ".join(f"{k} {v[0]}/{v[1]}" for k, v in counts.items())
    ok = not problems and not mismatches and bool(counts)
    if require_both:
        ok = ok and all(counts.get(k, [0, 0]) == [len(want), len(want)] for k in ("기대 = D", "기대 = C", "C = D"))
        mode = "세 겹(A2)"
    else:
        mode = "세 겹" if len(by_side) == 2 else "단독 점검"
    return {"n_vectors": len(want), "counts": {k: {"same": v[0], "n": v[1]} for k, v in counts.items()},
            "summary": summary, "problems": problems, "mismatches": mismatches, "ok": ok, "mode": mode}


def _check_head(head: dict, vector_sha: str, where: str) -> list[str]:
    return [] if head["vector_sha256"] == vector_sha else [f"{where}: 머리 줄의 벡터 sha 가 {head['vector_sha256']} 다"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--impl", default="evaluation.narrative_extract:extract")
    r.add_argument("--vectors", default=VECTOR_REL)
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--vectors", default=VECTOR_REL)
    c.add_argument("--d", type=Path)
    c.add_argument("--c", type=Path)
    c.add_argument("--out", type=Path)
    c.add_argument("--require-both", action="store_true",
                   help="세 겹(A2 입구) — C · D 가 둘 다 있고 호출 지문을 싣고 세 줄이 모두 전부 같아야 통과")
    a = ap.parse_args(argv)
    vec_path = Path(a.vectors) if Path(a.vectors).is_absolute() else REPO / a.vectors
    try:
        vectors = load_vectors(vec_path)
        if a.cmd == "run":
            if a.out.exists():
                print(f"{a.out} 가 이미 있다 — 새 파일로만 쓴다", file=sys.stderr)
                return 2
            fn, impl_id, impl_sha = resolve(a.impl)
            rows = run(fn, vectors)
            write_jsonl(a.out, head_for(impl_id, impl_sha, VECTOR_REL, VECTOR_SHA256), rows)
            res = compare(vectors, d_rows=rows)
            print(f"{impl_id} · 기대 대조 {res['summary']}")
        else:
            heads = {}
            d_rows = c_rows = None
            if a.d:
                heads["D"], d_rows = read_jsonl(a.d)
            if a.c:
                heads["C"], c_rows = read_jsonl(a.c)
            if not heads:
                print("--d · --c 가운데 하나는 준다", file=sys.stderr)
                return 2
            res = compare(vectors, d_rows=d_rows, c_rows=c_rows, require_both=a.require_both)
            for name, h in heads.items():
                res["problems"] += _check_head(h, VECTOR_SHA256, name)
            res["heads"] = heads
            res["ok"] = res["ok"] and not res["problems"]
            print(res["summary"])
            if a.out:
                a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    except ConformanceError as exc:
        print(f"적합성: {exc}", file=sys.stderr)
        return 1
    for p in res["problems"]:
        print("  -", p, file=sys.stderr)
    for m in res["mismatches"]:
        print("  ≠", json.dumps(m, ensure_ascii=False), file=sys.stderr)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
