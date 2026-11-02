"""판정 v7 결정적 grounded 의 동치 하니스 — 미니스펙 16번 7판 §13 보강(총괄 계약 5).

두 구현이 **같은 입력에서 같은 출력**을 내는지 행 단위로 본다. 정본은 평가 쪽 참조 구현(`evaluation/grounded_v7_ref.py`)이고, 정식
구현(외부 검토 도구가 `corpus/production/judge_policy.py` 로 옮긴 것)이 같은 벡터를 다시 내야 재탐침 신호가 나간다.

    # 벡터 만들기 — 참조 구현으로(사례 묶음은 합성 사례만, 여럿이면 --cases 를 되풀이)
    python -X utf8 -B -m scripts.probe.check_grounded_v7_equivalence build --cases <cases_v6_g150.json> \\
        --cases evaluation/vectors/grounded_v7_structural_cases.json --out evaluation/vectors/grounded_v7_vectors.jsonl
    # 대조 — 모듈 이름 또는 파일 경로, 함수 이름이 다르면 --map 으로
    python -X utf8 -B -m scripts.probe.check_grounded_v7_equivalence check --vectors evaluation/vectors/grounded_v7_vectors.jsonl \\
        --candidate corpus.production.judge_grounded_v7
    # 정식 구현은 정본의 바이트 사본이라 함수 이름이 같다 — 이름이 다른 후보는 네 칸 모두 --map 으로 맞댄다
    #   --map grounded_codes=<이름> --map contract_codes=<이름> --map attribute_codes=<이름> --map normal_codes=<이름>
    # 정본이 바뀐 뒤 핀 다시 걸기 — 행(입력)은 그대로 두고 기대값을 지금 정본으로 다시 낸다. 앞 핀과 바뀐 기대값은 meta 에 남는다
    python -X utf8 -B -m scripts.probe.check_grounded_v7_equivalence rebase --vectors evaluation/vectors/grounded_v7_vectors.jsonl \\
        --out evaluation/vectors/grounded_v7_vectors.jsonl --note <무엇이 바뀌었나>
    # 행 더하기 — 사례 묶음에서 벡터에 없는 id 의 행만 덧붙인다. 기존 행과 기대값은 그대로다. 더한 행 · 원천 · 까닭은 meta 에 남는다
    python -X utf8 -B -m scripts.probe.check_grounded_v7_equivalence extend --vectors evaluation/vectors/grounded_v7_vectors.jsonl \\
        --cases evaluation/vectors/grounded_v7_structural_cases.json --out evaluation/vectors/grounded_v7_vectors.jsonl --note <무엇을 더하나>

후보의 함수 계약: `f(task: dict, text: str) -> list[str]` — 사유 코드 목록, 빈 목록이 통과. 비교는 정렬한 목록으로 한다. 출력에는
행의 id(합성 사례 id) · 칸 · 기대 · 얻은 값만 싣는다. 불일치가 하나라도 있으면 종료 코드 1.

**핀은 정본 바이트 하나다.** 벡터 meta 의 `reference_sha256`(결정적 grounded 정본의 LF 바이트)이 지금 정본과 다르면 `stale` 에
적고 — 불일치가 없어도 — 종료 코드 2 다(벡터를 다시 만든다). 정본이 부르는 의존(`judge_policy` · `generation_policy`)의 지문은
`deps_at_build` 에 **기록으로만** 남긴다 — 다르면 `deps_changed` 에 적을 뿐 종료 코드를 바꾸지 않는다. 의존이 판정 동작을 바꾸면
벡터 대조의 불일치(종료 1)로 드러나므로, 생성 정책이 바뀔 때마다 벡터를 다시 봉인하지 않는다(게이트 72, 2026-10-13 — 앞 판은
두 의존 지문까지 핀으로 두어 생성 v2 와 만나 깨졌다).

**정본이 바뀌면 `rebase` 로 핀을 다시 건다.** 행(합성 사례의 입력)은 그대로 두고 기대값을 지금 정본으로 다시 낸다. 앞 핀(정본
바이트 · 벡터 파일의 해시 · 판)과 그때 바뀐 기대값(앞 값 · 새 값)은 meta 의 `reference_history` 에 쌓는다 — 당시 값을 덮지 않는다.

**행은 `extend` 로만 더한다.** 사례 묶음에서 벡터에 없는 id 의 행만 덧붙이고 기대값은 지금 정본의 출력이다. 벡터에 이미 있는
id 는 입력(과제 · 글 · 무리)이 같아야 하고, 정본이 벡터의 핀과 다르면 먼저 `rebase` 한다. 더한 행의 id · 원천 묶음의 해시(앞 해시 ·
새 해시) · 까닭은 meta 의 `extensions` 에 쌓는다.

**벡터는 독립 증거가 아니다.** G150 150 행의 기대값은 정본의 출력을 그대로 적은 것이고, 구조 사례의 기대값은 정본을 보며 정했다.
벡터가 고정하는 것은 "두 구현이 같은 동작" 이지 "정본이 옳다" 가 아니다 — 옳음은 사람 판독 · 조정 집합 · 최종 집합이 잰다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("grounded_codes", "contract_codes", "attribute_codes", "normal_codes")
SCHEMA = "grounded_v7_vectors"
VERSION = 6
"""판 6 — meta 에 `extensions`(더한 행 · 원천 · 까닭)가 붙는다. 판 5 에서 `reference_history`(앞 핀 · 바뀐 기대값)가, 판 4 까지는 둘 다
없었다."""
REF_MODULE = "evaluation.grounded_v7_ref"


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_candidate(module: str | None, file: str | None):
    if file:
        spec = importlib.util.spec_from_file_location("grounded_candidate", file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, lf_sha256(Path(file))
    mod = importlib.import_module(module or REF_MODULE)
    return mod, lf_sha256(Path(mod.__file__))


PINNED = ("reference_sha256",)
"""벡터를 묶는 지문 — 결정적 grounded 정본의 바이트뿐이다."""
DEPS = {"judge_policy_sha256": "corpus/production/judge_policy.py",
        "generation_policy_sha256": "corpus/production/generation_policy.py"}
"""정본이 부르는 의존 — 기록만 한다(핀이 아니다)."""


def deps_now() -> dict:
    return {k: lf_sha256(ROOT / rel) for k, rel in DEPS.items()}


def read_vectors(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines or "meta" not in lines[0] or lines[0]["meta"].get("schema") != SCHEMA:
        raise SystemExit("벡터 파일의 첫 줄이 grounded_v7_vectors 의 meta 가 아니다")
    meta, rows = lines[0]["meta"], lines[1:]
    if len(rows) != meta.get("n_rows"):
        raise SystemExit(f"벡터 행 수 {len(rows)} 가 meta 의 {meta.get('n_rows')} 와 다르다")
    return meta, rows


def outputs(mod, mapping: dict, task: dict, text: str) -> dict:
    return {f: sorted(getattr(mod, mapping.get(f, f))(task, text)) for f in FIELDS}


def build(cases_paths: list[Path], out: Path) -> dict:
    ref, ref_sha = load_candidate(REF_MODULE, None)
    rows, sources, seen = [], [], set()
    for cases_path in cases_paths:
        raw = cases_path.read_bytes()
        cases = json.loads(raw)["cases"]
        sources.append({"name": cases_path.name, "sha256": hashlib.sha256(raw).hexdigest(), "n": len(cases)})
        for c in cases:
            if c["id"] in seen:
                raise SystemExit(f"사례 id 가 겹친다: {c['id']}")
            seen.add(c["id"])
            task, text = c["task"], c["candidate"]
            rows.append({"id": c["id"], "group": c["group"], "row_class": ref.P.row_class(task), "task": task,
                         "candidate": text, "expected": outputs(ref, {}, task, text)})
    meta = {"schema": SCHEMA, "version": VERSION, "reference_module": REF_MODULE, "reference_sha256": ref_sha,
            "pinned": list(PINNED), "deps_at_build": deps_now(), "reference_history": [],
            "cases": sources, "cases_kind": "G150 v6 합성 사례(조정에 쓰인 집합) + 구조 사례(구역별 수 · 합 · 모순 · 정상 행 음성 · 보강 6)",
            "caveat": "독립 증거가 아니다 — G150 행의 기대값은 정본 출력이고 구조 사례의 기대값은 정본을 보며 정했다. 두 구현의 동치만 고정한다",
            "fields": list(FIELDS), "n_rows": len(rows), "builder_sha256": lf_sha256(Path(__file__))}
    _write(out, meta, rows)
    return {"out": str(out), "n_rows": len(rows), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "reference_sha256": ref_sha}


def _write(out: Path, meta: dict, rows: list[dict]) -> None:
    body = [json.dumps({"meta": meta}, ensure_ascii=False, sort_keys=True)]
    body += [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(("\n".join(body) + "\n").encode("utf-8"))


def rebase(vectors: Path, out: Path, note: str) -> dict:
    """정본이 바뀐 뒤 같은 행으로 핀을 다시 건다 — 기대값은 지금 정본의 출력, 앞 핀과 바뀐 기대값은 `reference_history` 에.

    정본이 그대로면 다시 걸 핀이 없다(거부). 행의 입력 · id · 무리는 바꾸지 않는다."""
    raw = Path(vectors).read_bytes()
    meta, rows = read_vectors(vectors)
    ref, ref_sha = load_candidate(REF_MODULE, None)
    if ref_sha == meta["reference_sha256"]:
        raise SystemExit("정본 바이트가 벡터의 핀과 같다 — 다시 걸 핀이 없다")
    if not note.strip():
        raise SystemExit("--note 가 비었다 — 무엇이 바뀌었는지 적는다")
    changed, new_rows = [], []
    for r in rows:
        exp = outputs(ref, {}, r["task"], r["candidate"])
        changed += [{"id": r["id"], "field": f, "before": sorted(r["expected"][f]), "after": exp[f]}
                    for f in FIELDS if sorted(r["expected"][f]) != exp[f]]
        new_rows.append({**r, "expected": exp})
    history = [*meta.get("reference_history", []),
               {"version": meta["version"], "reference_sha256": meta["reference_sha256"],
                "vectors_sha256": hashlib.sha256(raw).hexdigest(), "changed": changed, "note": note}]
    new_meta = {**meta, "version": VERSION, "reference_sha256": ref_sha, "deps_at_build": deps_now(),
                "builder_sha256": lf_sha256(Path(__file__)), "reference_history": history}
    _write(out, new_meta, new_rows)
    return {"out": str(out), "n_rows": len(new_rows), "sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest(),
            "reference_sha256": ref_sha, "previous_reference_sha256": meta["reference_sha256"], "changed": changed}


def extend(vectors: Path, cases_path: Path, out: Path, note: str) -> dict:
    """사례 묶음에서 벡터에 없는 id 의 행만 덧붙인다 — 기대값은 지금 정본의 출력이고, 기존 행 · 기대값은 그대로다.

    벡터에 이미 있는 id 는 입력이 같아야 한다(다르면 거부). 정본이 벡터의 핀과 다르면 먼저 `rebase` 한다(거부). 더할 행이 없거나
    까닭(--note)이 비면 거부한다."""
    raw = Path(vectors).read_bytes()
    meta, rows = read_vectors(vectors)
    ref, ref_sha = load_candidate(REF_MODULE, None)
    if ref_sha != meta["reference_sha256"]:
        raise SystemExit("정본이 벡터의 핀과 다르다 — 먼저 rebase 로 핀을 다시 건다")
    if not note.strip():
        raise SystemExit("--note 가 비었다 — 무엇을 더하는지 적는다")
    craw = Path(cases_path).read_bytes()
    cases = json.loads(craw)["cases"]
    have = {r["id"]: r for r in rows}
    added = []
    for c in cases:
        if c["id"] in have:
            r = have[c["id"]]
            if (r["task"], r["candidate"], r["group"]) != (c["task"], c["candidate"], c["group"]):
                raise SystemExit(f"같은 id 의 입력이 다르다: {c['id']}")
            continue
        added.append({"id": c["id"], "group": c["group"], "row_class": ref.P.row_class(c["task"]), "task": c["task"],
                      "candidate": c["candidate"], "expected": outputs(ref, {}, c["task"], c["candidate"])})
    if not added:
        raise SystemExit("더할 행이 없다")
    name = Path(cases_path).name
    source = {"name": name, "sha256": hashlib.sha256(craw).hexdigest(), "n": len(cases)}
    before = next((x for x in meta["cases"] if x["name"] == name), None)
    sources = [source if x["name"] == name else x for x in meta["cases"]] + ([] if before else [source])
    extensions = [*meta.get("extensions", []),
                  {"cases": name, "sha256": source["sha256"], "previous_sha256": before["sha256"] if before else None,
                   "vectors_sha256": hashlib.sha256(raw).hexdigest(), "added": [r["id"] for r in added], "note": note}]
    new_meta = {**meta, "version": VERSION, "cases": sources, "extensions": extensions, "n_rows": len(rows) + len(added),
                "deps_at_build": deps_now(), "builder_sha256": lf_sha256(Path(__file__))}
    _write(out, new_meta, rows + added)
    return {"out": str(out), "n_rows": len(rows) + len(added), "added": [r["id"] for r in added],
            "sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest(), "reference_sha256": ref_sha}


def check(vectors: Path, module: str | None, file: str | None, mapping: dict) -> dict:
    meta, rows = read_vectors(vectors)
    mod, sha = load_candidate(module, file)
    missing = [f for f in FIELDS if not callable(getattr(mod, mapping.get(f, f), None))]
    if missing:
        raise SystemExit(f"후보에 함수가 없다: {[mapping.get(f, f) for f in missing]}")
    bad = []
    for r in rows:
        got = outputs(mod, mapping, r["task"], r["candidate"])
        for f in FIELDS:
            if got[f] != sorted(r["expected"][f]):
                bad.append({"id": r["id"], "field": f, "expected": r["expected"][f], "got": got[f]})
    ref_now = lf_sha256(Path(importlib.import_module(REF_MODULE).__file__))
    stale = ["reference_sha256"] if meta.get("reference_sha256") != ref_now else []
    built = meta.get("deps_at_build") or {k: meta.get(k) for k in DEPS}       # 판 3 까지는 meta 꼭대기에 있었다
    deps_changed = sorted(k for k, v in deps_now().items() if built.get(k) != v)
    return {"vectors_sha256": hashlib.sha256(Path(vectors).read_bytes()).hexdigest(), "n_rows": len(rows),
            "candidate": file or module or REF_MODULE, "candidate_sha256": sha,
            "reference_sha256_at_build": meta["reference_sha256"], "stale": stale, "deps_changed": deps_changed,
            "mismatches": bad, "equal": not bad}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cases", type=Path, action="append", required=True)
    b.add_argument("--out", type=Path, required=True)
    rb = sub.add_parser("rebase")
    rb.add_argument("--vectors", type=Path, required=True)
    rb.add_argument("--out", type=Path, required=True)
    rb.add_argument("--note", required=True, help="무엇이 바뀌었나(예: 보강 7 — 절 범위 · 속성 판독 불가)")
    ex = sub.add_parser("extend")
    ex.add_argument("--vectors", type=Path, required=True)
    ex.add_argument("--cases", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--note", required=True, help="무엇을 더하나(예: 보강 7 — 배분 서술 '각각 N개씩')")
    c = sub.add_parser("check")
    c.add_argument("--vectors", type=Path, required=True)
    c.add_argument("--candidate", default=None, help="모듈 이름(기본: 평가 쪽 참조 구현)")
    c.add_argument("--candidate-file", default=None, help="모듈 파일 경로(추적 밖 사본을 맞댈 때)")
    c.add_argument("--map", action="append", default=[], help="칸=함수 이름(예: grounded_codes=deterministic_grounded)")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    if a.cmd == "build":
        print(json.dumps(build(a.cases, a.out), ensure_ascii=False))
        return 0
    if a.cmd == "rebase":
        print(json.dumps(rebase(a.vectors, a.out, a.note), ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "extend":
        print(json.dumps(extend(a.vectors, a.cases, a.out, a.note), ensure_ascii=False, indent=1))
        return 0
    mapping = dict(x.split("=", 1) for x in a.map)
    res = check(a.vectors, a.candidate, a.candidate_file, mapping)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 1 if not res["equal"] else (2 if res["stale"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
