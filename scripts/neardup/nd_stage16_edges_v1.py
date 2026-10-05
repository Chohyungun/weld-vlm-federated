"""16. 판본 있는 근사 중복 간선 `neardup_edges_v1` — 얼린 규칙 `X2_v1` 을 이미 단 특징에 적용한다(계산은 새로 하지 않는다).

    uv run python -X utf8 scripts/neardup/nd_stage16_edges_v1.py --src _workspace/2026-09-21-neardup \\
        --out data/interim/neardup_edges_v1

규칙 재설계안(`docs/dev_log/2026-09-24-후속정리/08_규칙재설계_설계안_A.md`)의 단계 4 · 5 와 §5 의 2 를 따른다.
- **간선 = 자동 제외를 지난 짝**이다. 얼린 식(`stage12_frozen_rule.json` 의 `고른_자동제외.식`, 식 문자열의 sha256 이
  같은 파일의 `식의_sha256` 과 같아야 한다)을 짝마다 평가한다. 식을 그대로 평가한 값과 같은 식을 비교문으로 풀어 쓴
  값이 한 짝이라도 다르면 멈춘다.
- **값이 없는 짝은 간선이 아니다 — 판정 못 함으로 따로 적는다**(단계 4 — "모르는 것을 뺄 수 없다"). 다만 빠진 값이
  식의 값을 바꾸지 못하는 짝(예: 문자 띠가 없지만 `blk_med >= 0.2` 라 안전장치가 걸리지 않는 짝)은 판정한다 —
  빠진 값을 양 끝으로 바꿔 넣어 식의 값이 같을 때만.
- **v1 은 큰 이동 후보(G4)를 덮지 않는다** — 그 짝에는 규칙의 입력이 없다(3e 미실행). 메타에 적는다.
- 간선의 모든 id 가 RT 동결 매니페스트(`data/interim/manifest_v1/manifest.csv`)에 있어야 한다 — 없으면 멈춘다.
- 산출 폴더가 이미 있으면 멈춘다(다시 만들지 않는다 — 불변조건 1-6). 쓰고 나서 계약서(`SNAPSHOT.sha256`)를 남기고
  파일을 읽기 전용으로 둔다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.frozen_guard import assert_writable  # noqa: E402

#: 짝 집합 — 세 계열의 특징 파일(이동 · 블록 중앙값 · 문자 띠)과 후보 파일. 이름은 재설계안 08 의 입력 표와 같다.
SETS: dict[str, dict[str, list[str]]] = {
    "cross_eval_vs_trainval": {
        "shift": ["feat3_cross_eval_vs_trainval.csv", "feat3_rest_cross_eval_vs_trainval.csv", "feat3_new_cross.csv"],
        "blk": ["feat5_cross_eval_vs_trainval.csv", "feat5_rest_cross_eval_vs_trainval.csv", "feat5_new_cross.csv"],
        "txt": ["feat7_cross_eval_vs_trainval.csv", "feat7_rest_cross_eval_vs_trainval.csv", "feat7_new_cross.csv"],
        "cand": ["cand_cross_eval_vs_trainval.csv", "bpcand_new_cross.csv"],
    },
    "within_eval": {"shift": ["feat3_within_eval.csv"], "blk": ["feat5_within_eval.csv"], "txt": [],
                    "cand": ["cand_within_eval.csv"]},
    "within_trainval": {"shift": ["feat3d_within_trainval.csv"], "blk": ["feat5_within_trainval.csv"], "txt": [],
                        "cand": ["cand_within_trainval.csv"]},
}
#: 덮지 않는 후보 — 규칙의 입력이 없다
NOT_COVERED = {"G4_large_shift_cross": "shiftcand_cross.csv"}
FROZEN_RULE = "stage12_frozen_rule.json"
EXPECTED_EXPR_SHA256_PREFIX = "79662c2e"        # 재설계 기록(10번 · 11번)이 인용한 얼린 식의 지문 앞자리
VARS = ("rd_dx", "rd_dy", "blk_med", "txt_gap", "txt_energy")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v                  # NaN 은 값이 없는 것이다


def pair_key(r: dict) -> tuple[str, str]:
    a, b = r["image_id_a"], r["image_id_b"]
    return (a, b) if a <= b else (b, a)


def read_rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def rule_explicit(v: dict) -> bool:
    """얼린 식을 비교문으로 풀어 쓴 것 — 식 평가와 맞대는 둘째 구현."""
    if max(abs(v["rd_dx"]), abs(v["rd_dy"])) > 48:
        return False
    if v["blk_med"] < 0.15:
        return False
    guard = v["txt_gap"] >= 0.3 and v["txt_energy"] >= 1.0 and v["blk_med"] < 0.2
    return not guard


def decide(v: dict, expr_code) -> tuple[bool | None, str]:
    """(간선 여부, 사유). 값이 모자라 정할 수 없으면 (None, 사유)."""
    if v.get("rd_dx") is None or v.get("rd_dy") is None:
        return None, "no_shift"
    if v.get("blk_med") is None:
        return None, "no_blk_med"
    missing = [k for k in ("txt_gap", "txt_energy") if v.get(k) is None]
    trials = [dict(v)] if not missing else [{**v, **{k: e for k in missing}} for e in (-1e18, 1e18)]
    if missing:                                   # 두 값의 엇갈린 조합도 본다
        trials += [{**v, missing[0]: -1e18, **({missing[1]: 1e18} if len(missing) > 1 else {})},
                   {**v, missing[0]: 1e18, **({missing[1]: -1e18} if len(missing) > 1 else {})}]
    outs_e = {bool(eval(expr_code, {"__builtins__": {}}, {"max": max, "abs": abs, **t})) for t in trials}
    outs_x = {rule_explicit(t) for t in trials}
    if outs_e != outs_x:
        raise SystemExit(f"식 평가와 풀어 쓴 비교문이 다르다: {v}")
    if len(outs_e) != 1:
        return None, "needs_txt"
    return outs_e.pop(), ("txt_missing_irrelevant" if missing else "full")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=REPO / "data/interim/manifest_v1/manifest.csv")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"산출 폴더가 이미 있다 — 다시 만들지 않는다: {args.out}")
    assert_writable(args.out.parent, what="간선 판본의 부모")

    frozen = json.loads((args.src / FROZEN_RULE).read_text(encoding="utf-8"))
    expr = frozen["고른_자동제외"]["식"]
    expr_sha = hashlib.sha256(expr.encode("utf-8")).hexdigest()
    if expr_sha != frozen["식의_sha256"] or not expr_sha.startswith(EXPECTED_EXPR_SHA256_PREFIX):
        raise SystemExit(f"얼린 식의 지문이 다르다: {expr_sha[:16]}…")
    code = compile(expr, "<X2_v1>", "eval")

    inputs, per_set, edges, undecided = {}, {}, set(), []
    seen: dict[tuple[str, str], str] = {}
    for tag, spec in SETS.items():
        vals: dict[tuple[str, str], dict] = {}
        cand: set[tuple[str, str]] = set()
        for kind, cols in (("shift", ("rd_dx", "rd_dy")), ("blk", ("blk_med",)), ("txt", ("txt_gap", "txt_energy"))):
            for name in spec[kind]:
                p = args.src / name
                rows = read_rows(p)
                inputs[name] = {"sha256": sha256_file(p), "rows": len(rows)}
                for r in rows:
                    k = pair_key(r)
                    d = vals.setdefault(k, {})
                    for c in cols:
                        x = num(r.get(c))
                        if c in d and d[c] != x:
                            raise SystemExit(f"같은 짝의 {c} 값이 파일마다 다르다 ({tag})")
                        d[c] = x
        for name in spec["cand"]:
            p = args.src / name
            rows = read_rows(p)
            inputs[name] = {"sha256": sha256_file(p), "rows": len(rows)}
            cand |= {pair_key(r) for r in rows}
        allk = cand | set(vals)
        c = {"pairs": len(allk), "edges": 0, "no_edge": 0}
        for k in sorted(allk):
            if k[0] == k[1]:
                raise SystemExit("자기 자신과의 짝이 있다")
            if k in seen:
                raise SystemExit(f"한 짝이 두 집합에 있다: {seen[k]} · {tag}")
            seen[k] = tag
            v = {x: vals.get(k, {}).get(x) for x in VARS}
            ok, why = decide(v, code)
            if ok is None:
                undecided.append((k[0], k[1], tag, why))
                c[f"undecided:{why}"] = c.get(f"undecided:{why}", 0) + 1
            elif ok:
                edges.add(k)
                c["edges"] += 1
                if why != "full":
                    c["edges_txt_missing_irrelevant"] = c.get("edges_txt_missing_irrelevant", 0) + 1
            else:
                c["no_edge"] += 1
        per_set[tag] = c
    for tag, name in NOT_COVERED.items():
        p = args.src / name
        inputs[name] = {"sha256": sha256_file(p), "rows": len(read_rows(p))}

    with args.manifest.open(encoding="utf-8", newline="") as fh:
        known = {r["image_id"] for r in csv.DictReader(fh)}
    unknown = {i for k in edges for i in k} - known
    if unknown:
        raise SystemExit(f"간선에 매니페스트에 없는 id 가 {len(unknown)} 개 있다")

    args.out.mkdir(parents=True)
    files = {}
    rows = sorted(edges)
    data = ("a_id,b_id\n" + "".join(f"{a},{b}\n" for a, b in rows)).encode("utf-8")
    (args.out / "edges.csv").write_bytes(data)
    und = ("a_id,b_id,set,reason\n" + "".join(f"{a},{b},{t},{w}\n" for a, b, t, w in sorted(undecided))).encode("utf-8")
    (args.out / "undecided.csv").write_bytes(und)
    meta = {
        "name": "neardup_edges_v1", "made_at": time.strftime("%Y-%m-%d %H:%M"),
        "rule": {"name": "X2_v1", "expr": expr, "expr_sha256": expr_sha,
                 "frozen_file": FROZEN_RULE, "frozen_file_sha256": sha256_file(args.src / FROZEN_RULE),
                 "dev_pairs": frozen.get("개발_쌍"), "dev_performance": frozen["고른_자동제외"].get("개발_성능")},
        "edge_meaning": "자동 제외를 지난 짝(같은 장면으로 보는 짝). 같은 묶음 간선은 싣지 않는다 — 읽는 쪽이 같은 묶음으로 잇는다",
        "undecided_meaning": "규칙의 입력이 모자라 간선으로도 아님으로도 정하지 않은 짝 — 간선이 아니다(재설계안 08 단계 4)",
        "not_covered": {k: {"file": v, "pairs": inputs[v]["rows"], "why": "규칙의 입력(이동 · 블록 중앙값 · 문자 띠)이 없다 — 3e 미실행"}
                        for k, v in NOT_COVERED.items()},
        "holdout_scoring": "하지 않았다 — 보류 120 쌍의 판독은 끝났고 얼린 식을 보류에 대고 채점하는 일은 남아 있다(11번 (ㄷ))",
        "manifest": {"path": "data/interim/manifest_v1/manifest.csv", "sha256": sha256_file(args.manifest)},
        "per_set": per_set, "edges_total": len(rows), "undecided_total": len(undecided),
        "edge_image_ids": len({i for k in rows for i in k}), "inputs": inputs,
        "script": "scripts/neardup/nd_stage16_edges_v1.py",
    }
    (args.out / "META.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                                        newline="\n")
    for n in ("edges.csv", "undecided.csv", "META.json"):
        files[n] = sha256_file(args.out / n)
    digest = hashlib.sha256("".join(files.values()).encode()).hexdigest()
    (args.out / "SNAPSHOT.sha256").write_text(
        "".join(f"{h}  {n}\n" for n, h in files.items()) + f"# snapshot_digest {digest}\n", encoding="utf-8",
        newline="\n")
    for n in list(files) + ["SNAPSHOT.sha256"]:
        os.chmod(args.out / n, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps({"edges": len(rows), "undecided": len(undecided), "edges_sha256": files["edges.csv"],
                      "snapshot_digest": digest, "per_set": per_set}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
