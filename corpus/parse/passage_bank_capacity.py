"""구절 은행 · 허용치 표의 생성 용량 (2026-10-07, v3).

    python -m corpus.parse.passage_bank_capacity \
        --bank corpus/parse/passage_bank_v3.jsonl \
        --full-bank <미추적 본문 은행> --registry corpus/parse/sources.yaml \
        --limits corpus/rules/limits_v0_pilot.csv \
        --out corpus/parse/passage_bank_v3.capacity.json

용량은 **상한**이다. 지금 생성기에 넘어가는 구절 수는 본문 은행에 소비 경계를 실제로 돌려 잰다
(본문 은행을 주지 않으면 null). 권리 결정 · 사람 검수 기록은 넣지 않는다 — 그런 기록은 아직 없다.
목표에 못 미치면 그 수를 적고, 반복으로 채운 수를 만들지 않는다.

- 원문 QA: 구절 수 × 구절당 질문 상한(생산 설정 `maximum_questions_per_passage`).
  지식 면 표지가 상한보다 적은 구절은 면 수만큼만 센 값도 함께 낸다(면 표지는 키워드 후보다).
  계보 묶음 하나는 train · val 가운데 한쪽에만 들어가야 하므로, 묶음 하나를 val 로 뺀 경우마다 train 쪽을 센다.
- 판정추론: 허용치 표의 조합(`skeleton_gen.enumerate_combos`) × 조합별 상한(`combo_cap`).
  같은 조합 안의 골격은 측정값 · 두께만 다른 같은 규칙이다 — 서로 다른 규칙 단위 수를 따로 적는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from corpus.parse.passage_bank import flag_names, load_jsonl, select_for_generation

QA_TARGET = 20_000
QA_MINIMUM = 10_000
REASONING_TARGET = 10_000
TIERS_STRICT = ("defect_specific", "weld_quality_general")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qa_capacity(rows: Sequence[dict], per_passage: int, minimum_distinct: int) -> dict:
    def block(sub: Sequence[dict]) -> dict:
        n = len(sub)
        cap = n * per_passage
        facet_bound = sum(min(per_passage, len(r.get("facets") or [])) for r in sub)
        return {
            "passages": n,
            "qa_upper": cap,
            "qa_upper_facet_bounded": facet_bound,
            "short_of_minimum_10000": max(0, QA_MINIMUM - cap),
            "short_of_target_20000": max(0, QA_TARGET - cap),
            "distinct_passages_short_of_minimum": max(0, minimum_distinct - n),
        }

    out: dict = {"per_passage": per_passage, "minimum_distinct_train_passages": minimum_distinct}
    variants = {
        "all_tiers": list(rows),
        "excluding_welding_general": [r for r in rows if r.get("relevance") in TIERS_STRICT],
        "excluding_historical_only": [r for r in rows if "historical_only" not in flag_names(r)],
        "excluding_policy_holds": [r for r in rows if not (r.get("source_policy_holds")
                                                           or r.get("context_dependency_policy_holds"))],
        "defect_specific_only": [r for r in rows if r.get("relevance") == "defect_specific"],
    }
    lineages = sorted({r["lineage_group_id"] for r in rows})
    for name, sub in variants.items():
        v = {"pool": block(sub), "by_doc": {}, "train_if_val_is": {}}
        for d in sorted({r["doc_id"] for r in sub}):
            v["by_doc"][d] = len([r for r in sub if r["doc_id"] == d])
        for lg in lineages:
            v["train_if_val_is"][lg] = block([r for r in sub if r["lineage_group_id"] != lg])
        out[name] = v
    out["lineage_groups"] = dict(sorted(Counter(r["lineage_group_id"] for r in rows).items()))
    out["note"] = ("상한이다. 권리 결정 · 사람 검수 기록이 없어 지금 생성기로 넘어가는 구절은 0 이다. "
                   "면 표지는 키워드 후보라 qa_upper_facet_bounded 도 검수 전 값이다.")
    return out


def reasoning_capacity(limits_csv: Path) -> dict:
    from corpus.rules.limits_loader import load_limits
    from corpus.rules.skeleton_gen import _DEFAULT_CONFIG, enumerate_combos

    table = load_limits(limits_csv, pilot=True)
    cap = _DEFAULT_CONFIG.combo_cap
    out: dict = {"limits_csv_sha256": _sha(limits_csv), "combo_cap": cap, "by_method": {}}
    for method in ("RT", "VT"):
        combos = enumerate_combos(table, inspection_method=method)
        rows = [c.row for c in combos]
        out["by_method"][method] = {
            "combos": len(combos),
            "skeleton_upper": len(combos) * cap,
            "short_of_target_10000": max(0, REASONING_TARGET - len(combos) * cap),
            "source_tables": dict(sorted(Counter(f"{r.source_doc}:{r.source_table_id}" for r in rows).items())),
            "source_pages": sorted({str(r.source_page) for r in rows}),
        }
    total = sum(v["skeleton_upper"] for v in out["by_method"].values())
    out["skeleton_upper_total"] = total
    out["combos_needed_for_10000_at_cap"] = -(-REASONING_TARGET // cap)
    out["note"] = ("조합 하나의 골격 40 건은 측정값 · 두께만 다른 같은 규칙이다. 상한을 올려 1만을 채우면 "
                   "반복으로 채운 수다. 서로 다른 규칙 단위는 combos 다.")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="구절 은행 · 허용치 표 생성 용량")
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--limits", required=True, type=Path)
    ap.add_argument("--full-bank", type=Path, default=None)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--per-passage", type=int, default=3)
    ap.add_argument("--minimum-distinct", type=int, default=3334)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    rows = load_jsonl(a.bank)
    report = {
        "bank_sha256": _sha(a.bank),
        "qa": qa_capacity(rows, a.per_passage, a.minimum_distinct),
        "reasoning": reasoning_capacity(a.limits),
        "passing_generation_gate_now": None,
    }
    if a.full_bank is not None:
        if a.registry is None:
            raise SystemExit("--full-bank 에는 --registry 가 필요하다")
        selected, why = select_for_generation(load_jsonl(a.full_bank), pointer_rows=rows, registry=a.registry)
        report["passing_generation_gate_now"] = len(selected)
        report["gate_reasons_now"] = dict(sorted(why.items()))
        report["full_bank_sha256"] = _sha(a.full_bank)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes((json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    q = report["qa"]["all_tiers"]["pool"]
    r = report["reasoning"]
    print(json.dumps({"passages": q["passages"], "qa_upper": q["qa_upper"],
                      "reasoning_upper": r["skeleton_upper_total"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
