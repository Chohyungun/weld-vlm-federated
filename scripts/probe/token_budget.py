"""`max_new_tokens` 재산정 — C 의 고장 7(256 부족으로 다결함 이미지 절단) 처방. 66번 과제 3.

두 예산을 따로 낸다. 형식이 다르므로 한 숫자로 묶으면 한쪽이 반드시 틀린다.

1. **통합형 추론 예산** — 출력이 `defects[]` 배열이라 결함 수에 선형으로 늘어난다.
   C 의 성공 생성문을 실토크나이저로 세어 `기본 + 결함당` 을 회귀로 뽑고, 표본의
   이미지당 최대 결함 수로 외삽한 뒤 여유를 둔다.
2. **판정부(⑤) 예산** — 출력이 `{verdict, cited_clauses, basis}` 고정 구조라 결함 수에
   비례하지 않는다. 최악 형태(top-k 조항 전부 인용 + 긴 한국어 기준 문장)를 직접 세운다.

**추정하지 않고 실측한다.** 토크나이저는 `Qwen/Qwen3.5-0.8B` 실물이다.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.eval_set import read_manifest

MODEL_ID = "Qwen/Qwen3.5-0.8B"
SNAP = Path("data/processed/aihub71761_rt_v1_pilot3000")
PRED = Path("outputs/pilot_c/predictions")
OUT = Path("outputs/pilot_d")
SAFETY = 1.5          # 여유 계수. 회귀 잔차와 미관측 결함 수 구간을 덮는다
GEN_CFG_MAX = 256     # C 가 쓴 값 (고장 7)


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731

    # --- 1. 통합형: 결함 수 대 토큰 수 실측 -----------------------------------------
    obs: list[tuple[int, int]] = []
    truncated: list[int] = []
    for cell in ("uni_central", "uni_fed"):
        for line in (PRED / f"{cell}.generations.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            t = n_tok(row["text"])
            if row.get("parse_error") is not None:
                truncated.append(t)
                continue
            obs.append((len(row["bbox_px_parsed"]["defects"]), t))

    counts = Counter(n for n, _ in obs)
    # 결함 수별 최대 토큰 — 예산은 평균이 아니라 상한으로 잡는다
    per_n_max = {n: max(t for k, t in obs if k == n) for n in sorted(counts)}
    ns = sorted(per_n_max)
    # 두 점(최소·최대 관측 결함 수)에서 기울기·절편을 잡고 전 구간 잔차를 확인한다
    if len(ns) >= 2:
        lo, hi = ns[0], ns[-1]
        slope = (per_n_max[hi] - per_n_max[lo]) / (hi - lo)
        base = per_n_max[lo] - slope * lo
    else:
        slope, base = 0.0, float(per_n_max[ns[0]])
    resid = {n: per_n_max[n] - (base + slope * n) for n in ns}

    rows = read_manifest(SNAP)
    defect_counts = Counter(int(r["n_defects"] or 0) for r in rows)
    max_defects = max(defect_counts)
    eval_max = max(int(r["n_defects"] or 0) for r in rows if r["split"] == "eval")

    need_uni = base + slope * max_defects + max(resid.values(), default=0.0)
    budget_uni = int(math.ceil(need_uni * SAFETY / 64.0) * 64)

    # --- 2. 판정부: 최악 형태를 직접 센다 -------------------------------------------
    worst_basis = (
        "제시된 조항에 따르면 모재 두께 구간별로 최대 결함 크기 한계가 4 mm 이하, 5 mm 이하, "
        "두께의 5분의 1 이하, 10 mm 이하로 규정되어 있으며 합계 길이 기준은 6 mm 이하, "
        "두께의 2분의 1 이하, 24 mm 이하로 규정되어 있으나 모재 두께와 화소당 실치수 정보가 "
        "없어 치수 기준의 합부 판정은 내리지 않는다"
    )
    worst_judge = json.dumps(
        {"verdict": "판정불가",
         "cited_clauses": ["KRA27-T15", "KRA27-T16", "KRA27-3D"],
         "basis": worst_basis},
        ensure_ascii=False,
    )
    need_judge = n_tok(worst_judge)
    budget_judge = int(math.ceil(need_judge * SAFETY / 64.0) * 64)

    payload = {
        "tokenizer": MODEL_ID,
        "unified": {
            "n_observed": len(obs),
            "n_truncated_excluded": len(truncated),
            "observed_defect_counts": {str(k): v for k, v in sorted(counts.items())},
            "max_tokens_per_defect_count": {str(k): v for k, v in per_n_max.items()},
            "fit": {"base": base, "per_defect": slope,
                    "residual_max": max(resid.values(), default=0.0),
                    "residuals": {str(k): v for k, v in resid.items()}},
            "snapshot_defect_counts": {str(k): v for k, v in sorted(defect_counts.items())},
            "max_defects_snapshot": max_defects,
            "max_defects_eval": eval_max,
            "need_raw": need_uni,
            "safety": SAFETY,
            "recommended_max_new_tokens": budget_uni,
            "previous": GEN_CFG_MAX,
            "truncated_token_counts_observed": sorted(set(truncated))[:5],
        },
        "judge": {
            "worst_case_output": worst_judge,
            "need_raw": need_judge,
            "safety": SAFETY,
            "recommended_max_new_tokens": budget_judge,
            "note": (
                "판정부 출력은 결함 수에 비례하지 않는다(verdict·cited_clauses·basis 고정 "
                "구조). 통합형 예산을 그대로 쓰면 낭비이고, 통합형에 판정부 예산을 쓰면 "
                "고장 7 이 재발한다"
            ),
        },
        "caveat": (
            "결함 수 상한은 파일럿 표본(3,279장) 관측치다. 본실험 70,000장의 상한은 "
            "A 가 전수로 다시 내야 하며, 그 값으로 이 식(base + per_defect × N)을 다시 "
            "평가하는 것이 본실험 착수 조건이다"
        ),
    }
    dest = OUT / "token_budget_v1.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"통합형: 기본 {base:.1f} + 결함당 {slope:.1f} · 잔차 최대 {max(resid.values(), default=0):.1f}")
    print(f"  표본 최대 결함 {max_defects}개(eval {eval_max}) → 필요 {need_uni:.0f} "
          f"→ 권고 {budget_uni} (기존 {GEN_CFG_MAX})")
    print(f"판정부: 최악 {need_judge} 토큰 → 권고 {budget_judge}")
    print(f"저장: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
