"""`max_new_tokens` 재산정 v2 — **train+val 전용 역산.** 67번 §5-4 · 71번 과제 5.

66번은 파일럿 생성문(평가셋 추론 결과)에서 회귀식을 뽑고 파일럿 표본 전체의 최대 결함
수로 외삽했다. 67번 §5-4 가 그것을 금지한다:

> 반드시 **train+val 라벨 통계에서만** 역산하고 eval 은 열지 않는다. 평가셋 통계로 5칸
> 공통 고정 디코딩 상수를 맞추면 불변조건 1-4 를 실질에서 깬다.

그래서 여기서는 **동결본의 train+val 어노테이션만** 읽어 학습 타깃 JSON 을 그대로 만들고
실토크나이저로 센다. 평가셋 행은 한 줄도 읽지 않는다(코드가 그것을 강제한다).

좌표는 C 의 `vlm/coords.py` 를 그대로 통과시킨다 — 자릿수가 토큰 수를 바꾸므로 규약이
다르면 예산도 달라진다.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.eval_set import read_manifest
from evaluation.gold import read_derived_csv
from evaluation.params import COORD_SPACE
from vlm.coords import CoordCfg, ImageGeom, quantize, to_model

FROZEN = Path("data/interim/manifest_v1")
OUT = Path("outputs/pilot_d")
MODEL_ID = "Qwen/Qwen3.5-0.8B"
# **ABS_ORIG.** 총괄 판정 1 (2026-09-02) · C 의 전환 커밋 main 47c4dbc.
# 자릿수가 토큰 수를 바꾸므로 규약이 바뀌면 예산도 다시 재야 한다 — 절대 픽셀은
# 0~1000 정규화보다 자릿수가 길어 타깃 토큰이 늘어난다(C 추정 +0.52개/박스).
# 이 상수는 `evaluation.params.COORD_SPACE` 와 같아야 하고 시험이 그것을 고정한다.
COORD_CFG = CoordCfg(coord_space=COORD_SPACE)
SAFETY = 1.5
CURRENT = 256          # C 가 쓴 값 (고장 7)
PREV_RECOMMEND = 1536  # 66번 §8 권고 (평가셋 포함 관측에서 유도)


def main() -> int:
    from transformers import AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    rows = read_manifest(FROZEN)
    trainval = {r["image_id"]: r for r in rows if r["split"] in ("train", "val")}
    n_eval = sum(1 for r in rows if r["split"] == "eval")
    print(f"동결본 {len(rows)}장 — train+val {len(trainval)}장 사용, eval {n_eval}장 미열람")

    clause_of: dict[str, str] = {}
    for g in read_derived_csv("corpus/derived/gold_clauses.csv"):
        if g["inspection_method"] == "RT":
            clause_of.setdefault(g["defect_code"], g["clause_id"])

    boxes: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    n_geom_invalid = 0
    with (FROZEN / "annotations.csv").open(encoding="utf-8", newline="") as fh:
        for a in csv.DictReader(fh):
            iid = a["image_id"]
            if iid not in trainval:
                continue                      # eval 행은 여기서 전부 걸러진다
            if a.get("geom_valid") == "False":
                n_geom_invalid += 1
                continue
            if not a.get("bbox_x1_px"):
                continue
            boxes[iid].append((
                a["iso_code"],
                (float(a["bbox_x1_px"]), float(a["bbox_y1_px"]),
                 float(a["bbox_x2_px"]), float(a["bbox_y2_px"])),
            ))

    counts = Counter()
    per_n_max: dict[int, int] = {}
    for iid, row in trainval.items():
        bs = boxes.get(iid, [])
        geom = ImageGeom(orig_w=int(row["width_px"]), orig_h=int(row["height_px"]))
        target = json.dumps(
            {
                "defects": [
                    {"iso_code": code, "bbox_2d": list(quantize(to_model(b, geom, COORD_CFG)))}
                    for code, b in bs
                ],
                "verdict": "판정불가",
                "cited_clauses": sorted({clause_of[c] for c, _ in bs if c in clause_of}),
            },
            ensure_ascii=False, separators=(",", ":"),
        )
        n = len(bs)
        t = len(tok(target, add_special_tokens=False)["input_ids"])
        counts[n] += 1
        per_n_max[n] = max(per_n_max.get(n, 0), t)

    ns = sorted(k for k in per_n_max if k >= 1)
    # 회귀 — 결함 1개와 최대 관측 사이 두 점. JSON 이 결함당 가산이므로 선형이다.
    lo, hi = ns[0], ns[-1]
    slope = (per_n_max[hi] - per_n_max[lo]) / (hi - lo)
    base = per_n_max[lo] - slope * lo
    resid = {n: per_n_max[n] - (base + slope * n) for n in ns}
    max_resid = max(resid.values())

    n_max = max(counts)
    need = base + slope * n_max + max_resid
    recommend = int(math.ceil(need * SAFETY / 64.0) * 64)

    # 상위 꼬리 — 예산을 어디서 자르면 몇 장이 절단되는지
    tail = {}
    for cap in (256, 512, 1024, 1536, 2048, recommend):
        n_defect_cap = int((cap - base) / slope) if slope else 0
        n_over = sum(v for k, v in counts.items() if k > n_defect_cap)
        tail[str(cap)] = {
            "결함_수_한계": n_defect_cap,
            "초과_이미지": n_over,
            "초과_비율": round(n_over / len(trainval), 6),
        }

    payload = {
        "rule": "67번 §5-4 — train+val 라벨 통계만. eval 미열람",
        "snapshot": str(FROZEN),
        "tokenizer": MODEL_ID,
        "coord_space": COORD_CFG.coord_space,
        "population": {
            "n_trainval": len(trainval), "n_eval_untouched": n_eval,
            "n_annotations_geom_invalid_skipped": n_geom_invalid,
        },
        "fit": {"base": base, "per_defect": slope, "residual_max": max_resid,
                "max_tokens_per_defect_count": {str(k): v for k, v in sorted(per_n_max.items())}},
        "defect_counts": {str(k): v for k, v in sorted(counts.items())},
        "max_defects_trainval": n_max,
        "need_raw": need,
        "safety": SAFETY,
        "recommended_max_new_tokens": recommend,
        "previous": {"C_used": CURRENT, "report66_recommend": PREV_RECOMMEND},
        "truncation_tail_trainval": tail,
    }
    dest = OUT / "token_budget_trainval_v1.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"식: {base:.1f} + {slope:.2f}×N · 잔차 최대 {max_resid:.1f}")
    print(f"train+val 최대 결함 {n_max}개 → 필요 {need:.0f} → 권고 {recommend} "
          f"(C {CURRENT} · 66번 {PREV_RECOMMEND})")
    print(f"저장: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
