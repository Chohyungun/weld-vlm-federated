"""하한 1회 추론 + 사후 필터 ≡ 임계 직접 추론 — **본실험 규모 재확인** (17번 §7-4, 22번 §4).

    uv run python scripts/probe/verify_filter_parity.py --snapshot data/interim/manifest_v1 \
        --pilot outputs/main_c/seed1 --out outputs/main_d/seed1 --profile main --n 200

## 왜 다시 재나

곡선 전 구간(판정 1)은 **하한 레코드를 사후 필터해서** 만든다. 임계마다 다시 추론하지
않는 근거는 NMS 의 성질이다 — 더 높은 점수의 박스만이 억제자가 되므로, 하한을 낮춰도
`점수 ≥ t` 인 생존 집합은 변하지 않는다. 파일럿에서 이 동치를 비트 대조로 실측했지만
(65번 산출물과 대조), **본실험 규모에서는 확인한 적이 없다** — C 가 하한으로만 내보냈기
때문이다(17번 §1-2). 곡선이 헤드라인이 된 이상 그 전제를 규모에서 한 번은 밟아야 한다.

## 무엇을 하나

평가셋에서 **결정적으로** 표본 N 장을 뽑아(정렬된 id 위의 고정 시드 셔플) 칸마다 두 경로를
만든다.

    A. 직접 추론:  load_yolo_from_npz → predict_cell(conf=운용임계)
    B. 하한 + 필터: sweep/{tag}_raw_s{seed}.jsonl 되읽기 → filter_by_conf(운용임계)

같은 이미지에 대해 `(iso_code, bbox 4개, score)` 다중집합이 같은지 본다. **좌표는 C 가
소수 2자리, 점수는 4자리로 반올림해 내보냈으므로**(13_spec_D §2-3) A 쪽도 같은 자리로
맞춰 비교한다 — 반올림 차이를 동치 위반으로 오독하지 않으려는 것이다. 그 반올림이 곧
저장 계약이고, 채점은 저장본으로 한다.

GPU 를 쓰지 않는다(`device=cpu`). 학습 중인 run 에 닿지 않는다 — 읽는 것은 C 의 npz 와
동결 스냅샷뿐이다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.adapters import DETECTION_EXPORT_TAGS, read_records
from evaluation.cells import load_population
from evaluation.detect_infer import cell_tag, filter_by_conf, load_yolo_from_npz, predict_cell
from evaluation.params import add_common_args, params_from_args

#: C 의 export 반올림 자리 (13_spec_D §2-3 · `scripts/main_det.py:cmd_export`).
BOX_NDIGITS = 2
SCORE_NDIGITS = 4

#: 표본 추출 시드. 값 자체는 임의지만 **고정**이라 재실행이 같은 표본을 본다.
SAMPLE_SEED = 20260908

#: 칸 → C 가 낸 가중치 상대경로. `_export_targets` 와 같은 자리를 본다.
NPZ_BY_TAG: dict[str, str] = {
    "sep_local_C1": "sep_local/sep_local_c0.npz",
    "sep_local_C2": "sep_local/sep_local_c1.npz",
    "sep_local_C3": "sep_local/sep_local_c2.npz",
    "sep_central": "sep_central/sep_central.npz",
    "sep_fed": "fl/sep_fed/global_r050.npz",
}


def _key(rec) -> list[tuple]:
    """레코드 하나의 박스 다중집합. 정렬해 순서 의존을 없앤다."""
    return sorted(
        (d.iso_code,
         *(round(float(v), BOX_NDIGITS) for v in d.bbox_px),
         round(float(d.score or 0.0), SCORE_NDIGITS))
        for d in rec.defects if d.bbox_px
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--root", default=".")
    ap.add_argument("--at-conf", action="store_true")
    ap.add_argument("--n", type=int, default=200, help="표본 이미지 수")
    args = ap.parse_args()

    params = params_from_args(args)
    pop = load_population(params)
    conf = params.conf.value

    rows = sorted(pop.rows, key=lambda r: r["image_id"])
    rng = random.Random(SAMPLE_SEED)
    sample = sorted(rng.sample(rows, min(int(args.n), len(rows))),
                    key=lambda r: r["image_id"])
    ids = {r["image_id"] for r in sample}
    print(f"표본 {len(sample)}장 / 평가셋 {len(rows):,}장 · 임계 {conf} "
          f"(하한 {params.conf_floor}) · 프로파일 {params.profile} "
          f"({params.model_cfg} · imgsz {params.imgsz} · {params.device})")

    cells: dict[str, dict] = {}
    for _, cell, client in DETECTION_EXPORT_TAGS:
        tag = cell_tag(cell, client)
        raw_p = params.out / "sweep" / f"{tag}_raw_s{params.seed}.jsonl"
        npz = params.pilot / NPZ_BY_TAG[tag]
        if not raw_p.exists() or not npz.exists():
            print(f"[{tag}] 입력 없음(raw={raw_p.exists()} npz={npz.exists()}) — 건너뜀")
            cells[tag] = {"checked": False,
                          "reason": f"raw={raw_p.exists()} npz={npz.exists()}"}
            continue

        # B — 하한 되읽기 + 사후 필터 (표본만)
        raw = [r for r in read_records(raw_p.read_text(encoding="utf-8").splitlines())
               if r.image_id in ids]
        b = {r.image_id: _key(r) for r in filter_by_conf(raw, conf)}

        # A — 직접 추론
        t0 = time.perf_counter()
        yolo = load_yolo_from_npz(npz, params.class_names, params.imgsz,
                                  model_cfg=params.model_cfg)
        recs = predict_cell(yolo, sample, Path(args.root), cell, client, params, conf=conf)
        a = {r.image_id: _key(r) for r in recs}
        wall = time.perf_counter() - t0

        diff = sorted(i for i in ids if a.get(i, []) != b.get(i, []))
        cells[tag] = {
            "checked": True,
            "npz": str(npz),
            "n_images": len(ids),
            "n_boxes_direct": sum(len(v) for v in a.values()),
            "n_boxes_refiltered": sum(len(v) for v in b.values()),
            "n_images_differing": len(diff),
            "identical": not diff,
            "wall_s": round(wall, 1),
            "example_diffs": [
                {"image_id": i, "direct": a.get(i), "refiltered": b.get(i)}
                for i in diff[:3]
            ],
        }
        print(f"[{tag}] 직접 {cells[tag]['n_boxes_direct']}박스 / 재필터 "
              f"{cells[tag]['n_boxes_refiltered']}박스 · 다른 이미지 {len(diff)}장 "
              f"→ {'동일' if not diff else '불일치'} ({wall:.1f}s)")

    checked = [v for v in cells.values() if v.get("checked")]
    ok = bool(checked) and all(v["identical"] for v in checked)
    payload = {
        "params": params.as_dict(),
        "sample_seed": SAMPLE_SEED,
        "n_sample": len(sample),
        "n_eval": pop.n_eval,
        "conf_operating": conf,
        "conf_floor": params.conf_floor,
        "rounding": {"bbox_ndigits": BOX_NDIGITS, "score_ndigits": SCORE_NDIGITS,
                     "note": "C 의 export 저장 계약과 같은 자리로 맞춰 비교한다"},
        "cells": cells,
        "n_cells_checked": len(checked),
        "all_identical": ok,
        "claim": ("하한 1회 추론 + filter_by_conf(t) 가 임계 t 직접 추론과 같은 생존 집합을 "
                  "준다 — 곡선 전 구간(판정 1)이 이 동치 위에 서 있다"),
    }
    dest = params.out / "verify_filter_parity_v1.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n칸 {len(checked)} 확인 · {'전부 동일' if ok else '불일치 있음'} → {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
