"""검출 임계 스윕 — 채점 비대칭(감사 D-1) 해소. 77번 과제 1.

    uv run python scripts/probe/sweep_detection_conf.py --stage predict
    uv run python scripts/probe/sweep_detection_conf.py --stage sweep

**문제.** 분리형 3칸은 `conf 0.25` 로 저신뢰 박스를 버린 뒤 채점되는데 통합형 2칸에는
대응하는 임계가 없다. RQ2 의 핵심 대비(결함 놓침 0.8142 대 0.2646)가 그 비대칭 위에
있다. 부정이 아니라 두 출력 형식이 근본적으로 다른 데서 온다.

**처방.** 임계 하나를 고르지 않는다. 곡선을 낸다.
- 분리형은 임계 스윕(0.05~0.50), 통합형은 임계가 없으므로 한 점.
- 통합형의 **예측 박스 수와 같은 개수가 나오는 검출 임계**를 찾아 그 지점을 함께 낸다.
  예측량을 맞춘 비교다.
- 어느 임계에서도 결론이 같은지, 뒤집히는 구간이 있는지를 밝힌다.

**GPU 를 쓰지 않는다.** predict 단계는 CPU 추론이고 sweep 단계는 순수 채점이다.
재학습은 없다 — 기존 last 체크포인트를 그대로 쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.adapters import read_records
from evaluation.cells import load_population
from evaluation.detect_infer import (
    CHECKPOINTS,
    cell_tag,
    checkpoint_paths,
    filter_by_conf,
    load_yolo_from_npz,
    predict_cell,
)
from evaluation.params import add_common_args, params_from_args
from evaluation.score import score_records
from tracking.mlflow_local import reject_best_checkpoint

UNI_CELLS = ("uni_central", "uni_fed")


def raw_path(out: Path, tag: str, seed: int) -> Path:
    return out / "sweep" / f"{tag}_raw_s{seed}.jsonl"


def n_boxes(records) -> int:
    return sum(len(r.defects) for r in records)


# --------------------------------------------------------------------------------------
# 1단계 — 하한 추론. 임계별 재추론 대신 하한 1회 + 사후 필터.
# --------------------------------------------------------------------------------------

def stage_predict(params, eval_rows, root: Path) -> dict:
    ckpts = checkpoint_paths(params.pilot)
    for p in ckpts.values():
        reject_best_checkpoint(p)
        if not p.exists():
            raise SystemExit(f"체크포인트 없음: {p}")
    (params.out / "sweep").mkdir(parents=True, exist_ok=True)

    timing = {}
    for (cell, client), ckpt in ckpts.items():
        tag = cell_tag(cell, client)
        t0 = time.time()
        yolo = load_yolo_from_npz(ckpt, params.class_names, params.imgsz,
                                  model_cfg=params.model_cfg)
        recs = predict_cell(yolo, eval_rows, root, cell, client, params,
                            conf=params.conf_floor)
        dest = raw_path(params.out, tag, params.seed)
        with dest.open("w", encoding="utf-8", newline="\n") as fh:
            for r in recs:
                fh.write(r.model_dump_json() + "\n")
        timing[tag] = {"seconds": round(time.time() - t0, 1), "n_boxes": n_boxes(recs)}
        print(f"[{tag}] 하한 {params.conf_floor} 추론 {n_boxes(recs)}박스 "
              f"· {timing[tag]['seconds']}s -> {dest.name}")
    return timing


# --------------------------------------------------------------------------------------
# 2단계 — 하한 결과를 65번 산출물(0.25)과 대조. 사후 필터가 재추론과 같은가.
# --------------------------------------------------------------------------------------

def _box_key(record):
    return sorted(
        (d.iso_code, *(round(v, 4) for v in d.bbox_px), round(d.score or 0.0, 6))
        for d in record.defects
    )


def verify_parity(params, tags: list[str]) -> dict:
    """`filter_by_conf(하한, 0.25)` 가 65번의 0.25 직접 추론과 같은 집합을 주는가.

    **같아야 스윕 전체가 성립한다.** 다르면 하한 추론 자체가 임계에 의존한다는 뜻이라
    스윕의 각 점을 따로 추론해야 한다. 실패해도 자동 폴백하지 않는다 — 보고 대상이다.
    """
    report: dict[str, dict] = {}
    for tag in tags:
        prev_p = params.out / f"{tag}_s{params.seed}.jsonl"
        raw_p = raw_path(params.out, tag, params.seed)
        if not prev_p.exists():
            report[tag] = {"checked": False, "reason": f"65번 산출물 없음: {prev_p.name}"}
            continue
        prev = read_records(prev_p.read_text(encoding="utf-8").splitlines())
        cut = filter_by_conf(
            read_records(raw_p.read_text(encoding="utf-8").splitlines()),
            params.conf.value,
        )
        prev_box = {r.image_id: _box_key(r) for r in prev}
        cut_box = {r.image_id: _box_key(r) for r in cut}
        diff_ids = sorted(i for i in set(prev_box) | set(cut_box)
                          if prev_box.get(i) != cut_box.get(i))
        report[tag] = {
            "checked": True,
            "n_boxes_65": sum(len(v) for v in prev_box.values()),
            "n_boxes_refiltered": sum(len(v) for v in cut_box.values()),
            "n_images_differing": len(diff_ids),
            "identical": not diff_ids,
            "example_diffs": [
                {"image_id": i, "65번": prev_box.get(i), "재필터": cut_box.get(i)}
                for i in diff_ids[:3]
            ],
        }
    return report


# --------------------------------------------------------------------------------------
# 3단계 — 스윕 + 예측량 정합점
# --------------------------------------------------------------------------------------

def parity_threshold(records, target_boxes: int) -> dict:
    """박스 수가 `target_boxes` 에 가장 가까워지는 임계.

    개수는 임계의 계단함수라 정확히 맞는 값이 없을 수 있다. 실제 점수값 위에서만
    후보를 세우고(그 사이 구간은 개수가 같다), 가장 가까운 점을 고른다.
    """
    scores = sorted((d.score or 0.0 for r in records for d in r.defects), reverse=True)
    total = len(scores)
    if total == 0:
        return {"threshold": None, "n_boxes": 0, "target": target_boxes,
                "exact": False, "note": "박스 0건 — 정합점 없음"}
    if target_boxes >= total:
        return {"threshold": None, "n_boxes": total, "target": target_boxes,
                "exact": False,
                "note": (f"하한 {total}박스로도 통합형 {target_boxes}박스에 못 미친다 — "
                         "임계를 낮춰서는 예측량을 맞출 수 없다")}
    # k 번째 점수 바로 위/아래에서 개수가 k / k+1 이 된다.
    idx = min(max(target_boxes - 1, 0), total - 1)
    thr = scores[idx]
    kept = sum(1 for s in scores if s >= thr)
    return {"threshold": round(float(thr), 6), "n_boxes": kept, "target": target_boxes,
            "exact": kept == target_boxes}


def rq2_verdict(sweep: dict, uni: dict, params) -> dict:
    """임계 전 구간에서 분리형↔통합형 대소가 유지되는가.

    유지되면 단일 임계로도 방향은 말할 수 있고, 뒤집히면 **단일 임계 비교는 불가**다.
    비교 대상은 두 지표다 — 결함 놓침(miss_rate, 낮을수록 좋음)과 Macro-F1.
    """
    det_tags = ("sep_central", "sep_fed", "sep_local_C1", "sep_local_C2", "sep_local_C3")
    rows = []
    for metric, better in (("miss_rate", "low"), ("macro_f1", "high")):
        for det in det_tags:
            for cell in uni:
                signs = set()
                per = {}
                for thr in params.conf_sweep:
                    k = f"{thr:.2f}"
                    a = sweep[det]["by_threshold"][k][metric]
                    b = uni[cell]["metrics"][metric]
                    s = (a > b) - (a < b)
                    signs.add(s)
                    per[k] = round(a - b, 6)
                rows.append({
                    "metric": metric, "better": better,
                    "detection": det, "unified": cell,
                    "flips": len(signs) > 1,
                    "signs": sorted(signs),
                    "delta_by_threshold": per,
                })
    n_flip = sum(1 for r in rows if r["flips"])
    return {
        "single_threshold_comparable": "가능" if n_flip == 0 else "불가",
        "n_pairs": len(rows),
        "n_flipping_pairs": n_flip,
        "pairs": rows,
        "summary": (
            f"임계 {params.conf_sweep[0]}~{params.conf_sweep[-1]} 구간에서 "
            f"{len(rows)}개 대비 중 {n_flip}개가 부호를 바꾼다"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--stage", choices=("predict", "sweep", "all"), default="all")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    params = params_from_args(args)
    root = Path(args.root)
    params.out.mkdir(parents=True, exist_ok=True)

    # 모집단은 `load_population` 한 지점에서만 만든다 — 여기서 따로 만들면 정상 이미지
    # back-fill 이 빠져 스윕만 다른 모집단에서 채점된다(80번 D9 의 재발 경로).
    pop = load_population(params)
    classes = pop.classes
    ev = pop.rows
    gold_codes, gold_boxes = pop.gold_codes, pop.gold_boxes
    print(f"평가셋 {len(ev)}장 · 임계 {params.conf.value} ({params.conf.source})")

    tags = [cell_tag(c, cl) for c, cl, _ in CHECKPOINTS]

    timing = {}
    if args.stage in ("predict", "all"):
        timing = stage_predict(params, ev, root)
    if args.stage == "predict":
        return 0

    parity = verify_parity(params, tags)
    for tag, r in parity.items():
        if r.get("checked") and not r["identical"]:
            print(f"[{tag}] 재필터가 65번과 다르다 — {r['n_images_differing']}장")

    # --- 통합형 2칸: 66번 산출물 되읽기. 임계가 없으므로 한 점 -----------------------
    uni: dict[str, dict] = {}
    for cell in UNI_CELLS:
        p = params.out / f"{cell}_s{params.seed}.jsonl"
        if not p.exists():
            print(f"통합형 산출물 없음: {p} — 66번을 먼저 돌린다")
            return 1
        recs = read_records(p.read_text(encoding="utf-8").splitlines())
        uni[cell] = {
            "n_boxes": n_boxes(recs),
            "metrics": score_records(recs, gold_codes, gold_boxes, classes),
        }
        m = uni[cell]["metrics"]
        print(f"[{cell}] 박스 {uni[cell]['n_boxes']} · miss {m['miss_rate']:.4f} "
              f"· macroF1 {m['macro_f1']:.4f}")

    # --- 분리형 스윕 -----------------------------------------------------------------
    sweep: dict[str, dict] = {}
    parity_points: dict[str, dict] = {}
    for tag in tags:
        raw = read_records(
            raw_path(params.out, tag, params.seed).read_text(encoding="utf-8").splitlines()
        )
        per_thr = {}
        for thr in params.conf_sweep:
            cut = filter_by_conf(raw, thr)
            per_thr[f"{thr:.2f}"] = {
                "n_boxes": n_boxes(cut),
                **score_records(cut, gold_codes, gold_boxes, classes),
            }
        sweep[tag] = {"n_boxes_floor": n_boxes(raw), "by_threshold": per_thr}
        parity_points[tag] = {}
        for cell in UNI_CELLS:
            pt = parity_threshold(raw, uni[cell]["n_boxes"])
            if pt.get("threshold") is not None:
                cut = filter_by_conf(raw, pt["threshold"])
                pt["metrics"] = score_records(cut, gold_codes, gold_boxes, classes)
            parity_points[tag][cell] = pt
        print(f"[{tag}] 스윕 완료 · 하한 {sweep[tag]['n_boxes_floor']}박스")

    verdict = rq2_verdict(sweep, uni, params)

    payload = {
        "params": params.as_dict(),
        "n_eval": len(ev),
        "timing_predict": timing,
        "parity_vs_65": parity,
        "unified": uni,
        "sweep": sweep,
        "parity_points": parity_points,
        "rq2": verdict,
    }
    dest = params.out / "sweep_detection_conf_v1.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"저장: {dest}")
    print(f"RQ2: {verdict['single_threshold_comparable']} — {verdict['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
