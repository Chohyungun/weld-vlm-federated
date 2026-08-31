"""검출 3칸(②③④) 선채점 — 추론(CPU) + 공통 스키마 + 단일 채점기. 65번 지시.

    uv run python scripts/probe/score_detection_cells.py \
        --snapshot data/processed/aihub71761_rt_v1_pilot3000 \
        --pilot outputs/pilot_c --out outputs/pilot_d

가중치 로드는 트랙 C 의 `detection.serialize` 를 **그대로 쓴다** — 재구현 금지.
좌표는 Ultralytics 표준 predict 의 원본 픽셀 복원(Q20 실측 봉인)을 그대로 받는다.
D 는 좌표를 변환하지 않는다.

**GPU 를 쓰지 않는다.** device 는 cpu 로 못박는다 — C 의 학습이 GPU 를 20시간 점유 중이다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.label_map import load_label_map
from detection import serialize
from evaluation.metrics.detection import class_jaccard, score_detection
from evaluation.metrics.localization import coco_map, score_bbox_iou
from evaluation.probes.metadata_probe import MetaSample, trivial_bound
from evaluation.probes.p9_runner import contexts_from_snapshot, p9_all_cells
from evaluation.schema import SCHEMA_VERSION, PredictionRecord
from tracking.mlflow_local import reject_best_checkpoint

SEED = 20260828        # C 의 파일럿 시드 (meta.json 실측)
CONF = 0.25            # Q4 기본값 — 5칸 공통. configs 확정 전 문서 기본값
IMGSZ = 416
DEVICE = "cpu"         # 고정. GPU 금지


def load_yolo_from_npz(npz_path: Path, class_names: list[str]):
    """C 의 npz 상태를 YOLO11n(nc=4) 에 주입한다. 전부 C 의 serialize 경유다."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    dm = DetectionModel(cfg="yolo11n.yaml", nc=len(class_names), verbose=False)
    keys = serialize.canonical_keys(dm.state_dict())
    z = np.load(npz_path)
    arrays = [z[f"arr_{i}"] for i in range(len(z.files))]
    ref = dm.state_dict()
    serialize.assert_compatible(arrays, keys, ref)
    dm.load_state_dict(serialize.ndarrays_to_state_dict(arrays, keys, ref), strict=True)
    dm.names = dict(enumerate(class_names))
    # ultralytics 체크포인트 형태로 감싸 YOLO 의 정식 로드 경로를 태운다
    tmp = npz_path.with_suffix(".tmp.pt")
    torch.save({"model": dm.float(), "train_args": {"imgsz": IMGSZ}}, tmp)
    yolo = YOLO(str(tmp))
    tmp.unlink(missing_ok=True)
    return yolo


def read_manifest(snapshot: Path):
    with (snapshot / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_gold(snapshot: Path, eval_ids: set[str]):
    """평가셋 GT — 이미지 수준 클래스 집합과 bbox 목록."""
    codes: dict[str, set[str]] = defaultdict(set)
    boxes: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    with (snapshot / "annotations.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            iid = r["image_id"]
            if iid not in eval_ids:
                continue
            codes[iid].add(r["iso_code"])
            if r.get("bbox_x1_px"):
                boxes[iid].append((
                    r["iso_code"],
                    (float(r["bbox_x1_px"]), float(r["bbox_y1_px"]),
                     float(r["bbox_x2_px"]), float(r["bbox_y2_px"])),
                ))
    return codes, boxes


def predict_cell(
    yolo, rows: list[dict], root: Path, cell: str, client: str | None
) -> list[PredictionRecord]:
    """평가셋 전량 추론 → 계약 #4 레코드. 좌표는 predict 가 복원한 원본 픽셀 그대로."""
    records: list[PredictionRecord] = []
    paths = [str(root / r["rel_path"]) for r in rows]
    results = yolo.predict(
        source=paths, imgsz=IMGSZ, conf=CONF, device=DEVICE,
        verbose=False, stream=True,
    )
    for row, res in zip(rows, results, strict=True):
        defects = []
        names = res.names
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            if x1 >= x2 or y1 >= y2:
                continue  # 퇴화 박스는 스키마가 거부한다 — 만들지 않는 게 아니라 세지 않는다
            dt = names[int(b.cls)]
            defects.append({
                "iso_code": LM.iso_code(dt),
                "bbox_px": [x1, y1, x2, y2],
                "score": float(b.conf),
                "size_px": max(x2 - x1, y2 - y1),
                "size_basis": "major_axis",
                "retrieved": None,
            })
        records.append(PredictionRecord(
            schema_version=SCHEMA_VERSION,
            image_id=row["image_id"], cell=cell, client=client, seed=SEED,
            defects=defects,
            verdict="판정불가",           # 판정부(⑤) 미실행 — 검출 축 선채점
            cited_clauses=[],
            parse_ok=True,
            coord_space="ABS_ORIG",       # Ultralytics 표준 predict, Q20 봉인 경로
        ))
    return records


def score_records(records, gold_codes, gold_boxes, classes):
    pred_codes = {r.image_id: sorted(r.iso_codes) for r in records}
    pred_boxes = {
        r.image_id: [(d.iso_code, tuple(d.bbox_px)) for d in r.defects if d.bbox_px]
        for r in records
    }
    pred_scored = {
        r.image_id: [
            (d.iso_code, tuple(d.bbox_px), d.score or 0.0)
            for d in r.defects if d.bbox_px
        ]
        for r in records
    }
    det = score_detection(pred_codes, {k: sorted(v) for k, v in gold_codes.items()}, classes)
    loc = score_bbox_iou(pred_boxes, gold_boxes)
    ap = coco_map(pred_scored, gold_boxes, classes)
    return {
        **det.as_dict(),
        "miss_rate": 1.0 - det.defect_recall,
        "class_jaccard": class_jaccard(pred_codes, gold_codes),
        **loc.as_dict(),
        **ap,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="data/processed/aihub71761_rt_v1_pilot3000")
    ap.add_argument("--pilot", default="outputs/pilot_c")
    ap.add_argument("--out", default="outputs/pilot_d")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    snapshot, pilot, root = Path(args.snapshot), Path(args.pilot), Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    global LM
    LM = load_label_map()
    class_names = ["crack", "porosity", "lack_of_fusion", "slag_inclusion"]
    classes = [LM.iso_code(n) for n in class_names]

    rows = read_manifest(snapshot)
    eval_rows = [r for r in rows if r["split"] == "eval"]
    eval_ids = {r["image_id"] for r in eval_rows}
    gold_codes, gold_boxes = read_gold(snapshot, eval_ids)
    for iid in eval_ids:
        gold_codes.setdefault(iid, set())
    print(f"평가셋 {len(eval_rows)}장 (정상 {sum(1 for r in eval_rows if r['has_defect'] == 'False')})")

    # last 확인 — best 파일명 거부 + 산출물 목록에 best 부재 확인
    checkpoints = {
        ("sep_central", None): pilot / "sep_central" / "sep_central.npz",
        ("sep_local", "C1"): pilot / "sep_local" / "sep_local_c0.npz",
        ("sep_local", "C2"): pilot / "sep_local" / "sep_local_c1.npz",
        ("sep_local", "C3"): pilot / "sep_local" / "sep_local_c2.npz",
        ("sep_fed", None): pilot / "sep_fed" / "global_r003.npz",
    }
    for p in checkpoints.values():
        reject_best_checkpoint(p)
        if not p.exists():
            print(f"체크포인트 없음: {p} — 그 칸은 채점하지 않는다")
            return 1
    stray_best = list(pilot.rglob("best*.pt")) + list(pilot.rglob("best*.npz"))
    if stray_best:
        print(f"best 체크포인트 발견 {stray_best[:3]} — 채점 중단, 보고 필요")
        return 1

    all_records: list[PredictionRecord] = []
    metrics: dict[str, dict] = {}
    for (cell, client), ckpt in checkpoints.items():
        tag = cell if client is None else f"{cell}_{client}"
        print(f"[{tag}] 추론 시작 ({ckpt.name})")
        yolo = load_yolo_from_npz(ckpt, class_names)
        records = predict_cell(yolo, eval_rows, root, cell, client)
        all_records.extend(records)
        with (out / f"{tag}_s{SEED}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in records:
                fh.write(r.model_dump_json() + "\n")
        metrics[tag] = score_records(records, gold_codes, gold_boxes, classes)
        m = metrics[tag]
        print(f"  macro_f1 {m['macro_f1']:.4f} · miss {m['miss_rate']:.4f} · "
              f"IoU {m['bbox_iou']:.4f} · mAP50 {m['map_50']:.4f}")

    # 회복률 중간값 (Macro-F1 기준, 로컬은 3사 단순 평균)
    local_mean = float(np.mean([
        metrics["sep_local_C1"]["macro_f1"],
        metrics["sep_local_C2"]["macro_f1"],
        metrics["sep_local_C3"]["macro_f1"],
    ]))
    central = metrics["sep_central"]["macro_f1"]
    fed = metrics["sep_fed"]["macro_f1"]
    denom = central - local_mean
    recovery = (fed - local_mean) / denom * 100 if denom > 0 else None

    # 사전등록 대조 — relative (축소 표본)
    samples = [
        MetaSample(
            image_id=r["image_id"], width_px=int(r["width_px"]),
            height_px=int(r["height_px"]), file_bytes=0, n_channels=1,
            quant_table_id=0,
            iso_codes=tuple(sorted(gold_codes.get(r["image_id"], ()))),
        )
        for r in eval_rows
    ]
    bound = trivial_bound(samples, classes)

    # P9 첫 실측 — 세 칸(로컬은 클라이언트별) 전부, 같은 러너
    with (snapshot / "tiles.csv").open(encoding="utf-8", newline="") as fh:
        prov = {r["image_id"]: r["provenance"] for r in csv.DictReader(fh)}
    contexts, missing = contexts_from_snapshot(eval_rows, prov)
    p9 = p9_all_cells(all_records, contexts)

    payload = {
        "snapshot": str(snapshot), "seed": SEED, "conf": CONF,
        "n_eval": len(eval_rows),
        "metrics": metrics,
        "recovery": {
            "basis": "macro_f1",
            "central": central, "fed": fed, "local_mean": local_mean,
            "locals": {k: metrics[k]["macro_f1"] for k in
                       ("sep_local_C1", "sep_local_C2", "sep_local_C3")},
            "denominator": denom,
            "recovery_pct": recovery,
            "note": "시드 1세트 중간값 — 결론으로 쓰지 않는다",
        },
        "prereg": {
            "sample_trivial_bound": bound,
            "gate_basis": "표본 상대(relative)",
            "checks": {
                tag: {"macro_f1": m["macro_f1"], "above_trivial": m["macro_f1"] > bound}
                for tag, m in metrics.items()
            },
        },
        "p9": p9.as_dict(),
        "p9_context_missing": list(missing),
    }
    with (out / "score_detection_v1.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"저장: {out / 'score_detection_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
