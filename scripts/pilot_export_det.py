"""검출 칸 예측 raw 내보내기 — 평가 담당이 집어 갈 원시 출력.

원시 출력 계약(평가 스펙 §2-3):
    detections.jsonl  {image_id, boxes:[{cls, xyxy_px, conf}], coord_space, coord_cfg_hash}

- 대상: 글로벌 평가셋(eval split). **이것이 이 실험에서 평가셋에 접근하는 최초이자
  마지막 학습측 접점이다** — 추론만 하고 어떤 학습·선택에도 쓰지 않는다.
- 가중치: 각 칸의 last(npz). best 개념 자체가 없다.
- conf 임계 0.25 (5칸 공통 고정, 스펙 Q4).
- imgsz 416 (파일럿 프로파일 — 학습과 동일해야 한다).
- CPU 로 돈다. GPU 는 통합형 학습이 쓰고 있고, 예측 생성은 nano 모델이라 CPU 로 충분하다.
- Ultralytics 표준 predict 는 좌표를 원본 픽셀로 복원한다(coord_space=ABS_ORIG 선언,
  평가 담당의 Q20 실측 봉인 대상).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.manifest_io import load_snapshot, split_view
from detection import serialize
from detection.init_weights import build_initial_weights
from vlm.coords import CoordCfg, coord_cfg_hash

SNAPSHOT_DIR = "data/processed/aihub71761_rt_v1_pilot3000"
OUT_ROOT = Path("outputs/pilot_c").resolve()
CONF = 0.25
IMGSZ = 416


def _cells(root: Path) -> dict[str, Path]:
    return {
        "sep_local_c0": root / "sep_local" / "sep_local_c0.npz",
        "sep_local_c1": root / "sep_local" / "sep_local_c1.npz",
        "sep_local_c2": root / "sep_local" / "sep_local_c2.npz",
        "sep_central": root / "sep_central" / "sep_central.npz",
        "sep_fed": root / "sep_fed" / "latest.npz",
    }


CELLS = _cells(OUT_ROOT)


def load_model_from_npz(npz_path: Path):
    """npz(raw fp32, 정본 키 순서) → 추론 가능한 YOLO 객체."""
    from ultralytics import YOLO

    # 구조·키는 초기 가중치 빌더와 같은 경로로 만든다 (nc=4, 시드는 값에 무관 — 곧 덮어쓴다)
    _, keys, ref = build_initial_weights(pretrained="yolo11n.pt", nc=4, seed=0,
                                         cache_path=None)
    loaded = np.load(npz_path)
    arrays = [loaded[k] for k in loaded.files] if loaded.files[0] in keys else \
             [loaded[f] for f in loaded.files]
    # sep_* npz 는 save_cell_weights 가 위치 인자(arr_0..)로 저장했다 — 순서가 정본이다
    serialize.assert_compatible(arrays, keys, ref)
    sd = serialize.ndarrays_to_state_dict(arrays, keys, ref)

    from detection.init_weights import _build
    det = _build("yolo11n.pt", nc=4, seed=0)
    det.load_state_dict(sd, strict=True)
    det.eval()

    import tempfile
    tmp = Path(tempfile.mkstemp(suffix=".pt")[1])
    torch.save({"model": det.half(), "train_args": {}}, tmp)
    return YOLO(tmp), tmp


def main() -> None:
    sn = load_snapshot(SNAPSHOT_DIR)
    eval_m = split_view(sn.manifest, "eval")
    print(f"평가셋 {len(eval_m)}장 · conf {CONF} · imgsz {IMGSZ} · CPU")
    cfg_hash = coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG"))

    repo = Path.cwd().resolve()
    paths = [str(repo / p) for p in eval_m["rel_path"]]
    ids = list(eval_m["image_id"])

    for cell, npz in CELLS.items():
        if not npz.exists():
            print(f"  {cell}: 가중치 없음({npz}) — 건너뜀")
            continue
        t0 = time.perf_counter()
        model, tmp = load_model_from_npz(npz)
        out_path = OUT_ROOT / "predictions" / f"{cell}.detections.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_boxes = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for i in range(0, len(paths), 64):
                results = model.predict(paths[i:i+64], imgsz=IMGSZ, conf=CONF,
                                        device="cpu", verbose=False)
                for image_id, res in zip(ids[i:i+64], results):
                    boxes = [
                        {"cls": int(c), "xyxy_px": [round(float(v), 2) for v in xy],
                         "conf": round(float(cf), 4)}
                        for c, xy, cf in zip(res.boxes.cls, res.boxes.xyxy, res.boxes.conf)
                    ]
                    n_boxes += len(boxes)
                    fh.write(json.dumps(
                        {"image_id": image_id, "boxes": boxes,
                         "coord_space": "ABS_ORIG", "coord_cfg_hash": cfg_hash},
                        ensure_ascii=False) + "\n")
        del model
        try:
            tmp.unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows 가 핸들을 놓기 전이다. C: 임시 폴더라 잔류해도 무해하다
        print(f"  {cell}: {len(paths)}장 → 박스 {n_boxes} ({time.perf_counter()-t0:.0f}s) → {out_path}")


if __name__ == "__main__":
    # 어블레이션은 팔마다 평가셋이 다르다(431 / 418장). 같은 코드로 스냅샷·산출 경로만 바꾼다.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.snapshot:
        SNAPSHOT_DIR = a.snapshot
    if a.out:
        OUT_ROOT = Path(a.out).resolve()
        CELLS = _cells(OUT_ROOT)
    main()
