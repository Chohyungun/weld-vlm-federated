"""검출 3칸 추론 — 65번 스크립트에 인라인돼 있던 것을 모듈로 올렸다 (77번 과제 6).

옮긴 이유는 두 가지다.
- 65·66번의 채점 경로가 스크립트 두 벌에 나뉘어 있어 "단일 채점기" 주장을 코드 구조가
  받쳐 주지 못했다. 추론은 여기, 채점은 `evaluation.score`, 파라미터는
  `evaluation.params` 로 갈라 두면 진입점이 몇 개든 같은 함수를 탄다.
- **임계를 인자로 받아야 스윕이 가능하다.** 65번은 `CONF = 0.25` 를 모듈 상수로 박아
  분리형만 잘린 뒤 채점됐다(감사 D-1).

가중치 로드는 C 의 `detection.serialize` 를 그대로 쓴다 — 재구현 금지. 좌표는
Ultralytics 표준 predict 의 원본 픽셀 복원(Q20 실측 봉인)을 그대로 받는다.
D 는 좌표를 변환하지 않는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from data.label_map import load_label_map
from detection import serialize
from evaluation.params import ScoringParams
from evaluation.schema import SCHEMA_VERSION, PredictionRecord

CHECKPOINTS: tuple[tuple[str, str | None, str], ...] = (
    ("sep_central", None, "sep_central/sep_central.npz"),
    ("sep_local", "C1", "sep_local/sep_local_c0.npz"),
    ("sep_local", "C2", "sep_local/sep_local_c1.npz"),
    ("sep_local", "C3", "sep_local/sep_local_c2.npz"),
    ("sep_fed", None, "sep_fed/global_r003.npz"),
)
"""칸 → last 체크포인트. **best 는 없다**(불변조건 3-2). 경로는 C 의 산출 규약이다."""


def cell_tag(cell: str, client: str | None) -> str:
    return cell if client is None else f"{cell}_{client}"


def checkpoint_paths(pilot: Path) -> dict[tuple[str, str | None], Path]:
    return {(cell, client): pilot / rel for cell, client, rel in CHECKPOINTS}


def load_yolo_from_npz(npz_path: Path, class_names: Sequence[str], imgsz: int):
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
    torch.save({"model": dm.float(), "train_args": {"imgsz": imgsz}}, tmp)
    yolo = YOLO(str(tmp))
    tmp.unlink(missing_ok=True)
    return yolo


def predict_cell(
    yolo,
    rows: Sequence[dict],
    root: Path,
    cell: str,
    client: str | None,
    params: ScoringParams,
    conf: float,
) -> list[PredictionRecord]:
    """평가셋 전량 추론 → 계약 #4 레코드.

    `conf` 를 인자로 받는다. 스윕용 하한 추론에서는 `params.conf_floor` 를,
    65번 재현에서는 `params.conf.value` 를 준다 — **호출부가 무엇을 쓰는지 밝히게 한다.**
    """
    lm = load_label_map()
    records: list[PredictionRecord] = []
    paths = [str(root / r["rel_path"]) for r in rows]
    results = yolo.predict(
        source=paths, imgsz=params.imgsz, conf=conf, device=params.device,
        max_det=params.max_det, verbose=False, stream=True,
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
                "iso_code": lm.iso_code(dt),
                "bbox_px": [x1, y1, x2, y2],
                "score": float(b.conf),
                "size_px": max(x2 - x1, y2 - y1),
                "size_basis": "major_axis",
                "retrieved": None,
            })
        records.append(PredictionRecord(
            schema_version=SCHEMA_VERSION,
            image_id=row["image_id"], cell=cell, client=client, seed=params.seed,
            defects=defects,
            verdict="판정불가",           # 판정부(⑤) 미실행 — 검출 축 선채점
            cited_clauses=[],
            parse_ok=True,
            coord_space="ABS_ORIG",       # Ultralytics 표준 predict, Q20 봉인 경로
        ))
    return records


def filter_by_conf(
    records: Sequence[PredictionRecord], conf: float
) -> list[PredictionRecord]:
    """점수 ≥ conf 인 박스만 남긴 사본. **원 레코드를 바꾸지 않는다.**

    NMS 는 더 높은 점수의 박스만이 억제자가 되므로, 하한 추론 결과를 여기서 자르면
    그 임계로 직접 추론한 것과 같은 생존 집합이 된다. 이 동치는 가정이 아니라
    65번 산출물(0.25)과의 비트 대조로 실측한다.
    """
    out: list[PredictionRecord] = []
    for r in records:
        kept = [d for d in r.defects if (d.score or 0.0) >= conf]
        out.append(r.model_copy(update={"defects": kept}))
    return out
