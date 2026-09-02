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

**본실험 규모에서 죽던 결함 둘을 여기서 고쳤다**(82번 §10-3-1, C 가 §4-6 게이트에서 실측).
- 모델 규모가 `yolo11n.yaml` 로 박혀 있어 본실험 프로파일(YOLO11s) 가중치를 못 읽었다
  → `model_cfg` 인자. 프로파일이 틀리면 npz 가 어느 규모인지 말하며 실패한다.
- 평가셋 전량의 경로를 한 번에 넘겨 전처리가 36.7GB 단일 텐서가 됐다 → 내부 청킹.
  호출부는 여전히 `predict_cell` 하나만 부른다(추론 경로 한 벌).
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from data.label_map import load_label_map
from detection import serialize
from evaluation.params import ScoringParams, detection_profiles
from evaluation.schema import SCHEMA_VERSION, PredictionRecord


class ProfileMismatchError(RuntimeError):
    """npz 의 형상이 요청한 모델 cfg 와 맞지 않는다 — 채점 프로파일이 학습 프로파일과 다르다."""


_FIRST_CONV_OUT_TO_SCALE: dict[int, str] = {16: "n", 32: "s", 64: "m 또는 l", 96: "x"}


def npz_scale_hint(arrays: Sequence[np.ndarray]) -> str:
    """npz 첫 배열(`model.0.conv.weight`)의 출력 채널로 YOLO11 규모를 짐작한다. **오류 문구 전용** —
    판정은 `serialize.assert_compatible` 의 전수 형상 대조가 한다."""
    if not arrays or getattr(arrays[0], "ndim", 0) != 4:
        return "규모 판별 불가"
    c = int(arrays[0].shape[0])
    return f"첫 합성곱 출력 채널 {c} → yolo11{_FIRST_CONV_OUT_TO_SCALE.get(c, '?')} 계열"

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


def load_yolo_from_npz(
    npz_path: Path, class_names: Sequence[str], imgsz: int, *, model_cfg: str
):
    """C 의 npz 상태를 **`model_cfg` 가 지정한 규모**의 YOLO11(nc=4) 에 주입한다.

    전부 C 의 serialize 경유다. `model_cfg` 는 호출부가 `ScoringParams.model_cfg`
    (프로파일에서 유도)로 준다 — 이전 판은 `yolo11n.yaml` 을 박아 두어 본실험 프로파일
    가중치에서 `shape 불일치 (32,3,3,3) != (16,3,3,3)` 로 죽었다(82번 §10-3-1 결함 1).
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    dm = DetectionModel(cfg=model_cfg, nc=len(class_names), verbose=False)
    keys = serialize.canonical_keys(dm.state_dict())
    z = np.load(npz_path)
    arrays = [z[f"arr_{i}"] for i in range(len(z.files))]
    ref = dm.state_dict()
    try:
        serialize.assert_compatible(arrays, keys, ref)
    except serialize.SerializeError as e:
        table = ", ".join(
            f"{name}→{p['model_cfg']}" for name, p in detection_profiles().items()
        )
        raise ProfileMismatchError(
            f"{npz_path} 를 {model_cfg} 에 주입할 수 없다 — {e}. npz 는 {npz_scale_hint(arrays)}. "
            f"채점 프로파일(--profile)이 학습 프로파일과 같은지 확인하라: {table}"
        ) from e
    dm.load_state_dict(serialize.ndarrays_to_state_dict(arrays, keys, ref), strict=True)
    dm.names = dict(enumerate(class_names))
    # ultralytics 체크포인트 형태로 감싸 YOLO 의 정식 로드 경로를 태운다. 임시 파일은
    # npz 옆이 아니라 임시 디렉터리에 둔다 — npz 가 있는 곳은 C 의 산출 폴더다.
    tmp_dir = Path(tempfile.mkdtemp(prefix="weld_yolo_"))
    try:
        tmp = tmp_dir / (npz_path.stem + ".pt")
        torch.save({"model": dm.float(), "train_args": {"imgsz": imgsz}}, tmp)
        yolo = YOLO(str(tmp))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return yolo


def _record_from_result(
    row: dict, res, lm, cell: str, client: str | None, params: ScoringParams
) -> PredictionRecord:
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
    return PredictionRecord(
        schema_version=SCHEMA_VERSION,
        image_id=row["image_id"], cell=cell, client=client, seed=params.seed,
        defects=defects,
        verdict="판정불가",           # 판정부(⑤) 미실행 — 검출 축 선채점
        cited_clauses=[],
        parse_ok=True,
        coord_space="ABS_ORIG",       # Ultralytics 표준 predict, Q20 봉인 경로
    )


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

    **`params.predict_chunk` 장씩 나눠 넘긴다.** 전량의 경로를 한 번에 주면 Ultralytics
    가 전처리를 한 텐서로 뭉쳐 평가셋 12,461장에서 36.7GB 를 단일 할당하려다 죽는다
    (82번 §10-3-1 결함 2). 레코드 순서는 입력 행 순서와 같다.
    """
    lm = load_label_map()
    records: list[PredictionRecord] = []
    chunk = int(params.predict_chunk)
    rows = list(rows)
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        paths = [str(root / r["rel_path"]) for r in part]
        results = yolo.predict(
            source=paths, imgsz=params.imgsz, conf=conf, device=params.device,
            max_det=params.max_det, verbose=False, stream=True,
        )
        for row, res in zip(part, results, strict=True):
            records.append(_record_from_result(row, res, lm, cell, client, params))
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
