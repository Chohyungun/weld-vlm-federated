"""매니페스트 → Ultralytics 학습 뷰 (계약 #2 소비 지점).

학습 코드는 원본 폴더를 직접 읽지 않는다. 이 모듈이 **데이터 담당의 로더만 통해**
스냅샷을 읽고, Ultralytics 가 요구하는 형태(images/ + labels/ + data.yaml)로 뷰를 만든다.

## 지키는 것

- **조인은 `join_defects` 로만 한다.** 자체 merge 를 쓰면 정상 이미지 1행 유지, 결측
  판별(row_kind) 같은 조인 의미론이 두 벌이 되고, 언젠가 조용히 갈라진다.
- **원본 불변.** 이미지는 하드링크로 뷰에 건다(같은 NTFS 볼륨). 하드링크가 안 되면
  복사로 물러난다. 원본 디렉터리에는 어떤 파일도 쓰지 않는다 — Ultralytics 는 라벨
  txt 를 이미지 옆에서 찾으므로, 뷰 디렉터리를 따로 만들지 않으면 라벨이 데이터 담당
  소유 폴더에 흘러 들어간다.
- **라벨 문자열 하드코딩 금지.** 클래스 id 는 `configs/label_map.yaml` 의
  `train_class_id` 에서 온다.
- `split == "eval"` 뷰는 학습용으로 만들지 않는다. 요청 자체를 거부한다.

## geom_valid=False 처리

기하가 깨진 어노테이션(면적 0, 좌표 역전 등)은 라벨에서 **제외하되 건수를 센다.**
조용히 떨어뜨리면 "라벨이 적은 것"과 "떨어뜨린 것"을 사후에 구분할 수 없다.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from data.manifest_io import Snapshot, join_defects, split_view

__all__ = ["ViewResult", "build_yolo_view", "load_class_names"]

_LABEL_MAP = Path("configs/label_map.yaml")


def load_class_names(label_map_path: str | Path = _LABEL_MAP) -> dict[int, str]:
    """`train_class_id` → 영문 클래스명. 이 스냅샷(RT)에 나타날 수 있는 클래스만.

    RIAWELC 전용 클래스(lack_of_penetration, id 4)는 RT 뷰에 넣지 않는다.
    nc 가 5가 되면 검출 헤드 모양이 바뀌고, 나타나지 않는 클래스가 지표 분모에 낀다.
    """
    doc = yaml.safe_load(Path(label_map_path).read_text(encoding="utf-8"))
    aihub_types = set(doc["sources"]["aihub71761"]["mapping"].values())
    names = {
        int(spec["train_class_id"]): key
        for key, spec in doc["defect_types"].items()
        if key in aihub_types
    }
    if sorted(names) != list(range(len(names))):
        raise ValueError(f"train_class_id 가 0부터 연속이어야 한다: {sorted(names)}")
    return names


@dataclass
class ViewResult:
    data_yaml: Path
    n_images: dict[str, int] = field(default_factory=dict)
    n_boxes: dict[str, int] = field(default_factory=dict)
    n_geom_invalid: int = 0
    n_background: int = 0     # 결함 0개 이미지 (빈 라벨 파일)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def build_yolo_view(
    snapshot: Snapshot,
    *,
    out_dir: str | Path,
    train_client: str | None,
    label_map_path: str | Path = _LABEL_MAP,
) -> ViewResult:
    """한 칸의 학습 뷰를 만든다.

    Args:
        train_client: `"C1"`·`"C2"`·`"C3"` 또는 중앙집중 칸이면 `None`(학습 풀 전체).
            val 은 항상 전체 val split 이다 — 로깅 전용이고 훈련 중 접근하지 않지만
            data.yaml 형식이 요구한다.
    """
    names = load_class_names(label_map_path)
    type_to_id = {v: k for k, v in names.items()}

    out = Path(out_dir).resolve()
    joined = join_defects(snapshot)

    result = ViewResult(data_yaml=out / "data.yaml")
    repo_root = Path.cwd().resolve()

    for split, client in (("train", train_client), ("val", None)):
        if split == "eval":  # 방어적 — 아래 split_view 호출은 리터럴이지만 규칙을 코드로 남긴다
            raise ValueError("평가셋으로 학습 뷰를 만들 수 없다")
        m = split_view(snapshot.manifest, split, client=client)
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        rows = joined[joined["image_id"].isin(set(m["image_id"]))]
        n_boxes = 0
        for image_id, g in rows.groupby("image_id", sort=True):
            first = g.iloc[0]
            src = (repo_root / first["rel_path"]).resolve()
            stem = Path(str(first["rel_path"])).stem
            _link_or_copy(src, img_dir / src.name)

            w, h = float(first["width_px"]), float(first["height_px"])
            lines: list[str] = []
            for _, r in g.iterrows():
                if r["row_kind"] != "defect":
                    continue
                if not bool(r["geom_valid"]):
                    result.n_geom_invalid += 1
                    continue
                cls = type_to_id[str(r["defect_type"])]
                x1, y1, x2, y2 = (float(r[k]) for k in
                                  ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px"))
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_boxes += 1
            if not lines:
                result.n_background += 1
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                 encoding="ascii")
        result.n_images[split] = int(len(rows["image_id"].unique()))
        result.n_boxes[split] = n_boxes

    data = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "names": {k: names[k] for k in sorted(names)},
    }
    result.data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                encoding="utf-8")
    return result
