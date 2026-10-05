"""RIAWELC 어댑터 — 분류 패치 24,407장. 스펙 §3-2.

원본에 위치 라벨이 없으므로 `has_localization=false` 이고, 결함 이미지는 annotations 1행
(기하 컬럼 전부 null)을 갖는다 — N1(위치 없음)과 N2(결함 없음)가 데이터 구조에서 갈린다.

## 실물 레이아웃 (2026-08-21 전수 실측)

    <raw_root>/riawelc/dataset/DB - Copy/{training|validation|testing}/{클래스}/*.png

클래스 디렉터리는 **이미지의 부모**이고, 그 위가 저자 분할이다. 중간 경로 이름
(`dataset`, `DB - Copy`)은 판본마다 다를 수 있으므로 **깊이를 가정하지 않고** 이미지를
재귀 탐색한 뒤 부모 디렉터리명을 클래스로 읽는다.

## 저자 제공 분할은 쓰지 않는다

`training/validation/testing` 이 원본에 들어 있지만 **분할 정보로 해석하지 않는다.**
이 연구는 글로벌 평가셋을 회사별 분할보다 **먼저** 직접 선분리한다(불변조건 1-3).
저자 분할을 섞으면 다섯 칸이 서로 다른 기준으로 채점되어 비교가 무너진다.
이 디렉터리들은 경로의 일부일 뿐이고 `split` 컬럼은 이 저장소의 분할 단계만 채운다.

## 묶음 ID 는 파일명에서 온다

24,407장은 독립 표본이 아니라 **479개 모원본**을 격자로 자른 타일이다
(`bam5_Img2_A80_S5_[3][10].png` → 모원본 `bam5_Img2_A80_S5`, 타일 [3][10]).
같은 모원본의 타일이 학습과 평가로 갈리면 누수다(불변조건 1-5). pHash 는 비겹침 타일을
원리적으로 못 잡으므로(스펙 §6-10), **파일명 접두사가 이 데이터셋의 실질 누수 방어선**이다.
어댑터가 접두사를 `group_key` 로 채우고 dedup 단계가 E2 엣지로 받는다. pHash 는 보조다.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path

from data.ingest.base import (
    Capabilities,
    DefectRecord,
    ImageRecord,
    RawItem,
    derive_capabilities,
    file_sha256,
    image_size,
)
from data.label_map import LabelMap

IMAGE_SUFFIXES = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")

#: 저자 제공 분할 디렉터리. **분할 정보로 쓰지 않는다** — 클래스 탐색에서 건너뛰기 위한 목록.
AUTHOR_SPLIT_DIRS = frozenset({"training", "validation", "testing", "train", "val", "test"})

#: `{모원본}_[행][열]{...}.png` — 접두사가 모원본 식별자다.
#: 뒤쪽 `.*` 가 ` - Copia`(사본 30건) 같은 접미를 흡수한다. 흡수하지 않으면 그 30건이
#: 각각 독립 묶음이 되어(479 → 509) 같은 모원본의 타일이 학습·평가로 갈린다.
TILE_RE = re.compile(r"^(?P<prefix>.+?)_\[\d+\]\[\d+\].*$")


def mother_image_key(filename: str) -> str | None:
    """파일명 → 모원본 접두사. 타일 패턴이 아니면 None."""
    m = TILE_RE.match(Path(filename).stem)
    return m.group("prefix") if m else None


class RiawelcAdapter:
    source = "riawelc"
    version = "riawelc-v2.0"

    #: 원본에 재질 정보가 없다. 추정하지 않는다 — UNK 는 결측이 아니라 "원본이 안 준다"다.
    material = "UNK"
    modality = "RT"
    label_type = "classification"
    has_localization = False

    def discover(self, raw_root: Path, *, rel_base: Path | None = None) -> Iterator[RawItem]:
        """`<raw_root>/riawelc/` 이하를 재귀 탐색한다. 읽기만 한다 — 쓰기·이동·삭제 없음.

        클래스 = 이미지의 **부모 디렉터리명**. 저자 분할 디렉터리는 클래스로 보지 않는다.
        """
        raw_root = Path(raw_root)
        base = Path(rel_base) if rel_base is not None else raw_root.parent.parent
        root = raw_root / self.source
        if not root.is_dir():
            raise FileNotFoundError(
                f"{root} 가 없다. RIAWELC 를 <raw_root>/riawelc/ 아래에 두어라 "
                "(원본 불변 — 다른 경로에 쓰지 않는다)"
            )

        for img in sorted(root.rglob("*")):
            if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = img.parent.name
            if label.lower() in AUTHOR_SPLIT_DIRS:
                raise ValueError(
                    f"{img} 의 부모가 저자 분할 디렉터리({label})다. 클래스 디렉터리가 없는 "
                    "레이아웃이므로 어댑터를 점검해야 한다 — 저자 분할을 클래스로 읽으면 안 된다"
                )
            yield RawItem(
                path=img,
                rel_path=img.relative_to(base).as_posix(),
                label=label,
                extra={"group_key": mother_image_key(img.name)},
            )

    def parse(self, item: RawItem, label_map: LabelMap) -> ImageRecord:
        """부모 디렉터리명 = 클래스. 사상표에 없으면 예외다(unmapped_policy=fail)."""
        width, height = image_size(item.path)
        defect_type = label_map.to_defect_type(self.source, item.label)   # None = 정상

        defects: list[DefectRecord] = []
        if defect_type is not None:
            # 결함이 있으면 1행. 종류·코드는 있고 기하만 null 이다.
            defects.append(
                DefectRecord(
                    seq=0,
                    src_label_raw=item.label,
                    defect_type=defect_type,
                    iso_code=label_map.iso_code(defect_type),
                    geom_valid=True,
                    geom_flags="",
                )
            )

        return ImageRecord(
            image_id=f"{self.source}:{item.rel_path}",
            source=self.source,
            rel_path=item.rel_path,
            sha256=file_sha256(item.path),
            width_px=width,
            height_px=height,
            modality=self.modality,
            material=self.material,
            label_type=self.label_type,
            has_localization=self.has_localization,
            ingest_version=self.version,
            defects=defects,
            group_key=item.extra.get("group_key"),
            notes="",
        )

    def capabilities(self, records: Sequence[ImageRecord]) -> Capabilities:
        return derive_capabilities(records)
