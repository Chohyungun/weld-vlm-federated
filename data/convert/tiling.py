"""정상 파노라마 타일링 + RT 전량 재인코딩 (확정 규격).

8/25 게이트에서 잠긴 규격이다. 값은 `configs/base.yaml` 의 `preprocess.tile` 에 있고
이 모듈은 그 값을 받아 쓸 뿐 자체 기본값을 두지 않는다.

| 규칙 | 확정값 |
|---|---|
| 타일 규격 | 1280×720, 원본 1장당 1장(k=1) |
| τ | 밴드 포함 (band ⊆ tile) |
| 밴드 초과분 | 밴드 중심 크롭 (tile ⊆ band 라 전 화소가 검사 영역이다) |
| 타일 선택 | `sha256(image_id + seed) mod N_cand`. **split 을 입력으로 받지 않는다** |
| 내용 기반 선택 | **금지.** 중앙 선택도 밝기·대비 기준 선택도 쓰지 않는다 |
| 패딩 | **금지.** 경계를 넘는 후보는 기각한다 |

평가셋에도 예외 없이 같은 규칙·같은 시드를 적용한다. 평가셋 정상만 파노라마로 남으면
헤드라인이 결함이 아니라 이미지 크기를 재게 된다.

**원본을 고치지 않는다.** 읽기만 하고 결과는 새 경로에 쓴다(불변조건 1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

#: 폐기·처리 사유 코드. 회계는 이 코드 단위로 집계한다.
REASON_OK = "ok"
REASON_TILED = "tiled"                     # 파노라마에서 타일을 떴다
REASON_CROPPED_BAND = "oversized_band_cropped"   # 밴드가 타일보다 커서 중심 크롭
REASON_NO_BAND = "no_band_polygon"         # 정상인데 밴드 폴리곤이 없다
REASON_NO_CANDIDATE = "no_valid_candidate"  # 패딩 금지 조건에서 후보가 0개
REASON_SMALL = "smaller_than_tile"         # 원본이 타일보다 작다


@dataclass(frozen=True)
class TilePlan:
    """이미지 한 장을 어떻게 처리할지. 픽셀을 읽기 전에 라벨만으로 정해진다."""

    image_id: str
    reason: str
    box: tuple[int, int, int, int] | None      # (x0, y0, x1, y1) 원본 좌표. None 이면 폐기
    n_candidates: int = 0

    @property
    def keep(self) -> bool:
        return self.box is not None


def _align(v: int, mult: int) -> int:
    return (v // mult) * mult


def candidate_origins(
    width: int, tile_w: int, stride_x: int, align: int
) -> list[int]:
    """가로 후보 격자. 패딩 금지라 마지막 후보는 오른쪽 경계에 붙인다."""
    if width < tile_w:
        return []
    xs = list(range(0, width - tile_w + 1, stride_x))
    last = width - tile_w
    if xs[-1] != last:
        xs.append(last)
    if align > 1:
        xs = sorted({min(_align(x, align), _align(last, align)) for x in xs})
    return xs


def select_index(image_id: str, seed: int, n: int) -> int:
    """`sha256(image_id + seed) mod n`.

    **split 을 받지 않는다.** 학습·평가에 같은 함수가 같은 시드로 적용된다.
    내용(밝기·대비·중앙 여부)을 보지 않는다 — 정상 쪽에만 걸리는 편향이 되기 때문이다.
    """
    if n <= 0:
        raise ValueError("후보가 없다")
    h = hashlib.sha256(f"{image_id}|{seed}".encode()).digest()
    return int.from_bytes(h[:8], "big") % n


def plan_tile(
    *,
    image_id: str,
    width: int,
    height: int,
    is_normal: bool,
    band: tuple[int, int, int, int] | None,
    tile_w: int,
    tile_h: int,
    stride_x: int,
    align: int,
    seed: int,
) -> TilePlan:
    """이 이미지를 어떻게 자를지 정한다. 픽셀을 읽지 않는다.

    - 결함 이미지는 이미 타일 규격이면 그대로 두고, 아니면 폐기한다(좌표 재계산 경로를
      만들지 않는다).
    - 정상 이미지는 밴드에 세로를 맞추고 가로만 후보 격자에서 고른다.
    """
    if (width, height) == (tile_w, tile_h):
        return TilePlan(image_id, REASON_OK, (0, 0, tile_w, tile_h), 1)
    if width < tile_w or height < tile_h:
        return TilePlan(image_id, REASON_SMALL, None)
    if not is_normal:
        # 결함인데 규격이 다르다. 폐기 대상 (확정 규격의 비-R1 결함 처리)
        return TilePlan(image_id, REASON_SMALL if width < tile_w else REASON_NO_CANDIDATE, None)
    if band is None:
        return TilePlan(image_id, REASON_NO_BAND, None)

    _, by0, _, by1 = band
    band_h = by1 - by0
    cy = (by0 + by1) // 2
    y0 = cy - tile_h // 2
    y0 = max(0, min(y0, height - tile_h))        # 패딩 금지: 경계 안으로 민다
    if align > 1:
        y0 = min(_align(y0, align), _align(height - tile_h, align))

    xs = candidate_origins(width, tile_w, stride_x, align)
    if not xs:
        return TilePlan(image_id, REASON_NO_CANDIDATE, None)
    x0 = xs[select_index(image_id, seed, len(xs))]

    reason = REASON_CROPPED_BAND if band_h > tile_h else REASON_TILED
    return TilePlan(image_id, reason, (x0, y0, x0 + tile_w, y0 + tile_h), len(xs))


def encode_tile(
    src: Path,
    box: tuple[int, int, int, int],
    dst: Path,
    *,
    mode: str,
    quality: int,
    progressive: bool,
    optimize: bool,
) -> tuple[str, int, int]:
    """원본에서 상자를 떠 새 경로에 다시 인코딩한다. 반환값은 (sha256, 폭, 높이).

    **RT 전량**이 이 경로를 지난다. 타일만 재인코딩하면 양자화표가 정상 쪽에만 달라져
    지름길이 그대로 이사한다. 결함 크롭도 같은 설정으로 다시 쓴다.
    """
    with Image.open(src) as im:
        im = im.convert(mode)
        tile = im.crop(box)
        buf = BytesIO()
        tile.save(
            buf, format="JPEG", quality=quality,
            progressive=progressive, optimize=optimize,
        )
    data = buf.getvalue()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)                        # 새 경로에만 쓴다. 원본은 건드리지 않는다
    return hashlib.sha256(data).hexdigest(), box[2] - box[0], box[3] - box[1]
