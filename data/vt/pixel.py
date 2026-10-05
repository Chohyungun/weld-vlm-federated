"""VT 화소 꼴 — 계획(라벨만으로)과 다시 인코딩 (2판 4-1 · 4-2 · 4판 5-3).

계획은 픽셀을 읽지 않는다. 장마다 처리(그대로 · 창으로 자르기 · 돌려서 타일 · 빼기)와 상자를 정한다.

| 무리 | 처리 | 출처(`tiles.csv` 어휘) |
|---|---|---|
| 결함 1280×720 | 그대로 | `N-crop` |
| 결함 · 설정 `tile.defect_crop` 의 크기(1800×720 융합불량) | 모든 결함의 바깥 정수화 상자가 드는 후보 창(`candidate_origins`)으로 자른다. 둘 다 들면 해시로. 드는 창이 없으면 뺀다 | `N-tile` |
| 결함 · 설정 `tile.defect_exclude` 의 크기 | 뺀다 | — |
| 결함 · 그 밖의 크기 | **멈춘다** | — |
| 정상 · 설정 `tile.normal_exclude` 의 크기 | 뺀다 | — |
| 정상 세로(높이 > 폭) | 반시계 90° 로 돌린 뒤 P1 — 원점은 돌린 뒤 프레임 | `N-tile` · `N-band` |
| 정상 1280×720 | 그대로 | `N-crop` |
| 정상 그 밖 | P1 — RT 와 같은 함수(`data.convert.tiling.plan_tile`), 비드는 세로 폭이 가장 큰 정상 다각형의 상자 | `N-tile` · `N-band` |

**돌림의 좌표.** 폭 W 인 원천을 반시계 90° 돌리면 연속 좌표 (x, y) 는 (y, W − x) 로 간다. 영상은
`Image.Transpose.ROTATE_90` 이고 이것은 `np.rot90(k=1)` 과 같다(시험 VT-4).

다시 인코딩은 결함 · 정상이 같은 경로다 — 해독 → (돌림) → 자르기 → 설정의 모드 · 품질로 다시 인코딩.
해독한 크기가 라벨의 선언과 다르거나, EXIF 방향 태그가 1 이 아니거나, 원천 모드가 설정과 다르면 **멈춘다**(VT-3b).
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass

from PIL import Image

from data.convert.tiling import (
    REASON_CROPPED_BAND,
    REASON_OK,
    REASON_TILED,
    candidate_origins,
    plan_tile,
    select_index,
)
from data.manifest_io import REASON_TO_PROVENANCE
from data.vt.labels import VtLabel

# 출처 어휘는 계약의 단일 소스(`data/manifest_io.REASON_TO_PROVENANCE`)에서 읽는다 — 다시 적지 않는다
PROV_CROP = REASON_TO_PROVENANCE[REASON_OK]
PROV_TILE = REASON_TO_PROVENANCE[REASON_TILED]
PROV_BAND = REASON_TO_PROVENANCE[REASON_CROPPED_BAND]
REASON_DEFECT_WINDOW = "defect_window"
REASON_NO_WINDOW = "defect_no_window"
EXIF_ORIENTATION = 0x0112


class PixelFormError(RuntimeError):
    """계획 · 인코딩이 규칙 밖의 장을 만났다 — 멈춘다."""


@dataclass(frozen=True)
class VtPlan:
    image_id: str
    vt_id: int
    keep: bool
    reason: str
    provenance: str | None = None
    rot90_k: int = 0
    box: tuple[int, int, int, int] | None = None   # 돌린 뒤 프레임의 (x0, y0, x1, y1)
    src_w: int = 0
    src_h: int = 0
    n_candidates: int = 0


def rotate_ccw_points(xs, ys, width: int) -> tuple[list[float], list[float]]:
    """반시계 90°. 폭 `width` 인 원천의 연속 좌표 (x, y) → (y, width − x)."""
    return [float(y) for y in ys], [float(width) - float(x) for x in xs]


def outer_int_box(xs, ys) -> tuple[int, int, int, int]:
    return (math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys)))


#: 설정의 규칙 문자열 — 동작은 RT 의 `plan_tile` 에서 온다. 문자열이 이 값이 아니면 그 동작을 이 코드가 하지 않는다(06b m-15)
TILE_SEMANTICS = {"vertical_anchor": "band_center", "tau_mode": "band_containment", "padding": "forbidden",
                  "selection": "sha256(image_id + seed) mod N_cand"}


def _tile_spec(cfg: dict) -> dict:
    t = cfg["tile"]
    for k, want in TILE_SEMANTICS.items():
        if t[k] != want:
            raise PixelFormError(f"설정 tile.{k} = {t[k]!r} — 이 코드는 {want!r} 만 한다")
    return {"tile_w": int(t["tile_size"][0]), "tile_h": int(t["tile_size"][1]), "stride_x": int(t["stride_x"]),
            "align": 8 if t["align8"] else 1, "seed": int(t["selection_seed"])}


def _match_size(entries: list[dict], folder: str | None, size: tuple[int, int]) -> dict | None:
    for e in entries:
        if tuple(e["size"]) == size and (folder is None or e.get("folder") in (None, folder)):
            return e
    return None


def plan_defect(lab: VtLabel, cfg: dict) -> VtPlan:
    s = _tile_spec(cfg)
    tw, th = s["tile_w"], s["tile_h"]
    base = {"image_id": lab.image_id, "vt_id": lab.vt_id, "src_w": lab.width, "src_h": lab.height}
    if lab.size == (tw, th):
        return VtPlan(**base, keep=True, reason=REASON_OK, provenance=PROV_CROP, box=(0, 0, tw, th), n_candidates=1)
    if _match_size(cfg["tile"]["defect_crop"], lab.folder, lab.size):
        boxes = [outer_int_box(a.xs, a.ys) for a in lab.anns
                 if a.cls == cfg["labels"]["defect_class"] and a.xs]
        if not boxes:
            raise PixelFormError(f"창으로 자를 결함에 좌표가 없다: {lab.image_id}")
        x0s, y0s = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1s, y1s = max(b[2] for b in boxes), max(b[3] for b in boxes)
        if lab.height != th:
            raise PixelFormError(f"창으로 자르는 결함의 높이가 타일과 다르다: {lab.image_id} {lab.size}")
        cands = candidate_origins(lab.width, tw, s["stride_x"], s["align"])
        fit = [c for c in cands if x0s >= c and x1s <= c + tw and y0s >= 0 and y1s <= th]
        if not fit:
            return VtPlan(**base, keep=False, reason=REASON_NO_WINDOW, n_candidates=len(cands))
        x0 = fit[select_index(lab.image_id, s["seed"], len(fit))]
        return VtPlan(**base, keep=True, reason=REASON_DEFECT_WINDOW, provenance=PROV_TILE,
                      box=(x0, 0, x0 + tw, th), n_candidates=len(fit))
    ex = _match_size(cfg["tile"]["defect_exclude"], lab.folder, lab.size)
    if ex:
        return VtPlan(**base, keep=False, reason=ex["reason"])
    raise PixelFormError(f"규칙에 없는 결함 크기다: {lab.image_id} {lab.folder} {lab.size}")


def plan_normal(lab: VtLabel, cfg: dict) -> VtPlan:
    s = _tile_spec(cfg)
    tw, th = s["tile_w"], s["tile_h"]
    base = {"image_id": lab.image_id, "vt_id": lab.vt_id, "src_w": lab.width, "src_h": lab.height}
    ex = _match_size(cfg["tile"]["normal_exclude"], None, lab.size)
    if ex:
        return VtPlan(**base, keep=False, reason=ex["reason"])
    k = int(cfg["tile"]["portrait_normal_rot90_k"]) if lab.height > lab.width else 0
    if k not in (0, 1):
        raise PixelFormError("돌림은 반시계 90° 한 번만 정의했다")
    if k == 1 and lab.width % 8:
        # 돌린 프레임의 8 의 배수 원점이 원천 JPEG 의 블록 격자와 맞는 것은 원천 폭이 8 의 배수일 때뿐이다(06b m-19)
        raise PixelFormError(f"돌릴 세로 사진의 폭이 8 의 배수가 아니다: {lab.image_id} {lab.size}")
    polys = [(list(a.xs), list(a.ys)) for a in lab.anns if a.cls == cfg["labels"]["normal_class"] and a.xs]
    w, h = lab.width, lab.height
    if k == 1:
        polys = [rotate_ccw_points(xs, ys, lab.width) for xs, ys in polys]
        w, h = lab.height, lab.width
    band = None
    if polys:
        xs, ys = max(polys, key=lambda p: max(p[1]) - min(p[1]))
        band = (min(xs), min(ys), max(xs), max(ys))
    p = plan_tile(image_id=lab.image_id, width=w, height=h, is_normal=True, band=band,
                  tile_w=tw, tile_h=th, stride_x=s["stride_x"], align=s["align"], seed=s["seed"])
    if not p.keep:
        return VtPlan(**base, keep=False, reason=p.reason, rot90_k=k)
    prov = REASON_TO_PROVENANCE[p.reason]
    # 돌린 좌표는 실수다 — plan_tile 을 지난 상자도 실수로 남는다. 정렬 뒤라 값은 정수여야 하고 정수로 확정한다
    if any(float(v) != int(v) for v in p.box):
        raise PixelFormError(f"타일 상자가 정수가 아니다: {lab.image_id} {p.box}")
    box = tuple(int(v) for v in p.box)
    if k == 1 and prov == PROV_CROP:
        raise PixelFormError(f"돌린 장은 원래부터 1280×720 일 수 없다: {lab.image_id}")
    return VtPlan(**base, keep=True, reason=p.reason, provenance=prov, rot90_k=k, box=box,
                  n_candidates=p.n_candidates)


def plan(lab: VtLabel, cfg: dict) -> VtPlan:
    return plan_normal(lab, cfg) if lab.is_normal else plan_defect(lab, cfg)


def band_inside(lab: VtLabel, p: VtPlan, cfg: dict) -> bool | None:
    """비드 ⊆ 타일(세로)인가 — 비드가 타일보다 낮을 때만 묻는다. 기록용(RT 와 같은 함수의 귀결을 잰다)."""
    if not lab.is_normal or not p.keep or p.provenance != PROV_TILE:
        return None
    polys = [(list(a.xs), list(a.ys)) for a in lab.anns if a.cls == cfg["labels"]["normal_class"] and a.xs]
    if p.rot90_k:
        polys = [rotate_ccw_points(xs, ys, lab.width) for xs, ys in polys]
    _, ys = max(polys, key=lambda q: max(q[1]) - min(q[1]))
    return p.box[1] <= min(ys) and max(ys) <= p.box[3]


# ------------------------------------------------------------------------------------------
# 다시 인코딩
# ------------------------------------------------------------------------------------------
def encode(raw: bytes, p: VtPlan, cfg: dict) -> tuple[bytes, dict]:
    """원천 바이트 하나를 계획대로 다시 인코딩한다. 반환값은 (JPEG 바이트, 기록)."""
    e = cfg["encode"]
    q = e["quality"]
    if q is None:
        raise PixelFormError("encode.quality 가 비어 있다 — R-9 측정 뒤 채운다")
    if int(q) < int(e["quality_floor_stop"]):
        raise PixelFormError(f"품질 {q} 가 멈춤 문턱 {e['quality_floor_stop']} 아래다(R-9)")
    if not e["strip_exif"]:
        raise PixelFormError("메타데이터를 남기는 재인코딩은 정의하지 않았다 — strip_exif 는 true 여야 한다")
    with Image.open(io.BytesIO(raw)) as im:
        if im.size != (p.src_w, p.src_h):
            raise PixelFormError(f"해독한 크기가 라벨의 선언과 다르다: {p.image_id} {im.size} ≠ {(p.src_w, p.src_h)}")
        orient = im.getexif().get(EXIF_ORIENTATION)
        if orient not in (None, 1):
            raise PixelFormError(f"EXIF 방향 태그가 있다: {p.image_id} {orient}")
        if im.mode != e["mode"]:
            raise PixelFormError(f"원천 모드가 {im.mode} 다 — 설정 {e['mode']}: {p.image_id}")
        im.load()
        work = im.transpose(Image.Transpose.ROTATE_90) if p.rot90_k == 1 else im
        # 패딩 금지를 인코딩 단계에서도 단언한다 — 프레임 밖 상자는 crop 이 검은색으로 채운다(06b I-2)
        tw, th = (int(v) for v in cfg["tile"]["tile_size"])
        x0, y0, x1, y1 = p.box
        if not (0 <= x0 < x1 <= work.size[0] and 0 <= y0 < y1 <= work.size[1]) or (x1 - x0, y1 - y0) != (tw, th):
            raise PixelFormError(f"상자가 (돌린) 프레임 밖이거나 타일 크기가 아니다: {p.image_id} {p.box} {work.size}")
        tile = work.crop(p.box)
        # 원천의 메타데이터(주석 COM · EXIF · ICC · dpi)를 싣지 않는다 — Pillow 는 `info` 의 주석을 다시 쓴다.
        # 결함 크롭과 정상 원판의 저장 소프트웨어가 다르면 그 흔적이 클래스를 가르는 비트가 된다(함정 11)
        tile.info = {}
        buf = io.BytesIO()
        tile.save(buf, format="JPEG", quality=int(q), progressive=bool(e["progressive"]),
                  optimize=bool(e["optimize"]))
    data = buf.getvalue()
    return data, {"image_id": p.image_id, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                  "w": p.box[2] - p.box[0], "h": p.box[3] - p.box[1]}
