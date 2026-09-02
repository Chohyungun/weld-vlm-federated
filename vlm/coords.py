"""좌표 규약 단일 모듈 — 지정 함정 구간 #4.

이 모듈이 이 저장소에서 bbox 좌표계를 변환하는 **유일한 곳**이다.
트랙 C가 소유하고, 트랙 D는 import만 하며 재구현하지 않는다
(게이트 웨이브 #5 §5-1: C = 규약 소유자, D = 검증 소유자).

## 의존성 제로 리프 모듈

표준 라이브러리만 쓴다. torch·transformers·ultralytics·numpy 어느 것도 import하지
않는다. C(학습·추론)와 D(채점)의 패키지 구성이 달라도 같은 코드가 돌아야 하기 때문이다.
이 파일에 서드파티 import를 추가하면 계약 위반이다 — tests/test_coords.py가 이를 검사한다.

같은 이유로 `transformers` 버전은 이 모듈이 조회하지 않고 호출자가 문자열로 주입한다.
리사이즈 치수(`ImageGeom.resized_w/h`)도 이 모듈이 계산하지 않고 인자로만 받는다.
`smart_resize`를 여기서 재구현하거나 import하면, 프로세서가 실제로 쓴 값과 어긋나도
아무도 모르는 상태가 된다 — 그것이 선행 연구에서 IoU가 0.938에서 0.055로 붕괴한 경로다.

## 단일 정본은 원본 픽셀이다

저장·전송되는 모든 좌표는 원본 이미지 픽셀이다. 모델 좌표는 어떤 파일에도 저장하지
않는다. 정변환은 학습 타깃을 만들 때, 역변환은 추론 직후에만 일어난다.

## 좌표 보정을 하지 않는다

이 모듈은 좌표를 판정할 뿐 고치지 않는다. `validate_box_px`는 사유 문자열을 돌려줄 뿐
입력을 바꾸지 않고, 유일한 예외인 `snap_to_bounds`도 호출자가 명시적으로 부를 때만
동작한다. 채점기가 좌표를 고쳐 주면 그것은 답을 고쳐 주는 것이다.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from typing import Sequence

__all__ = [
    "COORD_SPACES",
    "CoordCfg",
    "ImageGeom",
    "to_model",
    "to_px",
    "quantize",
    "is_degenerate",
    "roundtrip_error_px",
    "roundtrip_budget_px",
    "validate_box_px",
    "snap_to_bounds",
    "coords_source_sha256",
    "coord_cfg_hash",
    "CoordError",
]

#: 지원하는 좌표 규약. 계약 #4 `coord_space` enum과 같은 어휘를 쓴다
#: (게이트 #6 결정 F로 `convention` 표기는 폐기).
#:
#: - ``ABS_ORIG``    : 원본 이미지 절대 픽셀. 변환이 항등이다. **우리 기본값이다**
#:                     (총괄 판정 1, 2026-09-02 — 카나리아-1 실측 6/6). 통합형 타깃과
#:                     분리형 검출 출력이 같은 규약이 되므로 다섯 칸 대조가 한 공간에서 선다.
#:                     Ultralytics ``Results.boxes.xyxy`` 처럼 프레임워크가 이미 원본으로
#:                     복원해 주는 경로가 여기 해당한다(D의 Q20 실측 봉인 대상).
#: - ``NORM_1000``   : 원본 기준 0~1000 정규화. Qwen2-VL 계열 규약. **파일럿까지의 기본값**
#:                     이었고 지금은 쓰지 않는다 — 실측 네이티브 규약과 어긋나 라벨 자체에
#:                     계통 오차를 심었다(왕복 IoU median 0.98119, IoU<0.95 가 379건).
#: - ``ABS_RESIZED`` : 리사이즈된 입력 이미지의 절대 픽셀. Qwen2.5-VL 계열 규약.
#:                     배율이 이미지마다, 그리고 x축과 y축이 서로 다르다.
COORD_SPACES = ("ABS_ORIG", "NORM_1000", "ABS_RESIZED")


class CoordError(ValueError):
    """좌표 규약 위반. 조용히 넘어가면 안 되는 것만 이 예외를 쓴다."""


@dataclass(frozen=True)
class CoordCfg:
    """좌표 규약 선언. 코드가 아니라 configs 데이터다.

    Qwen3.5의 네이티브 좌표 규약이 공개 자료에 없으므로, 규약을 코드에 박지 않고
    이 설정값 하나로 흡수했다. 카나리아-1(파인튜닝 전 제로샷 grounding 판별)이 실측으로
    규약을 판별하면 ``coord_space`` 값만 바꾸며 모듈·채점기 코드는 불변이라고 적어 두었고,
    **2026-09-02 에 실제로 그렇게 됐다** — 실측이 ABS_ORIG 를 가리켰고 바뀐 것은
    ``vlm/pilot_vlm.COORD_CFG`` 한 줄과 프롬프트의 좌표 문장 하나뿐이다. 설계가 값을 했다.

    Attributes:
        coord_space: ``COORD_SPACES`` 중 하나.
        min_pixels: 프로세서 ``min_pixels``. 좌표 공간을 결정하므로 해시 입력에 들어간다.
        max_pixels: 프로세서 ``max_pixels``. GPU 프로파일에서 override 금지(Q15, 5칸 공통 고정).
        factor: 리사이즈 격자 단위(patch_size × merge_size). 기록용이며 이 모듈은 쓰지 않는다.
        norm_denominator: ``NORM_1000``의 분모. 규약상 1000 고정.
        transformers_version: 호출자가 주입한다. 이 모듈은 transformers를 import하지 않는다.
    """

    coord_space: str
    min_pixels: int | None = None
    max_pixels: int | None = None
    factor: int | None = None
    norm_denominator: int = 1000
    transformers_version: str = ""

    def __post_init__(self) -> None:
        if self.coord_space not in COORD_SPACES:
            raise CoordError(
                f"알 수 없는 coord_space: {self.coord_space!r}. "
                f"허용값은 {COORD_SPACES}. 기본값 추측 없이 즉시 실패한다."
            )
        if self.norm_denominator <= 0:
            raise CoordError(f"norm_denominator는 양수여야 한다: {self.norm_denominator}")

    def canonical_json(self) -> str:
        """해시 입력용 결정론적 직렬화. 키 정렬 고정."""
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ImageGeom:
    """한 이미지의 기하 정보.

    ``resized_w/h``는 **프로세서가 실제로 쓴 값**이어야 한다. 이 모듈이 재계산하지 않고
    호출자가 실호출 산출물(``image_grid_thw`` × patch_size 복원값)에서 얻어 넣는다.
    ``ABS_RESIZED``에서 이 값이 없으면 변환하지 않고 실패한다 — 추정하면 조용히 어긋난다.
    """

    orig_w: int
    orig_h: int
    resized_w: int | None = None
    resized_h: int | None = None

    def __post_init__(self) -> None:
        if self.orig_w <= 0 or self.orig_h <= 0:
            raise CoordError(f"원본 크기가 양수가 아니다: {self.orig_w}x{self.orig_h}")
        for name in ("resized_w", "resized_h"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise CoordError(f"{name}가 양수가 아니다: {v}")

    def require_resized(self) -> tuple[int, int]:
        if self.resized_w is None or self.resized_h is None:
            raise CoordError(
                "ABS_RESIZED 변환에는 프로세서 실측 리사이즈 치수가 필요하다. "
                "이 모듈은 smart_resize를 재계산하지 않는다 — 실호출 산출물을 넣어라."
            )
        return self.resized_w, self.resized_h


def _as_box(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """길이 4의 유한한 수치 시퀀스인지 확인하고 float 튜플로 만든다."""
    if bbox is None:
        raise CoordError("bbox가 None이다.")
    values = list(bbox)
    if len(values) != 4:
        raise CoordError(f"bbox 원소가 4개가 아니다: {len(values)}개")
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise CoordError(f"bbox 원소가 수치가 아니다: {v!r}")
        f = float(v)
        if not math.isfinite(f):
            raise CoordError(f"bbox 원소가 유한하지 않다: {v!r}")
        out.append(f)
    return out[0], out[1], out[2], out[3]


def to_model(
    bbox_px: Sequence[float], geom: ImageGeom, cfg: CoordCfg
) -> tuple[float, float, float, float]:
    """정변환: 원본 픽셀 → 모델 네이티브 좌표. 학습 타깃 생성이 호출한다.

    결과는 float다. 타깃 문자열에 넣을 정수화는 호출자가 `quantize`로 따로 한다 —
    정수화 지점을 하나로 묶어 두어야 반올림이 두 번 일어나지 않는다.
    """
    x1, y1, x2, y2 = _as_box(bbox_px)
    space = cfg.coord_space

    if space == "ABS_ORIG":
        return x1, y1, x2, y2

    if space == "NORM_1000":
        d = float(cfg.norm_denominator)
        sx = d / float(geom.orig_w)
        sy = d / float(geom.orig_h)
        return x1 * sx, y1 * sy, x2 * sx, y2 * sy

    # ABS_RESIZED — 축별 배율. 단일 scale로 묶으면 비정방 이미지에서 어긋난다.
    rw, rh = geom.require_resized()
    sx = float(rw) / float(geom.orig_w)
    sy = float(rh) / float(geom.orig_h)
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def to_px(
    bbox_model: Sequence[float], geom: ImageGeom, cfg: CoordCfg
) -> tuple[float, float, float, float]:
    """역변환: 모델 네이티브 좌표 → 원본 픽셀. 추론 직후 C가 호출한다.

    `to_model`과 짝을 이루는 같은 클래스의 함수로 두는 이유는, 순변환과 역변환이 서로
    다른 파일에서 각자 진화하는 것이 좌표계 붕괴의 구조이기 때문이다.
    """
    x1, y1, x2, y2 = _as_box(bbox_model)
    space = cfg.coord_space

    if space == "ABS_ORIG":
        return x1, y1, x2, y2

    if space == "NORM_1000":
        d = float(cfg.norm_denominator)
        sx = float(geom.orig_w) / d
        sy = float(geom.orig_h) / d
        return x1 * sx, y1 * sy, x2 * sx, y2 * sy

    rw, rh = geom.require_resized()
    sx = float(geom.orig_w) / float(rw)
    sy = float(geom.orig_h) / float(rh)
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def quantize(bbox: Sequence[float]) -> tuple[int, int, int, int]:
    """정수화 단일 지점. ``floor(v + 0.5)``.

    파이썬 내장 `round`를 쓰지 않는 이유는 그것이 banker's rounding(짝수로 반올림)이라
    0.5가 값에 따라 위아래로 갈리기 때문이다. 채점 재현성이 서브픽셀 정밀도보다 우선이므로
    플랫폼·값 무관하게 결정적인 규칙을 쓴다. 음수 좌표에서도 규칙은 동일하다.
    """
    x1, y1, x2, y2 = _as_box(bbox)
    return (
        int(math.floor(x1 + 0.5)),
        int(math.floor(y1 + 0.5)),
        int(math.floor(x2 + 0.5)),
        int(math.floor(y2 + 0.5)),
    )


def is_degenerate(bbox: Sequence[float]) -> bool:
    """면적이 0 이하인가(x1 >= x2 또는 y1 >= y2).

    정수화 뒤 이것이 참이면 박스가 뭉개진 것이다. **넓혀서 살리지 않는다** — 학습 타깃
    생성 경로에서는 폐기하고 건수를 보고하며(초과 샘플 폐기 원칙과 같은 결), 예측 경로에서는
    D가 `bbox_invalid` 오답으로 계상한다.

    ``NORM_1000``에서는 원본 폭이 분모(1000)의 2배를 넘으면 1px 박스가 구조적으로 뭉개질
    수 있다. 이것은 규약의 성질이지 구현 버그가 아니며, 골든 픽스처가 이를 고정한다.
    """
    x1, y1, x2, y2 = _as_box(bbox)
    return x1 >= x2 or y1 >= y2


def roundtrip_error_px(
    bbox_px: Sequence[float], geom: ImageGeom, cfg: CoordCfg, *, quantize_model: bool = True
) -> float:
    """왕복 오차(픽셀). 원본 → 모델 → (정수화) → 원본의 좌표별 최대 절대 오차.

    `quantize_model=True`가 실제 운용 경로다. 모델은 정수 좌표를 출력하므로 양자화를
    포함한 오차가 우리가 실제로 겪는 오차다.
    """
    box = _as_box(bbox_px)
    model = to_model(box, geom, cfg)
    if quantize_model:
        model = quantize(model)
    back = to_px(model, geom, cfg)
    return max(abs(a - b) for a, b in zip(box, back))


def roundtrip_budget_px(geom: ImageGeom, cfg: CoordCfg) -> float:
    """이 이미지·규약에서 왕복 오차가 넘지 말아야 할 상한(원본 픽셀).

    모델 좌표는 정수라, 양자화 1단위가 원본 픽셀로 환산되는 크기의 절반이 오차 상한이다.
    이 값이 규약마다 다르다는 점이 중요하다.

    - ``ABS_ORIG``    : 모델 공간이 곧 원본이므로 0.5px.
    - ``NORM_1000``   : 모델 공간이 분모(1000)이므로 ``0.5 × 원본축 / 1000``.
                        원본 한 변이 2000px를 넘으면 0.5px를 초과한다.
    - ``ABS_RESIZED`` : 모델 공간이 리사이즈 치수이므로 ``0.5 × 원본축 / 리사이즈축``.
                        ``max_pixels`` 캡으로 크게 축소되는 큰 이미지에서 이 값이 픽셀 단위로
                        커진다. 규약 선택이 좌표 정밀도에 직접 영향을 준다는 뜻이며,
                        ``ABS_RESIZED``를 채택한다면 이 상한을 한계 절에 기록해야 한다.

    예산은 규약의 성질이므로 여기서 단 한 번 정의하고, 테스트·픽스처 스크립트·트랙 D가
    모두 이 함수를 쓴다. 각자 상수를 적어 두면 그 순간 기준이 갈라진다.
    """
    if cfg.coord_space == "ABS_ORIG":
        return 0.5
    if cfg.coord_space == "NORM_1000":
        mx = my = float(cfg.norm_denominator)
    else:
        rw, rh = geom.require_resized()
        mx, my = float(rw), float(rh)
    return max(0.5, 0.5 * float(geom.orig_w) / mx, 0.5 * float(geom.orig_h) / my)


def validate_box_px(
    bbox: Sequence[float], geom: ImageGeom, *, tol: float = 0.0
) -> str | None:
    """원본 픽셀 박스를 판정한다. **입력을 고치지 않는다.**

    Returns:
        문제가 없으면 ``None``, 있으면 사유 문자열. 사유 어휘는 계약 #4의
        ``parse_error`` 열거값과 맞춘다(호출자가 그대로 기록할 수 있게).

    Args:
        tol: 경계 이탈 허용치(픽셀). 기본 0이면 이미지 밖으로 나간 좌표를 전부 보고한다.
             부분 이탈을 오답으로 만들지 여부는 채점 정책이므로 호출자가 정한다.
    """
    try:
        x1, y1, x2, y2 = _as_box(bbox)
    except CoordError as exc:
        return f"bbox_invalid: {exc}"

    if x1 >= x2 or y1 >= y2:
        return "bbox_invalid: 좌표 순서가 뒤바뀌었거나 면적이 0이다 (x1<x2, y1<y2 위반)"

    w, h = float(geom.orig_w), float(geom.orig_h)
    if x2 <= -tol or y2 <= -tol or x1 >= w + tol or y1 >= h + tol:
        return "bbox_invalid: 이미지와 교집합이 없다"

    if x1 < -tol or y1 < -tol or x2 > w + tol or y2 > h + tol:
        return "out_of_bounds: 이미지 경계를 벗어난 좌표가 있다"

    return None


def snap_to_bounds(
    bbox: Sequence[float], geom: ImageGeom, *, tol: float = 1.0
) -> tuple[tuple[float, float, float, float], bool]:
    """경계 이탈이 ``tol`` 이내일 때만 이미지 경계로 당긴다.

    이 모듈에서 좌표를 바꾸는 **유일한 함수**이며, 호출자가 명시적으로 부를 때만 동작한다.
    허용하는 근거는 이것이 모델의 오차가 아니라 우리 변환의 수치 오차이기 때문이다.
    ``tol``을 넘는 이탈은 손대지 않는다 — 클램프는 IoU를 올리는 방향으로만 작동해서
    답을 고쳐 주는 셈이 된다.

    Returns:
        (스냅된 박스, 스냅이 실제로 일어났는가). 두 번째 값을 카운터로 집계하라.
        발생률이 높으면 수치 오차가 아니라 좌표계 오류의 신호다.
        스냅 뒤 퇴화했는지는 호출자가 `is_degenerate`로 다시 확인해야 한다.
    """
    x1, y1, x2, y2 = _as_box(bbox)
    w, h = float(geom.orig_w), float(geom.orig_h)
    snapped = False

    def pull(v: float, lo: float, hi: float) -> float:
        nonlocal snapped
        if lo - tol <= v < lo:
            snapped = True
            return lo
        if hi < v <= hi + tol:
            snapped = True
            return hi
        return v

    return (pull(x1, 0.0, w), pull(y1, 0.0, h), pull(x2, 0.0, w), pull(y2, 0.0, h)), snapped


def coords_source_sha256() -> str:
    """이 파일 자신의 소스 sha256.

    `coord_cfg_hash`의 입력에 들어간다. 설정이 같아도 구현이 바뀌면 해시가 달라져야
    골든 픽스처 단계까지 가기 전에 잡힌다(게이트 #7 결정 H, 계약 #4 §3-2).
    """
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def coord_cfg_hash(cfg: CoordCfg) -> str:
    """계약 #4가 정한 입력 셋으로 해시를 만든다.

    ``sha256(CoordCfg 직렬화 + coords.py 소스 sha256 + transformers 버전)``

    무엇의 해시인지 계약에 못 박혀 있어야 C와 D가 같은 값을 계산한다. 5칸의 값이
    갈리면 채점기가 실패하고, 카나리아-1 재실행으로 규약 동일을 확인한 뒤에만
    새 해시로 재봉인한다.
    """
    payload = "\n".join(
        (cfg.canonical_json(), coords_source_sha256(), cfg.transformers_version)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
