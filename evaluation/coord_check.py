"""평가 쪽의 **독립 역변환** · 골든 픽스처 지문 · 통합형 좌표 설정 기록 — 07번 §14-2 7 · §31-6 · 진입점 미니스펙 3판 1-1-다 · 8-5.

쓰는 쪽의 좌표 코드(`vlm.coords`)를 **가져오지 않는다**(§13-2 나). 같은 규약을 여기서 따로 짠다 — 두 구현이 같은 답을 내는지가
문자 일치 카나리아와 에코 관문이 보는 것이다. 한쪽 코드를 빌려 쓰면 둘이 같은 실수를 같이 한다.

| 규약 | 모델 좌표 → 원본 픽셀 |
|---|---|
| `ABS_ORIG` | 그대로 |
| `NORM_1000` | x · 원본 폭 / 1000, y · 원본 높이 / 1000 |
| `ABS_RESIZED` | x · 원본 폭 / 입력 폭, y · 원본 높이 / 입력 높이(입력 크기는 곁 파일의 `model_input_wh`) |

**골든 픽스처 지문**(`coord_fixture_digest`)은 픽스처 파일의 사례마다 양자화된 모델 좌표를 이 역변환으로 풀어 **손계산 기대값**과 맞대고,
하나라도 허용치 밖이면 지문을 내지 않는다. 지문은 사례 id · 규약 · 결과(소수 여섯째 자리로 반올림)의 정규 JSON 의 sha256 이다 —
반올림은 부동소수 끝자리의 흔들림이 지문을 가르지 않게 하는 자리다(허용치 비교는 반올림 전에 한다).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

import yaml

from evaluation.prereg_unified import CoordCfgRecord

COORD_SPACES = ("ABS_ORIG", "NORM_1000", "ABS_RESIZED")
DIGEST_DECIMALS = 6


class FixtureMismatch(ValueError):
    """골든 픽스처의 손계산 기대값과 독립 역변환의 결과가 다르다. 어긋난 사례 id 를 든다."""

    def __init__(self, case_ids: list[str]):
        self.case_ids = list(case_ids)
        super().__init__("골든 픽스처와 독립 역변환이 다르다: " + ", ".join(self.case_ids))


def inverse_px(box: Sequence[float], coord_space: str, *, orig_wh: Sequence[int],
               model_wh: Sequence[int] | None = None) -> tuple[float, float, float, float]:
    """모델 좌표 `[x1, y1, x2, y2]` → 원본 픽셀. `ABS_RESIZED` 는 `model_wh`(입력 크기)가 있어야 한다."""
    if coord_space not in COORD_SPACES:
        raise ValueError(f"모르는 좌표 규약: {coord_space!r}")
    if len(box) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                                for v in box):
        raise ValueError("상자는 유한한 수 넷이다")
    w, h = orig_wh
    if coord_space == "ABS_ORIG":
        sx = sy = 1.0
    elif coord_space == "NORM_1000":
        sx, sy = w / 1000, h / 1000
    else:
        if model_wh is None:
            raise ValueError("ABS_RESIZED 의 역변환에는 입력 크기(model_input_wh)가 있어야 한다")
        sx, sy = w / model_wh[0], h / model_wh[1]
    x1, y1, x2, y2 = box
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def coord_fixture_digest(fixture_bytes: bytes) -> str:
    """골든 픽스처 파일의 바이트(읽기만) → 지문. 기대값과 하나라도 다르면 `FixtureMismatch`."""
    doc = json.loads(fixture_bytes.decode("utf-8"))
    tol = float(doc["tolerance"]["abs_tol_px"])
    rows: list[dict] = []
    bad: list[str] = []
    for case in doc["cases"]:
        space = case["cfg"]["coord_space"]
        geom = case["geom"]
        model_wh = (geom["resized_w"], geom["resized_h"]) if space == "ABS_RESIZED" else None
        got = inverse_px(case["expect"]["model_quantized"], space,
                         orig_wh=(geom["orig_w"], geom["orig_h"]), model_wh=model_wh)
        want = case["expect"]["back_px"]
        if any(abs(g - float(e)) > tol for g, e in zip(got, want, strict=True)):
            bad.append(case["id"])
        rows.append({"id": case["id"], "coord_space": space,
                     "back_px": [round(v, DIGEST_DECIMALS) for v in got]})
    if bad:
        raise FixtureMismatch(bad)
    body = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _lf_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def unified_coord_cfg_record(coord_cfg_hash: str, *, fixture_digest: str, registered_on: str,
                             coords_source: bytes, hash_record: bytes) -> CoordCfgRecord:
    """통합형의 좌표 설정 기록(§31-6) — 동결 export 가 없으므로 동결 값과 현재 값이 같다.

    `coord_cfg_hash` 는 등록하는 쪽이 통합형 좌표 설정으로 계산한 값이다(이 모듈은 쓰는 쪽 코드를 부르지 않는다).
    좌표 코드의 바이트(`coords_source`)가 이력 파일(`hash_record`, `HASH_RECORD.yaml`)의 마지막 값과 같을 때만 만든다 —
    이력이 낡은 채로 등록하면 "지금 바이트" 의 근거가 서지 않는다.
    """
    record = yaml.safe_load(hash_record.decode("utf-8"))
    entry = record["files"]["vlm/coords.py"]
    latest = entry.get("after_public_cleanup", entry)["sha256"]
    now = _lf_sha256(coords_source)
    if now != latest:
        raise ValueError("좌표 코드의 바이트가 HASH_RECORD.yaml 의 마지막 값과 다르다 — 이력을 먼저 갱신한다")
    return CoordCfgRecord(
        accepted=(coord_cfg_hash,), frozen_value=coord_cfg_hash, frozen_at=registered_on,
        current_value=coord_cfg_hash, change_scope="first_registration",
        equivalence_evidence=("동결 export 없음 — 첫 등록",
                              f"골든 픽스처 통과 · 지문 {fixture_digest}",
                              f"좌표 코드 바이트 {now} = HASH_RECORD.yaml 의 마지막 값"),
    )
