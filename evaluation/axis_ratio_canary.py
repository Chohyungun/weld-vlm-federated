"""축 배율 카나리아 — 모델 묶음의 생성 좌표가 정답과 같은 축 배율에 있는가. 07번 §12-3 의 ② · 리허설 명세 2판 §3 ② 나.

| 칸 | 정의 |
|---|---|
| 쌍 | 정답 박스가 있는 장 안에서 예측과 정답이 **서로의 최대 IoU 상대**(IoU > 0, 동률이면 쌍이 아니다) — 프레임 진단 검사기의 `mutual_pairs` 그대로. 클래스는 보지 않는다 |
| 축 배율 `m` | 쌍마다 중심 좌표 비(예측 ÷ 정답)의 축별 중앙값. 정답 중심이 0 인 축이 있는 쌍은 비를 낼 수 없어 빼고 수를 센다 |
| 상태 | 등록 칸(`canary.axis_ratio_threshold` · `canary.axis_ratio_min_pairs`)이 비면 **판정 불가 — 등록 칸 없음**. 쌍이 하한 미만이면 **판정 불가 — 표본 부족**. 두 축 `\\|m − 1\\| ≤ 문턱` 이면 **통과**, 아니면 **불통과**(어긋난 축) |

**내는 것은 상태 · 사유 · 표본 수뿐이다** — 비의 값은 싣지 않는다(리허설 산출물에 지표 값을 싣지 않는다 · 진입점 미니스펙 3판 4-2).
쓰는 쪽 판정기는 이 상태로 판정하지 않고 "호출되었는가" 만 본다(판정 불가도 호출이다).

본채점(엄격)의 카나리아는 진단 규칙 파일의 문턱 · 표준오차 게이트를 쓴다(07번 §31-1 의 6) — 엄격 채점 단계와 함께다.
이 함수는 등록 칸만 쓰는 **관측**이다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from evaluation import frame_diag as F

REASON_NO_FIELDS = "등록 칸 없음"
PASS, FAIL, INDETERMINATE = F.PASS, F.FAIL, F.INDETERMINATE
STATUSES: tuple[str, ...] = (PASS, FAIL, INDETERMINATE)
"""상태의 세 어휘 — 쓰는 쪽 판정기가 문자열 대신 이 상수를 가져다 쓴다(프레임 진단 검사기의 것과 같은 값)."""


def axis_ratio_canary(images: Sequence[F.FrameImage], *, threshold: float | None, min_pairs: int | None) -> dict:
    """묶음 하나의 축 배율 카나리아 — `{status, reason, n_defect_images, n_pairs, n_pair_groups, n_pairs_unusable}`."""
    defect = [im for im in images if im.gold]
    ratios, groups, unusable = [], set(), 0
    for im in defect:
        if not im.pred:
            continue
        pcs, gcs = F._centers(im.pred), F._centers(im.gold)
        for i, j in F.mutual_pairs(im.pred, im.gold):
            if np.any(gcs[j] == 0):
                unusable += 1
                continue
            ratios.append(pcs[i] / gcs[j])
            groups.add(im.group_id)
    counts = {"n_defect_images": len(defect), "n_pairs": len(ratios), "n_pair_groups": len(groups),
              "n_pairs_unusable": unusable}
    if threshold is None or min_pairs is None:
        return {"status": F.INDETERMINATE, "reason": REASON_NO_FIELDS, **counts}
    if len(ratios) < min_pairs:
        return {"status": F.INDETERMINATE, "reason": F.REASON_SAMPLE, **counts}
    m = np.median(np.asarray(ratios, dtype=float), axis=0)
    bad = [F.AXES[a] for a in (0, 1) if not abs(float(m[a]) - 1.0) <= threshold]
    return {"status": F.FAIL if bad else F.PASS, "reason": "·".join(bad), **counts}
