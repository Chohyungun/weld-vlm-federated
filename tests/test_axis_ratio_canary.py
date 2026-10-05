"""축 배율 카나리아 — 쌍 · 축별 중앙값 · 상태 넷 · 값을 싣지 않는다(07번 §12-3 의 ② · 리허설 명세 2판 §3 ② 나)."""

from __future__ import annotations

import pytest

from evaluation import frame_diag as F
from evaluation.axis_ratio_canary import REASON_NO_FIELDS, axis_ratio_canary


def ims(n: int, pred_of=lambda b: b, *, groups: int | None = None):
    out = []
    for i in range(n):
        g = (10.0 + i, 20.0 + 2 * i, 40.0 + i, 60.0 + 2 * i)
        out.append(F.FrameImage(f"i{i}", f"g{i % (groups or n)}", 0, 0, (pred_of(g),), (g,)))
    return out


def resize_y(b):
    k = 704 / 720
    return (b[0], b[1] * k, b[2], b[3] * k)


def test_원본_프레임은_통과하고_수만_싣는다() -> None:
    got = axis_ratio_canary(ims(30), threshold=0.01, min_pairs=10)
    assert got == {"status": F.PASS, "reason": "", "n_defect_images": 30, "n_pairs": 30, "n_pair_groups": 30,
                   "n_pairs_unusable": 0}
    assert all(isinstance(v, (str, int)) and not isinstance(v, float) for v in got.values()), "비의 값은 싣지 않는다"


def test_리사이즈_프레임은_y_축_불통과() -> None:
    got = axis_ratio_canary(ims(30, resize_y), threshold=0.01, min_pairs=10)
    assert (got["status"], got["reason"]) == (F.FAIL, "y")


def test_쌍이_하한_미만이면_표본_부족() -> None:
    got = axis_ratio_canary(ims(9), threshold=0.01, min_pairs=10)
    assert (got["status"], got["reason"], got["n_pairs"]) == (F.INDETERMINATE, F.REASON_SAMPLE, 9)


@pytest.mark.parametrize(("threshold", "min_pairs"), [(None, 10), (0.01, None), (None, None)])
def test_등록_칸이_비면_판정_불가지만_수는_센다(threshold, min_pairs) -> None:
    got = axis_ratio_canary(ims(30), threshold=threshold, min_pairs=min_pairs)
    assert (got["status"], got["reason"], got["n_pairs"]) == (F.INDETERMINATE, REASON_NO_FIELDS, 30)


def test_정답_중심이_0_인_쌍은_빼고_센다() -> None:
    zero = F.FrameImage("z", "gz", 0, 0, ((-5.0, 0.0, 5.0, 10.0),), ((-5.0, 0.0, 5.0, 10.0),))
    got = axis_ratio_canary(ims(10) + [zero], threshold=0.01, min_pairs=10)
    assert (got["n_pairs"], got["n_pairs_unusable"], got["status"]) == (10, 1, F.PASS)


def test_예측이_없거나_정답이_없는_장은_쌍이_아니다() -> None:
    images = ims(5) + [F.FrameImage("np", "g", 0, 0, (), ((1.0, 1.0, 5.0, 5.0),)),
                       F.FrameImage("ng", "g", 0, 0, ((1.0, 1.0, 5.0, 5.0),), ())]
    got = axis_ratio_canary(images, threshold=0.01, min_pairs=1)
    assert (got["n_defect_images"], got["n_pairs"]) == (6, 5)


def test_상태_어휘는_공개_상수다() -> None:
    """쓰는 쪽 판정기가 문자열 대신 가져다 쓰는 자리 — 프레임 진단 검사기의 값과 같다."""
    from evaluation import axis_ratio_canary as A
    assert A.STATUSES == (F.PASS, F.FAIL, F.INDETERMINATE) == (A.PASS, A.FAIL, A.INDETERMINATE)
    assert axis_ratio_canary(ims(30), threshold=0.01, min_pairs=10)["status"] in A.STATUSES
