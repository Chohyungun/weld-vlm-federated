"""`map_50` 두 번째 구현 시험 — 53번 §5 시험 1·2·3, 57번 00:15 판정 1, I 모드 입력 규약.

1. **손계산 픽스처** — 정의표(53번 §3 + A 58번 보강·#18)의 갈림길마다 값이 달라지게 만든 작은 입력. 기대값은 정의를 손으로
   따라가 적고, 같은 입력을 pycocotools 에도 넣어 비트로 맞댄다.
2. **합성 무작위 대조** — 동점·절단·경계 IoU·IoU 동률·정답 없는 카테고리·빈 이미지·목록 밖 클래스가 섞인 입력 여러 벌에서
   `map_50` 과 정밀도 격자가 pycocotools 와 **비트로** 같다. pycocotools 는 이 파일 안에서만 기준으로 쓴다.
3. **import 경계** — 모듈이 pycocotools·기존 채점 경로를 import 하지 않고, 이 모듈을 import 하는 파일은 시험과 검산 실행기뿐이다.
4. **I 모드 입력** — 파일 형식 문서의 규약대로 읽고, 어긋나면 멈춘다. S·I 두 입구가 같은 데이터에서 같은 비트를 낸다.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluation.metrics import map_independent as mi
from evaluation.metrics.map_independent import (
    CONVENTIONS,
    EPS,
    MAX_DETS,
    REC_THRS,
    evaluate_map50,
    map50_independent,
    map50_shared,
    read_gold_independent,
    read_pred_independent,
)

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "evaluation" / "metrics" / "map_independent.py"
CLASSES = ("100", "2011", "401", "301")


# --------------------------------------------------------------------------------------
# pycocotools 기준 — 이 시험 파일 안에서만 쓴다
# --------------------------------------------------------------------------------------

def coco_reference(gold, pred, classes):
    """pycocotools 2.0.11 로 같은 입력을 채점한다. id 는 전부 1 부터(#18). 반환 `(AP50, (101, K) 정밀도)`."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ids = sorted(gold)
    img_id = {i: n + 1 for n, i in enumerate(ids)}
    cat_id = {c: n + 1 for n, c in enumerate(classes)}
    anns = []
    for i in ids:
        for code, (x, y, w, h) in gold[i]:
            if code in cat_id:
                anns.append({"id": len(anns) + 1, "image_id": img_id[i], "category_id": cat_id[code],
                             "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    dets = []
    for i in ids:
        for code, (x, y, w, h), s in pred.get(i, ()):
            if code in cat_id:
                dets.append({"image_id": img_id[i], "category_id": cat_id[code], "bbox": [x, y, w, h], "score": s})
    assert dets, "pycocotools loadRes 는 빈 검출 목록을 받지 못한다 — 픽스처에 검출을 넣어라"
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": [{"id": img_id[i]} for i in ids], "annotations": anns,
                      "categories": [{"id": v, "name": k} for k, v in cat_id.items()]}
        gt.createIndex()
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[1]), np.array(ev.eval["precision"][0, :, :, 0, 2])


def assert_same_as_coco(gold, pred, classes=CLASSES):
    res = evaluate_map50(gold, pred, classes)
    ref_ap, ref_prec = coco_reference(gold, pred, classes)
    assert res.map_50 == ref_ap, (res.map_50, ref_ap)
    assert np.array_equal(res.precision, ref_prec)
    valid = [k for k in range(len(classes)) if ref_prec[0, k] > -1]
    if valid:
        ref_class_mean = float(np.mean([np.mean(ref_prec[:, k]) for k in valid]))
        assert abs(res.map_50_class_mean - ref_class_mean) <= 1e-12
    return res


def _grid_mean(values_by_point):
    return float(np.mean(np.array(values_by_point, dtype=np.float64)))


# ======================================================================================
# 1. 손계산 픽스처
# ======================================================================================

def test_완벽_매칭도_AP_가_1_이_아니다_ε():
    """#4 — 정밀도 분모의 ε 때문에 101점이 모두 1/(1+ε) 다(A 58번 §3: 0.9999999999999999)."""
    gold = {"a": [("100", (0.0, 0.0, 10.0, 10.0))]}
    pred = {"a": [("100", (0.0, 0.0, 10.0, 10.0), 0.9)]}
    res = assert_same_as_coco(gold, pred)
    expect = _grid_mean([1.0 / (0.0 + 1.0 + EPS)] * len(REC_THRS))
    assert res.map_50 == expect
    assert res.map_50 != 1.0
    assert res.ap_by_class == {"100": expect, "2011": None, "401": None, "301": None}


def test_IoU_가_정확히_0_5_면_매칭된다():
    """#9·#11 — 10×10 정답과 10×20 검출: 교집합 100, 합집합 100 + 200 − 100 = 200 → 0.5. 같음은 매칭이다."""
    gold = {"a": [("100", (0.0, 0.0, 10.0, 10.0))]}
    pred = {"a": [("100", (0.0, 0.0, 10.0, 20.0), 0.9)]}
    res = assert_same_as_coco(gold, pred)
    assert res.counts["boundary_pairs"] == 1
    assert res.map_50 == _grid_mean([1.0 / (1.0 + EPS)] * len(REC_THRS))


def test_IoU_식에_플러스_1_이_없다():
    """#11 — [0,0,10,10] 과 [5,5,10,10]: 25/175. `+1` 규약이면 36/206 이다. 여기서는 문턱을 못 넘는다."""
    d = np.array([[5.0, 5.0, 10.0, 10.0]])
    g = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert mi._iou_matrix(d, g)[0, 0] == 25.0 / 175.0
    assert mi._iou_matrix(np.array([[10.0, 0.0, 5.0, 5.0]]), g)[0, 0] == 0.0     # 맞닿기만 하면 0


def test_IoU_행렬이_maskUtils_iou_와_비트가_같다():
    """#11 — 연산 순서까지 같은지 본다(합집합 `(da + ga) − i`). IoU 끝자리 차이는 매칭 결정이 바뀔 때만 AP 에 드러나므로
    AP 대조만으로는 못 잡는다(60번 변이 T18). 소수 좌표 박스 여러 벌에서 pycocotools `maskUtils.iou` 와 행렬 전체를 맞댄다."""
    from pycocotools import mask as mask_utils

    rng = random.Random(3)
    for trial in range(30):
        def box():
            if rng.random() < 0.5:
                x1, y1 = _r2(rng, 0, 300), _r2(rng, 0, 300)
                x2, y2 = x1 + _r2(rng, 0.5, 90), y1 + _r2(rng, 0.5, 90)
                return [x1, y1, x2 - x1, y2 - y1]                               # xyxy 소수 → xywh(실물 형태)
            return [rng.uniform(0, 300), rng.uniform(0, 300), rng.uniform(0.1, 90), rng.uniform(0.1, 90)]

        d = [box() for _ in range(rng.randint(1, 40))]
        g = [box() for _ in range(rng.randint(1, 20))]
        ref = np.asarray(mask_utils.iou(d, g, [0] * len(g)))
        got = mi._iou_matrix(np.array(d, dtype=np.float64), np.array(g, dtype=np.float64))
        assert got.shape == ref.shape
        assert np.array_equal(got.view(np.uint64), ref.view(np.uint64)), trial


def test_IoU_동률이면_뒤쪽_정답에_매칭한다():
    """#10 — 검출 1(점수 0.9)은 정답 1·2 와 IoU 가 둘 다 0.5 → 뒤쪽 정답 2. 그러면 검출 2(0.8)가 정답 1 과 IoU 1 로 매칭된다.

    뒤쪽: TP, TP → rc [0.5, 1.0], 포락선 뒤 전 격자 2/(2+ε).
    앞쪽(틀린 구현): 검출 1 이 정답 1 을 가져가 검출 2 는 FP → rc [0.5, 0.5] → r > 0.5 인 격자점이 0.
    """
    gold = {"a": [("100", (0.0, 0.0, 10.0, 10.0)), ("100", (10.0, 0.0, 10.0, 10.0))]}
    pred = {"a": [("100", (0.0, 0.0, 20.0, 10.0), 0.9), ("100", (0.0, 0.0, 10.0, 10.0), 0.8)]}
    res = assert_same_as_coco(gold, pred)
    assert res.counts["iou_ties"] == 1
    p2 = 2.0 / (0.0 + 2.0 + EPS)
    assert res.map_50 == _grid_mean([p2] * len(REC_THRS))
    front = [1.0 / (1.0 + EPS) if r <= 0.5 else 0.0 for r in REC_THRS]
    assert abs(res.map_50 - _grid_mean(front)) > 0.4


def test_문턱_이상_중_최대_IoU_를_고른다():
    """A 58번 §5-a — 첫 통과 정답에서 멈추지 않는다.

    검출 1(0.9)은 정답 1 과 IoU 0.5, 정답 2 와 0.8 → 정답 2. 검출 2(0.8)는 정답 1 과 IoU 1 → TP. 둘 다 TP.
    첫 통과에서 멈추는 구현이면 검출 1 이 정답 1 을 가져가고, 검출 2 는 정답 2 와 IoU 0.3 이라 FP 가 된다.
    """
    g1, g2 = (0.0, 0.0, 10.0, 10.0), (4.0, 0.0, 16.0, 10.0)
    d1, d2 = (0.0, 0.0, 20.0, 10.0), (0.0, 0.0, 10.0, 10.0)
    assert mi._iou_matrix(np.array([d1, d2]), np.array([g1, g2])).tolist() == [[0.5, 0.8], [1.0, 0.3]]
    res = assert_same_as_coco({"a": [("100", g1), ("100", g2)]}, {"a": [("100", d1, 0.9), ("100", d2, 0.8)]})
    assert res.map_50 == _grid_mean([2.0 / (0.0 + 2.0 + EPS)] * len(REC_THRS))  # 둘 다 TP
    first_pass = [1.0 / (0.0 + 1.0 + EPS) if r <= 0.5 else 0.0 for r in REC_THRS]
    assert abs(res.map_50 - _grid_mean(first_pass)) > 0.4


def test_이미지_경계를_넘는_동점은_이미지_문자열_순서를_따른다():
    """#7·#8 — 점수 0.9 동점: `img10` 의 FP 와 `img9` 의 TP. 문자열 순서로 img10 이 먼저라 FP 가 앞선다.

    FP 먼저: tp [0,1], fp [1,1] → pr [0, 1/(2+ε)] → 포락선 [1/(2+ε), 1/(2+ε)] → 전 격자 1/(2+ε).
    숫자 순서(img9 먼저)였다면 TP 먼저라 전 격자 1/(1+ε).
    """
    gold = {"img9": [("100", (0.0, 0.0, 10.0, 10.0))], "img10": []}
    pred = {"img10": [("100", (100.0, 100.0, 10.0, 10.0), 0.9)], "img9": [("100", (0.0, 0.0, 10.0, 10.0), 0.9)]}
    res = assert_same_as_coco(gold, pred)
    assert res.map_50 == _grid_mean([1.0 / (1.0 + 1.0 + EPS)] * len(REC_THRS))


def test_이미지_안_동점은_레코드_순서를_따른다():
    """#6 — 한 이미지 안 점수 0.5 동점 FP·TP. 레코드 순서 FP 가 먼저면 AP 가 절반쯤이다."""
    gold = {"a": [("100", (0.0, 0.0, 10.0, 10.0))]}
    fp_first = {"a": [("100", (50.0, 50.0, 10.0, 10.0), 0.5), ("100", (0.0, 0.0, 10.0, 10.0), 0.5)]}
    tp_first = {"a": [fp_first["a"][1], fp_first["a"][0]]}
    a = assert_same_as_coco(gold, fp_first)
    b = assert_same_as_coco(gold, tp_first)
    assert a.map_50 == _grid_mean([1.0 / (2.0 + EPS)] * len(REC_THRS))
    assert b.map_50 == _grid_mean([1.0 / (1.0 + EPS)] * len(REC_THRS))


def _truncation_case():
    """#5 — 한 이미지에 카테고리 `100` 검출 111건(12위·111위가 참), 카테고리 `2011` 검출 150건(1위가 참, 점수 0.3).

    `100`: 상위 100 = FP 11 · TP · FP 88 → 111위 TP 는 잘린다. 정답 2 라 재현율 최대 0.5.
    `2011`: 카테고리마다 100 이라 점수 0.3 인 TP 가 남는다. **이미지 전체 100** 이었다면 `100` 의 110건이 앞서 잘렸을 것이다.
    """
    g1, g2, g3 = (0.0, 0.0, 10.0, 10.0), (500.0, 500.0, 10.0, 10.0), (200.0, 200.0, 10.0, 10.0)
    d100 = [("100", (900.0 + k, 600.0, 5.0, 5.0), 0.99 - 0.001 * k) for k in range(11)]
    d100.append(("100", g1, 0.5))
    d100 += [("100", (900.0 + k, 300.0, 5.0, 5.0), 0.49 - 0.001 * k) for k in range(98)]
    d100.append(("100", g2, 0.01))
    d2011 = [("2011", g3, 0.3)] + [("2011", (700.0 + k, 50.0, 5.0, 5.0), 0.29 - 0.001 * k) for k in range(149)]
    gold = {"a": [("100", g1), ("100", g2), ("2011", g3)]}
    return gold, {"a": d100 + d2011}


def test_절단은_이미지와_카테고리마다_상위_100건이다():
    gold, pred = _truncation_case()
    res = assert_same_as_coco(gold, pred)
    assert res.counts["det_truncated"] == 11 + 50
    assert res.counts["det_evaluated_by_class"] == {"100": 100, "2011": 100, "401": 0, "301": 0}
    p12 = 1.0 / (11.0 + 1.0 + EPS)
    ap100 = _grid_mean([p12 if r <= 0.5 else 0.0 for r in REC_THRS])
    ap2011 = _grid_mean([1.0 / (1.0 + EPS)] * len(REC_THRS))
    assert res.ap_by_class["100"] == ap100
    assert res.ap_by_class["2011"] == ap2011
    assert MAX_DETS == 100


def test_정답_없는_카테고리는_빼고_검출_없는_카테고리는_0_으로_넣는다():
    """#14·#15·#16 — `401` 은 검출만 있고 정답 0 → −1(제외, 그 FP 는 어디에도 안 들어감). `301` 은 정답만 있고 검출 0 → 0 포함."""
    gold = {"a": [("100", (0.0, 0.0, 10.0, 10.0)), ("301", (40.0, 40.0, 10.0, 10.0))], "b": []}
    pred = {"a": [("100", (0.0, 0.0, 10.0, 10.0), 0.9), ("401", (40.0, 40.0, 10.0, 10.0), 0.95)],
            "b": [("401", (1.0, 1.0, 5.0, 5.0), 0.7)]}
    res = assert_same_as_coco(gold, pred)
    assert res.ap_by_class["401"] is None and res.ap_by_class["2011"] is None
    assert res.ap_by_class["301"] == 0.0
    assert (res.precision[:, 2] == -1).all() and (res.precision[:, 3] == 0).all()
    one = 1.0 / (1.0 + EPS)
    flat = [v for r in range(len(REC_THRS)) for v in (one, 0.0)]                 # R 우선, 그 안에서 K(100, 301)
    assert res.map_50 == float(np.mean(np.array(flat)))
    assert res.counts["pairs_empty"] == 4                                          # a: 2011 · b: 100·2011·301


def test_목록_밖_클래스는_버리고_센다():
    """#17 — `402` 정답·검출은 어느 카테고리에도 들어가지 않는다."""
    gold = {"a": [("100", (0.0, 0.0, 10.0, 10.0)), ("402", (0.0, 0.0, 10.0, 10.0))]}
    pred = {"a": [("402", (0.0, 0.0, 10.0, 10.0), 0.99), ("100", (0.0, 0.0, 10.0, 10.0), 0.5)]}
    res = assert_same_as_coco(gold, pred)
    assert res.counts["dropped_class_gt"] == 1 and res.counts["dropped_class_det"] == 1
    assert res.map_50 == _grid_mean([1.0 / (1.0 + EPS)] * len(REC_THRS))


def test_재현율이_격자점과_같으면_그_점에_포함된다():
    """#1·#2 — 정답 4, 검출 TP·FP·TP·TP. 재현율 0.25 는 격자점 `REC_THRS[25]` 와 **같은지**가 그 점의 값을 정한다.

    `searchsorted(left)` 는 재현율 ≥ r 인 첫 순위를 고른다. 기대값을 같은 `linspace` 값으로 계산해 두고, 같은 입력을
    pycocotools 와 맞댄다(부동소수 격자를 십진 정확값으로 바꾸는 구현은 여기서 갈린다).
    """
    gts = [("100", (float(10 * k), 0.0, 5.0, 5.0)) for k in range(4)]
    dets = [("100", gts[0][1], 0.9), ("100", (500.0, 500.0, 5.0, 5.0), 0.8),
            ("100", gts[1][1], 0.7), ("100", gts[2][1], 0.6)]
    res = assert_same_as_coco({"a": gts}, {"a": dets})
    tp = np.array([1.0, 1.0, 2.0, 3.0])
    fp = np.array([0.0, 1.0, 1.0, 1.0])
    rc = tp / 4
    pr = (tp / (fp + tp + EPS)).tolist()
    for i in range(3, 0, -1):
        pr[i - 1] = max(pr[i - 1], pr[i])
    q = [pr[i] if i < 4 else 0.0 for i in np.searchsorted(rc, REC_THRS, side="left")]
    assert res.map_50 == _grid_mean(q)
    assert rc[0] == 0.25 and 0.2 < REC_THRS[25] < 0.3


def test_검출이_하나도_없으면_정답_있는_카테고리는_0():
    gold = {"a": [("100", (0.0, 0.0, 10.0, 10.0))]}
    res = evaluate_map50(gold, {}, CLASSES)
    assert res.map_50 == 0.0 and res.ap_by_class["100"] == 0.0


def test_정답에_없는_이미지의_검출은_거부():
    with pytest.raises(ValueError, match="정답에 없는 이미지"):
        evaluate_map50({"a": []}, {"b": [("100", (0.0, 0.0, 1.0, 1.0), 0.5)]}, CLASSES)


@pytest.mark.parametrize("bad", [
    (float("nan"), 0.0, 1.0, 1.0), (0.0, 0.0, float("inf"), 1.0), (0.0, 0.0, 1.0), ("0", 0.0, 1.0, 1.0),
    (True, 0.0, 1.0, 1.0),
])
def test_유한하지_않거나_형식이_깨진_박스는_거부(bad):
    with pytest.raises(ValueError):
        evaluate_map50({"a": [("100", bad)]}, {}, CLASSES)


@pytest.mark.parametrize("score", [float("nan"), None, "0.5", True])
def test_점수가_유한_실수가_아니면_거부(score):
    with pytest.raises(ValueError):
        evaluate_map50({"a": []}, {"a": [("100", (0.0, 0.0, 1.0, 1.0), score)]}, CLASSES)


def test_S_모드는_형식만_보고_그대로_계산한다():
    gold, pred = _truncation_case()
    assert map50_shared(gold, pred, CLASSES).map_50 == evaluate_map50(gold, pred, CLASSES).map_50
    with pytest.raises(TypeError):
        map50_shared([("a", [])], {}, CLASSES)


# ======================================================================================
# 2. 합성 무작위 — pycocotools 와 비트 대조
# ======================================================================================

def _r2(rng, lo, hi):
    return round(rng.uniform(lo, hi), 2)


def _random_case(seed):
    """실물과 비슷한 형태(정수 정답, 소수 둘째 자리 검출)에 갈림길을 일부러 섞는다."""
    rng = random.Random(seed)
    n_img = rng.randint(3, 24)
    ids = set()
    while len(ids) < n_img:
        ids.add(f"synth:{rng.choice([rng.randint(1, 99), rng.randint(10**7, 10**8 - 1)])}")
    absent = rng.choice([None, None, "401", "301"])
    gold, pred = {}, {}
    for img in sorted(ids):
        g, p = [], []
        for code in (*CLASSES, "402"):
            ng = 0 if code == absent else rng.choice([0, 0, 0, 1, 1, 2, 3, 5])
            if code == "402" and rng.random() < 0.7:
                ng = 0
            boxes = []
            for _ in range(ng):
                x, y = rng.randint(0, 1200), rng.randint(0, 650)
                w, h = rng.randint(4, 80), rng.randint(4, 70)
                boxes.append((float(x), float(y), float(w), float(h)))
                if rng.random() < 0.15:
                    boxes.append((float(x), float(y), float(w), float(h)))          # 같은 정답 둘 → IoU 동률
            g += [(code, b) for b in boxes]
            nd = rng.choice([0, 0, 1, 2, 4, 9, 30, 104, 131])
            for _ in range(nd):
                mode = rng.random()
                if boxes and mode < 0.45:
                    x, y, w, h = rng.choice(boxes)
                    x1, y1 = x + _r2(rng, -6, 6), y + _r2(rng, -6, 6)
                    x2, y2 = x + w + _r2(rng, -6, 6), y + h + _r2(rng, -6, 6)
                    if x2 <= x1 + 0.5 or y2 <= y1 + 0.5:
                        x2, y2 = x1 + w, y1 + h
                    box = (x1, y1, x2 - x1, y2 - y1)
                elif boxes and mode < 0.55:
                    box = rng.choice(boxes)                                            # IoU 1
                elif boxes and mode < 0.65:
                    x, y, w, h = rng.choice(boxes)
                    box = (x, y, 2 * w, h)                                             # IoU 정확히 0.5
                else:
                    x1, y1 = _r2(rng, 0, 1250), _r2(rng, 0, 700)
                    box = (x1, y1, _r2(rng, 1, 60), _r2(rng, 1, 60))
                score = rng.choice([round(rng.random(), 2), round(rng.random(), 4), 0.5, 0.25, 0.0101])
                p.append((code, box, score))
        rng.shuffle(g)
        rng.shuffle(p)
        gold[img] = g
        if p:
            pred[img] = p
    if not any(code in CLASSES for v in pred.values() for code, _, _ in v):
        first = min(gold)
        pred.setdefault(first, []).append(("100", (1.0, 1.0, 3.0, 3.0), 0.3))
    return gold, pred


N_RANDOM = 48


def test_합성_무작위_입력에서_pycocotools_와_비트가_같다():
    seen = {"iou_ties": 0, "boundary_pairs": 0, "det_truncated": 0, "pairs_empty": 0,
            "dropped_class_gt": 0, "dropped_class_det": 0, "absent_class": 0, "zero_ap_class": 0}
    for seed in range(N_RANDOM):
        gold, pred = _random_case(seed)
        res = assert_same_as_coco(gold, pred)
        for k in ("iou_ties", "boundary_pairs", "det_truncated", "pairs_empty", "dropped_class_gt", "dropped_class_det"):
            seen[k] += res.counts[k]
        seen["absent_class"] += sum(v is None for v in res.ap_by_class.values())
        seen["zero_ap_class"] += sum(v == 0.0 for v in res.ap_by_class.values())
        assert res.counts["ignored_gt"] == 0 and res.counts["ignored_det"] == 0
    # 이 대조가 빈 대조가 아니다 — 갈림길을 실제로 지났다
    assert all(v > 0 for v in seen.values()), seen


# ======================================================================================
# 3. import 경계
# ======================================================================================

FORBIDDEN = ("pycocotools", "evaluation.recovery_ci", "evaluation.score", "evaluation.metrics.localization",
             "scripts.probe")


def _imported_names(source: str) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = ("." * node.level) + (node.module or "")
            names.append(base)
            names += [f"{base}.{a.name}" for a in node.names]
    return names


def test_모듈은_pycocotools_와_기존_채점_경로를_import_하지_않는다():
    names = _imported_names(MODULE.read_text(encoding="utf-8"))
    for n in names:
        assert not any(n == f or n.startswith(f + ".") for f in FORBIDDEN), n
        assert not n.startswith("."), n                                          # 상대 import 없음
        assert n.split(".")[0] not in {"evaluation", "data", "scripts"}, n       # 입력 규약도 기존 코드에서 가져오지 않는다
    top = {n.split(".")[0] for n in names}
    assert top <= {"__future__", "csv", "hashlib", "json", "math", "collections", "dataclasses", "pathlib",
                   "typing", "numpy", "yaml"}, top


def test_모듈을_불러도_pycocotools_가_올라오지_않는다():
    code = ("import sys; import evaluation.metrics.map_independent as m; "
            "bad = [k for k in sys.modules if k.startswith(('pycocotools', 'evaluation.score', "
            "'evaluation.recovery_ci', 'evaluation.metrics.localization', 'scripts'))]; print(bad)")
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO, capture_output=True, text=True,
                         encoding="utf-8", check=True)
    assert out.stdout.strip() == "[]", out.stdout


ALLOWED_IMPORTERS = {"tests/test_map_independent.py", "scripts/probe/map50_independent.py"}
"""57번 00:15 판정 1 — 이 모듈(특히 I 모드 입력 함수)을 import 할 수 있는 파일. 실행기 이름이 바뀌면 여기도 바꾼다."""


def test_이_모듈을_import_하는_파일은_시험과_검산_실행기뿐이다():
    try:
        out = subprocess.run(["git", "grep", "--untracked", "-l", "map_independent", "--", "*.py"], cwd=REPO,
                             capture_output=True, text=True, encoding="utf-8", check=False)
    except FileNotFoundError:
        pytest.skip("git 없음")
    if out.returncode not in (0, 1):
        pytest.skip(f"git 저장소가 아니다(사본): {out.stderr.strip()[:80]}")
    importers = set()
    for rel in out.stdout.split():
        if rel == "evaluation/metrics/map_independent.py":
            continue
        names = _imported_names((REPO / rel).read_text(encoding="utf-8"))
        if any("map_independent" in n for n in names):
            importers.add(rel)
    assert importers <= ALLOWED_IMPORTERS, importers - ALLOWED_IMPORTERS
    assert "tests/test_map_independent.py" in importers


# ======================================================================================
# 4. I 모드 입력
# ======================================================================================

LABEL_MAP = """version: 1
defect_types:
  crack: {iso_code: "100"}
  porosity: {iso_code: "2011"}
  lack_of_penetration: {iso_code: "402"}
eval_spaces:
  main_rt:
    defect_types: [porosity, crack]
    excluded: [lack_of_penetration]
"""
"""평가 공간 순서(`porosity`, `crack`)가 `defect_types` 선언 순서와 다르다 — 클래스 순서는 평가 공간을 따른다(D8)."""

MANIFEST_HEAD = "image_id,source,split,has_localization,width_px,height_px,notes\n"
ANN_HEAD = ("ann_id,image_id,iso_code,polygon_json,bbox_x1_px,bbox_y1_px,bbox_x2_px,bbox_y2_px,"
            "area_px,geom_valid,geom_flags\n")


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("utf-8"))
    return p


def _snapshot(tmp_path: Path, *, manifest_extra: str = "", ann_rows: str | None = None) -> tuple[Path, Path]:
    snap = tmp_path / "snap"
    _write(snap / "manifest.csv", MANIFEST_HEAD
           + "x:9,x,eval,True,100,80,\n"
           + "x:10,x,eval,True,100,80,\n"
           + "x:11,x,eval,True,100,80,\n"
           + "x:12,x,train,True,100,80,\n"
           + manifest_extra)
    rows = ann_rows if ann_rows is not None else (
        # 파일은 ann_id 문자열 순서다 — x:10#10 이 x:10#2 보다 앞선다(D9)
        'x:10#0,x:10,100,"[]",0,0,10,10,1.0,True,\n'
        'x:10#1,x:10,2011,"[]",20,20,30,30,1.0,True,\n'
        'x:10#10,x:10,100,"[]",40,40,50,50,1.0,True,\n'
        'x:10#2,x:10,100,"[]",60,10,70,20,1.0,True,\n'
        'x:10#3,x:10,100,"[]",5,5,9,9,1.0,False,self_intersect\n'          # bbox 있지만 geom_valid False → 제외(D10)
        'x:10#4,x:10,2011,"[]",,,,,,False,self_intersect;zero_area\n'
        'x:11#0,x:11,402,"[]",1,1,5,5,1.0,True,\n'                        # 목록 밖 → 평가에서 버림(D12)
        'x:12#0,x:12,100,"[]",0,0,10,10,1.0,True,\n')                     # train — 읽지 않는다
    _write(snap / "annotations.csv", ANN_HEAD + rows)
    _write(snap / "SNAPSHOT.sha256", "0" * 64 + "  manifest.csv\n# snapshot_digest " + "a" * 64 + "\n")
    lm = _write(tmp_path / "label_map.yaml", LABEL_MAP)
    return snap, lm


def _rec(iid, defects, **kw):
    base = {"schema_version": "1.3", "image_id": iid, "cell": "sep_local", "client": "C1", "seed": 7,
            "defects": defects, "verdict": "판정불가", "cited_clauses": [], "parse_ok": True, "parse_error": None,
            "coord_space": "ABS_ORIG", "coord_cfg_hash": "h"}
    base.update(kw)
    return base


def _defect(code, xyxy, score):
    return {"iso_code": code, "bbox_px": list(xyxy), "score": score, "size_mm": None, "size_px": None,
            "size_basis": None, "retrieved": None}


def _records(tmp_path: Path, recs, name="rec.jsonl") -> Path:
    return _write(tmp_path / name, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))


def _good_records():
    return [
        _rec("x:11", [_defect("402", (1, 1, 5, 5), 0.9), _defect("100", (0.5, 0.25, 9.75, 10.5), 0.4)]),
        _rec("x:10", [_defect("100", (0.12, 0.34, 10.01, 9.87), 0.93),
                      {**_defect("100", (1, 1, 2, 2), 0.5), "bbox_px": None},
                      _defect("2011", (20.5, 20.5, 30.25, 29.75), 0.66),
                      _defect("100", (60.0, 10.0, 70.0, 20.0), 0.66)]),
        _rec("x:9", [_defect("100", (1, 1, 2, 2), 0.2)], parse_ok=False, parse_error="bbox_invalid"),
    ]


def test_I_모드_정답은_문서_규약대로_읽는다(tmp_path):
    snap, lm = _snapshot(tmp_path)
    g = read_gold_independent(snap, lm)
    assert g.classes == ("2011", "100")                                       # D8 평가 공간 순서
    assert g.image_ids == ("x:10", "x:11", "x:9")                             # D5 문자열 정렬
    assert g.gold["x:10"] == [("100", (0.0, 0.0, 10.0, 10.0)), ("2011", (20.0, 20.0, 10.0, 10.0)),
                              ("100", (40.0, 40.0, 10.0, 10.0)), ("100", (60.0, 10.0, 10.0, 10.0))]   # D9 파일 행 순서
    assert g.gold["x:11"] == [("402", (1.0, 1.0, 4.0, 4.0))] and g.gold["x:9"] == []
    assert "x:12" not in g.gold
    r = g.report
    assert r["gold_excluded"] == {"geom_valid=False": 2}
    assert r["gold_rule_disagreement"] == {"excluded_but_has_bbox": 1}         # bbox 유무 기준이면 남았을 행
    assert r["n_gold_boxes"] == 5 and r["n_images_with_box"] == 2
    assert r["snapshot_digest_line"] == "a" * 64
    assert set(r["files_sha256"]) == {"manifest.csv", "annotations.csv", "label_map.yaml"}


def test_I_모드_레코드는_파일값_그대로_읽고_센다(tmp_path):
    snap, lm = _snapshot(tmp_path)
    g = read_gold_independent(snap, lm)
    p = read_pred_independent(_records(tmp_path, _good_records()), g,
                              expect={"cell": "sep_local", "client": "C1", "seed": 7})
    assert p.pred["x:10"][0] == ("100", (0.12, 0.34, 10.01 - 0.12, 9.87 - 0.34), 0.93)   # D4 반올림 없음
    assert len(p.pred["x:10"]) == 3                                            # bbox null 1건 제외(D7)
    assert "x:9" not in p.pred                                                 # 파싱 실패 → 결함 없음(D19)
    c = p.report["counts"]
    assert c["records"] == 3 and c["bbox_null"] == 1 and c["parse_fail_records"] == 1
    assert c["parse_fail_defects_ignored"] == 1
    assert p.report["coord_space"] == {"ABS_ORIG": 3}


def test_I_모드와_S_모드가_같은_데이터에서_같은_비트를_낸다(tmp_path):
    snap, lm = _snapshot(tmp_path)
    g = read_gold_independent(snap, lm)
    p = read_pred_independent(_records(tmp_path, _good_records()), g)
    res_i, report = map50_independent(g, p)
    shared_gold = {k: list(v) for k, v in g.gold.items()}
    shared_pred = {k: list(v) for k, v in p.pred.items()}
    res_s = map50_shared(shared_gold, shared_pred, ("2011", "100"))
    assert res_i.map_50 == res_s.map_50
    assert np.array_equal(res_i.precision, res_s.precision)
    assert res_i.counts["dropped_class_gt"] == 1 and res_i.counts["dropped_class_det"] == 1
    assert report["not_independent"] == ["D9_gold_order"]
    assert {c["id"] for c in report["conventions"]} == set(CONVENTIONS)
    ref, _ = coco_reference(shared_gold, shared_pred, ("2011", "100"))
    assert res_i.map_50 == ref


def test_I_모드_점수_None_은_기본이_멈춤이고_0_치환은_독립_아님으로_적힌다(tmp_path):
    snap, lm = _snapshot(tmp_path)
    g = read_gold_independent(snap, lm)
    recs = _good_records()
    recs[1]["defects"][0]["score"] = None
    path = _records(tmp_path, recs)
    with pytest.raises(ValueError, match="점수가 없다"):
        read_pred_independent(path, g)
    p = read_pred_independent(path, g, score_none="zero")
    assert p.pred["x:10"][0][2] == 0.0 and p.report["counts"]["score_none"] == 1
    _, report = map50_independent(g, p)
    assert report["not_independent"] == ["D6_score_none", "D9_gold_order"]
    with pytest.raises(ValueError, match="score_none"):
        read_pred_independent(path, g, score_none="one")


@pytest.mark.parametrize("breakage,match", [
    ("coord", "ABS_ORIG"),
    ("missing", "집합이 다르다"),
    ("extra", "집합이 다르다"),
    ("duplicate", "중복"),
    ("identity", "한 값이 아니다"),
    ("expect", "기대한 칸"),
    ("degenerate", "퇴화"),
    ("nan", "유한"),
    ("blank", "빈 줄"),
    ("bom", "BOM"),
    ("field", "필수 필드"),
])
def test_I_모드_레코드_형식이_어긋나면_멈춘다(tmp_path, breakage, match):
    snap, lm = _snapshot(tmp_path)
    g = read_gold_independent(snap, lm)
    recs = _good_records()
    expect = None
    if breakage == "coord":
        recs[0]["coord_space"] = "NORM_1000"
    elif breakage == "missing":
        recs = recs[:2]
    elif breakage == "extra":
        recs.append(_rec("x:12", []))
    elif breakage == "duplicate":
        recs.append(_rec("x:9", []))
    elif breakage == "identity":
        recs[0]["seed"] = 8
    elif breakage == "expect":
        expect = {"cell": "sep_fed"}
    elif breakage == "degenerate":
        recs[1]["defects"][0]["bbox_px"] = [5, 5, 5, 9]
    elif breakage == "nan":
        recs[1]["defects"][0]["bbox_px"] = [0, 0, float("inf"), 9]
    elif breakage == "field":
        del recs[0]["parse_ok"]
    path = _records(tmp_path, recs)
    if breakage == "blank":
        path.write_bytes(path.read_bytes() + b"\n\n")
    if breakage == "bom":
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    with pytest.raises(ValueError, match=match):
        read_pred_independent(path, g, expect=expect)


@pytest.mark.parametrize("breakage,match", [
    ("bom", "BOM"),
    ("valid_without_bbox", "bbox 칸이 비었다"),
    ("out_of_bounds", "IV10"),
    ("not_int", "정수가 아니다"),
    ("no_localization", "has_localization"),
    ("dup_image", "중복"),
    ("missing_col", "열이 없다"),
    ("no_space", "eval_spaces"),
])
def test_I_모드_스냅샷_형식이_어긋나면_멈춘다(tmp_path, breakage, match):
    extra, rows = "", None
    if breakage == "valid_without_bbox":
        rows = 'x:10#0,x:10,100,"[]",,,,,,True,\n'
    elif breakage == "out_of_bounds":
        rows = 'x:10#0,x:10,100,"[]",0,0,101,10,1.0,True,\n'
    elif breakage == "not_int":
        rows = 'x:10#0,x:10,100,"[]",0,0,10.5,10,1.0,True,\n'
    elif breakage == "no_localization":
        extra = "x:13,x,eval,False,100,80,\n"
    elif breakage == "dup_image":
        extra = "x:9,x,eval,True,100,80,\n"
    snap, lm = _snapshot(tmp_path, manifest_extra=extra, ann_rows=rows)
    if breakage == "bom":
        p = snap / "annotations.csv"
        p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())
    if breakage == "missing_col":
        p = snap / "annotations.csv"
        p.write_bytes(p.read_bytes().replace(b"geom_valid", b"geom_ok", 1))
    if breakage == "no_space":
        lm.write_bytes(LABEL_MAP.replace("main_rt", "other").encode("utf-8"))
    with pytest.raises(ValueError, match=match):
        read_gold_independent(snap, lm)
