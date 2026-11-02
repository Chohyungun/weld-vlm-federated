"""등록 M1 대조선(`evaluation/box_gate_baseline.py`) — 게이트의 뜻은 합성 자료로, 봉인 예측 파일은 실물로(없으면 건너뛴다)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import box_gate_baseline as G
from evaluation.strict_main import BaselineFileError, read_baseline_predictions

SNAP = ROOT / "data/interim/manifest_v1"
LISTS = ROOT / "docs/dev_log/2026-10-16-본실험실행/eval_lists_D"
LIST = LISTS / "eval_sample_v1_1000/eval_sample_1000.txt"
BASE = LISTS / "baseline_v1_1000"
CLS = ("A", "B")


def _fit(rows: dict[str, tuple[int, list[tuple[str, list | None]]]]):
    """`id → (구간, [(코드, 상자 또는 None)])` 로 적합한다. 상자가 None 인 행은 상자 없는 주석 행이다."""
    codes = {i: {c for c, _ in v} for i, (_, v) in rows.items()}
    boxes = {i: [(c, b) for c, b in v if b is not None] for i, (_, v) in rows.items()}
    bin_of = {i: b for i, (b, _) in rows.items()}
    return G.fit_gate(list(rows), codes, boxes, bin_of, CLS), bin_of


def test_문턱은_이상이고_정확한_정수식으로_비교한다() -> None:
    rows = {f"i{n}": (0, [("A", [0, 0, 10, 10])] if n < 3 else []) for n in range(10)}       # 10 장 중 3 장 — 빈도 3/10
    fit, _ = _fit(rows)
    assert fit.fires(0, "A", 3) and not fit.fires(0, "A", 4)
    assert fit.exact_boundary_cells() == 1                                                   # 3/10 이 후보 0.3 과 같다
    assert not fit.fires(1, "A", 1)                                                          # 적합 이미지가 없는 구간은 내지 않는다


def test_상자는_좌표별_중앙값이고_상자_없는_행은_빈도에만_센다() -> None:
    rows = {"i1": (0, [("A", [0, 0, 10, 10])]), "i2": (0, [("A", [2, 2, 14, 12])]), "i5": (0, [("A", [9, 9, 30, 40])]),
            "i3": (0, [("A", None), ("B", None)]), "i4": (0, [])}
    fit, _ = _fit(rows)
    assert fit.n_with[(0, "A")] == 4 and fit.n_images[0] == 5
    assert fit.box[(0, "A")] == (2.0, 2.0, 14.0, 12.0)                                       # 평균이면 (3.67, 3.67, 18, 20.67)
    assert (0, "B") not in fit.box and not fit.fires(0, "B", 1)                              # 빈도는 있어도 상자가 없으면 내지 않는다


def test_채점_클래스_밖_코드는_버리고_예측은_구간_번호만_쓴다() -> None:
    rows = {"i1": (0, [("A", [0, 0, 4, 4]), ("Z", [0, 0, 9, 9])]), "i2": (1, [("B", [1, 1, 3, 3])]),
            "i6": (2, [("B", [5, 5, 6, 6]), ("A", [7, 7, 8, 8])])}
    fit, _ = _fit(rows)
    assert all(c in CLS for _b, c in fit.box)
    pred = G.predict_gate(fit, ["e1", "e2", "e3", "e4"], {"e1": 0, "e2": 1, "e3": 7, "e4": 2}, 5)
    assert pred == {"e1": [("A", [0.0, 0.0, 4.0, 4.0])], "e2": [("B", [1.0, 1.0, 3.0, 3.0])], "e3": [],
                    "e4": [("A", [7.0, 7.0, 8.0, 8.0]), ("B", [5.0, 5.0, 6.0, 6.0])]}                # 두 클래스면 클래스 순서


def test_문턱은_M1_최대이고_같으면_작은_t_다() -> None:
    assert G.pick_threshold({1: 0.1, 2: 0.3, 3: 0.3, 4: None}) == 2
    with pytest.raises(ValueError):
        G.pick_threshold({1: None, 2: None})


def test_예측_줄은_엄격_판독기를_지나고_목록_순서다() -> None:
    pred = {"e2": [("A", [0.5, 1.0, 3.0, 4.5])], "e1": [], "e3": [("B", [1.0, 1.0, 2.0, 2.0])]}
    order = ["e3", "e1", "e2"]                                            # 정렬 순서가 아니다 — 목록 순서를 지키는지 본다
    raw = G.prediction_bytes(pred, order)
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert [json.loads(x)["image_id"] for x in raw.decode("utf-8").splitlines()] == order
    got, info = read_baseline_predictions(raw, population=order, allowed_ids=set(order), known_codes=set(CLS), classes=CLS)
    assert got == {i: [(c, list(b)) for c, b in pred[i]] for i in order} and info["n_lines"] == 3


# ---------------------------------------------------------------- 실물(동결 매니페스트 · 봉인 목록 · 봉인 대조선)
REAL = (SNAP / "manifest.csv").is_file() and LIST.is_file() and (BASE / "BASELINE.json").is_file()


def _eval_ids() -> set[str]:
    from data.manifest_view import read_manifest_columns

    return set(read_manifest_columns(SNAP, ["image_id"], split_filter=("eval",))["image_id"].astype(str))


@pytest.mark.skipif(not REAL, reason="동결 매니페스트 · 봉인 목록 · 봉인 대조선이 이 트리에 없다")
def test_봉인_대조선은_봉인_목록과_행_단위로_맞물리고_엄격_판독기를_지난다() -> None:
    from data.label_map import load_label_map
    from evaluation.conformance_v14 import SCORING_CODES
    from evaluation.eval_list import validate_eval_list

    meta = json.loads((BASE / "BASELINE.json").read_text(encoding="utf-8"))
    raw = (BASE / meta["file"]).read_bytes()
    el = validate_eval_list(LIST.read_bytes(), where="eval_sample_1000")
    assert hashlib.sha256(raw).hexdigest() == meta["file_sha256"]
    assert (meta["list_file_sha256"], meta["list_set_sha256"], meta["n_lines"]) == (el.file_sha256, el.set_sha256, 1059)
    assert [json.loads(x)["image_id"] for x in raw.decode("utf-8").splitlines()] == list(el.ids)
    assert (meta["rule_id"], meta["strata_k"], meta["fit_split"]) == (G.RULE_ID, G.STRATA_K, G.FIT_SPLIT)
    known, ev = set(load_label_map().iso_codes()), _eval_ids()
    _, info = read_baseline_predictions(raw, population=el.ids, allowed_ids=ev, known_codes=known, classes=SCORING_CODES)
    assert info == {"n_lines": 1059, "n_outside_population": 0, "n_out_of_scope_dropped": 0}
    extra = min(ev - set(el.ids))                             # 파일이 덮지 않는 평가 id 가 모집단에 들면 거부한다
    with pytest.raises(BaselineFileError):
        read_baseline_predictions(raw, population=[*el.ids, extra], allowed_ids=ev, known_codes=known, classes=SCORING_CODES)


@pytest.mark.skipif(not REAL, reason="동결 매니페스트 · 봉인 목록 · 봉인 대조선이 이 트리에 없다")
def test_봉인_대조선은_지금_코드가_등록_문턱으로_같은_바이트를_다시_낸다() -> None:
    """train+val 재적합 · 기록한 문턱으로 다시 낸다(문턱 선택 곡선은 기록 파일과 실행 스크립트의 대조가 본다)."""
    from data.manifest_view import read_manifest_columns
    from evaluation.conformance_v14 import SCORING_CODES
    from evaluation.eval_list import validate_eval_list
    from evaluation.strata import bins_for

    meta = json.loads((BASE / "BASELINE.json").read_text(encoding="utf-8"))
    ids = list(validate_eval_list(LIST.read_bytes(), where="eval_sample_1000").ids)
    m = read_manifest_columns(SNAP, ["image_id", "split"], split_filter=G.FIT_SPLITS)
    trainval = sorted(m["image_id"].astype(str))
    codes, boxes, _ = G.read_fit_gold(SNAP, trainval)
    bin_of = bins_for([*trainval, *ids], G.STRATA_K, SNAP)
    fit = G.fit_gate(trainval, codes, boxes, bin_of, SCORING_CODES)
    k = round(meta["threshold_t"] * 10)
    assert G.prediction_bytes(G.predict_gate(fit, ids, bin_of, k), ids) == (BASE / meta["file"]).read_bytes()


@pytest.mark.skipif(not (SNAP / "manifest.csv").is_file() or not LIST.is_file(), reason="동결 매니페스트 · 봉인 목록이 이 트리에 없다")
def test_적합_정답_판독은_평가_id_를_주석_파일을_열기_전에_거부한다() -> None:
    from data.manifest_view import ManifestViewError
    from evaluation.eval_list import validate_eval_list

    one = validate_eval_list(LIST.read_bytes(), where="eval_sample_1000").ids[0]
    with pytest.raises(ManifestViewError):
        G.read_fit_gold(SNAP, [one])


@pytest.mark.skipif(not (SNAP / "manifest.csv").is_file() or not LIST.is_file(), reason="동결 매니페스트 · 봉인 목록이 이 트리에 없다")
@pytest.mark.parametrize("bad", ["train", "dup", "unknown"])
def test_예측_목록이_평가_분할의_겹치지_않는_id_가_아니면_주석을_읽기_전에_거부한다(bad, monkeypatch) -> None:
    from data.manifest_view import read_manifest_columns
    from evaluation.conformance_v14 import SCORING_CODES
    from evaluation.eval_list import validate_eval_list

    ids = list(validate_eval_list(LIST.read_bytes(), where="eval_sample_1000").ids)
    train_id = str(read_manifest_columns(SNAP, ["image_id"], split_filter=("train",))["image_id"].iloc[0])
    lst = {"train": ids[:3] + [train_id], "dup": ids[:3] + ids[:1], "unknown": ids[:3] + ["aihub71761:없는장"]}[bad]
    called = []
    monkeypatch.setattr(G, "read_fit_gold", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="겹치지 않는 평가 분할 id"):
        G.build(SNAP, lst, SCORING_CODES)
    assert called == []                                                            # 주석 파일을 열기 전이다
