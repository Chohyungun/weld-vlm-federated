"""경계 규약 감사 스크립트 — 산출이 무엇을 쟀는지 스스로 말하는가.

여기서 지키는 것 셋.

1. **재현식을 라이브러리의 판정이라 부르지 않는다.** 첫째 열은 이 스크립트가 다시 쓴 식이다.
2. **빠진 입력을 0 뒤에 숨기지 않는다.** 없으면 멈추고, 허용하면 목록과 수를 싣는다.
3. **채점 클래스를 코드에 적지 않는다.** 사상표에서 읽는다.

합성 스냅샷과 합성 예측만 쓴다 — 동결 자산과 사전실험 산출물을 읽지 않는다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.probe import boundary_convention_audit as bca


def snapshot(tmp: Path) -> Path:
    snap = tmp / "snap"
    snap.mkdir()
    with open(snap / "manifest.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "split"])
        w.writerows([["i1", "eval"], ["i2", "train"]])
    code = bca.SCORING[0]
    with open(snap / "annotations.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "iso_code", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px"])
        w.writerow(["i1", code, 0, 0, 10, 10])
        w.writerow(["i2", code, 0, 0, 10, 10])
    return snap


def predictions(root: Path, seed: int, value: str, tags) -> None:
    d = root / f"seed{seed}" / "sweep"
    d.mkdir(parents=True, exist_ok=True)
    row = {"image_id": "i1", "defects": [{"iso_code": bca.SCORING[0], "bbox_px": [0, 0, 10, 10]}]}
    for tag in tags:
        (d / f"{tag}_raw_s{value}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_빠진_입력이_있으면_멈춘다(tmp_path: Path) -> None:
    """앞 판은 한 줄 찍고 건너뛰어 합계가 '갈림 없음' 으로 읽혔다."""
    snap = snapshot(tmp_path)
    predictions(tmp_path / "root", 1, "11", bca.TAGS[:2])
    with pytest.raises(SystemExit, match="3/5 개가 없다"):
        bca.audit(tmp_path / "root", snap, [1], ["11"], allow_missing=False, log=lambda *_: None)


def test_전부_빠져도_멈춘다(tmp_path: Path) -> None:
    snap = snapshot(tmp_path)
    with pytest.raises(SystemExit, match="5/5"):
        bca.audit(tmp_path / "root", snap, [1], ["11"], allow_missing=False, log=lambda *_: None)


def test_허용하면_빠진_목록과_수를_싣는다(tmp_path: Path) -> None:
    snap = snapshot(tmp_path)
    predictions(tmp_path / "root", 1, "11", bca.TAGS[:2])
    got = bca.audit(tmp_path / "root", snap, [1], ["11"], allow_missing=True, log=lambda *_: None)
    assert got["inputs"]["expected"] == 5 and got["inputs"]["read"] == 2
    assert got["inputs"]["complete"] is False
    assert sorted(got["inputs"]["missing"]) == sorted(f"seed1/{t}" for t in bca.TAGS[2:])
    assert set(got["per_seed"]["1"]) == set(bca.TAGS[:2])


def test_다_있으면_완결이다(tmp_path: Path) -> None:
    snap = snapshot(tmp_path)
    predictions(tmp_path / "root", 1, "11", bca.TAGS)
    got = bca.audit(tmp_path / "root", snap, [1], ["11"], allow_missing=False, log=lambda *_: None)
    assert got["inputs"] == {"expected": 5, "read": 5, "missing": [], "complete": True}
    assert got["per_seed"]["1"][bca.TAGS[0]]["n_pairs"] == 1


def test_시드_값의_수가_다르면_멈춘다(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="시드 값의 수"):
        bca.input_paths(tmp_path, [1, 2], ["11"])


def test_재현식을_라이브러리_이름으로_부르지_않는다() -> None:
    """앞 판의 산출은 이 열을 `coco` 라 부르고 'pycocotools 가 쓴 것' 이라 적었다."""
    assert set(bca.CONVENTIONS) == {"float_ratio", "binary", "decimal"}
    assert "pycocotools 의 판정이 아니다" in bca.CONVENTIONS["float_ratio"]


def test_갈림_이름에도_라이브러리_이름이_없다(tmp_path: Path) -> None:
    snap = snapshot(tmp_path)
    predictions(tmp_path / "root", 1, "11", bca.TAGS)
    got = bca.audit(tmp_path / "root", snap, [1], ["11"], allow_missing=False, log=lambda *_: None)
    for per_tag in got["per_seed"]["1"].values():
        assert not any("coco" in k for k in per_tag["disagree"])


def test_채점_클래스는_사상표에서_읽는다() -> None:
    from data.label_map import load_label_map
    from evaluation.params import CLASS_NAMES
    lm = load_label_map()
    assert bca.SCORING == tuple(lm.iso_code(n) for n in CLASS_NAMES)


def test_실행_때의_코드_상태를_단언하지_않는다() -> None:
    """`why` 가 '부르는 곳은 자기 자신과 시험뿐' 을 매번 찍으면 진입점이 생긴 뒤 조용히 거짓이 된다."""
    assert "시험뿐" not in bca.WHY and "재지 않는다" in bca.WHY


def test_판정은_경계에서_셋이_갈릴_수_있다() -> None:
    """정확히 절반 겹침 — 유리수 규약은 후보, 재현식은 부동소수 결과에 따른다."""
    a, b = (0.0, 0.0, 3.0, 1.0), (1.0, 0.0, 4.0, 1.0)      # inter 2, union 4
    fr, binary, dec, near = bca.judge(a, b)
    assert binary is True and dec is True and near is True
    assert fr is True
