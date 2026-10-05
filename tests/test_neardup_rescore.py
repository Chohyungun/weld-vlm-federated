"""근사 중복 재채점 도구 — §27-24·26 의 지적마다 대응하는 작은 회귀.

여기서 지키는 것 셋.

1. **미정의와 0 을 가른다.** NaN 이 섞인 배열에 그냥 `np.percentile` 을 걸면 "가장 낮았다" 와
   "잴 수 없었다" 가 같은 값으로 나간다.
2. **맞추지 않은 축을 기록한다.** 대조 추첨이 맞춘 것은 칸별 제외 장수뿐이다. 남은 구성 차이를
   숫자로 남겨야 "구성은 같다" 는 과한 문장을 쓰지 않게 된다.
3. **입력이 모집단 안인지 본다.** 목록에 밖 ID 가 있으면 조용히 지나가고 목록 장수와 실제
   적용 수가 갈라진다.
4. **구간이 어느 계약으로 났는지 산출이 스스로 말한다.** 통합형 등록 규약의 값이 아니라는 문장까지.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from scripts.probe import neardup_rescore as nr

# ---------------------------------------------------------------- 미정의 처리

def test_미정의_추첨을_세고_순위에서_뺀다() -> None:
    got = nr._against_control([1.0, 2.0, float("nan")], 0.5)
    assert (got["n_draws"], got["n_defined"], got["n_undefined"]) == (3, 2, 1)
    assert got["control_mean"] == pytest.approx(1.5), "미정의를 평균에 섞지 않는다"
    assert got["percentile_of_actual"] == 0.0


def test_유효_추첨이_없으면_백분위를_숫자로_내지_않는다() -> None:
    """0.0 으로 내보내면 '가장 낮았다' 로 읽힌다."""
    got = nr._against_control([float("nan"), float("nan")], 1.0)
    assert got["percentile_of_actual"] is None
    assert got["control_mean"] is None
    assert got["reason"] == "유효 추첨 0"
    assert got["n_undefined"] == 2


def test_실제값이_미정의면_그렇다고_말한다() -> None:
    got = nr._against_control([1.0, 2.0], float("nan"))
    assert got["actual"] is None and got["percentile_of_actual"] is None
    assert got["reason"] == "실제값 미정의"


def test_전부_유효하면_사유를_붙이지_않는다() -> None:
    got = nr._against_control([1.0, 2.0, 3.0], 4.0)
    assert "reason" not in got
    assert got["percentile_of_actual"] == 100.0
    assert got["gap_vs_control_mean"] == pytest.approx(2.0)


def test_구성_퍼짐은_None_과_NaN_을_걸러낸다() -> None:
    got = nr._spread([0.39, 0.40, None, float("nan")])
    assert got["n"] == 2
    assert (got["min"], got["max"]) == (0.39, 0.40)


def test_구성_퍼짐이_빌_수_있다() -> None:
    assert nr._spread([None, float("nan")]) == {"n": 0}


# ---------------------------------------------------------------- 지지량 클래스

@dataclass
class _Cat:
    npig: np.ndarray


@dataclass
class _Cache:
    per_class: list


def test_가중이_0_인_이미지의_클래스는_지지량에서_빠진다() -> None:
    """`weighted_map` 이 GT 0 클래스를 평균에서 빼므로 고정 support macro AP 가 아니다."""
    cache = _Cache([_Cat(np.array([2, 0])), _Cat(np.array([0, 3]))])
    assert nr.n_support_classes(cache, np.array([1, 1])) == 2
    assert nr.n_support_classes(cache, np.array([1, 0])) == 1
    assert nr.n_support_classes(cache, np.array([0, 0])) == 0


# ---------------------------------------------------------------- 대조 칸과 서명

@dataclass
class _Pop:
    rows: list


def _pop(rows) -> _Pop:
    return _Pop([{"image_id": i, "strata_key": s} for i, s in rows])


def test_칸은_층과_출처를_함께_잡는다(monkeypatch) -> None:
    prov = {"a": "N-crop", "b": "N-tile", "c": "N-crop"}
    monkeypatch.setattr(nr, "load_provenance", lambda _p: prov)
    pop = _pop([("a", "ST|porosity"), ("b", "ST|porosity"), ("c", "ST|porosity")])
    members, n_drop, meta = nr.control_cells(pop, frozenset({"a"}), nr.Path("."))
    assert set(members) == {("ST|porosity", "N-crop"), ("ST|porosity", "N-tile")}
    assert n_drop[("ST|porosity", "N-crop")] == 1
    assert n_drop[("ST|porosity", "N-tile")] == 0
    assert meta["cells"]["ST|porosity|N-crop"] == {"n": 2, "n_dropped": 1}


def test_출처를_모르는_이미지가_있으면_멈춘다(monkeypatch) -> None:
    monkeypatch.setattr(nr, "load_provenance", lambda _p: {"a": "N-crop"})
    pop = _pop([("a", "ST|porosity"), ("b", "ST|porosity")])
    with pytest.raises(SystemExit, match="출처 미상"):
        nr.control_cells(pop, frozenset(), nr.Path("."))


def test_같은_입력이면_칸_서명이_같다(monkeypatch) -> None:
    prov = {"a": "N-crop", "b": "N-crop"}
    monkeypatch.setattr(nr, "load_provenance", lambda _p: prov)
    pop = _pop([("a", "ST|porosity"), ("b", "ST|porosity")])
    one = nr.control_cells(pop, frozenset({"a"}), nr.Path("."))[2]["signature"]
    two = nr.control_cells(pop, frozenset({"a"}), nr.Path("."))[2]["signature"]
    assert one == two


def test_칸_안_순서가_바뀌면_서명이_갈린다(monkeypatch) -> None:
    """서명이 같아야 'b 번째 추첨이 세 시드에서 같은 장을 뺀다' 가 성립한다."""
    prov = {"a": "N-crop", "b": "N-crop"}
    monkeypatch.setattr(nr, "load_provenance", lambda _p: prov)
    fwd = _pop([("a", "ST|porosity"), ("b", "ST|porosity")])
    rev = _pop([("b", "ST|porosity"), ("a", "ST|porosity")])
    a = nr.control_cells(fwd, frozenset({"a"}), nr.Path("."))[2]["signature"]
    b = nr.control_cells(rev, frozenset({"a"}), nr.Path("."))[2]["signature"]
    assert a != b


def test_제외_수가_바뀌면_서명이_갈린다(monkeypatch) -> None:
    prov = {"a": "N-crop", "b": "N-crop"}
    monkeypatch.setattr(nr, "load_provenance", lambda _p: prov)
    pop = _pop([("a", "ST|porosity"), ("b", "ST|porosity")])
    a = nr.control_cells(pop, frozenset({"a"}), nr.Path("."))[2]["signature"]
    b = nr.control_cells(pop, frozenset({"a", "b"}), nr.Path("."))[2]["signature"]
    assert a != b


# ---------------------------------------------------------------- 제외 목록

def test_중복이_있는_목록은_거부한다(tmp_path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("a\nb\na\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="중복"):
        nr.load_exclusion(p)


def test_목록의_해시와_장수를_함께_돌려준다(tmp_path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes(b"a\nb\n")
    ids, meta = nr.load_exclusion(p)
    assert ids == frozenset({"a", "b"})
    assert meta["n"] == 2
    assert len(meta["sha256"]) == 64


# ---------------------------------------------------------------- 출처 결속

def test_채점_경로_파일의_해시를_전부_낸다() -> None:
    got = nr.code_provenance()
    assert set(got) == set(nr.SCORER_FILES)
    assert all(len(v) == 64 for v in got.values())


def test_이_스크립트_자신이_결속_대상에_있다() -> None:
    """도구가 바뀌면 값도 바뀔 수 있다 — 빠지면 그 사실이 산출물에 남지 않는다."""
    assert "scripts/probe/neardup_rescore.py" in nr.SCORER_FILES


# ---------------------------------------------------------------- 구간의 계약을 산출이 싣는다

def test_옛_등록_블록을_통째로_싣는다() -> None:
    """같은 함수로 구간을 내는 사전실험 스크립트가 그렇게 한다 — 값 옆에서 출처와 지위를 읽게."""
    from evaluation.prereg import RECOVERY_CI_REGISTRATION
    got = nr.interval_convention()
    assert got["registration"] == RECOVERY_CI_REGISTRATION


def test_신뢰수준과_방식과_미정의_처리의_이름을_싣는다() -> None:
    got = nr.interval_convention()
    assert got["alpha"] == 0.05
    assert got["interval"] == "백분위"
    assert "linear" in got["percentile_method"] and "numpy" in got["percentile_method"]
    assert got["undefined_denominator"].startswith("drop_undefined")


def test_통합형_등록_규약의_값이_아니라고_말한다() -> None:
    assert "통합형 등록 규약으로 낸 값이 아니다" in nr.interval_convention()["not_unified_registration"]


def test_대조_분포의_백분위는_신뢰구간이_아니라고_말한다() -> None:
    cp = nr.interval_convention()["control_percentiles"]
    assert cp["levels"] == [2.5, 97.5]
    assert "신뢰구간이 아니다" in cp["what_this_is_not"]


def test_구간_계산이_등록된_신뢰수준을_쓴다(monkeypatch) -> None:
    """상수 0.05 를 따로 적지 않는다 — 등록이 바뀌면 같이 바뀌어야 한다."""
    seen = []
    monkeypatch.setattr(nr, "percentile_ci", lambda draws, alpha: seen.append(alpha) or {})
    curves = {1: {"before": {"r": [0.1, 0.2], "d": [1.0, 1.0]}, "after": {"r": [0.1, 0.2], "d": [1.0, 1.0]}}}
    out = nr.combine_curves(curves, [1], 2)
    assert set(seen) == {nr.RECOVERY_CI_ALPHA}
    assert out["alpha"] == nr.RECOVERY_CI_ALPHA
    assert "interval_convention" in out["convention"]


def test_no_ci_도움말이_대조_분포가_남는다고_말한다() -> None:
    """앞 판은 '점추정만 낸다' 였다 — --control 과 함께 주면 사실이 아니다."""
    ap = nr.build_parser()
    (act,) = [a for a in ap._actions if "--no-ci" in a.option_strings]
    assert "--control" in act.help and "신뢰구간이 아닌" in act.help
    assert "점추정만" not in act.help


def test_노출_설명은_수치가_무엇인지를_말한다() -> None:
    """실행 때 산출 파일에 찍히는 문자열이다(`exposure_note`). 앞 판은 누가 준 수치인지를 적었다."""
    assert nr.EXPOSURE_NOTE.startswith("train 에 상대가 있는 eval 장수. 이 스크립트에 적어 둔 고정 배경 수치")


# ---------------------------------------------------------------- 점추정의 미정의

def test_시드_하나가_미정의면_평균도_미정의다() -> None:
    """앞 판은 NaN 평균을 쓰다가 표준 JSON 쓰기에서 멈췄다 — 계산을 다 한 뒤였다."""
    per_seed = {1: {"recovery_before": 0.2, "recovery_after": nr.finite_or_none(float("nan"))},
                2: {"recovery_before": 0.4, "recovery_after": 0.3}}
    got = nr.summarize_recovery(per_seed, [1, 2])
    assert got["recovery_mean_before"] == pytest.approx(0.3)
    assert got["recovery_mean_after"] is None
    assert got["n_seeds_undefined_after"] == 1
    assert "null" in got["undefined_rule"]
    import json
    json.dumps(got, allow_nan=False)


def test_유한하지_않은_값은_None_이다() -> None:
    assert nr.finite_or_none(float("nan")) is None
    assert nr.finite_or_none(float("inf")) is None
    assert nr.finite_or_none(None) is None
    assert nr.finite_or_none(0.0) == 0.0


# ---------------------------------------------------------------- 결함 여부 표기

@pytest.mark.parametrize(("raw", "want"), [("True", True), ("False", False)])
def test_결함_여부는_두_글자만_받는다(raw: str, want: bool) -> None:
    assert nr.is_defect(raw) is want


@pytest.mark.parametrize("raw", ["true", "1", "", "TRUE"])
def test_표기가_바뀌면_전부_정상으로_세지_않고_멈춘다(raw: str) -> None:
    with pytest.raises(SystemExit, match="has_defect"):
        nr.is_defect(raw)


def test_칸_서명은_한_함수가_낸다(monkeypatch) -> None:
    """대조 재현이 같은 함수로 서명을 내 맞댄다 — 식을 두 곳에 적지 않는다."""
    prov = {"a": "N-crop", "b": "N-crop"}
    monkeypatch.setattr(nr, "load_provenance", lambda _p: prov)
    pop = _pop([("a", "ST|porosity"), ("b", "ST|porosity")])
    members, n_drop, meta = nr.control_cells(pop, frozenset({"a"}), nr.Path("."))
    assert meta["signature"] == nr.cells_signature(members, n_drop)
