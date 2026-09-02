"""사전등록 상수 + 결정론적 프로브 P1·P4 + 메타데이터 프로브 P0·P1′ 테스트.

근거 `40_규격지름길_대응판정.md` §5. **모든 게이트가 헤드라인 4결함 Macro-F1 축 위에
있는지**가 핵심이다 — 이진이나 5클래스 값으로 게이트를 세우면 미처리 원본이 통과한다.
"""

from __future__ import annotations

import pytest

from evaluation.prereg import (
    BEFORE_PREPROCESS,
    PREREG,
    R1_COUNT,
    RT_TOTAL,
    all_positive_macro_f1,
    recovery_denominator_ok,
    shortcut_contribution,
    spec_only_macro_f1,
    verify_against_prereg,
)
from evaluation.probes.deterministic import (
    ImageHeader,
    compare_header_to_manifest,
    gate,
    p1_spec_identity,
    p4_encoding_fingerprint,
)
from evaluation.probes.metadata_probe import (
    MetaSample,
    p0_baseline,
    p1_prime,
    trivial_bound,
)

CLASSES = ("100", "2011", "301", "401")

#: **동결 스냅샷 실측** 클래스별 결함 이미지 수 (`prereg_recomputed_v1.json`).
#: 이전 값 {2349, 26967, 2062, 3229} 은 동결 전 집계이고 전량도 62,998 로 690 많았다.
COUNTS = {"100": 2376, "2011": 26970, "301": 2108, "401": 3327}


# --- 사전등록 상수 재현 ---------------------------------------------------------

def test_all_positive_reproduces_registered_constant():
    macro, _ = all_positive_macro_f1(COUNTS, RT_TOTAL)
    assert macro == pytest.approx(PREREG.all_positive_macro_f1, abs=0.001)


def test_spec_only_reproduces_registered_constant():
    macro, _ = spec_only_macro_f1(COUNTS, R1_COUNT)
    assert macro == pytest.approx(PREREG.spec_only_macro_f1, abs=0.001)


def test_shortcut_contribution_reproduces():
    assert shortcut_contribution(COUNTS) == pytest.approx(
        PREREG.shortcut_contribution, abs=0.001
    )


def test_shortcut_was_real_before_preprocessing():
    """**처리 전** 부등식이 '규격이 지름길'이라는 정량 증거였다. 기록으로 남긴다."""
    assert BEFORE_PREPROCESS["spec_only_macro_f1"] > BEFORE_PREPROCESS[
        "all_positive_macro_f1"]
    assert BEFORE_PREPROCESS["shortcut_contribution"] == pytest.approx(0.0944, abs=1e-9)


def test_shortcut_is_dead_after_preprocessing():
    """**처리 후** 규격전용 규칙이 전량양성과 같아진다 — 함정 #11 대응의 성공 조건.

    재인코딩으로 전 이미지가 1280×720 이 되어 "1280×720 에만 주장하는 규칙"이
    "전부에 주장하는 규칙"과 구별되지 않는다. 순 기여 0 이 목표값이었다.
    """
    base, _ = all_positive_macro_f1(COUNTS, RT_TOTAL)
    spec, _ = spec_only_macro_f1(COUNTS, R1_COUNT)
    assert spec == pytest.approx(base, abs=1e-12)
    assert PREREG.shortcut_contribution == 0.0


def test_untreated_original_would_have_cleared_the_gate():
    """미처리 원본이 통과선을 넘었다는 것이 게이트가 필요했던 이유다(처리 전 값)."""
    assert BEFORE_PREPROCESS["spec_only_macro_f1"] > BEFORE_PREPROCESS[
        "all_positive_macro_f1"] + 0.005


def test_treated_shortcut_no_longer_clears_the_gate():
    """처리 후에는 규격 규칙이 통과선을 넘지 못한다."""
    assert PREREG.spec_only_macro_f1 < PREREG.p1_prime_gate


def test_p1_prime_gate_is_lower_bound_plus_tolerance():
    assert PREREG.p1_prime_gate == pytest.approx(0.2161, abs=1e-9)


def test_gate_line_sits_above_the_scoring_population_bound():
    """**등록 통과선이 채점 모집단 자명하한 아래에 있으면 안 된다**(80번 D2).

    이전 판은 통과선 0.2131 이 동결 평가셋 자명하한 0.21595 보다 낮았다 — 자명하한도
    못 넘는 예측기가 "통과"로 찍히는 상태였다.
    """
    from evaluation.prereg import EVAL_ALL_POSITIVE, pass_line_for

    assert pass_line_for(EVAL_ALL_POSITIVE) > EVAL_ALL_POSITIVE
    assert BEFORE_PREPROCESS["all_positive_macro_f1"] + 0.005 < EVAL_ALL_POSITIVE


def test_verify_accepts_reproduced_values():
    ok, msg = verify_against_prereg(
        PREREG.all_positive_macro_f1, PREREG.spec_only_macro_f1)
    assert ok and "재현 확인" in msg


def test_verify_rejects_drifted_measurement():
    """재현 실패는 지름길 판정 이전에 계측이 틀렸다는 뜻이다."""
    ok, msg = verify_against_prereg(PREREG.all_positive_macro_f1, 0.2500)
    assert not ok and "계측을 먼저" in msg


def test_verify_rejects_the_pre_freeze_constants():
    """**이빨 시험.** 낡은 상수(0.2081/0.3025)를 넣으면 반드시 실패해야 한다."""
    ok, _ = verify_against_prereg(0.2081, 0.3025)
    assert not ok


def test_header_rows_carry_all_three_constants():
    rows = PREREG.as_header_rows()
    joined = " ".join(v for _, v in rows)
    assert "0.2111" in joined and "+0.0000" in joined


# --- 회복률 분모 규칙 (§5-4) -----------------------------------------------------

def test_recovery_denominator_allowed_when_wide():
    ok, d, _ = recovery_denominator_ok(central=0.80, local_mean=0.60, seed_sd=0.01)
    assert ok and d == pytest.approx(0.20)


def test_recovery_denominator_blocked_when_narrow():
    """지름길이 죽으면 분모가 좁아진다 — 분산 폭증한 비율을 헤드라인에 싣지 않는다."""
    ok, _, msg = recovery_denominator_ok(central=0.62, local_mean=0.60, seed_sd=0.02)
    assert not ok and "산출하지 않고" in msg


def test_recovery_denominator_blocked_when_negative():
    ok, _, _ = recovery_denominator_ok(central=0.55, local_mean=0.60, seed_sd=0.001)
    assert not ok


# --- P1 규격 항등성 --------------------------------------------------------------

def hdr(iid: str, w=1280, h=720, **over) -> ImageHeader:
    base = {
        "mode": "L", "n_channels": 1, "subsampling": "4:4:4",
        "progressive": False, "quant_table_hash": "qt-a", "file_bytes": 100_000,
    }
    base.update(over)
    return ImageHeader(image_id=iid, width_px=w, height_px=h, **base)


def test_p1_passes_when_all_uniform():
    r = p1_spec_identity([hdr("a"), hdr("b")])
    assert r.passed and r.stats["conformance"] == pytest.approx(1.0)


def test_p1_fails_on_single_violation():
    """위반 1건이면 빌드 실패 — 확률적 판정이 아니다."""
    r = p1_spec_identity([hdr("a"), hdr("panorama", w=4096, h=720)])
    assert not r.passed
    assert r.violations == ("panorama(4096x720)",)


def test_p1_empty_input_is_not_a_pass():
    """검사 대상이 0장인데 통과로 처리하면 '검증 미도달'을 '통과'처럼 쓰게 된다."""
    assert not p1_spec_identity([]).passed


def test_header_manifest_mismatch_detected():
    """RIAWELC 227 대 224 전례 — 라벨을 믿으면 통일 안 된 채 통일됐다고 보고한다."""
    r = compare_header_to_manifest([hdr("a", w=227, h=227)], {"a": (224, 224)})
    assert not r.passed and "헤더 227x227" in r.violations[0]


def test_header_manifest_missing_entry_detected():
    r = compare_header_to_manifest([hdr("ghost")], {})
    assert not r.passed


# --- P4 인코딩 지문 --------------------------------------------------------------

def test_p4_passes_with_single_fingerprint():
    r = p4_encoding_fingerprint([hdr("a"), hdr("b"), hdr("c")])
    assert r.passed and r.stats["n_fingerprints"] == 1


def test_p4_fails_on_mixed_quant_tables():
    """압축 세대 차이가 남으면 출처가 새고, P2 가 그것을 잡는다 — 미리 막는 편이 싸다."""
    r = p4_encoding_fingerprint([hdr("a"), hdr("b"), hdr("c", quant_table_hash="qt-b")])
    assert not r.passed
    assert r.stats["n_fingerprints"] == 2
    assert r.violations == ("c",)


def test_p4_detects_channel_count_drift():
    r = p4_encoding_fingerprint([hdr("a"), hdr("rgb", mode="RGB", n_channels=3)])
    assert not r.passed


def test_read_header_extracts_jpeg_quant_table(tmp_path):
    """JPEG 양자화표는 `info` 가 아니라 `im.quantization` 에 있다.

    `info` 만 읽으면 전 파일이 빈 지문이 되어 **P4 가 공허하게 통과**한다.
    실제 tiles_v1 실행에서 이 경로로 잘못된 통과가 나왔고, 그래서 회귀로 고정한다.
    """
    from PIL import Image

    from evaluation.probes.deterministic import read_header

    path = tmp_path / "t.jpg"
    Image.new("L", (1280, 720), color=128).save(path, format="JPEG", quality=85)
    h = read_header(path, image_id="t")
    assert h.quant_table_hash, "양자화표 해시가 비어 있으면 P4 가 무력화된다"
    assert (h.width_px, h.height_px, h.mode) == (1280, 720, "L")


def test_p4_separates_different_jpeg_qualities(tmp_path):
    """압축 세대가 다르면 지문이 갈려야 한다. 갈리지 않으면 P4 가 아무것도 못 잡는다."""
    from PIL import Image

    from evaluation.probes.deterministic import p4_encoding_fingerprint, read_header

    headers = []
    for name, q in (("a", 95), ("b", 95), ("c", 40)):
        fp = tmp_path / f"{name}.jpg"
        Image.new("L", (1280, 720), color=128).save(fp, format="JPEG", quality=q)
        headers.append(read_header(fp, image_id=name))
    r = p4_encoding_fingerprint(headers)
    assert not r.passed
    assert r.stats["n_fingerprints"] == 2
    assert r.violations == ("c",)


def test_deterministic_gate_blocks_on_any_failure():
    ok, msg = gate([
        p1_spec_identity([hdr("a")]),
        p4_encoding_fingerprint([hdr("a"), hdr("b", quant_table_hash="qt-b")]),
    ])
    assert not ok and "머지 금지" in msg


def test_deterministic_gate_passes_when_all_clean():
    ok, _ = gate([p1_spec_identity([hdr("a")]), p4_encoding_fingerprint([hdr("a")])])
    assert ok


# --- P0 / P1′ 메타데이터 프로브 ----------------------------------------------------

def sample(iid: str, w: int, h: int, codes=()) -> MetaSample:
    return MetaSample(
        image_id=iid, width_px=w, height_px=h,
        file_bytes=w * h // 10, n_channels=1, quant_table_id=0,
        iso_codes=tuple(codes),
    )


def shortcut_dataset(n: int = 60) -> list[MetaSample]:
    """결함은 전량 1280×720, 정상은 파노라마 — 창원 데이터의 구조를 축소 재현."""
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(sample(f"d{i}", 1280, 720, ["2011"]))
        else:
            out.append(sample(f"n{i}", 4096, 720))
    return out


def uniform_dataset(n: int = 60) -> list[MetaSample]:
    """타일링 후 — 규격이 상수라 메타데이터에 클래스 정보가 없다."""
    return [
        sample(f"d{i}", 1280, 720, ["2011"]) if i % 2 == 0 else sample(f"n{i}", 1280, 720)
        for i in range(n)
    ]


def test_p0_detects_shortcut_before_treatment():
    """처리 전에는 픽셀 없이도 결함을 맞힌다 — 그 크기가 고쳐야 할 양이다."""
    data = shortcut_dataset()
    r = p0_baseline(data, data, CLASSES)
    assert r.macro_f1 > 0.9
    assert r.passed is None                # P0 는 크기를 재는 프로브라 판정이 없다


def test_p1_prime_passes_after_uniform_spec():
    """규격이 상수면 메타데이터로는 자명하한을 넘지 못한다.

    이 축소 표본은 결함 유병률이 실제 RT(≈54%)와 달라 자명하한도 다르다. 고정 상수가
    아니라 표본 상대 통과선으로 재야 사과 대 사과 비교가 된다.
    """
    data = uniform_dataset()
    r = p1_prime(data, data, CLASSES, relative=True)
    assert r.passed is True
    assert r.macro_f1 <= r.gate


def test_trivial_bound_depends_only_on_prevalence():
    """F1 = 2p/(1+p) — 층화 추출이면 비율이 보존돼 사전등록 상수가 재현된다."""
    small = uniform_dataset(60)
    large = uniform_dataset(600)
    assert trivial_bound(small, CLASSES) == pytest.approx(
        trivial_bound(large, CLASSES), abs=1e-9
    )


def test_fixed_gate_is_wrong_for_different_prevalence():
    """유병률이 다른 표본에 고정 상수를 대면 통과선이 틀린다 — 실측으로 드러난 함정."""
    data = uniform_dataset()
    assert trivial_bound(data, CLASSES) > PREREG.p1_prime_gate


def test_p1_prime_fails_on_untreated_data():
    """미처리 원본이 게이트를 통과하면 안 된다 — 이 정정의 핵심."""
    data = shortcut_dataset()
    r = p1_prime(data, data, CLASSES)
    assert r.passed is False
    assert "살아 있다" in r.detail


def test_probe_uses_headline_four_defect_axis():
    """정상이 5번째 클래스로 새지 않는지 — 계약 #4 축 위에서 재는지 확인."""
    data = uniform_dataset()
    r = p1_prime(data, data, CLASSES)
    assert set(r.per_class) <= set(CLASSES)
    assert "정상" not in r.per_class


def test_probe_is_deterministic_across_runs():
    """시드 고정 — 프로브 값이 흔들리면 게이트가 흔들린다."""
    data = shortcut_dataset()
    a = p0_baseline(data, data, CLASSES).macro_f1
    b = p0_baseline(data, data, CLASSES).macro_f1
    assert a == b
