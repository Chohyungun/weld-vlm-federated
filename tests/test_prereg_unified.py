"""등록 블록 — 07번 미니스펙 §12-5·§16-5·§18-6·§29 가 요구하는 성질.

여기서 지키는 것 여섯.

1. **두 지문이 따로 움직인다.** 채점 규칙을 고쳐도 생성 지문은 그대로여야 한다. 아니면
   지표를 손볼 때마다 전량 재추론이 된다.
2. **빈 값은 `None` 뿐이다.** `False`·`0` 을 빈 값으로 세면 "쓰지 않기로 정했다"는 결정이
   등록 누락으로 둔갑한다. 출발점 두 필드는 짝이다 — 정확히 하나가 값이다.
3. **값이 있다고 등록인 것은 아니다.** 자료형·해시 꼴·어휘·범위·칸 사이의 관계를 본다.
   앞 판은 정수 칸에 문자열을 넣은 등록을 완결로 받았다.
4. **무엇이 비었는지, 무엇이 틀렸는지 이름으로 말한다.**
5. **영수증은 지금 등록을 가리켜야 한다.** 커밋 시각은 근거가 못 된다 — 이력이 다시 쓰인다.
   영수증의 종류와 등록의 목적이 같아야 하고, 등록은 영수증이 가리키는 파일 바이트에서 만든다.
6. **목록 해시 두 벌이 실제 목록에 묶인다.** 채점 목록은 생성 목록과 같거나 그 부분집합이다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields as dc_fields
from dataclasses import replace

import pytest

from evaluation import prereg_unified as P
from evaluation.eval_list import canonical_bytes, validate_eval_list
from evaluation.prereg_unified import (
    DEPENDENCIES,
    BaselineSpec,
    CanarySpec,
    CiSpec,
    CoordCfgRecord,
    EchoGateFailed,
    GenerationSpec,
    ListBindingError,
    MetricSpec,
    PopulationSpec,
    Receipt,
    ReceiptMissing,
    RegistrationFileError,
    RegistrationIncomplete,
    RegistrationInvalid,
    UnifiedRegistration,
    check_echo_gate,
    check_list_binding,
    dump_registration,
    load_registration,
    read_receipt,
    registration_for_receipt,
    require_receipt,
)

SPECS = {"metric": MetricSpec, "baseline": BaselineSpec, "ci": CiSpec,
         "population": PopulationSpec, "generation": GenerationSpec, "canary": CanarySpec}


def leaf_names() -> set[str]:
    """등록이 담는 잎 항목의 이름 전량 — 스펙 정의에서 직접 센다."""
    out = {f"{part}.{f.name}" for part, cls in SPECS.items() for f in dc_fields(cls)}
    out |= {"seeds", "planned_seed_count", "adoption_anchor"}
    return out


def hx(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


GEN_LIST = validate_eval_list(canonical_bytes(["a3", "a1", "a2"]))


def cfg(**over) -> CoordCfgRecord:
    kw = {"accepted": ("c598d549", "0cc62b38"), "frozen_value": "c598d549",
          "frozen_at": "2026-09-02", "current_value": "0cc62b38",
          "change_scope": "comments_only",
          "equivalence_evidence": ("AST 동일", "골든 픽스처 지문 동일")}
    kw.update(over)
    return CoordCfgRecord(**kw)


_SPECIAL = {
    "metric.iou_threshold": "1/2",
    "metric.class_codes": ("100", "2011", "401", "301"),
    "ci.alpha": 0.05,
    "ci.n_resamples": 2000,
    "ci.min_defined_draws": 1900,
    "ci.rule4_max_undefined_fraction": 0.1,
    "ci.rule1_multiplier_by_seed_count": {"3": 3, "2": 6},
    "baseline.full_population_m1": 0.16,
    "generation.init_adapter_digest": None,
    "generation.eval_list_file_sha256": GEN_LIST.file_sha256,
    "generation.eval_list_set_sha256": GEN_LIST.set_sha256,
    "population.eval_list_file_sha256": GEN_LIST.file_sha256,
    "population.eval_list_set_sha256": GEN_LIST.set_sha256,
    "generation.processor_min_pixels": 3136,
    "generation.processor_max_pixels": 1_003_520,
    "generation.decoding": {"do_sample": False, "num_beams": 1},
    "generation.frame_diag_rules_path": "configs/registration/frame_diag_rules-20261001-1.json",
    "generation.impl_ids": ("export_generator=vlm.export_run:load_generator@vlm/export_run.py",
                            "export_preflight=vlm.export_run:preflight@vlm/export_run.py"),
    "canary.coord_cfg": None,          # 아래에서 객체를 넣는다
}


def valid_value(name: str, kind: str):
    """칸 하나에 올 수 있는 값. **내용은 보기 값**이다 — 어휘의 첫째를 고른 것이 결정이 아니다."""
    if name in _SPECIAL:
        return _SPECIAL[name]
    if name in P.CI_ADOPTED:
        return P.CI_ADOPTED[name]
    if name in P.VOCAB:
        return P.VOCAB[name][0]
    if name in P.HEX64_FIELDS:
        return hx(name)
    return {"str": f"v_{name}", "int": 7, "float": 0.5, "bool": False, "dict": {"k": 1},
            "str_tuple": ("x", "y"), "str_int_dict": {"3": 3}}[kind]


def filled(*, generation_only: bool = False, purpose: str = "main") -> UnifiedRegistration:
    """모든 칸이 유효한 값으로 찬 등록. 출발점은 **유지** 쪽 하나만 찬다.

    목적이 null 을 요구하는 칸(판 3 — 본실험의 진단 규칙 두 칸 · 학습 행 digest · 계획 파일 해시)은 비운다.
    """
    r = UnifiedRegistration()
    kinds = P._kinds()
    for area in ("generation",) if generation_only else SPECS:
        spec = getattr(r, area)
        for f in dc_fields(spec):
            name = f"{area}.{f.name}"
            setattr(spec, f.name, valid_value(name, kinds[name]))
    r.generation.purpose, r.generation.list_split = purpose, P.SPLIT_OF_PURPOSE[purpose]
    for name in P.PURPOSE_NULL[purpose]:
        setattr(r.generation, name.split(".", 1)[1], None)
    if not generation_only:
        r.canary.coord_cfg = cfg()
        r.seeds = {"1": 20260828, "2": 20260829, "3": 20260830}
        r.planned_seed_count = 3
        r.adoption_anchor = "reg-2026-10-01"
    return r


# ---------------------------------------------------------------- 비어 있는 항목

def test_새_등록은_모든_항목이_비어_있다() -> None:
    assert set(UnifiedRegistration().missing()) == leaf_names()


def test_비어_있는_항목을_이름으로_준다() -> None:
    gaps = UnifiedRegistration().missing()
    assert "metric.iou_threshold" in gaps
    assert "ci.cluster_unit" in gaps
    assert "population.exclusion_list_sha256" in gaps
    assert gaps == sorted(gaps), "정렬돼 있어야 diff 가 읽힌다"


@pytest.mark.parametrize("name", ["ci.percentile_method", "ci.infinity_handling",
                                  "ci.interval_absence_representation", "ci.min_defined_draws",
                                  "ci.endpoint_states", "ci.half_width_states", "ci.numerator_zero_rule",
                                  "ci.numerator_zero_test", "ci.percentile_n_definition",
                                  "ci.point_denominator_failure", "ci.interval_scope", "ci.convention_id"])
def test_구간_규약_칸이_채점_쪽_완결에_요구된다(name: str) -> None:
    """앞 판에는 칸이 없어서 보간·무한·부재 표현의 결정이 하나도 없는 등록이 채점 쪽 완결을 지났다."""
    r = filled()
    area, key = name.split(".")
    setattr(getattr(r, area), key, None)
    assert name in r.missing(side="scoring")
    with pytest.raises(RegistrationIncomplete, match=key):
        r.require_complete(side="scoring")


@pytest.mark.parametrize("name", ["generation.purpose", "generation.list_split",
                                  "generation.snapshot_digest"])
def test_목적과_분할과_동결본_칸이_생성_쪽_완결에_요구된다(name: str) -> None:
    r = filled(generation_only=True)
    setattr(r.generation, name.split(".")[1], None)
    assert name in r.missing(side="generation")


def test_목적과_분할과_동결본이_생성_지문에_들어간다() -> None:
    """목록이 우연히 같아도 목적·분할·동결본이 다르면 지문이 갈린다."""
    base = filled(generation_only=True)
    for key, other in (("purpose", "rehearsal"), ("list_split", "val"), ("snapshot_digest", hx("다른"))):
        r = filled(generation_only=True)
        setattr(r.generation, key, other)
        assert r.generation_sha256() != base.generation_sha256(), key


def test_False_와_0_은_값이다() -> None:
    """'쓰지 않기로 정했다'가 '등록하지 않았다'로 둔갑하면 안 된다."""
    r = UnifiedRegistration()
    r.ci.rule3_denominator_excludes_zero = False
    r.ci.rule4_max_undefined_fraction = 0.0
    r.canary.echo_max_coord_diff = 0.0
    r.generation.batch_size = 0
    gaps = set(r.missing())
    for name in ("ci.rule3_denominator_excludes_zero", "ci.rule4_max_undefined_fraction",
                 "canary.echo_max_coord_diff", "generation.batch_size"):
        assert name not in gaps
    assert r.invalid() == []


def test_빈_문자열과_빈_사전은_빈_칸이_아니지만_무효다() -> None:
    """빈 칸(`missing`)과 무효(`invalid`)는 다른 물음이다. 앞 판은 빈 사전 디코딩을 완결로 받았다."""
    r = UnifiedRegistration()
    r.generation.padding_side = ""
    r.generation.decoding = {}
    gaps = set(r.missing())
    assert "generation.padding_side" not in gaps
    assert "generation.decoding" not in gaps
    bad = r.invalid()
    assert any(b.startswith("generation.padding_side:") for b in bad)
    assert any(b.startswith("generation.decoding:") and "빈 사전" in b for b in bad)


def test_한쪽만_물을_수_있다() -> None:
    r = filled(generation_only=True)
    assert r.missing(side="generation") == []
    assert r.missing(side="scoring"), "채점 쪽은 아직 비어 있다"
    assert not any(g.startswith("generation.") for g in r.missing(side="scoring"))


def test_모르는_영역은_거부한다() -> None:
    with pytest.raises(ValueError, match="영역"):
        UnifiedRegistration().missing(side="chaejeom")


def test_비었으면_거부하고_이름을_들고_있다() -> None:
    with pytest.raises(RegistrationIncomplete) as exc:
        UnifiedRegistration().require_complete()
    assert "metric.metric_id" in exc.value.missing
    assert "metric.metric_id" in str(exc.value)


def test_생성만_채우면_생성_쪽은_통과한다() -> None:
    """채점 쪽 개정이 남아 있어도 내보내기는 돌 수 있어야 한다 (§12-5)."""
    r = filled(generation_only=True)
    r.require_complete(side="generation")
    with pytest.raises(RegistrationIncomplete):
        r.require_complete(side="scoring")


def test_전부_채운_보기_등록은_완결이다() -> None:
    filled().require_complete()


# ---------------------------------------------------------------- 출발점 두 필드

def test_출발점이_둘_다_비면_둘_다_빈_칸이다() -> None:
    r = filled(generation_only=True)
    r.generation.start_checkpoint_sha256 = None
    gaps = r.missing(side="generation")
    assert "generation.start_checkpoint_sha256" in gaps and "generation.init_adapter_digest" in gaps


@pytest.mark.parametrize("keep", ["start_checkpoint_sha256", "init_adapter_digest"])
def test_출발점은_하나만_값이면_완결이다(keep: str) -> None:
    """앞 판은 둘 다 비지 않기를 요구했다 — 배경지식 단계를 폐지하면 등록이 영원히 미완이었다."""
    r = filled(generation_only=True)
    r.generation.start_checkpoint_sha256 = None
    r.generation.init_adapter_digest = None
    setattr(r.generation, keep, hx("출발점"))
    r.require_complete(side="generation")


def test_출발점이_둘_다_값이면_무효다() -> None:
    r = filled(generation_only=True)
    r.generation.init_adapter_digest = "digest-v1"
    with pytest.raises(RegistrationInvalid, match="출발점 두 필드가 둘 다 값이다"):
        r.require_complete(side="generation")


# ---------------------------------------------------------------- 유효성

@pytest.mark.parametrize(("name", "bad", "why"), [
    ("generation.max_new_tokens", "v_max_new_tokens", "정수"),   # 앞 판의 시험이 이 값을 완결로 받았다
    ("generation.max_new_tokens", True, "정수"),
    ("generation.max_new_tokens", 2368.0, "정수"),
    ("generation.max_new_tokens", 0, "1 이상"),
    ("generation.batch_size", -1, "0 이상"),
    ("generation.prompt_sha256", "abc", "sha256"),
    ("generation.prompt_sha256", "A" * 64, "sha256"),
    ("generation.coord_space", "abs_orig", "허용 어휘"),
    ("generation.purpose", "probe", "허용 어휘"),
    ("metric.iou_threshold", "0.5", "유리수"),
    ("metric.iou_threshold", "3/2", "1 이하"),
    ("metric.coupling_rule", "coupled_v1", "허용 어휘"),
    ("metric.class_codes", ("100", "100"), "중복"),
    ("ci.alpha", 1.0, "0 과 1 사이"),
    ("ci.alpha", True, "유한한 수"),
    ("ci.rule4_max_undefined_fraction", 1.5, "1 이하"),
    ("ci.undefined_denominator_policy", "drop_undefined", "채택값"),
    ("ci.percentile_method", "linear", "채택값"),
    ("ci.percentile_method", "lower_lo_higher_hi", "채택값"),
    ("ci.infinity_handling", "drop_nonfinite", "채택값"),
    ("ci.interval_absence_representation", "null", "채택값"),
    ("ci.numerator_zero_rule", "pos_inf", "채택값"),
    ("ci.numerator_zero_test", "abs_tol_1e-12", "채택값"),
    ("ci.convention_id", "interval-20261001-2", "채택값"),
    ("ci.endpoint_states", ("finite", "absent"), "채택값"),
    ("ci.half_width_states", ("finite", "infinite", "absent", "undefined"), "채택값"),
    ("ci.rule4_counts", ("support", "indeterminate", "denominator"), "채택값"),
    ("ci.min_defined_draws", 0, "1 이상"),
    ("generation.impl_ids", ("export.main_uni",), "열쇠"),
    ("generation.impl_ids", ("a=b:c",), "열쇠"),
    ("ci.rule3_denominator_excludes_zero", 1, "참·거짓"),
    ("canary.echo_max_coord_diff", -0.1, "0 이상"),
    ("canary.axis_ratio_threshold", 0.0, "0 보다"),
    ("population.weighting", "inverse_probability", "허용 어휘"),
    ("baseline.threshold_t", float("nan"), "유한한 수"),
])
def test_그_칸에_올_수_없는_값은_무효다(name: str, bad, why: str) -> None:
    r = filled()
    area, key = name.split(".")
    setattr(getattr(r, area), key, bad)
    assert r.missing() == [], "빈 칸이 아니다 — 앞 판은 여기서 끝났다"
    with pytest.raises(RegistrationInvalid) as exc:
        r.require_complete()
    hit = [p for p in exc.value.problems if p.startswith(f"{name}:")]
    assert hit and why in hit[0], exc.value.problems


def test_등록_거부는_사유_코드를_싣는다() -> None:
    """결정 04 의 4절 13 — 다른 거부와 같은 `[코드]` 꼴. 쓰는 쪽이 예외 이름이 아니라 코드로 맞댄다."""
    inv, inc = RegistrationInvalid(["a: b"]), RegistrationIncomplete(["x"])
    assert inv.code == "REGISTRATION_INVALID" and str(inv).startswith("[REGISTRATION_INVALID] ")
    assert inc.code == "REGISTRATION_INCOMPLETE" and str(inc).startswith("[REGISTRATION_INCOMPLETE] ")
    assert inv.problems == ["a: b"] and inc.missing == ["x"]


def test_무효는_사유를_전부_낸다() -> None:
    r = filled()
    r.generation.max_new_tokens = "많이"
    r.ci.alpha = 2.0
    r.metric.iou_threshold = "0.5"
    assert len(r.invalid()) == 3


def test_재표집_수보다_큰_유효_추첨_하한은_무효다() -> None:
    r = filled()
    r.ci.min_defined_draws = r.ci.n_resamples + 1
    assert any(p.startswith("ci.min_defined_draws:") for p in r.invalid())


@pytest.mark.parametrize(("seeds", "why"), [
    ({"1": 20260828, "2": 20260828}, "겹친다"),
    ({"0": 1}, "키가"),
    ({"a": 1}, "키가"),
    ({"1": True}, "정수"),
    ({}, "비지 않은"),
])
def test_시드_사상의_꼴(seeds, why: str) -> None:
    r = filled()
    r.seeds = seeds
    bad = [p for p in r.invalid() if p.startswith("seeds:")]
    assert bad and why in bad[0]


@pytest.mark.parametrize(("purpose", "split"), [("main", "val"), ("rehearsal", "eval"), ("frame_diag", "eval")])
def test_목적과_분할이_한_줄로_서지_않으면_무효다(purpose: str, split: str) -> None:
    r = filled(generation_only=True, purpose=purpose)
    r.generation.list_split = split
    with pytest.raises(RegistrationInvalid, match="기준 분할"):
        r.require_complete(side="generation")


def test_리허설_목적은_val_분할과_선다() -> None:
    r = filled(generation_only=True)
    r.generation.purpose, r.generation.list_split = "rehearsal", "val"
    r.require_complete(side="generation")


def test_구간_규약_칸은_채택값만_받는다() -> None:
    """판 4 — 2026-10-01 에 채택했다(의사결정로그 10-01 · 결정표 11행). 판 3 까지의 이 시험은 "어휘는 고른 값이 아니라
    선택지다" 였다 — 고르기 전이었다. 선택지 기록은 남기되 검사에 쓰지 않는다."""
    from evaluation import percentile_rule as PR
    from evaluation import recovery_interval as RI
    assert P.CI_ADOPTED == {f"ci.{k}": v for k, v in RI.REGISTERED_CI.items()}, "값의 정본은 집계 모듈 하나"
    assert P.CI_ADOPTED["ci.percentile_method"] == PR.PERCENTILE_RULE
    assert "((n-1)*25)//1000" in PR.PERCENTILE_RULE and "no interpolation" in PR.PERCENTILE_RULE
    assert P.CI_ADOPTED["ci.undefined_denominator_policy"] == "signed_infinity"
    assert P.CI_ADOPTED["ci.infinity_handling"] == "include_as_extreme"
    assert P.CI_ADOPTED["ci.endpoint_states"] == ("finite", "pos_inf", "neg_inf", "absent")
    assert P.CI_ADOPTED["ci.half_width_states"] == ("finite", "infinite", "undefined", "absent")
    assert P.CI_ADOPTED["ci.numerator_zero_rule"] == "indeterminate"
    assert P.CI_ADOPTED["ci.numerator_zero_test"] == "exact_float_equality"
    assert P.CI_ADOPTED["ci.point_denominator_failure"] == "absent"
    assert P.CI_ADOPTED["ci.rule4_counts"] == ("support", "indeterminate")
    assert not any(k in P.VOCAB for k in P.CI_ADOPTED), "검사가 두 곳이면 어느 것이 정본인지 갈린다"
    assert len(P.UNDEFINED_DENOMINATOR_POLICIES) == 3 and "linear" in P.PERCENTILE_METHODS, "선택지 기록은 남는다"
    assert UnifiedRegistration().ci.percentile_method is None, "새 등록의 칸은 여전히 비어 있다 — 채우는 것은 등록이다"
    assert UnifiedRegistration().ci.min_defined_draws is None, "채택 행이 수를 정하지 않았다"


def test_순위_식은_α_가_0_05_일_때만_선다() -> None:
    r = filled()
    assert not r.invalid(side="scoring")
    r.ci.alpha = 0.1
    assert any(p.startswith("ci.alpha:") and "25/1000" in p for p in r.invalid(side="scoring"))


# ---------------------------------------------------------------- 목록 해시 두 벌

def test_파일_해시가_같은데_집합_해시가_다르면_무효다() -> None:
    r = filled()
    r.population.eval_list_set_sha256 = hx("딴집합")
    assert any(p.startswith("population.eval_list_set_sha256:") for p in r.invalid(side="scoring"))


def test_같은_목록이면_두_벌이_묶인다() -> None:
    check_list_binding(filled(), GEN_LIST, GEN_LIST)


def test_부분집합_재채점은_받는다() -> None:
    """§13-2 가 — 채점 모집단은 생성 목록의 부분집합이면 된다. 두 벌이 늘 같아야 한다고 하면 이 길이 막힌다."""
    sub = validate_eval_list(canonical_bytes(["a1", "a2"]))
    r = filled()
    r.population.eval_list_file_sha256, r.population.eval_list_set_sha256 = sub.file_sha256, sub.set_sha256
    r.require_complete()
    check_list_binding(r, GEN_LIST, sub)


def test_생성_목록_밖의_id_가_채점_목록에_있으면_거부한다() -> None:
    other = validate_eval_list(canonical_bytes(["a1", "z9"]))
    r = filled()
    r.population.eval_list_file_sha256, r.population.eval_list_set_sha256 = other.file_sha256, other.set_sha256
    with pytest.raises(ListBindingError, match="부분집합이 아니다"):
        check_list_binding(r, GEN_LIST, other)


@pytest.mark.parametrize("side", ["generation", "population"])
def test_등록의_목록_해시가_실제_목록과_다르면_거부한다(side: str) -> None:
    r = filled()
    getattr(r, side).eval_list_file_sha256 = hx("딴파일")
    with pytest.raises(ListBindingError, match="목록 해시가"):
        check_list_binding(r, GEN_LIST, GEN_LIST)


# ---------------------------------------------------------------- 두 지문

def test_채점_개정이_생성_지문을_바꾸지_않는다() -> None:
    r = filled(generation_only=True)
    before_gen, before_score = r.generation_sha256(), r.scoring_sha256()
    r.metric.metric_id = "M1"
    r.ci.n_resamples = 2000
    r.population.exclusion_list_sha256 = hx("제외")
    assert r.generation_sha256() == before_gen, "생성 지문이 채점 개정에 끌려갔다"
    assert r.scoring_sha256() != before_score


def test_생성_개정이_채점_지문을_바꾸지_않는다() -> None:
    r = UnifiedRegistration()
    before = r.scoring_sha256()
    r.generation.prompt_sha256 = hx("프롬프트")
    assert r.scoring_sha256() == before


def test_같은_값이면_지문이_같다() -> None:
    a = UnifiedRegistration(metric=MetricSpec(metric_id="M1", iou_threshold="1/2"))
    b = UnifiedRegistration(metric=MetricSpec(iou_threshold="1/2", metric_id="M1"))
    assert a.scoring_sha256() == b.scoring_sha256()


def test_문턱을_유리수_문자열로_적는_것이_지문에_남는다() -> None:
    """0.5 를 부동소수로 적으면 경계 규칙과 어긋난다 — 지문이 갈려서 드러나야 한다."""
    a = UnifiedRegistration(metric=MetricSpec(iou_threshold="1/2"))
    b = UnifiedRegistration(metric=MetricSpec(iou_threshold="0.5"))
    assert a.scoring_sha256() != b.scoring_sha256()


def test_시드는_채점_쪽이다() -> None:
    """생성 지문에 시드가 들어가면 시드를 더할 때마다 기존 예측이 무효가 된다 (§13-2 가)."""
    r = filled(generation_only=True)
    before = r.generation_sha256()
    r.seeds = {"1": 20260828}
    assert r.generation_sha256() == before
    assert "seeds" in json.dumps(r.scoring_payload(), ensure_ascii=False)


def test_구간_규약_칸이_채점_지문에_들어간다() -> None:
    a, b = filled(), filled()
    b.ci.percentile_method = P.PERCENTILE_METHODS[1]
    assert a.scoring_sha256() != b.scoring_sha256()
    assert a.generation_sha256() == b.generation_sha256()


# ---------------------------------------------------------------- 의존

def test_좌표_규약을_바꾸면_카나리아를_다시_정해야_한다() -> None:
    gaps = UnifiedRegistration().dependency_gaps(["generation.coord_space"])
    assert "canary.echo_criterion" in gaps
    assert "canary.literal_match_tolerance" in gaps


def test_제외_목록을_바꾸면_모집단_상수가_끌려온다() -> None:
    gaps = UnifiedRegistration().dependency_gaps(["population.exclusion_list_sha256"])
    for name in ("population.n_images", "population.n_clusters", "population.class_support",
                 "population.totals_error_table", "baseline.full_population_m1",
                 "population.eval_list_file_sha256"):
        assert name in gaps


def test_모집단이_끌고_가는_목록은_채점_쪽이다() -> None:
    """앞 판은 끝 이름만 적어 생성 쪽 목록인지 채점 쪽 목록인지 몰랐다 — 생성 쪽을 다시 보면 재추론이 된다."""
    gaps = UnifiedRegistration().dependency_gaps(["population.population_id"])
    assert "population.eval_list_file_sha256" in gaps
    assert not any(g.startswith("generation.") for g in gaps)


def test_여러_항목을_바꾸면_합집합이다() -> None:
    r = UnifiedRegistration()
    both = set(r.dependency_gaps(["generation.coord_space", "generation.prompt_sha256"]))
    assert both == set(r.dependency_gaps(["generation.coord_space"])) | \
        set(r.dependency_gaps(["generation.prompt_sha256"]))


def test_의존이_없는_항목은_빈_목록이다() -> None:
    assert UnifiedRegistration().dependency_gaps(["adoption_anchor"]) == []


def test_등록에_없는_이름을_주면_거부한다() -> None:
    """끝 이름만 주거나 오타를 내면 빈 목록이 나와 '다시 볼 것이 없다' 로 읽혔다."""
    with pytest.raises(ValueError, match="등록에 없는 항목"):
        UnifiedRegistration().dependency_gaps(["coord_space"])


def test_의존표의_이름이_전부_실재하는_항목이다() -> None:
    """오타로 적힌 이름은 영원히 안 걸린다. **영역까지 맞댄다** — 앞 판은 끝 이름만 맞대 모호함을 못 잡았다."""
    known = leaf_names()
    for src, targets in DEPENDENCIES.items():
        assert src in known, f"의존표의 출발 항목 {src} 가 등록에 없다"
        for t in targets:
            assert t in known, f"의존표의 대상 {t} 가 등록에 없다"


# ---------------------------------------------------------------- 영수증

def receipt_for(r: UnifiedRegistration, **over) -> Receipt:
    data = {"generation_sha256": r.generation_sha256(),
            "scoring_sha256": r.scoring_sha256(),
            "registered_at": "2026-09-21T10:00:00+09:00",
            "main_commit": "0123456789abcdef0123456789abcdef01234567",
            "kind": r.generation.purpose or "main",
            "registration_path": "docs/registration/unified.json",
            "registration_file_sha256": hashlib.sha256(dump_registration(r)).hexdigest()}
    data.update(over)
    return Receipt(**data)


def test_지금_등록을_가리키면_통과하고_지문을_돌려준다() -> None:
    r = filled()
    rc = receipt_for(r)
    assert require_receipt(r, rc) == r.generation_sha256()
    assert require_receipt(r, rc, side="scoring") == r.scoring_sha256()


def test_등록이_바뀐_뒤의_옛_영수증은_거부한다() -> None:
    r = filled()
    old = receipt_for(r)
    r.generation.prompt_sha256 = hx("바뀐 프롬프트")
    with pytest.raises(ReceiptMissing, match="지문이 지금 등록과 다르다"):
        require_receipt(r, old)


def test_naive_시각은_받지_않는다() -> None:
    r = filled()
    with pytest.raises(ReceiptMissing, match="오프셋"):
        receipt_for(r, registered_at="2026-09-21T10:00:00")


def test_채점_쪽_영수증은_채점_완결까지_본다() -> None:
    r = filled(generation_only=True)
    with pytest.raises(RegistrationIncomplete):
        require_receipt(r, receipt_for(r), side="scoring")


def test_채점_쪽_영수증은_채점_쪽_유효성까지_본다() -> None:
    r = filled()
    r.ci.alpha = 2.0
    with pytest.raises(RegistrationInvalid):
        require_receipt(r, receipt_for(r), side="scoring")


def test_둘을_한꺼번에_묻는_영역은_받지_않는다() -> None:
    """앞 판은 `all` 로 부르면 완결은 둘 다 보고 지문은 채점 쪽만 맞댔다."""
    r = filled()
    with pytest.raises(ValueError, match="두 번 부른다"):
        require_receipt(r, receipt_for(r), side="all")


def test_영수증의_종류와_등록의_목적이_다르면_거부한다() -> None:
    """리허설 영수증으로 본실험 등록을 지나가는 길 — 종류는 접두 문자열이 아니라 필드다."""
    r = filled()
    with pytest.raises(ReceiptMissing, match="종류"):
        require_receipt(r, receipt_for(r, kind="rehearsal"))


@pytest.mark.parametrize(("key", "bad"), [
    ("main_commit", "0" * 39),
    ("main_commit", "rehearsal:" + "0" * 40),
    ("main_commit", "A" * 40),
    ("generation_sha256", "a"),
    ("kind", "probe"),
    ("registration_path", "/abs/path.json"),
    ("registration_path", "docs\\registration.json"),
    ("registration_path", "../outside.json"),
    ("registration_path", "C:/x.json"),
    ("registration_path", ""),
])
def test_영수증의_꼴이_틀리면_만들_수_없다(key: str, bad: str) -> None:
    with pytest.raises(ReceiptMissing, match="꼴이 틀렸다"):
        receipt_for(filled(), **{key: bad})


def test_영수증_파일에_항목이_빠지면_이름을_말한다(tmp_path) -> None:
    r = filled()
    data = {k: v for k, v in vars(receipt_for(r)).items() if k != "kind"}
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReceiptMissing, match="kind"):
        read_receipt(p)


def test_영수증_파일을_읽으면_장치는_없어도_된다(tmp_path) -> None:
    r = filled()
    data = {k: v for k, v in vars(receipt_for(r)).items() if k != "device"}
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    assert read_receipt(p).device == ""


# ---------------------------------------------------------------- 등록 파일

def test_등록_파일은_왕복해도_지문이_같다() -> None:
    r = filled()
    back = load_registration(dump_registration(r))
    assert back.generation_sha256() == r.generation_sha256()
    assert back.scoring_sha256() == r.scoring_sha256()
    assert back.canary.coord_cfg == r.canary.coord_cfg
    assert isinstance(back.metric.class_codes, tuple)


def test_빈_등록도_파일로_적고_읽힌다() -> None:
    r = UnifiedRegistration()
    assert load_registration(dump_registration(r)).missing() == r.missing()


def test_등록_파일은_정규형만_받는다() -> None:
    """같은 바이트면 같은 등록이다 — 손으로 고친 파일(공백·키 순서)은 멈춘다."""
    raw = dump_registration(filled())
    pretty = json.dumps(json.loads(raw), ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    with pytest.raises(RegistrationFileError, match="정규형"):
        load_registration(pretty)
    with pytest.raises(RegistrationFileError, match="CR"):
        load_registration(raw.replace(b"\n", b"\r\n"))


def test_모르는_키나_빠진_키가_있으면_멈춘다() -> None:
    body = json.loads(dump_registration(filled()))
    body["ci"]["percentile_methd"] = "linear"          # 오타 난 칸
    with pytest.raises(RegistrationFileError, match="ci"):
        load_registration(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8") + b"\n")
    body = json.loads(dump_registration(filled()))
    del body["seeds"]
    with pytest.raises(RegistrationFileError, match="빠진 키"):
        load_registration(json.dumps(body).encode("utf-8") + b"\n")


def test_다른_판의_등록_파일은_읽지_않는다() -> None:
    raw = dump_registration(replace(filled(), version="1"))
    with pytest.raises(RegistrationFileError, match="등록 판"):
        load_registration(raw)


def test_영수증이_가리키는_바이트로_등록을_만든다() -> None:
    r = filled()
    raw = dump_registration(r)
    rc = receipt_for(r)
    got, fp = registration_for_receipt(raw, rc, side="scoring")
    assert fp == r.scoring_sha256() and got.scoring_sha256() == fp


def test_작업_트리의_다른_등록으로는_영수증을_지나지_못한다() -> None:
    """영수증은 커밋의 바이트를 가리킨다. 작업 트리에서 고친 등록을 넣으면 지문이 갈린다."""
    r = filled()
    rc = receipt_for(r)
    edited = filled()
    edited.generation.max_new_tokens = 4096
    with pytest.raises(ReceiptMissing):
        registration_for_receipt(dump_registration(edited), rc, side="generation")


# ================================================================ 09-25 판정 반영
#
# I-1 매칭 규약 결속 · I-2 갈린 coord_cfg_hash 의 병기와 에코 관문.


# ---------------------------------------------------------------- I-1 매칭 규약

def test_매칭_규약을_바꾸면_기지답과_대조선이_딸려_온다() -> None:
    """값이 하나뿐인 지금은 안 걸린다. **걸릴 날에 같은 지문으로 통과하는 것**을 막는 자리다."""
    gaps = UnifiedRegistration().dependency_gaps(["metric.matching_rule_id"])
    for name in ("metric.iou_threshold", "canary.known_answer_sha256",
                 "canary.conformance_vector_sha256", "baseline.full_population_m1"):
        assert name in gaps


def test_매칭_규약이_의존표의_출발점이다() -> None:
    assert "metric.matching_rule_id" in DEPENDENCIES


# ---------------------------------------------------------------- I-2 병기 기록

def test_병기는_두_값을_다_담는다() -> None:
    r = cfg()
    assert r.frozen_value in r.accepted and r.current_value in r.accepted


@pytest.mark.parametrize(("field", "value"), [
    ("frozen_value", "없는값"),
    ("current_value", "없는값"),
])
def test_담기지_않은_값은_병기가_아니다(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="accepted 에 없다"):
        cfg(**{field: value})


def test_받아들일_값이_비면_거부한다() -> None:
    with pytest.raises(ValueError, match="비어 있다"):
        cfg(accepted=(), frozen_value="a", current_value="a")


def test_동작_동일성_근거_없이는_병기하지_않는다() -> None:
    """근거 없이 둘 다 받으면 그냥 검사를 끈 것이다."""
    with pytest.raises(ValueError, match="동작 동일성 근거"):
        cfg(equivalence_evidence=())


def test_기록에_네_가지가_다_있다() -> None:
    d = cfg().as_dict()
    assert d["frozen_value"] == "c598d549" and d["frozen_at"] == "2026-09-02"   # 가
    assert d["current_value"] == "0cc62b38"                                     # 나
    assert d["change_scope"] == "comments_only"                                 # 다
    assert len(d["equivalence_evidence"]) == 2                                  # 라
    assert d["frozen_exports_keep_old"] is True
    assert "재고정하지 않고" in d["note"]


# ---------------------------------------------------------------- I-2 에코 관문

def canary(**over) -> CanarySpec:
    """에코 관문은 `coord_fixture_sha256` 을 본다(07번 §31-6). `known_answer_sha256` 은 M1 의 기지답으로 돌아가 관문과 무관하다."""
    kw = {"coord_fixture_sha256": "k" * 64, "known_answer_sha256": "m" * 64, "coord_cfg": cfg()}
    kw.update(over)
    return CanarySpec(**kw)


@pytest.mark.parametrize("observed", ["c598d549", "0cc62b38"])
def test_등록된_값_가운데_하나면_통과한다(observed: str) -> None:
    """동결 export 는 옛 값을 기록한다 — 그것을 거부하면 동결 자산이 통째로 막힌다."""
    check_echo_gate(observed, "k" * 64, canary())


def test_해시만_맞으면_막는다() -> None:
    with pytest.raises(EchoGateFailed, match="골든 픽스처"):
        check_echo_gate("c598d549", "x" * 64, canary())


def test_픽스처만_맞으면_막는다() -> None:
    with pytest.raises(EchoGateFailed, match="coord_cfg_hash"):
        check_echo_gate("deadbeef", "k" * 64, canary())


def test_둘_다_틀리면_사유를_둘_다_낸다() -> None:
    with pytest.raises(EchoGateFailed) as exc:
        check_echo_gate("deadbeef", "x" * 64, canary())
    assert len(exc.value.reasons) == 2


def test_관측값이_없어도_막는다() -> None:
    with pytest.raises(EchoGateFailed):
        check_echo_gate(None, None, canary())


@pytest.mark.parametrize("missing", ["coord_cfg", "coord_fixture_sha256"])
def test_등록이_비어_있으면_관문을_열어_두지_않는다(missing: str) -> None:
    with pytest.raises(ValueError, match="에코 관문을 걸 등록이 없다"):
        check_echo_gate("c598d549", "k" * 64, canary(**{missing: None}))


def test_에코_관문은_M1_기지답_칸을_보지_않는다() -> None:
    """한 칸의 두 뜻을 갈랐다(07번 §31-6). 기지답이 비어도, 골든 지문과 같은 값이어도 관문의 판정은 그대로다."""
    check_echo_gate("c598d549", "k" * 64, canary(known_answer_sha256=None))
    with pytest.raises(EchoGateFailed, match="골든 픽스처"):
        check_echo_gate("c598d549", "m" * 64, canary())


# ---------------------------------------------------------------- 지문 결속

def test_좌표_설정_기록이_채점_지문에_들어간다() -> None:
    a = UnifiedRegistration(canary=canary())
    b = UnifiedRegistration(canary=canary(coord_cfg=cfg(accepted=("c598d549", "0cc62b38", "zz"))))
    assert a.scoring_sha256() != b.scoring_sha256()
    assert a.generation_sha256() == b.generation_sha256(), "생성 쪽은 안 끌려간다"


def test_좌표_설정_기록이_없으면_빈_항목이다() -> None:
    assert "canary.coord_cfg" in UnifiedRegistration().missing()


def test_기록이_채워지면_빈_항목이_아니다() -> None:
    assert "canary.coord_cfg" not in UnifiedRegistration(canary=canary()).missing()


# ================================================================ 판 3 (07번 §31 · §32, 진입점 미니스펙 3판)
#
# 목적 셋 · 목적별 완결 · 진단 규칙의 자리 · 영수증의 파일 해시 · 리허설·진단의 채점 쪽.

def test_판_3_이다() -> None:
    """판 3 의 목적 셋은 판 4 에서도 그대로다. 판 4 는 구간 규약 칸만 바꿨다."""
    assert P.REGISTRATION_VERSION == "4"
    assert P.PURPOSES == ("main", "rehearsal", "frame_diag")
    assert P.SPLIT_OF_PURPOSE["frame_diag"] == "val"
    assert P.VOCAB["generation.list_split"] == ("eval", "val"), "어휘에 같은 값이 두 번 들지 않는다"


@pytest.mark.parametrize("purpose", ["main", "rehearsal", "frame_diag"])
def test_목적마다_가득_찬_등록은_완결이다(purpose: str) -> None:
    filled(purpose=purpose).require_complete(side="generation")


@pytest.mark.parametrize("name", sorted(P.PURPOSE_NULL["main"]))
def test_본실험에서_진단_칸에_값이_있으면_무효다(name: str) -> None:
    """본실험의 학습 행은 칸마다 달라 어댑터 meta 에만 있다. 진단 규칙은 진단 등록의 것이다(§31-5)."""
    r = filled(generation_only=True, purpose="main")
    setattr(r.generation, name.split(".", 1)[1], valid_value(name, P._kinds()[name]))
    with pytest.raises(RegistrationInvalid, match="null 이어야 한다"):
        r.require_complete(side="generation")


def test_본실험은_진단_칸을_빈_항목으로_세지_않는다() -> None:
    gaps = filled(generation_only=True, purpose="main").missing(side="generation")
    assert not set(gaps) & P.PURPOSE_NULL["main"]


@pytest.mark.parametrize("name", ["generation.frame_diag_rules_sha256", "generation.frame_diag_rules_path",
                                  "generation.train_rows_digest", "generation.plan_sha256",
                                  "generation.echo_list_file_sha256", "generation.echo_list_set_sha256"])
def test_진단은_규칙_학습행_계획_에코목록이_필수다(name: str) -> None:
    r = filled(generation_only=True, purpose="frame_diag")
    setattr(r.generation, name.split(".", 1)[1], None)
    assert name in r.missing(side="generation")


@pytest.mark.parametrize("name", ["generation.train_rows_digest", "generation.plan_sha256"])
def test_리허설은_학습행과_계획_해시가_비어도_된다(name: str) -> None:
    r = filled(generation_only=True, purpose="rehearsal")
    setattr(r.generation, name.split(".", 1)[1], None)
    r.require_complete(side="generation")


def test_리허설에서_진단_규칙_칸에_값이_있으면_무효다() -> None:
    r = filled(generation_only=True, purpose="rehearsal")
    r.generation.frame_diag_rules_sha256 = hx("rules")
    r.generation.frame_diag_rules_path = _SPECIAL["generation.frame_diag_rules_path"]
    with pytest.raises(RegistrationInvalid, match="null 이어야 한다"):
        r.require_complete(side="generation")


def test_목적이_비면_조건부_칸도_빈_항목으로_남긴다() -> None:
    """무엇을 채워야 하는지 모르는 채로 빈 항목을 줄이지 않는다."""
    gaps = UnifiedRegistration().missing()
    assert P.PURPOSE_NULL["main"] <= set(gaps)


@pytest.mark.parametrize("bad", ["configs/registration/rules.json", "configs/frame_diag_rules-20261001-1.json",
                                 "configs/registration/frame_diag_rules-2026101-1.json",
                                 "configs/registration/frame_diag_rules-20261001-0.json"])
def test_진단_규칙_파일의_경로는_정한_자리와_이름이다(bad: str) -> None:
    r = filled(generation_only=True, purpose="frame_diag")
    r.generation.frame_diag_rules_path = bad
    with pytest.raises(RegistrationInvalid, match="frame_diag_rules_path"):
        r.require_complete(side="generation")


def test_진단_규칙의_경로와_digest_는_짝이다() -> None:
    r = filled(generation_only=True, purpose="frame_diag")
    r.generation.frame_diag_rules_path = None
    assert any("둘 다 값이거나" in p for p in r.invalid(side="generation"))


def test_진단_규칙은_생성_지문에_든다() -> None:
    """규칙을 바꾸면 생성 지문이 바뀐다 — 채점 등록만 고쳐 같은 묶음을 다시 판정하는 길을 막는다(§31-1)."""
    a = filled(purpose="frame_diag")
    b = filled(purpose="frame_diag")
    b.generation.frame_diag_rules_sha256 = hx("다른 규칙")
    assert a.generation_sha256() != b.generation_sha256()
    assert a.scoring_sha256() == b.scoring_sha256()


@pytest.mark.parametrize("name", ["canary.coord_fixture_sha256", "canary.frame_diag_sha256"])
def test_본실험의_채점_쪽은_골든_지문과_진단_산출물_해시를_요구한다(name: str) -> None:
    r = filled(purpose="main")
    setattr(r.canary, name.split(".", 1)[1], None)
    with pytest.raises(RegistrationIncomplete, match=name.split(".", 1)[1]):
        r.require_complete(side="scoring")


@pytest.mark.parametrize("field", P.FRAME_DEFINING_FIELDS)
def test_프레임을_정하는_칸이_바뀌면_진단_산출물_해시를_다시_본다(field: str) -> None:
    assert "canary.frame_diag_sha256" in UnifiedRegistration().dependency_gaps([f"generation.{field}"])


# ---------------------------------------------------------------- 영수증의 파일 해시

def test_영수증에_파일_해시가_없으면_읽지_않는다(tmp_path) -> None:
    r = filled()
    data = {k: v for k, v in vars(receipt_for(r)).items() if k != "registration_file_sha256"}
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReceiptMissing, match="registration_file_sha256"):
        read_receipt(p)


def test_영수증의_파일_해시는_sha256_꼴이다() -> None:
    with pytest.raises(ReceiptMissing, match="registration_file_sha256"):
        receipt_for(filled(), registration_file_sha256="abc")


def test_진단_영수증을_만들_수_있다() -> None:
    r = filled(purpose="frame_diag")
    assert receipt_for(r).kind == "frame_diag"
    assert require_receipt(r, receipt_for(r)) == r.generation_sha256()


def test_파일_바이트가_영수증과_다르면_등록을_만들지_않는다() -> None:
    """지문을 계산하기 전에 바이트를 본다 — 깨진 바이트도 파일 해시에서 먼저 멈춘다."""
    rc = receipt_for(filled())
    with pytest.raises(ReceiptMissing, match="registration_file_sha256"):
        registration_for_receipt(b"not json\n", rc, side="generation")


def test_영수증의_파일_해시가_다른_파일을_가리키면_멈춘다() -> None:
    r = filled()
    rc = receipt_for(r, registration_file_sha256=hx("다른 파일"))
    with pytest.raises(ReceiptMissing, match="registration_file_sha256"):
        registration_for_receipt(dump_registration(r), rc, side="generation")


# ---------------------------------------------------------------- 리허설·진단의 채점 쪽

def probe_registration(purpose: str = "rehearsal") -> UnifiedRegistration:
    """생성 쪽은 가득 차고 채점 쪽은 에코 관문 두 칸만 찬 등록."""
    r = filled(generation_only=True, purpose=purpose)
    r.canary.coord_cfg = cfg()
    r.canary.coord_fixture_sha256 = hx("golden")
    return r


@pytest.mark.parametrize("purpose", ["rehearsal", "frame_diag"])
def test_리허설과_진단은_에코_관문_두_칸만_요구하고_빈_칸의_이름을_준다(purpose: str) -> None:
    r = probe_registration(purpose)
    gaps = P.require_probe_scoring(r, receipt_for(r))
    assert "ci.percentile_method" in gaps, "구간 규약 칸이 비어도 돈다(§31-8 의 11)"
    assert not set(gaps) & set(P.PROBE_SCORING_REQUIRED)


@pytest.mark.parametrize("name", P.PROBE_SCORING_REQUIRED)
def test_에코_관문_칸이_비면_리허설도_멈춘다(name: str) -> None:
    r = probe_registration()
    setattr(r.canary, name.split(".", 1)[1], None)
    with pytest.raises(RegistrationIncomplete):
        P.require_probe_scoring(r, receipt_for(r))


def test_리허설의_채점_지문도_영수증과_같아야_한다() -> None:
    r = probe_registration()
    rc = receipt_for(r)
    r.canary.coord_fixture_sha256 = hx("바뀐 골든")
    with pytest.raises(ReceiptMissing, match="scoring 지문"):
        P.require_probe_scoring(r, rc)


def test_리허설의_채점_쪽에_틀린_값이_있으면_멈춘다() -> None:
    """빈 칸과 틀린 칸은 다르다 — 채워진 값은 유효해야 한다."""
    r = probe_registration()
    r.ci.alpha = 2.0
    with pytest.raises(RegistrationInvalid):
        P.require_probe_scoring(r, receipt_for(r))


def test_본실험_등록은_리허설_채점_쪽_검사를_받지_않는다() -> None:
    r = filled(purpose="main")
    with pytest.raises(ValueError, match="두 영역"):
        P.require_probe_scoring(r, receipt_for(r))


def test_리허설_채점_쪽_검사도_영수증_종류를_본다() -> None:
    r = probe_registration("frame_diag")
    with pytest.raises(ReceiptMissing, match="종류"):
        P.require_probe_scoring(r, receipt_for(r, kind="rehearsal"))
