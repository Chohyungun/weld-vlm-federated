"""평가 등록 설정 판독기 — `evaluation/eval_registration.py`(1013o 의 4). 후보 yaml(28번)과 코드 상수를 맞댄다."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import eval_registration as ER  # noqa: E402

CAND = ROOT / ER.CANDIDATE_REL
BASE = ROOT / ER.BASE_REL
DECIDED_MAX_NEW_TOKENS = 512
"""평가 생성 한도 L — 의사결정로그 2026-10-13 본실험 등록 조건 ②. 데이터 쪽이 base.yaml `uni_max_new_tokens` 에 적는 값이다."""
NULL_PROBLEM = "generation.max_new_tokens 를 base.yaml 과 맞댈 수 없다 — fixed_before_main_runs.uni_max_new_tokens 가 비어 있다"


def _doc():
    return ER.load_eval_registration(CAND.read_bytes())


def _base():
    return yaml.safe_load(BASE.read_text(encoding="utf-8"))


def _base_value():
    return _base()["fixed_before_main_runs"].get("uni_max_new_tokens")


def test_후보_yaml_은_코드_상수와_맞고_결정값_512_와_맞는다() -> None:
    """후보의 값은 전부 코드 상수와 같고 생성 한도는 결정값 512 다. 실제 base.yaml 이 512 면(데이터 쪽 반영 뒤) 문제가 없다.
    이 트랙 브랜치가 그 반영을 아직 받지 않아 null 이면 남는 문제는 그 칸 하나다 — 다른 값이면 어긋남으로 잡힌다(31번 I-1)."""
    assert _doc()["generation"]["max_new_tokens"] == DECIDED_MAX_NEW_TOKENS
    assert ER.registration_problems(_doc()) == []
    probs = ER.registration_problems(_doc(), base_doc=_base())
    if _base_value() == DECIDED_MAX_NEW_TOKENS:
        assert probs == []
    else:
        assert _base_value() is None, f"base.yaml uni_max_new_tokens 가 결정값 {DECIDED_MAX_NEW_TOKENS} 도 null 도 아니다"
        assert probs == [NULL_PROBLEM]


def test_base_yaml_이_채워지면_맞댄다() -> None:
    base = _base()
    base["fixed_before_main_runs"]["uni_max_new_tokens"] = _doc()["generation"]["max_new_tokens"]
    assert ER.registration_problems(_doc(), base_doc=base) == []
    base["fixed_before_main_runs"]["uni_max_new_tokens"] = 128
    assert any("uni_max_new_tokens 128" in p for p in ER.registration_problems(_doc(), base_doc=base))


@pytest.mark.parametrize(("path", "value", "needle"), [
    ("representative.metric_id", "m1_other", "metric_id"),
    ("representative.sensitivity_variant", "fp_per_class_normal_failed_only", "sensitivity_variant"),
    ("representative.coupling_rule", "coupled", "coupling_rule"),
    ("population.sample.seed", "eval_sample_v2|x|y", "seed"),
    ("population.sample.weighting", "none", "weighting"),
    ("population.sample.side_file_header", "image_id,stratum", "side_file_header"),
    ("population.sample.main_size", 2500, "main_size"),
    ("population.sample.full_addendum_d_min", 0, "full_addendum_d_min"),
])
def test_어긋난_값은_칸_이름으로_잡힌다(path, value, needle) -> None:
    doc = copy.deepcopy(_doc())
    cur = doc
    keys = path.split(".")
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value
    probs = ER.registration_problems(doc)
    assert any(needle in p for p in probs), probs


def test_봉인_키_집합은_표본_추출기의_크기와_같아야_한다() -> None:
    doc = copy.deepcopy(_doc())
    del doc["population"]["sample"]["sealed"]["2000"]
    assert any("sealed" in p for p in ER.registration_problems(doc))
    doc = copy.deepcopy(_doc())
    doc["population"]["sample"]["sealed"]["3000"]["file_sha256"] = "zz"
    assert any("sealed[3000]" in p for p in ER.registration_problems(doc))


def test_등록_객체의_모집단은_봉인한_주_표본과_같아야_한다() -> None:
    from evaluation import sample_weights as SW
    from evaluation.prereg_unified import UnifiedRegistration

    doc = _doc()
    sealed = doc["population"]["sample"]["sealed"][str(doc["population"]["sample"]["main_size"])]
    r = UnifiedRegistration()
    r.population.weighting = SW.WEIGHTING_SAMPLE
    r.population.eval_list_file_sha256, r.population.eval_list_set_sha256 = sealed["file_sha256"], sealed["set_sha256"]
    r.population.n_images = sealed["images"]
    assert ER.check_population_binding(r, doc) == []
    r.population.n_images = sealed["images"] + 1
    assert any("n_images" in p for p in ER.check_population_binding(r, doc))
    r.population.weighting = SW.WEIGHTING_NONE
    assert ER.check_population_binding(r, doc) == []            # 전량이면 봉인 표본과 묶지 않는다


def test_등록_어휘에_층화_가중이_있다() -> None:
    from evaluation import sample_weights as SW
    from evaluation.prereg_unified import VOCAB

    assert VOCAB["population.weighting"] == ("none", SW.WEIGHTING_SAMPLE)


def test_꼴이_틀리거나_없는_파일은_문제다(tmp_path, capsys) -> None:
    with pytest.raises(ER.EvalRegistrationError):
        ER.load_eval_registration(b"kind: other\n")
    assert ER.main(["--file", str(tmp_path / "없음.yaml")]) == 1
    assert "없다" in capsys.readouterr().out
    # 실제 base.yaml — 결정값 512 면 종료 0, 반영 전(null)이면 그 칸 하나로 종료 1
    assert ER.main(["--file", str(CAND), "--base", str(BASE)]) == (0 if _base_value() == DECIDED_MAX_NEW_TOKENS else 1)
    base = tmp_path / "base.yaml"
    b = _base()
    b["fixed_before_main_runs"]["uni_max_new_tokens"] = _doc()["generation"]["max_new_tokens"]
    base.write_text(yaml.safe_dump(b, allow_unicode=True), encoding="utf-8")
    assert ER.main(["--file", str(CAND), "--base", str(base)]) == 0
