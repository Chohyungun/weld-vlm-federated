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
    """1차 후보는 보존한 10월 설정에 묶는다. 현재 2차 설정과 섞으면 거부한다."""
    from tests.registration_history import historical_base

    old = yaml.safe_load(historical_base().read_bytes())
    assert _doc()["generation"]["max_new_tokens"] == DECIDED_MAX_NEW_TOKENS
    assert old["fixed_before_main_runs"]["uni_max_new_tokens"] == DECIDED_MAX_NEW_TOKENS
    assert ER.registration_problems(_doc()) == []
    assert ER.registration_problems(_doc(), base_doc=old) == []
    bad = ER.registration_problems(_doc(), base_doc=_base())
    assert any("max_new_tokens" in p for p in bad), bad


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


# ---------------------------------------------------------------- base 설정 — CLI 는 생략하지 않는다(외부 검토 137-2 M-04)

def test_CLI_는_없는_base_를_대조_생략으로_넘기지_않는다(tmp_path, capsys) -> None:
    """앞 판은 없는 `--base` 를 None 으로 넘겨 대조 없이 종료 0 이 될 수 있었다."""
    assert ER.registration_problems(_doc()) == []                  # 후보 자체는 맞다 — 남는 것은 base 부재뿐
    assert ER.main(["--file", str(CAND), "--base", str(tmp_path / "없는_base.yaml")]) == 1
    assert "base 설정이 없다" in capsys.readouterr().out


@pytest.mark.parametrize(("body", "needle"), [
    ("- a\n- b\n", "객체(사상)가 아니다"),
    ("그냥 글\n", "객체(사상)가 아니다"),
    ("", "객체(사상)가 아니다"),
    ("a: [\n", "읽지 못한다"),
    ("other: 1\n", "fixed_before_main_runs"),
    ("fixed_before_main_runs: 3\n", "fixed_before_main_runs"),
])
def test_CLI_는_객체가_아닌_base_를_거부한다(tmp_path, capsys, body, needle) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(body, encoding="utf-8")
    assert ER.main(["--file", str(CAND), "--base", str(base)]) == 1
    assert needle in capsys.readouterr().out


def test_라이브러리의_base_doc_은_생략과_잘못된_값을_가른다() -> None:
    assert ER.registration_problems(_doc(), base_doc=None) == []    # 부르는 쪽이 고른 생략
    assert any("객체(사상)가 아니다" in p for p in ER.registration_problems(_doc(), base_doc=["a"]))
    assert any("fixed_before_main_runs" in p for p in ER.registration_problems(_doc(), base_doc={}))
    with pytest.raises(ER.EvalRegistrationError):
        ER.load_base(b"- a\n")
    assert ER.load_base(BASE.read_bytes())["fixed_before_main_runs"] is not None


# ---------------------------------------------------------------- 엄격 표본 모드의 결속(35번 §1)

def _bound_registration():
    """후보 yaml 의 봉인 주 표본 · 대표 지표와 맞는 등록 객체 — 칸 하나씩 흔들어 본다."""
    from evaluation import sample_weights as SW
    from evaluation.prereg_unified import UnifiedRegistration

    doc = _doc()
    sealed = doc["population"]["sample"]["sealed"][str(doc["population"]["sample"]["main_size"])]
    r = UnifiedRegistration()
    p, m = r.population, r.metric
    p.weighting, p.cluster_unit, r.ci.cluster_unit = SW.WEIGHTING_SAMPLE, "group", "group"
    p.eval_list_file_sha256, p.eval_list_set_sha256 = sealed["file_sha256"], sealed["set_sha256"]
    p.n_images, p.n_clusters = sealed["images"], sealed["groups"]
    rep = doc["representative"]
    m.metric_id, m.iou_threshold, m.coupling_rule = rep["metric_id"], rep["iou_threshold"], rep["coupling_rule"]
    m.sensitivity_variant_id, m.macro_support_rule = rep["sensitivity_variant"], rep["macro_support_rule"]
    m.class_codes = tuple(rep["class_codes"])
    r.seeds = {str(s): 20260827 + s for s in doc["seeds"]}
    r.generation.max_new_tokens = doc["generation"]["max_new_tokens"]
    return r, doc


def test_표본_결속은_봉인_묶음_수_시드_대표_지표_칸까지_맞댄다() -> None:
    r, doc = _bound_registration()
    assert ER.strict_binding_problems(r, doc) == []


@pytest.mark.parametrize(("edit", "needle"), [
    (lambda r: setattr(r.population, "n_clusters", 299), "population.n_clusters"),
    (lambda r: setattr(r.population, "cluster_unit", "component"), "population.cluster_unit"),
    (lambda r: setattr(r.ci, "cluster_unit", "component"), "ci.cluster_unit"),
    (lambda r: setattr(r, "seeds", {"1": 5, "2": 6}), "seeds"),
    (lambda r: setattr(r.metric, "metric_id", "m1_other"), "metric.metric_id"),
    (lambda r: setattr(r.metric, "iou_threshold", "1/3"), "metric.iou_threshold"),
    (lambda r: setattr(r.metric, "sensitivity_variant_id", "fp_per_class_normal_failed_only"), "metric.sensitivity_variant_id"),
    (lambda r: setattr(r.metric, "coupling_rule", "coupled"), "metric.coupling_rule"),
    (lambda r: setattr(r.metric, "macro_support_rule", "fixed"), "metric.macro_support_rule"),
    (lambda r: setattr(r.metric, "class_codes", ("100", "2011", "301")), "metric.class_codes"),
    (lambda r: setattr(r.population, "n_images", 3081), "population.n_images"),
    (lambda r: setattr(r.population, "weighting", "none"), "population.weighting"),
    (lambda r: setattr(r.generation, "max_new_tokens", 256), "generation.max_new_tokens"),
])
def test_표본_결속이_어긋나면_칸_이름으로_잡힌다(edit, needle) -> None:
    r, doc = _bound_registration()
    edit(r)
    probs = ER.strict_binding_problems(r, doc)
    assert any(needle in p for p in probs), probs


def test_재표집_단위는_코드가_실제로_쓰는_단위여야_한다() -> None:
    """설정과 등록이 서로 같아도(둘 다 component) 재표집은 묶음(group)을 쓴다 — 두 파일의 일치만으로 통과시키지 않는다."""
    r, doc = _bound_registration()
    doc = copy.deepcopy(doc)
    doc["population"]["sample"]["cluster_unit"] = "component"
    r.population.cluster_unit = r.ci.cluster_unit = "component"
    assert any("재표집이 실제로 쓰는 단위" in p for p in ER.strict_binding_problems(r, doc))


# ---------------------------------------------------------------- 크기 1000 봉인(본실험 평가 표본, 설계 결정 기록 2026-10-17)
SEALED_1000 = ROOT / "docs/dev_log/2026-10-16-본실험실행/eval_lists_D/eval_sample_v1_1000"
MANIFEST = ROOT / "data/interim/manifest_v1"


def test_후보는_본_설정과_같은_바이트이고_크기_1000_을_주_크기로_봉인한다() -> None:
    raw = CAND.read_bytes()
    assert raw == (ROOT / ER.CONFIG_REL).read_bytes()
    sample = _doc()["population"]["sample"]
    assert sample["main_size"] == 1000 and set(sample["sealed"]) == {"1000", "2000", "3000", "4000"}
    assert (sample["sealed"]["1000"]["images"], sample["sealed"]["1000"]["groups"]) == (1059, 101)


@pytest.mark.parametrize("edit", ["extra_key", "main_size_unsealed", "drop_1000"])
def test_봉인_밖_크기는_키도_주_크기도_되지_않는다(edit) -> None:
    doc = copy.deepcopy(_doc())
    sample = doc["population"]["sample"]
    if edit == "extra_key":
        sample["sealed"]["1500"] = dict(sample["sealed"]["1000"])
    elif edit == "main_size_unsealed":
        sample["main_size"] = 1500
    else:
        del sample["sealed"]["1000"]
    assert ER.registration_problems(doc), edit


@pytest.mark.skipif(not (MANIFEST / "manifest.csv").is_file(), reason="동결 매니페스트가 이 트리에 없다")
def test_크기_1000_봉인은_추출기의_같은_시드_같은_규칙이고_다른_크기의_접두다() -> None:
    from data import eval_sample as E

    res = E.sample_eval(E.read_manifest_columns(MANIFEST / "manifest.csv"), [1000, *E.SIZES])
    sealed = _doc()["population"]["sample"]["sealed"]
    for size, smp in res["samples"].items():
        got = {"images": smp["images"], "groups": smp["groups"],
               "file_sha256": __import__("hashlib").sha256(E.list_bytes(smp["ids"])).hexdigest(),
               "side_sha256": __import__("hashlib").sha256(E.side_bytes(smp["side_rows"])).hexdigest()}
        assert got == {k: sealed[str(size)][k] for k in got}, size
    assert set(res["samples"][1000]["ids"]) <= set(res["samples"][2000]["ids"])
    for name in ("eval_sample_1000.txt", "eval_sample_1000_strata.csv"):             # 봉인 파일은 추출기가 낸 바이트 그대로
        if (SEALED_1000 / name).is_file():
            raw = (SEALED_1000 / name).read_bytes()
            want = E.list_bytes(res["samples"][1000]["ids"]) if name.endswith(".txt") else E.side_bytes(res["samples"][1000]["side_rows"])
            assert raw == want, name


@pytest.mark.skipif(not ((MANIFEST / "manifest.csv").is_file() and (SEALED_1000 / "eval_sample_1000.txt").is_file()),
                    reason="동결 매니페스트 · 크기 1000 봉인 파일이 이 트리에 없다")
def test_본실험_표본_계약은_1059_장_목록과_곁_파일로_서고_봉인_밖_목록은_거부한다() -> None:
    """채점 쪽 `check_main_sample` — 주 크기 1000 의 봉인으로 1,059 장 목록 · 곁 파일(`G_h` · `n_h` 는 크기 1000 의 배분)이 선다.
    한 묶음을 뺀 목록 · 크기 2000 의 목록은 봉인한 주 표본이 아니라 거부한다."""
    from evaluation.entry_gate import EntryRejected, check_main_sample, read_manifest_meta
    from evaluation.eval_list import canonical_bytes, validate_eval_list
    from evaluation.sample_weights import read_side

    doc = _doc()
    sealed = doc["population"]["sample"]["sealed"][str(doc["population"]["sample"]["main_size"])]
    list_raw = (SEALED_1000 / "eval_sample_1000.txt").read_bytes()
    side_raw = (SEALED_1000 / "eval_sample_1000_strata.csv").read_bytes()
    gen = validate_eval_list(list_raw, where="eval_sample_1000")
    assert (gen.file_sha256, gen.set_sha256, len(gen)) == (sealed["file_sha256"], sealed["set_sha256"], 1059)
    assert __import__("hashlib").sha256(side_raw).hexdigest() == sealed["side_sha256"]
    meta = read_manifest_meta(MANIFEST)
    check_main_sample(gen, gen, meta, sealed=sealed, side=read_side(side_raw, list_raw))
    first = meta.group_of[gen.ids[0]]
    cut = validate_eval_list(canonical_bytes([i for i in gen.ids if meta.group_of[i] != first]), where="cut")
    with pytest.raises(EntryRejected) as e1:
        check_main_sample(cut, cut, meta, sealed=sealed, side=read_side(side_raw, list_raw))
    assert e1.value.code == "MAIN_SAMPLE_NOT_SEALED"
    from data import eval_sample as E

    ids2000 = E.sample_eval(E.read_manifest_columns(MANIFEST / "manifest.csv"), [2000])["samples"][2000]["ids"]
    g2000 = validate_eval_list(E.list_bytes(ids2000), where="eval_sample_2000")
    with pytest.raises(EntryRejected) as e2:
        check_main_sample(g2000, g2000, meta, sealed=sealed, side=read_side(side_raw, list_raw))
    assert e2.value.code == "MAIN_SAMPLE_NOT_SEALED"


# ---------------------------------------------------------------- 판 6 연결 2 판 — 결정적 곁 파일 · 공통 판독(§2 · R4 · R6 공통 · R7 · R11 출처)

from dataclasses import replace

from evaluation import narrative_registration as NR
from evaluation import sample_weights as SW
from evaluation.prereg_unified import dump_registration
from tests import v6_link_world as VW


def _load(w, **kw):
    return ER.load_bound_eval(repo=w.root, anchor_rev=w.head, registration_rel=w.stem_rel, registration=w.reg, **kw)


def _refused(fn, *args, **kw) -> str:
    with pytest.raises(ER.EvalBindingRefused) as exc:
        fn(*args, **kw)
    return exc.value.code


def test_v6_link_bound_eval_reads_anchor_blob(tmp_path: Path) -> None:
    """네 소비자의 공통 판독 — anchor 커밋의 결정적 곁 파일 바이트만 읽고, 넘긴 등록 객체도 anchor 의 등록 바이트와 맞댄다."""
    w = VW.build(tmp_path)
    b = _load(w)
    assert (b.relative_path, b.raw, b.sha256, b.anchor_rev, b.profile_id) == (
        ER.eval_sidecar_rel(w.stem_rel), w.eval_raw, VW.sha(w.eval_raw), w.head, NR.PROFILE_REGIME2)
    assert b.doc["auxiliary"][-5:] == list(ER.AUXILIARY_V6) and b.doc["generation"]["max_new_tokens"] == 1280
    assert b.doc["generation"]["registration_version"] == "6" and b.doc["population"]["sample"]["main_size"] == 1000
    (w.root / b.relative_path).write_bytes(b"kind: eval_registration\n")                 # 작업 트리를 바꿔도 anchor 를 읽는다
    assert _load(w).raw == w.eval_raw
    other = replace(w.reg, planned_seed_count=2)
    assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev=w.head, registration_rel=w.stem_rel, registration=other) == "eval_binding"
    # 등록의 결속 칸이 설정과 다르면(시드 · 한도 · 표본) eval_binding — 등록 바이트도 anchor 에 함께 있어야 한다
    for edit, _ in ((lambda r: replace(r, seeds={"1": 20260828, "2": 20260829}), "seeds"),
                    (lambda r: replace(r, generation=replace(r.generation, max_new_tokens=512)), "max_new_tokens"),
                    (lambda r: replace(r, population=replace(r.population, n_images=1058)), "n_images")):
        reg2 = edit(w.reg)
        head = VW.commit_files(w.root, {w.stem_rel: dump_registration(reg2)}, "결속 어긋남")
        assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev=head, registration_rel=w.stem_rel, registration=reg2) == "eval_binding"


def test_v6_link_sidecar_required(tmp_path: Path) -> None:
    """R4 — 새 체제 anchor 에 결정적 곁 파일이 없으면 `eval_sidecar_missing`. HEAD 의 후보 · 작업 트리의 곁 파일 · anchor 의 1 차 설정이
    있어도 구판 fallback 0."""
    w = VW.build(tmp_path, with_sidecar=False)
    cand_rel = "configs/registration/candidates/main-20261028-1.eval.yaml"
    VW.commit_files(w.root, {cand_rel: w.eval_raw}, "후보")                              # HEAD 에는 후보
    (w.root / ER.eval_sidecar_rel(w.stem_rel)).write_bytes(w.eval_raw)                    # 작업 트리에만(미추적)
    assert (w.root / ER.CONFIG_REL).is_file()
    assert _refused(_load, w) == "eval_sidecar_missing"
    assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev="HEAD", registration_rel=w.stem_rel,
                    registration=w.reg) == "eval_sidecar_missing"


def test_v6_link_closed_path_and_sources(tmp_path: Path) -> None:
    """R7 — 절대 경로 · 상위 이동 · 역슬래시 · 다른 디렉터리 · 잘못된 줄기 · 날짜 · n 은 `eval_path`; 출처 운반 파일의 모르는 키 · 역할 ·
    순서 · 규칙 · 프로필 · 결정 · 공개 칸의 변경은 `v6_sources_mismatch`. 임의 파일 탐색 · 설정 override 는 없다."""
    for bad in ("E:/x/configs/registration/main-20261029-1.json", "/configs/registration/main-20261029-1.json",
                "configs/registration/../registration/main-20261029-1.json", "configs\\registration\\main-20261029-1.json",
                "configs/registration/candidates/main-20261029-1.json", "configs/registration/main-20261029-1.eval.yaml",
                "configs/registration/main-20261029-1.receipt.json", "configs/registration/main-20261029-1.train_config.json",
                "configs/registration/frame_diag-20261029-1.json", "configs/registration/main-20261332-1.json",
                "configs/registration/main-20261029-0.json", "configs/registration/main-20261029-01.json",
                "configs/registration/main-2026102-1.json", "main-20261029-1.json", ""):
        with pytest.raises(ER.EvalPathRefused):
            ER.eval_sidecar_rel(bad)
        with pytest.raises(ER.EvalPathRefused):
            ER.v6_sources_rel(bad)
    assert ER.eval_sidecar_rel("configs/registration/main-20261029-1.json") == "configs/registration/main-20261029-1.eval.yaml"
    assert ER.v6_sources_rel("configs/registration/main-20261029-1.json") == "configs/registration/main-20261029-1.v6_sources.json"
    assert _refused(ER.load_bound_eval, repo=tmp_path, anchor_rev="HEAD",
                    registration_rel="configs/registration/candidates/main-20261029-1.json", registration=None) == "eval_path"
    w = VW.build(tmp_path)
    doc = VW.sources_doc(w.files[ER.v6_sources_rel(w.stem_rel)])
    assert set(doc) == NR.V6_SOURCES_KEYS and [r["role"] for r in doc["artifacts"]] == [r.role for r in NR.V6_EVIDENCE_V4]
    variants = {
        "unknown_key": {**doc, "approved_at": "2026-10-28T03:41:33+09:00"},
        "missing_key": {k: v for k, v in doc.items() if k != "disclosure"},
        "unknown_role": {**doc, "artifacts": [*doc["artifacts"], {"role": "extra", "path": "configs/base.yaml",
                                                                 "sha256": "0" * 64, "hash_rule": "raw_sha256"}]},
        "role_swapped": {**doc, "artifacts": [doc["artifacts"][1], doc["artifacts"][0], *doc["artifacts"][2:]]},
        "rule_changed": {**doc, "artifacts": [{**doc["artifacts"][0], "hash_rule": "lf_sha256"}, *doc["artifacts"][1:]]},
        "path_changed": {**doc, "artifacts": [{**doc["artifacts"][0], "path": "evaluation/vectors/narrative_v4_list_probes.v1.json"},
                                              *doc["artifacts"][1:]]},
        "profile_changed": {**doc, "profile_id": NR.PROFILE_LEGACY},
        "decision_changed": {**doc, "decision": {**doc["decision"], "commit": "0" * 40}},
        "disclosure_changed": {**doc, "disclosure": {"comparison_round": 1, "sample_previously_observed": False}},
        "registration_sha": {**doc, "registration_sha256": "0" * 64},
    }
    for name, v in variants.items():
        head = VW.commit_files(w.root, {ER.v6_sources_rel(w.stem_rel): NR.dump_v6_sources(v)}, name)
        assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev=head, registration_rel=w.stem_rel,
                        registration=w.reg) == "v6_sources_mismatch", name
    head = VW.commit_files(w.root, {ER.v6_sources_rel(w.stem_rel): NR.dump_v6_sources(doc).replace(b"\n", b"\r\n")}, "정규형 밖")
    assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev=head, registration_rel=w.stem_rel,
                    registration=w.reg) == "v6_sources_mismatch"


def test_v6_link_eval_bytes_and_sources_binding(tmp_path: Path) -> None:
    """R6 의 공통 함수 · R11 의 출처 결속 — 곁 파일 바이트가 출처의 eval_sha256 과 다르면 `eval_bytes`, 출처 파일이 없으면
    `v6_sources_missing`, 13b · 기대 변경표 v3 행이 빠지거나 anchor 의 원본이 다르면 `v6_sources_mismatch`."""
    w = VW.build(tmp_path)
    good = w.files[ER.v6_sources_rel(w.stem_rel)]
    altered = w.eval_raw + "# 주석 한 줄 — YAML 값은 같다\n".encode()
    head = VW.commit_files(w.root, {ER.eval_sidecar_rel(w.stem_rel): altered}, "곁 파일 바이트만 다름")
    assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev=head, registration_rel=w.stem_rel, registration=w.reg) == "eval_bytes"
    head = VW.commit_files(w.root, {ER.eval_sidecar_rel(w.stem_rel): w.eval_raw, ER.v6_sources_rel(w.stem_rel): None}, "출처 없음")
    assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev=head, registration_rel=w.stem_rel, registration=w.reg) == "v6_sources_missing"
    doc = VW.sources_doc(good)
    for role in ("regression_receipt", "expectation_changes"):
        v = {**doc, "artifacts": [r for r in doc["artifacts"] if r["role"] != role]}
        head = VW.commit_files(w.root, {ER.v6_sources_rel(w.stem_rel): NR.dump_v6_sources(v)}, role)
        assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev=head, registration_rel=w.stem_rel,
                        registration=w.reg) == "v6_sources_mismatch", role
    reg_row = NR.V6_EVIDENCE_V4[2]
    tampered = (w.root / reg_row.path).read_bytes() + b" "
    head = VW.commit_files(w.root, {ER.v6_sources_rel(w.stem_rel): good, reg_row.path: tampered}, "원본 변조")
    assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev=head, registration_rel=w.stem_rel, registration=w.reg) == "v6_sources_mismatch"
    with pytest.raises(NR.ProfileRefused) as exc:                      # 두 구현 — 작성기도 변조된 원본으로 쓰지 않는다
        NR.build_v6_sources(registration_raw=w.reg_raw, eval_raw=w.eval_raw, repo=w.root)
    assert exc.value.code == "v6_sources_mismatch"
    (w.root / reg_row.path).unlink()
    with pytest.raises(NR.ProfileRefused) as exc2:
        NR.build_v6_sources(registration_raw=w.reg_raw, eval_raw=w.eval_raw, repo=w.root)
    assert exc2.value.code == "v6_sources_missing"
    head = VW.commit_files(w.root, {reg_row.path: tampered[:-1]}, "복구")
    assert _load(w) is not None and ER.load_bound_eval(repo=w.root, anchor_rev=head, registration_rel=w.stem_rel,
                                                        registration=w.reg).sha256 == VW.sha(w.eval_raw)


def test_v6_link_derivation_from_first_edition_bytes(tmp_path: Path) -> None:
    """2 차 설정은 anchor 의 1 차 설정 바이트(보존 지문)에서 도출된 것이어야 한다 — 1 차 바이트가 다르면 `eval_bytes`, 도출 값과 다르면
    `eval_binding`, 보조 이름이 다르면 `aux_names`."""
    w = VW.build(tmp_path)
    first = (w.root / ER.CONFIG_REL).read_bytes()
    assert VW.sha(first) == ER.SOURCE_EVAL_SHA256_V1
    head = VW.commit_files(w.root, {ER.CONFIG_REL: first + b"# comment\n"}, "1 차 바이트 변경")
    assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev=head, registration_rel=w.stem_rel, registration=w.reg) == "eval_bytes"
    VW.commit_files(w.root, {ER.CONFIG_REL: first}, "복구")
    doc = dict(w.doc)
    doc["generation"] = {**doc["generation"], "pairs_digest": "0" * 64}
    head = VW.commit_files(w.root, {ER.eval_sidecar_rel(w.stem_rel): VW.eval_bytes(doc),
                                    ER.v6_sources_rel(w.stem_rel): NR.build_v6_sources(registration_raw=w.reg_raw,
                                                                                       eval_raw=VW.eval_bytes(doc), repo=w.root)},
                           "도출 밖 값")
    assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev=head, registration_rel=w.stem_rel, registration=w.reg) == "eval_binding"
    doc = dict(w.doc)
    doc["auxiliary"] = [*w.doc["auxiliary"][:6], *reversed(w.doc["auxiliary"][6:])]
    head = VW.commit_files(w.root, {ER.eval_sidecar_rel(w.stem_rel): VW.eval_bytes(doc),
                                    ER.v6_sources_rel(w.stem_rel): NR.build_v6_sources(registration_raw=w.reg_raw,
                                                                                       eval_raw=VW.eval_bytes(doc), repo=w.root)},
                           "보조 순서")
    assert _refused(ER.load_bound_eval, repo=w.root, anchor_rev=head, registration_rel=w.stem_rel, registration=w.reg) == "aux_names"


def test_v6_link_legacy_paths_preserved(tmp_path: Path) -> None:
    """판 5 등록과 보존 기준 커밋의 역사적 판 6 은 옛 `CONFIG_REL` 경로 그대로다. 옛 등록에 새 설정 값을 맞추면 `eval_binding`, 보존
    기준이 아닌 판 6 에 옛 핀을 붙이면 곁 파일 없이 서지 않는다(`profile_mismatch`)."""
    VW.copy_pinned(tmp_path)
    base = (tmp_path / ER.BASE_REL).read_bytes()                       # 1 차 anchor 의 base 는 한도 512 였다 — 그 시점의 값으로
    assert base.count(b"uni_max_new_tokens: 1280") == 1
    (tmp_path / ER.BASE_REL).write_bytes(base.replace(b"uni_max_new_tokens: 1280", b"uni_max_new_tokens: 512"))
    VW.commit_files(tmp_path, {}, "코드 · 설정")
    doc1 = ER.load_eval_registration((tmp_path / ER.CONFIG_REL).read_bytes())
    r5 = VW.legacy_registration(tmp_path, doc1, version6=False)
    rel5 = "configs/registration/main-20261017-9.json"
    head = VW.commit_files(tmp_path, {rel5: dump_registration(r5)}, "판 5")
    b = ER.load_bound_eval(repo=tmp_path, anchor_rev=head, registration_rel=rel5, registration=r5)
    assert (b.relative_path, b.profile_id, b.doc["generation"]["max_new_tokens"], b.sha256) == (
        ER.CONFIG_REL, None, 512, ER.SOURCE_EVAL_SHA256_V1)
    r5b = replace(r5, generation=replace(r5.generation, max_new_tokens=1280))                # 옛 등록에 새 값
    head = VW.commit_files(tmp_path, {rel5: dump_registration(r5b)}, "옛 등록 새 값")
    assert _refused(ER.load_bound_eval, repo=tmp_path, anchor_rev=head, registration_rel=rel5, registration=r5b) == "eval_binding"
    r6 = VW.legacy_registration(tmp_path, doc1)
    rel6 = "configs/registration/main-20261018-9.json"
    head6 = VW.commit_files(tmp_path, {rel6: dump_registration(r6)}, "역사적 판 6")
    b6 = ER.load_bound_eval(repo=tmp_path, anchor_rev=head6, registration_rel=rel6, registration=r6, history_rev=head6)
    assert (b6.relative_path, b6.profile_id) == (ER.CONFIG_REL, NR.PROFILE_LEGACY)
    assert _refused(ER.load_bound_eval, repo=tmp_path, anchor_rev=head6, registration_rel=rel6, registration=r6) == "profile_mismatch"
    r6b = replace(r6, adoption_anchor="reg-2026-10-02")                                   # 보존 기준과 바이트가 다른 판 6
    assert dump_registration(r6b) != dump_registration(r6)
    head7 = VW.commit_files(tmp_path, {rel6: dump_registration(r6b)}, "바이트가 다른 판 6")
    assert _refused(ER.load_bound_eval, repo=tmp_path, anchor_rev=head7, registration_rel=rel6,
                    registration=r6b, history_rev=head6) == "profile_mismatch"
    assert r6.population.weighting == SW.WEIGHTING_SAMPLE
