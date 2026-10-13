"""본실험 주 분석의 전량 목록 결속 — 07번 §40.

본실험(`purpose=main`)의 생성 목록과 주채점 목록은 **같은 파일**이고, 그 집합은 검증한 동결 매니페스트의 `split == eval`
id 전체다. 리허설 · 진단의 부분집합 채점은 그대로 받는다. 합성 매니페스트만 쓴다 — 실물 평가 자산을 열지 않는다.

계약 시험의 여섯 경우: 미등록 · 한 장 빠진 목록 · 같은 집합의 다른 순서 · 생성/채점 한 장 차이 · 지문 불일치 · 정상.
등록한 장 수(`population.n_images`)와 목록 길이의 대조는 아직 걸지 않았다 — 07번 §40-1 의 3.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from evaluation import entry_gate as EG
from evaluation import prereg_unified as P
from evaluation.entry_gate import EntryRejected, ManifestMeta
from evaluation.eval_list import canonical_bytes, validate_eval_list
from scripts.probe import score_unified as SU

EVAL = ("e01", "e02", "e03", "e04", "e05")
VAL = ("v01", "v02", "v03")
TRAIN = ("t01",)


def meta() -> ManifestMeta:
    split = {**{i: "eval" for i in EVAL}, **{i: "val" for i in VAL}, **{i: "train" for i in TRAIN}}
    return ManifestMeta(split_of=MappingProxyType(split), sha256_of=MappingProxyType({i: "0" * 64 for i in split}),
                        group_of=MappingProxyType({i: f"g_{i}" for i in split}))


def elist(ids):
    return validate_eval_list(canonical_bytes(list(ids)))


def ctx_for(tmp: Path, *, gen_ids, pop_ids, reg_gen=None, reg_pop=None, purpose="main", registration="strict",
            n_images=None, echo_ids=VAL):
    """`_read_lists` 가 읽는 것만 가진 문맥. `reg_gen` · `reg_pop` 은 등록에 적은 목록(없으면 파일과 같은 것)."""
    paths = {}
    for name, ids in (("generation", gen_ids), ("scoring", pop_ids), ("echo", echo_ids)):
        p = tmp / f"{name}.list"
        p.write_bytes(canonical_bytes(list(ids)))
        paths[name] = p
    g_l, e_l = elist(reg_gen or gen_ids), elist(echo_ids)
    p_l = elist(reg_pop or pop_ids)
    gen = SimpleNamespace(eval_list_file_sha256=g_l.file_sha256, eval_list_set_sha256=g_l.set_sha256,
                          echo_list_file_sha256=e_l.file_sha256, echo_list_set_sha256=e_l.set_sha256)
    pop = SimpleNamespace(eval_list_file_sha256=p_l.file_sha256, eval_list_set_sha256=p_l.set_sha256, n_images=n_images)
    args = SimpleNamespace(generation_list=paths["generation"], scoring_list=paths["scoring"], echo_list=paths["echo"],
                           registration=registration, purpose=purpose, snapshot=tmp)
    return SimpleNamespace(args=args, reg=SimpleNamespace(generation=gen, population=pop), notes=[])


@pytest.fixture(autouse=True)
def _meta(monkeypatch):
    monkeypatch.setattr(SU, "read_manifest_meta", lambda root: meta())


def code_of(exc: pytest.ExceptionInfo) -> str:
    return exc.value.args[0] if exc.value.args else ""


# ---------------------------------------------------------------- 진입점의 여섯 경우

def test_정상_전량은_지난다(tmp_path) -> None:
    gen, pop, _echo, _meta = SU._read_lists(ctx_for(tmp_path, gen_ids=EVAL, pop_ids=EVAL))
    assert gen.file_sha256 == pop.file_sha256 and gen.id_set == set(EVAL)


def test_미등록_채점_목록은_거부한다(tmp_path) -> None:
    c = ctx_for(tmp_path, gen_ids=EVAL, pop_ids=EVAL)
    c.reg.population.eval_list_file_sha256 = c.reg.population.eval_list_set_sha256 = None
    with pytest.raises(EntryRejected) as e:
        SU._read_lists(c)
    assert "LIST_BINDING" in str(e.value)


def test_한_장_빠진_목록은_전량이_아니다(tmp_path, capsys) -> None:
    short = EVAL[:-1]
    with pytest.raises(EntryRejected) as e:
        SU._read_lists(ctx_for(tmp_path, gen_ids=short, pop_ids=short))
    msg = str(e.value)
    assert "MAIN_LIST_NOT_FULL_EVAL" in msg and "빠짐 1개" in msg
    assert not any(i in msg for i in EVAL), "사유에 평가 id 를 싣지 않는다 — 건수만"


def test_같은_집합의_다른_순서는_등록과_다르면_거부한다(tmp_path) -> None:
    with pytest.raises(EntryRejected) as e:
        SU._read_lists(ctx_for(tmp_path, gen_ids=EVAL[::-1], pop_ids=EVAL, reg_gen=EVAL))
    assert "LIST_BINDING" in str(e.value)


def test_생성과_채점이_같은_집합의_다른_순서면_거부한다(tmp_path) -> None:
    """두 목록이 각자의 등록 해시와는 맞지만 파일이 다르다 — 주 분석은 같은 파일이어야 한다."""
    with pytest.raises(EntryRejected) as e:
        SU._read_lists(ctx_for(tmp_path, gen_ids=EVAL, pop_ids=EVAL[::-1]))
    assert "MAIN_LIST_NOT_IDENTICAL" in str(e.value)


def test_생성과_채점이_한_장_다르면_거부한다(tmp_path) -> None:
    """채점 목록이 생성 목록의 부분집합이어도 — 부분집합 재채점은 주 분석의 경로가 아니다."""
    with pytest.raises(EntryRejected) as e:
        SU._read_lists(ctx_for(tmp_path, gen_ids=EVAL, pop_ids=EVAL[:-1]))
    assert "MAIN_LIST_NOT_IDENTICAL" in str(e.value)


def test_등록_지문이_다르면_거부한다(tmp_path) -> None:
    with pytest.raises(EntryRejected) as e:
        SU._read_lists(ctx_for(tmp_path, gen_ids=EVAL, pop_ids=EVAL, reg_pop=EVAL[:-1]))
    assert "LIST_BINDING" in str(e.value)


def test_진단의_부분집합_채점은_그대로_받는다(tmp_path) -> None:
    """07b 의 m-4 — 진단(`frame_diag`) 목적의 합법적 부분집합도 전량 계약 밖이다."""
    gen, pop, _e, _m = SU._read_lists(ctx_for(tmp_path, gen_ids=VAL, pop_ids=VAL[:-1], purpose="frame_diag",
                                              registration="probe"))
    assert pop.id_set < gen.id_set


def test_리허설의_부분집합_채점은_그대로_받는다(tmp_path) -> None:
    """전량 계약은 본실험에만 건다 — 리허설 · 진단의 합법적 부분집합은 지킨다."""
    gen, pop, _e, _m = SU._read_lists(ctx_for(tmp_path, gen_ids=VAL, pop_ids=VAL[:-1], purpose="rehearsal",
                                              registration="probe"))
    assert pop.id_set < gen.id_set


# ---------------------------------------------------------------- 관문 함수

def test_전량_관문은_평가_행_전체와_같은_파일을_받는다() -> None:
    EG.check_main_full_eval(elist(EVAL), elist(EVAL), meta())


@pytest.mark.parametrize(("gen_ids", "pop_ids", "code"), [
    (EVAL[:-1], EVAL[:-1], "MAIN_LIST_NOT_FULL_EVAL"),
    (EVAL, EVAL[:-1], "MAIN_LIST_NOT_IDENTICAL"),
    (EVAL, EVAL[::-1], "MAIN_LIST_NOT_IDENTICAL"),
])
def test_전량_관문의_거부(gen_ids, pop_ids, code: str) -> None:
    with pytest.raises(EntryRejected) as e:
        EG.check_main_full_eval(elist(gen_ids), elist(pop_ids), meta())
    assert code in str(e.value)


# ---------------------------------------------------------------- 등록

def test_본실험_등록은_두_목록_해시가_같아야_한다() -> None:
    from tests.test_prereg_unified import filled  # 같은 트랙의 시험 도우미
    r = filled(purpose="main")
    r.population.eval_list_file_sha256 = "a" * 64
    r.population.eval_list_set_sha256 = "b" * 64
    assert any(x.startswith("population.eval_list_file_sha256") for x in r.invalid(side="scoring"))


def test_본실험의_check_list_binding_은_같은_파일을_요구한다() -> None:
    from tests.test_prereg_unified import filled
    r = filled(purpose="main")
    g, p = elist(EVAL), elist(EVAL[:-1])
    r.generation.eval_list_file_sha256, r.generation.eval_list_set_sha256 = g.file_sha256, g.set_sha256
    r.population.eval_list_file_sha256, r.population.eval_list_set_sha256 = p.file_sha256, p.set_sha256
    with pytest.raises(P.ListBindingError, match="같은 파일"):
        P.check_list_binding(r, g, p)


# ---------------------------------------------------------------- 생성 쪽에 넘기는 시험 벡터

VECTORS = Path(__file__).resolve().parents[1] / "evaluation" / "full_eval_vectors.json"


def _vectors() -> dict:
    import json
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def _vector_meta(v: dict) -> ManifestMeta:
    split = {**{i: "eval" for i in v["eval_ids"]}, **{i: "val" for i in v["other_ids"]}}
    return ManifestMeta(split_of=MappingProxyType(split), sha256_of=MappingProxyType({i: "0" * 64 for i in split}),
                        group_of=MappingProxyType({i: f"g_{i}" for i in split}))


@pytest.mark.parametrize("case", _vectors()["cases"] if VECTORS.is_file() else [None], ids=lambda c: c["name"] if c else "없음")
def test_시험_벡터는_채점_쪽_관문과_같은_판정이다(case) -> None:
    """생성 쪽 관문이 맞댈 벡터가 채점 쪽 관문의 판정과 어긋나면 여기서 먼저 떨어진다."""
    assert case is not None, "evaluation/full_eval_vectors.json 이 없다"
    meta_ = _vector_meta(_vectors())
    if case["expect"] == "pass":
        EG.check_main_full_eval(elist(case["gen"]), elist(case["pop"]), meta_)
    else:
        with pytest.raises(EntryRejected) as e:
            EG.check_main_full_eval(elist(case["gen"]), elist(case["pop"]), meta_)
        assert case["expect"] in str(e.value)
    if case["export_side"]:
        assert case["gen"] == case["pop"], "생성 쪽 경우는 목록 하나다"


def test_시험_벡터는_다섯_거부와_정상을_덮는다() -> None:
    kinds = {c["expect"] for c in _vectors()["cases"]}
    assert kinds == {"pass", "MAIN_LIST_NOT_FULL_EVAL", "MAIN_LIST_NOT_IDENTICAL"}
    assert any(c["export_side"] and c["expect"] == "MAIN_LIST_NOT_FULL_EVAL" for c in _vectors()["cases"])
