"""본실험 주 분석의 **표본 목록 결속**(`entry_gate.check_main_sample`) — 25번 §2 · 실험설계 8-5(2026-10-13 등록) · 35번 §1.

모집단 가중이 `stratified_g_over_n` 인 등록만 이 길이다. 전량 계약(`check_main_full_eval`)은 그대로 두고 따로 시험한다
(`tests/test_full_eval_binding.py`). 합성 매니페스트만 쓴다 — 실물 평가 자산을 열지 않는다. 사유에는 id 를 싣지 않는다.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from evaluation.entry_gate import EntryRejected, ManifestMeta, check_main_sample
from evaluation.eval_list import canonical_bytes, validate_eval_list
from evaluation.sample_weights import SideRow

GROUPS = {"gA": ("e01", "e02"), "gB": ("e03",), "gC": ("e04", "e05"), "gZ": ("e06", "e07")}
SAMPLE = ("e01", "e02", "e03", "e04", "e05")
STRATUM = {"gA": ("S1", 3, 2), "gB": ("S1", 3, 2), "gC": ("S2", 2, 1)}


def meta() -> ManifestMeta:
    group_of = {i: g for g, ids in GROUPS.items() for i in ids} | {"v01": "gv"}
    split = {i: "eval" for i in group_of} | {"v01": "val"}
    return ManifestMeta(split_of=MappingProxyType(split), sha256_of=MappingProxyType({i: "0" * 64 for i in split}),
                        group_of=MappingProxyType(group_of))


def elist(ids):
    return validate_eval_list(canonical_bytes(list(ids)))


def side_for(ids, *, over=None):
    m = meta()
    out = {i: SideRow(*STRATUM[m.group_of[i]]) for i in ids}
    out.update(over or {})
    return {i: out[i] for i in ids}


def sealed_for(lst, **over):
    return {"images": len(lst), "groups": len({meta().group_of[i] for i in lst.ids}), "file_sha256": lst.file_sha256,
            "set_sha256": lst.set_sha256, "side_sha256": "0" * 64} | over


def check(gen_ids=SAMPLE, pop_ids=None, *, sealed=None, side=None):
    gen = elist(gen_ids)
    pop = elist(gen_ids if pop_ids is None else pop_ids)
    check_main_sample(gen, pop, meta(), sealed=sealed or sealed_for(gen), side=side or side_for(gen.ids))


def code_of(**kw) -> str:
    with pytest.raises(EntryRejected) as exc:
        check(**kw)
    return exc.value.code


def test_봉인한_표본은_지난다() -> None:
    check()


def test_표본도_채점_목록은_생성_목록과_같은_파일이다() -> None:
    assert code_of(pop_ids=SAMPLE[::-1]) == "MAIN_LIST_NOT_IDENTICAL"


def test_봉인한_목록이_아니면_거부한다() -> None:
    other = elist(SAMPLE[::-1])
    assert code_of(sealed=sealed_for(other)) == "MAIN_SAMPLE_NOT_SEALED"


def test_장수가_봉인과_다르면_거부한다() -> None:
    assert code_of(sealed=sealed_for(elist(SAMPLE), images=6)) == "MAIN_SAMPLE_COUNT"


def test_뽑힌_묶음의_평가_이미지가_다_들지_않으면_거부한다() -> None:
    ids = ("e01", "e03", "e04", "e05")                    # gA 의 한 장만
    lst = elist(ids)
    with pytest.raises(EntryRejected) as exc:
        check_main_sample(lst, lst, meta(), sealed=sealed_for(lst), side=side_for(ids))
    assert exc.value.code == "MAIN_SAMPLE_GROUP_SPLIT" and "e02" not in str(exc.value)


def test_묶음_수가_봉인과_다르면_거부한다() -> None:
    assert code_of(sealed=sealed_for(elist(SAMPLE), groups=4)) == "MAIN_SAMPLE_GROUPS"


def test_곁_파일의_순서가_목록과_다르면_거부한다() -> None:
    side = side_for(SAMPLE)
    assert code_of(side={i: side[i] for i in reversed(SAMPLE)}) == "MAIN_SAMPLE_SIDE"


@pytest.mark.parametrize("over", [{"e02": SideRow("S2", 2, 1)}, {"e01": SideRow("S1", 3, 3), "e02": SideRow("S1", 3, 3),
                                                                  "e03": SideRow("S1", 3, 3)}])
def test_층이_묶음을_가르거나_n_h_가_뽑힌_묶음_수와_다르면_거부한다(over) -> None:
    assert code_of(side=side_for(SAMPLE, over=over)) == "MAIN_SAMPLE_STRATA"
