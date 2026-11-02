"""본실험 표본 목록 관문의 공유 시험 벡터 — 생성 쪽 `vlm.export_preflight.check_sample_eval_set` 과 채점 쪽
`evaluation.entry_gate.check_main_sample` 이 같은 경우에서 같은 판정을 내는지 본다(1016k, 03 §0 ②).

합성 id 다 — 실물 평가 자산을 열지 않는다. 세계는 채점 쪽 시험(`tests/test_main_sample_binding.py`)과 같은 꼴이다.
채점 쪽의 '채점 목록 = 생성 목록' 판정은 생성 쪽에 없다(목록이 하나다) — 벡터는 채점 목록을 생성 목록과 같은 파일로 둔다.
`expect` 는 채점 쪽 거부 사유 코드(대문자)이고 생성 쪽 코드는 그 소문자다. None 은 지난다.
"""

from __future__ import annotations

from types import MappingProxyType

GROUPS = {"gA": ("e01", "e02"), "gB": ("e03",), "gC": ("e04", "e05"), "gZ": ("e06", "e07")}
SAMPLE = ("e01", "e02", "e03", "e04", "e05")
STRATUM = {"gA": ("S1", 3, 2), "gB": ("S1", 3, 2), "gC": ("S2", 2, 1)}

CASES: tuple[dict, ...] = (
    {"name": "봉인한_표본", "ids": SAMPLE, "expect": None},
    {"name": "봉인한_목록의_파일_해시가_아니다", "ids": SAMPLE, "sealed": {"file_sha256": "f" * 64},
     "expect": "MAIN_SAMPLE_NOT_SEALED"},
    {"name": "봉인한_목록의_집합_해시가_아니다", "ids": SAMPLE, "sealed": {"set_sha256": "f" * 64},
     "expect": "MAIN_SAMPLE_NOT_SEALED"},
    {"name": "장수가_봉인과_다르다", "ids": SAMPLE, "sealed": {"images": 6}, "expect": "MAIN_SAMPLE_COUNT"},
    {"name": "뽑힌_묶음의_평가_이미지가_다_들지_않았다", "ids": ("e01", "e03", "e04", "e05"),
     "expect": "MAIN_SAMPLE_GROUP_SPLIT"},
    {"name": "묶음_수가_봉인과_다르다", "ids": SAMPLE, "sealed": {"groups": 4}, "expect": "MAIN_SAMPLE_GROUPS"},
    {"name": "곁_파일의_순서가_목록과_다르다", "ids": SAMPLE, "side_reversed": True, "expect": "MAIN_SAMPLE_SIDE"},
    {"name": "층이_묶음을_가른다", "ids": SAMPLE, "side": {"e02": ("S2", 2, 1)}, "expect": "MAIN_SAMPLE_STRATA"},
    {"name": "층의_n_h_가_뽑힌_묶음_수와_다르다", "ids": SAMPLE,
     "side": {"e01": ("S1", 3, 3), "e02": ("S1", 3, 3), "e03": ("S1", 3, 3)}, "expect": "MAIN_SAMPLE_STRATA"},
)


def meta():
    from evaluation.entry_gate import ManifestMeta

    group_of = {i: g for g, ids in GROUPS.items() for i in ids} | {"v01": "gv"}
    split = {i: "eval" for i in group_of} | {"v01": "val"}
    return ManifestMeta(split_of=MappingProxyType(split), sha256_of=MappingProxyType({i: "0" * 64 for i in split}),
                        group_of=MappingProxyType(group_of))


def elist(ids):
    from evaluation.eval_list import canonical_bytes, validate_eval_list

    return validate_eval_list(canonical_bytes(list(ids)))


def inputs(case: dict):
    """(목록, 매니페스트, 봉인, 곁 파일) — 봉인은 목록에서 짓고 `sealed` 로 덮는다. 곁 파일은 목록 순서이고 `side` 로 덮는다."""
    from evaluation.sample_weights import SideRow

    m = meta()
    el = elist(case["ids"])
    sealed = {"images": len(el), "groups": len({m.group_of[i] for i in el.ids}), "file_sha256": el.file_sha256,
              "set_sha256": el.set_sha256, "side_sha256": "0" * 64} | dict(case.get("sealed") or {})
    rows = {i: SideRow(*STRATUM[m.group_of[i]]) for i in el.ids}
    rows.update({i: SideRow(*v) for i, v in (case.get("side") or {}).items()})
    order = list(reversed(el.ids)) if case.get("side_reversed") else list(el.ids)
    return el, m, sealed, {i: rows[i] for i in order}
