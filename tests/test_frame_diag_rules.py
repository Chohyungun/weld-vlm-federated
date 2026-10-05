"""프레임 진단의 규칙 파일 — 정규형 · 칸 · 판 · 검사기 지문(07번 §31-1 의 1 · §32-9)."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from evaluation import frame_diag as F
from evaluation import frame_diag_rules as FR

RULES = F.FrameRules(n_img=50, s_min=0.74, w_s=0.11, w_r=0.0075, w_mix=0.0019, delta=0.0055, trim=0.0,
                     n_boot=400, boot_seed=1)


REQS = [
    {"noise_step": 0.1, "stat": "r", "axis": "x", "width": "w_r", "n0": 800, "n_required": 300, "g0": 400,
     "groups_required": 150, "q0": None, "pairs_required": None},
    {"noise_step": 0.1, "stat": "m", "axis": "x", "width": "axis_ratio_width", "n0": 800, "n_required": 120,
     "g0": 380, "groups_required": 60, "q0": 900, "pairs_required": 140},
    {"noise_step": 0.4, "stat": "r", "axis": "x", "width": "w_r", "n0": 800, "n_required": None, "g0": 400,
     "groups_required": None, "q0": None, "pairs_required": None},
]


def raw(**over) -> bytes:
    return FR.build_rules_file(RULES, sample_requirements=over.pop("reqs", REQS), expected_sha256="e" * 64,
                               cases_sha256="c" * 64, procedure_sha256="p" * 64,
                               noise_ladder=[0.1], summary={"x": 1}, sources={"y": "z"}, **over)


def test_만들고_읽으면_같은_규칙이다() -> None:
    rules, body = FR.load_rules_file(raw())
    assert rules == RULES and body["checker_sha256"] == FR.lf_sha256(FR.CHECKER_PATH)


def test_정규형이_아니면_받지_않는다() -> None:
    body = json.loads(raw())
    pretty = (json.dumps(body, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode("utf-8")
    with pytest.raises(FR.RulesFileError, match="정규형"):
        FR.load_rules_file(pretty)


@pytest.mark.parametrize(("edit", "why"), [
    (lambda b: b.pop("sources"), "칸"),
    (lambda b: b.update(extra=1), "칸"),
    (lambda b: b.update(version="1"), "판"),
    (lambda b: b.pop("sample_requirements"), "칸"),
    (lambda b: b.update(sample_requirements=[]), "비었다"),
    (lambda b: b["sample_requirements"][0].pop("g0"), "행의 칸"),
    (lambda b: b["sample_requirements"][0].update(n_required=-1), "정수"),
    (lambda b: b["sample_requirements"][0].update(n_required=1.5), "정수"),
    (lambda b: b["sample_requirements"][0].update(q0=10), "쌍 수"),
    (lambda b: b["sample_requirements"][1].update(q0=None), "쌍 수"),
    (lambda b: b["rules"].pop("trim"), "칸"),
    (lambda b: b.update(checker_sha256="0" * 64), "검사기"),
])
def test_꼴이_다르면_받지_않는다(edit, why: str) -> None:
    body = json.loads(raw())
    edit(body)
    with pytest.raises(FR.RulesFileError, match=why):
        FR.load_rules_file(FR.canonical(body))


def test_JSON_이_아니면_받지_않는다() -> None:
    with pytest.raises(FR.RulesFileError, match="JSON"):
        FR.load_rules_file(b"\xff")


def test_규칙의_칸은_검사기의_칸과_같다() -> None:
    assert set(json.loads(raw())["rules"]) == set(asdict(RULES))


def _verdict(**over) -> dict:
    return {"n_images": 300, "n_image_groups": 150, "n_pairs": 140, "n_pair_groups": 60} | over


def test_요구_표본_수는_계단마다_가장_큰_요구와_맞대_충족_미달만_낸다() -> None:
    """검토(10-02 병합 전) 4 — 규칙 파일의 요구 장 · 묶음 · 쌍 수가 산출의 충족 · 미달로 이어진다."""
    _, body = FR.load_rules_file(raw())
    got = FR.requirement_status(body, _verdict())
    assert got["0.1"] == {"images": F.MET, "image_groups": F.MET, "pairs": F.MET, "pair_groups": F.MET}
    assert got["0.4"]["images"] == FR.UNDECIDABLE and got["0.4"]["pairs"] == FR.UNDECIDABLE
    short = FR.requirement_status(body, _verdict(n_images=299, n_pair_groups=59))
    assert short["0.1"] == {"images": F.UNMET, "image_groups": F.MET, "pairs": F.MET, "pair_groups": F.UNMET}
    assert all(isinstance(v, str) for s in got.values() for v in s.values()), "상태만 — 수는 싣지 않는다"
