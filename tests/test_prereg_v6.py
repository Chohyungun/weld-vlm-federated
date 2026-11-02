"""등록 판 6 — 서술 블록 `narrative` · 조항 축 블록 `clause_axis`(지시 1022b · 03 2판 §5a · 195 M5).

판 5 의 바이트와 지문은 바뀌지 않아야 한다 — main `e1820b2d` 의 옛 모듈로 실제로 맞댄다. 판 6 은 블록의 칸마다 채점 지문이 바뀌고,
어휘 밖의 값과 다른 판의 꼴을 거부한다.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest

from evaluation import prereg_unified as P
from evaluation.narrative_registration import EXPLORATORY_KINDS, OPEN_HOLES
from tests.test_prereg_unified import filled

REPO = Path(__file__).resolve().parents[1]
MAIN_BEFORE_V6 = "e1820b2d"


def narrative_spec(**kw) -> P.NarrativeSpec:
    vocab = {f.name: P.VOCAB[f"narrative.{f.name}"][0] for f in fields(P.NarrativeSpec) if f"narrative.{f.name}" in P.VOCAB}
    hexes = {f.name: f"{i:x}" * 64 for i, f in enumerate(fields(P.NarrativeSpec), start=1)
             if f"narrative.{f.name}" in P.HEX64_FIELDS}
    base = {**vocab, **{k: v[:64] for k, v in hexes.items()}, "extractor_id": "narrative_extract_v1",
            "train_extractor_id": "vlm_narrative_v1", "checker_id": "narrative_checker_v3", "label_map_version": 1,
            "kinds": P.NARRATIVE_KINDS, "reasons": P.NARRATIVE_REASONS, "open_holes": OPEN_HOLES,
            "exploratory_kinds": EXPLORATORY_KINDS}
    base.update(kw)
    return P.NarrativeSpec(**base)


def axis_spec(**kw) -> P.ClauseAxisSpec:
    base = {f.name: P.VOCAB[f"clause_axis.{f.name}"][0] for f in fields(P.ClauseAxisSpec)
            if f"clause_axis.{f.name}" in P.VOCAB}
    base.update(metrics=P.CLAUSE_AXIS_METRICS, limits_sha256=narrative_spec().limits_sha256)
    base.update(kw)
    return P.ClauseAxisSpec(**base)


def v6(**kw) -> P.UnifiedRegistration:
    return replace(filled(), version=P.REGISTRATION_VERSION_V6, narrative=narrative_spec(), clause_axis=axis_spec(), **kw)


def _old_module():
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{MAIN_BEFORE_V6}:evaluation/prereg_unified.py"],
                         capture_output=True, check=False)
    if src.returncode != 0:
        pytest.skip(f"이 체크아웃에 {MAIN_BEFORE_V6} 가 없다")
    return src.stdout


def test_판_5_등록의_바이트와_두_지문은_판_6_전의_모듈과_같다(tmp_path: Path) -> None:
    path = tmp_path / "prereg_before_v6.py"
    path.write_bytes(_old_module())
    spec = importlib.util.spec_from_file_location("prereg_before_v6", path)
    old = importlib.util.module_from_spec(spec)
    sys.modules["prereg_before_v6"] = old
    try:
        spec.loader.exec_module(old)
        for reg in (filled(), filled(generation_only=True), filled(purpose="rehearsal")):
            raw = P.dump_registration(reg)
            back = old.load_registration(raw)                       # 옛 정규형과 바이트가 같아야 읽힌다
            assert old.dump_registration(back) == raw
            assert back.scoring_sha256() == reg.scoring_sha256()
            assert back.generation_sha256() == reg.generation_sha256()
    finally:
        sys.modules.pop("prereg_before_v6", None)


def test_기본_판은_여전히_5_이고_판_6_은_읽기만_넓힌다() -> None:
    assert P.REGISTRATION_VERSION == "5" and P.READABLE_VERSIONS == ("5", "6")
    assert "narrative" not in filled().scoring_payload() and "clause_axis" not in filled().scoring_payload()


def test_판_6_은_완결이면_지나고_파일에서_그대로_돌아온다() -> None:
    reg = v6()
    reg.require_complete()
    assert P.load_registration(P.dump_registration(reg)) == reg
    assert reg.scoring_sha256() != replace(reg, version="5", narrative=None, clause_axis=None).scoring_sha256()


def test_판_6_의_빈_블록과_빈_칸은_빈_항목이다() -> None:
    assert "narrative" in replace(v6(), narrative=None).missing()
    assert "narrative.scope" in replace(v6(), narrative=narrative_spec(scope=None)).missing()
    assert "clause_axis.limits_sha256" in replace(v6(), clause_axis=axis_spec(limits_sha256=None)).missing()


def test_판_5_에_블록이_있으면_무효다() -> None:
    bad = replace(filled(), narrative=narrative_spec())
    assert any(p.startswith("narrative: 판 5") for p in bad.invalid())


@pytest.mark.parametrize("area,name", [("narrative", f.name) for f in fields(P.NarrativeSpec)]
                         + [("clause_axis", f.name) for f in fields(P.ClauseAxisSpec)])
def test_블록의_칸마다_채점_지문이_바뀐다(area: str, name: str) -> None:
    """정책 · 기대값 · 어휘 목록 · 별칭 표(`lists_sha256`) · 표 · 로더 · 벡터 — 어느 칸이 바뀌어도 지문이 바뀐다(195 M5)."""
    reg = v6()
    block = getattr(reg, area)
    v = getattr(block, name)
    changed = ((v[::-1] if len(v) > 1 else (*v, "가")) if isinstance(v, tuple) else v + 1 if isinstance(v, int) else
               ("0" * 64 if v != "0" * 64 else "1" * 64) if f"{area}.{name}" in P.HEX64_FIELDS else v + "x")
    other = replace(reg, **{area: replace(block, **{name: changed})})
    assert other.scoring_sha256() != reg.scoring_sha256()
    assert other.generation_sha256() == reg.generation_sha256()             # 채점 쪽 블록이다


@pytest.mark.parametrize("area,name,value,word", [
    ("narrative", "layout", "json_then_text", "어휘"),
    ("narrative", "separator", " ", "어휘"),
    ("narrative", "scope", "현 허용치 표 기준", "어휘"),
    ("narrative", "kinds", ("가", "나", "다", "라"), "이어야"),
    ("narrative", "extractor_sha256", "abc", "sha256"),
    ("clause_axis", "metrics", ("citation_match_rate",), "이어야"),
    ("clause_axis", "limits_sha256", "9" * 64, "같은 허용치 표"),
    ("narrative", "exploratory_kinds", ("마", "가"), "차례"),
    ("narrative", "exploratory_kinds", ("바",), "부분"),
    ("clause_axis", "weighting", "unweighted", "어휘"),
    ("clause_axis", "ci", "paired_cluster_bootstrap_aux", "어휘"),
])
def test_어휘_밖의_값과_어긋난_관계는_무효다(area: str, name: str, value, word: str) -> None:
    reg = v6()
    bad = replace(reg, **{area: replace(getattr(reg, area), **{name: value})})
    assert any(word in p for p in bad.invalid()), bad.invalid()


def test_잔여_한계와_탐색적_종류는_빈_목록을_받고_다른_목록_칸은_받지_않는다() -> None:
    """04 §2 의 5 — 구멍이 없다 · 탐색적 종류가 없다는 빈 목록으로 적는다. 다른 문자열 목록 칸의 '비지 않음' 은 그대로다."""
    reg = replace(v6(), narrative=narrative_spec(open_holes=(), exploratory_kinds=()))
    assert not [p for p in reg.invalid() if "open_holes" in p or "exploratory_kinds" in p], reg.invalid()
    bad = replace(v6(), narrative=narrative_spec(kinds=()))
    assert any(p.startswith("narrative.kinds") for p in bad.invalid())


def test_판_5_파일에_판_6_의_열쇠가_있으면_읽지_않는다() -> None:
    raw = P.dump_registration(v6()).replace(b'"version":"6"', b'"version":"5"')
    with pytest.raises(P.RegistrationFileError, match="최상위 키"):
        P.load_registration(raw)
