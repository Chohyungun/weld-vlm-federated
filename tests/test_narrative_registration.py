"""판 6 신원 대조(`evaluation.narrative_registration`) · 서술 배치 결속(`evaluation.eval_registration.narrative_layout_problems`).

적힌 해시가 실제 바이트와 같으면 통과하고, 추출기 · 허용치 표 · 로더 · 벡터 · 생성기 · 프롬프트 가운데 하나라도 다르면 거부한다(195 M5).
실제 저장소를 고치지 않으려고 필요한 파일만 임시 폴더에 복사해 바꾼다.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation import narrative_checker as NCK
from evaluation import narrative_registration as NR
from evaluation import prereg_unified as P
from evaluation.eval_registration import narrative_layout_problems
from evaluation.narrative_conformance import VECTOR_REL
from tests.test_prereg_v6 import axis_spec, narrative_spec, v6

REPO = Path(__file__).resolve().parents[1]
COPIED = (NR.EXTRACTOR_REL, VECTOR_REL, NR.LIMITS_CSV_REL, NR.GENERATOR_REL, "configs/label_map.yaml",
          NR.CHECKER_REL, NCK.LISTS_REL, *NR.LIMITS_LOADER_FILES)


@pytest.fixture()
def world(tmp_path: Path):
    for rel in COPIED:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, tmp_path / rel)
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes("가상 프롬프트 — 시험용\n".encode())
    act = NR.actual_identity(tmp_path)
    reg = v6()
    reg = replace(reg, narrative=narrative_spec(**act), clause_axis=axis_spec(limits_sha256=act["limits_sha256"]),
                  generation=replace(reg.generation, prompt_sha256=NR.prompt_sha256(prompt)))
    return tmp_path, reg, prompt


def test_로더_묶음의_해시는_경로_정렬_탭_줄이다() -> None:
    lines = "".join(f"{rel}\t{NR.lf_sha256(REPO / rel)}\n" for rel in sorted(NR.LIMITS_LOADER_FILES))
    assert NR.limits_loader_bundle_sha256(REPO) == hashlib.sha256(lines.encode("utf-8")).hexdigest()
    assert NR.limits_loader_bundle_sha256(REPO, files=tuple(reversed(NR.LIMITS_LOADER_FILES))) == \
        NR.limits_loader_bundle_sha256(REPO)


def test_적힌_값이_실제와_같으면_통과한다(world) -> None:
    root, reg, prompt = world
    reg.require_complete()
    assert NR.identity_problems(reg, repo=root, prompt_path=prompt) == []


@pytest.mark.parametrize("rel,field", [
    (NR.EXTRACTOR_REL, "narrative.extractor_sha256"),
    (NR.LIMITS_CSV_REL, "narrative.limits_sha256"),
    ("corpus/rules/limit_eval.py", "narrative.limits_loader_sha256"),
    (VECTOR_REL, "narrative.extract_vectors_sha256"),
    (NR.GENERATOR_REL, "narrative.generator_sha256"),
    (NR.CHECKER_REL, "narrative.checker_sha256"),
    (NCK.LISTS_REL, "narrative.lists_sha256"),
])
def test_한_파일이_바뀌면_그_칸이_거부된다(world, rel: str, field: str) -> None:
    root, reg, prompt = world
    with (root / rel).open("ab") as fh:
        fh.write(b"\n# x\n" if rel.endswith(".py") else b" ")
    problems = NR.identity_problems(reg, repo=root, prompt_path=prompt)
    assert any(p.startswith(field) for p in problems), problems
    if rel in (VECTOR_REL, NCK.LISTS_REL):
        assert any("핀" in p for p in problems)                        # 정본 핀과도 어긋난다


def test_프롬프트가_바뀌면_거부된다(world) -> None:
    root, reg, prompt = world
    prompt.write_bytes("가상 프롬프트 — 바뀜\n".encode())
    assert any(p.startswith("generation.prompt_sha256") for p in NR.identity_problems(reg, repo=root, prompt_path=prompt))


def test_선택_칸은_파일을_줄_때만_맞댄다(world, tmp_path: Path) -> None:
    """학습 쪽 추출기는 파일을 줄 때만 맞댄다 — LF sha256 과 식별자(`EXTRACTOR_ID`, 구문 트리로 읽음) 둘 다. 검사기 · 목록은 저장소
    정본으로 늘 맞대고, 다른 사본을 주면 그 사본과 맞댄다."""
    root, reg, prompt = world
    train = tmp_path / "narrative.py"
    train.write_bytes(b'EXTRACTOR_ID = "vlm_narrative_v1"\nx = 1\n')
    assert any(p.startswith("narrative.train_extractor_sha256") for p in
               NR.identity_problems(reg, repo=root, train_extractor_path=train))
    ok = replace(reg, narrative=replace(reg.narrative, train_extractor_sha256=NR.lf_sha256(train)))
    assert NR.identity_problems(ok, repo=root, train_extractor_path=train, prompt_path=prompt) == []
    renamed = tmp_path / "renamed.py"
    renamed.write_bytes(b'EXTRACTOR_ID = "other_v9"\nx = 1\n')
    bad_id = replace(ok, narrative=replace(ok.narrative, train_extractor_sha256=NR.lf_sha256(renamed)))
    assert [p for p in NR.identity_problems(bad_id, repo=root, train_extractor_path=renamed)
            if p.startswith("narrative.train_extractor_id")]
    other = tmp_path / "checker_copy.py"
    other.write_bytes(b"y = 2\n")
    assert any(p.startswith("narrative.checker_sha256") for p in NR.identity_problems(ok, repo=root, checker_path=other))


def test_확인_파일은_실제_바이트로_맞댄다(world, tmp_path: Path) -> None:
    """확인 파일(정본 추출 벡터의 다른 사본)을 주면 그 실제 바이트를 `extract_vectors_sha256` 과 맞댄다(지시 1023c 과제 2)."""
    root, reg, _ = world
    same = tmp_path / "copy.json"
    same.write_bytes((REPO / VECTOR_REL).read_bytes())
    assert NR.identity_problems(reg, repo=root, vectors_path=same) == []
    other = tmp_path / "other.json"
    other.write_bytes((REPO / VECTOR_REL).read_bytes() + b" ")
    assert any(p.startswith("narrative.extract_vectors_sha256") for p in NR.identity_problems(reg, repo=root, vectors_path=other))


def test_잔여_한계와_탐색적_종류는_검사기_판의_정본과_맞댄다(world) -> None:
    root, reg, _ = world
    assert NR.HOLES_CHECKER_ID == NCK.CHECKER_ID and NR.EXPLORATORY_KINDS == ("마",)
    for name, value in (("open_holes", NR.OPEN_HOLES[:-1]), ("exploratory_kinds", ())):
        bad = replace(reg, narrative=replace(reg.narrative, **{name: value}))
        assert any(p.startswith(f"narrative.{name}") for p in NR.identity_problems(bad, repo=root)), name


def test_식별자는_파일을_실행하지_않고_읽는다(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_bytes('raise SystemExit("실행하면 안 된다")\nEXTRACTOR_ID: str = "vlm_narrative_v1"\n'.encode())
    assert NR.module_str_constant(src, "EXTRACTOR_ID") == "vlm_narrative_v1"
    assert NR.module_str_constant(src, "NONE_SUCH") is None
    assert NR.module_str_constant(REPO / NR.TRAIN_EXTRACTOR_REL, "EXTRACTOR_ID") == "vlm_narrative_v1"


def test_작성기는_신원을_실제_바이트로_채우고_경로가_없는_칸은_비운다(tmp_path: Path) -> None:
    """판 6 블록 작성기 — 어휘 · 신원 · 잔여 한계 · 탐색적 종류를 채우고, 확인 사례 · 검사기 벡터 · 학습 전 계약 묶음은 경로를 줄 때만
    (지어 넣지 않는다)."""
    n = NR.narrative_block(REPO)
    act = NR.actual_identity(REPO)
    assert all(getattr(n, k) == v for k, v in act.items())
    assert (n.train_extractor_id, n.train_extractor_sha256) == ("vlm_narrative_v1", NR.lf_sha256(REPO / NR.TRAIN_EXTRACTOR_REL))
    assert (n.open_holes, n.exploratory_kinds) == (NR.OPEN_HOLES, NR.EXPLORATORY_KINDS)
    assert n.checker_vectors_sha256 is None and n.confirmation_cases_sha256 is None and n.inputs_contract_sha256 is None
    conf = tmp_path / "conf.json"
    conf.write_bytes(b"{}")
    part = replace(v6(), narrative=NR.narrative_block(REPO, confirmation_path=conf, checker_vectors_path=conf),
                   clause_axis=NR.clause_axis_block(REPO))
    assert [m for m in part.missing() if m.startswith(("narrative", "clause_axis"))] == ["narrative.inputs_contract_sha256"]
    full = NR.narrative_block(REPO, confirmation_path=conf, checker_vectors_path=conf, inputs_contract_path=conf)
    reg = replace(v6(), narrative=full, clause_axis=NR.clause_axis_block(REPO))
    assert not [m for m in reg.missing() if m.startswith(("narrative", "clause_axis"))]
    assert not [p for p in reg.invalid() if p.startswith(("narrative", "clause_axis"))], reg.invalid()
    assert reg.clause_axis.weighting == "design_and_unweighted"
    assert reg.clause_axis.ci == "paired_cluster_bootstrap_aux_cells_and_baseline"       # 대응 구간 = 같은 W(모델 짝 · 대조선)


def test_판_5_등록은_서술_블록이_없다고_거부한다() -> None:
    from tests.test_prereg_unified import filled

    assert NR.identity_problems(filled())[0].startswith("version")


def test_서술_배치는_계약_v2_의_문자열과_같아야_한다() -> None:
    reg = v6()
    assert narrative_layout_problems(reg) == []
    assert narrative_layout_problems(reg, {"layout": P.NARRATIVE_LAYOUT, "separator": "\n"}) == []
    assert narrative_layout_problems(reg, {"layout": "json_then_text", "separator": "\n"})
    bad = replace(reg, narrative=replace(reg.narrative, layout="json_then_text"))
    assert any(p.startswith("narrative.layout") for p in narrative_layout_problems(bad))
    from tests.test_prereg_unified import filled

    assert narrative_layout_problems(filled()) == []                   # 판 5 — 볼 것이 없다


# ════════════════════════════════════════════════════════════ 판 6 연결 2 판 — 프로필 regime2_v4(§3-2 · §3-3 · R10 · R11)

from evaluation.narrative_conformance import VECTOR_SHA256
from tests import v6_link_world as VW


@pytest.fixture()
def world2(tmp_path: Path):
    """신판 프로필의 세계 — 명시한 파일을 복사하고 프로필 작성기로 판 6 블록을 짓는다."""
    VW.copy_pinned(tmp_path)
    reg = replace(v6(), narrative=NR.narrative_block(tmp_path, profile=NR.PROFILE_REGIME2),
                  clause_axis=NR.clause_axis_block(tmp_path, profile=NR.PROFILE_REGIME2))
    reg = replace(reg, generation=replace(reg.generation, prompt_sha256=NR.prompt_sha256(tmp_path / VW.PROMPT_REL)))
    return tmp_path, reg


def _codes2(reg, root: Path, **kw) -> list[str]:
    return sorted({p.split(":", 1)[0] for p in NR.identity_problems(reg, repo=root, profile=NR.PROFILE_REGIME2, **kw)})


def test_v6_link_profile_block_matches_actual_bytes(world2) -> None:
    root, reg = world2
    reg.require_complete()
    assert NR.identity_problems(reg, repo=root, profile=NR.PROFILE_REGIME2, prompt_path=root / VW.PROMPT_REL,
                                inputs_contract_path=root / NR.INPUTS_CONTRACT_REL) == []
    n, pin = reg.narrative, NR.CHECKER_PINS[1]
    assert (n.checker_id, n.limits_sha256, n.lists_sha256) == (pin.checker_id, pin.limits_sha256, pin.lists_sha256)
    assert n.generator_sha256 == NR.generator_bundle_identity(root)["sha256"] != NR.lf_sha256(root / NR.GENERATOR_REL)
    assert (n.checker_vectors_sha256, n.confirmation_cases_sha256) == (NR.V6_EVIDENCE_V4[0].sha256, NR.V6_EVIDENCE_V4[1].sha256)
    assert n.inputs_contract_sha256 == NR.raw_sha256(root / NR.INPUTS_CONTRACT_REL)
    assert (len(n.open_holes), n.exploratory_kinds) == (19, ("나", "마"))
    assert reg.clause_axis.limits_sha256 == n.limits_sha256 == pin.limits_sha256
    assert not [p for p in reg.invalid() if p.startswith(("narrative", "clause_axis"))], reg.invalid()
    # 프로필 없는 호출은 옛 동작 그대로 — 구판 블록 · 구판 신원. 신판 등록을 구판 잣대로 보면 어긋난다(자동 승격 없음)
    old = NR.narrative_block(root)
    assert (old.checker_id, old.open_holes, old.exploratory_kinds) == (NCK.CHECKER_ID, NR.OPEN_HOLES, NR.EXPLORATORY_KINDS)
    assert NR.actual_identity(root) == NR.actual_identity(root, profile=NR.PROFILE_LEGACY)
    assert NR.identity_problems(reg, repo=root)
    assert NR.NARRATIVE_PROFILES[NR.PROFILE_REGIME2].checker_pin is NR.CHECKER_PINS[1]      # 승인 행을 참조한다(중복 작성 0)


def _holes_from_document(path: Path) -> list[str]:
    """§15 확정 표의 등록 문구 열(백틱 안)을 문서에서 직접 뽑는다 — 코드 상수와의 두 구현 대조."""
    import re

    text = path.read_bytes().decode("utf-8")
    section = text.split("## 15. ")[1].split("\n## 16. ")[0]
    rows = [line for line in section.splitlines() if re.match(r"^\| \d+ \| `", line)]
    return [re.match(r"^\| (\d+) \| `(.*)` \| ", line).group(2) for line in rows]


def test_v6_link_holes_verbatim(world2) -> None:
    """R10 — v4 의 잔여 한계는 10 9판 §15 의 19 줄 · 순서 그대로, 탐색적 종류는 (나)(마). 옛 11 줄 · 빈 목록 · 한 줄 변경 · 순서 변경 ·
    옛 목록 덧붙임 · 종류 변경은 `profile_holes`."""
    root, reg = world2
    src = REPO / NR.HOLES_SOURCE_REL
    assert NR.raw_sha256(src) == NR.HOLES_SOURCE_SHA256
    assert _holes_from_document(src) == list(NR.OPEN_HOLES_V4) and len(NR.OPEN_HOLES_V4) == 19
    assert NR.NARRATIVE_PROFILES[NR.PROFILE_REGIME2].open_holes == NR.OPEN_HOLES_V4
    assert NR.NARRATIVE_PROFILES[NR.PROFILE_LEGACY].open_holes == NR.OPEN_HOLES and len(NR.OPEN_HOLES) == 11
    assert _codes2(reg, root) == []
    h = NR.OPEN_HOLES_V4
    for name, value in (("open_holes", NR.OPEN_HOLES), ("open_holes", ()), ("open_holes", (*h[:16], h[16] + " ", *h[17:])),
                        ("open_holes", (*h[1:], h[0])), ("open_holes", (*h, *NR.OPEN_HOLES)), ("open_holes", None),
                        ("exploratory_kinds", ("마",)), ("exploratory_kinds", ("나", "마", "다")), ("exploratory_kinds", ())):
        bad = replace(reg, narrative=replace(reg.narrative, **{name: value}))
        assert "profile_holes" in _codes2(bad, root), (name, value)


def test_v6_link_evidence_roles_and_bytes(world2, tmp_path: Path) -> None:
    """R11 — 검사기 벡터 칸에 추출 벡터 · 13c · 확인 사례를, 확인 사례 칸에 탐침 · 13c 를 대입하면 `evidence_role`; 빈 칸 · 다른 바이트 ·
    정본 파일의 변조는 `evidence_bytes`. 작성기는 변조된 정본으로 블록을 짓지 않고, 경로 인자를 따로 받지 않는다."""
    root, reg = world2
    probes, confirm = NR.V6_EVIDENCE_V4[0], NR.V6_EVIDENCE_V4[1]
    c13 = next(r for r in NR.V6_EVIDENCE_V4 if r.role == "confirmation_output")
    for field, value in (("checker_vectors_sha256", VECTOR_SHA256), ("checker_vectors_sha256", c13.sha256),
                         ("checker_vectors_sha256", confirm.sha256), ("confirmation_cases_sha256", probes.sha256),
                         ("confirmation_cases_sha256", c13.sha256)):
        got = _codes2(replace(reg, narrative=replace(reg.narrative, **{field: value})), root)
        assert "evidence_role" in got and "evidence_bytes" not in got, (field, value, got)
    for field in ("checker_vectors_sha256", "confirmation_cases_sha256"):
        assert "evidence_bytes" in _codes2(replace(reg, narrative=replace(reg.narrative, **{field: None})), root)
        assert "evidence_bytes" in _codes2(replace(reg, narrative=replace(reg.narrative, **{field: "f" * 64})), root)
    with (root / confirm.path).open("ab") as fh:                       # 정본 파일 자체의 변조 — 등록 칸은 핀과 같아도 거부
        fh.write(b" ")
    assert "evidence_bytes" in _codes2(reg, root)
    with pytest.raises(NR.ProfileRefused) as exc:
        NR.narrative_block(root, profile=NR.PROFILE_REGIME2)
    assert exc.value.code == "evidence_bytes"
    with pytest.raises(NR.ProfileRefused) as exc2:
        NR.narrative_block(REPO, profile=NR.PROFILE_REGIME2, confirmation_path=tmp_path / "x.json")
    assert exc2.value.code == "profile_paths"
    assert NR.evidence_problems(reg.narrative, NR.NARRATIVE_PROFILES[NR.PROFILE_LEGACY], root) == []   # 구판은 볼 것이 없다


def test_v6_link_writer_checks_contract_files(world2) -> None:
    """프로필 작성은 학습 전 계약의 실물 대조(`contract_code_problems`)를 지난다 — 생성 경로 한 파일이 바뀌면 `code_bytes` 로 멈춘다."""
    root, _ = world2
    assert NR.contract_code_problems((root / NR.INPUTS_CONTRACT_REL).read_bytes(), repo=root) == []
    with (root / NR.GENERATOR_FILES_V1[2]).open("ab") as fh:
        fh.write(b"\n# x\n")
    with pytest.raises(NR.ProfileRefused) as exc:
        NR.narrative_block(root, profile=NR.PROFILE_REGIME2)
    assert exc.value.code == "code_bytes"


def test_v6_link_decision_and_evidence_pins_are_real_blobs() -> None:
    """출처 운반 파일이 가리키는 아홉 원본은 이 트리의 실제 바이트이고, 결정 blob 은 그 커밋의 의사결정로그다(커밋이 없는 체크아웃은
    건너뛴다 — 통과로 세지 않는다)."""
    from evaluation.entry_gate import EntryRejected, blob_at

    for r in NR.V6_EVIDENCE_V4:
        assert NR.raw_sha256(REPO / r.path) == r.sha256 and r.hash_rule == "raw_sha256", r.role
    assert [r.role for r in NR.V6_EVIDENCE_V4] == ["checker_vectors", "confirmation_cases", "regression", "expectation_changes",
                                                   "regression_receipt", "holes_source", "disposition", "confirmation_output",
                                                   "uncertain_classification"]
    try:
        raw = blob_at(REPO, NR.V6_DECISION["commit"], NR.V6_DECISION["path"])
    except EntryRejected:
        pytest.skip(f"이 체크아웃에 결정 커밋 {NR.V6_DECISION['commit'][:8]} 이 없다")
    assert hashlib.sha256(raw).hexdigest() == NR.V6_DECISION["sha256"]
    assert "판 6 연결 결정 D1 ~ D6 확정(D7 보류)" in raw.decode("utf-8")
