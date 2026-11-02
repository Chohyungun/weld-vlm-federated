"""판 6 서술 · 조항 축 블록의 신원 대조 — 등록에 적은 해시가 실제 바이트와 같은가(195 M5 · 04 §7-5 ⑤ · ⑧).

등록은 값을 적을 뿐이다. 이 모듈이 추출기 · 공통 벡터 · 허용치 표 · 로더 묶음 · 정답 서술 생성기 · (있으면) 검사기 · 닫힌 목록 · 학습 쪽
추출기 · 프롬프트의 바이트를 다시 재어 적힌 값과 맞대고, 하나라도 다르면 문제로 적는다 — 다른 입력을 받지 않는다. 문제가 없다는 것은
"적힌 파일로 돈다" 까지다. 그 파일들이 옳다는 뜻은 아니다(그것은 벡터 대조 · 회귀 · 확인 사례의 일이다).

## 해시 꼴

| 칸 | 꼴 |
|---|---|
| `extractor_sha256` · `checker_sha256` · `train_extractor_sha256` · `generator_sha256` | 파일의 LF 정규화 바이트(`\\r\\n` → `\\n`) sha256 — 채점 지문과 같은 꼴 |
| `extract_vectors_sha256` · `lists_sha256` · `limits_sha256` · `confirmation_cases_sha256` · `inputs_contract_sha256` | 파일의 원시 바이트 sha256 — 자료 파일 |
| `limits_loader_sha256` | 로더 묶음 — 파일마다 LF 정규화 sha256 을 `경로 + 탭 + sha256 + LF` 줄로, 저장소 상대 경로의 정렬 순으로 이은 바이트의 sha256 |
| `generation.prompt_sha256` | `vlm.pilot_vlm.load_prompt` 와 같은 식 — 글 모드로 읽은 문자열(줄끝 LF)의 UTF-8 sha256 |

## 학습 전 계약 묶음(07 4판 §17 · §18 — 인터페이스 문서 07e)

판 6 은 학습이 끝난 뒤에 발급된다. 그래서 정답 서술을 만든 표 · 로더 · 생성 경로 · 조건 자료와 학습이 읽은 페어를 **학습 전에** 한 파일
(`configs/narrative_inputs.v1.json`)로 묶고, 학습 등록이 그 sha256 을 묶는다. 대조는 셋이다 — 모두 파일을 쓰지 않는 순수 함수이고 빈 목록이면
받음, 줄마다 `<사유 코드>: <설명>` 이다.

- `narrative_inputs_problems` — 학습 등록 검증 · 모델 적재 전 공통 점검이 부른다. 계약 바이트와 등록의 sha, 실제 코드 바이트, 호출자가 이미
  검증한 페어의 사실(`VerifiedPairs`), PAIRS_META 의 신원 블록을 맞댄다.
- `pairs_identity_problems` — PAIRS_META 바이트의 신원 블록 ↔ 계약. `identity.limits` 는 `limits_table` 에서 `{path, sha256}` 만 추출한
  값과 같아야 한다(`status` 는 `limits_table` 에 보존한다).
- `v6_contract_problems` — 판 6 발급 · 내보내기가 부른다. 판 6 의 신원 칸 · `narrative.inputs_contract_sha256` ↔ 넘긴 계약 ↔ 학습 착수 고정의
  `common.registration.narrative_inputs.sha256`. 지금 학습 등록 파일을 다시 읽은 값으로 학습 당시의 기대값을 대신하지 않는다.

메타 문자열끼리 맞대고 끝내지 않는다 — 계약 · PAIRS_META · 착수 고정은 바이트로 받고, 표 · 로더 · 생성 경로는 `repo` 에서 다시 잰다.

## 프로필(판 6 연결 2 판 — 263 §3)

판 6 의 검사기 판 · 표 · 닫힌 목록 · 생성 경로 해시 규칙 · 증거 자료 · 잔여 한계는 **닫힌 조합**(`NARRATIVE_PROFILES`) 둘뿐이다 —
`legacy_v3_1`(구판 재현 · 이 모듈의 옛 동작)과 `regime2_v4`(검사기 v4 · 2 판 표 · 세 파일 묶음 해시 · 19 줄 한계 · (나)(마) · 아홉 출처).
`profile_for` 가 등록 × 학습 등록의 단서로 하나를 고르고 섞임은 거부한다. `profile` 을 주지 않은 호출은 모두 옛 동작이다 — 구판 fixture 와
역사적 판 6 의 바이트 · 사유 코드는 바뀌지 않는다. D3 의 출처 아홉 행은 `{stem}.v6_sources.json`(`build_v6_sources` · `load_v6_sources`)이
운반하며 등록 JSON 의 스키마를 늘리지 않는다.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import NamedTuple

from evaluation.narrative_checker import CHECKER_ID, LIMITS_REL
from evaluation.narrative_conformance import VECTOR_REL, VECTOR_SHA256
from evaluation.narrative_extract import EXTRACTOR_ID
from evaluation.prereg_unified import (
    CLAUSE_AXIS_METRICS,
    NARRATIVE_KINDS,
    NARRATIVE_REASONS,
    REGISTRATION_VERSION_V6,
    VOCAB,
    ClauseAxisSpec,
    NarrativeSpec,
    UnifiedRegistration,
)
from evaluation.provenance import LIMITS_RULE_FILES

REPO = Path(__file__).resolve().parents[1]
EXTRACTOR_REL = "evaluation/narrative_extract.py"
CHECKER_REL = "evaluation/narrative_checker.py"
LIMITS_CSV_REL = LIMITS_REL
"""허용치 표의 경로 — 검사기의 표 핀(`narrative_checker.LIMITS_REL`)을 그대로 쓴다. 두 상수가 갈리면 신원 대조가 채점과 다른 표를 잰다."""
GENERATOR_REL = "corpus/generate/make_pairs_pilot.py"
"""정답 서술(`target_text`)을 만든 생성 경로 — `clause_basis` → `defect_target_text`(193 B-2)."""
LIMITS_LOADER_FILES: tuple[str, ...] = LIMITS_RULE_FILES
"""로더 묶음에 드는 파일 — 채점 지문의 트리 밖 목록에 더한 허용치 규칙 모듈과 같다."""
TRAIN_EXTRACTOR_REL = "vlm/narrative.py"
"""학습 쪽 추출기 — 식별자는 그 파일의 `EXTRACTOR_ID`(구문 트리로 읽는다 — 실행하지 않는다), 해시는 LF sha256."""
HOLES_CHECKER_ID = "narrative_checker_v3_1"
"""아래 두 목록이 속한 검사기 판 — 검사기가 바뀌면 목록도 새로 정한다(그 전에는 신원 대조가 거부한다)."""
OPEN_HOLES: tuple[str, ...] = (
    "r_scope 정답 꼴 문장이 r_dim 까지 채우는 과다 인정",
    "따로 선 '강' + '도'(강철도)의 진짜 쓰임을 잡지 않음",
    "'강의' 동음 — '강 + 의' 로 걸림(199-H3-N10)",
    "H1 · H3 단정 예외를 연결 어미 '고 · 며 · 므로' 로 우회",
    "r_note 조건 1 이 단서의 있음을 보지 않음",
    "(라) 표지의 '경우' 과대 분류",
    "'준용' 의 활용 꼴",
    "이중 부정 창 +12 상수",
    "U+318D 가운뎃점의 형제 자리(_SEP · _TOPIC) — '합격ㆍ불합격' 유보가 (나)로 걸림",
    "r_dim 정보 낱말에 환산 · mm/px 없음",
    "재질 복합어('강판' 등)를 재질 낱말로 받지 않음(199-H3-U01)",
)
"""검사기 v3.1 의 잔여 한계(03 §9-16 의 표 · 04 §7-8 · §7-11 · §7-12 · 199 결정 문서 §4) — 등록 `narrative.open_holes` 는 이 목록이어야
한다. v4(2차 공식 채점 전 별도 회차)가 닫을 것과 이 판의 대가를 함께 적는다."""
EXPLORATORY_KINDS: tuple[str, ...] = ("마",)
"""확인 사례 199 에 음성 오탐이 남은 종류(199-H3-N10 × 다섯 자리) — 그 종류의 금지 주장률은 탐색적이다(199 결정 문서 §2)."""

INPUTS_CONTRACT_KIND = "narrative_inputs"
INPUTS_CONTRACT_VERSION = 1
INPUTS_CONTRACT_REL = "configs/narrative_inputs.v1.json"
"""학습 전 계약 묶음 파일의 자리 — 데이터 쪽 설정이라 2 차 등록 때 둔다. 이 모듈은 그 꼴을 읽기만 한다."""
INPUTS_REGISTRATION_KEY = "narrative_inputs"
"""학습 등록의 칸 `narrative_inputs: {version, sha256}` — 착수 고정의 `common.registration` 에 그대로 실린다."""
BUNDLE_RULE = "lf_bundle_v1"
GENERATOR_FILES_V1: tuple[str, ...] = ("corpus/generate/make_pairs_main.py", "corpus/generate/make_pairs_pilot.py",
                                       "corpus/generate/pairs_text_v2.py")
"""계약 판 1 의 생성 경로 묶음 — 2 판 문장 계약의 정답 서술을 만드는 세 파일(08 §9-4)."""
TEXT_CONTRACT_V2 = "v2"
"""계약 판 1 이 가리키는 정답 서술의 문장 계약 판 — PAIRS_META 의 `text_contract`."""
SEALED_EXPECTED = "expected"
CONTRACT_KEYS = frozenset({"kind", "contract_version", "limits", "limits_loader", "generator", "conditions", "pairs"})
IDENTITY_KEYS = frozenset({"contract_version", "limits", "limits_loader", "generator", "conditions"})
REASON_CODES = ("contract_sha", "contract_form", "hash_rule", "code_bytes", "pairs_unverified", "pairs_digest",
                "pairs_meta_bytes", "pairs_meta_identity", "conditions", "v6_form", "v6_contract_sha", "v6_identity",
                "pin_bytes", "pin_contract", "pin_combo")
"""사유 코드 — 앞의 열넷은 사례 파일 `evaluation/vectors/narrative_inputs_cases.v1.json` 의 목록과 같다. `pin_combo` 는 핀 조합의 사유다."""


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def raw_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_sha256(path: Path) -> str:
    """`vlm.pilot_vlm.load_prompt` 와 같은 식 — 모델이 받는 문자열의 UTF-8 바이트."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def module_str_constant(path: Path, name: str) -> str | None:
    """파이썬 파일의 맨 위 `NAME = "…"` 문자열 상수 — 구문 트리로 읽는다(그 파일을 실행하지 않는다). 없으면 None."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else (
            node.target if isinstance(node, ast.AnnAssign) else None)
        if isinstance(target, ast.Name) and target.id == name and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            return node.value.value
    return None


def limits_loader_bundle_sha256(repo: Path | None = None, files: tuple[str, ...] = LIMITS_LOADER_FILES) -> str:
    """로더 묶음의 해시(04 §7-5 ⑧) — `경로\\tsha256\\n` 줄을 경로 정렬로 이은 바이트의 sha256. 파일이 없으면 멈춘다."""
    root = Path(repo or REPO)
    lines = "".join(f"{rel}\t{lf_sha256(root / rel)}\n" for rel in sorted(files))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def actual_identity(repo: Path | None = None, *, profile: NarrativeProfile | str | None = None) -> dict:
    """지금 트리에서 잰 값 — 등록 블록에 적을 후보이기도 하다.

    `profile` 을 주지 않으면 구판(검사기 v3.1 · 1 판 표 · 닫힌 목록 v3-3 · 생성 경로 한 파일의 LF sha)이다 — 기존 호출과 구판 fixture 의
    옛 동작 보존용. 신규 호출은 `regime2_v4` 를 명시한다(263 §3-2): 검사기 v4 행의 모듈 · 2 판 표 · 닫힌 목록 v4-2, 생성 경로는 세 파일
    묶음의 sha(`generator_bundle_identity`). 어느 쪽이든 문자열 복사가 아니라 실제 파일에서 잰다.
    """
    from data.label_map import load_label_map

    root = Path(repo or REPO)
    p = resolve_profile(profile)
    pin = p.checker_pin
    generator = (generator_bundle_identity(root)["sha256"] if p.generator_rule == BUNDLE_RULE
                 else lf_sha256(root / GENERATOR_REL))
    return {
        "extractor_id": EXTRACTOR_ID,
        "extractor_sha256": lf_sha256(root / EXTRACTOR_REL),
        "extract_vectors_sha256": raw_sha256(root / VECTOR_REL),
        "limits_sha256": raw_sha256(root / pin.limits_rel),
        "limits_loader_sha256": limits_loader_bundle_sha256(root),
        "generator_sha256": generator,
        "label_map_version": load_label_map(root / "configs" / "label_map.yaml").version,
        "checker_id": pin.checker_id,
        "checker_sha256": lf_sha256(root / pin.module),
        "lists_sha256": raw_sha256(root / pin.lists_rel),
    }


def identity_problems(registration: UnifiedRegistration, *, repo: Path | None = None,
                      checker_path: Path | None = None, lists_path: Path | None = None,
                      train_extractor_path: Path | None = None, prompt_path: Path | None = None,
                      confirmation_path: Path | None = None, vectors_path: Path | None = None,
                      inputs_contract_path: Path | None = None, profile: NarrativeProfile | str | None = None,
                      checker_vectors_path: Path | None = None) -> list[str]:
    """등록의 두 블록이 실제 바이트와 맞는가. 문제마다 한 줄 — 비었으면 맞다.

    검사기와 닫힌 목록은 프로필의 정본 파일로 늘 맞대고, 잔여 한계 · 탐색적 종류는 그 프로필의 정본과 늘 맞댄다. `profile` 을 주지 않으면
    구판(`legacy_v3_1` — 검사기 v3.1 · `OPEN_HOLES` · `EXPLORATORY_KINDS`)의 옛 동작 그대로다. `regime2_v4` 를 주면(263 §3-2) 검사기 v4 행 ·
    생성 묶음 · 19 줄 한계 · (나)(마) · 증거 자료 두 칸(검사기 벡터 = 쓰임 탐침 v2 · 확인 사례 v1 — 역할 `evidence_role` · 바이트
    `evidence_bytes`)까지 본다. 선택 인자는 파일이 있을 때 준다 — 검사기 · 목록의 다른 사본, 학습 쪽 추출기(식별자 `EXTRACTOR_ID` 와 LF
    sha256 둘 다), 프롬프트, 확인 사례, 정본 추출 벡터의 다른 사본(실제 바이트), 학습 전 계약 묶음 파일, 검사기 벡터의 다른 사본
    (`checker_vectors_path` — `vectors_path` 는 추출 벡터 전용). 주면 맞대고, 주지 않으면 그 칸은 여기서 보지 않는다 — 무엇을 보았는지는
    부르는 쪽이 적는다. 검사기 판 · 표 · 닫힌 목록의 짝은 고정표의 한 행이어야 한다(`registration_pin_problems`).
    """
    if registration.version != REGISTRATION_VERSION_V6:
        return [f"version: 판 {registration.version} 에는 서술 · 조항 축 블록이 없다 — 판 6 이어야 한다"]
    n, c = registration.narrative, registration.clause_axis
    out: list[str] = []
    if n is None or c is None:
        return [f"{name}: 블록이 없다" for name, b in (("narrative", n), ("clause_axis", c)) if b is None]
    p = resolve_profile(profile)
    pin = p.checker_pin
    root = Path(repo or REPO)
    act = actual_identity(root, profile=p)
    if act["extract_vectors_sha256"] != VECTOR_SHA256:
        out.append(f"벡터 정본 {VECTOR_REL} 의 바이트가 핀({VECTOR_SHA256[:12]}…)과 다르다")
    if act["lists_sha256"] != pin.lists_sha256:
        out.append(f"닫힌 목록 정본 {pin.lists_rel} 의 바이트가 핀({pin.lists_sha256[:12]}…)과 다르다")
    if profile is not None and act["limits_sha256"] != pin.limits_sha256:
        out.append(f"표 정본 {pin.limits_rel} 의 바이트가 핀({pin.limits_sha256[:12]}…)과 다르다")
    bundle_files = ({r["sha256"] for r in generator_bundle_identity(root)["files"]} if p.generator_rule == BUNDLE_RULE
                    else set())
    for name in ("extractor_id", "extractor_sha256", "extract_vectors_sha256", "limits_sha256",
                 "limits_loader_sha256", "generator_sha256", "label_map_version", "checker_id", "checker_sha256",
                 "lists_sha256"):
        if getattr(n, name) == act[name]:
            continue
        if name == "generator_sha256" and getattr(n, name) in bundle_files:
            out.append(f"hash_rule: narrative.generator_sha256 이 묶음이 아니라 한 파일의 LF sha 다 — 프로필 {p.profile_id} 의 생성 경로는 "
                       f"세 파일 묶음({BUNDLE_RULE})")
        else:
            out.append(f"narrative.{name}: 등록 {getattr(n, name)!r} · 실제 {act[name]!r}")
    if c.limits_sha256 != act["limits_sha256"]:
        out.append(f"clause_axis.limits_sha256: 등록 {c.limits_sha256!r} · 실제 {act['limits_sha256']!r}")
    for name, path, fn in (("checker_sha256", checker_path, lf_sha256), ("lists_sha256", lists_path, raw_sha256),
                           ("train_extractor_sha256", train_extractor_path, lf_sha256),
                           ("confirmation_cases_sha256", confirmation_path, raw_sha256),
                           ("extract_vectors_sha256", vectors_path, raw_sha256),
                           ("inputs_contract_sha256", inputs_contract_path, raw_sha256),
                           ("checker_vectors_sha256", checker_vectors_path, raw_sha256)):
        if path is not None and getattr(n, name) != fn(path):
            out.append(f"narrative.{name}: 등록 {getattr(n, name)!r} · 실제 {fn(path)!r}")
    if train_extractor_path is not None:
        tid = module_str_constant(train_extractor_path, "EXTRACTOR_ID")
        if n.train_extractor_id != tid:
            out.append(f"narrative.train_extractor_id: 등록 {n.train_extractor_id!r} · 실제 {tid!r}")
    if profile is None:                                                     # 구판 — 옛 줄 그대로
        if CHECKER_ID != HOLES_CHECKER_ID:
            out.append(f"잔여 한계 목록은 {HOLES_CHECKER_ID} 의 것이다 — 지금 검사기는 {CHECKER_ID}")
        if n.open_holes is None or tuple(n.open_holes) != OPEN_HOLES:      # 빈 칸도 어긋남 — 칸의 있음으로 대조를 대신하지 않는다
            out.append("narrative.open_holes: 검사기 판의 잔여 한계 목록(OPEN_HOLES)과 다르다"
                       + (" — 칸이 비었다" if n.open_holes is None else ""))
        if n.exploratory_kinds is None or tuple(n.exploratory_kinds) != EXPLORATORY_KINDS:
            got = None if n.exploratory_kinds is None else list(n.exploratory_kinds)
            out.append(f"narrative.exploratory_kinds: 등록 {got} · 검사기 판 {list(EXPLORATORY_KINDS)}")
    else:
        out += profile_holes_problems(n, p)
        out += evidence_problems(n, p, root)
        if (n.checker_id, n.limits_sha256, n.lists_sha256) != (pin.checker_id, pin.limits_sha256, pin.lists_sha256):
            out.append(f"profile_mismatch: 판 6 의 검사기 판 · 표 · 닫힌 목록 짝이 프로필 {p.profile_id} 의 행이 아니다 — "
                       f"checker_id {n.checker_id!r}")
    if prompt_path is not None and registration.generation.prompt_sha256 != prompt_sha256(prompt_path):
        out.append(f"generation.prompt_sha256: 등록 {registration.generation.prompt_sha256!r} · "
                   f"실제 {prompt_sha256(prompt_path)!r}")
    out += registration_pin_problems(registration)
    return out


def narrative_block(repo: Path | None = None, *, profile: NarrativeProfile | str | None = None,
                    train_extractor_path: Path | None = None, confirmation_path: Path | None = None,
                    checker_vectors_path: Path | None = None, inputs_contract_path: Path | None = None) -> NarrativeSpec:
    """판 6 서술 블록의 작성기 — 어휘 고정 칸은 `VOCAB` 의 값, 신원 칸은 지금 트리의 실제 바이트, 잔여 한계 · 탐색적 종류는 프로필의 정본.

    `profile` 을 주지 않으면 구판(`legacy_v3_1`)의 옛 동작이다 — 학습 쪽 추출기는 경로를 주지 않으면 트리의 `TRAIN_EXTRACTOR_REL`, 확인 사례 ·
    검사기 벡터 · 학습 전 계약 묶음 파일은 경로를 줄 때만 채운다(주지 않으면 빈 칸이고 등록이 완결되지 않아 거부된다 — 값을 지어 넣지 않는다).
    `regime2_v4` 를 주면(263 §4-1) 세 자료의 경로를 **따로 받지 않고** 프로필 정본(쓰임 탐침 v2 · 확인 사례 v1 · `INPUTS_CONTRACT_REL`)에서
    읽어 바이트를 핀과 맞댄 뒤에 적는다 — 계약은 실물 파일 대조(`contract_code_problems`)까지 지난다. 임의 문자열 sha 를 받는 입구는 없다.
    """
    root = Path(repo or REPO)
    p = resolve_profile(profile)
    if profile is not None and p.evidence_roles:
        given = [k for k, v in (("confirmation_path", confirmation_path), ("checker_vectors_path", checker_vectors_path),
                                ("inputs_contract_path", inputs_contract_path)) if v is not None]
        if given:
            raise ProfileRefused("profile_paths", f"프로필 {p.profile_id} 의 작성은 {given} 를 따로 받지 않는다 — 프로필 정본에서 읽는다")
        roles = {r.role: r for r in p.evidence_roles}
        for role in ("confirmation_cases", "checker_vectors"):
            f = root / roles[role].path
            if not f.is_file() or raw_sha256(f) != roles[role].sha256:
                raise ProfileRefused("evidence_bytes", f"{role} 의 정본 {roles[role].path} 가 없거나 바이트가 핀"
                                                       f"({roles[role].sha256[:12]}…)과 다르다")
        contract = root / INPUTS_CONTRACT_REL
        if not contract.is_file():
            raise ProfileRefused("v6_form", f"학습 전 계약 파일이 없다: {INPUTS_CONTRACT_REL}")
        problems = contract_code_problems(contract.read_bytes(), repo=root)
        if problems:
            raise ProfileRefused(problems[0].split(":", 1)[0], problems)
        confirmation_path, checker_vectors_path = root / roles["confirmation_cases"].path, root / roles["checker_vectors"].path
        inputs_contract_path = contract
    tex = Path(train_extractor_path) if train_extractor_path is not None else root / TRAIN_EXTRACTOR_REL
    vocab = {f.name: VOCAB[f"narrative.{f.name}"][0] for f in fields(NarrativeSpec) if f"narrative.{f.name}" in VOCAB}
    return NarrativeSpec(**vocab, **actual_identity(root, profile=p), train_extractor_id=module_str_constant(tex, "EXTRACTOR_ID"),
                         train_extractor_sha256=lf_sha256(tex),
                         checker_vectors_sha256=None if checker_vectors_path is None else raw_sha256(checker_vectors_path),
                         confirmation_cases_sha256=None if confirmation_path is None else raw_sha256(confirmation_path),
                         inputs_contract_sha256=None if inputs_contract_path is None else raw_sha256(inputs_contract_path),
                         kinds=NARRATIVE_KINDS, reasons=NARRATIVE_REASONS, open_holes=p.open_holes,
                         exploratory_kinds=p.exploratory_kinds)


def clause_axis_block(repo: Path | None = None, *, profile: NarrativeProfile | str | None = None) -> ClauseAxisSpec:
    """판 6 조항 축 블록의 작성기 — 어휘 고정 칸(가중 `design_and_unweighted` · 대응 구간
    `paired_cluster_bootstrap_aux_cells_and_baseline` — 같은 W 에서 모델 짝과 모델 − 대조선 포함 · 복수 조항 정책
    `primary_one_auxiliary_breakdown`)과 허용치 표의 바이트 sha256. 표는 프로필의 표 핀(`profile` 없음 = 구판 1 판 표)이다."""
    root = Path(repo or REPO)
    vocab = {f.name: VOCAB[f"clause_axis.{f.name}"][0] for f in fields(ClauseAxisSpec) if f"clause_axis.{f.name}" in VOCAB}
    return ClauseAxisSpec(**vocab, metrics=CLAUSE_AXIS_METRICS,
                          limits_sha256=raw_sha256(root / resolve_profile(profile).checker_pin.limits_rel))


# ════════════════════════════════════════════════════════════════════ 학습 전 계약 묶음

@dataclass(frozen=True)
class VerifiedPairs:
    """학습 전 계약 대조에 호출자가 넘기는, **이미 검증한** 페어의 사실(07 §17-3). 함수는 페어 파일을 스스로 열지 않는다.

    한 칸이라도 `None` 이면 필수 검증 사실의 누락이다 — 검증을 수행한 뒤의 빈 문제 목록(`snapshot_problems=()`)과 다르다.
    `sealed_status` 의 `None` 은 봉인 명부에 그 폴더가 없다는 뜻으로도 쓴다(둘 다 받지 않는다).
    """

    path: str | None
    """학습이 실제로 읽을 페어 폴더 — 저장소 상대 posix(봉인 명부의 열쇠와 같은 꼴)."""
    snapshot_digest: str | None
    """그 폴더 `SNAPSHOT.sha256` 의 `snapshot_digest`."""
    snapshot_problems: tuple[str, ...] | None
    """`vlm.uni_config.pairs_snapshot_problems` 의 결과 — 비어야 한다."""
    sealed_status: str | None
    """`data.frozen_guard.EXPECTED_SEALED` 에서 그 폴더의 `status`."""
    pairs_meta_raw: bytes | None
    """그 폴더 `PAIRS_META.json` 의 바이트."""
    snapshot_pairs_meta_sha256: str | None
    """`SNAPSHOT.sha256` 에 적힌 `PAIRS_META.json` 의 sha256."""
    config_pairs_digest: str | None
    """설정의 페어 digest."""
    token_table_pairs_digest: str | None
    """행별 토큰 표 출처의 페어 digest."""


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _hex64(v) -> bool:
    return isinstance(v, str) and bool(_HEX64.match(v))


def _rel_posix(v) -> bool:
    """저장소 상대 posix 경로 — 빈 마디 · `.` · `..` · 역슬래시 · 드라이브 문자 · 앞 `/` 가 없다."""
    if not isinstance(v, str) or not v or "\\" in v or v.startswith("/") or re.match(r"^[A-Za-z]:", v):
        return False
    return all(part not in ("", ".", "..") for part in v.split("/"))


def _sha(b: bytes) -> str:
    return hashlib.sha256(bytes(b)).hexdigest()


def _rows_sha(rows: Sequence[Mapping]) -> str:
    return _sha("".join(f"{r['path']}\t{r['sha256']}\n" for r in rows).encode("utf-8"))


def _json_object(raw: bytes | None) -> dict | None:
    if raw is None:
        return None
    try:
        obj = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def bundle_block(repo: Path | None, files: Sequence[str]) -> dict:
    """lf_bundle_v1 묶음 블록 — `files` = 경로 정렬 `{path, sha256}`(파일마다 LF sha256), `sha256` = 그 줄을 이은 바이트의 sha256.

    블록의 `sha256` 은 같은 파일 목록의 `limits_loader_bundle_sha256` 과 같다. 파일마다의 sha 를 함께 싣는 까닭은 실제 바이트가 다를 때
    어느 파일인지까지 말하게 하려는 것이다(07 §17-3).
    """
    root = Path(repo or REPO)
    rows = [{"path": rel, "sha256": lf_sha256(root / rel)} for rel in sorted(files)]
    return {"rule": BUNDLE_RULE, "files": rows, "sha256": _rows_sha(rows)}


def narrative_inputs_block(repo: Path | None = None, *, limits_rel: str, pairs_path: str, snapshot_digest: str,
                           pairs_meta_raw: bytes, loader_files: Sequence[str] = LIMITS_LOADER_FILES,
                           generator_files: Sequence[str] = GENERATOR_FILES_V1) -> dict:
    """학습 전 계약 묶음(판 1)의 작성기 — 표 · 로더 · 생성 경로는 `repo` 의 실제 바이트에서, PAIRS_META 의 sha 는 넘긴 바이트에서 잰다.

    손으로 옮기지 않는다. 파일을 쓰지 않는다 — 계약 파일을 두는 것은 2 차 등록이다(`dump_narrative_inputs` 의 바이트로).
    """
    root = Path(repo or REPO)
    return {"kind": INPUTS_CONTRACT_KIND, "contract_version": INPUTS_CONTRACT_VERSION,
            "limits": {"path": limits_rel, "sha256": raw_sha256(root / limits_rel)},
            "limits_loader": bundle_block(root, loader_files), "generator": bundle_block(root, generator_files),
            "conditions": {"in_table": True},
            "pairs": {"path": pairs_path, "snapshot_digest": snapshot_digest, "pairs_meta_sha256": _sha(pairs_meta_raw)}}


def dump_narrative_inputs(contract: Mapping) -> bytes:
    """계약의 정규 바이트 — 키 정렬 · UTF-8 · 들여쓰기 1 · 끝 LF. 대조 함수는 이 꼴만 받는다."""
    return (json.dumps(contract, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def _closed(obj, keys: set[str]) -> bool:
    return isinstance(obj, dict) and set(obj) == keys


def _bundle_form(name: str, block, code: str) -> list[str]:
    """묶음 블록의 꼴. 꼴의 잘못은 `code`(계약이면 contract_form), 규칙의 섞임은 hash_rule."""
    if not _closed(block, {"rule", "files", "sha256"}):
        return [f"{code}: {name} 은 {{rule, files, sha256}} 의 닫힌 객체여야 한다"]
    out: list[str] = []
    if block["rule"] != BUNDLE_RULE:
        out.append(f"hash_rule: {name}.rule {block['rule']!r} — 계약 판 {INPUTS_CONTRACT_VERSION} 의 묶음 규칙은 {BUNDLE_RULE}")
    files = block["files"]
    rows_ok = isinstance(files, list) and bool(files) and all(
        _closed(r, {"path", "sha256"}) and _rel_posix(r["path"]) and _hex64(r["sha256"]) for r in files)
    if not rows_ok:
        out.append(f"{code}: {name}.files 는 비지 않은 {{path, sha256}} 목록이어야 한다(저장소 상대 posix · 소문자 16진 64 자)")
    elif [r["path"] for r in files] != sorted({r["path"] for r in files}):
        out.append(f"{code}: {name}.files 가 경로 정렬이 아니거나 겹친다")
    if not _hex64(block["sha256"]):
        out.append(f"{code}: {name}.sha256 이 소문자 16진 64 자가 아니다")
    elif rows_ok:
        if block["sha256"] in {r["sha256"] for r in files}:
            out.append(f"hash_rule: {name}.sha256 이 묶음이 아니라 그 목록의 한 파일 sha 다(신구 해시 꼴 섞임)")
        elif block["sha256"] != _rows_sha(files):
            out.append(f"{code}: {name}.sha256 이 그 files 에서 낸 묶음 sha 와 다르다")
    return out


def _contract_form(contract) -> list[str]:
    """계약의 꼴(단계 ②) — contract_form · conditions · 꼴로 보이는 hash_rule."""
    if not isinstance(contract, dict):
        return ["contract_form: 계약이 JSON 객체가 아니다"]
    out: list[str] = []
    keys = set(contract)
    missing, unknown = sorted(CONTRACT_KEYS - keys), sorted(keys - CONTRACT_KEYS)
    if missing:
        out.append(f"contract_form: 칸이 빠졌다 {missing}")
    if unknown:
        out.append(f"contract_form: 모르는 칸 {unknown}")
    if "conditions" in missing:
        out.append("conditions: 조건 자료 칸을 생략했다 — 표 안의 조건이어도 {in_table: true} 를 적는다")
    if "kind" in keys and contract["kind"] != INPUTS_CONTRACT_KIND:
        out.append(f"contract_form: kind {contract['kind']!r} — {INPUTS_CONTRACT_KIND!r} 이어야 한다")
    if "contract_version" in keys and (type(contract["contract_version"]) is not int
                                       or contract["contract_version"] != INPUTS_CONTRACT_VERSION):
        out.append(f"contract_form: 모르는 계약 판 {contract['contract_version']!r} — 이 코드는 판 {INPUTS_CONTRACT_VERSION} 을 읽는다")
    if "limits" in keys:
        lim = contract["limits"]
        if not (_closed(lim, {"path", "sha256"}) and _rel_posix(lim["path"]) and _hex64(lim["sha256"])):
            out.append("contract_form: limits 는 {path, sha256} 의 닫힌 객체(저장소 상대 posix · 소문자 16진 64 자)여야 한다")
    for name in ("limits_loader", "generator"):
        if name in keys:
            out += _bundle_form(name, contract[name], "contract_form")
    if "conditions" in keys:
        cond = contract["conditions"]
        if _closed(cond, {"in_table"}) and cond["in_table"] is True:
            pass
        elif _closed(cond, {"path", "sha256"}):
            out.append("conditions: 따로 둔 조건 파일의 꼴이다 — 이 계약 판은 조건을 표 안의 칸으로 둔다({in_table: true})")
        elif _closed(cond, {"in_table"}):
            out.append(f"conditions: in_table {cond['in_table']!r} — 참이어야 한다")
        else:
            out.append("contract_form: conditions 의 꼴이 틀리다 — {in_table: true}")
    if "pairs" in keys:
        p = contract["pairs"]
        if not (_closed(p, {"path", "snapshot_digest", "pairs_meta_sha256"}) and _rel_posix(p["path"])
                and _hex64(p["snapshot_digest"]) and _hex64(p["pairs_meta_sha256"])):
            out.append("contract_form: pairs 는 {path, snapshot_digest, pairs_meta_sha256} 의 닫힌 객체(저장소 상대 posix · "
                       "소문자 16진 64 자)여야 한다")
    return out


def _canonical_problem(contract: dict, raw: bytes) -> list[str]:
    if dump_narrative_inputs(contract) != bytes(raw):
        return ["contract_form: 정규형이 아니다 — dump_narrative_inputs 가 적은 바이트만 받는다(손으로 고친 파일)"]
    return []


def _contract_bytes(contract: dict, root: Path) -> list[str]:
    """실제 바이트(단계 ③) — 표의 원시 sha · 묶음의 파일 목록과 파일마다의 LF sha."""
    out: list[str] = []
    lim = contract["limits"]
    table = root / lim["path"]
    if not table.is_file():
        out.append(f"code_bytes: 표 {lim['path']} 가 없다")
    else:
        raw, lf = raw_sha256(table), lf_sha256(table)
        if raw != lim["sha256"]:
            if lim["sha256"] == lf:
                out.append("hash_rule: limits.sha256 이 원시 바이트가 아니라 LF 정규화 sha 다(원시 ↔ LF 섞임)")
            else:
                out.append(f"code_bytes: 표 {lim['path']} 의 원시 sha {raw[:12]}… ≠ 계약 {lim['sha256'][:12]}…")
    for name, want in (("limits_loader", LIMITS_LOADER_FILES), ("generator", GENERATOR_FILES_V1)):
        rows = contract[name]["files"]
        listed = [r["path"] for r in rows]
        if listed != sorted(want):
            out.append(f"code_bytes: {name} 의 파일 목록이 계약 판 {INPUTS_CONTRACT_VERSION} 의 목록과 다르다 — 계약 {listed} · "
                       f"판 {sorted(want)}")
            continue
        for r in rows:
            f = root / r["path"]
            if not f.is_file():
                out.append(f"code_bytes: {name} 의 {r['path']} 가 없다")
            elif lf_sha256(f) != r["sha256"]:
                out.append(f"code_bytes: {name} 의 {r['path']} — 실제 LF sha {lf_sha256(f)[:12]}… ≠ 계약 {r['sha256'][:12]}…")
    return out


_VERIFIED_TYPES = {"path": str, "snapshot_digest": str, "snapshot_problems": (tuple, list), "pairs_meta_raw": bytes,
                   "snapshot_pairs_meta_sha256": str, "config_pairs_digest": str, "token_table_pairs_digest": str}


def _pairs_facts(contract: dict, pairs: VerifiedPairs) -> list[str]:
    """호출자가 검증한 사실(단계 ④) — pairs_unverified · pairs_digest · pairs_meta_bytes."""
    out: list[str] = []
    lost = [n for n in _VERIFIED_TYPES if getattr(pairs, n, None) is None]
    if lost:
        out.append(f"pairs_unverified: 필수 검증 사실이 빠졌다 {lost} — 검증을 수행한 뒤의 빈 목록과 다르다")
    bad = [n for n, t in _VERIFIED_TYPES.items() if getattr(pairs, n, None) is not None and not isinstance(getattr(pairs, n), t)]
    if bad:
        out.append(f"pairs_unverified: 검증 사실의 꼴이 틀리다 {bad}")
    sp = pairs.snapshot_problems
    if isinstance(sp, (tuple, list)) and sp:
        out.append(f"pairs_unverified: 스냅샷 검사에 문제가 {len(sp)} 개 남았다")
    if pairs.sealed_status != SEALED_EXPECTED:
        out.append(f"pairs_unverified: 봉인 명부의 상태가 {pairs.sealed_status!r} 다 — {SEALED_EXPECTED!r} 여야 한다(없으면 None)")
    want = contract["pairs"]
    if isinstance(pairs.path, str) and pairs.path != want["path"]:
        out.append(f"pairs_unverified: 실제로 읽을 페어 경로 {pairs.path!r} ≠ 계약 {want['path']!r}")
    for name in ("snapshot_digest", "config_pairs_digest", "token_table_pairs_digest"):
        v = getattr(pairs, name)
        if isinstance(v, str) and v != want["snapshot_digest"]:
            out.append(f"pairs_digest: {name} {v[:12]}… ≠ 계약의 snapshot_digest {want['snapshot_digest'][:12]}…")
    if isinstance(pairs.pairs_meta_raw, bytes):
        got = _sha(pairs.pairs_meta_raw)
        if isinstance(pairs.snapshot_pairs_meta_sha256, str) and got != pairs.snapshot_pairs_meta_sha256:
            out.append("pairs_meta_bytes: PAIRS_META 바이트의 sha256 ≠ 스냅샷 계약에 적힌 값")
        if got != want["pairs_meta_sha256"]:
            out.append("pairs_meta_bytes: PAIRS_META 바이트의 sha256 ≠ 계약의 pairs_meta_sha256")
    return out


def pairs_identity_problems(pairs_meta_raw: bytes, contract: Mapping) -> list[str]:
    """PAIRS_META 바이트의 신원 블록 ↔ 학습 전 계약. 줄은 모두 `pairs_meta_identity: …` 다.

    `limits_table` 의 기존 `status` 를 보존한다. `identity.limits` 는 `{path, sha256}` 의 닫힌 객체이며, 기존 `limits_table` 에서 그
    두 필드만 추출한 값과 같아야 한다(I-02 — 08 §9-4 와 같은 글자). 묶음 칸은 `files` 의 경로 목록부터 맞대고, 목록이 다르면 해시를 보기
    전에 거부한다.
    """
    tag = "pairs_meta_identity"
    meta = _json_object(pairs_meta_raw)
    if meta is None:
        return [f"{tag}: PAIRS_META 바이트가 JSON 객체가 아니다"]
    out: list[str] = []
    if meta.get("text_contract") != TEXT_CONTRACT_V2:
        out.append(f"{tag}: text_contract {meta.get('text_contract')!r} — 계약 판 {INPUTS_CONTRACT_VERSION} 의 문장 계약은 "
                   f"{TEXT_CONTRACT_V2!r}")
    lt = meta.get("limits_table")
    if not (isinstance(lt, dict) and {"path", "sha256"} <= set(lt)):
        out.append(f"{tag}: limits_table 에 path · sha256 이 없다")
        lt = None
    ident = meta.get("identity")
    if not isinstance(ident, dict):
        out.append(f"{tag}: 신원 블록(identity)이 없다 — 2 판 메타가 아니다")
        return out
    missing, unknown = sorted(IDENTITY_KEYS - set(ident)), sorted(set(ident) - IDENTITY_KEYS)
    if missing:
        out.append(f"{tag}: identity 의 칸이 빠졌다 {missing}")
    if unknown:
        out.append(f"{tag}: identity 에 모르는 칸 {unknown}")
    cv = ident.get("contract_version")
    if "contract_version" in ident and (type(cv) is not int or cv != contract.get("contract_version")):
        out.append(f"{tag}: identity.contract_version {cv!r} ≠ 계약 {contract.get('contract_version')!r}")
    if "limits" in ident:
        lim = ident["limits"]
        if not _closed(lim, {"path", "sha256"}):
            got = sorted(lim) if isinstance(lim, dict) else type(lim).__name__
            out.append(f"{tag}: identity.limits 는 {{path, sha256}} 의 닫힌 객체여야 한다 — 받은 칸 {got}")
        else:
            if lt is not None and lim["path"] != lt["path"]:
                out.append(f"{tag}: identity.limits.path ≠ limits_table.path")
            if lt is not None and lim["sha256"] != lt["sha256"]:
                out.append(f"{tag}: identity.limits.sha256 ≠ limits_table.sha256")
            if lim != contract.get("limits"):
                out.append(f"{tag}: identity.limits ≠ 계약의 limits")
    for name in ("limits_loader", "generator"):
        if name not in ident:
            continue
        got, want = ident[name], contract.get(name) or {}
        if not _closed(got, {"rule", "files", "sha256"}) or not isinstance(got["files"], list) \
                or not all(isinstance(r, dict) for r in got["files"]):
            out.append(f"{tag}: identity.{name} 는 {{rule, files, sha256}} 의 닫힌 묶음 블록이어야 한다")
            continue
        if got["rule"] != BUNDLE_RULE:
            out.append(f"{tag}: identity.{name}.rule {got['rule']!r} — {BUNDLE_RULE}")
        if [r.get("path") for r in got["files"]] != [r.get("path") for r in want.get("files") or []]:
            out.append(f"{tag}: identity.{name} 의 파일 목록이 계약과 다르다 — 해시는 보지 않았다")
            continue
        if got["files"] != want.get("files") or got["sha256"] != want.get("sha256"):
            out.append(f"{tag}: identity.{name} 의 sha 가 계약과 다르다")
    if "conditions" in ident:
        cond = ident["conditions"]
        if not (_closed(cond, {"in_table"}) and cond["in_table"] is True) or cond != contract.get("conditions"):
            out.append(f"{tag}: identity.conditions 는 {{in_table: true}} 이고 계약과 같아야 한다")
    return out


def narrative_inputs_problems(contract_raw: bytes, *, registered_sha256: str | None, repo: Path,
                              pairs: VerifiedPairs) -> list[str]:
    """학습 전 계약 대조 — 학습 등록 검증과 모델 적재 전 공통 점검이 부른다(07 §17-3 · §17-4). 빈 목록이면 받음.

    단계: ① 학습 등록이 묶은 sha ↔ 계약 바이트(어긋나면 멈춘다) → ② 계약의 꼴(잘못이면 멈춘다) → ③ 표 · 로더 · 생성 경로의 실제 바이트
    (어긋나면 멈춘다 — 실제 코드와 맞지 않는 계약으로 페어를 맞대지 않는다) → ④ 호출자가 검증한 페어의 사실 → ⑤ PAIRS_META 의 신원 블록.
    `repo` 는 학습이 쓸 코드 트리의 뿌리다 — 연합 서버 길에서는 실행 설정의 `uni-repo-root`(FAB 에 `corpus/**` 가 없다).
    """
    raw = bytes(contract_raw)
    if not _hex64(registered_sha256):
        return [f"contract_sha: 학습 등록의 계약 sha {registered_sha256!r} 가 소문자 16진 64 자가 아니다 — 등록이 계약을 묶지 않았다"]
    if _sha(raw) != registered_sha256:
        return [f"contract_sha: 학습 등록이 묶은 sha256 {registered_sha256[:12]}… ≠ 계약 바이트 {_sha(raw)[:12]}…"]
    contract = _json_object(raw)
    if contract is None:
        return ["contract_form: 계약 바이트가 JSON 객체가 아니다"]
    form = _contract_form(contract) + _canonical_problem(contract, raw)
    if form:
        return form
    code = _contract_bytes(contract, Path(repo))
    if code:
        return code
    out = _pairs_facts(contract, pairs)
    if isinstance(pairs.pairs_meta_raw, bytes):
        out += pairs_identity_problems(pairs.pairs_meta_raw, contract)
    return out


def v6_contract_problems(registration: UnifiedRegistration, contract_raw: bytes, *, pin_raw: bytes,
                         adapter_pin_sha256: Sequence[str]) -> list[str]:
    """판 6 ↔ 학습 전 계약 ↔ 학습 착수 고정(07 §17-6). 판 6 발급 · 내보내기 사전 점검이 부른다. 빈 목록이면 받음.

    학습 당시의 계약 sha 는 착수 고정의 `common.registration.narrative_inputs.sha256` 에서 읽는다 — 지금 학습 등록 파일을 다시 읽은 값으로
    대신하지 않는다. 착수 고정 바이트의 sha256 은 다섯 칸 어댑터 meta 의 `start_gate.pin_sha256`(`adapter_pin_sha256`)과 같아야 한다.
    단계: ① 판 6 의 꼴 → ② 넘긴 계약의 꼴(①② 는 잘못이면 멈춘다) → ③ 착수 고정 → ④ 판 6 의 계약 sha → ⑤ 판 6 의 신원 칸.
    """
    if registration.version != REGISTRATION_VERSION_V6:
        return [f"v6_form: 판 {registration.version} 이다 — 판 6 이어야 한다"]
    n, c = registration.narrative, registration.clause_axis
    if n is None or c is None:
        return ["v6_form: 판 6 의 서술 · 조항 축 블록이 없다"]
    if n.inputs_contract_sha256 is None:
        return ["v6_form: narrative.inputs_contract_sha256 이 비었다 — 학습 전 계약에 묶이지 않은 판 6 이다"]
    if not _hex64(n.inputs_contract_sha256):
        return ["v6_form: narrative.inputs_contract_sha256 이 소문자 16진 64 자가 아니다"]
    raw = bytes(contract_raw)
    contract = _json_object(raw)
    if contract is None:
        return ["contract_form: 넘긴 계약 바이트가 JSON 객체가 아니다"]
    form = _contract_form(contract) + _canonical_problem(contract, raw)
    if form:
        return form
    out: list[str] = []
    csha, psha = _sha(raw), _sha(pin_raw)
    adapters = tuple(adapter_pin_sha256 or ())
    if not adapters:
        out.append("pin_bytes: 어댑터 meta 의 start_gate.pin_sha256 을 받지 않았다")
    elif any(a != psha for a in adapters):
        out.append(f"pin_bytes: 착수 고정 바이트의 sha {psha[:12]}… ≠ 어댑터 meta 의 start_gate.pin_sha256 "
                   f"({sum(a != psha for a in adapters)}/{len(adapters)} 칸)")
    pin = _json_object(pin_raw)
    if pin is None:
        out.append("pin_contract: 착수 고정 바이트가 JSON 객체가 아니다")
    else:
        reg = (pin.get("common") or {}).get("registration") if isinstance(pin.get("common"), dict) else None
        ni = reg.get(INPUTS_REGISTRATION_KEY) if isinstance(reg, dict) else None
        if not isinstance(ni, dict):
            out.append("pin_contract: 착수 고정의 common.registration.narrative_inputs 가 없다 — 학습이 계약에 묶이지 않았다")
        else:
            if type(ni.get("version")) is not int or ni.get("version") != contract["contract_version"]:
                out.append(f"pin_contract: 학습 등록의 narrative_inputs.version {ni.get('version')!r} ≠ 계약 판 "
                           f"{contract['contract_version']!r}")
            if ni.get("sha256") != csha:
                out.append("pin_contract: 학습 착수 고정의 계약 sha ≠ 넘긴 계약 바이트 — 학습 당시의 계약이 아니다")
    if n.inputs_contract_sha256 != csha:
        out.append(f"v6_contract_sha: narrative.inputs_contract_sha256 {n.inputs_contract_sha256[:12]}… ≠ 넘긴 계약 {csha[:12]}…")
    want = contract["limits"]["sha256"]
    for label, got in (("narrative.limits_sha256", n.limits_sha256), ("clause_axis.limits_sha256", c.limits_sha256)):
        if got != want:
            out.append(f"v6_identity: {label} ≠ 계약의 limits.sha256")
    for label, got, name in (("narrative.limits_loader_sha256", n.limits_loader_sha256, "limits_loader"),
                             ("narrative.generator_sha256", n.generator_sha256, "generator")):
        block = contract[name]
        if got != block["sha256"]:
            if got in {r["sha256"] for r in block["files"]}:
                out.append(f"hash_rule: {label} 이 한 파일의 sha 다 — 계약의 {name} 은 묶음({BUNDLE_RULE})")
            else:
                out.append(f"v6_identity: {label} ≠ 계약의 {name}.sha256")
    return out


# ════════════════════════════════════════════════════════════════════ 핀 조합(07 §19-1)

class CheckerPin(NamedTuple):
    """검사기 모듈 하나의 고정 조합 — 판 · 표 핀 · 닫힌 목록."""

    module: str
    checker_id: str
    limits_rel: str
    limits_sha256: str
    lists_rel: str
    lists_sha256: str


CHECKER_PINS: tuple[CheckerPin, ...] = (
    CheckerPin("evaluation/narrative_checker.py", "narrative_checker_v3_1", "corpus/rules/limits_v0_pilot.csv",
               "334c78811677accb86a2ec14dad24716e16add3b6a14b47abe33ca6fe9e0cccb",
               "evaluation/vectors/narrative_lists.v3-3.json",
               "c0a4be1055b84091a823a8eeac12e35efd6a30258d0b52daae812ec15df1e77e"),
    CheckerPin("evaluation/narrative_checker_v4.py", "narrative_checker_v4", "corpus/rules/limits_v2_pilot.csv",
               "0c76b6a639fc2d617115122fbc10204fcc0ccf665e9b4c6d27f679057294ef4d",
               "evaluation/vectors/narrative_lists.v4-2.json",
               "651b7d6ca7d55f07ede6894c92fb111abd6ecb2a4e5f619daac7f8bb9dcc4709"),
)
"""검사기 판 · 표 핀 · 닫힌 목록의 고정 조합. 한 경로가 같다는 사실만으로 내용이 호환되지 않는다 — 표 sha 만 바꾸고 판 · 목록을 그대로 둔
변경은 여기서 멈춘다. 구판 재현용(v3.1 파일 · 그 상수 · 닫힌 목록 v3-3)은 새 표로 바꾸지 않는다. 다음 판은 새 행으로 더한다."""

_PIN_CONSTANTS = ("CHECKER_ID", "LIMITS_REL", "LIMITS_SHA256", "LISTS_REL", "LISTS_SHA256")


def checker_pin_problems(repo: Path | None = None) -> list[str]:
    """검사기 모듈마다 상수(구문 트리로 읽는다 — 실행하지 않는다)가 고정표의 한 행인가, 닫힌 목록의 출처 표가 그 행의 표 핀인가."""
    root = Path(repo or REPO)
    out: list[str] = []
    pinned = {p.module for p in CHECKER_PINS}
    for path in sorted((root / "evaluation").glob("narrative_checker*.py")):
        rel = path.relative_to(root).as_posix()
        if rel not in pinned:
            out.append(f"pin_combo: 검사기 모듈 {rel} 이 고정표에 없다")
    for pin in CHECKER_PINS:
        module = root / pin.module
        if not module.is_file():
            out.append(f"pin_combo: 고정표의 검사기 모듈 {pin.module} 이 없다")
            continue
        want = dict(zip(_PIN_CONSTANTS, (pin.checker_id, pin.limits_rel, pin.limits_sha256, pin.lists_rel, pin.lists_sha256),
                        strict=True))
        for name, value in want.items():
            got = module_str_constant(module, name)
            if got != value:
                out.append(f"pin_combo: {pin.module} 의 {name} {got!r} ≠ 고정표 {value!r}")
        lists = root / pin.lists_rel
        if not lists.is_file():
            out.append(f"pin_combo: 닫힌 목록 {pin.lists_rel} 이 없다")
        else:
            if raw_sha256(lists) != pin.lists_sha256:
                out.append(f"pin_combo: 닫힌 목록 {pin.lists_rel} 의 바이트가 고정표와 다르다")
            src = _json_object(lists.read_bytes()) or {}
            csv = (src.get("sources") or {}).get("limits_csv") or {}
            if (csv.get("path"), csv.get("sha256")) != (pin.limits_rel, pin.limits_sha256):
                out.append(f"pin_combo: 닫힌 목록의 출처 표 {csv.get('sha256')!r} ≠ 그 행의 표 핀 {pin.limits_sha256!r}")
        table = root / pin.limits_rel
        if not table.is_file() or raw_sha256(table) != pin.limits_sha256:
            out.append(f"pin_combo: 표 {pin.limits_rel} 의 바이트가 고정표의 표 핀과 다르다")
    return out


def registration_pin_problems(registration: UnifiedRegistration) -> list[str]:
    """판 6 의 (`narrative.checker_id`, `narrative.limits_sha256`, `narrative.lists_sha256`)가 고정표의 한 행이어야 한다."""
    n = registration.narrative
    if n is None:
        return []
    rows = {(p.checker_id, p.limits_sha256, p.lists_sha256) for p in CHECKER_PINS}
    if (n.checker_id, n.limits_sha256, n.lists_sha256) not in rows:
        return [f"pin_combo: 판 6 의 검사기 판 · 표 · 닫힌 목록 짝이 고정표의 한 행이 아니다 — checker_id {n.checker_id!r}"]
    return []


# ════════════════════════════════════════════════════════════════════ 프로필(263 §3-2 · §3-3 · §4-1) — 판 6 의 닫힌 조합

class ProfileRefused(ValueError):
    """프로필 · 증거 · 출처 대조의 거부 — `code` 가 안정된 사유 코드, `problems` 가 설명 줄. 설명 문장과 코드는 구분한다."""

    def __init__(self, code: str, problems: Sequence[str] | str):
        self.code = code
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__(f"[{code}] " + " · ".join(self.problems))


class EvidenceRole(NamedTuple):
    """D3 출처 운반 파일의 한 행 — 역할마다 유일, 해시는 원본 바이트(`raw_sha256`)에서."""

    role: str
    path: str
    sha256: str
    hash_rule: str = "raw_sha256"


class NarrativeProfile(NamedTuple):
    """판 6 의 검사기 판 · 표 · 목록(`checker_pin` — `CHECKER_PINS` 의 행을 참조한다) · 생성 경로 해시 규칙 · 증거 자료 · 잔여 한계 ·
    탐색적 종류의 닫힌 조합. 외부 입력으로 모듈 경로를 받지 않는다 — 두 식별자뿐이다."""

    profile_id: str
    checker_pin: CheckerPin
    generator_rule: str
    checker_vectors_rel: str | None
    confirmation_rel: str | None
    open_holes: tuple[str, ...]
    exploratory_kinds: tuple[str, ...]
    evidence_roles: tuple[EvidenceRole, ...]


GENERATOR_RULE_SINGLE = "lf_single_v1"
"""구판의 생성 경로 해시 — `make_pairs_pilot.py` 한 파일의 LF sha. 신판은 `BUNDLE_RULE`(세 파일 묶음)."""
PROFILE_LEGACY = "legacy_v3_1"
PROFILE_REGIME2 = "regime2_v4"
HOLES_SOURCE_REL = "docs/dev_log/2026-10-20-설계복구/10_검사기v4_미니스펙_D.md"
HOLES_SOURCE_SHA256 = "c996555937ee61120d4963d4ff56a9d1da2aa084f30711bd89528efdea038f81"
HOLES_SOURCE_SECTION = "15"
"""v4 잔여 한계 19 줄 · 탐색적 종류 (나)(마)의 정본 — 그 문서 9 판 §15 의 확정 표(raw sha 고정). 아래 상수는 그 표를 옮긴 것이며 시험이
문서에서 다시 뽑아 맞댄다. 줄을 순화 · 축약하거나 옛 11 줄을 붙이면 `profile_holes` 다."""
OPEN_HOLES_V4: tuple[str, ...] = (
    "r_scope 정답 꼴 문장이 r_dim 까지 채우는 과다 인정",
    "따로 선 '강' + '도'(강철도)의 진짜 쓰임을 잡지 않음",
    "강의의 동음 꼴을 재질 강과 조사 의로 읽어 타 재질 기준 적용으로 과다 검출함(199-H3-N10, 기존 (마) 음성 오탐).",
    "'준용' 의 활용 꼴 — 적용 술어의 '준용' 낱말은 '준용한다' · '준용된다' · '준용함' 뿐이라, 그 밖 활용의 적용 단정은 보조 면제로 빠질 수 있음",
    "이중 부정 창 +12 상수",
    "U+318D 가운뎃점의 형제 자리(_SEP · _TOPIC) — '합격ㆍ불합격' 유보가 (나)로 걸림",
    "r_dim 정보 낱말에 환산 · mm/px 없음",
    "재질 복합어('강판' 등)를 재질 낱말로 받지 않음(199-H3-U01)",
    "다른 작은 마디의 단정이 보조 조항의 면제를 막는 과다 셈 — 진단 표지 '보조 면제 막힘' 으로 상한을 싣는다",
    "예외가 있는 대조 마디에서 뒤 작은 마디가 대상 낱말을 생략하면 H1 · H3 · (다)의 단정을 잡지 않음(미검출)",
    "앞 작은 마디의 조건이 뒤 작은 마디로 이어지지 않아 참 조건문의 뒤 서술이 걸리는 과다 셈 — 진단 표지 '조건 범위 단절 가능' 으로 상한을 싣는다",
    "(다) 새 세부 표지의 인과 꼴('…없으므로 … 적용된다') — 뒤 작은 마디의 술어로만 서는 단정을 잡지 않음(미검출)",
    "(ii) 일반 조건 어미 — 제외 목록 밖의 '면' 으로 끝나는 이름씨를 조건 어미로 읽어, 적용 단정 없는 보조 조항 언급이 면제될 수 있음",
    "(다)의 '경우' — 받침 ㄴ · ㄹ 로 끝나는 이름씨 뒤의 '경우' 를 조건절로 읽어 촉발 낱말이 예외가 될 수 있음",
    "r_note — 허용 꼴 앞에 다른 말이 먼저 오면 존재를 말해도 인정하지 않음(과소 인정)",
    "(다) 군집 · 간격 단정의 촉발 낱말 '군집을 이룬 · 군집으로 본 · 군집에 든' 이 '이루는 · 보는 · 드는' 꼴을 잡지 않음(알려진 미검출)",
    "r_note의 거부 꼴 고 보기 어렵이 어려워 활용을 받지 않아, 단서가 있다고 보기 어려워라는 존재 단정의 부정을 인정함(알려진 과다 인정).",
    "쉼표를 동반한 대조 이음(지만,)에서 앞 유보와 뒤 단정을 나누지 못해, 명시적 대상을 다시 적은 허용·안전 단정을 놓침(053, (나) 미검출).",
    "쉼표를 동반한 대조 이음 뒤 보조 전용 조항의 적용 단정을 앞 유보가 덮어, (라)는 검출하되 조건부 대응의 단정 세부 표지를 빠뜨림(071).",
)
EXPLORATORY_KINDS_V4: tuple[str, ...] = ("나", "마")
V6_EVIDENCE_V4: tuple[EvidenceRole, ...] = (
    EvidenceRole("checker_vectors", "evaluation/vectors/narrative_v4_list_probes.v2.json",
                 "e983a0098e8bc75a71e262cc89e1f49ec481c6f7b7dccd742699134c0f14aafe"),
    EvidenceRole("confirmation_cases", "evaluation/vectors/narrative_v4_confirm_cases.v1.json",
                 "8111533cd5685287a60d57b5272840e1b33d78c463b0ac9d5ae5617b4c7d1820"),
    EvidenceRole("regression", "evaluation/vectors/narrative_b_regression.v1.json",
                 "d75253aedf68747dd478d02fe7ba7cd8d15fc4dc446b17aaaa02d309593e04c6"),
    EvidenceRole("expectation_changes", "evaluation/vectors/narrative_v4_expectation_changes.v3.json",
                 "29f8b7764b4b289ca988824dc0ba3aa75a0062a06da7557af7e8f88a5c96343f"),
    EvidenceRole("regression_receipt", "docs/dev_log/2026-10-20-설계복구/13b_검사기v4_회귀_2회_D.json",
                 "32e66d60b46a5332e6f51f0e74503c5cd957e5e786617da158fcb9e24d535314"),
    EvidenceRole("holes_source", HOLES_SOURCE_REL, HOLES_SOURCE_SHA256),
    EvidenceRole("disposition", "docs/dev_log/2026-10-20-설계복구/07_허용치표_부분B_미니스펙_D.md",
                 "d20462a2568a280f86ca70a9dbc845d47e8e2a9309d98948d57ee1aad801bf42"),
    EvidenceRole("confirmation_output", "docs/dev_log/2026-10-20-설계복구/13c_검사기v4_확인배치_D.json",
                 "4382c492fa73593e3d60dc60c068515e5e8999713309f3061f8ce0e89cde65e0"),
    EvidenceRole("uncertain_classification", "docs/dev_log/2026-10-20-설계복구/13f_확인배치_uncertain_분류_D.json",
                 "8095d3dc16848257360ff555dfed7873562470dae750001e610967a5d778728f"),
)
"""D3 의 아홉 행(263 §3-3) — 순서 고정. 검사기 벡터 = 쓰임 탐침 v2(쓰임 탐침이지 전체 독립 확인 성적이 아니다), 확인 사례 = 원 사례 v1,
기대 변경표 v3 는 사후 관측을 거친 기대 변경(사전 독립 기대표로 재명명하지 않는다), 13c · 13f 는 보존 근거 · 사후 분류(확인 사례 · 기대
변경표 칸에 넣지 않는다). 해시는 2026-10-28 의 기준 커밋 blob 에서 잰 원본 바이트다."""
NARRATIVE_PROFILES: dict[str, NarrativeProfile] = {
    PROFILE_LEGACY: NarrativeProfile(PROFILE_LEGACY, CHECKER_PINS[0], GENERATOR_RULE_SINGLE, None, None, OPEN_HOLES,
                                     EXPLORATORY_KINDS, ()),
    PROFILE_REGIME2: NarrativeProfile(PROFILE_REGIME2, CHECKER_PINS[1], BUNDLE_RULE, V6_EVIDENCE_V4[0].path,
                                      V6_EVIDENCE_V4[1].path, OPEN_HOLES_V4, EXPLORATORY_KINDS_V4, V6_EVIDENCE_V4),
}
"""닫힌 프로필 — 구판(검사기 v3.1 · 옛 11 줄 · (마))과 신판(검사기 v4 · 2 판 표 · 묶음 생성 해시 · 19 줄 · (나)(마) · 아홉 출처)."""
CHECKER_MODULES: dict[str, str] = {PROFILE_LEGACY: "evaluation.narrative_checker", PROFILE_REGIME2: "evaluation.narrative_checker_v4"}
"""프로필 → 실제로 불러 쓰는 검사기 모듈(닫힌 사상 — `load_profile_checker`)."""
V6_SOURCES_KIND = "v6_link_sources"
V6_SOURCES_VERSION = 1
V6_SOURCES_KEYS = frozenset({"kind", "version", "profile_id", "registration_sha256", "eval_sha256", "decision", "artifacts",
                             "disclosure"})
V6_DECISION: dict[str, str] = {"commit": "af44b4719c4eecd73e257c543216a9459e58d2ef", "path": "docs/의사결정로그.md",
                               "sha256": "d96dd9c4f6c42375e396bb4d046992811719d2ff5f934cb4249d92c3cbe57d3a",
                               "section": "2026-10-28 판 6 연결 결정 D1 ~ D6 확정(D7 보류)"}
"""결정의 자리 — 그 커밋의 의사결정로그 blob(raw sha)과 항목. 승인 시각을 지어내는 키는 두지 않는다(기록 시각은 커밋 시각이다)."""
V6_DISCLOSURE: dict[str, object] = {"comparison_round": 2, "sample_previously_observed": True}
"""D5 — 이미 관측한 표본의 2 차 비교임을 적는다(옛 전량 대조선 선언을 새 표본 실측값으로 바꾸지 않는다)."""
REASON_CODES_V6_LINK = ("profile_mismatch", "profile_holes", "profile_runtime", "profile_paths", "evidence_role",
                        "evidence_bytes", "v6_sources_missing", "v6_sources_mismatch", "eval_bytes")
"""이 절이 새로 정의한 사유 코드(263 §8). 기존 코드(`hash_rule` · `code_bytes` · `contract_form` · `v6_form` · `pin_combo`)의 뜻은 그대로다."""


def resolve_profile(profile: NarrativeProfile | str | None) -> NarrativeProfile:
    """`None` 은 구판, 문자열은 두 식별자 가운데 하나, 레코드는 등록된 것이어야 한다 — 밖이면 `profile_mismatch`."""
    if profile is None:
        return NARRATIVE_PROFILES[PROFILE_LEGACY]
    if isinstance(profile, str):
        if profile not in NARRATIVE_PROFILES:
            raise ProfileRefused("profile_mismatch", f"모르는 프로필 {profile!r} — 아는 것 {sorted(NARRATIVE_PROFILES)}")
        return NARRATIVE_PROFILES[profile]
    if isinstance(profile, NarrativeProfile) and NARRATIVE_PROFILES.get(profile.profile_id) == profile:
        return profile
    raise ProfileRefused("profile_mismatch", "등록된 프로필 레코드가 아니다")


def contract_code_problems(contract_raw: bytes, *, repo: Path) -> list[str]:
    """학습 전 계약 바이트의 꼴 · 정규형 · 실제 파일(표 raw · 로더 묶음 · 생성 묶음의 LF sha) — B4 의 공개 합성 함수(263 §3-1).

    꼴에 잘못이 있으면 바이트 단계로 가지 않는다. 페어의 사실이 필요한 `narrative_inputs_problems` 를 증거 없이 부르지 않는다 — 그쪽은
    학습 등록 검증 · 적재 전 점검의 일이다. 단일 파일 sha 대입은 `hash_rule`, 실제 파일 누락 · 변경 · 목록 차이는 `code_bytes`.
    """
    raw = bytes(contract_raw)
    contract = _json_object(raw)
    if contract is None:
        return ["contract_form: 계약 바이트가 JSON 객체가 아니다"]
    form = _contract_form(contract) + _canonical_problem(contract, raw)
    if form:
        return form
    return _contract_bytes(contract, Path(repo))


def generator_bundle_identity(repo: Path | None = None) -> dict:
    """정답 서술 생성 경로 세 파일(`GENERATOR_FILES_V1`)의 `lf_bundle_v1` 묶음 — `{rule, files, sha256}`, 실측이다(문자열 복사가 아니다)."""
    return bundle_block(repo, GENERATOR_FILES_V1)


def profile_for(registration: UnifiedRegistration, *, train_entry: Mapping | None) -> NarrativeProfile:
    """판 6 등록 × 학습 등록의 조합에서 프로필 하나 — 단서가 섞이면 `profile_mismatch`(263 §2-1 · §3-2).

    `train_entry` 는 학습 등록의 값(`vlm.registration.registration_from_bytes` 의 사전 — 새 작성은 실제 PIN 의 학습 등록, 커밋 후에는
    같은 anchor 의 `configs/train_registration.yaml`). 신규 단서 = 검사기 v4 행(판 · 2 판 표 · 닫힌 목록 v4-2)과 계약 결속(등록의
    `narrative.inputs_contract_sha256` 이 학습 등록의 `narrative_inputs.sha256`). 전부면 `regime2_v4`, 하나라도 있으면서 나머지가 빠지거나
    옛 값이면 거부, 하나도 없고 v3.1 행이면 `legacy_v3_1`. 곁 파일의 유무 · 파일 날짜 · 생성 한도로는 가르지 않는다.
    """
    if registration.version != REGISTRATION_VERSION_V6:
        raise ProfileRefused("profile_mismatch", f"판 {registration.version} 에는 서술 블록이 없다 — 프로필을 고를 수 없다")
    n = registration.narrative
    if n is None:
        raise ProfileRefused("profile_mismatch", "판 6 의 서술 블록이 없다")
    new_pin, old_pin = NARRATIVE_PROFILES[PROFILE_REGIME2].checker_pin, NARRATIVE_PROFILES[PROFILE_LEGACY].checker_pin
    ni = train_entry.get(INPUTS_REGISTRATION_KEY) if isinstance(train_entry, Mapping) else None
    bound = isinstance(ni, Mapping) and _hex64(n.inputs_contract_sha256) and ni.get("sha256") == n.inputs_contract_sha256
    new = {"checker_v4": n.checker_id == new_pin.checker_id, "limits_v2": n.limits_sha256 == new_pin.limits_sha256,
           "lists_v4_2": n.lists_sha256 == new_pin.lists_sha256, "contract_bound": bool(bound)}
    old = {"checker_v3_1": n.checker_id == old_pin.checker_id, "limits_v0": n.limits_sha256 == old_pin.limits_sha256,
           "lists_v3_3": n.lists_sha256 == old_pin.lists_sha256}
    if all(new.values()):
        return NARRATIVE_PROFILES[PROFILE_REGIME2]
    if any(new.values()):
        have, lack = [k for k, v in new.items() if v], [k for k, v in new.items() if not v]
        raise ProfileRefused("profile_mismatch", f"신규 단서 {have} 가 있는데 {lack} 가 빠졌거나 옛 값이다 — 판 6 의 검사기 판 · 표 · "
                                                 f"닫힌 목록 · 계약 결속은 함께 바뀐다")
    if all(old.values()):
        return NARRATIVE_PROFILES[PROFILE_LEGACY]
    raise ProfileRefused("profile_mismatch", "검사기 판 · 표 · 닫힌 목록이 어느 프로필의 행도 아니다")


def profile_holes_problems(narrative: NarrativeSpec, profile: NarrativeProfile) -> list[str]:
    """등록의 잔여 한계 · 탐색적 종류가 프로필 정본과 **글자 · 순서 그대로** 같은가 — 어긋나면 `profile_holes`(263 §3-2 · R10)."""
    out: list[str] = []
    holes = None if narrative.open_holes is None else tuple(narrative.open_holes)
    if holes != profile.open_holes:
        why = ("칸이 비었다" if holes is None else
               f"등록 {len(holes)} 줄 · 정본 {len(profile.open_holes)} 줄" if len(holes) != len(profile.open_holes) else
               "같은 수의 줄인데 문구나 순서가 다르다")
        out.append(f"profile_holes: narrative.open_holes 가 프로필 {profile.profile_id} 의 정본 목록({HOLES_SOURCE_REL} "
                   f"§{HOLES_SOURCE_SECTION})과 다르다 — {why}")
    kinds = None if narrative.exploratory_kinds is None else tuple(narrative.exploratory_kinds)
    if kinds != profile.exploratory_kinds:
        out.append(f"profile_holes: narrative.exploratory_kinds 등록 {None if kinds is None else list(kinds)} · "
                   f"프로필 {list(profile.exploratory_kinds)}")
    return out


def evidence_problems(narrative: NarrativeSpec, profile: NarrativeProfile, repo: Path | None = None) -> list[str]:
    """프로필의 증거 칸 둘(검사기 벡터 · 확인 사례) — 다른 역할의 파일이면 `evidence_role`, 빈 칸 · 다른 바이트 · 정본 파일의 변조는
    `evidence_bytes`(263 §3-3 · R11). 구판 프로필(출처 행 없음)은 호출자가 경로를 줄 때만 맞대므로 여기서 볼 것이 없다."""
    if not profile.evidence_roles:
        return []
    root = Path(repo or REPO)
    roles = {r.role: r for r in profile.evidence_roles}
    out: list[str] = []
    for field, role in (("checker_vectors_sha256", "checker_vectors"), ("confirmation_cases_sha256", "confirmation_cases")):
        want, got = roles[role], getattr(narrative, field)
        others = {r.sha256: r.role for r in profile.evidence_roles if r.role != role}
        others[VECTOR_SHA256] = "extract_vectors"                                  # 추출 벡터는 검사기 벡터 자리에 못 온다
        if got is None:
            out.append(f"evidence_bytes: narrative.{field} 이 비었다 — 프로필 {profile.profile_id} 는 {want.path} 를 요구한다")
        elif got in others:
            out.append(f"evidence_role: narrative.{field} 이 {others[got]} 역할의 파일이다 — {role} 자리에는 {want.path}")
        elif got != want.sha256:
            out.append(f"evidence_bytes: narrative.{field} {got[:12]}… ≠ 프로필 정본 {want.path} 의 sha {want.sha256[:12]}…")
        actual = root / want.path
        if not actual.is_file() or raw_sha256(actual) != want.sha256:
            out.append(f"evidence_bytes: 정본 {want.path} 가 없거나 바이트가 프로필 핀({want.sha256[:12]}…)과 다르다")
    return out


def load_profile_checker(profile: NarrativeProfile | str, *, repo: Path):
    """프로필이 가리키는 검사기 모듈의 `load_checker(repo)` — 닫힌 사상(`CHECKER_MODULES`)에서만 고른다(263 §3-2).

    먼저 `checker_pin_problems`(모듈 상수 · 표 · 닫힌 목록의 실제 바이트)를 지나고, 불러온 모듈의 핀 상수 · 파일 바이트가 그 프로필의
    행 · 저장소의 핀 모듈과 같은지 본다 — 다르면 `profile_runtime`(대역 모듈 · 바뀐 상수). 구판 판 6 를 새 프로필로 올리지 않는다.
    """
    import importlib

    p = resolve_profile(profile)
    root = Path(repo)
    problems = checker_pin_problems(root)
    if problems:
        raise ProfileRefused("pin_combo", problems)
    mod = importlib.import_module(CHECKER_MODULES[p.profile_id])
    pin = p.checker_pin
    want = dict(zip(_PIN_CONSTANTS, (pin.checker_id, pin.limits_rel, pin.limits_sha256, pin.lists_rel, pin.lists_sha256),
                    strict=True))
    got = {name: getattr(mod, name, None) for name in _PIN_CONSTANTS}
    if got != want:
        bad = sorted(k for k in want if got[k] != want[k])
        raise ProfileRefused("profile_runtime", f"불러온 모듈 {mod.__name__} 의 핀 상수 {bad} 가 프로필 {p.profile_id} 의 행과 다르다")
    src = Path(getattr(mod, "__file__", "") or "")
    if not src.is_file() or lf_sha256(src) != lf_sha256(root / pin.module):
        raise ProfileRefused("profile_runtime", f"불러온 모듈 {mod.__name__} 의 바이트가 저장소의 핀 모듈 {pin.module} 과 다르다")
    return mod.load_checker(root)


def dump_v6_sources(doc: Mapping) -> bytes:
    """출처 운반 파일의 정규 바이트 — 키 정렬 · UTF-8 · 들여쓰기 1 · 끝 LF(263 §3-3)."""
    return (json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def build_v6_sources(*, registration_raw: bytes, eval_raw: bytes, repo: Path) -> bytes:
    """`{stem}.v6_sources.json` 의 바이트 — 아홉 원본을 `repo` 의 실물과 맞댄 뒤 적는다(263 §3-3). 원자료 · 사례 본문 · 성능값은 싣지 않는다.

    등록 · 설정에는 출처 파일의 sha 를 다시 넣지 않으므로 해시 순환은 없다. 원본이 없으면 `v6_sources_missing`, 바이트가 다르면
    `v6_sources_mismatch` — 핀 불일치를 새 sha 재기입으로 덮지 않는다.
    """
    root = Path(repo)
    problems: list[str] = []
    for r in V6_EVIDENCE_V4:
        f = root / r.path
        if not f.is_file():
            problems.append(f"v6_sources_missing: {r.role} 의 원본 {r.path} 가 없다")
        elif raw_sha256(f) != r.sha256:
            problems.append(f"v6_sources_mismatch: {r.role} 의 원본 {r.path} 의 바이트가 승인 sha({r.sha256[:12]}…)와 다르다")
    if problems:
        raise ProfileRefused(problems[0].split(":", 1)[0], problems)
    doc = {"kind": V6_SOURCES_KIND, "version": V6_SOURCES_VERSION, "profile_id": PROFILE_REGIME2,
           "registration_sha256": _sha(registration_raw), "eval_sha256": _sha(eval_raw), "decision": dict(V6_DECISION),
           "artifacts": [r._asdict() for r in V6_EVIDENCE_V4], "disclosure": dict(V6_DISCLOSURE)}
    return dump_v6_sources(doc)


def v6_sources_problems(raw: bytes, *, registration_raw: bytes, eval_raw: bytes, read) -> list[str]:
    """출처 운반 파일의 바이트 ↔ 등록 · 설정 바이트 ↔ anchor 의 원본(`read(rel) -> bytes | None`). 순수 함수 — 비면 맞다.

    `eval_sha256` 이 설정 바이트와 다르면 `eval_bytes`(네 소비자가 모두 거부한다 — R6), 그 밖의 어긋남은 `v6_sources_mismatch`, anchor 에
    원본이 없으면 `v6_sources_missing`.
    """
    doc = _json_object(raw)
    if doc is None:
        return ["v6_sources_mismatch: 출처 파일이 JSON 객체가 아니다"]
    keys = set(doc)
    if keys != V6_SOURCES_KEYS:
        return [f"v6_sources_mismatch: 최상위 키가 다르다 — 모르는 {sorted(keys - V6_SOURCES_KEYS)} · 빠진 {sorted(V6_SOURCES_KEYS - keys)}"]
    out: list[str] = []
    if dump_v6_sources(doc) != bytes(raw):
        out.append("v6_sources_mismatch: 정규형이 아니다 — build_v6_sources 가 적은 바이트만 받는다")
    for key, want in (("kind", V6_SOURCES_KIND), ("version", V6_SOURCES_VERSION), ("profile_id", PROFILE_REGIME2),
                      ("decision", V6_DECISION), ("disclosure", V6_DISCLOSURE)):
        if doc[key] != want:
            out.append(f"v6_sources_mismatch: {key} 가 정본과 다르다 — 받은 {doc[key]!r}")
    if doc["registration_sha256"] != _sha(registration_raw):
        out.append("v6_sources_mismatch: registration_sha256 ≠ 등록 바이트의 sha")
    if doc["eval_sha256"] != _sha(eval_raw):
        out.append("eval_bytes: eval_sha256 ≠ 평가 설정 바이트의 sha — 출처가 묶은 설정과 다른 설정이다")
    rows = doc["artifacts"]
    if not isinstance(rows, list) or [r.get("role") if isinstance(r, dict) else None for r in rows] != [r.role for r in V6_EVIDENCE_V4]:
        out.append("v6_sources_mismatch: artifacts 의 역할 목록 · 순서가 정해진 아홉 행과 다르다")
        return out
    for got, want in zip(rows, V6_EVIDENCE_V4, strict=True):
        if got != want._asdict():
            out.append(f"v6_sources_mismatch: {want.role} 행이 승인 경로 · sha · 규칙과 다르다")
            continue
        b = read(want.path)
        if b is None:
            out.append(f"v6_sources_missing: anchor 에 {want.role} 의 원본 {want.path} 가 없다")
        elif _sha(b) != want.sha256:
            out.append(f"v6_sources_mismatch: anchor 의 {want.path} 바이트가 승인 sha({want.sha256[:12]}…)와 다르다")
    return out


def load_v6_sources(*, repo: Path, anchor_rev: str, registration_rel: str, registration_raw: bytes, eval_raw: bytes) -> dict:
    """anchor 커밋의 `{stem}.v6_sources.json` 을 읽어 등록 · 설정 바이트 · 같은 anchor 의 아홉 원본과 맞댄다(263 §3-3). 거부는 `ProfileRefused`."""
    from evaluation.entry_gate import EntryRejected, blob_at
    from evaluation.eval_registration import v6_sources_rel

    root = Path(repo)
    rel = v6_sources_rel(registration_rel)
    try:
        raw = blob_at(root, anchor_rev, rel)
    except EntryRejected:
        raise ProfileRefused("v6_sources_missing", f"anchor 에 출처 운반 파일 {rel} 이 없다") from None

    def read(path: str) -> bytes | None:
        try:
            return blob_at(root, anchor_rev, path)
        except EntryRejected:
            return None

    problems = v6_sources_problems(raw, registration_raw=registration_raw, eval_raw=eval_raw, read=read)
    if problems:
        raise ProfileRefused(problems[0].split(":", 1)[0], problems)
    return json.loads(raw.decode("utf-8"))


__all__ = ["BUNDLE_RULE", "CHECKER_MODULES", "CHECKER_PINS", "CHECKER_REL", "EXPLORATORY_KINDS", "EXPLORATORY_KINDS_V4",
           "EXTRACTOR_REL", "GENERATOR_FILES_V1", "GENERATOR_REL", "GENERATOR_RULE_SINGLE", "HOLES_CHECKER_ID", "HOLES_SOURCE_REL",
           "HOLES_SOURCE_SECTION", "HOLES_SOURCE_SHA256", "INPUTS_CONTRACT_KIND", "INPUTS_CONTRACT_REL", "INPUTS_CONTRACT_VERSION",
           "INPUTS_REGISTRATION_KEY", "LIMITS_CSV_REL", "LIMITS_LOADER_FILES", "NARRATIVE_PROFILES", "OPEN_HOLES", "OPEN_HOLES_V4",
           "PROFILE_LEGACY", "PROFILE_REGIME2", "REASON_CODES", "REASON_CODES_V6_LINK", "TEXT_CONTRACT_V2", "TRAIN_EXTRACTOR_REL",
           "V6_DECISION", "V6_DISCLOSURE", "V6_EVIDENCE_V4", "V6_SOURCES_KEYS", "V6_SOURCES_KIND", "V6_SOURCES_VERSION", "CheckerPin",
           "EvidenceRole", "NarrativeProfile", "ProfileRefused", "VerifiedPairs", "actual_identity", "build_v6_sources", "bundle_block",
           "checker_pin_problems", "clause_axis_block", "contract_code_problems", "dump_narrative_inputs", "dump_v6_sources",
           "evidence_problems", "generator_bundle_identity", "identity_problems", "lf_sha256", "limits_loader_bundle_sha256",
           "load_profile_checker", "load_v6_sources", "module_str_constant", "narrative_block", "narrative_inputs_block",
           "narrative_inputs_problems", "pairs_identity_problems", "profile_for", "profile_holes_problems", "prompt_sha256",
           "raw_sha256", "registration_pin_problems", "resolve_profile", "v6_contract_problems", "v6_sources_problems"]
