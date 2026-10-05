"""통합형 **본채점 진입점** — 진입점 미니스펙 3판(wt/D `d22c7bf`) · 계약 07번 §32.

    python -m scripts.probe.score_unified --registration probe --purpose rehearsal --receipt R --snapshot S \\
        --bundles B --generation-list G --scoring-list P --echo-list E --train-root T --out O --device cpu:0

| 단계 | 무엇 | 걸리면 |
|---|---|---|
| 0 | 인자 · 모드와 목적(`probe` + `main` 거부) · 출력 루트(엄격은 `outputs/main_u` 뿐, 탐색은 `outputs/rehearsal_u/…`) · 엄격의 입력이 리허설 루트 밖인가 | 종료 3 |
| 1 | 영수증 · 커밋의 등록(파일 해시 먼저) · 동결본의 닻 · 프로세서 설정 · 완결(엄격은 `require_receipt` 두 영역, 탐색은 생성 쪽 + 에코 관문 두 칸) · 스냅샷 검증과 흡수 상태 · 잠정 블록 | 종료 3 |
| 2 | 정본 채점 원장에 `call` 한 줄(계산 **전**) — 원장의 자리는 인자가 아니다 | — |
| 2′ | 목록 셋을 바이트로 검사 · 등록의 목록 해시와 결속 · 매니페스트 네 열로 분할 대조 | 종료 3 |
| 3 | 묶음마다 `verify_bundle`, 묶음 사이 환경 대조(모델끼리 · 에코끼리) | 그 칸 거부 · 종료 2 |
| 5 | 에코 관문과 에코 채점 · 문자 일치 | 엄격의 에코 불통과는 3, 탐색은 2 |
| 6 | 리허설 — ③ · ⑤ · 적대적 한 장 · 지표 확인(값 없음) | 종료 2 |
| 7 | 산출물 — 고정 이름 · 배타 생성 | 있으면 거부 |

**산출 경로.** `--out` 아래 산출 경로의 조각(`score` · 파일)이 정션 · 기호 링크로 밖을 가리키면 `[OUT_PATH_LINK]`, 산출 파일에 하드 링크가 있으면
`[OUT_FILE_HARDLINK]` 로 거부한다 — 0 단계(원장 줄 전)에서 미리 보고, 읽기 · 생성 직전에 다시 본다(공유 회신 §57).

**에코 전용(`--echo-only`) — 학습 전 관문.** 탐색(`probe`)의 리허설 · 진단 목적만 받는다(본실험 · 엄격은 진입 전 종료 3). 0 · 1 은 위와 같고
(진단의 재생성 분기와 규칙 파일 읽기는 하지 않는다), **2 를 하지 않는다 — 정본 원장에 아무것도 쓰지 않으므로 시도로 세지 않는다.**
2′ 목록 결속 → 3 은 **에코 묶음만** → 5 의 에코 관문 · 에코 채점 · 문자 일치. 결과는 `score/echo_gate.json` 하나 — 통과면 종료 0,
아니면 사유 코드(`ECHO_REASONS`)를 싣고 종료 2. 같은 입력으로 다시 부르면 같은 바이트라 받는다(재개) — 다른 바이트가 있으면 종료 3.

**지금 서지 않는 둘**(보고 13 의 3-3). **엄격 경로**는 5 까지 돌고 6 앞에서 종료 3 이다 — 등록 지표(주 지표 1b) · 규칙 ⑤ 등록 판정 모듈(07번 §29-5) ·
구간 규약(보류-4)의 구현 · 평가 정답 뷰(데이터 쪽 `read_annotations_view_eval`)가 없다. **진단**(`--purpose frame_diag`)은 검사기(과제 4) 전이라 0 에서 종료 3 이다.

**시험의 이음새.** `main(argv, _seam=Seam(...))` 로 본체 체크아웃과 저장소를 tmp 에 줄 수 있다. 커밋된 영수증(종류 `main` · `frame_diag` 이고
커밋이 본체 `refs/heads/main` 의 조상)이면 이음새를 거부한다(`entry_gate.guard_seam`) — 그 조상 검사는 이음새를 거치지 않는다. 명령줄에는 원장 ·
출력 부모 · 저장소의 경로 인자가 없다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from data.manifest_view import read_annotations_view, read_annotations_view_eval
from evaluation.actuals import registered_values
from evaluation.adapters_v14 import UpstreamContractError
from evaluation.bundle_v14 import (MODE_ECHO, MODE_MODEL, BundleFiles, Registration, TrainingArtifact,
                                   check_env_consistency, verify_bundle)
from evaluation.content_key import ContentKeyError, generations_content_sha256
from evaluation.coord_check import coord_fixture_digest
from evaluation.entry_gate import (COMMITTED_KINDS, STRICT_OUT_REL, EntryRejected, blob_at, check_anchor, check_list_split,
                                   check_mode, check_processor, check_receipt_kind, check_snapshot, guard_seam,
                                   canonical_ledger, load_anchor, load_processor_kwargs, load_registration_for,
                                   main_checkout, read_manifest_meta)
from evaluation.eval_list import validate_eval_list
from evaluation.ledger_v14 import FED_CLIENTS, LedgerRejected, read_ledger_view
from evaluation.prereg_unified import (EchoGateFailed, ReceiptMissing, RegistrationIncomplete, RegistrationInvalid,
                                       check_echo_gate, read_receipt, require_probe_scoring, require_receipt)
from evaluation.provenance import scorer_code_digest, write_new_text
from evaluation.reject_v14 import BundleRejected
from evaluation.scoring_layer import (adversarial_probe, check_literal, judge_rehearsal_gates, rehearsal_metric_smoke,
                                      score_echo)
from evaluation.frame_diag import PASS as FRAME_PASS
from evaluation.frame_diag import FrameImage, judge, mutual_pairs
from evaluation.axis_ratio_canary import axis_ratio_canary
from evaluation.frame_diag_rules import RulesFileError, load_rules_file, requirement_status
from evaluation.scoring_ledger import LedgerBusy, append_line, find_verdicts, ledger_lock, read_ledger

EXIT_OK, EXIT_REJECTED, EXIT_ENTRY = 0, 2, 3
REHEARSAL_OUT_REL = "outputs/rehearsal_u"
FORBIDDEN_OUT_REL = ("outputs/main_d",)
GOLDEN_FIXTURES_REL = "vlm/coords_fixtures/golden_fixtures.json"
OUT_SCORE = "score/score_unified_rehearsal.json"
OUT_GATES = "score/rehearsal_gates.json"
OUT_DIAG = "score/frame_diag_score.json"
"""진단 산출물 — **판정 줄의 결정적 함수**다(계약 §32-8). 판정 줄이 없으면 이 이름으로 쓰지 않는다."""
OUT_DIAG_NOT_JUDGED = "score/frame_diag_not_judged.json"
OUT_ECHO_GATE = "score/echo_gate.json"
"""에코 전용 호출의 산출 — 원장에 줄을 쓰지 않는 학습 전 관문의 결과."""
ECHO_ONLY_PURPOSES = ("rehearsal", "frame_diag")
ECHO_REASONS = ("no_echo_bundle", "echo_bundle_rejected", "echo_gate_failed", "echo_mismatch", "literal_mismatch",
                "stand_in")
"""에코 전용 호출이 통과가 아닐 때의 사유 코드 — 묶음이 없다 · 묶음 검증 거부 · 좌표 규약 관문 · 에코 채점(항등) ·
문자 일치 · 승인 구현이 아니다(대역 실행)."""
"""판정하지 않은 진단 호출의 산출 — 묶음 거부 · 대역 실행 · 카나리아 불통과. 판정 줄을 쓰지 않으므로 한 번 규칙을 쓰지 않는다."""
FRAME_INPUT_FIELDS = ("prompt_sha256", "chat_template_kwargs", "gen_prefix_sha256", "processor_config_sha256",
                      "processor_min_pixels", "processor_max_pixels", "patch_size", "merge_size", "coord_space",
                      "coord_cfg_hash", "base_model_id", "base_model_revision", "init_adapter_digest",
                      "transformers_version")
"""진단 통과가 서는 본실험 등록의 칸(계약 §31-7) — 진단 산출물이 그 **입력 해시**를 싣는다."""
TRAIN_CONFIG_UNBOUND = ("num_rounds", "local_epochs", "total_epochs", "pairs_digest")
"""`train_config` 에서 프레임 결속에서 빼는 키 — 예산 · 페어(계약 §31-7 의 "예산 · 페어 · 행 목록을 뺀 나머지"). 행 목록은 `train_config` 에 없다."""
DEFAULT_COUPLING = "decoupled_v2"
"""결합 규칙의 어휘는 하나다(`prereg_unified.VOCAB`). 리허설 등록의 채점 쪽이 비었으면 그 값을 쓰고 산출에 그 사실을 적는다."""

EVAL_VIEW = read_annotations_view_eval
"""엄격 경로의 평가 정답 뷰 — 데이터 쪽 판독기(판정 10 의 1절, A `40c1316` 이 본줄기에 들였다). 엄격 채점(단계 6)의 정답은
**이것으로만** 읽는다 — 영수증 **파일 경로**를 `receipt=` 로 준다. 부르는 조건(영수증 종류 `main` · 커밋이 본줄기의 조상 · 등록의
목적 `main` · 기준 분할 `eval` · 닻 · 평가 id 만)은 판독기가 코드로 걸고, 어기면 주석 파일을 열기 전에 거부한다.
단계 6 이 아직 없어 지금 부르는 자리는 없다 — 그 단계가 설 때 이 이름을 부른다."""
EVAL_VIEW_READER = f"{EVAL_VIEW.__module__}.{EVAL_VIEW.__qualname__}"
EVAL_VIEW_CONDITIONS = ("영수증 종류가 main", "영수증 커밋이 본체 refs/heads/main 의 조상", "등록의 목적이 main")
"""그 뷰를 부르는 조건 — 셋이 모두 설 때만 부른다. 판독기도 같은 조건(과 그 밖의 넷)을 코드로 건다(판정 10 의 1절)."""
STRICT_PENDING = ("등록 지표의 구현(주 지표 1b)", "규칙 ⑤ 등록 판정 모듈(07번 §29-5)",
                  "축 배율 카나리아의 진단 규칙 판(07번 §31-1 의 6)")
"""엄격 채점(단계 6 · 7)이 서지 않는 까닭. 구간 규약(지시 j)과 평가 정답 뷰(데이터 쪽)는 들어왔다."""

_NAME = re.compile(r"^(?P<tag>[A-Za-z0-9_]+)_s(?P<seed>\d+)(?P<echo>\.echo)?\.generations\.jsonl$")


@dataclass(frozen=True)
class Seam:
    """**시험 전용** 이음새 — 본체 체크아웃(원장 · 출력의 부모)과 git 저장소. 커밋된 영수증과 함께 쓰면 거부한다."""

    checkout: Path
    repo: Path


@dataclass
class Ctx:
    args: argparse.Namespace
    checkout: Path
    repo: Path
    receipt: object = None
    reg: object = None
    anchor: object = None
    absorption: str | None = None
    provisional: list = field(default_factory=list)
    scoring_gaps: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    diag_rules: object = None
    diag_rules_body: dict | None = None
    content_errors: dict = field(default_factory=dict)
    """내용 해시를 낼 수 없는 묶음 → 사유. 그 묶음은 검증 · 채점으로 가지 않는다."""


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="통합형 본채점 진입점")
    ap.add_argument("--registration", required=True, choices=("strict", "probe"))
    ap.add_argument("--purpose", required=True, choices=("main", "rehearsal", "frame_diag"))
    for name in ("receipt", "snapshot", "bundles", "generation-list", "scoring-list", "echo-list", "train-root", "out"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    ap.add_argument("--plan", type=Path, default=None, help="계획 파일 — 등록에 plan_sha256 이 있으면 필수")
    ap.add_argument("--device", required=True)
    ap.add_argument("--echo-only", action="store_true",
                    help="학습 전 에코 관문만 — 원장에 쓰지 않는다(리허설 · 진단 목적만)")
    return ap


def _under(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ── 0 · 1 진입 전 ────────────────────────────────────────────────────────────

OUT_FILES = (OUT_SCORE, OUT_GATES, OUT_DIAG, OUT_DIAG_NOT_JUDGED, OUT_ECHO_GATE)
"""이 진입점이 `--out` 아래에 쓰는 산출 — 0 단계에서 경로를 미리 보고, 읽기 · 생성 직전에 다시 본다."""


def _out_file(ctx: Ctx, rel: str) -> Path:
    """산출 파일의 자리 — `--out` 에서 그 파일까지 **있는 조각마다 해석한 최종 경로가 그 자리 그대로**여야 한다.

    `--out` 자체는 0 단계가 본다. 그 아래 `score` 가 정션 · 기호 링크로 밖을 가리키면 `mkdir(exist_ok)` 와 배타 생성이 그 연결을
    따라가 허용 영역 밖에 산출을 만든다(공유 회신 §57) — 읽기 · 생성 전에 거부한다. 산출 파일이 이미 있으면 다른 이름(하드 링크)이
    없어야 한다 — 같은 바이트를 비교하려고 읽는 것도 밖의 파일을 읽는 일이다.
    """
    out = ctx.args.out
    root = out.resolve()
    q = out
    for part in Path(rel).parts:
        q = q / part
        if os.path.lexists(q) and q.resolve() != root / q.relative_to(out):
            raise EntryRejected("OUT_PATH_LINK",
                                f"산출 경로 {q.relative_to(out).as_posix()} 가 --out 밖으로 이어진다(정션 · 링크) — 쓰거나 읽지 않는다")
    path = out / rel
    if path.is_file() and os.stat(path).st_nlink > 1:
        raise EntryRejected("OUT_FILE_HARDLINK", f"산출 파일 {rel} 에 다른 이름이 있다(하드 링크) — 쓰거나 읽지 않는다")
    return path


def _stage0(ctx: Ctx) -> None:
    a = ctx.args
    check_mode(a.registration, a.purpose)
    co = ctx.checkout
    out = a.out
    if any(_under(out, co / r) for r in FORBIDDEN_OUT_REL):
        raise EntryRejected("OUT_ROOT", "분리형 사전실험의 루트에는 쓰지 않는다")
    if a.registration == "strict":
        if out.resolve() != (co / STRICT_OUT_REL).resolve():
            raise EntryRejected("OUT_ROOT", "엄격의 출력 루트는 outputs/main_u 뿐이다(가-4′)")
        for name in ("bundles", "generation_list", "scoring_list", "echo_list", "train_root", "receipt"):
            if _under(getattr(a, name), co / REHEARSAL_OUT_REL):
                raise EntryRejected("INPUT_UNDER_REHEARSAL", f"엄격의 입력({name})이 리허설 루트 아래다(가-4)")
    else:
        if _under(out, co / STRICT_OUT_REL) or not _under(out, co / REHEARSAL_OUT_REL) \
                or out.resolve() == (co / REHEARSAL_OUT_REL).resolve():
            raise EntryRejected("OUT_ROOT", "탐색 채점의 산출은 outputs/rehearsal_u/<id>/ 아래다 — main_u 에는 원장 한 줄뿐")
        for rel in OUT_FILES:          # 원장 줄 전에 — 산출 경로가 밖으로 이어지면 시도가 아니다
            _out_file(ctx, rel)


def _provisional(ctx: Ctx) -> list[dict]:
    """잠정 블록 — 이 진입점이 **지금 읽을 수 있는** 항목만 적는다(흡수 상태 · 계획 파일의 생성 한도). 나머지 잠정의 표시를 읽는 자리는
    보고 13 의 3-3 에 비워 둔 자리로 올렸다."""
    out: list[dict] = []
    if ctx.absorption == "provisional":
        out.append({"id": "snapshot_absorption", "status": "provisional", "source": str(ctx.args.snapshot),
                    "sha256": ctx.anchor.snapshot_digest, "reason": "동결본의 흡수 상태가 잠정이다(1-5 의 7)"})
    g = ctx.reg.generation
    if g.plan_sha256 is not None:
        if ctx.args.plan is None:
            raise EntryRejected("PLAN_MISSING", "등록에 plan_sha256 이 있는데 계획 파일이 주어지지 않았다")
        raw = ctx.args.plan.read_bytes()
        if hashlib.sha256(raw).hexdigest() != g.plan_sha256:
            raise EntryRejected("PLAN_HASH", "계획 파일의 바이트가 등록의 plan_sha256 과 다르다")
        plan = yaml.safe_load(raw.decode("utf-8")) or {}
        if (plan.get("gen_limit") or {}).get("provisional"):
            out.append({"id": "generation_limit", "status": "provisional", "source": str(ctx.args.plan),
                        "sha256": g.plan_sha256, "reason": "생성 한도가 잠정값이다(결정표 5c)"})
    return out


def _stage1_receipt(ctx: Ctx, seam: Seam | None) -> None:
    """영수증 검사(3판 1-1-가 의 1~3) — 영수증 · 종류 · 조상 · 등록 파일의 바이트 · **생성 지문**. 재생성은 여기까지만 한다(§32-8).

    3 의 `registration_for_receipt` 는 파일 해시 뒤에 영수증과 등록의 생성 지문을 맞댄다 — 같은 대조를 여기서 한다.
    재생성이 등록 객체의 지문으로 판정 줄을 찾으므로, 이 대조가 없으면 생성 지문만 다른 영수증으로도 복원한다.
    """
    a = ctx.args
    rc = read_receipt(a.receipt)
    check_receipt_kind(a.registration, a.purpose, rc)
    if seam is not None:
        guard_seam(rc)
    reg = load_registration_for(rc, repo=ctx.repo)
    require_receipt(reg, rc, side="generation")
    ctx.receipt, ctx.reg = rc, reg


def _stage1(ctx: Ctx) -> None:
    a = ctx.args
    if not a.bundles.is_dir():
        # 원장 줄 전에 멈춘다 — 묶음 폴더가 없는 호출은 시도가 아니다(결정 04 의 4절 12)
        raise EntryRejected("BUNDLES_MISSING", "--bundles 가 폴더가 아니다(없다)")
    rc, reg = ctx.receipt, ctx.reg
    anchor = load_anchor(rc, repo=ctx.repo)
    check_anchor(reg, anchor)
    check_processor(reg, load_processor_kwargs(rc, repo=ctx.repo))
    if a.registration == "strict":
        require_receipt(reg, rc, side="scoring")
    else:
        ctx.scoring_gaps = require_probe_scoring(reg, rc)
    ctx.anchor = anchor
    ctx.absorption = check_snapshot(a.snapshot, anchor)
    ctx.provisional = _provisional(ctx)
    if a.registration == "strict" and ctx.provisional:
        raise EntryRejected("PROVISIONAL", "엄격 경로는 잠정 입력을 받지 않는다: " + ", ".join(p["id"] for p in ctx.provisional))


# ── 2 원장 · 2′ 목록 ─────────────────────────────────────────────────────────

def _bundle_files(bundles: Path) -> list[Path]:
    """묶음 이름의 꼴에 맞는 생성 파일만 집는다 — 다른 파일이 섞여도 입력이 늘지 않는다(3판 3-2)."""
    return sorted(p for p in bundles.iterdir() if p.is_file() and _NAME.match(p.name))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _content_keys(gens: list[Path]) -> tuple[dict[Path, str], dict[Path, str]]:
    """묶음마다 내용 해시 — 낼 수 없으면 그 사유. 분류 밖 키 · 빠진 내용 키면 해시를 내지 않는다(07번 §32-2)."""
    keys, errors = {}, {}
    for p in gens:
        try:
            keys[p] = generations_content_sha256(p.read_bytes())
        except ContentKeyError as exc:
            errors[p] = str(exc)
    return keys, errors


def _write_call(ctx: Ctx, gens: list[Path], now: datetime) -> str:
    a = ctx.args
    keys, errors = _content_keys(gens)
    ctx.content_errors = errors
    bundles = []
    for p in gens:
        m = _NAME.match(p.name)
        row = {"tag": m["tag"], "seed_index": int(m["seed"]), "mode": MODE_ECHO if m["echo"] else MODE_MODEL,
               "generations_sha256": _sha(p), "generations_content_sha256": keys.get(p)}
        if p in errors:
            row["content_key_error"] = errors[p][:300]
        bundles.append(row)
    inputs = {str(p.name): _sha(p) for p in gens}
    for name in ("generation_list", "scoring_list", "echo_list", "receipt"):
        inputs[name] = _sha(getattr(a, name))
    payload = {"registration_mode": a.registration, "run_kind": a.purpose, "purpose": a.purpose,
               "generation_sha256": ctx.reg.generation_sha256(),
               "scoring_sha256": "미완" if ctx.scoring_gaps else ctx.receipt.scoring_sha256,
               "bundles": bundles, "generation_list_file_sha256": inputs["generation_list"], "inputs": inputs,
               "scorer_code": scorer_code_digest(ctx.checkout, include_files=False)["combined"], "device": a.device}
    return append_line(canonical_ledger(ctx.checkout), "call", payload, at=now)


def _read_lists(ctx: Ctx):
    a, g, p = ctx.args, ctx.reg.generation, ctx.reg.population
    gen = validate_eval_list(a.generation_list.read_bytes(), where="generation_list")
    pop = validate_eval_list(a.scoring_list.read_bytes(), where="scoring_list")
    echo = validate_eval_list(a.echo_list.read_bytes(), where="echo_list")
    if (g.eval_list_file_sha256, g.eval_list_set_sha256) != (gen.file_sha256, gen.set_sha256):
        raise EntryRejected("LIST_BINDING", "생성 쪽 등록의 목록 해시가 생성 목록과 다르다")
    if p.eval_list_file_sha256 is None and a.registration == "probe":
        if pop.file_sha256 != gen.file_sha256:
            raise EntryRejected("LIST_BINDING", "채점 쪽 목록이 등록되지 않은 탐색 채점은 생성 목록 그대로를 채점한다")
        ctx.notes.append("채점 쪽 목록 해시가 등록되지 않아 채점 목록 = 생성 목록으로 돌았다")
    elif (p.eval_list_file_sha256, p.eval_list_set_sha256) != (pop.file_sha256, pop.set_sha256):
        raise EntryRejected("LIST_BINDING", "채점 쪽 등록의 목록 해시가 채점 목록과 다르다")
    if pop.id_set - gen.id_set:
        raise EntryRejected("LIST_BINDING", "채점 목록이 생성 목록의 부분집합이 아니다")
    if (g.echo_list_file_sha256, g.echo_list_set_sha256) != (echo.file_sha256, echo.set_sha256):
        raise EntryRejected("LIST_BINDING", "등록의 에코 목록 해시가 에코 목록과 다르다(07번 §31-3)")
    if a.registration == "strict" and not len(pop):
        raise EntryRejected("EMPTY_POPULATION", "엄격 경로의 채점 모집단이 비었다(가-3)")
    meta = read_manifest_meta(a.snapshot)
    check_list_split(gen.ids, meta, a.purpose, where="생성 목록")
    check_list_split(pop.ids, meta, a.purpose, where="채점 목록")
    check_list_split(echo.ids, meta, a.purpose, where="에코 목록", echo=True)
    return gen, pop, echo, meta


# ── 3 묶음 ───────────────────────────────────────────────────────────────────

def _projection(ctx: Ctx, gen, echo, mode: str) -> Registration:
    r = ctx.reg
    if r.seeds is None:
        raise EntryRejected("SEEDS_MISSING", "등록에 시드표가 없다 — 곁 파일의 시드를 맞댈 수 없다")
    return Registration(generation_sha256=r.generation_sha256(),
                        coupling_rule=r.metric.coupling_rule or DEFAULT_COUPLING,
                        seeds={int(k): v for k, v in r.seeds.items()}, values=registered_values(r, mode=mode),
                        eval_list=gen, purpose=r.generation.purpose, echo_list=echo, coord_space=r.generation.coord_space,
                        train_rows_digest=r.generation.train_rows_digest)


def _training_inputs(ctx: Ctx, gen_path: Path, sidecar: dict):
    m = _NAME.match(gen_path.name)
    meta_path = ctx.args.train_root / f"{m['tag']}_s{m['seed']}" / "adapter_last.meta.json"
    artifact = None
    if meta_path.is_file():
        am = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(am.get("adapter_sha256"), str) and type(am.get("adapter_step")) is int:
            artifact = TrainingArtifact(am["adapter_sha256"], am["adapter_step"], am.get("train_rows_digest"))
    ledger = None
    rel = sidecar.get("train_ledger_path")
    if isinstance(rel, str) and rel:
        lp = Path(rel) if Path(rel).is_absolute() else ctx.checkout / rel
        try:
            # 연합 원장은 등록의 R 과 칸 정의의 참여자 수로 읽는다 — 기본값이 없다(로컬 · 중앙은 보지 않는다)
            ledger = read_ledger_view(lp, n_rounds=ctx.reg.generation.budget_r,
                                      n_clients=FED_CLIENTS) if lp.is_file() else None
        except LedgerRejected as exc:
            ctx.notes.append(f"{gen_path.name}: 원장을 읽지 못했다 [{exc.code}]")
    return artifact, ledger


def _verify_all(ctx: Ctx, gens: list[Path], gen, pop, echo, meta):
    verified, rejected = [], {}
    seen: dict = {}
    proj = {MODE_MODEL: _projection(ctx, gen, echo, MODE_MODEL), MODE_ECHO: _projection(ctx, gen, echo, MODE_ECHO)}
    for p in gens:
        files = BundleFiles(p)
        if p in ctx.content_errors:
            # 내용 해시를 낼 수 없는 묶음은 채점하지 않는다 — 원시 해시가 맞아도 분류 밖 키가 채점으로 새지 않게(§32-2)
            rejected[files.stem] = ["content_key"]
            continue
        is_echo = bool(_NAME.match(p.name)["echo"])
        try:
            sidecar = json.loads(files.sidecar.read_text(encoding="utf-8")) if files.sidecar.exists() else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            sidecar = {}
        artifact, ledger = (None, None) if is_echo else _training_inputs(ctx, p, sidecar)
        try:
            vb = verify_bundle(files, registration=proj[MODE_ECHO if is_echo else MODE_MODEL], artifact=artifact,
                               ledger=ledger, scoring_population=None if is_echo else list(pop.ids),
                               group_of=dict(meta.group_of), seen_adapters=seen, receipt_time=ctx.receipt.time,
                               image_sha256_of=dict(meta.sha256_of))
        except BundleRejected as exc:
            rejected[files.stem] = sorted(c.value for c in exc.codes())
            continue
        if not is_echo:
            seen[vb.sidecar["scored_adapter_sha256"]] = (vb.tag, vb.seed_index)
        verified.append(vb)
    for mode in (MODE_MODEL, MODE_ECHO):
        group = [vb for vb in verified if vb.mode == mode]
        if len(group) > 1:
            for r in check_env_consistency(group):
                rejected.setdefault(f"env:{mode}", []).append(r.code.value)
    return verified, rejected


# ── 5 카나리아 ───────────────────────────────────────────────────────────────

def _gold(snapshot: Path, ids) -> dict:
    """정답 뷰 — **주어진 id 의 주석 행만**(데이터 쪽 판독기). 정상 이미지는 빈 목록."""
    ids = list(ids)
    df = read_annotations_view(snapshot, ids)
    out: dict[str, list] = {i: [] for i in ids}
    for row in df.itertuples(index=False):
        out[row.image_id].append((str(row.iso_code), [float(row.bbox_x1_px), float(row.bbox_y1_px),
                                                      float(row.bbox_x2_px), float(row.bbox_y2_px)]))
    return out


def _canaries(ctx: Ctx, verified, echo_list) -> tuple[dict, bool]:
    r = ctx.reg
    known = list(_known_codes())
    classes = list(r.metric.class_codes or ())
    if not classes:
        raise EntryRejected("CLASSES_MISSING", "등록에 채점 클래스(metric.class_codes)가 없다")
    fixture = coord_fixture_digest((ctx.checkout / GOLDEN_FIXTURES_REL).read_bytes())
    out: dict = {"echo": [], "literal": []}
    ok = True
    echo_vbs = [vb for vb in verified if vb.mode == MODE_ECHO]
    if echo_vbs:
        gold = _gold(ctx.args.snapshot, echo_list.ids)
        tol = r.canary.echo_max_coord_diff if r.canary.echo_max_coord_diff is not None else 0.0
        for vb in echo_vbs:
            try:
                check_echo_gate(vb.sidecar["coord_cfg_hash"], fixture, r.canary)
            except EchoGateFailed as exc:
                out["echo"].append({"bundle": vb.files.stem, "status": "gate_failed", "reasons": exc.reasons})
                ok = False
                continue
            res = score_echo(vb, gold, classes=classes, max_coord_diff=tol,
                             coupling_rule=r.metric.coupling_rule or DEFAULT_COUPLING, known_iso_codes=known)
            out["echo"].append({"bundle": vb.files.stem, **res.as_dict()})
            ok = ok and res.status == "pass"
    else:
        out["echo"].append({"status": "no_echo_bundle"})
        ok = False
    lit_tol = r.canary.literal_match_tolerance if r.canary.literal_match_tolerance is not None else 0.0005
    for vb in verified:
        res = check_literal(vb, known_iso_codes=known, scoring_iso_codes=classes, tolerance=lit_tol)
        out["literal"].append({"bundle": vb.files.stem, **res.as_dict()})
        ok = ok and res.status == "pass"
    return out, ok


def _known_codes():
    from data.label_map import load_label_map
    return load_label_map().iso_codes()


# ── 진단 (목적 frame_diag) ───────────────────────────────────────────────────

def _diag_rules(ctx: Ctx) -> None:
    """규칙 파일을 **영수증의 커밋에서** 읽는다 — 작업 트리의 규칙 파일을 읽지 않는다(3판 8-1 의 1 · §32-9)."""
    g = ctx.reg.generation
    raw = blob_at(ctx.repo, ctx.receipt.main_commit, g.frame_diag_rules_path)
    if hashlib.sha256(raw).hexdigest() != g.frame_diag_rules_sha256:
        raise EntryRejected("DIAG_RULES_HASH", "규칙 파일의 바이트가 등록의 frame_diag_rules_sha256 과 다르다")
    try:
        ctx.diag_rules, ctx.diag_rules_body = load_rules_file(raw)
    except RulesFileError as exc:
        raise EntryRejected("DIAG_RULES", str(exc)) from None


def _diag_output_rel(ctx: Ctx) -> str:
    return _out_file(ctx, OUT_DIAG).resolve().relative_to(ctx.checkout.resolve()).as_posix()


def _diag_output_text(verdict_sha: str, row: dict) -> str:
    """판정 줄 → 진단 산출물. **줄의 값만으로** 정규 JSON 을 적는다 — 첫 실행과 재생성이 같은 바이트를 낸다(§32-8)."""
    body = {k: v for k, v in row.items() if k not in ("kind", "at", "prev_sha256")}
    body["verdict_line_sha256"] = verdict_sha
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _diag_exit(status: str) -> int:
    return EXIT_OK if status == FRAME_PASS else EXIT_REJECTED


def _diag_existing(ctx: Ctx, now: datetime) -> int | None:
    """이 등록 · 이 산출물 이름의 판정 줄이 있으면 다시 판정하지 않는다. 산출물이 있으면 종료 3, 없으면 **그 줄에서** 다시 만든다(§32-8).

    다시 만드는 호출은 **영수증 검사(1~3)만** 하고 부른다 — 닻 · 스냅샷 · 계획 파일 · 규칙 파일 · 지금 검사기의 지문에 기대지 않는다.
    묶음 · 정답 · 이미지를 열지 않으며 계산하지 않는다. `call` 줄을 쓰지 않는다. 원장 잠금 안에서 부른다.
    """
    ledger = canonical_ledger(ctx.checkout)
    rel, gen_sha = _diag_output_rel(ctx), ctx.reg.generation_sha256()
    hits = [(sha, row) for sha, row in read_ledger(ledger)
            if row["kind"] == "frame_diag_verdict" and row.get("generation_sha256") == gen_sha and row.get("output") == rel]
    if not hits:
        return None
    sha, row = hits[-1]
    out = ctx.checkout / rel
    if out.exists():
        raise EntryRejected("DIAG_ALREADY_JUDGED", "이 진단은 판정이 나 있다 — 같은 묶음을 다시 판정하지 않는다(§31-1 의 4)")
    text = _diag_output_text(sha, row)
    append_line(ledger, "frame_diag_regenerated",
                {"verdict_sha256": sha, "output": rel,
                 "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}, at=now)
    out.parent.mkdir(parents=True, exist_ok=True)
    _out_file(ctx, OUT_DIAG)
    write_new_text(out, text)
    print(f"판정 줄에서 진단 산출물을 다시 만들었다: {rel}", file=sys.stderr)
    return _diag_exit(row["status"])


def _diag_once(ctx: Ctx, gens: list[Path]) -> None:
    """판정 한 번 — 모델 묶음의 **내용 해시**로 판정 줄을 찾는다. 있으면 종료 3(§32-2). 계산 전이다."""
    keys = []
    for p in gens:
        if _NAME.match(p.name)["echo"]:
            continue
        try:
            keys.append(generations_content_sha256(p.read_bytes()))
        except ContentKeyError:
            continue
    if keys and find_verdicts(canonical_ledger(ctx.checkout), keys):
        raise EntryRejected("DIAG_ALREADY_JUDGED", "같은 내용의 묶음에 판정이 이미 나 있다(§32-2) — 다시 판정하지 않는다")


def _diag_not_judged(ctx: Ctx, call_sha: str, why: str, detail) -> int:
    body = {"kind": "frame_diag_not_judged", "call_sha256": call_sha, "reason": why, "detail": detail,
            "generation_sha256": ctx.reg.generation_sha256(), "provisional": ctx.provisional}
    path = _out_file(ctx, OUT_DIAG_NOT_JUDGED)
    path.parent.mkdir(parents=True, exist_ok=True)
    _out_file(ctx, OUT_DIAG_NOT_JUDGED)
    write_new_text(path, json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n")
    print(f"진단을 판정하지 않았다: {why}", file=sys.stderr)
    return EXIT_REJECTED


def _image_sizes(snapshot: Path, ids) -> dict[str, tuple[int, int]]:
    from data.manifest_view import read_manifest_columns
    df = read_manifest_columns(snapshot, ["image_id", "width_px", "height_px"], split_filter=["val"])
    want = set(ids)
    return {str(i): (int(w), int(h)) for i, w, h in zip(df["image_id"], df["width_px"], df["height_px"]) if str(i) in want}


def _pred_boxes(row: dict) -> tuple:
    """줄의 역변환된 예측 박스 — 클래스를 보지 않는다."""
    parsed = row.get("bbox_px_parsed") or {}
    return tuple(tuple(float(x) for x in d["bbox_px"]) for d in parsed.get("defects", [])
                 if isinstance(d, dict) and isinstance(d.get("bbox_px"), list) and len(d["bbox_px"]) == 4)


def _frame_images(vb, gold: dict, meta, sizes: dict) -> tuple[list[FrameImage], int, tuple[int, int]]:
    images, n_limit, whs = [], 0, set()
    for iid, line in zip(vb.image_ids, vb.lines, strict=True):
        row = json.loads(line)
        whs.add(tuple(row["model_input_wh"]))
        pred = _pred_boxes(row)
        golds = tuple(tuple(b) for _, b in gold.get(iid, []))
        w, h = sizes[iid]
        images.append(FrameImage(iid, meta.group_of[iid], w, h, pred, golds))
        if golds and row.get("gen_stop") == "length" and not mutual_pairs(pred, golds):
            n_limit += 1
    if len(whs) != 1:
        raise EntryRejected("DIAG_MODEL_INPUT_WH", "모델 묶음의 줄마다 model_input_wh 가 다르다 — 배율을 하나로 정할 수 없다")
    return images, n_limit, next(iter(whs))


def _frame_inputs(sidecar: dict) -> dict:
    def h(v) -> str:
        return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    out = {k: h(sidecar[k]) for k in FRAME_INPUT_FIELDS}
    cfg = {k: v for k, v in json.loads(sidecar["train_config"]).items() if k not in TRAIN_CONFIG_UNBOUND}
    out["train_config_bound"] = h(cfg)
    return out


def _diag(ctx: Ctx, call_sha: str, verified, rejected, canary_ok: bool, gen, meta, now: datetime) -> int:
    """진단 판정 — 판정 줄을 **산출물보다 먼저** 쓰고 되읽은 줄에서 산출물을 만든다(3판 8-1 의 4)."""
    models = [vb for vb in verified if vb.mode == MODE_MODEL]
    if rejected:
        return _diag_not_judged(ctx, call_sha, "묶음이 거부됐다", rejected)
    if len(models) != 1:
        return _diag_not_judged(ctx, call_sha, "진단은 모델 묶음 하나를 판정한다", [vb.files.stem for vb in models])
    vb = models[0]
    # 대역 실행(승인 구현이 아닌 묶음)은 진단에서 묶음 검증이 거부한다(목적이 리허설이 아니다) — 위의 "묶음이 거부됐다" 로 온다
    if not canary_ok:
        return _diag_not_judged(ctx, call_sha, "카나리아(에코 · 문자 일치)가 통과가 아니다", None)
    gold = _gold(ctx.args.snapshot, gen.ids)
    images, n_limit, wh = _frame_images(vb, gold, meta, _image_sizes(ctx.args.snapshot, gen.ids))
    try:
        v = judge(images, wh, ctx.diag_rules)
    except ValueError as exc:
        raise EntryRejected("DIAG_RULES", str(exc)) from None
    g = ctx.reg.generation
    verdict = v.as_dict()
    payload = {"run_kind": "frame_diag", "call_sha256": call_sha, "generation_sha256": ctx.reg.generation_sha256(),
               "bundles": [{"tag": vb.tag, "seed_index": vb.seed_index,
                            "generations_sha256": vb.sidecar["generations_sha256"],
                            "generations_content_sha256": generations_content_sha256(vb.files.generations.read_bytes())}],
               **verdict, "design_samples": requirement_status(ctx.diag_rules_body, verdict), "n_limit_no_pair": n_limit,
               "rules_path": g.frame_diag_rules_path, "rules_sha256": g.frame_diag_rules_sha256,
               "frame_inputs": _frame_inputs(vb.sidecar), "provisional": ctx.provisional, "output": _diag_output_rel(ctx)}
    ledger = canonical_ledger(ctx.checkout)
    sha = append_line(ledger, "frame_diag_verdict", payload, at=now)
    row = next(r for s, r in read_ledger(ledger) if s == sha)
    out = ctx.checkout / row["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    _out_file(ctx, OUT_DIAG)
    write_new_text(out, _diag_output_text(sha, row))
    return _diag_exit(row["status"])


# ── 5′ 축 배율 카나리아 ─────────────────────────────────────────────────────

def _axis_ratio(ctx: Ctx, verified, gen, meta) -> list[dict]:
    """모델 묶음마다 축 배율 카나리아의 상태 · 표본 수(리허설 명세 2판 §3 ② 나). 정답은 생성 목록의 val 행에서만.

    쓰는 쪽 판정기가 `canaries.axis_ratio` 를 읽는다 — 블록이 없으면 호출되지 않은 것으로 본다.
    """
    models = [vb for vb in verified if vb.mode == MODE_MODEL]
    if not models:
        return []
    gold = _gold(ctx.args.snapshot, gen.ids)
    c = ctx.reg.canary
    out = []
    for vb in models:
        images = [FrameImage(iid, meta.group_of[iid], 0, 0, _pred_boxes(json.loads(line)),
                             tuple(tuple(b) for _, b in gold.get(iid, [])))
                  for iid, line in zip(vb.image_ids, vb.lines, strict=True)]
        out.append({"bundle": vb.files.stem,
                    **axis_ratio_canary(images, threshold=c.axis_ratio_threshold, min_pairs=c.axis_ratio_min_pairs)})
    return out


# ── 6 · 7 리허설 ─────────────────────────────────────────────────────────────

def _rehearsal(ctx: Ctx, verified, gen) -> tuple[dict, bool]:
    r = ctx.reg
    known = list(_known_codes())
    classes = list(r.metric.class_codes or ())
    coupling = r.metric.coupling_rule or DEFAULT_COUPLING
    models = [vb for vb in verified if vb.mode == MODE_MODEL]
    gates = judge_rehearsal_gates(models, coupling_rule=coupling, known_iso_codes=known, scoring_iso_codes=classes) \
        if models else {}
    probe = adversarial_probe(models[0], coupling_rule=coupling, known_iso_codes=known, scoring_iso_codes=classes) \
        if models else {"status": "no_model_bundle"}
    gold = _gold(ctx.args.snapshot, gen.ids) if models else {}
    smoke = [{"bundle": vb.files.stem, **rehearsal_metric_smoke(vb, gold, classes=classes, coupling_rule=coupling,
                                                                  known_iso_codes=known)} for vb in models]
    ok = bool(models) and all(g.status == "pass" for g in gates.values()) and probe.get("status") == "pass" \
        and all(s["completed"] for s in smoke)
    return {"gates": {k: g.as_dict() for k, g in gates.items()}, "adversarial_probe": probe, "metric_smoke": smoke}, ok


def _status(verified, rejected, canary_ok: bool, rehearsal_ok: bool) -> str:
    """리허설의 판정 상태. **대역 실행**(승인 구현이 아닌 묶음 — `stand_in`)은 통과를 내지 않는다(3판 1-1-다 · 07번 §30-4)."""
    if any(vb.notes.get("stand_in") for vb in verified):
        return "stand_in"
    return "pass" if not rejected and canary_ok and rehearsal_ok else "fail"


def _write(ctx: Ctx, call_sha: str, verified, rejected, canaries, rehearsal, status: str) -> None:
    a, r = ctx.args, ctx.reg
    common = {"run_kind": "rehearsal", "registration_mode": a.registration, "purpose": a.purpose, "status": status,
              "population_id": r.population.population_id, "call_sha256": call_sha,
              "fingerprints": {"generation_sha256": r.generation_sha256(),
                               "scoring_sha256": "미완" if ctx.scoring_gaps else ctx.receipt.scoring_sha256,
                               "registration_file_sha256": ctx.receipt.registration_file_sha256},
              "provisional": ctx.provisional}
    score = {**common, "scoring_gaps": ctx.scoring_gaps, "notes": ctx.notes,
             "bundles": {"verified": [vb.files.stem for vb in verified], "rejected": rejected,
                         "stand_in": [vb.files.stem for vb in verified if vb.notes.get("stand_in")],
                         "fault_events": {vb.files.stem: vb.notes["fault_events"] for vb in verified
                                          if vb.notes.get("fault_events")}},
             "canaries": canaries, "metric_smoke": rehearsal["metric_smoke"],
             "scorer_code": scorer_code_digest(ctx.checkout),
             "note": "리허설 산출물에는 지표 값을 싣지 않는다(진입점 미니스펙 3판 4-2). 대표 채택에 쓰지 않는다"}
    gates = {**common, "gates": rehearsal["gates"], "adversarial_probe": rehearsal["adversarial_probe"]}
    for rel, body in ((OUT_SCORE, score), (OUT_GATES, gates)):
        path = _out_file(ctx, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        _out_file(ctx, rel)
        write_new_text(path, json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n")


# ── 에코 전용 — 학습 전 관문 ─────────────────────────────────────────────────

def _echo_reasons(gens: list[Path], verified, rejected: dict, canaries: dict) -> list[str]:
    """통과가 아닌 까닭 — `ECHO_REASONS` 의 순서로, 겹치지 않게. 비면 통과다."""
    out = set()
    if not gens:
        out.add("no_echo_bundle")
    if rejected:
        out.add("echo_bundle_rejected")
    for e in canaries["echo"]:
        if e.get("status") == "gate_failed":
            out.add("echo_gate_failed")
        elif e.get("status") not in ("pass", "no_echo_bundle"):
            out.add("echo_mismatch")
    if any(x.get("status") != "pass" for x in canaries["literal"]):
        out.add("literal_mismatch")
    if any(vb.notes.get("stand_in") for vb in verified):
        out.add("stand_in")
    return [r for r in ECHO_REASONS if r in out]


def _write_echo_gate(ctx: Ctx, body: dict) -> None:
    """배타 생성 — 다만 **같은 바이트**가 이미 있으면 받는다(같은 입력의 재개). 다른 바이트면 덮지 않고 종료 3."""
    path = _out_file(ctx, OUT_ECHO_GATE)
    text = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == text.encode("utf-8"):
            return
        raise EntryRejected("ECHO_GATE_OUT_EXISTS", "에코 관문 산출이 이미 있고 이번 결과와 다르다 — 덮지 않는다")
    path.parent.mkdir(parents=True, exist_ok=True)
    _out_file(ctx, OUT_ECHO_GATE)
    write_new_text(path, text)


def _echo_only(ctx: Ctx, seam: Seam | None) -> int:
    """학습 전 에코 관문 — **정본 원장에 아무것도 쓰지 않는다**(시도로 세지 않는다). 리허설 · 진단 목적만 받는다."""
    a = ctx.args
    try:
        if a.registration != "probe" or a.purpose not in ECHO_ONLY_PURPOSES:
            raise EntryRejected("ECHO_ONLY_PURPOSE",
                                f"에코 전용 검사는 탐색(probe)의 {list(ECHO_ONLY_PURPOSES)} 목적만 받는다 — 본실험 엄격 경로와 섞지 않는다")
        _stage0(ctx)
        _stage1_receipt(ctx, seam)
        _stage1(ctx)
        found = _bundle_files(a.bundles)
        gens = [p for p in found if _NAME.match(p.name)["echo"]]
        ctx.content_errors = _content_keys(gens)[1]          # 원장 줄 없이 내용 해시만 — 낼 수 없는 묶음은 검증으로 가지 않는다
        gen, pop, echo, meta = _read_lists(ctx)
        verified, rejected = _verify_all(ctx, gens, gen, pop, echo, meta)
        canaries, _ok = _canaries(ctx, verified, echo)
        reasons = _echo_reasons(gens, verified, rejected, canaries)
        status = "fail" if reasons else "pass"
        inputs = {p.name: _sha(p) for p in gens}
        for name in ("generation_list", "echo_list", "receipt"):
            inputs[name] = _sha(getattr(a, name))
        body = {"kind": "echo_gate", "status": status, "reasons": reasons, "purpose": a.purpose,
                "registration_mode": a.registration,
                "fingerprints": {"generation_sha256": ctx.reg.generation_sha256(),
                                 "registration_file_sha256": ctx.receipt.registration_file_sha256},
                "inputs": inputs,
                # 모델 묶음은 싣지 않는다 — 학습 전에 남긴 산출을, 모델 묶음이 붙은 뒤의 재개가 같은 바이트로 다시 받게(재개 동일성)
                "bundles": {"verified": [vb.files.stem for vb in verified], "rejected": rejected},
                "canaries": canaries, "provisional": ctx.provisional, "notes": ctx.notes,
                "scorer_code": scorer_code_digest(ctx.checkout, include_files=False)["combined"],
                "ledger": "쓰지 않았다 — 이 호출은 시도로 세지 않는다"}
        _write_echo_gate(ctx, body)
        others = [p.name for p in found if p not in gens]
        if others:
            print("에코 전용 — 열지 않은 모델 묶음: " + " · ".join(others), file=sys.stderr)
    except (EntryRejected, ReceiptMissing, RegistrationIncomplete, RegistrationInvalid, BundleRejected,
            UpstreamContractError) as exc:
        print(f"진입 전 거부: {exc}", file=sys.stderr)
        return EXIT_ENTRY
    return EXIT_OK if status == "pass" else EXIT_REJECTED


# ── 입구 ─────────────────────────────────────────────────────────────────────

def _diag_judge(ctx: Ctx, gens: list[Path], call_sha: str, now: datetime) -> int:
    """판정 한 번 — 내용 해시 조회부터 판정 줄 추가까지 **원장 잠금 하나 안에서**(병합 전 검토 3). 동시 호출은 기다렸다가 조회에서 멈춘다."""
    with ledger_lock(canonical_ledger(ctx.checkout)):
        _diag_once(ctx, gens)
        gen, pop, echo, meta = _read_lists(ctx)
        verified, rejected = _verify_all(ctx, gens, gen, pop, echo, meta)
        _canaries_out, canary_ok = _canaries(ctx, verified, echo)
        return _diag(ctx, call_sha, verified, rejected, canary_ok, gen, meta, now)


def main(argv: list[str] | None = None, *, _seam: Seam | None = None, _now: datetime | None = None) -> int:
    args = _parser().parse_args(argv)
    checkout = _seam.checkout if _seam is not None else main_checkout()
    repo = _seam.repo if _seam is not None else checkout
    ctx = Ctx(args, checkout, repo)
    if args.echo_only:
        return _echo_only(ctx, _seam)
    now = _now or datetime.now().astimezone()
    diag = args.purpose == "frame_diag"
    try:
        _stage0(ctx)
        _stage1_receipt(ctx, _seam)
        if diag:
            # 재생성은 영수증 검사 바로 뒤에서 가른다 — 스냅샷 · 계획 · 지금 검사기가 없거나 바뀌어도 판정 줄에서 되살린다(§32-8)
            with ledger_lock(canonical_ledger(ctx.checkout)):
                regenerated = _diag_existing(ctx, now)
            if regenerated is not None:
                return regenerated
        _stage1(ctx)
        if diag:
            _diag_rules(ctx)
    except (EntryRejected, ReceiptMissing, RegistrationIncomplete, RegistrationInvalid, LedgerBusy) as exc:
        print(f"진입 전 거부: {exc}", file=sys.stderr)
        return EXIT_ENTRY
    gens = _bundle_files(args.bundles)
    try:
        call_sha = _write_call(ctx, gens, now)
    except LedgerBusy as exc:
        print(f"진입 전 거부: {exc}", file=sys.stderr)
        return EXIT_ENTRY
    try:
        if diag:
            return _diag_judge(ctx, gens, call_sha, now)
        gen, pop, echo, meta = _read_lists(ctx)
        verified, rejected = _verify_all(ctx, gens, gen, pop, echo, meta)
        canaries, canary_ok = _canaries(ctx, verified, echo)
    except (EntryRejected, BundleRejected, UpstreamContractError, LedgerBusy) as exc:
        print(f"진입 전 거부: {exc}", file=sys.stderr)
        return EXIT_ENTRY
    if args.registration == "strict":
        if not canary_ok:
            print("엄격 경로의 카나리아가 통과가 아니다 — 진입 전 관문", file=sys.stderr)
            return EXIT_ENTRY
        print("엄격 경로의 채점 단계가 아직 없다: " + " · ".join(STRICT_PENDING), file=sys.stderr)
        return EXIT_ENTRY
    canaries["axis_ratio"] = _axis_ratio(ctx, verified, gen, meta)
    rehearsal, rehearsal_ok = _rehearsal(ctx, verified, gen)
    status = _status(verified, rejected, canary_ok, rehearsal_ok)
    try:
        _write(ctx, call_sha, verified, rejected, canaries, rehearsal, status)
    except EntryRejected as exc:            # 0 단계 뒤에 산출 경로가 밖으로 이어졌다 — 통제된 사유로 멈춘다
        print(f"산출을 쓰지 않았다: {exc}", file=sys.stderr)
        return EXIT_ENTRY
    return EXIT_OK if status == "pass" else EXIT_REJECTED


if __name__ == "__main__":
    raise SystemExit(main())
