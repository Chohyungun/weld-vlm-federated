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

import pandas as pd
import yaml

from data import manifest_view as _MV
from data.manifest_io import ManifestError
from data.manifest_view import read_annotations_view, read_annotations_view_eval
from evaluation.actuals import registered_values
from evaluation.adapters_v14 import UpstreamContractError
from evaluation.bundle_v14 import (DEVICE_LABEL_CANONICAL, DEVICE_LABEL_NONCANONICAL, MODE_ECHO, MODE_MODEL, BundleFiles,
                                   Registration, TrainingArtifact, check_env_consistency, verify_bundle)
from evaluation.content_key import ContentKeyError, generations_content_sha256
from evaluation import known_answer as KA
from evaluation.coord_check import coord_fixture_digest
from evaluation.early_coord_check import EarlyCheckRejected, load_rules
from evaluation.ecc_frame_check import frame_check_by_a3_substitute, frame_check_by_model_free, frame_check_by_permit
from evaluation.entry_gate import (COMMITTED_KINDS, STRICT_OUT_REL, EntryRejected, anchor_rev, blob_at, check_anchor,
                                   check_list_split, check_main_full_eval, check_main_sample, check_mode, check_processor,
                                   check_receipt_kind, check_snapshot, guard_seam,
                                   canonical_ledger, load_anchor, load_processor_kwargs, load_registration_for,
                                   main_checkout, read_manifest_meta)
from evaluation import strict_main as SM
from evaluation.eval_list import validate_eval_list
from evaluation.eval_registration import AUXILIARY_V6, EvalBindingRefused, load_bound_eval
from evaluation.ledger_v14 import FED_CLIENTS, LedgerRejected, read_ledger_view
from evaluation.prereg_unified import (MAIN_FRAME_DIAG_A3_SUBSTITUTE, MAIN_FRAME_DIAG_MEANING, MAIN_FRAME_DIAG_MODEL_FREE,
                                       EchoGateFailed, ReceiptMissing,
                                       RegistrationIncomplete, RegistrationInvalid, check_echo_gate, read_receipt,
                                       require_probe_scoring, require_receipt)
from evaluation.provenance import scorer_code_digest, write_new_text
from evaluation.reject_v14 import BundleRejected
from evaluation.recovery_interval import registered_mismatches as ci_convention_mismatches
from evaluation.rule5 import VERDICT_AUTHORITY as RULE5_REGISTERED
from evaluation.rule5 import registered_inputs as rule5_inputs
from evaluation.sample_weights import WEIGHTING_NONE, WEIGHTING_SAMPLE, SampleSideError, read_side
from evaluation.scoring_layer import (adversarial_probe, check_literal, judge_rehearsal_gates, rehearsal_metric_smoke,
                                      records_from_verified, score_echo, score_registered)
from evaluation.frame_diag import PASS as FRAME_PASS
from evaluation.frame_diag import FrameImage, judge, mutual_pairs
from evaluation.axis_ratio_canary import axis_ratio_canary
from evaluation.frame_diag_rules import RulesFileError, load_rules_file, requirement_status
from evaluation.scoring_ledger import LedgerBusy, LedgerChainBroken, append_line, find_verdicts, ledger_lock, read_ledger

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
부르는 자리는 `_gold_eval` 하나다(표본 모드의 단계 6, 35번 §1)."""
EVAL_VIEW_READER = f"{EVAL_VIEW.__module__}.{EVAL_VIEW.__qualname__}"
EVAL_VIEW_CONDITIONS = ("영수증 종류가 main", "영수증 커밋이 본체 refs/heads/main 의 조상", "등록의 목적이 main")
"""그 뷰를 부르는 조건 — 셋이 모두 설 때만 부른다. 판독기도 같은 조건(과 그 밖의 넷)을 코드로 건다(판정 10 의 1절)."""
STRICT_PENDING = ("전량 모드의 채점 단계 — 전량의 재표집 층(07번 §39-4-1)이 정해지지 않았다",
                  "축 배율 카나리아의 진단 규칙 판(07번 §31-1 의 6) — 엄격에서는 사람 확인 대기로 싣고 막지 않는다(3판 1-1-다)")
"""엄격 채점에 아직 없는 것. 등록 지표(M1) · 규칙 ⑤ 등록 판정 · 구간 규약 · 평가 정답 뷰 · 표본 모드는 들어왔다(35번 §1).
첫째가 남아 있는 동안 전량 모드(가중 `none`)의 엄격 채점은 카나리아 뒤에서 종료 3 이다. 둘째는 산출에 `not_run` 으로 적는다."""
MODE_FULL, MODE_SAMPLE = "full", "sample"
"""엄격 · 본실험의 모집단 모드 — 등록의 `population.weighting` 이 정한다(`none` → 전량, `stratified_g_over_n` → 표본)."""
STRICT_OUT_PREFIX = "score/main_"
"""엄격 산출의 이름 — `score/main_<population_id>_<채점 지문 앞 12자>.json`. 배타 생성이라 같은 등록의 두 번째 산출은 쓰지 않는다 —
채점 규칙만 고쳐 다시 등록하면(채점 지문이 바뀐다) 새 이름이 된다(실험설계 8-5 의 채점 규칙 개정 재채점)."""
_POPULATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ECC_RULES_REL = "configs/early_coord_check.yaml"
"""조기 좌표 점검의 규칙 파일. 본실험 산출의 프레임 표지는 등록 `canary.frame_diag_sha256` 이 가리키는 허가 파일(`--ecc-permit`)을
**영수증 커밋의** 이 규칙으로 다시 본다(`evaluation.ecc_frame_check` — 의사결정로그 10-07, 등록 판 5 추기 10-17).
산출의 표지는 `FRAME_CHECKED_BY_ECC` 다(앞 판의 `FRAME_NOT_CHECKED` — 칸이 있는지만 보던 것 — 를 대신한다). 허가가 사용자 결정으로
면제되면 모델 없는 통과 산출 · 면제 파일(`--frame-admission` · `--ecc-waiver`)의 묶음을 보고 `FRAME_CHECKED_MODEL_FREE_ONLY` 를 싣는다."""
DECISION_LOG_REL = "docs/의사결정로그.md"
"""면제 파일의 절 제목을 찾는 의사결정로그 — **영수증 커밋의** 것을 읽는다."""
GOLD_READER_REL = "data/manifest_view.py"
"""엄격 정답 판독기의 자리. 채점 지문(`provenance.SCORER_FILES`) 밖이라 산출에 그 파일의 해시를 따로 싣는다 — 지문 목록에 더하는 것은
고정 해시 파일(`provenance.py`)을 고치는 일이라 따로 묻는다(35번 §1)."""

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
    mode: str | None = None
    """엄격 · 본실험의 모집단 모드(`MODE_FULL` · `MODE_SAMPLE`). 단계 1 이 등록의 가중으로 정한다."""
    eval_doc: dict | None = None
    eval_doc_sha256: str | None = None
    eval_bound: object = None
    """네 소비자 공통 판독의 결과(`evaluation.eval_registration.BoundEval`) — 경로 · 바이트 · 프로필. 단계 1 이 정한다."""
    sealed: dict | None = None
    side_raw: bytes | None = None
    side: dict | None = None
    baseline_raw: bytes | None = None
    out_rel: str | None = None
    """엄격 산출의 자리(`--out` 아래) — 단계 1 이 정하고 원장 줄 전에 비어 있는지 본다."""
    frame: dict | None = None
    """본실험 산출의 프레임 표지(`FRAME_CHECKED_BY_ECC` · 허가 sha) — 단계 1 이 허가 파일을 본 뒤 정한다."""
    known_answer: dict | None = None
    """기지답 재현의 기록 — 단계 1 이 영수증 커밋의 픽스처로 다시 낸 계수표 sha 가 등록과 같을 때만 선다."""
    call_scorer: str | None = None
    """원장 `call` 줄에 적은 채점기 지문 — 산출을 쓸 때의 지문과 맞대 `code_drift` 를 싣는다(3판 3-4)."""


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="통합형 본채점 진입점")
    ap.add_argument("--registration", required=True, choices=("strict", "probe"))
    ap.add_argument("--purpose", required=True, choices=("main", "rehearsal", "frame_diag"))
    for name in ("receipt", "snapshot", "bundles", "generation-list", "scoring-list", "echo-list", "train-root", "out"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    ap.add_argument("--plan", type=Path, default=None, help="계획 파일 — 등록에 plan_sha256 이 있으면 필수")
    ap.add_argument("--side", type=Path, default=None,
                    help="표본 곁 파일(image_id,stratum,G_h,n_h) — 엄격 표본 모드에서 필수, 봉인한 side_sha256 과 맞댄다")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="등록 대조선의 예측(JSON 줄) — 엄격 표본 모드에서 필수, 등록의 baseline.prediction_sha256 과 맞댄다")
    ap.add_argument("--ecc-permit", type=Path, default=None,
                    help="조기 좌표 점검 허가 파일(시드 폴더의 ecc_permit.json) — 엄격 표본 모드에서 필수, 등록의 "
                         "canary.frame_diag_sha256 과 맞대고 영수증 커밋의 규칙으로 통과를 다시 낸다")
    ap.add_argument("--frame-admission", type=Path, default=None,
                    help="허가 면제 때 — 모델 없는 네 검사 통과 산출(ECC_MODEL_FREE_ADMISSION.json). --ecc-waiver 와 함께, --ecc-permit 과 함께 받지 않는다")
    ap.add_argument("--ecc-waiver", type=Path, default=None,
                    help="허가 면제 파일(학습 쪽 --ecc-permit-waiver 의 그 파일) — --frame-admission 과 함께")
    ap.add_argument("--ecc-substitute", type=Path, default=None,
                    help="2 차 체제 — A3 대체 승인서(ecc_a3_substitute.json, 증거는 같은 폴더). 등록 canary.frame_diag_sha256 과 맞대고 "
                         "영수증 커밋 · A3 커밋 · 증거로 다시 검증한다. --ecc-permit · --frame-admission · --ecc-waiver 와 함께 받지 않는다")
    ap.add_argument("--device", required=True)
    ap.add_argument("--echo-only", action="store_true",
                    help="학습 전 에코 관문만 — 원장에 쓰지 않는다(리허설 · 진단 목적만)")
    ap.add_argument("--aux-train-extractor", type=Path, default=None,
                    help="판 6 보조 지표 — 학습 쪽 추출기 파일(vlm/narrative.py). 주면 식별자(EXTRACTOR_ID) · LF sha256 을 등록과 맞댄다")
    ap.add_argument("--aux-prompt", type=Path, default=None,
                    help="판 6 보조 지표 — 생성 프롬프트 파일. 주면 generation.prompt_sha256 과 맞댄다")
    ap.add_argument("--aux-vectors", type=Path, default=None,
                    help="판 6 보조 지표 — 확인 파일(정본 추출 벡터 narrative_extract.v1.json). 주면 실제 바이트를 "
                         "narrative.extract_vectors_sha256 과 맞댄다")
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
        for name in ("bundles", "generation_list", "scoring_list", "echo_list", "train_root", "receipt", "side", "baseline",
                     "ecc_permit", "frame_admission", "ecc_waiver", "ecc_substitute"):
            if getattr(a, name) is not None and _under(getattr(a, name), co / REHEARSAL_OUT_REL):
                raise EntryRejected("INPUT_UNDER_REHEARSAL", f"엄격의 입력({name})이 리허설 루트 아래다(가-4)")
    else:
        if any(getattr(a, n) is not None for n in ("side", "baseline", "ecc_permit", "frame_admission", "ecc_waiver",
                                                   "ecc_substitute")):
            raise EntryRejected("STRICT_ONLY_INPUT", "표본 곁 파일 · 대조선 예측 · 조기 점검 허가 파일은 엄격 본채점의 입력이다")
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
    if a.registration == "strict" and a.purpose == "main":
        _strict_population(ctx)


def _strict_population(ctx: Ctx) -> None:
    """엄격 · 본실험의 모집단 모드(35번 §1). 등록의 `population.weighting` 이 정한다 — 원장 줄 전이라 거부는 시도가 아니다.

    - `none` → **전량 모드**. 목록은 `check_main_full_eval`(전량 계약)로 본다. 곁 파일을 주면 거부한다.
    - `stratified_g_over_n` → **표본 모드**. 평가 등록 설정(`configs/eval_registration.yaml`)과 base 설정을 **영수증의 커밋에서**
      읽어(작업 트리를 읽지 않는다) 코드 상수 · base · 등록 객체와 맞댄다. 곁 파일의 바이트가 봉인한 `side_sha256` 이고, 대조선
      예측의 바이트가 등록의 `baseline.prediction_sha256` 이어야 한다.

    표본 모드의 관문 순서 — 아래 일곱이 이 함수(원장 줄 전)이고, 표본 관문 `check_main_sample` 은 그 **뒤** 목록 단계다. 생성 쪽 사전
    점검이 1 · 2 를 돌지 않으면 결속이 어긋난 등록이 생성(다섯 칸 × 표본 전부)을 다 쓴 뒤 여기서 거부된다 — 본실험 등록 짓기
    (`vlm.main_registration`)는 짓기 · 영수증에서 같은 2 를 미리 돈다.

    | # | 검사 | 사유 코드 |
    |---|---|---|
    | 1 | 평가 등록 설정의 꼴 · 코드 상수 · base(`registration_problems`) | `EVAL_REGISTRATION` |
    | 2 | 등록 ↔ 설정 결속 — 봉인 주 표본의 목록 두 해시 · 장수 · 묶음 수, 묶음 단위, 시드, 대표 지표 칸, 생성 한도
    |   | (`strict_binding_problems`) | `EVAL_REGISTRATION` |
    | 3 | 곁 파일 바이트 = 봉인 `side_sha256` | `SIDE_MISSING` · `SIDE_HASH` |
    | 4 | 대조선 예측 바이트 = 등록 `baseline.prediction_sha256` | `BASELINE_MISSING` · `BASELINE_HASH` |
    | 5 | 규칙 ⑤ 의 등록 입력 · 구간 규약 · 모집단 id 꼴 · 산출 자리 | `RULE5_REGISTRATION` · `CI_CONVENTION` · `POPULATION_ID` ·
    |   |   | `OUT_EXISTS` |
    | 6 | 기지답 — 영수증 커밋의 픽스처로 다시 낸 계수표 sha = 등록 `canary.known_answer_sha256` | `KNOWN_ANSWER_FIXTURE` ·
    |   |   | `KNOWN_ANSWER_MISMATCH` |
    | 7 | 프레임 — 허가 파일(`--ecc-permit`) sha = 등록 `canary.frame_diag_sha256`, 영수증 커밋의 규칙으로 통과를 다시 낸다. 허가 면제면 |
    |   | 통과 산출 · 면제 파일(`--frame-admission` · `--ecc-waiver`)의 묶음 sha = 등록 칸, 네 검사 통과 · 규칙 지문 · 면제 칸 · 절 제목 |
    |   | (`evaluation.ecc_frame_check`) | `FRAME_PERMIT_MISSING` · `FRAME_INPUT_CONFLICT` · `FRAME_PERMIT` |
    | 뒤 | 표본 관문 — 생성 목록 = 채점 목록 = 봉인 주 표본, 묶음이 쪼개지지 않음, 곁 파일이 목록과 같은 행 | `MAIN_*` |
    """
    a, reg, rc = ctx.args, ctx.reg, ctx.receipt
    weighting = reg.population.weighting
    if weighting == WEIGHTING_NONE:
        if a.side is not None:
            raise EntryRejected("SIDE_IN_FULL_MODE", "전량 모드(가중 none)의 등록에 표본 곁 파일이 주어졌다")
        if a.baseline is not None:
            raise EntryRejected("BASELINE_IN_FULL_MODE", "전량 모드는 채점 단계가 없다 — 대조선 예측을 받지 않는다")
        ctx.mode = MODE_FULL
        return
    if weighting != WEIGHTING_SAMPLE:
        raise EntryRejected("WEIGHTING", f"모집단 가중 {weighting!r} 은 엄격 본채점의 어휘가 아니다")
    rev = anchor_rev(rc)
    # 네 소비자 공통 판독(판 6 연결 2 판 §2-2) — 영수증 커밋의 blob 만: 판 5 는 configs/eval_registration.yaml, 신판 판 6 은 같은
    # 줄기의 .eval.yaml(없으면 거부 — 옛 경로 · 작업 트리 · 명령줄로 대신하지 않는다). 꼴 · 코드 상수 · base · 등록 결속 · 출처까지.
    try:
        bound = load_bound_eval(repo=ctx.repo, anchor_rev=rev, registration_rel=rc.registration_path, registration=reg)
    except EvalBindingRefused as exc:
        raise EntryRejected("EVAL_REGISTRATION", f"[{exc.code}] 영수증 커밋의 평가 설정 결속 — {len(exc.problems)}건: "
                                                 + " · ".join(exc.problems[:3])) from None
    doc = dict(bound.doc)
    sample = doc["population"]["sample"]
    sealed = dict(sample["sealed"][str(sample["main_size"])])
    if a.side is None or not a.side.is_file():
        raise EntryRejected("SIDE_MISSING", "표본 모드는 봉인한 곁 파일(--side)이 있어야 한다 — 없거나 파일이 아니다")
    side_raw = a.side.read_bytes()
    if hashlib.sha256(side_raw).hexdigest() != sealed["side_sha256"]:
        raise EntryRejected("SIDE_HASH", "곁 파일의 바이트가 봉인한 side_sha256 과 다르다")
    if a.baseline is None or not a.baseline.is_file():
        raise EntryRejected("BASELINE_MISSING", "표본 모드는 등록 대조선의 예측(--baseline)이 있어야 한다(규칙 ⑤) — 없거나 파일이 아니다")
    baseline_raw = a.baseline.read_bytes()
    if hashlib.sha256(baseline_raw).hexdigest() != reg.baseline.prediction_sha256:
        raise EntryRejected("BASELINE_HASH", "대조선 예측의 바이트가 등록의 baseline.prediction_sha256 과 다르다")
    try:
        rule5_inputs(reg)
    except ValueError as exc:
        raise EntryRejected("RULE5_REGISTRATION", f"규칙 ⑤ 의 등록 입력이 쓸 수 없다: {exc}") from None
    bad_ci = ci_convention_mismatches(reg.ci)
    if bad_ci:
        raise EntryRejected("CI_CONVENTION", f"등록의 구간 규약이 구현과 다르다 — {len(bad_ci)}칸: " + " · ".join(bad_ci[:3]))
    pid = reg.population.population_id
    if not isinstance(pid, str) or not _POPULATION_ID.match(pid):
        raise EntryRejected("POPULATION_ID", "population_id 가 산출 이름에 쓸 수 있는 꼴이 아니다([A-Za-z0-9_.-])")
    rel = f"{STRICT_OUT_PREFIX}{pid}_{reg.scoring_sha256()[:12]}.json"
    if _out_file(ctx, rel).exists():
        raise EntryRejected("OUT_EXISTS", f"같은 모집단의 엄격 산출({rel})이 이미 있다 — 덮지 않고 시도로 세지 않는다")
    try:                                 # 기지답 — 영수증 커밋의 픽스처를 지금 채점기로 다시 센다(07번 597행)
        ka_raw = blob_at(ctx.repo, rev, KA.FIXTURE_REL)
        ka_sha = KA.known_answer_sha256(ka_raw)
    except (EntryRejected, KA.KnownAnswerError) as exc:
        raise EntryRejected("KNOWN_ANSWER_FIXTURE", f"영수증 커밋의 기지답 픽스처를 읽지 못했다: {exc}") from None
    if ka_sha != reg.canary.known_answer_sha256:
        raise EntryRejected("KNOWN_ANSWER_MISMATCH", "기지답 픽스처로 다시 낸 계수표의 sha256 이 등록 canary.known_answer_sha256 과 "
                                                    "다르다 — 매칭 · 경계 규약이 등록 때와 다르다")
    waived = (a.frame_admission, a.ecc_waiver)
    sub = getattr(a, "ecc_substitute", None)
    if sum(x is not None for x in (a.ecc_permit, sub)) + (1 if any(x is not None for x in waived) else 0) > 1:
        raise EntryRejected("FRAME_INPUT_CONFLICT", "허가 파일 · 허가 면제(통과 산출 · 면제 파일) · A3 대체 승인서는 하나만 준다")
    use_permit = a.ecc_permit is not None
    if sub is not None:
        if not sub.is_file():
            raise EntryRejected("FRAME_PERMIT_MISSING", "A3 대체 승인서(--ecc-substitute)가 없다")
        try:
            frame = frame_check_by_a3_substitute(sub.read_bytes(), evidence_dir=sub.parent,
                                                 registered_sha256=reg.canary.frame_diag_sha256, repo=ctx.repo, rev=rev)
        except EarlyCheckRejected as exc:
            raise EntryRejected("FRAME_PERMIT", f"A3 대체 승인서가 등록 · 영수증 커밋 · 증거와 맞지 않는다 — {exc}") from None
        meaning = MAIN_FRAME_DIAG_A3_SUBSTITUTE
    elif not all(x is not None and x.is_file() for x in ((a.ecc_permit,) if use_permit else waived)):
        raise EntryRejected("FRAME_PERMIT_MISSING", "표본 모드는 등록이 가리키는 조기 좌표 점검 허가 파일(--ecc-permit), 허가 "
                                                   "면제의 통과 산출 · 면제 파일(--frame-admission · --ecc-waiver), 또는 A3 대체 "
                                                   "승인서(--ecc-substitute)가 있어야 한다")
    else:
        meaning = MAIN_FRAME_DIAG_MEANING if use_permit else MAIN_FRAME_DIAG_MODEL_FREE
    try:
        if sub is not None:
            pass                                                    # 위에서 판정했다 — 규칙 파일을 읽지 않는다(점검 규칙 밖)
        elif use_permit:
            rules = load_rules(blob_at(ctx.repo, rev, ECC_RULES_REL))
            frame = frame_check_by_permit(a.ecc_permit.read_bytes(), registered_sha256=reg.canary.frame_diag_sha256, rules=rules)
        else:
            rules = load_rules(blob_at(ctx.repo, rev, ECC_RULES_REL))
            frame = frame_check_by_model_free(a.frame_admission.read_bytes(), a.ecc_waiver.read_bytes(),
                                              registered_sha256=reg.canary.frame_diag_sha256, rules=rules,
                                              decision_log=blob_at(ctx.repo, rev, DECISION_LOG_REL).decode("utf-8"))
    except EarlyCheckRejected as exc:
        raise EntryRejected("FRAME_PERMIT", f"좌표 진단 칸의 파일이 등록 · 규칙 · 의사결정로그와 맞지 않는다 — {exc}") from None
    ctx.frame = {**frame, "meaning": meaning}
    ctx.known_answer = {"status": "reproduced", "sha256": ka_sha, "fixture": KA.FIXTURE_REL,
                        "fixture_sha256": hashlib.sha256(ka_raw).hexdigest()}
    ctx.mode, ctx.eval_doc, ctx.sealed, ctx.out_rel = MODE_SAMPLE, doc, sealed, rel
    ctx.eval_doc_sha256, ctx.eval_bound = bound.sha256, bound
    ctx.side_raw, ctx.baseline_raw = side_raw, baseline_raw


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
    if ctx.mode == MODE_SAMPLE:
        inputs["side"] = hashlib.sha256(ctx.side_raw).hexdigest()
        inputs["baseline"] = hashlib.sha256(ctx.baseline_raw).hexdigest()
        inputs["eval_registration"] = ctx.eval_doc_sha256
    payload = {"registration_mode": a.registration, "run_kind": a.purpose, "purpose": a.purpose,
               "generation_sha256": ctx.reg.generation_sha256(),
               "scoring_sha256": "미완" if ctx.scoring_gaps else ctx.receipt.scoring_sha256,
               "bundles": bundles, "generation_list_file_sha256": inputs["generation_list"], "inputs": inputs,
               "scorer_code": scorer_code_digest(ctx.checkout, include_files=False)["combined"], "device": a.device}
    ctx.call_scorer = payload["scorer_code"]
    return append_line(canonical_ledger(ctx.checkout), "call", payload, at=now)


def _read_lists(ctx: Ctx):
    a, g, p = ctx.args, ctx.reg.generation, ctx.reg.population
    gen = validate_eval_list(a.generation_list.read_bytes(), where="generation_list")
    pop_raw = a.scoring_list.read_bytes()
    pop = validate_eval_list(pop_raw, where="scoring_list")
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
    if a.purpose == "main":
        if getattr(ctx, "mode", None) == MODE_SAMPLE:
            try:
                side = read_side(ctx.side_raw, pop_raw)
            except SampleSideError as exc:
                raise EntryRejected("SIDE_FILE", f"곁 파일을 읽지 못했다: {exc}") from None
            check_main_sample(gen, pop, meta, sealed=ctx.sealed, side=side)
            ctx.side = side
        else:
            check_main_full_eval(gen, pop, meta)
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


# ── 6 · 7 엄격 · 표본 모드 ─────────────────────────────────────────────────────

def _gold_eval(ctx: Ctx, ids) -> tuple[dict, dict, int]:
    """엄격 정답 — 평가 정답 뷰(`EVAL_VIEW`)로만 읽는다. `(상자, 이미지 수준 코드, 상자 없는 주석 행 수)`. 정상 이미지는 빈 목록.

    상자 네 칸 가운데 하나라도 비면(판독기는 정수 열의 빈 칸을 `NA` 로 준다) 그 행은 상자로 세지 않고 이미지 수준 코드에만 남긴다 —
    공통 채점기의 정답 판독(`evaluation.eval_set.read_gold`)과 같은 처리다. 센 수는 산출에 싣는다.
    """
    ids = list(ids)
    df = EVAL_VIEW(ctx.args.snapshot, ids, receipt=ctx.args.receipt)
    boxes: dict[str, list] = {i: [] for i in ids}
    codes: dict[str, set] = {i: set() for i in ids}
    no_box = 0
    for row in df.itertuples(index=False):
        codes[row.image_id].add(str(row.iso_code))
        vals = (row.bbox_x1_px, row.bbox_y1_px, row.bbox_x2_px, row.bbox_y2_px)
        if any(pd.isna(v) for v in vals):
            no_box += 1
            continue
        boxes[row.image_id].append((str(row.iso_code), [float(v) for v in vals]))
    return boxes, codes, no_box


def _json_default(o):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"JSON 으로 쓸 수 없는 값: {type(o).__name__}")


def _prior_calls(ctx: Ctx, call_sha: str) -> list[str] | dict:
    """같은 영수증 · 같은 생성 목록으로 앞서 남은 `call` 줄 — 시도가 몇 번이었는지 산출이 스스로 말하게(07번 §12-5).
    원장은 잠금 안에서 읽는다. 읽지 못하면 수를 지어내지 않고 상태로 적는다."""
    want = (_sha(ctx.args.receipt), _sha(ctx.args.generation_list))
    path = canonical_ledger(ctx.checkout)
    try:
        with ledger_lock(path):
            rows = read_ledger(path)
    except (LedgerBusy, LedgerChainBroken) as exc:
        return {"status": "unreadable", "reason": type(exc).__name__}
    out = []
    for sha, row in rows:
        inp = row.get("inputs") or {}
        if row.get("kind") == "call" and sha != call_sha and (inp.get("receipt"), inp.get("generation_list")) == want:
            out.append(sha)
    return out


def _gold_reader_identity(ctx: Ctx) -> dict:
    """정답을 실제로 읽은 판독기 모듈의 바이트 — 본체 체크아웃의 같은 이름 파일이 아니라 불러온 파일을 잰다(검토 F8)."""
    src = Path(_MV.__file__)
    here = ctx.checkout / GOLD_READER_REL
    return {"name": EVAL_VIEW_READER, "path": GOLD_READER_REL,
            "file_sha256_lf": hashlib.sha256(src.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "loaded_from_checkout": here.is_file() and src.resolve() == here.resolve(),
            "note": "채점 지문 밖의 정답 판독기 — 지문 목록에 더할지는 따로 묻는다(35번 §1)"}


def _device_label(verified) -> dict:
    """산출의 장비 표지 — 정본 꼴이 아닌 장비 문자열(`bundle_v14.DEVICE_LABEL_NONCANONICAL`)이면 그 문자열과 묶음을 싣는다(2026-10-18)."""
    rows = {vb.files.stem: vb.notes["device_label"]["value"] for vb in verified if vb.notes.get("device_label")}
    if not rows:
        return {"status": DEVICE_LABEL_CANONICAL}
    return {"status": DEVICE_LABEL_NONCANONICAL, "values": sorted(set(rows.values())), "bundles": sorted(rows),
            "scope": "출처 기록(도장 · 시작 줄 · 곁 파일)의 장비 문자열이 '{이름}:{인덱스}' 꼴이 아니다 — 한 문자열임만 확인했고 몇 번 장치인지는 모른다"}


def _aux_scoring(ctx: Ctx, verified, scores, ids, gold_codes, b_pred, known, classes):
    """2 차 보조 지표(서술 · 조항 축)의 입력 — 판 6 등록에 블록이 있고 평가 등록 `auxiliary` 에 그 이름이 있을 때만(03 2판 §5a). 아니면 None.

    검사기 · 닫힌 목록 · 추출기 · 허용치 표 · 로더 · 정답 서술 생성기 · 잔여 한계 · 탐색적 종류의 신원은 **지금 불러 쓰는 모듈의
    파일**로 등록과 맞대고(`narrative_registration.identity_problems`), 선택 검사 인자(`--aux-train-extractor` · `--aux-prompt` ·
    `--aux-vectors`)를 주면 그 파일도 맞댄다. 다르면 보조 지표를 내지 않고 사유를 싣는다(`AuxRefused` — M1 산출은 그대로). 무엇을
    맞댔는지는 `identity_checks` 에 적는다. 묶음 · 정답 · 대조선은 이미 검증 · 판독한 것을 그대로 쓴다. 1.4 레코드는 채점 층과 같은
    함수(`records_from_verified`)로 같은 줄에서 다시 낸다. 판 5 등록이면 보조 모듈을 불러오기 **전에** None 을 낸다(203 §2).
    """
    # 판 5 길도 쓰는 모듈 — 새 모듈이 아니다
    from evaluation.prereg_unified import CLAUSE_AXIS_METRICS, REGISTRATION_VERSION_V6

    reg = ctx.reg
    if reg.version != REGISTRATION_VERSION_V6:
        return None
    # 판 6 — 여기서부터 보조 모듈을 부른다
    from corpus.rules import limits_loader
    from evaluation import narrative_checker as NCK
    from evaluation import narrative_registration as NR
    from evaluation import strict_narrative as SN
    from evaluation.clause_axis import entries_from_limits
    from evaluation.sample_bootstrap import design_weights
    names = set(ctx.eval_doc.get("auxiliary") or [])
    wanted = set()
    if reg.narrative is not None:
        wanted |= names & set(SN.NARRATIVE_NAMES)
    if reg.clause_axis is not None:
        wanted |= names & set(CLAUSE_AXIS_METRICS)
    if not wanted:
        return None
    code_root = Path(NCK.__file__).resolve().parents[1]
    opt = {"train_extractor_path": getattr(ctx.args, "aux_train_extractor", None),
           "prompt_path": getattr(ctx.args, "aux_prompt", None), "vectors_path": getattr(ctx.args, "aux_vectors", None)}
    checks = {"code_tree": "checked",
              **{k.removesuffix("_path"): "checked" if v is not None else "not_given" for k, v in opt.items()}}
    profile = _aux_profile(ctx)                 # 신판이면 regime2_v4 프로필, 구판이면 None(옛 동작), 섞였으면 거부 사유 문자열
    if isinstance(profile, str):
        return SN.AuxRefused(wanted=frozenset(wanted), problems=(profile,), identity_checks=checks)
    pkw = {} if profile is None else {"profile": profile}
    problems = NR.identity_problems(reg, repo=code_root, **opt, **pkw)
    if problems:
        return SN.AuxRefused(wanted=frozenset(wanted), problems=tuple(problems), identity_checks=checks)
    if profile is None:
        checker, limits_rel = NCK.load_checker(code_root), NCK.LIMITS_REL
    else:                                       # 실제 검사기 · 표 · 신원이 같은 프로필 — 닫힌 사상 · 핀 · 실행 모듈 대조
        try:
            checker = NR.load_profile_checker(profile, repo=code_root)
        except NR.ProfileRefused as exc:
            return SN.AuxRefused(wanted=frozenset(wanted), problems=(f"{exc.code}: " + " · ".join(exc.problems),),
                                 identity_checks=checks)
        limits_rel = profile.checker_pin.limits_rel
    material_of = SN.material_of_side(ctx.side, ids, SM.MATERIALS)
    w0 = design_weights(ids, ctx.side)
    models = {}
    for vb in verified:
        key = (vb.tag, vb.seed_index)
        if vb.mode != MODE_MODEL or key not in scores:
            continue
        rep = records_from_verified(vb, coupling_rule=reg.metric.coupling_rule, known_iso_codes=known,
                                    scoring_iso_codes=list(classes))
        models[key] = SN.model_inputs(vb.tag, vb.seed_index, vb.lines, rep.records, image_ids=ids,
                                      material_of=material_of, gold_codes=gold_codes, weights=w0)
    entries = entries_from_limits(limits_loader.load_limits(code_root / limits_rel, pilot=True))
    act = NR.actual_identity(code_root, **pkw)
    identity = {k: act[k] for k in ("checker_id", "checker_sha256", "lists_sha256", "extractor_id", "extractor_sha256")}
    return SN.AuxScoring(models=models, checker=checker, checker_identity=identity, entries=entries,
                         index_clause_ids=frozenset(checker.t.clause_ids), material_of=material_of,
                         gold_codes={i: frozenset(gold_codes.get(i, ())) for i in ids},
                         baseline_codes={i: sorted({c for c, _ in b_pred.get(i, ())}) for i in ids},
                         wanted=frozenset(wanted), image_weights={i: float(w) for i, w in zip(ids, w0, strict=True)},
                         exploratory_kinds=tuple(reg.narrative.exploratory_kinds or ()) if reg.narrative is not None else (),
                         identity_checks=checks)


def _aux_profile(ctx: Ctx):
    """판 6 등록의 프로필 — 영수증 커밋의 학습 등록으로 고른다(평가 쪽 `profile_for`, 판 6 연결 2 판 §3-2). 신판(`regime2_v4`)이면 그
    프로필, 구판(역사적 판 6 · 학습 등록 없음)이면 None(옛 동작 — 산출 바이트 보존), 조합이 섞였으면 `profile_mismatch: …` 문자열."""
    from evaluation import narrative_registration as NR
    from vlm.registration import REG_PATH, RegistrationRejected, registration_from_bytes

    try:
        entry = registration_from_bytes(blob_at(ctx.repo, anchor_rev(ctx.receipt), REG_PATH.as_posix()))
    except (EntryRejected, RegistrationRejected):
        entry = None
    try:
        profile = NR.profile_for(ctx.reg, train_entry=entry)
    except NR.ProfileRefused as exc:
        pin, n = NR.NARRATIVE_PROFILES[NR.PROFILE_REGIME2].checker_pin, ctx.reg.narrative
        if n is not None and (n.checker_id == pin.checker_id or n.limits_sha256 == pin.limits_sha256
                              or n.lists_sha256 == pin.lists_sha256):
            return f"{exc.code}: " + " · ".join(exc.problems)          # 신판 단서가 섞였다 — 거부 사유로
        return None                                                     # 신판 단서 0 — 구판 신원 대조가 칸 이름으로 사유를 낸다
    return profile if profile.profile_id == NR.PROFILE_REGIME2 else None


def v6_completion(*, profile_id: str, required_cells, verified_cells, aux, report: dict) -> dict:
    """신판 판 6 산출의 완료 표지(판 6 연결 2 판 §5 · D4) — 필수 칸 전량 검증과 보조 다섯 이름의 **실제 계산**을 모두 요구한다.

    `wanted` 만 채운 것은 계산이 아니다 — 계산된 이름 · 결과의 모델 포함 범위 · 대응 비교의 산출 여부를 report 블록과 맞댄다. 수학적으로
    분모가 없는 값의 기존 '정의 불가' 처리는 그대로이고 0 으로 메우지 않는다. `aux is None` → aux_not_computed, 신원 거부 →
    aux_identity_refused, 일부 이름 · 모델 · 비교가 빠짐 → aux_incomplete, 칸 거부 → cells_missing. `complete` 는 보조 계산의 완료
    표지이지 성능 성공 · 가설 채택이 아니다. 별도 문턱 · 채택 기준을 두지 않는다."""
    from evaluation.strict_main import cell_name

    reasons: list[str] = []
    missing = sorted(set(required_cells) - set(verified_cells))
    if missing:
        reasons.append(f"cells_missing:{len(missing)}")
    if aux is None:
        reasons.append("aux_not_computed")
    elif getattr(aux, "problems", None):
        reasons.append("aux_identity_refused")
    else:
        names = set(aux.computed_names())
        cells = {cell_name(*k) for k in verified_cells}
        incomplete = not set(AUXILIARY_V6) <= names
        for key in ("narrative", "clause_axis"):
            block = report.get(key)
            if not isinstance(block, dict) or set(block.get("models") or {}) != cells:
                incomplete = True
                continue
            paired = block.get("paired")
            if not isinstance(paired, dict) or paired.get("status") == "산출 불가":
                incomplete = True
        if incomplete:
            reasons.append("aux_incomplete")
    return {"profile_id": profile_id, "complete": not reasons, "reasons": reasons, "required_names": list(AUXILIARY_V6),
            "note": "보조 계산의 완료 표지 — 성능 성공 · 연구 가설 채택이 아니다. 정의 불가 값은 그대로 두고 0 으로 메우지 않는다"}


def _strict_sample(ctx: Ctx, call_sha: str, gens: list[Path], verified, rejected: dict, canaries: dict, gen, pop,
                   meta) -> int:
    """표본 모드의 단계 6(등록 지표 · 회복률 · 참여자 · 규칙 ⑤) · 7(배타 생성). 거부된 칸이 있으면 산출은 쓰고 종료 2 다(3판 1-4)."""
    reg = ctx.reg
    pid, rel = reg.population.population_id, ctx.out_rel
    classes = [str(c) for c in reg.metric.class_codes]
    known = list(_known_codes())
    try:                                 # 대조선 예측은 정답보다 먼저 읽는다 — 꼴이 틀린 등록 입력으로 평가 정답을 열지 않는다
        b_pred, b_info = SM.read_baseline_predictions(
            ctx.baseline_raw, population=pop.ids, allowed_ids={i for i, s in meta.split_of.items() if s == "eval"},
            known_codes=set(known), classes=classes)
    except SM.BaselineFileError as exc:
        print(f"채점 단계에서 멈췄다(산출 없음): 대조선 예측 — {exc}", file=sys.stderr)
        return EXIT_ENTRY
    try:
        _out_file(ctx, rel)
        gold, gold_codes, n_no_box = _gold_eval(ctx, pop.ids)
    except (EntryRejected, ManifestError, OSError) as exc:       # 판독기의 거부 · 스냅샷 검증 · 읽기 실패
        print(f"채점 단계에서 멈췄다(산출 없음): {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ENTRY
    seeds = {int(k) for k in reg.seeds}
    rej = {k: list(v) for k, v in rejected.items()}
    scores = {}
    for vb in verified:
        if vb.mode != MODE_MODEL:
            continue
        name = SM.cell_name(vb.tag, vb.seed_index)
        if vb.seed_index not in seeds:
            rej.setdefault(name, []).append("unregistered_seed")
            continue
        try:
            scores[(vb.tag, vb.seed_index)] = score_registered(
                vb, gold, population=pop.ids, classes=classes, coupling_rule=reg.metric.coupling_rule,
                known_iso_codes=known, sensitivity_variant=reg.metric.sensitivity_variant_id, gold_codes=gold_codes)
        except (UpstreamContractError, ValueError) as exc:
            rej.setdefault(name, []).append(f"score_failed:{type(exc).__name__}")
    for tag in SM.TAGS:                  # 등록한 칸인데 묶음이 없으면 그것도 거부된 칸이다(검토 F1) — 분모를 조용히 줄이지 않는다
        for s in sorted(seeds):
            name = SM.cell_name(tag, s)
            if (tag, s) not in scores and name not in rej:
                rej[name] = ["missing_bundle"]
    env = sorted(k for k in rej if k == f"env:{MODE_MODEL}")      # 에코 묶음의 환경은 모델 비교를 무효로 하지 않는다
    try:
        aux = _aux_scoring(ctx, verified, scores, list(pop.ids), gold_codes, b_pred, known, classes)
    except (ValueError, OSError) as exc:  # 보조 입력의 계약 위반 — 산출을 쓰지 않는다(신원 어긋남은 아래 거부 — 보조 지표만 뺀다)
        print(f"채점 단계에서 멈췄다(산출 없음): 보조 지표 — {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ENTRY
    if getattr(aux, "problems", None):   # strict_narrative.AuxRefused — 보조 지표만 빼고 M1 은 그대로 낸다
        print("보조 지표를 내지 않는다 — 판 6 블록의 신원이 맞지 않는다(산출의 aux_identity): " + " · ".join(aux.problems),
              file=sys.stderr)
    try:
        gold_sub = {i: [(c, b) for c, b in gold[i] if c in set(classes)] for i in pop.ids}
        report = SM.compute(scores=scores, baseline=SM.baseline_table(b_pred, gold_sub, classes), side=ctx.side,
                            group_of=meta.group_of, reg=reg, eval_doc=ctx.eval_doc, rejected=rej,
                            invalid_comparison=env, aux_scoring=aux)
    except (ValueError, AssertionError) as exc:  # 대조선 파일 · 등록 규약 · 재표집 항등 · 추첨 회계 — 산출을 쓰지 않는다
        print(f"채점 단계에서 멈췄다(산출 없음): {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ENTRY
    r5 = report.get("rule5") or {}
    if r5.get("status") == "computed" and r5.get("verdict_authority") != RULE5_REGISTERED:
        print("규칙 ⑤ 가 등록 판정이 아니다 — 산출을 쓰지 않는다(3판 1-1-라)", file=sys.stderr)
        return EXIT_ENTRY
    completion = None
    if getattr(ctx.eval_bound, "profile_id", None) == "regime2_v4":     # 신판 판 6 산출에만 — 판 5 · 역사적 판 6 의 바이트는 그대로
        completion = v6_completion(profile_id=ctx.eval_bound.profile_id,
                                   required_cells=[(t, s) for t in SM.TAGS for s in sorted(seeds)],
                                   verified_cells=list(scores), aux=aux, report=report)
    status = "pass" if not rej and (completion is None or completion["complete"]) else "partial"
    scorer = scorer_code_digest(ctx.checkout)
    inputs = {p.name: _sha(p) for p in gens}
    for name in ("generation_list", "scoring_list", "echo_list", "receipt", "ecc_permit", "frame_admission", "ecc_waiver",
                 "ecc_substitute"):
        if getattr(ctx.args, name, None) is not None:
            inputs[name] = _sha(getattr(ctx.args, name))
    body = {"run_kind": "main", "registration_mode": "strict", "purpose": "main", "mode": MODE_SAMPLE, "status": status,
            "population_id": pid, "call_sha256": call_sha, "prior_calls": _prior_calls(ctx, call_sha),
            "fingerprints": {"generation_sha256": reg.generation_sha256(), "scoring_sha256": ctx.receipt.scoring_sha256,
                             "registration_file_sha256": ctx.receipt.registration_file_sha256,
                             "eval_registration_sha256": ctx.eval_doc_sha256, "eval_registration_rev": anchor_rev(ctx.receipt),
                             "side_sha256": hashlib.sha256(ctx.side_raw).hexdigest(),
                             "baseline_prediction_sha256": hashlib.sha256(ctx.baseline_raw).hexdigest(),
                             "generation_list_file_sha256": gen.file_sha256, "scoring_list_file_sha256": pop.file_sha256,
                             "sealed": ctx.sealed},
            "gold_reader": _gold_reader_identity(ctx),
            "bundles": {"verified": [vb.files.stem for vb in verified], "rejected": rej}, "inputs": inputs,
            "device_label": _device_label(verified),
            "canaries": {**canaries, "axis_ratio": {"status": "not_run", "reason": STRICT_PENDING[1]},
                         "frame": ctx.frame, "known_answer": ctx.known_answer},
            "gold": {"n_annotation_rows_without_box": n_no_box,
                     "rule": "상자 네 칸 가운데 하나라도 빈 행은 상자로 세지 않고 이미지 수준 코드에만 남긴다(eval_set.read_gold 와 같다)"},
            "baseline_file": b_info, "scorer_code": scorer,
            "code_drift": {"call": ctx.call_scorer, "write": scorer["combined"], "drifted": ctx.call_scorer != scorer["combined"]},
            "notes": ctx.notes, "report": report}
    if completion is not None:
        body["v6_completion"] = completion
        body["fingerprints"]["eval_registration_path"] = ctx.eval_bound.relative_path
    text = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False, default=_json_default) + "\n"
    try:
        path = _out_file(ctx, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        _out_file(ctx, rel)
        write_new_text(path, text)
    except FileExistsError:
        print(f"산출을 쓰지 않았다: {rel} 가 이미 있다 — 덮지 않는다", file=sys.stderr)
        return EXIT_ENTRY
    except (EntryRejected, OSError) as exc:
        print(f"산출을 쓰지 않았다: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ENTRY
    return EXIT_OK if status == "pass" else EXIT_REJECTED


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
        echo_stems = {vb.files.stem for vb in verified if vb.mode == MODE_ECHO}
        bad_literal = {x.get("bundle") for x in canaries["literal"] if x.get("status") != "pass"}
        echo_ok = bool(canaries["echo"]) and all(e.get("status") == "pass" for e in canaries["echo"])
        if not echo_ok or bad_literal & echo_stems:
            print("엄격 경로의 에코 카나리아가 통과가 아니다 — 진입 전 관문", file=sys.stderr)
            return EXIT_ENTRY
        if ctx.mode != MODE_SAMPLE:
            print("엄격 경로의 채점 단계는 표본 모드만 있다: " + STRICT_PENDING[0], file=sys.stderr)
            return EXIT_ENTRY
        for stem in sorted(bad_literal):          # 모델 묶음의 문자 일치가 어긋나면 그 칸을 거부한다 — 실행을 멈추지 않는다
            rejected.setdefault(stem, []).append("literal_mismatch")
        kept = [vb for vb in verified if vb.files.stem not in bad_literal]
        return _strict_sample(ctx, call_sha, gens, kept, rejected, canaries, gen, pop, meta)
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
