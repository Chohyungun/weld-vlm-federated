"""export 의 사전 점검 — 리허설 2판 §2-4 의 1 을 **평가 쪽 실제 함수**로 한다. 이미지를 열기 전 · 모델을 올리기 전에 끝난다.

등록 객체는 평가 쪽 로더로 읽는다 — 이 모듈은 등록을 만들지 않는다. 순서는 평가 쪽 진입점(진입점 미니스펙 3판 1-1-가 · 1-1-나)과 같다.

1. 영수증(`read_receipt`) — 종류가 목적과 같다(`check_receipt_kind`). 이음새(저장소)를 받았으면 커밋된 영수증을 거부한다(`guard_seam`).
2. 등록(`load_registration_for`) — 본실험 · 진단은 영수증 커밋이 본줄기의 조상이고 그 커밋의 등록 파일 바이트로, 리허설은 로컬 파일로.
   어느 쪽이든 파일 바이트의 해시가 영수증과 같아야 한다. 리허설의 로컬 등록 파일은 그 루트 아래여야 한다.
3. 동결본의 닻(`load_anchor` · `check_anchor`) · 프로세서 인자(`check_processor`) — 영수증 커밋(리허설은 `refs/heads/main`)의 설정 바이트로.
4. 생성 쪽 완결과 지문(`require_receipt(side="generation")`).
5. 스냅샷 검증(`check_snapshot`) — 본실험은 잠정(`provisional`) 동결본을 받지 않는다.
6. 계획 파일 — 등록에 `plan_sha256` 이 있으면 그 바이트여야 하고 `kind` 가 목적이다. 본실험은 계획을 받지 않는다.
7. 목록(`validate_eval_list`) — 파일 **바이트**로 읽고, 모델 묶음은 생성 목록 칸 · 에코 묶음은 에코 목록 칸의 두 해시와 같다.
   그 뒤 동결 매니페스트의 분할(`read_manifest_meta` · `check_list_split`) — 모델은 목적의 분할, 에코는 언제나 val.
   **본실험의 모델 묶음은 등록의 모집단 모드를 따른다** — 가중이 표본(`stratified_g_over_n`)이 아니면 전량이다: 목록의 집합이
   검증한 동결 매니페스트의 `split == eval` 전체와 같다(평가 쪽 07번 §40 · Codex 12번 "예측 생성 · 본채점 전에 실패"). 표본이면
   영수증 커밋의 평가 등록 설정이 봉인한 주 표본과 같다 — 채점 쪽 `check_main_sample` 과 같은 식(봉인 목록 두 해시 · 장수 ·
   뽑힌 묶음의 평가 이미지 전부 · 묶음 수 · 곁 파일의 층과 `n_h`, 곁 파일 바이트는 봉인한 `side_sha256`)이다(1016k).
   파일 해시는 앞의 등록 대조가 묶는다. 리허설 · 진단 · 에코는 부분집합 그대로다. 셋 다 `check_generation_list` 가 한다.
8. 노출 기록(리허설 · 진단) — 이미지를 열기 전에 같은 실행 id · 목록 종류 · 목록 해시의 `planned` 줄이 있어야 한다(§5-4).
9. 실측값 — 설정(닻과 같은 리비전)의 모델 · 템플릿 · 좌표 규약 · 프롬프트 파일과 프로세서에서 잰 값(이음새 `measure_env`)으로
   곁 파일에 실을 값을 모으고 `check_actuals` 로 등록과 맞댄다. 리허설은 구현 식별자의 어긋남을 "대역 실행" 으로 적고 지나간다.
10. 모델 모드 — 어댑터 meta · 어댑터 파일 sha · 닫힌 원장(`read_ledger_view`) · 원장 끝 스텝 = meta 의 스텝 · 학습 설정 · 프로세서 ·
    좌표 해시 · 학습 행 digest(본실험은 페어의 참여자 행 전체에서 영수증 커밋의 학습 등록이 정한 예산 비율 몫을 등록 시드로 다시
    고른다 — `check_main_train_rows`, 비율 1.0 이면 전체 행 그대로) · meta 의 목적 · 태그 · 시드 번째.
    조기 점검(`check_step`)은 별도 분기 — step 파일 넷 · 점검 원장의 그 step 줄, 그리고 시드는 번호가 아니라 등록 시드표의
    **실제 시드**를 신원의 `seed` 와 맞대고 `round_idx == client_idx == 0` 을 함께 본다(§137 C-01).
    학습 설정 원문의 계약에서 레시피를 되찾고(모르는 계약은 거부), 본실험은 등록의 판이 그 레시피의 판이어야 한다 — 1 차는 판 5,
    새 체제는 판 6(`registration_version`, 지시 1023c).
11. 작업 트리 청결 — 추적 코드(`*.py`)와 `vlm/prompts/` 의 작업 트리 바이트가 커밋 blob 과 같고 추적 밖 `*.py` 가 없다.
12. 지금 시각 ≥ 영수증 시각.

거부는 `ExportRefused` 다. 평가 쪽 거부의 사유 코드는 `entry_<코드>` 로 옮긴다.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from vlm.export_writer import ExportRefused

__all__ = ["FED_CLIENTS", "GenerationConfig", "PreflightResult", "check_full_eval_set", "check_generation_list",
           "check_main_train_rows", "check_sample_eval_set", "code_commit", "export_preflight"]

FED_CLIENTS = 3
_CODE_DIRS = ("vlm", "fl", "scripts", "evaluation", "detection", "data", "corpus", "rag", "tracking")


@dataclass(frozen=True)
class GenerationConfig:
    """생성기를 여는 값 — 등록과 설정에서 온다. 실제 생성기와 대역이 같은 꼴로 받는다."""

    mode: str
    tag: str
    model_id: str
    model_revision: str | None
    processor_kwargs: Mapping[str, Any]
    prompt_text: str
    chat_template_kwargs: Mapping[str, Any]
    max_new_tokens: int
    decoding: Mapping[str, Any]
    batch_size: int
    padding_side: str
    coord_space: str
    pairs_path: str | None = None
    adapter_path: Path | None = None
    """모델 모드의 `adapter_last.npz`. 에코는 None."""
    purpose: str = "main"
    """목적 — 생성기가 안쪽 구현(모델 적재기)을 부르기 전에 이것과 맞댄다. 비우면 가장 엄격한 `main` 으로 본다."""
    target_recipe: str = "json_only"
    """학습 레시피 — 등록된 학습 설정 원문의 계약 해시에서 되찾는다. 에코 글(`vlm.export_echo`)이 이 레시피로 타깃을 만든다."""


@dataclass(frozen=True)
class PreflightResult:
    registration: Any
    receipt: Any
    generation_sha256: str
    eval_list: Any
    seed_value: int
    gen_cfg: GenerationConfig
    stamp_fields: dict[str, Any]
    sidecar_values: dict[str, Any]
    stand_in: bool = False
    """리허설에서 구현 식별자가 등록과 다르다(대역 실행). 리허설 판정기는 통과를 내지 않는다."""
    notes: list[str] = field(default_factory=list)


def _refuse_from(exc: Exception) -> ExportRefused:
    code = getattr(exc, "code", None) or type(exc).__name__
    return ExportRefused(f"entry_{code}", str(exc))


def _under(p: Path, root: Path) -> bool:
    q, r = Path(p).resolve(), Path(root).resolve()
    return q == r or q.is_relative_to(r)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)


def code_commit(repo: Path) -> tuple[str, list[str]]:
    """(HEAD 커밋, 청결하지 않은 까닭). 추적 코드 · 프롬프트의 작업 트리 바이트를 커밋 blob 과 맞대고 추적 밖 `*.py` 를 센다.

    `git status` 만으로는 줄끝 정규화에 가린 차이를 못 잡으므로 바이트로 잰 blob id 를 맞댄다(2판 §2-4 의 1 · W-17).
    """
    head = _git(repo, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise ExportRefused("code_commit", f"코드 저장소의 HEAD 를 읽지 못했다: {repo}")
    commit = head.stdout.decode().strip()
    out: list[str] = []
    tree = _git(repo, "ls-tree", "-r", "-z", "HEAD")
    for entry in tree.stdout.decode("utf-8", "replace").split("\0"):
        if not entry:
            continue
        meta, _, path = entry.partition("\t")
        parts = meta.split()
        if len(parts) != 3 or parts[1] != "blob":
            continue
        if not (path.endswith(".py") or path.startswith("vlm/prompts/")):
            continue
        f = Path(repo) / path
        if not f.is_file():
            out.append(f"없다: {path}")
            continue
        data = f.read_bytes()
        if hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != parts[2]:
            out.append(f"커밋과 다르다: {path}")
    others = _git(repo, "ls-files", "--others", "--exclude-standard", "-z", "--", *_CODE_DIRS)
    for path in others.stdout.decode("utf-8", "replace").split("\0"):
        if path.endswith(".py"):
            out.append(f"추적 밖: {path}")
    return commit, out


def _uni_config_at(raw: bytes):
    import yaml

    from vlm.uni_config import UniConfigIncomplete, uni_config_from

    try:
        return uni_config_from(yaml.safe_load(raw.decode("utf-8")) or {})
    except UniConfigIncomplete as exc:
        raise ExportRefused("config_form", str(exc)) from None


def _plan(plan_path: Path | None, g, purpose: str) -> tuple[str | None, dict]:
    import yaml

    if purpose == "main":
        if plan_path is not None:
            raise ExportRefused("main_with_plan", "본실험 export 는 계획 파일을 받지 않는다")
        return None, {}
    if g.plan_sha256 is None:
        return (None, {}) if plan_path is None else (hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(), {})
    if plan_path is None:
        raise ExportRefused("plan_missing", "등록에 plan_sha256 이 있는데 계획 파일이 주어지지 않았다")
    raw = Path(plan_path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != g.plan_sha256:
        raise ExportRefused("plan_hash", "계획 파일의 바이트가 등록의 plan_sha256 과 다르다")
    plan = yaml.safe_load(raw.decode("utf-8")) or {}
    if plan.get("kind") != purpose:
        raise ExportRefused("plan_kind", f"계획의 kind {plan.get('kind')!r} 가 목적 {purpose!r} 가 아니다")
    return sha, plan


def _training(adapter_dir: Path, *, tag: str, seed_index: int, purpose: str, reg, measured: Mapping,
              coord_hash: str, cfg, run_root: Path | None, seed_value: int | None = None,
              budget_fraction: float = 1.0) -> tuple[dict, dict, dict]:
    """모델 모드의 학습 쪽 출처 — (meta, 도장 값, 곁 파일 값). `seed_value` 는 등록 시드표의 값, `budget_fraction` 은 영수증
    커밋의 학습 등록이 정한 공통 학습 예산 비율이다(본실험의 학습 행 대조 — `check_main_train_rows`)."""
    from evaluation.ledger_v14 import LedgerRejected, read_ledger_view
    from evaluation.schema_v14 import tag_parts
    from vlm.train_cell import ADAPTER_FILE, LEDGER_FILE, META_FILE, _record_path, _sha256_file

    d = Path(adapter_dir)
    meta_p, npz = d / META_FILE, d / ADAPTER_FILE
    led = d / (LEDGER_FILE if tag != "uni_fed" else "atomic_log.csv")
    for p in (meta_p, npz, led):
        if not p.is_file():
            raise ExportRefused("adapter_missing", f"모델 모드의 학습 산출물이 없다: {p.name}")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    ident = meta.get("identity") or {}
    for key, want in (("purpose", purpose), ("tag", tag), ("seed_index", seed_index)):
        if ident.get(key) != want:
            raise ExportRefused("adapter_identity", f"어댑터 meta 의 {key} {ident.get(key)!r} ≠ {want!r}")
    if _sha256_file(npz) != meta.get("adapter_sha256"):
        raise ExportRefused("adapter_sha", "어댑터 파일의 sha 가 meta 의 adapter_sha256 과 다르다")
    g = reg.generation
    try:
        view = read_ledger_view(led, n_rounds=g.budget_r, n_clients=FED_CLIENTS) if tag == "uni_fed" \
            else read_ledger_view(led)
    except LedgerRejected as exc:
        raise ExportRefused(f"ledger_{exc.code}", str(exc)) from None
    cell, client = tag_parts(tag)
    if (view.cell, view.client) != (cell, client):
        raise ExportRefused("ledger_cell", f"원장의 칸 · 참여자 {(view.cell, view.client)} ≠ {(cell, client)}")
    if view.final_step != meta.get("adapter_step"):
        raise ExportRefused("ledger_step", "원장 끝 행의 스텝이 meta 의 adapter_step 과 다르다 — 중간 체크포인트다")
    if view.run_id != meta.get("train_run_id"):
        raise ExportRefused("ledger_run_id", "원장의 실행 신원이 meta 의 train_run_id 와 다르다")
    if meta.get("processor_config_sha256") != measured.get("processor_config_sha256"):
        raise ExportRefused("adapter_processor", "학습 쪽 프로세서 설정이 export 의 프로세서와 다르다(§3 ② 다)")
    if meta.get("coord_cfg_hash") != coord_hash:
        raise ExportRefused("adapter_coord", "학습 쪽 좌표 해시가 export 의 좌표 규약과 다르다")
    # 학습 행 — 등록에 값이 있으면 그것과, 본실험은 페어의 그 참여자 행 전체에서 학습 등록의 예산 몫을 다시 고른 값과
    # 맞댄다(§2-7 · §5-2 · 1016k — 비율 1.0 이면 전체 행 그대로).
    rows_got = meta.get("train_rows_digest")
    if g.train_rows_digest is not None and rows_got != g.train_rows_digest:
        raise ExportRefused("train_rows", "어댑터 meta 의 train_rows_digest 가 등록과 다르다")
    if purpose == "main":
        if cfg.pairs_path is None:
            raise ExportRefused("train_rows", "본실험 학습 행을 다시 낼 페어 경로가 설정에 없다")
        check_main_train_rows(rows_got, tag=tag, client=client, ident=ident, pairs_path=cfg.pairs_path,
                              seed_value=seed_value, fraction=budget_fraction)
    if purpose != "main" and run_root is not None and not _under(d, run_root):
        raise ExportRefused("run_root", "어댑터가 이 루트 아래가 아니다")
    stamp = {"train_run_id": view.run_id, "train_ledger_sha256": view.file_sha256,
             "scored_adapter_sha256": meta["adapter_sha256"]}
    side = {"train_ledger_path": _record_path(led), "scored_adapter_step": int(meta["adapter_step"]),
            "train_config": meta.get("train_config")}
    return meta, stamp, side


def check_main_train_rows(rows_got: Any, *, tag: str, client: str | None, ident: Mapping[str, Any], pairs_path: Any,
                          seed_value: int | None, fraction: float) -> None:
    """본실험 어댑터의 학습 행 — 페어의 참여자 행 **전체**에서 학습 등록의 예산 비율 몫을 등록 시드로 다시 고른 행
    (`vlm.budget.apply_budget` — 학습 칸 · 연합 클라이언트와 같은 함수)의 digest 와 맞댄다(1016k, 03 §2 (나)). 연합은 참여자별 사전이다.

    비율 1.0 이면 몫이 전체 행이라 이 판 전과 같은 값이다. 비율이 1 이 아니면(등록이든 meta 든) meta 신원의 비율 · 정책 · 시드 값이
    등록과 같아야 한다 — 몫은 시드로 고른다. 어긋나면 `ExportRefused`. 사유에는 id 를 싣지 않는다.
    """
    from vlm.budget import BUDGET_POLICY, BudgetRejected, apply_budget
    from vlm.pilot_vlm import load_pairs, train_rows_digest

    got_f = ident.get("budget_fraction", 1.0)
    if fraction != 1.0 or got_f != 1.0:
        if got_f != fraction or ident.get("budget_policy") != BUDGET_POLICY:
            raise ExportRefused("train_budget", f"어댑터 meta 의 예산 비율 {got_f!r} · 정책 {ident.get('budget_policy')!r} 이 학습 "
                                                f"등록의 비율 {fraction!r} · 정책 {BUDGET_POLICY!r} 과 다르다")
        if type(ident.get("seed_value")) is not int or ident.get("seed_value") != seed_value:
            raise ExportRefused("adapter_seed", "어댑터 meta 의 시드 값이 등록의 시드표와 다르다 — 예산 몫은 시드로 고른다")

    def digest(c: str | None) -> str:
        rows = load_pairs("train", c, pairs_path=pairs_path)
        try:
            rows = apply_budget(rows, fraction=fraction, seed_value=seed_value, pairs_path=pairs_path)
        except BudgetRejected as exc:
            raise ExportRefused("train_budget", str(exc)) from None
        return train_rows_digest(rows)

    want = {c: digest(c) for c in ("C1", "C2", "C3")} if tag == "uni_fed" else digest(client)
    if rows_got != want:
        if fraction == 1.0:
            raise ExportRefused("train_rows", "본실험 어댑터가 페어의 참여자 행 전체로 학습되지 않았다(부분 목록)")
        raise ExportRefused("train_rows", "본실험 어댑터가 페어의 참여자 행 전체에서 학습 등록의 예산 몫으로 학습되지 않았다 "
                                          "— 부분 목록 · 다른 비율 · 다른 시드")


def _train_budget_fraction_at(root: Path, rev: str) -> float:
    """영수증 커밋의 학습 등록(`configs/train_registration.yaml`)이 정한 공통 학습 예산 비율 — 작업 트리가 아니라 그 커밋에서 읽는다.
    그 커밋에 학습 등록이 없으면 1.0 이다 — 전체 행 대조 그대로이고, 비율 칸이 있는 어댑터는 그때 `train_budget` 으로 거부된다.
    등록의 꼴이 틀리면 거부한다."""
    import yaml

    from evaluation.entry_gate import EntryRejected, blob_at
    from vlm.budget import BudgetRejected
    from vlm.registration import (
        REG_PATH,
        RegistrationRejected,
        budget_fraction_of,
        registration_from_bytes,
    )

    rel = REG_PATH.as_posix()
    if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{rev}:{rel}"], capture_output=True,
                      check=False).returncode != 0:
        return 1.0
    try:
        return budget_fraction_of(registration_from_bytes(blob_at(root, rev, rel)))
    except (EntryRejected, RegistrationRejected, BudgetRejected, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ExportRefused("train_registration", f"영수증 커밋의 학습 등록을 읽지 못했다: {exc}") from None


def _main_sample_inputs(reg, *, root: Path, rev: str, list_raw: bytes, side_path: Path | None,
                        registration_rel: str) -> dict | None:
    """본실험 모델 묶음의 모집단 모드 — 등록의 `population.weighting` 이 정한다(채점 쪽 `_strict_population` 과 같은 갈래, 1016k).

    표본(`stratified_g_over_n`)이면 영수증 커밋의 평가 설정을 네 소비자 공통 판독(`evaluation.eval_registration.load_bound_eval` —
    판 5 는 `configs/eval_registration.yaml`, 신판 판 6 은 같은 줄기의 `.eval.yaml`, 모두 그 커밋의 blob)으로 읽어 등록과의 결속까지
    본 뒤 주 표본 크기(`population.sample.main_size`)의 봉인을 고르고, 곁 파일 바이트가 봉인한 `side_sha256` 이며 목록과 같은 순서인지
    본다 — `{"sealed", "side"}`. 설정 부재 · 결속 어긋남은 모델 적재 · 시작 도장 · 생성보다 먼저 거부한다(판 6 연결 2 판 §2-2). 그 밖은
    전량 모드라 None 이고, 곁 파일을 주면 거부한다."""
    from evaluation.eval_registration import (
        EvalBindingRefused,
        EvalRegistrationError,
        load_bound_eval,
    )
    from evaluation.sample_weights import WEIGHTING_SAMPLE, SampleSideError, read_side

    if getattr(reg.population, "weighting", None) != WEIGHTING_SAMPLE:
        if side_path is not None:
            raise ExportRefused("side_in_full_mode", "전량 모드(가중이 표본이 아니다)의 등록에 표본 곁 파일이 주어졌다")
        return None
    try:
        doc = load_bound_eval(repo=root, anchor_rev=rev, registration_rel=registration_rel, registration=reg).doc
    except EvalBindingRefused as exc:
        raise ExportRefused(exc.code, f"영수증 커밋의 평가 설정 결속 — {len(exc.problems)}건: " + " · ".join(exc.problems[:3])) from None
    except EvalRegistrationError as exc:
        raise ExportRefused("eval_registration", f"영수증 커밋의 평가 등록 설정을 읽지 못했다: {exc}") from None
    sample = (doc.get("population") or {}).get("sample") if isinstance(doc.get("population"), Mapping) else None
    table = sample.get("sealed") if isinstance(sample, Mapping) else None
    sealed = table.get(str(sample.get("main_size"))) if isinstance(table, Mapping) else None
    if not isinstance(sealed, Mapping):
        size = sample.get("main_size") if isinstance(sample, Mapping) else None
        raise ExportRefused("main_sample_not_sealed", f"평가 등록 설정에 주 표본 크기 {size!r} 의 봉인이 없다")
    if side_path is None or not Path(side_path).is_file():
        raise ExportRefused("side_missing", "표본 모드의 본실험 모델 묶음은 봉인한 곁 파일(--side)이 있어야 한다 — 없거나 파일이 아니다")
    side_raw = Path(side_path).read_bytes()
    if hashlib.sha256(side_raw).hexdigest() != sealed.get("side_sha256"):
        raise ExportRefused("side_hash", "곁 파일의 바이트가 봉인한 side_sha256 과 다르다")
    try:
        side = read_side(side_raw, list_raw)
    except SampleSideError as exc:
        raise ExportRefused("side_file", f"곁 파일을 읽지 못했다: {exc}") from None
    return {"sealed": dict(sealed), "side": side}


def _positive_int(v: Any) -> bool:
    return type(v) is int and v > 0


def _ecc_step_paths(adapter_dir: Path, step: int) -> dict[str, Path]:
    """점검 step 파일의 자리 — 검사(`_training_check_step`)와 생성 설정이 같은 함수를 쓴다(`vlm.ecc.step_paths`)."""
    from vlm.ecc import step_paths

    return step_paths(Path(adapter_dir), step)


def _ecc_seed_value(reg, seed_index: Any) -> int:
    """조기 점검 export 의 시드 — 영수증에 묶인 등록의 시드표에서 고른 **실제 시드**(외부 검토 §137 C-01).

    점검 체크포인트의 신원은 시드 번호가 아니라 실제 시드(`seed`)를 싣는다. 번호는 bool 아닌 양의 정수, 시드표의 값은 모두 bool
    아닌 0 이상의 정수이고 서로 달라야 한다 — 등록 판독기(`evaluation.prereg_unified` 의 채점 쪽 검사)의 계약을 이 자리에서 다시
    본다(형 변환으로 넘기지 않는다). 생성 쪽 영수증 검사는 그 채점 쪽 검사를 부르지 않으므로, 값이 겹친 등록에서는 한 체크포인트가
    두 번호로 내보내질 수 있다(적대 검증).
    """
    seeds = getattr(reg, "seeds", None)
    if not _positive_int(seed_index) or not isinstance(seeds, Mapping) or str(seed_index) not in seeds:
        raise ExportRefused("seed_index", f"조기 점검 export 의 시드 번호 {seed_index!r} 가 등록의 시드표에 없다")
    values = list(seeds.values())
    if any(type(v) is not int or v < 0 for v in values) or len(set(values)) != len(values):
        raise ExportRefused("seed_index", f"등록의 시드표 값이 서로 다른 0 이상의 정수가 아니다: {seeds!r}")
    return seeds[str(seed_index)]


def _training_check_step(adapter_dir: Path, *, step: int, tag: str, measured: Mapping, coord_hash: str, cfg,
                         reg, run_root: Path | None, seed_index: int) -> tuple[dict, dict, dict]:
    """조기 좌표 점검의 학습 쪽 출처 — **별도 분기**다(11번 §3-3). `_training`(본실험 · 리허설 · 진단)의 검사는 고치지 않는다.

    받는 것은 점검 실행의 step 파일이다 — meta 의 목적 · step · 등록 step 목록, step 어댑터 · 소비 행 목록의 sha256 이 meta 와 같고,
    점검 원장에 그 step 줄이 정확히 하나 있고 그 줄의 지문이 meta 와 같다. 학습 행은 본실험 중앙과 같은(페어의 학습 분할 전체) 행이다.

    시드(§137 C-01): 실제 작성자(`vlm.pilot_vlm` 의 `UniResumeIdentity`)는 시드 번호를 쓰지 않는다. 등록의 시드표에서 고른 실제
    시드와 신원의 `seed` 를, 그리고 `round_idx == client_idx == 0` 을 맞댄다 — 점검은 라운드 0 · 참여자 0 으로 돌고
    `derive_seed(base, r, c) = base + 10007 r + 101 c` 이므로 그때만 실제 시드가 등록값과 같다(base 7 · 참여자 1 = 108 과 base 108 의
    겹침을 막는다). 실제 시드에서 번호를 역산하지 않는다. 신원에 옛 `seed_index` 칸이 있으면 같아야 한다.
    """
    from evaluation.schema_v14 import tag_parts
    from vlm.ecc import LEDGER_FILE as ECC_LEDGER
    from vlm.ecc import PURPOSE, ledger_rows
    from vlm.pilot_vlm import load_pairs, train_rows_digest
    from vlm.train_cell import CENTRAL_TAG, _record_path, _sha256_file

    if not _positive_int(step):
        raise ExportRefused("check_step", f"점검 step 은 bool 아닌 양의 정수다: {step!r}")
    seed_value = _ecc_seed_value(reg, seed_index)
    if tag != CENTRAL_TAG:
        raise ExportRefused("check_tag", f"조기 좌표 점검은 중앙 칸 자체만 받는다: {tag!r}")
    d = Path(adapter_dir)
    paths = _ecc_step_paths(d, step)
    led = d / ECC_LEDGER
    for p in (paths["meta"], paths["adapter"], paths["consumed"], led):
        if not p.is_file():
            raise ExportRefused("check_step_missing", f"점검 step 의 산출물이 없다: {p.name}")
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ExportRefused("check_step", "점검 step meta 가 객체가 아니다")
    if meta.get("purpose") != PURPOSE:
        raise ExportRefused("check_purpose", "점검 실행의 step 이 아니다")
    registered = meta.get("registered_steps")
    if (not _positive_int(meta.get("step")) or meta["step"] != step or not isinstance(registered, list) or not registered
            or not all(_positive_int(s) for s in registered) or step not in registered):
        raise ExportRefused("check_step", f"step {step} 이 meta 의 step · 등록 step 목록과 맞지 않는다(bool 아닌 양의 정수)")
    ident = meta.get("identity")
    if not isinstance(ident, dict):
        raise ExportRefused("adapter_identity", "점검 step meta 의 신원이 객체가 아니다")
    if ident.get("cell") != tag:
        raise ExportRefused("adapter_identity", f"점검 step meta 의 칸 {ident.get('cell')!r} ≠ {tag!r}")
    for key, want in (("round_idx", 0), ("client_idx", 0), ("seed", seed_value)):
        if type(ident.get(key)) is not int or ident[key] != want:
            raise ExportRefused("adapter_identity", f"점검 step 신원의 {key} {ident.get(key)!r} ≠ {want!r}"
                                                    f"(등록의 시드표 {seed_index} 번 · 라운드 0 · 참여자 0)")
    if "seed_index" in ident and (type(ident["seed_index"]) is not int or ident["seed_index"] != seed_index):
        raise ExportRefused("adapter_identity", f"점검 step 신원의 seed_index {ident['seed_index']!r} ≠ {seed_index!r}")
    if _sha256_file(paths["adapter"]) != (meta.get("adapter") or {}).get("sha256"):
        raise ExportRefused("adapter_sha", "점검 step 어댑터의 sha 가 meta 와 다르다")
    if _sha256_file(paths["consumed"]) != (meta.get("consumed") or {}).get("sha256"):
        raise ExportRefused("check_consumed", "소비 행 목록의 sha 가 meta 와 다르다")
    lines = [r for r in ledger_rows(d) if r.get("event") == "step" and r.get("step") == str(step)]
    if len(lines) != 1 or lines[0].get("adapter_sha256") != meta["adapter"]["sha256"] \
            or lines[0].get("consumed_sha256") != meta["consumed"]["sha256"]:
        raise ExportRefused("check_ledger", "점검 원장에 이 step 의 줄이 정확히 하나가 아니거나 지문이 meta 와 다르다")
    if ident.get("processor_config_sha256") != measured.get("processor_config_sha256"):
        raise ExportRefused("adapter_processor", "점검 학습의 프로세서 설정이 export 의 프로세서와 다르다")
    if ident.get("coord_cfg_hash") != coord_hash:
        raise ExportRefused("adapter_coord", "점검 학습의 좌표 해시가 export 의 좌표 규약과 다르다")
    g = reg.generation
    rows_got = meta.get("train_rows_digest")
    if g.train_rows_digest is not None and rows_got != g.train_rows_digest:
        raise ExportRefused("train_rows", "점검 step meta 의 train_rows_digest 가 등록과 다르다")
    if cfg.pairs_path is None:
        raise ExportRefused("train_rows", "점검 학습 행을 다시 낼 페어 경로가 설정에 없다")
    if rows_got != train_rows_digest(load_pairs("train", None, pairs_path=cfg.pairs_path)):
        raise ExportRefused("train_rows", "점검 실행이 본실험 중앙의 학습 행(페어의 학습 분할 전체)으로 학습되지 않았다")
    if not isinstance(meta.get("train_config"), str):
        raise ExportRefused("train_config", "점검 step meta 에 학습 설정 원문이 없다")
    if run_root is None or not _under(d, run_root):
        raise ExportRefused("run_root", "점검 step 이 이 루트 아래가 아니다")
    cell, client = tag_parts(tag)
    stamp = {"train_run_id": ident.get("run_id"), "train_ledger_sha256": _sha256_file(led),
             "scored_adapter_sha256": meta["adapter"]["sha256"], "check_step": step}
    side = {"train_ledger_path": _record_path(led), "scored_adapter_step": step, "train_config": meta["train_config"],
            "check_consumed_sha256": meta["consumed"]["sha256"], "check_consumed_rows": int(meta["consumed"]["rows"]),
            "check_pin_sha256": meta.get("pin_sha256")}
    out = {"init_adapter_digest": ident.get("input_adapter_digest"), "adapter_step": int(step), "cell": cell,
           "client": client}
    return out, stamp, side


#: 커밋된 등록 옆의 학습 설정 원문 — 등록 파일과 같은 줄기 이름(`{purpose}-{YYYYMMDD}-{n}`)에 이 꼬리를 붙인다(평가 쪽 `configs/registration/README.md`).
TRAIN_CONFIG_SUFFIX = ".train_config.json"


def _v6_code(line: str) -> str:
    """학습 쪽 대조 줄 `narrative_inputs: <사유 코드>: <설명>` 에서 안정된 사유 코드만 — 꼴이 다르면 `narrative_inputs`."""
    rest = line.split(": ", 1)[1] if line.startswith("narrative_inputs: ") else line
    head = rest.split(": ", 1)[0]
    return head if ": " in rest and " " not in head else "narrative_inputs"


def _v6_regime_checks(reg, *, root: Path, rev: str, mode: str, adapter_dir, adapter_meta: Mapping[str, Any]) -> None:
    """판 6 본실험 묶음의 체제 검사(판 6 연결 2 판 §4-3) — 영수증 커밋의 학습 등록으로 프로필을 고르고(평가 쪽 `profile_for`), 신판
    (`regime2_v4`)이면 계약 칸(`v6_form`) · 판 6 신원(검사기 v4 행 · 생성 묶음 · 증거 자료 — `identity_problems`) · 모델 모드는 계약 ↔
    PIN(`vlm.narrative_inputs.v6_problems` — 영수증 커밋의 계약 blob · 시드 폴더의 착수 고정 · meta 의 `start_gate.pin_sha256`, `repo` 로
    계약 실물까지) · 에코 모드는 계약 실물만 본다(에코에 PIN · 어댑터를 지어 넣지 않는다). 모두 생성기를 열기 전 · writer 를 시작하기
    전이다. 판 5 와 구판 프로필(역사적 판 6)은 옛 동작 그대로다."""
    from evaluation import narrative_registration as NR
    from evaluation.entry_gate import EntryRejected, blob_at
    from evaluation.prereg_unified import REGISTRATION_VERSION_V6
    from vlm.narrative_inputs import ENTRY_KEY, contract_rel, v6_problems
    from vlm.registration import REG_PATH, RegistrationRejected, registration_from_bytes

    if reg.version != REGISTRATION_VERSION_V6:
        return
    try:
        entry = registration_from_bytes(blob_at(root, rev, REG_PATH.as_posix()))
    except EntryRejected:
        entry = None
    except RegistrationRejected as exc:
        raise ExportRefused("train_registration", f"영수증 커밋의 학습 등록을 읽지 못했다: {exc}") from None
    def contract_lines() -> list[str]:
        """계약 ↔ PIN(모델 모드 — `v6_problems`, 계약 실물까지) · 계약 실물(에코 모드)의 거부 줄."""
        if mode == "model":
            return v6_problems(reg, read_at_receipt=lambda p: blob_at(root, rev, p), adapter_dir=Path(adapter_dir),
                               adapter_meta=adapter_meta, repo=root)
        ni = entry.get(ENTRY_KEY) if isinstance(entry, Mapping) else None
        if not isinstance(ni, Mapping) or type(ni.get("version")) is not int:
            raise ExportRefused("pin_contract", "영수증 커밋의 학습 등록에 학습 전 계약 칸이 없다 — 에코도 계약 실물을 맞댄다")
        try:
            contract_raw = blob_at(root, rev, contract_rel(ni["version"]))
        except EntryRejected:
            raise ExportRefused("v6_form", "영수증 커밋에 학습 전 계약 파일이 없다") from None
        lines = NR.contract_code_problems(contract_raw, repo=root)
        if hashlib.sha256(contract_raw).hexdigest() != reg.narrative.inputs_contract_sha256:
            lines = ["v6_contract_sha: 영수증 커밋의 계약 바이트가 판 6 의 narrative.inputs_contract_sha256 과 다르다", *lines]
        return lines

    try:
        profile = NR.profile_for(reg, train_entry=entry)
    except NR.ProfileRefused as exc:
        pin, n = NR.NARRATIVE_PROFILES[NR.PROFILE_REGIME2].checker_pin, reg.narrative
        if n is not None and n.inputs_contract_sha256 and \
                (n.checker_id, n.limits_sha256, n.lists_sha256) == (pin.checker_id, pin.limits_sha256, pin.lists_sha256):
            lines = contract_lines()                          # v4 행은 맞고 계약 결속만 어긋났다 — 더 좁은 사유(v6_contract_sha · pin_contract)
            if lines:
                raise ExportRefused(_v6_code(lines[0]), f"판 6 계약 — {len(lines)}건: " + " · ".join(lines[:3])) from None
        raise ExportRefused(exc.code, str(exc)) from None
    if profile.profile_id != NR.PROFILE_REGIME2:
        return
    if not reg.narrative.inputs_contract_sha256:
        raise ExportRefused("v6_form", "신판 판 6 의 narrative.inputs_contract_sha256 이 비었다 — 계약에 묶이지 않은 등록으로 export 하지 않는다")
    lines = contract_lines()
    if lines:
        raise ExportRefused(_v6_code(lines[0]), f"판 6 계약 — {len(lines)}건: " + " · ".join(lines[:3]))
    try:
        problems = NR.identity_problems(reg, repo=root, profile=profile)
    except OSError as exc:
        raise ExportRefused("code_bytes", f"판 6 신원을 잴 파일이 저장소에 없다: {exc}") from None
    if problems:
        raise ExportRefused(problems[0].split(":", 1)[0], f"판 6 신원 — {len(problems)}건: " + " · ".join(problems[:3]))


def registration_version_of_recipe(recipe: str) -> str:
    """본실험 등록의 판 — 1 차 레시피는 판 5, 새 체제(`json_line_then_basis`)는 판 6(서술 · 조항 축 블록, `evaluation.prereg_unified`).
    export 는 학습 설정 원문의 계약에서 되찾은 레시피의 판으로만 본실험 묶음을 낸다(지시 1023c). 판을 정하지 않은 레시피는 거부한다
    (레시피가 늘면 여기에 판을 적어야 export 가 열린다)."""
    from evaluation.prereg_unified import REGISTRATION_VERSION, REGISTRATION_VERSION_V6
    from vlm.pilot_vlm import RECIPE_V1, RECIPE_V2

    versions = {RECIPE_V1: REGISTRATION_VERSION, RECIPE_V2: REGISTRATION_VERSION_V6}
    if recipe not in versions:
        raise ValueError(f"본실험 등록의 판을 정하지 않은 레시피다: {recipe!r}")
    return versions[recipe]


def echo_train_config_rel(registration_path: str) -> str:
    """커밋된 등록의 학습 설정 원문 경로 — `configs/registration/main-20261001-1.json` → `….main-20261001-1.train_config.json`."""
    rel = PurePosixPath(registration_path)
    if rel.suffix != ".json" or rel.name.endswith(TRAIN_CONFIG_SUFFIX):
        raise ExportRefused("echo_train_config_place", f"등록 파일 이름이 `{{줄기}}.json` 꼴이 아니다: {registration_path}")
    return rel.with_name(rel.stem + TRAIN_CONFIG_SUFFIX).as_posix()


def _echo_train_config(rc, root: Path) -> str:
    """에코의 학습 설정 원문(§2-5). 본실험 · 진단(커밋된 영수증)은 **영수증 커밋에서** 등록 옆의 `{줄기}.train_config.json` 을 읽는다 —
    작업 트리가 아니다(`git show <main_commit>:<경로>`, 평가 쪽 README 의 "등록 파일 옆에 두는 것"). 리허설은 등록 파일 옆의 `train_config.json`.
    어느 쪽이든 그 바이트의 sha256 이 등록의 `train_config_sha256` 인지는 실측 대조(`check_actuals`)가 본다."""
    from evaluation.entry_gate import COMMITTED_KINDS, EntryRejected, blob_at

    if rc.kind in COMMITTED_KINDS:
        rel = echo_train_config_rel(rc.registration_path)
        try:
            raw = blob_at(root, rc.main_commit, rel)
        except EntryRejected:
            raise ExportRefused("echo_train_config",
                                f"영수증 커밋에 등록 옆의 학습 설정 원문이 없다: {rel}") from None
        return raw.decode("utf-8")
    p = root / PurePosixPath(rc.registration_path).parent / "train_config.json"
    if not p.is_file():
        raise ExportRefused("echo_train_config", f"등록 옆의 학습 설정 원문이 없다: {p.name}")
    return p.read_bytes().decode("utf-8")


def check_full_eval_set(el, meta) -> None:
    """본실험 모델 묶음의 전량 — 목록의 집합이 검증한 동결 매니페스트의 `split == eval` 전체와 같다. 어긋나면 `ExportRefused`.

    평가 쪽 채점 진입점의 관문(`evaluation.entry_gate.check_main_full_eval`)의 집합 판정과 같은 식이다 — 두 쪽이 같은 시험 벡터
    (`evaluation/full_eval_vectors.json` 의 생성 쪽 경우)를 같은 판정으로 지난다(`tests/test_export_full_eval.py`). 그 함수를 부르지는 않는다
    (소관을 나눈다). 같은 파일의 판정은 생성 쪽에 없다 — 생성 쪽은 목록이 하나다. 사유에는 건수만 싣는다.
    """
    full = {i for i, s in meta.split_of.items() if s == "eval"}
    missing, extra = len(full - el.id_set), len(el.id_set - full)
    if missing or extra:
        raise ExportRefused("main_list_not_full_eval", f"본실험 생성 목록이 동결 매니페스트의 평가 행 전체가 아니다 — 빠짐 {missing}개 · "
                                                       f"평가 행 밖 {extra}개 · 목록 {len(el)}개 · 평가 행 {len(full)}개")


def check_sample_eval_set(el, meta, *, sealed: Mapping[str, Any], side: Mapping[str, Any]) -> None:
    """본실험 모델 묶음의 표본 — 목록이 평가 등록 설정에 봉인한 주 표본이다. 어긋나면 `ExportRefused`(1016k).

    평가 쪽 채점 진입점의 표본 관문(`evaluation.entry_gate.check_main_sample`)과 같은 식이다 — 두 쪽이 같은 시험 벡터
    (`tests/main_sample_vectors.py`)를 같은 판정으로 지난다. 그 함수를 부르지는 않는다(소관을 나눈다, `check_full_eval_set` 과 같다).
    생성 쪽은 목록이 하나라 '채점 목록 = 생성 목록' 판정은 없다. 본다: 목록 = 봉인한 목록(두 해시) · 장수 = 봉인 · 뽑힌 묶음의 평가
    이미지 전부 · 묶음 수 = 봉인 · 곁 파일의 id 가 목록과 같은 순서 · 층이 묶음 단위이고 층마다 뽑힌 묶음 수가 `n_h`. 사유에는 건수만.
    """
    from evaluation.sample_bootstrap import SampleDesignError, strata_groups

    if (el.file_sha256, el.set_sha256) != (sealed.get("file_sha256"), sealed.get("set_sha256")):
        raise ExportRefused("main_sample_not_sealed", "본실험 표본 목록이 평가 등록 설정에 봉인한 목록이 아니다(해시)")
    if len(el) != sealed.get("images"):
        raise ExportRefused("main_sample_count", f"표본 목록 {len(el)}장 ≠ 봉인한 장수 {sealed.get('images')!r}")
    members: dict[str, set[str]] = {}
    for i, s in meta.split_of.items():
        if s == "eval":
            members.setdefault(meta.group_of[i], set()).add(i)
    groups = {meta.group_of[i] for i in el.ids}
    split_groups = sum(1 for grp in groups if not members.get(grp, set()) <= el.id_set)
    if split_groups:
        raise ExportRefused("main_sample_group_split", f"뽑힌 묶음 {split_groups}개의 평가 이미지가 표본에 다 들지 않았다")
    if len(groups) != sealed.get("groups"):
        raise ExportRefused("main_sample_groups", f"표본의 묶음 {len(groups)}개 ≠ 봉인한 묶음 수 {sealed.get('groups')!r}")
    if list(side) != list(el.ids):
        raise ExportRefused("main_sample_side", "곁 파일의 id 가 표본 목록과 같은 순서로 같지 않다")
    try:
        strata_groups(el.ids, meta.group_of, side)
    except SampleDesignError as exc:
        raise ExportRefused("main_sample_strata", str(exc)) from None


def check_generation_list(el, g, *, mode: str, purpose: str, meta_of: Callable[[], Any],
                          sample: Mapping[str, Any] | None = None) -> None:
    """7 의 목록 검사 — 등록의 칸 → 동결 매니페스트의 분할 → 본실험 모델 묶음의 전량 또는 봉인 표본. 어긋나면 `ExportRefused`.

    `el` 은 `validate_eval_list` 가 만든 목록, `g` 는 등록의 생성 쪽, `meta_of` 는 검증한 동결 매니페스트를 읽는 함수다(등록 대조가
    먼저 거부하면 읽지 않는다). 전량의 기준 집합은 호출자의 목록이 아니라 매니페스트에서 만든다 — 평가 쪽 채점 진입점의 전량 관문
    (`check_main_full_eval`)과 같은 식이다. `sample`(`{"sealed", "side"}`, 등록의 가중이 표본일 때 사전 점검이 낸다)이 있으면 전량
    대신 표본 관문(`check_sample_eval_set`)을 본다. 사유에는 건수만 싣고 id 는 싣지 않는다.
    """
    from evaluation.entry_gate import EntryRejected, check_list_split

    want = ((g.echo_list_file_sha256, g.echo_list_set_sha256) if mode == "echo"
            else (g.eval_list_file_sha256, g.eval_list_set_sha256))
    if (el.file_sha256, el.set_sha256) != want:
        raise ExportRefused("list_binding", f"목록의 두 해시가 등록의 {'에코' if mode == 'echo' else '생성'} 목록 칸과 다르다")
    try:
        meta = meta_of()
        check_list_split(el.ids, meta, purpose, where="export 목록", echo=(mode == "echo"))
    except EntryRejected as exc:
        raise _refuse_from(exc) from None
    if purpose == "main" and mode == "model":
        if sample is None:
            check_full_eval_set(el, meta)
        else:
            check_sample_eval_set(el, meta, sealed=sample["sealed"], side=sample["side"])


def export_preflight(*, purpose: str, mode: str, tag: str, seed_index: int, receipt_path: Path, list_path: Path,
                     snapshot_root: Path, adapter_dir: Path | None, plan_path: Path | None, run_root: Path | None,
                     measure_env: Callable, impl_ids: Mapping[str, Any], parser_sha256: str,
                     repo: Path | None = None, code_repo: Path | None = None, exposure_ledger: Path | None = None,
                     now: datetime | None = None, check_step: int | None = None,
                     side_path: Path | None = None) -> PreflightResult:
    """§2-4 의 1. 통과하면 도장 · 곁 파일에 실을 값과 생성기를 여는 값을 돌려준다. 어긋나면 `ExportRefused`.

    `side_path` 는 표본 모드 본실험 모델 묶음의 곁 파일(`image_id,stratum,G_h,n_h`)이다 — 그 밖에서는 받지 않는다."""
    from evaluation.actuals import check_actuals
    from evaluation.entry_gate import (
        COMMITTED_KINDS,
        EntryRejected,
        anchor_rev,
        blob_at,
        check_anchor,
        check_processor,
        check_receipt_kind,
        check_snapshot,
        guard_seam,
        load_anchor,
        load_processor_kwargs,
        load_registration_for,
        main_checkout,
        read_manifest_meta,
    )
    from evaluation.eval_list import validate_eval_list
    from evaluation.prereg_unified import (
        ReceiptMissing,
        RegistrationIncomplete,
        RegistrationInvalid,
        read_receipt,
        require_receipt,
    )
    from evaluation.reject_v14 import BundleRejected
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.export_measure import MeasureSpec
    from vlm.exposure import default_ledger as exposure_default
    from vlm.exposure import find_planned
    from vlm.pilot_vlm import load_prompt
    from vlm.run_root import REPO_ROOT

    if side_path is not None and not (purpose == "main" and mode == "model"):
        raise ExportRefused("side_scope", "표본 곁 파일은 본실험 모델 묶음에서만 받는다")
    notes: list[str] = []
    entry_errors = (EntryRejected, ReceiptMissing, RegistrationIncomplete, RegistrationInvalid)
    try:
        # 1 · 2 영수증과 등록
        rc = read_receipt(Path(receipt_path))
        check_receipt_kind("strict" if purpose == "main" else "probe", purpose, rc)
        if repo is not None:
            guard_seam(rc)
        root = Path(repo) if repo is not None else main_checkout()
        reg = load_registration_for(rc, repo=repo)
        if rc.kind not in COMMITTED_KINDS and run_root is not None \
                and not _under(root / PurePosixPath(rc.registration_path), run_root):
            raise ExportRefused("run_root", "리허설 등록 파일이 이 루트 아래가 아니다")
        # 3 닻 · 프로세서 인자 — 같은 리비전의 설정 바이트
        anchor = load_anchor(rc, repo=repo)
        check_anchor(reg, anchor)
        processor_kwargs = load_processor_kwargs(rc, repo=repo)
        check_processor(reg, processor_kwargs)
        cfg = _uni_config_at(blob_at(root, anchor_rev(rc), "configs/base.yaml"))
        # 4 생성 쪽 완결과 지문
        gen_sha = require_receipt(reg, rc, side="generation")
        # 5 스냅샷
        absorption = check_snapshot(Path(snapshot_root), anchor)
    except ReceiptMissing as exc:
        raise ExportRefused("entry_RECEIPT", str(exc)) from None
    except entry_errors as exc:
        raise _refuse_from(exc) from None
    g = reg.generation
    if purpose == "main" and absorption == "provisional":
        raise ExportRefused("provisional", "본실험 export 는 잠정 동결본을 받지 않는다")
    if reg.seeds is None or str(seed_index) not in reg.seeds:
        raise ExportRefused("seed_index", f"등록의 시드표에 {seed_index} 번이 없다")
    seed_value = int(reg.seeds[str(seed_index)])
    # 본실험 모델 묶음의 학습 행 대조에 쓰는 공통 학습 예산 비율 — 영수증 커밋의 학습 등록에서(1016k)
    budget_fraction = _train_budget_fraction_at(root, anchor_rev(rc)) if purpose == "main" and mode == "model" else 1.0
    if now is not None and now < rc.time:
        raise ExportRefused("clock_before_receipt", "지금 시각이 영수증 시각보다 이르다")

    # 6 계획 파일
    plan_sha, plan = _plan(plan_path, g, purpose)

    # 7 목록 — 바이트로 · 등록의 칸과 · 동결 매니페스트의 분할과
    list_raw = Path(list_path).read_bytes()
    try:
        el = validate_eval_list(list_raw, where=Path(list_path).name)
    except BundleRejected as exc:
        raise ExportRefused("list_form", str(exc)) from None
    sample = (_main_sample_inputs(reg, root=root, rev=anchor_rev(rc), list_raw=list_raw, side_path=side_path,
                                  registration_rel=rc.registration_path)
              if purpose == "main" and mode == "model" else None)
    check_generation_list(el, g, mode=mode, purpose=purpose, meta_of=lambda: read_manifest_meta(Path(snapshot_root)),
                          sample=sample)

    # 8 노출 기록 — 이미지를 열기 전에
    if purpose != "main":
        if run_root is None:
            raise ExportRefused("run_root", f"{purpose} 는 루트가 있어야 한다")
        ledger = Path(exposure_ledger) if exposure_ledger is not None else exposure_default()
        if find_planned(ledger, run_id=Path(run_root).name, list_kind="echo" if mode == "echo" else "gen",
                        list_file_sha256=el.file_sha256) is None:
            raise ExportRefused("exposure_not_planned", "노출 원장에 이 목록의 planned 줄이 없다 — 노출 기록이 먼저다(§5-4)")

    # 9 실측값 — 설정 · 프롬프트 파일 · 프로세서
    for name in ("model_id", "prompt_path", "coord_space", "chat_template_kwargs"):
        if getattr(cfg, name) in (None, ""):
            raise ExportRefused("config_incomplete", f"설정의 {name} 이 비어 있다")
    prompt_text, prompt_sha = load_prompt(REPO_ROOT / cfg.prompt_path)
    max_new = (plan.get("gen_limit") or {}).get("max_new_tokens") if purpose != "main" else cfg.max_new_tokens
    if max_new is None:
        raise ExportRefused("gen_limit", "생성 한도가 정해지지 않았다(본실험은 설정, 리허설 · 진단은 계획의 gen_limit)")
    measured = dict(measure_env(MeasureSpec(model_id=cfg.model_id, model_revision=cfg.model_revision,
                                            processor_kwargs=dict(processor_kwargs), prompt_text=prompt_text,
                                            chat_template_kwargs=dict(cfg.chat_template_kwargs))))
    chash = coord_cfg_hash(CoordCfg(coord_space=cfg.coord_space))

    # 10 모델 모드의 학습 출처 · 에코의 학습 설정 원문
    stamp: dict[str, Any] = {}
    side: dict[str, Any] = {}
    if (check_step is not None) != (purpose == "early_coord_check" and mode == "model"):
        # 점검 step 어댑터는 점검 목적의 모델 모드만 받는다 — 본실험 · 리허설 · 진단의 export 는 step 어댑터를 받지 않는다(11번 T6).
        raise ExportRefused("check_step_scope", "점검 step 은 early_coord_check 의 모델 모드에서만(그때는 반드시) 준다")
    if check_step is not None and not _positive_int(check_step):
        raise ExportRefused("check_step", f"점검 step 은 bool 아닌 양의 정수다: {check_step!r}")
    if mode == "model":
        if adapter_dir is None:
            raise ExportRefused("adapter_missing", "모델 모드는 어댑터 폴더가 있어야 한다")
        if purpose == "early_coord_check":
            meta, stamp, side = _training_check_step(Path(adapter_dir), step=check_step, tag=tag, measured=measured,
                                                     coord_hash=chash, cfg=cfg, reg=reg, run_root=run_root,
                                                     seed_index=seed_index)
        else:
            meta, stamp, side = _training(Path(adapter_dir), tag=tag, seed_index=seed_index, purpose=purpose, reg=reg,
                                          measured=measured, coord_hash=chash, cfg=cfg, run_root=run_root,
                                          seed_value=seed_value, budget_fraction=budget_fraction)
        tc_text = side["train_config"]
        init_digest = meta.get("init_adapter_digest")
    else:
        if adapter_dir is not None:
            raise ExportRefused("echo_with_adapter", "에코는 어댑터를 받지 않는다")
        tc_text = _echo_train_config(rc, root)
        init_digest = None
    try:
        tc = json.loads(tc_text) if isinstance(tc_text, str) else None
    except (json.JSONDecodeError, ValueError):
        tc = None
    if not isinstance(tc, dict):
        raise ExportRefused("train_config", "학습 설정 원문을 읽지 못했다")
    from vlm.pilot_vlm import recipe_of_contract

    try:
        target_recipe = recipe_of_contract(tc.get("target_contract_sha256"))
    except ValueError as exc:
        raise ExportRefused("train_config_contract", f"학습 설정 원문의 타깃 계약이 아는 레시피의 것이 아니다: {exc}") from None
    # 본실험 등록의 판 ↔ 학습 레시피 — 생성기를 열기 전이다(지시 1023c — 같은 임시 세계의 사슬에서 '판 5 등록에 2 차 산출' 이
    # 어디서도 멈추지 않았다).
    if purpose == "main":
        try:
            want_version = registration_version_of_recipe(target_recipe)
        except ValueError as exc:
            raise ExportRefused("registration_version", str(exc)) from None
        if reg.version != want_version:
            raise ExportRefused("registration_version", f"학습 레시피 {target_recipe!r} 의 산출은 본실험 등록 판 {want_version} 으로만 "
                                                        f"export 한다 — 이 등록은 판 {reg.version!r}")
    if purpose == "main":
        _v6_regime_checks(reg, root=root, rev=anchor_rev(rc), mode=mode, adapter_dir=adapter_dir,
                          adapter_meta=meta if mode == "model" else {})
    if init_digest is None:
        init_digest = tc.get("init_adapter_digest")
    side["train_config"] = tc_text

    actual: dict[str, Any] = {
        "purpose": purpose, "list_split": g.list_split, "snapshot_digest": anchor.snapshot_digest,
        "prompt_sha256": prompt_sha, "chat_template_kwargs": dict(cfg.chat_template_kwargs),
        "max_new_tokens": max_new, "decoding": g.decoding, "batch_size": cfg.batch_size,
        "padding_side": g.padding_side, "coord_space": cfg.coord_space, "coord_cfg_hash": chash,
        "base_model_id": cfg.model_id, "base_model_revision": cfg.model_revision,
        "train_config_sha256": hashlib.sha256(tc_text.encode("utf-8")).hexdigest(),
        "budget_n": tc.get("total_epochs"), "budget_r": tc.get("num_rounds"), "budget_e": tc.get("local_epochs"),
        "impl_ids": dict(impl_ids), "init_adapter_digest": init_digest, "start_checkpoint_sha256": None,
        "eval_list_file_sha256": el.file_sha256, "eval_list_set_sha256": el.set_sha256,
        **{k: measured.get(k) for k in ("processor_config_sha256", "processor_min_pixels", "processor_max_pixels",
                                        "patch_size", "merge_size", "gen_prefix_sha256", "transformers_version")},
    }
    if plan_sha is not None:
        actual["plan_sha256"] = plan_sha
    stand_in = False
    bad = []
    for m in check_actuals(reg, actual, mode=mode):
        if m.item == "impl_ids" and purpose == "rehearsal":
            stand_in = True
            notes.append("구현 식별자가 등록과 다르다 — 대역 실행으로 적힌다")
            continue
        bad.append(m.item)
    if bad:
        raise ExportRefused("actuals", f"실측값이 등록과 다르다: {sorted(bad)}")

    # 11 작업 트리 청결
    commit, dirty = code_commit(Path(code_repo) if code_repo is not None else REPO_ROOT)
    if dirty:
        raise ExportRefused("tree_dirty", f"코드가 커밋과 다르다 — export_commit 이 코드를 가리키지 못한다: {dirty[:5]}")

    stamp_fields = {"purpose": purpose, "export_commit": commit, "parser_sha256": parser_sha256,
                    "generation_sha256": gen_sha, **stamp}
    sidecar_values = {**actual, **side, "generation_sha256": gen_sha}
    gen_cfg = GenerationConfig(mode=mode, tag=tag, model_id=cfg.model_id, model_revision=cfg.model_revision,
                               processor_kwargs=dict(processor_kwargs), prompt_text=prompt_text,
                               chat_template_kwargs=dict(cfg.chat_template_kwargs), max_new_tokens=int(max_new),
                               decoding=dict(g.decoding or {}), batch_size=int(cfg.batch_size),
                               padding_side=str(g.padding_side), coord_space=cfg.coord_space,
                               pairs_path=cfg.pairs_path,
                               # 점검 step 어댑터는 검사한 그 파일 — 이름 규칙을 손으로 다시 쓰지 않고 검사기와 같은 함수로(적대 검증)
                               adapter_path=(None if mode != "model" else
                                             _ecc_step_paths(Path(adapter_dir), check_step)["adapter"]
                                             if check_step is not None else Path(adapter_dir) / "adapter_last.npz"),
                               purpose=purpose, target_recipe=target_recipe)
    return PreflightResult(registration=reg, receipt=rc, generation_sha256=gen_sha, eval_list=el,
                           seed_value=seed_value, gen_cfg=gen_cfg, stamp_fields=stamp_fields,
                           sidecar_values=sidecar_values, stand_in=stand_in, notes=notes)
