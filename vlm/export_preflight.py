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
8. 노출 기록(리허설 · 진단) — 이미지를 열기 전에 같은 실행 id · 목록 종류 · 목록 해시의 `planned` 줄이 있어야 한다(§5-4).
9. 실측값 — 설정(닻과 같은 리비전)의 모델 · 템플릿 · 좌표 규약 · 프롬프트 파일과 프로세서에서 잰 값(이음새 `measure_env`)으로
   곁 파일에 실을 값을 모으고 `check_actuals` 로 등록과 맞댄다. 리허설은 구현 식별자의 어긋남을 "대역 실행" 으로 적고 지나간다.
10. 모델 모드 — 어댑터 meta · 어댑터 파일 sha · 닫힌 원장(`read_ledger_view`) · 원장 끝 스텝 = meta 의 스텝 · 학습 설정 · 프로세서 ·
    좌표 해시 · 학습 행 digest(본실험은 페어에서 다시 낸다) · meta 의 목적 · 태그 · 시드 번째.
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

__all__ = ["FED_CLIENTS", "GenerationConfig", "PreflightResult", "code_commit", "export_preflight"]

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
              coord_hash: str, cfg, run_root: Path | None) -> tuple[dict, dict, dict]:
    """모델 모드의 학습 쪽 출처 — (meta, 도장 값, 곁 파일 값)."""
    from evaluation.ledger_v14 import LedgerRejected, read_ledger_view
    from evaluation.schema_v14 import tag_parts
    from vlm.pilot_vlm import load_pairs, train_rows_digest
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
    # 학습 행 — 등록에 값이 있으면 그것과, 본실험은 페어의 그 참여자 행 전체에서 다시 낸 값과 맞댄다(§2-7 · §5-2).
    rows_got = meta.get("train_rows_digest")
    if g.train_rows_digest is not None and rows_got != g.train_rows_digest:
        raise ExportRefused("train_rows", "어댑터 meta 의 train_rows_digest 가 등록과 다르다")
    if purpose == "main":
        if cfg.pairs_path is None:
            raise ExportRefused("train_rows", "본실험 학습 행을 다시 낼 페어 경로가 설정에 없다")
        if tag == "uni_fed":
            want_rows = {c: train_rows_digest(load_pairs("train", c, pairs_path=cfg.pairs_path)) for c in ("C1", "C2", "C3")}
        else:
            want_rows = train_rows_digest(load_pairs("train", client, pairs_path=cfg.pairs_path))
        if rows_got != want_rows:
            raise ExportRefused("train_rows", "본실험 어댑터가 페어의 참여자 행 전체로 학습되지 않았다(부분 목록)")
    if purpose != "main" and run_root is not None and not _under(d, run_root):
        raise ExportRefused("run_root", "어댑터가 이 루트 아래가 아니다")
    stamp = {"train_run_id": view.run_id, "train_ledger_sha256": view.file_sha256,
             "scored_adapter_sha256": meta["adapter_sha256"]}
    side = {"train_ledger_path": _record_path(led), "scored_adapter_step": int(meta["adapter_step"]),
            "train_config": meta.get("train_config")}
    return meta, stamp, side


#: 커밋된 등록 옆의 학습 설정 원문 — 등록 파일과 같은 줄기 이름(`{purpose}-{YYYYMMDD}-{n}`)에 이 꼬리를 붙인다(평가 쪽 `configs/registration/README.md`).
TRAIN_CONFIG_SUFFIX = ".train_config.json"


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


def export_preflight(*, purpose: str, mode: str, tag: str, seed_index: int, receipt_path: Path, list_path: Path,
                     snapshot_root: Path, adapter_dir: Path | None, plan_path: Path | None, run_root: Path | None,
                     measure_env: Callable, impl_ids: Mapping[str, Any], parser_sha256: str,
                     repo: Path | None = None, code_repo: Path | None = None, exposure_ledger: Path | None = None,
                     now: datetime | None = None) -> PreflightResult:
    """§2-4 의 1. 통과하면 도장 · 곁 파일에 실을 값과 생성기를 여는 값을 돌려준다. 어긋나면 `ExportRefused`."""
    from evaluation.actuals import check_actuals
    from evaluation.entry_gate import (
        COMMITTED_KINDS,
        EntryRejected,
        anchor_rev,
        blob_at,
        check_anchor,
        check_list_split,
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
    if now is not None and now < rc.time:
        raise ExportRefused("clock_before_receipt", "지금 시각이 영수증 시각보다 이르다")

    # 6 계획 파일
    plan_sha, plan = _plan(plan_path, g, purpose)

    # 7 목록 — 바이트로 · 등록의 칸과 · 동결 매니페스트의 분할과
    try:
        el = validate_eval_list(Path(list_path).read_bytes(), where=Path(list_path).name)
    except BundleRejected as exc:
        raise ExportRefused("list_form", str(exc)) from None
    want = ((g.echo_list_file_sha256, g.echo_list_set_sha256) if mode == "echo"
            else (g.eval_list_file_sha256, g.eval_list_set_sha256))
    if (el.file_sha256, el.set_sha256) != want:
        raise ExportRefused("list_binding", f"목록의 두 해시가 등록의 {'에코' if mode == 'echo' else '생성'} 목록 칸과 다르다")
    try:
        meta_m = read_manifest_meta(Path(snapshot_root))
        check_list_split(el.ids, meta_m, purpose, where="export 목록", echo=(mode == "echo"))
    except EntryRejected as exc:
        raise _refuse_from(exc) from None

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
    if mode == "model":
        if adapter_dir is None:
            raise ExportRefused("adapter_missing", "모델 모드는 어댑터 폴더가 있어야 한다")
        meta, stamp, side = _training(Path(adapter_dir), tag=tag, seed_index=seed_index, purpose=purpose, reg=reg,
                                      measured=measured, coord_hash=chash, cfg=cfg, run_root=run_root)
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
                               adapter_path=(Path(adapter_dir) / "adapter_last.npz") if mode == "model" else None,
                               purpose=purpose)
    return PreflightResult(registration=reg, receipt=rc, generation_sha256=gen_sha, eval_list=el,
                           seed_value=seed_value, gen_cfg=gen_cfg, stamp_fields=stamp_fields,
                           sidecar_values=sidecar_values, stand_in=stand_in, notes=notes)
