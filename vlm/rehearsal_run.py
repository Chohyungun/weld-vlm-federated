"""리허설의 단계 본체 — 목록(0) · 등록(0′) · 학습(1 · 2 · 2′ · 2″) · 합성 구판(2′ 의 재료). 리허설 2판 §1-2.

명령줄은 `scripts/rehearsal_uni.py` 다. 이 모듈의 함수는 **파이썬 인자**로 이음새(실측 · 모델 적재 · 생성기 적재)와 경로(본체 체크아웃 ·
노출 원장 · 비본실험 부모)를 받는다 — 합성 시험이 임시 폴더를 준다. 명령줄에는 구현을 고르는 인자가 없다.

**학습은 본실험 실행기(`scripts/main_uni.py`)가 아니라 이 오케스트레이터의 하위 명령으로 돈다.** 본실험 실행기는 목적이 언제나 `main` 이고
계획 · 학습 행 목록 인자를 아예 받지 않는다 — 받고 거부하는 것보다 길이 없는 편이 단단하다. 2판 §1-2 의 표는 진입점을 `main_uni.py central` 로
적었다(2판과 다르게 한 자리). 칸의 본체(`vlm.train_cell`)는 같은 것을 부른다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "LIST_FILES",
    "REHEARSAL_NEEDS",
    "StageRefused",
    "check_train_list",
    "diag_registration",
    "diag_rules_bytes",
    "local_registration",
    "purpose_of_root",
    "read_jsonl",
    "read_under",
    "rehearsal_spec",
    "stage_lists",
    "stage_register",
    "stage_train",
    "synth_negatives",
    "target_token_counts",
    "train_list_bytes",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
#: 리허설이 비워 둘 수 없는 설정 칸 — 본실험 키를 그대로 읽는다(2판 §4-4). 학습량 · 감독 토큰 기대값은 계획 쪽이다.
REHEARSAL_NEEDS = ("model_id", "model_revision", "pairs_path", "pairs_digest", "chat_template_kwargs", "seeds",
                   "snapshot_digest", "processor_kwargs", "prompt_path", "coord_space", "batch_size")
TV_COLUMNS = ("image_id", "split", "group_id", "strata_key", "client", "sha256")
EVAL_COLUMNS = ("image_id", "split", "sha256", "group_id")
LIST_FILES = {"gen": "gen.txt", "echo": "echo.txt"}
TRAIN_LIST = "train_{tag}.jsonl"
RECORD_FILE = "lists_record.json"
LOCAL_TAGS = {"uni_local_C1": "C1", "uni_local_C2": "C2", "uni_local_C3": "C3"}
CENTRAL_TAG = "uni_central"
NEGATIVE_CASES = ("policy_blank", "identity_key_missing", "identity_key_extra", "format_mismatch")
POLICY_FIELDS = ("shuffle_policy", "template_mode", "gen_prefix_digest")
FIXTURES_REL = "vlm/coords_fixtures/golden_fixtures.json"


class StageRefused(ValueError):
    def __init__(self, code: str, msg: str):
        self.code = code
        super().__init__(f"[{code}] {msg}")


# ---------------------------------------------------------------- 공통
def _abs(p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _fsync_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except FileExistsError:
        raise StageRefused("exists", f"이미 있다 — 덮지 않는다: {path}") from None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


def _fault_guard(root: Path, stage: str, env: Mapping[str, str] | None) -> None:
    """2판 §1-4 의 순서 2 — 이 단계에는 중단 지점이 없다. 두 변수가 없거나, 둘 다 있고 계획 기록과 같으며 이 단계의 중단이어야 한다.
    어긋나면 아무것도 쓰기 전에 거부한다(외부 검토 7)."""
    from vlm.fault import FaultRefused, check_fault_env

    try:
        check_fault_env("rehearsal", run_root=root, stage=stage, tags=[], env=env)
    except FaultRefused as exc:
        raise StageRefused(exc.code, str(exc)) from None


def _config_problems(cfg: Any) -> list[str]:
    return [f"{k} 이 비어 있다" for k in REHEARSAL_NEEDS if getattr(cfg, k) in (None, (), "", {})]


def _require_config(cfg: Any) -> None:
    bad = _config_problems(cfg)
    if bad:
        raise StageRefused("config_incomplete", "리허설이 읽을 본실험 설정이 비어 있다: " + " · ".join(bad))


def purpose_of_root(root: Path) -> str:
    """루트 이름 → 목적. `reh-*` 는 리허설, `fdiag-*` 는 진단이다(`vlm.run_root.ROOT_KIND`). 계획의 `kind` 는 읽는 쪽이 이 목적과 맞댄다."""
    from vlm.run_root import ROOT_KIND

    name = Path(root).name
    for purpose, pat in ROOT_KIND.items():
        if pat.fullmatch(name):
            return purpose
    raise StageRefused("run_root", f"루트 이름 {name!r} 가 리허설(reh-*) · 진단(fdiag-*) 꼴이 아니다")


def _head(checkout: Path) -> str:
    return subprocess.run(["git", "-C", str(checkout), "rev-parse", "refs/heads/main"], capture_output=True,
                          check=True).stdout.decode().strip()


def _rel(p: Path, base: Path) -> str:
    return Path(os.path.abspath(p)).relative_to(Path(os.path.abspath(base))).as_posix()


# ---------------------------------------------------------------- 0 목록
def train_list_bytes(rows: list[dict]) -> bytes:
    """학습 행 목록 파일의 바이트 — 받은 순서(페어 파일의 순서)대로 한 줄에 페어 행 하나(`json.dumps(행, ensure_ascii=False)`), UTF-8 · LF · 끝 개행."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def target_token_counts(rows: list[dict], *, config: Any) -> dict[str, int]:
    """**승인된 실제 구현** — 페어 행의 학습 타깃(`build_target`)을 프로세서의 토크나이저로 센 토큰 수(특수 토큰 없이).
    프로세서는 학습 · 실측과 같은 호출(`AutoProcessor.from_pretrained(모델, revision, **프로세서 인자)`)로 연다 — 모델은 올리지 않는다.
    원본 크기는 이미지 머리만 읽어 낸다. 좌표 규약이 `ABS_ORIG` 가 아니면 리사이즈 크기가 필요해 세지 않는다."""
    from PIL import Image
    from transformers import AutoProcessor

    from vlm.coords import CoordCfg, ImageGeom
    from vlm.pilot_vlm import build_target

    if config.coord_space != "ABS_ORIG":
        raise StageRefused("target_tokens_coord", f"타깃 길이는 ABS_ORIG 에서만 센다: {config.coord_space!r}")
    proc = AutoProcessor.from_pretrained(config.model_id, revision=config.model_revision, **dict(config.processor_kwargs))
    tok = proc.tokenizer
    cfg = CoordCfg(coord_space=config.coord_space)
    out: dict[str, int] = {}
    for r in rows:
        p = Path(r["image_path"])
        with Image.open(p if p.is_absolute() else REPO_ROOT / p) as im:
            w, h = im.size
        out[str(r["image_id"])] = len(tok(build_target(r, ImageGeom(w, h), cfg), add_special_tokens=False)["input_ids"])
    return out


def stage_lists(*, root: Path, plan: Any, config: Any, snapshot: Path, checkout: Path,
                exposure_ledger: Path | None = None, now: datetime | None = None,
                env: Mapping[str, str] | None = None, token_counter: Callable | None = None) -> dict:
    """목록 단계. 쓰는 순서는 노출 원장의 목록 사본 → `planned` 줄 → 루트의 `lists/` 다(2판 §5-4).

    매니페스트는 열을 고르는 읽기 함수로만 읽는다. 스냅샷의 digest 는 설정(닻)과 같아야 한다.
    계획에 `gen.max_target_tokens` 가 값이면(진단) 생성 분할의 페어 행마다 학습 타깃의 토큰 수를 `token_counter`(비면 실제 것)로 센다.
    """
    from data.manifest_view import read_manifest_columns
    from evaluation.eval_list import canonical_bytes, validate_eval_list
    from vlm.exposure import append_row, default_ledger, write_list_copy
    from vlm.rehearsal_lists import ListsRefused, build_lists, read_edges
    from vlm.uni_config import pairs_snapshot_problems

    _fault_guard(Path(root), "lists", env)
    _require_config(config)
    from vlm.rehearsal_lists import diag_hold

    try:
        diag_hold(plan)                                          # 진단 목록은 보류 — 매니페스트를 읽기 전에(결정 21 의 4)
    except ListsRefused as exc:
        raise StageRefused(exc.code, str(exc)) from None
    lists = Path(root) / "lists"
    if lists.exists():
        raise StageRefused("lists_exist", f"목록 폴더가 이미 있다: {lists}")
    edges_spec = plan.require("exclude.near_dup_edges")
    tv = read_manifest_columns(snapshot, TV_COLUMNS, split_filter={"train", "val"})
    ev = read_manifest_columns(snapshot, EVAL_COLUMNS, split_filter={"eval"})
    for df in (tv, ev):
        if df.attrs.get("snapshot_digest") != config.snapshot_digest:
            raise StageRefused("snapshot_not_anchor", "스냅샷의 digest 가 설정의 동결본과 다르다")
    manifest = tv.to_dict("records") + [{**r, "strata_key": None, "client": None} for r in ev.to_dict("records")]
    pairs_path = _abs(config.pairs_path)
    bad = pairs_snapshot_problems(pairs_path, config.pairs_digest)
    if bad:
        raise StageRefused("pairs_snapshot", " · ".join(bad))
    pairs = [r for r in read_jsonl(pairs_path) if r.get("split") in ("train", "val")]
    eraw = _abs(edges_spec["path"]).read_bytes()
    if hashlib.sha256(eraw).hexdigest() != edges_spec["sha256"]:
        raise StageRefused("edges_hash", "근사 중복 간선 파일의 바이트가 계획의 sha256 과 다르다")
    target_tokens = None
    if plan.get("gen.max_target_tokens") is not None:
        counter = token_counter or target_token_counts
        target_tokens = counter([r for r in pairs if r.get("split") == plan.get("gen.split")], config=config)
    try:
        res = build_lists(manifest, pairs, read_edges(eraw), plan, target_tokens=target_tokens)
    except ListsRefused as exc:
        raise StageRefused(exc.code, str(exc)) from None

    run_id = Path(root).name
    ledger = Path(exposure_ledger) if exposure_ledger is not None else default_ledger()
    commit = _head(checkout)
    at = (now or datetime.now().astimezone()).isoformat(timespec="microseconds")
    group_of = {str(r["image_id"]): str(r["group_id"]) for r in manifest}
    raws = {"gen": canonical_bytes(res.gen_ids), "echo": canonical_bytes(res.echo_ids)}
    els = {k: validate_eval_list(v) for k, v in raws.items()}
    for kind, raw in raws.items():                                   # ① 목록 사본과 묶음 곁 파일
        write_list_copy(ledger, run_id, kind, raw)
        groups = {i: group_of[i] for i in els[kind].ids}
        write_list_copy(ledger, run_id, kind, (json.dumps(groups, sort_keys=True) + "\n").encode("utf-8"),
                        suffix=".groups.json")
    for kind in raws:                                                 # ② 노출 원장의 planned 줄
        append_row(ledger, {"event": "planned", "run_id": run_id, "purpose": plan.purpose, "tag": None,
                            "seed_index": int(plan.get("seed_index")), "mode": None, "list_kind": kind,
                            "list_path": (lists / LIST_FILES[kind]).as_posix(),
                            "list_file_sha256": els[kind].file_sha256, "list_set_sha256": els[kind].set_sha256,
                            "n": len(els[kind].ids), "snapshot_digest": config.snapshot_digest, "commit": commit,
                            "at": at})
    files: dict[str, str] = {}
    for kind, raw in raws.items():                                    # ③ 루트의 lists/
        _fsync_new(lists / LIST_FILES[kind], raw)
        files[LIST_FILES[kind]] = hashlib.sha256(raw).hexdigest()
    for tag, rows in res.train_rows.items():
        raw = train_list_bytes(rows)
        name = TRAIN_LIST.format(tag=tag)
        _fsync_new(lists / name, raw)
        files[name] = hashlib.sha256(raw).hexdigest()
    record = {**res.record, "files": files, "edges_file_sha256": edges_spec["sha256"],
              "snapshot_digest": config.snapshot_digest, "plan_sha256": plan.sha256, "commit": commit,
              "before_3c": True}
    _fsync_new(lists / RECORD_FILE, (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=1) + "\n")
               .encode("utf-8"))
    return record


# ---------------------------------------------------------------- 0′ 등록
def local_registration(root: Path):
    """루트의 리허설 등록 · 영수증 · 학습 설정 원문 — 평가 쪽 로더로 읽는다."""
    from evaluation.prereg_unified import load_registration, read_receipt

    d = Path(root) / "registration"
    receipts = sorted(d.glob("rehearsal-*.receipt.json"))
    if len(receipts) != 1:
        raise StageRefused("registration_missing", f"루트의 리허설 영수증이 하나가 아니다: {[p.name for p in receipts]}")
    rp = receipts[0]
    reg_p = rp.with_name(rp.name.replace(".receipt.json", ".json"))
    return load_registration(reg_p.read_bytes()), read_receipt(rp), rp, d / "train_config.json"


def diag_rules_bytes(checkout: Path, rules_rel: str | None) -> bytes:
    """진단 규칙 파일 — **본줄기(`refs/heads/main`)의 커밋에서** 읽고(작업 트리가 아니다) 평가 쪽 판독기로 꼴을 본다.
    평가 쪽 진단 경로는 영수증 커밋에서 같은 경로를 읽어 등록의 digest 와 맞댄다."""
    from evaluation.entry_gate import MAIN_REF, EntryRejected, blob_at
    from evaluation.frame_diag_rules import RulesFileError, load_rules_file

    if not rules_rel:
        raise StageRefused("rules_missing", "진단은 규칙 파일(--rules, 저장소 상대 경로)이 있어야 한다")
    try:
        raw = blob_at(Path(checkout), MAIN_REF, rules_rel)
    except EntryRejected as exc:
        raise StageRefused("rules_not_in_main", f"규칙 파일 {rules_rel} 을 본줄기에서 읽지 못했다: {exc}") from None
    try:
        load_rules_file(raw)
    except RulesFileError as exc:
        raise StageRefused("rules_form", str(exc)) from None
    return raw


def stage_register(*, root: Path, plan: Any, config: Any, checkout: Path, measure_env: Callable,
                   model_loader: Callable | None, generator_loader: Callable, now: datetime | None = None,
                   standin_allowed: bool = False, env: Mapping[str, str] | None = None,
                   rules_rel: str | None = None) -> Path:
    """등록 단계 — 평가 쪽 `UnifiedRegistration` 을 채워 `registration/` 에 쓴다. 목적은 계획의 것(`plan.purpose`)이다.
    리허설은 영수증까지 쓰고 그 경로를 돌려준다. **진단은 등록 파일과 학습 설정 원문만 쓰고 멈춘다**(2판 §3-1 의 흐름 1) —
    둘을 본줄기에 커밋하고 영수증을 적는 것은 그 뒤의 일이다. 돌려주는 값은 등록 파일 경로다.

    | 칸 | 어디서 |
    |---|---|
    | 목적 · 기준 분할 · 동결본 · 목록 해시 넷 | 목적 · `val` · 설정 · 루트의 `lists/` |
    | 프롬프트 · 템플릿 · 모델 · 배치 · 좌표 | 설정(본실험 키) — 프롬프트 해시는 학습 · export 와 같은 `load_prompt` |
    | 생성 접두 · 프로세서 일곱 | 실측 이음새(`measure_env`) |
    | 생성 한도 · 학습량 · 계획 해시 | 계획(`gen_limit` · `amount`) |
    | 디코딩 · 채움 방향 | 모델 생성기가 쓰는 값(`vlm.export_run.DECODING` · `PADDING_SIDE`) |
    | 초기 어댑터 digest · 학습 설정 원문 | 루트의 `init/initial.npz`(모델 적재 1) · 실제로 만든 모델과 최적화기의 다섯 묶음(모델 적재 1) |
    | 진단만 — 학습 행 digest · 진단 규칙 | 루트의 `lists/train_uni_central.jsonl` 의 `image_id` 수열 · 본줄기의 규칙 파일(`--rules`) |
    | 구현 식별자 | 실측과 생성기 적재 이음새 — 두 모드가 같은 적재 함수를 지난다(`vlm.export_generator`) |
    | 에코 관문 두 칸 | 평가 쪽 함수(`unified_coord_cfg_record` · `coord_fixture_digest`)가 낸 값 |

    모델을 두 번 올린다(초기 어댑터 캐시가 있으면 한 번). 초기 어댑터의 캐시 · proof 규칙은 `build_initial_adapter` 의 것이다.
    """
    import gc

    from evaluation.actuals import impl_key
    from evaluation.conformance_v14 import SCORING_CODES
    from evaluation.coord_check import coord_fixture_digest, unified_coord_cfg_record
    from evaluation.eval_list import validate_eval_list
    from evaluation.prereg_unified import UnifiedRegistration, dump_registration
    from vlm import export_run as XR
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.export_measure import MeasureSpec
    from vlm.export_preflight import TRAIN_CONFIG_SUFFIX
    from vlm.init_adapter import build_initial_adapter, init_adapter_digest
    from vlm.pilot_vlm import (
        TARGET_CONTRACT_SHA256,
        _config_blocks,
        canonical_json,
        load_prompt,
        make_optimizer,
        processor_config_sha256,
        resolve_model_loader,
        template_mode_of,
        train_rows_digest,
    )
    from vlm.seams import impl_id
    from vlm.train_cell import train_config_object

    _fault_guard(Path(root), "register", env)
    _require_config(config)
    root = Path(root)
    purpose = plan.purpose
    diag = purpose == "frame_diag"
    if not diag and rules_rel is not None:
        raise StageRefused("rules_not_for_purpose", "규칙 파일은 진단 등록만 받는다")
    rules_raw = diag_rules_bytes(checkout, rules_rel) if diag else None
    d = root / "registration"
    if d.exists():
        raise StageRefused("registration_exists", f"등록 폴더가 이미 있다: {d}")
    rows_digest = None
    if diag:                                                     # 학습 행 — 목록 단계의 기록과 같은 바이트의 수열
        name = TRAIN_LIST.format(tag=CENTRAL_TAG)
        raw = read_under(root / "lists" / name, root, code="train_list_path")
        rec = json.loads(read_under(root / "lists" / RECORD_FILE, root, code="train_list_path").decode("utf-8"))
        if hashlib.sha256(raw).hexdigest() != (rec.get("files") or {}).get(name):
            raise StageRefused("train_list_hash", f"학습 행 목록 {name} 의 바이트가 목록 단계의 기록과 다르다")
        rows_digest = train_rows_digest([json.loads(x) for x in raw.decode("utf-8").splitlines() if x])
    raws = {k: read_under(root / "lists" / f, root, code="list_path") for k, f in LIST_FILES.items()}
    el = {k: validate_eval_list(v) for k, v in raws.items()}
    now = now or datetime.now().astimezone()
    n, r, e = plan.amount
    idx = int(plan.get("seed_index"))
    if not 1 <= idx <= len(config.seeds):
        raise StageRefused("seed_index", f"계획의 seed_index {idx} 가 시드표 1..{len(config.seeds)} 밖이다")
    seed_value = int(config.seeds[idx - 1])
    prompt_text, prompt_sha = load_prompt(_abs(config.prompt_path))
    tmpl = dict(config.chat_template_kwargs)
    measured = measure_env(MeasureSpec(model_id=config.model_id, model_revision=config.model_revision,
                                       processor_kwargs=dict(config.processor_kwargs), prompt_text=prompt_text,
                                       chat_template_kwargs=tmpl))
    loader, _ = resolve_model_loader(model_loader, purpose=purpose, standin_allowed=standin_allowed)
    arrays, _, _ = build_initial_adapter(model_id=config.model_id, seed=seed_value, cache_path=root / "init" / "initial.npz",
                                         revision=config.model_revision, purpose=purpose, model_loader=model_loader,
                                         standin_allowed=standin_allowed)
    init_digest = init_adapter_digest(arrays)
    model, proc = loader(config.model_id, init_seed=seed_value, revision=config.model_revision,
                         processor_kwargs=dict(config.processor_kwargs))
    blocks = _config_blocks(model, make_optimizer(model), supervised_logits_only=True)
    proc_sha = processor_config_sha256(proc)
    del model
    gc.collect()
    if proc_sha != measured["processor_config_sha256"]:
        raise StageRefused("processor_mismatch", "학습 쪽 프로세서와 실측한 프로세서의 설정 해시가 다르다 — 학습과 생성이 다른 입력 크기를 본다")
    chash = coord_cfg_hash(CoordCfg(coord_space=config.coord_space))
    metrics = {"processor_config_sha256": proc_sha, "config_blocks": blocks, "prompt_sha256": prompt_sha,
               "template_mode": template_mode_of(tmpl), "coord_cfg_hash": chash,
               "target_contract_sha256": TARGET_CONTRACT_SHA256}
    tc_text = canonical_json(train_config_object(
        model_id=config.model_id, model_revision=config.model_revision, pairs_digest=config.pairs_digest,
        chat_template_kwargs=tmpl, coord_space=config.coord_space, num_rounds=r, local_epochs=e, total_epochs=n,
        prompt_sha256=prompt_sha, init_digest=init_digest, metrics=metrics))

    reg = UnifiedRegistration()
    g = reg.generation
    g.purpose, g.list_split, g.snapshot_digest = purpose, "val", config.snapshot_digest
    g.eval_list_file_sha256, g.eval_list_set_sha256 = el["gen"].file_sha256, el["gen"].set_sha256
    g.echo_list_file_sha256, g.echo_list_set_sha256 = el["echo"].file_sha256, el["echo"].set_sha256
    g.prompt_sha256, g.chat_template_kwargs = prompt_sha, tmpl
    g.gen_prefix_sha256 = measured["gen_prefix_sha256"]
    g.max_new_tokens = int(plan.require("gen_limit.max_new_tokens"))
    g.decoding, g.batch_size, g.padding_side = dict(XR.DECODING), int(config.batch_size), XR.PADDING_SIDE
    for k in ("processor_config_sha256", "processor_min_pixels", "processor_max_pixels", "patch_size", "merge_size",
              "transformers_version"):
        setattr(g, k, measured[k])
    g.coord_space, g.coord_cfg_hash = config.coord_space, chash
    g.base_model_id, g.base_model_revision = config.model_id, config.model_revision
    g.init_adapter_digest, g.start_checkpoint_sha256 = init_digest, None
    g.train_config_sha256 = hashlib.sha256(tc_text.encode("utf-8")).hexdigest()
    g.budget_n, g.budget_r, g.budget_e = n, r, e
    g.impl_ids = tuple(sorted(
        impl_key(impl_id(fn, seam=seam, approved=any(fn is a for a in approved)))
        for fn, seam, approved in ((measure_env, "export_measure", XR.APPROVED_MEASURES),
                                   (generator_loader, "export_generator",
                                    XR.APPROVED_GENERATOR_LOADERS["model"] + XR.APPROVED_GENERATOR_LOADERS["echo"]))))
    from vlm.export_generator import inner_impl_ids
    from vlm.export_generator import load_generator as generator_real

    if generator_loader is generator_real:                       # 실제 적재 함수면 안쪽 구현 셋도 등록한다(export 와 같은 꼴)
        g.impl_ids = tuple(sorted({*g.impl_ids, *(impl_key(v) for v in inner_impl_ids(
            purpose=purpose, standin_allowed=True).values())}))
    g.train_rows_digest, g.plan_sha256 = rows_digest, plan.sha256
    if diag:
        g.frame_diag_rules_sha256, g.frame_diag_rules_path = hashlib.sha256(rules_raw).hexdigest(), rules_rel
    fd = coord_fixture_digest((REPO_ROOT / FIXTURES_REL).read_bytes())
    reg.canary.coord_cfg = unified_coord_cfg_record(
        chash, fixture_digest=fd, registered_on=now.date().isoformat(),
        coords_source=(REPO_ROOT / "vlm/coords.py").read_bytes(),
        hash_record=(REPO_ROOT / "vlm/coords_fixtures/HASH_RECORD.yaml").read_bytes())
    reg.canary.coord_fixture_sha256 = fd
    reg.metric.class_codes, reg.metric.coupling_rule = tuple(SCORING_CODES), "decoupled_v2"
    reg.seeds = {str(idx): seed_value}
    reg.require_complete(side="generation")

    name = f"{purpose}-{now.strftime('%Y%m%d')}-1"
    raw = dump_registration(reg)
    _fsync_new(d / f"{name}.json", raw)
    if diag:                                                     # 커밋할 꼴 — 등록 파일과 같은 줄기 이름 + 꼬리(평가 쪽 README)
        _fsync_new(d / f"{name}{TRAIN_CONFIG_SUFFIX}", tc_text.encode("utf-8"))
        return d / f"{name}.json"
    _fsync_new(d / "train_config.json", tc_text.encode("utf-8"))
    receipt = {"generation_sha256": reg.generation_sha256(), "scoring_sha256": reg.scoring_sha256(),
               "registered_at": now.isoformat(timespec="microseconds"), "main_commit": _head(checkout),
               "kind": "rehearsal", "registration_path": _rel(d / f"{name}.json", checkout),
               "registration_file_sha256": hashlib.sha256(raw).hexdigest()}
    rp = d / f"{name}.receipt.json"
    _fsync_new(rp, (json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return rp


# ---------------------------------------------------------------- 1 · 2 · 2′ · 2″ 학습
def rehearsal_spec(*, root: Path, plan: Any, config: Any, stamp: str | None = None, negative: str | None = None,
                   non_main_parent: Path | None = None, standin_allowed: bool = False,
                   config_path: Path | None = None):
    """리허설 칸의 설정 — 값은 본실험 설정, 학습량은 계획, 경로는 루트 아래다. `negative` 는 2′ 의 새 폴더(`negative/<경우>/`)."""
    from vlm.train_cell import UniRunSpec

    _require_config(config)
    root = Path(root)
    n, r, e = plan.amount
    idx = int(plan.get("seed_index"))
    if not 1 <= idx <= len(config.seeds):
        raise StageRefused("seed_index", f"계획의 seed_index {idx} 가 시드표 1..{len(config.seeds)} 밖이다")
    if negative is not None and negative not in NEGATIVE_CASES:
        raise StageRefused("negative_case", f"모르는 구판 경우: {negative!r}")
    base = root / "negative" / negative if negative else root
    return UniRunSpec(
        purpose=plan.purpose, run_stamp=stamp or f"{root.name}_s{idx}", seed_index=idx,
        seed_value=int(config.seeds[idx - 1]), snapshot_digest=str(config.snapshot_digest),
        model_id=str(config.model_id), model_revision=config.model_revision, pairs_path=str(config.pairs_path),
        pairs_digest=str(config.pairs_digest), chat_template_kwargs=dict(config.chat_template_kwargs),
        num_rounds=r, local_epochs=e, total_epochs=n, train_root=base / "train",
        init_adapter_path=root / "init" / "initial.npz", resume_root=base / "resume",
        prompt_path=str(_abs(config.prompt_path)), coord_space=str(config.coord_space),
        processor_kwargs=dict(config.processor_kwargs), plan_sha256=plan.sha256, standin_allowed=standin_allowed,
        expected_supervised_tokens=None, config_path=config_path, run_root=root, non_main_parent=non_main_parent)


def check_train_list(root: Path, tag: str, *, config: Any, snapshot: Path) -> list[dict]:
    """학습 행 목록을 **학습 전에** 허용된 원본 행에 다시 묶는다(외부 검토 1). 돌려주는 값은 검사를 지난 행이다.

    0. 읽는 두 파일(학습 행 목록 · `lists_record.json`)의 최종 경로가 이 루트 아래이고 이름이 하나인 보통 파일이다(`_read_under`) —
       루트 안의 `lists/` 에 다른 루트의 정상 파일로 가는 링크 · 정션을 두면 해시 · 원본 행 대조를 다 지난다.
    1. 목록 파일의 sha256 이 목록 단계의 기록(`lists_record.json` 의 `files`)과 같다.
    2. 행마다 `image_id` 가 동결 매니페스트(열을 고르는 판독기, digest = 설정)의 **train** 행이고, 로컬 칸이면 그 참여자의 행이다.
    3. 행이 본실험 페어(스냅샷 계약을 지난)의 같은 id 의 train 행과 **통째로 같다** — `image_path` 를 바꿔 끼운 행을 받지 않는다.

    어긋나면 모델을 올리기 전에 거부한다. 목록 단계가 만든 목록만 이 길을 지난다.
    """
    from data.manifest_view import read_manifest_columns
    from vlm.uni_config import pairs_snapshot_problems

    lists = Path(root) / "lists"
    name = TRAIN_LIST.format(tag=tag)
    try:
        raw = _read_under(lists / name, Path(root))
        rec_raw = _read_under(lists / RECORD_FILE, Path(root))
    except ValueError as exc:
        raise StageRefused("train_list_path", str(exc)) from None
    rec = json.loads(rec_raw.decode("utf-8"))
    if hashlib.sha256(raw).hexdigest() != (rec.get("files") or {}).get(name):
        raise StageRefused("train_list_hash", f"학습 행 목록 {name} 의 바이트가 목록 단계의 기록과 다르다")
    tv = read_manifest_columns(snapshot, ("image_id", "split", "client"), split_filter={"train"})
    if tv.attrs.get("snapshot_digest") != config.snapshot_digest:
        raise StageRefused("snapshot_not_anchor", "스냅샷의 digest 가 설정의 동결본과 다르다")
    client_of = {str(r["image_id"]): str(r["client"]) for r in tv.to_dict("records")}
    pairs_path = _abs(config.pairs_path)
    bad = pairs_snapshot_problems(pairs_path, config.pairs_digest)
    if bad:
        raise StageRefused("pairs_snapshot", " · ".join(bad))
    original = {str(r["image_id"]): r for r in read_jsonl(pairs_path) if r.get("split") == "train"}
    rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x]
    want_client = LOCAL_TAGS.get(tag)
    for r in rows:
        iid = str(r.get("image_id"))
        if iid not in client_of:
            raise StageRefused("train_list_split", "학습 행 목록에 동결 매니페스트의 train 행이 아닌 id 가 있다")
        if want_client is not None and client_of[iid] != want_client:
            raise StageRefused("train_list_client", f"학습 행 목록에 참여자 {want_client} 의 행이 아닌 id 가 있다")
        if r != original.get(iid):
            raise StageRefused("train_list_row", "학습 행이 본실험 페어의 원본 행과 다르다")
    return rows


def diag_registration(root: Path, receipt_path: Path, *, checkout: Path) -> tuple[Any, Any, bool]:
    """진단 — 영수증(`kind="frame_diag"`)이 가리키는 **커밋의 등록**을 평가 쪽 관문 함수로 만들고, 루트의 준비 산출(`fdiag prepare`)과
    바이트를 맞댄다. 돌려주는 값은 `(등록, 영수증, 커밋된 영수증인가)` 다.

    1. 영수증은 이 루트의 `registration/` 아래에 있다.
    2. 등록은 영수증 커밋에서 읽는다(조상 검사 · 파일 해시 · 생성 지문 — `load_registration_for` · `require_receipt`).
    3. 루트의 `registration/<줄기>.json` · `<줄기>.train_config.json` 이 그 커밋의 등록 · 학습 설정 원문과 같은 바이트다 —
       prepare 뒤에 루트의 등록이 바뀌었으면 거부한다(2판 §3-1 의 흐름 3).
    커밋된 영수증(실제 본줄기의 조상)이면 대역을 받지 않는다(2판 §1-4 의 순서 3) — 부르는 쪽이 셋째 값으로 가른다.
    """
    from evaluation.entry_gate import (
        EntryRejected,
        blob_at,
        is_ancestor,
        load_registration_for,
        main_checkout,
    )
    from evaluation.prereg_unified import (
        ReceiptMissing,
        RegistrationIncomplete,
        RegistrationInvalid,
        read_receipt,
        require_receipt,
    )
    from vlm.export_preflight import TRAIN_CONFIG_SUFFIX, echo_train_config_rel

    root, rp = Path(root), Path(receipt_path)
    if not rp.resolve().is_relative_to((root / "registration").resolve()):
        raise StageRefused("receipt_outside_root", f"영수증 {rp} 가 이 루트의 registration/ 아래가 아니다")
    try:
        rc = read_receipt(rp)
        if rc.kind != "frame_diag":
            raise StageRefused("receipt_kind", f"진단 루트의 영수증 종류가 frame_diag 가 아니다: {rc.kind!r}")
        reg = load_registration_for(rc, repo=checkout)
        require_receipt(reg, rc, side="generation")
        reg_blob = blob_at(Path(checkout), rc.main_commit, rc.registration_path)
        tc_blob = blob_at(Path(checkout), rc.main_commit, echo_train_config_rel(rc.registration_path))
    except (EntryRejected, ReceiptMissing, RegistrationIncomplete, RegistrationInvalid) as exc:
        raise StageRefused("receipt", str(exc)) from None
    stem = rc.registration_path.rsplit("/", 1)[-1]
    if not stem.endswith(".json"):
        raise StageRefused("receipt", f"등록 경로가 .json 이 아니다: {rc.registration_path}")
    stem = stem[: -len(".json")]
    local = root / "registration" / f"{stem}.json"
    local_tc = root / "registration" / f"{stem}{TRAIN_CONFIG_SUFFIX}"
    if not local.is_file() or read_under(local, root, code="registration_path") != reg_blob:
        raise StageRefused("registration_vs_root", "루트의 등록 파일이 영수증 커밋의 등록과 다르다 — prepare 뒤에 바뀌었다")
    if not local_tc.is_file() or read_under(local_tc, root, code="registration_path") != tc_blob:
        raise StageRefused("train_config_vs_root", "루트의 학습 설정 원문이 영수증 커밋의 것과 다르다")
    return reg, rc, is_ancestor(main_checkout(), rc.main_commit)


def _against_registration(root: Path, plan: Any, spec: Any, *, require_init: bool = False, reg: Any = None) -> Any:
    """학습 전 — 등록의 학습량 · 계획 바이트 · 초기 어댑터 digest 와 맞댄다. 돌려주는 값은 생성 쪽 등록이다.
    `reg` 가 비면 루트의 리허설 등록을 읽는다(진단은 영수증 커밋의 등록을 준다)."""
    from vlm.init_adapter import init_adapter_digest, read_initial_cache

    if reg is None:
        reg, _, _, _ = local_registration(Path(root))
    g = reg.generation
    if (g.budget_n, g.budget_r, g.budget_e) != plan.amount:
        raise StageRefused("budget_vs_registration", f"계획의 학습량 {plan.amount} 이 등록 {(g.budget_n, g.budget_r, g.budget_e)} 과 다르다")
    if g.plan_sha256 != plan.sha256:
        raise StageRefused("plan_vs_registration", "계획 파일의 바이트가 등록의 plan_sha256 과 다르다")
    init_p = Path(spec.init_adapter_path)
    if init_p.exists():
        arrays, _ = read_initial_cache(init_p, model_id=spec.model_id, seed=spec.seed_value,
                                       revision=spec.model_revision, purpose=spec.purpose)
        if init_adapter_digest(arrays) != g.init_adapter_digest:
            raise StageRefused("init_vs_registration", "초기 어댑터의 digest 가 등록과 다르다")
    elif require_init:
        raise StageRefused("init_adapter_missing", f"등록 단계가 만든 공통 초기 어댑터가 없다: {init_p}")
    return g


def stage_train(*, root: Path, plan: Any, config: Any, tag: str, snapshot: Path, stamp: str | None = None,
                negative: str | None = None, model_loader: Callable | None = None, standin_allowed: bool = False,
                non_main_parent: Path | None = None, config_path: Path | None = None, receipt: Path | None = None,
                checkout: Path | None = None) -> Any:
    """한 칸의 학습. 학습 행은 루트의 `lists/train_<tag>.jsonl`. **모델을 올리기 전에** 등록의 학습량 · 초기 어댑터 digest 와 맞대고,
    학습이 낸 학습 설정 원문의 해시를 등록과 맞댄다(2판 §1-4 의 진입점 표 — 리허설은 등록 파일과 언제나 맞댄다).
    연합 칸은 `stage_train_fed` 다.

    **진단**(계획의 목적 `frame_diag`)은 중앙 칸 하나다. 등록은 영수증(`receipt`) 커밋의 것이고(`diag_registration`), 학습 행 목록의
    `image_id` 수열이 등록의 `train_rows_digest` 와 같아야 한다. 커밋된 영수증이면 대역을 받지 않는다. 스탬프 · 구판 경우를 받지 않는다."""
    from vlm.pilot_vlm import train_rows_digest
    from vlm.train_cell import CellRejected, run_uni_central_cell, run_uni_local_cell

    if tag not in (CENTRAL_TAG, *LOCAL_TAGS):
        raise StageRefused("tag", f"이 단계의 학습 칸은 로컬 · 중앙이다(연합은 stage_train_fed): {tag!r}")
    reg = None
    diag = plan.purpose == "frame_diag"
    if diag:
        if tag != CENTRAL_TAG:
            raise StageRefused("tag", f"진단은 중앙 칸 하나다: {tag!r}")
        if stamp is not None or negative is not None:
            raise StageRefused("fdiag_args", "진단 학습은 스탬프 · 구판 경우를 받지 않는다")
        if receipt is None:
            raise StageRefused("receipt_missing", "진단 학습은 영수증(--receipt)이 있어야 한다")
        reg, _rc, committed = diag_registration(Path(root), Path(receipt), checkout=Path(checkout))
        if committed:
            standin_allowed = False                              # 커밋된 진단 영수증으로는 대역을 받지 않는다
    elif receipt is not None:
        raise StageRefused("receipt_not_for_purpose", "리허설 학습은 영수증 인자를 받지 않는다 — 루트의 등록과 맞댄다")
    spec = rehearsal_spec(root=root, plan=plan, config=config, stamp=stamp, negative=negative,
                          non_main_parent=non_main_parent, standin_allowed=standin_allowed, config_path=config_path)
    g = _against_registration(Path(root), plan, spec, reg=reg)
    rows = check_train_list(Path(root), tag, config=config, snapshot=Path(snapshot))
    if diag and train_rows_digest(rows) != g.train_rows_digest:
        raise StageRefused("train_rows_vs_registration", "학습 행 목록의 수열이 등록의 train_rows_digest 와 다르다")
    try:
        if tag == CENTRAL_TAG:
            res = run_uni_central_cell(spec=spec, model_loader=model_loader, rows=rows)
        else:
            c = LOCAL_TAGS[tag]
            res = run_uni_local_cell([c], spec=spec, model_loader=model_loader, rows_by_client={c: rows})[c]
    except CellRejected as exc:
        raise StageRefused(exc.code, str(exc)) from None
    meta = json.loads((Path(res.cell_dir) / "adapter_last.meta.json").read_text(encoding="utf-8"))
    if hashlib.sha256(str(meta.get("train_config", "")).encode("utf-8")).hexdigest() != g.train_config_sha256:
        raise StageRefused("train_config_vs_registration", "학습이 낸 학습 설정 원문의 해시가 등록과 다르다")
    return res


# ---------------------------------------------------------------- 줄 B 의 연합 칸
FED_TAG = "uni_fed"
FED_CLIENT_TAGS = ("C1", "C2", "C3")


def _read_under(path: Path, root: Path) -> bytes:
    """`path` 의 바이트 — 최종 경로(링크 · 정션을 푼 것)가 `root` 아래이고 다른 이름의 하드 링크가 없는 보통 파일일 때만.
    하드 링크는 경로를 풀어도 드러나지 않는다 — 목록 단계가 만든 파일은 이름이 하나다(외부 검토 회신 `rehearsal_blockers2` 의 3)."""
    real = Path(path).resolve()
    if not real.is_relative_to(Path(root).resolve()):
        raise ValueError(f"{Path(path).name} 의 최종 경로 {real} 가 실행 루트 {root} 아래가 아니다 — 읽지 않는다")
    st = os.lstat(real)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise ValueError(f"{Path(path).name} 이 이름 하나의 보통 파일이 아니다(링크 수 {st.st_nlink}) — 읽지 않는다")
    return real.read_bytes()


def read_under(path: Path, root: Path, *, code: str) -> bytes:
    """`_read_under` 를 단계의 거부(`StageRefused(code)`)로 — 오케스트레이터 · 단계가 루트의 파일을 읽는 자리에 쓴다."""
    try:
        return _read_under(Path(path), Path(root))
    except ValueError as exc:
        raise StageRefused(code, str(exc)) from None


def fed_client_rows(lists_dir: str | Path, *, client: str, pairs_path: str | Path | None,
                    run_root: str | Path | None = None) -> list[dict]:
    """연합 클라이언트의 학습 행 — 목록 단계가 만든 그 참여자의 목록(로컬 칸과 같은 행, 2판 §4-1 의 `fed_rows: same_as_local`).

    오케스트레이터가 연합 단계 앞에서 `check_train_list` 로 동결 매니페스트와 맞댄 목록이다. 클라이언트는 받은 쪽에서 다시 본다 —
    목록 바이트가 목록 단계의 기록과 같고, 행마다 페어의 **그 참여자의 train 행**과 통째로 같다. 어긋나면 학습하지 않는다."""
    from vlm.pilot_vlm import load_pairs

    lists = Path(lists_dir)
    # 입력 경로 격리(2판 §1-4) — 목록 폴더가 서버가 내려보낸 실행 루트 아래여야 한다. 다른 루트의 정상 목록을 읽지 않는다
    if run_root is None or not lists.resolve().is_relative_to(Path(run_root).resolve()):
        raise ValueError(f"학습 행 목록 폴더 {lists} 가 실행 루트 {run_root} 아래가 아니다 — 읽지 않는다")
    name = TRAIN_LIST.format(tag=f"uni_local_{client}")
    # 읽을 두 파일도 각각의 최종 경로가 루트 아래인 보통 파일이어야 한다 — 폴더 안의 파일 링크가 다른 루트의 정상 파일을 가리키면 폴더 검사를 지난다
    raw = _read_under(lists / name, Path(run_root))
    rec = json.loads(_read_under(lists / RECORD_FILE, Path(run_root)).decode("utf-8"))
    if hashlib.sha256(raw).hexdigest() != (rec.get("files") or {}).get(name):
        raise ValueError(f"학습 행 목록 {name} 의 바이트가 목록 단계의 기록과 다르다 — 학습하지 않는다")
    original = {str(r["image_id"]): r for r in load_pairs("train", client=client, pairs_path=pairs_path)}
    rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x]
    if not rows:
        raise ValueError(f"학습 행 목록 {name} 이 비었다")
    for r in rows:
        if r != original.get(str(r.get("image_id"))):
            raise ValueError(f"학습 행이 페어의 참여자 {client} 의 train 행과 다르다 — 학습하지 않는다")
    return rows


def fed_run_items(*, root: Path, plan_path: Path, spec: Any, config: Any) -> dict[str, Any]:
    """연합 단계의 실행 설정(파이썬 값). 키는 모두 `pyproject.toml` 에 선언된 것이다. 본실험 실행기(`scripts/main_uni.py` 의
    `fed_run_config`)와 같은 키에 목적 · 계획 · 루트 · 학습 행 목록 · 초기 어댑터를 더한다. 학습량은 계획의 것이다."""
    from vlm.pilot_vlm import canonical_json

    root = Path(os.path.abspath(root))
    return {
        "cell": FED_TAG, "num-server-rounds": spec.num_rounds, "local-epochs": spec.local_epochs,
        "total-epochs": spec.total_epochs, "num-clients": len(FED_CLIENT_TAGS), "base-seed": spec.seed_value,
        "run-stamp": spec.run_stamp, "split-hash": spec.snapshot_digest, "model": spec.model_id,
        "project": root.as_posix(), "client-tags": ",".join(FED_CLIENT_TAGS), "resume-root": "",
        "uni-model": spec.model_id, "uni-model-revision": str(spec.model_revision or ""),
        "uni-pairs": Path(_abs(config.pairs_path)).as_posix(), "uni-pairs-digest": spec.pairs_digest,
        "uni-prompt": Path(spec.prompt_path).as_posix(),
        "uni-chat-template-kwargs": canonical_json(dict(spec.chat_template_kwargs)),
        "uni-coord-space": spec.coord_space, "uni-seed-index": str(spec.seed_index),
        "uni-processor-kwargs": canonical_json(dict(spec.processor_kwargs)), "uni-expected-supervised-tokens": "",
        "purpose": "rehearsal", "plan": Path(os.path.abspath(plan_path)).as_posix(), "rehearsal-root": root.as_posix(),
        "uni-train-root": (root / "train").as_posix(), "uni-train-lists": (root / "lists").as_posix(),
        "uni-init-adapter": (root / "init" / "initial.npz").as_posix(),
    }


def fed_run_config_text(items: dict[str, Any]) -> str:
    """`flwr run --run-config` 문자열 — 정수는 그대로, 나머지는 본실험 실행기와 같은 따옴표 규칙(`scripts/main_uni.py` 의 `_toml_str`)."""
    from scripts.main_uni import _toml_str

    return " ".join(f"{k}={v if type(v) is int else _toml_str(str(v))}" for k, v in items.items())


def flwr_items_runner(items: dict[str, Any], log_path: Path) -> None:
    """실제 실행 — 본실험 · 검출과 같은 flwr 실행 함수(`scripts/main_det.py` 의 `_flwr_run`, 2판 §1-3 의 연합 행 ①)."""
    from scripts.main_det import _flwr_run

    _flwr_run(fed_run_config_text(items), Path(log_path))


def stage_train_fed(*, root: Path, plan: Any, plan_path: Path, config: Any, snapshot: Path,
                    runner: Callable | None = None, procs: Callable | None = None, stop: Callable | None = None,
                    standin_allowed: bool = False, non_main_parent: Path | None = None,
                    config_path: Path | None = None) -> Any:
    """줄 B 의 연합 칸 — 본실험과 같은 **`flwr run` 경로**(2판 §1-1 · §1-3 의 연합 행). 중단을 넣지 않는다.

    순서: 등록 대조(학습량 · 계획 바이트 · 초기 어댑터) → 참여자 셋의 학습 행 목록을 동결 매니페스트 · 페어와 맞댐(`check_train_list`) →
    **상주 SuperLink · 시뮬레이션 프로세스를 내리고 남은 것이 없는지 본다** → 실행 → 다시 내리고 남은 것이 없는지 본다 → 출력이 닫혔는가 ·
    학습 설정 원문이 등록과 같은가. 출력은 `<루트>/train/uni_fed_s<n>/`(2판 §13-4 — 로컬 · 중앙과 같은 꼴),
    초기 어댑터는 등록 단계가 만든 `<루트>/init/initial.npz` 를 서버에 넘긴다(`uni-init-adapter`)."""
    from scripts.main_uni import fed_state
    from vlm.rehearsal_orch import flwr_processes, stop_processes

    procs = procs or flwr_processes
    stop = stop or stop_processes
    runner = runner or flwr_items_runner
    root = Path(root)
    spec = rehearsal_spec(root=root, plan=plan, config=config, stamp=None, negative=None,
                          non_main_parent=non_main_parent, standin_allowed=standin_allowed, config_path=config_path)
    g = _against_registration(root, plan, spec, require_init=True)
    for tag in LOCAL_TAGS:                                       # 연합 클라이언트 = 로컬과 같은 행
        check_train_list(root, tag, config=config, snapshot=Path(snapshot))
    out = root / "train" / f"{FED_TAG}_s{spec.seed_index}"
    state = fed_state(out)
    if state == "closed":
        return _fed_result(out, g, status="skipped")
    if state != "none":
        raise StageRefused("fed_not_fresh", f"연합 출력이 {state} 이다 — 연합은 재개가 없다: {out}")

    def clear(when: str) -> None:
        left = procs()
        unclear = [p for p in left if p.get("kind") != "target"]
        if unclear:                                              # 신원이 불명확하면 죽이지 않고 시작을 거부한다(외부 검토 §50 요청 2)
            raise StageRefused(f"flwr_identity_unclear_{when}",
                               f"이 실행의 것인지 가릴 수 없는 flwr 프로세스가 있다({when}) — 내리지 않고 시작하지 않는다: "
                               + ", ".join(f"{p['pid']}:{p['name']}" for p in unclear[:5]))
        if left:
            try:
                stop(left)
            except Exception as exc:  # noqa: BLE001 - 권한 거부 따위 — 정리 실패로 알린다(부모가 다시 정리한다)
                raise StageRefused(f"flwr_stop_failed_{when}", f"상주 프로세스를 내리지 못했다({when}): {exc}") from None
            left = procs()
        if left:
            raise StageRefused(f"flwr_residual_{when}",
                               f"상주 SuperLink · 시뮬레이션 프로세스가 남았다({when}) — 다음 단계를 시작하지 않는다: "
                               + ", ".join(f"{p['pid']}:{p['name']}" for p in left[:5]))

    clear("before")
    items = fed_run_items(root=root, plan_path=Path(plan_path), spec=spec, config=config)
    try:
        runner(items, root / "logs" / "flwr_uni_fed.log")
    except BaseException:
        try:                                                     # 실행이 실패했으면 정리 실패가 그 실패를 덮지 않게 한다
            clear("after")
        except Exception:  # noqa: BLE001, S110 - 부모가 다시 정리하고 기록한다
            pass
        raise
    clear("after")
    if fed_state(out) != "closed":
        raise StageRefused("fed_not_closed", f"연합이 끝났는데 출력이 닫히지 않았다: {out}")
    return _fed_result(out, g, status="trained")


def _fed_result(out: Path, g: Any, *, status: str) -> Any:
    from types import SimpleNamespace

    meta = json.loads((out / "adapter_last.meta.json").read_text(encoding="utf-8"))
    if hashlib.sha256(str(meta.get("train_config", "")).encode("utf-8")).hexdigest() != g.train_config_sha256:
        raise StageRefused("train_config_vs_registration", "연합이 낸 학습 설정 원문의 해시가 등록과 다르다")
    return SimpleNamespace(status=status, adapter_step=meta.get("adapter_step"), cell_dir=out)


# ---------------------------------------------------------------- 2′ 합성 구판
def synth_negatives(root: Path, *, tag: str, seed_index: int, source: Path) -> dict[str, dict[str, Any]]:
    """앞 시도가 남긴 재개 체크포인트 하나에서 합성 구판 넷(2판 §3 ①)을 `negative/<경우>/resume/<tag>_s<n>/` 에 만든다.

    ㉮ 정책 셋을 빈 값으로 · ㉯ 신원 키 하나를 뺀다 · ㉰ 모르는 신원 키를 더한다 · ㉱ 형식 문자열을 바꾼다. 돌려주는 값은 경우 → 경로 · sha256 · 바꾼 것.
    합성은 지금의 신원 클래스로 저장된 파일에서 만든다. 원본은 고치지 않는다.
    """
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    out: dict[str, dict[str, Any]] = {}
    for case in NEGATIVE_CASES:
        p = copy.deepcopy(payload)
        ident = p["identity"]
        if case == "policy_blank":
            for f in POLICY_FIELDS:
                ident[f] = ""
            changed = list(POLICY_FIELDS)
        elif case == "identity_key_missing":
            key = "processor_config_sha256"
            ident.pop(key)
            changed = [key]
        elif case == "identity_key_extra":
            ident["unknown_identity_key"] = "x"
            changed = ["unknown_identity_key"]
        else:
            p["format"] = str(p.get("format")) + "-old"
            changed = ["format"]
        dst = Path(root) / "negative" / case / "resume" / f"{tag}_s{seed_index}" / Path(source).name
        if dst.exists():
            raise StageRefused("exists", f"합성 구판이 이미 있다: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        torch.save(p, tmp)
        os.rename(tmp, dst)
        out[case] = {"path": dst, "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "changed": changed}
    return out


def plan_faults(plan: Any) -> Mapping[str, Any]:
    return plan.get("faults") or {}
