"""통합형 로컬·중앙 칸 실행 모듈 — 미니스펙 12번 §1-2, 리허설 2판 §2-6 · §2-7.

`scripts/pilot_c.py` 의 ⑥ 은 파일럿 상수에 묶여 있고 저장이 `np.savez` 직행이다. 검출의 `run_local_cell`
이 갖춘 것(선행 산출물 대조 · 내구 저장 · 신원 블록 · 멱등)을 통합형에 맞춰 여기 둔다. `pilot_c.py` 는
고치지 않았다 — 다만 그 ⑦(통합·연합) 재현은 연합 서버가 목적과 필수 키를 먼저 보게 된 뒤로 시작 전에 거부된다
(목적을 넘기지 않아 본실험으로 읽히고, 인프로세스 경로는 본실험을 받지 않는다). 파일럿 결과는 보존된 산출물로만 인용한다.

## 한 칸이 도는 순서

1. **이음새** — 모델 적재기를 부르기 전에 목적과 맞댄다(`vlm/seams.py`). 본실험은 대역을 받지 않는다.
2. **목적의 입력** — 본실험은 계획 파일도 학습 행 목록도 받지 않는다. 리허설·진단은 계획 해시가 있어야 한다.
3. **초기 어댑터** — 캐시를 읽기만 한다. 없으면 멈춘다(만드는 것은 실행기의 `initadapter` 단계다).
4. **멱등 판정** — 어댑터 파일이 있으면 meta 의 신원을 **먼저** 맞댄다. 같고 원장이 닫혔으면 건너뛰고,
   같은데 열려 있으면 멈추며 `close-ledger` 를 안내하고, 다르면 멈춘다. 모두 모델 적재 전이다.
5. **원장** — 모델마다 파일 하나(`<tag>_s<n>/train_ledger.csv`). 신원이 다른 원장이면 열지 않는다.
6. **학습** — `train_rounds` 를 `R=1, E=N` 으로 부른다. 재개 신원을 지나면 이음매 행을 쓰고, epoch 마다 행을 쓴다.
7. **대조** — 주입이 공통 초기 어댑터와 같은지, 실제 epoch · 갱신 수가 예산과 같은지 본다.
8. **저장** — tmp 저장 → 재로드 대조 → sha256 → meta(LF) → `os.replace`. 어댑터 파일이 완료 표지다.
9. **끝 행** — 저장이 끝난 뒤 원장을 닫는다. 그 앞에서 죽으면 `close_ledger` 가 세 대조를 지날 때만 닫는다.

로컬 세 모델은 **각자 공통 초기 어댑터에서 출발한다** — 앞 모델의 어댑터를 이어 쓰지 않는다(74번 감사 C-1).
재개 파일은 지우지 않는다 — 기존 체크포인트를 지우지 않는다는 규칙을 학습 쪽에서도 지킨다.

## 여기서 하지 않는 것

경로 규칙(비본실험 부모 · 루트 종류) · 등록과의 대조 · 고의 중단 장치는 이 모듈 밖이다. 목적별 가드 가운데
이음새(순서 3)와 계획·학습 목록(순서 5의 일부)만 여기서 건다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from detection import serialize
from detection.round_runner import derive_seed
from fl.atomic_log import END_METRIC, AtomicLog, last_data_row, new_run_id
from fl.uni_run_config import PURPOSES
from vlm.coords import CoordCfg, coord_cfg_hash

__all__ = [
    "LOCAL_TAGS", "CENTRAL_TAG", "CLIENT_ORDER", "cell_of", "client_of",
    "UniRunSpec", "UniCellResult", "CellRejected", "LedgerOpen",
    "run_uni_local_cell", "run_uni_central_cell", "save_adapter_cell",
    "cell_dir", "cell_run_id", "read_ledger_rows", "ledger_file_sha256", "ledger_view",
    "check_local_end_row", "close_ledger", "resume_segments_from_rows", "build_train_config",
    "train_config_object", "check_before_load",
]

CLIENT_ORDER = ("C1", "C2", "C3")
LOCAL_TAGS = {c: f"uni_local_{c}" for c in CLIENT_ORDER}
CENTRAL_TAG = "uni_central"
ADAPTER_FILE = "adapter_last.npz"
META_FILE = "adapter_last.meta.json"
LEDGER_FILE = "train_ledger.csv"
SEAM_METRICS = ("process_start", "process_started_unix")
#: epoch 행의 지표. 원장 콜백이 넘기는 값에서 이 일곱을 쓴다(리허설 2판 §2-6).
EPOCH_METRICS = ("epochs_ran", "optimizer_steps", "supervised_tokens", "mean_ce", "lr",
                 "param_l2", "peak_vram_gb")
REPO_ROOT = Path(__file__).resolve().parents[1]


def cell_of(tag: str) -> str:
    """원장 `cell` 열의 값 — 평가 쪽의 칸 이름(`uni_local`·`uni_central`)."""
    if tag in LOCAL_TAGS.values():
        return "uni_local"
    if tag == CENTRAL_TAG:
        return "uni_central"
    raise ValueError(f"로컬·중앙 칸의 태그가 아니다: {tag!r}")


def client_of(tag: str) -> str:
    """원장 `client_id` 열의 값 — 참여자 이름(`C1`…). 칸 태그가 아니다(12번 정정 94행)."""
    cell_of(tag)
    return "central" if tag == CENTRAL_TAG else tag.removeprefix("uni_local_")


class CellRejected(ValueError):
    """이 칸을 시작하지 않는다. `code` 가 거부 사유다 — 시험은 종료 여부뿐 아니라 사유를 맞댄다."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


class LedgerOpen(CellRejected):
    """어댑터 파일은 있는데 원장이 닫히지 않았다 — 건너뜀(종료 0)으로 숨기지 않는다."""


@dataclass(frozen=True)
class UniRunSpec:
    """한 시드의 통합형 칸들이 공유하는 실행 설정. **상수를 모듈에 박지 않는다** — 실행기가 설정에서 채운다.

    `num_rounds` · `local_epochs` · `total_epochs` 는 **등록 예산**이다(칸과 무관). 로컬·중앙은 `total_epochs`
    를 한 번에 돈다(`R=1, E=N`). 학습 설정 원문에는 이 셋이 그대로 실린다.
    """

    purpose: str
    run_stamp: str
    seed_index: int
    seed_value: int
    snapshot_digest: str
    model_id: str
    model_revision: str | None
    pairs_path: str
    pairs_digest: str
    chat_template_kwargs: Mapping[str, Any]
    num_rounds: int
    local_epochs: int
    total_epochs: int
    train_root: Path
    init_adapter_path: Path
    resume_root: Path | None = None
    prompt_path: str | None = None
    coord_space: str = "ABS_ORIG"
    processor_kwargs: Mapping[str, Any] | None = None
    plan_sha256: str | None = None
    standin_allowed: bool = False
    #: epoch 하나의 감독 토큰 기대값 — 참여자 이름 셋(`C1` · `C2` · `C3`)이 정확히 있어야 한다. 중앙은 셋의 합이다.
    #: 본실험은 비워 둘 수 없다. 리허설 · 진단은 비우면 대조하지 않고 meta 에 `null` 로 적는다.
    #: 키가 빠진 기대값은 적재 전 가드가 거부한다 — 그 칸의 학습을 다 돈 뒤에 멈추지 않게.
    expected_supervised_tokens: Mapping[str, int] | None = None
    #: 본실험이 맞댈 설정 파일. 없으면 작업 트리의 `configs/base.yaml` 이다(시험이 임시 파일을 준다).
    config_path: Path | None = None
    #: 리허설 · 진단의 루트(`reh-*` · `fdiag-*`). 출력 · 초기 어댑터 · 재개 폴더가 모두 이 아래여야 한다(2판 §1-4 의 순서 4).
    run_root: Path | None = None
    #: 비본실험 부모 — 이음새처럼 **파이썬 인자로만** 바꾼다(합성 시험이 임시 폴더를 준다). 없으면 `outputs/rehearsal_u`.
    non_main_parent: Path | None = None

    def __post_init__(self) -> None:
        if self.purpose not in PURPOSES:
            raise CellRejected("purpose_unknown", f"purpose 가 {list(PURPOSES)} 가운데 하나가 아니다: {self.purpose!r}")
        for name in ("seed_index", "seed_value", "num_rounds", "local_epochs", "total_epochs"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise CellRejected("spec_type", f"{name} 이 정수가 아니다: {v!r}")
        if self.seed_index < 1:
            raise CellRejected("spec_value", f"seed_index 는 1 부터다: {self.seed_index}")
        if self.total_epochs != self.num_rounds * self.local_epochs or self.total_epochs < 1:
            raise CellRejected("budget", f"total_epochs {self.total_epochs} ≠ num_rounds {self.num_rounds} × "
                                         f"local_epochs {self.local_epochs}")
        if not isinstance(self.chat_template_kwargs, Mapping):
            raise CellRejected("spec_type", "chat_template_kwargs 가 사전이 아니다")
        for name in ("run_stamp", "snapshot_digest", "model_id", "pairs_path", "pairs_digest"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise CellRejected("spec_value", f"{name} 이 비어 있다")


@dataclass
class UniCellResult:
    tag: str
    status: str
    """`trained` 또는 `skipped`(같은 신원 · 닫힌 원장)."""
    cell_dir: Path
    adapter_sha256: str
    adapter_step: int
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- 경로 · 신원
def cell_dir(spec: UniRunSpec, tag: str) -> Path:
    return Path(spec.train_root) / f"{tag}_s{spec.seed_index}"


def cell_run_id(spec: UniRunSpec, tag: str) -> str:
    """원장 · 재개 신원 · meta 의 실행 신원 — `new_run_id(tag, seed_value, stamp)`(리허설 2판 §2-6)."""
    return new_run_id(tag, spec.seed_value, spec.run_stamp)


def _record_path(p: Path) -> str:
    """기록용 경로 — **`resolve()` 하기 전의 경로**를 작업 트리 루트 기준으로 낸다(리허설 2판 §5-1).
    작업 트리 밖이면 절대 경로를 적는다(합성 시험의 임시 폴더)."""
    ab = Path(os.path.abspath(p))
    try:
        return ab.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return ab.as_posix()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj: Any) -> str:
    from vlm.pilot_vlm import canonical_json

    return canonical_json(obj)


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _run_identity(spec: UniRunSpec, tag: str, *, client_idx: int, rows_digest: str, n_rows: int,
                  prompt_sha256: str, init_digest: str) -> dict[str, Any]:
    """실행 신원 가운데 **모델을 올리기 전에 낼 수 있는 블록**. meta 의 `identity` 는 이 블록에 재개 신원 전체
    (`resume` — `vlm/resume_uni.UniResumeIdentity`, 정책 셋 포함)를 품은 것이고 `identity_sha256` 은 그 전체의 해시다
    (리허설 2판 §2-7). 멱등 판정과 원장 복구는 모델을 올리기 전에 돌므로 `_verify_prior` 가 이 블록과 재개 신원의
    적재 전 필드(`_expected_resume`)를 맞댄다. 재개 사슬은 적재 뒤에 전체 해시를 맞댄다.
    """
    from vlm.pilot_vlm import (GRAD_ACCUM, LR, MICRO_BATCH, SHUFFLE_POLICY,
                               TARGET_CONTRACT_SHA256, template_mode_of)
    from vlm.schedule import LRF

    return {
        "run_id": cell_run_id(spec, tag),
        "tag": tag, "cell": cell_of(tag), "client": client_of(tag),
        "seed_index": spec.seed_index, "seed_value": spec.seed_value, "purpose": spec.purpose,
        "round_idx": 0, "client_idx": int(client_idx),
        "derived_seed": derive_seed(spec.seed_value, 0, int(client_idx)),
        "epochs": spec.total_epochs, "num_rounds_trained": 1,
        "budget": {"num_rounds": spec.num_rounds, "local_epochs": spec.local_epochs,
                   "total_epochs": spec.total_epochs},
        "model_id": spec.model_id, "model_revision": spec.model_revision,
        "pairs_digest": spec.pairs_digest, "prompt_sha256": prompt_sha256,
        "template_mode": template_mode_of(spec.chat_template_kwargs),
        "shuffle_policy": SHUFFLE_POLICY,
        "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space=spec.coord_space)),
        "target_contract_sha256": TARGET_CONTRACT_SHA256,
        "init_adapter_digest": init_digest,
        "train_rows_digest": rows_digest, "n_train_rows": int(n_rows),
        "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM, "lr0": LR, "lrf": LRF,
        "snapshot_digest": spec.snapshot_digest,
        "plan_sha256": spec.plan_sha256,
        # 프로세서 설정 — 해상도만 바꿔도 다른 입력이다. 멱등 · 재개 판정이 모델을 올리기 전에 가른다.
        "processor_kwargs": None if spec.processor_kwargs is None else dict(spec.processor_kwargs),
        # 감독 토큰 기대값 — 기대값만 바꿔 다시 요청하면 완료 산출물을 건너뛰지 않고 신원이 어긋난다.
        "expected_supervised_tokens": (None if spec.expected_supervised_tokens is None
                                       else {str(k): int(v) for k, v in spec.expected_supervised_tokens.items()}),
    }


#: 재개 신원 가운데 **프로세서가 있어야 나오는** 두 필드. 모델을 올리기 전에는 다시 낼 수 없다.
RESUME_POST_LOAD = ("gen_prefix_digest", "processor_config_sha256")


def _expected_resume(spec: UniRunSpec, tag: str, *, client_idx: int, prompt_sha256: str,
                     init_digest: str) -> dict[str, Any]:
    """재개 신원(`UniResumeIdentity`) 가운데 **모델을 올리기 전에 낼 수 있는** 필드의 기대값.

    `train_rounds` 가 이 칸에서 만드는 값과 같아야 한다. 기록을 되짚을 때(멱등 · 원장 복구) meta 에 실린 재개 신원을
    이것과 맞댄다. 프로세서 쪽 두 필드(`RESUME_POST_LOAD`)는 다시 낼 수 없어 보지 않는다 — 그 둘은 모델 판 ·
    프로세서 설정(둘 다 앞 블록에 있다)에서 정해진다.
    """
    from vlm.pilot_vlm import (GRAD_ACCUM, LR, MICRO_BATCH, SHUFFLE_POLICY,
                               TARGET_CONTRACT_SHA256, template_mode_of)
    from vlm.schedule import LRF

    return {
        "run_id": cell_run_id(spec, tag), "round_idx": 0, "client_idx": int(client_idx),
        "seed": derive_seed(spec.seed_value, 0, int(client_idx)),
        "total_epochs": spec.total_epochs, "local_epochs": spec.total_epochs,
        "model": str(spec.model_id), "data": str(spec.pairs_path),
        "loader_reseed_per_epoch": False, "loader_seed": None, "profile": "",
        "shuffle_policy": SHUFFLE_POLICY, "template_mode": template_mode_of(spec.chat_template_kwargs),
        "cell": tag, "client_tag": client_of(tag), "num_rounds": 1,
        "pairs_digest": str(spec.pairs_digest), "prompt_sha256": prompt_sha256,
        "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space=spec.coord_space)),
        "target_contract_sha256": TARGET_CONTRACT_SHA256, "input_adapter_digest": init_digest,
        "micro_batch": int(MICRO_BATCH), "grad_accum": int(GRAD_ACCUM), "lr0": float(LR), "lrf": float(LRF),
    }


def build_train_config(spec: UniRunSpec, *, prompt_sha256: str, init_digest: str,
                       metrics: Mapping[str, Any]) -> dict[str, Any]:
    """학습 설정 원문의 객체 — 12번 §7-1 의 항목 + `processor_config_sha256` · `target_contract_sha256` +
    다섯 묶음(리허설 2판 §2-5). **칸과 무관하다** — 같은 시드의 다섯 칸이 같은 값을 낸다. 칸별 학습 목록은
    여기 넣지 않는다(meta 의 `train_rows_digest`).
    """
    return train_config_object(
        model_id=spec.model_id, model_revision=spec.model_revision, pairs_digest=spec.pairs_digest,
        chat_template_kwargs=spec.chat_template_kwargs, coord_space=spec.coord_space,
        num_rounds=spec.num_rounds, local_epochs=spec.local_epochs, total_epochs=spec.total_epochs,
        prompt_sha256=prompt_sha256, init_digest=init_digest, metrics=metrics)


def train_config_object(*, model_id: str, model_revision: str | None, pairs_digest: str | None,
                        chat_template_kwargs: Mapping[str, Any], coord_space: str,
                        num_rounds: int, local_epochs: int, total_epochs: int,
                        prompt_sha256: str, init_digest: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """`build_train_config` 의 본체 — 연합 서버도 같은 함수로 낸다(칸마다 따로 짜면 값이 갈린다).

    `metrics` 는 학습이 실제로 쓴 값이다 — `processor_config_sha256` · `config_blocks` 와 대조용
    `prompt_sha256` · `template_mode` · `coord_cfg_hash` · `target_contract_sha256`.
    """
    from vlm.pilot_vlm import (GRAD_ACCUM, LR, MICRO_BATCH, SHUFFLE_POLICY,
                               TARGET_CONTRACT_SHA256, template_mode_of)
    from vlm.schedule import LRF

    cfg = {
        "model_id": model_id, "model_revision": model_revision,
        "pairs_digest": pairs_digest, "prompt_sha256": prompt_sha256,
        "template_mode": template_mode_of(chat_template_kwargs),
        "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space=coord_space)),
        "num_rounds": num_rounds, "local_epochs": local_epochs, "total_epochs": total_epochs,
        "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM, "lr0": LR, "lrf": LRF,
        "init_adapter_digest": init_digest, "shuffle_policy": SHUFFLE_POLICY,
        "processor_config_sha256": metrics["processor_config_sha256"],
        "target_contract_sha256": TARGET_CONTRACT_SHA256,
    }
    blocks = dict(metrics["config_blocks"])
    if set(blocks) != {"lora", "quant", "optimizer", "loss", "lr_schedule"}:
        raise ValueError(f"학습 설정 원문의 다섯 묶음이 아니다: {sorted(blocks)}")
    cfg.update(blocks)
    # 학습이 실제로 쓴 값과 설정에서 낸 값이 같은지 본다 — 원문이 설정을 적고 학습은 다른 값으로 돌면 안 된다.
    for k in ("prompt_sha256", "template_mode", "coord_cfg_hash", "target_contract_sha256"):
        if metrics.get(k) != cfg[k]:
            raise ValueError(f"학습이 쓴 {k} {metrics.get(k)!r} ≠ 설정의 값 {cfg[k]!r}")
    return cfg


# ---------------------------------------------------------------- 원장 읽기
def read_ledger_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh) if any(r.values())]


def ledger_file_sha256(path: str | Path) -> str:
    """원장 해시 — **원시 바이트**의 sha256. 줄끝(CRLF)을 정규화하지 않는다(리허설 2판 §2-6)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _effective_epoch_rows(rows: Sequence[Mapping[str, str]]) -> dict[tuple[int, str, str], float]:
    """겹친 행 규칙을 적용한 epoch 행 — 같은 `(round, client_id, metric_name)` 은 **이음매 행 뒤의 것이 이긴다.**

    이음매 행 없이 같은 열쇠가 두 번 나오면 원장이 깨진 것이라 거부한다 — 한 프로세스는 한 epoch 를 한 번만 쓴다.
    """
    out: dict[tuple[int, str, str], float] = {}
    seen_in_segment: set[tuple[int, str, str]] = set()
    for r in rows:
        name = r["metric_name"]
        if name in SEAM_METRICS:
            seen_in_segment = set()
            continue
        if name == END_METRIC:
            continue
        key = (int(r["round"]), r["client_id"], name)
        if key in seen_in_segment:
            raise ValueError(f"이음매 행 없이 같은 행이 두 번 있다: {key}")
        seen_in_segment.add(key)
        out[key] = float(r["metric_value"])
    return out


def resume_segments_from_rows(rows: Sequence[Mapping[str, str]], *, verified: Sequence[float],
                              identity_sha256: str) -> list[dict[str, Any]]:
    """프로세스마다 `{start_epoch, end_epoch, identity_sha256, started_at, ended_at}` — **원장의 이음매 행에서** 만든다.

    체크포인트를 남기지 못하고 죽은 프로세스도 이음매 행은 남긴다. `end_epoch` 는 그 뒤 epoch 행에서 읽고,
    epoch 행이 없으면 None 이다. `ended_at` 은 언제나 None 이다 — 종료 시각을 추정해 넣지 않는다. 끝낸
    프로세스의 종료 시각은 meta 를 쓰는 쪽이 채운다. 시각은 유닉스 초다.

    `identity_sha256` 은 **재개 사슬로 이어진 프로세스에만** 적는다 — 각 프로세스는 이어 받은 체크포인트의
    신원 해시를 자기 것과 맞대고 이음매 행을 썼으므로(`verified` 는 그 사슬의 시작 시각 목록) 사슬 위의 값은
    지금 신원과 같다. 사슬 밖(체크포인트 없이 죽은 뒤 처음부터 다시 선 경우)은 None 이다.
    """
    segs: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    pending_start: int | None = None
    for r in rows:
        name = r["metric_name"]
        if name == "process_start":
            pending_start = int(float(r["metric_value"]))
            continue
        if name == "process_started_unix":
            if pending_start is None:
                raise ValueError("process_started_unix 앞에 process_start 가 없다")
            t = float(r["metric_value"])
            cur = {"start_epoch": pending_start, "end_epoch": None,
                   "identity_sha256": identity_sha256 if t in set(verified) else None,
                   "started_at": t, "ended_at": None}
            segs.append(cur)
            pending_start = None
            continue
        if name == "epochs_ran" and cur is not None:
            ep = int(r["round"])
            cur["end_epoch"] = ep if cur["end_epoch"] is None else max(cur["end_epoch"], ep)
    return segs


def ledger_view(path: str | Path) -> dict[str, Any]:
    """평가 쪽 `LedgerView` 로 옮길 값 넷 — 리허설 2판 §2-6 의 규칙.

    마지막 데이터 행이 끝 행이 아니면 거부한다. `final_step` 은 그 행의 값을 정수로 읽는다(`round` 는 읽지
    않는다). `client` 는 `central`·`server` 면 None 이다.
    """
    last = last_data_row(path)
    if last is None or last["metric_name"] != END_METRIC:
        raise ValueError(f"닫히지 않은 원장이다(마지막 데이터 행이 {END_METRIC} 가 아니다): {path}")
    v = float(last["metric_value"])
    if not v.is_integer():
        raise ValueError(f"끝 행의 값이 정수가 아니다: {v}")
    client = last["client_id"]
    return {"run_id": last["run_id"], "file_sha256": ledger_file_sha256(path), "final_step": int(v),
            "cell": last["cell"], "client": None if client in ("central", "server") else client}


def check_local_end_row(path: str | Path) -> int:
    """로컬·중앙 원장의 끝 행을 같은 원장의 학습 행과 한 번 더 맞댄다 — 겹친 행 규칙 뒤 **마지막 epoch 의
    `optimizer_steps`** 와 같아야 한다. 맞으면 그 값을 돌려준다."""
    rows = read_ledger_rows(path)
    view = ledger_view(path)
    eff = _effective_epoch_rows(rows)
    steps = {rd: v for (rd, cid, name), v in eff.items()
             if name == "optimizer_steps" and cid == rows[-1]["client_id"]}
    if not steps:
        raise ValueError(f"epoch 행이 없는 원장이다: {path}")
    last_ep = max(steps)
    if int(steps[last_ep]) != view["final_step"]:
        raise ValueError(f"끝 행 {view['final_step']} ≠ 마지막 epoch({last_ep})의 optimizer_steps {int(steps[last_ep])}")
    return view["final_step"]


# ---------------------------------------------------------------- 저장
def save_adapter_cell(out_dir: Path, arrays: Sequence[np.ndarray], keys: Sequence[str],
                      meta: dict[str, Any]) -> dict[str, Any]:
    """최종 어댑터를 **키가 있는** `adapter_last.npz` 로 내구 저장하고 meta 를 쓴다. 순서가 계약이다.

    tmp 저장 → **tmp 재로드 후 키 · 배열 수 · shape · dtype · 값 대조** → sha256(tmp) → `meta.json`(LF) 내구 저장
    → `os.replace(tmp, npz)`. 어댑터 파일이 완료 표지이므로 맨 마지막에 확정한다 — "표지가 있으면 meta 도 있다".
    `os.replace` 는 바이트를 바꾸지 않아 tmp 에서 잰 해시가 최종 파일의 값이다. 검증에 실패한 tmp 는 남긴다.
    돌려주는 값은 meta 에 채운 파일 필드다.
    """
    from detection.train_cell import _write_durable

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = list(arrays)
    keys = list(keys)
    if len(arrays) != len(keys) or len(set(keys)) != len(keys):
        raise ValueError("배열과 키의 수가 다르거나 키가 겹친다")
    path = out_dir / ADAPTER_FILE

    def _verify(tmp: Path) -> None:
        with np.load(tmp) as z:
            if list(z.files) != keys:
                raise RuntimeError(f"{tmp}: 재로드 키가 저장한 키와 다르다 — 저장이 깨졌다")
            for k, a in zip(keys, arrays):
                b = z[k]
                if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
                    raise RuntimeError(f"{tmp}: {k} 재로드 불일치(shape/dtype/값) — 저장이 깨졌다")

    tmp = _write_durable(path, lambda fh: np.savez(fh, **dict(zip(keys, arrays))), _verify, replace=False)
    files = {"adapter_file": ADAPTER_FILE, "adapter_sha256": _sha256_file(tmp), "n_arrays": len(arrays),
             "file_bytes": tmp.stat().st_size, "canonical_keys_sha256": serialize.keys_digest(keys)}
    full = {**meta, **files}
    payload = (json.dumps(full, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    _write_durable(out_dir / META_FILE, lambda fh: fh.write(payload))
    os.replace(tmp, path)
    return files


# ---------------------------------------------------------------- 한 칸
def _load_init(spec: UniRunSpec, model_loader: Callable | None) -> tuple[list[np.ndarray], list[str], dict]:
    """공통 초기 어댑터를 **읽기만** 한다. proof 규칙은 `build_initial_adapter` 가 건다(본실험은 proof 필수)."""
    from vlm.init_adapter import adapter_proof, build_initial_adapter, init_adapter_digest

    p = Path(spec.init_adapter_path)
    if not p.exists():
        raise CellRejected("init_adapter_missing",
                           f"공통 초기 어댑터가 없다: {p}. 실행기의 initadapter 단계를 먼저 돌린다")
    arrays, keys, _ = build_initial_adapter(
        model_id=spec.model_id, seed=spec.seed_value, cache_path=p, revision=spec.model_revision,
        purpose=spec.purpose, model_loader=model_loader, standin_allowed=spec.standin_allowed)
    return arrays, keys, {"init_adapter_digest": init_adapter_digest(arrays),
                          "init_adapter_tensor_digest": adapter_proof(arrays, keys)["tensor_digest"],
                          "proof": adapter_proof(arrays, keys)}


def _verify_prior(meta: Mapping[str, Any], identity: dict[str, Any], expected_resume: Mapping[str, Any], *,
                  where: Path) -> None:
    """앞 산출물의 meta 가 지금 실행의 것인지 — `identity`(재개 신원 포함)와 `identity_sha256` 을 맞댄다.

    1. meta 의 해시가 meta 의 신원 **전체**(재개 신원 포함)에서 다시 난 값과 같다 — 기록이 온전하다.
    2. 신원의 적재 전 블록이 지금 요청의 것과 같다.
    3. 재개 신원이 있고, 그 가운데 적재 전에 낼 수 있는 필드가 지금 요청의 기대값과 같다. 프로세서 쪽 두 필드
       (`RESUME_POST_LOAD`)만 다시 내지 못하고, 그 둘은 1 의 해시가 지킨다.

    멱등 판정과 `close_ledger` 가 같은 함수를 쓴다.
    """
    old = meta.get("identity")
    old_sha = meta.get("identity_sha256")
    if not isinstance(old, dict) or not isinstance(old_sha, str):
        raise CellRejected("identity_missing", f"앞 산출물의 meta 에 identity · identity_sha256 이 없다: {where}")
    if _sha(old) != old_sha:
        raise CellRejected("identity_sha_broken", f"앞 산출물 meta 의 identity_sha256 이 그 identity 와 맞지 않는다: {where}")
    pre = {k: v for k, v in old.items() if k != "resume"}
    if pre != identity:
        diff = sorted(k for k in set(pre) | set(identity) if pre.get(k) != identity.get(k))
        raise CellRejected("identity_mismatch",
                           f"다른 실행의 산출물이 이미 있다: {where} — 다른 칸 {diff}. 새 스탬프는 새 출력 폴더로 돌린다")
    res = old.get("resume")
    if not isinstance(res, dict) or any(k not in res for k in RESUME_POST_LOAD):
        raise CellRejected("resume_identity_missing", f"앞 산출물 meta 의 identity 에 재개 신원이 없다: {where}")
    diff = sorted(k for k, v in expected_resume.items() if res.get(k) != v)
    extra_keys = sorted(set(res) - set(expected_resume) - set(RESUME_POST_LOAD))
    if diff or extra_keys:
        raise CellRejected("identity_mismatch",
                           f"앞 산출물의 재개 신원이 지금 요청과 다르다: {where} — 다른 칸 {diff} · 모르는 칸 {extra_keys}")


def _idempotence(d: Path, identity: dict[str, Any], expected_resume: Mapping[str, Any]) -> dict[str, Any] | None:
    """어댑터 파일이 있으면 meta 의 신원을 **먼저** 맞댄다(리허설 2판 §2-6 의 멱등 판정의 순서).

    같고 원장이 닫혔으면 meta 를 돌려준다(건너뜀). 같은데 원장이 열려 있으면 `LedgerOpen`. 다르면 거부.
    어댑터 파일이 없으면 None — 학습한다.
    """
    npz, meta_p, led = d / ADAPTER_FILE, d / META_FILE, d / LEDGER_FILE
    if not npz.exists():
        return None
    if not meta_p.exists():
        raise CellRejected("meta_missing", f"완료 표지는 있는데 meta 가 없다: {d}")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    _verify_prior(meta, identity, expected_resume, where=d)
    if _sha256_file(npz) != meta.get("adapter_sha256"):
        raise CellRejected("adapter_sha_mismatch", f"어댑터 파일의 sha 가 meta 의 adapter_sha256 과 다르다: {d}")
    last = last_data_row(led) if led.exists() else None
    if last is None or last["metric_name"] != END_METRIC:
        raise LedgerOpen("ledger_open",
                         f"어댑터는 있는데 원장이 닫히지 않았다: {led}. `main_uni.py close-ledger` 로 세 대조를 거쳐 닫는다")
    return meta


def _run_cell(spec: UniRunSpec, tag: str, *, client_idx: int, rows: list[dict],
              init: tuple[list[np.ndarray], list[str], dict], model_loader: Callable | None,
              hooks: Mapping[str, Callable] | None, armed: Any = None) -> UniCellResult:
    from vlm.fault import fault_events
    from vlm.init_adapter import assert_injected_matches
    from vlm.pilot_vlm import GRAD_ACCUM, load_prompt, train_rounds, train_rows_digest

    init_arrays, init_keys, init_info = init
    d = cell_dir(spec, tag)
    _, prompt_sha256 = load_prompt(spec.prompt_path)
    rows_digest = train_rows_digest(rows)
    identity = _run_identity(spec, tag, client_idx=client_idx, rows_digest=rows_digest, n_rows=len(rows),
                             prompt_sha256=prompt_sha256, init_digest=init_info["init_adapter_digest"])

    expected_resume = _expected_resume(spec, tag, client_idx=client_idx, prompt_sha256=prompt_sha256,
                                       init_digest=init_info["init_adapter_digest"])
    prior = _idempotence(d, identity, expected_resume)
    if prior is not None:
        return UniCellResult(tag=tag, status="skipped", cell_dir=d, adapter_sha256=prior["adapter_sha256"],
                             adapter_step=int(prior["adapter_step"]), metrics=prior.get("metrics") or {})

    # "한 번만" — 가드를 지나고 멱등 판정을 한 뒤에 선다(2판 §1-4 의 순서 7). 같은 열쇠의 활성화 기록이 있으면 켜지 않는다.
    if armed is not None and armed.spec.tag == tag and armed.already_activated():
        print(f"[fault] {armed.spec.raw} 의 활성화 기록이 이미 있다 — 다시 켜지 않는다", file=sys.stderr, flush=True)
        armed = None
    run_id = cell_run_id(spec, tag)
    client = client_of(tag)
    ledger = AtomicLog(d / LEDGER_FILE, run_id=run_id, seed=spec.seed_value, cell=cell_of(tag),
                       split_hash=spec.snapshot_digest)
    if ledger.closed:
        raise CellRejected("ledger_closed_without_adapter",
                           f"원장은 닫혔는데 어댑터 파일이 없다: {d}. 산출물을 지우지 말고 원인을 먼저 본다")

    extra: dict[str, Any] = {}
    chain: dict[str, Any] = {}

    def on_identity_ok(start_epoch: int, payload: Mapping[str, Any] | None,
                       resume_identity: Mapping[str, Any] | None = None) -> None:
        res = dict(resume_identity or {})
        bad = sorted(k for k, v in expected_resume.items() if res.get(k) != v)
        if bad:
            raise CellRejected("resume_identity_unexpected", f"학습 루프가 만든 재개 신원이 기대와 다르다: {bad}")
        full = {**identity, "resume": res}
        sha = _sha(full)
        verified: list[float] = []
        if payload is not None:
            # 이어 받는 체크포인트가 **지금과 같은 신원**으로 쓰였고, 그 프로세스의 이음매 행이 원장에 있는지 맞댄다.
            if payload.get("identity_sha256") != sha:
                raise CellRejected("resume_identity_mismatch",
                                   "재개 체크포인트의 신원 해시가 지금 실행과 다르다 — 적용하지 않는다")
            seams = {float(r["metric_value"]) for r in read_ledger_rows(d / LEDGER_FILE)
                     if r["metric_name"] == "process_started_unix"}
            t_prev = payload.get("process_started_unix")
            if t_prev is None or float(t_prev) not in seams:
                raise CellRejected("resume_seam_missing",
                                   "재개 체크포인트를 쓴 프로세스의 이음매 행이 원장에 없다 — 적용하지 않는다")
            verified = [float(x) for x in payload.get("verified_processes") or []] + [float(t_prev)]
        t = time.time()
        ledger.log_round(round_idx=int(start_epoch), client_id=client, n_train_samples=len(rows),
                         metrics={"process_start": float(start_epoch), "process_started_unix": t})
        extra.update({"identity_sha256": sha, "process_started_unix": t, "verified_processes": verified})
        chain.update({"identity": full, "identity_sha256": sha, "t": t, "verified": verified + [t]})

    def ledger_cb(v: Mapping[str, Any]) -> None:
        ledger.log_round(round_idx=int(v["epoch"]), client_id=client, n_train_samples=len(rows),
                         metrics={k: float(v[k]) for k in EPOCH_METRICS}, wall_time=float(v["wall_s"]))

    hooks = dict(hooks or {})
    if armed is not None and armed.spec.tag == tag:
        # 체크포인트 앞뒤의 중단 지점. 훅은 목적과 루트를 **문맥에서 다시 받는다**(2판 §1-3 의 막는 자리).
        from vlm.fault import FaultContext

        ctx = FaultContext(purpose=spec.purpose, run_root=Path(spec.run_root))
        for point in ("before_ckpt", "after_ckpt"):
            prev = hooks.get(point)

            def _hook(ep, ckpt, _point=point, _prev=prev):
                if _prev is not None:
                    _prev(ep, ckpt)
                armed.fire(ctx, point=_point, tag=tag, k=int(ep), run_id=run_id)

            hooks[point] = _hook
    arrays, keys, m, _ = train_rounds(
        rows=rows, epochs=spec.total_epochs, round_idx=0, client_idx=int(client_idx),
        base_seed=spec.seed_value, adapter_in=list(init_arrays), adapter_keys=list(init_keys),
        model_id=spec.model_id, resume_dir=(str(Path(spec.resume_root) / f"{tag}_s{spec.seed_index}")
                                            if spec.resume_root is not None else None),
        run_id=run_id, init_seed=spec.seed_value, num_rounds=1,
        model_revision=spec.model_revision, pairs_path=spec.pairs_path, prompt_path=spec.prompt_path,
        chat_template_kwargs=dict(spec.chat_template_kwargs),
        coord_cfg=CoordCfg(coord_space=spec.coord_space), processor_kwargs=spec.processor_kwargs,
        model_loader=model_loader, purpose=spec.purpose, standin_allowed=spec.standin_allowed,
        tag=tag, seed_index=spec.seed_index,
        on_identity_ok=on_identity_ok, ledger_cb=ledger_cb,
        before_ckpt_cb=hooks.get("before_ckpt"), after_ckpt_cb=hooks.get("after_ckpt"),
        resume_extra=extra, pairs_digest=spec.pairs_digest, client_tag=client,
    )

    # -- 대조: 주입 · 키 · 예산 ------------------------------------------------
    if keys != list(init_keys):
        raise RuntimeError(f"{tag}: 학습이 낸 정본 키가 초기 어댑터의 키와 다르다")
    if m.get("injected_proof") is None:
        raise RuntimeError(f"{tag}: 초기 어댑터 주입 증빙이 없다")
    assert_injected_matches(list(init_arrays), list(init_keys), m["injected_proof"], who=tag)
    want_steps = math.ceil(len(rows) / GRAD_ACCUM) * spec.total_epochs
    if int(m["epochs_ran"]) != spec.total_epochs or int(m["optimizer_steps"]) != want_steps:
        raise RuntimeError(f"{tag}: 회계 불일치 — epoch {m['epochs_ran']}/{spec.total_epochs}, "
                           f"갱신 {m['optimizer_steps']}/{want_steps}")
    if m.get("train_input_grid_observed") is None or sum(m["train_input_grid_observed"].values()) != len(rows):
        raise RuntimeError(f"{tag}: 관측 격자가 모든 학습 행을 세지 않았다")
    # 감독 토큰 — 갱신 수와 함께 **둘 다** 기대값과 맞댄다(12번 §2-4). 기대값이 없으면(리허설 · 진단) 대조하지 않는다.
    tokens_expected = _expected_tokens(spec, client)
    if tokens_expected is not None and int(m["supervised_tokens"]) != tokens_expected:
        raise RuntimeError(f"{tag}: 감독 토큰 {m['supervised_tokens']} ≠ 기대값 {tokens_expected} "
                           f"(epoch 당 {tokens_expected // spec.total_epochs} × {spec.total_epochs})")

    # -- meta · 저장 · 끝 행 ----------------------------------------------------
    train_config = build_train_config(spec, prompt_sha256=prompt_sha256,
                                      init_digest=init_info["init_adapter_digest"], metrics=m)
    tc_text = _canonical(train_config)
    segs = resume_segments_from_rows(read_ledger_rows(d / LEDGER_FILE), verified=chain["verified"],
                                     identity_sha256=chain["identity_sha256"])
    if not segs or segs[-1]["started_at"] != chain["t"]:
        raise RuntimeError(f"{tag}: 이 프로세스의 이음매 행이 원장의 마지막 세그먼트가 아니다")
    segs[-1]["ended_at"] = time.time()
    meta = {
        "adapter_step": int(m["optimizer_steps"]), "adapter_step_unit": "optimizer_steps",
        "identity": chain["identity"], "identity_sha256": chain["identity_sha256"],
        "supervised_tokens": int(m["supervised_tokens"]),
        "supervised_tokens_expected": tokens_expected,
        "train_rows_digest": rows_digest,
        "train_input_grid_observed": m["train_input_grid_observed"],
        "processor_config_sha256": m["processor_config_sha256"],
        "coord_cfg_hash": m["coord_cfg_hash"],
        "resume_segments": segs,
        "train_config": tc_text,
        "train_config_sha256": hashlib.sha256(tc_text.encode("utf-8")).hexdigest(),
        "init_adapter_digest": init_info["init_adapter_digest"],
        "init_adapter_tensor_digest": init_info["init_adapter_tensor_digest"],
        "train_run_id": run_id,
        "train_ledger_path": _record_path(d / LEDGER_FILE),
        "metrics": m,
        "impl_ids": m["impl_ids"],
        # 이 학습에서 켜진 고의 중단의 활성화 · 발화 기록과 그 뒤의 복구 세그먼트. 본실험은 언제나 빈 목록이다.
        "fault_events": ([] if spec.purpose == "main" else
                         fault_events(spec.run_root, stage="train", tags=[tag], segments=segs)),
    }
    if spec.purpose != "main":
        meta["plan_sha256"] = spec.plan_sha256
    files = save_adapter_cell(d, arrays, keys, meta)
    ledger.close(client_id=client, step=int(m["optimizer_steps"]), n_train_samples=len(rows))
    return UniCellResult(tag=tag, status="trained", cell_dir=d, adapter_sha256=files["adapter_sha256"],
                         adapter_step=int(m["optimizer_steps"]), metrics=m)


def _expected_tokens(spec: UniRunSpec, client: str) -> int | None:
    """이 칸의 감독 토큰 기대값(전 epoch 합). 기대값을 받지 않았으면 None."""
    exp = spec.expected_supervised_tokens
    if exp is None:
        return None
    if sorted(exp) != sorted(CLIENT_ORDER):
        # 적재 전 가드(`_preflight`)가 먼저 거부한다. 여기는 가드를 거치지 않은 호출을 막는 두 번째 자리다.
        raise CellRejected("tokens_expected_keys", f"감독 토큰 기대값의 참여자 {sorted(exp)} ≠ {list(CLIENT_ORDER)}")
    per = sum(int(exp[c]) for c in CLIENT_ORDER) if client == "central" else int(exp[client])
    return int(per) * spec.total_epochs


#: 본실험이 비워 둘 수 없는 칸. 비면 파일럿 기본값(모델 판 · 프롬프트 · 프로세서)으로 내려간다.
MAIN_REQUIRED = ("model_revision", "prompt_path", "processor_kwargs", "expected_supervised_tokens")


def _preflight(spec: UniRunSpec, *, model_loader: Callable | None, rows_given: bool,
               hooks: Mapping[str, Callable] | None, fault_tags: Sequence[str] = (),
               env: Mapping[str, str] | None = None) -> Any:
    """모델도 파일도 건드리기 전의 가드 — 2판 §1-4 의 순서대로다. 켤 중단이 있으면 그것을 돌려준다.

    2 중단 변수와 표식 → 3 이음새 → 4 루트와 모든 경로 → 5 계획 · 학습 목록 · 훅(본실험은 설정 대조까지).
    본실험은 필수 칸이 비어 있거나, 값이 설정 파일(`configs/base.yaml` 의 `uni_*` · 시드표 · 스냅샷)과
    다르거나, 출력이 비본실험 부모 아래면 시작하지 않는다 — 파일럿 기본값 · 옛 실행 설정으로 내려가지 않는다.
    `fault_tags` 는 이 호출이 학습하는 칸이다 — 비었으면 이 호출은 중단을 받지 않는다(원장 닫기 · 적재 전 검사).
    """
    from vlm.fault import FaultRefused, check_fault_env
    from vlm.pilot_vlm import load_prompt, resolve_model_loader
    from vlm.run_root import run_root_problems
    from vlm.uni_config import load_uni_config, main_run_problems

    try:                                                        # 순서 2
        armed = check_fault_env(spec.purpose, run_root=spec.run_root, stage="train", tags=fault_tags,
                                bounds={"ep": spec.total_epochs}, env=env)
    except FaultRefused as exc:
        raise CellRejected(exc.code, str(exc)) from None
    resolve_model_loader(model_loader, purpose=spec.purpose, standin_allowed=spec.standin_allowed)  # 순서 3
    bad_root = run_root_problems(spec.purpose, spec.run_root,                                          # 순서 4
                                 [spec.train_root, spec.init_adapter_path]
                                 + ([spec.resume_root] if spec.resume_root is not None else []),
                                 parent=spec.non_main_parent)
    if bad_root:
        raise CellRejected("run_root", " · ".join(bad_root))
    if armed is not None and spec.resume_root is None:
        # 중단 지점은 체크포인트 앞뒤다 — 재개 폴더가 없으면 체크포인트가 없어 켜진 중단이 조용히 발화하지 않는다.
        raise CellRejected("fault_needs_resume_root", f"켜진 중단 {armed.spec.raw} 에 재개 폴더가 없다")
    exp = spec.expected_supervised_tokens                                                            # 순서 5
    if exp is not None and sorted(exp) != sorted(CLIENT_ORDER):
        raise CellRejected("tokens_expected_keys", f"감독 토큰 기대값의 참여자 {sorted(exp)} ≠ {list(CLIENT_ORDER)}")
    if spec.purpose == "main":
        if spec.plan_sha256 is not None:
            raise CellRejected("main_with_plan", "본실험은 계획 파일을 받지 않는다")
        if rows_given:
            raise CellRejected("main_with_train_list", "본실험은 학습 행 목록 인자를 받지 않는다 — 페어의 참여자 행 전체로 돈다")
        if hooks:
            raise CellRejected("main_with_hooks", "본실험은 학습 훅을 받지 않는다")
        empty = [k for k in MAIN_REQUIRED if getattr(spec, k) in (None, "")]
        if empty:
            raise CellRejected("main_incomplete", f"본실험은 이 칸을 비워 둘 수 없다: {empty}")
        load_prompt(spec.prompt_path)                         # 파일이 없으면 여기서 멈춘다
        bad = main_run_problems(
            load_uni_config(spec.config_path), model_id=spec.model_id, model_revision=spec.model_revision,
            pairs_path=spec.pairs_path, pairs_digest=spec.pairs_digest,
            chat_template_kwargs=spec.chat_template_kwargs, num_rounds=spec.num_rounds,
            local_epochs=spec.local_epochs, total_epochs=spec.total_epochs, seed_index=spec.seed_index,
            seed_value=spec.seed_value, snapshot_digest=spec.snapshot_digest, prompt_path=spec.prompt_path,
            coord_space=spec.coord_space, processor_kwargs=spec.processor_kwargs,
            expected_supervised_tokens=spec.expected_supervised_tokens,
            output_paths=[Path(spec.train_root), Path(spec.init_adapter_path)]
            + ([Path(spec.resume_root)] if spec.resume_root is not None else []))
        if bad:
            raise CellRejected("main_config_mismatch", "본실험 설정과 다르다: " + " · ".join(bad))
    elif not spec.plan_sha256:
        raise CellRejected("plan_missing", f"{spec.purpose} 는 계획 파일의 해시가 있어야 한다")
    return armed


def check_before_load(spec: UniRunSpec, *, model_loader: Callable | None = None) -> None:
    """모델을 올리기 전의 가드만 부른다 — 실행기가 초기 어댑터를 만들기 전에 쓴다(같은 가드를 두 벌 짜지 않는다)."""
    _preflight(spec, model_loader=model_loader, rows_given=False, hooks=None)


def run_uni_local_cell(clients: Sequence[str] = CLIENT_ORDER, *, spec: UniRunSpec,
                       model_loader: Callable | None = None,
                       rows_by_client: Mapping[str, list[dict]] | None = None,
                       hooks: Mapping[str, Callable] | None = None) -> dict[str, UniCellResult]:
    """통합·로컬 — 참여자마다 **공통 초기 어댑터에서** 따로 학습한다. 결과를 참여자 이름으로 돌려준다.

    `rows_by_client` 는 리허설·진단의 부분 목록이다. 본실험은 받지 않고 페어의 참여자 행 전체로 돈다.
    """
    from vlm.pilot_vlm import load_pairs

    bad = [c for c in clients if c not in CLIENT_ORDER]
    if bad or len(set(clients)) != len(clients):
        raise CellRejected("client_unknown", f"참여자가 {list(CLIENT_ORDER)} 가운데 하나가 아니거나 겹친다: {list(clients)}")
    armed = _preflight(spec, model_loader=model_loader, rows_given=rows_by_client is not None, hooks=hooks,
                       fault_tags=[LOCAL_TAGS[c] for c in clients])
    init = _load_init(spec, model_loader)
    out: dict[str, UniCellResult] = {}
    for c in clients:
        rows = (list(rows_by_client[c]) if rows_by_client is not None
                else load_pairs("train", c, pairs_path=spec.pairs_path))
        out[c] = _run_cell(spec, LOCAL_TAGS[c], client_idx=CLIENT_ORDER.index(c), rows=rows, init=init,
                           model_loader=model_loader, hooks=hooks, armed=armed)
    return out


def run_uni_central_cell(*, spec: UniRunSpec, model_loader: Callable | None = None,
                         rows: list[dict] | None = None,
                         hooks: Mapping[str, Callable] | None = None) -> UniCellResult:
    """통합·중앙 — 학습 분할 전체로 한 번 학습한다. 참여자 번호는 0 이다(연합의 첫 클라이언트와 같은 파생 시드)."""
    from vlm.pilot_vlm import load_pairs

    armed = _preflight(spec, model_loader=model_loader, rows_given=rows is not None, hooks=hooks,
                       fault_tags=[CENTRAL_TAG])
    init = _load_init(spec, model_loader)
    rows = list(rows) if rows is not None else load_pairs("train", None, pairs_path=spec.pairs_path)
    return _run_cell(spec, CENTRAL_TAG, client_idx=0, rows=rows, init=init,
                     model_loader=model_loader, hooks=hooks, armed=armed)


def close_ledger(spec: UniRunSpec, tag: str, *, rows: list[dict] | None = None,
                 model_loader: Callable | None = None) -> int:
    """끝 행 앞에서 죽은 칸의 원장을 닫는다. **셋을 먼저 본다** — 이 함수가 끝 행을 지어내는 도구가 되지 않게.

    1. 어댑터 파일의 sha 가 meta 의 `adapter_sha256` 과 같다.
    2. 원장의 epoch 행(겹친 행 규칙 적용)이 예산의 epoch 를 빠짐없이 덮는다.
    3. 마지막 epoch 행의 `optimizer_steps` 가 meta 의 `adapter_step` 과 같다.

    하나라도 어긋나면 쓰지 않는다. 이미 닫혔으면 쓰지 않는다. 돌려주는 값은 쓴 끝 행의 스텝이다.

    그보다 먼저 **학습할 때와 같은 신원 검증**을 한다 — 같은 가드(`_preflight`)를 지나고, 지금 요청에서 신원을
    다시 내 meta 의 `identity` · `identity_sha256` 과 맞댄다. 목적이나 모델 판을 바꾼 요청은 닫지 못한다.
    `rows` 는 학습에 쓴 행이다(본실험은 페어에서 다시 읽고, 인자를 받지 않는다).
    """
    from vlm.pilot_vlm import load_pairs, load_prompt, train_rows_digest

    d = cell_dir(spec, tag)
    npz, meta_p, led = d / ADAPTER_FILE, d / META_FILE, d / LEDGER_FILE
    _preflight(spec, model_loader=model_loader, rows_given=rows is not None, hooks=None)     # 가드가 먼저다
    if not (npz.exists() and meta_p.exists() and led.exists()):
        raise CellRejected("close_inputs_missing", f"어댑터 · meta · 원장 가운데 없는 것이 있다: {d}")
    client = client_of(tag)
    if rows is None:
        rows = load_pairs("train", None if tag == CENTRAL_TAG else client, pairs_path=spec.pairs_path)
    _, _, init_info = _load_init(spec, model_loader)
    _, prompt_sha256 = load_prompt(spec.prompt_path)
    client_idx = 0 if tag == CENTRAL_TAG else CLIENT_ORDER.index(client)
    identity = _run_identity(spec, tag, client_idx=client_idx, rows_digest=train_rows_digest(rows),
                             n_rows=len(rows), prompt_sha256=prompt_sha256,
                             init_digest=init_info["init_adapter_digest"])
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    _verify_prior(meta, identity, _expected_resume(spec, tag, client_idx=client_idx, prompt_sha256=prompt_sha256,
                                                   init_digest=init_info["init_adapter_digest"]), where=d)
    # 감독 토큰 — 학습할 때와 같이 기대값과 맞댄다. 끝 행 앞에서 죽은 칸도 기대값을 지나야 닫힌다.
    tokens_expected = _expected_tokens(spec, client)
    if tokens_expected is not None and int(meta.get("supervised_tokens", -1)) != tokens_expected:
        raise CellRejected("close_tokens", f"meta 의 감독 토큰 {meta.get('supervised_tokens')} ≠ 기대값 {tokens_expected}")
    if _sha256_file(npz) != meta.get("adapter_sha256"):
        raise CellRejected("close_adapter_sha", "어댑터 파일의 sha 가 meta 와 다르다")
    led_rows = read_ledger_rows(led)
    eff = _effective_epoch_rows(led_rows)
    eps = {rd for (rd, cid, name) in eff if name == "epochs_ran" and cid == client}
    if eps != set(range(spec.total_epochs)):
        raise CellRejected("close_epochs", f"원장의 epoch 행 {sorted(eps)} 이 예산 0..{spec.total_epochs - 1} 을 덮지 않는다")
    last_steps = eff.get((spec.total_epochs - 1, client, "optimizer_steps"))
    if last_steps is None or int(last_steps) != int(meta.get("adapter_step", -1)):
        raise CellRejected("close_step", f"마지막 epoch 의 optimizer_steps {last_steps} ≠ meta 의 adapter_step {meta.get('adapter_step')}")
    ledger = AtomicLog(led, run_id=cell_run_id(spec, tag), seed=spec.seed_value, cell=cell_of(tag),
                       split_hash=spec.snapshot_digest)
    if ledger.closed:
        raise CellRejected("already_closed", f"이미 닫힌 원장이다: {led}")
    ledger.close(client_id=client, step=int(meta["adapter_step"]), n_train_samples=int(meta["identity"]["n_train_rows"]))
    return int(meta["adapter_step"])
