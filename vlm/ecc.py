"""조기 좌표 점검(`early_coord_check`, ECC)의 학습 쪽 — 11번 미니스펙 §1 ~ §5 · §10, 평가 쪽 10번 §14-3 · §14-5.

점검 실행은 **본실험 중앙 칸 자체**를 공통 초기 어댑터에서 학습하다가 등록한 점검 step 에서 세우는 **버리는 실행**이다.

- **멈춤**(`CheckStop`) — `vlm.pilot_vlm.train_rounds` 가 최적화 스텝마다 부른다. 등록한 step 이면 그 스텝의 최적화가 끝난 뒤
  step 어댑터 · 재개 상태 · 소비한 행 목록 · meta 를 쓰고(meta 가 마지막 — 완료 표지), `stop=True` 면 `CheckStepReached` 로 멈춘다.
- **epoch 중간 재개** — 재개 상태에 epoch · 그 epoch 순열의 다음 표본 커서 · 순열 지문 · 누적 스텝 · 누적 감독 토큰 · 최적화기 ·
  난수를 싣는다. 수치 궤적의 동치는 요구하지 않지만 **소비 행 수열 · 누적 스텝 · 누적 감독 토큰 · 비용 누적은 같아야 한다**(T13).
- **원장**(`ecc_ledger.csv`) — 시작 · step · 재개 · 멈춤 · 비용 가드 줄. 비용 누적(`cum_wall_s`)은 앞 줄에서 이어 가고 재개로 줄지 않는다.
- **비용 가드**(`budget_check`) — 다음 일을 시작하기 전에 `쓴 시간 + 다음 일의 예상` 이 상한을 넘으면 시작하지 않는다. 상한의
  정본은 평가 쪽 규칙 파일의 `limits`(상한 · 단가 · 계수)이고 실행 인자는 그보다 좁히기만 한다 — 평가 쪽 `effective_limits` 를
  부른다. 금액은 시간 × 단가 × 계수다. 쓴 시간은 점검 원장의 누적 처리 시간보다 작을 수 없다. 단가를 코드 기본값 · 실행 인자로 두지 않는다.
- **결과별 동작**(`action_for`) — 평가 쪽 순차 결정(`decide` 의 결과: 통과 · 불통과 · 판정 불가 · 이어 간다)을 학습 쪽 다음
  동작으로 옮긴다. 결정(한 번 통과 · 두 step 연속의 보류)은 평가 쪽 몫이고, 학습 쪽은 판정 문자열을 따로 두지 않는다.
- **시도 기록**(`record_attempt`) — 끝난 점검 실행마다 `ecc_attempts.jsonl` 에 한 줄(루트는 지우지 않는다). 시작한 시도의
  수(`attempts_started`)는 점검 루트의 준비 기록으로 센다 — 규칙 파일의 허용 시도 수를 넘으면 새 루트를 만들지 않는다.
- **규칙 · 착수 허가**(`load_check_rules` · `check_permit`) — 평가 쪽 `load_rules` · `require_ready` · `verify_permit`(3판 서명
  `raw, *, expected_sha256, rules, expected, file_sha256_of` — 대조값은 출발 어댑터 · 착수 고정 · 점검 실행 신원 · 시드 번호 넷,
  프로세서 설정은 착수 고정이 묶는다)을 **부르는 자리**다. 평가 쪽 모듈이 없으면 점검도 본실험 칸도 시작하지 않는다(닫힌 쪽으로).
  학습 쪽은 부르기 전에 둘을 더 본다 — 규칙의 판별력 영수증 실물(경로 · 바이트 지문)과 통과 파일의 네 검사 산출 파일 실물(§119 I-04 · I-05).
- **착수 허가의 키**(`accept_permit` · `read_permit_key`) — 점검이 통과하면 평가 쪽 통과 파일을 평가 쪽 검증으로 받아 시드 폴더에
  사본과 키 기록(`ECC_PERMIT_KEY.json` — 통과 파일의 sha256 · 점검 실행 신원 · …)을 덮지 않고 쓴다. 본실험 칸은 그 키를 **통과
  파일 밖에서** 읽어 `expected_sha256` 으로 넘긴다(20b 의 I-1).

점검 실행의 체크포인트는 채점 대상이 아니다. 본실험 중앙은 점검 실행을 잇지 않고 공통 초기 어댑터부터 새로 돈다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ATTEMPTS_FILE",
    "ECC_PARENT",
    "EVAL_SIDE",
    "LEDGER_FILE",
    "PERMIT_KEY_FIELDS",
    "PERMIT_KEY_FILE",
    "PREPARE_FILE",
    "PURPOSE",
    "ROOT_RE",
    "RULE_PATH",
    "STEP_EXIT",
    "CheckStepReached",
    "CheckStop",
    "EccRefused",
    "accept_permit",
    "action_for",
    "append_ledger",
    "attempts_started",
    "budget_check",
    "check_permit",
    "consumed_ids",
    "cum_wall_s",
    "effective_limits",
    "epoch_order",
    "identity_sha256",
    "ledger_rows",
    "load_check_rules",
    "load_step_state",
    "perm_sha256",
    "read_permit_key",
    "record_attempt",
    "save_check_step",
    "step_paths",
]

PURPOSE = "early_coord_check"
#: 점검 루트의 부모 — 평가 쪽 10번 §8. 본실험 출력(`outputs/main_u/`)과 리허설 부모(`outputs/rehearsal_u/`) 밖이다.
ECC_PARENT = Path("outputs/early_check")
ROOT_RE = re.compile(r"ecc-[0-9]{8}T[0-9]{6}")
#: 점검 step 도달의 종료 코드(11번 §3-2).
STEP_EXIT = 4
LEDGER_FILE = "ecc_ledger.csv"
ATTEMPTS_FILE = "ecc_attempts.jsonl"
#: 점검 루트의 준비 기록(`scripts/main_uni.py ecc-prepare`) — 시도 수를 세는 단위다.
PREPARE_FILE = "ECC_PREPARE.json"
#: 평가 쪽 규칙 파일(10번 §3). 읽는 쪽은 평가 쪽 `load_rules` 하나이고, 학습 쪽은 그것을 부른다.
RULE_PATH = Path("configs/early_coord_check.yaml")
#: 평가 쪽 점검 모듈. 학습 쪽은 규칙 읽기 · 착수 허가 대조 · 결정 문자열을 여기서만 가져온다.
EVAL_SIDE = "evaluation.early_coord_check"
#: 시드 폴더의 착수 허가 키 기록 — 통과 파일 사본과 나란히 둔다. 덮지 않는다.
PERMIT_KEY_FILE = "ECC_PERMIT_KEY.json"
PERMIT_KEY_FIELDS = ("permit_sha256", "check_run_id", "seed", "pin_sha256", "start_adapter_digest",
                     "rules_sha256", "attempt", "ecc_root", "accepted_at")
CLOCK_FILE = "step_clock.csv"
LEDGER_FIELDS = ("at", "event", "step", "epoch", "cursor", "optimizer_steps", "supervised_tokens", "consumed_rows",
                 "consumed_sha256", "adapter_sha256", "cum_wall_s", "detail")


class EccRefused(ValueError):
    """점검 실행의 가드 — 시작 · 재개 · export 를 하지 않는다."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


class CheckStepReached(Exception):
    """등록한 점검 step 에 닿았다 — 파일을 다 쓴 뒤에 올린다. 실행기는 종료 코드 4 로 바꾼다."""

    def __init__(self, step: int, meta_path: Path):
        self.step = int(step)
        self.meta_path = Path(meta_path)
        super().__init__(f"점검 step {step} 에 닿았다: {meta_path.name}")


@dataclass(frozen=True)
class CheckStop:
    """학습 루프에 넘기는 멈춤 지정. `stop=False` 는 파일만 쓰고 잇는다(관찰 — 시험의 끊지 않은 실행)."""

    steps: tuple[int, ...]
    out_dir: Path
    stop: bool = True
    resume_from: Path | None = None
    pin_sha256: str = ""
    #: 이 프로세스 앞의 비용 누적(초) — 원장의 마지막 줄에서 읽는다. 재개로 처음으로 돌아가지 않는다.
    prior_wall_s: float = 0.0
    #: 학습 설정 원문(`vlm.train_cell.train_config_object`)의 설정 쪽 인자 — 점검 step 의 meta 에 원문을 싣는다(점검 export 의 곁 파일).
    train_config_args: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        s = tuple(self.steps)
        if not s or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in s) \
                or list(s) != sorted(set(s)):
            raise EccRefused("check_steps", f"점검 step 목록은 겹치지 않는 양의 정수의 오름차순이어야 한다: {list(s)}")


# ---------------------------------------------------------------- 순열 · 소비 행
def epoch_order(n: int, seed: int, ep: int) -> list[int]:
    """학습 루프와 같은 식의 epoch 순열 — `random.Random(f"{seed}:{ep}")` 로 행 색인을 섞는다(`vlm/pilot_vlm.py`)."""
    order = list(range(int(n)))
    random.Random(f"{seed}:{ep}").shuffle(order)
    return order


def perm_sha256(order: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(i) for i in order], separators=(",", ":")).encode("utf-8")).hexdigest()


def consumed_ids(rows: Sequence[Mapping[str, Any]], seed: int, ep: int, cursor: int) -> list[str]:
    """지금까지 학습에 나온 행의 `image_id` 수열 — 앞 epoch 전부 + 이 epoch 순열의 앞 `cursor` 개."""
    out: list[str] = []
    n = len(rows)
    for e in range(int(ep)):
        out.extend(str(rows[i]["image_id"]) for i in epoch_order(n, seed, e))
    out.extend(str(rows[i]["image_id"]) for i in epoch_order(n, seed, ep)[:int(cursor)])
    return out


def identity_sha256(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(identity), sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 파일
def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_new(path: Path, write: Callable) -> str:
    """덮지 않는 원자 쓰기 — tmp(배타) → fsync → 제 이름. 쓴 파일의 sha256."""
    p = Path(path)
    if p.exists():
        raise EccRefused("step_file_exists", f"점검 step 파일이 이미 있다 — 덮지 않는다: {p.name}")
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "xb") as fh:
        write(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return _sha_file(p)


def step_paths(out_dir: Path, step: int) -> dict[str, Path]:
    d = Path(out_dir)
    return {"adapter": d / f"adapter_step{step:06d}.npz", "state": d / f"resume_step{step:06d}.pt",
            "consumed": d / f"consumed_upto_s{step:06d}.txt", "meta": d / f"step{step:06d}.meta.json"}


def append_ledger(out_dir: Path, row: Mapping[str, Any]) -> None:
    p = Path(out_dir) / LEDGER_FILE
    new = not p.exists()
    with p.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in LEDGER_FIELDS})
        fh.flush()
        os.fsync(fh.fileno())


def ledger_rows(out_dir: Path) -> list[dict[str, str]]:
    p = Path(out_dir) / LEDGER_FILE
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def cum_wall_s(out_dir: Path) -> float:
    """원장의 마지막 비용 누적(초). 없으면 0."""
    vals = [float(r["cum_wall_s"]) for r in ledger_rows(out_dir) if r.get("cum_wall_s")]
    return max(vals) if vals else 0.0


def save_check_step(cs: CheckStop, *, step: int, epoch: int, cursor: int, order: Sequence[int],
                    rows: Sequence[Mapping[str, Any]], seed: int, arrays: Sequence[np.ndarray], keys: Sequence[str],
                    state: Mapping[str, Any], identity: Mapping[str, Any], supervised_tokens: int,
                    process_wall_s: float, extra: Mapping[str, Any] | None = None) -> Path:
    """한 점검 step 의 파일 넷 — 어댑터 · 재개 상태 · 소비 행 목록 · meta(마지막). meta 경로를 돌려준다."""
    import torch

    out = Path(cs.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = step_paths(out, step)
    if perm_sha256(order) != perm_sha256(epoch_order(len(rows), seed, epoch)):
        raise EccRefused("perm_formula", "학습 루프의 순열이 점검 쪽 식과 다르다 — 소비 행 목록을 낼 수 없다")
    ids = consumed_ids(rows, seed, epoch, cursor)
    consumed_raw = ("\n".join(ids) + "\n").encode("utf-8") if ids else b""
    adapter_sha = _write_new(paths["adapter"], lambda fh: np.savez(fh, **{k: a for k, a in zip(keys, arrays)}))
    state_sha = _write_new(paths["state"], lambda fh: torch.save(dict(state), fh))
    consumed_sha = _write_new(paths["consumed"], lambda fh: fh.write(consumed_raw))
    cum = float(cs.prior_wall_s) + float(process_wall_s)
    meta = {"purpose": PURPOSE, "step": int(step), "epoch": int(epoch), "cursor": int(cursor),
            "perm_sha256": perm_sha256(order), "optimizer_steps": int(step), "supervised_tokens": int(supervised_tokens),
            "identity": dict(identity), "identity_sha256": identity_sha256(identity), "pin_sha256": cs.pin_sha256,
            "registered_steps": list(cs.steps), "cum_wall_s": round(cum, 6),
            "adapter": {"file": paths["adapter"].name, "sha256": adapter_sha, "keys": list(keys)},
            "state": {"file": paths["state"].name, "sha256": state_sha},
            "consumed": {"file": paths["consumed"].name, "sha256": consumed_sha, "rows": len(ids),
                         "unique_rows": len(set(ids))},
            **dict(extra or {})}
    _write_new(paths["meta"], lambda fh: fh.write(json.dumps(meta, ensure_ascii=False, sort_keys=True,
                                                             indent=1).encode("utf-8") + b"\n"))
    append_ledger(out, {"at": time.time(), "event": "step", "step": step, "epoch": epoch, "cursor": cursor,
                        "optimizer_steps": step, "supervised_tokens": supervised_tokens, "consumed_rows": len(ids),
                        "consumed_sha256": consumed_sha, "adapter_sha256": adapter_sha, "cum_wall_s": round(cum, 6)})
    return paths["meta"]


def load_step_state(meta_path: Path, *, identity: Mapping[str, Any], pin_sha256: str,
                    steps: Sequence[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    """재개할 step 의 (meta, 상태). 적용하기 **전에** 신원 · 착수 고정 · 파일 지문 · step 목록을 맞댄다."""
    import torch

    mp = Path(meta_path)
    if not mp.is_file():
        raise EccRefused("resume_missing", f"재개할 점검 step 의 meta 가 없다: {mp.name}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if meta.get("purpose") != PURPOSE:
        raise EccRefused("resume_purpose", "점검 실행의 step 이 아니다")
    if meta.get("identity_sha256") != identity_sha256(identity) or meta.get("identity") != dict(identity):
        raise EccRefused("resume_identity", "재개할 step 의 신원이 지금 실행과 다르다 — 적용하지 않는다")
    if meta.get("pin_sha256") != pin_sha256:
        raise EccRefused("pin_mismatch", "재개할 step 의 착수 고정이 지금과 다르다")
    if list(meta.get("registered_steps") or []) != list(steps):
        raise EccRefused("resume_steps", "재개할 step 의 등록 step 목록이 지금과 다르다")
    if int(meta["step"]) not in steps or int(meta["step"]) >= max(steps):
        raise EccRefused("resume_step", f"step {meta['step']} 에서는 이어 갈 다음 점검 step 이 없다")
    d = mp.parent
    for part in ("adapter", "state", "consumed"):
        f = d / meta[part]["file"]
        if not f.is_file() or _sha_file(f) != meta[part]["sha256"]:
            raise EccRefused("resume_file", f"점검 step 의 {part} 파일이 없거나 meta 의 지문과 다르다")
    state = torch.load(d / meta["state"]["file"], map_location="cpu", weights_only=False)
    for key in ("epoch", "cursor", "perm_sha256", "optimizer_steps"):
        if state.get(key) != meta.get(key):
            raise EccRefused("resume_state", f"재개 상태의 {key} 가 meta 와 다르다")
    return meta, state


# ---------------------------------------------------------------- 비용 · 결과 · 시도
def effective_limits(rules: Any, *, max_hours: float | None = None, max_won: float | None = None) -> dict[str, Any]:
    """점검 실행의 상한과 금액 단가 — 평가 쪽 `effective_limits(rules, max_hours=…, max_cost_krw=…)` 를 부른다(3판, 13-2 의 정본).

    실행 인자는 규칙 파일의 상한을 **좁히기만** 한다(넓히면 `ecc_budget_widen`). 돌려주는 `cost_krw_of_hour` 는 단가 × 계수다 —
    학습 쪽은 단가를 따로 두지 않는다. 실행 인자는 유한 양수여야 한다 — NaN · ±무한 · 음수 · bool 은 받지 않는다(§119 I-09).
    """
    for name, v in (("max_hours", max_hours), ("max_won", max_won)):
        if v is not None and not _finite(v, positive=True):
            raise EccRefused("ecc_budget_args", f"실행 인자 {name} 은 유한 양수여야 한다: {v!r}")
    D = _eval_side()
    try:
        return dict(D.effective_limits(rules, max_hours=max_hours, max_cost_krw=max_won))
    except D.EarlyCheckRejected as exc:
        code = "ecc_budget_widen" if getattr(exc, "code", "") == "LIMITS_WIDENED" else "ecc_rules_rejected"
        raise EccRefused(code, str(exc)) from None


def budget_check(out_dir: Path, *, spent_hours: float, next_hours: float, max_hours: float | None,
                 max_won: float | None, won_per_hour: float, what: str,
                 price_detail: Mapping[str, Any] | None = None) -> None:
    """다음 일(`what` — 재개 · export)을 시작하기 전. 넘으면 원장에 세 수를 적고 `EccRefused("ecc_budget_exceeded")`.

    쓴 시간(`spent_hours`)은 운영자가 준다(인스턴스가 켜져 있던 시간). 점검 원장의 누적 처리 시간보다 작으면 받지 않는다(20b 의 m-3).
    `won_per_hour` 는 평가 쪽 `effective_limits` 의 `cost_krw_of_hour`(단가 × 계수)다 — 기본값이 없다(20b 의 I-2).
    `price_detail`(단가 · 계수 · 출처)은 원장 줄에 함께 적는다.
    """
    if won_per_hour is None or not _finite(won_per_hour, positive=True):
        raise EccRefused("ecc_rate_missing", "시간당 금액 단가가 없거나 유한 양수가 아니다 — 규칙 파일의 limits 에서 온다")
    for name, v in (("spent_hours", spent_hours), ("next_hours", next_hours)):
        if not _finite(v, positive=False):
            raise EccRefused("ecc_budget_args", f"{name} 은 유한 비음수여야 한다: {v!r}")
    for name, v in (("max_hours", max_hours), ("max_won", max_won)):
        if v is not None and not _finite(v, positive=True):
            raise EccRefused("ecc_budget_args", f"{name} 은 유한 양수여야 한다: {v!r}")
    ledger_s = cum_wall_s(out_dir)
    if float(spent_hours) * 3600 < ledger_s:
        raise EccRefused("ecc_spent_below_ledger",
                         f"쓴 시간 {spent_hours:.3f} h 가 점검 원장의 누적 처리 시간 {ledger_s / 3600:.3f} h 보다 작다")
    total_h = float(spent_hours) + float(next_hours)
    over_h = max_hours is not None and total_h > float(max_hours)
    over_w = max_won is not None and total_h * float(won_per_hour) > float(max_won)
    detail = {"what": what, "spent_hours": round(float(spent_hours), 4), "next_hours": round(float(next_hours), 4),
              "max_hours": max_hours, "max_won": max_won, "won_per_hour": won_per_hour, **dict(price_detail or {})}
    if over_h or over_w:
        append_ledger(out_dir, {"at": time.time(), "event": "ecc_budget_exceeded",
                                "detail": json.dumps(detail, ensure_ascii=False, sort_keys=True)})
        raise EccRefused("ecc_budget_exceeded", f"{what}: 쓴 {spent_hours:.3f} h + 예상 {next_hours:.3f} h 가 상한을 넘는다")
    append_ledger(out_dir, {"at": time.time(), "event": "budget_ok",
                            "detail": json.dumps(detail, ensure_ascii=False, sort_keys=True)})


def _finite(v: Any, *, positive: bool) -> bool:
    """bool 이 아닌 유한 실수. `positive` 면 0 보다 커야, 아니면 0 이상이어야 한다."""
    import math

    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return False
    return v > 0 if positive else v >= 0


def _permit_files_problems(raw: bytes, file_sha256_of: Callable[[Any], str | None]) -> list[str]:
    """통과 파일의 네 검사 산출 파일 — 비지 않은 경로 · 64 자리 16진 sha · 실물이 지금 있어야 한다(§119 I-04). 평가 쪽 `verify_permit`
    은 지문 콜백의 None 과 빈 기대값을 같다고 볼 수 있으므로, 학습 쪽이 부르기 전에 먼저 본다."""
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []                                                   # 꼴 오류는 평가 쪽이 PERMIT_FORMAT 으로 거부한다
    files = ((obj.get("evidence") or {}) if isinstance(obj, dict) else {}).get("model_free_files")
    if not isinstance(files, dict):
        return ["model_free_files 가 사전이 아니다"]
    out = []
    for name, f in sorted(files.items()):
        path, sha = (f.get("path"), f.get("sha256")) if isinstance(f, dict) else (None, None)
        if not isinstance(path, str) or not path:
            out.append(f"{name}: 경로가 비었다")
        elif not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
            out.append(f"{name}: sha256 이 64 자리 16진이 아니다")
        elif file_sha256_of(path) is None:
            out.append(f"{name}: 산출 파일이 지금 없다")
    return out


def _eval_side() -> Any:
    """평가 쪽 점검 모듈. 없으면 점검도 착수 허가도 열지 않는다(닫힌 쪽으로). 시험은 이 함수를 바꿔 끼운다."""
    import importlib

    try:
        return importlib.import_module(EVAL_SIDE)
    except ImportError:
        raise EccRefused("ecc_eval_side_missing", f"평가 쪽 점검 모듈({EVAL_SIDE})이 없다 — 시작하지 않는다") from None


#: 학습 쪽 다음 동작 — 평가 쪽 결정의 넷에 하나씩.
ACTIONS = ("end_pass", "stop_fail", "stop_undecidable", "continue")


def action_for(decision: Mapping[str, Any], *, step: int, steps: Sequence[int]) -> str:
    """평가 쪽 순차 결정(`evaluation.early_coord_check.decide` 의 반환) → 학습 쪽 다음 동작.

    통과 `end_pass`(점검 실행을 끝내고 본 중앙은 공통 출발점에서 새로) · 불통과 `stop_fail`(모든 학습을 멈춘다) ·
    판정 불가 `stop_undecidable`(멈추고 보고) · 이어 간다 `continue`(다음 점검 step 까지 재개). 결정의 step 자리가 지금
    step 과 다르거나, 마지막 step 뒤에 잇자는 결정이면 거부한다.
    """
    D = _eval_side()
    table = {D.PASS: "end_pass", D.FAIL: "stop_fail", D.INDETERMINATE: "stop_undecidable", D.CONTINUE: "continue"}
    out = decision.get("outcome")
    if out not in table:
        raise EccRefused("decision", f"평가 쪽 결정의 값이 아니다: {out!r}")
    steps = tuple(int(s) for s in steps)
    if int(step) not in steps:
        raise EccRefused("decision_step", f"step {step} 은 등록한 점검 step 이 아니다")
    idx = decision.get("step_index")
    if idx is not None and (type(idx) is not int or not 0 <= idx < len(steps) or steps[idx] != int(step)):
        raise EccRefused("decision_step", f"결정의 step 자리 {idx!r} 가 지금 step {step} 과 다르다")
    act = table[out]
    if act == "continue" and int(step) == max(steps):
        raise EccRefused("decision_step", "마지막 점검 step 뒤에는 이을 step 이 없다")
    return act


def attempts_started(parent: Path, pin_sha256: str) -> int:
    """이 착수 고정(= 이 시드)으로 준비한 점검 루트의 수 — 부모 바로 아래 루트의 준비 기록으로 센다."""
    n = 0
    for root in sorted(Path(parent).glob("ecc-*")):
        rec = root / PREPARE_FILE
        if (ROOT_RE.fullmatch(root.name) and rec.is_file()
                and json.loads(rec.read_text(encoding="utf-8")).get("pin_sha256") == pin_sha256):
            n += 1
    return n


def record_attempt(root_parent: Path, row: Mapping[str, Any]) -> None:
    """끝난 점검 실행의 한 줄 — 실행 신원 · 착수 고정 해시 · 마지막 step · 소요 · 사유. 루트는 지우지 않는다."""
    need = ("run_id", "pin_sha256", "last_step", "hours", "won", "reason")
    missing = [k for k in need if k not in row]
    if missing:
        raise EccRefused("attempt_fields", f"시도 기록의 칸이 빠졌다: {missing}")
    p = Path(root_parent) / ATTEMPTS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_check_rules(rule_path: Path | None = None) -> dict[str, Any]:
    """평가 쪽 규칙 파일 → 학습 쪽이 쓰는 것: 점검 step 목록 · 허용 시도 수 · 상한(`limits`) · 규칙 바이트의 sha256(평가 쪽
    `raw_sha256`) · 평가 쪽 규칙 객체(`rules` — `verify_permit` 에 그대로 넘긴다).

    평가 쪽 `load_rules`(꼴 · 종류 · 판 · 검사기 지문)와 `require_ready`(순차 규칙 · 허용 시도 수 · 상한이 정해졌는가)를 부른다 —
    학습 쪽은 규칙 파일을 따로 해석하지 않는다.
    """
    D = _eval_side()
    p = Path(rule_path) if rule_path is not None else _repo_root() / RULE_PATH
    if not p.is_file():
        raise EccRefused("ecc_rules_missing", f"조기 좌표 점검의 규칙 파일이 없다: {p.name}")
    raw = p.read_bytes()
    try:
        rules = D.load_rules(raw)
        D.require_ready(rules)
    except D.EarlyCheckRejected as exc:
        raise EccRefused("ecc_rules_rejected", str(exc)) from None
    disc = getattr(rules, "discrimination", None)
    if isinstance(disc, dict):                                      # §119 I-05 — 영수증의 경로 · 바이트 지문을 실물과 맞댄다
        q = Path(str(disc.get("path") or ""))
        q = q if q.is_absolute() else _repo_root() / q
        if not disc.get("path") or not q.is_file():
            raise EccRefused("ecc_discrimination_missing", f"규칙의 판별력 영수증이 없다: {disc.get('path')!r}")
        if hashlib.sha256(q.read_bytes()).hexdigest() != str(disc.get("sha256", "")).lower():
            raise EccRefused("ecc_discrimination_changed", "규칙의 판별력 영수증 바이트가 적힌 sha256 과 다르다")
    steps = tuple(int(s) for s in rules.steps)
    CheckStop(steps=steps, out_dir=Path("."))                  # 꼴 검사(양의 정수 · 오름차순 · 겹침 없음)
    return {"steps": steps, "max_attempts": int(rules.max_attempts), "rules_sha256": str(rules.raw_sha256),
            "limits": dict(rules.limits), "rules": rules}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_lf_sha256(path: Any) -> str | None:
    """평가 쪽 통과 파일이 싣는 네 검사 산출 파일의 지금 지문(LF 정규화 · 저장소 기준 상대 경로). 없으면 None — 어긋남으로 센다."""
    if not isinstance(path, str) or not path:
        return None
    q = Path(path)
    q = q if q.is_absolute() else _repo_root() / q
    if not q.is_file():
        return None
    return hashlib.sha256(q.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_permit(permit_path: Path | None, *, expected_sha256: str, rules: Any, expected: Mapping[str, Any],
                 file_sha256_of: Callable[[Any], str | None] = _repo_lf_sha256) -> str:
    """네 칸 · 통과 뒤 본 중앙의 착수 허가 — 평가 쪽 `verify_permit(raw, *, expected_sha256, rules, expected, file_sha256_of)` 를
    **부른다**(10번 §14-3, 평가 쪽 2판). 허가 파일의 sha256 을 돌려준다.

    `expected_sha256` 은 통과 파일 **밖**의 키(`ECC_PERMIT_KEY.json`)에서 오고, `rules` 는 평가 쪽 `load_rules` 의 객체,
    `expected` 는 이 칸이 잰 값(출발 어댑터 · 프로세서 설정 · 착수 고정 · 점검 실행 신원 · 시드 번호)이다. 검증 함수는 평가 쪽
    소관이다 — 없으면 시작하지 않는다. 학습 쪽이 판정을 대신하지 않는다.
    """
    if permit_path is None:
        raise EccRefused("ecc_permit_missing", "조기 좌표 점검의 통과 파일이 주어지지 않았다 — 네 칸을 열지 않는다")
    p = Path(permit_path)
    if not p.is_file():
        raise EccRefused("ecc_permit_missing", f"조기 좌표 점검의 통과 파일이 없다: {p.name}")
    D = _eval_side()
    raw = p.read_bytes()
    bad = _permit_files_problems(raw, file_sha256_of)
    if bad:
        raise EccRefused("ecc_permit_files", "통과 파일의 네 검사 산출 파일이 서지 않는다 — " + " · ".join(bad))
    try:
        D.verify_permit(raw, expected_sha256=expected_sha256, rules=rules, expected=dict(expected),
                        file_sha256_of=file_sha256_of)
    except D.EarlyCheckRejected as exc:
        raise EccRefused("ecc_permit_rejected", str(exc)) from None
    return hashlib.sha256(raw).hexdigest()


def _write_bytes_once(path: Path, data: bytes) -> None:
    """덮지 않는 쓰기 — 같은 바이트가 이미 있으면 그대로 둔다(중간에 멈춘 받기를 다시 잇게), 다르면 거부."""
    if path.exists():
        if path.read_bytes() != data:
            raise EccRefused("ecc_permit_exists", f"{path.name} 가 이미 있고 바이트가 다르다 — 덮지 않는다")
        return
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def accept_permit(permit_src: Path, *, rules: Any, expected: Mapping[str, Any], seed_dir: Path, permit_name: str,
                  attempts_parent: Path, attempt_row: Mapping[str, Any], extra: Mapping[str, Any],
                  file_sha256_of: Callable[[Any], str | None] = _repo_lf_sha256) -> dict[str, Any]:
    """점검 통과 뒤 착수 허가의 키를 세운다 — 평가 쪽 통과 파일을 평가 쪽 `verify_permit` 으로 받고(이 바이트의 sha256 을 키로),
    시드 폴더에 사본(`permit_name`)과 키 기록(`PERMIT_KEY_FILE`, 마지막 — 완료 표지)을 덮지 않고 쓴 뒤 시도 기록에 한 줄 덧붙인다.

    `expected` 는 **점검 실행 쪽에서 잰** 값이다(통과 파일에서 읽지 않는다). 키 기록이 이미 있으면 받지 않는다 — 한 시드에 한 번.
    """
    key_path = Path(seed_dir) / PERMIT_KEY_FILE
    if key_path.exists():
        raise EccRefused("ecc_permit_exists", "이 시드의 착수 허가 키가 이미 있다 — 덮지 않는다")
    raw = Path(permit_src).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    D = _eval_side()
    bad = _permit_files_problems(raw, file_sha256_of)
    if bad:
        raise EccRefused("ecc_permit_files", "통과 파일의 네 검사 산출 파일이 서지 않는다 — " + " · ".join(bad))
    try:
        D.verify_permit(raw, expected_sha256=sha, rules=rules, expected=dict(expected), file_sha256_of=file_sha256_of)
    except D.EarlyCheckRejected as exc:
        raise EccRefused("ecc_permit_rejected", str(exc)) from None
    key = {"permit_sha256": sha, **{k: expected[k] for k in ("check_run_id", "seed", "pin_sha256", "start_adapter_digest")},
           "rules_sha256": str(rules.raw_sha256), **dict(extra), "accepted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    missing = [k for k in PERMIT_KEY_FIELDS if k not in key]
    if missing:
        raise EccRefused("ecc_permit_key_fields", f"키 기록의 칸이 빠졌다: {missing}")
    Path(seed_dir).mkdir(parents=True, exist_ok=True)
    _write_bytes_once(Path(seed_dir) / permit_name, raw)
    _write_bytes_once(key_path, (json.dumps(key, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode("utf-8"))
    record_attempt(attempts_parent, {**dict(attempt_row), "reason": "pass", "permit_sha256": sha})
    return key


def read_permit_key(seed_dir: Path) -> dict[str, Any]:
    """시드 폴더의 착수 허가 키 기록. 없거나 칸이 빠지면 본실험 칸을 열지 않는다."""
    p = Path(seed_dir) / PERMIT_KEY_FILE
    if not p.is_file():
        raise EccRefused("ecc_permit_key_missing", f"착수 허가의 키 기록이 없다: {p.name} — 점검 통과 뒤 ecc-accept 가 쓴다")
    key = json.loads(p.read_text(encoding="utf-8"))
    missing = [k for k in PERMIT_KEY_FIELDS if k not in key]
    if missing:
        raise EccRefused("ecc_permit_key_fields", f"키 기록의 칸이 빠졌다: {missing}")
    return key


# ---------------------------------------------------------------- 점검 실행(중앙 칸 자체)
def main_config_problems(spec: Any) -> list[str]:
    """점검 실행의 설정 대조 — **본실험 중앙과 같은 값**이어야 한다(`vlm.uni_config.main_run_problems` · 본실험 필수 칸).

    출력 경로는 여기서 보지 않는다 — 점검 루트(`outputs/early_check/`)는 본실험 출력 금지 부모라 `main_run_problems` 가 거부한다.
    루트는 `run_root_problems` 가 따로 본다.
    """
    from vlm.train_cell import MAIN_REQUIRED
    from vlm.uni_config import load_uni_config, main_run_problems

    empty = [k for k in MAIN_REQUIRED if getattr(spec, k) in (None, "")]
    if empty:
        return [f"본실험 중앙과 같은 설정이 비었다: {empty}"]
    return main_run_problems(
        load_uni_config(spec.config_path), model_id=spec.model_id, model_revision=spec.model_revision,
        pairs_path=spec.pairs_path, pairs_digest=spec.pairs_digest, chat_template_kwargs=spec.chat_template_kwargs,
        num_rounds=spec.num_rounds, local_epochs=spec.local_epochs, total_epochs=spec.total_epochs,
        seed_index=spec.seed_index, seed_value=spec.seed_value, snapshot_digest=spec.snapshot_digest,
        prompt_path=spec.prompt_path, coord_space=spec.coord_space, processor_kwargs=spec.processor_kwargs,
        expected_supervised_tokens=spec.expected_supervised_tokens, output_paths=[])


def run_ecc_train(spec: Any, *, steps: Sequence[int], registered_steps: Sequence[int], pin_path: Path,
                  resume_step: int | None = None, model_loader: Callable | None = None,
                  pin_probes: Mapping[str, Callable[[], dict[str, Any]]] | None = None,
                  config_check: Callable[[Any], list[str]] | None = None, rows: list[dict] | None = None,
                  budget: Mapping[str, Any] | None = None, stop: bool = True) -> dict[str, Any]:
    """점검 학습 — 공통 초기 어댑터에서 본실험 중앙의 학습 행 · 순서 · 설정 그대로 학습하고 다음 점검 step 에서 멈춘다.

    `resume_step` 이 있으면 그 step 의 재개 상태에서 **다음 점검 step 까지** 잇는다(판정 불가 뒤). 돌려주는 사전의 `exit_code` 는
    점검 step 도달이면 4, 등록 예산을 끝까지 돌았으면 0 이다. 가드에 걸리면 `EccRefused`(아무것도 학습하지 않는다).

    `model_loader` · `pin_probes` · `config_check` · `rows` 는 시험의 이음새다(파이썬 인자로만). `rows` 는 대역을 허용한 실행에서만
    받는다 — 본 점검은 페어의 학습 분할 전체(본실험 중앙과 같은 행)로 돈다. `budget` 은 `spent_hours` · `next_hours` · `max_hours` ·
    `max_won` · `won_per_hour`(단가 × 계수) · `price_detail` 이다(11번 §5 — 인스턴스가 켜져 있던 시간으로 센다. 상한과 단가는
    평가 쪽 `effective_limits` 가 규칙 파일에서 낸다).
    """
    from vlm.coords import CoordCfg
    from vlm.pilot_vlm import load_pairs, resolve_model_loader, train_rounds
    from vlm.pin import PinMismatch, build_pin, check_pin
    from vlm.run_root import run_root_problems
    from vlm.train_cell import CENTRAL_TAG, _load_init, cell_dir, cell_run_id, client_of

    if spec.purpose != PURPOSE:
        raise EccRefused("purpose", f"점검 실행은 purpose={PURPOSE!r} 다: {spec.purpose!r}")
    steps = tuple(int(s) for s in steps)
    CheckStop(steps=steps, out_dir=Path("."))
    if steps != tuple(int(s) for s in registered_steps):
        raise EccRefused("steps_not_registered", f"점검 step {list(steps)} 이 규칙 파일의 {list(registered_steps)} 와 다르다")
    resolve_model_loader(model_loader, purpose=PURPOSE, standin_allowed=spec.standin_allowed)       # 이음새 — 부르기 전
    out = cell_dir(spec, CENTRAL_TAG)
    bad = run_root_problems(PURPOSE, spec.run_root, [spec.train_root, spec.init_adapter_path, Path(pin_path)],
                            parent=spec.non_main_parent)
    if bad:
        raise EccRefused("run_root", " · ".join(bad))
    if spec.resume_root is not None:
        raise EccRefused("resume_root", "점검 실행은 epoch 경계 재개 폴더를 쓰지 않는다 — 점검 step 의 재개 상태로 잇는다")
    problems = (config_check or main_config_problems)(spec)
    if problems:
        raise EccRefused("config_mismatch", "본실험 중앙과 다른 설정이다: " + " · ".join(problems))
    try:                                                                                         # 착수 고정 — 시작 · 재개마다
        pin_sha = check_pin(Path(pin_path), build_pin(init_adapter_path=spec.init_adapter_path,
                                                      config_path=spec.config_path, probes=pin_probes))
    except PinMismatch as exc:
        raise EccRefused("pin_mismatch", str(exc)) from None
    resume_meta = None
    done = sorted(int(p.name[4:10]) for p in out.glob("step*.meta.json")) if out.exists() else []
    if resume_step is None:
        if done:
            raise EccRefused("ecc_run_exists", f"이 점검 실행에 이미 step {done} 이 있다 — 새 점검 실행은 새 루트로 연다")
    else:
        if int(resume_step) not in steps[:-1]:
            raise EccRefused("resume_step", f"step {resume_step} 에서 이어 갈 다음 점검 step 이 없다")
        if done != [s for s in steps if s <= int(resume_step)]:
            raise EccRefused("resume_order", f"이어 갈 step {resume_step} 앞뒤의 점검 step 파일이 등록 순서와 맞지 않는다: {done}")
        resume_meta = step_paths(out, int(resume_step))["meta"]
    out.mkdir(parents=True, exist_ok=True)
    if budget is not None:
        budget_check(out, spent_hours=float(budget["spent_hours"]), next_hours=float(budget["next_hours"]),
                     max_hours=budget.get("max_hours"), max_won=budget.get("max_won"),
                     won_per_hour=budget.get("won_per_hour"), what="resume" if resume_step else "start",
                     price_detail=budget.get("price_detail"))
    init_arrays, init_keys, init_info = _load_init(spec, model_loader)
    if rows is not None and not spec.standin_allowed:
        raise EccRefused("rows_override", "본 점검은 페어의 학습 분할 전체로 돈다 — 학습 행 목록을 받지 않는다")
    rows = list(rows) if rows is not None else load_pairs("train", None, pairs_path=spec.pairs_path)
    prior = cum_wall_s(out)
    append_ledger(out, {"at": time.time(), "event": "resume" if resume_step else "start", "step": resume_step,
                        "cum_wall_s": round(prior, 6),
                        "detail": json.dumps({"pin_sha256": pin_sha, "steps": list(steps),
                                              "init_adapter_digest": init_info["init_adapter_digest"]}, sort_keys=True)})
    from vlm.pilot_vlm import load_prompt

    _, prompt_sha = load_prompt(spec.prompt_path)
    tc_args = {"model_id": spec.model_id, "model_revision": spec.model_revision, "pairs_digest": spec.pairs_digest,
               "chat_template_kwargs": dict(spec.chat_template_kwargs), "coord_space": spec.coord_space,
               "num_rounds": spec.num_rounds, "local_epochs": spec.local_epochs, "total_epochs": spec.total_epochs,
               "prompt_sha256": prompt_sha, "init_digest": init_info["init_adapter_digest"]}
    cs = CheckStop(steps=steps, out_dir=out, stop=stop, resume_from=resume_meta, pin_sha256=pin_sha,
                   prior_wall_s=prior, train_config_args=tc_args)
    try:
        arrays, _, m, _ = train_rounds(
            rows=rows, epochs=spec.total_epochs, round_idx=0, client_idx=0, base_seed=spec.seed_value,
            adapter_in=list(init_arrays), adapter_keys=list(init_keys), model_id=spec.model_id, resume_dir=None,
            run_id=cell_run_id(spec, CENTRAL_TAG), init_seed=spec.seed_value, num_rounds=1,
            model_revision=spec.model_revision, pairs_path=spec.pairs_path, prompt_path=spec.prompt_path,
            chat_template_kwargs=dict(spec.chat_template_kwargs), coord_cfg=CoordCfg(coord_space=spec.coord_space),
            processor_kwargs=spec.processor_kwargs, model_loader=model_loader, purpose=PURPOSE,
            standin_allowed=spec.standin_allowed, tag=CENTRAL_TAG, seed_index=spec.seed_index,
            pairs_digest=spec.pairs_digest, client_tag=client_of(CENTRAL_TAG), check_stop=cs)
    except CheckStepReached as hit:
        meta = json.loads(hit.meta_path.read_text(encoding="utf-8"))
        append_ledger(out, {"at": time.time(), "event": "stop", "step": hit.step, "cum_wall_s": meta["cum_wall_s"],
                            "detail": "check_step_reached"})
        return {"exit_code": STEP_EXIT, "step": hit.step, "meta": str(hit.meta_path), "pin_sha256": pin_sha,
                "init_adapter_digest": init_info["init_adapter_digest"], "cum_wall_s": meta["cum_wall_s"]}
    append_ledger(out, {"at": time.time(), "event": "complete", "optimizer_steps": m["optimizer_steps"],
                        "supervised_tokens": m["supervised_tokens"], "cum_wall_s": round(prior + float(m["wall_s"]), 6)})
    from vlm.init_adapter import init_adapter_digest

    return {"exit_code": 0, "step": None, "pin_sha256": pin_sha, "init_adapter_digest": init_info["init_adapter_digest"],
            "optimizer_steps": int(m["optimizer_steps"]), "metrics": m,
            "final_adapter_digest": init_adapter_digest(list(arrays))}        # 끝 어댑터의 원소까지 묶는 지문(T2)
