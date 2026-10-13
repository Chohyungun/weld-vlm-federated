"""본실험 등록값 — 데이터 쪽이 쓰는 `configs/train_registration.yaml` 을 읽어 코드 · 설정과 맞대고 착수 고정에 싣는다.

다섯 칸 공통 고정의 확정 수치는 그 파일 하나에 둔다(값을 코드에 적지 않는다 — 의사결정로그 2026-10-13 13:5x). 학습 쪽은

- 파일을 정확한 칸 집합으로 읽고(`load_registration`),
- 코드가 **실제로 쓰는 값**(마이크로 배치 · 누적 · 학습률 · 4bit 적재 인자 · 학습 레시피), 통합형 설정 블록(라운드 · 국소 epoch ·
  총 epoch · 생성 한도 · 시드 수), 조기 좌표 점검 규칙 파일(판 · 바이트 지문)과 맞대며(`registration_problems`),
- 어긋남이 없을 때만 착수 고정의 `common` 칸에 파일 지문과 값을 싣는다(`registration_probe` — `vlm.pin.common_probe` 가 부른다).

파일이 없거나 하나라도 어긋나면 착수 고정을 쓰지 않는다(`RegistrationRejected` — 닫힌 쪽). 본 칸은 시작 · 재개마다 착수 고정을
다시 재므로, 등록 파일이나 코드 · 설정이 바뀌면 그 자리에서 멈춘다.

데이터 쪽이 파일을 쓴 뒤 스스로 확인하는 명령(착수 고정을 쓰지 않는다 — 읽고 맞대기만):

    python -X utf8 -m vlm.registration [--config configs/base.yaml] [--path configs/train_registration.yaml]

맞으면 종료 0 과 파일 지문, 어긋나면 종료 2 와 어긋남을 줄마다 낸다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from vlm.pin import PinMismatch

__all__ = [
    "KEYS",
    "KIND",
    "QUANT_KEYS",
    "REG_PATH",
    "TRAIN_RECIPES",
    "RegistrationRejected",
    "load_registration",
    "registration_probe",
    "registration_problems",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
#: 등록 파일의 자리(데이터 쪽이 쓴다).
REG_PATH = Path("configs/train_registration.yaml")
KIND = "train_registration"
VERSION = 1
#: 등록 파일의 칸 — 정확히 이 집합(빠진 칸 · 모르는 칸은 거부).
KEYS = ("kind", "version", "seed_indices", "num_rounds", "local_epochs", "total_epochs", "recipe", "eval_gen_limit",
        "quantization", "micro_batch", "grad_accum", "lr", "early_coord_check_rules", "init_adapter_digest")
QUANT_KEYS = ("load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant")
#: 학습 코드가 구현한 레시피. 두 과제 혼합 · JSON 뒤 산문은 아직 학습 코드에 없다(02 2판은 명세다).
TRAIN_RECIPES = ("json_only",)


class RegistrationRejected(PinMismatch):
    """등록 파일이 없거나 코드 · 설정과 다르다 — 착수 고정을 쓰지 않는다."""

    code = "registration_rejected"

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__(["registration"], " · ".join(self.problems))


def load_registration(path: Path | None = None) -> tuple[dict[str, Any], str]:
    """(값, 파일 바이트의 sha256). 칸 집합 · 종류 · 판을 본다."""
    import yaml

    p = Path(path) if path is not None else REPO_ROOT / REG_PATH
    if not p.is_file():
        raise RegistrationRejected([f"등록 파일이 없다: {REG_PATH.as_posix()} — 데이터 쪽이 쓴다"])
    raw = p.read_bytes()
    body = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(body, dict) or body.get("kind") != KIND:
        raise RegistrationRejected([f"등록 파일의 종류가 {KIND!r} 가 아니다"])
    if set(body) != set(KEYS):
        raise RegistrationRejected([f"칸이 다르다 — 빠짐 {sorted(set(KEYS) - set(body))} · 모름 {sorted(set(body) - set(KEYS))}"])
    if body["version"] != VERSION:
        raise RegistrationRejected([f"판 {body['version']!r} 은 이 판독기의 판 {VERSION} 이 아니다"])
    return body, hashlib.sha256(raw).hexdigest()


def _rules_identity(rule_path: Path) -> dict[str, Any] | None:
    import yaml

    if not rule_path.is_file():
        return None
    raw = rule_path.read_bytes()
    body = yaml.safe_load(raw.decode("utf-8")) or {}
    return {"version": body.get("version"), "sha256": hashlib.sha256(raw).hexdigest()}


def registration_problems(reg: dict[str, Any], *, uni: Any, rule_path: Path | None = None) -> list[str]:
    """등록값과 코드 · 설정 · 규칙 파일의 어긋남. 빈 목록이면 맞다."""
    from vlm import pilot_vlm as V
    from vlm.ecc import RULE_PATH

    out: list[str] = []
    for key, actual in (("micro_batch", V.MICRO_BATCH), ("grad_accum", V.GRAD_ACCUM), ("lr", V.LR)):
        if isinstance(reg[key], bool) or reg[key] != actual:
            out.append(f"{key} {reg[key]!r} ≠ 학습 코드 {actual!r}")
    quant = reg["quantization"]
    if not isinstance(quant, dict) or set(quant) != set(QUANT_KEYS):
        out.append(f"quantization 의 칸은 {list(QUANT_KEYS)} 이다")
    elif quant != dict(V.QUANT_4BIT):
        out.append(f"quantization {quant!r} ≠ 학습 코드의 적재 인자 {dict(V.QUANT_4BIT)!r}")
    if reg["recipe"] not in TRAIN_RECIPES:
        out.append(f"recipe {reg['recipe']!r} 는 학습 코드가 구현한 {list(TRAIN_RECIPES)} 가 아니다")
    for key, actual in (("num_rounds", uni.num_rounds), ("local_epochs", uni.local_epochs),
                        ("total_epochs", uni.total_epochs), ("eval_gen_limit", uni.max_new_tokens)):
        if isinstance(reg[key], bool) or reg[key] != actual:
            src = "uni_max_new_tokens" if key == "eval_gen_limit" else f"uni_train_budget.{key}"
            out.append(f"{key} {reg[key]!r} ≠ 통합형 설정 {src} {actual!r}")
    seeds = reg["seed_indices"]
    if not (isinstance(seeds, list) and seeds and all(type(s) is int and 1 <= s <= len(uni.seeds) for s in seeds)
            and len(set(seeds)) == len(seeds)):
        out.append(f"seed_indices {seeds!r} 는 통합형 설정의 시드 {len(uni.seeds)} 개 안의 서로 다른 번호여야 한다")
    want = reg["early_coord_check_rules"]
    have = _rules_identity(Path(rule_path) if rule_path is not None else REPO_ROOT / RULE_PATH)
    if have is None:
        out.append("조기 좌표 점검 규칙 파일이 없다")
    elif not isinstance(want, dict) or want != have:
        out.append(f"early_coord_check_rules {want!r} ≠ 규칙 파일 {have!r}")
    dig = reg["init_adapter_digest"]
    if dig is not None and not (isinstance(dig, str) and len(dig) == 64):
        out.append("init_adapter_digest 는 null 이거나 sha256 16진 64 자다")
    return out


def registration_probe(config_path: Path | None = None, *, path: Path | None = None,
                       rule_path: Path | None = None) -> dict[str, Any]:
    """착수 고정의 `common` 칸에 싣는 둘 — 등록 파일의 sha256 과 값. 어긋나면 `RegistrationRejected`."""
    from vlm.uni_config import load_uni_config

    reg, sha = load_registration(path)
    bad = registration_problems(reg, uni=load_uni_config(config_path), rule_path=rule_path)
    if bad:
        raise RegistrationRejected(bad)
    return {"registration_sha256": sha, "registration": reg}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="본실험 등록 파일을 코드 · 설정 · 규칙 파일과 맞댄다(착수 고정을 쓰지 않는다)")
    ap.add_argument("--config", type=Path, default=None, help="통합형 설정 파일(기본: 학습 쪽 기본 설정)")
    ap.add_argument("--path", type=Path, default=None, help=f"등록 파일(기본: {REG_PATH.as_posix()})")
    a = ap.parse_args(argv)
    try:
        got = registration_probe(a.config, path=a.path)
    except RegistrationRejected as exc:
        for line in exc.problems:
            print(f"어긋남: {line}", file=sys.stderr)
        return 2
    print(f"맞다 — {REG_PATH.as_posix()} sha256 {got['registration_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
