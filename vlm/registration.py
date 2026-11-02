"""본실험 등록값 — 데이터 쪽이 쓰는 `configs/train_registration.yaml` 을 읽어 코드 · 설정과 맞대고 착수 고정에 싣는다.

다섯 칸 공통 고정의 확정 수치는 그 파일 하나에 둔다(값을 코드에 적지 않는다 — 의사결정로그 2026-10-13 13:5x). 학습 쪽은

- 파일을 정확한 칸 집합으로 읽고(`load_registration`),
- 코드가 **실제로 쓰는 값**(마이크로 배치 · 누적 · 학습률 · 4bit 적재 인자 · 학습 레시피), 통합형 설정 블록(라운드 · 국소 epoch ·
  총 epoch · 생성 한도 · 시드 수), 조기 좌표 점검 규칙 파일(판 · 바이트 지문), 레시피의 행별 토큰 표(출처의 페어 · 계측 코드 ·
  토크나이저 신원 — 지금 환경과)와 맞대며(`registration_problems`),
- 어긋남이 없을 때만 착수 고정의 `common` 칸에 파일 지문과 값을 싣는다(`registration_probe` — `vlm.pin.common_probe` 가 부른다).

파일이 없거나 하나라도 어긋나면 착수 고정을 쓰지 않는다(`RegistrationRejected` — 닫힌 쪽). 본 칸은 시작 · 재개마다 착수 고정을
다시 재므로, 등록 파일이나 코드 · 설정이 바뀌면 그 자리에서 멈춘다.

데이터 쪽이 파일을 쓴 뒤 스스로 확인하는 명령(착수 고정을 쓰지 않는다 — 읽고 맞대기만):

    python -X utf8 -m vlm.registration [--config configs/base.yaml] [--path configs/train_registration.yaml]

맞으면 종료 0 과 파일 지문, 어긋나면 종료 2 와 어긋남을 줄마다 낸다.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vlm.pin import PinMismatch

__all__ = [
    "KEYS",
    "KIND",
    "PROFILE_FILE",
    "QUANT_KEYS",
    "REGIME_LIMITS",
    "REGIME_PROMPTS",
    "REGIME_PROMPT_SHA256",
    "REG_PATH",
    "TRAIN_RECIPES",
    "RegistrationRejected",
    "binding_problems",
    "budget_fraction_of",
    "load_registration",
    "read_run_profile",
    "regime_problems",
    "regime_prompt_sha256",
    "registered_budget_fraction",
    "registered_recipe",
    "registered_seed_indices",
    "registration_from_bytes",
    "registration_probe",
    "registration_problems",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
#: 등록 파일의 자리(데이터 쪽이 쓴다).
REG_PATH = Path("configs/train_registration.yaml")
#: 출력 뿌리의 실행 묶음(1017b) — 그 뿌리가 쓰는 학습 등록 벌과 실행 이름. 없으면 본실험(기본 등록)이다. 실행기와 연합 서버가 같은
#: 판독기(`read_run_profile`)를 쓴다(지시 1022c 덧붙임 B — 서버가 실행 설정의 레시피를 등록과 맞댄다).
PROFILE_FILE = "RUN_PROFILE.json"
KIND = "train_registration"
VERSION = 1
#: 등록 파일의 칸 — 정확히 이 집합(빠진 칸 · 모르는 칸은 거부).
KEYS = ("kind", "version", "seed_indices", "num_rounds", "local_epochs", "total_epochs", "recipe", "eval_gen_limit",
        "quantization", "micro_batch", "grad_accum", "lr", "early_coord_check_rules", "init_adapter_digest")
#: 있어도 되는 칸 — 공통 학습 예산 비율(`vlm.budget`. 없으면 1.0)과 학습 전 계약(`narrative_inputs: {version, sha256}` — C 08 2 판 §5-3,
#: 있으면 평가 쪽 대조 함수로 맞댄다 — `vlm.narrative_inputs`). 없으면 이 칸들을 더하기 전의 등록 파일 바이트가 그대로 읽힌다.
OPTIONAL_KEYS = ("budget_fraction", "narrative_inputs")
QUANT_KEYS = ("load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant")
#: 학습 코드가 구현한 레시피 — `vlm.pilot_vlm.TARGET_CONTRACTS` 의 열쇠와 같은 집합(시험이 맞댄다). `json_only` 은 1 차 등록,
#: `json_line_then_basis` 는 JSON 한 줄 + 줄바꿈 + `target_text`(2026-10-20 설계 복구 02 2판 §3-5). 두 과제 혼합은 아직 없다.
TRAIN_RECIPES = ("json_only", "json_line_then_basis")
#: 체제 짝(지시 1022c 덧붙임 A) — 레시피마다 받는 프롬프트(저장소 상대 자리 · 모델이 받는 문자열의 sha256)와 평가 생성 한도.
#: 하네스 04 `resolve_regime` 과 같은 짝이다(하네스는 이 두 상수를 읽는다). 1 차는 1 차 등록의 값(프롬프트 v2 · 한도 512), 2 차는 프롬프트
#: v3.1 과 **명시한** 한도(사용자 결정 — 값을 짝에 묶지 않는다). v3.1 은 v3 초안(`unified_v3_basis.txt` — 바이트 그대로 남는다)의 '적용 조항'
#: 을 정답 서술 2 판의 낱말 '대응 조항' 으로 맞춘 새 판이다(C 08 2 판 Q5 · 09 §7 결정 6 — 02a · 하네스 · 등록이 같은 자리 · 해시를 읽는다).
#: 프롬프트는 자리가 아니라 문자열의 sha256(`vlm.pilot_vlm.load_prompt`)으로 맞대고, 지문은 고정 상수다 — 연합 서버의 앱 묶음(FAB)에는
#: 프롬프트 파일이 없고, 문안을 고치면 이 상수 · 시험 8 이 함께 바뀐다(시험이 파일과 맞댄다).
REGIME_PROMPTS = {"json_only": "vlm/prompts/unified_v2_absorig.txt",
                  "json_line_then_basis": "vlm/prompts/unified_v3_1_basis.txt"}
REGIME_PROMPT_SHA256 = {"json_only": "08d3bda5777bb79583377b37194ee7c5bff33834a3aa564900cd0eb74c894a2b",
                        "json_line_then_basis": "dad1769abda33cb766556ee48f0973db298d4468251d7a94f0484e305623eb68"}
REGIME_LIMITS: dict[str, int | None] = {"json_only": 512, "json_line_then_basis": None}
#: 코드 자산(프롬프트 파일)의 뿌리 — 이 모듈이 있는 저장소다. `REPO_ROOT`(등록 파일의 기본 자리 — 시험이 옮긴다)와 따로 둔다.
_CODE_ROOT = Path(__file__).resolve().parents[1]
_HEX64 = re.compile(r"[0-9a-f]{64}")
_UNSET = object()


def regime_prompt_sha256(recipe: str) -> str:
    """그 레시피가 받는 프롬프트의 sha256(고정 상수 — `load_prompt` 의 값과 같다는 것은 시험이 본다)."""
    return REGIME_PROMPT_SHA256[recipe]


def regime_problems(recipe: Any, *, prompt_sha256: Any, gen_limit: Any = _UNSET) -> list[str]:
    """레시피 × 프롬프트 × 한도의 짝(지시 1022c 덧붙임 A). 빈 목록이면 맞다 — 섞인 조합('2 차 레시피 + v2 프롬프트' ·
    '1 차 레시피 + v3')은 기동 전에 거부한다. `gen_limit` 을 주지 않으면 한도는 보지 않는다 — 한도를 갖는 자리(학습 등록의
    `eval_gen_limit`)에서만 준다. 1 차는 한도가 1 차 등록의 값이어야 하고, 2 차는 명시한 양의 정수여야 한다(값은 사용자 결정)."""
    if recipe not in REGIME_PROMPTS:
        return [f"레시피 {recipe!r} 의 체제 짝이 없다 — 아는 것 {list(REGIME_PROMPTS)}"]
    out: list[str] = []
    want = regime_prompt_sha256(recipe)
    if prompt_sha256 != want:
        other = next((r for r in REGIME_PROMPTS if r != recipe and regime_prompt_sha256(r) == prompt_sha256), None)
        what = f"레시피 {other!r} 의 프롬프트" if other else "어느 체제의 프롬프트도 아니다"
        out.append(f"레시피 {recipe!r} 는 프롬프트 {REGIME_PROMPTS[recipe]}(sha256 {want[:16]}…)만 받는다 — 받은 프롬프트 "
                   f"sha256 {str(prompt_sha256)[:16]}…는 {what}")
    if gen_limit is not _UNSET:
        lim = REGIME_LIMITS[recipe]
        if isinstance(gen_limit, bool) or not isinstance(gen_limit, int) or gen_limit < 1:
            out.append(f"레시피 {recipe!r} 의 평가 생성 한도가 명시한 양의 정수가 아니다: {gen_limit!r}")
        elif lim is not None and gen_limit != lim:
            out.append(f"레시피 {recipe!r} 의 평가 생성 한도는 {lim} 이다(1 차 등록의 값) — 받은 값 {gen_limit!r}")
    return out


class RegistrationRejected(PinMismatch):
    """등록 파일이 없거나 코드 · 설정과 다르다 — 착수 고정을 쓰지 않는다."""

    code = "registration_rejected"

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__(["registration"], " · ".join(self.problems))


def load_registration(path: Path | None = None) -> tuple[dict[str, Any], str]:
    """(값, 파일 바이트의 sha256). 칸 집합 · 종류 · 판을 본다."""
    p = Path(path) if path is not None else REPO_ROOT / REG_PATH
    if not p.is_file():
        raise RegistrationRejected([f"등록 파일이 없다: {REG_PATH.as_posix()} — 데이터 쪽이 쓴다"])
    raw = p.read_bytes()
    return registration_from_bytes(raw), hashlib.sha256(raw).hexdigest()


def registration_from_bytes(raw: bytes) -> dict[str, Any]:
    """등록 파일 바이트의 값 — `load_registration` 과 같은 검사(칸 집합 · 종류 · 판). 커밋의 blob 을 읽는 쪽(export 사전 점검)이 쓴다."""
    import yaml

    body = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(body, dict) or body.get("kind") != KIND:
        raise RegistrationRejected([f"등록 파일의 종류가 {KIND!r} 가 아니다"])
    if not set(KEYS) <= set(body) <= set(KEYS) | set(OPTIONAL_KEYS):
        unknown = sorted(set(body) - set(KEYS) - set(OPTIONAL_KEYS))
        raise RegistrationRejected([f"칸이 다르다 — 빠짐 {sorted(set(KEYS) - set(body))} · 모름 {unknown}"])
    if body["version"] != VERSION:
        raise RegistrationRejected([f"판 {body['version']!r} 은 이 판독기의 판 {VERSION} 이 아니다"])
    return body


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
    else:
        # 체제 짝 — 등록의 레시피가 설정의 프롬프트 · 등록의 생성 한도와 같은 체제여야 한다(지시 1022c 덧붙임 A).
        prompt = getattr(uni, "prompt_path", None)
        pp = None if not prompt else (Path(prompt) if Path(prompt).is_absolute() else _CODE_ROOT / prompt)
        if pp is None or not pp.is_file():
            out.append(f"통합형 설정의 프롬프트 파일(uni_prompt_path {prompt!r})이 없다 — 레시피의 프롬프트와 맞댈 수 없다")
        else:
            out.extend(regime_problems(reg["recipe"], prompt_sha256=V.load_prompt(pp)[1], gen_limit=reg["eval_gen_limit"]))
        # 레시피의 행별 토큰 표 — 출처가 설정의 페어 · 지금 계측 코드 · 설정 모델의 지금 토크나이저 신원과 맞아야 착수 고정을 쓴다
        # (외부 검토 200 M-01). 1 차 표는 출처 파일이 없어 지문 고정만 본다(v1: 신원 미기록 — `token_table_problems`).
        from vlm.budget import BudgetRejected, token_table_problems

        try:
            table_bad = token_table_problems(reg["recipe"], pairs_digest=getattr(uni, "pairs_digest", None),
                                             model_id=getattr(uni, "model_id", None),
                                             model_revision=getattr(uni, "model_revision", None))
        except BudgetRejected as exc:
            table_bad = [str(exc)]
        out.extend(f"레시피 {reg['recipe']!r} 의 행별 토큰 표 — {b}" for b in table_bad)
    for key, actual in (("num_rounds", uni.num_rounds), ("local_epochs", uni.local_epochs),
                        ("total_epochs", uni.total_epochs), ("eval_gen_limit", uni.max_new_tokens)):
        if isinstance(reg[key], bool) or reg[key] != actual:
            src = "uni_max_new_tokens" if key == "eval_gen_limit" else f"uni_train_budget.{key}"
            out.append(f"{key} {reg[key]!r} ≠ 통합형 설정 {src} {actual!r}")
    if "budget_fraction" in reg:
        from vlm.budget import BudgetRejected, check_fraction

        try:
            check_fraction(reg["budget_fraction"])
        except BudgetRejected as exc:
            out.append(str(exc))
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
    if dig is not None and not (isinstance(dig, str) and _HEX64.fullmatch(dig)):
        out.append("init_adapter_digest 는 null 이거나 소문자 16진 64 자(sha256)다")
    elif dig is not None and not (isinstance(seeds, list) and len(seeds) == 1):
        out.append("init_adapter_digest 는 시드 번호가 하나일 때만 둔다 — 공통 초기 어댑터는 시드마다 다르다")
    # 학습 전 계약(C 08 2 판 §5-3 · D 07 §17-4) — 칸이 있을 때만 평가 쪽 대조 함수로 맞댄다. 하나라도 어긋나면 착수 고정을 쓰지 않는다.
    from vlm import narrative_inputs as NI

    if NI.ENTRY_KEY in reg:
        out.extend(NI.problems(reg[NI.ENTRY_KEY], repo=REPO_ROOT, pairs_path=getattr(uni, "pairs_path", None),
                               config_pairs_digest=getattr(uni, "pairs_digest", None), recipe=reg["recipe"]))
    return out


def binding_problems(reg: Mapping[str, Any], *, seed_index: Any, init_digest: str | None = None,
                     recipe: str | None = None) -> list[str]:
    """등록과 이 실행의 결속(외부 검토 §137 M-01 · M-02). 빈 목록이면 맞다.

    요청한 시드 번호가 등록의 `seed_indices` 안이어야 하고, 등록에 `init_adapter_digest` 가 있으면 실제로 읽은 공통 초기
    어댑터의 digest(`init_digest`)와 같아야 한다. `init_digest` 를 주지 않으면 시드만 본다(명령줄의 이른 경계).
    `recipe` 를 주면 그 실행의 학습 레시피가 등록의 `recipe` 와 같아야 한다 — 실행 설정의 레시피가 등록과 갈린 채 학습하지 않는다.
    """
    out: list[str] = []
    if recipe is not None and reg.get("recipe") != recipe:
        out.append(f"실행의 학습 레시피 {recipe!r} 가 등록의 recipe {reg.get('recipe')!r} 와 다르다")
    seeds = reg.get("seed_indices")
    if type(seed_index) is not int or not isinstance(seeds, list) or seed_index not in seeds:
        out.append(f"시드 번호 {seed_index!r} 는 등록의 seed_indices {seeds!r} 밖이다 — 등록하지 않은 시드는 시작하지 않는다")
    want = reg.get("init_adapter_digest")
    if want is not None and init_digest is not None and want != init_digest:
        out.append("실제로 읽은 공통 초기 어댑터의 digest 가 등록의 init_adapter_digest 와 다르다")
    return out


def budget_fraction_of(reg: Mapping[str, Any]) -> float:
    """등록의 공통 학습 예산 비율 — 칸이 없으면 1.0."""
    from vlm.budget import check_fraction

    return check_fraction(reg.get("budget_fraction", 1.0))


def registered_budget_fraction(path: Path | None = None) -> float:
    """등록 파일의 예산 비율(꼴만 본다 — 설정과의 맞댐은 `registration_probe`)."""
    reg, _ = load_registration(path)
    return budget_fraction_of(reg)


def read_run_profile(out_root: Path, *, repo_root: Path | None = None) -> dict[str, Any] | None:
    """출력 뿌리의 실행 묶음 — `{"train_registration": 경로, "run_name": 이름}`, 묶음 파일이 없으면 None(본실험 기본 등록).

    적힌 등록 벌(저장소 상대 경로 — `repo_root` 아래)의 바이트가 적힌 sha256 과 다르거나 없으면 `RegistrationRejected`."""
    import json

    p = Path(out_root) / PROFILE_FILE
    if not p.is_file():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    reg = Path(repo_root if repo_root is not None else REPO_ROOT) / rec["train_registration"]
    if not reg.is_file() or hashlib.sha256(reg.read_bytes()).hexdigest() != rec["train_registration_sha256"]:
        raise RegistrationRejected([f"실행 묶음의 등록 벌이 없거나 처음 받은 바이트와 다르다: {rec['train_registration']}"])
    return {"train_registration": reg, "run_name": str(rec["run_name"])}


def registered_recipe(path: Path | None = None) -> str:
    """등록 파일의 학습 레시피 — 실행 묶음이 고른 등록(`path`)을 읽는다. 코드가 구현하지 않은 레시피면 `RegistrationRejected`.

    설정과의 맞댐(생성 한도 · 시드표 · 규칙 파일)은 착수 고정이 `registration_probe` 로 한다. 여기서는 레시피만 본다."""
    reg, _ = load_registration(path)
    recipe = reg["recipe"]
    if recipe not in TRAIN_RECIPES:
        raise RegistrationRejected([f"recipe {recipe!r} 는 학습 코드가 구현한 {list(TRAIN_RECIPES)} 가 아니다"])
    return recipe


def registered_seed_indices(path: Path | None = None) -> list[int]:
    """등록 파일의 시드 번호들(정렬) — 실행기 명령줄의 시드 경계. 서로 다른 양의 정수 목록이 아니면 `RegistrationRejected`.

    설정과의 대조(시드표 크기 · 예산 · 규칙 파일)는 착수 고정이 `registration_probe` 로 한다. 여기서는 꼴만 본다."""
    reg, _ = load_registration(path)
    seeds = reg["seed_indices"]
    if not (isinstance(seeds, list) and seeds and all(type(s) is int and s >= 1 for s in seeds)
            and len(set(seeds)) == len(seeds)):
        raise RegistrationRejected([f"seed_indices {seeds!r} 는 서로 다른 양의 정수 목록이어야 한다"])
    return sorted(seeds)


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
    recipe = got["registration"]["recipe"]
    if recipe == TRAIN_RECIPES[0]:
        print(f"행별 토큰 표({recipe}): v1: 신원 미기록 — 봉인 표에 출처 파일이 없어 표 지문 고정만 본다")
    else:
        print(f"행별 토큰 표({recipe}): 출처의 계측 코드 · 토크나이저 신원이 지금 코드 · 지금 환경과 같다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
