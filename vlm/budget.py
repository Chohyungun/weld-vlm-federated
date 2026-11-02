"""공통 학습 예산 비율(`budget_fraction`) — 학습량 등가 R × E = N 의 N 을 1 보다 작게 둘 때(실험설계 6-4 조건 2).

조기 종료가 아니라 **착수 전에 고정한 예산**이다. 비율 f 는 등록 파일(`configs/train_registration.yaml` 의 `budget_fraction`,
없으면 1.0)이 정하고 착수 고정이 등록 바이트째 묶는다. 다섯 칸이 같은 f 다.

- **어느 행을 보나** — 참여자 c 마다 학습 행(봉인 페어의 split=train, client=c, 파일 순서)을 `Random(f"{seed}:budget-v1:{c}")` 로
  섞어 앞 k_c 행을 고른다. k_c = min(n_c, ceil(f × n_c / (micro × accum)) × micro × accum) — 참여자의 갱신 수가
  ceil(f × n_c / 32) 가 되게 마지막 누적 창을 채운다. 고른 행은 파일 순서로 되돌려 학습기에 넘기고, 학습기는 그 행으로 1 epoch
  (자기 셔플 그대로)를 돈 뒤 **last** 를 저장한다. 시드는 칸과 무관하다 — 로컬 C_k · 중앙(세 몫의 합집합) · 연합 클라이언트 C_k
  가 **같은 행**을 본다(칸 사이의 학습 자료가 같다는 비교의 전제를 지킨다).
- **갱신 수** — 칸의 갱신 수는 고른 행 수에서 학습기의 회계 그대로 ceil(Σ k_c / 32) 다(로컬 · 연합 클라이언트는
  ceil(f × n_c / 32), 중앙은 세 몫 합의 ceil).
- **감독 토큰 기대값** — 고른 행의 행별 감독 토큰(봉인 페어 계측 `per_row.csv`, 아래 `TOKEN_TABLE`)의 합이다. 참여자별 값을
  설정의 epoch 당 기대값 자리에 넣으므로 칸 · 연합 서버의 기존 대조(로컬 = 참여자 값, 중앙 = 세 값의 합)가 그대로 선다.
- f = 1.0 이면 아무것도 바꾸지 않는다 — 행 · 신원 · 기대값이 지금과 바이트로 같다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["BUDGET_POLICY", "PROVENANCE_SUFFIX", "TOKEN_TABLE", "TOKEN_TABLES", "TOKEN_TABLE_SHA256", "BudgetRejected",
           "apply_budget", "budget_identity", "budget_steps", "check_fraction", "expected_tokens", "select_ids",
           "token_table_of", "token_table_problems"]

REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET_POLICY = "client-prefix-v1"
#: 봉인 페어 pairs_main_v1(pairs.jsonl f4e44131…)의 학습 행별 감독 토큰 — `supervised_tokens.json`(10e62663…)과 같은 계측의 행 표.
TOKEN_TABLE = Path("outputs/B_seal_20261001/tokens/per_row.csv")
TOKEN_TABLE_SHA256 = "c88155c48c7c9a5e3ce226a3ae5441d78373c13c7e98afb6116053ba6371a62b"
#: 학습 레시피 → (행별 감독 토큰 표의 자리, sha256). `json_only` 은 1 차 레시피의 표(위 두 상수 — 봉인 폴더 안, 바꾸지 않는다).
#: 새 레시피의 표는 봉인 폴더 밖이고 `vlm.token_table` 이 낸다 — 곁의 출처 파일(`<표>` + `PROVENANCE_SUFFIX`)이 레시피 · 페어 digest ·
#: 계측 코드 · 토크나이저 신원을 든다. 표 해시가 맞는 것과 지금 등록 · 지금 환경에 맞는 표인 것은 따로 본다(`token_table_problems`).
#: 2 차 표는 2026-10-27 봉인 페어를 잰 S15 산출이다. S10 digest와 곁 출처의 per_client를 S13 독립 계측과 대조했다.
#: 이전 임시 페어의 표와 출처는 보존한다. 현재 등록에는 최종 봉인 페어의 표만 연결한다.
TOKEN_TABLES: dict[str, tuple[Path, str]] = {
    "json_only": (TOKEN_TABLE, TOKEN_TABLE_SHA256),
    "json_line_then_basis": (Path("outputs/target_tokens_v3_20261027/per_row.csv"),
                             "295603f08002e17ae6f3de0387f85fea74968abbf2e9233955792cb82788c8f1"),
}
PROVENANCE_SUFFIX = ".provenance.json"


class BudgetRejected(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def check_fraction(f: Any) -> float:
    """0 < f ≤ 1 의 실수(불 · 문자열 아님). 틀리면 `BudgetRejected`."""
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(float(f)) or not 0 < float(f) <= 1:
        raise BudgetRejected("budget_fraction", f"budget_fraction 은 0 보다 크고 1 이하인 수다: {f!r}")
    return float(f)


def _window() -> int:
    from vlm.pilot_vlm import GRAD_ACCUM, MICRO_BATCH

    return int(MICRO_BATCH) * int(GRAD_ACCUM)


def budget_steps(n_rows: int, fraction: float) -> int:
    """참여자 하나의 갱신 수 — ceil(f × n / (micro × accum))."""
    return math.ceil(check_fraction(fraction) * int(n_rows) / _window())


def _keep(n_rows: int, fraction: float) -> int:
    w = _window()
    return min(int(n_rows), budget_steps(n_rows, fraction) * w)


def select_ids(ids_by_client: Mapping[str, Sequence[str]], *, fraction: float, seed_value: int) -> dict[str, list[str]]:
    """참여자마다 고른 행의 id(파일 순서). f = 1 이면 전부."""
    f = check_fraction(fraction)
    out: dict[str, list[str]] = {}
    for c, ids in ids_by_client.items():
        ids = list(ids)
        if f == 1.0:
            out[c] = ids
            continue
        order = list(range(len(ids)))
        random.Random(f"{int(seed_value)}:budget-v1:{c}").shuffle(order)
        keep = set(order[:_keep(len(ids), f)])
        out[c] = [ids[i] for i in range(len(ids)) if i in keep]
    return out


def _row_id(r: Mapping[str, Any]) -> str:
    return str(r["image_id"])


def apply_budget(rows: list[dict], *, fraction: float, seed_value: int, pairs_path: str | Path | None = None) -> list[dict]:
    """칸의 학습 행에서 예산 몫만(파일 순서 유지). 몫은 **참여자의 전체 학습 행**에서 고른다 — 로컬 · 중앙 · 연합이 같은 행.

    f = 1 이면 받은 목록을 그대로 돌려준다. f < 1 이면 행마다 참여자(`client`)를 보고, 그 참여자의 전체 학습 행
    (봉인 페어 · `pairs_path`)에서 고른 id 집합에 든 행만 남긴다. 받은 행이 참여자 전체가 아니면(리허설 목록 등) 거부한다.
    """
    f = check_fraction(fraction)
    if f == 1.0:
        return rows
    from vlm.pilot_vlm import load_pairs

    clients = sorted({str(r["client"]) for r in rows})
    full = {c: [_row_id(r) for r in load_pairs("train", client=c, pairs_path=pairs_path)] for c in clients}
    for c in clients:
        got = [_row_id(r) for r in rows if str(r["client"]) == c]
        if got != full[c]:
            raise BudgetRejected("budget_rows", f"예산 비율은 참여자 {c} 의 학습 행 전체에만 건다 — 받은 행이 전체와 다르다")
    keep = {i for ids in select_ids(full, fraction=f, seed_value=seed_value).values() for i in ids}
    return [r for r in rows if _row_id(r) in keep]


def token_table_problems(recipe: str, *, pairs_digest: str | None, table_path: Path | None = None,
                         model_id: str | None = None, model_revision: str | None = None) -> list[str]:
    """'지금 등록 · 지금 환경에 맞는 표인가'. 빈 목록이면 맞다.

    표 곁의 출처 파일이 레시피 · 페어 digest · 표 sha256 · 타깃 계약을 맞게 들고, 출처의 계측 코드 sha256 과 토크나이저 신원
    (`vlm.token_table.IDENTITY_KEYS`)이 지금 이 코드 · 지금 환경(`tokenizer_identity(model_id, model_revision)` — 로컬 캐시의 작은
    파일 · 꾸러미 판, 모델 가중치 0)과 같아야 한다(외부 검토 200 M-01 — 페어 · 레시피 · 합을 그대로 두고 토크나이저 판만 바꿔도
    학습 전에 멈춘다). 표 바이트의 해시 대조(`_token_table`)와 따로 본다.

    1 차 레시피의 표는 봉인 폴더 안이라 출처 파일이 없다 — 지문 고정만 본다(v1: 신원 미기록)."""
    if recipe == "json_only":
        return []
    rel, want_sha = token_table_of(recipe)
    p = Path(table_path) if table_path is not None else REPO_ROOT / rel
    prov = p.with_name(p.name + PROVENANCE_SUFFIX)
    if not prov.is_file():
        return [f"행별 토큰 표의 출처 파일이 없다: {prov.name}"]
    try:
        rec = json.loads(prov.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"출처 파일을 읽지 못했다: {exc}"]
    out = []
    if not isinstance(rec, Mapping):
        return ["출처 파일이 객체가 아니다"]
    if rec.get("recipe") != recipe:
        out.append(f"표의 레시피 {rec.get('recipe')!r} ≠ 지금 레시피 {recipe!r}")
    if pairs_digest is None or rec.get("pairs_digest") != pairs_digest:
        out.append(f"표의 페어 digest {rec.get('pairs_digest')!r} ≠ 설정의 페어 digest {pairs_digest!r}")
    if rec.get("table_sha256") != want_sha:
        out.append(f"출처 파일의 표 sha256 {rec.get('table_sha256')!r} ≠ 고정값 {want_sha!r}")
    from vlm import token_table as T
    from vlm.pilot_vlm import target_contract_sha256

    if rec.get("contract_sha256") != target_contract_sha256(recipe):
        out.append("출처 파일의 타깃 계약 해시가 레시피의 계약과 다르다")
    if rec.get("schema") != T.PROVENANCE_SCHEMA:
        out.append(f"출처 파일의 꼴 {rec.get('schema')!r} 이 지금 꼴 {T.PROVENANCE_SCHEMA} 가 아니다 — 토크나이저 파일 신원이 없는 옛 출처다")
    counter = rec.get("counter")
    if not isinstance(counter, Mapping) or counter.get("sha256_lf") != T.source_sha256():
        out.append("출처의 계측 코드 sha256 이 지금 계측 코드(vlm/token_table.py)와 다르다 — 지금 코드로 표를 다시 낸다")
    if not model_id:
        out.append("토크나이저 신원을 맞댈 모델(model_id · revision)을 받지 못했다")
    else:
        try:
            now = T.tokenizer_identity(model_id, model_revision)
        except (OSError, ValueError) as exc:
            out.append(f"지금 환경의 토크나이저 신원을 재지 못했다 — {exc}")
        else:
            out.extend(T.identity_problems(rec.get("tokenizer"), now))
    return out


def _token_table(path: Path | None = None, *, sha256: str | None = None, recipe: str = "json_only") -> dict[str, int]:
    """행별 감독 토큰 표. `path` 가 없으면 이 모듈의 저장소 뿌리 아래의 레시피 표(`TOKEN_TABLES`)이고 지문은 늘 그 고정값이다.
    `path` 를 주면 `sha256` 을 줄 때만 맞댄다(연합 서버는 등록 지문을 준다 — 시험의 합성 표는 주지 않는다)."""
    rel, pinned = token_table_of(recipe)
    p = Path(path) if path is not None else REPO_ROOT / rel
    if not p.is_file():
        raise BudgetRejected("token_table_missing", f"행별 감독 토큰 표가 없다: {rel.as_posix()}")
    raw = p.read_bytes()
    want = pinned if path is None else sha256
    if want is not None and hashlib.sha256(raw).hexdigest() != want:
        raise BudgetRejected("token_table_changed", "행별 감독 토큰 표의 바이트가 등록한 지문과 다르다")
    rows = csv.DictReader(raw.decode("utf-8").splitlines())
    return {str(r["image_id"]): int(r["supervised_tokens"]) for r in rows}


def token_table_of(recipe: str) -> tuple[Path, str]:
    """레시피의 행별 토큰 표 (저장소 상대 자리, sha256). 모르는 레시피는 `BudgetRejected("recipe_unknown")`."""
    if recipe not in TOKEN_TABLES:
        raise BudgetRejected("recipe_unknown", f"행별 토큰 표가 없는 레시피다: {recipe!r} — 아는 것 {list(TOKEN_TABLES)}")
    return TOKEN_TABLES[recipe]


def expected_tokens(per_epoch: Mapping[str, int] | None, *, fraction: float, seed_value: int,
                    pairs_path: str | Path | None = None, table_path: Path | None = None,
                    table_sha256: str | None = None, recipe: str = "json_only",
                    pairs_digest: str | None = None, model_id: str | None = None,
                    model_revision: str | None = None) -> dict[str, int] | None:
    """참여자별 감독 토큰 기대값(epoch 당). `table_path` · `table_sha256` 은 표의 자리와 지문이다(`_token_table`).

    - 레시피 `json_only`(1 차): f = 1 이면 설정 값 그대로, f < 1 이면 고른 행의 행별 토큰 합 — 지금과 같다.
    - 새 레시피: f 와 무관하게 그 레시피의 표를 읽는다. 표의 참여자 합이 설정의 기대값과 같고(f = 1 에서도 — 설정 값이 틀리면 칸이
      학습을 다 돈 뒤 끝 대조에서야 멈추던 길을 착수 전으로 당긴다), 표의 출처가 지금 레시피 · 페어 digest · 계측 코드 · 이 모델
      (`model_id` · `model_revision`)의 지금 토크나이저 신원과 맞아야 쓴다(`token_table_problems`)."""
    f = check_fraction(fraction)
    token_table_of(recipe)
    if per_epoch is None or (f == 1.0 and recipe == "json_only"):
        return None if per_epoch is None else {str(k): int(v) for k, v in per_epoch.items()}
    from vlm.pilot_vlm import load_pairs

    if recipe != "json_only":
        bad = token_table_problems(recipe, pairs_digest=pairs_digest, table_path=table_path, model_id=model_id,
                                   model_revision=model_revision)
        if bad:
            raise BudgetRejected("token_table_provenance", " · ".join(bad))
    table = _token_table(table_path, sha256=table_sha256, recipe=recipe)
    full = {c: [_row_id(r) for r in load_pairs("train", client=c, pairs_path=pairs_path)] for c in per_epoch}
    # 표가 참여자의 전체 학습 행을 덮고 그 합이 설정의 epoch 당 기대값과 같아야 쓴다(표와 설정이 같은 계측인지).
    for c, ids in full.items():
        miss = [i for i in ids if i not in table]
        if miss or sum(table[i] for i in ids) != int(per_epoch[c]):
            raise BudgetRejected("token_table_mismatch", f"행별 감독 토큰 표가 참여자 {c} 의 설정 기대값과 맞지 않는다")
    chosen = select_ids(full, fraction=f, seed_value=seed_value)
    return {c: sum(table[i] for i in ids) for c, ids in chosen.items()}


def budget_identity(fraction: float) -> dict[str, Any]:
    """신원 · 학습 설정에 덧붙일 칸 — f = 1 이면 빈 사전(지금과 바이트가 같다)."""
    f = check_fraction(fraction)
    return {} if f == 1.0 else {"budget_fraction": f, "budget_policy": BUDGET_POLICY}
