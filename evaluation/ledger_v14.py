"""통합형 학습 원장 → `LedgerView` — 평가 쪽의 **독립 구현**. 07번 §30-7 · 리허설 미니스펙 2판 반영판(`13d7f4f`) §2-6.

쓰는 쪽에도 같은 규칙을 옮긴 함수가 있다(학습 쪽 `ledger_view`). 이 모듈은 그 코드를 보지 않고 **계약의 글에서** 짰다 —
두 구현이 같은 원장에서 같은 값을 내는지 맞대는 것이 목적이다. 쓰는 쪽 모듈을 가져오지 않는다(열 이름도 여기 적는다).

| 규칙 | 무엇 |
|---|---|
| 해시 | `file_sha256` = 파일 **원시 바이트**의 sha256. 줄끝을 정규화하지 않는다(원장은 CRLF 로 쓰인다) · CR 을 거부하지 않는다 |
| 끝 행 | 마지막 데이터 행의 `metric_name` 이 `final_adapter_step` 이어야 한다. `final_step` = 그 행의 `metric_value` 를 실수로 읽어 정수인지 본 값. 끝 행의 `round` 는 읽지 않는다 |
| 신원 | `run_id` · `cell` = 끝 행의 값. `client` = 끝 행의 `client_id` — `central` · `server` 는 `None` |
| 로컬 · 중앙 | 겹친 행 규칙(이음매 행 `process_start` 뒤의 것이 앞의 것을 대체) 뒤 **마지막 epoch 의 `optimizer_steps`** 가 끝 행의 값과 같다 |
| 연합 | `R`(등록의 `budget_r`)과 `K`(참여자 수)를 **호출부가 준다** — 기본값이 없다. 끝 행은 `client_id="server"` · `round = R` · 값 `= R`. 끝 행을 뺀 데이터 행에 — ① `server` 행의 `round` 집합 = `{0,…,R−1}` ② 같은 `(round, client_id, metric_name)` 이 둘이면 거부(재개가 없다) ③ 라운드마다 `optimizer_steps` 행을 가진 정수 `client_id` 집합 = `{0,…,K−1}`(기본값 `-1` 은 받지 않는다) |

규칙의 글은 리허설 미니스펙 2판 반영판 §2-6 이다. 계약 §30-7 의 문면(③ "서로 다른 정수", 끝 행 제외 없음)과 다르다 —
개정 전에는 "계약대로" 라고 인용하지 않는다.

**쓰는 쪽 규칙 함수보다 더 보는 것**(쓰는 쪽은 받는데 여기서만 거부한다 — 모두 쓰는 쪽이 쓰지 않는 원장이다. 사례와 까닭은
`tests/ledger_cases_v14.py` 의 `stricter` 가 들고, 대조 시험이 그 밖의 어긋남을 떨어뜨린다).
- 모든 데이터 행의 `run_id` · `cell` 이 끝 행과 같아야 한다 — 원장은 실행 하나 · 칸 하나의 것이다.
- 로컬 · 중앙에 다른 참여자의 행이 있으면 거부한다 — 모델마다 파일 하나다. 끝 행의 `cell` 이 통합형 칸이 아니거나 끝 행의 참여자가 그 칸의 것이 아니면 거부한다.
- 로컬 · 중앙에서 **한 세그먼트 안**에 같은 `(round, client_id, metric_name)` 이 둘이면 거부한다. 세그먼트는 `process_start` 에서만 끊는다(쓰는 쪽은 이음매 두 행 모두에서 끊는다 — 두 행은 함께 쓰인다).
- 연합 참여자 행의 `client_id` 는 **정수의 정규 표기**여야 한다(`01` · ` 1` · `C1` 거부 — 원시 문자열 열쇠의 겹침을 ② 가 못 잡는다). 라운드 밖의 참여자 행을 거부한다.
- `round` 는 정수의 정규 표기여야 한다. 끝 행의 값은 1 이상이다. 모든 칸이 빈 행을 거부한다(쓰는 쪽은 건너뛴다 — 원시 바이트 해시가 그 행을 품는다).
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from pathlib import Path

from evaluation.bundle_v14 import LedgerView

LEDGER_FIELDS: tuple[str, ...] = (
    "run_id", "seed", "cell", "split_hash", "client_id", "round", "n_train_samples",
    "metric_name", "metric_value", "bytes_up", "bytes_down", "wall_time",
)
"""원장의 12열 — 계약 §30-7 "꼴" 행이 가리키는 쓰는 쪽 정의와 같아야 한다(시험이 맞댄다)."""
END_METRIC = "final_adapter_step"
SEAM_METRICS = ("process_start", "process_started_unix")
"""이음매 행 — 프로세스가 신원 대조를 지난 뒤에 쓴다. 세그먼트의 경계다."""
LOCAL_CLIENTS = ("C1", "C2", "C3")
CELL_END_CLIENT = {"uni_local": LOCAL_CLIENTS, "uni_central": ("central",), "uni_fed": ("server",)}
"""칸 → 끝 행이 가질 수 있는 `client_id`."""
FED_CLIENTS = len(LOCAL_CLIENTS)
"""통합형 연합의 참여자 수 — 칸 정의(`C1` · `C2` · `C3`)에서 온다. 등록에 참여자 수 칸이 없다. 호출부가 이 값을 넘긴다."""
_INT = re.compile(r"^-?(0|[1-9][0-9]*)$")


class LedgerRejected(ValueError):
    """원장이 규칙을 어긴다. `code` 가 어느 규칙인지 말한다."""

    def __init__(self, code: str, reason: str):
        self.code = code
        self.reason = reason
        super().__init__(f"[{code}] {reason}")


def _rows(raw: bytes) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise LedgerRejected("LEDGER_FORM", "UTF-8 이 아니다") from None
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration:
        raise LedgerRejected("LEDGER_FORM", "빈 파일이다") from None
    if tuple(header) != LEDGER_FIELDS:
        raise LedgerRejected("LEDGER_FORM", "머리행이 원장의 12열이 아니다")
    out: list[dict[str, str]] = []
    for n, row in enumerate(reader, 2):
        if not row:
            continue
        if len(row) != len(LEDGER_FIELDS):
            raise LedgerRejected("LEDGER_FORM", f"{n}번째 줄의 칸 수가 12 가 아니다")
        if not any(row):
            raise LedgerRejected("LEDGER_FORM", f"{n}번째 줄의 모든 칸이 비었다")
        out.append(dict(zip(LEDGER_FIELDS, row)))
    return out


def _int_text(value: str, what: str) -> int:
    """정수의 **정규 표기**(`3`, `-1`) — `3.0` · `03` · ` 3` 은 받지 않는다. 쓰는 쪽은 `int(…)` 로 읽는다."""
    if not _INT.match(value):
        raise LedgerRejected("LEDGER_VALUE", f"{what} 가 정수의 정규 표기가 아니다")
    return int(value)


def _integral(value: str, what: str) -> int:
    try:
        v = float(value)
    except ValueError:
        raise LedgerRejected("LEDGER_VALUE", f"{what} 가 수가 아니다") from None
    if not math.isfinite(v) or v != int(v):
        raise LedgerRejected("LEDGER_VALUE", f"{what} 가 정수가 아니다")
    return int(v)


def _local_check(rows: list[dict[str, str]], end_value: int) -> None:
    """겹친 행 규칙을 적용한 뒤 마지막 epoch 의 `optimizer_steps` = 끝 행의 값."""
    latest: dict[tuple[str, str, str], str] = {}
    in_segment: set[tuple[str, str, str]] = set()
    for r in rows:
        if r["metric_name"] in SEAM_METRICS:
            if r["metric_name"] == "process_start":
                in_segment = set()
            continue
        key = (r["round"], r["client_id"], r["metric_name"])
        if key in in_segment:
            raise LedgerRejected("LEDGER_DUPLICATE", "한 세그먼트 안에 같은 (round, client_id, metric_name) 행이 둘이다")
        in_segment.add(key)
        latest[key] = r["metric_value"]
    steps: dict[int, str] = {}
    for (rnd, _client, metric), value in latest.items():
        if metric == "optimizer_steps":
            steps[_int_text(rnd, "epoch 행의 round")] = value
    if not steps:
        raise LedgerRejected("LEDGER_NO_STEPS", "optimizer_steps 행이 없다")
    last = _integral(steps[max(steps)], "마지막 epoch 의 optimizer_steps")
    if last != end_value:
        raise LedgerRejected("LEDGER_END_MISMATCH", "끝 행의 값이 마지막 epoch 의 optimizer_steps 와 다르다")


def _fed_check(rows: list[dict[str, str]], end_value: int, n_clients: int) -> None:
    """라운드 행의 규칙 ①~③ — 끝 행을 뺀 데이터 행에 건다. ④(끝 행 = R)는 부르는 쪽이 먼저 봤다."""
    seen: set[tuple[str, str, str]] = set()
    server_rounds: set[int] = set()
    steps_by_round: dict[int, set[int]] = {}
    for r in rows:
        key = (r["round"], r["client_id"], r["metric_name"])
        if key in seen:
            raise LedgerRejected("LEDGER_DUPLICATE", "연합 원장에 같은 (round, client_id, metric_name) 행이 둘이다 — 재개가 없다")
        seen.add(key)
        rnd = _int_text(r["round"], "라운드 행의 round")
        cid = r["client_id"]
        if cid == "server":
            server_rounds.add(rnd)
            continue
        if not _INT.match(cid):
            raise LedgerRejected("LEDGER_CLIENT", "연합 원장의 참여자 client_id 가 정수의 정규 표기가 아니다")
        client = int(cid)
        if r["metric_name"] == "optimizer_steps":
            steps_by_round.setdefault(rnd, set()).add(client)
    want_rounds = set(range(end_value))
    if server_rounds != want_rounds:
        raise LedgerRejected("LEDGER_ROUNDS", "server 행의 round 집합이 {0, …, R−1} 이 아니다(R = 끝 행의 값)")
    want_clients = set(range(n_clients))
    for rnd in sorted(want_rounds):
        if steps_by_round.get(rnd, set()) != want_clients:
            raise LedgerRejected("LEDGER_CLIENTS", f"라운드 {rnd} 의 optimizer_steps 참여자가 {{0, …, {n_clients - 1}}} 이 아니다")
    extra = set(steps_by_round) - want_rounds
    if extra:
        raise LedgerRejected("LEDGER_ROUNDS", "참여자 행의 round 가 서버 라운드 밖이다")


def read_ledger_view(source: Path | str | bytes, *, n_rounds: int | None = None,
                     n_clients: int | None = None) -> LedgerView:
    """원장 파일(경로 또는 바이트) → `LedgerView`. 규칙을 어기면 `LedgerRejected`.

    연합 원장이면 `n_rounds`(등록의 `budget_r`) · `n_clients`(`FED_CLIENTS`) 가 **둘 다** 있어야 한다 — 없으면
    `ValueError`(호출부의 잘못이다). 로컬 · 중앙은 두 값을 보지 않는다.
    """
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    rows = _rows(raw)
    if not rows:
        raise LedgerRejected("LEDGER_NOT_CLOSED", "데이터 행이 없다")
    end = rows[-1]
    if end["metric_name"] != END_METRIC:
        raise LedgerRejected("LEDGER_NOT_CLOSED", "마지막 데이터 행이 final_adapter_step 이 아니다 — 닫히지 않은 원장이다")
    final_step = _integral(end["metric_value"], "끝 행의 metric_value")
    if final_step < 1:
        raise LedgerRejected("LEDGER_VALUE", "끝 행의 값이 1 보다 작다 — 학습하지 않은 어댑터다")
    cell = end["cell"]
    if cell not in CELL_END_CLIENT:
        raise LedgerRejected("LEDGER_CELL", "끝 행의 cell 이 통합형 칸이 아니다")
    if end["client_id"] not in CELL_END_CLIENT[cell]:
        raise LedgerRejected("LEDGER_CLIENT", "끝 행의 client_id 가 그 칸의 참여자가 아니다")
    body = rows[:-1]
    for r in body:
        if r["run_id"] != end["run_id"] or r["cell"] != cell:
            raise LedgerRejected("LEDGER_IDENTITY", "데이터 행의 run_id · cell 이 끝 행과 다르다 — 실행 하나의 원장이 아니다")
        if r["metric_name"] == END_METRIC:
            raise LedgerRejected("LEDGER_NOT_CLOSED", "끝 행 뒤에 행이 있다 — 닫힌 원장에 덧붙였다")
    if cell == "uni_fed":
        if type(n_rounds) is not int or type(n_clients) is not int or n_rounds < 1 or n_clients < 1:
            raise ValueError("연합 원장은 등록의 R 과 참여자 수 K 를 받아야 읽는다 — 기본값이 없다")
        if final_step != n_rounds or _int_text(end["round"], "끝 행의 round") != n_rounds:
            raise LedgerRejected("LEDGER_ROUNDS", "연합 끝 행의 round · 값이 등록의 R 이 아니다")
        _fed_check(body, final_step, n_clients)
    else:
        if any(r["client_id"] != end["client_id"] for r in body):
            raise LedgerRejected("LEDGER_CLIENT", "로컬 · 중앙 원장에 다른 참여자의 행이 있다 — 모델마다 파일 하나다")
        _local_check(body, final_step)
    client = None if end["client_id"] in ("central", "server") else end["client_id"]
    return LedgerView(run_id=end["run_id"], file_sha256=hashlib.sha256(raw).hexdigest(),
                      final_step=final_step, cell=cell, client=client)
