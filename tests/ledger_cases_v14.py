"""통합형 학습 원장의 사례 — 평가 쪽 읽기와 쓰는 쪽 규칙 함수를 **같은 원장**에 넣어 맞대는 데 쓴다.

원장은 손으로 만든다(쓰는 쪽 코드를 부르지 않는다). 줄끝은 csv 기본(CRLF) — 쓰는 쪽과 같다.
사례마다 평가 쪽이 낼 결과(`want`)를 적는다. 쓰는 쪽이 받는데 평가 쪽만 거부하는 사례는 **까닭**(`stricter`)을 단다 —
그 목록 밖에서 두 구현이 갈리면 시험이 떨어진다. 쓰는 쪽이 거부하는데 평가 쪽이 받는 사례는 하나도 두지 않는다.

쓰는 쪽의 판정은 시험이 쓰는 쪽 규칙 함수를 **직접 불러** 낸다(`tests/test_ledger_v14.py`).
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field

from evaluation.ledger_v14 import LEDGER_FIELDS

RUN = "uni_local_C1_20260828_20261001T0900"
FED_RUN = "uni_fed_20260828_20261001T0900"
FED_CLIENTS = 3


def ledger(rows: list[dict], *, header: tuple[str, ...] = LEDGER_FIELDS) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf)                    # 기본 줄끝 CRLF — 쓰는 쪽과 같다
    w.writerow(header)
    for r in rows:
        w.writerow([r.get(k, "") for k in LEDGER_FIELDS])
    return buf.getvalue().encode("utf-8")


def row(rnd, client, metric, value, *, run=RUN, cell="uni_local") -> dict:
    return {"run_id": run, "seed": "20260828", "cell": cell, "split_hash": "d" * 64, "client_id": client,
            "round": str(rnd), "n_train_samples": "100", "metric_name": metric, "metric_value": str(value),
            "bytes_up": "0", "bytes_down": "0", "wall_time": "1.0"}


def local_rows(client="C1", cell="uni_local", run=RUN, epochs=3, per_epoch=4) -> list[dict]:
    out = [row(0, client, "process_start", 0.0, run=run, cell=cell),
           row(0, client, "process_started_unix", 1.7e9, run=run, cell=cell)]
    for e in range(1, epochs + 1):
        out += [row(e, client, "epochs_ran", float(e), run=run, cell=cell),
                row(e, client, "optimizer_steps", float(e * per_epoch), run=run, cell=cell),
                row(e, client, "mean_ce", 0.5, run=run, cell=cell)]
    return out


def end(value, client="C1", cell="uni_local", run=RUN, rnd=None) -> dict:
    return row(value if rnd is None else rnd, client, "final_adapter_step", float(value), run=run, cell=cell)


def fed_rows(rounds=2, clients=FED_CLIENTS) -> list[dict]:
    out = []
    for r in range(rounds):
        for c in range(clients):
            out += [row(r, str(c), "optimizer_steps", 4.0, run=FED_RUN, cell="uni_fed"),
                    row(r, str(c), "mean_ce", 0.5, run=FED_RUN, cell="uni_fed")]
        out.append(row(r, "server", "bytes_round", 100.0, run=FED_RUN, cell="uni_fed"))
    return out


def fed_end(value, rnd=None) -> dict:
    return row(value if rnd is None else rnd, "server", "final_adapter_step", float(value), run=FED_RUN, cell="uni_fed")


def seam(epoch, client="C1") -> list[dict]:
    return [row(epoch, client, "process_start", float(epoch)), row(epoch, client, "process_started_unix", 1.8e9)]


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    """`local` · `central` · `fed`. 쓰는 쪽에서 부를 규칙 함수가 갈린다."""
    raw: bytes
    want: int | str
    """평가 쪽의 결과 — 받으면 `final_step`(정수), 거부하면 사유 코드(문자열)."""
    n_rounds: int | None = None
    """연합만 — 등록의 R(`budget_r`)."""
    n_clients: int | None = None
    """연합만 — 참여자 수 K."""
    stricter: str | None = None
    """쓰는 쪽은 받는데 평가 쪽만 거부하는 까닭. 계약보다 엄격한 쪽이어야 하고, 쓰는 쪽이 쓰는 원장이 아니어야 한다."""
    extra: dict = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()


def _resumed() -> list[dict]:
    rows = local_rows(epochs=2)
    return rows + seam(2) + [row(2, "C1", "optimizer_steps", 8.0), row(3, "C1", "optimizer_steps", 12.0)]


def _resumed_overwrites() -> list[dict]:
    """이음매 뒤의 값이 앞의 값을 대체한다 — 앞 · 뒤 값을 다르게 둔다(M-5). 뒤의 것이 이기면 끝 값 13 과 맞는다."""
    rows = local_rows(epochs=3)
    return rows + seam(3) + [row(3, "C1", "optimizer_steps", 13.0)]


def _with_short_line() -> bytes:
    """끝 행 앞에 칸이 11 개인 행을 하나 넣는다(마지막 칸을 뺀다)."""
    head = ledger(local_rows())
    short = ",".join(row(3, "C1", "lr", 0.1)[k] for k in LEDGER_FIELDS[:-1]).encode("utf-8") + b"\r\n"
    tail = ledger([end(12)]).split(b"\r\n", 1)[1]
    return head + short + tail


def _fed_bad_client(value: str) -> list[dict]:
    rows = fed_rows(2)
    return rows + [row(1, value, "mean_ce", 0.5, run=FED_RUN, cell="uni_fed")]


CASES: tuple[Case, ...] = (
    # ── 받는다 ───────────────────────────────────────────────────────────
    Case("로컬", "local", ledger(local_rows() + [end(12)]), 12),
    Case("로컬-끝행-round-무시", "local", ledger(local_rows() + [end(12, rnd=999)]), 12),
    Case("로컬-재개", "local", ledger(_resumed() + [end(12)]), 12),
    Case("로컬-재개-뒤가-이긴다", "local", ledger(_resumed_overwrites() + [end(13)]), 13),
    Case("중앙", "central", ledger(local_rows("central", "uni_central", "uni_central_x")
                                   + [end(12, "central", "uni_central", "uni_central_x")]), 12),
    Case("연합", "fed", ledger(fed_rows() + [fed_end(2)]), 2, n_rounds=2, n_clients=3),
    Case("연합-참여자-둘", "fed", ledger(fed_rows(2, clients=2) + [fed_end(2)]), 2, n_rounds=2, n_clients=2),
    # ── 둘 다 거부한다 ───────────────────────────────────────────────────
    Case("닫히지-않음", "local", ledger(local_rows()), "LEDGER_NOT_CLOSED"),
    Case("끝행-뒤-덧붙임", "local", ledger(local_rows() + [end(12), row(4, "C1", "mean_ce", 0.5)]), "LEDGER_NOT_CLOSED"),
    Case("끝행-둘", "local", ledger(local_rows() + [end(12), end(12)]), "LEDGER_NOT_CLOSED",
         stricter="끝 행이 둘 — 쓰는 쪽 로컬 규칙은 마지막 행 앞의 끝 행을 보지 않는다. 닫힌 원장에는 덧붙이지 않는다(2판 §2-6 의 닫힘)"),
    Case("끝값-비정수", "local", ledger(local_rows() + [row(12, "C1", "final_adapter_step", "12.5")]), "LEDGER_VALUE"),
    Case("로컬-끝값-불일치", "local", ledger(local_rows() + [end(11)]), "LEDGER_END_MISMATCH"),
    Case("로컬-재개-앞값으로-끝행", "local", ledger(_resumed_overwrites() + [end(12)]), "LEDGER_END_MISMATCH"),
    Case("한-세그먼트-겹침", "local", ledger(local_rows() + [row(3, "C1", "optimizer_steps", 12.0), end(12)]),
         "LEDGER_DUPLICATE"),
    Case("로컬-스텝행-없음", "local",
         ledger([r for r in local_rows() if r["metric_name"] != "optimizer_steps"] + [end(12)]), "LEDGER_NO_STEPS"),
    Case("연합-빠진-라운드", "fed",
         ledger([r for r in fed_rows(3) if not (r["client_id"] == "server" and r["round"] == "1")] + [fed_end(3)]),
         "LEDGER_ROUNDS", n_rounds=3, n_clients=3),
    Case("연합-끝값-라운드수-다름", "fed", ledger(fed_rows(2) + [fed_end(3)]), "LEDGER_ROUNDS", n_rounds=2, n_clients=3),
    Case("연합-등록R-다름", "fed", ledger(fed_rows(2) + [fed_end(2)]), "LEDGER_ROUNDS", n_rounds=3, n_clients=3),
    Case("연합-끝행-round-다름", "fed", ledger(fed_rows(2) + [fed_end(2, rnd=5)]), "LEDGER_ROUNDS", n_rounds=2, n_clients=3),
    Case("연합-겹친-행", "fed", ledger(fed_rows(2) + [row(1, "0", "mean_ce", 0.5, run=FED_RUN, cell="uni_fed"), fed_end(2)]),
         "LEDGER_DUPLICATE", n_rounds=2, n_clients=3),
    Case("연합-마이너스1-참여자", "fed",
         ledger([r for r in fed_rows(2) if not (r["client_id"] == "2" and r["round"] == "0")]
                + [row(0, "-1", "optimizer_steps", 4.0, run=FED_RUN, cell="uni_fed"), fed_end(2)]),
         "LEDGER_CLIENTS", n_rounds=2, n_clients=3),
    Case("연합-참여자수-다름", "fed", ledger(fed_rows(2, clients=2) + [fed_end(2)]), "LEDGER_CLIENTS", n_rounds=2, n_clients=3),
    Case("연합-라운드밖-참여자행", "fed",
         ledger(fed_rows(2) + [row(2, "0", "optimizer_steps", 4.0, run=FED_RUN, cell="uni_fed"), fed_end(2)]),
         "LEDGER_ROUNDS", n_rounds=2, n_clients=3,
         stricter="라운드 밖의 참여자 행 — 쓰는 쪽은 R 안의 라운드만 세고 밖의 행을 보지 않는다. 서버가 쓰지 않는 원장이다"),
    Case("로컬-round-실수표기", "local",
         ledger([dict(r, round=f"{int(r['round'])}.0") for r in local_rows()] + [end(12)]), "LEDGER_VALUE"),
    Case("머리행-다름", "local", ledger(local_rows() + [end(12)], header=LEDGER_FIELDS[:-1] + ("x",)), "LEDGER_FORM"),
    Case("빈-파일", "local", b"", "LEDGER_FORM"),
    Case("칸-수-다름", "local", _with_short_line(), "LEDGER_FORM",
         stricter="칸이 11 개인 행 — 쓰는 쪽 csv 읽기는 빠진 칸을 None 으로 채워 받는다. 쓰는 쪽은 12열로만 쓴다"),
    Case("머리행만", "local", ledger([]), "LEDGER_NOT_CLOSED"),
    Case("UTF-8-아님", "local", ledger(local_rows() + [end(12)]).replace(b"uni_local_C1_", b"uni_local_C1\xff", 1),
         "LEDGER_FORM"),
    # ── 평가 쪽만 거부한다 — 계약보다 엄격한 쪽 ─────────────────────────────
    Case("다른-실행-행", "local", ledger(local_rows() + [row(3, "C1", "lr", 0.1, run="other_run"), end(12)]),
         "LEDGER_IDENTITY", stricter="데이터 행의 run_id 가 끝 행과 다르다 — 원장은 실행 하나의 것이다"),
    Case("다른-칸-행", "local", ledger(local_rows() + [row(3, "C1", "lr", 0.1, cell="uni_central"), end(12)]),
         "LEDGER_IDENTITY", stricter="데이터 행의 cell 이 끝 행과 다르다 — 원장은 칸 하나의 것이다"),
    Case("로컬-다른-참여자-행", "local", ledger(local_rows() + [row(3, "C2", "lr", 0.1), end(12)]),
         "LEDGER_CLIENT", stricter="로컬 원장에 다른 참여자의 행 — 모델마다 파일 하나다(2판 §2-6 의 자리)"),
    Case("끝행-참여자가-칸의-것이-아님", "local", ledger(local_rows() + [end(12, client="central")]), "LEDGER_CLIENT"),
    Case("통합형-칸이-아님", "local",
         ledger(local_rows(cell="det_local") + [end(12, cell="det_local")]),
         "LEDGER_CELL", stricter="끝 행의 cell 이 통합형 칸이 아니다"),
    Case("연합-비정수-참여자", "fed", ledger(_fed_bad_client("C1") + [fed_end(2)]), "LEDGER_CLIENT",
         n_rounds=2, n_clients=3, stricter="연합 참여자 행의 client_id 가 정수가 아니다 — 쓰는 쪽은 optimizer_steps 행만 센다"),
    Case("연합-참여자-01", "fed", ledger(_fed_bad_client("01") + [fed_end(2)]), "LEDGER_CLIENT",
         n_rounds=2, n_clients=3, stricter="client_id 가 정수의 정규 표기가 아니다(`01`) — `1` 과의 겹침을 ② 가 못 잡는다"),
    Case("연합-참여자-공백", "fed", ledger(_fed_bad_client(" 1") + [fed_end(2)]), "LEDGER_CLIENT",
         n_rounds=2, n_clients=3, stricter="client_id 가 정수의 정규 표기가 아니다(` 1`)"),
    Case("로컬-끝값-0", "local", ledger(local_rows(per_epoch=0) + [end(0)]), "LEDGER_VALUE",
         stricter="끝 행의 값이 0 — 옵티마이저 갱신이 없는 어댑터다. 예산이 양수라 쓰는 쪽이 쓰지 않는 원장이다"),
    Case("이음매-절반", "local",
         ledger(local_rows() + [row(3, "C1", "process_started_unix", 1.8e9), row(3, "C1", "optimizer_steps", 12.0), end(12)]),
         "LEDGER_DUPLICATE",
         stricter="process_start 없이 process_started_unix 만으로 세그먼트를 끊지 않는다 — 쓰는 쪽은 두 행을 함께 쓴다"),
    Case("빈-데이터-행", "local", ledger(local_rows()[:3] + [{k: "" for k in LEDGER_FIELDS}] + local_rows()[3:] + [end(12)]),
         "LEDGER_FORM", stricter="모든 칸이 빈 행 — 쓰는 쪽은 건너뛰지만 쓰지 않는 행이다. 원시 바이트 해시가 그 행을 품는다"),
)


def case(name: str) -> Case:
    return next(c for c in CASES if c.name == name)
