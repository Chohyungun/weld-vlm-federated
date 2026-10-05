"""통합형 학습 원장 → `LedgerView` — 평가 쪽 독립 구현(07번 §30-7 · 리허설 미니스펙 2판 반영판 §2-6).

원장은 손으로 만든다 — 쓰는 쪽 코드를 부르지 않는다. 줄끝은 csv 기본(CRLF)이다. 사례와 만드는 도구는 `tests/ledger_cases_v14.py`.

**두 구현의 대조**(검수 16번 I-1). 쓰는 쪽의 **규칙 함수**(로컬 · 중앙 `check_local_end_row`, 연합 `check_fed_ledger`)에
같은 원장을 넣는다 — 양성 · 음성 사례 전부. 쓰는 쪽 모듈을 **바로 가져온다** — 없으면 건너뛰지 않고 수집에서 떨어진다.
(쓰는 쪽 코드가 본줄기에 들어오기 전에는 그 판정을 고정물로 두었다. 들어온 뒤 고정물을 걷었다 — 읽는 곳이 없는 표시를 남기지 않는다.)
"""

from __future__ import annotations

import hashlib
import importlib

import pytest

from fl.uni_fed import check_fed_ledger
from vlm.train_cell import check_local_end_row, ledger_view

from evaluation.bundle_v14 import LedgerView
from evaluation.ledger_v14 import FED_CLIENTS, LEDGER_FIELDS, LedgerRejected, read_ledger_view
from tests.ledger_cases_v14 import CASES, FED_RUN, RUN, end, fed_end, fed_rows, ledger, local_rows, row


def fed_view(raw: bytes, *, n_rounds: int = 2, n_clients: int = FED_CLIENTS) -> LedgerView:
    return read_ledger_view(raw, n_rounds=n_rounds, n_clients=n_clients)


# ---------------------------------------------------------------- 정상

def test_로컬_원장() -> None:
    raw = ledger(local_rows() + [end(12)])
    view = read_ledger_view(raw)
    assert view == LedgerView(run_id=RUN, file_sha256=hashlib.sha256(raw).hexdigest(), final_step=12,
                              cell="uni_local", client="C1")


def test_중앙_원장의_참여자는_None_이다() -> None:
    run = "uni_central_20260828_x"
    raw = ledger(local_rows("central", "uni_central", run) + [end(12, "central", "uni_central", run)])
    assert read_ledger_view(raw).client is None


def test_해시는_원시_바이트이고_CR_을_거부하지_않는다() -> None:
    raw = ledger(local_rows() + [end(12)])
    assert b"\r\n" in raw
    assert read_ledger_view(raw).file_sha256 == hashlib.sha256(raw).hexdigest()
    lf = raw.replace(b"\r\n", b"\n")
    assert read_ledger_view(lf).file_sha256 != read_ledger_view(raw).file_sha256, "줄끝을 정규화하지 않는다"


def test_끝_행의_값은_12_점_0_꼴로_적혀도_정수로_읽는다() -> None:
    assert read_ledger_view(ledger(local_rows() + [end(12)])).final_step == 12


def test_끝_행의_round_는_읽지_않는다() -> None:
    assert read_ledger_view(ledger(local_rows() + [end(12, rnd=999)])).final_step == 12


def test_재개로_겹친_epoch_는_이음매_뒤의_것이_대체한다() -> None:
    rows = local_rows(epochs=2)
    rows += [row(2, "C1", "process_start", 2.0), row(2, "C1", "process_started_unix", 1.7e9),
             row(2, "C1", "optimizer_steps", 8.0), row(3, "C1", "optimizer_steps", 12.0)]
    assert read_ledger_view(ledger(rows + [end(12)])).final_step == 12


def test_연합_원장() -> None:
    raw = ledger(fed_rows() + [fed_end(2)])
    view = fed_view(raw)
    assert (view.final_step, view.cell, view.client, view.run_id) == (2, "uni_fed", None, FED_RUN)


def test_원장의_열은_쓰는_쪽_정의와_같다() -> None:
    atomic_log = importlib.import_module("fl.atomic_log")
    assert LEDGER_FIELDS == tuple(atomic_log.FIELDS)


# ---------------------------------------------------------------- 거부

def _rejected(raw: bytes, code: str, **kw) -> None:
    with pytest.raises(LedgerRejected) as exc:
        read_ledger_view(raw, **kw)
    assert exc.value.code == code, exc.value


FED = {"n_rounds": 2, "n_clients": FED_CLIENTS}


def test_닫히지_않은_원장() -> None:
    _rejected(ledger(local_rows()), "LEDGER_NOT_CLOSED")


def test_끝_행_뒤에_덧붙인_원장() -> None:
    _rejected(ledger(local_rows() + [end(12), row(4, "C1", "mean_ce", 0.4)]), "LEDGER_NOT_CLOSED")


def test_끝_행이_둘인_원장() -> None:
    _rejected(ledger(local_rows() + [end(12), end(12)]), "LEDGER_NOT_CLOSED")


@pytest.mark.parametrize("value", ["12.5", "nan", "inf", "x"])
def test_끝_행의_값이_정수가_아니면_거부한다(value: str) -> None:
    bad = end(12)
    bad["metric_value"] = value
    _rejected(ledger(local_rows() + [bad]), "LEDGER_VALUE")


def test_로컬_끝_행이_마지막_epoch_의_스텝과_다르면_거부한다() -> None:
    _rejected(ledger(local_rows() + [end(11)]), "LEDGER_END_MISMATCH")


def test_한_세그먼트_안의_겹친_행은_거부한다() -> None:
    rows = local_rows() + [row(3, "C1", "optimizer_steps", 12.0)]
    _rejected(ledger(rows + [end(12)]), "LEDGER_DUPLICATE")


def test_다른_실행의_행이_섞이면_거부한다() -> None:
    rows = local_rows()
    rows[3] = row(1, "C1", "optimizer_steps", 4.0, run="other_run")
    _rejected(ledger(rows + [end(12)]), "LEDGER_IDENTITY")


def test_로컬_원장에_다른_참여자가_섞이면_거부한다() -> None:
    rows = local_rows() + [row(1, "C2", "mean_ce", 0.5)]
    _rejected(ledger(rows + [end(12)]), "LEDGER_CLIENT")


@pytest.mark.parametrize(("cell", "client"), [("uni_local", "uni_local_C1"), ("uni_local", "central"),
                                              ("uni_central", "C1"), ("uni_fed", "0")])
def test_끝_행의_참여자가_그_칸의_것이_아니면_거부한다(cell: str, client: str) -> None:
    rows = [row(1, client, "optimizer_steps", 4.0, cell=cell)]
    _rejected(ledger(rows + [end(4, client, cell)]), "LEDGER_CLIENT")


def test_통합형_칸이_아니면_거부한다() -> None:
    rows = [row(1, "C1", "optimizer_steps", 4.0, cell="sep_local")]
    _rejected(ledger(rows + [end(4, "C1", "sep_local")]), "LEDGER_CELL")


def test_머리행이_12열이_아니면_거부한다() -> None:
    raw = ledger(local_rows() + [end(12)]).replace(b"wall_time", b"wall")
    _rejected(raw, "LEDGER_FORM")


def test_연합_빠진_라운드() -> None:
    rows = [r for r in fed_rows(3) if not (r["client_id"] == "server" and r["round"] == "1")]
    _rejected(ledger(rows + [fed_end(3)]), "LEDGER_ROUNDS", n_rounds=3, n_clients=FED_CLIENTS)


def test_연합_끝_행의_값이_라운드_수와_다르면_거부한다() -> None:
    _rejected(ledger(fed_rows(2) + [fed_end(3)]), "LEDGER_ROUNDS", **FED)
    _rejected(ledger(fed_rows(3) + [fed_end(3)]), "LEDGER_ROUNDS", **FED)


def test_연합_겹친_행은_거부한다() -> None:
    rows = fed_rows(2) + [row(1, "0", "optimizer_steps", 4.0, run=FED_RUN, cell="uni_fed")]
    _rejected(ledger(rows + [fed_end(2)]), "LEDGER_DUPLICATE", **FED)


def test_연합_기본값_마이너스_1_참여자는_받지_않는다() -> None:
    rows = fed_rows(2)
    for r in rows:
        if r["client_id"] == "2":
            r["client_id"] = "-1"
    _rejected(ledger(rows + [fed_end(2)]), "LEDGER_CLIENTS", **FED)


def test_연합_참여자_수가_등록과_다르면_거부한다() -> None:
    _rejected(ledger(fed_rows(2, clients=2) + [fed_end(2)]), "LEDGER_CLIENTS", **FED)
    assert fed_view(ledger(fed_rows(2, clients=2) + [fed_end(2)]), n_clients=2).final_step == 2


def test_연합_참여자가_정수가_아니면_거부한다() -> None:
    rows = fed_rows(2) + [row(0, "C1", "mean_ce", 0.5, run=FED_RUN, cell="uni_fed")]
    _rejected(ledger(rows + [fed_end(2)]), "LEDGER_CLIENT", **FED)


@pytest.mark.parametrize("kw", [{}, {"n_rounds": 2}, {"n_clients": 3}, {"n_rounds": 0, "n_clients": 3},
                                {"n_rounds": True, "n_clients": 3}])
def test_연합_원장은_R_과_K_를_받아야_읽는다(kw: dict) -> None:
    """기본값이 없다 — 등록에서 읽지 않은 참여자 수가 조용히 3 으로 서지 않게(검수 16번 M-6)."""
    with pytest.raises(ValueError, match="R"):
        read_ledger_view(ledger(fed_rows() + [fed_end(2)]), **kw)


def test_로컬_원장은_R_과_K_를_보지_않는다() -> None:
    raw = ledger(local_rows() + [end(12)])
    assert read_ledger_view(raw, n_rounds=99, n_clients=7) == read_ledger_view(raw)


# ---------------------------------------------------------------- 사례표

@pytest.mark.parametrize("c", CASES, ids=[c.name for c in CASES])
def test_사례마다_평가_쪽의_결과(c) -> None:
    try:
        got = read_ledger_view(c.raw, n_rounds=c.n_rounds, n_clients=c.n_clients).final_step
    except LedgerRejected as exc:
        got = exc.code
    assert got == c.want


# ---------------------------------------------------------------- 두 구현 대조 — 규칙 함수

def _mine(c) -> dict:
    try:
        v = read_ledger_view(c.raw, n_rounds=c.n_rounds, n_clients=c.n_clients)
    except LedgerRejected as exc:
        return {"accept": False, "code": exc.code}
    return {"accept": True, "view": vars(v)}


def _writer(c, tmp_path) -> dict:
    """쓰는 쪽 규칙 함수의 판정. 거부의 꼴은 함수마다 달라 종류만 적는다."""
    p = tmp_path / "train_ledger.csv"
    p.write_bytes(c.raw)
    try:
        if c.kind == "fed":
            check_fed_ledger(p, num_rounds=c.n_rounds, num_clients=c.n_clients, closed=True)
            view = ledger_view(p)
        else:
            step = check_local_end_row(p)
            view = ledger_view(p)
            assert step == view["final_step"]
    except Exception as exc:  # noqa: BLE001 — 거부의 꼴이 함수마다 다르다
        return {"accept": False, "error": type(exc).__name__}
    return {"accept": True, "view": view}


@pytest.mark.parametrize("c", CASES, ids=[c.name for c in CASES])
def test_두_구현이_같은_원장에서_같은_판정을_낸다(c, tmp_path) -> None:
    writer, mine = _writer(c, tmp_path), _mine(c)
    if writer["accept"] and mine["accept"]:
        assert mine["view"] == writer["view"], "둘 다 받으면 다섯 값이 같다"
        assert c.stricter is None
    elif writer["accept"]:
        assert c.stricter, f"쓰는 쪽은 받고 평가 쪽만 거부한다 — 까닭이 사례에 없다: {mine['code']}"
    elif mine["accept"]:
        pytest.fail("쓰는 쪽은 거부하는데 평가 쪽이 받는다 — 평가 쪽이 더 느슨하다")
    else:
        assert c.stricter is None, "둘 다 거부하는 사례에 '평가 쪽만' 까닭이 붙어 있다"


def test_사례에는_양성과_음성이_둘_다_있다(tmp_path) -> None:
    """양성만 맞대면 규칙 논리를 보지 못한다 — 쓰는 쪽이 받는 사례와 거부하는 사례가 둘 다 있어야 한다."""
    verdicts = []
    for i, c in enumerate(CASES):
        d = tmp_path / f"c{i:02d}"
        d.mkdir()
        verdicts.append(_writer(c, d)["accept"])
    assert any(verdicts) and not all(verdicts)
