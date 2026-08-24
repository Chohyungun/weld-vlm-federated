"""fl/atomic_log.py 테스트 — 원자 로그 스키마와 결측 검사.

이 로그를 지금 남기지 않으면 나중에 전 실험을 다시 돌려야 한다. 그래서 검사할 것은
"기록이 되는가"가 아니라 **"빠진 것을 찾아낼 수 있는가"** 다.
"""

from __future__ import annotations

import csv

import pytest

from fl.atomic_log import FIELDS, AtomicLog, AtomicRecord, RoundTimer, new_run_id


def _log(tmp_path, cell="sep_fed"):
    return AtomicLog(
        tmp_path / "atomic.csv",
        run_id=new_run_id(cell, 0, "260824"),
        seed=0,
        cell=cell,
        split_hash="abc123",
    )


def test_열_구성이_지표설계_스키마와_같다():
    """열 순서를 바꾸면 이전 실험 로그와 합칠 수 없다."""
    assert FIELDS == (
        "run_id", "seed", "cell", "split_hash", "client_id", "round",
        "n_train_samples", "metric_name", "metric_value",
        "bytes_up", "bytes_down", "wall_time",
    )


def test_지표_dict이_줄_단위로_펼쳐진다(tmp_path):
    log = _log(tmp_path)
    n = log.log_round(
        round_idx=0, client_id=2, n_train_samples=8000,
        metrics={"param_l2": 1.5, "epochs_ran": 2.0},
        bytes_up=1000, bytes_down=1000, wall_time=12.5,
    )
    assert n == 2
    rows = log.read_rows()
    assert {r["metric_name"] for r in rows} == {"param_l2", "epochs_ran"}
    assert all(r["client_id"] == "2" and r["round"] == "0" for r in rows)
    assert all(r["split_hash"] == "abc123" for r in rows)
    assert rows[0]["run_id"] == "sep_fed_s0_260824"


def test_append로_라운드가_쌓인다(tmp_path):
    log = _log(tmp_path)
    for r in range(3):
        log.log_round(round_idx=r, client_id=0, n_train_samples=100, metrics={"m": float(r)})
    rows = log.read_rows()
    assert [r["round"] for r in rows] == ["0", "1", "2"]
    # 헤더는 한 번만
    with (tmp_path / "atomic.csv").open(encoding="utf-8") as fh:
        assert sum(1 for line in fh if line.startswith("run_id,")) == 1


def test_결측_라운드를_찾아낸다(tmp_path):
    """RQ3 은 라운드별 궤적에서 나온다. 중간이 비면 곡선을 그릴 수 없다."""
    log = _log(tmp_path)
    for r in range(3):
        for c in (0, 1, 2):
            if (r, c) == (1, 2):      # 라운드 1 에서 C3 이탈
                continue
            log.log_round(round_idx=r, client_id=c, n_train_samples=1, metrics={"m": 1.0})
    gaps = log.audit_rounds(3, [0, 1, 2])
    assert gaps and "결측 1건" in gaps[0]
    assert "(1, '2')" in gaps[0]


def test_전부_있으면_결측이_없다(tmp_path):
    log = _log(tmp_path)
    for r in range(2):
        for c in (0, 1):
            log.log_round(round_idx=r, client_id=c, n_train_samples=1, metrics={"m": 1.0})
    assert log.audit_rounds(2, [0, 1]) == []


def test_서버_행도_클라이언트와_같은_스키마다(tmp_path):
    """집계 진단을 별도 파일로 빼면 나중에 조인해야 한다."""
    log = _log(tmp_path)
    log.log_round(round_idx=0, client_id="server", n_train_samples=400,
                  metrics={"global_l2": 3.0})
    row = log.read_rows()[0]
    assert row["client_id"] == "server" and row["metric_name"] == "global_l2"
    assert set(row) == set(FIELDS)


def test_열_구성이_다른_기존_파일은_거부한다(tmp_path):
    p = tmp_path / "atomic.csv"
    p.write_text("run_id,seed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="열 구성이 다르다"):
        AtomicLog(p, run_id="x", seed=0, cell="sep_fed", split_hash="h")


def test_run_id는_같은_run에서_동일하다():
    """서버와 클라이언트가 각자 시각을 읽으면 run 이 쪼개진다."""
    assert new_run_id("sep_fed", 0, "260824") == new_run_id("sep_fed", 0, "260824")
    assert new_run_id("sep_fed", 0, "260824") != new_run_id("sep_fed", 1, "260824")


def test_타이머는_구간을_나눠_잰다():
    t = RoundTimer()
    a = t.lap()
    b = t.lap()
    assert a >= 0 and b >= 0
    # lap 은 누적이 아니라 구간이다
    assert b < 1.0


def test_로컬중앙_칸은_통신량이_0이다(tmp_path):
    """교환이 없는 칸에서 0 은 결측이 아니라 정의상 참이다."""
    log = _log(tmp_path, cell="sep_central")
    log.log_round(round_idx=0, client_id="central", n_train_samples=3000,
                  metrics={"param_l2": 2.0}, bytes_up=0, bytes_down=0)
    row = log.read_rows()[0]
    assert row["bytes_up"] == "0" and row["bytes_down"] == "0"
    assert row["cell"] == "sep_central"
