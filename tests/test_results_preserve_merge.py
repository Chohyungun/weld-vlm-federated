"""34번 §3-1 8-1 — 재개 시 loss CSV 보존(rename) + 읽기 전용 병합.

상류 `BaseTrainer.__init__` 은 `args.resume` 가 거짓이면 같은 save_dir 의 `results.csv` 를 지운다
(`engine/trainer.py:199-201`). 우리 재개는 상류 resume 을 쓰지 않으므로 이전 프로세스의 loss 행이
사라졌다. 지금은 `train_round` 가 트레이너를 만들기 **전에** 옆 이름으로 옮기고, 병합은 학습
경로 밖의 읽기 전용 유틸이 파생 테이블로만 한다(원본 CSV 무수정, F03 15 열 불변).

**구판 짧은 행을 오른쪽으로 패딩하면 안 된다**(정적 검토 Critical). F03 의 마지막 epoch 행은
`metrics/*`·`val/*` 7 열이 **가운데에서** 빠진 8 열이고 뒤 3 값이 LR 이다 — 오른쪽 패딩하면 LR 이
`metrics/mAP50(B)` 값으로 들어앉아, 재지 않은 성능 지표가 실측값처럼 파생 테이블에 실린다.
아래 골든 행은 동결 산출물 `outputs/main_c/seed1/sep_central/r000_c0/results.csv` 의 마지막 줄이다.
"""

from __future__ import annotations

import builtins
import csv
import io
import math
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from detection import results_merge
from detection.results_merge import (MergeReport, before_resume_files, fragment_name,
                                     merge_results, read_results, sanitize_run_id)
from detection.round_runner import _preserve_results_csv

HEADER = ["epoch", "time", "train/box_loss", "train/cls_loss", "train/dfl_loss",
          "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
          "val/box_loss", "val/cls_loss", "val/dfl_loss", "lr/pg0", "lr/pg1", "lr/pg2"]
#: 시드 1 ③ 동결 CSV 의 마지막 줄(8 열). 뒤 3 값은 LR 이고 epoch 99 행의 lr/pg0 과 같은 자리다.
SEED1_F03_ROW = "100,38198.9,1.82596,1.30662,1.21453,0.000102443,0.000102443,0.000102443"
SEED1_F03_LR = 0.000102443


def _write(path: Path, rows: list, header=HEADER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def _row(epoch: int, box: float) -> list:
    return [epoch, 10.0 * epoch, box, 1.0, 1.0, math.nan, math.nan, math.nan, math.nan,
            math.nan, math.nan, math.nan, 0.01, 0.01, 0.01]


def _raw(path: Path, lines: list[str], header=HEADER) -> Path:
    """원문 그대로 쓴다(짧은 행·찢긴 행·BOM 을 만들기 위해)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(header) + "\n" + "".join(l + "\n" for l in lines), encoding="utf-8")
    return path


# ==========================================================================
# 구판 F03 짧은 행 — 오른쪽 패딩 금지
# ==========================================================================

def test_구판_8열_행의_뒤_3값은_LR_이고_metrics_는_NaN_이다(tmp_path):
    """Critical. 오른쪽 패딩하면 LR 이 precision·recall·mAP50 값이 된다."""
    p = _raw(tmp_path / "results.csv", ["99,37833.1,1.82438,1.30773,1.21218,0,0,0,0,0,0,0,"
                                        "0.000109768,0.000109768,0.000109768", SEED1_F03_ROW])
    rep = MergeReport()
    header, rows = read_results(p, rep)
    assert header == HEADER
    last = rows[-1]
    assert last["epoch"] == 100
    assert last["train/box_loss"] == 1.82596 and last["train/dfl_loss"] == 1.21453
    assert last["lr/pg0"] == last["lr/pg1"] == last["lr/pg2"] == SEED1_F03_LR
    for k in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
              "metrics/mAP50-95(B)", "val/box_loss", "val/cls_loss", "val/dfl_loss"):
        assert math.isnan(last[k]), k
    assert rep.remapped == [("results.csv", 100)]      # 되맞췄다는 사실이 보고된다


def test_되맞출_수_없는_열_수는_버리고_보고한다(tmp_path):
    """부분 기록된 마지막 줄(재개는 비정상 종료의 산물이라 이 입력이 온다)."""
    p = _raw(tmp_path / "results.csv", ["9,10.0,1.0,1.0,1.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.01,0.01,0.01",
                                        "10,20.0,1.0"])
    rep = MergeReport()
    _, rows = read_results(p, rep)
    assert [r["epoch"] for r in rows] == [9]
    assert rep.dropped_rows and rep.dropped_rows[0][0] == "results.csv"
    assert "열 수" in rep.dropped_rows[0][2]


def test_찢긴_행이_앞_조각의_정상_행을_덮지_않는다(tmp_path):
    """"짧은 행 패딩" + "뒤 파일 승" 이 결합하면 정상 값이 전 열 NaN 으로 사라진다."""
    old = _write(tmp_path / fragment_name(8, "s", "20260916-000000"), [_row(9, 1.0)])
    cur = _raw(tmp_path / "results.csv", ["9"])
    rep = MergeReport()
    rows = merge_results([old, cur], rep)
    assert [r["epoch"] for r in rows] == [9]
    assert rows[0]["train/box_loss"] == 1.0          # 정상 값이 살아 있다
    assert len(rep.dropped_rows) == 1


def test_긴_행도_버리고_보고한다(tmp_path):
    p = _raw(tmp_path / "results.csv", [",".join(["1"] * 16)])
    rep = MergeReport()
    _, rows = read_results(p, rep)
    assert rows == [] and "헤더" in rep.dropped_rows[0][2]


# ==========================================================================
# 손상 조각·헤더 경계
# ==========================================================================

def test_0바이트_조각은_건너뛰고_나머지를_합친다(tmp_path):
    """append 모드로 파일을 만든 직후 죽으면 헤더조차 없다 — 그 조각 하나가 전체를 죽이면 안 된다."""
    empty = tmp_path / "results.before_resume_1__s__20260916-000000.csv"
    empty.write_bytes(b"")
    good = _write(tmp_path / "results.csv", [_row(3, 1.0)])
    rep = MergeReport()
    rows = merge_results([empty, good], rep)
    assert [r["epoch"] for r in rows] == [3]
    assert rep.skipped_files and rep.skipped_files[0][0] == empty.name


def test_다른_열_구성_조각은_건너뛴다(tmp_path):
    ok = _write(tmp_path / "results.csv", [_row(1, 2.0)])
    other = _write(tmp_path / "other.csv", [[1, 1.0]], header=["epoch", "loss"])
    rep = MergeReport()
    rows = merge_results([ok, other], rep)
    assert [r["epoch"] for r in rows] == [1]
    assert any("열 구성" in why for _, why in rep.skipped_files)


def test_BOM_은_읽고_중복_열은_거부한다(tmp_path):
    bom = tmp_path / "bom.csv"
    bom.write_bytes(b"\xef\xbb\xbf" + (",".join(HEADER) + "\n"
                                       + ",".join(str(v) for v in _row(1, 2.0)) + "\n").encode())
    header, rows = read_results(bom)
    assert header == HEADER and rows[0]["epoch"] == 1
    dup = _write(tmp_path / "dup.csv", [], header=HEADER[:-1] + ["lr/pg1"])
    with pytest.raises(ValueError, match="중복 열"):
        read_results(dup)


def test_헤더가_아니면_거부하고_epoch_오류는_파일_줄을_말한다(tmp_path):
    not_results = _write(tmp_path / "x.csv", [[1]], header=["step"])
    with pytest.raises(ValueError, match="헤더가 아니다"):
        read_results(not_results)
    bad = _raw(tmp_path / "nan.csv", [",".join(["nan"] + ["1"] * 14)])
    with pytest.raises(ValueError, match="nan.csv:2.*epoch"):
        read_results(bad)


def test_빈_행은_건너뛴다(tmp_path):
    p = _raw(tmp_path / "results.csv", [",".join(str(v) for v in _row(1, 2.0)), "", "   ,  "])
    _, rows = read_results(p)
    assert [r["epoch"] for r in rows] == [1]


# ==========================================================================
# 병합 규칙 · 조각 순서 · 실행 신원
# ==========================================================================

def test_병합은_전_epoch_을_빠짐없이_중복없이_내고_뒤_파일이_이긴다(tmp_path):
    old = _write(tmp_path / fragment_name(3, "s", "20260916-000000"),
                 [_row(3, 1.8), _row(2, 1.9), _row(1, 2.0)])          # 역순 — 정렬 보장 검사
    new = _write(tmp_path / "results.csv", [_row(3, 7.7), _row(4, 1.6), _row(5, 1.5)])
    rows = merge_results([old, new])
    assert [r["epoch"] for r in rows] == [1, 2, 3, 4, 5]
    assert rows[2]["train/box_loss"] == 7.7                            # epoch 3: 재개 뒤 파일 채택
    assert all(list(r) == HEADER for r in rows)


def test_조각은_epoch_done_숫자순이고_현재_파일이_마지막이다(tmp_path):
    for ep in (10, 9, 3):
        _write(tmp_path / fragment_name(ep, "main_s1", "20260916-000000"), [_row(ep, 1.0)])
    _write(tmp_path / "results.csv", [_row(11, 1.0)])
    names = [p.name for p in before_resume_files(tmp_path)]
    assert names == [fragment_name(3, "main_s1", "20260916-000000"),
                     fragment_name(9, "main_s1", "20260916-000000"),
                     fragment_name(10, "main_s1", "20260916-000000"), "results.csv"]
    assert [r["epoch"] for r in merge_results(before_resume_files(tmp_path))] == [3, 9, 10, 11]


def test_run_id_로_조각을_거르고_섞인_신원을_보고한다(tmp_path):
    """검토 I2. save_dir 이름(`r000_c0`)은 실행이 달라도 같다 — 신원이 없으면 옛 실행 조각이
    `epoch_done` 이 크다는 이유로 새 실행 행보다 뒤에 놓여 표를 오염시킨다."""
    _write(tmp_path / fragment_name(50, "old_run", "20260901-000000"), [_row(50, 9.9)])
    _write(tmp_path / fragment_name(10, "main_s1", "20260916-000000"), [_row(10, 2.0)])
    _write(tmp_path / "results.csv", [_row(11, 3.0)])

    mine = before_resume_files(tmp_path, run_id="main_s1")
    assert [p.name for p in mine] == [fragment_name(10, "main_s1", "20260916-000000"), "results.csv"]
    assert [r["epoch"] for r in merge_results(mine)] == [10, 11]

    rep = MergeReport()
    everything = before_resume_files(tmp_path, report=rep)
    assert len(everything) == 3
    assert any("신원이" in why for _, why in rep.skipped_files)
    assert [r["epoch"] for r in merge_results(everything)] == [10, 11, 50]   # 오염이 눈에 보인다


def test_규약과_다른_조각_이름은_보고하고_건너뛴다(tmp_path):
    _write(tmp_path / "results.before_resume_7_oldstyle.csv", [_row(7, 1.0)])
    _write(tmp_path / "results.csv", [_row(8, 1.0)])
    rep = MergeReport()
    files = before_resume_files(tmp_path, report=rep)
    assert [p.name for p in files] == ["results.csv"]
    assert any("이름 규약" in why for _, why in rep.skipped_files)


def test_run_id_는_파일_이름에_안전하게_다듬어진다():
    assert sanitize_run_id("main_s1_rs1") == "main-s1-rs1"
    assert sanitize_run_id("") == "norun"
    assert sanitize_run_id("a/b\\c:d") == "a-b-c-d"
    assert "__" not in sanitize_run_id("x__y")          # 구분자와 충돌하지 않는다


# ==========================================================================
# 읽기 전용 + F03 15 열 불변(게이트 조건)
# ==========================================================================

def test_병합_유틸은_파일을_만들거나_고치지_않는다(monkeypatch, tmp_path):
    """소스 문자열 검사가 아니라 **행동**으로 고정한다 — 쓰기 모드 open 자체를 막고,
    디렉터리 스냅샷(이름·크기·mtime)을 전후 비교한다."""
    a = _write(tmp_path / fragment_name(1, "s", "20260916-000000"), [_row(1, 2.0)])
    b = _write(tmp_path / "results.csv", [_row(2, 1.0)])
    snap = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.iterdir()}

    real_open, real_io_open = builtins.open, io.open

    def guard(factory):
        def opener(file, mode="r", *a, **kw):
            if any(c in str(mode) for c in "wax+"):
                raise AssertionError(f"병합 유틸이 쓰기 모드로 열었다: {file!r} mode={mode!r}")
            return factory(file, mode, *a, **kw)
        return opener

    monkeypatch.setattr(builtins, "open", guard(real_open))
    monkeypatch.setattr(io, "open", guard(real_io_open))
    rows = merge_results(before_resume_files(tmp_path, run_id="s"))
    read_results(a)
    monkeypatch.undo()

    assert [r["epoch"] for r in rows] == [1, 2]
    assert {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.iterdir()} == snap
    assert {p.name for p in tmp_path.iterdir()} == {a.name, b.name}


def test_실제_트레이너가_쓴_CSV_는_15열이고_병합_뒤에도_15열이다(tmp_path):
    """F03: `FedDetectionTrainer.save_metrics` 가 마지막 epoch 에도 15 열을 유지한다 —
    병합이 그것을 깨지 않는다. 상류는 `self.epoch + 1` 을 기록한다(1-기반)."""
    from detection.fed_trainer import FedDetectionTrainer

    tr = object.__new__(FedDetectionTrainer)
    tr.csv = tmp_path / "results.csv"
    tr.train_time_start = time.time()
    tr.metrics = dict.fromkeys(HEADER[5:12], 0.0)
    losses = {"train/box_loss": 2.2, "train/cls_loss": 1.7, "train/dfl_loss": 1.4}
    lrs = {"lr/pg0": 0.0001, "lr/pg1": 0.0001, "lr/pg2": 0.0001}
    tr.epoch = 98                                   # → CSV epoch 99
    tr.save_metrics({**losses, **tr.metrics, **lrs})
    tr.epoch = 99                                   # → CSV epoch 100, 검증 미실행 → NaN
    tr.metrics, fitness = tr.validate()
    assert fitness is None
    tr.save_metrics({**losses, **tr.metrics, **lrs})

    old = _write(tmp_path / fragment_name(96, "t", "20260916-000000"),
                 [_row(98, 9.9), _row(97, 2.5)])    # 역순 조각 — 정렬 보장
    rows = merge_results([old, tr.csv])
    assert [r["epoch"] for r in rows] == [97, 98, 99, 100]
    assert all(len(r) == 15 and list(r) == HEADER for r in rows)
    assert rows[1]["train/box_loss"] == 9.9         # 조각의 epoch 98
    assert rows[2]["train/box_loss"] == 2.2         # 트레이너가 쓴 epoch 99
    assert math.isnan(rows[3]["val/box_loss"]) and rows[3]["lr/pg0"] == 0.0001
    with tr.csv.open(newline="", encoding="utf-8") as fh:
        assert [len(r) for r in csv.reader(fh)] == [15, 15, 15]


def test_행_수_증거_소비자가_병합을_참조한다():
    """검토 I3. "조기 종료 부재의 증거는 results.csv 행 수" 를 말하는 곳이 조각을 합치지
    않으면, 재개한 런은 여전히 미달로 보인다."""
    root = Path(results_merge.__file__).resolve().parent.parent
    for rel in ("scripts/gate_reduced_pilot.py", "detection/budget_audit.py"):
        assert "results_merge" in (root / rel).read_text(encoding="utf-8"), rel


# ==========================================================================
# train_round — 재개면 트레이너 생성 전에 rename, 아니면 무접촉
# ==========================================================================

def test_preserve_는_이름만_바꾸고_덮어쓰지_않는다(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "strftime", lambda *a, **k: "20260916-010203")
    _write(tmp_path / "results.csv", [_row(1, 2.0)])
    first = _preserve_results_csv(tmp_path, epoch_done=4, run_id="main_s1")
    assert first == fragment_name(4, "main_s1", "20260916-010203")
    assert not (tmp_path / "results.csv").exists()
    _write(tmp_path / "results.csv", [_row(5, 1.0)])
    second = _preserve_results_csv(tmp_path, epoch_done=4, run_id="main_s1")
    assert second == fragment_name(4, "main_s1", "20260916-010203-1")
    assert (tmp_path / first).read_text(encoding="utf-8") != (tmp_path / second).read_text(encoding="utf-8")
    assert _preserve_results_csv(tmp_path, epoch_done=4, run_id="main_s1") is None


def _checkpoint(rdir: Path, data: Path, *, epoch_done: int, consumed: int) -> None:
    """`train_round` 가 읽을 수 있는 실제 체크포인트. consumed == local_epochs 면 완주다."""
    from detection.resume import ResumeCheckpointer, ResumeIdentity
    from detection.round_runner import _LRTrace, derive_seed
    from tests.test_detection_resume import _Counter, _FakeTrainer, _Net

    ident = ResumeIdentity(run_id="t", round_idx=0, client_idx=0, seed=derive_seed(0, 0, 0),
                           total_epochs=2, local_epochs=2, model="m.pt", data=str(data.resolve()),
                           loader_reseed_per_epoch=False, loader_seed=None, profile="pilot")
    tr = _FakeTrainer(_Net(), epoch=epoch_done, start_epoch=epoch_done)
    tr.n_optimizer_updates = 3
    tr._resume_lr_trace = _LRTrace([(e, 0.01) for e in range(epoch_done + 1)])
    ResumeCheckpointer(rdir, identity=ident, step_counter=_Counter(5),
                       resumed_epochs=consumed - 1).save(tr)


@pytest.fixture
def upstream_like_trainer(monkeypatch):
    """상류처럼 생성 시 `save_dir/results.csv` 를 지우는 트레이너. 무엇을 봤는지 기록한다."""
    import detection.fed_trainer as ft

    from tests._round_fakes import RoundTrainer

    seen: dict = {}

    class Tr(RoundTrainer):
        def __init__(self, overrides=None, **kw):
            super().__init__(overrides, **kw)
            save_dir = Path(overrides["project"]) / overrides["name"]
            save_dir.mkdir(parents=True, exist_ok=True)
            seen["csv_at_ctor"] = (save_dir / "results.csv").exists()
            seen["fragments_at_ctor"] = sorted(p.name for p in
                                               save_dir.glob("results.before_resume_*.csv"))
            if (save_dir / "results.csv").exists():
                (save_dir / "results.csv").unlink()          # 상류 199-201 행의 동작
            with (save_dir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerows([HEADER, _row(2, 0.5)])

    monkeypatch.setattr(ft, "FedDetectionTrainer", Tr)
    return seen


def _round(tmp_path, **kw):
    from detection.round_runner import train_round

    return train_round(data_yaml=str(tmp_path / "d.yaml"), model="m.pt", total_epochs=2,
                       local_epochs=2, project=tmp_path / "runs", profile="pilot", run_id="t", **kw)


def test_재개_진입은_트레이너_생성_전에_CSV_를_옮겨_상류_unlink_에_걸리지_않는다(upstream_like_trainer, tmp_path):
    """(a)"""
    rdir = tmp_path / "_resume"
    _checkpoint(rdir, tmp_path / "d.yaml", epoch_done=0, consumed=1)
    save_dir = tmp_path / "runs" / "r000_c0"
    _write(save_dir / "results.csv", [_row(1, 2.0)])
    old_bytes = (save_dir / "results.csv").read_bytes()

    res = _round(tmp_path, resume_dir=rdir)
    assert res.resumed_from_epoch == 1
    assert upstream_like_trainer["csv_at_ctor"] is False              # 상류가 지울 파일이 없었다
    assert upstream_like_trainer["fragments_at_ctor"] == [res.results_csv_preserved]
    assert res.results_csv_preserved.startswith("results.before_resume_0__t__")
    assert (save_dir / res.results_csv_preserved).read_bytes() == old_bytes   # 바이트 그대로
    assert [r["epoch"] for r in merge_results(before_resume_files(save_dir, run_id="t"))] == [1, 2]


def test_재개가_아니면_CSV_를_건드리지_않는다(upstream_like_trainer, tmp_path):
    """(c)"""
    save_dir = tmp_path / "runs" / "r000_c0"
    _write(save_dir / "results.csv", [_row(1, 2.0)])
    res = _round(tmp_path)
    assert res.results_csv_preserved is None
    assert upstream_like_trainer["csv_at_ctor"] is True               # 그대로 상류에 넘어갔다
    assert list(save_dir.glob("results.before_resume_*.csv")) == []


def test_재개지만_이전_CSV_가_없으면_None_이다(upstream_like_trainer, tmp_path):
    rdir = tmp_path / "_resume"
    _checkpoint(rdir, tmp_path / "d.yaml", epoch_done=0, consumed=1)
    res = _round(tmp_path, resume_dir=rdir)
    assert res.resumed_from_epoch == 1 and res.results_csv_preserved is None


def test_거부될_재개는_CSV_를_건드리지_않는다(upstream_like_trainer, tmp_path):
    """보존은 `validate_detection_resume` **뒤**에 있어야 한다 — 완주 체크포인트로 들어온
    실행은 학습도 보존도 하지 않고 죽는다(동결 산출물의 마지막 방어선)."""
    rdir = tmp_path / "_resume"
    _checkpoint(rdir, tmp_path / "d.yaml", epoch_done=1, consumed=2)   # 2/2 = 완주
    save_dir = tmp_path / "runs" / "r000_c0"
    _write(save_dir / "results.csv", [_row(1, 2.0)])
    before = {p.name: p.read_bytes() for p in save_dir.iterdir()}
    with pytest.raises(ValueError, match="예산이 이미 완료"):
        _round(tmp_path, resume_dir=rdir)
    assert {p.name: p.read_bytes() for p in save_dir.iterdir()} == before
    assert "csv_at_ctor" not in upstream_like_trainer                  # 트레이너도 만들지 않았다


def test_project_를_주지_않으면_보존하지_않는다(upstream_like_trainer, tmp_path, monkeypatch):
    """`overrides` 에 project 가 없으면 save_dir 을 알 수 없다 — 추측하지 않는다."""
    from detection.round_runner import train_round

    calls = []
    monkeypatch.setattr("detection.round_runner._preserve_results_csv",
                        lambda *a, **k: calls.append(a) or "x")
    rdir = tmp_path / "_resume"
    _checkpoint(rdir, tmp_path / "d.yaml", epoch_done=0, consumed=1)

    class NoProjectTrainer:
        def __init__(self, overrides=None, **kw):
            raise RuntimeError("stop-here")

    import detection.fed_trainer as ft
    monkeypatch.setattr(ft, "FedDetectionTrainer", NoProjectTrainer)
    with pytest.raises(RuntimeError, match="stop-here"):
        train_round(data_yaml=str(tmp_path / "d.yaml"), model="m.pt", total_epochs=2,
                    local_epochs=2, profile="pilot", run_id="t", resume_dir=rdir)
    assert calls == []
