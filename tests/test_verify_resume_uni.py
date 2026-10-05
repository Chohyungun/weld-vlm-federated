"""통합형 재개 등가 시험 실행기(`scripts/verify_resume_uni.py`, 12번 §4-2) — 합성 대역으로 단계 함수를 돈다.

CLI 는 자식 프로세스를 `os._exit(9)` 로 죽였다 이어 가지만, CLI 에는 구현을 고르는 인자가 없어 합성 시험은 단계 함수를
**프로세스 안에서** 부른다 — 죽음 대신 예외를 올린다. 그래서 이 시험은 판정 산술과 단계의 순서 · 폴더 규칙을 본다.
자식 프로세스의 사망 · 재기동과 실제 모델의 비트 동일은 GPU 창에서 실행기로 본다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import verify_resume_uni as V  # noqa: E402
from tests.uni_fakes import CountingLoader, make_rows  # noqa: E402


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    import torch

    was = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    assert was or not torch.cuda.is_initialized(), "이 시험이 CUDA 를 초기화했다"


class _Died(RuntimeError):
    pass


def _die():
    raise _Died("os._exit(9) 대신")


def _all_stages(root: Path, rows, loader, *, epochs=3, die_after=1):
    kw = dict(rows=rows, epochs=epochs, die_after=die_after, model_loader=loader, purpose="rehearsal", die=_die)
    V.run_stage("baseline", root, **kw)
    V.run_stage("baseline2", root, **kw)
    with pytest.raises(_Died):
        V.run_stage("crash", root, **kw)
    V.run_stage("resume", root, **kw)


def test_죽였다_이어_간_학습이_중단_없는_학습과_비트_동일하다(tmp_path):
    rows = make_rows(tmp_path / "img", "C1", 40)          # 누적 32 → epoch 당 2 스텝(마지막 부분 창 포함)
    loader = CountingLoader()
    _all_stages(tmp_path / "root", rows, loader)
    rep = V.judge(tmp_path / "root", epochs=3, die_after=1)
    assert rep["passed"], rep
    pair = rep["resume_pair"]
    assert pair["adapter_bit_identical"] and pair["adapter_max_abs_diff"] == 0.0
    assert pair["optimizer_steps"] == [6, 6] and pair["per_epoch_equal"]
    assert rep["resumed_from_epoch"] == 2
    assert rep["nondeterminism_floor"]["adapter_bit_identical"]
    # crash 가 남긴 epoch 행과 resume 이 쓴 행이 한 파일에 이어진다 — 판정은 epoch 마다 마지막 기록을 쓴다
    lines = (tmp_path / "root" / "resume_run" / "epochs.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["stage"] for x in lines] == ["crash", "crash", "resume"]


def test_어댑터가_한_값이라도_다르면_통과하지_않고_크기를_적는다(tmp_path):
    rows = make_rows(tmp_path / "img", "C1", 8)
    _all_stages(tmp_path / "root", rows, CountingLoader())
    fin = tmp_path / "root" / "resume_run" / "final.npz"
    with np.load(fin) as z:
        arrs = {k: z[k].copy() for k in z.files}
    k0 = next(iter(arrs))
    arrs[k0].flat[0] += 1e-3
    np.savez(fin, **arrs)
    rep = V.judge(tmp_path / "root", epochs=3, die_after=1)
    assert not rep["passed"]
    assert rep["resume_pair"]["adapter_bit_identical"] is False
    assert rep["resume_pair"]["adapter_max_abs_diff"] == pytest.approx(1e-3, rel=1e-3)


def test_단계의_순서와_폴더를_지킨다(tmp_path):
    rows = make_rows(tmp_path / "img", "C1", 4)
    root = tmp_path / "root"
    kw = dict(rows=rows, epochs=3, die_after=1, model_loader=CountingLoader(), purpose="rehearsal", die=_die)
    with pytest.raises(FileNotFoundError):
        V.run_stage("resume", root, **kw)                 # crash 전에 이어 갈 수 없다
    V.run_stage("baseline", root, **kw)
    with pytest.raises(FileExistsError):
        V.run_stage("baseline", root, **kw)               # 같은 루트에 다시 쓰지 않는다
    with pytest.raises(ValueError):
        V.run_stage("crash", root, **{**kw, "die_after": 2})   # 이어 갈 epoch 가 남지 않는다


def test_본실험_목적은_대역을_받지_않는다(tmp_path):
    loader = CountingLoader()
    from vlm.seams import SeamRejected

    with pytest.raises(SeamRejected):
        V.run_stage("baseline", tmp_path / "root", rows=make_rows(tmp_path / "img", "C1", 2), epochs=3,
                    die_after=1, model_loader=loader)
    assert loader.n == 0


def test_CLI_는_비어_있지_않은_루트에서_시작하지_않는다(tmp_path, capsys):
    root = tmp_path / "root"
    root.mkdir()
    (root / "x").write_text("x", encoding="utf-8")
    assert V.main([ "--root", str(root)]) == 2
    assert "비어 있지 않다" in capsys.readouterr().err
