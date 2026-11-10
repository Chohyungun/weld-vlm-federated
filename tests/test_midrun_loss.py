"""학습 중 손실 기록(`vlm.midrun_loss`) — 값 · 기울기 · 학습 궤적을 바꾸지 않고, 창 손실을 손으로 센 값과 같게 적는다.

엘리스 멈춤 지점 (나) 의 입력(설계 결정 기록 10-31 c0c418f2). 환경 변수가 없으면 아무 일도 없어야 한다(지금 동작 그대로).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.uni_fakes import CountingLoader, make_rows
from vlm import loss_norm, midrun_loss, pilot_vlm


def _logits_targets(seed: int, n: int = 6, v: int = 5, n_ignored: int = 2):
    import torch

    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, v, generator=g, requires_grad=True)
    targets = torch.randint(0, v, (n,), generator=g)
    targets[:n_ignored] = loss_norm.IGNORE_INDEX
    return logits, targets


def _lines(d):
    files = sorted(d.glob("midrun_loss_*.jsonl"))
    return files, [json.loads(x) for f in files for x in f.read_text(encoding="utf-8").splitlines()]


def test_관찰자는_값과_기울기를_바꾸지_않는다():
    got = []
    logits, targets = _logits_targets(1)
    ce0, n0 = loss_norm.supervised_ce_sum(logits, targets)
    ce0.backward()
    g0 = logits.grad.clone()

    logits.grad = None
    loss_norm._OBSERVERS.append(lambda ce, n: got.append((float(ce), n, ce.requires_grad)))
    try:
        ce1, n1 = loss_norm.supervised_ce_sum(logits, targets)
        ce1.backward()
    finally:
        loss_norm._OBSERVERS.clear()
    assert (float(ce1.detach()), n1) == (float(ce0.detach()), n0) and n0 == 4
    assert np.array_equal(logits.grad.numpy(), g0.numpy())
    assert got == [(float(ce0), n0, False)]                      # 떼어 낸 값 — 그래프에 매달리지 않는다


def test_감독_토큰이_없는_호출은_관찰하지_않는다():
    got = []
    logits, targets = _logits_targets(2, n=3, n_ignored=3)
    loss_norm._OBSERVERS.append(lambda ce, n: got.append(n))
    try:
        _, n = loss_norm.supervised_ce_sum(logits, targets)
    finally:
        loss_norm._OBSERVERS.clear()
    assert n == 0 and got == []


def test_환경_변수가_없으면_아무_일도_없다(tmp_path, monkeypatch):
    monkeypatch.delenv(midrun_loss.ENV_DIR, raising=False)
    monkeypatch.chdir(tmp_path)
    with midrun_loss.observe("central_s1") as rec:
        assert rec is None and loss_norm._OBSERVERS == []
        loss_norm.supervised_ce_sum(*_logits_targets(3))
    assert list(tmp_path.rglob("*")) == []


def test_창_손실은_손으로_센_토큰_가중_평균과_같다(tmp_path, monkeypatch):
    monkeypatch.setenv(midrun_loss.ENV_DIR, str(tmp_path / "obs"))
    monkeypatch.setenv(midrun_loss.ENV_EVERY, "3")
    calls = []
    for seed in range(7):                                         # 호출 7 → 창 3 + 3 + 끝 1
        lt = _logits_targets(10 + seed, n_ignored=seed % 3)
        ce, n = loss_norm.supervised_ce_sum(*lt)                  # 관찰 밖에서 기대값을 먼저 센다
        calls.append((float(ce), n))
    with midrun_loss.observe("central_s1"):
        for seed in range(7):
            loss_norm.supervised_ce_sum(*_logits_targets(10 + seed, n_ignored=seed % 3))
    assert loss_norm._OBSERVERS == []
    files, rows = _lines(tmp_path / "obs")
    assert len(files) == 1 and files[0].name.startswith("midrun_loss_central_s1_")
    assert [r["window_calls"] for r in rows] == [3, 3, 1] and [r["final"] for r in rows] == [False, False, True]
    assert [r["calls"] for r in rows] == [3, 6, 7]
    for r, win in zip(rows, (calls[0:3], calls[3:6], calls[6:7])):
        tok = sum(n for _, n in win)
        assert r["window_tokens"] == tok
        assert r["window_ce"] == pytest.approx(sum(c for c, _ in win) / tok, rel=1e-12)
    assert rows[-1]["cum_tokens"] == sum(n for _, n in calls)
    assert rows[0]["window_ce"] != rows[1]["window_ce"]            # 창마다 다른 입력 — 같은 값이 우연히 맞지 않는다


def test_블록이_예외로_끝나도_관찰자를_떼고_남은_창을_쓴다(tmp_path, monkeypatch):
    monkeypatch.setenv(midrun_loss.ENV_DIR, str(tmp_path))
    monkeypatch.setenv(midrun_loss.ENV_EVERY, "100")
    with pytest.raises(RuntimeError), midrun_loss.observe("uni_fed_r000_c1_C2"):
        loss_norm.supervised_ce_sum(*_logits_targets(4))
        raise RuntimeError("학습 실패")
    assert loss_norm._OBSERVERS == []
    _, rows = _lines(tmp_path)
    assert len(rows) == 1 and rows[0]["final"] is True and rows[0]["window_calls"] == 1


def test_잘못된_묶음_수는_거부한다(tmp_path, monkeypatch):
    monkeypatch.setenv(midrun_loss.ENV_DIR, str(tmp_path))
    monkeypatch.setenv(midrun_loss.ENV_EVERY, "0")
    with pytest.raises(ValueError), midrun_loss.observe("x"):
        pass
    assert loss_norm._OBSERVERS == []


def test_관찰을_켜도_학습_궤적이_같고_기록이_원장과_맞는다(tmp_path, monkeypatch):
    """대역 모델 학습 — 관찰 켬/끔의 끝 어댑터가 원소까지 같고, 기록한 토큰 · 손실이 학습기의 지표와 맞는다."""
    import torch

    from vlm.init_adapter import build_initial_adapter

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pilot_vlm, "GRAD_ACCUM", 2)
    rows = make_rows(tmp_path / "img", "C1", 6)
    init_path = tmp_path / "init" / "initial.npz"
    build_initial_adapter(model_id="tiny", seed=7, cache_path=init_path, revision="r1", purpose="rehearsal",
                          model_loader=CountingLoader(), standin_allowed=True)
    init = np.load(init_path)

    def run(ces):
        return pilot_vlm.train_rounds(
            rows=rows, epochs=1, round_idx=0, client_idx=0, base_seed=7,
            adapter_in=[init[k] for k in init.files], adapter_keys=list(init.files), model_id="tiny", run_id="x",
            init_seed=7, num_rounds=1, model_revision="r1", model_loader=CountingLoader(),
            purpose="rehearsal", standin_allowed=True, tag="uni_central",
            chat_template_kwargs={"enable_thinking": False}, processor_kwargs={},
            log_cb=lambda ep, mean_ce, steps, wall: ces.append(float(mean_ce)))

    monkeypatch.delenv(midrun_loss.ENV_DIR, raising=False)
    ce_plain, ce_seen = [], []
    plain, _, m_plain, _ = run(ce_plain)
    monkeypatch.setenv(midrun_loss.ENV_DIR, str(tmp_path / "obs"))
    monkeypatch.setenv(midrun_loss.ENV_EVERY, "4")
    with midrun_loss.observe("uni_central_s1"):
        seen, _, m_seen, _ = run(ce_seen)
    assert len(plain) == len(seen) and all(np.array_equal(a, b) for a, b in zip(plain, seen))
    assert ce_plain == ce_seen and len(ce_seen) == 1
    for key in ("optimizer_steps", "supervised_tokens", "param_l2", "lr"):
        assert m_plain[key] == m_seen[key], key
    _, recs = _lines(tmp_path / "obs")
    assert [r["window_calls"] for r in recs] == [4, 2] and recs[-1]["calls"] == len(rows)
    assert recs[-1]["cum_tokens"] == m_seen["supervised_tokens"]
    pooled = sum(r["window_ce"] * r["window_tokens"] for r in recs) / recs[-1]["cum_tokens"]
    assert pooled == pytest.approx(ce_seen[0], rel=1e-5)
