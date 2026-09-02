"""LoRA 초기화 회귀 시험 — 지정 함정 구간 #3 / 74번 감사 C-1.

⑦ 통합·연합 r0 에서 세 클라이언트가 각자 난수 A 로 출발해 r0 로컬 학습 144스텝(전체
432의 33%)이 상쇄로 폐기됐다. 이 파일은 **그 사고가 두 번 나지 않게** 하는 것이 목적이다.

세 겹으로 건다.

1. `fl.seeding.seeded` 가 실제로 난수를 고정하고 **원래 상태로 되돌리는가** — 되돌리지
   않으면 초기화 한 번이 학습 전체의 난수 흐름을 호출 순서에 묶는다.
2. 같은 시드로 만든 LoRA A 가 **비트 단위로 같은가**, 그리고 시드가 없으면 **실제로
   달라지는가**. 뒤쪽이 없으면 시험이 이빨을 잃는다 — 늘 통과하는 시험이 이번 사고의
   다른 얼굴이다(P9).
3. 집계 산술이 두 경우를 구분하는가. 공유 초기값이면 norm 이 보존되고, 독립 난수면
   `sqrt(sum w^2)` 배로 줄어든다. 실측 수치를 픽스처로 박아 둔다.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fl.seeding import seed_all, seeded, shared_init_seed  # noqa: E402
from vlm.init_adapter import adapter_proof, assert_same_start  # noqa: E402

peft = pytest.importorskip("peft")


# --------------------------------------------------------------------------
# 1. seeded() 는 고정하고 되돌린다
# --------------------------------------------------------------------------

def test_seeded_는_블록_안에서만_난수를_고정한다():
    with seeded(1234):
        a = torch.randn(4)
    with seeded(1234):
        b = torch.randn(4)
    assert torch.equal(a, b), "같은 시드인데 다른 값이 나왔다"


def test_seeded_는_빠져나올_때_전역_상태를_되돌린다():
    """되돌리지 않으면 초기화 한 번이 그 뒤 셔플 전체를 호출 순서에 묶는다."""
    seed_all(999)
    before_py = random.random()
    before_torch = torch.randn(3)

    seed_all(999)
    with seeded(1234):
        torch.randn(100)
        random.random()
    after_py = random.random()
    after_torch = torch.randn(3)

    assert before_py == after_py
    assert torch.equal(before_torch, after_torch)


def test_공유_시드는_라운드_클라이언트_인자를_받지_않는다():
    """`derive_seed` 와 헷갈리면 반드시 한쪽이 틀린다. 항등임을 못박아 둔다."""
    import inspect

    from detection.round_runner import derive_seed

    assert list(inspect.signature(shared_init_seed).parameters) == ["base_seed"]
    assert shared_init_seed(20260828) == 20260828
    # 파생 시드는 반대로 라운드·클라이언트마다 **달라야** 한다
    assert derive_seed(20260828, 0, 0) != derive_seed(20260828, 1, 0)
    assert derive_seed(20260828, 0, 0) != derive_seed(20260828, 0, 1)


# --------------------------------------------------------------------------
# 2. LoRA A 가 클라이언트 간 비트 단위로 같은가
# --------------------------------------------------------------------------

class _Tiny(nn.Module):
    """LoRA 접미사만 흉내 내는 최소 모델. 실물 VLM 을 띄우지 않고 초기화 규약만 본다."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(32, 32, bias=False)
        self.v_proj = nn.Linear(32, 32, bias=False)

    def forward(self, x):  # pragma: no cover - 학습하지 않는다
        return self.v_proj(self.q_proj(x))


def _lora_state(*, seed: int | None) -> dict[str, torch.Tensor]:
    """`vlm.pilot_vlm._load_model` 과 **같은 방식**으로 어댑터를 만든다."""
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "v_proj"])
    if seed is None:
        model = get_peft_model(_Tiny(), cfg)
    else:
        with seeded(shared_init_seed(seed)):
            model = get_peft_model(_Tiny(), cfg)
    return {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}


def test_같은_시드면_세_클라이언트의_lora_A_가_비트_단위로_같다():
    """이것이 ⑦ r0 사고의 직접 회귀 시험이다."""
    states = [_lora_state(seed=20260828) for _ in range(3)]
    a_keys = [k for k in states[0] if "lora_A" in k]
    assert a_keys, "lora_A 키가 없다 — 시험이 아무것도 보고 있지 않다"
    for k in a_keys:
        for other in states[1:]:
            assert torch.equal(states[0][k], other[k]), f"{k} 가 클라이언트마다 다르다"


def test_시드가_없으면_실제로_달라진다__시험의_이빨():
    """늘 통과하는 시험을 통과 근거로 두지 않는다(74번 P9 와 같은 실수 방지)."""
    seed_all(0)
    states = [_lora_state(seed=None) for _ in range(3)]
    a_keys = [k for k in states[0] if "lora_A" in k]
    assert any(
        not torch.equal(states[0][k], other[k])
        for k in a_keys
        for other in states[1:]
    ), "시드 없이도 같다면 이 시험은 아무것도 검증하지 못한다"


def test_lora_B_는_영행렬로_출발한다():
    """A 만 고정하면 되는 근거. B 가 0 이 아니면 전제가 바뀐다."""
    sd = _lora_state(seed=20260828)
    for k, v in sd.items():
        if "lora_B" in k:
            assert torch.count_nonzero(v) == 0, f"{k} 가 0 이 아니다"


# --------------------------------------------------------------------------
# 3. 런타임 가드
# --------------------------------------------------------------------------

def test_assert_same_start_는_같은_증빙을_통과시킨다():
    sd = _lora_state(seed=20260828)
    keys = list(sd)
    arrays = [v.numpy() for v in sd.values()]
    proof = adapter_proof(arrays, keys)
    assert_same_start({0: proof, 1: dict(proof), 2: dict(proof)})


def test_assert_same_start_는_갈라진_출발을_잡는다():
    seed_all(0)
    proofs = {}
    for i in range(3):
        sd = _lora_state(seed=None)
        proofs[i] = adapter_proof([v.numpy() for v in sd.values()], list(sd))
    with pytest.raises(RuntimeError, match="함정 #3"):
        assert_same_start(proofs)


# --------------------------------------------------------------------------
# 4. 집계 산술 — 사고의 지문
# --------------------------------------------------------------------------

def test_공유_초기값이면_집계가_norm_을_보존한다():
    from fl.aggregate import weighted_fedavg

    rng = np.random.default_rng(7)
    shared = [rng.normal(size=(16, 16)).astype(np.float32)]
    keys = ["a"]
    ref = {"a": torch.zeros(16, 16)}
    n = [1275, 656, 342]
    agg = weighted_fedavg([[s.copy() for s in shared] for _ in n], n, keys, ref)
    src = float(np.linalg.norm(shared[0].astype(np.float64)))
    assert agg.global_norm == pytest.approx(src, rel=1e-6)


def test_독립_난수면_집계가_sqrt_가중치제곱합_배로_준다():
    """⑦ r0 의 지문. 31.68 × 0.64852 = 20.55 이고 실측 global_l2 는 20.5755 였다."""
    from fl.aggregate import weighted_fedavg

    rng = np.random.default_rng(11)
    keys = ["a"]
    ref = {"a": torch.zeros(256, 256)}
    n = [1275, 656, 342]
    payloads = [[rng.normal(size=(256, 256)).astype(np.float32)] for _ in n]
    agg = weighted_fedavg(payloads, n, keys, ref)

    w = np.array(n, dtype=np.float64) / sum(n)
    shrink = float(np.sqrt((w ** 2).sum()))
    assert shrink == pytest.approx(0.64852, abs=1e-4)   # 감사 보고서와 같은 값
    mean_src = float(np.mean([np.linalg.norm(p[0].astype(np.float64)) for p in payloads]))
    assert agg.global_norm == pytest.approx(mean_src * shrink, rel=0.02)


def test_파일럿_실측이_독립난수_가설과_맞는다():
    """61번 §1 의 '함정 #3 3라운드 정상'이 왜 사실이 아닌지를 수치로 고정해 둔다."""
    n = np.array([1275, 656, 342], dtype=np.float64)
    w = n / n.sum()
    shrink = float(np.sqrt((w ** 2).sum()))
    client_l2 = np.mean([31.7405, 31.6717, 31.6238])
    observed_global = 20.5755
    assert client_l2 * shrink == pytest.approx(observed_global, rel=0.005)
    # 공유 초기값 가설이었다면 클라이언트 norm 근처여야 한다 — 실측은 거기서 35% 낮다
    assert observed_global < 0.7 * client_l2
