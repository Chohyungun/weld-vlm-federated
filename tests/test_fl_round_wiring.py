"""연합 배선 스모크 + 무이빨 가드의 이빨 시험 — 80번 체크리스트 14·15·16항.

## 무엇을 사는가

80번의 한 줄 답이 "검사를 만들어 두고 부르지 않았다"였다. 그 형태의 사고는 **경로를
한 번도 안 돌려 봐서** 난다. `flwr run` 진입점은 라운드 1 에서 죽는 상태로 커밋돼
있었고(F1), 아무도 그것을 몰랐다. 시험이 없어서가 아니라 **경로를 돌리는 시험이**
없어서다.

그래서 이 파일은 두 가지를 한다.

1. **완주 스모크** — 더미 2텐서·3클라이언트·R=2·E=1 로 두 진입점(`pilot_sim` ·
   `server_app`)을 실제 Flower 런타임 위에서 끝까지 돌린다. 통합형 칸(⑦)도 같은 경로로
   돌려 인프로세스 루프 우회가 닫혔음을 확인한다(15항).
2. **이빨 시험** — 가드마다 "이 입력이면 반드시 실패한다"를 하나씩 붙인다. G2-9 가
   의무화한 것이고, 픽스처는 80번이 실측으로 통과시켰던 것들이다(fp16 페이로드,
   NaN 한 원소, 가중치 합 0.99, keys-digest 부재).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("flwr")

from detection import serialize  # noqa: E402
from fl.aggregate import weighted_fedavg  # noqa: E402
from fl.round_wiring import SERVER_ROUND_KEY  # noqa: E402

# 더미 교환 단위 — 2텐서. 실물 모델을 띄우지 않고 배선만 본다.
KEYS = ["layer.w", "layer.b"]
REF = {"layer.w": torch.zeros(4, 3), "layer.b": torch.zeros(3)}


def _dummy_arrays(scale: float = 1.0) -> list[np.ndarray]:
    return [
        (np.arange(12, dtype=np.float32).reshape(4, 3) * scale),
        (np.arange(3, dtype=np.float32) * scale),
    ]


# ==========================================================================
# 16항 — 이빨 시험. 각 가드마다 반드시 실패하는 입력이 하나씩 있다.
# ==========================================================================

def test_fp16_페이로드가_실패한다():
    """G2-1. 이전 판은 '부동소수 계열'만 봐서 통과했다 — 최대 편차 5.44e-04 였다."""
    bad = [a.astype(np.float16) for a in _dummy_arrays()]
    with pytest.raises(serialize.SerializeError, match="float32"):
        serialize.assert_compatible(bad, KEYS, REF)


def test_fp32_페이로드는_통과한다():
    serialize.assert_compatible(_dummy_arrays(), KEYS, REF)


def test_NaN_한_원소가_실패한다():
    """G2-2 ①. 이전에는 통과해 `global_l2 = nan` 만 남고 출처를 잃었다."""
    bad = _dummy_arrays()
    bad[0][1, 1] = np.nan
    with pytest.raises(ValueError, match="유한하지 않은"):
        weighted_fedavg([bad, _dummy_arrays()], [1, 1], KEYS, REF)


def test_inf_한_원소도_실패한다():
    bad = _dummy_arrays()
    bad[1][0] = np.inf
    with pytest.raises(ValueError, match="유한하지 않은"):
        weighted_fedavg([bad, _dummy_arrays()], [1, 1], KEYS, REF)


def test_가중치_합이_1_이_아니면_실패한다(monkeypatch):
    """G2-2 ②. 실패 클라이언트를 빼고 재정규화하는 경로를 산술로 잡는다."""
    import fl.aggregate as agg_mod

    real = np.array

    def fake_array(obj, *a, **k):
        arr = real(obj, *a, **k)
        # 가중치 배열만 0.99 로 흔든다(합 검사가 실제로 발화하는지 본다).
        if arr.dtype == np.float64 and arr.ndim == 1 and arr.size == 2:
            return arr * 0.99
        return arr

    monkeypatch.setattr(agg_mod.np, "array", fake_array)
    with pytest.raises(ValueError, match="가중치 합"):
        weighted_fedavg([_dummy_arrays(), _dummy_arrays()], [1, 1], KEYS, REF)


def test_출력_norm_이_클라이언트_최대를_넘으면_실패한다(monkeypatch):
    """G2-2 ③. 볼록 가중 평균에서는 일어날 수 없다 — 일어나면 평균이 아닌 산술이 돈 것."""
    import fl.aggregate as agg_mod

    real_norm = agg_mod.serialize.params_l2_norm
    calls = {"n": 0}

    def fake_norm(arrays):
        calls["n"] += 1
        v = real_norm(arrays)
        return v * 10.0 if calls["n"] > 2 else v      # 마지막(글로벌) 호출만 부풀린다

    monkeypatch.setattr(agg_mod.serialize, "params_l2_norm", fake_norm)
    with pytest.raises(ValueError, match="넘는다"):
        weighted_fedavg([_dummy_arrays(), _dummy_arrays()], [1, 1], KEYS, REF)


def test_tensor_digest_가_A_와_B_를_함께_뽑는다():
    """G2-4. LoRA 는 A·B 가 교대로 배열되는데 step 이 짝수라 전부 A 만 뽑혔다."""
    # A 는 크고 B 는 0 — 실제 LoRA 초기 상태의 모양이다.
    arrays = []
    for i in range(20):
        arrays.append(np.full((4, 4), 1.0 if i % 2 == 0 else 0.0, dtype=np.float32))
    digest = serialize.tensor_digest(arrays)
    assert digest, "빈 다이제스트면 아무것도 대조하지 못한다"
    # 마지막 원소는 전 텐서 L2 — 표집이 무엇을 놓치든 여기서 잡힌다.
    assert digest[-1] == pytest.approx(serialize.params_l2_norm(arrays))


def test_B_만_바뀌면_digest_가_움직인다__구판이_못_잡던_고장():
    """`lora_B` 만 누락되는 부분 주입이 이전 판에서는 원리적으로 안 잡혔다."""
    base = [np.full((4, 4), 1.0 if i % 2 == 0 else 0.0, dtype=np.float32) for i in range(20)]
    only_b_changed = [a.copy() for a in base]
    for i in range(1, 20, 2):                     # 홀수 = B 계열
        only_b_changed[i] += 0.5
    assert serialize.tensor_digest(base) != serialize.tensor_digest(only_b_changed)


def test_assert_same_start_가_l2_만_다른_경우도_잡는다():
    """G2-4 / F14 — 계산해 두고 비교하지 않던 값."""
    from vlm.init_adapter import assert_same_start

    ref = {"keys_digest": "abc", "tensor_digest": [1.0, 2.0], "l2": 3.0, "n_tensors": 2}
    same = dict(ref)
    assert_same_start({0: ref, 1: same})
    drifted = dict(ref, l2=3.5)
    with pytest.raises(RuntimeError, match="함정 #3"):
        assert_same_start({0: ref, 1: drifted})


def test_주입이_서버_페이로드와_다르면_잡는다():
    """G2-5 — 세 클라이언트 모두 no-op 이면 클라이언트끼리 비교로는 통과한다."""
    from vlm.init_adapter import adapter_proof, assert_injected_matches

    sent = _dummy_arrays()
    good = adapter_proof(sent, KEYS)
    assert_injected_matches(sent, KEYS, good, who="c0")

    noop = adapter_proof(_dummy_arrays(scale=2.0), KEYS)   # 주입이 안 먹은 상태
    with pytest.raises(RuntimeError, match="주입이 서버 페이로드와 다르다"):
        assert_injected_matches(sent, KEYS, noop, who="c0")


def test_keys_digest_부재가_실패한다():
    """G2-3 — 기본값을 기대값으로 채우면 필드를 안 실은 클라이언트가 무조건 통과한다."""
    src = Path("fl/strategy.py").read_text(encoding="utf-8")
    assert 'strings.get("keys-digest", self.keys_digest)' not in src, (
        "기본값이 기대값이면 이 검사는 공허하다"
    )
    assert 'strings.get("keys-digest", "")' in src


# ==========================================================================
# 14항 — 진입점 등록·키 통일
# ==========================================================================

def test_pyproject_에_flwr_앱이_등록돼_있다():
    """이 절이 없으면 `flwr run` 이 앱을 **해석조차 못 한다**(F1)."""
    import tomllib

    cfg = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    flwr = cfg["tool"]["flwr"]
    assert flwr["app"]["components"]["serverapp"] == "fl.server_app:app"
    assert flwr["app"]["components"]["clientapp"] == "fl.client_det:app"
    assert flwr["federations"]["local-sim"]["options"]["num-supernodes"] == 3


def test_라운드_키를_리터럴로_쓰지_않는다():
    """두 배선이 다른 이름(`round` 대 `server-round`)을 써서 진입점이 죽었다(F1)."""
    for name in ("fl/server_app.py", "fl/pilot_sim.py", "fl/client_vlm.py"):
        src = Path(name).read_text(encoding="utf-8")
        body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        assert '"server-round"' not in body, f"{name}: 상수 대신 리터럴을 썼다"
        assert '"round"' not in body, f"{name}: 옛 키 리터럴이 남아 있다"
    assert SERVER_ROUND_KEY == "server-round"


def test_서버가_보내는_설정에_정본_키가_실린다():
    """`ArrayRecord` 는 리스트 경로에서 이름을 인덱스로 바꾼다 — 이름은 따로 실려야 한다."""
    from fl.round_wiring import CANONICAL_KEYS_KEY

    for name in ("fl/server_app.py", "fl/pilot_sim.py"):
        assert "CANONICAL_KEYS_KEY" in Path(name).read_text(encoding="utf-8"), name
    assert CANONICAL_KEYS_KEY == "canonical-keys"


def test_회계_마감이_finally_에_있다():
    """학습 밖 단계가 죽어도 회계는 남아야 한다 — 파일럿에서 라운드를 날린 그 고장."""
    import ast

    for name in ("fl/server_app.py", "fl/pilot_sim.py"):
        tree = ast.parse(Path(name).read_text(encoding="utf-8"))
        tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
        called = any(
            isinstance(c, ast.Call) and getattr(c.func, "id", "") == "finalize_accounting"
            for t in tries for stmt in t.finalbody for c in ast.walk(stmt)
        )
        assert called, f"{name}: finalize_accounting 이 finally 에 없다"


def test_load_initial_이_더_이상_NotImplementedError_가_아니다():
    """F1 — 이 함수 때문에 `flwr run` 진입점이 아예 못 떴다."""
    src = Path("fl/server_app.py").read_text(encoding="utf-8")
    assert "초기 파라미터 로더는 칸 진입 시 연결한다" not in src


# ==========================================================================
# 14·15항 — 완주 스모크. 실제 Flower 런타임 위에서 두 진입점을 돌린다.
# ==========================================================================

def _run(tmp_path: Path, *, out_name: str, fail_at: str = "") -> Path:
    """스모크 칸으로 실제 Flower 런타임을 돈다.

    **`monkeypatch` 로 클라이언트를 바꿔치기할 수 없다.** Ray 액터는 별도 프로세스라
    부모의 패치가 닿지 않고, 실물 클라이언트가 그대로 돌아 GPU 를 잡는다(실측: 스모크가
    게이트 학습과 자원 경합으로 10분 넘게 굶었고, num_gpus=0 을 주면 클라이언트가
    `Invalid device id` 로 죽었다). 그래서 스모크 경로가 **코드에** 있어야 한다.
    """
    from fl.pilot_sim import SMOKE_BACKEND, SMOKE_CELL, run_pilot_fed

    out = tmp_path / out_name
    run_pilot_fed({
        "cell": SMOKE_CELL, "out_dir": out, "num_rounds": 2, "local_epochs": 1,
        "total_epochs": 2, "num_clients": 3, "base_seed": 1, "run_stamp": "smoke",
        "split_hash": "deadbeef", "canonical_keys": KEYS,
        "initial_arrays": _dummy_arrays(), "reference_sd": REF,
        "smoke_fail_at": fail_at,
    }, backend_config=SMOKE_BACKEND)
    return out


def test_run_simulation_스모크가_완주한다(tmp_path):
    """더미 2텐서·3클라이언트·R=2·E=1 로 서버 루프를 끝까지 돌린다(G10-2).

    이 시험이 실제로 도는 것: `WeldFedAvg` 라운드 루프 · 실패 검사 3종 · `weighted_fedavg`
    가드 · `round_wiring` 기록 · `finalize_accounting`. ⑦ 도 이제 이 루프를 탄다(15항).
    """
    out = _run(tmp_path, out_name="ok")

    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["ok"], audit["failures"]
    assert (out / "accounting.csv").exists()
    assert (out / "atomic_log.csv").exists()
    assert (out / "global_r002.npz").exists(), "라운드별 글로벌이 남아야 궤적을 본다"
    assert (out / "DO_NOT_CITE.md").exists(), "배선 산출물이 인용 가능한 상태로 남으면 안 된다"
    assert {int(k): v for k, v in audit["total_epochs_by_client"].items()} == {0: 2, 1: 2, 2: 2}


def test_가중_단위가_회계에_기록된다(tmp_path):
    """총괄 판정 2 — 회계가 단위를 말해야 RQ3 을 해석할 수 있다."""
    import csv

    out = _run(tmp_path, out_name="unit")
    rows = list(csv.DictReader((out / "accounting.csv").open(encoding="utf-8")))
    assert rows and all(r["fedavg_weight_unit"] == "supervised_tokens" for r in rows)
    assert all(float(r["fedavg_weight"]) > 0 for r in rows)
    assert all(int(r["supervised_tokens"]) > 0 for r in rows)


def test_원자_로그에_epochs_ran_과_lr_이_남는다(tmp_path):
    """F9 — ⑦ 로그에 이 둘이 없어 R×E=N 을 복원할 수 없었고, 그래서 회계가 상수로 채워졌다."""
    import csv

    out = _run(tmp_path, out_name="log")
    rows = list(csv.DictReader((out / "atomic_log.csv").open(encoding="utf-8")))
    names = {r["metric_name"] for r in rows}
    for need in ("epochs_ran", "lr", "optimizer_steps", "supervised_tokens", "fedavg_weight"):
        assert need in names, f"원자 로그에 {need} 가 없다 — 사후 복원이 불가능해진다"


def test_클라이언트를_죽이면_실패하되_회계는_디스크에_남는다(tmp_path):
    """`finally` 가 실제로 값을 하는지 — 파일럿에서 라운드를 날린 그 고장이다."""
    out = tmp_path / "killed"
    with pytest.raises(Exception):
        _run(tmp_path, out_name="killed", fail_at="1,2")

    # 여기가 요점이다 — 실패했어도 회계가 남아 있어야 "어디서 끊겼는가"를 답할 수 있다.
    assert (out / "audit.json").exists(), "회계가 유실됐다 — finally 가 값을 못 했다"
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert not audit["ok"], "라운드가 깨졌는데 회계가 통과로 남았다"
    assert audit["failures"], "실패 사유가 비어 있다"
    # 1라운드는 정상이었으므로 그 셀은 남아 있어야 한다 — 어디까지 갔는지가 산출물이다.
    assert (out / "accounting.csv").exists()
