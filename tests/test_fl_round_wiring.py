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

# **skip 이 아니라 fail 이다** (85번 ⑨). `importorskip` 이면 `uv sync --extra` 사고로
# flwr 가 걷혔을 때 이 파일의 이빨 시험 전체가 조용히 skip 되어 초록으로 보인다 —
# CLAUDE.md 가 경고하는 바로 그 사고다. 의존성 부재는 환경 고장이므로 시끄럽게 죽인다.
try:
    import flwr  # noqa: F401
except ImportError as _exc:  # pragma: no cover
    pytest.fail(
        f"flwr 가 없다({_exc}) — `uv sync --extra detection --extra vlm --extra fl "
        "--extra corpus` 를 돌려라. skip 하면 항목 16 이빨 시험 전체가 무이빨이 된다."
    )

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
    # 85번 ⑤·⑥ — 등록 클라이언트는 칸 디스패처 하나다. 구판의 fl.client_det:app 은
    # train 핸들러가 죽어 있었다("등록된 것은 죽어 있고 도는 것은 등록돼 있지 않다").
    assert flwr["app"]["components"]["clientapp"] == "fl.client_app:app"

    # SuperLink 연결 설정은 pyproject 에 **없어야 한다.** flwr 1.33 CLI 가 이 절을
    # 발견하면 pyproject 를 스스로 재작성해 사용자 저장소로 이관한다(실측 — 그 재작성이
    # 착수 커밋에 딸려 들어가 이 시험이 KeyError 로 죽었다). 절이 되살아나면 다음
    # flwr run 이 실험 도중 또 파일을 고친다.
    assert "federations" not in flwr, (
        "[tool.flwr.federations] 가 되살아났다 — flwr run 이 pyproject 를 재작성한다. "
        "연결 설정은 scripts/main_det.py FEDERATION_CONFIG 로만 준다."
    )
    # --run-config 덮어쓰기가 허용되려면 키가 기본값에 선언돼 있어야 한다(flwr 실측).
    for key in ("resume-root", "smoke-fail-at"):
        assert key in flwr["app"]["config"], f"app.config 에 {key} 기본값이 없다"

    # 연결 설정 값의 정본은 이제 러너다 — num-supernodes=3 과 GPU 동시성 1 을 고정한다.
    from scripts.main_det import FEDERATION_CONFIG

    assert "options.num-supernodes=3" in FEDERATION_CONFIG
    assert "options.backend.client-resources.num-gpus=1.0" in FEDERATION_CONFIG
    src = Path("scripts/main_det.py").read_text(encoding="utf-8")
    assert '"--federation-config", FEDERATION_CONFIG' in src, (
        "러너가 연결 설정을 명시하지 않으면 값이 사용자 저장소(저장소 밖)에 산다"
    )


def test_라운드_키를_리터럴로_쓰지_않는다():
    """두 배선이 다른 이름(`round` 대 `server-round`)을 써서 진입점이 죽었다(F1)."""
    # 85번 ⑤ — 구판 목록에는 하필 client_det.py 만 빠져 있었고, 그 파일의 죽은
    # 핸들러가 옛 키 리터럴을 읽고 있었다. **정의 지점(fl/round_wiring.py)만 빼고**
    # fl/ 의 배선 파일 전부를 검사한다 — 상수의 집은 리터럴을 한 번은 가져야 한다.
    for name in ("fl/server_app.py", "fl/pilot_sim.py", "fl/client_vlm.py",
                 "fl/client_det.py", "fl/client_app.py"):
        src = Path(name).read_text(encoding="utf-8")
        body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        assert '"server-round"' not in body, f"{name}: 상수 대신 리터럴을 썼다"
        assert '"round"' not in body, f"{name}: 옛 키 리터럴이 남아 있다"
    assert SERVER_ROUND_KEY == "server-round"
    # 정의 지점 검사 — 리터럴은 round_wiring 에 정확히 한 번(정의행)만 산다.
    home = Path("fl/round_wiring.py").read_text(encoding="utf-8")
    body = "\n".join(l for l in home.splitlines() if not l.strip().startswith("#"))
    assert body.count('SERVER_ROUND_KEY = "server-round"') == 1


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

# --------------------------------------------------------------------------
# Ray 기동을 부하에 견디게 한다 — **건너뛰지 않는다**
#
# 머지 게이트에서 이 스모크가 raylet 기동 타임아웃(GCS overloaded)으로 죽었다. 그때 같은
# 기계에서 §4-6 파일럿이 GPU·CPU 를 물고 있었다. 부하에서 죽는 스모크는 게이트를
# 불안정하게 만든다.
#
# 처방은 셋이고 **어느 것도 skip 이 아니다.**
#   1. Ray 발자국을 묶고 기동 타임아웃을 올린다 (`SMOKE_BACKEND` · `ray_startup_env`)
#   2. 시작 전에 자원 여유를 **명시적으로 기다린다** — 기다려도 안 나면 실패시킨다
#   3. **기동 실패에 한해** 재시도한다. 단언 실패는 그대로 올린다
#
# skip 하면 무이빨이 된다. 기다리다 못 얻으면 그것도 보고할 사실이므로 실패로 남긴다.
# --------------------------------------------------------------------------

#: 자원이 빌 때까지 기다리는 한도. 넘으면 skip 이 아니라 **fail** 이다.
HEADROOM_WAIT_S = 600
#: 스모크 3 클라이언트 + Ray 오버헤드에 필요한 최소 여유.
NEED_FREE_GB = 6.0

#: Ray 기동 실패로만 판별할 문구. 여기 없는 실패는 재시도하지 않는다 —
#: 배선 결함을 재시도로 덮으면 이 시험이 하는 일이 없어진다.
_RAY_STARTUP_SIGNS = (
    "raylet", "gcs", "overloaded", "Failed to start", "timed out while waiting",
    "connection refused", "RaySystemError", "Cannot connect", "ray.init",
)


def _free_gb() -> float:
    import psutil

    vm = psutil.virtual_memory()
    return vm.available / 1e9


def _wait_for_headroom(deadline_s: float = HEADROOM_WAIT_S) -> float:
    """자원이 빌 때까지 기다린다. **못 얻으면 예외** — 호출부가 실패시킨다."""
    import time as _t

    end = _t.monotonic() + deadline_s
    last = _free_gb()
    while _t.monotonic() < end:
        last = _free_gb()
        if last >= NEED_FREE_GB:
            return last
        _t.sleep(10)
    raise RuntimeError(
        f"자원 여유를 {deadline_s:.0f}초 기다렸지만 {last:.1f}GB 뿐이다 "
        f"(필요 {NEED_FREE_GB}GB). 이 기계에서 다른 장기 작업이 돌고 있다 — "
        "스모크를 건너뛰지 않고 실패로 남긴다."
    )


def _is_ray_startup_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(sign.lower() in text for sign in _RAY_STARTUP_SIGNS)


def _run(tmp_path: Path, *, out_name: str, fail_at: str = "", _attempts: int = 3) -> Path:
    """스모크 칸으로 실제 Flower 런타임을 돈다.

    **`monkeypatch` 로 클라이언트를 바꿔치기할 수 없다.** Ray 액터는 별도 프로세스라
    부모의 패치가 닿지 않고, 실물 클라이언트가 그대로 돌아 GPU 를 잡는다(실측: 스모크가
    게이트 학습과 자원 경합으로 10분 넘게 굶었고, num_gpus=0 을 주면 클라이언트가
    `Invalid device id` 로 죽었다). 그래서 스모크 경로가 **코드에** 있어야 한다.

    기동 실패는 재시도하고 그 외 실패는 그대로 올린다.
    """
    import os
    import time as _t

    from fl.pilot_sim import (SMOKE_BACKEND, SMOKE_CELL, ray_startup_env,
                              run_pilot_fed)

    os.environ.update(ray_startup_env())
    out = tmp_path / out_name
    cfg = {
        "cell": SMOKE_CELL, "out_dir": out, "num_rounds": 2, "local_epochs": 1,
        "total_epochs": 2, "num_clients": 3, "base_seed": 1, "run_stamp": "smoke",
        "split_hash": "deadbeef", "canonical_keys": KEYS,
        "initial_arrays": _dummy_arrays(), "reference_sd": REF,
        "smoke_fail_at": fail_at,
    }

    last_startup_exc: BaseException | None = None
    for attempt in range(1, _attempts + 1):
        free = _wait_for_headroom()          # 못 기다리면 여기서 예외 → 실패
        try:
            run_pilot_fed(dict(cfg), backend_config=SMOKE_BACKEND)
            return out
        except BaseException as exc:                                 # noqa: BLE001
            if not _is_ray_startup_failure(exc):
                raise            # **배선 실패는 재시도하지 않는다.** 그대로 올린다
            last_startup_exc = exc
            print(f"[smoke] 시도 {attempt}/{_attempts}: Ray 기동 실패 "
                  f"(여유 {free:.1f}GB) — {type(exc).__name__}: {str(exc)[:120]}")
            try:
                import ray

                if ray.is_initialized():
                    ray.shutdown()
            except Exception:                                        # noqa: BLE001
                pass
            _t.sleep(15 * attempt)

    raise AssertionError(
        f"Ray 기동이 {_attempts}회 연속 실패했다 — 마지막: {last_startup_exc}. "
        "스모크를 건너뛰지 않는다(무이빨이 된다). 기계가 계속 이 상태면 CI 러너의 "
        "자원 배정을 고쳐야 한다."
    )


@pytest.mark.resource_heavy
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


@pytest.mark.resource_heavy
def test_가중_단위가_회계에_기록되고_거짓말하지_않는다(tmp_path):
    """총괄 판정 2 / 85번 ① — 실제 전송 가중이 단위가 가리키는 값과 같아야 한다.

    구판 스모크는 가중과 페어 수에 **같은 값**을 실어 키 충돌을 원리적으로 못 봤다.
    지금은 토큰(1000+c)과 페어 수(100+c)가 다른 값이라, 충돌이 재발해 페어 수가
    전송되면 여기서 fedavg_weight 가 100+c 로 잡혀 즉시 실패한다.
    """
    import csv

    out = _run(tmp_path, out_name="unit")
    rows = list(csv.DictReader((out / "accounting.csv").open(encoding="utf-8")))
    assert rows and all(r["fedavg_weight_unit"] == "supervised_tokens" for r in rows)
    for r in rows:
        c = int(r["client_idx"])
        assert float(r["fedavg_weight"]) == float(1000 + c), (
            f"전송 가중이 감독 토큰이 아니다: {r['fedavg_weight']} — 85번 ① 재발"
        )
        assert int(r["supervised_tokens"]) == 1000 + c
        assert int(r["num_examples"]) == 100 + c, "페어 수가 의미 키에서 사라졌다"
        assert float(r["fedavg_weight"]) != float(r["num_examples"]), (
            "가중과 페어 수가 같은 값이면 이 시험은 충돌을 구분하지 못한다"
        )


@pytest.mark.resource_heavy
def test_원자_로그에_epochs_ran_과_lr_이_남는다(tmp_path):
    """F9 — ⑦ 로그에 이 둘이 없어 R×E=N 을 복원할 수 없었고, 그래서 회계가 상수로 채워졌다."""
    import csv

    out = _run(tmp_path, out_name="log")
    rows = list(csv.DictReader((out / "atomic_log.csv").open(encoding="utf-8")))
    names = {r["metric_name"] for r in rows}
    for need in ("epochs_ran", "lr", "optimizer_steps", "supervised_tokens", "fedavg_weight"):
        assert need in names, f"원자 로그에 {need} 가 없다 — 사후 복원이 불가능해진다"


@pytest.mark.resource_heavy
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


# ==========================================================================
# 15항 — ⑦ 경로의 설정 키 계약
#
# 스모크는 더미 칸으로 돌므로 통합형 분기의 **키 이름**을 지나가지 않는다. F1 이
# 정확히 그 형태였다 — 서버가 보내는 키와 클라이언트가 읽는 키가 어긋나 라운드 1 에서
# 죽었고, 그 경로를 아무도 안 돌려 봐서 몰랐다. 실물 학습 없이 계약만 대조한다.
# ==========================================================================

def test_uni_fed_서버가_보내는_키와_클라이언트가_읽는_키가_맞는다():
    import ast

    from fl.pilot_sim import _cell_train_config

    cfg = {"client_tags": ["C1", "C2", "C3"]}
    sent = set(_cell_train_config("uni_fed", cfg, Path("/tmp/out")))
    # 서버 공통부가 함께 싣는 키
    sent |= {"cell", "canonical-keys", "local-epochs", "total-epochs", "num-rounds",
             "base-seed", "run-stamp", "resume-root"}

    # **등록 디스패처**(fl/client_app.py)가 실제로 읽는 키를 소스에서 뽑는다.
    src = Path("fl/client_app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "train")
    read: set[str] = set()
    for node in ast.walk(fn):
        # cfg["x"] / cfg.get("x")
        if isinstance(node, ast.Subscript) and getattr(node.value, "id", "") == "cfg":
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                read.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and getattr(node.func.value, "id", "") == "cfg"
                and node.args and isinstance(node.args[0], ast.Constant)):
            read.add(node.args[0].value)

    # f-string 으로 만드는 키(client-tag-N, num-examples-N)는 위 추출에 안 잡힌다.
    read |= {"client-tag-0", "client-tag-1", "client-tag-2"}
    # sep_fed·스모크 전용 키는 uni_fed 계약이 아니다
    read -= {"views-root", "model", "project", "profile", "num-examples-0", "smoke-fail-at"}

    missing = read - sent
    assert not missing, f"클라이언트가 읽는데 서버가 안 보내는 키: {sorted(missing)}"


def _fake_train_metrics() -> dict:
    return {
        "supervised_tokens": 4321, "epochs_ran": 2, "optimizer_steps": 7,
        "resumed_from_epoch": None, "param_l2": 1.5, "payload_bytes": 999,
        "seed": 11, "lr": 1e-4, "peak_vram_gb": 0.1, "optimizer": "AdamW",
        "init_proof": {"l2": 31.5},
    }


def test_uni_fed_가중은_감독_토큰이고_페어수는_별도_키다__행동_시험():
    """총괄 판정 2 / 85번 ① — **문자열이 아니라 실구성 dict 를 검사한다.**

    구판 시험은 `'WEIGHT_KEY: float(...)' in src` 문자열 검사라, dict 리터럴 중복 키로
    가중이 페어 수로 바뀐 깨진 코드에서도 통과했다. 여기서는 페이로드를 실제로 만들어
    전송될 값을 본다.
    """
    from fl.client_vlm import payload_metrics
    from fl.strategy import WEIGHT_KEY

    metrics, strings = payload_metrics(
        _fake_train_metrics(), n_pairs=77, canonical_keys=KEYS, client_idx=2)

    assert metrics[WEIGHT_KEY] == 4321.0, "전송 가중이 감독 토큰 총합이 아니다"
    assert metrics["num-examples"] == 77.0, "페어 수가 의미 키에 없다"
    assert metrics["supervised-tokens"] == 4321.0
    assert strings["weight-unit"] == "supervised_tokens"
    # 키 자체가 분리돼 있어야 한다 — 같으면 dict 구성에서 충돌이 재발할 수 있다.
    assert WEIGHT_KEY != "num-examples", "가중 키가 의미 키와 같다 — 85번 ① 의 뿌리"


def test_가중_키_분리_상수_고정():
    """`WEIGHT_KEY` 가 의미 키로 되돌아가는 회귀를 막는다."""
    from fl.round_wiring import WEIGHT_KEY as RW
    from fl.strategy import WEIGHT_KEY as ST

    assert ST == "fedavg-weight"
    assert RW is ST, "두 모듈이 다른 상수를 들고 있으면 한쪽만 바뀌는 사고가 재발한다"


def test_fl_배선의_dict_리터럴에_중복_키가_없다():
    """85번 ① 을 **원리적으로** 막는다 — 상수로 쓴 키를 값으로 풀어 문자열 키와 겹치는지
    fl/ 전체의 dict 리터럴에서 검사한다. 파이썬은 중복 키를 조용히 뒤가 이기게 둔다."""
    import ast as _ast

    known = {
        "WEIGHT_KEY": "fedavg-weight", "CANONICAL_KEYS_KEY": "canonical-keys",
        "SERVER_ROUND_KEY": "server-round", "ARRAYS_KEY": "arrays",
        "METRICS_KEY": "metrics", "CONFIG_KEY": "config",
    }
    for name in sorted(Path("fl").glob("*.py")):
        tree = _ast.parse(name.read_text(encoding="utf-8"))
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Dict):
                continue
            seen: dict[str, str] = {}
            for k in node.keys:
                if k is None:                      # **spread
                    continue
                if isinstance(k, _ast.Constant) and isinstance(k.value, str):
                    resolved, shown = k.value, f'"{k.value}"'
                elif isinstance(k, _ast.Name) and k.id in known:
                    resolved, shown = known[k.id], k.id
                else:
                    continue
                assert resolved not in seen, (
                    f"{name}:{k.lineno} dict 리터럴 중복 키 {resolved!r} "
                    f"({seen[resolved]} 와 {shown}) — 뒤가 조용히 이긴다(85번 ①)"
                )
                seen[resolved] = shown


def test_uni_fed_클라이언트가_주입을_서버_기준으로_검증한다():
    """G2-5 — 클라이언트끼리 비교로는 셋 다 no-op 일 때 통과한다."""
    src = Path("fl/client_vlm.py").read_text(encoding="utf-8")
    assert "assert_injected_matches" in src


# ==========================================================================
# 85번 ④ — 등록 진입점(`fl.server_app:app`)을 실제 런타임에서 끝까지 돌린다
#
# 지금까지의 스모크는 `pilot_sim` 서버만 돌았다. server_app 고유 구간(run_config 파싱·
# `_load_initial`·`_as_list`)은 실행 이력 0 이었다 — F1(죽은 채 커밋)과 같은 부류.
# `flwr run` 이 하는 일을 재현한다: 등록된 두 컴포넌트 + run_config 를 실은 Context.
# ==========================================================================

@pytest.mark.resource_heavy
def test_flwr_run_진입점이_끝까지_돈다(tmp_path):
    import os

    from flwr.common import Context, RecordDict
    from flwr.supercore.telemetry import EventType
    from flwr.simulation.run_simulation import _run_simulation
    from flwr.supercore.run import Run

    from fl.client_app import app as registered_client
    from fl.pilot_sim import SMOKE_BACKEND, ray_startup_env
    from fl.server_app import app as registered_server

    os.environ.update(ray_startup_env())
    _wait_for_headroom()

    run = Run.create_empty(run_id=20260902)
    run.primary_task_id = 1
    run.federation_id = "@none/default"          # NOOP_FEDERATION_ID 실측값

    project = tmp_path / "proj"
    ctx = Context(
        run_id=run.run_id, node_id=0, node_config={}, state=RecordDict(),
        run_config={
            # pyproject [tool.flwr.app.config] 와 같은 모양 — flwr run 이 넣어 주는 것
            "cell": "smoke", "num-server-rounds": 2, "local-epochs": 1,
            "total-epochs": 2, "num-clients": 3, "num-classes": 4,
            "base-seed": 1, "run-stamp": "entry", "split-hash": "deadbeef",
            "model": "yolo11s.pt", "project": str(project),
            "views-root": str(project / "views"), "profile": "main",
            "num-examples": "1,1,1", "client-tags": "C1,C2,C3",
        },
    )
    _run_simulation(
        num_supernodes=3,
        exit_event=EventType.PYTHON_API_RUN_SIMULATION_LEAVE,
        client_app=registered_client,
        server_app=registered_server,
        backend_config=dict(SMOKE_BACKEND),
        server_app_context=ctx,
        run=run,
    )

    out = project / "fl" / "smoke"
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["ok"], audit["failures"]
    assert (out / "accounting.csv").exists()
    assert (out / "DO_NOT_CITE.md").exists(), "스모크 산출물에 인용 금지 표식이 없다"
    assert (out / "global_r002.npz").exists()


# ==========================================================================
# 85번 ⑦ — run_gates 지문 1개는 통과가 아니다
# ==========================================================================

def _fp(tag: str, coord_hash: str = "h1"):
    from tracking.mlflow_local import CellFingerprint

    return CellFingerprint(cell=tag, base_ckpt_sha256="b", coords_sha256="c",
                           coord_cfg_hash=coord_hash, rag_snapshot_sha256="r")


def test_지문_1개는_cells_identical_이_None_이다():
    """구판은 `if prints:` 라 지문 1개(통합형 학습이 정확히 그렇다)가 대조 없이
    true 로 기록됐다 — 이 게이트가 잡으려던 무이빨 형태의 재발(85번 ⑦)."""
    from fl.run_gates import apply_run_gates

    out = apply_run_gates(cell="t", fingerprints=[_fp("a")])
    assert out["gate_results"]["cells_identical"] is None
    assert "check_cells_identical" not in out["gates_evaluated"], (
        "부르지 않은 대조를 불렀다고 적으면 gates_evaluated 가 거짓말한다"
    )


def test_지문_2개_동일은_true_다():
    from fl.run_gates import apply_run_gates

    out = apply_run_gates(cell="t", fingerprints=[_fp("a"), _fp("b")])
    assert out["gate_results"]["cells_identical"] is True
    assert "check_cells_identical" in out["gates_evaluated"]


def test_지문_2개_상이는_죽는다__이빨():
    from fl.run_gates import apply_run_gates

    with pytest.raises(RuntimeError, match="같아야 할 값"):
        apply_run_gates(cell="t", fingerprints=[_fp("a"), _fp("b", coord_hash="h2")])


# ==========================================================================
# 85번 ① — 회계의 단위 일치 감사 (이빨)
# ==========================================================================

def test_가중_단위가_거짓말하면_회계가_실패한다():
    """85번 ① 그 자체를 픽스처로 — unit=supervised_tokens 인데 가중이 페어 수."""
    from detection.budget_audit import AccountingCell, AccountingMatrix

    m = AccountingMatrix(num_rounds=1, client_ids=[0], local_epochs=1, total_epochs=1)
    m.record(AccountingCell(
        round_idx=0, client_idx=0, epochs_ran=1, optimizer_steps=5, num_examples=100,
        seed=1, optimizer="AdamW", lr=1e-4, arg_optimizer="AdamW", arg_lr0=1e-4,
        fedavg_weight=100.0,                     # 페어 수가 전송됐다
        fedavg_weight_unit="supervised_tokens",  # 단위는 토큰이라고 주장한다
        supervised_tokens=4321,
    ))
    rep = m.audit()
    assert not rep.ok
    assert any("거짓말" in f for f in rep.failures), rep.failures


def test_가중_단위가_정직하면_통과한다():
    from detection.budget_audit import AccountingCell, AccountingMatrix

    m = AccountingMatrix(num_rounds=1, client_ids=[0], local_epochs=1, total_epochs=1)
    m.record(AccountingCell(
        round_idx=0, client_idx=0, epochs_ran=1, optimizer_steps=5, num_examples=100,
        seed=1, optimizer="AdamW", lr=1e-4, arg_optimizer="AdamW", arg_lr0=1e-4,
        fedavg_weight=4321.0, fedavg_weight_unit="supervised_tokens",
        supervised_tokens=4321,
    ))
    rep = m.audit()
    assert rep.ok, rep.failures
