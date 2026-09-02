"""스모크 — LoRA A 공유 초기화가 실제로 고쳐졌는가 (74번 감사 C-1).

## 무엇을 보는가

한 숫자다. **`global_l2 / mean(client_l2)`**.

- 고장 상태(클라이언트마다 독립 난수 A): 가중 평균이 상쇄라 비가 `sqrt(sum w^2)` 로 떨어진다.
  파일럿 ⑦ 에서 w=(1275,656,342)/2273 이었고 그 값이 0.64852, 실측 비가
  20.5755/31.6787 = **0.6495** 였다. 두 자리까지 맞았다.
- 고쳐진 상태(같은 A 에서 출발): 세 클라이언트가 같은 기저 위에서 조금씩 갈라졌을 뿐이라
  비가 **1 근처**에 머문다.

즉 이 비 하나가 고쳐졌는지 아닌지를 정확히 말해 준다. 10시간짜리 ⑦ 재실행이 필요 없다.

## 왜 소표본으로 충분한가

이 스모크는 **성능을 재지 않는다.** 초기화 규약과 집계 산술만 본다. 둘 다 표본 크기와
무관하다. 클라이언트당 40행이면 라운드당 옵티마이저 갱신이 2회씩 일어나 어댑터가 실제로
움직이고, 그 상태에서 비를 보면 된다.

파일럿 성능 수치는 논문에 싣지 않으므로 ⑦ 전면 재실행은 하지 않는다(지시서).

    uv run python scripts/smoke_lora_init.py

산출: outputs/probe_c/smoke_lora_init/report.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

BASE_SEED = 20260828       # 파일럿과 같은 상수
R, E = 3, 1                # 라운드 3 · 로컬 epoch 1
N_PER_CLIENT = 40
OUT = Path("outputs/probe_c/smoke_lora_init")

#: 파일럿 ⑦ r0 실측. 고장 상태의 지문이며 이 스모크의 대조군이다(재실행하지 않는다).
PILOT_R0 = {
    "client_param_l2": [31.7405, 31.6717, 31.6238],
    "global_l2": 20.5755,
    "n_k": [1275, 656, 342],
}


def shrink_if_independent(weights: list[int]) -> float:
    """독립 난수 가정에서 예측되는 축소율 `sqrt(sum w^2)`."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(np.sqrt((w**2).sum()))


def main() -> None:
    from detection import serialize
    from fl.aggregate import weighted_fedavg
    from vlm.init_adapter import assert_same_start, build_initial_adapter
    from vlm.pilot_vlm import load_pairs, train_rounds

    OUT.mkdir(parents=True, exist_ok=True)
    clients = ["C1", "C2", "C3"]
    shards = {i: load_pairs("train", client=c)[:N_PER_CLIENT] for i, c in enumerate(clients)}
    n_train = {i: len(shards[i]) for i in shards}
    print(f"표본 {n_train} (클라이언트당 최대 {N_PER_CLIENT})", flush=True)

    t0 = time.perf_counter()
    init_arrays, keys, _ = build_initial_adapter(
        seed=BASE_SEED, cache_path=OUT / "adapter_initial.npz"
    )
    print(f"초기 어댑터 {len(keys)} 텐서 ‖·‖ {serialize.params_l2_norm(init_arrays):.4f} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    rounds: list[dict] = []
    global_arrays, ref_sd = init_arrays, None
    for r in range(R):
        payloads, weights, cl2, init_proofs = [], [], [], {}
        for i in sorted(shards):
            arrays, keys, m, ref_sd = train_rounds(
                rows=shards[i], epochs=E, round_idx=r, client_idx=i,
                base_seed=BASE_SEED, adapter_in=global_arrays, adapter_keys=keys,
            )
            payloads.append(arrays); weights.append(n_train[i])
            cl2.append(m["param_l2"]); init_proofs[i] = m["init_proof"]
            print(f"  r{r} c{i}: ‖param‖ {m['param_l2']:.4f} steps {m['optimizer_steps']} "
                  f"ep {m['epochs_ran']} opt {m['optimizer']} lr {m['lr']} "
                  f"({m['wall_s']:.0f}s)", flush=True)

        # 런타임 가드 — 세 클라이언트가 같은 A 에서 출발했는가
        assert_same_start(init_proofs)

        agg = weighted_fedavg(payloads, weights, keys,
                              {k: torch.as_tensor(v) for k, v in ref_sd.items()})
        global_arrays = agg.ndarrays
        mean_l2 = float(np.mean(cl2))
        ratio = agg.global_norm / mean_l2
        rounds.append({
            "round": r,
            "client_param_l2": cl2,
            "client_mean_l2": round(mean_l2, 4),
            "global_l2": round(agg.global_norm, 4),
            "ratio_global_over_client_mean": round(ratio, 4),
            "shrink_if_independent": round(shrink_if_independent(weights), 5),
            "init_digest_all_equal": True,          # assert_same_start 를 통과했다
        })
        print(f"  r{r} 집계: global_l2 {agg.global_norm:.4f} / 클라이언트 평균 {mean_l2:.4f} "
              f"= {ratio:.4f}  (독립난수라면 {shrink_if_independent(weights):.4f})", flush=True)

    r0 = rounds[0]
    verdict = {
        "판정": "고쳐짐" if r0["ratio_global_over_client_mean"] > 0.95 else "여전히 상쇄",
        "근거": "r0 의 global/클라이언트평균 비가 1 근처면 공유 초기값, "
                "sqrt(sum w^2) 근처면 독립 난수 상쇄다.",
        "r0_비": r0["ratio_global_over_client_mean"],
        "독립난수_예측": r0["shrink_if_independent"],
        "파일럿_고장_상태_비": round(PILOT_R0["global_l2"] / float(np.mean(PILOT_R0["client_param_l2"])), 4),
        "파일럿_독립난수_예측": round(shrink_if_independent(PILOT_R0["n_k"]), 5),
    }
    rep = {
        "설정": {"base_seed": BASE_SEED, "R": R, "E": E, "n_per_client": n_train,
                 "model": "Qwen/Qwen3.5-0.8B", "note": "성능 측정이 아니다 — 초기화 규약과 집계 산술만 본다"},
        "대조군_파일럿_r0": PILOT_R0,
        "라운드": rounds,
        "판정": verdict,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print("\n" + json.dumps(verdict, ensure_ascii=False, indent=2))
    print(f"\n총 {rep['wall_s']:.0f}s → {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
