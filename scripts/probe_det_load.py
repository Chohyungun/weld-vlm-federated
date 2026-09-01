"""프로브 1c — GPU 부하 레버 실측 (과제 5).

프로브 1 에서 본실험 프로파일의 활용률은 62.9~82.5%(중앙값 88~93%)였고 로더 대기는
2.3~9.6%였다. **데이터 경로는 병목이 아니다.** 그러면 남는 여지는 연산 쪽이다.

여기서 재는 것은 두 가지다.

1. **결정성 비용.** `deterministic=True` 는 `torch.use_deterministic_algorithms` 와
   cudnn 결정 알고리즘을 켠다. 빠른 커널이 배제되므로 대가가 있다. **얼마인지 모른 채로
   '재현성을 위해 감수한다'고 말할 수는 없다.** 계측 전용 `probe_nondet` 프로파일로 잰다.
   이 프로파일의 가중치는 실험 산출물이 아니다.
2. **동시 실행 처리량.** 칸·클라이언트 간 독립 실행을 한 GPU 에 겹칠 때 총 처리량이
   실제로 오르는지. 겹치면 **개별 벽시계는 반드시 늘어난다** — 그래서 논문의 처리 속도
   표에는 비경합 단독 실측치만 쓴다. 여기서 보려는 것은 총 처리량이지 개별 속도가 아니다.

동시 실행분은 `concurrent` 표시를 붙여 남긴다. 표시 없는 값만 성능 비교에 쓸 수 있다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_det_steptime import BatchTimer, GpuSampler, WARMUP_STEPS  # noqa: E402

VIEW = Path("outputs/probe_c/view_central").resolve()
OUT = Path("outputs/probe_c/probe1c_load.json").resolve()
FRACTION = 0.2141          # 프로브 1 과 같은 표본 — 비교 가능해야 한다


def one_leg(tag: str, profile: str, client_idx: int = 0) -> dict:
    from detection.round_runner import train_round

    sampler = GpuSampler()
    sampler.start()
    bt = BatchTimer()
    t0 = time.perf_counter()
    try:
        res = train_round(
            data_yaml=VIEW / "data.yaml", model="yolo11s.pt",
            total_epochs=100, local_epochs=1, round_idx=90, client_idx=client_idx,
            base_seed=0, project=Path(f"outputs/probe_c/probe1c_{tag}").resolve(),
            profile=profile, extra_overrides={"fraction": FRACTION},
            callbacks={"on_train_batch_start": bt.on_start, "on_train_batch_end": bt.on_end},
        )
        wall = time.perf_counter() - t0
    finally:
        sampler.stop()
    split = bt.split(WARMUP_STEPS)
    out = {"leg": tag, "profile": profile, "wall_s": round(wall, 1),
           "peak_vram_gb_torch": round(res.peak_vram_gb, 3),
           **{k: v for k, v in split.items() if not k.startswith("steady_")}}
    if "steady_t0" in split:
        out["gpu"] = sampler.window(split["steady_t0"], split["steady_t1"])
    return out


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("det", "nondet"):
        tag = "single_det" if stage == "det" else "single_nondet"
        prof = "main" if stage == "det" else "probe_nondet"
        r = one_leg(tag, prof)
        (Path(f"outputs/probe_c/_leg_{stage}.json")).write_text(
            json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
        return
    if stage.startswith("conc"):
        idx = int(stage[-1])
        r = one_leg(f"conc{idx}", "main", client_idx=idx)
        r["concurrent"] = True
        (Path(f"outputs/probe_c/_leg_conc{idx}.json")).write_text(
            json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
        return

    py = sys.executable
    legs = []
    print("=== 1) 결정성 켬 (단독) ===", flush=True)
    subprocess.run([py, __file__, "det"], check=True)
    legs.append(json.loads(Path("outputs/probe_c/_leg_det.json").read_text(encoding="utf-8")))

    print("=== 2) 결정성 끔 (단독, 계측 전용) ===", flush=True)
    subprocess.run([py, __file__, "nondet"], check=True)
    legs.append(json.loads(Path("outputs/probe_c/_leg_nondet.json").read_text(encoding="utf-8")))

    print("=== 3) 동시 2개 (본실험 프로파일) ===", flush=True)
    t0 = time.perf_counter()
    procs = [subprocess.Popen([py, __file__, f"conc{i}"]) for i in (0, 1)]
    rcs = [p.wait() for p in procs]
    conc_wall = time.perf_counter() - t0
    for i in (0, 1):
        f = Path(f"outputs/probe_c/_leg_conc{i}.json")
        if f.exists():
            legs.append(json.loads(f.read_text(encoding="utf-8")))

    by = {l["leg"]: l for l in legs}
    rep = {"fraction": FRACTION, "legs": legs, "concurrent_rcs": rcs,
           "gpu_tool": "nvidia-smi utilization.gpu, 0.5s 표본, 정상상태 스텝 구간만"}
    if "single_det" in by and "single_nondet" in by:
        a, b = by["single_det"], by["single_nondet"]
        rep["determinism_cost"] = {
            "step_s_p50_deterministic": a["step_s_p50"],
            "step_s_p50_nondeterministic": b["step_s_p50"],
            "slowdown_from_determinism": round(a["step_s_p50"] / b["step_s_p50"], 3),
            "util_deterministic": a.get("gpu", {}).get("util_gpu_p50"),
            "util_nondeterministic": b.get("gpu", {}).get("util_gpu_p50"),
        }
    if "single_det" in by and "conc0" in by and "conc1" in by:
        s = by["single_det"]["step_s_p50"]
        c = max(by["conc0"]["step_s_p50"], by["conc1"]["step_s_p50"])
        rep["concurrency"] = {
            "single_step_s_p50": s,
            "concurrent_step_s_p50_worst": c,
            "per_run_slowdown": round(c / s, 3),
            "aggregate_throughput_gain": round(2 * s / c, 3),
            "concurrent_wall_s": round(conc_wall, 1),
            "peak_vram_gb_sum_est": round(
                by["conc0"]["peak_vram_gb_torch"] + by["conc1"]["peak_vram_gb_torch"], 2),
            "주의": "동시 실행분 벽시계는 논문 처리 속도 표에 쓰지 않는다",
        }
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in rep if k in ("determinism_cost", "concurrency")},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
