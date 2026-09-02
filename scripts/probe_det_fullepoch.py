"""프로브 1b — 전체 규모 1 epoch 실측. 표본 축소 편향과 런당 기동비를 함께 닫는다.

프로브 1 은 `fraction` 으로 9,602장만 돌렸다. 두 가지가 남는다.

1. **표본 편향.** `fraction` 은 무작위 표집이 아니라 목록 앞부분을 자른다. 그래서 부분집합의
   배경 이미지 비율이 68.6% 로 전체 49.8% 와 다르다. 배경 이미지는 박스가 없어 손실 계산이
   가볍다 — 스텝시간이 낙관적으로 나올 수 있다.
2. **런당 기동비.** 라벨 캐시 적재·모델 구성·AMP 점검·로더 기동은 표본 크기에 따라 다르고,
   ④ 연합은 이 비용을 **라운드 × 클라이언트 수만큼** 낸다. 67번 합산식의 마지막 항이 이것이다.

전체 중앙 뷰(44,846장)로 **정확히 1 epoch** 을 돌려 둘 다 실측한다.
`total_epochs=100` 은 유지한다 — 스케줄과 close_mosaic 위치를 본실험과 같게 두기 위해서다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("outputs/probe_c/probe1b_fullepoch.json").resolve()
VIEW = Path("outputs/probe_c/view_central").resolve()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_det_steptime import BatchTimer, GpuSampler, WARMUP_STEPS  # noqa: E402


def main() -> None:
    from detection.round_runner import train_round

    data_yaml = VIEW / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"중앙 뷰가 없다: {data_yaml}. probe_det_steptime.py 를 먼저 돌려라")

    sampler = GpuSampler()
    sampler.start()
    bt = BatchTimer()
    t0 = time.perf_counter()
    try:
        res = train_round(
            data_yaml=data_yaml, model="yolo11s.pt",
            total_epochs=100, local_epochs=1, round_idx=0, client_idx=0, base_seed=0,
            project=Path("outputs/probe_c/probe1b").resolve(), profile="main",
            callbacks={"on_train_batch_start": bt.on_start, "on_train_batch_end": bt.on_end},
        )
        wall = time.perf_counter() - t0
    finally:
        sampler.stop()

    split = bt.split(WARMUP_STEPS)
    n = split["n_steps_total"]
    # 스텝 총합은 정상상태 p50 로 환산하지 않는다 — 실제로 흐른 시간을 쓴다.
    step_span = bt.ends[n - 1] - bt.starts[0]
    report = {
        "view": str(VIEW), "profile": "main", "model": "yolo11s.pt",
        "batch": 32, "imgsz": 640, "epochs_ran": res.epochs_ran,
        "wall_s": round(wall, 1),
        "step_span_s": round(step_span, 1),
        "startup_s": round(wall - step_span, 1),
        "peak_vram_gb_torch": round(res.peak_vram_gb, 3),
        "optimizer_steps": res.optimizer_steps,
        "gpu_tool": "nvidia-smi utilization.gpu, 0.5s 표본, 정상상태 스텝 구간만",
        **{k: v for k, v in split.items() if not k.startswith("steady_")},
    }
    if "steady_t0" in split:
        report["gpu"] = sampler.window(split["steady_t0"], split["steady_t1"])
    report["extrapolation_100ep_h"] = round(
        (split["step_s_p50"] * n * 100 + (wall - step_span)) / 3600, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
