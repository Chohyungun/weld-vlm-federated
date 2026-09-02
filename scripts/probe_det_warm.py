"""프로브 1d — 첫 epoch 과 둘째 epoch 을 갈라 잰다. 콜드 읽기 비용을 분리한다.

## 왜 필요한가

프로브 1b(전체 중앙 뷰 1 epoch)에서 **로더 대기가 38.5%** 로 나왔다. 부분표본(9,602장)의
2.3~9.6% 와 크게 다르다. 스텝시간도 중앙값 0.273초인데 평균이 0.465초, p90 이 1.033초다 —
**꼬리가 두껍다.** 디스크에서 처음 읽는 이미지가 만드는 지연이다.

여기서 갈리는 것이 외삽의 폭이다.

- 콜드 읽기가 **첫 epoch 에만** 있다면 → 100 epoch 은 중앙값에 가깝다 (약 10.6시간)
- **매 epoch 반복된다면** → 평균에 가깝다 (약 18.1시간)

1.7배 차이고, 유료 GPU 결정의 입력이다. **추측하지 않는다** — 2 epoch 을 돌려서 본다.
학습 풀 이미지 총량이 약 2GB 라 OS 페이지 캐시에 들어갈 수 있지만, 들어갔는지는
재봐야 안다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_det_steptime import GpuSampler, WARMUP_STEPS  # noqa: E402

VIEW = Path("outputs/probe_c/view_central").resolve()
OUT = Path("outputs/probe_c/probe1d_warm.json").resolve()


class EpochSplitTimer:
    """배치 경계 시각을 epoch 별로 나눠 담는다."""

    def __init__(self) -> None:
        self.epochs: list[dict] = []

    def on_epoch_start(self, trainer) -> None:  # noqa: ANN001
        self.epochs.append({"epoch": int(trainer.epoch), "starts": [], "ends": []})

    def on_batch_start(self, trainer) -> None:  # noqa: ANN001
        if self.epochs:
            self.epochs[-1]["starts"].append(time.perf_counter())

    def on_batch_end(self, trainer) -> None:  # noqa: ANN001
        if self.epochs:
            self.epochs[-1]["ends"].append(time.perf_counter())

    def summary(self, warmup: int) -> list[dict]:
        out = []
        for e in self.epochs:
            s, t = e["starts"], e["ends"]
            n = min(len(s), len(t))
            if n <= warmup + 5:
                continue
            waits = [s[i] - t[i - 1] for i in range(warmup, n) if i > 0]
            comps = [t[i] - s[i] for i in range(warmup, n)]
            steps = [t[i] - t[i - 1] for i in range(warmup + 1, n)]
            ss = sorted(steps)
            out.append({
                "epoch": e["epoch"], "n_steps": n, "n_steady": len(steps),
                "epoch_span_s": round(t[n - 1] - s[0], 1),
                "step_s_mean": round(sum(steps) / len(steps), 4),
                "step_s_p50": round(ss[len(ss) // 2], 4),
                "step_s_p90": round(ss[int(len(ss) * 0.9)], 4),
                "wait_ratio": round(sum(waits) / max(sum(waits) + sum(comps), 1e-9), 3),
                "_t0": s[warmup], "_t1": t[n - 1],
            })
        return out


def main() -> None:
    from detection.round_runner import train_round

    data_yaml = VIEW / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"중앙 뷰가 없다: {data_yaml}")

    sampler = GpuSampler()
    sampler.start()
    et = EpochSplitTimer()
    t0 = time.perf_counter()
    try:
        res = train_round(
            data_yaml=data_yaml, model="yolo11s.pt",
            total_epochs=100, local_epochs=2, round_idx=0, client_idx=0, base_seed=0,
            project=Path("outputs/probe_c/probe1d").resolve(), profile="main",
            callbacks={"on_train_epoch_start": et.on_epoch_start,
                       "on_train_batch_start": et.on_batch_start,
                       "on_train_batch_end": et.on_batch_end},
        )
        wall = time.perf_counter() - t0
    finally:
        sampler.stop()

    eps = et.summary(WARMUP_STEPS)
    for e in eps:
        e["gpu"] = sampler.window(e.pop("_t0"), e.pop("_t1"))

    report = {
        "view": str(VIEW), "profile": "main", "model": "yolo11s.pt", "batch": 32,
        "wall_s": round(wall, 1), "epochs_ran": res.epochs_ran,
        "optimizer_steps_batches": res.optimizer_steps,
        "optimizer_updates": getattr(res, "optimizer_updates", None),
        "peak_vram_gb_torch": round(res.peak_vram_gb, 3),
        "gpu_tool": "nvidia-smi utilization.gpu, 0.5s 표본, epoch 별 정상상태 구간",
        "epochs": eps,
    }
    if len(eps) >= 2:
        a, b = eps[0], eps[1]
        report["판정"] = {
            "첫_epoch_대기비율": a["wait_ratio"], "둘째_epoch_대기비율": b["wait_ratio"],
            "첫_epoch_평균스텝s": a["step_s_mean"], "둘째_epoch_평균스텝s": b["step_s_mean"],
            "둘째가_더_빠른가": b["step_s_mean"] < a["step_s_mean"],
            "개선율": round(1 - b["step_s_mean"] / a["step_s_mean"], 3),
            "정상상태_추정_100ep_h": round(b["step_s_mean"] * b["n_steps"] * 100 / 3600, 2),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report.get("판정", report), ensure_ascii=False, indent=2), flush=True)
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
