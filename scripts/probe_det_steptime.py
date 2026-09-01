"""프로브 1 — 검출 스텝시간·활용률·peak VRAM·디코드 대 연산 비율.

## 무엇을 재는가

본실험 프로파일(`profile='main'` = YOLO11s·640·batch 32)에서

1. **정상상태 스텝시간** — 워밍업 구간(앞 20스텝)을 뺀 중앙값·평균
2. **GPU 활용률** — `nvidia-smi --query-gpu=utilization.gpu` 를 0.5초 간격으로 표본,
   구간은 **정상상태 스텝의 첫 배치 시작 ~ 마지막 배치 종료** 로 잘라서 집계한다.
   (61번이 활용률을 적으면서 도구와 구간을 기재하지 않아 검증이 불가능했다)
3. **peak VRAM** — `torch.cuda.max_memory_allocated`(할당분)와 `nvidia-smi memory.used`
   (컨텍스트·캐시 포함)를 **둘 다** 낸다. 두 값은 다른 것을 재며 후자가 실제 한계선이다.
4. **디코드 대 연산 비율** — Ultralytics 배치 콜백 사이 간격으로 분해한다.
       t_wait[i]    = batch_start[i] - batch_end[i-1]   (로더 대기 = 디코드·증강·전송)
       t_compute[i] = batch_end[i]   - batch_start[i]   (순전파·역전파·옵티마이저)
   이 분해가 저활용의 원인을 데이터·연산 중 어디로 볼지 가른다. **추측하지 않는다.**

## mosaic on/off 를 따로 재는 이유

`close_mosaic=10` 이라 본실험 100 epoch 중 **90 epoch 은 mosaic 이 켜진 채** 돈다.
`epochs=2` 로 부르면 2 > 100-10 이 되어 처음부터 mosaic 이 꺼진 채 돌고, 실제 본실험이
쓰는 파이프라인을 한 번도 재지 못한다. 그래서 `total_epochs=100` 을 유지한 채
`round_idx` 로 위치를 옮겨 두 구간을 각각 잰다.

    mosaic ON  : round_idx=0  → start_epoch 0   (epoch 0..89 구간의 대표)
    mosaic OFF : round_idx=90 → start_epoch 90  (epoch 90..99 구간의 대표)

외삽은 `90 × ep_on + 10 × ep_off` 다.

## 표본 축소

스텝시간 측정에 전체 epoch(44,846장 = 1,402스텝)을 돌 이유가 없다. `fraction` 으로
학습 목록만 줄인다 — 5칸 공통 고정 항목이 아니고(계측 전용 실행이다), 배치 구성·증강·
모델·이미지 크기는 본실험과 동일하다. 이 실행의 가중치는 **버린다.**
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SNAPSHOT_DIR = "data/interim/manifest_v1"
OUT_ROOT = Path("outputs/probe_c").resolve()
VIEW_DIR = Path("outputs/probe_c/view_central").resolve()
WARMUP_STEPS = 20
TARGET_STEPS = 300          # batch 32 기준 → fraction 으로 맞춘다


class GpuSampler:
    """nvidia-smi 0.5초 표본. 구간을 잘라 집계할 수 있도록 타임스탬프를 함께 남긴다."""

    INTERVAL = 0.5
    QUERY = "utilization.gpu,utilization.memory,memory.used,power.draw"

    def __init__(self) -> None:
        self.samples: list[tuple[float, float, float, float, float]] = []
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.QUERY}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()[0]
                vals = [float(v.strip()) for v in out.split(",")]
                self.samples.append((time.perf_counter(), *vals))
            except Exception:
                pass
            self._stop.wait(self.INTERVAL)

    def start(self) -> None:
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=3)

    def window(self, t0: float, t1: float) -> dict:
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return {"n_samples": 0}
        util = [x[1] for x in s]
        mem = [x[3] for x in s]
        return {
            "n_samples": len(s),
            "window_s": round(t1 - t0, 1),
            "util_gpu_mean": round(sum(util) / len(util), 1),
            "util_gpu_p50": round(sorted(util)[len(util) // 2], 1),
            "util_gpu_max": round(max(util), 1),
            "mem_used_mib_max": round(max(mem), 0),
            "power_w_mean": round(sum(x[4] for x in s) / len(s), 1),
        }


class BatchTimer:
    """배치 경계 시각만 기록한다. 학습에 영향 없음."""

    def __init__(self) -> None:
        self.starts: list[float] = []
        self.ends: list[float] = []

    def on_start(self, trainer) -> None:   # noqa: ANN001 - ultralytics 콜백 규약
        self.starts.append(time.perf_counter())

    def on_end(self, trainer) -> None:     # noqa: ANN001
        self.ends.append(time.perf_counter())

    def split(self, warmup: int) -> dict:
        n = min(len(self.starts), len(self.ends))
        if n <= warmup + 5:
            return {"n_steps": n, "note": "표본 부족"}
        waits, comps = [], []
        for i in range(warmup, n):
            comps.append(self.ends[i] - self.starts[i])
            if i > 0:
                waits.append(self.starts[i] - self.ends[i - 1])
        tot_w, tot_c = sum(waits), sum(comps)
        steps = [self.ends[i] - self.ends[i - 1] for i in range(warmup + 1, n)]
        steps_sorted = sorted(steps)
        return {
            "n_steps_total": n,
            "n_steps_steady": len(steps),
            "step_s_mean": round(sum(steps) / len(steps), 4),
            "step_s_p50": round(steps_sorted[len(steps) // 2], 4),
            "step_s_p90": round(steps_sorted[int(len(steps) * 0.9)], 4),
            "wait_s_mean": round(tot_w / max(len(waits), 1), 4),
            "compute_s_mean": round(tot_c / len(comps), 4),
            "wait_ratio": round(tot_w / max(tot_w + tot_c, 1e-9), 3),
            "steady_t0": self.starts[warmup],
            "steady_t1": self.ends[n - 1],
        }


def build_view() -> tuple[Path, int]:
    from data.manifest_io import load_snapshot, split_view
    from detection.dataset_view import build_yolo_view

    sn = load_snapshot(SNAPSHOT_DIR)
    n_train = len(split_view(sn.manifest, "train"))
    if (VIEW_DIR / "data.yaml").exists():
        print(f"뷰 재사용: {VIEW_DIR} (train {n_train})", flush=True)
        return VIEW_DIR / "data.yaml", n_train
    t0 = time.perf_counter()
    r = build_yolo_view(sn, out_dir=VIEW_DIR, train_client=None)
    print(f"뷰 생성 {time.perf_counter()-t0:.0f}s · train {r.n_images['train']} "
          f"박스 {r.n_boxes['train']} · 배경 {r.n_background} · geom 제외 {r.n_geom_invalid}",
          flush=True)
    return r.data_yaml, r.n_images["train"]


def run_leg(name: str, data_yaml: Path, round_idx: int, fraction: float,
            sampler: GpuSampler) -> dict:
    from detection.round_runner import train_round

    bt = BatchTimer()
    t_leg0 = time.perf_counter()
    res = train_round(
        data_yaml=data_yaml, model="yolo11s.pt",
        total_epochs=100, local_epochs=1,          # N=100 유지 — close_mosaic 위치가 정본
        round_idx=round_idx, client_idx=0, base_seed=0,
        project=OUT_ROOT / f"probe1_{name}", profile="main",
        extra_overrides={"fraction": fraction},
        callbacks={"on_train_batch_start": bt.on_start,
                   "on_train_batch_end": bt.on_end},
    )
    leg_wall = time.perf_counter() - t_leg0
    split = bt.split(WARMUP_STEPS)
    out = {
        "leg": name, "round_idx": round_idx, "fraction": fraction,
        "wall_s": round(leg_wall, 1),
        "optimizer_steps": res.optimizer_steps,
        "peak_vram_gb_torch": round(res.peak_vram_gb, 3),
        **split,
    }
    if "steady_t0" in split:
        out["gpu"] = sampler.window(split["steady_t0"], split["steady_t1"])
        out.pop("steady_t0"); out.pop("steady_t1")
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    data_yaml, n_train = build_view()
    # batch 32 기준 TARGET_STEPS 스텝이 되도록 학습 목록을 줄인다
    fraction = min(1.0, round(TARGET_STEPS * 32 / n_train, 4))
    print(f"train {n_train} · fraction {fraction} → 약 {int(n_train*fraction/32)} 스텝/epoch",
          flush=True)

    sampler = GpuSampler()
    sampler.start()
    legs = []
    try:
        for name, ridx in (("mosaic_on", 0), ("mosaic_off", 90)):
            print(f"\n=== {name} (round_idx={ridx}) ===", flush=True)
            legs.append(run_leg(name, data_yaml, ridx, fraction, sampler))
            print(json.dumps(legs[-1], ensure_ascii=False, indent=2), flush=True)
    finally:
        sampler.stop()

    report = {
        "snapshot": SNAPSHOT_DIR, "n_train": n_train, "fraction": fraction,
        "profile": "main", "model": "yolo11s.pt", "batch": 32, "imgsz": 640,
        "warmup_steps": WARMUP_STEPS,
        "gpu_tool": "nvidia-smi utilization.gpu, 0.5s 표본, 정상상태 스텝 구간만",
        "legs": legs,
    }
    if len(legs) == 2 and "step_s_p50" in legs[0]:
        on, off = legs[0], legs[1]
        steps_per_epoch = n_train / 32
        ep_on = on["step_s_p50"] * steps_per_epoch
        ep_off = off["step_s_p50"] * steps_per_epoch
        report["extrapolation"] = {
            "steps_per_epoch_full": round(steps_per_epoch, 1),
            "epoch_s_mosaic_on": round(ep_on, 1),
            "epoch_s_mosaic_off": round(ep_off, 1),
            "epochs_100_s": round(90 * ep_on + 10 * ep_off, 1),
            "epochs_100_h": round((90 * ep_on + 10 * ep_off) / 3600, 2),
        }
    path = OUT_ROOT / "probe1_det_steptime.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {path}", flush=True)
    if "extrapolation" in report:
        print(json.dumps(report["extrapolation"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
