"""프로브 1e — 런당 기동비만 따로 잰다. ④ 연합 예산의 지배 항이다.

## 왜 따로 재는가

재외삽에서 ④ 분리·연합의 절반이 **라운드당 기동비**다. 라운드마다 트레이너를 새로
만들기 때문이고, R=50·클라이언트 3이면 시드당 150번 낸다. 그런데 지금까지 이 값은
다른 측정에서 빼서 얻은 잔차였다.

- 프로브 1b: 256.5초 (라벨 **전수 스캔 포함**, 다른 트랙 작업 없음)
- 프로브 1d: 510.6초 (라벨 캐시 있음, **다른 트랙의 디스크 작업과 겹침**)

캐시가 있는 쪽이 더 큰 것은 경합 때문이지 기동비가 커서가 아니다. 지배 항을 잔차로
두면 안 된다 — 직접 잰다.

## 어떻게

`fraction` 을 아주 작게 줘서 **스텝을 거의 돌지 않는 런**을 연속으로 띄운다. 벽시계에서
실제 스텝 시간을 빼면 남는 것이 기동비다. 클라이언트 뷰마다 따로 재는 이유는 라벨 캐시
적재 비용이 표본 수를 타기 때문이다.

첫 런은 캐시를 만들고(콜드), 둘째·셋째는 캐시를 읽는다(웜). ④ 가 내는 것은 웜 쪽이다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.probe_det_steptime import BatchTimer  # noqa: E402

VIEW = Path("outputs/probe_c/view_central").resolve()
OUT = Path("outputs/probe_c/probe1e_startup.json").resolve()
N_REPEAT = 3
FRACTION = 0.005      # 44,846 × 0.005 ≈ 224장 → 7스텝


def one(i: int) -> dict:
    from detection.round_runner import train_round

    bt = BatchTimer()
    t0 = time.perf_counter()
    res = train_round(
        data_yaml=VIEW / "data.yaml", model="yolo11s.pt",
        total_epochs=100, local_epochs=1, round_idx=90, client_idx=i, base_seed=0,
        project=Path(f"outputs/probe_c/probe1e_{i}").resolve(), profile="main",
        extra_overrides={"fraction": FRACTION},
        callbacks={"on_train_batch_start": bt.on_start, "on_train_batch_end": bt.on_end},
    )
    wall = time.perf_counter() - t0
    n = min(len(bt.starts), len(bt.ends))
    span = bt.ends[n - 1] - bt.starts[0] if n else 0.0
    return {"run": i, "wall_s": round(wall, 1), "n_steps": n,
            "step_span_s": round(span, 1), "startup_s": round(wall - span, 1),
            "peak_vram_gb": round(res.peak_vram_gb, 3),
            "optimizer_updates": getattr(res, "optimizer_updates", None)}


def main() -> None:
    runs = [one(i) for i in range(N_REPEAT)]
    for r in runs:
        print(json.dumps(r, ensure_ascii=False), flush=True)
    warm = [r["startup_s"] for r in runs[1:]]
    rep = {
        "fraction": FRACTION, "view": str(VIEW), "runs": runs,
        "startup_cold_s": runs[0]["startup_s"],
        "startup_warm_s_mean": round(sum(warm) / len(warm), 1) if warm else None,
        "주의": "이 값이 ④ 연합의 라운드당 기동비다. 라벨 캐시가 있는 상태의 값을 쓴다",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "runs"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
