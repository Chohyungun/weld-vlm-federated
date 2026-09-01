"""GPU 작업을 순차로 돌린다 — 겹치면 시간 측정이 오염된다.

프로브 2(크기-시간 곡선)·프로브 1c(부하 레버)·승격 어블레이션은 전부 **시간을 재는**
작업이다. 겹쳐 돌리면 셋 다 못 쓰게 된다. 한 프로세스에서 차례로 띄우고, 하나가 실패해도
나머지는 계속한다 — 한 단계 실패로 두 시간을 잃지 않는다.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("프로브 2 — 모델 크기-시간 곡선 + 판정 11", ["scripts/probe_vlm_scale.py"]),
    ("프로브 1c — 결정성 비용·동시 실행", ["scripts/probe_det_load.py"]),
    ("승격 어블레이션 두 팔", ["scripts/run_ablation.py"]),
]


#: 앞선 GPU 작업이 끝나기를 기다린다. 겹치면 첫 항목(크기-시간 곡선)이 오염된다.
WAIT_FOR = Path("outputs/probe_c/probe1b_fullepoch.json")


def main() -> None:
    if WAIT_FOR.name and not WAIT_FOR.exists():
        print(f"[queue] {WAIT_FOR} 를 기다린다 (앞선 실측과 겹치지 않게)", flush=True)
        while not WAIT_FOR.exists():
            time.sleep(20)
        time.sleep(30)   # 프로세스가 GPU 를 놓을 여유
    py = sys.executable
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    for name, args in STEPS:
        tag = Path(args[0]).stem
        t0 = time.perf_counter()
        print(f"\n########## {name} ##########", flush=True)
        with open(log_dir / f"{tag}.log", "w", encoding="utf-8") as out:
            rc = subprocess.run([py, *args], stdout=out, stderr=subprocess.STDOUT).returncode
        print(f"[queue] {tag}: rc={rc} {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
