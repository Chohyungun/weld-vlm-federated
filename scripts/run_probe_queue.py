"""GPU 작업을 순차로 돌린다 — 겹치면 시간 측정이 오염된다.

프로브 2(크기-시간 곡선)·프로브 1c(부하 레버)·승격 어블레이션은 전부 **시간을 재는**
작업이다. 겹쳐 돌리면 셋 다 못 쓰게 된다. 한 프로세스에서 차례로 띄우고, 하나가 실패해도
나머지는 계속한다 — 한 단계 실패로 두 시간을 잃지 않는다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

#: 콘솔 기본 인코딩이 cp949 라 한글·기호가 섞이면 `UnicodeEncodeError` 로 **큐 자체가
#: 죽는다.** 실제로 한 번 죽었다(em dash). 자식 프로세스에도 물려준다.
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

STEPS = [
    ("프로브 1d: 첫 epoch 대 둘째 epoch (콜드 읽기 분리)", ["scripts/probe_det_warm.py"]),
    ("프로브 2: 모델 크기-시간 곡선 + 판정 11", ["scripts/probe_vlm_scale.py"]),
    ("프로브 1c: 결정성 비용·동시 실행", ["scripts/probe_det_load.py"]),
    ("승격 어블레이션 두 팔", ["scripts/run_ablation.py"]),
]


def main() -> None:
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
