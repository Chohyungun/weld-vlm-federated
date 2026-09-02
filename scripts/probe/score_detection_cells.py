"""**폐지 — `score_cells.py` 로 대체됐다.** 65번 재현 명령을 깨지 않으려고 남긴 껍데기다.

    구: uv run python scripts/probe/score_detection_cells.py
    신: uv run python scripts/probe/score_cells.py predict --at-conf
        uv run python scripts/probe/score_cells.py score

왜 옮겼나 — 이 파일에 `CONF = 0.25` 가 모듈 상수로 박혀 있었다. 분리형만 저신뢰 박스를
버린 뒤 채점되고 통합형에는 대응 임계가 없어, RQ2 의 핵심 대비가 그 비대칭 위에
있었다(감사 D-1 · P14). 임계를 인자로 만들려면 추론이 스크립트 상수와 분리돼야 한다.

지금 임계·시드·모집단은 `evaluation.params`, 추론은 `evaluation.detect_infer`,
정답·칸 목록은 `evaluation.cells`, 지표는 `evaluation.score` 에 있다. 진입점이 무엇이든
같은 함수를 탄다 — "단일 채점기"를 코드 구조가 받치게 한 것이 77번 과제 6이다.

산출물 경로가 바뀐다: `score_detection_v1.json` → `score_cells_v1.json`.
**옛 파일은 덮어쓰지 않는다.** 새 채점이 그 값을 완전 일치로 대조하는 회귀 시험의
기준이기 때문이다(`score_cells.py score` 의 `regression` 절).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.probe.score_cells import main as score_cells_main


def main() -> int:
    argv = sys.argv[1:]
    print(__doc__)
    print("→ score_cells.py predict --at-conf 로 넘긴다\n")
    sys.argv = ["score_cells.py", "predict", "--at-conf", *argv]
    rc = score_cells_main()
    if rc:
        return rc
    print("\n→ score_cells.py score 로 넘긴다\n")
    sys.argv = ["score_cells.py", "score", *argv]
    return score_cells_main()


if __name__ == "__main__":
    raise SystemExit(main())
