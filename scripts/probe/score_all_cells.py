"""**폐지 — `score_cells.py score` 로 대체됐다.** 66번 재현 명령을 위해 남긴 껍데기다.

    구: uv run python scripts/probe/score_all_cells.py
    신: uv run python scripts/probe/score_cells.py score

옮긴 내용은 전부 그대로다 — 검출 3칸 되읽기, 통합형 2칸 어댑터, 좌표계 건강, 회복률,
자명하한, P9, 인용 진단. 달라진 것은 임계·시드·모집단·게이트가 `evaluation.params`
한 곳에서 오고, 정답 생성이 `evaluation.cells.load_population` 한 지점이라는 것이다.

산출물 경로가 바뀐다: `score_all_cells_v1.json` → `score_cells_v1.json`.
**옛 파일은 덮어쓰지 않는다.** 새 채점이 그 값을 완전 일치로 대조한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.probe.score_cells import main as score_cells_main


def main() -> int:
    print(__doc__)
    print("→ score_cells.py score 로 넘긴다\n")
    sys.argv = ["score_cells.py", "score", *sys.argv[1:]]
    return score_cells_main()


if __name__ == "__main__":
    raise SystemExit(main())
