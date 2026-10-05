"""계약 #4 의 1.4 판 JSON Schema 생성. 손으로 고치지 말고 이 스크립트를 돌린다.

    uv run python scripts/gen_prediction_schema_v14.py

1.3 파일(`evaluation/prediction.schema.json`)은 **만들지도 열지도 않는다** — 그 파일은 동결이다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.schema_v14 import json_schema_v14

OUT = REPO / "evaluation/prediction.schema.v1_4.json"


def main() -> None:
    # newline="\n" 고정 — Windows 기본 CRLF 로 쓰면 플랫폼마다 파일 해시가 갈린다.
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(json_schema_v14(), ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
