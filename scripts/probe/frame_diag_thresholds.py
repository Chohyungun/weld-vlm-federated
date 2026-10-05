"""프레임 진단의 문턱 절차를 돌려 **제안**을 적는다 — `evaluation.frame_diag_thresholds.run()`.

합성 사례만 쓴다(모델 출력 · 평가 자산 · 동결 스냅샷을 열지 않는다). 산출은 배타 생성이다 — 있으면 덮지 않고 멈춘다.
제안은 등록이 아니다. 등록은 규칙 파일(`configs/registration/frame_diag_rules-*.json`)과 크기 재판정 · 사용자 승인을 거친다(07번 §31-1 · §32-7).

    python -m scripts.probe.frame_diag_thresholds --out <경로>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.frame_diag_thresholds import run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    report = run()
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    with args.out.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    r = report["result"]
    print(f"{r['status']} — {json.dumps(r.get('rules'), ensure_ascii=False)}", file=sys.stderr)
    return 0 if r["status"] == "centered" else 2


if __name__ == "__main__":
    raise SystemExit(main())
