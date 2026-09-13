"""python -m scripts.probe.discrimination_metadata_baseline --snapshot <동결본>

기존 결과 파일을 수정하지 않고 메타데이터 대조선 JSON을 표준 출력으로 보낸다.
"""

from __future__ import annotations

import argparse
import json

from evaluation.discrimination_baseline import compute_metadata_baseline
from evaluation.strata import ID_GRANULARITY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="data/interim/manifest_v1")
    parser.add_argument("--ladder", action="store_true", help="기존 K 사다리 전부를 사후 민감도 진단")
    args = parser.parse_args()
    result = ([compute_metadata_baseline(args.snapshot, k=k) for k in ID_GRANULARITY]
              if args.ladder else compute_metadata_baseline(args.snapshot))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
