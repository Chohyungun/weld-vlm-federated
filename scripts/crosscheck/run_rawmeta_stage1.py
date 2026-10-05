"""전수 메타데이터화 1단계를 실물로 돌린다 — 원본 라벨에서 새 판을 낸다.

    uv run python -m scripts.crosscheck.run_rawmeta_stage1 [--out 경로]
    uv run python -m scripts.crosscheck.run_rawmeta_stage1 --expect-digest <지문> --staging <임시 경로>

입력은 저장소 안(정션)의 동결본 v1 · 원본 라벨 zip · 계약 사상표다. 기본 출력은 새 경로
`data/interim/manifest_v3_rawlabels` 다. 잠긴 판이 있으면 쓰지 않고 멈춘다(빌더의 출력 경로 가드).
회계를 표준출력에 JSON 으로 낸다 — 평가 행의 값은 회계에 없다.

**정본 목적지(`data/interim/` 과 그 아래)에는 `--expect-digest` 와 `--staging` 이 필수다.** 빠지면 빌더를 부르기
전에 거부한다. 판을 `--staging` 에 먼저 내고, 지문이 기대값과 같을 때만 정본 경로로 옮긴다. 다르면 정본 경로에
아무것도 쓰지 않고 멈춘다. 임시 경로는 정본 뿌리 밖이어야 하고, 목적지와 같거나 서로 한쪽이 다른 쪽 아래면
거부한다 — 대조 전에 정본 경로가 생기지 않게. 경로는 정션을 풀어 비교한다. 옮긴 뒤 지문과 읽는 쪽 검사를 다시
보고, 구성 파일과 잠금 파일에 읽기 전용 속성을 준다(동결본 v1 과 같은 잠금).

`--check-v2` 는 쓰지 않고 잠정판 v2 가 바뀐 읽는 쪽 검사를 그대로 지나는지만 본다.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

from data import absorb, rawmeta
from data.manifest_io import SNAPSHOT_FILENAME, verify_snapshot

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "data/interim/manifest_v1"
V2 = ROOT / "data/interim/manifest_v2_absorbed"
LABELS = ROOT / "data/interim/aihub_labels"
OUT = ROOT / "data/interim/manifest_v3_rawlabels"
LABEL_MAP = ROOT / "configs/label_map.yaml"
#: 정본 뿌리 — 이 경로와 그 아래는 지문 대조를 거쳐서만 쓴다.
CANONICAL_ROOT = ROOT / "data/interim"


def _real(p: Path) -> str:
    return os.path.normcase(os.path.realpath(p))


def _within(a: Path, b: Path) -> bool:
    """`a` 가 `b` 와 같거나 `b` 아래인가 — 정션을 풀고 대소문자를 맞춘 경로로 본다."""
    ra, rb = _real(a), _real(b)
    return ra == rb or ra.startswith(rb.rstrip(os.sep) + os.sep)


def check_destination(out: Path, staging: Path | None, expect: str | None) -> None:
    """빌더를 부르기 **전에** 목적지 규칙을 본다. 어기면 아무것도 쓰지 않고 거부한다."""
    if bool(expect) != bool(staging):
        raise rawmeta.RawMetaError("--expect-digest 와 --staging 은 함께 준다")
    if _within(out, CANONICAL_ROOT) and not expect:
        raise rawmeta.RawMetaError(
            f"정본 목적지({out.name})에는 --expect-digest 와 --staging 이 필수다 — 지문 대조 없이 쓰지 않는다")
    if staging is None:
        return
    if _within(staging, CANONICAL_ROOT):
        raise rawmeta.RawMetaError("임시 경로가 정본 뿌리 안이다 — 정본 뿌리 밖에 준다")
    if _within(staging, out) or _within(out, staging):
        raise rawmeta.RawMetaError("임시 경로와 목적지가 같거나 한쪽이 다른 쪽 아래다 — 대조 전에 정본 경로가 생긴다")


def _locked_names(root: Path) -> list[str]:
    names = [SNAPSHOT_FILENAME]
    for line in (root / SNAPSHOT_FILENAME).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            names.append(line.split(maxsplit=1)[1].strip())
    return names


def promote(staging: Path, out: Path, expect: str) -> str:
    """임시 경로의 판을 지문이 같을 때만 정본 경로로 옮기고 잠근다. 다르면 정본 경로를 건드리지 않는다."""
    got = verify_snapshot(staging)
    if got != expect:
        raise rawmeta.RawMetaError(f"임시 판의 지문이 기대값과 다르다 — {got} ≠ {expect}. 정본 경로에 쓰지 않았다")
    if out.exists():
        raise rawmeta.RawMetaError(f"정본 경로가 이미 있다: {out}")
    os.replace(staging, out)
    if verify_snapshot(out) != expect:
        raise rawmeta.RawMetaError("옮긴 판의 지문이 기대값과 다르다")
    absorb.load_absorbed(out)
    for name in _locked_names(out):
        (out / name).chmod(stat.S_IREAD)
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-v2", action="store_true", help="쓰지 않고 잠정판 v2 를 읽는 쪽 검사로만 연다")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="출력 경로. 정본 경로가 아닌 후보 경로에 먼저 낼 때 준다 — 잠긴 판이 있는 경로에는 쓰지 않는다")
    ap.add_argument("--expect-digest", help="정본에 낼 때 기대하는 지문. 주면 --staging 에 먼저 내고 같을 때만 옮긴다")
    ap.add_argument("--staging", type=Path, help="--expect-digest 와 함께 — 판을 먼저 낼 임시 경로")
    args = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if args.check_v2:
        absorb.load_absorbed(V2)
        print(json.dumps({"v2_load_absorbed": "통과"}, ensure_ascii=False))
        return 0
    try:
        check_destination(args.out, args.staging, args.expect_digest)
    except rawmeta.RawMetaError as exc:
        print(f"★ {exc}", flush=True)
        return 2
    try:
        target = args.staging if args.expect_digest else args.out
        acct = rawmeta.build(V1, LABELS, target, LABEL_MAP, v2_root=V2)
        if args.expect_digest:
            acct["promoted_to"] = args.out.name
            acct["promoted_digest"] = promote(args.staging, args.out, args.expect_digest)
    except (rawmeta.RawMetaError, absorb.AbsorptionError) as exc:
        print(f"★ {exc}\n\n쓰지 않았거나 끝 검사에서 멈췄다.", flush=True)
        return 2
    print(json.dumps(acct, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
