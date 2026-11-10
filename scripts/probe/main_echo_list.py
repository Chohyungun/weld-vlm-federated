"""본실험 에코 목록을 만든다 — 정의 안(1016l, 연구 책임자 결정에 올린다): ECC 대상과 같은 규칙을 val 분할에 건 목록.

규칙은 `vlm.main_registration.echo_list_ids`(정본 `evaluation.early_coord_check.target_ids` 와 같은 식 — 봉인 페어의 그 분할 행 가운데
골격 결함이 하나 이상인 장에서 R0 가 뺀 장을 뺀 것, 정렬)다. 입력과 대조는 ECC 대상 빌더(`scripts/probe/early_coord_check_targets.py`)와
같다 — 동결 스냅샷의 digest 를 닻과, R0 목록의 sha256 을 인자와 맞대고, 만든 목록을 val 분할만 읽는 매니페스트 뷰로 다시 본다.
출력 목록에는 실물 id 가 들므로 추적하지 않는 경로에 쓴다. 표준 출력에는 건수와 해시만 낸다.

    python -X utf8 -B -m scripts.probe.main_echo_list --snapshot data/interim/manifest_v1 \\
        --pairs data/processed/pairs_main_v1/pairs.jsonl --removed <R0 목록> --removed-sha256 <hex> --out <목록 파일>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build(*, snapshot: Path, pairs: Path, removed: Path, removed_sha256: str, anchor_digest: str) -> tuple[bytes, dict]:
    """(목록 바이트, 보고). 어긋나면 `SystemExit`(사유 한 줄). 보고에는 id 를 싣지 않는다."""
    from data.manifest_io import verify_snapshot
    from data.manifest_view import read_manifest_columns
    from evaluation.eval_list import canonical_bytes, validate_eval_list
    from vlm.main_registration import ECHO_SPLIT, echo_list_ids

    if verify_snapshot(snapshot) != anchor_digest:
        raise SystemExit("스냅샷 digest 가 닻과 다르다")
    raw_removed = Path(removed).read_bytes()
    if hashlib.sha256(raw_removed).hexdigest() != removed_sha256:
        raise SystemExit("R0 목록의 sha256 이 다르다")
    gone = {x.strip() for x in raw_removed.decode("utf-8").splitlines() if x.strip()}
    h = hashlib.sha256()
    with Path(pairs).open("rb") as f:                 # 페어 파일을 한꺼번에 올리지 않는다
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    ids = echo_list_ids(_rows(Path(pairs)), gone)
    if not ids:
        raise SystemExit("에코 목록이 비었다")
    raw = canonical_bytes(ids)
    el = validate_eval_list(raw, where="에코 목록")
    df = read_manifest_columns(snapshot, ["image_id", "split"], split_filter=(ECHO_SPLIT,))
    val = {str(r.image_id) for r in df.itertuples() if str(r.split) == ECHO_SPLIT}
    outside = sum(1 for i in el.ids if i not in val)
    if outside:
        raise SystemExit(f"에코 목록에 동결 매니페스트의 val 이 아닌 장이 {outside}개 있다")
    return raw, {"n": len(el), "file_sha256": el.file_sha256, "set_sha256": el.set_sha256, "pairs_sha256": h.hexdigest(),
                 "removed_sha256": removed_sha256, "removed_n": len(gone), "split": ECHO_SPLIT}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--removed", type=Path, required=True)
    ap.add_argument("--removed-sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))["fixed_before_main_runs"]
    raw, rep = build(snapshot=a.snapshot, pairs=a.pairs, removed=a.removed, removed_sha256=a.removed_sha256,
                     anchor_digest=base["snapshot_digest"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "xb") as fh:
        fh.write(raw)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
