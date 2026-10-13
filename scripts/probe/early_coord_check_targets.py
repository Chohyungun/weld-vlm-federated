"""조기 좌표 점검의 대상 목록을 만든다 — 미니스펙 10번 13-1 의 m-4 정의, `evaluation/eval_list` 정규형.

봉인 페어를 **한 줄씩** 읽어 학습 행 가운데 골격 결함이 하나 이상인 장을 모으고, 데이터 쪽 R0 목록(sha256 대조)의 장을 뺀다.
만든 목록은 학습 분할만 읽는 매니페스트 뷰로 다시 대조한다 — 평가 id 는 그 뷰에 없으므로 섞이면 거부된다(m-2).
목록의 sha256 이 규칙 파일(`configs/early_coord_check.yaml`)의 대상과 같아야 성공이다. 출력 목록에는 실물 id 가 들므로
추적하지 않는 경로에 쓴다. 표준 출력에는 건수와 해시만 낸다.

    python -X utf8 -B -m scripts.probe.early_coord_check_targets --snapshot data/interim/manifest_v1 \\
        --pairs data/processed/pairs_main_v1/pairs.jsonl --removed <R0 목록> --removed-sha256 <hex> --out <목록 파일>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from evaluation import early_coord_check as E
from evaluation.eval_list import canonical_bytes, validate_eval_list

ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--removed", type=Path, required=True)
    ap.add_argument("--removed-sha256", required=True)
    ap.add_argument("--rules", type=Path, default=E.RULES_PATH)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from data.manifest_io import verify_snapshot
    from data.manifest_view import read_manifest_columns

    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))["fixed_before_main_runs"]
    if verify_snapshot(a.snapshot) != base["snapshot_digest"]:
        raise SystemExit("스냅샷 digest 가 닻과 다르다")
    h = hashlib.sha256()
    with a.pairs.open("rb") as f:                     # 페어 파일을 한꺼번에 올리지 않는다
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    pairs_sha = h.hexdigest()
    raw_removed = a.removed.read_bytes()
    if hashlib.sha256(raw_removed).hexdigest() != a.removed_sha256:
        raise SystemExit("R0 목록의 sha256 이 다르다")
    removed = {x.strip() for x in raw_removed.decode("utf-8").splitlines() if x.strip()}

    ids = E.target_ids(_rows(a.pairs), removed)
    raw = canonical_bytes(ids)
    el = validate_eval_list(raw, where="조기 점검 대상")
    df = read_manifest_columns(a.snapshot, ["image_id", "split"], split_filter=("train",))
    split_of = {str(r.image_id): str(r.split) for r in df.itertuples()}
    rules = E.load_rules(a.rules.read_bytes())
    E.check_target_list(el, split_of, rules)          # 학습 뷰 밖 id · 규칙의 대상과 다른 목록이면 거부

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(raw)
    print(json.dumps({"n": len(el), "file_sha256": el.file_sha256, "set_sha256": el.set_sha256,
                      "rules_target_sha256": rules.target_sha256, "same_as_rules": el.file_sha256 == rules.target_sha256,
                      "pairs_sha256": pairs_sha, "removed_sha256": a.removed_sha256, "removed_n": len(removed)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
