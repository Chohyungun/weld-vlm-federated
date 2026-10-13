"""python -m scripts.probe.early_check_count --snapshot <스냅샷 루트> --removed <R0 가 뺀 학습검증 id 목록> \
    --removed-sha256 <그 sha256> --pairs <봉인 페어 pairs.jsonl> --out <미추적 폴더>

조기 좌표 점검의 대상 집합을 센다 — 07번 §41(조기 좌표 점검 미니스펙)의 집계. **학습 분할만 읽는다** — 매니페스트는 열을 고르는
판독기로 `split == train` 행만, 주석은 평가 id 를 열기 전에 거부하는 판독기로 대상 장만 읽는다. 평가 자산을 열지 않는다.

대상의 두 읽기를 함께 센다 — (가) R0 를 걸고 **남은** 학습 결함 장(지금의 작업 해석), (나) R0 가 **뺀** 학습 결함 장(결정 문구 그대로의
읽기). 둘 다 (1) 장 수 (2) 정답 박스 수(주석 줄 · 페어 골격의 결함 수) (3) 묶음 수 (4) 층마다 장 · 묶음 · 박스를 내고, (가)는 (5) 생성
한도의 타깃 거름(학습 타깃의 토큰 수 ≤ 한도) 뒤의 같은 넷도 낸다. 표준 출력에는 건수만 싣는다. id 목록은 `--out` 아래에 쓰고 그 sha256 만
표준 출력에 싣는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ("image_id", "split", "group_id", "strata_key", "has_defect", "material")
FLOORS = {"pairs": 100, "pair_groups": 20, "image_groups": 20}
"""진단 검사기의 기본 하한(`evaluation/frame_diag.py` 의 `min_pairs` · `min_pair_groups` · `min_groups`). 쌍 수의 상한은 정답 박스 수,
쌍의 묶음 수의 상한은 박스가 든 묶음 수다(짝짓기는 정답 하나에 쌍 하나)."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _summary(ids: list[str], meta: dict, boxes_ann: dict, boxes_sk: dict) -> dict:
    by = defaultdict(lambda: {"images": 0, "groups": set(), "boxes_ann": 0, "boxes_skeleton": 0})
    for i in ids:
        s = by[meta[i]["strata_key"]]
        s["images"] += 1
        s["groups"].add(meta[i]["group_id"])
        s["boxes_ann"] += boxes_ann.get(i, 0)
        s["boxes_skeleton"] += boxes_sk.get(i, 0)
    groups = {meta[i]["group_id"] for i in ids}
    groups_with_box = {meta[i]["group_id"] for i in ids if boxes_ann.get(i, 0) > 0}
    out = {"images": len(ids), "groups": len(groups), "groups_with_box": len(groups_with_box),
           "boxes_ann": sum(boxes_ann.get(i, 0) for i in ids), "boxes_skeleton": sum(boxes_sk.get(i, 0) for i in ids),
           "by_stratum": {k: {"images": v["images"], "groups": len(v["groups"]), "boxes_ann": v["boxes_ann"],
                              "boxes_skeleton": v["boxes_skeleton"]} for k, v in sorted(by.items())}}
    out["reaches_floor"] = {"pairs<=boxes_ann": out["boxes_ann"] >= FLOORS["pairs"],
                            "pair_groups<=groups_with_box": out["groups_with_box"] >= FLOORS["pair_groups"],
                            "image_groups": out["groups"] >= FLOORS["image_groups"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--removed", type=Path, required=True)
    ap.add_argument("--removed-sha256", required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--max-target-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from data.manifest_io import verify_snapshot
    from data.manifest_view import read_annotations_view, read_manifest_columns

    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))["fixed_before_main_runs"]
    digest = verify_snapshot(a.snapshot)
    if digest != base["snapshot_digest"]:
        raise SystemExit("스냅샷 digest 가 닻과 다르다")
    raw_removed = a.removed.read_bytes()
    if _sha(raw_removed) != a.removed_sha256:
        raise SystemExit("R0 목록의 sha256 이 다르다")
    removed = {x.strip() for x in raw_removed.decode("utf-8").splitlines() if x.strip()}

    df = read_manifest_columns(a.snapshot, COLUMNS, split_filter=("train",))
    meta = {str(r.image_id): {"group_id": str(r.group_id), "strata_key": str(r.strata_key)} for r in df.itertuples()}
    defect = {str(r.image_id) for r in df.itertuples() if bool(r.has_defect)}
    keep = sorted(defect - removed)
    gone = sorted(defect & removed)

    ann = read_annotations_view(a.snapshot, keep + gone)
    boxes_ann: dict[str, int] = defaultdict(int)
    for i in ann["image_id"].astype(str):
        boxes_ann[i] += 1
    want = set(keep) | set(gone)
    pair_rows: dict[str, dict] = {}
    with a.pairs.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if str(r["image_id"]) in want:
                pair_rows[str(r["image_id"])] = r
    boxes_sk = {i: len(r["skeleton"]["defects"]) for i, r in pair_rows.items()}

    from vlm.rehearsal_run import target_token_counts
    m = base["uni_model"]
    cfg = SimpleNamespace(coord_space="ABS_ORIG", model_id=m["id"],
                          model_revision=m["revision"], processor_kwargs=base["uni_processor_kwargs"])
    tokens = target_token_counts([pair_rows[i] for i in keep if i in pair_rows], config=cfg)
    kept_filtered = sorted(i for i in keep if tokens.get(i, a.max_target_tokens + 1) <= a.max_target_tokens)

    a.out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, ids in (("remaining_R0_train_defect", keep), ("remaining_R0_train_defect_le_tokens", kept_filtered),
                      ("removed_R0_train_defect", gone)):
        raw = ("\n".join(ids) + "\n").encode("utf-8")
        (a.out / f"{name}.txt").write_bytes(raw)
        files[name] = {"n": len(ids), "sha256": _sha(raw)}
    out = {"snapshot_digest": digest, "removed_list_sha256": a.removed_sha256, "max_target_tokens": a.max_target_tokens,
           "pairs_file_sha256": _sha(a.pairs.read_bytes()),
           "pairs_found": {"remaining": sum(1 for i in keep if i in pair_rows), "removed": sum(1 for i in gone if i in pair_rows)},
           "tokens_counted": len(tokens), "floors": FLOORS,
           "remaining": _summary(keep, meta, boxes_ann, boxes_sk),
           "remaining_le_tokens": _summary(kept_filtered, meta, boxes_ann, boxes_sk),
           "removed": _summary(gone, meta, boxes_ann, boxes_sk),
           "id_files": files}
    (a.out / "early_check_count.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                                                  newline="\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
