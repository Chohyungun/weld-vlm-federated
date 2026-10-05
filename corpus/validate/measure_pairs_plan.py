"""본실험 D4 페어 설계용 계측 — **읽기 전용.** 아무것도 만들지 않는다.

동결 매니페스트(`data/interim/manifest_v1`)의 train·val 에서

1. 예상 건수(분할 × 참여자 × 결함/정상, 재질, 결함 코드)를 세고,
2. 평가 격리를 프로그램으로 확인하고(image_id·group_id 의 eval 교차 0건),
3. 파일럿 빌더(`corpus.generate.make_pairs_pilot`)의 함수를 **그대로** 불러 메모리 안에서만
   페어를 조립해 검증 게이트 통과·폐기 사유·조항 미특정 건수와 소요 시간을 재고,
4. 고정 모델 토크나이저로 학습 타깃 길이 분포를 잰다(구조화 JSON · 기준 서술 · 둘의 합).

**eval 분할의 정답은 읽지 않는다** — 길이 계측은 train+val 만이다(개발규약 1-4, 검토 §24-4).
결과는 표준 출력의 JSON 하나다. 파일을 쓰지 않으므로 봉인 자산·산출 디렉터리와 부딪히지 않는다.

허용치 표는 파일럿 초판(`limits_v0_pilot.csv`, 미검수)이다. 여기서 나온 조항 수·길이는
"그 표로 만들었을 때" 의 값이고, 표가 바뀌면 다시 재야 한다.

실행:
  uv run python -m corpus.validate.measure_pairs_plan                     # 건수·격리·조립
  uv run python -m corpus.validate.measure_pairs_plan --tokenizer Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO / "data/interim/manifest_v1"


def _pct(values: list[int], qs=(50, 90, 95, 99)) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    out = {"n": len(v), "min": v[0], "max": v[-1], "mean": round(sum(v) / len(v), 1)}
    for q in qs:
        out[f"p{q}"] = v[min(len(v) - 1, round(q / 100 * (len(v) - 1)))]
    return out


def counts_and_isolation(m) -> dict:
    tv = m[m["split"].isin(["train", "val"])]
    ev = m[m["split"] == "eval"]
    by = Counter((r.split, r.client, bool(r.has_defect)) for r in tv.itertuples())
    table = {}
    for (split, client, has), n in sorted(by.items()):
        table.setdefault(split, {}).setdefault(str(client), {})["결함" if has else "정상"] = n
    eval_ids, eval_groups = set(ev["image_id"]), set(ev["group_id"])
    return {
        "n_manifest": len(m), "n_train_val": len(tv), "n_eval": len(ev),
        "by_split_client": table,
        "by_material_client": {f"{mat}|{cl}": n for (mat, cl), n in sorted(
            Counter((str(r.material), str(r.client)) for r in tv.itertuples()).items())},
        "isolation": {
            "image_id_overlap_with_eval": len(set(tv["image_id"]) & eval_ids),
            "group_id_overlap_with_eval": len(set(tv["group_id"]) & eval_groups),
            "train_val_rows_without_client": int(tv["client"].isna().sum()),
            "eval_rows_with_client": int(ev["client"].notna().sum()),
            "clients_in_train_val": sorted({str(c) for c in tv["client"].dropna().unique()}),
        },
    }


def dry_build(snap, *, check_paths: bool = False) -> tuple[dict, list[dict]]:
    """빌더의 `build_pairs` 를 **그대로** 부른다. 디스크에는 아무것도 쓰지 않는다.

    첫 판은 조립 루프를 복제했다 — 조립이 `main()` 안에 있어 import 할 수 없었기 때문이다.
    지금은 빌더·계측·독립 검산이 같은 `assemble_record` 를 지난다(06번 §2-나).
    입력 규약 단언(재질·참여자 표기, eval 교차, 어노테이션 수 정합)도 `build_pairs` 가 한다.
    """
    from corpus.generate import make_pairs_pilot as P
    from corpus.generate.run_cycle_corpus import defect_names
    from corpus.rules import limits_loader
    from corpus.rules.skeleton_gen import load_defect_lexicon

    table = limits_loader.load_limits(str(P.LIMITS_CSV), pilot=True)
    m = snap.manifest
    tv = m[m["split"].isin(["train", "val"])]
    assert len(tv) + int((m["split"] == "eval").sum()) == len(m), "train·val·eval 밖의 split 값이 있다"
    assert str(tv["has_defect"].dtype) in ("bool", "boolean"), tv["has_defect"].dtype
    assert "geom_valid" in snap.annotations.columns, "geom_valid 열이 없다 — 없으면 전부 유효로 열린다"

    t0 = time.perf_counter()
    res = P.build_pairs(m, snap.annotations, table, defect_names(), load_defect_lexicon(),
                        check_paths=check_paths)
    t_build = time.perf_counter() - t0
    made = res.made

    code_counts = Counter((str(r["client"]), d["type"]) for r in made for d in r["skeleton"]["defects"])
    by = Counter((r["split"], str(r["client"]), bool(r["skeleton"]["defects"])) for r in made)
    noclause = Counter(str(r["client"]) for r in made
                       if r["skeleton"]["defects"] and not r["skeleton"]["clauses"])
    n_def = [len(r["skeleton"]["defects"]) for r in made if r["skeleton"]["defects"]]
    # 크기와 무관한 기준(전량 불허)이 걸리는 코드 — 파일럿 표에서는 균열이다. 코드 문자열을 박지 않고 표에서 읽는다(규약 1-8).
    np_codes = {str(r.defect_code) for r in table.rows if getattr(r.limit_rule, "value", r.limit_rule) == "none_permitted"}
    crack = [r for r in made if any(d["type"] in np_codes for d in r["skeleton"]["defects"])]
    summary = {
        "limits_csv": str(P.LIMITS_CSV.relative_to(REPO)).replace("\\", "/"),
        "materials_present": res.materials, "materials_covered_by_limits": sorted(res.covered),
        "n_input_train_val": res.n_input, "n_made": len(made),
        "n_discarded": len(res.discarded),
        "discard_reasons": dict(res.reasons),
        "pass_rate": round(len(made) / res.n_input, 6),
        "made_by_split_client": {f"{s}|{c}|{'결함' if d else '정상'}": n
                                 for (s, c, d), n in sorted(by.items())},
        "defect_pairs_without_clause_by_client": dict(noclause),
        "n_annotations_skipped_geom_invalid": res.n_geom_skipped,
        "n_annotations_skipped_in_made_pairs": sum(r.get("n_annotations_skipped_geom_invalid", 0) for r in made),
        "n_made_with_11_or_more_defects": sum(1 for x in n_def if x >= 11),
        "none_permitted_codes": sorted(np_codes),
        "n_made_with_none_permitted_code": len(crack),
        "n_made_with_none_permitted_code_by_client": dict(Counter(str(r["client"]) for r in crack)),
        "n_made_none_permitted_code_only": sum(1 for r in crack
                                               if {d["type"] for d in r["skeleton"]["defects"]} <= np_codes),
        "eval_isolation": res.isolation,
        "defects_per_defect_image": _pct(n_def),
        "annotations_by_client_code": {f"{c}|{code}": n for (c, code), n in sorted(code_counts.items())},
        "seconds": {"build_pairs": round(t_build, 1), "paths_checked": check_paths},
    }
    return summary, made


def token_lengths(made: list[dict], tokenizer_ids: list[str]) -> dict:
    """학습 타깃 길이. JSON 타깃은 학습기의 `build_target` 을 **그대로** 부른다."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # 내려받지 않는다 — 캐시에 없으면 실패
    from transformers import AutoTokenizer

    from vlm.coords import ImageGeom
    from vlm.pilot_vlm import build_target

    json_targets = [build_target(r, ImageGeom(orig_w=r["width_px"], orig_h=r["height_px"])) for r in made]
    texts = [r["target_text"] for r in made]
    out: dict = {"n": len(made), "chars": {"json": _pct([len(x) for x in json_targets]),
                                           "target_text": _pct([len(x) for x in texts])}}
    for tid in tokenizer_ids:
        tok = AutoTokenizer.from_pretrained(tid)

        lj = [len(x) for x in tok(json_targets, add_special_tokens=False)["input_ids"]]
        lt = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
        both = [a + b for a, b in zip(lj, lt)]
        is_def = [bool(r["skeleton"]["defects"]) for r in made]
        ent = {"json": _pct(lj), "target_text": _pct(lt), "json_plus_text": _pct(both),
               "json_defect_only": _pct([x for x, d in zip(lj, is_def) if d])}
        for limit in (256, 512, 1024, 2048):
            ent[f"over_{limit}"] = {"json": sum(x > limit for x in lj),
                                    "target_text": sum(x > limit for x in lt),
                                    "json_plus_text": sum(x > limit for x in both)}
        by_client = defaultdict(list)
        for r, x in zip(made, lj):
            by_client[str(r["client"])].append(x)
        ent["json_by_client"] = {c: _pct(v) for c, v in sorted(by_client.items())}
        out[tid] = ent
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="본실험 D4 페어 계획 계측 (읽기 전용)")
    ap.add_argument("--tokenizer", action="append", default=[],
                    help="고정 모델 id (반복 가능). 없으면 길이는 재지 않는다")
    ap.add_argument("--no-build", action="store_true", help="건수·격리만 본다")
    args = ap.parse_args(argv)

    from data.manifest_io import load_snapshot

    t0 = time.perf_counter()
    snap = load_snapshot(MANIFEST_DIR)                    # 계약 해시 4/4 를 검증하고 읽는다
    report: dict = {"manifest_dir": str(MANIFEST_DIR.relative_to(REPO)).replace("\\", "/"),
                    "seconds_load_snapshot": round(time.perf_counter() - t0, 1),
                    "counts": counts_and_isolation(snap.manifest)}
    if not args.no_build:
        report["dry_build"], made = dry_build(snap)
        if args.tokenizer:
            report["token_lengths"] = token_lengths(made, args.tokenizer)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
