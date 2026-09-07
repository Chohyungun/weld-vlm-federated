"""첫 산출물 감사 — **채점기와 독립인 경로로** 핵심 지표 3개를 재계산해 대조한다.

    uv run python scripts/probe/verify_first_scoring.py \
        --snapshot data/interim/manifest_v1 --out outputs/main_d/seed1 --k 64

본실험 첫 수치다. "채점기가 그렇게 냈다"는 것으로는 부족하고, 같은 정의를 **다른 코드로**
구현해 소수 6자리까지 같은 수가 나오는지를 본다(지시서 과제 2).

## 무엇이 독립인가

| 축 | 채점기 경로 | 여기 |
|---|---|---|
| 평가셋 | `evaluation.eval_set.eval_rows(read_manifest(...))` | `manifest.csv` 를 직접 읽어 `split == "eval"` |
| 정답 | `evaluation.eval_set.read_gold` | `annotations.csv` 를 직접 읽어 image_id → iso_code 집합 |
| 예측 | `PredictionRecord` (pydantic) 되읽기 | `json.loads` 로 `defects[].iso_code` 만 |
| Macro-F1·놓침 | `evaluation.metrics.detection.score_detection` | 이 파일의 `_macro_f1`·`_recall`(순수 파이썬 카운터) |
| 층 배정 | `evaluation.strata.bins_for` → A 의 `stratum_of`(런타임 분위) | A 가 실체화한 `id_strata_k{K}.csv` 를 읽는다 |
| 층화 평균 | `evaluation.strata.stratified_score`(numpy) | 이 파일의 순수 파이썬 평균 |

`evaluation.*` 를 하나도 import 하지 않는다 — 그것이 이 스크립트의 존재 이유다. 두 경로가
같은 상수를 공유하면 대조가 자기 자신과의 대조가 된다.

**층 배정의 독립성이 특히 중요하다.** 채점기는 실행 시점에 동결본 train+val 의 분위를
다시 계산하고, 여기서는 A 가 그 계산으로 미리 써 둔 표(`id_strata_k64.csv`, 62,308행)를
읽는다. 둘이 어긋나면 층화 수치 전체가 다른 축 위에 서 있는 것이다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# 채점 클래스 4종의 ISO 코드. **채점기 상수를 import 하지 않는다** — 값이 갈리면 대조가
# 그 사실을 드러내야 한다(같은 상수를 공유하면 대조가 자기 자신과의 대조가 된다).
CLASSES: tuple[str, ...] = ("100", "2011", "401", "301")
TAGS: tuple[str, ...] = ("sep_local_C1", "sep_local_C2", "sep_local_C3",
                         "sep_central", "sep_fed")
TOL = 1e-6
"""소수 6자리 대조."""


# --------------------------------------------------------------------------------------
# 입력 — 전부 직접 읽는다
# --------------------------------------------------------------------------------------

def verify_snapshot(snapshot: Path) -> dict:
    """`SNAPSHOT.sha256` 을 **직접** 대조한다(감사 11 의 봉인 — 누락 방향 D-9).

    채점기는 `evaluation.eval_set.ensure_verified` 를 지나 파일을 열고, 여기서는 같은
    해시 파일을 자체 구현으로 다시 센다. 어느 한쪽만 통과하면 그 사실이 드러나야 한다.
    """
    lines = (snapshot / "SNAPSHOT.sha256").read_text(encoding="utf-8").splitlines()
    declared = {}
    digest = ""
    for line in lines:
        if line.startswith("# snapshot_digest"):
            digest = line.split()[-1]
        elif line.strip():
            h, name = line.split()
            declared[name] = h
    files = {}
    for name, want in declared.items():
        h = hashlib.sha256()
        with (snapshot / name).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        files[name] = {"declared": want, "actual": h.hexdigest(),
                       "match": h.hexdigest() == want}
    return {"snapshot_digest": digest, "files": files,
            "all_match": all(v["match"] for v in files.values())}


def read_eval_ids(snapshot: Path) -> list[str]:
    csv.field_size_limit(1 << 30)
    with (snapshot / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        return [r["image_id"] for r in csv.DictReader(fh) if r["split"] == "eval"]


def read_gold(snapshot: Path, eval_ids: set[str]) -> dict[str, set[str]]:
    gold: dict[str, set[str]] = {i: set() for i in eval_ids}
    with (snapshot / "annotations.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["image_id"] in gold:
                gold[r["image_id"]].add(r["iso_code"])
    return gold


def read_pred(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["image_id"]] = {d["iso_code"] for d in row["defects"]}
    return out


def read_bins(snapshot: Path, k: int, eval_ids: set[str]) -> dict[str, int]:
    """A 가 실체화한 층 표. **채점기의 런타임 분위 계산과 독립인 경로다.**"""
    p = snapshot / f"id_strata_k{k}.csv"
    if not p.exists():
        raise SystemExit(f"층 표 없음: {p} — A 의 `data.id_strata.materialize` 산출물")
    with p.open(encoding="utf-8", newline="") as fh:
        return {r["image_id"]: int(r["stratum"]) for r in csv.DictReader(fh)
                if r["image_id"] in eval_ids}


# --------------------------------------------------------------------------------------
# 지표 — 순수 파이썬
# --------------------------------------------------------------------------------------

def _macro_f1(pred: dict[str, set[str]], gold: dict[str, set[str]],
              ids: list[str]) -> float | None:
    """GT 표본이 있는 클래스만 평균한다. 하나도 없으면 None(구간이 채점 불가)."""
    f1s = []
    for code in CLASSES:
        tp = fp = fn = 0
        for i in ids:
            in_p = code in pred.get(i, ())
            in_g = code in gold.get(i, ())
            tp += in_p and in_g
            fp += in_p and not in_g
            fn += not in_p and in_g
        if tp + fn == 0:                       # support 0 — 평균 대상 아님
            continue
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return (sum(f1s) / len(f1s)) if f1s else None


def _miss_rate(pred: dict[str, set[str]], gold: dict[str, set[str]],
               ids: list[str]) -> float:
    """1 − (이미지×클래스) 양성쌍 micro recall."""
    pos = hits = 0
    for i in ids:
        for code in CLASSES:
            if code in gold.get(i, ()):
                pos += 1
                hits += code in pred.get(i, ())
    return 1.0 - (hits / pos if pos else 0.0)


def _majority(ids: list[str], gold: dict[str, set[str]]) -> frozenset[str]:
    """구간 최빈 정답 코드 집합. 동률이면 정렬 역순 튜플이 큰 쪽 — 채점기와 같은 규칙."""
    counts: dict[frozenset[str], int] = defaultdict(int)
    for i in ids:
        counts[frozenset(gold.get(i, ())) & frozenset(CLASSES)] += 1
    return max(counts.items(),
               key=lambda kv: (kv[1], tuple(sorted(kv[0], reverse=True))))[0]


def stratified(pred: dict[str, set[str]], gold: dict[str, set[str]],
               bins: dict[str, int], ids: list[str]) -> dict:
    by_bin: dict[int, list[str]] = defaultdict(list)
    for i in ids:
        by_bin[bins[i]].append(i)
    rows = []
    for b in sorted(by_bin):
        bin_ids = by_bin[b]
        distinct = {frozenset(gold.get(i, ())) & frozenset(CLASSES) for i in bin_ids}
        maj = _majority(bin_ids, gold)
        base_pred = {i: set(maj) for i in bin_ids}
        m = _macro_f1(pred, gold, bin_ids)
        base = _macro_f1(base_pred, gold, bin_ids)
        rows.append({"bin": b, "n": len(bin_ids), "pure": len(distinct) == 1,
                     "macro_f1": m, "baseline": base,
                     "lift": None if (m is None or base is None) else m - base})
    scored = [r for r in rows if r["macro_f1"] is not None and r["baseline"] is not None]
    impure = [r for r in scored if not r["pure"]]

    def mean(rs, key):
        return (sum(r[key] for r in rs) / len(rs)) if rs else 0.0

    return {
        "n_strata": len(rows),
        "n_strata_scored": len(scored),
        "n_pure_strata": sum(1 for r in rows if r["pure"]),
        "n_strata_impure_scored": len(impure),
        "stratified_macro_f1": mean(scored, "macro_f1"),
        "stratified_lift": mean(scored, "lift"),
        "shortcut_baseline_macro_f1": mean(scored, "baseline"),
        "stratified_macro_f1_impure": mean(impure, "macro_f1"),
        "stratified_lift_impure": mean(impure, "lift"),
    }


# --------------------------------------------------------------------------------------
# 대조
# --------------------------------------------------------------------------------------

def _cmp(name: str, mine: float, theirs: float | None, rows: list[dict]) -> bool:
    ok = theirs is not None and abs(mine - theirs) <= TOL
    rows.append({"key": name, "independent": round(mine, 9),
                 "scorer": None if theirs is None else round(theirs, 9),
                 "delta": None if theirs is None else round(mine - theirs, 12),
                 "match": ok})
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default="data/interim/manifest_v1")
    ap.add_argument("--out", default="outputs/main_d/seed1")
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--k", type=int, default=64)
    a = ap.parse_args()

    snapshot, out = Path(a.snapshot), Path(a.out)
    scored_path = out / "score_cells_v1.json"
    if not scored_path.exists():
        raise SystemExit(f"채점 산출물 없음: {scored_path} — 먼저 score 를 돌린다")
    payload = json.loads(scored_path.read_text(encoding="utf-8"))

    snap = verify_snapshot(snapshot)
    print(f"스냅샷 자체 대조: digest {snap['snapshot_digest'][:16]}… · "
          f"파일 {len(snap['files'])}종 {'전부 일치' if snap['all_match'] else '불일치!'}")

    ids = read_eval_ids(snapshot)
    eval_ids = set(ids)
    ids = sorted(eval_ids)                       # 채점기와 같은 정렬 기준
    gold = read_gold(snapshot, eval_ids)
    bins = read_bins(snapshot, a.k, eval_ids)
    n_normal = sum(1 for i in ids if not gold[i])
    print(f"독립 경로: 평가셋 {len(ids):,}장 (정답 코드 없는 이미지 {n_normal:,}) · "
          f"층 표 id_strata_k{a.k}.csv {len(bins):,}행")
    if len(bins) != len(ids):
        raise SystemExit(f"층 표가 평가셋을 전부 덮지 않는다: {len(bins)} != {len(ids)}")

    rows: list[dict] = []
    ok = True
    ok &= _cmp("n_eval", float(len(ids)), float(payload.get("n_eval", -1)), rows)
    if not snap["all_match"]:
        rows.append({"key": "snapshot_sha256", "independent": None, "scorer": None,
                     "delta": None, "match": False})
        ok = False

    strata_scorer = payload.get("stratified", {}).get("by_k", {}).get(str(a.k), {})
    per_cell: dict[str, dict] = {}
    for tag in TAGS:
        src = out / f"{tag}_s{a.seed}.jsonl"
        if not src.exists():
            print(f"[{tag}] 레코드 없음 — 건너뜀")
            continue
        pred = read_pred(src)
        missing = eval_ids - set(pred)
        if missing:
            raise SystemExit(f"{tag}: 예측 결측 {len(missing)}장 — 모집단이 다르다")
        m = payload["metrics"].get(tag, {})
        s = strata_scorer.get(tag, {})

        cell_rows: list[dict] = []
        f1 = _macro_f1(pred, gold, ids)
        miss = _miss_rate(pred, gold, ids)
        st = stratified(pred, gold, bins, ids)
        good = _cmp(f"{tag}.macro_f1", f1 or 0.0, m.get("macro_f1"), cell_rows)
        good &= _cmp(f"{tag}.miss_rate", miss, m.get("miss_rate"), cell_rows)
        for key in ("stratified_macro_f1", "stratified_lift",
                    "stratified_macro_f1_impure", "stratified_lift_impure"):
            good &= _cmp(f"{tag}.{key}", st[key], s.get(key), cell_rows)
        for key in ("n_strata", "n_strata_scored", "n_pure_strata",
                    "n_strata_impure_scored"):
            good &= _cmp(f"{tag}.{key}", float(st[key]),
                         None if s.get(key) is None else float(s[key]), cell_rows)
        rows.extend(cell_rows)
        ok &= good
        per_cell[tag] = {"macro_f1": f1, "miss_rate": miss, **st}
        print(f"[{tag}] macroF1 {f1:.6f} · 놓침 {miss:.6f} · "
              f"층화 {st['stratified_macro_f1']:.6f} · lift {st['stratified_lift']:+.6f} "
              f"→ {'일치' if good else '불일치'}")

    # 지름길 규칙 — 구간 최빈 라벨 예측기의 lift 는 정의상 정확히 0 이어야 한다.
    shortcut_pred = {}
    by_bin: dict[int, list[str]] = defaultdict(list)
    for i in ids:
        by_bin[bins[i]].append(i)
    for bin_ids in by_bin.values():
        maj = _majority(bin_ids, gold)
        for i in bin_ids:
            shortcut_pred[i] = set(maj)
    sc = stratified(shortcut_pred, gold, bins, ids)
    sc_scorer = strata_scorer.get("__shortcut__", {})
    ok &= _cmp("__shortcut__.stratified_lift", sc["stratified_lift"],
               sc_scorer.get("stratified_lift"), rows)
    ok &= _cmp("__shortcut__.global_macro_f1", _macro_f1(shortcut_pred, gold, ids) or 0.0,
               sc_scorer.get("global_macro_f1"), rows)
    exact_zero = sc["stratified_lift"] == 0.0
    print(f"[지름길] 전역 {_macro_f1(shortcut_pred, gold, ids):.6f} · "
          f"층화 {sc['stratified_macro_f1']:.6f} · lift {sc['stratified_lift']:+.9f} "
          f"({'정확히 0' if exact_zero else '0 이 아니다 — 층 정의 고장'})")

    bad = [r for r in rows if not r["match"]]
    report = {
        "independent_of": "evaluation.* 전부 (정답·예측·지표·층 배정 모두 자체 구현)",
        "snapshot": str(snapshot), "scored": str(scored_path), "k": a.k,
        "n_eval": len(ids), "n_images_without_gold_code": n_normal,
        "strata_source": f"{snapshot}/id_strata_k{a.k}.csv (A 실체화본)",
        "snapshot_verification": snap,
        "classes": list(CLASSES),
        "tolerance": TOL,
        "all_match": not bad,
        "shortcut_lift_exactly_zero": exact_zero,
        "n_compared": len(rows), "n_mismatch": len(bad),
        "mismatches": bad,
        "comparisons": rows,
        "independent_values": per_cell,
    }
    dest = out / "verify_first_scoring_v1.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n대조 {len(rows)}건 · 불일치 {len(bad)}건 → {dest}")
    for r in bad[:10]:
        print(f"  [불일치] {r['key']}: 독립 {r['independent']} vs 채점기 {r['scorer']}")
    return 0 if (not bad and exact_zero) else 1


if __name__ == "__main__":
    sys.exit(main())
