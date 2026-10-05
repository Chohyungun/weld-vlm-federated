"""대조 추첨이 **맞추지 못한 구성**을 잰다 — 이미 낸 판을 다시 계산하지 않고.

대조 추첨은 순수 조합이다. 같은 씨앗·같은 칸·같은 칸별 제외 수를 주면 어느 장이 빠지는지가
그대로 재현된다. mAP 는 필요 없다. 그래서 이미 낸 산출물의 추첨을 되틀어 **그때 남은
결함 유병률·클래스 구성·정답 박스 수**를 사후에 잴 수 있다.

## 왜 필요한가

`neardup_rescore` 의 대조군이 맞춘 것은 `strata_key`(묶음 **대표** 클래스) × `tiles.csv` 출처의
**칸별 제외 장수** 둘뿐이다. `strata_key` 는 이미지별 실제 라벨이 아니라 묶음 구성원 라벨의
대표라서, 같은 칸 안에 정상과 결함이 섞여 있다(정상 대표층에 기공 1,466장). 따라서 칸별 장수를
맞춰도 **실제 결함 유병률은 맞춰지지 않는다.**

"맞췄다" 고 쓸 수 없다면 얼마나 안 맞았는지를 숫자로 내야 한다. 그것이 이 스크립트다.

## 무엇을 재현하는가

`control_draws` 의 소비 순서를 그대로 따른다 — `n_drop` 은 `pop.rows` 순서의 삽입 순서이고,
칸 안의 후보 순서도 같다. 추첨 하나에서 빠지는 **이미지 id 집합**이 같으면 구성 통계도 같다.

**기존 산출물을 고치지 않는다.** 이 값은 새 파일로 낸다.

## 같은 추첨인지 먼저 맞댄다

"그대로 재현한다" 는 같은 스냅샷 · 같은 행 순서 · 같은 제외 목록 · 같은 씨앗일 때만 선다. 그래서 되틀기 전에
재현 대상인 `neardup_rescore` 산출물(`--rescore-artifact`)을 읽고 넷을 맞댄다 — **칸 서명**(칸 키·칸 안 순서·
제외 수), 입력 **매니페스트의 sha256**, **제외 목록의 sha256**, **씨앗**. 하나라도 다르면 다른 추첨의 구성을
조용히 내는 것이라 멈춘다. 추첨 수는 재현 대상의 수 이하여야 한다(앞에서부터 같은 추첨이다).

`lo`·`hi` 는 추첨 분포의 2.5·97.5 백분위다. **신뢰구간이 아니다.**
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


from evaluation.eval_set import eval_rows, read_manifest
from evaluation.probes.source_probe import load_provenance
from evaluation.provenance import write_new_text
from evaluation.stats import BOOTSTRAP_SEED
from scripts.probe.neardup_rescore import (
    CONTROL_PERCENTILES,
    cells_signature,
    is_defect,
    load_exclusion,
    sha256_of,
)


class ReplayMismatch(SystemExit):
    """재현 대상과 다른 입력이다 — 되틀면 다른 추첨의 구성이 나온다."""


def check_against_rescore(artifact: dict, *, signature: str, manifest_sha256: str,
                          exclusion_sha256: str, rng_seed: int, draws: int) -> dict:
    """재현 대상 산출물과 넷을 맞댄다. 맞으면 맞댄 값을 돌려주고, 다르면 **전부 모아** 멈춘다."""
    ctl = artifact.get("composition_control") or {}
    if not ctl.get("computed"):
        raise ReplayMismatch("재현 대상에 대조 추첨이 없다(--control 0 으로 돌린 산출물이다)")
    seeds = sorted(artifact.get("per_seed") or {}, key=int)
    want_manifest = {artifact["per_seed"][n]["inputs"].get("manifest.csv") for n in seeds}
    reasons = []
    if ctl.get("cells_signature") != signature:
        reasons.append("칸 서명이 다르다")
    if want_manifest != {manifest_sha256}:
        reasons.append("매니페스트 sha256 이 다르다")
    if (artifact.get("exclusion_list") or {}).get("sha256") != exclusion_sha256:
        reasons.append("제외 목록 sha256 이 다르다")
    if ctl.get("rng_seed") != rng_seed:
        reasons.append("씨앗이 다르다")
    if not isinstance(ctl.get("n_draws"), int) or draws > ctl["n_draws"]:
        reasons.append(f"추첨 수 {draws} 가 재현 대상의 추첨 수 {ctl.get('n_draws')} 를 넘는다")
    if reasons:
        raise ReplayMismatch("재현 대상과 맞지 않는다: " + " · ".join(reasons))
    return {"cells_signature": signature, "manifest_sha256": manifest_sha256,
            "exclusion_sha256": exclusion_sha256, "rng_seed": rng_seed,
            "rescore_n_draws": ctl["n_draws"]}


def cells_of(rows, excluded: frozenset[str], prov: dict[str, str]):
    """`control_cells` 와 **같은 순서**로 칸을 만든다. 순서가 곧 난수 소비 순서다."""
    members: dict[tuple[str, str], list[str]] = {}
    n_drop: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r["strata_key"], prov[r["image_id"]])
        members.setdefault(key, []).append(r["image_id"])
        n_drop.setdefault(key, 0)
        if r["image_id"] in excluded:
            n_drop[key] += 1
    return members, n_drop


def facts_of(kept: set[str], defect_of: dict[str, bool], codes_of: dict[str, tuple[str, ...]]):
    """남은 모집단의 구성. 대조와 실제를 같은 함수로 잰다."""
    n = len(kept)
    n_def = sum(1 for i in kept if defect_of[i])
    cls = Counter(c for i in kept for c in codes_of[i])
    return {"n_kept": n, "n_defect": n_def,
            "prevalence": (n_def / n) if n else None,
            "class_images": dict(sorted(cls.items()))}


def spread(values) -> dict:
    """추첨 분포의 퍼짐. `lo`·`hi` 는 백분위이고 그 수준을 함께 싣는다 — 구간처럼 읽히지 않게."""
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()),
            "lo": float(np.percentile(a, CONTROL_PERCENTILES[0])),
            "hi": float(np.percentile(a, CONTROL_PERCENTILES[1])),
            "lo_hi_percentiles": list(CONTROL_PERCENTILES),
            "min": float(a.min()), "max": float(a.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", default="data/interim/manifest_v1", type=Path)
    ap.add_argument("--exclude", required=True, type=Path)
    ap.add_argument("--rescore-artifact", required=True, type=Path,
                    help="재현 대상 neardup_rescore 산출물 — 칸 서명·매니페스트·제외 목록·씨앗을 맞댄다")
    ap.add_argument("--draws", type=int, default=None,
                    help="되틀 추첨 수. 주지 않으면 재현 대상의 수. 그보다 크면 멈춘다")
    ap.add_argument("--rng-seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--dest", default="outputs/main_d/seed3set/neardup_control_replay_v1.json")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    snap = REPO / args.snapshot
    rows = eval_rows(read_manifest(snap))
    prov = load_provenance(snap / "tiles.csv")
    # 제외 목록은 재채점과 **같은 함수**로 읽는다 — 중복을 거부하고 파일 바이트의 sha256 을 낸다.
    excluded, list_meta = load_exclusion(args.exclude)
    artifact = json.loads(args.rescore_artifact.read_text(encoding="utf-8"))

    all_ids = {r["image_id"] for r in rows}
    outside = excluded - all_ids
    if outside:
        raise SystemExit(f"제외 목록에 평가 모집단 밖 ID {len(outside)}개")
    defect_of = {r["image_id"]: is_defect(r["has_defect"]) for r in rows}
    codes_of = {r["image_id"]: tuple(sorted({c for f in r["defect_types"].split("|")
                                             for c in f.split(";") if c}))
                for r in rows}

    actual = facts_of(all_ids - excluded, defect_of, codes_of)
    full = facts_of(all_ids, defect_of, codes_of)
    print(f"전량 {full['n_kept']} · 실제 제외 뒤 {actual['n_kept']} "
          f"(유병률 {actual['prevalence'] * 100:.2f}%)")

    members, n_drop = cells_of(rows, excluded, prov)
    n_draws = (artifact.get("composition_control") or {}).get("n_draws") if args.draws is None else args.draws
    matched = check_against_rescore(
        artifact, signature=cells_signature(members, n_drop),
        manifest_sha256=sha256_of(snap / "manifest.csv"), exclusion_sha256=list_meta["sha256"],
        rng_seed=args.rng_seed, draws=n_draws if isinstance(n_draws, int) else -1)
    rng = np.random.default_rng(args.rng_seed)
    draws = []
    for b in range(n_draws):
        dropped: set[str] = set()
        for key, k in n_drop.items():
            if k == 0:
                continue
            pool = members[key]
            for j in rng.choice(len(pool), size=k, replace=False):
                dropped.add(pool[j])
        draws.append(facts_of(all_ids - dropped, defect_of, codes_of))
        if (b + 1) % 20 == 0:
            print(f"  추첨 {b + 1}/{n_draws}")

    classes = sorted({c for d in draws for c in d["class_images"]})
    payload = {
        "kind": "neardup_control_replay",
        "note": ("이미 낸 대조 추첨을 난수만으로 되틀어 **맞추지 못한 구성**을 잰 것이다. "
                 "mAP 를 다시 계산하지 않았고 기존 산출물도 고치지 않았다"),
        "rng_seed": args.rng_seed, "n_draws": n_draws,
        "matched_against_rescore": {**matched, "artifact": args.rescore_artifact.name},
        "matched_on": ["strata_key(묶음 대표 클래스)", "tiles.csv 출처"],
        "matched_quantity": "칸별 제외 장수",
        "n_cells": len(members),
        "exclusion_list": list_meta,
        "full_population": full,
        "actual_after_exclusion": actual,
        "control": {
            "prevalence": spread([d["prevalence"] for d in draws]),
            "n_kept": spread([d["n_kept"] for d in draws]),
            "n_defect": spread([d["n_defect"] for d in draws]),
            "class_images": {c: spread([d["class_images"].get(c, 0) for d in draws])
                             for c in classes},
        },
        "residual": {
            "prevalence_actual_minus_control_mean": (
                actual["prevalence"] - float(np.mean([d["prevalence"] for d in draws]))),
            "n_defect_actual_minus_control_mean": (
                actual["n_defect"] - float(np.mean([d["n_defect"] for d in draws]))),
            "reading": ("0 에서 멀수록 대조가 실제 제외의 구성을 못 맞췄다는 뜻이다. "
                        "맞춘 축은 칸별 장수뿐이고 실제 라벨은 맞추지 않았다"),
        },
    }
    out = REPO / args.dest
    write_new_text(out, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"\n{out.relative_to(REPO)}")
    c = payload["control"]["prevalence"]
    print(f"대조 유병률 {c['mean'] * 100:.2f}% [{c['lo'] * 100:.2f}, {c['hi'] * 100:.2f}] "
          f"· 실제 {actual['prevalence'] * 100:.2f}% "
          f"· 차 {payload['residual']['prevalence_actual_minus_control_mean'] * 100:+.2f}%p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
