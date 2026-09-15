"""회복률 CI — 시드 3세트 × 칸 5 의 짝지은 묶음 부트스트랩 (36번 미니스펙, 총괄 게이트 09-16).

    python scripts/probe/recovery_bootstrap.py --root outputs/main_d --seeds 1,2,3 \\
        --artifact score_cells_v2.json --dest outputs/main_d/seed3set

순서(스펙 §2): (칸, 시드) 15개마다 `COCOeval.evaluate()` 한 번 → 캐시(중간물) → **항등 검사**
(중복도 전부 1 = 채점기 `map_50`·`map_50_95` 비트 일치, 아니면 멈춤) → 묶음 추첨 2,000회를
15개 전부에 같이 써서 `R_s`·`D_s`·`R̄`·칸별 mAP 의 백분위 CI → 규칙 ②③④ 판정.

**채점 기준을 바꾸지 않는다.** 시드별 채점 JSON 은 읽기만 하고, 캐시(`coco_evalimgs_*.npz`)는
중간물이라 봉인하지 않는다. 산출물에는 채점기 코드 지문과 입력 레코드 해시를 싣는다.
GPU 무접촉.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.adapters import read_records
from evaluation.params import ScoringParams
from evaluation.prereg import (
    RECOVERY_CI_DENOMINATOR_MUST_EXCLUDE_ZERO,
    RECOVERY_CI_MAX_HALF_WIDTH,
    RECOVERY_CI_MAX_UNDEFINED_FRACTION,
    RECOVERY_CI_REGISTRATION,
)
from evaluation.provenance import (
    hash_files,
    relpath,
    scorer_code_digest,
    stable_digest,
)
from evaluation.recovery_ci import (
    EvalCache,
    build_cache,
    group_weights,
    identity_check,
    recovery_from,
    weighted_map,
)
from evaluation.stats import BOOTSTRAP_N, BOOTSTRAP_SEED, MAX_RECOVERY_HALF_WIDTH

DET_TAGS = ("sep_central", "sep_local_C1", "sep_local_C2", "sep_local_C3", "sep_fed")
LOCALS = ("sep_local_C1", "sep_local_C2", "sep_local_C3")
ALPHA = 0.05


def _percentile_ci(draws: np.ndarray) -> dict:
    ok = np.isfinite(draws)
    kept = draws[ok]
    if kept.size == 0:
        return {"ci_lo": None, "ci_hi": None, "half_width": None,
                "n_defined": 0, "n_undefined": int((~ok).sum())}
    lo, hi = np.percentile(kept, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"ci_lo": float(lo), "ci_hi": float(hi), "half_width": float((hi - lo) / 2),
            "n_defined": int(kept.size), "n_undefined": int((~ok).sum())}


def _load_seed(root: Path, n: int, artifact: str) -> tuple[dict, ScoringParams]:
    art_path = root / f"seed{n}" / artifact
    art = json.loads(art_path.read_text(encoding="utf-8"))
    p = art["params"]
    params = ScoringParams(
        snapshot=Path(p["snapshot"]), pilot=Path(p["pilot"]), out=root / f"seed{n}",
        seed=int(p["seed"]), profile=p["profile"],
    )
    return art, params


def _caches_for_seed(params: ScoringParams, art: dict, cache_dir: Path,
                     log=print) -> tuple[dict[str, EvalCache], dict, dict]:
    """(칸 → 캐시, 항등 검사 결과, 입력 레코드 해시). 항등 실패는 예외로 올라간다."""
    from scripts.probe.score_cells import load_population, raw_record_path

    pop = load_population(params)
    gold = {i: pop.gold_boxes.get(i, []) for i in pop.eval_ids}
    caches, checks, inputs = {}, {}, []
    for tag in DET_TAGS:
        src = raw_record_path(params, tag)
        inputs.append(src)
        recs = read_records(src.read_text(encoding="utf-8").splitlines())
        pred = {r.image_id: [(d.iso_code, tuple(d.bbox_px), d.score or 0.0)
                             for d in r.defects if d.bbox_px] for r in recs}
        t0 = time.perf_counter()
        cache = build_cache(pred, gold, pop.classes)
        if cache is None:
            raise SystemExit(f"{tag}: GT 박스 0 — 캐시를 만들 수 없다")
        exp = art["threshold_independent"]["per_tag"][tag]
        checks[tag] = identity_check(cache, exp["map_50"], exp["map_50_95"])
        checks[tag]["build_seconds"] = round(time.perf_counter() - t0, 1)
        np.savez_compressed(
            cache_dir / f"coco_evalimgs_{tag}_s{params.seed}.npz",
            note="중간물 — 봉인하지 않는다. 재실행하면 다시 만든다 (36번 §4)",
            **{f"c{k}_{name}": getattr(c, name) for k, c in enumerate(cache.per_class)
               for name in ("scores", "dt_match", "dt_ignore", "start", "length", "npig")},
        )
        caches[tag] = cache
        log(f"  [{tag}] 캐시 {checks[tag]['build_seconds']}s · 항등 {checks[tag]['passed']} "
            f"· map_50 {checks[tag]['map_50']:.6f} (차 {checks[tag]['abs_diff_map_50']:.1e})")
    groups = {r["image_id"]: r["group_id"] for r in pop.rows}
    ids = next(iter(caches.values())).image_ids
    gnames = sorted({groups[i] for i in ids})
    gidx = {g: k for k, g in enumerate(gnames)}
    group_of_image = np.asarray([gidx[groups[i]] for i in ids], dtype=np.int64)
    meta = {"group_of_image": group_of_image, "n_groups": len(gnames), "image_ids": ids,
            "input_records": hash_files(inputs)}
    return caches, checks, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/main_d")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--artifact", default="score_cells_v2.json",
                    help="시드별 채점 산출물 파일명. 항등 검사의 기준값이 여기서 온다")
    ap.add_argument("--dest", default="outputs/main_d/seed3set")
    ap.add_argument("--n-resamples", type=int, default=BOOTSTRAP_N)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--identity-only", action="store_true",
                    help="항등 검사(T1)까지만 돌리고 CI 는 내지 않는다 — v1 검사용")
    args = ap.parse_args()

    if RECOVERY_CI_MAX_HALF_WIDTH != MAX_RECOVERY_HALF_WIDTH:
        raise SystemExit("prereg 와 stats 의 반폭 상한이 다르다 — 등록값을 먼저 맞춰라")

    code_start = scorer_code_digest()
    root, dest = Path(args.root), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    t_all = time.perf_counter()

    caches: dict[int, dict[str, EvalCache]] = {}
    checks: dict[int, dict] = {}
    inputs: dict[str, dict] = {}
    artifact_hash: dict[str, dict] = {}
    group_of_image = None
    n_groups = 0
    for n in seeds:
        art, params = _load_seed(root, n, args.artifact)
        art_path = root / f"seed{n}" / args.artifact
        artifact_hash[str(n)] = {"path": relpath(art_path), **hash_files([art_path]),
                                 "scorer_code_in_artifact": (art.get("scorer_code") or {}).get("combined")}
        print(f"[시드 {n}] {args.artifact} · 채점기 지문 {(art.get('scorer_code') or {}).get('combined', '없음')[:16]}")
        c, chk, meta = _caches_for_seed(params, art, root / f"seed{n}")
        caches[n], checks[str(n)] = c, chk
        inputs[str(n)] = meta["input_records"]
        if group_of_image is None:
            group_of_image, n_groups, image_ids = meta["group_of_image"], meta["n_groups"], meta["image_ids"]
        elif meta["image_ids"] != image_ids:
            raise SystemExit(f"시드 {n} 의 이미지 순서가 다르다 — 같은 평가셋이 아니다")
    n_checks = sum(len(v) for v in checks.values())
    print(f"항등 검사 {n_checks}/{n_checks} 통과 · 묶음 {n_groups:,} · 이미지 {len(image_ids):,}")
    if args.identity_only:
        out = dest / f"identity_check_{Path(args.artifact).stem}.json"
        out.write_text(json.dumps({"artifact": args.artifact, "identity_check": checks,
                                   "inputs": {"artifacts": artifact_hash, "records_by_seed": inputs},
                                   "scorer_code": code_start}, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"항등 검사만 — 저장: {out}")
        return 0

    # ---- 짝지은 재표집 — 한 번의 추첨을 15개에 같이 쓴다
    rng = np.random.default_rng(args.seed)
    n_res = args.n_resamples
    draws_map = {n: {t: np.empty(n_res) for t in DET_TAGS} for n in seeds}
    draws_R = {n: np.empty(n_res) for n in seeds}
    draws_D = {n: np.empty(n_res) for n in seeds}
    draws_Rbar = np.empty(n_res)
    t_bs = time.perf_counter()
    for b in range(n_res):
        w = group_weights(group_of_image, n_groups, rng)
        rs = []
        for n in seeds:
            m = {t: weighted_map(caches[n][t], w)[0] for t in DET_TAGS}
            for t in DET_TAGS:
                draws_map[n][t][b] = m[t]
            r, d = recovery_from(m["sep_central"], m["sep_fed"], [m[t] for t in LOCALS])
            draws_R[n][b], draws_D[n][b] = r, d
            rs.append(r)
        draws_Rbar[b] = float(np.mean(rs)) if all(np.isfinite(rs)) else float("nan")
        if (b + 1) % 200 == 0:
            el = time.perf_counter() - t_bs
            print(f"  재표집 {b + 1}/{n_res} · {el:.0f}s 경과 · 예상 잔여 {el / (b + 1) * (n_res - b - 1):.0f}s")

    # ---- 점추정(중복도 1)과 CI
    ones = np.ones(len(image_ids), dtype=np.int64)
    point_map = {n: {t: weighted_map(caches[n][t], ones)[0] for t in DET_TAGS} for n in seeds}
    point_R, point_D = {}, {}
    for n in seeds:
        r, d = recovery_from(point_map[n]["sep_central"], point_map[n]["sep_fed"],
                             [point_map[n][t] for t in LOCALS])
        point_R[n], point_D[n] = r, d
    point_Rbar = float(np.mean([point_R[n] for n in seeds]))

    ci_Rbar = _percentile_ci(draws_Rbar)
    ci_R = {str(n): {"point": point_R[n], **_percentile_ci(draws_R[n])} for n in seeds}
    ci_D = {str(n): {"point": point_D[n], **_percentile_ci(draws_D[n])} for n in seeds}
    ci_map = {str(n): {t: {"point": point_map[n][t], **_percentile_ci(draws_map[n][t])}
                       for t in DET_TAGS} for n in seeds}

    # ---- 규칙 ②③④ (evaluation/prereg.py 에 등록, 값 보기 전)
    undefined_frac = ci_Rbar["n_undefined"] / n_res
    rule2 = (ci_Rbar["half_width"] is not None
             and ci_Rbar["half_width"] <= RECOVERY_CI_MAX_HALF_WIDTH)
    rule3 = all(v["ci_lo"] is not None and v["ci_lo"] > 0 for v in ci_D.values())
    rule4 = undefined_frac <= RECOVERY_CI_MAX_UNDEFINED_FRACTION
    passed = rule2 and (rule3 or not RECOVERY_CI_DENOMINATOR_MUST_EXCLUDE_ZERO) and rule4
    verdict = {
        "2_half_width": {"value": ci_Rbar["half_width"], "max": RECOVERY_CI_MAX_HALF_WIDTH, "pass": rule2},
        "3_denominator_ci_excludes_zero": {
            "by_seed": {k: (v["ci_lo"] is not None and v["ci_lo"] > 0) for k, v in ci_D.items()},
            "required": RECOVERY_CI_DENOMINATOR_MUST_EXCLUDE_ZERO, "pass": rule3},
        "4_undefined_fraction": {"value": undefined_frac, "max": RECOVERY_CI_MAX_UNDEFINED_FRACTION, "pass": rule4},
        "ci_pass": passed,
        "reading": ("CI 판정만이다. 게재 가부(recovery_reportable)는 집계기가 트립와이어 ①과 "
                    "함께 낸다. 대표 채택은 총괄 판정"),
    }
    print(f"[R̄] {point_Rbar * 100:+.1f}% · 95% CI [{(ci_Rbar['ci_lo'] or float('nan')) * 100:+.1f}%, "
          f"{(ci_Rbar['ci_hi'] or float('nan')) * 100:+.1f}%] · 반폭 {(ci_Rbar['half_width'] or float('nan')):.3f} "
          f"(≤ {RECOVERY_CI_MAX_HALF_WIDTH} → {'통과' if rule2 else '불합격'}) · 미정의 {ci_Rbar['n_undefined']}/{n_res}")
    for n in seeds:
        print(f"[시드 {n}] R {point_R[n] * 100:+.1f}% CI [{ci_R[str(n)]['ci_lo'] * 100:+.1f}%, {ci_R[str(n)]['ci_hi'] * 100:+.1f}%] "
              f"· D {point_D[n]:.4f} CI [{ci_D[str(n)]['ci_lo']:.4f}, {ci_D[str(n)]['ci_hi']:.4f}]")
    print(f"규칙 ② {rule2} · ③ {rule3} · ④ {rule4} → CI 판정 {'통과' if passed else '불합격'}")

    code_end = scorer_code_digest()
    payload = {
        "spec": "36번 미니스펙 · 총괄 게이트 2026-09-16 00:35 (조건부 통과)",
        "registration": RECOVERY_CI_REGISTRATION,
        "statistic": {
            "axis": "map_50 (export 하한)",
            "judged": "R̄ = mean_s (fed_s − localmean_s) / (central_s − localmean_s)",
            "localmean": "로컬 세 칸의 비가중 산술평균",
            "resample_unit": "group_id", "n_groups": n_groups, "n_images": len(image_ids),
            "n_resamples": n_res, "rng_seed": args.seed, "alpha": ALPHA, "interval": "백분위",
            "paired": "한 번의 묶음 추첨을 칸 5 × 시드 3 전부에 같이 썼다",
            "undefined_rule": "D_s ≤ 0 이면 R_s 는 nan — 버리고 센다",
        },
        "seed_caveat": ("시드 3개는 분산 추정이 아니다. 이 CI 는 평가셋 묶음의 재표집 분산이지 "
                        "학습 시드의 분산이 아니며, R̄ 의 CI 는 세 시드를 고정한 채 평가셋만 흔든 것이다"),
        "identity_check": checks,
        "point": {"R_bar": point_Rbar, "R_by_seed": {str(n): point_R[n] for n in seeds},
                  "D_by_seed": {str(n): point_D[n] for n in seeds},
                  "map_50_by_seed": {str(n): point_map[n] for n in seeds}},
        "ci": {"R_bar": ci_Rbar, "R_by_seed": ci_R, "D_by_seed": ci_D, "map_50_by_seed": ci_map},
        "verdict": verdict,
        "inputs": {"artifacts": artifact_hash, "records_by_seed": inputs,
                   "artifact_name": args.artifact},
        "scorer_code": stable_digest(code_start, code_end),
        "cache": {"path_pattern": "outputs/main_d/seed{n}/coco_evalimgs_{tag}_s{seed}.npz",
                  "role": "중간물 — 봉인하지 않는다"},
        "runtime_seconds": round(time.perf_counter() - t_all, 1),
    }
    out = dest / "recovery_ci_v1.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {out} · {payload['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
