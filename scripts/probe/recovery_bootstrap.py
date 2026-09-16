"""회복률 CI — 시드 3세트 × 칸 5 의 짝지은 묶음 부트스트랩 (36번 미니스펙, 총괄 게이트 09-16).

    python scripts/probe/recovery_bootstrap.py --root outputs/main_d --seeds 1,2,3 \\
        --artifact score_cells_v3.json --dest outputs/main_d/seed3set

출력 이름은 입력의 판을 따른다(`score_cells_v3.json` → `recovery_ci_v3.json`). 같은 이름이 이미 있으면
계산 전에 멈춘다 — 옛 판은 보존한다(C 42번 I-2).

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
    RECOVERY_CI_ALPHA,
    RECOVERY_CI_DENOMINATOR_MUST_EXCLUDE_ZERO,
    RECOVERY_CI_INTERVAL,
    RECOVERY_CI_MAX_HALF_WIDTH,
    RECOVERY_CI_MAX_UNDEFINED_FRACTION,
    RECOVERY_CI_REGISTRATION,
)
from evaluation.provenance import (
    hash_files,
    scorer_code_digest,
    stable_digest,
    versioned_output,
    write_new_text,
)
from evaluation.recovery_ci import (
    ArtifactNameMismatch,
    EvalCache,
    build_cache,
    identity_check,
    make_binding,
    paired_bootstrap,
    percentile_ci,
    read_artifact,
    recovery_from,
    weighted_map,
)
from evaluation.stats import BOOTSTRAP_N, BOOTSTRAP_SEED, MAX_RECOVERY_HALF_WIDTH

DET_TAGS = ("sep_central", "sep_local_C1", "sep_local_C2", "sep_local_C3", "sep_fed")
LOCALS = ("sep_local_C1", "sep_local_C2", "sep_local_C3")


def _percentile_ci(draws: np.ndarray) -> dict:
    """신뢰수준은 등록 상수(`RECOVERY_CI_ALPHA`) — 집계기 재판정이 같은 값을 읽는다(C 42번 I-1)."""
    return percentile_ci(draws, RECOVERY_CI_ALPHA)


def _load_seed(root: Path, n: int, artifact: str) -> tuple[dict, ScoringParams, dict]:
    """산출물을 **한 번만** 읽는다 — 결속 해시와 파싱이 같은 바이트에서 나온다."""
    try:
        art, entry = read_artifact(root / f"seed{n}" / artifact)
    except ArtifactNameMismatch as e:        # C 42번 §9-6 n-5 — 계산 전에 멈춘다
        raise SystemExit(f"시드 {n}: {e}") from None
    p = art["params"]
    params = ScoringParams(
        snapshot=Path(p["snapshot"]), pilot=Path(p["pilot"]), out=root / f"seed{n}",
        seed=int(p["seed"]), profile=p["profile"],
    )
    return art, params, entry


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

    root, dest = Path(args.root), Path(args.dest)
    # **옛 판을 덮지 않는다**(C 42번 I-2). 출력 이름은 입력 산출물의 판에서 뽑는다 —
    # score_cells_v3.json → recovery_ci_v3.json. 30분 계산 전에 확인하고, 쓸 때 배타 생성으로 한 번 더 막는다.
    identity_out = dest / f"identity_check_{Path(args.artifact).stem}.json"
    try:
        out = (None if args.identity_only
               else versioned_output(dest, "recovery_ci", args.artifact))
    except (ValueError, FileExistsError) as e:
        raise SystemExit(f"{e} (dest={dest})") from None
    if args.identity_only and identity_out.exists():
        raise SystemExit(f"{identity_out.name} 이 이미 있다 — 덮지 않는다 (dest={dest})")

    code_start = scorer_code_digest()
    dest.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    t_all = time.perf_counter()

    caches: dict[int, dict[str, EvalCache]] = {}
    checks: dict[int, dict] = {}
    inputs: dict[str, dict] = {}
    artifact_hash: dict[str, dict] = {}
    group_of_image = None
    n_groups = 0
    entries: dict[int, dict] = {}
    for n in seeds:
        art, params, entry = _load_seed(root, n, args.artifact)
        entries[n] = entry
        artifact_hash[str(n)] = entry
        print(f"[시드 {n}] {args.artifact} · 채점기 지문 {str(entry['scorer_code_combined'])[:16]}")
        # CI 는 **채점과 같은 코드**로만 낸다. 다르면 집계기가 어차피 쓰지 않으므로 계산 전에 멈춘다.
        if entry["scorer_code_combined"] != code_start["combined"]:
            raise SystemExit(
                f"시드 {n} 채점 지문 {str(entry['scorer_code_combined'])[:16]} 이 지금 코드 "
                f"{code_start['combined'][:16]} 와 다르다 — 채점한 커밋에서 CI 를 내라")
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
        write_new_text(identity_out, json.dumps(
            {"artifact": args.artifact, "identity_check": checks,
             "inputs": {"artifacts": artifact_hash, "records_by_seed": inputs},
             "binding": make_binding(args.artifact, entries),
             "scorer_code": code_start}, ensure_ascii=False, indent=2) + "\n")
        print(f"항등 검사만 — 저장: {identity_out}")
        return 0

    # ---- 짝지은 재표집 — 한 번의 추첨을 15개에 같이 쓴다
    rng = np.random.default_rng(args.seed)
    n_res = args.n_resamples
    t_bs = time.perf_counter()

    def _progress(k: int) -> None:
        if k % 200 == 0:
            el = time.perf_counter() - t_bs
            print(f"  재표집 {k}/{n_res} · {el:.0f}s 경과 · 예상 잔여 {el / k * (n_res - k):.0f}s")

    draws = paired_bootstrap(caches, group_of_image, n_groups, seeds=seeds,
                             central="sep_central", locals_=LOCALS, fed="sep_fed",
                             n_resamples=n_res, rng=rng, progress=_progress)
    draws_map, draws_R, draws_D, draws_Rbar = draws["map"], draws["R"], draws["D"], draws["R_bar"]

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
        # 자기 파일명(판) — 채점 산출물과 같은 필드(A 54번 m-1). 집계기가 읽은 파일 이름과 맞댄다
        "artifact_version": out.name,
        "spec": "36번 미니스펙 · 총괄 게이트 2026-09-16 00:35 (조건부 통과)",
        "registration": RECOVERY_CI_REGISTRATION,
        "statistic": {
            "axis": "map_50 (export 하한)",
            "judged": "R̄ = mean_s (fed_s − localmean_s) / (central_s − localmean_s)",
            "localmean": "로컬 세 칸의 비가중 산술평균",
            "resample_unit": "group_id", "n_groups": n_groups, "n_images": len(image_ids),
            "n_resamples": n_res, "rng_seed": args.seed,
            "alpha": RECOVERY_CI_ALPHA, "interval": RECOVERY_CI_INTERVAL,
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
        # **입력 결속.** 집계기는 이 블록이 자기가 읽은 산출물과 하나라도 다르면 CI 를 쓰지 않는다
        # (codex_reply §18). 집계기와 같은 함수(`make_binding`)로 만든다.
        "binding": make_binding(args.artifact, entries),
        "scorer_code": stable_digest(code_start, code_end),
        "cache": {"path_pattern": "outputs/main_d/seed{n}/coco_evalimgs_{tag}_s{seed}.npz",
                  "role": "중간물 — 봉인하지 않는다"},
        "runtime_seconds": round(time.perf_counter() - t_all, 1),
    }
    try:
        write_new_text(out, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError:
        raise SystemExit(f"{out} 이 계산 도중 생겼다 — 덮지 않는다") from None
    print(f"저장: {out} · {payload['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
