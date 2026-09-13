"""시드 3세트 집계 — 칸별 평균·sd·범위, 회복률(시드별 + 평균), 분모 게이트, 대조선 병기.

    python scripts/probe/aggregate_seeds.py --root outputs/main_d --seeds 1,2,3 \\
        --metadata-ladder <촬영 ID 빈도 규칙 Δ_AUC 사다리 JSON> --dest outputs/main_d/seed3set

**입력은 시드별 본채점 산출물(`score_cells_v1.json`)뿐이다.** 여기서 채점을 다시 하지 않는다 —
같은 코드(wt/D 4dc868e)·같은 격자로 낸 세 산출물을 모으는 것이 이 스크립트의 전부다.

## 무엇을 지키나

1. **대조선 불변.** 무내용 대조선은 동결본 `train+val` 에만 의존하므로 세 시드가 **비트 단위로
   같아야** 한다. 다르면 동결본이 바뀐 것이고, 집계를 멈추고 보고한다(24번 §4).
2. **회복률은 시드별 값과 평균을 둘 다 싣는다.** "평균의 회복률"과 "회복률의 평균"은 다른
   수다 — 둘 다 내고 어느 쪽인지 표에 적는다.
3. **분모 게이트(사전등록 §5-4, `D ≥ 3·시드 sd`)를 판정한다.** 시드 sd 는 다섯 칸의 시드 간
   sd 를 풀링한 값(자유도 10)을 주로 쓰고, 분모 D 자체의 시드 sd(자유도 2)를 병기한다.
   41번이 처방한 CI 보조 판정은 회복률 부트스트랩 CI 가 있는 축에서만 적용한다.
4. **Δ_AUC 는 촬영 ID 빈도 규칙 대조선을 반드시 병기한다**(감사 F06). "지름길 면역" 근거는
   철회됐고, 후보 지위 판단은 총괄 판정이다 — 여기서는 판정 재료만 낸다.
5. **총괄 판정 22번 §6:** 위치 축 `map_50` 회복률만 대표 후보. 분류 축(`macro_ap`)은 대조선
   병기, 대표 숫자 없음.

GPU 무접촉. 원장·predictions·동결본 읽기 전용.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from math import sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.prereg import recovery_denominator_ok
from evaluation.stats import recovery_denominator_verdict

DET_TAGS = ("sep_central", "sep_local_C1", "sep_local_C2", "sep_local_C3", "sep_fed")
LOCALS = ("sep_local_C1", "sep_local_C2", "sep_local_C3")
FLOOR_METRICS = ("map_50", "map_50_95", "macro_ap")
OPERATING_METRICS = ("macro_f1", "miss_rate", "defect_recall", "class_jaccard", "bbox_iou")
HEADLINE_AXIS = "map_50"
"""총괄 판정 22번 §6-2-3 — 유일한 확정 대표 후보 축."""

VARIANCE_INTERPRETATION = (
    "세 시드의 분산은 **기존 난수 정책(라운드마다 같은 첫 두 epoch 순열, 감사 F01) 아래의 "
    "변동**이다. 데이터 순서·증강 다양성을 바꾼 분산으로 설명하지 않는다. 시드 3세트가 "
    "완성됐다는 사실이 F06 의 메타데이터 교락 반례를 없애지도 않는다(codex_reply Q1·Q2)."
)


# --------------------------------------------------------------------------------------
# 통계 보조
# --------------------------------------------------------------------------------------

def _summary(values: list[float]) -> dict:
    n = len(values)
    return {
        "by_seed": values,
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if n >= 2 else None,
        "min": min(values), "max": max(values), "range": max(values) - min(values),
    }


def _pooled_sd(per_cell_values: dict[str, list[float]], alpha: float = 0.05) -> dict:
    """다섯 칸의 시드 간 sd 를 풀링한다 — 분산 평균의 제곱근, 자유도 Σ(n_i − 1).

    시드 3개 한 칸의 sd(자유도 2)보다 안정적이다. 그래도 카이제곱 CI 를 함께 낸다 —
    41번이 요구한 "sd 를 보고할 때 그 자체의 CI 를 병기"다.
    """
    from scipy import stats

    ss, df = 0.0, 0
    for vals in per_cell_values.values():
        if len(vals) >= 2:
            ss += statistics.variance(vals) * (len(vals) - 1)
            df += len(vals) - 1
    if df == 0:
        return {"sd": None, "df": 0, "ci": None}
    s = sqrt(ss / df)
    lo = s * sqrt(df / stats.chi2.ppf(1 - alpha / 2, df))
    hi = s * sqrt(df / stats.chi2.ppf(alpha / 2, df))
    return {"sd": s, "df": df, "ci": [lo, hi],
            "note": f"다섯 칸 시드 간 분산 풀링(자유도 {df}). σ 의 95% CI [{lo:.4f}, {hi:.4f}]"}


def _recovery(central: float, fed: float, local_mean: float) -> float | None:
    d = central - local_mean
    return None if d <= 0 else 100.0 * (fed - local_mean) / d


# --------------------------------------------------------------------------------------
# 집계
# --------------------------------------------------------------------------------------

def load_seed(root: Path, n: int) -> dict:
    p = root / f"seed{n}" / "score_cells_v1.json"
    if not p.exists():
        raise SystemExit(f"시드 {n} 본채점 산출물이 없다: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def check_baselines_identical(payloads: dict[int, dict]) -> dict:
    """무내용 대조선이 세 시드에서 비트 단위로 같은가. 다르면 동결본이 바뀐 것이다."""
    ref_seed = min(payloads)
    ref = payloads[ref_seed]["content_free_baseline"]
    key = ("classification_axis", "position_axis", "primary_rule", "self_check_reproduced")
    mismatch = {
        s: [k for k in key if p["content_free_baseline"].get(k) != ref.get(k)]
        for s, p in payloads.items()
    }
    bad = {s: ks for s, ks in mismatch.items() if ks}
    if bad:
        raise SystemExit(
            f"무내용 대조선이 시드 사이에 다르다 {bad} — 동결본이 바뀐 것이다. "
            "집계를 멈추고 보고한다(24번 §4)"
        )
    return {"identical_across_seeds": True, "reference_seed": ref_seed,
            "classification_axis": ref["classification_axis"],
            "position_axis": ref["position_axis"],
            "self_check_reproduced": ref["self_check_reproduced"]}


def axis_table(payloads: dict[int, dict]) -> dict:
    """칸 × 지표 — 하한 축(임계 독립)과 운용점 축을 따로 낸다."""
    seeds = sorted(payloads)
    out: dict[str, dict] = {"floor": {}, "operating": {}}
    for tag in DET_TAGS:
        out["floor"][tag] = {
            m: _summary([payloads[s]["threshold_independent"]["per_tag"][tag][m] for s in seeds])
            for m in FLOOR_METRICS
        }
        out["operating"][tag] = {
            m: _summary([payloads[s]["metrics"][tag][m] for s in seeds])
            for m in OPERATING_METRICS
        }
    return out


def recovery_table(payloads: dict[int, dict], table: dict) -> dict:
    """축별 회복률 — 시드별 값, 그 평균, 평균의 회복률, 분모 게이트."""
    seeds = sorted(payloads)
    out: dict[str, dict] = {}
    for tier, metrics in (("floor", FLOOR_METRICS), ("operating", OPERATING_METRICS)):
        for m in metrics:
            per_seed = []
            for s in seeds:
                src = (payloads[s]["threshold_independent"]["per_tag"] if tier == "floor"
                       else payloads[s]["metrics"])
                c, f = src["sep_central"][m], src["sep_fed"][m]
                lm = statistics.fmean(src[t][m] for t in LOCALS)
                # 놓침처럼 낮을수록 좋은 지표는 부호를 뒤집어 같은 식으로 잰다
                sign = -1.0 if m == "miss_rate" else 1.0
                per_seed.append({
                    "seed": s, "central": c, "fed": f, "local_mean": lm,
                    "denominator": sign * (c - lm),
                    "recovery_pct": _recovery(sign * c, sign * f, sign * lm),
                })
            cm = statistics.fmean(p["central"] for p in per_seed)
            fm = statistics.fmean(p["fed"] for p in per_seed)
            lmm = statistics.fmean(p["local_mean"] for p in per_seed)
            sign = -1.0 if m == "miss_rate" else 1.0
            recs = [p["recovery_pct"] for p in per_seed if p["recovery_pct"] is not None]
            cell_vals = {t: [(payloads[s]["threshold_independent"]["per_tag"] if tier == "floor"
                              else payloads[s]["metrics"])[t][m] for s in seeds] for t in DET_TAGS}
            pooled = _pooled_sd(cell_vals)
            d_per_seed = [p["denominator"] for p in per_seed]

            # 사전등록 트립와이어 — 풀링 sd 로 (주), 분모 D 의 시드 sd 로 (병기)
            ok_pooled, d_val, msg_pooled = recovery_denominator_ok(
                sign * cm, sign * lmm, pooled["sd"] or 0.0)
            verdict_dsd = recovery_denominator_verdict(
                sign * cm, [sign * lmm], seed_values=d_per_seed)
            out[m] = {
                "tier": tier,
                "by_seed": per_seed,
                "mean_of_seed_recoveries_pct": statistics.fmean(recs) if recs else None,
                "sd_of_seed_recoveries_pct": statistics.stdev(recs) if len(recs) >= 2 else None,
                "n_seeds_recovery_defined": len(recs),
                "recovery_of_seed_means_pct": _recovery(sign * cm, sign * fm, sign * lmm),
                "means": {"central": cm, "fed": fm, "local_mean": lmm, "denominator": sign * (cm - lmm)},
                "denominator_gate": {
                    "rule": "사전등록 §5-4: D ≥ 3·시드 sd 아니면 회복률을 헤드라인으로 싣지 않는다",
                    "primary": {
                        "seed_sd_definition": "다섯 칸 시드 간 sd 풀링(자유도 10)",
                        "seed_sd": pooled["sd"], "seed_sd_ci": pooled["ci"],
                        "denominator": d_val, "threshold_3sd": 3 * (pooled["sd"] or 0.0),
                        "pass": ok_pooled, "detail": msg_pooled,
                    },
                    "secondary_denominator_sd": {
                        "seed_sd_definition": "분모 D 자체의 시드 간 sd(자유도 2)",
                        **verdict_dsd.as_dict(),
                    },
                    "ci_auxiliary": ("이 축의 회복률 부트스트랩 CI 는 본채점이 내지 않는다 — "
                                     "41번 보조 판정은 미적용"),
                },
            }
    return out


def _floor(payloads: dict[int, dict], s: int, tag: str, m: str) -> float:
    return float(payloads[s]["threshold_independent"]["per_tag"][tag][m])


def secondary_ratios(payloads: dict[int, dict], axis: str = HEADLINE_AXIS) -> dict:
    """부차 지표 두 개 — 대표 지표(회복률)를 대체하지 않는다. 산출물에서 계산만 더한다.

    - 중앙집중 대비 연합 성능 = 연합 ÷ 중앙집중
    - 개별학습 대비 향상률 = (연합 − 로컬 평균) ÷ 로컬 평균

    둘 다 같은 축(`axis`, 기본 `map_50` 하한)에서 낸다. 정의는 위 일반식이 전부다.
    """
    seeds = sorted(payloads)
    per_seed = []
    for s in seeds:
        c, f = _floor(payloads, s, "sep_central", axis), _floor(payloads, s, "sep_fed", axis)
        lm = statistics.fmean(_floor(payloads, s, t, axis) for t in LOCALS)
        per_seed.append({
            "seed": s, "central": c, "fed": f, "local_mean": lm,
            "fed_over_central_pct": None if c <= 0 else 100.0 * f / c,
            "improvement_over_local_mean_pct": None if lm <= 0 else 100.0 * (f - lm) / lm,
        })
    cm = statistics.fmean(p["central"] for p in per_seed)
    fm = statistics.fmean(p["fed"] for p in per_seed)
    lmm = statistics.fmean(p["local_mean"] for p in per_seed)
    r1 = [p["fed_over_central_pct"] for p in per_seed if p["fed_over_central_pct"] is not None]
    r2 = [p["improvement_over_local_mean_pct"] for p in per_seed
          if p["improvement_over_local_mean_pct"] is not None]
    return {
        "axis": axis,
        "role": "부차 지표 — 대표 지표는 회복률이다. 채점 기준 불변, 산출물에서 계산만 추가",
        "definitions": {
            "fed_over_central_pct": "연합 ÷ 중앙집중 × 100",
            "improvement_over_local_mean_pct": "(연합 − 로컬 평균) ÷ 로컬 평균 × 100",
        },
        "by_seed": per_seed,
        "fed_over_central_pct": {**_summary(r1), "of_seed_means": 100.0 * fm / cm if cm > 0 else None},
        "improvement_over_local_mean_pct": {
            **_summary(r2), "of_seed_means": 100.0 * (fm - lmm) / lmm if lmm > 0 else None},
        "participating_clients": "3/3",
    }


def client_improvement(payloads: dict[int, dict], axis: str = HEADLINE_AXIS) -> dict:
    """클라이언트별 향상률 = (연합 − 로컬_c) ÷ 로컬_c. 최악 클라이언트와 편차를 함께 낸다.

    평균 향상률은 참여자 간 비대칭을 가린다 — 로컬이 강한 참여자는 연합으로 손해를 볼 수
    있다. 그래서 평균이 아니라 **최악 참여자**와 최대−최소 편차를 따로 싣는다.
    """
    seeds = sorted(payloads)
    per_seed = []
    for s in seeds:
        f = _floor(payloads, s, "sep_fed", axis)
        row = {"seed": s, "fed": f, "by_client": {}}
        for t in LOCALS:
            loc = _floor(payloads, s, t, axis)
            row["by_client"][t] = {
                "local": loc,
                "improvement_pct": None if loc <= 0 else 100.0 * (f - loc) / loc,
            }
        vals = {t: v["improvement_pct"] for t, v in row["by_client"].items()
                if v["improvement_pct"] is not None}
        worst = min(vals, key=vals.get)
        row["worst_client"] = worst
        row["worst_improvement_pct"] = vals[worst]
        row["spread_pct"] = max(vals.values()) - min(vals.values())
        row["n_clients_losing"] = sum(1 for v in vals.values() if v < 0)
        per_seed.append(row)
    by_client = {
        t: _summary([p["by_client"][t]["improvement_pct"] for p in per_seed]) for t in LOCALS
    }
    worst_by_seed = [p["worst_client"] for p in per_seed]
    return {
        "axis": axis,
        "role": "부차 지표 — 대표 지표는 회복률이다. 채점 기준 불변, 산출물에서 계산만 추가",
        "definition": "(연합 − 로컬_c) ÷ 로컬_c × 100, c ∈ {C1, C2, C3}",
        "by_seed": per_seed,
        "by_client": by_client,
        "worst_client_by_seed": worst_by_seed,
        "worst_client_stable": len(set(worst_by_seed)) == 1,
        "worst_improvement_pct": _summary([p["worst_improvement_pct"] for p in per_seed]),
        "spread_pct": _summary([p["spread_pct"] for p in per_seed]),
        "n_seeds_worst_client_loses": sum(1 for p in per_seed if p["worst_improvement_pct"] < 0),
        "reading": ("최악 참여자의 향상률이 전 시드 음수면, 평균 향상률이 양수라도 그 참여자는 "
                    "연합으로 손해를 본다 — 평균이 그 사실을 가린다"),
    }


def discrimination_tables(payloads: dict[int, dict], ladder: dict | None) -> dict:
    seeds = sorted(payloads)
    delta = {t: {"point": [], "ci": []} for t in DET_TAGS}
    for s in seeds:
        for r in payloads[s]["discrimination"]["results"]:
            tag = r["cell"] if not r.get("client") else f"{r['cell']}_{r['client']}"
            if tag in delta:
                delta[tag]["point"].append(r["delta"]["point"])
                delta[tag]["ci"].append([r["delta"]["ci_lo"], r["delta"]["ci_hi"]])
    delta_out = {t: {**_summary(v["point"]), "ci_by_seed": v["ci"],
                     "all_ci_above_zero": all(lo > 0 for lo, _ in v["ci"])}
                 for t, v in delta.items()}

    auc = {t: {"point": [], "ci": []} for t in DET_TAGS}
    contrasts: dict[str, dict] = {}
    for s in seeds:
        dt = payloads[s]["discrimination_threshold_free"]
        for t in DET_TAGS:
            v = dt["by_cell"][t]
            auc[t]["point"].append(v["point"]); auc[t]["ci"].append([v["ci_lo"], v["ci_hi"]])
        for k, v in dt["contrasts"].items():
            contrasts.setdefault(k, {"point": [], "ci": []})
            contrasts[k]["point"].append(v["point"]); contrasts[k]["ci"].append([v["ci_lo"], v["ci_hi"]])
    auc_out = {t: {**_summary(v["point"]), "ci_by_seed": v["ci"]} for t, v in auc.items()}
    contrast_out = {k: {**_summary(v["point"]), "ci_by_seed": v["ci"],
                        "n_seeds_ci_excludes_zero": sum(1 for lo, hi in v["ci"] if lo > 0 or hi < 0)}
                    for k, v in contrasts.items()}

    baseline_512 = None
    if ladder:
        baseline_512 = ladder.get("512")
    fed_mean = auc_out["sep_fed"]["mean"]
    best_mean = max(v["mean"] for v in auc_out.values())
    return {
        "delta_operating": {
            "definition": "결함 발화율 − 정상 발화율 (N-crop 고정, conf 0.25 운용점)",
            "role": "선별용 — '다섯 칸 전부 CI 하한이 0 위' 한 문장으로만 (22번 §6-2-4)",
            "by_cell": delta_out,
            "all_cells_all_seeds_ci_above_zero": all(v["all_ci_above_zero"] for v in delta_out.values()),
        },
        "delta_auc": {
            "definition": "2·AUROC − 1 (N-crop 고정, 하한 레코드)",
            "status": ("후보 지위 총괄 재판정 대기. '지름길 면역' 근거는 감사 F06 으로 철회 — "
                       "촬영 ID 빈도 규칙이 같은 모집단에서 양수를 낸다"),
            "by_cell": auc_out,
            "contrasts": contrast_out,
            "metadata_baseline": {
                "rule": "idq{K}: train+val 적합 촬영 ID 구간 빈도 규칙, 이미지 미열람 (감사 F06)",
                "population": "같은 N-crop 평가 모집단 (결함 7,086 · 정상 654)",
                "delta_auc_k512": baseline_512,
                "ladder": ladder,
                "reproduced_by": "D 독립 재현 — Codex 스크립트(read-only) + 자체 드라이버·쌍별 AUROC 둘 다 소수 6자리 일치",
                "fed_mean_minus_baseline_k512": (None if baseline_512 is None else fed_mean - baseline_512),
                "best_cell_mean_minus_baseline_k512": (None if baseline_512 is None else best_mean - baseline_512),
                "reading": (
                    "모든 칸의 3시드 평균이 K=512 대조선 아래면, 이 축에서도 '메타데이터만으로 도달 가능한 "
                    "수준'을 모델이 넘지 못한 것이다. 대조선은 상한이 아니라 반례 하나다(K=4 는 음수)"
                ),
            },
        },
    }


def stratified_table(payloads: dict[int, dict]) -> dict:
    seeds = sorted(payloads)
    k = str(payloads[seeds[0]]["stratified"]["default_k"])
    out = {}
    for tag in [*DET_TAGS, "__shortcut__"]:
        lifts = [payloads[s]["stratified"]["by_k"][k][tag]["stratified_lift"] for s in seeds]
        out[tag] = {**_summary(lifts), "n_seeds_negative": sum(1 for v in lifts if v < 0)}
    return {"k": int(k), "lift": out,
            "shortcut_lift_exactly_zero_all_seeds": all(v == 0.0 for v in out["__shortcut__"]["by_seed"])}


def p9_tables(payloads: dict[int, dict]) -> dict:
    out = {}
    for s in sorted(payloads):
        rows = []
        for r in payloads[s]["p9"]["results"]:
            rows.append({
                "tag": r.get("tag") or (r["cell"] if not r.get("client") else f"{r['cell']}_{r['client']}"),
                "fp_rate_crop": r["fp_rate_crop"], "fp_rate_tile": r["fp_rate_tile"],
                "fp_rate_diff": r["fp_rate_diff"], "n_crop": r["n_crop"], "n_tile": r["n_tile"],
                "tost_equivalent": (r.get("tost") or {}).get("equivalent"),
            })
        out[str(s)] = {"rows": rows, "all_equivalent": payloads[s]["p9"]["all_equivalent"]}
    n_eq = {s: sum(1 for r in v["rows"] if r["tost_equivalent"]) for s, v in out.items()}
    return {"by_seed": out, "n_cells_equivalent_by_seed": n_eq,
            "note": "판정 2(22번 §2-2): 승격 보류·분리 보고 필수·시드 3세트 후 재판정. 여기는 재판정 재료다"}


def decomposition_table(payloads: dict[int, dict]) -> dict:
    seeds = sorted(payloads)
    out = {}
    for mat in payloads[seeds[0]]["decomposition"]["by_group"]:
        per_tag = {t: _summary([payloads[s]["decomposition"]["by_group"][mat]["per_tag"][t]["map_50"] for s in seeds])
                   for t in DET_TAGS}
        recs = [payloads[s]["decomposition"]["by_group"][mat]["recovery"]["map_50"]["recovery_pct"] for s in seeds]
        out[mat] = {"map_50_by_tag": per_tag,
                    "map_50_recovery_by_seed": recs,
                    "map_50_recovery_mean": statistics.fmean(r for r in recs if r is not None) if any(r is not None for r in recs) else None,
                    "n_images": payloads[seeds[0]]["decomposition"]["by_group"][mat]["n_images"]}
    return {"axis": payloads[seeds[0]]["decomposition"]["axis"],
            "limitation": payloads[seeds[0]]["decomposition"]["limitation"], "by_group": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/main_d")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--metadata-ladder", default=None,
                    help="촬영 ID 빈도 규칙 Δ_AUC 사다리 JSON (Codex 스크립트 출력 형식 또는 {K: 값})")
    ap.add_argument("--dest", default="outputs/main_d/seed3set")
    args = ap.parse_args()

    root = Path(args.root)
    seeds = [int(s) for s in args.seeds.split(",")]
    payloads = {s: load_seed(root, s) for s in seeds}

    # 시드 값·격자·코드 동일성 — 세 산출물이 한 기준인지
    seed_values = {s: p["params"]["seed"] for s, p in payloads.items()}
    if len(set(seed_values.values())) != len(seeds):
        raise SystemExit(f"시드 값이 겹친다: {seed_values}")
    grids = {s: tuple(p["curve"]["grid"]) for s, p in payloads.items()}
    if len(set(grids.values())) != 1:
        raise SystemExit(f"격자가 시드 사이에 다르다: {grids}")
    for s, p in payloads.items():
        if p["exit_code"] != 0 or p["gates_evaluated"]["blocking_failures"]:
            raise SystemExit(f"시드 {s} 본채점이 결과로 쓸 수 없는 상태다: exit {p['exit_code']} "
                             f"차단 {p['gates_evaluated']['blocking_failures']}")

    baseline = check_baselines_identical(payloads)
    print(f"시드 {seeds} · 시드값 {seed_values} · 격자 {len(grids[seeds[0]])}점 · 대조선 3시드 동일")

    ladder = None
    if args.metadata_ladder:
        raw = json.loads(Path(args.metadata_ladder).read_text(encoding="utf-8"))
        if isinstance(raw, list):          # Codex 스크립트 --ladder 출력 형식
            ladder = {r["name"].split("_")[0][3:]: r["delta_auc"] for r in raw}
        else:
            ladder = {str(k): v for k, v in raw.items()}

    table = axis_table(payloads)
    rec = recovery_table(payloads, table)
    disc = discrimination_tables(payloads, ladder)
    strat = stratified_table(payloads)
    p9 = p9_tables(payloads)
    decomp = decomposition_table(payloads)
    sec = secondary_ratios(payloads)
    cli = client_improvement(payloads)

    head = rec[HEADLINE_AXIS]
    print(f"[{HEADLINE_AXIS}] 시드별 회복률 "
          + " / ".join(f"{p['recovery_pct']:+.1f}%" for p in head["by_seed"])
          + f" · 평균 {head['mean_of_seed_recoveries_pct']:+.1f}% "
          + f"· 평균의 회복률 {head['recovery_of_seed_means_pct']:+.1f}%")
    g = head["denominator_gate"]["primary"]
    print(f"[{HEADLINE_AXIS}] 분모 게이트: {g['detail']} (sd 정의: {g['seed_sd_definition']})")
    print("[macro_ap] 시드별 회복률 " + " / ".join(f"{p['recovery_pct']:+.1f}%" for p in rec['macro_ap']['by_seed'])
          + f" — 대조선 {baseline['classification_axis']['macro_ap_freq']:.4f} 병기, 대표 아님")
    mb = disc["delta_auc"]["metadata_baseline"]
    print("[Δ_AUC] 3시드 평균 " + ", ".join(f"{t} {v['mean']:+.4f}" for t, v in disc["delta_auc"]["by_cell"].items())
          + (f" · 촬영 ID 대조선 K=512 {mb['delta_auc_k512']:+.4f}" if mb["delta_auc_k512"] is not None else " · 대조선 미제공"))
    fc = disc["delta_auc"]["contrasts"].get("fed_minus_central", {})
    if fc:
        print("[Δ_AUC] 연합−중앙집중 시드별 " + " / ".join(f"{v:+.4f}" for v in fc["by_seed"])
              + f" · CI 가 0 을 배제한 시드 {fc['n_seeds_ci_excludes_zero']}/{len(seeds)}")
    print(f"[판별력 Δ] 전 칸·전 시드 CI 하한 > 0: {disc['delta_operating']['all_cells_all_seeds_ci_above_zero']}")
    print(f"[층화 K={strat['k']}] lift 음수 시드 수: " + ", ".join(f"{t} {v['n_seeds_negative']}" for t, v in strat["lift"].items() if t != "__shortcut__")
          + f" · 지름길 행 정확히 0: {strat['shortcut_lift_exactly_zero_all_seeds']}")
    print(f"[P9] TOST 동등 칸 수(시드별): {p9['n_cells_equivalent_by_seed']}")
    print(f"[부차·{HEADLINE_AXIS}] 연합÷중앙집중 시드별 "
          + " / ".join(f"{p['fed_over_central_pct']:.1f}%" for p in sec["by_seed"])
          + f" · 평균 {sec['fed_over_central_pct']['mean']:.1f}% ; 개별학습 대비 향상률 "
          + " / ".join(f"{p['improvement_over_local_mean_pct']:+.1f}%" for p in sec["by_seed"])
          + f" · 평균 {sec['improvement_over_local_mean_pct']['mean']:+.1f}%")
    for t, v in cli["by_client"].items():
        print(f"[클라이언트별·{HEADLINE_AXIS}] {t}: "
              + " / ".join(f"{x:+.1f}%" for x in v["by_seed"]) + f" · 평균 {v['mean']:+.1f}%")
    print(f"[클라이언트별] 최악 참여자 {cli['worst_client_by_seed']} · 최악 향상률 평균 "
          f"{cli['worst_improvement_pct']['mean']:+.1f}% · 손해 보는 시드 "
          f"{cli['n_seeds_worst_client_loses']}/{len(seeds)} · 편차 평균 {cli['spread_pct']['mean']:.1f}pp")

    payload = {
        "seeds": seeds, "seed_values": seed_values,
        "scorer_code": "wt/D 4dc868e (세 시드 동일 코드·동일 격자)",
        "grid": list(grids[seeds[0]]),
        "headline_policy": payloads[seeds[0]].get("headline_policy"),
        "headline_axis": HEADLINE_AXIS,
        "content_free_baseline": baseline,
        "cells": table,
        "recovery": rec,
        "discrimination": disc,
        "stratified": strat,
        "p9": p9,
        "decomposition": decomp,
        "secondary_ratios": sec,
        "client_improvement": cli,
        "variance_interpretation": VARIANCE_INTERPRETATION,
        "gates_by_seed": {s: {"exit_code": p["exit_code"], "ok": p["gates_evaluated"]["ok"],
                              "n_evaluated": p["gates_evaluated"]["n_evaluated"]}
                          for s, p in payloads.items()},
    }
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "aggregate_v1.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
