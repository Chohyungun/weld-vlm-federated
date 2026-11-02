"""본채점(엄격 · **표본 모드**)의 지표 단계 — 진입점 미니스펙 3판 1-1 의 6 단계 계산. 실험설계 8-5(2026-10-13 등록) · 25번 · 07번.

입력은 **검증을 지난 묶음에서 채점 층이 만든 계수표**(`scoring_layer.score_registered`)와 등록된 대조선 예측의 계수표다. 이 모듈은
묶음 · 정답 · 이미지 · 파일을 열지 않는다(대조선 예측의 바이트만 받아 읽는다). 등록되지 않은 값을 코드에서 정하지 않는다 —
재표집 수 · 난수 시드 · 유효 추첨 하한 · 규칙 문턱은 등록 객체에서, 표본의 봉인 · 덧붙임 문턱은 평가 등록 설정에서 읽는다.
**코드가 정한 것**은 재표집의 꼴 하나다 — 산출의 `unregistered_choices` 가 그 사실과 근거를 싣는다(35번 §1-3).

## 산출의 뼈대

| 칸 | 무엇 | 근거 |
|---|---|---|
| `models` | 모델 × 시드마다 대표 M1 — 주 · 민감도, 설계 가중 · 가중 없음(점추정), 실패율 | 실험설계 8-5 · `sample_weights.representative` |
| `baseline` | 등록 대조선 예측을 같은 목록 · 같은 가중 · 같은 채점기로 | 25번 §4-4 · 07번 §15-7 |
| `bootstrap` | 층 안 묶음 복원추출 — 층마다 `n_h` · 중복도 행렬의 sha256 · NumPy 판 · 항등 검사 | 25번 §3-1 |
| `rq1_recovery` | 회복률 `R = (F − L̄)/(C − L̄)` — 시드별 점추정 · 구간(10-01 규약) · 다섯 카운터 · R①~R④ 상태 | 07번 §24-3 · §24-4 · §33 |
| `rq2_participants` | 참여자마다 `연합 − 로컬_k` 대응 차와 방향(위 · 아래 · 구분 불가). **참여자 평균을 내지 않는다** | 실험설계 8-5 · 07번 §19-4 |
| `rule5` | 규칙 ⑤ 등록 판정과 근거. 로컬 셋과 로컬 평균은 진단 모델로 병기한다(공식 판정에 들지 않는다) | `evaluation.rule5` · 07번 §19-1 · §23-2 |
| `main_vs_sensitivity` | 두 채점에서 세 방식의 순위 · 회복률 부호가 바뀌는가 | 25번 §4-3 |
| `full_addendum` | 시드 1 표본의 분모 점추정이 등록 문턱보다 작으면 전량을 덧붙인다 — 표시만 한다 | 25번 §3-2 |
| `not_judged_on_sample` | 표본에서 판정하지 않는 것 — AL 부분집합 매크로 M1 은 점추정과 구간만, P9 등가는 출처 축 입력이 없어 내지 않는다 | 25번 §3-3 |
| `not_computed` | 이 판이 내지 않는 것 — 키를 빼지 않고 상태로 적는다 | 3판 1-4 |
| `narrative` · `clause_axis` | 2 차 보조 지표 — 판 6 등록에 블록이 있고 평가 등록 `auxiliary` 에 이름이 있을 때만(`aux_scoring`). 같은 추첨 `W` 를 쓴다 | 03 2판 §5a · `evaluation.strict_narrative` |

**같은 추첨.** 재표집 중복도 하나를 다섯 모델 · 모든 시드 · 대조선이 같이 쓴다. 추첨이 잴 수 없는지는 정답 지지량으로만 정해지므로
모델과 상관없다 — 다르면 배관이 틀린 것이라 거부한다.

**거부된 칸.** 검증 · 채점에서 거부된 모델 칸(묶음이 없는 등록 칸 포함)은 값 없이 `rejected` 와 사유이고, 그 칸이 필요한 파생값은
`산출 불가` 와 빠진 칸의 이름이다(3판 1-4). 분모를 조용히 줄이지 않는다. 모델 묶음 사이 환경이 다르면 모델을 맞대는 값 전부가 `산출 불가` 다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

import numpy as np

from evaluation import percentile_rule as PR
from evaluation import rule5 as R5
from evaluation.metrics.box_f1 import CountTable, build_counts, score_m1
from evaluation.recovery_interval import point_recovery, recovery_interval_registered
from evaluation.sample_bootstrap import (IDENTITY_TOL, design_weights, draw_multiplicities, identity_problems,
                                         m1_draws, multiplicity_problems, multiplicity_sha256, strata_groups,
                                         weights_from)
from evaluation.sample_weights import WEIGHTING_SAMPLE, representative

CENTRAL, FED = "uni_central", "uni_fed"
LOCALS: tuple[str, ...] = ("uni_local_C1", "uni_local_C2", "uni_local_C3")
TAGS: tuple[str, ...] = (CENTRAL, FED, *LOCALS)
"""본실험 통합형 다섯 모델(`schema_v14.UNI_MAIN_TAGS` 와 같은 이름 — 시험이 맞댄다)."""
LOCAL_MEAN = "local_mean"
"""규칙 ⑤ 의 진단 자리 — 로컬 셋의 시드별 평균(07번 §19-1 의 '로컬 평균은 조건에서 빼고 대조선과의 관계를 병기')."""
VARIANTS = ("main", "sensitivity")
RESAMPLING = "within_stratum_group"
"""재표집의 꼴 — 층 안 묶음 복원추출, 보정 없음(`sample_bootstrap`). 표본 모드만 이 판에 있다."""
UNAVAILABLE = "산출 불가"
DIRECTION_KO = {"above": "위", "below": "아래", "indeterminate": "구분 불가"}
BASELINE_LINE_KEYS = frozenset({"image_id", "defects"})
BASELINE_DEFECT_KEYS = frozenset({"iso_code", "bbox_px"})
MATERIALS = ("AL", "ST")
"""곁 파일의 층 이름 `재질|구간` 의 앞 칸(`data.eval_sample` 의 층 정의). AL 부분집합은 이 칸으로 가른다 — 매니페스트를 읽지 않는다."""
UNREGISTERED_CHOICES = ({
    "what": "재표집의 꼴 — 층 안 묶음 복원추출(층마다 n_h 개를 복원추출, 가중 G_h/n_h × 중복도), 분산 보정(Rao–Wu 등) 없음",
    "basis": "25번 §3-1(승인된 크기 어림이 쓴 방법) · 외부 검토 §137-3-3('이미 결정된 층 안 group 재표집')",
    "registration": "등록 칸이 없다 — 판 6 의 ci.resampling 칸(07번 §39-4-1)에서 정한다. 그때까지 이 값은 코드가 정한 것이다",
    "affects": ["rq1_recovery.*.interval", "rq2_participants.*.rows.*.interval", "rule5 ⑤-②"],
    "known_bias": "보정이 없으면 층마다 분산이 (n_h − 1)/n_h 배로 작게 잡힌다 — bootstrap.strata 의 n_h 를 본다",
},)


def cell_name(tag: str, seed_index: int) -> str:
    return f"{tag}_s{seed_index}"


# ── 대조선 예측 ──────────────────────────────────────────────────────────────

class BaselineFileError(ValueError):
    """등록 대조선 예측 파일의 꼴이 틀렸다. 사유에는 줄 번호와 건수만 싣는다."""


def _box_ok(b) -> bool:
    return (isinstance(b, list) and len(b) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in b)
            and b[0] < b[2] and b[1] < b[3])


def read_baseline_predictions(raw: bytes, *, population: Sequence[str], allowed_ids: set[str],
                              known_codes: set[str], classes: Sequence[str]) -> tuple[dict, dict]:
    """대조선 예측(JSON 줄 — `{"image_id", "defects": [{"iso_code", "bbox_px"}]}`) → 채점 모집단의 예측과 회계.

    파일은 등록의 `baseline.prediction_sha256` 이 묶은 바이트다(진입점이 해시를 먼저 맞댄다). 한 줄 한 이미지, id 가 겹치지 않고,
    **채점 모집단을 전부 덮는다** — 빠진 이미지는 대조선의 미탐이 지표에서 사라지는 것이라 거부한다. 모집단 밖의 줄은 평가 분할의
    id 일 때만 받고(전량 예측 파일을 표본에 쓰는 경우) 채점하지 않는다. 채점 클래스 밖의 코드는 버리고 센다.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise BaselineFileError("대조선 예측이 UTF-8 이 아니다") from None
    if not text.endswith("\n") or "\r" in text:
        raise BaselineFileError("대조선 예측은 LF 줄끝 · 끝 개행이어야 한다")
    pred: dict[str, list] = {}
    n_out_of_scope = n_outside_population = 0
    pop = set(population)
    cls = set(classes)
    for no, line in enumerate(text.split("\n")[:-1], 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise BaselineFileError(f"{no} 번째 줄이 JSON 이 아니다") from None
        if not isinstance(row, dict) or set(row) != BASELINE_LINE_KEYS:
            raise BaselineFileError(f"{no} 번째 줄의 키가 {sorted(BASELINE_LINE_KEYS)} 가 아니다")
        iid, defects = row["image_id"], row["defects"]
        if not isinstance(iid, str) or iid in pred or iid not in allowed_ids:
            raise BaselineFileError(f"{no} 번째 줄의 image_id 가 겹치거나 평가 분할 밖이다")
        if not isinstance(defects, list):
            raise BaselineFileError(f"{no} 번째 줄의 defects 가 목록이 아니다")
        boxes = []
        for d in defects:
            if not isinstance(d, dict) or set(d) != BASELINE_DEFECT_KEYS or not isinstance(d["iso_code"], str) \
                    or d["iso_code"] not in known_codes:
                raise BaselineFileError(f"{no} 번째 줄에 꼴이 틀린 항목이 있다")
            if not _box_ok(d["bbox_px"]):
                raise BaselineFileError(f"{no} 번째 줄에 퇴화 · 비유한 상자가 있다")
            if d["iso_code"] not in cls:
                n_out_of_scope += 1
                continue
            boxes.append((d["iso_code"], [float(x) for x in d["bbox_px"]]))
        pred[iid] = boxes
        if iid not in pop:
            n_outside_population += 1
    missing = pop - set(pred)
    if missing:
        raise BaselineFileError(f"대조선 예측이 채점 모집단 {len(missing)} 장을 덮지 않는다")
    return ({i: pred[i] for i in population},
            {"n_lines": len(pred), "n_outside_population": n_outside_population, "n_out_of_scope_dropped": n_out_of_scope})


def baseline_table(pred: Mapping[str, Sequence], gold_sub: Mapping[str, Sequence], classes: Sequence[str]) -> CountTable:
    """모델과 **같은 채점기**(`build_counts`)로 — 단일 채점기(불변조건 3-7)."""
    return build_counts(pred, gold_sub, list(classes))


# ── 계산 ─────────────────────────────────────────────────────────────────────

def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs))


def _interval(draws: np.ndarray, *, alpha: float, reasons: Mapping[str, int]) -> dict:
    iv, n_def, n_undef = PR.defined_interval(draws, alpha)
    return {**iv.as_dict(), "n_defined": n_def, "n_undefined": n_undef, "undefined_reasons": dict(reasons)}


def _direction(iv: dict, min_draws: int) -> tuple[str | None, str | None]:
    """구간의 방향 — 유효 추첨이 등록 하한(`ci.min_defined_draws`)보다 적으면 방향을 말하지 않는다(규칙 ⑤ 와 같은 기준)."""
    if iv["n_defined"] < min_draws or iv["n_defined"] == 0:
        return None, f"유효 추첨 {iv['n_defined']} 이 등록 하한 {min_draws} 보다 적다"
    if iv["ci_lo_state"] != "finite" or iv["ci_hi_state"] != "finite":
        return None, f"구간 끝점이 유한하지 않다({iv['ci_lo_state']} · {iv['ci_hi_state']})"
    key = "above" if iv["ci_lo"] > 0 else "below" if iv["ci_hi"] < 0 else "indeterminate"
    return DIRECTION_KO[key], None


def compute(*, scores: Mapping[tuple[str, int], object], baseline: CountTable, side: Mapping, group_of: Mapping[str, str],
            reg, eval_doc: Mapping, rejected: Mapping[str, Sequence[str]], invalid_comparison: Sequence[str] = (),
            aux_scoring=None) -> dict:
    """표본 모드의 지표 단계. `scores` 는 `(태그, 시드 번호)` → `score_registered` 의 결과다. `rejected` 는 칸 이름 → 사유.

    `invalid_comparison` 이 비어 있지 않으면(모델 묶음 사이 환경 불일치 — `bundle_v14.check_env_consistency`) 모델마다의 값은 싣되
    모델을 맞대는 것(재표집 · 회복률 · 참여자 차 · 규칙 ⑤ · 순위 바뀜 · 덧붙임)은 `산출 불가` 다 — 다른 조건의 출력을 한 비교에 넣지 않는다.

    `aux_scoring`(`strict_narrative.AuxScoring`)을 주면 2 차 보조 지표 두 블록을 더하고 그 이름을 `not_computed` 에서 뺀다.
    `strict_narrative.AuxRefused`(판 6 블록의 신원이 맞지 않음)를 주면 블록 대신 `aux_identity` 를 싣고 그 이름의 `not_computed` 사유를
    거부로 바꾼다. 주지 않으면 산출은 이 인자가 없던 때와 바이트가 같다.
    """
    reg5 = R5.registered_inputs(reg)
    seeds = sorted(reg5.seeds)
    ci = reg.ci
    alpha = ci.alpha
    ids = baseline.image_ids
    support = baseline.tp + baseline.fn
    for key, sc in scores.items():
        if sc.table.image_ids != ids or sc.sens.image_ids != ids or sc.table.classes != baseline.classes:
            raise ValueError(f"{cell_name(*key)}: 계수표의 이미지 · 클래스가 대조선과 다르다 — 같은 모집단이 아니다")
        if not np.array_equal(sc.table.tp + sc.table.fn, support) or not np.array_equal(sc.sens.tp + sc.sens.fn, support):
            raise ValueError(f"{cell_name(*key)}: 이미지 × 클래스의 정답 지지량이 대조선과 다르다 — 같은 정답이 아니다")
    groups = strata_groups(ids, group_of, side)

    # 점추정 — 대표 M1(주 · 민감도, 설계 가중 · 가중 없음)
    reps = {key: representative(sc.table, gold_codes=sc.gold_codes, failed_image_ids=list(sc.failed), side=side)
            for key, sc in scores.items()}
    w0 = design_weights(ids, side)
    for key, sc in scores.items():          # 재표집이 쓰는 민감도 표 = 대표 산출의 민감도(등록 변형) — 다르면 두 숫자가 다른 것을 잰다
        got, want = score_m1(sc.sens, w0).macro_f1, reps[key]["sensitivity"]["m1_macro_f1"]
        if (got is None) != (want is None) or (got is not None and abs(got - want) > IDENTITY_TOL):
            raise ValueError(f"{cell_name(*key)}: 민감도 계수표가 대표 산출의 민감도와 다르다 — 변형이 등록값이 아니다")
    base_rep = score_m1(baseline, w0)
    classes_in_macro = tuple(base_rep.classes_in_macro)
    for key, rep in reps.items():
        for v in VARIANTS:
            if tuple(rep[v]["classes_in_macro"]) != classes_in_macro:
                raise ValueError(f"{cell_name(*key)}: 평균 대상이 대조선과 다르다 — 정답이 같은 모집단이 아니다")

    def point(tag: str, s: int, v: str) -> float | None:
        rep = reps.get((tag, s))
        return None if rep is None else rep[v]["m1_macro_f1"]

    def missing_for(tags: Sequence[str]) -> list[str]:
        return [cell_name(t, s) for t in tags for s in seeds if (t, s) not in scores]

    out: dict = {
        "estimand": "초모집단 — 표본 값은 전량 평가의 실측이 아니다(07번 §15-4)",
        "weighting": WEIGHTING_SAMPLE, "n_images": len(ids), "n_groups": sum(len(g) for g in groups.values()),
        "n_strata": len(groups), "classes_in_macro": list(classes_in_macro), "seeds": [str(s) for s in seeds],
        "unregistered_choices": [dict(c) for c in UNREGISTERED_CHOICES], "models": {},
    }
    for tag in TAGS:
        for s in seeds:
            name = cell_name(tag, s)
            if (tag, s) in scores:
                out["models"][name] = {"status": "scored", "representative": reps[(tag, s)],
                                       "failures": scores[(tag, s)].failures}
            else:
                out["models"][name] = {"status": "rejected", "reasons": list(rejected.get(name, ["missing_bundle"]))}
    out["baseline"] = {"main": base_rep.as_dict(), "main_unweighted": score_m1(baseline).as_dict(),
                       "note": "같은 목록 · 같은 가중 · 같은 채점기 — 전량의 등록 상수와 비교하지 않는다(25번 §4-4)"}
    aux = [a for a in (eval_doc.get("auxiliary") or []) if a not in ("m1_micro", "per_class_f1")]
    out["not_computed"] = {**{a: "이 판의 범위 밖 — 35번 §1" for a in aux},
                           "source_axis": "N-crop · N-tile 갈래 · 판별력 Δ — 출처 칸이 엄격 경로의 입력에 없다(35번 §1)",
                           "failure_rate_by_source": "출처 축 갈래는 같은 까닭으로 내지 않는다 — 정상 · 결함 갈래만 싣는다"}
    out["m1_micro_and_per_class"] = "models.*.representative.*.micro_f1 · per_class_f1 에 있다"
    if aux_scoring is not None:
        for name in aux_scoring.computed_names():
            out["not_computed"].pop(name, None)
        out["not_computed"].update(aux_scoring.not_computed_reasons())
    nj = list(((eval_doc.get("population") or {}).get("sample") or {}).get("not_judged_on_sample") or [])
    if invalid_comparison:
        why = {"status": UNAVAILABLE, "reason": "env_mismatch", "detail": list(invalid_comparison),
               "note": "묶음 사이 환경이 달라 모델을 한 비교에 넣지 않는다(check_env_consistency)"}
        for key in ("bootstrap", "rq1_recovery", "rq2_participants", "rule5", "main_vs_sensitivity", "full_addendum"):
            out[key] = dict(why)
        out["not_judged_on_sample"] = {k: dict(why) for k in nj}
        if aux_scoring is not None:
            out.update(aux_scoring.report(image_ids=ids, W=None, alpha=alpha, min_defined_draws=reg5.min_defined_draws,
                                  seeds=seeds, cell_name=cell_name, fed=FED, central=CENTRAL, locals_=LOCALS,
                                  comparable=False))
        return out

    # 재표집 — 중복도 하나를 모두에게
    M = draw_multiplicities(groups, n_resamples=ci.n_resamples, rng_seed=ci.rng_seed)
    problems = multiplicity_problems(M, groups) + identity_problems(baseline, side, group_of, classes_in_macro)
    W = weights_from(M, ids, group_of, side, groups)
    draws: dict[tuple[str, int, str], np.ndarray] = {}
    base_draws, support_ok = m1_draws(baseline, W, classes_in_macro)
    for key, sc in scores.items():
        problems += [f"{cell_name(*key)}: {p}" for p in identity_problems(sc.table, side, group_of, classes_in_macro)]
        for v, table in (("main", sc.table), ("sensitivity", sc.sens)):
            d, ok = m1_draws(table, W, classes_in_macro)
            if not np.array_equal(ok, support_ok):
                raise ValueError(f"{cell_name(*key)}: 추첨의 지지량 정의가 대조선과 다르다 — 정답이 같지 않다")
            draws[(*key, v)] = d
    if problems:
        raise ValueError("재표집 검사 실패: " + " · ".join(problems))
    g_of = {side[i].stratum: side[i].G_h for i in ids}
    out["bootstrap"] = {"resampling": RESAMPLING, "n_resamples": ci.n_resamples, "rng_seed": ci.rng_seed,
                        "n_excluded_support": int((~support_ok).sum()), "identity_check": "pass",
                        "multiplicity_check": "pass", "multiplicity_sha256": multiplicity_sha256(M),
                        "numpy_version": np.__version__,
                        "strata": {h: {"n_h": len(gs), "G_h": g_of[h],
                                       "variance_factor_without_correction": (len(gs) - 1) / len(gs)}
                                   for h, gs in groups.items()},
                        "shared_draws": "다섯 모델 · 모든 시드 · 대조선이 같은 중복도 행렬"}
    min_draws = reg5.min_defined_draws
    out["rq1_recovery"] = {v: _recovery(v, seeds, point, draws, support_ok, missing_for(TAGS), ci) for v in VARIANTS}
    out["rq2_participants"] = {v: _participants(v, seeds, point, draws, support_ok, missing_for, alpha, min_draws,
                                                base_rep.macro_f1) for v in VARIANTS}
    out["rule5"] = _rule5(reg5, seeds, point, draws, base_draws, support_ok, missing_for, base_rep.macro_f1, alpha)
    out["main_vs_sensitivity"] = _flips(seeds, point, out["rq1_recovery"], missing_for(TAGS))
    out["full_addendum"] = _addendum(seeds, point, eval_doc)
    out["not_judged_on_sample"] = {k: _not_judged(k, scores, side, W, w0, ids, alpha) for k in nj}
    if aux_scoring is not None:
        out.update(aux_scoring.report(image_ids=ids, W=W, alpha=alpha, min_defined_draws=min_draws, seeds=seeds,
                              cell_name=cell_name, fed=FED, central=CENTRAL, locals_=LOCALS, comparable=True))
    return out


def _recovery(v: str, seeds, point, draws, support_ok, missing: list[str], ci) -> dict:
    if missing:
        return {"status": UNAVAILABLE, "missing_cells": missing}
    F = np.array([point(FED, s, v) for s in seeds], dtype=object)
    C = np.array([point(CENTRAL, s, v) for s in seeds], dtype=object)
    L = np.array([[point(t, s, v) for t in LOCALS] for s in seeds], dtype=object)
    if any(x is None for x in (*F, *C, *L.ravel())):
        pt = {"R_bar": None, "state": "absent", "reason": "support", "convention_id": PR.CONVENTION_ID}
    else:
        pt = point_recovery(F.astype(float), C.astype(float), L.astype(float))
    # 지지량 제외 추첨의 자리는 분류 전에 0 으로 비운다 — 분류기가 지지량을 먼저 보고 그 추첨을 뺀다
    Fd = np.where(support_ok[:, None], np.stack([draws[(FED, s, v)] for s in seeds], axis=1), 0.0)
    Cd = np.where(support_ok[:, None], np.stack([draws[(CENTRAL, s, v)] for s in seeds], axis=1), 0.0)
    Ld = np.stack([np.stack([draws[(t, s, v)] for t in LOCALS], axis=-1) for s in seeds], axis=1)
    Ld = np.where(support_ok[:, None, None], Ld, 0.0)
    iv = recovery_interval_registered(ci, Fd, Cd, Ld, support_ok).as_dict()
    finite = iv["half_width_state"] == "finite"
    r2 = None if ci.rule2_max_half_width is None or not finite else bool(iv["half_width"] <= ci.rule2_max_half_width)
    r4 = None if ci.rule4_max_undefined_fraction is None else bool(iv["rule4_fraction"] <= ci.rule4_max_undefined_fraction)
    one = len(seeds) == 1
    hold = "미판정 — 한 시드(실험설계 8-5 · 25번 §3-6)" if one else "미판정 — 이 판에 구현이 없다(35번 §1)"
    Fm = _mean([float(x) for x in F]) if all(x is not None for x in F) else None
    Cm = _mean([float(x) for x in C]) if all(x is not None for x in C) else None
    r2_note = "반폭이 수가 아니면 판정하지 않는다 — 통과로 만들지 않는다(07번 §25-2 가2)" if not finite else None
    return {"status": "computed", "point": pt, "interval": iv,
            "rules": {"R1": hold, "R3": hold,
                      "R2": {"half_width_within": r2, "half_width_state": iv["half_width_state"],
                             "threshold": ci.rule2_max_half_width, "note": r2_note,
                             "adoption_use": "한 시드라 채택 판정에 쓰지 않는다" if one else None},
                      "R4": {"within": r4, "threshold": ci.rule4_max_undefined_fraction,
                             "adoption_use": "한 시드라 채택 판정에 쓰지 않는다" if one else None}},
            "adoption": hold,
            "companions": {"raw_gap_central_minus_fed": None if Fm is None or Cm is None else Cm - Fm,
                           "fed_over_central_pct": None if Fm is None or not Cm else 100.0 * Fm / Cm}}


def _participants(v: str, seeds, point, draws, support_ok, missing_for, alpha: float, min_draws: int, baseline_m1) -> dict:
    rows = {}
    for t in LOCALS:
        miss = missing_for((FED, t))
        if miss:
            rows[t] = {"status": UNAVAILABLE, "missing_cells": miss}
            continue
        f = [point(FED, s, v) for s in seeds]
        lk = [point(t, s, v) for s in seeds]
        if any(x is None for x in (*f, *lk)):
            rows[t] = {"status": UNAVAILABLE, "reason": "support"}
            continue
        diff = _mean([a - b for a, b in zip(f, lk, strict=True)])
        d = np.mean(np.stack([draws[(FED, s, v)] - draws[(t, s, v)] for s in seeds], axis=0), axis=0)
        d = np.where(support_ok, d, np.nan)
        iv = _interval(d, alpha=alpha, reasons={"support": int((~support_ok).sum())})
        direction, why = _direction(iv, min_draws)
        lbar = _mean(lk)
        rows[t] = {"status": "computed", "F": _mean(f), "L_k": lbar, "diff_F_minus_L_k": diff, "interval": iv,
                   "direction": direction, "direction_reason": why, "min_defined_draws": min_draws,
                   "relative_pct": None if lbar == 0 else 100.0 * (_mean(f) - lbar) / lbar,
                   "relative_note": "참고 — 로컬이 대조선보다 낮다" if baseline_m1 is not None and lbar < baseline_m1 else None}
    done = {t: r["diff_F_minus_L_k"] for t, r in rows.items() if r["status"] == "computed"}
    if len(done) == len(LOCALS):
        w = min(done, key=lambda t: (done[t], t))
        worst = {"participant": w, "diff": done[w], "interval": rows[w]["interval"], "direction": rows[w]["direction"],
                 "caveat": "결과를 본 뒤 정해지는 이름이다 — 사전에 고른 참여자가 아니다"}
    else:
        worst = {"status": UNAVAILABLE, "reason": "산출 불가인 참여자가 있어 가장 낮은 참여자를 정하지 않는다(3판 1-4)",
                 "computed": sorted(done)}
    return {"definition": "연합 M1 − 로컬_k M1(같은 표본 · 같은 가중 · 같은 추첨). 부호를 뒤집지 않는다",
            "rows": rows, "no_average": "참여자 평균을 내지 않는다(실험설계 8-5)", "worst": worst}


def _evidence(tag: str, per_seed: Mapping[int, float | None], mean_draw: np.ndarray, base_draws, support_ok,
              alpha: float) -> R5.ModelEvidence:
    d = np.where(support_ok, mean_draw - base_draws, np.nan)
    return R5.ModelEvidence(tag, dict(per_seed), R5.DiffCI.from_draws(d, {"support": int((~support_ok).sum())}, alpha=alpha))


def _rule5(reg5, seeds, point, draws, base_draws, support_ok, missing_for, baseline_m1, alpha) -> dict:
    miss = missing_for(reg5.condition_models)
    if miss:
        return {"status": UNAVAILABLE, "missing_cells": miss, "verdict_authority": None}

    def seed_mean(t: str) -> np.ndarray:
        return np.mean(np.stack([draws[(t, s, "main")] for s in seeds], axis=0), axis=0)

    evidence = {t: _evidence(t, {s: point(t, s, "main") for s in seeds}, seed_mean(t), base_draws, support_ok, alpha)
                for t in reg5.condition_models}
    present = [t for t in LOCALS if not missing_for((t,))]
    for t in present:                         # 진단 — 공식 판정에 들지 않는다(07번 §23-2)
        evidence[t] = _evidence(t, {s: point(t, s, "main") for s in seeds}, seed_mean(t), base_draws, support_ok, alpha)
    if len(present) == len(LOCALS):
        per_seed = {}
        for s in seeds:
            vals = [point(t, s, "main") for t in LOCALS]
            per_seed[s] = None if any(x is None for x in vals) else _mean(vals)
        mean_draw = np.mean(np.stack([seed_mean(t) for t in LOCALS], axis=0), axis=0)
        evidence[LOCAL_MEAN] = _evidence(LOCAL_MEAN, per_seed, mean_draw, base_draws, support_ok, alpha)
    verdict = R5.judge_rule5_registered(reg5, baseline_m1=baseline_m1, evidence=evidence).as_dict()
    # 직렬화한 꼴을 근거에서 다시 판정한다(§25-1) — 같은 실행 안이라 채점 지문의 대조는 없음을 명시해 부른다
    R5.rejudge(json.loads(json.dumps(verdict)), reg5, scoring_sha256=None, stored_scoring_sha256=None)
    return {"status": "computed", **verdict}


def _flips(seeds, point, rq1: Mapping, missing: list[str]) -> dict:
    if missing:
        return {"status": UNAVAILABLE, "missing_cells": missing}

    def order(v: str):
        vals = {"central": [point(CENTRAL, s, v) for s in seeds], "fed": [point(FED, s, v) for s in seeds],
                "local_mean": [_mean([point(t, s, v) for t in LOCALS]) if all(point(t, s, v) is not None for t in LOCALS)
                               else None for s in seeds]}
        if any(x is None for xs in vals.values() for x in xs):
            return None
        means = {k: _mean(xs) for k, xs in vals.items()}
        return [k for k, _ in sorted(means.items(), key=lambda kv: (-kv[1], kv[0]))]

    def sign(v: str):
        r = (rq1.get(v) or {}).get("point") or {}
        x = r.get("R_bar")
        return None if x is None else (1 if x > 0 else -1 if x < 0 else 0)

    om, os_ = order("main"), order("sensitivity")
    sm, ss = sign("main"), sign("sensitivity")
    return {"status": "computed", "ranking_main": om, "ranking_sensitivity": os_,
            "ranking_changed": None if om is None or os_ is None else om != os_,
            "recovery_sign_main": sm, "recovery_sign_sensitivity": ss,
            "recovery_sign_changed": None if sm is None or ss is None else sm != ss}


def _addendum(seeds, point, eval_doc: Mapping) -> dict:
    d_min = ((eval_doc.get("population") or {}).get("sample") or {}).get("full_addendum_d_min")
    s = seeds[0]
    c = point(CENTRAL, s, "main")
    ls = [point(t, s, "main") for t in LOCALS]
    if c is None or any(x is None for x in ls):
        return {"status": UNAVAILABLE, "seed": str(s), "d_min": d_min}
    d = float(c) - _mean(ls)
    return {"status": "computed", "seed": str(s), "D_point": d, "d_min": d_min, "full_addendum_required": bool(d < d_min),
            "rule": "시드 1 표본의 D = C − L̄ 점추정이 등록 문턱보다 작으면 전량을 덧붙인다 — 이 산출은 표시만 한다(25번 §3-2)"}


def _not_judged(key: str, scores, side, W, w0, ids, alpha) -> dict:
    """표본에서 판정하지 않는 것 — 판정 불가 표시와 함께, 낼 수 있는 것은 점추정과 구간만(25번 §3-3)."""
    if key == "al_subset_macro_m1":
        mats = {i: side[i].stratum.split("|", 1)[0] for i in ids}
        if not set(mats.values()) <= set(MATERIALS):
            return {"status": UNAVAILABLE, "reason": "곁 파일의 층 이름에서 재질을 읽지 못했다"}
        mask = np.array([mats[i] == "AL" for i in ids], dtype=np.float64)
        per = {}
        for (tag, s), sc in sorted(scores.items()):
            pt = score_m1(sc.table, w0 * mask)
            if pt.classes_in_macro:
                d, ok = m1_draws(sc.table, W * mask[None, :], pt.classes_in_macro)
            else:
                d, ok = np.full(W.shape[0], np.nan), np.zeros(W.shape[0], dtype=bool)
            per[cell_name(tag, s)] = {"point": pt.macro_f1, "label": pt.label, "classes_in_macro": list(pt.classes_in_macro),
                                      "interval": _interval(np.where(ok, d, np.nan), alpha=alpha,
                                                            reasons={"support": int((~ok).sum())})}
        return {"status": "판정 불가(표본) — 점추정과 구간만 싣는다(25번 §3-3)", "n_images": int(mask.sum()),
                "per_model": per}
    if key == "p9_equivalence":
        return {"status": "판정 불가(표본)", "computed": False,
                "reason": "출처 축(N-crop · N-tile) 칸이 엄격 경로의 입력에 없다 — 이 판은 점추정 · 구간도 내지 않는다(35번 §1)"}
    return {"status": "판정 불가(표본)", "computed": False, "reason": "이 판이 모르는 항목이다"}
