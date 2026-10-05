"""통합형 **채점 층** — 검증을 지난 묶음(`VerifiedBundle`)만 받는 함수들. 진입점 미니스펙 3판 1-2 나 · 채점 층(검토 10 의 M-5) ·
1-1-다 · 2 절(③ · ⑤) · 4-2.

목록은 `SCORING_LAYER` 다. 목록의 함수는 **받은 묶음마다 첫 문장에서 `require_verified` 를 부른다** — 경로 · `dict` · 레코드 목록을
받는 겹침 인자를 두지 않는다. 시험이 목록의 함수마다 `VerifiedBundle` 이 아닌 것을 넣어 `TypeError` 를 보고, 구문 트리로 첫 문장을 본다.
진입점 쪽 모듈에서 그물의 이름(`score_records` · `box_f1` · 옛 구간 함수 · 규칙 ⑤)을 부르는 함수는 모두 이 목록에 있어야 한다.

| 함수 | 무엇 |
|---|---|
| `records_from_verified` | 묶음의 줄 → 1.4 레코드(`adapt_unified_main` 을 부르는 유일한 자리) |
| `score_echo` | 에코 채점 — M1 계수(`box_f1`)로 FP = FN = 0 · TP = 기대 수 · 좌표 최대 차 ≤ 등록 허용. **다섯 수**를 싣는다(배관의 항등 검사라 값이다) |
| `check_literal` | 문자 일치 — 참조 파서로 `text` 를 다시 읽고 **독립 역변환**으로 원본 픽셀을 다시 내 줄의 좌표와 맞댄다 |
| `judge_rehearsal_gates` | 리허설 통과 조건 ③(스키마 · 실패율 기록) · ⑤(지연 · 실제 생성 길이) |
| `rehearsal_metric_smoke` | 리허설의 모델 묶음에 M1 계산을 끝까지 돌리고 **끝났는가 · 키 목록 · 미정의 사유만** 낸다 — 값은 싣지 않는다(4-2) |
| `adversarial_probe` | ③ 의 "0 이 나오면 의심한다" — 고의로 위반하는 한 장 셋이 **실제로 거부되는지**(3판 2-1). 리허설 표본이 아니라 합성이고 실패율에 넣지 않는다 |

**리허설 산출물에 지표 값을 싣지 않는다**(4-2). 이 목록의 함수가 내는 수는 배관의 검사(에코의 다섯 수 · 실패율 · 지연 요약)뿐이다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.adapters_v14 import OUTCOME_OK, UpstreamContractError, adapt_unified_main, reference_parse
from evaluation.bundle_v14 import MODE_ECHO, MODE_MODEL, VerifiedBundle, require_verified
from evaluation.coord_check import inverse_px
from evaluation.metrics.box_f1 import build_counts
from evaluation.reject_v14 import RejectCode

SCORING_LAYER: tuple[str, ...] = ("records_from_verified", "score_echo", "check_literal", "judge_rehearsal_gates",
                                  "adversarial_probe", "rehearsal_metric_smoke")
"""채점 층의 함수 이름. 진입점의 프레임 진단(`run_frame_diag`) · 엄격의 등록 지표(`score_registered`) · 규칙 ⑤(`judge_rule5_for`)는
그 구현이 들어오는 회차에 여기에 더한다 — 없는 함수를 목록에 두지 않는다."""

LENGTH_ALERT_RATIO = 0.05
"""한도에 닿은 비율이 이것을 넘으면 **경보**다 — 불통과가 아니다(쓰는 쪽 2판 3절 ⑤). 사람이 본다."""


# ── 레코드 ──────────────────────────────────────────────────────────────────

def records_from_verified(vb: VerifiedBundle, *, coupling_rule: str, known_iso_codes: Iterable[str],
                          scoring_iso_codes: Sequence[str], image_size: Mapping | None = None):
    """검증을 지난 묶음의 줄 → 1.4 레코드. 상류 줄이 계약을 어기면 `UpstreamContractError`(배관 고장)."""
    vb = require_verified(vb)
    return adapt_unified_main(list(vb.lines), tag=vb.tag, seed=vb.seed_value, coupling_rule=coupling_rule,
                              known_iso_codes=known_iso_codes, scoring_iso_codes=scoring_iso_codes,
                              image_size=image_size)


# ── 에코 채점 ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EchoOutcome:
    """에코의 판정과 다섯 수. 수는 모델의 지표가 아니라 배관의 항등 검사다(3판 1-1-다)."""

    status: str
    n_expected: int
    tp: int
    fp: int
    fn: int
    max_coord_diff: float
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"status": self.status, "n_expected": self.n_expected, "tp": self.tp, "fp": self.fp, "fn": self.fn,
                "max_coord_diff": self.max_coord_diff, "reasons": list(self.reasons)}


def _max_pair_diff(gold: list[Sequence[float]], pred: list[Sequence[float]]) -> float:
    """같은 클래스 안에서 좌표 차(체비쇼프)가 가장 작게 짝을 지은 뒤 그 짝들의 최대 차."""
    if not gold or not pred:
        return 0.0
    cost = np.array([[max(abs(a - b) for a, b in zip(g, p, strict=True)) for p in pred] for g in gold])
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].max()) if len(r) else 0.0


def score_echo(vb: VerifiedBundle, gold: Mapping[str, Sequence[tuple[str, Sequence[float]]]], *, classes: Sequence[str],
               max_coord_diff: float, coupling_rule: str, known_iso_codes: Iterable[str]) -> EchoOutcome:
    """에코 묶음을 M1 계수로 채점한다 — 통과는 FP = FN = 0 · TP = 기대 수(정답 박스 수) · 좌표 최대 차 ≤ 등록 허용(§14-2 8).

    `gold` 는 **목적의 분할로 좁힌 뷰**에서 만든 이미지 → `[(ISO 코드, [x1, y1, x2, y2])]` 다. 묶음의 모든 id 가 있어야 한다.
    """
    vb = require_verified(vb)
    if vb.mode != MODE_ECHO:
        raise ValueError("에코 채점은 에코 묶음만 받는다")
    missing = [i for i in vb.image_ids if i not in gold]
    if missing:
        raise ValueError(f"정답 뷰에 없는 에코 id 가 {len(missing)}개 — 뷰를 목록으로 만들었는지 본다")
    rep = records_from_verified(vb, coupling_rule=coupling_rule, known_iso_codes=known_iso_codes,
                                scoring_iso_codes=list(classes))
    pred = {r.image_id: [(d.iso_code, list(d.bbox_px)) for d in r.defects if d.bbox_px is not None] for r in rep.records}
    sub = {i: [(c, list(b)) for c, b in gold[i] if c in classes] for i in vb.image_ids}
    table = build_counts(pred, sub, list(classes))
    tp, fp, fn = int(table.tp.sum()), int(table.fp.sum()), int(table.fn.sum())
    n_expected = sum(len(v) for v in sub.values())
    diff = 0.0
    for i in vb.image_ids:
        for code in classes:
            g = [b for c, b in sub[i] if c == code]
            p = [b for c, b in pred.get(i, []) if c == code]
            diff = max(diff, _max_pair_diff(g, p))
    reasons = []
    if fp or fn:
        reasons.append("FP 또는 FN 이 0 이 아니다")
    if tp != n_expected:
        reasons.append("TP 가 사전에 센 정답 박스 수와 다르다")
    if not diff <= max_coord_diff:
        reasons.append("좌표 최대 차가 등록 허용을 넘는다")
    return EchoOutcome("pass" if not reasons else "fail", n_expected, tp, fp, fn, diff, tuple(reasons))


# ── 문자 일치 ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LiteralOutcome:
    """문자 일치의 결과 — 건수와 어긋난 줄 번호. 어긋나면 그 칸을 거부한다(3판 1-1-다)."""

    n_checked: int
    n_skipped_record_fail: int
    mismatches: tuple[tuple[int, str], ...] = ()

    @property
    def status(self) -> str:
        return "pass" if not self.mismatches else "rejected"

    def as_dict(self) -> dict:
        return {"status": self.status, "n_checked": self.n_checked,
                "n_skipped_record_fail": self.n_skipped_record_fail,
                "mismatches": [{"line": n, "code": c} for n, c in self.mismatches]}


def check_literal(vb: VerifiedBundle, *, known_iso_codes: Iterable[str], scoring_iso_codes: Sequence[str],
                  tolerance: float, orig_wh_of: Mapping[str, Sequence[int]] | None = None) -> LiteralOutcome:
    """생성문을 참조 파서로 다시 읽고 좌표를 **독립 역변환**해 줄의 `bbox_px` 와 허용오차로 맞댄다(§14-2 7).

    어긋남은 둘로 적는다 — 유지 · 폐기 판정(항목 수 · 코드 · 폐기 수)이 다르면 `PARSER_DISAGREEMENT`, 좌표가 다르면 `LITERAL_MATCH_FAILED`.
    `ABS_ORIG` 는 역변환이 항등이라 원본 크기가 필요 없다. 다른 규약은 `orig_wh_of` 를 준다(그 칸의 분할 것만).
    """
    vb = require_verified(vb)
    known = set(known_iso_codes)
    scoring = list(scoring_iso_codes)
    out: list[tuple[int, str]] = []
    n_checked = n_skip = 0
    for n, line in enumerate(vb.lines, 1):
        row = json.loads(line)
        if row.get("parse_error") is not None:
            n_skip += 1
            continue
        n_checked += 1
        ref = reference_parse(row.get("text", ""), known_iso_codes=known, scoring_iso_codes=scoring)
        upstream = (row.get("bbox_px_parsed") or {}).get("defects") or []
        if ref.outcome != OUTCOME_OK or len(ref.kept) != len(upstream) \
                or ref.dropped != row.get("n_bad_items_dropped", 0) \
                or [k[0] for k in ref.kept] != [d.get("iso_code") for d in upstream]:
            out.append((n, RejectCode.PARSER_DISAGREEMENT.value))
            continue
        space = row.get("coord_space")
        orig = orig_wh_of.get(row["image_id"]) if orig_wh_of is not None else None
        if space != "ABS_ORIG" and orig is None:
            raise ValueError(f"규약 {space} 의 역변환에는 원본 크기가 있어야 한다")
        for k, d in zip(ref.kept, upstream, strict=True):
            back = inverse_px(k[1:], space, orig_wh=orig or (0, 0), model_wh=row.get("model_input_wh"))
            if not all(abs(a - float(b)) <= tolerance for a, b in zip(back, d.get("bbox_px") or (), strict=True)):
                out.append((n, RejectCode.LITERAL_MATCH_FAILED.value))
                break
    return LiteralOutcome(n_checked, n_skip, tuple(out))


# ── 리허설 통과 조건 ③ · ⑤ ───────────────────────────────────────────────────

@dataclass
class GateJudgment:
    """③ 또는 ⑤ 의 판정. `checks` 는 번호(③-1 …)마다 통과 여부와 사유."""

    gate: str
    checks: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)

    @property
    def status(self) -> str:
        return "pass" if all(v["pass"] for v in self.checks.values()) else "fail"

    def as_dict(self) -> dict:
        return {"gate": self.gate, "status": self.status, "checks": self.checks, "summary": self.summary,
                "alerts": list(self.alerts)}


def _quantiles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"n": int(arr.size), "mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)), "max": float(arr.max()), "min": float(arr.min())}


def judge_rehearsal_gates(vbs: Sequence[VerifiedBundle], *, coupling_rule: str, known_iso_codes: Iterable[str],
                          scoring_iso_codes: Sequence[str]) -> dict[str, GateJudgment]:
    """리허설의 ③ · ⑤(3판 2 절). **모델 묶음만** 센다 — 에코는 모델을 부르지 않은 줄이다.

    ③ 은 "실패율이 0" 이 아니라 **기록된다**가 통과다. ⑤ 는 지연 · 생성 길이의 요약이 있는 것이 통과다 — 한도 도달 비율이
    5 % 를 넘으면 경보(불통과가 아니다)이고, **⑤ 가 생성 한도(5c)를 정하지 않는다.**
    """
    models = [require_verified(vb) for vb in vbs]
    models = [vb for vb in models if vb.mode == MODE_MODEL]
    known = list(known_iso_codes)
    g3 = GateJudgment("③")
    n_lines = n_record_fail = n_clause = 0
    schema_fail = vocab_fail = 0
    for vb in models:
        try:
            rep = records_from_verified(vb, coupling_rule=coupling_rule, known_iso_codes=known,
                                        scoring_iso_codes=scoring_iso_codes)
        except UpstreamContractError as exc:
            if "parse_error" in str(exc):
                vocab_fail += 1
            else:
                schema_fail += 1
            continue
        n_lines += rep.n_lines
        n_record_fail += sum(rep.upstream_failures.values())
        n_clause += sum(1 for r in rep.records if r.clause_error)
    g3.checks["③-1"] = {"pass": schema_fail == 0, "n_bundles_failed": schema_fail}
    g3.checks["③-2"] = {"pass": True, "note": "줄 수 · id 수열은 묶음 검증이 이미 본다"}
    g3.checks["③-3"] = {"pass": vocab_fail == 0, "n_bundles_failed": vocab_fail}
    rates = {"parse_fail_rate": (n_record_fail / n_lines) if n_lines else None,
             "clause_error_rate": (n_clause / n_lines) if n_lines else None,
             "combined_violation_rate": ((n_record_fail + n_clause) / n_lines) if n_lines else None}
    g3.checks["③-4"] = {"pass": n_lines > 0 and all(v is not None for v in rates.values())}
    recomputed = None if not n_lines else rates["parse_fail_rate"] + rates["clause_error_rate"]
    g3.checks["③-5"] = {"pass": recomputed is not None
                        and math.isclose(recomputed, rates["combined_violation_rate"], rel_tol=0, abs_tol=1e-12)}
    g3.summary = {"n_lines": n_lines, **rates}

    g5 = GateJudgment("⑤")
    lat, ntok, stops, batch = [], [], {}, set()
    bad_fields = 0
    for vb in models:
        batch.add((vb.sidecar.get("batch_size"), vb.sidecar.get("device")))
        for line in vb.lines:
            row = json.loads(line)
            v, t = row.get("latency_ms"), row.get("n_new_tokens")
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                    and type(t) is int and t >= 0):
                bad_fields += 1
                continue
            lat.append(float(v))
            ntok.append(t)
            stops[row.get("gen_stop")] = stops.get(row.get("gen_stop"), 0) + 1
    g5.checks["⑤-1"] = {"pass": bad_fields == 0 and bool(lat), "n_lines_bad": bad_fields}
    g5.checks["⑤-2"] = {"pass": bool(lat)}
    total = sum(stops.values())
    length_ratio = (stops.get("length", 0) / total) if total else None
    g5.checks["⑤-3"] = {"pass": total > 0}
    need_def = [b for b in batch if isinstance(b[0], int) and b[0] > 1]
    basis = {vb.sidecar.get("latency_basis") for vb in models}
    g5.checks["⑤-4"] = {"pass": bool(batch) and (not need_def or basis <= {"per_call", "per_image"}),
                        "reason": None if not need_def else "배치 > 1 — 곁 파일 latency_basis 로 장당의 정의를 받는다"}
    g5.summary = {"latency_ms": _quantiles(lat) if lat else None, "n_new_tokens": _quantiles(ntok) if ntok else None,
                  "gen_stop_counts": stops, "length_ratio": length_ratio,
                  "batch_device": sorted([list(b) for b in batch], key=str),
                  "note": ("이 비율은 모델이 실제로 생성한 길이가 한도에 닿은 비율이다. 저장된 타깃의 길이 분포가 아니다. "
                           "리허설 소표본의 값이다 — 전량 · 참여자별로 일반화하지 않는다")}
    if length_ratio is not None and length_ratio > LENGTH_ALERT_RATIO:
        g5.alerts.append("한도 도달 비율이 5 % 를 넘는다 — 경보(불통과가 아니다). 사람이 본다")
    return {"③": g3, "⑤": g5}


# ── ③ 의 적대적 한 장 ────────────────────────────────────────────────────────

def adversarial_probe(vb: VerifiedBundle, *, coupling_rule: str, known_iso_codes: Iterable[str],
                      scoring_iso_codes: Sequence[str]) -> dict:
    """검증을 지난 **모델 묶음의 첫 줄**을 본으로 위반 셋을 만들어 검사가 **실제로 거부하는지** 본다(3판 2-1).

    | 섞는 것 | 걸려야 하는 곳 |
    |---|---|
    | 필수 필드(`gen_stop`) 하나가 빠진 줄 | ③-1 — 어댑터가 배관 고장으로 멈춘다 |
    | `parse_error` 어휘 밖 값 | ③-3 — 어댑터가 멈춘다 |
    | 좌표 규약이 다른 줄 | 묶음 검증의 줄 검사(`COORD_SPACE_MISMATCH`) |

    이 셋은 리허설 표본이 아니라 합성 픽스처다 — **실패율 통계에 넣지 않고 검사 발화 여부만** 싣는다. 섞은 자리는 산출물에 적는다.
    """
    from evaluation.bundle_v14 import _check_line_env
    from evaluation.schema_v14 import tag_parts

    vb = require_verified(vb)
    if vb.mode != MODE_MODEL:
        raise ValueError("적대적 한 장은 모델 묶음의 줄을 본으로 만든다")
    base = json.loads(vb.lines[0])
    fired: dict[str, bool] = {}

    def adapter_rejects(row: dict) -> bool:
        try:
            adapt_unified_main([json.dumps(row, ensure_ascii=False)], tag=vb.tag, seed=vb.seed_value,
                               coupling_rule=coupling_rule, known_iso_codes=known_iso_codes,
                               scoring_iso_codes=scoring_iso_codes)
        except UpstreamContractError:
            return True
        return False

    missing = dict(base)
    missing.pop("gen_stop", None)
    fired["missing_required_field"] = adapter_rejects(missing)
    fired["parse_error_out_of_vocab"] = adapter_rejects({**base, "parse_error": "not_in_vocab",
                                                         "bbox_px_parsed": None})

    class _Reg:
        coord_space = vb.sidecar["coord_space"]
    other = "NORM_1000" if base.get("coord_space") != "NORM_1000" else "ABS_ORIG"
    rej: list = []
    _check_line_env([json.dumps({**base, "coord_space": other}, ensure_ascii=False)], vb.sidecar, _Reg(),
                    vb.files, rej)
    fired["coord_space_mismatch"] = any(r.code == RejectCode.COORD_SPACE_MISMATCH for r in rej)
    cell, _client = tag_parts(vb.tag)
    return {"status": "pass" if all(fired.values()) else "fail", "fired": fired,
            "template": {"tag": vb.tag, "cell": cell, "line": 1},
            "note": "합성 픽스처 셋 — 리허설 표본이 아니다. 실패율 통계에 넣지 않았다"}


# ── 리허설의 지표 — 값을 싣지 않는다(4-2) ────────────────────────────────────

def rehearsal_metric_smoke(vb: VerifiedBundle, gold: Mapping[str, Sequence[tuple[str, Sequence[float]]]], *,
                           classes: Sequence[str], coupling_rule: str, known_iso_codes: Iterable[str]) -> dict:
    """모델 묶음에 M1 계산을 **끝까지 돌리고** 끝났는가 · 키 목록 · 미정의 사유 어휘만 돌려준다 — **값은 싣지 않는다**(3판 4-2).

    값을 보면 그 수로 주 지표(1b)를 고르게 된다 — 원칙을 사람의 자제에 맡기지 않는 쪽을 골랐다. 이 함수의 반환에는 수가 없다.
    """
    from evaluation.metrics.box_f1 import score_m1

    vb = require_verified(vb)
    if vb.mode != MODE_MODEL:
        raise ValueError("리허설의 지표 확인은 모델 묶음만 받는다")
    try:
        rep = records_from_verified(vb, coupling_rule=coupling_rule, known_iso_codes=known_iso_codes,
                                    scoring_iso_codes=list(classes))
        pred = {r.image_id: [(d.iso_code, list(d.bbox_px)) for d in r.defects if d.bbox_px is not None]
                for r in rep.records}
        sub = {i: [(c, list(b)) for c, b in gold[i] if c in classes] for i in vb.image_ids}
        report = score_m1(build_counts(pred, sub, list(classes)))
    except Exception as exc:  # 리허설은 배관을 본다 — 어느 예외든 "끝나지 않았다" 로 적는다
        return {"metric": "M1", "completed": False, "error": type(exc).__name__}
    keys = sorted(k for k in vars(report) if not k.startswith("_"))
    return {"metric": "M1", "completed": True, "keys": keys,
            "undefined_reason": report.undefined_reason,
            "note": "값을 싣지 않는다 — 리허설은 계산이 끝났는지와 산출의 모양만 본다(진입점 미니스펙 3판 4-2)"}
