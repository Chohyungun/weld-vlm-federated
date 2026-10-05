"""`map_50` 독립 구현 검산 실행기 (53번 미니스펙 · 게이트 57번, 2026-09-17).

    python scripts/probe/map50_independent.py --root outputs/main_d --seeds 1,2,3 \\
        --artifact score_cells_v3.json --dest outputs/main_d/seed3set

v3 칸 × 시드 15개의 `map_50` 을 **두 번째 구현**(`evaluation/metrics/map_independent.py`)으로 다시 내고
허용 차 1e−12(53번 §4)로 맞댄다. 두 모드를 모두 돌리고 둘 다 판정한다(57번).

- **S(공유 입력)** — 기존 `load_population`·`read_records`·`to_coco_xywh` 결과를 새 알고리즘에 넣는다. 진단용이다.
- **I(독립 입력)** — 새 모듈이 동결 스냅샷과 하한 레코드를 직접 읽는다. **채택 조건을 닫는 것은 I 모드다.**
- **pycocotools 재계산**(진단) — S 입력으로 `COCOeval` 을 다시 돌려 v3 값이 재현되는지, 카테고리별 AP 가 얼마인지 본다.

실행기가 맡는 것(57번·서명 문서 3판): 경로 결정 · 입력 해시 30개 전후 대조 · 스냅샷 무결성(`ensure_verified`,
57번 00:15 판정 3) · 채점 코드 파일별 대조 · 판정 · 출력(판 번호, 덮어쓰기 거부, 배타 생성).
**채점 기준과 v3 산출물은 바꾸지 않는다.** 검산 결과는 대표 채택 조건의 판정에만 쓴다(53번 §9). GPU 무접촉.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.provenance import (
    artifact_version,
    hash_files,
    scorer_code_digest,
    versioned_output,
    write_new_text,
)
from evaluation.recovery_ci import (
    POINT_TOLERANCE,
    ArtifactNameMismatch,
    read_artifact,
    recovery_from,
)

DET_TAGS = ("sep_central", "sep_local_C1", "sep_local_C2", "sep_local_C3", "sep_fed")
LOCALS = ("sep_local_C1", "sep_local_C2", "sep_local_C3")

TOLERANCE = 1e-12
"""53번 §4 판정 허용 차. 근거는 `evaluation.recovery_ci.POINT_TOLERANCE` 와 같다(37번 §8-3) — 두 값이 갈리면 멈춘다."""

S_PATH_FILES = (
    "evaluation/cells.py",               # load_population
    "evaluation/eval_set.py",            # read_manifest · read_gold · ensure_verified
    "evaluation/adapters.py",            # read_records
    "evaluation/schema.py",              # PredictionRecord
    "evaluation/metrics/localization.py",  # to_coco_xywh · build_cocoeval
    "evaluation/params.py",              # ScoringParams · CLASS_NAMES
    "data/label_map.py",                 # iso_code 사상
    "data/manifest_io.py",               # verify_snapshot
)
"""S 모드 입력과 pycocotools 재계산이 지나는 채점 코드. 이 파일들이 v3 산출물의 파일별 해시와 같아야 S 모드가
v3 와 **같은 입력**을 새 알고리즘에 넣는다고 말할 수 있다(C 서명 문서 3판: 합산 지문은 새 모듈 때문에 바뀐다)."""

EXPECTED_ADDED = ("evaluation/metrics/map_independent.py",)
"""v3 이후 채점기 지문 대상에 **더해진** 파일로 기대하는 것 — 새 모듈 하나뿐이다."""

RUNNER_NOTES = (
    {"id": "R-integrity", "rule": "스냅샷 무결성은 실행기가 읽기 전에 기존 `ensure_verified` 로 확인한다",
     "source": "exception", "doc": "57번 00:15 판정 3 — 해시 대조는 파싱 규약이 아니다", "independent": True},
    {"id": "R-paths", "rule": "하한 레코드 경로와 해시는 v3 산출물의 `input_records` 에서 고정한다(`_raw_s` 15개)",
     "source": "code", "doc": "경로 규칙은 `score_cells.raw_record_path` 에만 있다(D 입력 규약 #1·#2)", "independent": False},
)
"""실행기 자신이 기존 코드·예외에서 가져온 규약. 모듈의 `conventions` 와 함께 산출물에 싣는다."""


# --------------------------------------------------------------------------------------
# 순수 함수 — 시험 대상
# --------------------------------------------------------------------------------------

def tag_identity(tag: str) -> tuple[str, str | None]:
    """`sep_local_C1` → (`sep_local`, `C1`), `sep_central` → (`sep_central`, None). 레코드 `cell`·`client` 와 맞댄다."""
    if tag.startswith("sep_local_"):
        return "sep_local", tag.rsplit("_", 1)[1]
    return tag, None


def raw_record_keys(input_records: Mapping[str, str | None], seed_value: int) -> dict[str, str]:
    """v3 `input_records` 에서 칸별 **하한** 레코드 경로를 고른다. 칸마다 정확히 하나여야 한다."""
    out: dict[str, str] = {}
    for tag in DET_TAGS:
        want = f"/sweep/{tag}_raw_s{seed_value}.jsonl"
        hits = [k for k in input_records if k.replace("\\", "/").endswith(want)]
        if len(hits) != 1:
            raise SystemExit(f"{tag} 하한 레코드를 input_records 에서 하나로 정할 수 없다: {hits}")
        out[tag] = hits[0]
    return out


def check_input_hashes(expected: Mapping[str, Mapping[str, str | None]], repo: Path = REPO) -> dict:
    """시드별 `input_records`(v3 기록)와 지금 파일 해시를 맞댄다. 하나라도 다르거나 없으면 멈춘다."""
    checked = 0
    for n, recs in expected.items():
        now = hash_files([repo / k for k in recs], repo)
        for k, v in recs.items():
            if v is None or now.get(k) != v:
                raise SystemExit(f"시드 {n} 입력 {k} 의 해시가 v3 기록과 다르다: 기록 {str(v)[:16]} · 지금 "
                                 f"{str(now.get(k))[:16]} — 계산하지 않는다")
            checked += 1
    return {"n_checked": checked, "all_equal": True}


def code_drift(v3_files: Mapping[str, str], now_files: Mapping[str, str]) -> dict:
    """v3 산출물의 채점 코드 파일별 해시와 지금 트리를 맞댄다. S 경로 파일이 바뀌었으면 `s_path_unchanged=False`."""
    changed = sorted(k for k in set(v3_files) & set(now_files) if v3_files[k] != now_files[k])
    added = sorted(set(now_files) - set(v3_files))
    removed = sorted(set(v3_files) - set(now_files))
    s_changed = sorted(set(changed + removed) & set(S_PATH_FILES))
    return {
        "n_v3": len(v3_files), "n_now": len(now_files),
        "changed": changed, "added": added, "removed": removed,
        "s_path_files": list(S_PATH_FILES), "s_path_changed": s_changed,
        "s_path_unchanged": not s_changed,
        "added_as_expected": added == list(EXPECTED_ADDED),
    }


def recoveries(values: Mapping[int, Mapping[str, float]]) -> dict:
    """시드별 칸 값 → R_s·D_s·R̄ (생성기·집계와 같은 `recovery_from`, np.mean)."""
    r, d = {}, {}
    for n in sorted(values):
        m = values[n]
        r[n], d[n] = recovery_from(m["sep_central"], m["sep_fed"], [m[t] for t in LOCALS])
    rs = [r[n] for n in sorted(r)]
    rbar = float(np.mean(rs)) if all(math.isfinite(x) for x in rs) else float("nan")
    return {"R": r, "D": d, "R_bar": rbar}


def _diff(a, b) -> float:
    if a is None or b is None:
        return math.inf if (a is None) != (b is None) else 0.0
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return abs(a - b)


def judge_mode(v3: Mapping[int, Mapping[str, float]], mode: Mapping[int, Mapping[str, dict]],
               reference_ap: Mapping[int, Mapping[str, Mapping[str, float | None]]] | None,
               tol: float = TOLERANCE) -> dict:
    """한 모드의 판정(53번 §4). **전부** 허용 안이어야 통과다.

    `mode[n][tag]` = `{"map_50", "map_50_class_mean", "ap_by_class"}`. 비교 대상:
    칸 × 시드 `map_50`(v3) · 같은 값의 카테고리 평균 판(v3) · R_s·R̄(v3 값으로 낸 것) ·
    카테고리별 AP(pycocotools 재계산, 있을 때).
    """
    rows, worst = [], {"map_50": 0.0, "map_50_class_mean": 0.0, "ap_by_class": 0.0, "R": 0.0, "R_bar": 0.0}
    n_bit_equal = 0
    for n in sorted(v3):
        for t in DET_TAGS:
            got = mode[n][t]
            d50 = _diff(got["map_50"], v3[n][t])
            # v3 에는 카테고리 평균 판이 따로 없다. 각 카테고리가 101점씩이라 두 평균은 수학적으로 같으므로
            # v3 의 평탄 평균 `map_50` 과 맞댄다(53번 #16 — 끝자리만 다를 수 있다. A 61번 Minor D m-2)
            dcm = _diff(got["map_50_class_mean"], v3[n][t])
            dap = 0.0
            if reference_ap is not None:
                ref = reference_ap[n][t]
                dap = max((_diff(got["ap_by_class"].get(c), ref.get(c)) for c in ref), default=0.0)
            n_bit_equal += int(got["map_50"] == v3[n][t])
            worst["map_50"] = max(worst["map_50"], d50)
            worst["map_50_class_mean"] = max(worst["map_50_class_mean"], dcm)
            worst["ap_by_class"] = max(worst["ap_by_class"], dap)
            rows.append({"seed": n, "tag": t, "v3": v3[n][t], "map_50": got["map_50"], "diff": d50,
                         "map_50_class_mean": got["map_50_class_mean"], "diff_class_mean": dcm,
                         "diff_ap_by_class_max": dap, "bit_equal": got["map_50"] == v3[n][t]})
    rv3 = recoveries(v3)
    rmode = recoveries({n: {t: mode[n][t]["map_50"] for t in DET_TAGS} for n in v3})
    for n in rv3["R"]:
        worst["R"] = max(worst["R"], _diff(rmode["R"][n], rv3["R"][n]))
    worst["R_bar"] = _diff(rmode["R_bar"], rv3["R_bar"])
    failures = [k for k, v in worst.items() if not v <= tol]
    first = next((r for r in rows if not (r["diff"] <= tol and r["diff_class_mean"] <= tol
                                          and r["diff_ap_by_class_max"] <= tol)), None)
    return {
        "pass": not failures, "tolerance": tol, "worst_abs_diff": worst, "failed_quantities": failures,
        "n_cells": len(rows), "n_bit_equal_map_50": n_bit_equal,
        "recovery_v3": {"R_by_seed": {str(k): v for k, v in rv3["R"].items()}, "R_bar": rv3["R_bar"]},
        "recovery_mode": {"R_by_seed": {str(k): v for k, v in rmode["R"].items()}, "R_bar": rmode["R_bar"],
                          "D_by_seed": {str(k): v for k, v in rmode["D"].items()}},
        "first_mismatch": first, "rows": rows,
    }


def overall_verdict(s: dict | None, i: dict | None) -> dict:
    """57번: **채택 조건을 닫는 것은 I 모드다.** S 는 진단이다. 둘이 갈리면 그 사실을 적는다."""
    closed = bool(i and i["pass"])
    note = None
    if s is not None and i is not None and s["pass"] != i["pass"]:
        note = ("S·I 판정이 갈린다 — I 만 불일치면 입력 규약 쪽, S 만 불일치면 공유 입력 경로 쪽을 먼저 좁힌다. "
                "어느 쪽이 맞다고 적기 전에 원인을 보고한다(57번)")
    return {"S_pass": None if s is None else s["pass"], "I_pass": None if i is None else i["pass"],
            "adoption_condition_closed_by_I": closed, "note": note,
            "decision": "대표 채택 판정은 이 검산의 몫이 아니다. 불일치가 있으면 채택을 다시 연다(57번)"}


def _result_view(res) -> dict:
    return {"map_50": float(res.map_50), "map_50_class_mean": float(res.map_50_class_mean),
            "ap_by_class": {k: (None if v is None else float(v)) for k, v in res.ap_by_class.items()},
            "counts": res.counts}


# --------------------------------------------------------------------------------------
# 무거운 단계 — 실데이터
# --------------------------------------------------------------------------------------

def _scoring_params(art: dict, root: Path, n: int):
    from evaluation.params import ScoringParams

    p = art["params"]
    return ScoringParams(snapshot=Path(p["snapshot"]), pilot=Path(p["pilot"]), out=root / f"seed{n}",
                         seed=int(p["seed"]), profile=p["profile"])


def _shared_inputs(pop, raw_path: Path):
    """S 모드 입력 — 기존 채점과 같은 변환(`score_records` 의 None→0.0 · bbox 있는 결함만 · `to_coco_xywh`)."""
    from evaluation.adapters import read_records
    from evaluation.metrics.localization import to_coco_xywh

    recs = read_records(raw_path.read_text(encoding="utf-8").splitlines())
    pred_xyxy = {r.image_id: [(d.iso_code, tuple(d.bbox_px), d.score or 0.0) for d in r.defects if d.bbox_px]
                 for r in recs}
    gold_xywh = {i: [(c, to_coco_xywh(b)) for c, b in v] for i, v in pop.gold_boxes.items()}
    pred_xywh = {i: [(c, to_coco_xywh(b), s) for c, b, s in v] for i, v in pred_xyxy.items()}
    return gold_xywh, pred_xywh, pred_xyxy


def _pycocotools_reference(pop, pred_xyxy) -> tuple[float, dict[str, float | None]]:
    """S 입력으로 `COCOeval` 재계산 — `map_50`(= `stats[1]`)과 카테고리별 AP(`precision[0,:,k,all,100]` 평균)."""
    from evaluation.metrics.localization import build_cocoeval

    ev, _, _ = build_cocoeval(pred_xyxy, pop.gold_boxes, pop.classes)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.accumulate()
        ev.summarize()
    prec = ev.eval["precision"][0, :, :, 0, 2]
    ap = {c: (float(np.mean(prec[:, k])) if bool((prec[:, k] > -1).all()) else None)
          for k, c in enumerate(pop.classes)}
    return float(ev.stats[1]), ap


def _peak_memory_gb() -> float | None:
    try:
        import psutil

        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 30, 2)
    except Exception:  # noqa: BLE001 — 측정 실패는 기록 누락일 뿐이다
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/main_d")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--artifact", default="score_cells_v3.json")
    ap.add_argument("--dest", default="outputs/main_d/seed3set")
    ap.add_argument("--label-map", default="configs/label_map.yaml")
    ap.add_argument("--modes", default="S,I", help="돌릴 모드. 기본은 둘 다(57번)")
    args = ap.parse_args()

    if TOLERANCE != POINT_TOLERANCE:
        raise SystemExit("판정 허용 차가 POINT_TOLERANCE 와 다르다 — 53번 §4 근거를 먼저 맞춰라")
    modes = [m.strip().upper() for m in args.modes.split(",") if m.strip()]
    if not set(modes) <= {"S", "I"} or not modes:
        raise SystemExit(f"--modes 는 S·I 중에서 고른다: {args.modes}")

    # ---- 1. 출력 이름 — 옛 판을 덮지 않는다. 계산 전에 확인한다
    root, dest = Path(args.root), Path(args.dest)
    try:
        out = versioned_output(dest, "map50_independent", args.artifact)
    except (ValueError, FileExistsError) as e:
        raise SystemExit(f"{e} (dest={dest})") from None

    t0 = time.perf_counter()
    seeds = [int(s) for s in args.seeds.split(",")]

    # ---- 2. v3 산출물 — 기준값 · 입력 해시 · 채점 코드 파일별 해시
    arts, entries = {}, {}
    for n in seeds:
        try:
            arts[n], entries[n] = read_artifact(root / f"seed{n}" / args.artifact)
        except ArtifactNameMismatch as e:
            raise SystemExit(f"시드 {n}: {e}") from None
    snapshots = {Path(a["params"]["snapshot"]).as_posix().replace("\\", "/") for a in arts.values()}
    if len(snapshots) != 1:
        raise SystemExit(f"시드마다 스냅샷이 다르다: {snapshots}")
    snapshot = Path(next(iter(snapshots)))
    v3 = {n: {t: float(arts[n]["threshold_independent"]["per_tag"][t]["map_50"]) for t in DET_TAGS} for n in seeds}
    expected_inputs = {n: dict(arts[n]["input_records"]) for n in seeds}
    raw_keys = {n: raw_record_keys(expected_inputs[n], int(arts[n]["params"]["seed"])) for n in seeds}
    v3_files = arts[seeds[0]]["scorer_code"]["files"]
    if any(arts[n]["scorer_code"]["files"] != v3_files for n in seeds):
        raise SystemExit("시드마다 채점 코드 파일별 해시가 다르다 — 한 코드 상태로 채점된 판이 아니다")

    # ---- 3. 입력 해시(전) · 스냅샷 무결성 · 코드 대조
    inputs_before = check_input_hashes(expected_inputs)
    from evaluation.eval_set import ensure_verified

    snapshot_digest = ensure_verified(REPO / snapshot)
    code_now = scorer_code_digest()
    drift = code_drift(v3_files, code_now["files"])
    print(f"입력 해시 {inputs_before['n_checked']}개 v3 와 일치 · 스냅샷 {snapshot_digest[:16]} · "
          f"S 경로 코드 v3 와 같음 {drift['s_path_unchanged']} · 더해진 파일 {drift['added']} · 바뀐 파일 {drift['changed']}")
    if not drift["s_path_unchanged"]:
        print(f"  ! S 경로 파일이 v3 이후 바뀌었다: {drift['s_path_changed']} — S 모드가 v3 입력과 같다고 말할 수 없다")

    from evaluation.metrics import map_independent as mi

    # ---- 4. S 모드 + pycocotools 재계산
    s_mode = ref_map = ref_ap = None
    if "S" in modes:
        from evaluation.cells import load_population

        s_mode, ref_map, ref_ap = {}, {}, {}
        for n in seeds:
            pop = load_population(_scoring_params(arts[n], root, n))
            s_mode[n], ref_map[n], ref_ap[n] = {}, {}, {}
            for t in DET_TAGS:
                gold_xywh, pred_xywh, pred_xyxy = _shared_inputs(pop, REPO / raw_keys[n][t])
                res = mi.map50_shared(gold_xywh, pred_xywh, pop.classes)
                s_mode[n][t] = _result_view(res)
                ref_map[n][t], ref_ap[n][t] = _pycocotools_reference(pop, pred_xyxy)
                print(f"[S 시드 {n}] {t}: 새 구현 {res.map_50:.12f} · v3 {v3[n][t]:.12f} · "
                      f"pycocotools 재계산 {ref_map[n][t]:.12f}", flush=True)
    # ---- 5. I 모드
    i_mode = i_reports = gold_report = None
    if "I" in modes:
        gold = mi.read_gold_independent(REPO / snapshot, REPO / args.label_map)
        gold_report = gold.report
        i_mode, i_reports = {}, {}
        for n in seeds:
            i_mode[n], i_reports[n] = {}, {}
            for t in DET_TAGS:
                cell, client = tag_identity(t)
                pred = mi.read_pred_independent(REPO / raw_keys[n][t], gold,
                                                expect={"cell": cell, "client": client,
                                                        "seed": int(arts[n]["params"]["seed"])})
                res, rep = mi.map50_independent(gold, pred)
                i_mode[n][t] = _result_view(res)
                i_reports[n][t] = rep
                print(f"[I 시드 {n}] {t}: 새 구현 {res.map_50:.12f} · v3 {v3[n][t]:.12f}", flush=True)

    # ---- 6. 판정
    ref_check = None
    if ref_map is not None:
        ref_check = {"pycocotools_equals_v3": all(ref_map[n][t] == v3[n][t] for n in seeds for t in DET_TAGS),
                     "max_abs_diff": max(abs(ref_map[n][t] - v3[n][t]) for n in seeds for t in DET_TAGS)}
    s_judge = judge_mode(v3, s_mode, ref_ap) if s_mode is not None else None
    i_judge = judge_mode(v3, i_mode, ref_ap) if i_mode is not None else None
    verdict = overall_verdict(s_judge, i_judge)

    # ---- 7. 입력 해시(후)
    inputs_after = check_input_hashes(expected_inputs)

    conventions = not_independent = None
    if i_reports:
        first = i_reports[seeds[0]][DET_TAGS[0]]
        conventions = list(first["conventions"]) + list(RUNNER_NOTES)
        not_independent = sorted(set(first["not_independent"])
                                  | {c["id"] for c in RUNNER_NOTES if not c["independent"]})
        for n in seeds:
            for t in DET_TAGS:
                if i_reports[n][t]["not_independent"] != first["not_independent"]:
                    not_independent = sorted(set(not_independent) | set(i_reports[n][t]["not_independent"]))

    payload = {
        "artifact_version": out.name,
        "spec": "53번 미니스펙(A 58번 정의표 승인) · 게이트 57번(00:15 추기 포함)",
        "purpose": "대표 채택 조건 판정용 검산 — map_50 을 대체하지 않는다(53번 §9)",
        "input_artifact": args.artifact, "input_artifact_version": artifact_version(args.artifact),
        "seeds": seeds, "modes": modes, "tolerance": TOLERANCE,
        "verdict": verdict,
        "judgement": {"S": s_judge, "I": i_judge},
        "pycocotools_reference": {"check": ref_check,
                                  "map_50": None if ref_map is None else
                                  {str(n): v for n, v in ref_map.items()},
                                  "ap_by_class": None if ref_ap is None else
                                  {str(n): v for n, v in ref_ap.items()}},
        "results": {"S": None if s_mode is None else {str(n): v for n, v in s_mode.items()},
                    "I": None if i_mode is None else {str(n): v for n, v in i_mode.items()}},
        "i_mode_reports": None if i_reports is None else
        {str(n): {t: {k: v for k, v in r.items() if k != "gold"} for t, r in d.items()} for n, d in i_reports.items()},
        "i_mode_gold_report": gold_report,
        "conventions": conventions,
        "not_independent": not_independent,
        "doc_mismatch_note": "D 입력 규약 #4 — 문서(13_spec §3-4)는 좌표 정수화, 실물 하한 레코드는 실수. 파일값을 그대로 썼다",
        "inputs": {"before": inputs_before, "after": inputs_after,
                   "records_by_seed": {str(n): expected_inputs[n] for n in seeds},
                   "raw_records_used": {str(n): raw_keys[n] for n in seeds},
                   "v3_artifacts": {str(n): {k: entries[n][k] for k in ("path", "sha256")} for n in seeds},
                   "snapshot": snapshot.as_posix(), "snapshot_digest": snapshot_digest,
                   "label_map": args.label_map},
        "code": {"scorer_tree_combined": code_now["combined"], "rule": code_now["rule"],
                 "v3_combined": arts[seeds[0]]["scorer_code"]["combined"], "drift": drift,
                 "git": code_now["git"]},
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "peak_memory_gb": _peak_memory_gb(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    try:
        write_new_text(out, json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    except FileExistsError:
        raise SystemExit(f"{out} 이 계산 도중 생겼다 — 덮지 않는다") from None

    print(f"판정: S {verdict['S_pass']} · I {verdict['I_pass']} · 채택 조건(I) 닫힘 {verdict['adoption_condition_closed_by_I']}")
    for name, j in (("S", s_judge), ("I", i_judge)):
        if j is not None:
            print(f"  [{name}] 최대 |차| {j['worst_abs_diff']} · 비트 일치 {j['n_bit_equal_map_50']}/{j['n_cells']}"
                  + ("" if j["pass"] else f" · 첫 불일치 {j['first_mismatch']}"))
    if ref_check is not None:
        print(f"  pycocotools 재계산 = v3: {ref_check['pycocotools_equals_v3']} (최대 차 {ref_check['max_abs_diff']:.1e})")
    if verdict["note"]:
        print(f"  ! {verdict['note']}")
    print(f"독립 확인 아님: {not_independent}")
    print(f"저장: {out} · {payload['runtime_seconds']}s · 최대 메모리 {payload['peak_memory_gb']} GB")
    return 0


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"직렬화할 수 없다: {type(o)}")


if __name__ == "__main__":
    raise SystemExit(main())
