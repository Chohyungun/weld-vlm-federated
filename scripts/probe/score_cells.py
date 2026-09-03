"""**다섯 칸 채점 단일 진입점.** 77번 과제 6.

    uv run python scripts/probe/score_cells.py score          # 다섯 칸 채점 + 65·66 대조
    uv run python scripts/probe/score_cells.py gate           # 게이트 재대조 (A 상수 도착 시)
    uv run python scripts/probe/score_cells.py gate --gate 0.5940
    uv run python scripts/probe/score_cells.py predict        # 스윕용 하한 추론 (CPU)
    uv run python scripts/probe/score_cells.py sweep          # 임계 스윕 + RQ2 판정

65·66번은 스크립트 두 벌로 나뉘어 있었고 임계·시드·칸 목록이 각자 살아 있었다.
지금은 전부 `evaluation.params`·`evaluation.cells` 에서 온다. 옛 스크립트는 이 파일로
넘기는 얇은 껍데기로 남겨 두었다 — 65·66 보고서의 재현 명령을 깨지 않기 위해서다.

**리팩토링의 통과 조건은 값 불변이다.** `score` 가 65번(`score_detection_v1.json`)과
66번(`score_all_cells_v1.json`)의 저장 지표를 완전 일치로 대조하고, 어긋나면 0 이
아닌 코드로 죽는다. 저장 파일을 덮어쓰지 않는다 — 증거를 지우면 대조가 무의미해진다.

**차단은 종료 코드다** (13번 D-7). `run_scoring_gates` 의 `blocking_failures` 가 비어
있지 않으면 산출물은 증거로 남기되 `score` 가 2 로 죽는다. 이전 판은 출력만 하고 0 을
돌려줘 "차단 ○" 열이 기록 이상이 아니었다. 종료 코드 표는 `evaluation/README.md`.

`score` 는 전역 지표 옆에 **같은 산출물 안에** id 구간 층화 블록을 싣고(총괄 판정 6 ·
13번 D-1), 채점 디렉터리에 `prereg_recomputed_v1.json` 이 없으면 동결본에서 재산출해
선배치한다(13번 D-8 파생). 둘 다 게이트가 매 채점마다 확인한다.

GPU 를 쓰지 않는다. `predict` 만 CPU 추론이고 나머지는 순수 채점이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import csv

from data.label_map import load_label_map
from evaluation.cells import (
    ALL_TAGS,
    DET_TAGS,
    UNI_TAGS,
    Population,
    gate_check,
    load_detection_records,
    load_population,
    load_unified_records,
    record_path,
    regression_diffs,
    score,
)
from evaluation.gates import GateContext, run_scoring_gates
from evaluation.params import (
    FROZEN_SNAPSHOT,
    ScoringParams,
    add_common_args,
    params_from_args,
    resolve_gate_status,
)
from evaluation.probes.metadata_probe import MetaSample, trivial_bound
from evaluation.probes.p9_runner import contexts_from_snapshot, p9_all_cells
from evaluation.score import coord_health, failure_breakdown
from evaluation.strata import (
    DEFAULT_K,
    ID_GRANULARITY,
    SHORTCUT_TAG,
    STRATUM_AXIS,
    stratified_table,
)

EXIT_OK = 0
EXIT_REGRESSION = 1
"""저장 지표(65·66번) 대조 불일치 — 리팩토링이 값을 바꿨다."""
EXIT_GATE_BLOCKED = 2
"""차단 게이트 실패. 산출물은 쓰이지만 **이 채점을 결과로 쓰지 마라.**"""


def exit_code(gates: dict, regressions: dict) -> tuple[int, str]:
    """`score` 의 종료 코드와 사유. **차단 게이트가 대조 불일치보다 앞선다** — 게이트
    실패는 채점 신뢰의 전제가 빈 것이고, 대조 불일치는 그 위의 정의 변경 문제다."""
    blocked = list(gates.get("blocking_failures") or [])
    if blocked:
        return EXIT_GATE_BLOCKED, (
            f"차단 게이트 실패 {blocked} — 산출물은 증거로 남겼으나 이 채점을 결과로 쓰지 마라")
    bad = [k for k, v in regressions.items() if v.get("checked") and not v["identical"]]
    if bad:
        return EXIT_REGRESSION, f"저장 지표 대조 불일치 {bad}"
    return EXIT_OK, "차단 실패 없음 · 저장 지표 대조 일치"


def stratified_block(pop: Population, by_cell: dict,
                     frozen: Path = Path(FROZEN_SNAPSHOT)) -> dict:
    """전역 지표 옆에 **같은 산출물 안에** 층화 지표를 싣는다 (총괄 판정 6 · 13번 D-1).

    절단점은 동결본 train+val 에서만 온다(A 의 `data.id_strata`). 채점 모집단이 파일럿
    부분집합이든 동결 평가셋 전량이든 같은 절단점을 받는다. K 사다리 전체를 내되 기본
    K 를 표시한다 — `stratified_scoring` 게이트가 기본 K 의 표와 지름길 행을 본다.
    구간별 상세는 `stratified_compare.py --ladder` 가 낸다(여기서는 표만).
    """
    gold = {i: sorted(pop.gold_codes.get(i, ())) for i in pop.eval_ids}
    preds = {tag: {r.image_id: sorted(r.iso_codes) for r in recs}
             for tag, recs in by_cell.items()}
    by_k = stratified_table(preds, gold, pop.classes, ID_GRANULARITY, snapshot=frozen)
    return {
        "axis": STRATUM_AXIS,
        "default_k": DEFAULT_K,
        "ladder": list(ID_GRANULARITY),
        "cut_points_from": f"{frozen} train+val (data.id_strata)",
        "by_k": by_k,
        "note": (
            "층화 Macro-F1 은 지시 문면, lift 는 완료 기준이 겨냥한 수다(83번 §1-2). "
            "지름길 규칙(__shortcut__) 행의 lift 가 0 이 아니면 층 정의나 기준선이 틀린 것이다"
        ),
    }


def score_all(
    params: ScoringParams, pop: Population
) -> tuple[dict, dict, dict, list, dict]:
    """다섯 칸 전부 채점. 검출은 저장 레코드 되읽기, 통합형은 어댑터 경유."""
    known = set(load_label_map().iso_codes())
    metrics: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    adapters: dict[str, dict] = {}
    all_records: list = []

    by_cell: dict[str, list] = {}
    for tag in DET_TAGS:
        recs = load_detection_records(params, tag)
        all_records.extend(recs)
        by_cell[tag] = recs
        metrics[tag] = score(pop, recs)
        failures[tag] = failure_breakdown(recs)

    for cell in UNI_TAGS:
        # 원시 생성문에서 매번 새로 어댑트한다. 저장본을 되읽으면 어댑터가 채점 경로에서
        # 빠져 "칸이 갈리는 유일한 지점"이 검증 대상 밖으로 나간다.
        rep = load_unified_records(params, pop, cell, known)
        recs = rep.records
        adapters[cell] = rep.as_dict()
        dest = record_path(params, cell)
        if dest.exists():
            stored = load_detection_records(params, cell)   # 같은 되읽기 경로
            adapters[cell]["stored_matches"] = (
                [r.model_dump_json() for r in stored] == [r.model_dump_json() for r in recs]
            )
        else:
            with dest.open("w", encoding="utf-8", newline="\n") as fh:
                for r in recs:
                    fh.write(r.model_dump_json() + "\n")
        missing = pop.eval_ids - {r.image_id for r in recs}
        extra = {r.image_id for r in recs} - pop.eval_ids
        if missing or extra:
            raise SystemExit(f"{cell}: 평가셋 불일치 결측 {len(missing)} 초과 {len(extra)}")
        all_records.extend(recs)
        by_cell[cell] = recs
        metrics[cell] = score(pop, recs)
        failures[cell] = failure_breakdown(recs)
        adapters[cell]["citations"] = rep.citations

    return metrics, failures, adapters, all_records, by_cell


def recovery(metrics: dict) -> dict:
    """회복률 = (연합 − 로컬) / (중앙집중 − 로컬). 헤드라인 숫자(개발규약).

    통합·로컬은 실험 구조상 '제외' 칸이라 분모가 없다. 그쪽은 유지율만 낸다 —
    **두 값은 다른 양이다.** 같은 표에 나란히 두면 읽는 사람이 섞는다.
    """
    def f1(tag: str) -> float:
        return float(metrics[tag]["macro_f1"])

    locals_ = ("sep_local_C1", "sep_local_C2", "sep_local_C3")
    local_mean = sum(f1(k) for k in locals_) / len(locals_)
    denom = f1("sep_central") - local_mean
    return {
        "basis": "macro_f1",
        "separated": {
            "central": f1("sep_central"), "fed": f1("sep_fed"), "local_mean": local_mean,
            "locals": {k: f1(k) for k in locals_},
            "denominator": denom,
            "recovery_pct": (f1("sep_fed") - local_mean) / denom * 100
            if denom > 0 else None,
        },
        "unified": {
            "central": f1("uni_central"), "fed": f1("uni_fed"),
            "local_mean": None, "recovery_pct": None,
            "retention_pct": f1("uni_fed") / f1("uni_central") * 100
            if f1("uni_central") > 0 else None,
            "note": "통합·로컬은 '제외' 칸이라 회복률 분모가 없다. 유지율만 낸다",
        },
        "caveat": "시드 1세트 · 표본 3,279 · R×E=N=6 (본실험의 1/17). 결론으로 쓰지 않는다",
    }


def diagnostics(params: ScoringParams, pop: Population, all_records: list,
                adapters: dict) -> dict:
    """P9(규격 지름길) · 자명하한 · 통합형 인용 진단. 66번이 내던 것을 그대로 옮겼다."""
    with (params.snapshot / "tiles.csv").open(encoding="utf-8", newline="") as fh:
        prov = {r["image_id"]: r["provenance"] for r in csv.DictReader(fh)}
    contexts, ctx_missing = contexts_from_snapshot(pop.rows, prov)
    p9 = p9_all_cells(all_records, contexts)

    samples = [
        MetaSample(
            image_id=r["image_id"], width_px=int(r["width_px"]),
            height_px=int(r["height_px"]), file_bytes=0, n_channels=1, quant_table_id=0,
            iso_codes=tuple(sorted(pop.gold_codes.get(r["image_id"], ()))),
        )
        for r in pop.rows
    ]

    from rag.index import load_chunks, load_rag_config

    index_ids = {c.chunk_id for c in load_chunks(load_rag_config().chunk_meta)}
    citation_diag = {}
    for cell in UNI_TAGS:
        cited = adapters.get(cell, {}).get("citations", {}) or {}
        flat = [c for v in cited.values() for c in v]
        citation_diag[cell] = {
            "n_citations": len(flat),
            "n_images_citing": sum(1 for v in cited.values() if v),
            "n_in_index": sum(1 for c in flat if c in index_ids),
            "n_not_in_index": sum(1 for c in flat if c not in index_ids),
            "distinct": sorted(set(flat))[:20],
        }
    return {
        "p9": p9.as_dict(),
        "p9_context_missing": list(ctx_missing),
        "sample_trivial_bound": trivial_bound(samples, pop.classes),
        "citation_diagnostic": citation_diag,
    }


def check_regressions(params: ScoringParams, metrics: dict) -> dict:
    """65·66번 저장 지표와의 완전 일치 대조. 저장 파일은 읽기만 한다."""
    out: dict[str, dict] = {}
    for label, fname, tags in (
        ("65번", "score_detection_v1.json", DET_TAGS),
        ("66번", "score_all_cells_v1.json", ALL_TAGS),
    ):
        p = params.out / fname
        if not p.exists():
            out[label] = {"checked": False, "reason": f"{fname} 없음"}
            continue
        stored = json.loads(p.read_text(encoding="utf-8"))["metrics"]
        diffs = regression_diffs(stored, metrics, tags)
        out[label] = {"checked": True, "n_cells": len(tags), "diffs": diffs,
                      "identical": not diffs}
    return out


def cmd_score(args) -> int:
    params = params_from_args(args)
    params.out.mkdir(parents=True, exist_ok=True)
    pop = load_population(params)
    print(f"평가셋 {pop.n_eval}장 (정상 {pop.n_normal})")

    metrics, failures, adapters, all_records, by_cell = score_all(params, pop)
    for tag in ALL_TAGS:
        m = metrics[tag]
        print(f"[{tag}] macroF1 {m['macro_f1']:.4f} · miss {m['miss_rate']:.4f} "
              f"· IoU {m['bbox_iou']:.4f}")

    reg = check_regressions(params, metrics)
    diag = diagnostics(params, pop, all_records, adapters)
    strata = stratified_block(pop, by_cell)
    k0 = str(strata["default_k"])
    for tag in [*ALL_TAGS, SHORTCUT_TAG]:
        s = strata["by_k"][k0][tag]
        print(f"[{tag}] 층화(K={k0}) macroF1 {s['stratified_macro_f1']:.4f} · "
              f"lift {s['stratified_lift']:+.5f} · 비순수 lift {s['stratified_lift_impure']:+.5f}")
    gates = run_scoring_gates(GateContext(
        metrics=metrics,
        records_by_cell=by_cell,
        expected_coord_space=params.coord_space,
        population_bound=diag["sample_trivial_bound"],
        n_eval=pop.n_eval,
        n_scored={t: len(v) for t, v in by_cell.items()},
        recovery=recovery(metrics),
        seed_sd=None,                 # 시드 1세트 — 게이트가 그 사실을 판정으로 남긴다
        env=None,                     # 실제 프로세스 환경을 본다
        tags=None,                    # 채점 단계에는 run 태그가 없다. 차단하지 않는다
        gate_status=resolve_gate_status(),
        extra={
            "gate_pass_line": params.gate_pass_line,
            "measured_prereg": _measured_prereg(params),
            "stratified": strata,
        },
    ))
    code, why = exit_code(gates, reg)
    payload = {
        "params": params.as_dict(),
        "n_eval": pop.n_eval,
        "scorer": "evaluation.score.score_records (단일)",
        "metrics": metrics,
        "stratified": strata,
        "failures": failures,
        "adapters": adapters,
        "coord_health": {t: coord_health(m) for t, m in metrics.items()},
        "recovery": recovery(metrics),
        "gate": gate_check(metrics, params),
        "regression": reg,
        "gates_evaluated": gates,
        "exit_code": code,
        "exit_reason": why,
        **diag,
    }
    dest = params.out / "score_cells_v1.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"저장: {dest}")

    print(f"게이트 {gates['n_evaluated']}/{gates['n_registered']} 평가 · "
          f"차단 실패 {gates['blocking_failures'] or '없음'} · "
          f"건너뜀 {gates['n_skipped']}")
    for r in gates["results"]:
        if not r["passed"] and not r["skipped"]:
            mark = "차단" if r["blocking"] else "기록"
            print(f"  [{mark}] {r['name']}: {r['detail'][:120]}")

    bad = [k for k, v in reg.items() if v.get("checked") and not v["identical"]]
    for k in bad:
        print(f"{k} 대조 불일치: {reg[k]['diffs'][:3]}")
    if not bad:
        print("65·66번 대조: 완전 일치" if any(v.get("checked") for v in reg.values())
              else "65·66번 대조: 저장본 없음 — 미대조")
    print(f"게이트: {payload['gate']['verdict']} (선 {params.gate_pass_line}, "
          f"{params.gate.source})")
    print(f"종료 코드 {code} — {why}")
    return code


def cmd_gate(args) -> int:
    """게이트 상수만 갈아 끼워 다섯 칸을 다시 판정한다. 채점 자체는 건드리지 않는다.

    A 의 content-free 천장 재산출이 도착하면 `--gate <값>` 한 번, 혹은
    `configs/base.yaml` 에 키가 생기면 플래그 없이도 그 값으로 돈다.
    """
    params = params_from_args(args)
    src = params.out / "score_cells_v1.json"
    if not src.exists():
        print(f"{src} 없음 — 먼저 `score` 를 돌린다")
        return 1
    metrics = json.loads(src.read_text(encoding="utf-8"))["metrics"]
    result = gate_check(metrics, params)
    best = best_case_gate(params)
    dest = params.out / f"gate_recheck_g{params.gate.value:.4f}_v1.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump({"params": params.as_dict(), "gate_result": result,
                   "best_case_over_sweep": best}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    for tag, v in result["cells"].items():
        mark = "통과" if v["above_pass_line"] else "미달"
        print(f"[{tag}] {v['macro_f1']:.4f} {mark} (여유 {v['margin_vs_pass_line']:+.4f})")
    print(f"판정(운용 임계 {params.conf.value}): {result['verdict']} · "
          f"통과선 {result['pass_line']} ({params.gate.source})")
    if best.get("checked"):
        print(f"판정(임계 최선): {best['verdict']} — "
              f"{best['n_above_pass_line']}/{best['n_cells']} 통과")
    print(f"저장: {dest}")
    return 0


def best_case_gate(params: ScoringParams) -> dict:
    """**임계를 가장 유리하게 잡아도** 게이트를 넘는 칸이 있는가.

    운용 임계 하나에서의 미달은 "임계를 잘못 골라서"라는 반론을 받는다. 스윕 전 구간의
    칸별 최댓값으로 대면 그 반론이 닫힌다 — 분리형에 최대한 유리한 조건이다.
    통합형은 임계가 없으므로 한 점 그대로다.
    """
    p = params.out / "sweep_detection_conf_v1.json"
    if not p.exists():
        return {"checked": False, "reason": "sweep_detection_conf_v1.json 없음"}
    d = json.loads(p.read_text(encoding="utf-8"))
    cells: dict[str, dict] = {}
    for tag, s in d["sweep"].items():
        arg, val = max(((k, v["macro_f1"]) for k, v in s["by_threshold"].items()),
                       key=lambda kv: kv[1])
        cells[tag] = {"macro_f1": val, "at_conf": arg,
                      "above_pass_line": val > params.gate_pass_line}
    for cell, u in d["unified"].items():
        val = u["metrics"]["macro_f1"]
        cells[cell] = {"macro_f1": val, "at_conf": "임계 없음",
                       "above_pass_line": val > params.gate_pass_line}
    n = sum(1 for v in cells.values() if v["above_pass_line"])
    return {
        "checked": True, "pass_line": params.gate_pass_line,
        "sweep_grid": d["params"]["conf_sweep"],
        "n_cells": len(cells), "n_above_pass_line": n, "cells": cells,
        "verdict": ("임계를 가장 유리하게 잡아도 전 칸 미달" if n == 0
                    else f"임계 최선에서 {n}/{len(cells)} 통과"),
    }


def cmd_predict(args) -> int:
    """검출 3칸 CPU 추론.

    기본은 **스윕용 하한**(`conf_floor`)이다. `--at-conf` 를 주면 운용 임계로 65번과
    같은 레코드를 만든다 — 임계가 인자인 것이 이 리팩토링의 핵심이다. 65번은 이 값을
    모듈 상수로 박아 분리형만 잘린 뒤 채점됐다(감사 D-1).
    """
    from evaluation.detect_infer import (
        cell_tag,
        checkpoint_paths,
        load_yolo_from_npz,
        predict_cell,
    )
    from evaluation.params import CONF_FLOOR
    from tracking.mlflow_local import reject_best_checkpoint

    params = params_from_args(args)
    pop = load_population(params)
    conf = params.conf.value if args.at_conf else params.conf_floor
    sub = params.out if args.at_conf else params.out / "sweep"
    sub.mkdir(parents=True, exist_ok=True)
    print(f"평가셋 {pop.n_eval}장 · conf {conf} "
          f"({'운용' if args.at_conf else f'하한 {CONF_FLOOR}'}) · "
          f"프로파일 {params.profile} ({params.model_cfg} · imgsz {params.imgsz} · "
          f"청크 {params.predict_chunk})")

    for (cell, client), ckpt in checkpoint_paths(params.pilot).items():
        reject_best_checkpoint(ckpt)
        if not ckpt.exists():
            print(f"체크포인트 없음: {ckpt}")
            return 1
        tag = cell_tag(cell, client)
        yolo = load_yolo_from_npz(ckpt, params.class_names, params.imgsz,
                                  model_cfg=params.model_cfg)
        recs = predict_cell(yolo, pop.rows, Path(args.root), cell, client, params, conf=conf)
        name = f"{tag}_s{params.seed}.jsonl" if args.at_conf \
            else f"{tag}_raw_s{params.seed}.jsonl"
        with (sub / name).open("w", encoding="utf-8", newline="\n") as fh:
            for r in recs:
                fh.write(r.model_dump_json() + "\n")
        print(f"[{tag}] {sum(len(r.defects) for r in recs)}박스 -> {name}")
    return 0


def cmd_sweep(args) -> int:
    from scripts.probe.sweep_detection_conf import main as sweep_main

    argv = ["sweep_detection_conf.py", "--stage", "sweep",
            "--snapshot", args.snapshot, "--pilot", args.pilot, "--out", args.out]
    if args.conf is not None:
        argv += ["--conf", str(args.conf)]
    sys.argv = argv
    return sweep_main()


def main() -> int:
    ap = argparse.ArgumentParser(description="다섯 칸 채점 단일 진입점")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("score", cmd_score), ("gate", cmd_gate),
                     ("predict", cmd_predict), ("sweep", cmd_sweep)):
        p = sub.add_parser(name)
        add_common_args(p)
        p.add_argument("--root", default=".")
        p.add_argument("--at-conf", action="store_true",
                       help="predict: 하한이 아니라 운용 임계로 추론한다(65번 레코드 생성)")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


def _measured_prereg(params: ScoringParams) -> dict:
    """채점 디렉터리의 `prereg_recomputed_v1.json`. **없으면 동결본에서 재산출해 선배치한다.**

    13번 D-8 파생: 파일이 없으면 `prereg_constants_reproduced` 게이트가 `skipped` 로
    갈렸다 — 검증에서 실측된 구멍이다. 상수는 동결 스냅샷의 결정론적 함수이므로 채점기가
    스스로 만들어 두는 것이 사람 손 선배치보다 안전하다. 파일은 남겨 다음 채점이 되읽고,
    digest 와 출처를 게이트 value 에 실어 어느 동결본에서 왔는지가 표에 남게 한다.
    """
    from scripts.probe.recompute_prereg import FILE_NAME, recompute, write_payload

    p = params.out / FILE_NAME
    if not p.exists():
        write_payload(recompute(), p)
        print(f"사전등록 상수 재산출 → 선배치 {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    tot = d["populations"]["frozen_total"]
    return {
        "all_positive_macro_f1": tot["all_positive_macro_f1"],
        "spec_only_macro_f1": tot["spec_only_macro_f1"],
        "snapshot_digest": d.get("snapshot_digest"),
        "source": str(p),
    }


if __name__ == "__main__":
    raise SystemExit(main())

