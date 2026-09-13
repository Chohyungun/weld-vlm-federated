"""검증기 정본 선정 — 사람 라벨 표본 + 정밀도·재현율 (체크리스트 9, 80번 G12-3·G12-4).

파일럿에서 검증기 둘의 통과율이 0.3876 대 0.9551 로 갈렸고 보고서는 그것을 "계열
차이"로 읽었다. 재검이 뒤집었다 — **계열 + 설정 차이**다. 한쪽은 사고를 프리필로
억제하고 예산을 32 토큰으로 잘랐고 다른 쪽은 자유 생성 64 토큰이었다 (B9).

설정 교락은 `configs/corpus_validation.yaml` 이 두 후보에게 **같은 예산·같은 사고
정책**을 주는 것으로 닫았다. 남는 것은 "그래서 어느 쪽이 맞는가" 인데, 그 답은 두
기계의 일치도로는 나오지 않는다. **정답이 없기 때문이다.**

그래서 사람 라벨 100건을 정답으로 놓고 후보별 정밀도·재현율을 잰다. 라벨이 없는
동안에는 어떤 후보도 정본이 아니고, 보고서는 통과율을 `pass_rate` 가 아니라
`judge_agreement` 로만 싣는다.

표본은 **축 2 × 판정 2 층화**다. 판정 축을 층으로 쓰는 이유는, 통과분만 보면 재현율을
잴 수 없고 기각분만 보면 정밀도를 잴 수 없기 때문이다.

실행:
  uv run python -m corpus.validate.judge_labels sheet   # 표본지 생성 (사람이 채운다)
  uv run python -m corpus.validate.judge_labels score   # 채워진 라벨로 지표 산출
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs/corpus_validation.yaml"
DEFAULT_CYCLE_DIR = REPO / "corpus/generate/cycle_pilot_v2"


def load_cfg(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _stratum(rec: dict, judge_id: str) -> Optional[tuple[str, str]]:
    v = rec.get(f"judge_{judge_id}_pass")
    if v is None:
        return None
    return (str(rec.get("axis") or "?"), "judge_pass" if v else "judge_fail")


def build_sheet(records: Sequence[dict], cfg: dict,
                judge_id: str) -> tuple[list[dict], dict]:
    """층별 균등 배분 표본. 시드 고정 — 같은 입력이면 같은 표본이다.

    한 층이 목표에 못 미치면 그 부족분을 다른 층에 채우지 않는다. 채우면 층 비율이
    무너지고, 그러면 정밀도·재현율이 어느 모집단의 값인지 말할 수 없다.

    **사람이 보는 것과 표집 기록을 가른다** (74번 F14, 09-13 판정). 예전에는 행마다
    `stratum: "…|judge_pass"` 가 실려서 라벨러가 후보 판정을 그대로 읽었고, 층별로 묶여
    나와 **순서만 봐도** 어디까지가 통과분인지 보였다. 블라인드가 아니었다.

    그렇다고 층 정보를 **지우면** 안 된다. 층별 표집확률 `p = k/N` 이 없으면 나중에
    모집단 지표를 가중할 수 없고, 그러면 정밀도·재현율이 어느 모집단의 값인지 다시
    말할 수 없게 된다 — 표집을 층화한 이유 자체가 사라진다. 그래서 **지우지 않고 옮긴다.**

    돌려주는 것은 `(사람용 시트, 표집 메타)` 둘이다. 시트에는 층이 없고 순서가 섞여
    있으며, 메타에는 층·모집단 크기·표집 수·표집확률·부족분이 전부 남는다. 메타는
    라벨러에게 주지 않는다.
    """
    n = int(cfg["labeling"]["n"])
    axes, verdicts = cfg["labeling"]["strata"]
    cells = [(a, v) for a in axes for v in verdicts]
    per = n // len(cells)
    rng = np.random.default_rng(int(cfg["labeling"]["seed"]))

    pool: dict[tuple[str, str], list[dict]] = {c: [] for c in cells}
    for r in records:
        key = _stratum(r, judge_id)
        if key in pool:
            pool[key].append(r)

    sheet: list[dict] = []
    strata: dict[str, str] = {}
    cell_meta: dict[str, dict] = {}
    for cell in cells:
        name = "|".join(cell)
        rows = sorted(pool[cell], key=lambda r: str(r.get("sample_id")))
        k = min(per, len(rows))
        cell_meta[name] = {
            "N_population": len(rows),      # 이 층의 모집단 크기
            "k_drawn": k,                   # 실제로 뽑은 수
            "target": per,                  # 목표
            "shortfall": per - k,           # 못 채운 수 (0 이면 충족)
            # 층 가중의 재료. 모집단이 0이면 확률을 말할 수 없다.
            "p_sampling": round(k / len(rows), 6) if rows else None,
        }
        idx = rng.choice(len(rows), size=k, replace=False) if k else []
        for i in sorted(int(x) for x in idx):
            r = rows[i]
            strata[str(r["sample_id"])] = name
            sheet.append({
                "sample_id": r["sample_id"],
                "axis": r.get("axis"),
                # 사람이 보는 것도 기계가 본 것과 **같은 자료**여야 한다 (G4-1).
                "basis": _basis_of(r),
                "text": r.get("text"),
                # 라벨러가 기계 판정에 끌려가지 않도록 후보 판정은 싣지 않는다.
                "human_ok": None,
                "labeler": None,
                "note": None,
            })
    # 층별 블록 순서 자체가 후보 판정을 드러낸다 — 표집과 같은 시드로 순서를 섞는다.
    rng.shuffle(sheet)

    meta = {
        "_meta": ("표집 집계. 층 가중 지표의 재료다 (74번 F14-P1). 항목별 층은 여기에"
                  " 없다 — 그것은 곧 항목별 후보 판정이라 따로 두고 추적하지 않는다."
                  " 층별 합계(N·k)는 모집단 사실이라 보고 대상이다."),
        "stratified_by_judge": judge_id,
        "seed": int(cfg["labeling"]["seed"]),
        "n_target": n,
        "n_drawn": len(sheet),
        "cells": cell_meta,
        "total_shortfall": sum(c["shortfall"] for c in cell_meta.values()),
        "stratum_of": strata,               # sample_id → 층
    }
    return sheet, meta


def _basis_of(rec: dict) -> str:
    from corpus.generate.basis import render_basis

    return render_basis(rec)


def _counts(pred: Sequence[bool], gold: Sequence[bool]) -> dict:
    tp = sum(1 for p, g in zip(pred, gold) if p and g)
    fp = sum(1 for p, g in zip(pred, gold) if p and not g)
    fn = sum(1 for p, g in zip(pred, gold) if not p and g)
    tn = sum(1 for p, g in zip(pred, gold) if not p and not g)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "n": tp + fp + fn + tn}


def score(records: Sequence[dict], labels: Sequence[dict], cfg: dict) -> dict:
    """후보별 정밀도·재현율. **사람 라벨이 정답이다.**

    `OK` 를 양성으로 본다 — 정밀도는 "통과시킨 것 중 진짜 통과할 것", 재현율은
    "통과할 것 중 통과시킨 것" 이다. 검증기의 실패 비용은 비대칭이라(오염이 학습에
    들어가는 쪽이 비싸다) 정밀도를 먼저 본다.
    """
    field = cfg["labeling"]["label_field"]
    gold_by_id = {str(x["sample_id"]): bool(x[field])
                  for x in labels if x.get("sample_id") and x.get(field) is not None}
    by_id = {str(r["sample_id"]): r for r in records}
    out: dict[str, Any] = {"n_labeled": len(gold_by_id), "candidates": {}}
    for cand in cfg["judges"]["candidates"]:
        cid = cand["id"]
        pred, gold, missing = [], [], 0
        for sid, g in sorted(gold_by_id.items()):
            r = by_id.get(sid)
            if r is None or r.get(f"judge_{cid}_pass") is None:
                missing += 1
                continue
            pred.append(bool(r[f"judge_{cid}_pass"]))
            gold.append(g)
        out["candidates"][cid] = {
            "model": cand["model"], "family": cand["family"],
            "n_missing_prediction": missing, **_counts(pred, gold),
        }
    labelers = {x.get("labeler") for x in labels if x.get("labeler")}
    out["n_labelers"] = len(labelers)
    if len(labelers) >= 2:
        out["note"] = "라벨러 2명 이상 — Cohen's kappa 를 함께 산출해야 한다"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "score"])
    ap.add_argument("--judge-id", default=None,
                    help="층화 기준 후보. 미지정 시 등록된 첫 후보")
    ap.add_argument("--cycle-dir", default=str(DEFAULT_CYCLE_DIR),
                    help="판정 레코드가 있는 사이클 산출 디렉터리")
    args = ap.parse_args()

    cfg = load_cfg()
    cyc = Path(args.cycle_dir)
    # 정본 미지정이면 채택분이 `reasoning_pending.jsonl` 로 간다 (F13). 셋 다 읽는다.
    records = (read_jsonl(cyc / "reasoning_accepted.jsonl")
               + read_jsonl(cyc / "reasoning_pending.jsonl")
               + read_jsonl(cyc / "discarded.jsonl"))
    records = [r for r in records if r.get("axis")]
    if not records:
        print("판정 레코드가 없다 — corpus 사이클을 먼저 돌려라 (체크리스트 19)",
              file=sys.stderr)
        return 2

    sheet_path = REPO / cfg["labeling"]["sheet"]
    labels_path = REPO / cfg["labeling"]["labels"]

    if args.cmd == "sheet":
        jid = args.judge_id or cfg["judges"]["candidates"][0]["id"]
        sheet, meta = build_sheet(records, cfg, jid)
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        with sheet_path.open("w", encoding="utf-8", newline="") as fh:
            for row in sheet:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        # 메타를 둘로 가른다. **집계는 남기고 항목별 층은 추적하지 않는다.**
        # 집계(층별 N·k·표집확률)는 감사가 보고하라고 한 것이고 판정을 담지 않는다.
        # 항목별 층은 곧 그 항목의 후보 판정이라, 시트 옆에 추적돼 있으면 라벨러가
        # 저장소만 열어도 답을 본다. 필요하면 `_stratum()` 으로 다시 계산하면 된다.
        per_sample = meta.pop("stratum_of")
        meta_path = sheet_path.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
        strata_path = sheet_path.with_suffix(".strata.json")
        with strata_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(json.dumps(
                {"_meta": "항목별 층 = 그 항목의 후보 판정이다. **추적하지 않는다.**",
                 "stratified_by_judge": meta["stratified_by_judge"],
                 "stratum_of": per_sample}, ensure_ascii=False, indent=1) + "\n")
        print(f"표본지 {len(sheet)}건: {sheet_path}")
        # 층 분포는 화면에도 찍지 않는다 — 터미널 기록도 누설 경로다.
        print(f"표집 집계(층별 N·k·표집확률): {meta_path}   [추적]")
        print(f"항목별 층: {strata_path}   [미추적 — 라벨러에게 주지 않는다]")
        if meta["total_shortfall"]:
            print(f"  ! 층 목표 미달 합계 {meta['total_shortfall']}건 —"
                  " 부족분을 다른 층에서 채우지 않았다 (층 비율 보존).")
        print(f"**사람이 {cfg['labeling']['label_field']} 을 채운 뒤** "
              f"{labels_path} 로 저장하고 score 를 돌려라.")
        return 0

    labels = read_jsonl(labels_path)
    if not labels:
        print(f"라벨 파일이 없다: {labels_path}", file=sys.stderr)
        print("정본 선정은 사람 라벨 없이 할 수 없다 (G12-3). 표본지를 먼저 채워라.",
              file=sys.stderr)
        return 3
    result = score(records, labels, cfg)
    out = labels_path.parent / "judge_selection.json"
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print("산출:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
