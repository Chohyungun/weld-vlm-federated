"""리허설 · 진단 계획의 단계 시한(`timeouts_s`)을 재고 제안한다 — 결정 19 의 R-1.

    python -X utf8 scripts/rehearsal_timeouts.py measure-load --out <JSON> [--repeats 2]      # GPU 창 — 모델 적재 시간을 잰다
    python -X utf8 scripts/rehearsal_timeouts.py propose --plan configs/rehearsal_uni.yaml (--load-json <JSON> | --load-s <초>) [--out <JSON>]

**제안이다 — 설정 파일을 쓰지 않는다.** 계획 파일(`configs/`)은 데이터 쪽 소관이고 값은 총괄이 승인한다. `propose` 는 단계마다 상한과
그것을 낸 식 · 입력을 JSON 으로 내고, 계획에 옮길 `timeouts_s` 블록을 함께 적는다. 시한은 넘으면 그 단계를 프로세스 트리째 끊는
상한이다(2판 §1-3) — 너무 짧으면 정상 실행을 끊고, 넉넉하면 멈춘 실행을 늦게 잡을 뿐이다. 그래서 모르는 값은 **넉넉한 쪽**으로 둔다.

| 단계 | 식(초) | 입력 |
|---|---|---|
| `lists` | 고정 1,800 | CPU 만 — 매니페스트 · 페어 · 간선 읽기와 뽑기. 잰 적이 없다 |
| `register` | 1.5 × (2 × 적재) + 600 | 공통 초기 어댑터(캐시가 없을 때 적재 1)와 학습 설정 원문(적재 1) |
| `train` | 1.5 × max(칸마다) + 60 | 로컬 · 중앙 = 행 × epoch(`amount.n`) × 표본 단가 + 적재. 연합 = R × (참여자 수 × 적재 + Σ 행 × E × 표본 단가) — GPU 1 장이라 클라이언트가 차례로 돈다. 60 은 SuperLink 기동 |
| `export` | 1.5 × (`gen.max_n` × (선행 + 한도 ÷ 토큰 속도) + 적재) | 모든 장이 한도에 닿는 최악(생성 길이를 모른다) |
| `score` | 고정 1,800 | 평가 쪽 본채점(CPU). 잰 적이 없다 |

표본 단가 3.621 s · 선행 0.816 s · 토큰 속도 5.87 tok/s 는 4B · 4bit 실측이다(예산표 `02_예산표_C.md` §1 의 b · d · e).
행 수는 계획의 목표(`train.rows`)다 — 묶음 단위로 뽑아 실제는 조금 다르다. 루트(`--root`)를 주면 그 루트의 `lists/` 에서 실제 행 수를 센다.
**그때 루트 · 목록 폴더 · 계획의 칸마다 필요한 목록(연합은 참여자 셋)과 비어 있지 않은 행을 먼저 보고, 하나라도 없으면 제안을 거부한다** —
빠진 칸을 0 으로 보거나 계획값으로 대신하지 않는다(외부 검토 §51).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

#: 4B · 4bit 실측(예산표 §1). 출처를 함께 싣는다.
PER_SAMPLE_S = 3.621
PREFILL_S = 0.816
TOK_PER_S = 5.87
MEASURED_FROM = "docs/dev_log/2026-09-24-후속정리/02_예산표_C.md §1 (b · d · e)"
MARGIN = 1.5
LISTS_S = SCORE_S = 1800
REGISTER_EXTRA_S = 600
SUPERLINK_S = 60
STAGES = ("lists", "register", "train", "export", "score")
EXIT_OK, EXIT_REFUSED = 0, 2
LOCAL_TAGS = ("uni_local_C1", "uni_local_C2", "uni_local_C3")


class ProposalRefused(ValueError):
    """제안을 낼 입력이 서지 않는다 — 값을 내지 않는다."""

    def __init__(self, code: str, msg: str):
        self.code = code
        super().__init__(f"[{code}] {msg}")


def _needed(cells) -> list[str]:
    """계획의 칸이 학습에 쓰는 목록의 태그 — 로컬은 그 칸, 중앙은 중앙, 연합은 참여자 셋."""
    out: list[str] = []
    for tag in cells:
        for t in (LOCAL_TAGS if tag == "uni_fed" else (tag,)):
            if t not in out:
                out.append(t)
    return out


def _ceil100(x: float) -> int:
    return int(math.ceil(x / 100.0) * 100)


def _rows(plan: Any, root: Path | None) -> dict[str, int]:
    """칸 → 학습 행 수. 루트가 있으면 목록 단계의 실제 행, 없으면 계획의 목표. 필요한 칸이 하나라도 없으면 `ProposalRefused`."""
    need = _needed(plan.get("cells") or [])
    if root is not None:
        root = Path(root)
        if not root.is_dir():
            raise ProposalRefused("root_missing", f"루트가 없다: {root}")
        lists = root / "lists"
        if not lists.is_dir():
            raise ProposalRefused("lists_missing", f"루트에 목록 폴더가 없다: {lists}")
        out = {}
        for tag in need:
            p = lists / f"train_{tag}.jsonl"
            if not p.is_file():
                raise ProposalRefused("train_list_missing", f"칸 {tag} 의 학습 행 목록이 없다: {p.name}")
            n = sum(1 for x in p.read_text(encoding="utf-8").splitlines() if x.strip())
            if n == 0:
                raise ProposalRefused("train_list_empty", f"칸 {tag} 의 학습 행 목록이 비었다: {p.name}")
            out[tag] = n
        return out
    target = dict(plan.get("train.rows") or {})
    out = {f"uni_local_{c}": int(v) for c, v in target.items() if c != "central"}
    if "central" in target:
        out["uni_central"] = int(target["central"])
    elif plan.get("train.central_rows") == "union":
        out["uni_central"] = sum(out.values())
    missing = [t for t in need if not out.get(t)]
    if missing:
        raise ProposalRefused("plan_rows_missing", f"계획에 학습 행 목표가 없는 칸이 있다: {missing}")
    return {t: out[t] for t in need}


def propose(plan: Any, *, load_s: float, root: Path | None = None) -> dict[str, Any]:
    """단계 시한의 제안 — 값 · 식 · 입력. 계획 파일은 쓰지 않는다."""
    n, r, e = plan.amount
    rows = _rows(plan, root)
    cells = list(plan.get("cells") or [])
    per_cell: dict[str, float] = {}
    for tag in cells:                                         # `_rows` 가 칸마다 필요한 행을 다 냈다 — 빠진 칸을 건너뛰지 않는다
        if tag == "uni_fed":
            locals_ = [rows[t] for t in LOCAL_TAGS]
            per_cell[tag] = r * (len(locals_) * load_s + sum(locals_) * e * PER_SAMPLE_S)
        else:
            per_cell[tag] = rows[tag] * n * PER_SAMPLE_S + load_s
    limit = int(plan.get("gen_limit.max_new_tokens"))
    per_image = PREFILL_S + limit / TOK_PER_S
    max_n = int(plan.get("gen.max_n"))
    values = {
        "lists": LISTS_S,
        "register": _ceil100(MARGIN * (2 * load_s) + REGISTER_EXTRA_S),
        "train": _ceil100(MARGIN * max(per_cell.values()) + SUPERLINK_S) if per_cell else None,
        "export": _ceil100(MARGIN * (max_n * per_image + load_s)),
        "score": SCORE_S,
    }
    return {
        "kind": "timeouts_proposal", "purpose": plan.purpose, "plan_sha256": plan.sha256, "timeouts_s": values,
        "inputs": {"load_s": load_s, "per_sample_s": PER_SAMPLE_S, "prefill_s": PREFILL_S, "tok_per_s": TOK_PER_S,
                   "measured_from": MEASURED_FROM, "margin": MARGIN, "rows": rows, "rows_from": "root" if root else "plan",
                   "amount": {"n": n, "r": r, "e": e}, "gen_max_n": max_n, "max_new_tokens": limit,
                   "per_image_worst_s": round(per_image, 3), "train_per_cell_s": {k: round(v, 1) for k, v in per_cell.items()}},
        "note": "제안이다 — 계획 파일을 쓰지 않는다. 값은 총괄이 승인하고 데이터 쪽이 계획에 옮긴다.",
    }


def measure_load(*, repeats: int = 2, loader: Callable | None = None, config_path: Path | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """모델 적재 시간 — 학습 · 생성이 쓰는 그 적재기(`vlm.pilot_vlm._load_model`)를 본실험 설정의 모델 · 판 · 프로세서 설정으로 부른다.
    `loader` 는 시험의 이음새다(파이썬 인자로만). 학습 · 생성은 하지 않는다."""
    from vlm.pilot_vlm import _load_model
    from vlm.uni_config import load_uni_config

    cfg = load_uni_config(config_path)
    fn = loader or _load_model
    times = []
    for _ in range(int(repeats)):
        t0 = clock()
        model, proc = fn(cfg.model_id, revision=cfg.model_revision, processor_kwargs=dict(cfg.processor_kwargs or {}))
        times.append(round(clock() - t0, 3))
        del model, proc
    return {"kind": "load_measurement", "model_id": cfg.model_id, "model_revision": cfg.model_revision,
            "repeats": len(times), "load_s": times, "load_s_max": max(times), "load_s_first": times[0],
            "at": datetime.now().astimezone().isoformat(timespec="seconds")}


def _write_new(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as fh:      # 배타 생성 — 있으면 덮지 않는다
        fh.write(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


def main(argv: list[str] | None = None, *, loader: Callable | None = None, config_path: Path | None = None) -> int:
    from vlm.rehearsal_plan import PlanRejected, load_plan

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure-load")
    m.add_argument("--out", required=True, type=Path)
    m.add_argument("--repeats", type=int, default=2)
    p = sub.add_parser("propose")
    p.add_argument("--plan", required=True, type=Path)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--load-json", type=Path)
    g.add_argument("--load-s", type=float)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "measure-load":
        if a.out.exists():
            print(f"[out_exists] 산출이 이미 있다: {a.out} — 덮지 않는다", file=sys.stderr)
            return EXIT_REFUSED
        rep = measure_load(repeats=a.repeats, loader=loader, config_path=config_path)
        _write_new(a.out, rep)
        print(f"load_s_max={rep['load_s_max']}", flush=True)
        return EXIT_OK
    import yaml

    try:
        purpose = str((yaml.safe_load(a.plan.read_text(encoding="utf-8")) or {}).get("kind"))
        plan = load_plan(a.plan, purpose=purpose)
    except PlanRejected as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED
    load_s = float(json.loads(a.load_json.read_text(encoding="utf-8"))["load_s_max"]) if a.load_json else float(a.load_s)
    try:
        out = propose(plan, load_s=load_s, root=a.root)
    except ProposalRefused as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED
    text = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if a.out is not None:
        if a.out.exists():
            print(f"[out_exists] 산출이 이미 있다: {a.out} — 덮지 않는다", file=sys.stderr)
            return EXIT_REFUSED
        _write_new(a.out, out)
    print(text, flush=True)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
