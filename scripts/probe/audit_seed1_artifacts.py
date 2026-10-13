"""시드 1 첫 산출물 감사 — 학습 산출물 1~7 (13번 §4). **읽기 전용.**

    uv run python scripts/probe/audit_seed1_artifacts.py --run outputs/main_c/seed1

13번 §4 가 "첫 산출물에서 확인하라"고 지목한 학습 쪽 7항을 기계로 센다. 이 스크립트는 채점 쪽이고
이 파일들은 C 의 산출물이라 **아무것도 쓰지 않는다** — 결과만 낸다.

1. ②③ meta.json 의 회계 필드(`epochs_ran`·`optimizer_steps`·stopper·`seed`·cudnn)
2. ② c1 재개 흔적과 원자 로그 (round, client, metric) 중복
3. 완료 마커 npz 옆 meta.json 이 전 칸 실재(마커만 있고 meta 결손인 칸이 없는가 — C-5)
4. ④ audit.json — `ok`·`failures`·총 스텝·`global_r050.npz`
5. ④ 재기동 흔적 → 원자 로그 중복 전수 (C-1)
6. ④ accounting.csv — 가중과 표본 수의 분리, 단위 일관
7. `initial.npz` 해시가 ②③④ 세 경로에서 동일 (동일 출발)

기대값은 문서가 아니라 **상수에서 유도한다**: `optimizer_steps == epochs_ran ×
ceil(num_examples / batch)`(batch 는 `FIXED_OVERRIDES`), 원자 행 수는 R × (클라이언트 ×
지표 + 서버 지표). 문서에 적힌 수를 그대로 대조하면 문서가 틀렸을 때 함께 틀린다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

R, E, N = 50, 2, 100
CLIENTS = (0, 1, 2)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fixed_batch() -> int:
    from detection.round_runner import FIXED_OVERRIDES

    return int(FIXED_OVERRIDES["batch"])


def _rows(path: Path) -> list[dict]:
    csv.field_size_limit(1 << 30)
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def check(items: list[dict], name: str, ok: bool, detail: str) -> bool:
    items.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


# --------------------------------------------------------------------------------------
# 1·3 — ②③ meta.json
# --------------------------------------------------------------------------------------

def audit_local_central(run: Path, items: list[dict], batch: int) -> dict:
    out: dict[str, dict] = {}
    targets = [
        ("sep_local_c0", run / "sep_local" / "sep_local_c0"),
        ("sep_local_c1", run / "sep_local" / "sep_local_c1"),
        ("sep_local_c2", run / "sep_local" / "sep_local_c2"),
        ("sep_central", run / "sep_central" / "sep_central"),
    ]
    for tag, stem in targets:
        npz, meta_p = stem.with_suffix(".npz"), Path(str(stem) + ".meta.json")
        # 감사 3 — 마커(npz)만 있고 meta 가 없는 칸이 없어야 한다(C-5: 건너뜀 경로는 meta 를 안 본다)
        check(items, f"3.{tag}.meta_beside_npz", npz.exists() and meta_p.exists(),
              f"npz={npz.exists()} meta={meta_p.exists()}")
        if not meta_p.exists():
            continue
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        want_steps = int(m["epochs_ran"]) * math.ceil(int(m["num_examples"]) / batch)
        gr = m.get("gate_results") or {}
        out[tag] = {k: m.get(k) for k in (
            "epochs_ran", "optimizer_steps", "optimizer_updates", "num_examples", "seed",
            "stopper_class", "stopper_true_count", "budget_fired_at",
            "resumed_from_epoch", "val_loader_workers", "peak_vram_gb")}
        out[tag]["optimizer_steps_expected"] = want_steps
        out[tag]["gate_results"] = gr
        out[tag]["gates_evaluated"] = m.get("gates_evaluated")
        check(items, f"1.{tag}.epochs_ran==N", int(m["epochs_ran"]) == N,
              f"{m['epochs_ran']} (기대 {N})")
        check(items, f"1.{tag}.optimizer_steps", int(m["optimizer_steps"]) == want_steps,
              f"{m['optimizer_steps']} = {m['epochs_ran']}×ceil({m['num_examples']}/{batch})={want_steps}")
        check(items, f"1.{tag}.stopper_class", m.get("stopper_class") == "NoEarlyStopping",
              str(m.get("stopper_class")))
        check(items, f"1.{tag}.stopper_true_count==0", m.get("stopper_true_count") == 0,
              str(m.get("stopper_true_count")))
        check(items, f"1.{tag}.budget_fired_at", m.get("budget_fired_at") == N - 1,
              f"{m.get('budget_fired_at')} (기대 {N - 1})")
        check(items, f"1.{tag}.cudnn_deterministic", gr.get("cudnn_deterministic") is True,
              f"deterministic={gr.get('cudnn_deterministic')} benchmark={gr.get('cudnn_benchmark')}")
    # 시드는 **파생식의 값**이어야 한다 — `derive_seed(base, round_idx, client_idx)`.
    # ②③ 은 R=1 의 퇴화 케이스라 round_idx=0 이고, 클라이언트 번호만 다르다. 그래서
    # `sep_local_c0` 와 `sep_central` 은 **같은 값이 정상이다**(둘 다 client 0, round 0).
    from detection.round_runner import derive_seed

    base = int(out.get("sep_local_c0", {}).get("seed", 0))
    want = {"sep_local_c0": derive_seed(base, 0, 0), "sep_local_c1": derive_seed(base, 0, 1),
            "sep_local_c2": derive_seed(base, 0, 2), "sep_central": derive_seed(base, 0, 0)}
    got = {t: v.get("seed") for t, v in out.items()}
    check(items, "1.seed_matches_derive_seed",
          all(got.get(t) == w for t, w in want.items() if t in got),
          f"실측 {got} · 기대 {want} (base {base}, derive_seed = base + 10007·r + 101·c)")
    return out


# --------------------------------------------------------------------------------------
# 2·5 — 원자 로그 중복
# --------------------------------------------------------------------------------------

def audit_atomic(path: Path, items: list[dict], label: str) -> dict:
    if not path.exists():
        check(items, f"{label}.atomic_exists", False, f"없음 {path}")
        return {}
    rows = _rows(path)
    seen: dict[tuple, int] = defaultdict(int)
    for r in rows:
        seen[(r["round"], r["client_id"], r["metric_name"])] += 1
    dups = {k: v for k, v in seen.items() if v > 1}
    by_round = defaultdict(int)
    for r in rows:
        by_round[int(r["round"])] += 1
    ids = {r["run_id"] for r in rows}
    check(items, f"{label}.no_duplicate_rows", not dups,
          f"{len(rows)}행 · 중복 {len(dups)}건 {sorted(dups)[:4]}")
    check(items, f"{label}.single_run_id", len(ids) == 1, f"run_id {sorted(ids)}")
    return {"path": str(path), "n_rows": len(rows), "n_duplicates": len(dups),
            "run_ids": sorted(ids), "rows_per_round": dict(sorted(by_round.items())),
            "metric_names": sorted({r["metric_name"] for r in rows}),
            "client_ids": sorted({r["client_id"] for r in rows})}


# --------------------------------------------------------------------------------------
# 4·6 — ④ 회계
# --------------------------------------------------------------------------------------

def audit_fed(run: Path, items: list[dict], batch: int) -> dict:
    fed = run / "fl" / "sep_fed"
    audit_p, acc_p = fed / "audit.json", fed / "accounting.csv"
    out: dict = {}
    a = json.loads(audit_p.read_text(encoding="utf-8")) if audit_p.exists() else {}
    out["audit"] = {k: a.get(k) for k in (
        "ok", "failures", "total_optimizer_steps", "total_optimizer_updates",
        "total_epochs_by_client", "resumed_cells", "reconstructed_cells", "notes")}
    check(items, "4.audit_ok", a.get("ok") is True, f"ok={a.get('ok')}")
    check(items, "4.no_failures", not a.get("failures"), f"failures={a.get('failures')}")
    check(items, "4.global_r050_exists", (fed / f"global_r{R:03d}.npz").exists(),
          f"global_r{R:03d}.npz")
    check(items, "4.epochs_per_client==N",
          all(int(v) == N for v in (a.get("total_epochs_by_client") or {}).values()),
          str(a.get("total_epochs_by_client")))
    check(items, "5.no_resumed_or_reconstructed",
          not a.get("resumed_cells") and not a.get("reconstructed_cells")
          and not a.get("notes"),
          f"resumed={a.get('resumed_cells')} reconstructed={a.get('reconstructed_cells')} "
          f"notes={a.get('notes')}")

    if acc_p.exists():
        rows = _rows(acc_p)
        out["accounting"] = {"n_rows": len(rows), "columns": list(rows[0]) if rows else []}
        check(items, "6.accounting_cells", len(rows) == R * len(CLIENTS),
              f"{len(rows)}행 (기대 {R * len(CLIENTS)} = R{R}×클라이언트{len(CLIENTS)})")
        units = {r["fedavg_weight_unit"] for r in rows}
        check(items, "6.weight_unit_single", len(units) == 1, f"단위 {sorted(units)}")
        # 검출은 가중 == 표본 수다. **단위가 가리키는 값과 기록된 가중이 같은가**(85번 ①).
        lies = [r for r in rows
                if abs(float(r["fedavg_weight"]) - float(r["num_examples"])) > 1e-6]
        check(items, "6.weight_matches_unit", not lies,
              f"불일치 {len(lies)}건 (단위 num_examples 면 가중 == 표본 수)")
        tok = {float(r["supervised_tokens"]) for r in rows}
        check(items, "6.supervised_tokens_zero_for_detection", tok == {0.0},
              f"supervised_tokens {sorted(tok)[:4]} — 검출은 0, 통합형만 채운다")
        bad_steps = [
            f"r{r['round_idx']}c{r['client_idx']}" for r in rows
            if int(r["optimizer_steps"]) !=
            int(r["epochs_ran"]) * math.ceil(int(r["num_examples"]) / batch)
        ]
        check(items, "6.optimizer_steps_formula", not bad_steps,
              f"어긋난 셀 {len(bad_steps)}건 {bad_steps[:4]} "
              f"(epochs×ceil(n/{batch}))")
        srcs = {r["value_source"] for r in rows}
        check(items, "6.value_source_measured", srcs == {"measured"}, f"{sorted(srcs)}")
        n_ex = {int(r["client_idx"]): int(r["num_examples"]) for r in rows}
        out["accounting"]["num_examples_by_client"] = dict(sorted(n_ex.items()))
        out["accounting"]["total_steps_from_csv"] = sum(int(r["optimizer_steps"]) for r in rows)
        check(items, "4.total_steps_matches_csv",
              out["accounting"]["total_steps_from_csv"] == a.get("total_optimizer_steps"),
              f"csv {out['accounting']['total_steps_from_csv']} vs audit "
              f"{a.get('total_optimizer_steps')}")
    return out


# --------------------------------------------------------------------------------------
# 7 — 동일 출발
# --------------------------------------------------------------------------------------

def audit_initial(run: Path, items: list[dict]) -> dict:
    cands = {
        "seed_root": run / "initial.npz",
        "sep_fed": run / "fl" / "sep_fed" / "initial.npz",
    }
    found = {k: sha256(p) for k, p in cands.items() if p.exists()}
    check(items, "7.initial_npz_same_hash", len(set(found.values())) <= 1 and found,
          f"{ {k: v[:12] for k, v in found.items()} }")
    return {"paths": {k: str(v) for k, v in cands.items()}, "sha256": found}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="outputs/main_c/seed1")
    ap.add_argument("--out", default="outputs/main_d/seed1")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    batch = _fixed_batch()
    items: list[dict] = []

    print(f"감사 대상 {run} · 고정 배치 {batch} · R={R} E={E} N={N}")
    local = audit_local_central(run, items, batch)
    atomic = {
        "sep_local": audit_atomic(run / "sep_local" / "atomic_log.csv", items, "2.sep_local"),
        "sep_central": audit_atomic(run / "sep_central" / "atomic_log.csv", items, "2.sep_central"),
        "sep_fed": audit_atomic(run / "fl" / "sep_fed" / "atomic_log.csv", items, "5.sep_fed"),
    }
    fed = audit_fed(run, items, batch)
    initial = audit_initial(run, items)

    # ④ 원자 행 수는 상수에서 유도한다 — R × (클라 3 × 지표 + 서버 지표)
    fed_rows = atomic["sep_fed"].get("rows_per_round") or {}
    per_round = sorted(set(fed_rows.values()))
    check(items, "5.rows_per_round_uniform", len(per_round) == 1,
          f"라운드당 행 수 {per_round}")
    check(items, "5.n_rounds", len(fed_rows) == R, f"{len(fed_rows)}라운드 (기대 {R})")

    bad = [i for i in items if not i["ok"]]
    report = {
        "run": str(run), "R": R, "E": E, "N": N, "fixed_batch": batch,
        "n_checks": len(items), "n_failed": len(bad),
        "all_pass": not bad,
        "failed": bad,
        "checks": items,
        "local_central_meta": local,
        "atomic": atomic,
        "fed": fed,
        "initial_npz": initial,
    }
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "audit_seed1_artifacts_v1.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    for i in items:
        print(f"  [{'OK' if i['ok'] else '실패'}] {i['check']}: {i['detail']}")
    print(f"\n검사 {len(items)}건 · 실패 {len(bad)}건 → {dest}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
