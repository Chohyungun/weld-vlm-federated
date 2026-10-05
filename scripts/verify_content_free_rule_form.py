"""무내용 대조의 두 규칙 형태를 같은 구간 위에서 재현한다 — 17번 §13-3 의 답.

등록 Macro-F1 0.9149 와 재구성 0.9128 이 0.0021 갈린 원인이 절차 잔차인지 규칙 형태
차이인지를 실측으로 가른다. 같은 족(idq512)·같은 절단점(train+val 분위)·같은 적합 모집단
(train+val)·같은 채점 모집단(eval 12,461) 위에서 두 형태를 적합해 eval Macro-F1 을 낸다.

  (A) 클래스별 F1 최적 접두사 — `scripts/recompute_baselines.fit_rule` (등록 0.9149 의 규칙)
  (D) 구간 최빈 정답 코드 집합 — `evaluation.content_free._majority_of` 와 같은 규칙

    uv run python scripts/verify_content_free_rule_form.py

CPU 전용, 읽기만 한다. 파일을 쓰지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from recompute_baselines import (
    Population,
    apply_rule,
    build_gold,
    fit_rule,
    make_cuts,
)

from data.id_strata import stratum_of
from data.manifest_io import load_snapshot

SNAPSHOT = REPO_ROOT / "data/interim/manifest_v1"
K = 512
MIN_SUPPORT = 30


def majority_rule(pop: Population, fam, gold, classes) -> np.ndarray:
    """구간별 최빈 정답 코드 집합 → 셀×클래스 마스크. 동률은 D 와 같은 규칙으로 깬다."""
    cell, size = pop.cell_ids(fam)
    scope = frozenset(classes)
    sel = np.zeros((size, len(classes)), dtype=bool)
    for b in range(size):
        idx = np.flatnonzero(cell == b)
        if idx.size == 0:
            continue
        counts: dict[frozenset, int] = {}
        for i in idx:
            key = frozenset(gold[pop.ids[i]]) & scope
            counts[key] = counts.get(key, 0) + 1
        maj = max(counts.items(), key=lambda kv: (kv[1], tuple(sorted(kv[0], reverse=True))))[0]
        for j, c in enumerate(classes):
            sel[b, j] = c in maj
    return sel


def main() -> int:
    snap = load_snapshot(SNAPSHOT)
    m, t = snap.manifest, snap.tiles
    prov = dict(zip(t["image_id"], t["provenance"], strict=True))
    m = m.assign(prov=m["image_id"].map(prov))
    m["id_num"] = m["image_id"].str.rsplit(":", n=1).str[-1].astype("int64")
    m["group_size"] = pd.to_numeric(m["group_size"], errors="coerce").fillna(1).astype("int64")
    m["file_bytes"] = 0  # fsz 축은 쓰지 않는다

    gold, _ = build_gold(snap, list(m["image_id"]))
    tv = m[m["split"] != "eval"].reset_index(drop=True)
    ev = m[m["split"] == "eval"].reset_index(drop=True)
    cuts = make_cuts(tv)
    classes = sorted({c for i in ev["image_id"] for c in gold[i]})
    print(f"클래스 {classes} · train+val {len(tv):,} · eval {len(ev):,}")

    P_tv = Population(tv, gold, classes, cuts)
    P_ev = Population(ev, gold, classes, cuts)
    fam = (("idq", K),)
    n_tv, pos_tv = P_tv.counts(fam)
    n_ev, pos_ev = P_ev.counts(fam)

    # (A) — search() 와 같이 미관측·과소표본 셀은 전역 규칙으로 떨어뜨린다
    gn, gpos = P_tv.counts(())
    fb = fit_rule(gn, gpos, 1)[0]
    sel_a = fit_rule(n_tv, pos_tv, MIN_SUPPORT)
    unseen = (n_tv < MIN_SUPPORT) & (n_ev > 0)
    sel_a[unseen] = fb
    f1_a = apply_rule(n_ev, pos_ev, sel_a)

    # (D)
    sel_d = majority_rule(P_tv, fam, gold, classes)
    f1_d = apply_rule(n_ev, pos_ev, sel_d)

    print(f"(A) 클래스별 F1 최적 접두사   eval Macro-F1 = {f1_a:.6f}   (등록 0.9149)")
    print(f"(D) 구간 최빈 코드 집합       eval Macro-F1 = {f1_d:.6f}   (D 재구성 0.912842)")
    print(f"차 = {f1_a - f1_d:.6f}")
    diff = int((sel_a != sel_d).any(axis=1).sum())
    print(f"두 규칙의 예측이 갈리는 구간 = {diff} / {int((n_tv > 0).sum())}")

    cell_ev, _ = P_ev.cell_ids(fam)
    strata = np.asarray(stratum_of(list(ev["image_id"]), K, SNAPSHOT))
    same = bool((cell_ev == strata).all())
    print(
        f"A 절단점 구간 == D stratum_of 구간 (eval 전 이미지): {same} · 불일치 {int((cell_ev != strata).sum())}"
    )

    prev = pos_ev.sum(axis=0) / n_ev.sum()
    print(f"eval 평균 유병률 = {float(prev.mean()):.6f}  (all_positive 의 AP, 정의상)")
    print(
        f"경계: 0.9149² = {0.9149**2:.4f} > H 0.8355 ≥ 0.9128² = {0.9128**2:.4f}"
        " → H 는 등록 F1 의 예측기가 아니라 재구성 F1 의 예측기다"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
