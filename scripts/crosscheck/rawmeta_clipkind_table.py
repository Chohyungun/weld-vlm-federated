"""미니스펙 2판 6절의 결정 규칙 — `clip_kind` 가 참여자와 상관되는가.

    uv run python -m scripts.crosscheck.rawmeta_clipkind_table <스냅샷> <출력 폴더>

규칙(수를 보기 전에 적은 것): (가) C1 · C2 · C3 의 학습 + 검증 주석 행으로 참여자 × `clip_kind` 표를 내고
독립성을 검정한다 — 기대 빈도가 5 미만인 칸이 있으면 정확 검정 대신 카이제곱 통계의 순열 검정(시드 고정)이다.
**p < 0.01 이면 `clip_kind` · `clip_applied` 를 진단 전용으로 내린다.** (나) 결함 종류 (다) ST 안의 id 구간별 분포는
해석용으로 보고만 한다. **평가 행은 표에 넣지 않는다.** 표는 스냅샷 밖 보고 파일에 쓴다.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

from data import absorb
from data.id_strata import stratum_of

ALPHA = 0.01
N_PERM = 20000
SEED = 20261001
KINDS = ("none", "boundary", "polygon_repair", "both", "empty_bbox")


def _chi2(table: np.ndarray) -> float:
    t = table[:, table.sum(axis=0) > 0]
    return float(chi2_contingency(t, correction=False)[0]) if t.shape[1] > 1 else 0.0


def independence(groups: list[str], kinds: list[str]) -> dict:
    gs, ks = sorted(set(groups)), [k for k in KINDS if k in set(kinds)]
    gi, ki = {g: i for i, g in enumerate(gs)}, {k: i for i, k in enumerate(ks)}
    table = np.zeros((len(gs), len(ks)), dtype=np.int64)
    for g, k in zip(groups, kinds):
        table[gi[g], ki[k]] += 1
    expected = chi2_contingency(table, correction=False)[3] if table.shape[1] > 1 else table.astype(float)
    small = bool((expected < 5).any())
    stat = _chi2(table)
    if small:
        rng = np.random.default_rng(SEED)
        g_arr = np.array([gi[g] for g in groups])
        k_arr = np.array([ki[k] for k in kinds])
        ge = 0
        ng, nk = table.shape
        for _ in range(N_PERM):
            perm = rng.permutation(k_arr)
            t = np.bincount(g_arr * nk + perm, minlength=ng * nk).reshape(ng, nk)
            ge += _chi2(t) >= stat - 1e-12
        p = (ge + 1) / (N_PERM + 1)
        method = f"카이제곱 통계의 순열 검정(N={N_PERM}, 시드 {SEED})"
    else:
        p = float(chi2_contingency(table, correction=False)[1])
        method = "카이제곱 독립성 검정"
    return {"groups": gs, "kinds": ks, "table": table.tolist(), "chi2": stat, "p": p, "method": method,
            "small_expected": small}


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    snap = absorb.load_absorbed(root)
    man = snap.manifest.set_index("image_id")
    ann = snap.annotations.set_index("ann_id")
    d = snap.absorb_defect
    d = d[d["image_id"].map(man["split"]) != "eval"]                 # 평가 행은 넣지 않는다
    client = d["image_id"].map(man["client"]).astype(str).tolist()
    kinds = d["clip_kind"].astype(str).tolist()
    primary = independence(client, kinds)
    by_type: dict[str, Counter] = defaultdict(Counter)
    for aid, k in zip(d["ann_id"], kinds):
        by_type[str(ann.loc[aid, "defect_type"])][k] += 1
    st = d[d["image_id"].map(man["material"]) == "ST"]
    strata = stratum_of(st["image_id"].tolist(), 16)
    by_stratum: dict[str, Counter] = defaultdict(Counter)
    for s, k in zip(strata, st["clip_kind"].astype(str)):
        by_stratum[str(int(s))][k] += 1
    decision = "강등 — clip_kind · clip_applied 를 진단 전용으로 내린다" if primary["p"] < ALPHA else "유지"
    result = {
        "규칙": f"C1·C2·C3 학습+검증 주석 행의 참여자 × clip_kind 독립성, p < {ALPHA} 이면 강등. 평가 행 제외",
        "primary_by_client": primary,
        "decision": decision,
        "C3_is_AL": "C3 ≡ AL 이라 참여자 차이는 재질 차이와 같은 값이다",
        "by_defect_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "by_st_id_stratum_k16": {k: dict(v) for k, v in sorted(by_stratum.items(), key=lambda x: int(x[0]))},
        "보고만": "결함 종류 · ST id 구간은 해석용이다. 결정에 쓰지 않는다",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "clipkind_table.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("primary_by_client", "decision")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
