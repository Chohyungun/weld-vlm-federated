"""평가 층화 표본의 크기별 신뢰구간 폭 어림 — 사전실험(검출기 다섯 모델) 예측으로. 25번 보고의 표를 낸다.

**무엇을 하나.** 동결 평가셋에서 표본 설계(25번 §2 의 규칙)를 **복제 시드**로 여러 번 다시 뽑고, 표본마다 층 안 묶음 부트스트랩으로
세 후보 지표(M1 · 이미지 수준 결함 유무 F1 · Class-Jaccard)의 95 % 구간 폭과 모델 간 대응 차 · 회복률의 구간 반폭을 잰다.
사전실험 예측은 통합형이 아니다 — **크기와 설계의 비교만** 읽는다.

**하지 않는 것.** 공식 표본 시드(`OFFICIAL_SEED`)로는 표본을 뽑지 않는다 — 공식 목록은 다른 두 구현이 내고, 그 목록 위의 성적은
이 어림에서 보지 않는다. 복제 시드는 `ci_estimate|<설계>|<크기>|<r>` 꼴이라 공식 시드와 겹치지 않는다(시험이 아니라 문자열 꼴로 보장).

    python -X utf8 -B -m scripts.probe.eval_sample_ci_estimate --out <추적 밖 경로>/eval_sample_ci_estimate.json

입력: 동결 스냅샷 `data/interim/manifest_v1`(매니페스트 · 주석), `outputs/main_d/seed{1,2,3}/sep_*_s*.jsonl`(사전실험 예측).
출력: 집계만(id 없음).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data/interim/manifest_v1"
PRED = ROOT / "outputs/main_d"
SEEDS = {1: 20260828, 2: 20260829, 3: 20260830}
CELLS = ("sep_central", "sep_fed", "sep_local_C1", "sep_local_C2", "sep_local_C3")
CLASSES = ("100", "2011", "401", "301")            # 균열 · 기공 · 융합불량 · 슬래그혼입
ALT = {"2012": "2011"}
RARE_TYPES = {"crack", "lack_of_fusion", "slag_inclusion"}
CONF = 0.25                                        # 사전실험의 판정 문턱(evaluation.params.CONF_FALLBACK)
N_FULL = 12461
OFFICIAL_SEED = "eval_sample_v1|aihub71761_rt_v1|1f80e98b"   # 25번 §2 — 여기서는 쓰지 않는다


# ---------------------------------------------------------------- 표본 설계(25번 §2 의 규칙)
def group_bucket(n_def: int, n_img: int) -> str:
    """묶음의 이미지당 평균 결함 수 m 의 구간. 정수 비교로만 — 0 · 1(0<m<1.5) · 2-3(1.5≤m<3.5) · 4+(m≥3.5)."""
    if n_def == 0:
        return "0"
    if 2 * n_def < 3 * n_img:
        return "1"
    if 2 * n_def < 7 * n_img:
        return "2-3"
    return "4+"


def stratum_of(g: dict, design: str) -> str:
    if design == "P":                              # 지시서 문자 그대로: 재질 × 결함 수 구간
        return f"{g['material']}|{group_bucket(g['n_def'], g['n_img'])}"
    if g["rare"]:                                  # S: 희소 클래스 보유 묶음은 재질별 한 층
        return f"{g['material']}|rare"
    return f"{g['material']}|{group_bucket(g['n_def'], g['n_img'])}"


def allocation(G_h: int, n_target: int) -> int:
    """n_h = min(G_h, max(2, floor(f·G_h + 1/2))), f = n_target / N_FULL — 정수 산술."""
    if n_target >= N_FULL:
        return G_h
    return min(G_h, max(2, (2 * n_target * G_h + N_FULL) // (2 * N_FULL)))


def order_key(seed: str, group_id: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}|{group_id}".encode("utf-8")).hexdigest(), group_id


def draw(groups: list[dict], design: str, n_target: int, seed: str) -> dict[str, tuple[list[int], int]]:
    """층마다 (뽑힌 묶음 색인, 층의 묶음 수). 같은 시드에서 작은 크기 ⊂ 큰 크기(접두)."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by[stratum_of(g, design)].append(i)
    out = {}
    for h, idx in sorted(by.items()):
        n_h = allocation(len(idx), n_target)
        chosen = sorted(idx, key=lambda i: order_key(seed, groups[i]["group_id"]))[:n_h]
        out[h] = (chosen, len(idx))
    return out


# ---------------------------------------------------------------- 입력
def load_inputs() -> tuple[list[dict], dict, dict]:
    sys.path.insert(0, str(ROOT))
    from data import manifest_io as M
    from evaluation.eval_set import read_gold
    from evaluation.metrics import box_f1 as BF

    M.verify_snapshot(SNAP)
    df = M.read_manifest(SNAP / M.MANIFEST_FILENAME)
    ev = df[df["split"] == "eval"]
    assert len(ev) == N_FULL, len(ev)
    gmap: dict[str, dict] = {}
    img_group: dict[str, str] = {}
    for r in ev.itertuples(index=False):
        g = gmap.setdefault(r.group_id, {"group_id": r.group_id, "material": r.material, "n_def": 0, "n_img": 0,
                                         "rare": False, "images": []})
        g["n_def"] += int(r.n_defects)
        g["n_img"] += 1
        g["rare"] |= bool(set(str(r.defect_types).split(";")) & RARE_TYPES)
        g["images"].append(r.image_id)
        img_group[r.image_id] = r.group_id
    groups = sorted(gmap.values(), key=lambda g: g["group_id"])
    gidx = {g["group_id"]: i for i, g in enumerate(groups)}
    ids = set(img_group)
    codes, boxes = read_gold(SNAP, ids)
    gold = {i: [(ALT.get(c, c), tuple(b)) for c, b in boxes.get(i, [])] for i in ids}
    gold_codes = {i: {ALT.get(c, c) for c in codes.get(i, set())} & set(CLASSES) for i in ids}

    feats = {}                                     # (seed, cell) → G × 17
    for s, sv in SEEDS.items():
        for cell in CELLS:
            path = PRED / f"seed{s}" / f"{cell}_s{sv}.jsonl"
            pred = {i: [] for i in ids}
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec["image_id"] not in ids:
                        continue
                    pred[rec["image_id"]] = [(ALT.get(d["iso_code"], d["iso_code"]), tuple(d["bbox_px"]))
                                             for d in rec.get("defects") or []
                                             if d.get("bbox_px") is not None and (d.get("score") or 0) >= CONF
                                             and ALT.get(d["iso_code"], d["iso_code"]) in CLASSES]
            table = BF.build_counts(pred, gold, CLASSES)
            X = np.zeros((len(groups), 17))
            rowg = np.array([gidx[img_group[i]] for i in table.image_ids])
            for k, arr in enumerate((table.tp, table.fp, table.fn)):
                np.add.at(X[:, 4 * k:4 * k + 4], rowg, arr)
            for i in table.image_ids:
                gi = gidx[img_group[i]]
                pc = {c for c, _ in pred[i]}
                gc = gold_codes[i]
                pp, gp = bool(pc), bool(gc)
                X[gi, 12] += pp and gp
                X[gi, 13] += pp and not gp
                X[gi, 14] += gp and not pp
                X[gi, 15] += 1.0 if not (pc | gc) else len(pc & gc) / len(pc | gc)
                X[gi, 16] += 1
            feats[(s, cell)] = X
    return groups, feats, {"n_groups": len(groups), "n_images": len(ids)}


# ---------------------------------------------------------------- 지표
def metrics(T: np.ndarray) -> dict[str, np.ndarray]:
    """T: (… × 17) 가중 합 → M1 macro(고정 4 클래스, 지지량 0 이면 NaN) · 이미지 F1 · Class-Jaccard."""
    tp, fp, fn = T[..., 0:4], T[..., 4:8], T[..., 8:12]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1c = np.where(tp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        m1 = f1c.mean(-1)                          # 지지량 0 인 클래스가 있으면 NaN(support 미정의)
        itp, ifp, ifn = T[..., 12], T[..., 13], T[..., 14]
        imf1 = 2 * itp / (2 * itp + ifp + ifn)
        cj = T[..., 15] / T[..., 16]
    return {"M1": m1, "imgF1": imf1, "CJ": cj}


def recovery(F, C, L):
    """R = (F − L̄)/(C − L̄). 분모 ≤ 0 이면 분자 부호의 ±무한(10-01 규약)."""
    lbar = np.mean(L, axis=0)
    num, den = F - lbar, C - lbar
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, np.where(num >= 0, np.inf, -np.inf))
    return np.where(np.isnan(num) | np.isnan(den), np.nan, r)


def ci(draws: np.ndarray) -> tuple[float, float, float]:
    """95 % 백분위(차수 통계). NaN(support 미정의)은 빼고 그 비율을 따로 낸다. ±무한은 남긴다."""
    d = draws[~np.isnan(draws)]
    if len(d) == 0:
        return float("nan"), float("nan"), 1.0
    s = np.sort(d)
    lo, hi = s[int(np.floor(0.025 * (len(s) - 1)))], s[int(np.ceil(0.975 * (len(s) - 1)))]
    return float(lo), float(hi), float(1 - len(d) / len(draws))


def evaluate(groups, feats, design: str, n_target: int, seed: str, B: int, rng_seed: int, seeds_used) -> dict:
    sel = draw(groups, design, n_target, seed)
    G = len(groups)
    w0 = np.zeros(G)
    W = np.zeros((B, G))
    rng = np.random.default_rng(rng_seed)
    n_img = 0
    for h, (chosen, G_h) in sel.items():
        n_h = len(chosen)
        wh = G_h / n_h
        w0[chosen] = wh
        n_img += sum(groups[i]["n_img"] for i in chosen)
        W[:, chosen] = rng.multinomial(n_h, np.full(n_h, 1.0 / n_h), size=B) * wh
    out = {"n_images": n_img, "n_groups": int(sum(len(c) for c, _ in sel.values())), "max_weight": float(w0.max())}
    any_x = next(iter(feats.values()))
    gold_cnt = (w0 > 0) @ (any_x[:, 0:4] + any_x[:, 8:12])     # 표본 안 클래스별 정답 박스 수(가중 없음, 모델 무관)
    out["support_boxes"] = {c: float(v) for c, v in zip(CLASSES, gold_cnt)}
    mat = np.array([g["material"] for g in groups])
    point, boot = {}, {}
    for key, X in feats.items():
        if key[0] not in seeds_used:
            continue
        point[key] = metrics(w0 @ X)
        boot[key] = metrics(W @ X)
    for m in ("M1", "imgF1", "CJ"):
        # 시드별 값 → 시드 평균(같은 추첨을 모든 모델 · 시드에)
        P = {c: np.mean([point[(s, c)][m] for s in seeds_used]) for c in CELLS}
        D = {c: np.mean([boot[(s, c)][m] for s in seeds_used], axis=0) for c in CELLS}
        lo, hi, und = ci(D["sep_central"])
        out[f"{m}.central"] = {"point": float(P["sep_central"]), "half": (hi - lo) / 2, "undef": und}
        if m == "M1":
            for sub in ("ST", "AL"):
                mask = (mat == sub).astype(float)
                ps = np.mean([metrics((w0 * mask) @ feats[(s, "sep_central")])["M1"] for s in seeds_used])
                ds = np.mean([metrics((W * mask) @ feats[(s, "sep_central")])["M1"] for s in seeds_used], axis=0)
                lo, hi, und = ci(ds)
                out[f"M1.central.{sub}"] = {"point": float(ps), "half": (hi - lo) / 2, "undef": und}
        lo, hi, und = ci(D["sep_central"] - D["sep_fed"])
        out[f"{m}.central_minus_fed"] = {"point": float(P["sep_central"] - P["sep_fed"]), "half": (hi - lo) / 2, "undef": und}
        rs_point = np.mean([recovery(point[(s, "sep_fed")][m], point[(s, "sep_central")][m],
                                     [point[(s, c)][m] for c in CELLS[2:]]) for s in seeds_used])
        rs = np.mean([recovery(boot[(s, "sep_fed")][m], boot[(s, "sep_central")][m],
                               np.stack([boot[(s, c)][m] for c in CELLS[2:]])) for s in seeds_used], axis=0)
        lo, hi, und = ci(rs)
        out[f"{m}.R"] = {"point": float(rs_point), "half": (hi - lo) / 2 if np.isfinite(hi - lo) else float("inf"),
                         "undef": und, "inf_share": float(np.mean(np.isinf(rs)))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=60)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    groups, feats, info = load_inputs()
    res = {"info": info, "conf": CONF, "B": a.B, "replicates": a.replicates, "classes": CLASSES,
           "strata": {}, "rows": []}
    for design in ("P", "S"):
        st = defaultdict(lambda: [0, 0])
        for g in groups:
            st[stratum_of(g, design)][0] += 1
            st[stratum_of(g, design)][1] += g["n_img"]
        res["strata"][design] = {h: {"groups": v[0], "images": v[1]} for h, v in sorted(st.items())}
    for seeds_used, tag in (((1,), "seed1"), ((1, 2, 3), "seed123")):
        for design in ("P", "S"):
            for n in (2000, 3000, 4000, N_FULL):
                reps = 1 if n == N_FULL else a.replicates
                rows = [evaluate(groups, feats, design, n, f"ci_estimate|{design}|{n}|{r}", a.B, 1000 * r + n, seeds_used)
                        for r in range(reps)]
                agg = {"seeds": tag, "design": design, "n_target": n,
                       "n_images_mean": float(np.mean([x["n_images"] for x in rows])),
                       "n_images_range": [int(min(x["n_images"] for x in rows)), int(max(x["n_images"] for x in rows))],
                       "n_groups": rows[0]["n_groups"], "max_weight": rows[0]["max_weight"],
                       "support_boxes_mean": {c: float(np.mean([x["support_boxes"][c] for x in rows])) for c in CLASSES},
                       "support_boxes_min": {c: float(min(x["support_boxes"][c] for x in rows)) for c in CLASSES}}
                for k in rows[0]:
                    if "." not in k:
                        continue
                    halves = np.array([x[k]["half"] for x in rows])
                    pts = np.array([x[k]["point"] for x in rows])
                    agg[k] = {"half_median": float(np.median(halves)), "half_p90": float(np.quantile(halves, 0.9)),
                              "point_mean": float(np.nanmean(pts)), "point_sd": float(np.nanstd(pts)),
                              "undef_mean": float(np.mean([x[k]["undef"] for x in rows]))}
                    if k.endswith(".R"):
                        agg[k]["share_half_le_0.25"] = float(np.mean(halves <= 0.25))
                        agg[k]["inf_share_mean"] = float(np.mean([x[k]["inf_share"] for x in rows]))
                res["rows"].append(agg)
                print(tag, design, n, f"{time.time() - t0:.0f}s", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
