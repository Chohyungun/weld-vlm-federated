"""평가 층화 표본의 크기별 신뢰구간 폭 어림 — 사전실험(검출기 다섯 모델) 예측으로. 25번 보고의 표를 낸다.

**무엇을 하나.** 동결 평가셋에서 표본 설계(25번 §2 의 규칙)를 **복제 시드**로 여러 번 다시 뽑고, 표본마다 층 안 묶음 부트스트랩으로
세 후보 지표(M1 · 이미지 수준 결함 유무 F1 · Class-Jaccard)의 95 % 구간 폭과 모델 간 대응 차 · 회복률의 구간 반폭을 잰다.
사전실험 예측은 통합형이 아니다 — **크기와 설계의 비교만** 읽는다.

**하지 않는 것.** 공식 표본 시드(`OFFICIAL_SEED`)로는 표본을 뽑지 않는다 — 공식 목록은 다른 두 구현이 내고, 그 목록 위의 성적은
이 어림에서 보지 않는다. 복제 시드는 `ci_estimate|<설계>|<크기>|<r>` 꼴이라 공식 시드와 겹치지 않는다(시험이 아니라 문자열 꼴로 보장).

    python -X utf8 -B -m scripts.probe.eval_sample_ci_estimate --out <추적 밖 경로>/eval_sample_ci_estimate.json

입력: 동결 스냅샷 `data/interim/manifest_v1`(매니페스트 · 주석), `outputs/main_d/seed{1,2,3}/sep_*_s*.jsonl`(사전실험 예측).
출력: 집계만(id 없음).

**구간 규약은 본채점과 같다**(외부 검토 137-2 M-03, 2026-10-14). 회복률은 `evaluation.recovery_interval` 이 추첨을 분류한다 —
분모 ≤ 0 이고 분자 == 0 이면 판정 불가(`numerator_zero`)이고 +무한이 아니다. 시드끼리 ±무한이 갈리면 판정 불가(`sign_conflict`),
어느 항이 수가 아니면(지지량 0) 지지량 제외다. 점추정은 분모 실패 · 지지량 0 이면 수를 내지 않는다(`absent`). 지표 구간은
`percentile_rule` 의 순위 규칙이다. **수가 아닌 값은 상태 칸으로 싣는다** — 산출 JSON 에 `NaN` · `Infinity` 를 쓰지 않는다.
앞 판(이 규약 전)은 분자 0 을 +무한으로, 시드끼리 갈린 ±무한을 지지량 미정의와 한 칸(`undef`)으로 셌다. 그 판의 산출은 그때
규약의 기록이다 — 고치지 않고, 이번 수정으로 다시 돌리지 않았다(실제 평가 입력을 열지 않는다).
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

from evaluation import percentile_rule as PR
from evaluation import recovery_interval as RI

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
    if design == "P":                              # 명세 문자 그대로: 재질 × 결함 수 구간
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


def value_state(x) -> dict:
    """수 하나를 표준 JSON 꼴로 — 유한이면 수, 아니면 `None` 과 상태(`undefined` · `pos_inf` · `neg_inf`)."""
    b = PR.bound_of(float(x))
    return {"value": b.value, "state": b.state}


def metric_interval(draws) -> dict:
    """지표 추첨의 95 % 구간 — 등록 순위 규칙(`percentile_rule`). `NaN`(지지량 0 · 분모 0)은 미정의로 세고 뺀다."""
    iv, n_def, n_undef = PR.defined_interval(draws)
    return {**iv.as_dict(), "n_defined": n_def, "n_undefined": n_undef}


def _support_ok(fed: np.ndarray, central: np.ndarray, locals_: np.ndarray) -> np.ndarray:
    return np.isfinite(fed).all(axis=-1) & np.isfinite(central).all(axis=-1) & np.isfinite(locals_).all(axis=(-2, -1))


def recovery_draws(fed, central, locals_) -> dict:
    """회복률 `R = (F − L̄)/(C − L̄)` 의 구간 — 본채점과 같은 분류(`recovery_interval`).

    모양은 연합 · 중앙 `(추첨, 시드)`, 로컬 `(추첨, 시드, 로컬 수)`. 어느 항이 수가 아닌 추첨은 지지량 제외다 —
    그 추첨의 값은 분류기에 넘기기 전에 0 으로 바꾸고 지지량 표시로 뺀다(분류가 지지량을 먼저 본다).
    """
    f, c, l = (np.asarray(x, dtype=float) for x in (fed, central, locals_))
    ok = _support_ok(f, c, l)
    f = np.where(ok[:, None], f, 0.0)
    c = np.where(ok[:, None], c, 0.0)
    l = np.where(ok[:, None, None], l, 0.0)
    return RI.recovery_interval(f, c, l, ok, alpha=PR.REGISTERED_ALPHA, n_resamples=int(f.shape[0])).as_dict()


def recovery_point(fed, central, locals_) -> dict:
    """시드별 `R_s` 와 `R̄` — 본채점의 점추정(`point_recovery`). 분모 실패면 `absent`(사유 `denominator`), 어느 항이 수가
    아니면 `absent`(사유 `support`). 무한을 적지 않는다."""
    f, c, l = (np.asarray(x, dtype=float) for x in (fed, central, locals_))
    if not (np.isfinite(f).all() and np.isfinite(c).all() and np.isfinite(l).all()):
        return {"R_bar": None, "state": "absent", "reason": "support", "convention_id": PR.CONVENTION_ID}
    p = RI.point_recovery(f, c, l)
    return {"R_bar": p["R_bar"], "state": p["state"], "reason": p["reason"], "convention_id": p["convention_id"]}


def _states(states) -> dict[str, int]:
    out: dict[str, int] = {}
    for st in states:
        out[st] = out.get(st, 0) + 1
    return dict(sorted(out.items()))


def _finite(xs) -> np.ndarray:
    return np.array([x for x in xs if x is not None], dtype=float)


def aggregate(cells: list[dict]) -> dict:
    """복제마다의 칸을 모은다. 수의 요약은 **유한한 것만**으로 내고, 수가 아닌 것은 상태별 건수로 함께 싣는다 — 덮지 않는다."""
    ivs = [c["interval"] for c in cells]
    half = _finite(i["half_width"] for i in ivs)
    out = {"replicates": len(cells), "half_states": _states(i["half_width_state"] for i in ivs),
           "half_median": float(np.median(half)) if half.size else None,
           "half_p90": float(np.quantile(half, 0.9)) if half.size else None}
    if "R_bar" in cells[0]["point"]:
        pts = _finite(c["point"]["R_bar"] for c in cells)
        out["point_states"] = _states(c["point"]["state"] for c in cells)
        for key in RI.COUNTERS:
            out[f"{key}_mean"] = float(np.mean([i[key] for i in ivs]))
        out["share_half_le_0.25"] = float(np.mean([i["half_width_state"] == "finite" and i["half_width"] <= 0.25
                                                   for i in ivs]))
    else:
        pts = _finite(c["point"]["value"] for c in cells)
        out["point_states"] = _states(c["point"]["state"] for c in cells)
        out["undefined_share_mean"] = float(np.mean([i["n_undefined"] / (i["n_defined"] + i["n_undefined"]) for i in ivs]))
    out["point_mean"] = float(np.mean(pts)) if pts.size else None
    out["point_sd"] = float(np.std(pts)) if pts.size else None
    return out


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
        out[f"{m}.central"] = {"point": value_state(P["sep_central"]), "interval": metric_interval(D["sep_central"])}
        if m == "M1":
            for sub in ("ST", "AL"):
                mask = (mat == sub).astype(float)
                ps = np.mean([metrics((w0 * mask) @ feats[(s, "sep_central")])["M1"] for s in seeds_used])
                ds = np.mean([metrics((W * mask) @ feats[(s, "sep_central")])["M1"] for s in seeds_used], axis=0)
                out[f"M1.central.{sub}"] = {"point": value_state(ps), "interval": metric_interval(ds)}
        out[f"{m}.central_minus_fed"] = {"point": value_state(P["sep_central"] - P["sep_fed"]),
                                         "interval": metric_interval(D["sep_central"] - D["sep_fed"])}
        # 회복률 — 시드를 한 축으로 묶어 본채점의 분류기에 넘긴다(시드 평균을 먼저 내지 않는다)
        fed = np.stack([boot[(s, "sep_fed")][m] for s in seeds_used], axis=1)
        cen = np.stack([boot[(s, "sep_central")][m] for s in seeds_used], axis=1)
        loc = np.stack([np.stack([boot[(s, c)][m] for c in CELLS[2:]], axis=-1) for s in seeds_used], axis=1)
        p_fed = np.array([point[(s, "sep_fed")][m] for s in seeds_used], dtype=float)
        p_cen = np.array([point[(s, "sep_central")][m] for s in seeds_used], dtype=float)
        p_loc = np.array([[point[(s, c)][m] for c in CELLS[2:]] for s in seeds_used], dtype=float)
        out[f"{m}.R"] = {"point": recovery_point(p_fed, p_cen, p_loc), "interval": recovery_draws(fed, cen, loc)}
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
                    agg[k] = aggregate([x[k] for x in rows])
                res["rows"].append(agg)
                print(tag, design, n, f"{time.time() - t0:.0f}s", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # 표준 JSON 만 — 수가 아닌 값이 상태 칸을 거치지 않고 새면 여기서 멈춘다
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float, allow_nan=False), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
