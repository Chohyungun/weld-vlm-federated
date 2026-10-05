"""프레임 진단의 학습 표본 뽑기 — 독립 구현(트랙 G, 2026-10-05).

리허설 미니스펙 2판 §13-3 의 두 구현 계약 가운데 독립 구현이다. **학습 쪽 구현을 열지 않고** 아래 세 글만 읽고 짰다.

- 총괄 결정 20(`docs/dev_log/2026-09-30-리허설/20_총괄결정_진단표본.md`, D26233880B72…)
- 총괄 결정 04 의 ② · ③(`docs/dev_log/2026-10-01-전수메타데이터/04_총괄결정_1단계와_리허설목록.md`, E1603CC8FC05…)
- 리허설 미니스펙 2판 §3-1 · §4-4 · §5-3 · §13

입력 규약(지시서 `dispatch_G_20261005c`): 매니페스트는 판독기의 **학습 · 검증 뷰만** 읽는다(평가 행 0).
층은 `strata_key`, 묶음은 `group_id`, 간선은 `configs/rehearsal_uni.yaml` 의 `exclude.near_dup_edges`,
T 와 시드는 진단 계획 `configs/frame_diag_uni.yaml` 의 `train.rows.central` · `sampling_seed` 다.

규칙(이 구현이 읽은 대로 — 아래가 기본안 A0 이다). 읽기가 갈릴 수 있는 자리는 같은 실행에서 변형으로 따로 낸다 —
A1 몫과 무관하게 합이 T 에 닿으면 모두 멈춤(2판 §13-2 의 3 처럼) · A2 무게를 격리 전 행 수로 ·
A3 후보를 페어 파일의 학습 행으로 좁힘 · A4 몫이 0 인 층에도 묶음 하나 · A5 묶음 안 순서를 매니페스트 순서로 ·
A6 후보에 검증 행까지. 정본은 A0 하나이고 변형은 대조가 갈릴 때 원인을 찾는 데만 쓴다.

1. 후보 = 학습 · 검증 뷰의 행 가운데 `split == train.split`(`train`). 평가 행은 뷰에 없다.
2. 격리 — 간선과 **같은 묶음 간선**으로 이은 성분 가운데 평가셋에 닿은 성분의 묶음 전체를 뺀다.
   이 구현은 평가 행을 읽지 않으므로 **뷰 밖의 끝점**(평가 id 이거나 매니페스트에 없는 id)을 모두
   "평가에 닿음" 으로 센다. 결정 04 ①(매니페스트에 없는 id 는 입력 오류)은 이 입력 규약으로는 가를 수 없다 —
   뷰 밖 끝점의 수를 기록에 싣는다.
3. 층 = `strata_key`. 한 묶음의 행은 한 층에만 있어야 한다(아니면 멈춘다). 층의 순서는 키의 문자열 오름차순.
4. 몫 — 층의 무게 `w_s` = 격리 뒤 그 층의 후보 행 수, `W = Σ w_s`. 최대 잔여 방식으로 `Σ q_s = T` 인 정수 몫.
   먼저 `⌊T·w_s/W⌋`, 남는 수를 잔여가 큰 층부터 하나씩, 잔여가 같으면 층 키 순서가 앞선 층에. 정확한 유리수로 센다.
5. 채우기 — 층마다(층 키 순서) 묶음을 `sha256(f"{group_id}|{sampling_seed}")` 16진 문자열 오름차순으로 보며,
   그 층에 넣은 행 수가 `q_s` 이상이 될 때까지 묶음째(그 묶음의 후보 행 전부) 넣는다. 마지막 묶음이 몫을 넘기면
   그 묶음까지 넣고 **다음 층의 몫에서 빼지 않는다** — 넘은 수(초과)를 층마다 기록한다. 몫을 못 채우고 묶음이
   떨어지면 멈춘다. `q_s = 0` 인 층은 넣지 않는다.
6. `train.avoid_multiple_of_accum` 이 참이면 결정 04 ② 를 건다(합이 32 의 배수가 아닐 때까지 다음 묶음). 진단 계획은
   거짓이라 걸지 않는다.
7. 출력 순서 = 층 키 순서 → 묶음 해시 순서 → 묶음 안 `image_id` 문자열 오름차순. 목록 파일은 한 줄에 `image_id`
   하나, UTF-8, LF, 끝 개행. 정렬한 집합 목록도 함께 낸다(순서와 무관한 값 대조용).

평가 행 · 주석 · 이미지 · 페어의 타깃 칸은 읽지 않는다. `--pairs` 를 주면 페어 파일에서 `image_id` · `split` 만 읽는다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from data.manifest_view import read_manifest_columns

VIEW_COLUMNS = ["image_id", "split", "group_id", "strata_key"]
ACCUM = 32
VARIANTS = ("A0", "A1_global_stop", "A2_weights_before_exclusion", "A3_pairs_pool",
            "A4_min_one_group", "A5_manifest_order_in_group", "A6_train_and_val_pool")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def group_key(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{group_id}|{seed}".encode()).hexdigest()


def read_edges(path: Path, want_sha: str) -> tuple[list[tuple[str, str]], str]:
    raw = path.read_bytes()
    got = sha256_bytes(raw)
    if got != want_sha:
        raise SystemExit(f"간선 파일의 sha256 이 계획과 다르다: {got} != {want_sha}")
    text = raw.decode("utf-8")
    rows = list(csv.reader(text.splitlines()))
    if not rows or rows[0] != ["a_id", "b_id"]:
        raise SystemExit("간선 파일의 머리행이 a_id,b_id 가 아니다")
    edges = []
    for n, r in enumerate(rows[1:], 2):
        if len(r) != 2 or not r[0] or not r[1]:
            raise SystemExit(f"간선 파일 {n} 줄의 꼴이 다르다")
        edges.append((r[0], r[1]))
    return edges, got


class UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def excluded_groups(view: list[dict], edges: list[tuple[str, str]]) -> tuple[set[str], dict]:
    """간선 + 같은 묶음 간선으로 이은 성분 가운데 뷰 밖 끝점(평가 · 미상)에 닿은 성분의 묶음 전체."""
    group_of = {r["image_id"]: r["group_id"] for r in view}
    uf = UF()
    for g in set(group_of.values()):
        uf.find(g)
    touched: set[str] = set()
    outside_ids: set[str] = set()
    n_in_in = n_in_out = n_out_out = n_self = 0
    for a, b in edges:
        if a == b:
            n_self += 1
        ga, gb = group_of.get(a), group_of.get(b)
        if ga is not None and gb is not None:
            uf.union(ga, gb)
            n_in_in += 1
        elif ga is not None or gb is not None:
            touched.add(ga if ga is not None else gb)
            outside_ids.add(b if ga is not None else a)
            n_in_out += 1
        else:
            outside_ids.update((a, b))
            n_out_out += 1
    bad_roots = {uf.find(g) for g in touched}
    drop = {g for g in set(group_of.values()) if uf.find(g) in bad_roots}
    comps = defaultdict(set)
    for g in drop:
        comps[uf.find(g)].add(g)
    stats = {"edges": len(edges), "edges_both_in_view": n_in_in, "edges_one_outside_view": n_in_out,
             "edges_both_outside_view": n_out_out, "self_loops": n_self,
             "outside_view_ids": len(outside_ids), "groups_touching_outside": len(touched),
             "components_dropped": len(comps), "groups_dropped": len(drop),
             "note": "뷰 밖 끝점 = 평가 id 이거나 매니페스트에 없는 id — 이 입력 규약으로는 가르지 않는다"}
    return drop, stats


def quotas(weights: dict[str, int], total: int) -> dict[str, dict]:
    keys = sorted(weights)
    W = sum(weights.values())
    out = {}
    for k in keys:
        exact = Fraction(total * weights[k], W)
        fl = exact.numerator // exact.denominator
        out[k] = {"w": weights[k], "exact": exact, "floor": fl, "rem": exact - fl, "q": fl}
    left = total - sum(v["floor"] for v in out.values())
    order = sorted(keys, key=lambda k: (-out[k]["rem"], k))
    for k in order[:left]:
        out[k]["q"] += 1
    for i, k in enumerate(order):
        out[k]["rem_rank"] = i + 1
    return out


def select(view: list[dict], drop: set[str], *, split: str, total: int, seed: int,
           avoid_multiple: bool, variant: str, pair_ids: set[str] | None,
           manifest_pos: dict[str, int]) -> tuple[list[str], dict]:
    rows = [r for r in view if r["split"] == split or variant == "A6_train_and_val_pool"]
    if variant == "A3_pairs_pool":
        rows = [r for r in rows if r["image_id"] in pair_ids]
    strata_of_group: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        strata_of_group[r["group_id"]].add(r["strata_key"])
    multi = {g: s for g, s in strata_of_group.items() if len(s) > 1}
    if multi:
        raise SystemExit(f"한 묶음이 여러 층에 걸쳤다: {len(multi)} 묶음")
    cand = [r for r in rows if r["group_id"] not in drop]
    by_stratum_group: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in cand:
        by_stratum_group[r["strata_key"]][r["group_id"]].append(r["image_id"])
    weights = {s: sum(len(v) for v in gs.values()) for s, gs in by_stratum_group.items()}
    if variant == "A2_weights_before_exclusion":
        pre: dict[str, int] = defaultdict(int)
        for r in rows:
            pre[r["strata_key"]] += 1
        weights = {s: pre[s] for s in weights}
    if total > sum(len(v) for gs in by_stratum_group.values() for v in gs.values()):
        raise SystemExit("후보 행이 T 보다 적다")
    Q = quotas(weights, total)
    picked: list[str] = []
    per = {}
    for s in sorted(by_stratum_group):
        groups = sorted(by_stratum_group[s], key=lambda g: group_key(g, seed))
        need = Q[s]["q"]
        if variant == "A4_min_one_group":
            need = max(need, 1)
        added, taken = 0, 0
        for g in groups:
            if added >= need:
                break
            ids = by_stratum_group[s][g]
            ids = (sorted(ids, key=lambda i: manifest_pos[i]) if variant == "A5_manifest_order_in_group"
                   else sorted(ids))
            picked.extend(ids)
            added += len(ids)
            taken += 1
            if variant == "A1_global_stop" and len(picked) >= total:
                break
        if variant != "A1_global_stop" and added < need:
            raise SystemExit(f"층 {s} 의 묶음이 몫 {need} 전에 떨어졌다({added})")
        per[s] = {"w": Q[s]["w"], "quota": Q[s]["q"], "floor": Q[s]["floor"],
                  "remainder": str(Q[s]["rem"]), "remainder_rank": Q[s]["rem_rank"],
                  "rows": added, "excess": added - Q[s]["q"], "groups_taken": taken,
                  "groups_available": len(groups)}
        if variant == "A1_global_stop" and len(picked) >= total:
            for s2 in sorted(by_stratum_group):
                if s2 not in per:
                    per[s2] = {"w": Q[s2]["w"], "quota": Q[s2]["q"], "rows": 0, "excess": -Q[s2]["q"],
                               "groups_taken": 0, "groups_available": len(by_stratum_group[s2])}
            break
    accum_added = 0
    if avoid_multiple:
        rest = [(s, g) for s in sorted(by_stratum_group)
                for g in sorted(by_stratum_group[s], key=lambda g: group_key(g, seed))
                if not set(by_stratum_group[s][g]) & set(picked)]
        it = iter(rest)
        while len(picked) % ACCUM == 0:
            nxt = next(it, None)
            if nxt is None:
                raise SystemExit("누적 창 배수를 피할 묶음이 없다")
            picked.extend(sorted(by_stratum_group[nxt[0]][nxt[1]]))
            accum_added += 1
    info = {"pool_rows": len(rows), "candidate_rows": len(cand),
            "candidate_groups": len({r['group_id'] for r in cand}),
            "rows_dropped_by_isolation": len(rows) - len(cand), "strata": per, "total_rows": len(picked),
            "total_excess": len(picked) - total, "accum_groups_added": accum_added,
            "steps_at_accum_32": -(-len(picked) // ACCUM)}
    return picked, info


def write_list(path: Path, ids: list[str]) -> str:
    data = "".join(f"{i}\n" for i in ids).encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-base", type=Path, required=True, help="data/ 를 담은 저장소(본체) 경로")
    ap.add_argument("--plan", type=Path, default=REPO / "configs/frame_diag_uni.yaml")
    ap.add_argument("--edges-plan", type=Path, default=REPO / "configs/rehearsal_uni.yaml")
    ap.add_argument("--base-config", type=Path, default=REPO / "configs/base.yaml")
    ap.add_argument("--pairs", type=Path, default=None, help="A3 만 쓴다 — image_id · split 만 읽는다")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    plan_raw = args.plan.read_bytes()
    plan = yaml.safe_load(plan_raw)
    if plan.get("kind") != "frame_diag" or plan["train"].get("central_rows") != "direct":
        raise SystemExit("진단 계획(kind frame_diag · central_rows direct)이 아니다")
    total = int(plan["train"]["rows"]["central"])
    seed = int(plan["sampling_seed"])
    split = plan["train"]["split"]
    avoid = bool(plan["train"]["avoid_multiple_of_accum"])
    if not plan["exclude"]["drop_components_touching_eval"]:
        raise SystemExit("진단 계획이 성분 제외를 끄고 있다 — 이 구현은 그 갈래를 짜지 않았다")
    edges_plan_raw = args.edges_plan.read_bytes()
    ned = yaml.safe_load(edges_plan_raw)["exclude"]["near_dup_edges"]
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    want_digest = base["fixed_before_main_runs"]["snapshot_digest"]

    snap_root = args.data_base / "data/interim/manifest_v1"
    df = read_manifest_columns(snap_root, VIEW_COLUMNS, split_filter={"train", "val"})
    got_digest = df.attrs.get("snapshot_digest")
    if got_digest != want_digest:
        raise SystemExit(f"스냅샷 지문이 설정과 다르다: {got_digest} != {want_digest}")
    view = df[VIEW_COLUMNS].astype(str).to_dict("records")
    splits = {r["split"] for r in view}
    if not splits <= {"train", "val"}:
        raise SystemExit(f"뷰에 학습 · 검증 밖의 분할이 있다: {splits}")
    manifest_pos = {r["image_id"]: i for i, r in enumerate(view)}
    if len(manifest_pos) != len(view):
        raise SystemExit("뷰에 같은 image_id 가 두 번 있다")
    gsplit: dict[str, set[str]] = defaultdict(set)
    for r in view:
        gsplit[r["group_id"]].add(r["split"])
    cross = sum(1 for s in gsplit.values() if len(s) > 1)
    if cross:
        raise SystemExit(f"학습 · 검증에 걸친 묶음이 있다: {cross}")

    edges, edges_sha = read_edges(args.data_base / ned["path"], ned["sha256"])
    drop, iso = excluded_groups(view, edges)

    pair_ids = None
    if args.pairs is not None:
        pair_ids = set()
        with args.pairs.open("rb") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    if d["split"] == split:
                        pair_ids.add(d["image_id"])

    args.out.mkdir(parents=True, exist_ok=False)
    meta = {"inputs": {"plan": str(args.plan.relative_to(REPO)).replace("\\", "/"),
                       "plan_sha256": sha256_bytes(plan_raw),
                       "edges_plan": str(args.edges_plan.relative_to(REPO)).replace("\\", "/"),
                       "edges_plan_sha256": sha256_bytes(edges_plan_raw),
                       "edges_path": ned["path"], "edges_sha256": edges_sha,
                       "snapshot_digest": got_digest, "view_columns": VIEW_COLUMNS,
                       "view_splits": sorted(splits),
                       "pairs_sha256": sha256_file(args.pairs) if args.pairs else None,
                       "implementation_sha256": sha256_file(Path(__file__))},
            "params": {"T": total, "sampling_seed": seed, "split": split,
                       "avoid_multiple_of_accum": avoid,
                       "group_order_key": 'sha256(f"{group_id}|{sampling_seed}") hex asc'},
            "view": {"rows": len(view), "train_rows": sum(r["split"] == "train" for r in view),
                     "val_rows": sum(r["split"] == "val" for r in view), "groups": len(gsplit)},
            "isolation": iso, "variants": {}}
    for v in VARIANTS:
        if v == "A3_pairs_pool" and pair_ids is None:
            continue
        try:
            ids, info = select(view, drop, split=split, total=total, seed=seed, avoid_multiple=avoid,
                               variant=v, pair_ids=pair_ids, manifest_pos=manifest_pos)
        except SystemExit as e:
            meta["variants"][v] = {"refused": str(e)}
            continue
        d = args.out / v
        d.mkdir()
        info["list_sha256"] = write_list(d / "train_rows.txt", ids)
        info["sorted_list_sha256"] = write_list(d / "train_rows.sorted.txt", sorted(ids))
        info["distinct_ids"] = len(set(ids))
        (d / "strata.json").write_bytes((json.dumps(info, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
        meta["variants"][v] = {k: info[k] for k in ("total_rows", "total_excess", "list_sha256",
                                                     "sorted_list_sha256", "candidate_rows",
                                                     "rows_dropped_by_isolation")}
    (args.out / "meta.json").write_bytes((json.dumps(meta, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({k: v for k, v in meta.items() if k != "inputs"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
