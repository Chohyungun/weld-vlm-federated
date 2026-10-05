#!/usr/bin/env python3
"""리허설 목록 독립 구현 B — 2판 반영판(§4-1 · §4-4 · §5-3) + **§13(3판 개정, sha256 a4cd67c8…)** 을 읽고 짰다.

표준 라이브러리만 쓴다.  python sample_b2.py <입력.json> <출력.json>
앞 판 sample_b.py(sha256 bc020784…)에서 §13-1 ① · ② 를 고쳤고, 생성 절차는 §13-2 의 문장에 맞췄다. 번호는 앞 판의 해석 번호를 잇는다.

== A. §13 이 정한 자리(앞 판 번호 → 정본) ==
 2'. [§13-1 ①] 근사 중복 간선의 끝점이 매니페스트에 없으면 **입력 오류로 거부**한다(목록을 쓰지 않는다). 앞 판의 "다리 꼭짓점" 은 폐기.
     성분 노드는 매니페스트의 모든 행(eval 포함)이다.
 6'. [§13-2 머리] 층 순서 = `strata_key` 문자열 오름차순. 층 안 묶음 순서 = `sha256(f"{group_id}|{sampling_seed}")` 16진 문자열 오름차순.
     묶음을 넣으면 그 묶음의 남은 후보 장을 모두 넣는다.
 7'. [§13-2 의 1] 최소 단계 — 층 순서대로 층마다 앞 `min_groups_per_stratum` 개 묶음(상한 검사 없이, 모자라면 있는 만큼). 장 수가 `max_n` 을 넘으면 거부.
     최소 단계만으로 T 에 닿아도 최소 단계는 끝까지 채운다.
 8'. [§13-2 의 2] 몫 — `R = max(0, T − 최소 단계 양)`, `w_s` = 층 후보 장 수(최소 단계 포함), 최대 잔여 방식(정확한 유리수), 잔여 동률은 층 순서 앞선 층.
 9'. [§13-2 의 3] 1차 채우기 — 층 순서대로, 이 단계에서 그 층에 넣은 양이 `q_s` 이상이 될 때까지 남은 묶음을 순서대로. `max_n` 을 넘기는 묶음은 건너뛰고(그 층에서 버림) 다음 묶음.
     합이 T 이상이면 그 자리에서 모두 멈춘다.
10'. [§13-2 의 4·5] 2차 채우기 — 1차 뒤 T 미달이면 몫을 보지 않고 층 순서 → 층 안 순서로 남은 묶음(건너뛰기 같음), T 이상이면 멈춤. 그래도 미달이면 거부.
11'. [§13-2 머리] T 는 리허설이면 `target_n`(장 수), 진단이면 `target_boxes`(장마다 `n_defects` 합). `max_n` 은 언제나 장 수. "닿는다" 는 `>=`.
12'. [§13-1 ②] 학습 — 참여자마다 묶음을 순서대로 넣어 행 수가 목표 이상이면 멈춘다. `avoid_multiple_of_accum` 이면 행 수가 `accum` 으로
     **나누어 떨어지지 않을 때까지** 다음 묶음을 더하고, 묶음이 다 떨어지면 거부한다(앞 판의 "하나만" 은 폐기).
13'. [§13-2 끝 문단] 중앙 = 세 참여자 행의 합집합. ② 를 다시 걸지 않는다.

== B. 여전히 명세가 정하지 않아 내가 고른 자리 ==
 1. 제외(eval 을 품은 성분)는 뽑기 전에 후보에서 뺀다. 생성·학습 양쪽에 같은 제외 묶음 집합.
 2. 같은 묶음 간선은 eval 행의 `group_id` 로도 잇는다(평가 행에서 식별자·분할·묶음은 읽는다).
 3. `excluded.n_components` 는 eval 노드와 train·val 노드를 둘 다 품은 성분만 센다. `n_groups` = 그 성분들의 train·val `group_id` 가짓수, `n_images` = 그 묶음의 train·val 행 수.
 4. 생성 후보 = 매니페스트 val ∧ 페어의 split=="val" 행 있음 ∧ (`defect_only` → `n_defects>=1`) ∧ (`max_target_tokens` 값이면 `target_tokens<=`) ∧ 제외 묶음 아님.
    학습 후보 = 매니페스트 train ∧ `client` 일치 ∧ 제외 묶음 아님 — 페어 train 행 유무는 보지 않는다.
 5. 생성의 묶음 단위는 (`strata_key`, `group_id`) 쌍. 묶음의 장 수·박스 수는 거른 뒤 남은 장으로 센다.
 6. 묶음 해시가 같으면(충돌) `group_id` 오름차순 — §13 은 동률을 정하지 않았다.
11. `target_n` · `target_boxes` 가운데 정확히 하나만 값이어야 하며 아니면 거부.
12. 학습에서 모든 묶음을 넣어도 목표 미달이면 거부(§13 은 ② 의 소진만 거부로 적었다).
14. train·val 행에 `group_id` 없음, 생성 후보에 `strata_key` 없음, 매니페스트·페어 `image_id` 중복 → 거부.
15. 거부하면 목록은 모두 비우고 `excluded` 는 계산이 끝났으면 그 값(아니면 0), `eval_intersection` 은 0.
16. `gen.split!="val"` · `train.split!="train"` 거부. 낸 전체 id ∩ eval ≠ 0 이면 거부(§5-3 증거 1).
17. ① 의 간선 검사는 `drop_components_touching_eval` 이 false 여도 건다(입력 오류는 쓰임과 무관). 간선 끝점이 자기 자신인 줄은 허용.
"""

import hashlib
import json
import sys
from collections import defaultdict
from fractions import Fraction


class Refuse(Exception):
    pass


def order_key(group_id, seed):
    digest = hashlib.sha256(f"{group_id}|{seed}".encode("utf-8")).hexdigest()
    return (digest, group_id)


class DSU:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_exclusion(manifest_rows, edges, enabled):
    by_id = {r["image_id"]: r for r in manifest_rows}
    # §13-1 ① — 매니페스트에 없는 id 가 간선에 있으면 입력 오류
    for a, b in edges:
        for x in (a, b):
            if x not in by_id:
                raise Refuse(f"edge endpoint not in manifest: {x}")
    if not enabled:
        return set(), {"n_components": 0, "n_groups": 0, "n_images": 0}
    dsu = DSU()
    groups = defaultdict(list)
    for r in manifest_rows:
        dsu.add(r["image_id"])
        g = r.get("group_id")
        if g is not None:
            groups[g].append(r["image_id"])
    for ids in groups.values():
        for other in ids[1:]:
            dsu.union(ids[0], other)
    for a, b in edges:
        dsu.union(a, b)
    comps = defaultdict(list)
    for node in list(dsu.parent):
        comps[dsu.find(node)].append(node)
    excluded_groups = set()
    n_components = 0
    for nodes in comps.values():
        has_eval = False
        tv_groups = set()
        for n in nodes:
            r = by_id[n]
            if r["split"] == "eval":
                has_eval = True
            elif r["split"] in ("train", "val"):
                if r.get("group_id") is None:
                    raise Refuse(f"train/val row without group_id: {n}")
                tv_groups.add(r["group_id"])
        if has_eval and tv_groups:
            n_components += 1
            excluded_groups |= tv_groups
    n_images = sum(
        1 for r in manifest_rows
        if r["split"] in ("train", "val") and r.get("group_id") in excluded_groups
    )
    return excluded_groups, {
        "n_components": n_components,
        "n_groups": len(excluded_groups),
        "n_images": n_images,
    }


def largest_remainder(total, weights, keys):
    """§13-2 의 2 — 정확한 유리수, 잔여 동률은 층 순서(keys 의 순서)."""
    quotas = {k: 0 for k in keys}
    wsum = sum(weights[k] for k in keys)
    if total <= 0 or wsum == 0:
        return quotas
    raw = {k: Fraction(total * weights[k], wsum) for k in keys}
    base = {k: int(raw[k]) for k in keys}  # 음이 아니므로 floor
    leftover = total - sum(base.values())
    pos = {k: i for i, k in enumerate(keys)}
    order = sorted(keys, key=lambda k: (-(raw[k] - base[k]), pos[k]))
    for k in order[:leftover]:
        base[k] += 1
    return base


def sample_gen(gen_cfg, seed, manifest_rows, pairs_by_id, excluded_groups):
    if gen_cfg.get("split") != "val":
        raise Refuse("gen.split must be 'val'")
    target_n = gen_cfg.get("target_n")
    target_boxes = gen_cfg.get("target_boxes")
    if (target_n is None) == (target_boxes is None):
        raise Refuse("exactly one of gen.target_n / gen.target_boxes must be set")
    measure_boxes = target_boxes is not None
    T = target_boxes if measure_boxes else target_n
    max_n = gen_cfg["max_n"]
    min_groups = gen_cfg.get("min_groups_per_stratum", 1)
    max_tok = gen_cfg.get("max_target_tokens")
    defect_only = bool(gen_cfg.get("defect_only"))

    strata = defaultdict(lambda: defaultdict(list))  # strata_key -> group_id -> [(image_id, n_defects)]
    for r in manifest_rows:
        if r["split"] != "val":
            continue
        p = pairs_by_id.get(r["image_id"])
        if p is None or p.get("split") != "val":
            continue
        if defect_only and p["n_defects"] < 1:
            continue
        if max_tok is not None and p["target_tokens"] > max_tok:
            continue
        if r.get("group_id") is None:
            raise Refuse(f"val row without group_id: {r['image_id']}")
        if r["group_id"] in excluded_groups:
            continue
        if r.get("strata_key") is None:
            raise Refuse(f"val candidate without strata_key: {r['image_id']}")
        strata[r["strata_key"]][r["group_id"]].append((r["image_id"], p["n_defects"]))

    def measure(imgs):
        return sum(nd for _, nd in imgs) if measure_boxes else len(imgs)

    stratum_keys = sorted(strata)
    ordered = {sk: sorted(strata[sk], key=lambda g: order_key(g, seed)) for sk in stratum_keys}
    weights = {sk: sum(len(v) for v in strata[sk].values()) for sk in stratum_keys}

    chosen = []
    taken = {sk: set() for sk in stratum_keys}
    total_m = 0
    total_img = 0

    def add(sk, g):
        nonlocal total_m, total_img
        imgs = strata[sk][g]
        chosen.extend(i for i, _ in imgs)
        taken[sk].add(g)
        total_m += measure(imgs)
        total_img += len(imgs)
        return measure(imgs)

    # 1. 최소 단계
    for sk in stratum_keys:
        for g in ordered[sk][:min_groups]:
            add(sk, g)
    if total_img > max_n:
        raise Refuse(f"minimum groups per stratum give {total_img} images > max_n {max_n}")

    # 2. 몫
    R = max(0, T - total_m)
    quotas = largest_remainder(R, weights, stratum_keys)

    # 3. 1차 채우기
    filled = {sk: 0 for sk in stratum_keys}
    stop = total_m >= T
    for sk in stratum_keys:
        if stop:
            break
        for g in ordered[sk]:
            if filled[sk] >= quotas[sk]:
                break
            if g in taken[sk]:
                continue
            if total_img + len(strata[sk][g]) > max_n:
                continue  # 그 층에서 버린다
            filled[sk] += add(sk, g)
            if total_m >= T:
                stop = True
                break

    # 4. 2차 채우기
    if total_m < T:
        for sk in stratum_keys:
            if total_m >= T:
                break
            for g in ordered[sk]:
                if g in taken[sk]:
                    continue
                if total_img + len(strata[sk][g]) > max_n:
                    continue
                add(sk, g)
                if total_m >= T:
                    break

    # 5.
    if total_m < T:
        raise Refuse(f"gen target {T} not reached (got {total_m}, {total_img} images, max_n {max_n})")
    return sorted(chosen)


def sample_train(train_cfg, accum, seed, manifest_rows, excluded_groups):
    if train_cfg.get("split") != "train":
        raise Refuse("train.split must be 'train'")
    avoid = bool(train_cfg.get("avoid_multiple_of_accum"))
    rows_target = train_cfg["rows"]
    result = {}
    for client in ("C1", "C2", "C3"):
        target = rows_target[client]
        groups = defaultdict(list)
        for r in manifest_rows:
            if r["split"] != "train" or r.get("client") != client:
                continue
            if r.get("group_id") is None:
                raise Refuse(f"train row without group_id: {r['image_id']}")
            if r["group_id"] in excluded_groups:
                continue
            groups[r["group_id"]].append(r["image_id"])
        order = sorted(groups, key=lambda g: order_key(g, seed))
        chosen = []
        idx = 0
        while len(chosen) < target and idx < len(order):
            chosen.extend(groups[order[idx]])
            idx += 1
        if len(chosen) < target:
            raise Refuse(f"train target {target} for {client} not reached (got {len(chosen)})")
        # §13-1 ② — 나누어 떨어지지 않을 때까지
        while avoid and len(chosen) % accum == 0:
            if idx >= len(order):
                raise Refuse(f"{client}: rows {len(chosen)} is a multiple of accum {accum} and no group remains")
            chosen.extend(groups[order[idx]])
            idx += 1
        result[client] = sorted(chosen)
    result["central"] = sorted(set().union(*(set(result[c]) for c in ("C1", "C2", "C3"))))
    return result


def run(inp):
    plan = inp["plan"]
    seed = plan["sampling_seed"]
    accum = inp["accum"]
    manifest_rows = inp["manifest"]
    pairs = inp["pairs"]
    edges = inp.get("edges", [])
    exclude_cfg = plan.get("exclude", {})

    out = {
        "status": "refused", "reason": None,
        "gen": [], "echo": [],
        "train": {"C1": [], "C2": [], "C3": [], "central": []},
        "excluded": {"n_components": 0, "n_groups": 0, "n_images": 0},
        "eval_intersection": 0,
    }
    try:
        seen = set()
        for r in manifest_rows:
            if r["image_id"] in seen:
                raise Refuse(f"duplicate manifest image_id: {r['image_id']}")
            seen.add(r["image_id"])
        pairs_by_id = {}
        for p in pairs:
            if p["image_id"] in pairs_by_id:
                raise Refuse(f"duplicate pairs image_id: {p['image_id']}")
            pairs_by_id[p["image_id"]] = p
        eval_ids = {r["image_id"] for r in manifest_rows if r["split"] == "eval"}

        excluded_groups, excluded = build_exclusion(
            manifest_rows, edges, bool(exclude_cfg.get("drop_components_touching_eval", False)))
        out["excluded"] = excluded

        gen = sample_gen(plan["gen"], seed, manifest_rows, pairs_by_id, excluded_groups)
        train = sample_train(plan["train"], accum, seed, manifest_rows, excluded_groups)

        emitted = set(gen)
        for v in train.values():
            emitted |= set(v)
        inter = len(emitted & eval_ids)
        if inter:
            raise Refuse(f"eval intersection {inter} != 0")

        out.update({"status": "ok", "gen": gen, "echo": list(gen), "train": train, "eval_intersection": inter})
    except Refuse as e:
        out["reason"] = str(e)
    return out


def main(argv):
    if len(argv) != 3:
        print("usage: python sample_b2.py <input.json> <output.json>", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        inp = json.load(f)
    out = run(inp)
    with open(argv[2], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
