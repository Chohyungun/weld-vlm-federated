"""리허설 계획 파일과 목록 뽑기 — 리허설 2판 §4-4(계획의 키 · 뽑는 법) · §5-3(제외) · §6-1 의 14 · 36 · 40.

두 구현을 맞댄다. `vlm.rehearsal_lists`(A)와 명세 문장만 읽고 따로 짠 `tests/rehearsal_lists_b.py`(B, sha256 아래 상수)다.
정본은 리허설 2판 §13(3판 개정 — 총괄 결정 04 의 2절)이다. 결정 전에 갈린 아홉 사례를 `xfail` 로 남겼던 표시는 개정에서 걷었다.
"""

from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm.rehearsal_lists import (
    ListsRefused,
    build_lists,
    exclusion,
    read_edges,
    sample_gen,
    sample_train,
)
from vlm.rehearsal_plan import PLAN_KEYS, PlanRejected, load_plan, parse_plan

B_PATH = ROOT / "tests" / "rehearsal_lists_b.py"
B_SHA256 = "a98e7b63b31ebe38a67f538e706ca24ab046e9c464de5b8d7f4f918007eb06c1"


# ================================================================ 계획 파일
PLANS = [("rehearsal", "rehearsal_uni.yaml"), ("frame_diag", "frame_diag_uni.yaml")]


@pytest.mark.parametrize(("purpose", "name"), PLANS)
def test_저장소의_계획_파일을_읽는다(purpose, name):
    """저장소 계획의 **지금 값**에 기대지 않는다 — 다른 쪽이 빈 칸을 채워도 깨지지 않게, 읽기 · 지문 · 학습량 · 간선 칸의 꼴만 본다."""
    p = load_plan(ROOT / "configs" / name, purpose=purpose)
    assert p.sha256 == hashlib.sha256((ROOT / "configs" / name).read_bytes()).hexdigest()
    n, r, e = p.amount
    assert n == r * e
    edges = p.get("exclude.near_dup_edges")
    assert edges is None or set(edges) == {"path", "sha256"}                   # 비었거나 정해진 꼴(읽는 쪽이 꼴을 이미 본다)


@pytest.mark.parametrize(("purpose", "name"), PLANS)
def test_빈_간선_칸은_필요한_단계에서_거부한다(tmp_path, purpose, name):
    """저장소 계획에서 간선 칸만 **명시적으로 비운** 합성 계획 — 그 값이 필요한 단계는 시작하지 않는다."""
    import yaml

    doc = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    doc["exclude"]["near_dup_edges"] = None
    syn = tmp_path / name
    syn.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    p = load_plan(syn, purpose=purpose)
    with pytest.raises(PlanRejected) as exc:
        p.require("exclude.near_dup_edges")
    assert exc.value.code == "plan_value_missing"


def test_계획의_키_표가_계획_파일_시험의_표와_같다():
    """두 표 모두 명세 §4-4 에서 뽑았다 — 한쪽만 바뀌면 여기서 드러난다."""
    from tests.test_plan_configs import PLAN_TABLE

    assert sorted(PLAN_KEYS) == sorted(PLAN_TABLE)


BASE = (ROOT / "configs" / "rehearsal_uni.yaml").read_text(encoding="utf-8")


@pytest.mark.parametrize(("edit", "code"), [
    (lambda t: t.replace("kind: rehearsal", "kind: frame_diag"), "plan_kind"),
    (lambda t: t.replace("version: 1", "version: 2"), "plan_version"),
    (lambda t: t + "\nmystery: 1\n", "plan_unknown_key"),
    (lambda t: t + "\nmodel: {id: x}\n", "plan_forbidden_key"),
    (lambda t: t.replace("gen:\n", "gen:\n  decoding: {do_sample: true}\n"), "plan_forbidden_key"),
    (lambda t: t + "\nattempts_allowed: 1\n", "plan_key_not_for_purpose"),
    (lambda t: t.replace("  n: 2\n", "  n: 3\n"), "plan_amount"),
    (lambda t: t.replace("split: val", "split: eval"), "plan_value"),
    (lambda t: t.replace("'export:after_lines:uni_central:n=12'", "'export:nowhere:uni_central:n=12'"), "plan_value"),
    (lambda t: t.replace("negatives: [policy_blank, ", "negatives: ["), "plan_value"),
])
def test_계획_파일의_규칙을_어기면_사유_코드로_거부한다(edit, code):
    with pytest.raises(PlanRejected) as exc:
        parse_plan(edit(BASE).encode("utf-8"), purpose="rehearsal")
    assert exc.value.code == code, str(exc.value)


def test_진단_계획에_중단_계획이_있으면_거부한다():
    t = (ROOT / "configs" / "frame_diag_uni.yaml").read_text(encoding="utf-8")
    with pytest.raises(PlanRejected) as exc:
        parse_plan((t + "\nfaults: {train: {uni_central: 'train:after_ckpt:uni_central:ep=0'}}\n").encode(),
                   purpose="frame_diag")
    assert exc.value.code == "plan_key_not_for_purpose"


# ================================================================ 제외 · 뽑기
def _m(i, split, g, s=None, c=None):
    return {"image_id": i, "split": split, "group_id": g, "strata_key": s, "client": c}


def test_묶음의_한_장만_평가와_이어져도_그_묶음_전체가_빠진다():
    man = [_m("v1", "val", "g1", "S|a"), _m("v2", "val", "g1", "S|a"), _m("v3", "val", "g2", "S|a"),
           _m("t1", "train", "g3", "S|a", "C1"), _m("e1", "eval", "ge")]
    drop, rec = exclusion(man, [("v1", "e1")])
    assert drop == {"v1", "v2"} and rec == {"n_components": 1, "n_groups": 1, "n_images": 2}
    drop, rec = exclusion(man, [("v1", "e1"), ("v3", "t1")])       # 평가와 닿지 않는 간선은 아무것도 빼지 않는다
    assert drop == {"v1", "v2"}
    drop, rec = exclusion(man, [("v1", "t1"), ("t1", "e1")])       # 사슬로 이어져도 성분이다
    assert drop == {"v1", "v2", "t1"} and rec["n_components"] == 1 and rec["n_groups"] == 2


def test_매니페스트에_없는_id_의_간선은_거부한다():
    with pytest.raises(ListsRefused) as exc:
        exclusion([_m("v1", "val", "g1", "S|a")], [("v1", "zz")])
    assert exc.value.code == "lists_edge_unknown_id"


def test_간선_파일의_꼴():
    assert read_edges(b"a_id,b_id\nx,y\n") == [("x", "y")]
    for bad in (b"a,b\nx,y\n", b"a_id,b_id\nx\n", b""):
        with pytest.raises(ListsRefused):
            read_edges(bad)


def _cands(spec):
    return [{"image_id": f"{s}{g}{j}", "group_id": f"{s}{g}", "strata_key": s, "n_defects": 1}
            for s, groups in spec.items() for g, n in enumerate(groups) for j in range(n)]


def test_층마다_최소_묶음을_먼저_넣고_목표에_닿으면_멈춘다():
    c = _cands({"A": [2, 2, 2], "B": [1, 1]})
    got = sample_gen(c, seed=1, target=4, by_boxes=False, max_n=8, min_groups=1)
    assert len(got) >= 4 and {i[0] for i in got} == {"A", "B"}
    assert sample_gen(c, seed=1, target=4, by_boxes=False, max_n=8, min_groups=1) == got      # 결정론


def test_상한을_넘기는_묶음은_건너뛰고_못_미치면_거부한다():
    c = _cands({"A": [5, 1, 1]})
    got = sample_gen(c, seed=3, target=2, by_boxes=False, max_n=3, min_groups=0)
    assert len(got) <= 3
    with pytest.raises(ListsRefused) as exc:
        sample_gen(_cands({"A": [1]}), seed=1, target=5, by_boxes=False, max_n=8, min_groups=1)
    assert exc.value.code == "lists_target_not_reached"
    with pytest.raises(ListsRefused) as exc:
        sample_gen(_cands({"A": [4], "B": [4]}), seed=1, target=2, by_boxes=False, max_n=6, min_groups=1)
    assert exc.value.code == "lists_min_groups_exceed_max"


def test_누적_창의_배수면_배수가_아닐_때까지_묶음을_더_넣는다():
    """묶음을 실제 순서대로 2 · 2 · 2 · 1 장으로 둔다 — 목표 4 에 닿으면 4(배수), 하나 더 넣어도 6(배수), 그다음이 7."""
    from vlm.rehearsal_lists import order_key

    order = sorted((f"g{k}" for k in range(4)), key=lambda g: order_key(g, 1))
    rows, group_of = [], {}
    for pos, g in enumerate(order):
        for j in range(2 if pos < 3 else 1):
            rows.append({"image_id": f"{g}_{j}"})
            group_of[f"{g}_{j}"] = g
    got = sample_train(rows, group_of=group_of, seed=1, target=4, avoid_multiple=True, accum=2)
    assert len(got) == 7
    assert len(sample_train(rows, group_of=group_of, seed=1, target=4, avoid_multiple=False, accum=2)) == 4


def test_누적_창의_배수를_피할_묶음이_없으면_거부한다():
    rows = [{"image_id": f"t{g}{j}"} for g in range(3) for j in range(2)]
    group_of = {r["image_id"]: f"g{r['image_id'][1]}" for r in rows}
    with pytest.raises(ListsRefused) as exc:
        sample_train(rows, group_of=group_of, seed=1, target=2, avoid_multiple=True, accum=2)
    assert exc.value.code == "lists_train_multiple"
    got = sample_train(rows + [{"image_id": "u0"}], group_of={**group_of, "u0": "gu"}, seed=1, target=2,
                       avoid_multiple=True, accum=2)
    assert len(got) % 2 == 1


# ================================================================ 두 구현
def _b():
    raw = B_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == B_SHA256, "독립 구현의 바이트가 고정 값과 다르다"
    spec = importlib.util.spec_from_file_location("rehearsal_lists_b", B_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _P:
    def __init__(self, d):
        self.d = d

    def get(self, dotted, default=None):
        x = self.d
        for k in dotted.split("."):
            if not isinstance(x, dict) or k not in x:
                return default
            x = x[k]
        return x


def run_a(inp):
    pairs = [{"image_id": r["image_id"], "split": r["split"], "client": r["client"],
              "skeleton": {"defects": [{}] * int(r["n_defects"])}} for r in inp["pairs"]]
    try:
        res = build_lists(inp["manifest"], pairs, [tuple(e) for e in inp["edges"]], _P(inp["plan"]),
                          target_tokens={r["image_id"]: int(r["target_tokens"]) for r in inp["pairs"]},
                          accum=int(inp["accum"]))
    except ListsRefused as exc:
        return {"status": "refused", "reason": exc.code}
    tr = {c: sorted(r["image_id"] for r in res.train_rows[t]) for c, t in
          (("C1", "uni_local_C1"), ("C2", "uni_local_C2"), ("C3", "uni_local_C3"), ("central", "uni_central"))}
    return {"status": "ok", "gen": res.gen_ids, "echo": res.echo_ids, "train": tr,
            "excluded": res.record["excluded"], "eval_intersection": res.record["eval_intersection"]}


def world(seed, *, boxes=False, unknown_edge=False):
    rnd = random.Random(seed)
    man, pairs, edges = [], [], []
    k = 0
    for g in range(24):
        s = f"{'ST' if g % 3 else 'AL'}|c{g % 4}"
        for _ in range(rnd.randint(1, 4)):
            k += 1
            man.append({"image_id": f"v{k:04d}", "split": "val", "group_id": f"gv{g}", "strata_key": s, "client": "C1"})
            pairs.append({"image_id": f"v{k:04d}", "split": "val", "client": "C1", "n_defects": rnd.choice([0, 1, 1, 2, 3]),
                          "target_tokens": rnd.randint(100, 700)})
    for g in range(40):
        c = ("C1", "C1", "C2", "C3")[g % 4]
        for _ in range(rnd.randint(1, 6)):
            k += 1
            man.append({"image_id": f"t{k:04d}", "split": "train", "group_id": f"gt{g}", "strata_key": "ST|x", "client": c})
            pairs.append({"image_id": f"t{k:04d}", "split": "train", "client": c, "n_defects": 1, "target_tokens": 200})
    for g in range(6):
        for _ in range(2):
            k += 1
            man.append({"image_id": f"e{k:04d}", "split": "eval", "group_id": f"ge{g}", "strata_key": None, "client": None})
    ids = [m["image_id"] for m in man]
    evals = [m["image_id"] for m in man if m["split"] == "eval"]
    edges = [[rnd.choice(ids), rnd.choice(evals)] for _ in range(8)] + [[rnd.choice(ids), rnd.choice(ids)] for _ in range(8)]
    if unknown_edge:
        edges.append(["zz_unknown", evals[0]])
    plan = {"sampling_seed": seed, "gen": {"split": "val", "defect_only": boxes, "target_n": None if boxes else 20,
                                           "target_boxes": 30 if boxes else None, "max_n": 40,
                                           "min_groups_per_stratum": 1, "max_target_tokens": 512 if boxes else None},
            "train": {"split": "train", "rows": {"C1": 30, "C2": 12, "C3": 8}, "avoid_multiple_of_accum": True,
                      "central_rows": "union"}, "exclude": {"drop_components_touching_eval": True}}
    return {"plan": plan, "accum": 4, "manifest": man, "pairs": pairs, "edges": edges}


CASES = [(f"reh{s}", s, False, False) for s in range(1, 9)] + [(f"diag{s}", 100 + s, True, False) for s in range(1, 5)] \
    + [("unknown_edge", 7, False, True)]


def _counterfactual_gen(inp) -> list[str] | None:
    """빈 층이 있으면 2판 §14-2 의 생성 목록을 **독립 구현 B 의 함수**로 따로 낸다 — B 의 제외 결과에서 빈 층의 묶음만 되돌린 세계로 B 의 생성 뽑기
    (`sample_gen`)를 목표 T 그대로 돌리고 빈 층의 장만 걷는다. 빈 층이 없으면 `None`(B 의 `run` 을 그대로 쓴다)."""
    B = _b()
    excluded_groups, _ex = B.build_exclusion(inp["manifest"], [tuple(e) for e in inp["edges"]], True)
    gen = inp["plan"]["gen"]
    pair_of = {r["image_id"]: r for r in inp["pairs"]}
    stratum_groups: dict = {}
    after = set()
    for m in inp["manifest"]:
        r = pair_of.get(m["image_id"])
        if m["split"] != "val" or r is None or r["split"] != "val" or (gen["defect_only"] and int(r["n_defects"]) < 1):
            continue
        if gen["max_target_tokens"] is not None and int(r["target_tokens"]) > int(gen["max_target_tokens"]):
            continue
        stratum_groups.setdefault(m["strata_key"], set()).add(m["group_id"])
        if m["group_id"] not in excluded_groups:
            after.add(m["strata_key"])
    empty = {s for s in stratum_groups if s not in after}
    if not empty:
        return None
    restored = set().union(*(stratum_groups[s] for s in empty))
    chosen = B.sample_gen(gen, inp["plan"]["sampling_seed"], inp["manifest"], pair_of, excluded_groups - restored)
    stratum_of = {m["image_id"]: m["strata_key"] for m in inp["manifest"]}
    return [i for i in chosen if stratum_of[i] not in empty]


@pytest.mark.parametrize("name,seed,boxes,unknown", CASES, ids=[c[0] for c in CASES])
def test_두_구현이_같은_입력에서_같은_목록을_낸다(name, seed, boxes, unknown):
    inp = world(seed, boxes=boxes, unknown_edge=unknown)
    a = run_a(inp)
    b = _b().run(inp)
    cf = None if unknown else _counterfactual_gen(inp)
    assert a["status"] == b["status"], (a.get("reason"), b.get("reason"))
    if a["status"] == "ok":
        want = b["gen"] if cf is None else cf                    # 빈 층이 있는 세계 — B 는 개정 전 판이라 되돌린 세계의 B 뽑기와 맞댄다
        assert a["gen"] == want and a["echo"] == want
        assert a["train"] == b["train"]
        assert a["excluded"] == b["excluded"] and a["eval_intersection"] == b["eval_intersection"] == 0


def test_최대_잔여_방식의_몫과_동률():
    """2판 §13-2 의 2 — 정수부를 먼저 주고 남은 것을 잔여가 큰 층부터. 잔여가 같으면 층 키 오름차순."""
    from vlm.rehearsal_lists import largest_remainder

    assert largest_remainder(4, {"a": 6, "b": 4}) == {"a": 2, "b": 2}           # 2.4 · 1.6 → 잔여 b 가 크다
    assert largest_remainder(1, {"b": 1, "a": 1}) == {"a": 1, "b": 0}           # 잔여 0.5 동률 → 층 키가 앞선 a
    assert largest_remainder(3, {"x": 1, "y": 1, "z": 1}) == {"x": 1, "y": 1, "z": 1}
    assert largest_remainder(0, {"a": 3}) == {"a": 0} and largest_remainder(5, {}) == {}


# ================================================================ 빈 층 — 총괄 결정 21 의 3 · 4 (2판 §14)
def _empty_world(*, drop_isolated_groups: bool = False, train_isolated: bool = False):
    """검증 층 셋 — `AL|crack` 의 묶음은 모두 평가와 이어져 제외로 빠진다(제외 전에는 후보가 있었다). `ST|__normal__` 은 결함이 없다.
    `drop_isolated_groups` 이면 그 묶음을 매니페스트 · 페어 · 간선에서 아예 지운 세계다(빈 층이 처음부터 없는 세계)."""
    man, pairs, edges = [], [], []
    spec = [("AL|crack", "ga", 2, 1), ("AL|crack", "gb", 1, 2), ("ST|porosity", "gc", 3, 1), ("ST|porosity", "gd", 2, 1),
            ("ST|porosity", "ge", 2, 2), ("ST|__normal__", "gf", 3, 0), ("ST|__normal__", "gg", 2, 0)]
    for s, g, n, d in spec:
        if drop_isolated_groups and s == "AL|crack":
            continue
        for j in range(n):
            i = f"v_{g}{j}"
            man.append(_m(i, "val", g, s, "C1"))
            pairs.append({"image_id": i, "split": "val", "client": "C1", "skeleton": {"defects": [{}] * d}})
    for c in ("C1", "C2", "C3"):
        for g in range(4):
            i = f"t{c}{g}"
            man.append(_m(i, "train", f"gt{c}{g}", "ST|porosity", c))
            pairs.append({"image_id": i, "split": "train", "client": c, "skeleton": {"defects": [{}]}})
    if train_isolated:                                           # 학습 풀의 한 층(AL|crack)이 통째로 평가와 이어진다
        man.append(_m("tx0", "train", "gtx", "AL|crack", "C1"))
        pairs.append({"image_id": "tx0", "split": "train", "client": "C1", "skeleton": {"defects": [{}]}})
    man += [_m("e1", "eval", "gev1"), _m("e2", "eval", "gev2")]
    if not drop_isolated_groups:
        edges = [("v_ga0", "e1"), ("v_gb0", "e2")]                   # AL|crack 의 두 묶음이 다 평가와 이어진다
    if train_isolated:
        edges = edges + [("tx0", "e1")]
    return man, pairs, edges


class _Plan(dict):
    def __init__(self, d, purpose=None):
        super().__init__(d)
        self.purpose = purpose

    def get(self, dotted, default=None):
        x = self
        for k in dotted.split("."):
            if not isinstance(x, dict) or k not in x:
                return default
            x = x[k]
        return x


def _reh_plan():
    return _Plan({"sampling_seed": 5, "gen": {"split": "val", "defect_only": False, "target_n": 6, "max_n": 20,
                                              "min_groups_per_stratum": 1, "max_target_tokens": None},
                  "train": {"split": "train", "rows": {"C1": 2, "C2": 2, "C3": 2}, "avoid_multiple_of_accum": False,
                            "central_rows": "union"}, "exclude": {"drop_components_touching_eval": True}}, "rehearsal")


def _diag_plan():
    return _Plan({"sampling_seed": 5, "gen": {"split": "val", "defect_only": True, "target_boxes": 3, "max_n": 20,
                                              "min_groups_per_stratum": 1, "max_target_tokens": None},
                  "train": {"split": "train", "rows": {"central": 4}, "avoid_multiple_of_accum": False,
                            "central_rows": "direct"}, "exclude": {"drop_components_touching_eval": True}}, "frame_diag")


def test_리허설은_빈_층의_몫만_비우고_다른_층의_몫은_빈_층이_있기_전_그대로다():
    """결정 21 덧붙임의 1 · 외부 검토 회신 emptystrata2 의 1 — `AL|crack` 은 제외(R0) 뒤 0 묶음이다. 층별 몫은 그 층이 비지 않았을 세계의 배분
    그대로이고 빈 층의 몫만 0 이다 — 다른 층의 행이 **그 세계와 같다**(늘지도 줄지도 않는다). T 미달을 기록한다."""
    from collections import Counter

    man, pairs, edges = _empty_world()
    plan = _reh_plan()
    res = build_lists(man, pairs, edges, plan)
    strata = {m["image_id"]: m["strata_key"] for m in man}
    alive = build_lists(man, pairs, [], plan)                           # 이 세계는 AL|crack 만 제외로 빠진다 — 제외가 없는 세계가 곧 비지 않았을 세계다
    got, ref = Counter(strata[i] for i in res.gen_ids), Counter(strata[i] for i in alive.gen_ids)
    assert got["AL|crack"] == 0 < ref["AL|crack"]
    assert {s: n for s, n in got.items()} == {s: n for s, n in ref.items() if s != "AL|crack"}      # 다른 층의 몫은 그대로
    assert [i for i in alive.gen_ids if strata[i] != "AL|crack"] == res.gen_ids                     # 같은 장이다
    assert res.record["unverified_strata"] == [{"stratum": "AL|crack", "share": ref["AL|crack"],
                                                "reason": "제외(R0) 뒤 후보 묶음 0 — 다른 곳에서도 남은 층으로도 채우지 않는다"}]
    assert res.record["gen_target"] == {"target": 6, "empty_share": ref["AL|crack"], "filled": len(res.gen_ids), "unit": "images"}
    assert len(res.gen_ids) == len(alive.gen_ids) - ref["AL|crack"]   # 빈 몫만큼 덜 낸다(묶음이 넘쳐 T 에 닿을 수도 있다 — 미달은 아래 반례가 본다)
    assert alive.record["gen_target"] == {"target": 6, "empty_share": 0, "filled": len(alive.gen_ids), "unit": "images"}


def test_빈_층이_생겨도_다른_층의_장이_늘지_않는다_회신의_반례():
    """회신의 반례 그대로 — 층 A · B 의 후보가 3 · 7 장(묶음마다 1 장), T 4, 층마다 최소 1 묶음. 제외 전 선택은 A 2 · B 2 다.
    A 가 통째로 제외되면 앞 판은 B 를 3 장으로 늘렸다 — 이 판은 B 2 장 그대로이고 목록은 2 장(T 미달), 빈 몫은 2 다."""
    from collections import Counter

    man, pairs = [], []
    for s, n in (("A|x", 3), ("B|x", 7)):
        for g in range(n):
            i = f"v_{s[0]}{g}"
            man.append(_m(i, "val", f"g{s[0]}{g}", s, "C1"))
            pairs.append({"image_id": i, "split": "val", "client": "C1", "skeleton": {"defects": [{}]}})
    for c in ("C1", "C2", "C3"):
        man.append(_m(f"t{c}", "train", f"gt{c}", "B|x", c))
        pairs.append({"image_id": f"t{c}", "split": "train", "client": c, "skeleton": {"defects": [{}]}})
    man += [_m("e1", "eval", "gev1")]
    plan = _Plan({"sampling_seed": 9, "gen": {"split": "val", "defect_only": False, "target_n": 4, "max_n": 20,
                                              "min_groups_per_stratum": 1, "max_target_tokens": None},
                  "train": {"split": "train", "rows": {"C1": 1, "C2": 1, "C3": 1}, "avoid_multiple_of_accum": False,
                            "central_rows": "union"}, "exclude": {"drop_components_touching_eval": True}}, "rehearsal")
    strata = {m["image_id"]: m["strata_key"] for m in man}
    alive = build_lists(man, pairs, [], plan)
    assert Counter(strata[i] for i in alive.gen_ids) == {"A|x": 2, "B|x": 2}
    res = build_lists(man, pairs, [(f"v_A{g}", "e1") for g in range(3)], plan)
    assert Counter(strata[i] for i in res.gen_ids) == {"B|x": 2}
    assert res.gen_ids == [i for i in alive.gen_ids if strata[i] == "B|x"]
    assert res.record["gen_target"] == {"target": 4, "empty_share": 2, "filled": 2, "unit": "images"}
    assert res.record["unverified_strata"][0]["share"] == 2
    man0, pairs0, edges0 = _empty_world(drop_isolated_groups=True)     # 처음부터 그 층이 없던 세계는 미검증이 아니다 — T 를 다 채운다
    res0 = build_lists(man0, pairs0, edges0, plan)
    assert res0.record["unverified_strata"] == [] and res0.record["gen_target"]["empty_share"] == 0


def _diag_variant(which):
    plan = _diag_plan()
    if which == "purpose_only":
        plan["train"] = {"split": "train", "rows": {"C1": 2, "C2": 2, "C3": 2}, "avoid_multiple_of_accum": False,
                         "central_rows": "union"}
    elif which == "direct_only":
        plan.purpose = None
    return plan


@pytest.mark.parametrize("which", ["both", "purpose_only", "direct_only"])
def test_진단_목적은_빈_층과_무관하게_목록_단계에서_보류로_거부한다(which):
    """결정 21 덧붙임의 2 — 빈 층이 없는 세계에서도 진단 목적의 계획은 `lists_diag_on_hold` 하나로 거부한다(진단의 두 표지 어느 하나만으로도)."""
    man, pairs, edges = _empty_world(drop_isolated_groups=True)
    with pytest.raises(ListsRefused) as exc:
        build_lists(man, pairs, edges, _diag_variant(which))
    assert exc.value.code == "lists_diag_on_hold"
    assert build_lists(man, pairs, edges, _reh_plan()).record["unverified_strata"] == []   # 리허설은 보류가 아니다


@pytest.mark.parametrize("which", ["both", "purpose_only", "direct_only"])
def test_진단은_빈_층을_미검증으로_넘기지_않고_멈춘다(which, monkeypatch):
    """보류 아래의 거부(진단 재개 조건의 자리) — 보류를 푼 세계에서 진단은 리허설 목록의 규칙을 물려받지 않는다. 두 표지 어느 하나만 있어도 멈춘다."""
    import vlm.rehearsal_lists as RL

    monkeypatch.setattr(RL, "DIAG_LISTS_ON_HOLD", False)
    man, pairs, edges = _empty_world()
    with pytest.raises(ListsRefused) as exc:
        build_lists(man, pairs, edges, _diag_variant(which))
    assert exc.value.code == "lists_stratum_empty" and "AL|crack" in str(exc.value)


def test_진단은_학습_풀의_빈_층도_거부한다(monkeypatch):
    """결정 21 덧붙임의 3 — 학습 풀(중앙 직접 뽑기의 후보)의 한 층이 통째로 제외로 빠지면 진단은 멈춘다(재개 조건의 자리 — 층별 지지량은
    평가 쪽 계약 §38). 보류를 푼 세계에서 본다. 생성 목록에 빈 층이 없어도 멈춘다. 리허설은 이 갈래가 아니다."""
    import vlm.rehearsal_lists as RL

    monkeypatch.setattr(RL, "DIAG_LISTS_ON_HOLD", False)
    man, pairs, edges = _empty_world(drop_isolated_groups=True, train_isolated=True)
    with pytest.raises(ListsRefused) as exc:
        build_lists(man, pairs, edges, _diag_plan())
    assert exc.value.code == "lists_stratum_empty" and "학습 풀" in str(exc.value)
    assert build_lists(man, pairs, edges, _reh_plan()).record["unverified_strata"] == []


def test_내용_거름으로_빈_층은_빈_층이_아니다(monkeypatch):
    """반례 — 진단의 결함 한정으로 정상 층(`ST|__normal__`)은 늘 후보가 없다. 그것은 제외로 빈 층이 아니므로 진단을 멈추지 않는다(보류를 푼 세계)."""
    import vlm.rehearsal_lists as RL

    monkeypatch.setattr(RL, "DIAG_LISTS_ON_HOLD", False)
    man, pairs, edges = _empty_world(drop_isolated_groups=True)
    res = build_lists(man, pairs, edges, _diag_plan())
    assert res.record["unverified_strata"] == []
    assert {m["strata_key"] for m in man if m["image_id"] in res.gen_ids} == {"ST|porosity"}


@pytest.mark.parametrize("name,seed,boxes,unknown", [c for c in CASES if not c[3]], ids=[c[0] for c in CASES if not c[3]])
def test_빈_층의_기록이_독립_구현의_제외로_낸_값과_같다(name, seed, boxes, unknown):
    """빈 층의 기록은 이 구현 하나가 낸다 — 독립 구현 B 에는 이 규칙이 없다(고정 바이트, 고치지 않는다). 그래서 **B 의 제외 결과**(뺀 묶음)로
    읽기 11 의 문장대로 따로 낸 빈 층과 맞댄다(리허설 꼴의 계획 — 진단의 멈춤은 위의 시험이 본다)."""
    inp = world(seed, boxes=boxes, unknown_edge=unknown)
    excluded_groups, _ex = _b().build_exclusion(inp["manifest"], [tuple(e) for e in inp["edges"]], True)
    gen = inp["plan"]["gen"]
    pair_of = {r["image_id"]: r for r in inp["pairs"] if r["split"] == "val"}
    before, after = set(), set()
    for m in inp["manifest"]:
        r = pair_of.get(m["image_id"])
        if m["split"] != "val" or r is None or (gen["defect_only"] and int(r["n_defects"]) < 1):
            continue
        if gen["max_target_tokens"] is not None and int(r["target_tokens"]) > int(gen["max_target_tokens"]):
            continue
        before.add(m["strata_key"])
        if m["group_id"] not in excluded_groups:
            after.add(m["strata_key"])
    want = sorted(before - after)
    a = run_a(inp)
    if a["status"] != "ok":
        pytest.skip(f"이 세계는 목록 단계가 다른 까닭으로 멈춘다: {a['reason']}")
    pairs = [{"image_id": r["image_id"], "split": r["split"], "client": r["client"],
              "skeleton": {"defects": [{}] * int(r["n_defects"])}} for r in inp["pairs"]]
    tok = {r["image_id"]: int(r["target_tokens"]) for r in inp["pairs"]}
    res = build_lists(inp["manifest"], pairs, [tuple(e) for e in inp["edges"]], _P(inp["plan"]), target_tokens=tok,
                      accum=int(inp["accum"]))
    assert [u["stratum"] for u in res.record["unverified_strata"]] == want
