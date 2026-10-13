"""착수 전 프레임 진단의 학습 표본 — 중앙 직접 뽑기(총괄 결정 20 — 층화) · 진단 계획의 칸 · 목록 단계의 타깃 길이.

뽑는 법의 정본은 총괄 결정 20(`docs/dev_log/2026-09-30-리허설/20_총괄결정_진단표본.md`)과 그 덧붙임(시드 흐름의 식)이고, 이 구현의 읽기는
`vlm/rehearsal_lists.py` 머리말의 10 이다. 두 구현의 맞대기(§13-3)는 메타데이터 검토 트랙의 독립 구현(`tests/independent/fdiag_lists_g.py`,
그 트랙의 커밋 바이트 그대로)을 부른다 — 결정 20 덧붙임대로 층 안 순서 키 한 줄만 정본 식으로 맞춘다.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm.rehearsal_lists import (
    CENTRAL_TAG,
    DIRECT_STREAM,
    ListsRefused,
    build_lists,
    direct_order_key,
    largest_remainder,
    order_key,
    sample_central_direct,
)
from vlm.rehearsal_plan import PlanRejected, parse_plan


@pytest.fixture(autouse=True)
def _diag_unheld(monkeypatch):
    """진단 목록은 보류다(결정 21 의 4 · `DIAG_LISTS_ON_HOLD`) — 이 파일은 보류가 풀린 뒤의 뽑기를 본다. 보류 자체는 `test_rehearsal_lists` 가 본다."""
    import vlm.rehearsal_lists as RL

    monkeypatch.setattr(RL, "DIAG_LISTS_ON_HOLD", False)


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


def _pool(spec):
    """`{층: [묶음의 행 수, ...]}` → 후보 행 · 묶음 · 층 표."""
    rows, group_of, stratum_of = [], {}, {}
    k = 0
    for s, sizes in spec.items():
        for gi, n in enumerate(sizes):
            g = f"g_{s}_{gi}"
            for _ in range(n):
                k += 1
                i = f"t{k:04d}"
                rows.append({"image_id": i})
                group_of[i], stratum_of[i] = g, s
    return rows, group_of, stratum_of


# ================================================================ 뽑는 법(결정 20)
def test_순서는_진단_시드의_다른_흐름이다():
    """`sha256(f"{group_id}|{seed}|central_direct")` — 생성 · 참여자 목록의 순서(`order_key`)와 다른 흐름이다."""
    assert DIRECT_STREAM == "central_direct"
    assert direct_order_key("g1", 7) == (hashlib.sha256(b"g1|7|central_direct").hexdigest(), "g1")
    groups = [f"g{i}" for i in range(30)]
    assert sorted(groups, key=lambda g: direct_order_key(g, 7)) != sorted(groups, key=lambda g: order_key(g, 7))


def test_층_크기에_비례한_몫을_최대_나머지로_나누고_층마다_묶음을_순서대로_넣는다():
    spec = {"AL|c1": [2, 3, 1], "ST|c1": [4, 4, 2, 3, 1], "ST|c2": [1, 1, 1]}
    rows, group_of, stratum_of = _pool(spec)
    seed, T = 11, 12
    ids, rec = sample_central_direct(rows, group_of=group_of, stratum_of=stratum_of, seed=seed, target=T)
    all_ids = [r["image_id"] for r in rows]
    size = Counter(group_of[i] for i in all_ids)
    weight = {s: sum(v) for s, v in spec.items()}
    quota = largest_remainder(T, weight)
    assert {s: v["quota"] for s, v in rec["by_stratum"].items()} == quota
    for s in spec:                                               # 층 안에서는 순서대로, 몫에 닿으면 멈춘다(마지막 묶음까지 넣는다)
        order = sorted({group_of[i] for i in all_ids if stratum_of[i] == s}, key=lambda g: direct_order_key(g, seed))
        want: list[str] = []
        for g in order:
            if sum(size[x] for x in want) >= quota[s]:
                break
            want.append(g)
        assert sorted({group_of[i] for i in ids if stratum_of[i] == s}) == sorted(want), s
        n = sum(1 for i in ids if stratum_of[i] == s)
        assert rec["by_stratum"][s]["taken"] == n and rec["by_stratum"][s]["over"] == n - quota[s] >= 0
    assert rec["n_rows"] == len(ids) and rec["over"] == len(ids) - T >= 0
    by_group = Counter(group_of[i] for i in ids)                 # 묶음은 통째로 들어간다
    assert all(by_group[g] == size[g] for g in by_group)


def test_몫을_넘은_묶음은_다음_층의_몫에서_빼지_않는다():
    """한 층에 큰 묶음 하나뿐이면 그 묶음까지 넣는다 — 넘은 수는 기록에 싣고 다음 층은 제 몫을 다 채운다."""
    rows, group_of, stratum_of = _pool({"A": [10], "B": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
    ids, rec = sample_central_direct(rows, group_of=group_of, stratum_of=stratum_of, seed=3, target=4)
    assert rec["by_stratum"]["A"] == {"rows": 10, "quota": 2, "taken": 10, "over": 8}
    assert rec["by_stratum"]["B"] == {"rows": 10, "quota": 2, "taken": 2, "over": 0}
    assert len(ids) == 12 and rec["over"] == 8


def test_몫이_0_인_층은_넣지_않는다():
    rows, group_of, stratum_of = _pool({"A": [1] * 9, "B": [1]})
    ids, rec = sample_central_direct(rows, group_of=group_of, stratum_of=stratum_of, seed=5, target=4)
    assert rec["by_stratum"]["B"]["quota"] == 0 and rec["by_stratum"]["B"]["taken"] == 0
    assert all(stratum_of[i] == "A" for i in ids)


@pytest.mark.parametrize(("case", "code"), [("spans", "lists_group_spans_strata"), ("over", "lists_train_short"),
                                            ("zero", "lists_train_short")])
def test_입력이_틀리면_거부한다(case, code):
    rows, group_of, stratum_of = _pool({"A": [2, 2], "B": [2]})
    T = 3
    if case == "spans":                                          # 한 묶음이 두 층에 걸친다
        stratum_of[rows[0]["image_id"]] = "B"
    elif case == "over":
        T = 7
    else:
        T = 0
    with pytest.raises(ListsRefused) as exc:
        sample_central_direct(rows, group_of=group_of, stratum_of=stratum_of, seed=1, target=T)
    assert exc.value.code == code


# ================================================================ 목록 전체 — 진단 계획
def world(seed, *, accum_avoid=False, central_rows="direct"):
    rnd = random.Random(seed)
    man, pairs = [], []
    k = 0
    for g in range(10):
        s = f"{'ST' if g % 3 else 'AL'}|c{g % 2}"
        for _ in range(rnd.randint(1, 3)):
            k += 1
            man.append({"image_id": f"v{k:04d}", "split": "val", "group_id": f"gv{g}", "strata_key": s, "client": "C1"})
            pairs.append({"image_id": f"v{k:04d}", "split": "val", "client": "C1",
                          "skeleton": {"defects": [{}] * rnd.choice([1, 1, 2, 3])}})
    for g in range(30):
        c = ("C1", "C1", "C2", "C3")[g % 4]
        s = ("ST|c0", "ST|c1", "AL|c0")[g % 3]
        for _ in range(rnd.randint(1, 5)):
            k += 1
            man.append({"image_id": f"t{k:04d}", "split": "train", "group_id": f"gt{g}", "strata_key": s, "client": c})
            pairs.append({"image_id": f"t{k:04d}", "split": "train", "client": c, "skeleton": {"defects": [{}]}})
    for g in range(3):
        k += 1
        man.append({"image_id": f"e{k:04d}", "split": "eval", "group_id": f"ge{g}", "strata_key": None, "client": None})
    evals = [m["image_id"] for m in man if m["split"] == "eval"]
    trains = [m["image_id"] for m in man if m["split"] == "train"]
    edges = [(trains[0], evals[0]), (trains[5], evals[1])]           # 학습 묶음 둘이 평가와 이어져 통째로 빠진다
    plan = {"sampling_seed": seed, "gen": {"split": "val", "defect_only": True, "target_boxes": 6, "max_n": 30,
                                           "min_groups_per_stratum": 1, "max_target_tokens": None},
            "train": {"split": "train", "rows": {"central": 20}, "avoid_multiple_of_accum": accum_avoid,
                      "central_rows": central_rows},
            "exclude": {"drop_components_touching_eval": True}}
    return man, pairs, edges, plan


def test_진단_계획은_중앙_칸_하나를_중앙_풀에서_바로_뽑는다():
    man, pairs, edges, plan = world(21)
    res = build_lists(man, pairs, edges, _P(plan))
    assert set(res.train_rows) == {CENTRAL_TAG}                  # 참여자 목록을 내지 않는다
    rows = res.train_rows[CENTRAL_TAG]
    meta = {m["image_id"]: m for m in man}
    dropped = {g for a, _ in edges for g in [meta[a]["group_id"]]}
    assert not {meta[r["image_id"]]["group_id"] for r in rows} & dropped     # 평가와 이어진 묶음은 빠진다
    assert [r["image_id"] for r in rows] == [p["image_id"] for p in pairs if p["image_id"] in {r["image_id"] for r in rows}]
    rec = res.record["central_direct"]
    assert rec["target"] == 20 and rec["n_rows"] == len(rows) >= 20
    assert res.record["eval_intersection"] == 0


@pytest.mark.parametrize(("over", "code"), [({"accum_avoid": True}, "lists_direct_accum"),
                                            ({"central_rows": "both"}, "lists_central_rows")])
def test_진단_계획의_정하지_않은_칸은_거부한다(over, code):
    man, pairs, edges, plan = world(22, **over)
    with pytest.raises(ListsRefused) as exc:
        build_lists(man, pairs, edges, _P(plan))
    assert exc.value.code == code


# ================================================================ 두 구현(§13-3) — 독립 구현과 목록 바이트까지
G_PATH = ROOT / "tests" / "independent" / "fdiag_lists_g.py"
G_SHA256 = "2e45fc8ad502722fc01da09655a9f44f71cc7a9cd575472762bf50a9905e400b"
"""메타데이터 검토 트랙의 독립 구현 — wt/G `ed1982e` 의 `docs/dev_log/2026-09-30-리허설/22_진단표본_독립구현_G.py` 바이트 그대로(그 트랙 22번 표의 해시)."""


def _g():
    raw = G_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == G_SHA256, "독립 구현의 바이트가 고정 값과 다르다"
    spec = importlib.util.spec_from_file_location("fdiag_lists_g", G_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 결정 20 덧붙임(10-05 06시) — 층 안 묶음 순서의 정본 식. 독립 구현은 계획 주석의 식(염 없음)으로 짰고, 22번 7-1 의 "순서 키만 바꾼 같은 구현" 이 이것이다
    mod.group_key = lambda group_id, seed: hashlib.sha256(f"{group_id}|{seed}|central_direct".encode()).hexdigest()
    return mod


def world_cmp(seed):
    """맞대기 세계 — 학습 묶음 · 층 · 크기 · 평가 이음을 무작위로. 검증에 평가와 잇지 않는 묶음을 하나 두어 생성 목록이 늘 선다."""
    rnd = random.Random(seed)
    man, pairs = [], []
    k = 0
    strata = ["AL|__normal__", "AL|crack", "ST|__normal__", "ST|porosity", "ST|crack"][: rnd.randint(1, 5)]
    for g in range(rnd.randint(3, 8)):
        for _ in range(rnd.randint(1, 3)):
            k += 1
            man.append({"image_id": f"v{k:04d}", "split": "val", "group_id": f"gv{g}", "strata_key": strata[0], "client": "C1"})
            pairs.append({"image_id": f"v{k:04d}", "split": "val", "client": "C1", "skeleton": {"defects": [{}]}})
    for g in range(rnd.randint(8, 40)):
        s = rnd.choice(strata)
        c = rnd.choice(("C1", "C2", "C3"))
        for _ in range(rnd.randint(1, 7)):
            k += 1
            man.append({"image_id": f"t{k:04d}", "split": "train", "group_id": f"gt{g}", "strata_key": s, "client": c})
            pairs.append({"image_id": f"t{k:04d}", "split": "train", "client": c, "skeleton": {"defects": [{}]}})
    rnd.shuffle(pairs)                                           # 페어 파일의 순서는 매니페스트와 다르다
    for g in range(rnd.randint(1, 5)):
        for _ in range(rnd.randint(1, 3)):
            k += 1
            man.append({"image_id": f"e{k:04d}", "split": "eval", "group_id": f"ge{g}", "strata_key": None, "client": None})
    ids = [m["image_id"] for m in man if m["group_id"] != "gv0"]  # gv0 은 평가와 잇지 않는다
    edges = [(rnd.choice(ids), rnd.choice(ids)) for _ in range(rnd.randint(0, 25))]
    n_train = sum(1 for m in man if m["split"] == "train")
    plan = {"sampling_seed": rnd.randint(1, 10 ** 9),
            "gen": {"split": "val", "defect_only": True, "target_boxes": 1, "max_n": 50, "min_groups_per_stratum": 0,
                    "max_target_tokens": None},
            "train": {"split": "train", "rows": {"central": rnd.randint(1, max(1, n_train))},
                      "avoid_multiple_of_accum": False, "central_rows": "direct"},
            "exclude": {"drop_components_touching_eval": True}}
    return man, pairs, edges, plan


def _g_side(mod, man, pairs, edges, plan):
    """독립 구현의 산출 — 평가 행을 뺀 학습 · 검증 뷰만 준다(그 구현의 입력 규약). 목록 파일은 정본의 꼴로 쓴다: 고른 장의 페어 행을
    **페어 파일의 순서**대로, 한 줄에 `json.dumps(행, ensure_ascii=False)`."""
    view = [{k: str(m[k]) for k in ("image_id", "split", "group_id", "strata_key")} for m in man if m["split"] != "eval"]
    drop, _iso = mod.excluded_groups(view, list(edges))
    try:
        picked, info = mod.select(view, drop, split="train", total=int(plan["train"]["rows"]["central"]),
                                  seed=int(plan["sampling_seed"]), avoid_multiple=False, variant="A0", pair_ids=None,
                                  manifest_pos={r["image_id"]: i for i, r in enumerate(view)})
    except SystemExit:
        return None
    chosen = set(picked)
    rows = [p for p in pairs if p["split"] == "train" and p["image_id"] in chosen]
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    dropped_rows = sum(1 for r in view if r["group_id"] in drop)
    return {"picked": picked, "rows": rows, "raw": raw, "strata": info["strata"], "drop": drop, "dropped_rows": dropped_rows}


@pytest.mark.parametrize("seed", range(60))
def test_두_구현이_같은_진단_학습_목록_바이트를_낸다(seed):
    """결정 20 의 두 구현 행 — **목록 파일의 바이트**까지 같다. 순서는 페어 파일의 순서이고(정본), 집합 · 층마다의 몫 · 넣은 행 · 초과 ·
    뺀 묶음 · 학습 행 digest 가 같다. 목표가 후보보다 크면 둘 다 거부한다. 목표 0 은 계획 판독기가 막아(`plan_value`) 맞대지 않는다."""
    from vlm.pilot_vlm import train_rows_digest
    from vlm.rehearsal_run import train_list_bytes

    man, pairs, edges, plan = world_cmp(seed)
    g = _g_side(_g(), man, pairs, edges, plan)
    try:
        a = build_lists(man, pairs, edges, _P(plan))
    except ListsRefused as exc:
        if exc.code == "lists_stratum_empty":
            # 학습 풀의 한 층이 제외로 통째로 빠졌다 — 진단 재개 조건의 자리라 이 구현은 멈추고(결정 21 덧붙임의 3) 독립 구현은 그 층 없이 뽑는다.
            # 독립 구현의 제외 결과로도 그런 층이 있는지 따로 본다
            view = [m for m in man if m["split"] == "train"]
            drop, _iso = _g().excluded_groups([{k: str(m[k]) for k in ("image_id", "split", "group_id", "strata_key")}
                                               for m in man if m["split"] != "eval"], list(edges))
            assert {m["strata_key"] for m in view} - {m["strata_key"] for m in view if m["group_id"] not in drop}
            return
        assert exc.code == "lists_train_short" and g is None, exc.code
        return
    assert g is not None
    rows = a.train_rows[CENTRAL_TAG]
    assert train_list_bytes(rows) == g["raw"]                                     # 직렬화 바이트
    assert [r["image_id"] for r in rows] == [r["image_id"] for r in g["rows"]]    # 최종 순서
    assert sorted(r["image_id"] for r in rows) == sorted(g["picked"])
    assert train_rows_digest(rows) == train_rows_digest(g["rows"])
    rec = a.record["central_direct"]
    assert {s: (v["quota"], v["taken"], v["over"]) for s, v in rec["by_stratum"].items()} \
        == {s: (v["quota"], v["rows"], v["excess"]) for s, v in g["strata"].items()}
    assert (a.record["excluded"]["n_groups"], a.record["excluded"]["n_images"]) == (len(g["drop"]), g["dropped_rows"])


# ================================================================ 진단 계획의 칸
FPLAN = (ROOT / "configs" / "frame_diag_uni.yaml").read_bytes()


def test_저장소의_진단_계획은_새_칸_검사를_지난다():
    p = parse_plan(FPLAN, purpose="frame_diag")
    assert p.get("train.central_rows") == "direct" and p.get("cells") == ["uni_central"]


@pytest.mark.parametrize(("old", "new"), [
    (b"central_rows: direct", b"central_rows: union"),
    (b"    central: 4000", b"    C1: 4000"),
    (b"cells: [uni_central]", b"cells: [uni_central, uni_local_C1]"),
    (b"attempts_allowed: 1 ", b"attempts_allowed: 0 "),
    (b"target_boxes: 1416", b"target_boxes: 0"),
])
def test_진단_계획의_칸이_틀리면_거부한다(old, new):
    assert FPLAN.count(old) == 1, old
    with pytest.raises(PlanRejected) as exc:
        parse_plan(FPLAN.replace(old, new), purpose="frame_diag")
    assert exc.value.code == "plan_value"


# ================================================================ 목록 단계의 타깃 길이(실제 구현의 셈)
def test_타깃_길이는_학습_타깃을_토크나이저로_센다(tmp_path, monkeypatch):
    """실제 구현 — 프로세서만 연다(모델을 올리지 않는다). 원본 크기는 이미지 머리에서, 타깃은 학습과 같은 `build_target` 이다."""
    import transformers
    from PIL import Image

    from vlm.coords import CoordCfg, ImageGeom
    from vlm.pilot_vlm import build_target
    from vlm.rehearsal_run import StageRefused, target_token_counts

    calls = []

    class _Tok:
        def __call__(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return {"input_ids": list(range(len(text)))}

    class _Proc:
        tokenizer = _Tok()

    def fake(model_id, revision=None, **kw):
        calls.append((model_id, revision, kw))
        return _Proc()

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(fake))
    p = tmp_path / "a.png"
    Image.new("L", (640, 360)).save(p)
    row = {"image_id": "v1", "image_path": str(p),
           "skeleton": {"defects": [{"type": "2011", "bbox_px": [10, 20, 30, 40]}], "verdict": "불합격", "clauses": []}}
    cfg = type("C", (), {"coord_space": "ABS_ORIG", "model_id": "m", "model_revision": "r",
                         "processor_kwargs": {"max_pixels": 9}})()
    got = target_token_counts([row], config=cfg)
    assert got == {"v1": len(build_target(row, ImageGeom(640, 360), CoordCfg(coord_space="ABS_ORIG")))}
    assert calls == [("m", "r", {"max_pixels": 9})]
    cfg.coord_space = "ABS_RESIZED"
    with pytest.raises(StageRefused) as exc:
        target_token_counts([row], config=cfg)
    assert exc.value.code == "target_tokens_coord"
