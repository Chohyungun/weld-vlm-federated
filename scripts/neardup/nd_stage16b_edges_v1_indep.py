# 추적 사본 — 근사 중복 간선 판본 neardup_edges_v1 의 **독립 구현**이다(원저자 아닌 하위 단위가 2026-10-05 에
# 생성기 `nd_stage16_edges_v1.py` 를 열지 않고 같은 입력에서 따로 짰다). 원본(추적 밖 스크래치 `indep_edges.py`)의
# sha256 은 abcf0cd5b3ffb0af71d30de908a05ce1206a531c322e6adf7d50e24db6b04af5 이다. 원본에서 바꾼 것은 이 주석, 머리 설명의
# 두 줄(실제 동작대로), 경로 세 줄(명령줄 인자로), 출력 경로 보호(외부 검토 §49 — 새 폴더만 · 동결 경로 밖, 디렉터리를
# 만들고 입력을 읽기 전에)뿐이다. 입력 조립과 판정은 원본 그대로다.
#     python nd_stage16b_edges_v1_indep.py <특징 폴더> <매니페스트 csv> <없는 새 출력 폴더>
# -*- coding: utf-8 -*-
"""근사 중복 간선 집합의 독립 재산출 — 입력을 수정하지 않고, 지정한 새 출력 폴더에 결과를 쓴다.

입력: S = _workspace/2026-09-21-neardup 의 CSV·규칙 JSON, 매니페스트 image_id 열.
출력: 명령줄로 받은 **없는 새 출력 폴더**에 edges.csv · undecided.csv · summary.json.
화면에는 수와 해시만 찍는다(식별자는 찍지 않는다).
"""
import ast
import csv
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

S = Path(sys.argv[1])            # 추적 사본에서 바꾼 줄 — 원본은 로컬 절대경로였다
MANIFEST = Path(sys.argv[2])     # 추적 사본에서 바꾼 줄
OUT = Path(sys.argv[3])          # 추적 사본에서 바꾼 줄


def _guard_out(out: Path) -> None:
    """출력 경로 보호(추적 사본에서 더한 것 — §49). 디렉터리를 만들고 입력을 읽기 **전에** 부른다.

    - 출력 폴더가 이미 있으면 멈춘다 — 이전 비교 결과나 봉인본을 덮어쓰지 않는다(원 생성기와 같다).
    - 동결 명부의 자리와 그 아래, 계약서(`SNAPSHOT.sha256`)가 있는 조상 폴더 아래에는 쓰지 않는다. 정션을 풀어 비교한다.
    """
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from data.frozen_guard import EXPECTED_SEALED, assert_writable

    if out.exists():
        sys.exit(f"출력 폴더가 이미 있다 — 없는 새 폴더만 쓴다: {out}")
    real = os.path.normcase(os.path.realpath(out))
    for rel in EXPECTED_SEALED:
        sealed = os.path.normcase(os.path.realpath(repo / rel))
        if real == sealed or real.startswith(sealed.rstrip(os.sep) + os.sep):
            sys.exit(f"동결 명부의 자리에는 쓰지 않는다: {rel}")
    for anc in [out.parent, *out.parent.parents]:
        if anc.exists():
            assert_writable(anc, what="출력 폴더의 조상")


_guard_out(OUT)
OUT.mkdir(parents=True)

csv.field_size_limit(1 << 30)

# ---------------------------------------------------------------- 1. 규칙
rule = json.loads((S / "stage12_frozen_rule.json").read_text(encoding="utf-8"))
EXPR = rule["고른_자동제외"]["식"]
expr_sha = hashlib.sha256(EXPR.encode("utf-8")).hexdigest()
expr_sha_ok = expr_sha == rule["식의_sha256"]

tree = ast.parse(EXPR, mode="eval")
names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
FUNCS = {"max", "abs"}
VARS = names - FUNCS
assert VARS == {"rd_dx", "rd_dy", "blk_med", "txt_gap", "txt_energy"}, VARS
TXT = ("txt_gap", "txt_energy")

# 문자 띠 변수는 상수와의 단순 비교에만 나오는가 — 그래야 문턱 양옆의 점만 보면 모든 값을 본 것이 된다.
thresholds = defaultdict(set)
txt_name_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in TXT]
txt_ok_nodes = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        l, r = node.left, node.comparators[0]
        if isinstance(l, ast.Name) and l.id in TXT and isinstance(r, ast.Constant):
            thresholds[l.id].add(float(r.value))
            txt_ok_nodes.add(id(l))
        elif isinstance(r, ast.Name) and r.id in TXT and isinstance(l, ast.Constant):
            thresholds[r.id].add(float(l.value))
            txt_ok_nodes.add(id(r))
assert all(id(n) in txt_ok_nodes for n in txt_name_nodes), "문자 띠 변수가 단순 비교 밖에 나온다"

TEST_VALUES = {}
for v in TXT:
    pts = {-1e300, 0.0, 1e300}
    for t in thresholds[v]:
        pts |= {t, math.nextafter(t, -math.inf), math.nextafter(t, math.inf)}
    TEST_VALUES[v] = sorted(pts)

CODE = compile(tree, "<frozen_rule>", "eval")
GLOBALS = {"__builtins__": {}, "max": max, "abs": abs}


def eval_rule(env):
    return bool(eval(CODE, GLOBALS, dict(env)))


def judge_eval(vals):
    """식 문자열을 그대로 평가. 문자 띠가 비면 문턱 양옆 점을 모두 넣어 결과가 하나인지 본다."""
    if vals["rd_dx"] is None or vals["rd_dy"] is None:
        return None, "no_shift"
    if vals["blk_med"] is None:
        return None, "no_blk_med"
    missing = [v for v in TXT if vals[v] is None]
    if not missing:
        return eval_rule(vals), None
    outs = set()
    for combo in itertools.product(*[TEST_VALUES[v] for v in missing]):
        env = dict(vals)
        env.update(zip(missing, combo))
        outs.add(eval_rule(env))
    if len(outs) == 1:
        return outs.pop(), None
    return None, "needs_txt"


def judge_struct(vals):
    """식을 손으로 풀어 쓴 판정 (교차 확인용)."""
    dx, dy, bm, tg, te = (vals[k] for k in ("rd_dx", "rd_dy", "blk_med", "txt_gap", "txt_energy"))
    if dx is None or dy is None:
        return None, "no_shift"
    if bm is None:
        return None, "no_blk_med"
    if not (max(abs(dx), abs(dy)) <= 48 and bm >= 0.15):
        return False, None
    if not (bm < 0.2):
        return True, None
    g1 = None if tg is None else (tg >= 0.3)
    g2 = None if te is None else (te >= 1.0)
    if g1 is False or g2 is False:
        return True, None
    if g1 is True and g2 is True:
        return False, None
    return None, "needs_txt"


# ---------------------------------------------------------------- 2. 짝 집합
SETS = {
    "cross_eval_vs_trainval": {
        "shift": ["feat3_cross_eval_vs_trainval.csv", "feat3_rest_cross_eval_vs_trainval.csv", "feat3_new_cross.csv"],
        "blk": ["feat5_cross_eval_vs_trainval.csv", "feat5_rest_cross_eval_vs_trainval.csv", "feat5_new_cross.csv"],
        "txt": ["feat7_cross_eval_vs_trainval.csv", "feat7_rest_cross_eval_vs_trainval.csv", "feat7_new_cross.csv"],
        "cand": ["cand_cross_eval_vs_trainval.csv", "bpcand_new_cross.csv"],
    },
    "within_eval": {
        "shift": ["feat3_within_eval.csv"],
        "blk": ["feat5_within_eval.csv"],
        "txt": [],
        "cand": ["cand_within_eval.csv"],
    },
    "within_trainval": {
        "shift": ["feat3d_within_trainval.csv"],
        "blk": ["feat5_within_trainval.csv"],
        "txt": [],
        "cand": ["cand_within_trainval.csv"],
    },
}
COLS = {"shift": ["rd_dx", "rd_dy"], "blk": ["blk_med"], "txt": ["txt_gap", "txt_energy"], "cand": []}
THRESH_OF = {"rd_dx": [48.0, -48.0], "rd_dy": [48.0, -48.0], "blk_med": [0.15, 0.2], "txt_gap": [0.3], "txt_energy": [1.0]}

parse_stats = Counter()


def parse(s):
    if s is None:
        parse_stats["none_field"] += 1
        return None
    t = s.strip()
    if t == "":
        parse_stats["empty"] += 1
        return None
    try:
        v = float(t)
    except ValueError:
        parse_stats["non_numeric"] += 1
        return None
    if math.isnan(v):
        parse_stats["nan"] += 1
        return None
    if math.isinf(v):
        parse_stats["inf"] += 1
    return v


def norm(a, b):
    return (a, b) if a <= b else (b, a)


diag = {}
set_pairs = {}
set_vals = {}
file_meta = {}

for sname, groups in SETS.items():
    pairs = set()
    vals = {c: defaultdict(list) for g in ("shift", "blk", "txt") for c in COLS[g]}
    d = Counter()
    group_pairs = {g: set() for g in groups}
    for g, files in groups.items():
        for fn in files:
            p = S / fn
            raw = p.read_bytes()
            fm = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "rows": 0,
                  "swapped": 0, "self": 0, "dup_rows_in_file": 0, "bom": raw.startswith(b"\xef\xbb\xbf")}
            seen_in_file = Counter()
            with open(p, newline="", encoding="utf-8-sig") as f:
                rd = csv.DictReader(f)
                for need in ["image_id_a", "image_id_b"] + COLS[g]:
                    assert need in rd.fieldnames, (fn, need)
                for row in rd:
                    fm["rows"] += 1
                    a, b = row["image_id_a"].strip(), row["image_id_b"].strip()
                    if a != row["image_id_a"] or b != row["image_id_b"]:
                        d["id_whitespace_stripped"] += 1
                    if a == b:
                        fm["self"] += 1
                    if a > b:
                        fm["swapped"] += 1
                    k = norm(a, b)
                    seen_in_file[k] += 1
                    group_pairs[g].add(k)
                    for c in COLS[g]:
                        v = parse(row[c])
                        vals[c][k].append((v, fn))
                        for t in THRESH_OF[c]:
                            if v is not None and v == t:
                                d[f"exact_threshold_{c}"] += 1
                            elif v is not None and abs(v - t) <= 1e-9:
                                d[f"near_threshold_{c}"] += 1
            fm["dup_rows_in_file"] = sum(n - 1 for n in seen_in_file.values() if n > 1)
            file_meta[f"{sname}/{fn}"] = fm
    pairs = group_pairs["cand"] | group_pairs["shift"]
    d["pairs"] = len(pairs)
    d["cand_pairs"] = len(group_pairs["cand"])
    d["shift_pairs"] = len(group_pairs["shift"])
    d["cand_not_in_shift"] = len(group_pairs["cand"] - group_pairs["shift"])
    d["shift_not_in_cand"] = len(group_pairs["shift"] - group_pairs["cand"])
    d["blk_pairs_outside_set"] = len(group_pairs["blk"] - pairs)
    d["txt_pairs_outside_set"] = len(group_pairs.get("txt", set()) - pairs)
    d["self_pairs_in_set"] = sum(1 for a, b in pairs if a == b)
    # 같은 짝 같은 열의 다중 값
    merged = {}
    for c, m in vals.items():
        mc = {}
        for k, lst in m.items():
            present = [v for v, _ in lst if v is not None]
            distinct = sorted(set(present))
            if len(lst) > 1:
                d[f"multi_rows_{c}"] += 1
            if len(distinct) > 1:
                d[f"conflict_{c}"] += 1
                mc[k] = ("CONFLICT", distinct)
            elif len(distinct) == 1:
                if len(present) < len(lst):
                    d[f"present_and_missing_{c}"] += 1
                mc[k] = distinct[0]
            else:
                mc[k] = None
        merged[c] = mc
    set_pairs[sname] = pairs
    set_vals[sname] = merged
    diag[sname] = d

# ---------------------------------------------------------------- 3. 판정
results = {}  # (set, pair) -> (verdict, reason)
mismatch_methods = 0
conflict_resolution = Counter()
missing_pattern = Counter()
for sname, pairs in set_pairs.items():
    merged = set_vals[sname]
    for k in pairs:
        base = {}
        conflict_cols = {}
        for c in ("rd_dx", "rd_dy", "blk_med", "txt_gap", "txt_energy"):
            v = merged[c].get(k) if c in merged else None
            if isinstance(v, tuple) and v and v[0] == "CONFLICT":
                conflict_cols[c] = v[1]
                base[c] = None
            else:
                base[c] = v
        missing_pattern[(sname,) + tuple(base[c] is None for c in ("rd_dx", "rd_dy", "blk_med", "txt_gap", "txt_energy"))] += 1
        if conflict_cols:
            outs = set()
            for combo in itertools.product(*conflict_cols.values()):
                env = dict(base)
                env.update(zip(conflict_cols.keys(), combo))
                outs.add(judge_eval(env))
            if len(outs) == 1:
                res = outs.pop()
                conflict_resolution["same_verdict"] += 1
            else:
                res = (None, "value_conflict")
                conflict_resolution["verdict_differs"] += 1
        else:
            res = judge_eval(base)
            res2 = judge_struct(base)
            if res != res2:
                mismatch_methods += 1
        results[(sname, k)] = res

# ---------------------------------------------------------------- 4. 확인
manifest_ids = set()
with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
    rd = csv.DictReader(f)
    assert "image_id" in rd.fieldnames
    n_manifest_rows = 0
    for row in rd:
        n_manifest_rows += 1
        manifest_ids.add(row["image_id"])

pair_sets_of = defaultdict(set)
for (sname, k) in results:
    pair_sets_of[k].add(sname)
multi_set_pairs = [k for k, ss in pair_sets_of.items() if len(ss) > 1]

edges = set()
edges_with_set = []
undecided = []
per_set = {}
for sname in SETS:
    c = Counter()
    for (s2, k), (v, reason) in results.items():
        if s2 != sname:
            continue
        c["pairs"] += 1
        if v is True:
            c["edge"] += 1
            edges.add(k)
            edges_with_set.append((k, sname))
        elif v is False:
            c["non_edge"] += 1
        else:
            c["undecided"] += 1
            c[f"undecided_{reason}"] += 1
            undecided.append((k[0], k[1], sname, reason))
    per_set[sname] = dict(c)

edge_ids = {x for k in edges for x in k}
missing_edge_ids = edge_ids - manifest_ids
all_pair_ids = {x for k in pair_sets_of for x in k}
missing_any_ids = all_pair_ids - manifest_ids

# multi-set 짝의 판정이 갈리는가
multi_set_verdict_split = sum(
    1 for k in multi_set_pairs if len({results[(s, k)] for s in pair_sets_of[k]}) > 1
)
multi_set_edge_in_any = sum(1 for k in multi_set_pairs if k in edges)

# ---------------------------------------------------------------- 5. 바이트·해시
edge_sorted = sorted(edges)
assert all(a < b for a, b in edge_sorted)
edge_text = "a_id,b_id\n" + "".join(f"{a},{b}\n" for a, b in edge_sorted)
edge_bytes = edge_text.encode("utf-8")
edge_sha = hashlib.sha256(edge_bytes).hexdigest()

und_sorted = sorted(undecided)
und_text = "a_id,b_id,set,reason\n" + "".join(f"{a},{b},{s},{r}\n" for a, b, s, r in und_sorted)
und_bytes = und_text.encode("utf-8")
und_sha = hashlib.sha256(und_bytes).hexdigest()

(OUT / "edges.csv").write_bytes(edge_bytes)
(OUT / "undecided.csv").write_bytes(und_bytes)

undecided_by_reason = Counter(r for *_, r in undecided)

summary = {
    "expr_sha256_computed": expr_sha,
    "expr_sha256_ok": expr_sha_ok,
    "txt_test_values": TEST_VALUES,
    "per_set": per_set,
    "edges_total_unique": len(edges),
    "edges_total_with_set": len(edges_with_set),
    "undecided_total": len(undecided),
    "undecided_by_reason": dict(undecided_by_reason),
    "edge_file_bytes": len(edge_bytes),
    "edge_file_sha256": edge_sha,
    "undecided_file_bytes": len(und_bytes),
    "undecided_file_sha256": und_sha,
    "edge_ids_unique": len(edge_ids),
    "edge_ids_missing_in_manifest": len(missing_edge_ids),
    "all_pair_ids_missing_in_manifest": len(missing_any_ids),
    "manifest_rows": n_manifest_rows,
    "manifest_unique_ids": len(manifest_ids),
    "pairs_in_multiple_sets": len(multi_set_pairs),
    "multi_set_verdict_split": multi_set_verdict_split,
    "multi_set_edge": multi_set_edge_in_any,
    "eval_vs_struct_mismatch": mismatch_methods,
    "conflict_resolution": dict(conflict_resolution),
    "parse_stats": dict(parse_stats),
    "diag": {k: dict(v) for k, v in diag.items()},
    "missing_pattern(rd_dx,rd_dy,blk_med,txt_gap,txt_energy)": {"|".join(map(str, k)): v for k, v in sorted(missing_pattern.items())},
    "file_meta": file_meta,
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1))
