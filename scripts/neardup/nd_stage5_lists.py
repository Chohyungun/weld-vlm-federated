"""근사 중복 5단계 — 정의별 집계와 eval 쪽 제외 목록.

입력: 세 생성기가 찾은 교차 쌍의 특징(feat3·feat4·feat5, 걸러졌던 쪽의 feat3_rest·feat5_rest, G3 만 찾은 쌍의 feat3_new·feat5_new)과
      같은 쪽 안의 쌍(eval 안쪽 feat3_within_eval, train·val 안쪽 feat3d_within_trainval).
정의: `scripts/neardup/definitions.json` — 이름과 규칙(특징 열에 대한 불리언 식). **규칙에는 화소·해시 특징만 쓴다.**
      정답 라벨·모델 출력은 규칙에 쓸 수 없다(아래 ALLOWED 밖의 이름은 거부한다). 라벨은 제외가 끝난 뒤의 분포 집계에만 쓴다.
출력: 제외 목록(식별자)은 `_workspace/2026-09-21-neardup/` 에만 쓰고, 요약 JSON 에는 건수·분포·목록의 sha256 만 적는다.

사용: python scripts/neardup/nd_stage5_lists.py
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
DEFS_PATH = Path(__file__).resolve().parent / "definitions.json"
csv.field_size_limit(10_000_000)

#: 규칙에 쓸 수 있는 열 — 전부 화소·해시에서 나온 값이다.
ALLOWED = {
    "hamming", "thumb_rmsd", "mad", "rmsd", "corr", "p99_absdiff", "frac_gt8", "grad_corr", "bp_corr", "bp_peak",
    "rd_peak", "rd_dx", "rd_dy", "rd_side", "rd_sharp", "ov", "reg_corr", "reg_mad_tone", "reg_fine", "reg_mid", "reg_coarse",
    "v1_mad", "v1_corr", "v1_reg_mad", "v1_reg_corr", "v1_reg_mad_gain", "blk_n", "blk_frac", "blk_med", "blk_min",
    "hi_n", "hi_min", "hi_min_energy", "hi_badfrac",
    "txt_corr", "body_corr", "txt_gap", "txt_energy",
    "bpdesc_corr", "shift_peak", "shift_px",
}
#: 규칙에 쓸 수 있는 함수.
FUNCS = {"abs": abs, "max": max, "min": min}
NO_LABEL = "(정상)"


def compile_rule(expr: str):
    """규칙 식을 검사하고 (함수, 쓰인 열 이름)을 돌려준다. 허용한 열 이름·숫자·비교·and/or/not·사칙·abs/max/min 만 통과시킨다."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED and node.id not in FUNCS:
                raise ValueError(f"규칙에 쓸 수 없는 이름: {node.id!r} — 화소·해시 특징만 쓴다")
        elif isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in FUNCS) or node.keywords:
                raise TypeError("규칙에서 부를 수 있는 것은 abs·max·min 뿐이다")
        elif not isinstance(node, (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.Compare,
                                   ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
                                   ast.Div, ast.Constant, ast.Load)):
            raise TypeError(f"규칙에 쓸 수 없는 구문: {type(node).__name__}")
    code = compile(tree, "<rule>", "eval")
    names = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in FUNCS})

    def rule(row: dict) -> bool:
        env = {}
        for k in names:
            if k not in row or row[k] == "":
                return False                       # 그 특징이 없는 쌍(G3 만 찾은 쌍의 mad 등)은 그 규칙에 들지 않는다
            env[k] = float(row[k])
        return bool(eval(code, {"__builtins__": {}, **FUNCS}, env))   # 위에서 구문을 걸렀다 — 허용한 열과 연산만 남는다

    return rule, names


def _read(name: str) -> list[dict]:
    p = W / name
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []


def _join(rows: list[dict], *names: str) -> None:
    for name in names:
        extra = {(r["image_id_a"], r["image_id_b"]): r for r in _read(name)}
        for r in rows:
            e = extra.get((r["image_id_a"], r["image_id_b"]))
            if e:
                r.update({k: v for k, v in e.items() if k not in r})


def _shift(rows: list[dict]) -> list[dict]:
    for r in rows:
        r["shift_px"] = f"{(float(r['rd_dx']) ** 2 + float(r['rd_dy']) ** 2) ** 0.5:.2f}"
    return rows


def load_cross() -> list[dict]:
    old = [r for r in _read("feat3_cross_eval_vs_trainval.csv") if r["selected"] == "1"]
    _join(old, "feat4_cross_eval_vs_trainval.csv", "feat5_cross_eval_vs_trainval.csv", "feat6_cross_eval_vs_trainval.csv")
    for r in old:
        r["generator"] = "G1∪G2"
    # 2d 의 사전 거름이 뺐던 쌍 — 2g 가 전량에 특징을 달았다. 없으면(2g 를 안 돌렸으면) 그 사실을 알린다.
    rest = _read("feat3_rest_cross_eval_vs_trainval.csv")
    if not rest:
        print("!! feat3_rest 가 없다 — 걸러졌던 쪽 후보에는 규칙이 적용되지 않는다(nd_stage2g_rest.py 를 먼저 돌려라)", flush=True)
    _join(rest, "feat5_rest_cross_eval_vs_trainval.csv", "feat6_rest_cross_eval_vs_trainval.csv")
    for r in rest:
        r["generator"] = "G1∪G2(거름 뒤)"
    new = _read("feat3_new_cross.csv")
    _join(new, "feat5_new_cross.csv", "feat6_new_cross.csv")
    for r in new:
        r["generator"] = "G3만"
    # G4(큰 이동)가 새로 올린 쌍 — 3e 가 정밀 특징을 단 것만. 전수가 끝나기 전에는 일부다.
    big = [r for r in _read("feat3_shift_cross.csv") if r.get("selected") == "1"]
    _join(big, "feat5_shift_cross.csv", "feat6_shift_cross.csv")
    for r in big:
        r["generator"] = "G4만"
    rows = _shift(old + rest + new + big)
    miss = sum("hi_min" not in r or r["hi_min"] == "" for r in rows)
    if miss:
        print(f"!! 고에너지 블록 값이 없는 쌍 {miss:,} — 그 쌍은 자동 제외 규칙을 통과하지 못한다(검토 띠로 간다)", flush=True)
    return rows


def load_within_eval() -> list[dict]:
    rows = [r for r in _read("feat3_within_eval.csv") if r["selected"] == "1"]
    _join(rows, "feat4_within_eval.csv", "feat5_within_eval.csv")
    return _shift(rows)


def load_within_trainval() -> list[dict]:
    rows = _read("feat3d_within_trainval.csv")
    _join(rows, "feat5_within_trainval.csv")
    return _shift(rows)


def _classes(m: dict) -> list[str]:
    return m["defect_types"].split(";") if m["defect_types"] else [NO_LABEL]


def _missing(names: list[str], rows: list[dict]) -> list[str]:
    """그 쌍 집합에 아예 없는 열 — 있으면 그 규칙은 그 집합에 적용할 수 없다(0 이 아니라 '해당 없음')."""
    return [k for k in names if not rows or k not in rows[0]]


def _components(edges: list[tuple[str, str]]) -> list[int]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    return sorted(Counter(find(x) for x in list(parent)).values(), reverse=True)


def main() -> int:
    defs = json.loads(DEFS_PATH.read_text(encoding="utf-8"))["definitions"]
    man = {r["image_id"]: r for r in csv.DictReader((ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline=""))}
    prov = {r["image_id"]: r["provenance"] for r in csv.DictReader((ROOT / "data/interim/manifest_v1/tiles.csv").open(encoding="utf-8", newline=""))}
    ev_all = [i for i, m in man.items() if m["split"] == "eval"]
    n_val = sum(m["split"] == "val" for m in man.values())

    def dist(ids) -> dict:
        by = defaultdict(Counter)
        for i in ids:
            m = man[i]
            by["provenance"][prov[i]] += 1
            by["material"][m["material"]] += 1
            by["has_defect"][m["has_defect"]] += 1
            by["provenance_x_has_defect"][f"{prov[i]}/{m['has_defect']}"] += 1
            for c in _classes(m):
                by["class"][c] += 1
                by["material_x_class"][f"{m['material']}/{c}"] += 1
        return {k: dict(sorted(v.items())) for k, v in by.items()}

    cross, w_eval, w_tv = load_cross(), load_within_eval(), load_within_trainval()
    summary = {"n_eval": len(ev_all), "n_eval_groups": len({man[i]["group_id"] for i in ev_all}),
               "n_pairs_examined": {"cross": len(cross), "within_eval": len(w_eval), "within_trainval": len(w_tv)},
               "eval_totals": dist(ev_all), "defs": {}}
    for d in defs:
        if d.get("기각됨"):
            continue          # 기각된 규칙은 식만 기록으로 남긴다 — 제외 목록을 만들지 않는다
        rule, names = compile_rule(d["rule"])
        key = d["key"]
        pairs = [r for r in cross if rule(r)]
        ev = sorted({r["image_id_a"] for r in pairs})
        ev_set = set(ev)
        groups = {man[i]["group_id"] for i in ev}
        ev_by_group = sorted(i for i in ev_all if man[i]["group_id"] in groups)
        sha = {}
        for suffix, lst in (("images", ev), ("groups_expanded", ev_by_group)):
            p = W / f"eval_exclude_{key}_{suffix}.txt"
            p.write_text("\n".join(lst) + "\n", encoding="utf-8", newline="\n")
            sha[suffix] = hashlib.sha256(p.read_bytes()).hexdigest()
        deg = Counter(r["image_id_a"] for r in pairs)
        s = {
            "name": d["name"], "rule": d["rule"],
            "n_pairs": len(pairs), "n_eval_images": len(ev), "pct_eval": round(100 * len(ev) / len(ev_all), 2),
            "n_trainval_images": len({r["image_id_b"] for r in pairs}),
            "n_eval_groups": len(groups), "n_eval_images_if_group_excluded": len(ev_by_group),
            "list_sha256": sha,
            "by": dist(ev),
            "remaining_eval": dist(i for i in ev_all if i not in ev_set),     # 제외 뒤에 남는 평가셋 — 층이 비는지 본다(사후 집계)
            "by_generator": dict(Counter(r["generator"] for r in pairs)),
            "partner_split_client": dict(Counter(f"{man[r['image_id_b']]['split']}/{man[r['image_id_b']]['client']}" for r in pairs)),
            "eval_images_by_partner_split": {
                "train_only_or_both": len({r["image_id_a"] for r in pairs if man[r["image_id_b"]]["split"] == "train"}),
                "val_only": len(ev_set - {r["image_id_a"] for r in pairs if man[r["image_id_b"]]["split"] == "train"}),
            },
            "label_mismatch_posthoc": {
                "defect_vs_normal": sum(man[r["image_id_a"]]["has_defect"] != man[r["image_id_b"]]["has_defect"] for r in pairs),
                "material": sum(man[r["image_id_a"]]["material"] != man[r["image_id_b"]]["material"] for r in pairs),
                "defect_types": sum(man[r["image_id_a"]]["defect_types"] != man[r["image_id_b"]]["defect_types"] for r in pairs),
            },
            "partners_per_eval_image": {"median": sorted(deg.values())[len(deg) // 2] if deg else 0, "max": max(deg.values()) if deg else 0},
        }

        # eval 안쪽 — 누출은 아니다. 묶음 독립 가정(CI)에 걸린다.
        miss = _missing(names, w_eval)
        if miss:
            s["within_eval"] = {"not_applicable": miss}
        else:
            we = [r for r in w_eval if rule(r)]
            xg = [r for r in we if man[r["image_id_a"]]["group_id"] != man[r["image_id_b"]]["group_id"]]
            comp = _components([(man[r["image_id_a"]]["group_id"], man[r["image_id_b"]]["group_id"]) for r in xg])
            rem = [r for r in xg if r["image_id_a"] not in ev_set and r["image_id_b"] not in ev_set]
            comp_rem = _components([(man[r["image_id_a"]]["group_id"], man[r["image_id_b"]]["group_id"]) for r in rem])
            s["within_eval"] = {
                "n_pairs": len(we), "n_images": len({i for r in we for i in (r["image_id_a"], r["image_id_b"])}),
                "n_pairs_across_groups": len(xg), "n_groups_linked": sum(comp), "largest_component_groups": comp[0] if comp else 0,
                "after_exclusion": {"n_pairs_across_groups": len(rem), "n_groups_linked": sum(comp_rem),
                                    "largest_component_groups": comp_rem[0] if comp_rem else 0},
            }

        # train·val 안쪽 — G1∪G2 만 돌렸다(하한).
        miss = _missing(names, w_tv)
        if miss:
            s["within_trainval"] = {"not_applicable": miss}
        else:
            wt = [r for r in w_tv if rule(r)]
            kind = Counter("↔".join(sorted((man[r["image_id_a"]]["split"], man[r["image_id_b"]]["split"]))) for r in wt)
            val_hit = {i for r in wt for i, j in ((r["image_id_a"], r["image_id_b"]), (r["image_id_b"], r["image_id_a"]))
                       if man[i]["split"] == "val" and man[j]["split"] == "train"}
            xc = Counter("↔".join(sorted((man[r["image_id_a"]]["client"], man[r["image_id_b"]]["client"]))) for r in wt
                         if man[r["image_id_a"]]["client"] != man[r["image_id_b"]]["client"])
            s["within_trainval"] = {
                "n_pairs": len(wt), "by_split_pair": dict(kind),
                "n_val_images_with_train_twin": len(val_hit), "pct_val": round(100 * len(val_hit) / n_val, 2),
                "across_clients": dict(xc), "n_pairs_across_clients": sum(xc.values()),
                "label_mismatch_posthoc": {
                    "defect_vs_normal": sum(man[r["image_id_a"]]["has_defect"] != man[r["image_id_b"]]["has_defect"] for r in wt),
                    "material": sum(man[r["image_id_a"]]["material"] != man[r["image_id_b"]]["material"] for r in wt),
                },
            }
        summary["defs"][key] = s
        print(f"[{key}] {d['name']}: 쌍 {s['n_pairs']:,} · eval {s['n_eval_images']:,}장({s['pct_eval']}%) · eval 묶음 {s['n_eval_groups']:,} "
              f"· 묶음째 빼면 {s['n_eval_images_if_group_excluded']:,}장", flush=True)
    (W / "stage5_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
