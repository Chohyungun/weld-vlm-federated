"""근사 중복 14단계 — **k 표.** k 마다 남는 잔여 누출과 층화 가능 여부를 한 표로 낸다.

k 가 무엇인가. 지금 묶음(`group_id`)은 sha256·연속 id·해시 일치로 만든 것이고, 근사 중복 간선은
그 묶음들 사이를 잇는다. **간선 하나만으로 묶음을 합치면 사슬이 생긴다** — 09-21 실측에서 묶음을
건너는 쌍의 96.6%가 간선 하나로 이어져 있었고 최대 성분이 36,816장(59.1%)까지 자랐다.
그래서 **두 묶음 사이에 간선이 k 개 이상일 때만 합친다.** k 를 임의로 고르지 않고 표를 낸 뒤 규칙대로 고른다.

**고르는 규칙 (2026-09-25 결정표 개정으로 바뀌었다).**

- **가장 작은 k 가 잔여 누출 최소라는 가정은 서지 않는다.** 후보 k 마다 **실제로 train–eval 경계를 넘는
  간선**을 재서 함께 적고, 성분을 건너는 간선은 그 **상한**으로만 둔다. 둘을 섞지 않는다.
- **층화 가능 여부를 최대 성분만으로 판정하지 않는다.** 성분별 층 개수와 전체 배정 제약으로 본다.
- **탐욕법이 배정을 찾지 못한 것과 수학적으로 불가능한 것은 다르다.** 못 찾았을 때 쓰는 문장은
  "이 데이터셋에 가능한 분할이 없다" 가 아니라 **"시험한 규칙·k 후보군과 제약 안에서 배정을 찾지 못했다"** 다.
- **현행 분할의 누출도 같은 기준으로 재서 나란히 놓는다.** 되는지만이 아니라 **할 값이 있는지**가 판정이다.
- 시한이 지나는 것이 현행 전량을 주 분석으로 **승격시키지 않는다.**

### ⚠ 위 규칙은 **아직 구현되지 않았다. 현 구현은 구판 선택 로직이다.**

이 독스트링만 새 정책으로 고쳤고 **실행 로직은 바꾸지 않았다**(2026-09-25, 실행 보류 결정).
새 규칙을 이미 구현한 것처럼 읽지 마라. **확인된 구판 자리는 아래 둘이다** — `stratify()` 는 §아래 문단 참조.

| 자리 | 지금 하는 일 | 새 정책에서 해야 할 일 |
|---|---|---|
| `table()` 의 `out["고르는_규칙"]` 문자열 | "층화가 가능해지는 **가장 작은 k**. 없으면 재분할 갈래는 닫힌다" 를 산출에 그대로 적는다 | 문구를 "시험한 규칙·k 후보군과 제약 안에서 배정을 찾지 못했다" 로 바꾸고, k 는 실측 누출로 고른다고 적어야 한다 |
| `table()` 의 `out["고른_k"] = min(ok) if ok else None` | **층화가 서는 후보 중 가장 작은 k** 를 그대로 고른다 | 후보마다 `잔여_누출(분할_경계를_넘는_간선)` 을 비교해 골라야 한다. 최소 k 가 누출 최소라는 가정은 서지 않는다 |
**확인된 차이는 위 둘뿐이다.** `stratify()` 는 새 정책의 요소를 **이미 일부 갖고 있다** —
`by_comp` 로 성분을 묶고, 성분마다 `Counter(stratum(i) …)` 로 **층 개수 벡터**를 만들며,
`want`(층별 목표)와 편차로 전체 배정 제약을 본다. 반환물의 `읽는_법` 에도
**"탐욕법으로 불가능하다고 나와도 최적해가 없다는 증명은 아니다"** 가 이미 적혀 있다.
그러니 성분별 벡터와 실패 해석을 **전부 새로 구현해야 하는 것처럼 묶어 읽지 마라.**
다만 이것이 새 정책 전체를 충족한다고 **검증한 것은 아니다** — 남은 차이는 아직 대조하지 않았다.

그래서 **지금 이 스크립트를 돌려 나온 `고른_k` 는 구판 기준의 값**이다. 새 정책의 답이 아니다.
고치는 것은 자원이 풀린 뒤 채점이 끝나고 나서 한다 — 지금 고치면 시험 없이 바꾸는 것이 된다.

층화 기준(얼린 값): **층별 허용 오차 ±2%p · 전체 eval 비율 18~22%**.

**규칙을 적용할 수 없는 쌍을 간선 없음으로 세지 않는다.** 열이 빠진 쌍은 따로 세어 표에 적는다 —
합치면 성분이 작아 보이고 층화가 실제보다 쉬워 보인다.

민감도 v2: **큰 이동 후보를 전부 진짜 간선으로 가정**하고 같은 표를 다시 낸다.
판독에서 큰 이동 18쌍이 **전부 다름**이었으므로 이 가정은 매우 보수적인 상한이다. 그 사실을 표와 함께 적는다.

사용: python nd_stage14_ktable.py [규칙키] --구판-선택기-임을-인정함      기본 규칙키 X2_v1
      **인정 인자가 없으면 산출물을 쓰지 않고 까닭만 적고 끝낸다**(종료 코드 2).
      선택 로직이 철회된 구판이라 산출물이 현행 결과로 읽히는 것을 막는다.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nd_stage5_lists import compile_rule  # noqa: E402

csv.field_size_limit(10_000_000)
KS = [1, 2, 3, 4, 5]
#: 실행 보류를 푸는 인자. **길고 읽히는 이름으로 둔다** — 짧은 깃발은 뜻을 모르고도 붙는다.
ACK = "--구판-선택기-임을-인정함"
EVAL_TARGET = 0.20
EVAL_BAND = (0.18, 0.22)      # 얼린 값
STRATUM_TOL = 0.02            # 층별 허용 오차 ±2%p

#: 쌍 묶음마다 (feat3 이름, 나머지 단계의 이름 꼬리)
FAMILIES = [
    ("feat3_cross_eval_vs_trainval.csv", "cross_eval_vs_trainval"),
    ("feat3_rest_cross_eval_vs_trainval.csv", "rest_cross_eval_vs_trainval"),
    ("feat3_new_cross.csv", "new_cross"),
    ("feat3_within_eval.csv", "within_eval"),
    ("feat3d_within_trainval.csv", "within_trainval"),
]
STAGES = ["feat5", "feat6", "feat7"]


def _rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


class DSU:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def load_manifest() -> dict:
    with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
        return {r["image_id"]: r for r in csv.DictReader(fh)}


def load_edges(rule: str) -> tuple[list, dict]:
    """(간선 목록, 회계). 규칙을 적용할 수 없는 쌍은 간선으로도 비간선으로도 세지 않는다."""
    fn, names = compile_rule(rule)
    edges, acct = [], {"쌍_전체": 0, "간선": 0, "비간선": 0, "규칙_적용불가": 0, "묶음별": {}}
    for f3, tail in FAMILIES:
        p = W / f3
        if not p.exists():
            acct["묶음별"][tail] = {"상태": "입력_없음"}
            continue
        base = _rows(p)
        for st in STAGES:
            q = W / f"{st}_{tail}.csv"
            if not q.exists():
                continue
            extra = {(r["image_id_a"], r["image_id_b"]): r for r in _rows(q)}
            for r in base:
                r.update({k: v for k, v in extra.get((r["image_id_a"], r["image_id_b"]), {}).items()
                          if k not in ("image_id_a", "image_id_b")})
        n_e = n_x = n_u = 0
        for r in base:
            acct["쌍_전체"] += 1
            if any(k not in r or r[k] in ("", None) for k in names):
                n_u += 1
                continue
            if fn(r):
                edges.append((r["image_id_a"], r["image_id_b"]))
                n_e += 1
            else:
                n_x += 1
        acct["간선"] += n_e
        acct["비간선"] += n_x
        acct["규칙_적용불가"] += n_u
        acct["묶음별"][tail] = {"쌍": len(base), "간선": n_e, "비간선": n_x, "규칙_적용불가": n_u,
                              "없는_단계": [st for st in STAGES if not (W / f"{st}_{tail}.csv").exists()]}
    return edges, acct


def components(man: dict, edges: list, k: int) -> tuple[dict, int]:
    """묶음 사이 간선이 k 개 이상일 때만 합친다. (영상→성분, 성분을 건너는 간선 수)."""
    between = Counter()
    for a, b in edges:
        ga, gb = man[a]["group_id"], man[b]["group_id"]
        if ga != gb:
            between[tuple(sorted((ga, gb)))] += 1
    d = DSU()
    for g in {m["group_id"] for m in man.values()}:
        d.find(g)
    for (ga, gb), n in between.items():
        if n >= k:
            d.union(ga, gb)
    comp = {i: d.find(m["group_id"]) for i, m in man.items()}
    crossing = sum(1 for a, b in edges if comp[a] != comp[b])
    return comp, crossing


def stratify(man: dict, comp: dict) -> dict:
    """성분 단위로 eval 20% 를 떼 본다. 층은 재질×결함유무다.

    탐욕법: 큰 성분부터, 넣었을 때 층별 편차의 최댓값이 작아지는 쪽으로 보낸다.
    최적해를 찾는 것이 목적이 아니라 **가능한지**를 보는 것이다 — 탐욕법으로도 되면 가능하다.
    """
    by_comp = defaultdict(list)
    for i in man:
        by_comp[comp[i]].append(i)

    def stratum(i):
        m = man[i]
        return f"{m['material']}/{m['has_defect']}"

    total = Counter(stratum(i) for i in man)
    want = {s: n * EVAL_TARGET for s, n in total.items()}
    got: Counter = Counter()
    n_eval = 0
    eval_comps = set()
    # 큰 성분부터, **넣는 쪽이 목표에서 덜 벗어나면** eval 로 보낸다. 결정론이고 씨앗이 없다.
    for c in sorted(by_comp, key=lambda c: (-len(by_comp[c]), str(c))):
        add = Counter(stratum(i) for i in by_comp[c])
        cost_skip = sum((got[s] - want[s]) ** 2 for s in total)
        cost_add = sum((got[s] + add[s] - want[s]) ** 2 for s in total)
        if cost_add < cost_skip:
            got += add
            n_eval += len(by_comp[c])
            eval_comps.add(c)
    frac = n_eval / len(man)
    dev = {s: round(got[s] / n - EVAL_TARGET, 4) for s, n in total.items() if n}
    ok_str = all(abs(v) <= STRATUM_TOL for v in dev.values())
    ok_tot = EVAL_BAND[0] <= frac <= EVAL_BAND[1]
    return {"eval_장수": n_eval, "eval_비율": round(frac, 4), "층별_편차": dev,
            "층별_최대편차": round(max((abs(v) for v in dev.values()), default=0), 4),
            "층별_±2%p_충족": ok_str, "전체_18~22%_충족": ok_tot, "층화_가능": ok_str and ok_tot,
            "읽는_법": "탐욕법으로 가능하면 가능하다. 불가능하다고 나와도 최적해가 없다는 증명은 아니다",
            "_eval_성분": eval_comps}


def table(man: dict, edges: list, label: str) -> dict:
    out = {"간선_수": len(edges), "k별": {}}
    for k in KS:
        comp, crossing = components(man, edges, k)
        sizes = Counter(comp.values())
        big = max(sizes.values())
        st = stratify(man, comp)
        ev = st.pop("_eval_성분")
        # **잔여 누출은 실제로 분할 경계를 넘는 간선이다.** 성분을 건너는 간선은 그 상한일 뿐이고,
        # 두 성분이 같은 쪽으로 가면 누출이 아니다. 둘을 함께 적어 상한과 실측을 구분한다.
        leak = sum(1 for a, b in edges if (comp[a] in ev) != (comp[b] in ev))
        out["k별"][k] = {
            "성분_수": len(sizes),
            "최대_성분_장수": big,
            "최대_성분_비율": round(big / len(man), 4),
            "잔여_누출(분할_경계를_넘는_간선)": leak,
            "성분을_건너는_간선(상한)": crossing,
            **st,
        }
    ok = [k for k in KS if out["k별"][k]["층화_가능"]]
    # 철회된 규칙이 **산출물에 현행처럼 실리지 않게** 한다. 받는 쪽은 산출물을 규칙으로 읽는다.
    out["고르는_규칙"] = {
        "상태": "구판 · 철회됨 · 실행 보류",
        "현재_구현이_쓰는_구판_규칙": "층화가 가능해지는 가장 작은 k",
        "왜_철회됐나": "가장 작은 k 가 잔여 누출 최소라는 가정이 서지 않는다. "
                   "후보 k 마다 실제로 분할 경계를 넘는 간선을 재서 골라야 한다",
        "새_정책_구현됨": False,
        "이_값을_현행_규칙으로_인용하지_마라": True,
    }
    # `고른_k` 는 **값을 싣지 않는다.** 그 이름의 숫자는 표로 옮겨질 때 경고를 달고 가지 못한다.
    # 구판 로직의 결과는 이름 자체가 경고인 칸에 둔다 — 이름을 지우지 않고는 옮길 수 없다.
    out["고른_k"] = None
    out["고른_k가_비어_있는_까닭"] = "새 k 선택 정책이 구현되지 않았다. 구판 값을 이 이름으로 실으면 현행 선택으로 읽힌다"
    out["구판_로직의_값_현행_선택_아님"] = min(ok) if ok else None
    out["라벨"] = label
    return out


def main() -> int:
    # **실행 보류를 코드로 건다.** 선택 로직이 철회된 구판이므로 명시적으로 그 사실을 인정한
    # 인자 없이는 산출물을 쓰지 않는다. 산출물이 남으면 나중에 누가 현행 결과로 읽는다.
    if ACK not in sys.argv[1:]:
        print("실행 보류 — k 선택 로직이 철회된 구판이고 새 정책은 구현되지 않았다.", flush=True)
        print("산출물을 쓰지 않는다. 구판 값을 보려면 그 사실을 인정하는 인자를 준다:", flush=True)
        print(f"    python {Path(__file__).name} [규칙키] {ACK}", flush=True)
        print("그렇게 얻은 산출물의 `고른_k` 는 비어 있고 구판 값은 따로 표시된다.", flush=True)
        return 2
    args = [a for a in sys.argv[1:] if a != ACK]
    key = args[0] if args else "X2_v1"
    defs = json.loads((HERE / "definitions.json").read_text(encoding="utf-8"))
    rule = next(d["rule"] for d in defs["definitions"] if d["key"] == key)
    man = load_manifest()
    edges, acct = load_edges(rule)

    rep = {
        "규칙": {"키": key, "식": rule,
               "상태": "**보류 표본으로 아직 확인하지 않았다.** 개발 180쌍에서 정해 얼린 규칙이다"},
        "영상_수": len(man), "간선_회계": acct,
        "층화_기준": {"층": "재질×결함유무", "층별_허용": f"±{STRATUM_TOL:.0%}p", "전체": f"{EVAL_BAND[0]:.0%}~{EVAL_BAND[1]:.0%}"},
        "v1": table(man, edges, "v1 — 얼린 규칙의 간선만"),
    }
    # v2 민감도 — 큰 이동 후보를 전부 진짜로 가정한다
    big = W / "shiftcand_cross.csv"
    if big.exists():
        extra = [(r["image_id_a"], r["image_id_b"]) for r in _rows(big)]
        rep["v2_민감도"] = {
            "가정": "큰 이동 후보를 **전부 진짜 간선으로** 가정한다",
            "더한_쌍": len(extra),
            "이_가정이_얼마나_보수적인가": "1차 판독에서 큰 이동 층 18쌍이 **전부 다름**이었다. "
                                "실제 참양성 비율은 이보다 훨씬 낮으므로 이 표는 상한이다. 점추정으로 읽지 않는다",
            **table(man, edges + extra, "v2 — 큰 이동 후보를 전부 진짜로 가정"),
        }
    (W / "stage14_ktable.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
