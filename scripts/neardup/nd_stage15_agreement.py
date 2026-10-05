"""근사 중복 15단계 — **판독자 일치율.** 라벨의 불확실성을 재는 것이지 규칙의 성능이 아니다.

두 판독은 A(1차)와 D(2차 **자동** 판독)다. **사람 검수가 아니다** — 이 수로 사람 판독을 대신하지 않고
`사람 판독 미측정` 은 한계 목록에 그대로 남는다.

**봉인을 먼저 본다.** A 는 넘기기 전에 자기 60쌍 판정의 지문을 남겼다. 그 지문이 지금 다시 계산해도
같아야 한다 — 2차 판독을 보고 자기 판정을 고쳤다면 일치율이 뜻을 잃는다. 어긋나면 값을 내지 않고 멈춘다.

**불확실을 한쪽에 합치지 않는다.** 셋을 따로 낸다.
  ① 전체 일치율 — 어휘 셋 그대로. 불확실도 하나의 답이다
  ② 같음/다름만 — 양쪽 다 확정한 쌍에서만. 규칙 문턱이 서는 자리와 같은 모집단이다
  ③ 불확실의 취급 — 누가 얼마나 불확실로 뒀고, 한쪽만 불확실인 쌍에서 다른 쪽은 무엇이라 했나

사용: python nd_stage15_agreement.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
OUT = ROOT / "outputs" / "neardup_second_read"
VOCAB = ("같음", "다름", "불확실")
COL = "두번째_자동판독_사람검수아님"       # 열 이름이 곧 표시다


def _rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def kappa(a: list, b: list) -> float | None:
    """Cohen 의 kappa. 우연 일치를 뺀 값이다 — 단순 일치율만 보면 층 구성에 속는다."""
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def main() -> int:
    seal = json.loads((W / "sr60_seal.json").read_text(encoding="utf-8"))
    mine = {r["pair_id"]: r["verdict"] for r in seal["mine"]}
    now = hashlib.sha256(json.dumps(seal["mine"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if now != seal["seal"]:
        raise SystemExit(f"봉인이 깨졌다 — A 의 판정이 바뀌었다. 일치율을 내지 않는다\n  {seal['seal']}\n  {now}")

    src = OUT / "판독할_쌍_D기입.csv"
    if not src.exists():
        raise SystemExit(f"D 의 기입본이 없다: {src}")
    theirs = {}
    for r in _rows(src):
        v = (r.get(COL) or "").strip()
        if v:
            theirs[r["pair_id"]] = v
    bad = sorted(v for v in set(theirs.values()) if v not in VOCAB)
    if bad:
        raise SystemExit(f"기준 밖 어휘가 있다: {bad}")

    ids = sorted(set(mine) & set(theirs))
    a = [mine[i] for i in ids]
    b = [theirs[i] for i in ids]
    rep = {
        "무엇을_재는가": "라벨의 불확실성이다. **규칙의 성능이 아니다.**",
        "사람_검수인가": "아니다. 2차 **자동** 판독이며 `사람 판독 미측정` 은 한계에 남는다",
        "봉인": {"A_지문": seal["seal"], "지금_다시_계산": now, "그대로인가": True},
        "쌍_수": len(ids),
        "A_분포": dict(Counter(a)), "D_분포": dict(Counter(b)),
    }
    # ① 전체 — 어휘 셋 그대로
    same = sum(x == y for x, y in zip(a, b))
    rep["①_전체_일치"] = {"일치": same, "쌍": len(ids), "일치율": round(same / len(ids), 4),
                      "kappa": kappa(a, b),
                      "읽는_법": "불확실도 하나의 답으로 센다. 합치면 일치율이 부풀거나 꺼진다"}
    # ② 같음/다름만 — 양쪽 다 확정한 쌍
    both = [(x, y) for x, y in zip(a, b) if x != "불확실" and y != "불확실"]
    if both:
        s2 = sum(x == y for x, y in both)
        rep["②_같음다름만"] = {"쌍": len(both), "일치": s2, "일치율": round(s2 / len(both), 4),
                          "kappa": kappa([x for x, _ in both], [y for _, y in both]),
                          "읽는_법": "규칙 문턱이 서는 자리와 같은 모집단이다. 확정한 쌍에서 둘이 얼마나 같은가"}
    # ③ 불확실의 취급 — 합치지 않고 그대로 드러낸다
    only_a = [(i, mine[i], theirs[i]) for i in ids if mine[i] == "불확실" and theirs[i] != "불확실"]
    only_b = [(i, mine[i], theirs[i]) for i in ids if theirs[i] == "불확실" and mine[i] != "불확실"]
    rep["③_불확실의_취급"] = {
        "A만_불확실": {"수": len(only_a), "그때_D의_답": dict(Counter(y for _i, _x, y in only_a))},
        "D만_불확실": {"수": len(only_b), "그때_A의_답": dict(Counter(x for _i, x, _y in only_b))},
        "둘_다_불확실": sum(1 for i in ids if mine[i] == theirs[i] == "불확실"),
        "읽는_법": "한쪽만 불확실인 쌍을 불일치로도 일치로도 세지 않는다. 그 수 자체가 결과다",
    }
    # 정면 충돌 — 한쪽 같음, 다른 쪽 다름. 라벨이 흔들리는 자리다
    clash = [i for i in ids if {mine[i], theirs[i]} == {"같음", "다름"}]
    rep["정면_충돌"] = {"수": len(clash), "쌍": clash,
                    "읽는_법": "이것이 라벨 불확실성의 핵심 수다. 문턱을 다시 볼지의 기준이 된다"}
    rep["교차표"] = {f"A={x}": {f"D={y}": sum(1 for i in ids if mine[i] == x and theirs[i] == y)
                             for y in VOCAB} for x in VOCAB}
    (W / "stage15_agreement.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
