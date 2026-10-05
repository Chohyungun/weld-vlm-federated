"""근사 중복 11단계 — **판독 표본 300쌍을 뽑는다.** 층을 나누고 개발·보류를 가른다.

설계는 `docs/dev_log/2026-09-24-후속정리/08_규칙재설계_설계안_A.md` 와 총괄 승인을 따른다.
층 목표는 확정 양성 100 · 적대적 음성 후보 120 · 경계 50 · 큰 이동 30 이고 개발 180 / 보류 120 이다.

**적대적 음성과 큰 이동은 옛 규칙이 띄우지 않은 구간에서만 뽑는다.** 10단계가 잰 대로
지금 가진 라벨 310쌍 가운데 옛 규칙 밖은 46쌍뿐이라, 그 표에는 옛 규칙이 안 띄운 계열의 오판이 거의 없다.
무작위 대조로는 그 구간이 잡히지 않는다 — 모집단의 대부분이 명백한 음성이기 때문이다. 구간을 지목해 뽑는다.

**이미 판정한 310쌍은 뽑지 않는다.** 보류가 한 번도 보지 않은 쌍이어야 하기 때문이다.

층은 겹칠 수 있으므로 **우선순위로 하나씩 배정**한다. 확정 양성 → 적대적 음성 → 큰 이동 → 경계 순이다.
쌍마다 옛 규칙 안/밖을 함께 적는다 — 보류를 채점할 때 안과 밖을 따로 내기 위해서다.
**층마다 모집단 크기를 함께 적는다.** 나중에 모집단 비율로 되돌리려면 그 수가 있어야 한다.

무작위는 고정 씨앗을 쓰고 씨앗을 산출에 적는다. 같은 씨앗이면 같은 표본이 나온다.

사용: python nd_stage11_sample.py [씨앗]   → sample_v1.csv · stage11_sample.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nd_stage5_lists import compile_rule  # noqa: E402
from nd_stage9_devfit import load_features, _rows  # noqa: E402
from nd_stage10_coverage import old_cover  # noqa: E402

DEFS = json.loads((Path(__file__).resolve().parent / "definitions.json").read_text(encoding="utf-8"))
RULE = {d["key"]: d["rule"] for d in DEFS["definitions"]}
SEED = 20260925
csv.field_size_limit(10_000_000)

#: 층 — (이름, 목표 수, 옛 규칙 밖만 쓰는가, 조건식). 위에서부터 배정한다.
STRATA = [
    ("확정양성", 100, False, "rd_sharp >= 0.50 and blk_frac >= 0.90 and blk_med >= 0.40"),
    ("적대적음성", 120, True, "(txt_gap >= 0.30 and txt_energy >= 1.0) or (blk_frac >= 0.30 and blk_frac < 0.50)"),
    ("큰이동", 30, True, "max(abs(rd_dx), abs(rd_dy)) > 48"),
    ("경계", 50, False, RULE["R"]),
]
DEV = 180                       # 나머지 120 이 보류다


def num(f: dict, k: str):
    try:
        return float(f[k])
    except (TypeError, ValueError, KeyError):
        return None


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
    feat = load_features()
    already = {(r["image_id_a"], r["image_id_b"]) for r in _rows(W / "labels_prior.csv")}

    # 쌍마다 옛 규칙 안/밖을 먼저 정한다.
    pool = []
    for key, f in feat.items():
        if f.get("_src") not in ("cross", "rest", "new"):
            continue
        hit, unk = old_cover(f)
        pool.append((key, f, "안" if (hit and not unk) else "밖", key in already))

    # 층을 우선순위로 배정한다. 모집단 크기와 이미 본 쌍 수를 같이 센다.
    assigned: dict[tuple, str] = {}
    frames: dict[str, list] = {}
    sizes: dict[str, dict] = {}
    for name, want, outside_only, expr in STRATA:
        fn, names = compile_rule(expr)
        pop = fresh = 0
        cands = []
        for key, f, frame, seen in pool:
            if outside_only and frame != "밖":
                continue
            if any(k not in f or f[k] in ("", None) for k in names) or not fn(f):
                continue
            pop += 1
            if seen or key in assigned:
                continue
            fresh += 1
            cands.append((key, f, frame))
        frames[name] = cands
        # 모집단 = 조건에 드는 전부. 뽑을 수 있는 쌍 = 그중 아직 판정하지 않고 앞 층에도 안 간 것.
        sizes[name] = {"모집단": pop, "이미_봤거나_앞_층에_감": pop - fresh, "뽑을_수_있는_쌍": fresh, "목표": want}
        rnd = random.Random(f"{seed}:{name}")
        rnd.shuffle(cands)
        for key, _f, _fr in cands[:want]:
            assigned[key] = name

    # 개발·보류를 층 안에서 비율대로 가른다. 층 구성이 양쪽에 같이 실리게.
    rows = []
    for name, want, _o, _e in STRATA:
        picked = [(k, f, fr) for k, f, fr in frames[name] if assigned.get(k) == name]
        rnd = random.Random(f"{seed}:split:{name}")
        rnd.shuffle(picked)
        n_dev = round(len(picked) * DEV / 300)
        for i, (key, f, frame) in enumerate(picked):
            rows.append({
                "image_id_a": key[0], "image_id_b": key[1],
                "층": name, "구간": frame, "쓰임": "개발" if i < n_dev else "보류",
                "rd_dx": f.get("rd_dx", ""), "rd_dy": f.get("rd_dy", ""),
                "rd_sharp": f.get("rd_sharp", ""), "blk_frac": f.get("blk_frac", ""),
                "blk_med": f.get("blk_med", ""), "hi_min": f.get("hi_min", ""),
                "txt_gap": f.get("txt_gap", ""), "txt_energy": f.get("txt_energy", ""),
                "body_corr": f.get("body_corr", ""),
                "판정": "", "메모": "",          # 판독이 채운다
            })
    out = W / "sample_v1.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rep = {
        "씨앗": seed, "설계_목표": {n: t for n, t, _o, _e in STRATA}, "뽑힌_수": len(rows),
        "층별": {n: {**sizes[n], "뽑힘": sum(r["층"] == n for r in rows),
                     "개발": sum(r["층"] == n and r["쓰임"] == "개발" for r in rows),
                     "보류": sum(r["층"] == n and r["쓰임"] == "보류" for r in rows),
                     "옛_규칙_밖": sum(r["층"] == n and r["구간"] == "밖" for r in rows)}
                for n, _t, _o, _e in STRATA},
        "쓰임별": {u: sum(r["쓰임"] == u for r in rows) for u in ("개발", "보류")},
        "보류의_구간별": {fr: sum(r["쓰임"] == "보류" and r["구간"] == fr for r in rows) for fr in ("안", "밖")},
        "이미_판정한_쌍_제외": len(already),
        "목록_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "주의": [
            "이 표본은 이미 뽑아 둔 후보 쌍 안에서의 것이다. 어떤 생성기도 못 본 쌍은 여기에도 없다.",
            "층별 모집단 크기를 같이 적었다 — 비율을 모집단으로 되돌릴 때 쓴다. 층 비율 그대로 읽으면 안 된다.",
            "보류는 한 번만 본다. 보고 나서 문턱을 고치면 보류가 아니다.",
        ],
    }
    (W / "stage11_sample.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
