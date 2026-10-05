"""이웃 id 의 닮음(VT 스냅샷 미니스펙의 이웃 표)을 따로 다시 센다 — 원저자 아닌 쪽의 검산.

표 머리의 정의만 따라 짰다. 라벨 JSON 만 읽는다. 독립성의 범위는 검산 보고에 적는다.
장마다의 값은 내지 않고 폴더마다 비율만 낸다. 무작위 짝은 고정 시드로 뽑으므로 다른 구현의 무작위 열과
소수 셋째 자리까지 같기를 기대하지 않는다 — 이웃 짝 수와 이웃 열은 같아야 한다.

**쓰기.** 출력은 `--out` 파일 하나이고, 저장소 `_workspace/` 아래의 **없는 새 경로**만 받는다(실제 경로로 본다 —
`scripts.crosscheck.xc_out_guard`). 입력 폴더와 겹치거나 동결 디렉터리 아래면 아무것도 읽기 전에 멈춘다.

정의:
- 이웃 짝: 같은 (촬영 방식 · 재질 · 폴더) 안에서 `info.id` 가 1 차이인 짝.
- 크기 같음: 두 장의 (width, height) 가 같다. 무작위 = 같은 폴더의 무작위 짝.
- 인스턴스 수 같음: 주석 수가 같다. 무작위 = 같은 폴더 · 같은 크기의 무작위 짝.
- 첫 상자 IoU ≥ 0.5: 각 장의 첫 주석(점 3 개 이상) 다각형의 최소 · 최대 상자의 IoU. 무작위 = 같은 폴더 · 같은 크기.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

FOLDER_RE = re.compile(r"_(RT|VT)(AL|ST)_(결함_\d\. [^.]+|정상)\.zip$")


def first_box(anns):
    for a in anns:
        c = a.get("coordinate") or {}
        xs, ys = c.get("x") or [], c.get("y") or []
        if len(xs) >= 3 and len(xs) == len(ys):
            return (min(xs), min(ys), max(xs), max(ys))
    return None


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="_workspace 아래의 없는 새 .json")
    ap.add_argument("--n-random", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261004)
    a = ap.parse_args()

    from scripts.crosscheck.xc_out_guard import guard_new_output, recheck_inside

    guard_new_output(a.out, repo=Path(__file__).resolve().parents[2], inputs=a.labels)

    groups: dict[str, dict[int, tuple]] = defaultdict(dict)
    for d in a.labels:
        for zp in sorted(d.glob("*.zip"), key=lambda p: p.name):
            m = FOLDER_RE.search(zp.name)
            if not m:
                continue
            key = f"{m.group(1)} {m.group(2)} {m.group(3)}"
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    if not name.lower().endswith(".json"):
                        continue
                    o = json.loads(z.read(name).decode("utf-8-sig"))
                    img = o["image_data"]
                    anns = o.get("annotations") or []
                    groups[key][int(o["info"]["id"])] = (
                        (int(img["width"]), int(img["height"])), len(anns), first_box(anns))

    rng = random.Random(a.seed)
    out = {}
    for key in sorted(groups):
        g = groups[key]
        ids = sorted(g)
        pairs = [(i, i + 1) for i in ids if i + 1 in g]
        n = len(pairs)
        if not n:
            continue
        nb_size = sum(g[x][0] == g[y][0] for x, y in pairs) / n
        same = [(x, y) for x, y in pairs if g[x][0] == g[y][0]]
        nb_inst = sum(g[x][1] == g[y][1] for x, y in same) / len(same) if same else None
        nb_iou = sum(iou(g[x][2], g[y][2]) >= 0.5 for x, y in same) / len(same) if same else None
        rs = [(rng.choice(ids), rng.choice(ids)) for _ in range(a.n_random)]
        rs = [(x, y) for x, y in rs if x != y]
        r_size = sum(g[x][0] == g[y][0] for x, y in rs) / len(rs)
        by_size = defaultdict(list)
        for i in ids:
            by_size[g[i][0]].append(i)
        sizes = [s for s, v in by_size.items() if len(v) > 1]
        weights = [len(by_size[s]) for s in sizes]
        rr = []
        for _ in range(a.n_random):
            s = rng.choices(sizes, weights=weights)[0]
            x, y = rng.sample(by_size[s], 2)
            rr.append((x, y))
        r_inst = sum(g[x][1] == g[y][1] for x, y in rr) / len(rr)
        r_iou = sum(iou(g[x][2], g[y][2]) >= 0.5 for x, y in rr) / len(rr)
        out[key] = {"neighbor_pairs": n, "size_same": [round(nb_size, 3), round(r_size, 3)],
                    "inst_same": [round(nb_inst, 3) if nb_inst is not None else None, round(r_inst, 3)],
                    "first_box_iou50": [round(nb_iou, 3) if nb_iou is not None else None, round(r_iou, 3)]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    recheck_inside(a.out.parent, repo=Path(__file__).resolve().parents[2])
    with a.out.open("x", encoding="utf-8", newline="\n") as fh:   # 배타 생성 — 그사이 생겼으면 덮지 않고 멈춘다
        fh.write(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
