"""M1b. 차선안(밴드 규격 타일링)의 규격 후보 산정.

M1 에서 밴드 높이 중앙값이 720 미만으로 나와 채택안(1280×720 k=1)이 분기했다.
차선안은 타일 규격을 밴드 높이 분위수 기반 `(w_t × h_t)` 로 낮추고 **결함 크롭도 같은
규격으로 중앙 크롭**한다. 그 규격을 고르려면 두 값을 같은 표에서 봐야 한다.

| 값 | 뜻 |
|---|---|
| 밴드 포함률 | 정상 파노라마 밴드가 높이 `h_t` 타일 안에 **통째로** 들어가는 비율 |
| 결함 절단율 | 결함 크롭(1280×720)을 높이 `h_t` 로 중앙 크롭할 때 결함 폴리곤이 잘리는 비율 |

결함 절단율 상한은 3% 다. 두 값은 반대 방향으로 움직인다. `h_t` 를 키우면
밴드는 더 들어가지만 정상 타일에 미검사 영역이 늘고, 줄이면 결함이 잘린다.

    uv run python scripts/measure_band_spec.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from scripts.measure_tiling_geometry import (
    PANORAMA_W,
    TILE_H,
    TILE_W,
    read_labels,
)

#: 후보 높이. 8의 배수(JPEG 블록 격자 정렬, 판정문서 4-3 align8)만 둔다.
CANDIDATES = (160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 720)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path("data/interim/aihub_labels"))
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    recs = [r for r in read_labels(args.labels) if r.modality == "RT"]
    out: list[str] = []
    out.append("# M1b: 밴드 규격 타일링 규격 후보 (차선안)\n")
    out.append("라벨 JSON 전수 집계. 밴드 높이 중앙값이 720 미만이어서 분기했다.\n")

    # --- 밴드 높이 (정상 파노라마) --------------------------------------------------
    bands = [
        (max(ys) - min(ys), (min(ys) + max(ys)) / 2 / r.height)
        for r in recs
        if r.is_normal and r.width > PANORAMA_W
        for xs, ys in r.polys
        if max(ys) > min(ys)
    ]
    heights = [h for h, _ in bands]
    out.append(f"정상 파노라마 밴드 폴리곤 **{len(heights):,}개**\n")

    # --- 결함 폴리곤 (1280×720 크롭 안) ---------------------------------------------
    dpolys = [
        (min(ys), max(ys))
        for r in recs
        if not r.is_normal and (r.width, r.height) == (TILE_W, TILE_H)
        for xs, ys in r.polys
    ]
    out.append(f"결함 폴리곤(1280×720 크롭 내) **{len(dpolys):,}개**\n")
    out.append("")

    out.append("| h_t | 밴드 포함률 | 결함 절단율 | 결함 절단 수 | 정상 타일 미검사 비율 | 판정 |")
    out.append("|---|---|---|---|---|---|")
    rows = []
    for h_t in CANDIDATES:
        contained = sum(1 for h in heights if h <= h_t) / len(heights) * 100
        lo, hi = (TILE_H - h_t) / 2, (TILE_H + h_t) / 2
        cut = sum(1 for y0, y1 in dpolys if y0 < lo or y1 > hi)
        cut_pct = cut / len(dpolys) * 100
        # 밴드 중앙값이 타일을 채우고 남는 비율 = 미검사(정상 승계 불가) 영역
        uninspected = max(0.0, (1 - statistics.median(heights) / h_t)) * 100
        ok = "통과" if cut_pct <= 3.0 else "초과"
        rows.append((h_t, contained, cut_pct, cut, uninspected, ok))
        out.append(
            f"| **{h_t}** | {contained:.1f}% | **{cut_pct:.2f}%** | {cut:,} | "
            f"{uninspected:.1f}% | {ok} |"
        )
    out.append("")

    ok_rows = [r for r in rows if r[2] <= 3.0]
    if ok_rows:
        best = max(ok_rows, key=lambda r: r[1])          # 절단율 통과 중 밴드 포함률 최대
        out.append(
            f"> **후보 권고: h_t = {best[0]}.** 결함 절단율 {best[2]:.2f}% (상한 3% 이내), "
            f"밴드 포함률 {best[1]:.1f}%.\n>\n"
            f"> 폭은 결함 크롭 폭과 같은 **1280** 을 유지한다(가로는 자르지 않는다). "
            f"규격은 **1280×{best[0]}**.\n"
        )
    else:
        out.append("> **모든 후보가 결함 절단율 3% 를 넘는다.**\n")
    out.append("")

    out.append("## 밴드 높이 분위수\n")
    out.append("| 분위 | 값(px) |")
    out.append("|---|---|")
    s = sorted(heights)
    for p in (0.50, 0.75, 0.90, 0.95, 0.99):
        out.append(f"| q{int(p*100)} | {s[int(p*(len(s)-1))]:.0f} |")
    out.append("")

    out.append("## 밴드 세로 중심: 정렬 기준이 성립하는가\n")
    cys = [c for _, c in bands]
    out.append(
        f"cy/H 중앙값 {statistics.median(cys):.3f} · "
        f"q25 {sorted(cys)[len(cys)//4]:.3f} · q75 {sorted(cys)[3*len(cys)//4]:.3f}\n"
    )
    out.append(
        "`vertical_anchor: band_center` 는 밴드 폴리곤에서 직접 중심을 읽으므로 "
        "이 분포가 넓어도 타일 자체는 밴드에 정렬된다. 분포는 참고값이다.\n"
    )

    text = "\n".join(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"기록: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
