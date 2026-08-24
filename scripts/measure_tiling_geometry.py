"""타일링 선행 측정 M0·M1·M5·M6 (라벨 JSON 전용).

원천 이미지를 열지 않는다. 라벨 zip 을 **읽기만** 하고 압축도 풀지 않는다(불변조건 1-1).

    uv run python scripts/measure_tiling_geometry.py --labels data/interim/aihub_labels \
        -o docs/dev_log/2026-08-22-데이터확정/41_선행측정_M0M1M5M6_A.md

측정 항목

| # | 무엇을 | 무엇을 결정하는가 |
|---|---|---|
| M0 | RT/AL 결함 4종의 해상도 분해 | 비-R1 결함의 클래스 쏠림. 단일 클래스 10% 이상이면 재검토 |
| M1 | 정상 폴리곤 기하 (밴드 폭·높이·면적비·종횡비·중심) | 밴드 높이 중앙값이 720 이상이면 τ 규칙 가능, 미만이면 차선안 |
| M5 | 결함 폴리곤 중심 (cx/W, cy/H) 분포 | 중앙 사전확률의 크기 |
| M6 | 정상 묶음(연속 id run) 크기 분포 | 오탐 지표의 유효 표본 수 |

**M5 를 임계 유도에 쓰지 않는다.** 결함·정상 통계를 대비해 판별 임계를 뽑으면 라벨로
전처리를 적합하는 것이고 규약 1-4 위반이다. 전처리 상수는 라벨 비의존 기하에서만 나온다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

TILE_W, TILE_H = 1280, 720
PANORAMA_W = 2500          # 판정문서 1-1 의 "대형(폭>2500)" 정의


@dataclass
class ImageRec:
    image_id: int
    modality: str
    material: str
    width: int
    height: int
    is_normal: bool
    cases: tuple[str, ...]
    polys: list[tuple[list[int], list[int]]] = field(default_factory=list)
    #: polys 와 같은 순서의 결함 종류. 한 이미지에 두 종류가 섞이는 경우가 있어
    #: 이미지 대표값으로 뭉뚱그리면 안 된다 (실측 180장 이상).
    poly_cases: list[str] = field(default_factory=list)


def read_labels(label_root: Path) -> list[ImageRec]:
    """RT 라벨 zip 을 전수 읽는다. 파싱 실패는 세어서 보고한다."""
    recs: list[ImageRec] = []
    failures = 0
    for zp in sorted(label_root.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.lower().endswith(".json"):
                    continue
                try:
                    o = json.loads(z.read(name).decode("utf-8-sig"))
                    info, img = o["info"], o["image_data"]
                    anns = o.get("annotations") or []
                    cases = tuple(sorted({a.get("case", "") for a in anns if a.get("case")}))
                    is_normal = all(a.get("class") != "defect" for a in anns) or not anns
                    rec = ImageRec(
                        image_id=int(info["id"]),
                        modality=str(info["type"]),
                        material=str(info["material"]),
                        width=int(img["width"]),
                        height=int(img["height"]),
                        is_normal=is_normal,
                        cases=cases,
                    )
                    for a in anns:
                        c = a.get("coordinate") or {}
                        xs, ys = c.get("x") or [], c.get("y") or []
                        if len(xs) >= 3 and len(xs) == len(ys):
                            rec.polys.append((xs, ys))
                            rec.poly_cases.append(str(a.get("case", "")))
                    recs.append(rec)
                except (KeyError, ValueError, TypeError, UnicodeDecodeError):
                    failures += 1
    if failures:
        print(f"  ! 파싱 실패 {failures}건")
    return recs


def _q(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = max(0, min(len(s) - 1, round(p * (len(s) - 1))))
    return s[i]


def _stat_line(name: str, vals: list[float]) -> str:
    if not vals:
        return f"| {name} | . | . | . | . | . | 0 |"
    return (
        f"| {name} | {min(vals):.1f} | {_q(vals, 0.25):.1f} | "
        f"{statistics.median(vals):.1f} | {_q(vals, 0.75):.1f} | {max(vals):.1f} | {len(vals)} |"
    )


def m0(recs: list[ImageRec], out: list[str]) -> None:
    out.append("## M0. RT/AL 결함의 해상도 분해\n")
    al_def = [r for r in recs if r.material == "AL" and not r.is_normal]
    st_def = [r for r in recs if r.material == "ST" and not r.is_normal]
    out.append("| 재질 | 결함 총수 | 1280×720 | 비-R1 | 비-R1 비율 |")
    out.append("|---|---|---|---|---|")
    for mat, group in (("ST", st_def), ("AL", al_def)):
        r1 = sum(1 for r in group if (r.width, r.height) == (TILE_W, TILE_H))
        non = len(group) - r1
        pct = (non / len(group) * 100) if group else 0.0
        out.append(f"| RT/{mat} | {len(group):,} | {r1:,} | {non:,} | {pct:.3f}% |")
    out.append("")

    non_r1 = [r for r in recs if not r.is_normal and (r.width, r.height) != (TILE_W, TILE_H)]
    out.append(f"**비-R1 결함 {len(non_r1)}장의 클래스 분해**\n")
    if not non_r1:
        out.append("해당 없음.\n")
        return
    cls = Counter(c for r in non_r1 for c in (r.cases or ("(무표기)",)))
    total_by_cls = Counter(c for r in recs if not r.is_normal for c in (r.cases or ()))
    out.append("| 클래스 | 비-R1 장수 | 해당 클래스 총수 | 비율 | 10% 초과 |")
    out.append("|---|---|---|---|---|")
    flagged = []
    for c, n in cls.most_common():
        tot = total_by_cls.get(c, 0)
        ratio = (n / tot * 100) if tot else 0.0
        hit = "**예 (재검토)**" if ratio >= 10.0 else "아니오"
        if ratio >= 10.0:
            flagged.append(c)
        out.append(f"| {c} | {n} | {tot:,} | {ratio:.3f}% | {hit} |")
    out.append("")
    out.append(
        f"**판정:** {'재검토 필요: ' + ', '.join(flagged) if flagged else '쏠림 없음. 폐기 진행 가능.'}\n"
    )
    res = Counter((r.width, r.height) for r in non_r1)
    out.append("비-R1 결함의 해상도 (상위 10):\n")
    out.append("| 해상도 | 장수 |")
    out.append("|---|---|")
    for (w, h), n in res.most_common(10):
        out.append(f"| {w}×{h} | {n} |")
    out.append("")


def m1(recs: list[ImageRec], out: list[str]) -> None:
    out.append("## M1. 정상 폴리곤 기하 (**차선안 분기점**)\n")
    normals = [r for r in recs if r.is_normal and r.polys]
    pano = [r for r in normals if r.width > PANORAMA_W]
    out.append(
        f"정상 이미지 {sum(1 for r in recs if r.is_normal):,}장 중 폴리곤 보유 {len(normals):,}장, "
        f"그중 파노라마(폭>{PANORAMA_W}) {len(pano):,}장\n"
    )

    for label, group in (("정상 전체", normals), ("정상 파노라마만", pano)):
        widths, heights, area_ratios, aspects, cys = [], [], [], [], []
        for r in group:
            for xs, ys in r.polys:
                w, h = max(xs) - min(xs), max(ys) - min(ys)
                if w <= 0 or h <= 0:
                    continue
                widths.append(w)
                heights.append(h)
                area_ratios.append(w * h / (r.width * r.height) * 100)
                aspects.append(w / h)
                cys.append((min(ys) + max(ys)) / 2 / r.height)
        out.append(f"**{label}** (폴리곤 {len(heights):,}개)\n")
        out.append("| 항목 | min | q25 | **중앙값** | q75 | max | n |")
        out.append("|---|---|---|---|---|---|---|")
        out.append(_stat_line("bbox 폭(px)", widths))
        out.append(_stat_line("**bbox 높이(px)**", heights))
        out.append(_stat_line("면적비(%)", area_ratios))
        out.append(_stat_line("종횡비 w/h", aspects))
        out.append(_stat_line("세로 중심 cy/H", cys))
        out.append("")
        if label == "정상 파노라마만" and heights:
            med = statistics.median(heights)
            ge = sum(1 for h in heights if h >= TILE_H) / len(heights) * 100
            out.append(
                f"> **분기 판정. 밴드 높이 중앙값 {med:.1f}px** "
                f"(720 이상 비율 {ge:.1f}%)\n>\n"
            )
            if med >= TILE_H:
                out.append(
                    "> 중앙값이 720 이상이다. **채택안(1280×720 k=1 타일링)으로 간다.**\n"
                    "> 타일 높이 720 이 밴드를 세로로 덮으므로 τ 규칙이 성립할 수 있다.\n"
                )
            else:
                out.append(
                    "> 중앙값이 **720 미만**이다. 차선안(밴드 규격 타일링)으로 분기한다.\n"
                    "> 타일 규격을 밴드 높이 분위수 기반 단일 규격으로 낮추고 결함 크롭도\n"
                    "> 같은 규격으로 중앙 크롭한다. 규격 확정 전 재검토가 필요하다.\n"
                )
            out.append("")


def m5(recs: list[ImageRec], out: list[str]) -> None:
    out.append("## M5. 결함 폴리곤 중심 분포 (중앙 사전확률)\n")
    out.append("> 이 값을 판별 임계 유도에 쓰지 않는다 (규약 1-4). 중앙 사전확률의 크기만 잰다.\n")
    defects = [r for r in recs if not r.is_normal and r.polys]
    for mat in ("ST", "AL"):
        group = [r for r in defects if r.material == mat]
        cxs, cys = [], []
        for r in group:
            for xs, ys in r.polys:
                cxs.append((min(xs) + max(xs)) / 2 / r.width)
                cys.append((min(ys) + max(ys)) / 2 / r.height)
        out.append(f"**RT/{mat}** (결함 폴리곤 {len(cxs):,}개)\n")
        out.append("| 항목 | min | q25 | 중앙값 | q75 | max | n |")
        out.append("|---|---|---|---|---|---|---|")
        out.append(_stat_line("cx/W", cxs))
        out.append(_stat_line("cy/H", cys))
        if cxs:
            central = sum(1 for x, y in zip(cxs, cys) if 0.25 <= x <= 0.75 and 0.25 <= y <= 0.75)
            out.append(f"\n중앙 50% 사각형 안: {central:,}/{len(cxs):,} = {central/len(cxs)*100:.1f}% "
                       f"(균일분포 기대 25.0%)\n")
        out.append("")


def m6(recs: list[ImageRec], out: list[str]) -> None:
    out.append("## M6. 정상 묶음(연속 id run) 크기 분포\n")
    out.append("> 오탐 지표의 유효 표본은 이미지 수가 아니라 묶음 수다.\n")
    buckets: dict[str, list[ImageRec]] = defaultdict(list)
    for r in recs:
        if r.is_normal:
            kind = "크롭 출신(1280×720)" if (r.width, r.height) == (TILE_W, TILE_H) else "타일 출신(그 외)"
            buckets[f"RT/{r.material} · {kind}"].append(r)

    out.append("| 구역 | 이미지 | 묶음(run) | 묶음 크기 중앙값 | 최대 | 이미지/묶음 |")
    out.append("|---|---|---|---|---|---|")
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda r: r.image_id)
        runs, cur = [], 1
        for a, b in itertools.pairwise(group):
            if b.image_id == a.image_id + 1:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        out.append(
            f"| {key} | {len(group):,} | {len(runs):,} | {statistics.median(runs):.0f} | "
            f"{max(runs)} | {len(group)/len(runs):.2f} |"
        )
    out.append("")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path("data/interim/aihub_labels"))
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"라벨 읽는 중: {args.labels}")
    recs = read_labels(args.labels)
    rt = [r for r in recs if r.modality == "RT"]
    print(f"  레코드 {len(recs):,} / RT {len(rt):,}")
    out: list[str] = []
    out.append("# 선행 측정 M0·M1·M5·M6: 타일링 규격 결정 재료\n")
    out.append("라벨 JSON 전수 집계. 원천 이미지는 열지 않았다.\n")
    out.append(f"입력 라벨 레코드 **{len(recs):,}건** 중 RT **{len(rt):,}건**을 집계했다.\n")
    n_norm = sum(1 for r in rt if r.is_normal)
    out.append(
        f"RT 내역: 결함 {len(rt)-n_norm:,} / 정상 {n_norm:,} · "
        f"ST {sum(1 for r in rt if r.material=='ST'):,} / AL {sum(1 for r in rt if r.material=='AL'):,}\n"
    )
    out.append("---\n")
    m0(rt, out)
    out.append("---\n")
    m1(rt, out)
    out.append("---\n")
    m5(rt, out)
    out.append("---\n")
    m6(rt, out)

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
