#!/usr/bin/env python
"""창원(AI허브 71761) 라벨 — 연합학습 적합성 · 판정 재료 분석기.

`inspect_aihub.py` 가 "무엇이 들어 있나"를 세었다면 이 스크립트는 두 가지를 판정한다.

  v2  Non-IID 관점: label / feature / quantity skew 와 concept drift 를 수치로 낸다
  v3  판정 재료: 원본 판정 라벨, 결함-정상 교차, 조치 지침, 자유 기술 필드

concept drift 는 축이 없어서 직접 못 잰다. 대신 `info.id` 가 전역 순번이라는 점을
이용해 **id 구간을 시간 대용축으로 삼아** 어노테이션 관행이 변했는지 본다.

사용:
    python scripts/analyze_aihub_fl.py <라벨_루트> -o <출력.txt>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict


def poly_area_px(xs, ys) -> float:
    """신발끈 공식. 단위는 **픽셀제곱이다. mm2 가 아니다.**
    실치수 환산에 필요한 mm/pixel 이 데이터에 없다 (v3 C-2)."""
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(a) / 2.0


def pct(x, n):
    return f"{x*100/n:5.1f}%" if n else "  n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    recs = []           # (id, type, material, kclass, w, h, n_inst, cases, areas, verts, crowd)
    str_fields = defaultdict(set)   # 문자열 필드별 고유값 (자유 기술 필드 탐지)
    str_len = defaultdict(list)

    for dp, _, fns in os.walk(args.root):
        for fn in fns:
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(dp, fn), encoding="utf-8") as fh:
                d = json.load(fh)
            info, im, meta = d["info"], d["image_data"], d["meta"]
            anns = d.get("annotations") or []
            cases, areas, verts = [], [], []
            for a in anns:
                c = a.get("coordinate") or {}
                xs, ys = c.get("x") or [], c.get("y") or []
                cases.append((a.get("class"), a.get("case")))
                areas.append(poly_area_px(xs, ys))
                verts.append(len(xs))
            recs.append((info["id"], info["type"], info["material"], im["information"],
                         im["width"], im["height"], len(anns), cases, areas, verts,
                         meta.get("is_crowd")))
            for k, v in (("image_data.file_name", im.get("file_name")),
                         ("image_data.format", im.get("format")),
                         ("image_data.information", im.get("information")),
                         ("info.type", info.get("type")),
                         ("info.material", info.get("material"))):
                if isinstance(v, str):
                    str_fields[k].add(v)
                    str_len[k].append(len(v))
            for a in anns:
                for k in ("class", "case", "tool"):
                    v = a.get(k)
                    if isinstance(v, str):
                        str_fields[f"annotations[].{k}"].add(v)
                        str_len[f"annotations[].{k}"].append(len(v))

    L: list[str] = []
    P = L.append
    N = len(recs)
    P(f"스캔 라벨 수: {N:,}")
    P("")

    # ------------------------------------------------------------------ v2 A-2
    P("=" * 74)
    P("[v2 A-2] 이질성 유형 — 재질 축(ST/AL) 기준")
    P("=" * 74)

    rt = [r for r in recs if r[1] == "RT"]
    by_m = defaultdict(list)
    for r in rt:
        by_m[r[2]].append(r)

    P(f"RT 전체 {len(rt):,}  (AL {len(by_m['AL']):,} / ST {len(by_m['ST']):,})")
    P("")
    P("--- (1) label skew : 재질별 클래스 비율 ---")
    classes = sorted({r[3] for r in rt})
    dist = {}
    for m in ("AL", "ST"):
        c = Counter(r[3] for r in by_m[m])
        n = len(by_m[m])
        dist[m] = {k: c[k] / n for k in classes}
        P(f"  {m} (n={n:,}): " + "  ".join(f"{k} {c[k]:,}({pct(c[k],n).strip()})" for k in classes))
    P("")
    P("  클래스별 비율 배수 (AL/ST):")
    for k in classes:
        a, s = dist["AL"][k], dist["ST"][k]
        P(f"    {k:<8} AL {a*100:5.1f}%  ST {s*100:5.1f}%   배수 {a/s if s else float('inf'):5.2f}x")
    # 총변이거리
    tvd = 0.5 * sum(abs(dist["AL"][k] - dist["ST"][k]) for k in classes)
    P(f"  총변이거리(TVD) AL vs ST = {tvd:.4f}   (0=동일, 1=완전분리)")
    P("")

    P("--- (2) quantity skew ---")
    P(f"  AL {len(by_m['AL']):,} : ST {len(by_m['ST']):,}  =  1 : {len(by_m['ST'])/len(by_m['AL']):.1f}")
    P("")

    P("--- (3) feature skew : 재질별 이미지 규격 ---")
    for m in ("AL", "ST"):
        dims = Counter(f"{r[4]}x{r[5]}" for r in by_m[m])
        areas = [r[4] * r[5] for r in by_m[m]]
        P(f"  {m}: 해상도 종수 {len(dims):,}  최빈 {dims.most_common(1)[0]}")
        P(f"      화소수 평균 {sum(areas)/len(areas)/1e6:.2f}MP  최소 {min(areas)/1e6:.2f}  최대 {max(areas)/1e6:.2f}")
    P("")
    P("  결함 클래스만(정상 제외) 비교 — 규격 지름길 배제 후:")
    for m in ("AL", "ST"):
        d = [r for r in by_m[m] if r[3] != "정상"]
        areas = [r[4] * r[5] for r in d]
        dims = Counter(f"{r[4]}x{r[5]}" for r in d)
        P(f"    {m}: n={len(d):,}  해상도 종수 {len(dims)}  화소수 평균 {sum(areas)/len(areas)/1e6:.3f}MP")
    P("")

    P("--- (4) 어노테이션 관행 : 재질별 폴리곤 특성 ---")
    for m in ("AL", "ST"):
        d = [r for r in by_m[m] if r[3] != "정상"]
        allv = [v for r in d for v in r[9]]
        alla = [a for r in d for a in r[8] if a > 0]
        ninst = [r[6] for r in d]
        crowd = Counter(r[10] for r in d)
        P(f"  {m}: 인스턴스/이미지 평균 {sum(ninst)/len(ninst):.2f}   꼭짓점 평균 {sum(allv)/len(allv):.1f}")
        P(f"      폴리곤 면적(px^2) 중앙값 {sorted(alla)[len(alla)//2]:,.0f}  평균 {sum(alla)/len(alla):,.0f}")
        P(f"      is_crowd=1 비율 {pct(crowd.get(1,0), len(d))}")
    P("")

    # ---------------------------------------------------------- v2 A-2 drift
    P("=" * 74)
    P("[v2 A-2] concept drift 대용 검사 — info.id 구간을 시간축으로 본다")
    P("=" * 74)
    P("회사·촬영일 축이 없어 직접 측정 불가. id 가 전역 순번이므로 구간별로")
    P("어노테이션 관행이 변했는지 본다. 변했다면 시기별 기준 차이의 방증이 된다.")
    P("")
    rts = sorted(rt, key=lambda r: r[0])
    B = 5
    sz = len(rts) // B
    for b in range(B):
        seg = rts[b*sz : (b+1)*sz if b < B-1 else len(rts)]
        d = [r for r in seg if r[3] != "정상"]
        if not d:
            P(f"  구간{b+1}: 결함 이미지 없음")
            continue
        allv = [v for r in d for v in r[9]]
        alla = [a for r in d for a in r[8] if a > 0]
        ninst = [r[6] for r in d]
        cc = Counter(r[3] for r in seg)
        mats = Counter(r[2] for r in seg)
        P(f"  구간{b+1} id {seg[0][0]}~{seg[-1][0]}  n={len(seg):,} (결함 {len(d):,})")
        P(f"      재질 {dict(mats)}")
        P(f"      클래스 {dict(cc)}")
        P(f"      꼭짓점 평균 {sum(allv)/len(allv):5.1f}   면적중앙값(px^2) {sorted(alla)[len(alla)//2]:>9,.0f}   인스턴스/장 {sum(ninst)/len(ninst):.2f}")
    P("")

    # ------------------------------------------------------------------ v3
    P("=" * 74)
    P("[v3 C-3] 원본 판정 라벨 · 결함/정상 교차")
    P("=" * 74)
    P("판정(합격/불합격) 필드는 스키마에 없다. 대신 폴더 클래스와 폴리곤 라벨의")
    P("정합성을 보아 '결함인데 정상' 류의 사례가 실재하는지 센다.")
    P("")
    normal_with_defect = 0
    defect_with_only_normal = 0
    defect_with_no_ann = 0
    normal_with_no_ann = 0
    mixed = 0
    for r in recs:
        kcls, cases = r[3], r[7]
        has_def = any(c == "defect" for c, _ in cases)
        has_nrm = any(c == "normal" for c, _ in cases)
        if kcls == "정상":
            if has_def:
                normal_with_defect += 1
            if not cases:
                normal_with_no_ann += 1
        else:
            if not cases:
                defect_with_no_ann += 1
            elif not has_def and has_nrm:
                defect_with_only_normal += 1
        if has_def and has_nrm:
            mixed += 1
    P(f"  '정상' 폴더인데 defect 폴리곤 보유 : {normal_with_defect:,}")
    P(f"  결함 폴더인데 normal 폴리곤만 보유 : {defect_with_only_normal:,}")
    P(f"  결함 폴더인데 폴리곤 0개           : {defect_with_no_ann:,}")
    P(f"  '정상' 폴더인데 폴리곤 0개         : {normal_with_no_ann:,}")
    P(f"  한 이미지에 defect+normal 혼재     : {mixed:,}")
    P("")

    # -------------------------------------------------------------- v3 C-5
    P("=" * 74)
    P("[v3 C-5] 문자열 필드 전수 — 자유 기술 필드가 있는가")
    P("=" * 74)
    for k in sorted(str_fields):
        vals = str_fields[k]
        ln = str_len[k]
        kind = "자유 기술" if len(vals) > 1000 else "열거형(고정 어휘)"
        P(f"  {k:<32} 고유값 {len(vals):>7,}  평균 길이 {sum(ln)/len(ln):5.1f}자   -> {kind}")
        if len(vals) <= 12:
            P(f"      값: {sorted(vals)}")
    P("")

    # -------------------------------------------------------------- v3 면적
    P("=" * 74)
    P("[v3] 결함 폴리곤 면적 — 단위는 픽셀제곱(px^2)이다. mm2 가 아니다")
    P("=" * 74)
    per_cls = defaultdict(list)
    for r in rt:
        if r[3] == "정상":
            continue
        for (cl, cs), a in zip(r[7], r[8]):
            if cl == "defect" and a > 0:
                per_cls[(r[2], r[3])].append(a)
    P(f"  {'재질/클래스':<20} {'n':>8} {'최소':>10} {'중앙값':>12} {'평균':>12} {'최대':>12}")
    for k in sorted(per_cls):
        v = sorted(per_cls[k])
        P(f"  {k[0]+'/'+k[1]:<20} {len(v):>8,} {v[0]:>10,.0f} {v[len(v)//2]:>12,.0f} {sum(v)/len(v):>12,.0f} {v[-1]:>12,.0f}")
    P("")
    P("  이 값들은 전부 px^2 다. mm2 로 바꾸려면 mm/pixel 이 필요한데 데이터에 없다.")
    P("  ISO 5817/10675 허용치는 mm 기준이므로 치수 판정에 직접 쓸 수 없다.")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
