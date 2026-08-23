#!/usr/bin/env python
"""AI허브 용접 데이터셋 라벨 전수 확인기.

드라이브에서 라벨 아카이브만 풀어 놓은 디렉터리를 받아, 스키마·클래스·좌표·
분할 축 후보를 전수 집계한다. 원천 이미지는 열지 않는다 (스트리밍이라 느리고,
세는 것만으로 충분하다).

사용:
    python scripts/inspect_aihub.py <라벨_루트> -o <출력.md 또는 .txt>

설계 원칙 — 추정하지 않는다. 센 것만 적는다.
필드가 없으면 "없다"고 적고, 못 센 것은 못 셌다고 적는다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

# 두께·물리 스케일·촬영 조건에 해당할 수 있는 모든 표기를 넓게 건다.
# 키 이름과 문자열 값 양쪽에 적용한다.
SCALE_RE = re.compile(
    r"thick|두께|scale|스케일|mm\b|pixel_?size|pixel_?spacing|spacing|resolution|dpi|"
    r"kv|kvp|\bma\b|exposure|노출|sod|sfd|sdd|distance|거리|film|source|voltage|current|"
    r"physical|calib|magnif|배율|specimen|plate|판재|재질두께|penetrant|energy",
    re.IGNORECASE,
)

# 연합 클라이언트 분할 축이 될 수 있는 필드 (회사·설비·촬영일 등)
SPLIT_AXIS_RE = re.compile(
    r"company|corp|vendor|공급|회사|업체|site|plant|공장|line|설비|equip|device|machine|"
    r"camera|sensor|date|datetime|time|일자|촬영|작업|operator|작업자|lot|batch|serial|"
    r"part|부재|joint|용접부|weld_?id|seam",
    re.IGNORECASE,
)


def flatten(obj, prefix, present, nonempty, types, samples, hits_key, hits_val, axis_hits):
    """JSON을 재귀로 훑어 키 경로를 평탄화하고 관심 패턴을 수집한다."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            present.add(path)
            if v not in (None, "", [], {}):
                nonempty.add(path)
            types[path][type(v).__name__] += 1
            if isinstance(v, (str, int, float)) and len(samples[path]) < 12:
                samples[path].add(str(v)[:80])
            if SCALE_RE.search(k):
                hits_key[path] += 1
            if isinstance(v, str) and SCALE_RE.search(v):
                hits_val[f"{path}={v[:50]}"] += 1
            if SPLIT_AXIS_RE.search(k):
                axis_hits[path] += 1
            flatten(v, path, present, nonempty, types, samples, hits_key, hits_val, axis_hits)
    elif isinstance(obj, list):
        for v in obj:
            flatten(v, prefix + "[]", present, nonempty, types, samples, hits_key, hits_val, axis_hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="라벨 JSON이 풀려 있는 루트 디렉터리")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    present_ct = Counter()
    nonempty_ct = Counter()
    types = defaultdict(Counter)
    samples = defaultdict(set)
    hits_key = Counter()
    hits_val = Counter()
    axis_hits = Counter()

    n_files = 0
    parse_fail = []
    per_archive = Counter()

    dims = Counter()
    dims_by_matl = defaultdict(Counter)
    fmt = Counter()
    itype = Counter()
    imatl = Counter()
    info_kr = Counter()
    kr_by_matl = defaultdict(Counter)
    tool = Counter()
    ann_class = Counter()
    ann_case = Counter()
    case_by_type = defaultdict(Counter)
    crowd = Counter()
    inst_per_img = Counter()
    verts = Counter()

    frac_coord = 0
    out_of_box = 0
    out_files = Counter()
    neg_files = set()
    xmin = ymin = 10**9
    xmax = ymax = -(10**9)

    # 파일명 접두사별 연속 id — 같은 용접부 연속 촬영 판정용
    ids_by_prefix = defaultdict(list)
    name_pat = Counter()

    for dirpath, _, filenames in os.walk(args.root):
        arch = os.path.relpath(dirpath, args.root).split(os.sep)[0]
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            n_files += 1
            per_archive[arch] += 1
            fp = os.path.join(dirpath, fn)
            try:
                with open(fp, encoding="utf-8") as fh:
                    d = json.load(fh)
            except Exception as e:  # 파싱 실패도 결과다. 조용히 넘기지 않는다.
                parse_fail.append((fp, repr(e)))
                continue

            present, nonempty = set(), set()
            flatten(d, "", present, nonempty, types, samples, hits_key, hits_val, axis_hits)
            for p in present:
                present_ct[p] += 1
            for p in nonempty:
                nonempty_ct[p] += 1

            info = d.get("info") or {}
            im = d.get("image_data") or {}
            meta = d.get("meta") or {}
            t = info.get("type")
            m = info.get("material")
            itype[t] += 1
            imatl[m] += 1
            w, h = im.get("width"), im.get("height")
            dims[f"{w}x{h}"] += 1
            dims_by_matl[f"{t}/{m}"][f"{w}x{h}"] += 1
            fmt[im.get("format")] += 1
            info_kr[im.get("information")] += 1
            kr_by_matl[f"{t}/{m}"][im.get("information")] += 1
            crowd[meta.get("is_crowd")] += 1

            name = im.get("file_name") or ""
            mm = re.match(r"^(.*?)_(\d+)$", name)
            if mm:
                ids_by_prefix[mm.group(1)].append(int(mm.group(2)))
                name_pat[re.sub(r"\d+", "#", name)] += 1

            anns = d.get("annotations") or []
            inst_per_img[len(anns)] += 1
            for a in anns:
                tool[a.get("tool")] += 1
                ann_class[a.get("class")] += 1
                c = a.get("case")
                ann_case[c] += 1
                case_by_type[t][c] += 1
                co = a.get("coordinate") or {}
                xs, ys = co.get("x") or [], co.get("y") or []
                verts[len(xs)] += 1
                bad = 0
                for v in xs:
                    if isinstance(v, float) and 0 < v < 1:
                        frac_coord += 1
                    xmin, xmax = min(xmin, v), max(xmax, v)
                    if isinstance(w, int) and (v < 0 or v > w):
                        bad += 1
                for v in ys:
                    if isinstance(v, float) and 0 < v < 1:
                        frac_coord += 1
                    ymin, ymax = min(ymin, v), max(ymax, v)
                    if isinstance(h, int) and (v < 0 or v > h):
                        bad += 1
                if bad:
                    out_of_box += bad
                    out_files[f"{t}/{m}"] += 1
                    if any(v < 0 for v in xs) or any(v < 0 for v in ys):
                        neg_files.add(fp)

    L: list[str] = []
    P = L.append

    P(f"라벨 JSON 전수 스캔: {n_files:,} 파일")
    P(f"파싱 실패: {len(parse_fail)}")
    for fp, e in parse_fail[:20]:
        P(f"    {fp}  {e}")
    P("")

    P("=" * 70)
    P("[1] 두께 · 픽셀-mm 스케일 · 촬영 조건 탐지")
    P("=" * 70)
    P(f"키 이름 매치 : {len(hits_key)}")
    for k, v in hits_key.most_common(30):
        P(f"    {k}  x{v}")
    P(f"문자열 값 매치 : {len(hits_val)}")
    for k, v in hits_val.most_common(30):
        P(f"    {k}  x{v}")
    if not hits_key and not hits_val:
        P("    *** 히트 0건 — 해당 필드가 스키마에 존재하지 않는다 ***")
    P("")

    P("=" * 70)
    P("[2] 연합 분할 축 후보 (회사·설비·촬영일 등)")
    P("=" * 70)
    if axis_hits:
        for k, v in axis_hits.most_common(30):
            P(f"    {k}  x{v}")
    else:
        P("    *** 히트 0건 — 회사·설비·촬영일 필드가 없다 ***")
    P("")

    P("=" * 70)
    P("[3] 스키마 전수 (키경로 : 보유파일 / 비어있지않음 / 타입)")
    P("=" * 70)
    for k in sorted(present_ct):
        P(f"  {k:<40} {present_ct[k]:>8,} / {nonempty_ct[k]:>8,}  {dict(types[k])}")
    P("")
    P("--- 값 표본 (원문 그대로, 번역·정규화하지 않음) ---")
    for k in sorted(samples):
        if len(samples[k]) <= 12:
            P(f"  {k:<40} {sorted(samples[k])}")
    P("")

    P("=" * 70)
    P("[4] 라벨 구조 · 위치 정보")
    P("=" * 70)
    P(f"annotation tool          : {dict(tool)}")
    P(f"annotation class         : {dict(ann_class)}")
    P(f"annotation case (원문)   : {dict(ann_case)}")
    P(f"case x 촬영방식          :")
    for t, c in case_by_type.items():
        P(f"    {t}: {dict(c)}")
    P(f"좌표 x 범위              : {xmin} .. {xmax}")
    P(f"좌표 y 범위              : {ymin} .. {ymax}")
    P(f"0~1 소수 좌표            : {frac_coord}   (>0 이면 정규화 좌표)")
    P(f"이미지 경계 밖 좌표값    : {out_of_box:,}")
    P(f"  영향 인스턴스(재질별)  : {dict(out_files)}")
    P(f"  음수 좌표 포함 파일    : {len(neg_files):,}")
    P(f"폴리곤 꼭짓점 수         : min={min(verts) if verts else 0} max={max(verts) if verts else 0}")
    P(f"  꼭짓점 3개 미만(퇴화)  : {sum(v for k, v in verts.items() if k < 3)}")
    P(f"meta.is_crowd            : {dict(crowd)}")
    P(f"이미지당 인스턴스 (상위) : {dict(sorted(inst_per_img.items())[:15])}")
    P(f"  최대                   : {max(inst_per_img) if inst_per_img else 0}")
    P(f"  총 인스턴스            : {sum(k * v for k, v in inst_per_img.items()):,}")
    P("")

    P("=" * 70)
    P("[5] 수량 · 분포")
    P("=" * 70)
    P(f"info.type      : {dict(itype)}")
    P(f"info.material  : {dict(imatl)}")
    P(f"image format   : {dict(fmt)}")
    P(f"클래스(원문 한글) 전체: {dict(info_kr)}")
    P("촬영방식/재질 x 클래스:")
    for k in sorted(kr_by_matl):
        tot = sum(kr_by_matl[k].values())
        P(f"    {k} (n={tot:,}) {dict(kr_by_matl[k])}")
    P("")
    P("해상도 (상위 12):")
    for k, v in dims.most_common(12):
        P(f"    {k:<14} {v:>8,}")
    P(f"  서로 다른 해상도 종수: {len(dims):,}")
    for k in sorted(dims_by_matl):
        P(f"    {k}: 종수={len(dims_by_matl[k]):,}  최빈={dims_by_matl[k].most_common(1)}")
    P("")
    P("아카이브별 파일 수:")
    for k, v in sorted(per_archive.items()):
        P(f"    {k:<45} {v:>8,}")
    P("")

    P("=" * 70)
    P("[6] 같은 용접부 연속 촬영 판정 (파일명 id 인접성)")
    P("=" * 70)
    P(f"파일명 패턴: {dict(name_pat.most_common(5))}")
    P(f"접두사 종수: {len(ids_by_prefix)}")
    all_ids = sorted(v for lst in ids_by_prefix.values() for v in lst)
    if all_ids:
        runs, cur = [], 1
        for a, b in zip(all_ids, all_ids[1:]):
            if b == a + 1:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        rc = Counter(runs)
        P(f"전체 id 범위: {all_ids[0]} .. {all_ids[-1]}  (고유 {len(set(all_ids)):,})")
        P(f"연속 id 런(run) 개수: {len(runs):,}")
        P(f"  런 길이 분포(상위): {dict(rc.most_common(12))}")
        P(f"  최장 런: {max(runs)}")
        P(f"  길이 1 (고립) 비율: {rc[1] / len(runs) * 100:.1f}%")
    P("")
    P("접두사별 id 개수 (상위 15):")
    for k, v in sorted(ids_by_prefix.items(), key=lambda kv: -len(kv[1]))[:15]:
        P(f"    {k:<20} {len(v):>8,}")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L[:8]))
    print(f"... 전체 결과 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
