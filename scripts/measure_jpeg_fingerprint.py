"""M2. JPEG 헤더 전수 스캔 — 색심도·채널수·양자화표·추정 품질.

재인코딩 목표 품질을 정하는 근거다. 확정 규격은 "양 모집단(결함·정상) 중 **낮은 쪽**에
맞춰 전량 재인코딩"이다. 높은 쪽에 맞추면 낮은 쪽을 업샘플하는 셈이라 없던 정보를 만든
것처럼 보이고, 양자화표 차이가 그대로 남아 지름길이 이사한다.

픽셀을 디코딩하지 않는다. **헤더(SOF·DQT)만 읽는다.**

    uv run python scripts/measure_jpeg_fingerprint.py --zips data/raw/aihub71761/_zips
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

#: libjpeg 표준 휘도 양자화표 (품질 50 기준). 추정 품질은 이 표와의 배율로 낸다.
STD_LUM = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]


def parse_jpeg_header(data: bytes) -> dict | None:
    """SOF 와 DQT 만 읽는다. 실패하면 None."""
    if not data.startswith(b"\xff\xd8"):
        return None
    i, out = 2, {"dqt": [], "channels": None, "precision": None, "w": None, "h": None}
    n = len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > n:
            break
        seg_len = int.from_bytes(data[i : i + 2], "big")
        seg = data[i + 2 : i + seg_len]
        if marker in (0xC0, 0xC1, 0xC2):                     # SOF0/1/2
            if len(seg) >= 6:
                out["precision"] = seg[0]
                out["h"] = int.from_bytes(seg[1:3], "big")
                out["w"] = int.from_bytes(seg[3:5], "big")
                out["channels"] = seg[5]
        elif marker == 0xDB:                                  # DQT
            p = 0
            while p < len(seg):
                pq, tq = seg[p] >> 4, seg[p] & 0x0F
                p += 1
                size = 64 * (2 if pq else 1)
                table = list(seg[p : p + size]) if not pq else []
                if table:
                    out["dqt"].append((tq, table))
                p += size
        elif marker == 0xDA:                                  # SOS. 이후는 압축 데이터
            break
        i += seg_len
    return out if out["w"] else None


def estimate_quality(table: list[int]) -> float:
    """표준 표와의 배율로 품질을 추정한다 (libjpeg 규칙의 역산).

    libjpeg 는 품질 q 를 배율 s(%) 로 바꿔 표준표에 곱한다.
        q >= 50 : s = 200 - 2q      → q = (200 - s) / 2
        q <  50 : s = 5000 / q      → q = 5000 / s
    s = 100 이 두 식의 경계이고 그때 q = 50 이다.
    """
    ratios = [t / s for t, s in zip(table, STD_LUM, strict=True) if s]
    scale = statistics.median(ratios) * 100
    if scale <= 0:
        return 100.0
    q = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return round(max(1.0, min(100.0, q)), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zips", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data/raw/aihub71761/_zips")
    ap.add_argument("--per-zip", type=int, default=400, help="zip 당 표본 수 (0 이면 전수)")
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    pops: dict[str, list[float]] = defaultdict(list)
    channels: Counter = Counter()
    precision: Counter = Counter()
    dqt_hashes: dict[str, Counter] = defaultdict(Counter)
    chan_by_pop: dict[str, Counter] = defaultdict(Counter)
    failures = 0

    incomplete: list[str] = []
    for zp in sorted(args.zips.glob("*.zip")):
        pop = "normal" if "정상" in zp.name else "defect"
        try:
            zipfile.ZipFile(zp).close()
        except zipfile.BadZipFile:
            # 복사가 끝나지 않은 파일. 조용히 건너뛰면 부분 통계가 전수처럼 보인다.
            incomplete.append(zp.name)
            continue
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg"))]
            if args.per_zip:
                names = names[: args.per_zip]
            for name in names:
                head = z.open(name).read(1 << 16)
                info = parse_jpeg_header(head)
                if not info or not info["dqt"]:
                    failures += 1
                    continue
                channels[info["channels"]] += 1
                chan_by_pop[pop][info["channels"]] += 1
                precision[info["precision"]] += 1
                lum = next((t for tq, t in info["dqt"] if tq == 0), info["dqt"][0][1])
                pops[pop].append(estimate_quality(lum))
                dqt_hashes[pop][hashlib.sha256(bytes(lum)).hexdigest()[:12]] += 1

    if incomplete:
        print(f"!! 열 수 없는 zip {len(incomplete)}개 (복사 미완?). 전수가 아니다:")
        for n in incomplete:
            print(f"     {n}")
    print(f"헤더 판독 실패 {failures}건")
    print(f"채널 수 분포 {dict(channels)}  (1=그레이스케일, 3=컬러)")
    print(f"비트 정밀도 {dict(precision)}")
    print()
    summary = {"channels": dict(channels), "precision": dict(precision), "populations": {}}
    for pop, qs in sorted(pops.items()):
        if not qs:
            continue
        s = sorted(qs)
        rec = {
            "n": len(qs), "min": s[0], "q25": s[len(s) // 4],
            "median": statistics.median(s), "q75": s[3 * len(s) // 4], "max": s[-1],
            "distinct_dqt": len(dqt_hashes[pop]),
            "channels": dict(chan_by_pop[pop]),
        }
        summary["populations"][pop] = rec
        ch = rec["channels"]
        gray = ch.get(1, 0) / max(1, sum(ch.values())) * 100
        print(f"{pop:7s} n={rec['n']:,} 추정품질 min {rec['min']} / 중앙 {rec['median']} / "
              f"max {rec['max']} · 양자화표 {rec['distinct_dqt']}종 · "
              f"채널 {ch} (그레이 {gray:.1f}%)")

    if len(summary["populations"]) == 2:
        gr = {k: v["channels"].get(1, 0) / max(1, sum(v["channels"].values())) * 100
              for k, v in summary["populations"].items()}
        gap = abs(gr.get("defect", 0) - gr.get("normal", 0))
        summary["grayscale_pct_by_population"] = {k: round(v, 2) for k, v in gr.items()}
        summary["grayscale_gap_pp"] = round(gap, 2)
        print(f"\n채널 수가 모집단을 가르는가: 그레이 비율 차 {gap:.2f}%p")
        print("  차이가 크면 색심도 자체가 지름길이다. mode L 강제가 이것을 없앤다.")
        target = min(v["min"] for v in summary["populations"].values())
        summary["recommended_quality"] = int(target)
        print(f"\n권고 재인코딩 품질: {int(target)} (양 모집단 최저치)")
        print("높은 쪽에 맞추면 낮은 쪽을 업샘플하는 셈이고 양자화표 차이가 남는다.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")
        print(f"기록: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
