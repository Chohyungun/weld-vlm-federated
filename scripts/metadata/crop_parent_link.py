"""결함 크롭과 **부모 파노라마**를 잇는 고리가 파일명에 있는가 — **없다는 것을 기록으로 남기는 조사다.**

## 왜 만들었고 무엇이 틀렸나

인쇄 표기는 파노라마에만 있고 1280×720 결함 크롭에는 없다. 크롭이 파노라마에서 잘라 낸 것이라면
부모를 찾아 두께를 물려받을 수 있다. 그 연결을 **파일명 접두**로 찾으려 했다.

**그 접근은 성립하지 않는다.** 파일명이 `RT_{ST|AL}_##_########` 꼴이라, 뒤의 숫자를 떼어 낸 접두는
`RT_ST` · `RT_AL` **두 값뿐**이다. 곧 "접두가 같다" 는 **"재질 토큰이 같다"** 와 같은 말이고 촬영 묶음을 가리키지 않는다.
이 스크립트가 내는 "부모 후보가 있는 크롭 100%" 도 그래서 **아무것도 말해 주지 않는다** — 재질이 같은 파노라마가 있다는 뜻일 뿐이다.

## 그러면 이 스크립트는 무엇에 쓰나

**부정 결과의 근거**다. 파일명에 촬영 묶음 정보가 없다는 사실을 수치로 남긴다(서로 다른 접두 수, 접두당 영상 수).
같은 결론을 화소 쪽에서도 따로 확인했다 — 근사 중복 쌍에 크롭↔파노라마 조합이 하나도 없다(10번 §3-1).
**두 경로 모두 연결을 세우지 못했다.** 다만 이것은 "연결이 없다" 의 증명이 아니라 **"이 두 방법으로는 찾지 못했다"** 이다.
취득 id 체계와 라벨 JSON 의 미사용 필드가 남은 후보다.

사용: python scripts/metadata/crop_parent_link.py
출력: stdout + crop_parent_link.json (건수·비율만. 식별자·파일명은 담지 않는다)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "neardup"))
import nd_stage6_zipsplit as Z

from scripts.measure_tiling_geometry import read_labels
from scripts.run_tiling import index_zip_members

W = ROOT / "_workspace" / "2026-09-25-metadata"
PANO = {"N-tile", "N-band"}


def main() -> int:
    man = {r["image_id"]: r for r in csv.DictReader((ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline=""))}
    prov = {r["image_id"]: r["provenance"] for r in csv.DictReader((ROOT / "data/interim/manifest_v1/tiles.csv").open(encoding="utf-8", newline=""))}
    recs = {f"aihub71761:{r.image_id}": r for r in read_labels(ROOT / "data/interim/aihub_labels") if r.modality == "RT"}
    members = index_zip_members(ROOT / "data/raw/aihub71761/_zips")

    by_prefix: dict[str, Counter] = defaultdict(Counter)
    prefix_of: dict[str, str] = {}
    unmatched = 0
    for iid, rec in recs.items():
        if iid not in man:
            continue
        if not members.get(Path(rec.file_name).stem):
            unmatched += 1
            continue
        pre = Z.prefix(rec.file_name)
        prefix_of[iid] = pre
        by_prefix[pre][prov.get(iid, "?")] += 1

    crops = [i for i in prefix_of if prov.get(i) == "N-crop"]
    panos = [i for i in prefix_of if prov.get(i) in PANO]
    sizes = sorted(sum(c.values()) for c in by_prefix.values())
    print(f"원본 매칭 실패 {unmatched:,} · 매칭된 영상 {len(prefix_of):,} · 크롭 {len(crops):,} · 파노라마 {len(panos):,}")
    print(f"**서로 다른 접두 {len(by_prefix)}개** · 접두당 영상 수 {sizes}")
    if len(by_prefix) <= 5:
        print("  → 접두가 이렇게 적으면 촬영 묶음을 가리키지 못한다. 아래 수치는 연결의 증거가 아니다.")

    linked = sum(any(by_prefix[prefix_of[i]][p] for p in PANO) for i in crops)
    ev_crops = [i for i in crops if man[i]["split"] == "eval"]
    ev_linked = sum(any(by_prefix[prefix_of[i]][p] for p in PANO) for i in ev_crops)
    print(f"\n(참고) 같은 접두에 파노라마가 있는 크롭 {linked:,}/{len(crops):,}"
          f" · eval 크롭 {ev_linked:,}/{len(ev_crops):,} — **재질이 같은 파노라마가 있다는 뜻일 뿐이다**")

    out = {
        "결론": "파일명 접두로는 크롭↔파노라마를 잇지 못한다. 접두가 재질 토큰과 같기 때문이다.",
        "주의": "아래 연결 수는 부모 확정이 아니다. 재질이 같은 파노라마의 존재일 뿐이다.",
        "서로_다른_접두": len(by_prefix), "접두당_영상수": sizes,
        "원본_매칭_실패": unmatched, "매칭된_영상": len(prefix_of),
        "크롭": len(crops), "파노라마": len(panos),
        "같은_접두에_파노라마_있는_크롭": linked, "eval_크롭": len(ev_crops), "eval_크롭_연결후보": ev_linked,
        "남은_후보_경로": ["취득 id 체계", "라벨 JSON 의 미사용 필드"],
    }
    W.mkdir(parents=True, exist_ok=True)
    # 다른 조사기와 같은 규약 — 있으면 멈춘다. 고정 파일명으로 조용히 덮지 않는다.
    stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    outp = W / f"crop_parent_link_{stamp}.json"
    if outp.exists():
        print(f"!! 이미 있다: {outp} — 덮지 않는다", file=sys.stderr)
        return 2
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {outp.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
