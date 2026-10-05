"""필름에 인쇄된 표기를 **로컬에서** 읽고 항목별 **후보**를 센다. 외부 API 를 쓰지 않는다(레드라인 2).

## 이 조사가 내는 것과 내지 않는 것

낸다: **패턴 적중 후보 장수.** 어떤 정규식이 어떤 층에서 몇 장에 걸렸는가.
내지 않는다: 확인된 두께·확인된 재질·확인된 투과도계. 글자가 읽혔다는 것과 그 값이 맞다는 것은 다르다.
**사람이 원문을 검수하기 전에는 어떤 항목도 "확보" 로 승격하지 않는다.**

항목마다 여는 것이 다르므로 따로 센다.

| 항목 | 찾는 모양 | 적중하면 **후보로** 열리는 것 |
|---|---|---|
| 두께 | `6T` · `12.6T` · `8mm` | 허용치 표의 두께 구간 선택 |
| 재질기호 | `SA283C` · `A312TP347H` | 재질 라벨 검증 · 허용치 표의 재질 축 |
| IQI | `IQI` · `DIN` · `EN 462` · `FE n` | **표준 이름이 읽혔을 뿐이다.** 선 번호·지름·화소 폭·투영 보정은 이 코드에 없다 |
| 스케일 | `10CM` · 눈금 | 같음 |
| 용접부번호 | `WD001` 꼴 | 같은 용접부 묶기. **재질기호와 겹쳐 잡힐 수 있다**(아래) |
| 날짜 | `21.08.10` · 여섯 자리 | 촬영 시기와 재질의 교락. 여섯 자리는 유효 날짜 검사를 하지 않는다 |

**정규식끼리 겹친다.** `SA283C` 는 재질기호이면서 용접부번호 모양(`SA283`)에도 걸린다.
그래서 항목별 후보를 **합산해 "메타데이터 확보 영상 수" 로 쓰지 않는다.** 겹친 장수를 따로 센다.

## 회계 — 단계마다 분모와 실패를 따로 적는다

선정 → 원본 매칭 → 읽기 → OCR → 후보 적중. 하나의 성공률로 합치지 않는다.
분모 102 가 "원본 매칭 102" 나 "OCR 성공 102" 를 뜻하지 않는다.

## 출력

실행마다 **새 경로**를 만든다(덮어쓰지 않는다). 표본별 판독 근거는 같은 폴더의 `*_표본별_비공개.json` 에 두고,
집계 JSON 에는 식별자·원문을 담지 않는다.

사용: python scripts/metadata/film_text_read.py [--n 102] [--threads 1] [--out-dir <경로>]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.measure_tiling_geometry import read_labels
from scripts.run_tiling import index_zip_members

BAND = 0.12
OUT_ROOT = ROOT / "_workspace" / "2026-09-25-metadata"

#: 항목 → (정규식, 적중하면 **후보로** 열리는 것)
PATTERNS: dict[str, tuple[re.Pattern, str]] = {
    "두께": (re.compile(r"(?<![A-Z0-9])(\d{1,3}(?:\.\d)?)\s*(?:T|MM)(?![A-Z])"), "허용치 표의 두께 구간 선택(후보)"),
    "재질기호": (re.compile(r"(?<![A-Z0-9])(S?A\s?\d{3}[A-Z]?|TP\s?\d{3}[A-Z]{0,2}|WP\s?[A-Z]{1,2}\d{0,2}|P\s?\d{1,2}(?![0-9]))"),
                 "재질 라벨 검증·허용치 표의 재질 축(후보)"),
    "IQI": (re.compile(r"(?<![A-Z])(IQI|ASTM|DIN|EN\s?462|FE\s?\d|WIRE|와이어|투과도)"),
            "표준 이름 적중일 뿐. 선지름·화소 보정은 이 코드에 없다"),
    "스케일": (re.compile(r"(?<![A-Z0-9])(\d{1,3}\s?CM|눈금|SCALE)(?![A-Z])"), "같음"),
    "용접부번호": (re.compile(r"(?<![A-Z0-9])([A-Z]{2}\s?\d{3,4})(?![0-9])"), "같은 용접부 묶기(후보). 재질기호와 겹친다"),
    "날짜": (re.compile(r"(?<!\d)(\d{2}\s?\.\s?\d{2}\s?\.\s?\d{2}|\d{6})(?!\d)"), "촬영 시기 교락(후보). 유효 날짜 검사 없음"),
}


def variants(a: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """원본 · 반전 · 대비 스트레치. 필름 인쇄는 밝을 수도 어두울 수도 있다. 이름을 붙여 출처를 남긴다."""
    lo, hi = np.percentile(a, [2, 98])
    st = np.clip((a.astype(np.float32) - lo) * (255.0 / max(1.0, hi - lo)), 0, 255).astype(np.uint8)
    return [("원본", a), ("반전", 255 - a), ("스트레치", st), ("반전스트레치", 255 - st)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="필름 인쇄 표기의 패턴 적중 후보를 센다")
    ap.add_argument("--n", type=int, default=102, help="표본 장수(층에 고르게 나눈다)")
    ap.add_argument("--threads", type=int, default=1, help="OCR 스레드. 여유 메모리를 먼저 보고 올린다")
    ap.add_argument("--out-dir", type=Path, default=None, help="출력 폴더. 비우면 실행 시각으로 새로 만든다")
    a = ap.parse_args(argv)
    if a.n <= 0:
        print("!! --n 은 1 이상이어야 한다", file=sys.stderr)
        return 2

    stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    outdir = a.out_dir or (OUT_ROOT / f"filmtext_{stamp}")
    if outdir.exists():
        print(f"!! 이미 있다: {outdir} — 덮지 않는다", file=sys.stderr)
        return 2
    outdir.mkdir(parents=True)

    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    cv2.setNumThreads(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    import easyocr

    man = {r["image_id"]: r for r in csv.DictReader((ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline=""))}
    prov = {r["image_id"]: r["provenance"] for r in csv.DictReader((ROOT / "data/interim/manifest_v1/tiles.csv").open(encoding="utf-8", newline=""))}
    recs = {f"aihub71761:{r.image_id}": r for r in read_labels(ROOT / "data/interim/aihub_labels") if r.modality == "RT"}
    members = index_zip_members(ROOT / "data/raw/aihub71761/_zips")

    strata = defaultdict(list)
    for iid, m in man.items():
        if iid in recs:
            strata[f"{prov.get(iid, '?')}/{m['material']}"].append(iid)
    if not strata:
        print("!! 층이 비었다 — 매니페스트와 라벨이 이어지지 않는다", file=sys.stderr)
        return 2
    rng = np.random.default_rng(20260921)
    picked: list[tuple[str, str]] = []
    per = max(1, a.n // len(strata))
    for k in sorted(strata):
        pool = strata[k]
        for i in rng.choice(len(pool), size=min(per, len(pool)), replace=False):
            picked.append((k, pool[i]))
    picked = picked[:a.n]

    # ---- 단계별 회계. 합치지 않는다.
    acct = Counter({"선정": len(picked)})
    seen_by_stratum = Counter(k for k, _ in picked)
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)

    hit_images: dict[str, set] = defaultdict(set)          # 항목 → 적중한 표본 번호
    hit_by_stratum: dict[str, Counter] = defaultdict(Counter)
    shapes: dict[str, Counter] = defaultdict(Counter)
    per_sample: list[dict] = []
    print(f"층 {len(strata)}개 · 선정 {len(picked)}장 · 스레드 {a.threads} · 출력 {outdir.name}", flush=True)

    for idx, (k, iid) in enumerate(picked):
        row: dict = {"표본": idx, "층": k, "image_id": iid}
        h = members.get(Path(recs[iid].file_name).stem)
        if not h:
            acct["원본_매칭_실패"] += 1
            row["상태"] = "원본_매칭_실패"
            per_sample.append(row)
            print(f"  [{idx:>3}] {k:14s} 원본 매칭 실패", flush=True)
            continue
        acct["원본_매칭"] += 1
        zp, name = h
        try:
            with zipfile.ZipFile(zp) as z:
                data = z.read(name)
            with Image.open(io.BytesIO(data)) as im:
                arr = np.asarray(im.convert("L"), dtype=np.uint8)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            acct["읽기_실패"] += 1
            row["상태"] = f"읽기_실패:{type(e).__name__}"
            per_sample.append(row)
            print(f"  [{idx:>3}] {k:14s} 읽기 실패 {type(e).__name__}", flush=True)
            continue
        acct["읽기"] += 1

        b = int(arr.shape[0] * BAND)
        found: list[dict] = []
        try:
            for strip_name, strip in (("위", arr[:b]), ("아래", arr[arr.shape[0] - b:])):
                for vname, v in variants(strip):
                    for _box, txt, conf in reader.readtext(v, detail=1, paragraph=False):
                        if conf >= 0.3 and txt.strip():
                            found.append({"띠": strip_name, "판": vname, "원문": txt.strip(), "확신": round(float(conf), 3)})
        except Exception as e:  # noqa: BLE001 — OCR 실패도 회계에 남긴다
            acct["OCR_실패"] += 1
            row["상태"] = f"OCR_실패:{type(e).__name__}"
            per_sample.append(row)
            print(f"  [{idx:>3}] {k:14s} OCR 실패 {type(e).__name__}", flush=True)
            continue
        acct["OCR"] += 1
        if found:
            acct["글자_읽힘"] += 1

        joined = re.sub(r"\s+", " ", " ".join(f["원문"] for f in found)).upper()
        this: list[str] = []                                # **이번 장의 적중만** 담는다
        for item, (pat, _why) in PATTERNS.items():
            m = pat.findall(joined)
            if not m:
                continue
            this.append(item)
            hit_images[item].add(idx)
            hit_by_stratum[item][k] += 1
            for x in m[:3]:
                shapes[item][re.sub(r"\d", "#", x if isinstance(x, str) else x[0])] += 1
        if this:
            acct["후보_적중"] += 1
        row.update({"상태": "정상", "읽힌_토막": len(found), "적중": this, "원문": found, "합친_문자열": joined})
        per_sample.append(row)
        print(f"  [{idx:>3}] {k:14s} 토막 {len(found):>3} · 적중 {' '.join(this) if this else '-'}", flush=True)

    # ---- 산출. 집계에는 식별자·원문을 담지 않는다.
    n = len(picked)
    overlap = Counter()
    for i in range(n):
        items = tuple(sorted(it for it, s in hit_images.items() if i in s))
        if len(items) > 1:
            overlap[" + ".join(items)] += 1

    print("\n[단계별 회계] 분모를 합치지 않는다")
    for stage in ("선정", "원본_매칭", "원본_매칭_실패", "읽기", "읽기_실패", "OCR", "OCR_실패", "글자_읽힘", "후보_적중"):
        print(f"  {stage:14s} {acct[stage]:>6,}")
    print(f"\n{'항목':10s} {'후보 적중 장수':>12} {'/ OCR 성공':>10}   모양(숫자는 #)")
    for item in PATTERNS:
        sh = " ".join(list(shapes[item])[:4])
        print(f"{item:10s} {len(hit_images[item]):>12,} {acct['OCR']:>10,}   {sh[:44]}")
    if overlap:
        print(f"\n겹쳐 잡힌 장: {dict(overlap)} — 항목별 후보를 합산하지 않는다")

    summary = {
        "생성": stamp, "표본": n, "층별_선정": dict(seen_by_stratum),
        "회계": dict(acct),
        "주의": ("값은 패턴 적중 후보 장수다. 확인된 두께·재질·투과도계가 아니다. "
                 "층마다 같은 장수를 뽑았으므로 모집단 보유율이 아니다. 항목별 후보는 겹칠 수 있어 합산하지 않는다."),
        "항목": {item: {"후보_적중_장수": len(hit_images[item]), "분모_OCR성공": acct["OCR"],
                        "층별": dict(hit_by_stratum[item]), "모양": dict(shapes[item]), "여는것": why}
                 for item, (_p, why) in PATTERNS.items()},
        "겹침": dict(overlap),
    }
    (outdir / "집계.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "표본별_비공개.json").write_text(json.dumps(
        {"경고": "표본별 식별자와 판독 원문이 들어 있다. 공개 문서에 옮기지 않는다. 사람 검수 상태를 여기에 적는다.",
         "표본": per_sample}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {outdir / '집계.json'} (식별자·원문 없음)")
    print(f"→ {outdir / '표본별_비공개.json'} **비공개**", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
