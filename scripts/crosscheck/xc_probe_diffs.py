"""남은 차이 둘의 성격을 본다 — **크기 불일치와 상자 불일치.**

세는 것이 아니라 **차이가 무엇으로 설명되는가**를 본다. 읽기만 하고 식별자를 찍지 않는다.
동결본은 읽기 전에 지문을 검증한다. 우리 쪽 빈 상자 행은 버리지 않고 센다.

  가. 크기가 다른 장 — 어느 쪽이 무엇을 적고 있나
  나. 상자가 다른 장 — 차이가 **경계 자르기**로 설명되는가, **폴리곤 보정**으로 설명되는가

사용: python xc_probe_diffs.py <jsonl 경로> [산출 json 경로]
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xc_common import (CLAMP_NOTE, MANI, clamp, load_manifest, safe_out,  # noqa: E402
                       verify_snapshot)
from xc_metadata_jsonl import stream_jsonl  # noqa: E402

csv.field_size_limit(10_000_000)


def main() -> int:
    seal = verify_snapshot()
    man = load_manifest()
    by_num, multi = {}, 0
    for iid, m in man.items():
        if m["num"] in by_num:
            multi += 1
        by_num.setdefault(m["num"], iid)

    size_pairs = Counter()
    jb = {}
    for d in stream_jsonl(Path(sys.argv[1])):
        k = str(d.get("orig_info_id", "")).strip()
        iid = by_num.get(k)
        if not iid:
            continue
        m = man[iid]
        jwh = (str(d.get("width_px", "")), str(d.get("height_px", "")))
        if m["wh"] != jwh:
            size_pairs[(f"우리 {m['wh'][0]}x{m['wh'][1]}", f"jsonl {jwh[0]}x{jwh[1]}",
                        "결함" if m["has_defect"] == "True" else "정상")] += 1
            continue
        bs = []
        for a in (d.get("annotations") or []):
            if a.get("annotation_role") != "defect":
                continue
            try:
                bs.append((int(a["bbox_xmin"]), int(a["bbox_ymin"]),
                           int(a["bbox_xmax"]), int(a["bbox_ymax"])))
            except (KeyError, TypeError, ValueError):
                pass
        jb[iid] = (sorted(bs), int(jwh[0]), int(jwh[1]))

    # 우리 상자 — **빈 행을 버리지 않고 표시와 함께 센다.** 개수 차이의 사유가 여기 있다.
    ours = defaultdict(list)
    flags = defaultdict(list)
    acct = Counter()
    with (MANI / "annotations.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["image_id"] not in jb:
                continue
            acct["겹친_장의_주석행"] += 1
            fl = (r.get("geom_flags") or "").strip()
            if fl:
                flags[r["image_id"]].append(fl)
            try:
                ours[r["image_id"]].append(
                    (int(float(r["bbox_x1_px"])), int(float(r["bbox_y1_px"])),
                     int(float(r["bbox_x2_px"])), int(float(r["bbox_y2_px"]))))
            except (KeyError, TypeError, ValueError):
                acct["상자가_빈_행"] += 1
                acct[f"빈_행의_표시:{fl or '(없음)'}"] += 1

    why = Counter()
    resid = []
    for iid, (jbox, w, h) in jb.items():
        a = sorted(ours.get(iid, []))
        if a == jbox:
            continue
        if len(a) != len(jbox):
            # 우리 쪽 빈 상자 행이 몇 개인지 같이 적는다 — 개수 차이가 그것으로 닫히는지 보이게.
            miss = sum(1 for f in flags.get(iid, []) if "zero_area" in f)
            why["상자_개수가_다르다"] += 1
            why["그중_빈_상자_행_수와_차가_같다" if len(jbox) - len(a) == miss
                else "그중_빈_상자_행으로_설명되지_않는다"] += 1
            continue
        if sorted(clamp(b, w, h) for b in jbox) == a:
            why["경계_자르기로_설명된다"] += 1
            continue
        fl = ";".join(sorted(set(flags.get(iid, []))))
        if "multipart_largest_kept" in fl:
            why["폴리곤_보정으로_설명된다"] += 1
            why[f"  표시:{fl}"] += 1
            continue
        why["설명되지_않는다"] += 1
        if len(resid) < 5:
            resid.append({"우리": a[:2], "jsonl": jbox[:2],
                          "자른_값": sorted(clamp(b, w, h) for b in jbox)[:2],
                          "크기": [w, h], "표시": fl or "(없음)"})

    rep = {
        "동결본_지문_검증": seal,
        "숫자부가_여럿에_걸린_장": multi,
        "가_크기_불일치의_조합": {f"{k[0]} ↔ {k[1]} · {k[2]}": v for k, v in size_pairs.most_common(10)},
        "가_크기_불일치_합": sum(size_pairs.values()),
        "가_결함유무별": dict(Counter(k[2] for k in size_pairs.elements())),
        "나_상자_불일치의_사유": dict(why),
        "나_설명되지_않는_표본(식별자_없음)": resid,
        "우리_주석행_회계": dict(acct),
        "자르기_규약": CLAMP_NOTE,
    }
    out = safe_out(Path(sys.argv[2])) if len(sys.argv) > 2 else None
    txt = json.dumps(rep, ensure_ascii=False, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt, encoding="utf-8", newline="\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
