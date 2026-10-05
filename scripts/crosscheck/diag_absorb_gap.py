"""잠정판 회계와 기대값이 어긋난 다섯 자리를 실물로 가른다.

기대값은 미니스펙 §5 의 예시 블록과 내 앞선 검산에서 왔다. 어느 쪽이 틀렸는지 재지 않고
한쪽을 고치면 수를 맞추려고 구현을 비트는 일이 된다. 그래서 **세는 단위부터 다시 센다.**

산출에 식별자가 섞이므로 추적 밖 경로에만 쓴다.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.absorb import _clamp, num_key, stream_jsonl  # noqa: E402
from data.manifest_io import load_snapshot  # noqa: E402

V1 = ROOT / "data/interim/manifest_v1"
V2 = ROOT / "data/interim/manifest_v2_absorbed"
#: 저장소 밖 입력은 환경 변수로 받는다. 로컬 절대경로를 코드에 두지 않는다.
JSONL = Path(os.environ.get("ABSORB_JSONL", ""))
OUT = ROOT / "_workspace/2026-09-30-흡수/gap.json"


def main() -> int:
    v1 = load_snapshot(V1, verify=True)
    v2 = load_snapshot(V2, verify=False)
    man, ann = v1.manifest, v1.annotations
    by_num = {}
    for iid in man["image_id"]:
        by_num.setdefault(num_key(iid), iid)
    wh = {iid: (int(r.width_px), int(r.height_px), bool(r.has_defect))
          for iid, r in zip(man["image_id"], man.itertuples(index=False))}

    # --- 1. 참여자 열의 결측 표현 -------------------------------------------------
    cl = man["client"]
    client_repr = {"dtype": str(cl.dtype), "na_rows": int(cl.isna().sum()),
                   "str_of_na": str(cl[cl.isna()].iloc[0]) if cl.isna().any() else "",
                   "빈문자열_행": int((cl.fillna("") == "").sum())}

    # --- 2. 아카이브 모수 — 전체 줄 대 겹치는 장 ----------------------------------
    zips_all, zips_cov, zips_by_mod = set(), set(), {}
    n_pair_uneven, uneven_examples = 0, []
    raw_by_ann: dict[str, tuple] = {}
    cov_ids = set()
    for d in stream_jsonl(JSONL):
        z = (d.get("source_zip") or "").strip()
        mod = d.get("modality", "")
        if z:
            zips_all.add(z)
            zips_by_mod.setdefault(mod, set()).add(z)
        oid = str(d.get("orig_info_id", "")).strip()
        iid = by_num.get(oid)
        if iid is None:
            continue
        cov_ids.add(iid)
        if z:
            zips_cov.add(z)
        ds = sorted((a for a in (d.get("annotations") or [])
                     if a.get("annotation_role") == "defect"),
                    key=lambda a: a.get("ann_idx", 0))
        rows = ann.loc[ann["image_id"] == iid] if False else None  # 느린 경로를 쓰지 않는다
        raw_by_ann[iid] = tuple((int(a["bbox_xmin"]), int(a["bbox_ymin"]),
                                 int(a["bbox_xmax"]), int(a["bbox_ymax"])) for a in ds)
        del rows

    ours_n = Counter(str(x) for x in ann["image_id"])
    for iid, raws in raw_by_ann.items():
        if ours_n.get(iid, 0) != len(raws):
            n_pair_uneven += 1
            if len(uneven_examples) < 5:
                uneven_examples.append({"우리": ours_n.get(iid, 0), "정리된파일": len(raws)})

    archives = {
        "정리된_파일_전체_줄에_나타난_아카이브": len(zips_all),
        "겹치는_장에만_나타난_아카이브": len(zips_cov),
        "양식별": {k: len(v) for k, v in sorted(zips_by_mod.items())},
        "짝수가_다른_이미지": n_pair_uneven,
        "짝수가_다른_예": uneven_examples,
    }

    # --- 3. 정상영역 — 장수와 영역수를 따로 센다 ----------------------------------
    reg = v2.absorb_region
    per_img = reg.groupby("image_id")["frame"].agg(["count", "first"])
    regions = {
        "영역_수": {k: int(v) for k, v in reg["frame"].value_counts().items()},
        "장_수": {k: int(v) for k, v in per_img["first"].value_counts().items()},
        "겹치는_정상_장수": int(sum(1 for i in cov_ids if not wh[i][2])),
        "영역이_둘_이상인_장": int((per_img["count"] > 1).sum()),
        "장당_영역수_분포": {int(k): int(v) for k, v in per_img["count"].value_counts().sort_index().items()},
        "여분_영역": int(len(reg) - len(per_img)),
    }

    # --- 4. 상자 — 자르기 분해를 다시 센다 ----------------------------------------
    # 이 진단은 곁파일이 덮인 주석만 담던 첫 판을 보려고 썼다. 지금은 모든 주석에 한 행이므로
    # 같은 모집단을 보려면 원좌표를 담은 행으로 거른다.
    dfc = v2.absorb_defect[v2.absorb_defect["raw_source"] == "absorbed"]
    a1 = ann.set_index("ann_id")
    both_present = dfc[dfc["raw_bbox_x1_px"].notna()]
    diff, boundary, poly_flag, unexplained, empty_ours = 0, 0, 0, 0, 0
    for r in dfc.itertuples(index=False):
        src = a1.loc[r.ann_id]
        try:
            mine = (int(src.bbox_x1_px), int(src.bbox_y1_px),
                    int(src.bbox_x2_px), int(src.bbox_y2_px))
        except (TypeError, ValueError):
            empty_ours += 1
            continue
        if pd.isna(r.raw_bbox_x1_px):
            continue
        raw = (int(r.raw_bbox_x1_px), int(r.raw_bbox_y1_px),
               int(r.raw_bbox_x2_px), int(r.raw_bbox_y2_px))
        if mine == raw:
            continue
        diff += 1
        w, h = wh[r.image_id][0], wh[r.image_id][1]
        is_b = mine == _clamp(raw, w, h)
        is_p = "multipart_largest_kept" in str(src.geom_flags or "")
        if is_b:
            boundary += 1
        elif is_p:
            poly_flag += 1
        else:
            unexplained += 1

    v1_empty = 0
    for r in ann.itertuples(index=False):
        try:
            int(r.bbox_x1_px), int(r.bbox_y1_px), int(r.bbox_x2_px), int(r.bbox_y2_px)
        except (TypeError, ValueError):
            v1_empty += 1

    boxes = {
        "곁파일_행수": int(len(dfc)),
        "원좌표가_있는_행": int(len(both_present)),
        "우리_상자가_빈_행_겹치는장": empty_ours,
        "우리_상자가_빈_행_동결본_전체": v1_empty,
        "값이_다른_행": diff,
        "경계로_설명": boundary,
        "폴리곤보정으로_설명": poly_flag,
        "설명되지_않음": unexplained,
        "회계의_clip_kinds": {k: int(v) for k, v in dfc["clip_kind"].value_counts().items()},
    }

    out = {"참여자_열": client_repr, "아카이브": archives, "정상영역": regions, "상자": boxes}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
