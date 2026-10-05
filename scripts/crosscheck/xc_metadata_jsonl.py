"""정리된 이미지 메타데이터 jsonl 을 동결 매니페스트와 맞대 수를 센다.

**읽기만 한다.** 드라이브 파일·동결 자산·이미지 파일을 열거나 고치지 않는다.
jsonl 은 한 줄씩 흘려 읽는다. 병렬을 쓰지 않는다. 동결본은 읽기 전에 지문을 검증한다.

세는 것
  1 겹치는 장 수 — **서로 다른 결합 키 둘**을 각각 시험해 차이를 적는다
  2 한쪽에만 있는 장 수(모달별)
  3 동결본 기준 참여자별·재질별 포함률
  4 jsonl 의 `source_zip` 이 가리키는 아카이브와 우리 원천 아카이브의 차
  5 겹치는 장의 결함 주석 수와 경계 상자 일치

사용: python xc_metadata_jsonl.py <jsonl 경로> [산출 json 경로]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xc_common import (CLAMP_NOTE, ZIPS, id_tail, load_boxes, load_manifest,  # noqa: E402
                       num_key, safe_out, verify_snapshot)


def stream_jsonl(path: Path):
    """한 줄씩 흘려 읽는다. 파일 전체를 메모리에 올리지 않는다."""
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"{n}번째 줄을 JSON 으로 읽지 못했다 — 값을 내지 않는다")


def main() -> int:
    jpath = Path(sys.argv[1])
    seal = verify_snapshot()          # 어긋나면 여기서 멈춘다
    man = load_manifest()
    by_num = defaultdict(list)
    by_stem = defaultdict(list)
    for iid, m in man.items():
        by_num[m["num"]].append(iid)
        by_stem[m["stem"]].append(iid)

    j_rows = 0
    j_by_oid: dict = {}
    j_by_tail: dict = {}
    dupe = Counter()
    j_modal = Counter()
    zips = Counter()
    j_boxes: dict = {}
    bad_id = 0

    for d in stream_jsonl(jpath):
        j_rows += 1
        j_modal[d.get("modality", "")] += 1
        zips[d.get("source_zip", "")] += 1
        oid = str(d.get("orig_info_id", "")).strip()
        tail = id_tail(str(d.get("image_id", "")))
        if not oid:
            bad_id += 1
            continue
        if oid in j_by_oid:
            dupe["orig_info_id_중복"] += 1
            continue
        rec = {
            "image_id": d.get("image_id", ""), "tail": tail,
            "modality": d.get("modality", ""), "material": d.get("material_nominal", ""),
            "n_def": d.get("n_defect_annotations", None),
            "wh": (str(d.get("width_px", "")), str(d.get("height_px", ""))),
        }
        j_by_oid[oid] = rec
        if tail in j_by_tail:
            dupe["image_id_꼬리_중복"] += 1
        else:
            j_by_tail[tail] = rec
        # 상자는 **겹치는 장에서만** 쓴다. 주석과 코드를 맞춘다.
        if oid in by_num:
            bs = []
            for a in (d.get("annotations") or []):
                if a.get("annotation_role") != "defect":
                    continue
                try:
                    bs.append((int(a["bbox_xmin"]), int(a["bbox_ymin"]),
                               int(a["bbox_xmax"]), int(a["bbox_ymax"])))
                except (KeyError, TypeError, ValueError):
                    dupe["jsonl_상자_값이_정수가_아님"] += 1
            j_boxes[oid] = bs

    # --- 1. 겹치는 장 — **서로 다른 결합 키 둘을 각각 시험한다** ----------------
    # 앞 판은 두 후보가 같은 값을 가리켜 차이가 구조적으로 0 이었다. 이제 키를 갈라 쓴다.
    #   A: jsonl `orig_info_id`     ↔ 매니페스트 image_id 의 숫자부
    #   B: jsonl `image_id` 의 맨 뒤 마디 ↔ 매니페스트 rel_path 의 파일명 어간
    hit_a = {k for k in j_by_oid if k in by_num}
    hit_b = {t for t in j_by_tail if t in by_stem}
    only_a = {j_by_oid[k]["image_id"] for k in hit_a} - {j_by_tail[t]["image_id"] for t in hit_b}
    only_b = {j_by_tail[t]["image_id"] for t in hit_b} - {j_by_oid[k]["image_id"] for k in hit_a}
    diff_rows = [{"어느_키만_잡았나": "A(orig_info_id)만", "jsonl_image_id": i} for i in sorted(only_a)]
    diff_rows += [{"어느_키만_잡았나": "B(image_id 꼬리)만", "jsonl_image_id": i} for i in sorted(only_b)]

    rep = {
        "동결본_지문_검증": seal,
        "입력": {"jsonl_줄": j_rows, "jsonl_고유_orig_info_id": len(j_by_oid),
               "빈_orig_info_id": bad_id, **dict(dupe), "매니페스트_장": len(man)},
        "1_겹치는_장": {
            "키A_orig_info_id ↔ image_id 숫자부": len(hit_a),
            "키B_image_id 꼬리 ↔ rel_path 어간": len(hit_b),
            "두_키가_갈리는_장": len(diff_rows),
            "갈리는_장_목록": diff_rows,
            "숫자부가_여럿에_걸린_경우": sum(1 for k in hit_a if len(by_num[k]) > 1),
            "어간이_여럿에_걸린_경우": sum(1 for t in hit_b if len(by_stem[t]) > 1),
            "본_대조에_쓴_키": "A",
        },
        "2_한쪽에만": {
            "jsonl_에만(모달별)": dict(Counter(j_by_oid[k]["modality"] for k in j_by_oid if k not in by_num)),
            "jsonl_에만_합": sum(1 for k in j_by_oid if k not in by_num),
            "매니페스트에만": len(man) - len({i for k in hit_a for i in by_num[k]}),
        },
    }

    # --- 3. 동결본 기준 포함률 -------------------------------------------------
    got = {i for k in hit_a for i in by_num[k]}

    def rate(field):
        tot, inc = Counter(), Counter()
        for iid, m in man.items():
            tot[m[field]] += 1
            if iid in got:
                inc[m[field]] += 1
        return {k: {"전체": tot[k], "jsonl_에_있음": inc[k],
                    "포함률": round(inc[k] / tot[k], 4) if tot[k] else None}
                for k in sorted(tot)}
    rep["3_포함률"] = {"참여자별": rate("client"), "재질별": rate("material"),
                    "전체": {"전체": len(man), "jsonl_에_있음": len(got),
                           "포함률": round(len(got) / len(man), 4)}}

    # --- 4. 아카이브 -----------------------------------------------------------
    src = sorted(p.name for p in ZIPS.glob("*.zip")) if ZIPS.exists() else []
    if not src:
        raise SystemExit("우리 원천 아카이브를 하나도 찾지 못했다 — 0건을 '빠진 것 없음' 으로 내지 않는다")
    expect = sorted({n.split("__", 1)[-1].replace("TS_", "TL_").replace("VS_", "VL_") for n in src})
    seen = sorted(z for z in zips if z)
    rep["4_아카이브"] = {
        "jsonl_의_source_zip_고유": len(seen),
        "우리_원천_아카이브": len(src),
        "이름을_옮겨_세운_라벨_아카이브": len(expect),
        "옮기는_규약": "Training__TS_X.zip → TL_X.zip · Validation__VS_X.zip → VL_X.zip. "
                  "우리 쪽에 라벨 아카이브가 없어 이름으로 세운다. **실물 대조가 아니고 RT 만이다**",
        "jsonl_에_없는_라벨_아카이브": [z for z in expect if z not in seen],
        "jsonl_에만_있는_source_zip": [z for z in seen if z not in expect],
    }

    # --- 5. 결함 주석 수와 상자 -------------------------------------------------
    want = {i for k in hit_a for i in by_num[k]}
    ann, acct = load_boxes(want)
    cmp_ = Counter()
    box_same = box_diff = box_skip = 0
    for k in sorted(hit_a):
        iid = by_num[k][0]
        m, j = man[iid], j_by_oid[k]
        n_m = int(m["n_defects"] or 0)
        n_j = int(j["n_def"] or 0)
        cmp_["주석수_같음" if n_m == n_j else "주석수_다름"] += 1
        if m["wh"] != j["wh"]:
            box_skip += 1
            cmp_["크기_다름"] += 1
            continue
        a = sorted(ann.get(iid, []))
        b = sorted(j_boxes.get(k, []))
        if a == b:
            box_same += 1
            cmp_["결함있음_상자일치" if m["has_defect"] == "True" else "결함없음_양쪽_상자0"] += 1
        else:
            box_diff += 1
    rep["5_주석과_상자"] = {
        "견준_장": len(hit_a), **dict(cmp_),
        "크기_같은_장에서_상자_완전일치": box_same,
        "크기_같은_장에서_상자_불일치": box_diff,
        "크기_달라_건너뜀": box_skip,
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
