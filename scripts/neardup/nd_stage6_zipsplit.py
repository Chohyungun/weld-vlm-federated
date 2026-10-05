"""근사 중복 6단계 — 참여자를 건너는 중복이 **원본 수록 단계**에서 온 것인지 본다.

물음: 같은 사진이 서로 다른 회사 몫으로 갈린 것이, 우리가 묶기에 실패해서인가 아니면
원본 자료가 애초에 같은 자료를 여러 묶음(zip)에 나눠 담았기 때문인가.

보는 것 (원본 zip 은 **읽기만** 한다. 구성원 목록과 파일명만 보고 이미지는 열지 않는다)
  - 같은 그림 쌍의 두 장이 **같은 zip** 에 있는가, 다른 zip 에 있는가
  - 파일명 접두가 같은가 — **주의: 이 자료에서 접두는 촬영 묶음이 아니라 재질 토큰이다**(아래 `prefix` 참조).
    그래서 '접두 같음' 은 '재질 같음' 과 같은 말이고 독립된 근거가 되지 못한다. 열은 기록으로 남기되 결론에 쓰지 않는다.
  - 참여자를 건너는 쌍과 같은 참여자 안의 쌍에서 그 비율이 다른가 — 다르면 수록 단계의 성질이다
  - zip 별로 어느 참여자에 배정됐는지의 분포 — 한 zip 이 여러 참여자로 갈리는가

사용: python nd_stage6_zipsplit.py [표본 쌍 수]    기본 전량
출력: stage6_zipsplit.json — 건수·비율만 담는다(식별자·파일명은 담지 않는다).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_stage5_lists as S

from scripts.measure_tiling_geometry import read_labels
from scripts.run_tiling import index_zip_members

W = S.W
RULE = "rd_sharp >= 0.35 and blk_frac >= 0.5"
csv.field_size_limit(10_000_000)


def prefix(name: str) -> str:
    """파일명에서 숫자 꼬리를 떼어 낸 앞부분.

    **이 자료에서는 촬영 묶음을 가리키지 못한다.** 파일명이 `RT_{ST|AL}_##_########` 꼴이라
    남는 값이 `RT_ST`·`RT_AL` 둘뿐이고, 곧 재질 토큰과 같다. 이 함수의 결과를 "같은 촬영" 의 근거로 쓰지 않는다.
    (2026-09-21 에 이 오해로 10번 §3-1 을 썼다가 정정했다.)
    """
    stem = Path(name).stem
    i = len(stem)
    while i > 0 and (stem[i - 1].isdigit() or stem[i - 1] in "_-"):
        i -= 1
    return stem[:i] or stem


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    man = {r["image_id"]: r for r in csv.DictReader((ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline=""))}
    recs = {f"aihub71761:{r.image_id}": r for r in read_labels(ROOT / "data/interim/aihub_labels") if r.modality == "RT"}
    members = index_zip_members(ROOT / "data/raw/aihub71761/_zips")
    print(f"라벨 {len(recs):,} · 원본 구성원 {len(members):,}", flush=True)

    def where(iid: str):
        """(zip 이름, 파일명 접두). 못 찾으면 None."""
        rec = recs.get(iid)
        if rec is None:
            return None
        hit = members.get(Path(rec.file_name).stem)
        return (hit[0].name, prefix(rec.file_name)) if hit else None

    # zip 이 어느 참여자로 갔는가 — 한 zip 이 여러 참여자로 갈리면 수록 단위와 배정 단위가 다르다는 뜻이다
    zip_client = defaultdict(Counter)
    for iid, m in man.items():
        if m["split"] == "eval":
            continue
        w = where(iid)
        if w:
            zip_client[w[0]][m["client"]] += 1
    split_zip = {z: dict(c) for z, c in zip_client.items() if len(c) > 1}
    print(f"학습 풀이 들어 있는 zip {len(zip_client)}개 · 그중 **여러 참여자로 갈린** zip {len(split_zip)}개", flush=True)

    rule, _ = S.compile_rule(RULE)
    pairs = [r for r in S.load_within_trainval() if rule(r)]
    if limit:
        pairs = pairs[:limit]
    out = {"rule": RULE, "n_pairs": len(pairs), "n_zips_with_pool": len(zip_client),
           "n_zips_spanning_clients": len(split_zip), "zip_client_spread": split_zip}

    for kind in ("참여자 건넘", "같은 참여자"):
        sel = [r for r in pairs
               if (man[r["image_id_a"]]["client"] != man[r["image_id_b"]]["client"]) == (kind == "참여자 건넘")]
        same_zip = diff_zip = same_pre = unknown = 0
        for r in sel:
            wa, wb = where(r["image_id_a"]), where(r["image_id_b"])
            if not wa or not wb:
                unknown += 1
                continue
            same_zip += wa[0] == wb[0]
            diff_zip += wa[0] != wb[0]
            same_pre += wa[1] == wb[1]
        n = len(sel) - unknown
        out[kind] = {"n": len(sel), "미확인": unknown, "같은 zip": same_zip, "다른 zip": diff_zip,
                     "파일명 접두 같음(재질 토큰과 같다 — 근거 아님)": same_pre,
                     "같은 zip 비율": round(same_zip / n, 4) if n else None,
                     "접두 같음 비율(근거 아님)": round(same_pre / n, 4) if n else None}
        print(f"[{kind}] {len(sel):,}쌍 · 같은 zip {same_zip:,} · 다른 zip {diff_zip:,} · 접두 같음 {same_pre:,}(재질 토큰과 같다) · 미확인 {unknown:,}", flush=True)

    (W / "stage6_zipsplit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→ stage6_zipsplit.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
