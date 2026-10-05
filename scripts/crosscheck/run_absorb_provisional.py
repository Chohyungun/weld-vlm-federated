"""잠정판을 낸다 — 흡수 미니스펙 §9.

동결본과 정리된 파일을 읽기만 하고 새 경로에만 쓴다. 회계가 두 검산의 수와 다르면 **쓰기 전에**
멈춘다. 같은 경로에 다시 내면 앞 판의 지문이 `digest_history` 에 남는다(§9-2).

경로는 저장소 안에서 푼다. 저장소 밖의 두 입력은 환경 변수로 받는다 — 로컬 절대경로를 코드에 두지 않는다.

  ABSORB_JSONL         정리된 파일(images.jsonl)
  ABSORB_DATASET_ROOT  원본 데이터셋 폴더(라벨 아카이브 이름만 센다)

사용: python run_absorb_provisional.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.absorb import AbsorptionError, build  # noqa: E402

V1 = ROOT / "data/interim/manifest_v1"
OUT = ROOT / "data/interim/manifest_v2_absorbed"
LABEL_MAP = ROOT / "configs/label_map.yaml"

#: 두 검산이 각각 낸 수와 명세 10-2 의 실물 검수. **다르면 흡수 로직이 틀린 것이다.**
#:
#: 그레인과 모수를 값 옆에 적는다. 명세 §5 의 `present: 21` 은 정리된 파일 전체 줄이고 덮인 장만 보면
#: 14 다. §3-2 의 53·39·9 와 §3-4 의 5,885·1,379 는 **장수**이고 곁파일은 **행**을 센다.
EXPECT = {
    # R-1. 정리된 파일의 신원
    "source_digest": "b7e0b417befb0fd26b9dcbb41992996677fc78d7973897e0f83c7ec0c7e8d28b",
    # R-2
    "covered_images": 19902,
    "total_images": 62308,
    "not_inserted": {"vt": 15168, "rt_missing_from_v1": 95},
    # R-3. (covered, total). 결측 참여자는 평가셋과 같은 집합임을 확인한 뒤 `eval` 로 모은다
    "client": {"C1": (6495, 26451), "C2": (2313, 16253), "C3": (6989, 7143), "eval": (4105, 12461)},
    "material": {"ST": (11160, 53379), "AL": (8742, 8929)},
    # 모수 = 정리된 파일 전체 줄 (RT 14 + VT 7). 덮인 장만 보면 14 다
    "archives": {"total_label_archives": 30, "present": 21, "missing_rt": 6, "missing_vt": 3,
                 "present_in_covered": 14},
    # 장수 — 명세 §3-4 표
    "region_images_by_frame": {"tile": 1379, "orig": 5885},
    # R-4. 장수 — 명세 §3-2 본문. 행으로는 54·43·9 다
    "clip_by_image": {"count_differs": 53, "boundary": 39, "polygon_repair": 9},
    # R-5. v2 매니페스트가 v1 과 다른 열은 이것 하나다
    "changed_columns_vs_v1": ["iso_codes"],
    # 명세 §8-1 3번 · §8-2 5번
    "iso_pair_fixed": 112,
    "missing_bbox_rows": 85,
    "recovered_from_absorb": 54,
    # 결함 곁파일은 모든 주석에 한 행
    "defect_rows": {"total": 109125, "absorbed": 30056, "not_covered": 79069},
    "defect_pairs_iso_checked": 30056,
}
#: 실행 뒤 사람이 읽는 행 그레인 값. 앞선 검산은 행으로 세지 않아 기대값을 두지 않는다.
NOTE_ROWS = ("clip_kinds", "region_frames", "regions_per_image", "regions_not_stored_on_defect_images")

#: §9-2 장치가 생기기 전에 같은 경로에 쓰였다가 덮인 판. 디스크에서 찾을 수 없으므로 여기서 준다.
#: 출처: 흡수 구현 보고의 고정 SHA 표. 그때의 `present` 는 덮인 장 모수(14)였다.
SEED_HISTORY = [{
    "status": "provisional",
    "digest": "64c87fa33c04ad7525c38b5c6daa3372a2cfe6dd1a04cdd2ce84a863b78a00ee",
    "at": None,
    "archives_present": 14,
    "note": "회계가 기대와 달라 반려했다. 쓰기가 대조보다 먼저라 파일이 남았고 다음 판이 같은 경로를 덮었다. "
            "바이트는 남지 않았고 시각은 기록되지 않았다",
}]


def _cmp(bad, path, got, want):
    if got != want:
        bad.append(f"{path}: 기대 {want} 실제 {got}")


def check(acct: dict) -> list[str]:
    bad: list[str] = []
    for key in ("source_digest", "covered_images", "total_images", "not_inserted",
                "changed_columns_vs_v1", "iso_pair_fixed", "missing_bbox_rows",
                "recovered_from_absorb", "defect_rows", "defect_pairs_iso_checked"):
        _cmp(bad, key, acct.get(key), EXPECT[key])
    for axis, want in (("coverage_by_client", "client"), ("coverage_by_material", "material")):
        for k, (c, t) in EXPECT[want].items():
            _cmp(bad, f"{axis}[{k}]", acct[axis].get(k), {"covered": c, "total": t})
    for group in ("archives", "region_images_by_frame", "clip_by_image"):
        for k, v in EXPECT[group].items():
            _cmp(bad, f"{group}.{k}", (acct.get(group) or {}).get(k), v)
    return bad


def _env_path(name: str) -> Path:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"환경 변수 {name} 가 비어 있다. 저장소 밖 입력의 경로를 준다")
    p = Path(v)
    if not p.exists():
        raise SystemExit(f"{name} 가 가리키는 경로가 없다")
    return p


def main() -> int:
    jsonl, dataset = _env_path("ABSORB_JSONL"), _env_path("ABSORB_DATASET_ROOT")
    try:
        acct = build(V1, jsonl, OUT, LABEL_MAP, dataset_root=dataset, check=check,
                     seed_history=SEED_HISTORY)
    except AbsorptionError as exc:
        print(f"★ {exc}\n\n쓰지 않았다.", flush=True)
        return 2
    acct["기대값_대조"] = "전건 일치"
    acct["행_그레인_참고"] = {k: acct[k] for k in NOTE_ROWS}
    print(json.dumps(acct, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
