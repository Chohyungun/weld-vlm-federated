"""사전등록 상수 재산출 — **N 이 690 틀렸다.** 체크리스트 18 (80번 D2).

    uv run python scripts/probe/recompute_prereg.py                 # → outputs/pilot_d/
    uv run python scripts/probe/recompute_prereg.py --out outputs/main_d

`evaluation/prereg.py` 의 상수는 `RT_TOTAL = 62_998` 위에 서 있는데 동결 스냅샷은
**62,308** 이다. 690장 차이가 자명하한을 흔들고, 그 자명하한에서 유도한 통과선
0.2131 이 동결 평가셋 자명하한보다 **낮아** 과엄격 헛경보를 만든다.

여기서 세 모집단의 값을 실측하고 `prereg.py` 가 그 값을 쓰게 한다. 산출은
`<채점 디렉터리>/prereg_recomputed_v1.json`.

**채점기가 직접 부른다.** `score_cells.py score` 는 채점 디렉터리에 이 파일이 없으면
`recompute()` 로 만들어 선배치한다(13번 D-8 파생 — 없으면 `prereg_constants_reproduced`
게이트가 `skipped` 로 갈리는 것이 검증에서 실측됐다). 상수는 동결 스냅샷의 결정론적
함수라 사람이 복사해 두는 것보다 코드가 만드는 쪽이 안전하다.

**규격전용(spec-only) 상수도 함께 재산출한다.** 재인코딩으로 전 이미지가 1280×720 이
됐으므로 "1280×720 에만 주장하는 규칙"은 전량양성과 같아진다 — 규격 지름길의 순 기여가
0 이 되는 것이 함정 #11 대응의 성공 증거다. 등록된 0.0944 는 처리 **전** 값이다.

평가셋 라벨을 읽는다. 이것은 **채점 상수 산출**이지 학습 투입이 아니며, 자명하한은
정의상 평가 모집단의 유병률 함수다(모집단이 다르면 값이 다르다는 것이 요점).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.label_map import load_label_map
from evaluation.eval_set import ensure_verified, parse_iso_codes, read_manifest
from evaluation.prereg import all_positive_macro_f1, spec_only_macro_f1

FROZEN = "data/interim/manifest_v1"
DEFAULT_OUT = Path("outputs/pilot_d")
FILE_NAME = "prereg_recomputed_v1.json"
R1_SIZE = (1280, 720)


def population(rows: list[dict], classes: list[str]) -> dict:
    counts = Counter()
    for r in rows:
        for c in parse_iso_codes(r.get("iso_codes")):
            if c in classes:
                counts[c] += 1
    n = len(rows)
    r1 = [r for r in rows
          if (int(r["width_px"]), int(r["height_px"])) == R1_SIZE]
    r1_ids = {r["image_id"] for r in r1}
    tp = Counter()
    for r in rows:
        if r["image_id"] not in r1_ids:
            continue
        for c in parse_iso_codes(r.get("iso_codes")):
            if c in classes:
                tp[c] += 1
    base, base_per = all_positive_macro_f1({c: counts[c] for c in classes}, n_images=n)
    spec, spec_per = spec_only_macro_f1(
        {c: counts[c] for c in classes},
        n_predicted_positive=len(r1),
        tp_counts={c: tp[c] for c in classes},
    )
    return {
        "n_images": n,
        "n_r1_1280x720": len(r1),
        "class_counts": {c: counts[c] for c in classes},
        "all_positive_macro_f1": base,
        "all_positive_per_class": base_per,
        "spec_only_macro_f1": spec,
        "spec_only_per_class": spec_per,
        "shortcut_contribution": spec - base,
    }


def recompute(frozen: str | Path = FROZEN) -> dict:
    """동결 스냅샷 → 세 모집단의 사전등록 상수. **평가셋 라벨을 읽되 학습에 넣지 않는다.**"""
    frozen = str(frozen)
    digest = ensure_verified(frozen)
    lm = load_label_map()
    classes = [lm.iso_code(n) for n in
               ("crack", "porosity", "lack_of_fusion", "slag_inclusion")]

    rows = read_manifest(frozen)
    pops = {
        "frozen_total": population(rows, classes),
        "frozen_eval": population([r for r in rows if r["split"] == "eval"], classes),
        "frozen_trainval": population(
            [r for r in rows if r["split"] in ("train", "val")], classes),
    }
    return {
        "snapshot": frozen,
        "snapshot_digest": digest,
        "classes": classes,
        "populations": pops,
        "registered_before": {
            "RT_TOTAL": 62_998, "R1_COUNT": 37_814,
            "all_positive_macro_f1": 0.2081, "spec_only_macro_f1": 0.3025,
            "shortcut_contribution": 0.0944,
        },
        "note": (
            "재인코딩 후 전 이미지가 1280×720 이라 규격전용 규칙이 전량양성과 같아진다. "
            "지름길 순 기여 0 은 함정 #11 대응이 성공했다는 증거이며, 등록값 0.0944 는 "
            "처리 전 값이다 — 같은 이름의 다른 양이다."
        ),
    }


def write_payload(payload: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="사전등록 상수 동결본 재산출")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="채점 디렉터리. 여기 prereg_recomputed_v1.json 을 쓴다")
    args = ap.parse_args(argv)
    payload = recompute(FROZEN)
    dest = write_payload(payload, Path(args.out) / FILE_NAME)
    for k, v in payload["populations"].items():
        print(f"[{k}] n={v['n_images']} r1={v['n_r1_1280x720']} "
              f"전량양성 {v['all_positive_macro_f1']:.8f} "
              f"규격전용 {v['spec_only_macro_f1']:.8f} "
              f"순기여 {v['shortcut_contribution']:+.8f}")
    print(f"저장: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
