"""R2-a 테두리 마스킹 — RT 전량, 각 변 8%, 동일 상대 규격.

P2·P3 불합격 대응이다. 출처 판별 AUC 0.9972 의 주 채널이 **1백분위 밝기**(어두운 쪽 끝)
였고, 파노라마에서 뜬 타일에만 필름 가장자리·배경의 어두운 영역이 들어와 있었다.
99백분위는 양쪽이 사실상 같으므로(162.2 대 164.4) 밝은 쪽은 손대지 않는다.

**대칭 조치다.** 크롭이든 타일이든 같은 규칙을 적용한다. 타일에만 적용하면 그 자체가
새 지름길이 된다.

    uv run python scripts/run_border_mask.py --stage calibrate
    uv run python scripts/run_border_mask.py --stage apply
    uv run python scripts/run_border_mask.py --stage manifest

**tiles_v1 을 덮지 않는다.** 결과는 `tiles_v1_masked/` 에 새로 쓴다.
**분할은 건드리지 않는다.** 매니페스트는 `rel_path` 와 `sha256` 만 갱신한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
V1 = REPO_ROOT / "data/interim/manifest_v1"
OUT = REPO_ROOT / "data/interim/tiles_v1_masked"
CAL = V1 / "border_mask_calibration.json"
PROGRESS = V1 / "mask_progress.jsonl"

#: 각 변 마스킹 비율. 사전 확정 사다리 R2-a 값이다.
BORDER_FRAC = 0.08


def border_px(width: int, height: int) -> tuple[int, int]:
    """각 변에서 걷어낼 화소 수. **상대 비율**이라 크기가 달라도 같은 규칙이다."""
    return round(BORDER_FRAC * width), round(BORDER_FRAC * height)


def interior_box(width: int, height: int) -> tuple[int, int, int, int]:
    bx, by = border_px(width, height)
    return bx, by, width - bx, height - by


def stage_calibrate(manifest) -> int:
    """마스킹 값을 **학습 풀에서만** 도출한다.

    평가셋 화소에서 전처리 상수를 뽑으면 불변조건 1-4 가 실질에서 깨진다. 캘리브레이션은
    타일링 때와 같은 규칙을 따른다.
    """
    pool = manifest[manifest["split"] != "eval"]
    print(f"학습 풀 {len(pool):,}장에서 내부 영역 평균을 낸다 (평가셋 제외)")
    total, count = 0.0, 0
    t0 = time.time()
    for k, r in enumerate(pool.itertuples(), 1):
        with Image.open(REPO_ROOT / r.rel_path) as im:
            arr = np.asarray(im.convert("L"))
        x0, y0, x1, y1 = interior_box(arr.shape[1], arr.shape[0])
        inner = arr[y0:y1, x0:x1]
        total += float(inner.sum())
        count += inner.size
        if k % 10000 == 0:
            print(f"  {k:,}/{len(pool):,}  평균 {total/count:.2f}", flush=True)
    mean = total / count
    bx, by = border_px(1280, 720)
    masked_frac = 1 - ((1280 - 2 * bx) * (720 - 2 * by)) / (1280 * 720)
    rec = {
        "border_frac": BORDER_FRAC,
        "border_px_1280x720": [bx, by],
        "masked_area_frac": round(masked_frac, 4),
        "fill_value": round(mean),
        "interior_mean_train_pool": round(mean, 3),
        "n_images": len(pool),
        "derived_from": "split != eval (학습 풀 전량)",
    }
    CAL.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"내부 평균 {mean:.3f} → 채움값 {rec['fill_value']} · "
          f"마스킹 면적 {masked_frac*100:.2f}% · {(time.time()-t0)/60:.1f}분")
    print(f"기록: {CAL}")
    return 0


def mask_and_encode(src: Path, dst: Path, fill: int, quality: int) -> tuple[str, int, int]:
    """테두리를 고정값으로 채우고 같은 규격으로 다시 인코딩한다.

    **채움값은 전 이미지 공통 상수 하나다.** 이미지별 평균으로 채우면 그 값 자체가
    이미지 통계를 실어 나르므로 지우려던 채널이 테두리에 그대로 남는다.
    """
    with Image.open(src) as im:
        arr = np.asarray(im.convert("L")).copy()
    h, w = arr.shape
    x0, y0, x1, y1 = interior_box(w, h)
    out = np.full_like(arr, fill)
    out[y0:y1, x0:x1] = arr[y0:y1, x0:x1]
    buf = BytesIO()
    Image.fromarray(out, mode="L").save(
        buf, format="JPEG", quality=quality, progressive=False, optimize=False)
    data = buf.getvalue()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), w, h


def load_progress() -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not PROGRESS.exists():
        return done
    for line in PROGRESS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if (REPO_ROOT / rec["rel_path"]).exists():
                done[rec["image_id"]] = rec
    return done


def stage_apply(manifest, quality: int) -> int:
    if not CAL.exists():
        print("!! 캘리브레이션이 없다. --stage calibrate 를 먼저 돌려라.")
        return 3
    cal = json.loads(CAL.read_text(encoding="utf-8"))
    fill = int(cal["fill_value"])
    done = load_progress()
    todo = [r for r in manifest.itertuples() if r.image_id not in done]
    print(f"마스킹 대상 {len(todo):,}장 (완료 {len(done):,}장) · 채움값 {fill}")
    if not todo:
        return 0
    t0 = time.time()
    with PROGRESS.open("a", encoding="utf-8", newline="\n") as fh:
        for k, r in enumerate(todo, 1):
            num = r.image_id.rsplit(":", 1)[-1]
            dst = OUT / r.modality / r.material / f"{num}.jpg"
            sha, w, h = mask_and_encode(REPO_ROOT / r.rel_path, dst, fill, quality)
            fh.write(json.dumps({
                "image_id": r.image_id,
                "rel_path": dst.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha, "width_px": w, "height_px": h,
            }, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                fh.flush()
            if k % 5000 == 0:
                rate = k / max(time.time() - t0, 1e-9)
                print(f"  {k:,}/{len(todo):,}  {rate:.0f}장/초  "
                      f"남은 시간 약 {(len(todo)-k)/max(rate,1e-9)/60:.0f}분", flush=True)
        fh.flush()
    print(f"완료 {len(todo):,}장 · {(time.time()-t0)/60:.1f}분")
    return 0


def stage_manifest(manifest) -> int:
    """`rel_path` 와 `sha256` 만 갈아 끼운다. 분할·묶음·클래스는 손대지 않는다."""
    from data.invariants import check_invariants
    from data.label_map import load_label_map
    from data.manifest_io import FLOAT_FORMAT, read_annotations

    prog = load_progress()
    missing = [i for i in manifest["image_id"] if i not in prog]
    if missing:
        print(f"!! 마스킹이 안 된 이미지 {len(missing):,}장. --stage apply 를 끝내라.")
        return 3

    m = manifest.copy()
    frozen = ["split", "client", "group_id", "group_size", "eval_subset", "strata_key",
              "defect_types", "iso_codes", "has_defect", "material", "width_px", "height_px"]
    before = {c: m[c].copy() for c in frozen}

    m["rel_path"] = m["image_id"].map(lambda i: prog[i]["rel_path"]).astype("string")
    m["sha256"] = m["image_id"].map(lambda i: prog[i]["sha256"]).astype("string")

    # 분할 배정이 실제로 안 바뀌는지 확인한다. 안 건드렸다는 주장 대신 대조한다.
    changed = [c for c in frozen if not before[c].fillna("~").eq(m[c].fillna("~")).all()]
    if changed:
        print(f"!! 불변이어야 할 컬럼이 바뀌었다: {changed}")
        return 4
    print(f"불변 확인: {len(frozen)}개 컬럼 전부 동일 (split·client·group_id·클래스 포함)")

    if m["sha256"].duplicated().any():
        dups = int(m["sha256"].duplicated().sum())
        print(f"   참고: 마스킹 후 바이트 동일 타일 {dups:,}행 (테두리를 걷어내 늘 수 있다)")

    lm = load_label_map()
    a = read_annotations(V1 / "annotations.csv")
    violations = check_invariants(m, a, lm, raw_root=REPO_ROOT)
    print(f"불변조건 위반 {len(violations)}건")
    for v in violations[:20]:
        print(f"   {v}")

    (V1 / "manifest_pre_mask.csv").write_bytes((V1 / "manifest.csv").read_bytes())
    out = m.sort_values("image_id", kind="stable").reset_index(drop=True)
    for col in out.columns:
        if str(out[col].dtype) == "boolean":
            out[col] = out[col].map({True: "True", False: "False"}).astype("string")
    with (V1 / "manifest.csv").open("w", encoding="utf-8", newline="\n") as fh:
        out.to_csv(fh, index=False, na_rep="", float_format=FLOAT_FORMAT,
                   lineterminator="\n")
    print(f"기록: {V1 / 'manifest.csv'}  ·  직전본은 manifest_pre_mask.csv")
    print("SNAPSHOT 은 잠그지 않았다.")
    return 7 if violations else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("calibrate", "apply", "manifest"), required=True)
    ap.add_argument("--quality", type=int, default=None)
    args = ap.parse_args()

    import yaml

    from data.manifest_io import read_manifest

    cfg = yaml.safe_load((REPO_ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    quality = args.quality or int(cfg["preprocess"]["tile"]["encode"]["quality"])
    manifest = read_manifest(V1 / "manifest.csv")
    print(f"매니페스트 v1 {len(manifest):,}행")

    if args.stage == "calibrate":
        return stage_calibrate(manifest)
    if args.stage == "apply":
        return stage_apply(manifest, quality)
    return stage_manifest(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
