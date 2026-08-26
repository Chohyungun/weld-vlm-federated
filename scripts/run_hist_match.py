"""히스토그램 정합 — 이미지별 단조 변환으로 전역 밝기 채널을 닫는 마지막 후보.

R2-a 재프로브에서 `lowres` 가 0.9961 로 거의 안 내려갔다. 32×32 로 줄여 전역 통계만
남겨도 출처가 판별된다는 뜻이다. 히스토그램 정합은 **모든 이미지의 주변 분포를 하나의
기준 분포로 강제**하므로 그 채널을 원리적으로 닫는다.

    uv run python scripts/run_hist_match.py --stage calibrate
    uv run python scripts/run_hist_match.py --stage apply
    uv run python scripts/run_hist_match.py --stage manifest
    uv run python scripts/run_hist_match.py --stage contrast

**입력은 `tiles_v1_masked` 다(R2-a 누적).** R2-a 는 통과선을 못 넘었지만 텍스처 채널을
실제로 깎았다(shuffle 0.9852 → 0.8476). 되돌리면 그 이득을 버린다. 모든 이미지가 같은
자리에 같은 상수를 갖고 있으므로 정합의 상대 관계는 왜곡되지 않는다.

**이미지별 단조 변환이다.** 이미지 안의 밝기 순서는 보존된다. 순서가 뒤집히는지는
`--stage contrast` 가 실측으로 확인한다.

**대칭 조치다.** 크롭과 타일에 같은 기준 분포를 쓴다.
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
OUT = REPO_ROOT / "data/interim/tiles_v1_histmatch"
REF = V1 / "histmatch_reference.json"
PROGRESS = V1 / "histmatch_progress.jsonl"


def build_lut(img_hist: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    """이미지 CDF 를 기준 CDF 에 맞추는 단조 사상표.

    `searchsorted` 는 비감소 입력에 비감소 출력을 준다. 따라서 **역전이 원리적으로
    불가능**하다. 같은 값이 같은 값으로 가므로 동점이 생길 뿐 순서는 뒤집히지 않는다.
    """
    total = img_hist.sum()
    if total == 0:
        return np.arange(256, dtype=np.uint8)
    img_cdf = np.cumsum(img_hist) / total
    lut = np.searchsorted(ref_cdf, img_cdf, side="left").astype(np.int32)
    return np.clip(lut, 0, 255).astype(np.uint8)


def stage_calibrate(manifest) -> int:
    """기준 분포 = **학습 풀 전체의 평균 히스토그램.**

    선택 근거는 셋이다.
    1. 모집단 평균이라 특정 이미지·특정 출처로 기울지 않는다. 크롭 쪽이나 타일 쪽 어느
       한쪽을 기준으로 삼으면 그 조치 자체가 새 비대칭이 된다.
    2. 모든 이미지를 같은 분포로 보내면 주변 분포 기반 통계량(1백분위·평균·저해상 축소)이
       설계상 같아진다. 닫으려는 채널을 직접 겨냥한다.
    3. 평가셋 화소를 쓰지 않는다. 전처리 상수를 평가셋에서 유도하면 불변조건 1-4 가
       실질에서 깨진다.

    모든 이미지가 1280×720 로 같으므로 화소 풀링과 이미지별 정규화 히스토그램의 평균은
    같은 값이다. 정의 모호성이 없다.
    """
    pool = manifest[manifest["split"] != "eval"]
    print(f"학습 풀 {len(pool):,}장에서 평균 히스토그램을 만든다 (평가셋 제외)")
    hist = np.zeros(256, dtype=np.int64)
    t0 = time.time()
    for k, r in enumerate(pool.itertuples(), 1):
        arr = np.asarray(Image.open(REPO_ROOT / r.rel_path).convert("L"))
        hist += np.bincount(arr.ravel(), minlength=256)
        if k % 10000 == 0:
            print(f"  {k:,}/{len(pool):,}", flush=True)
    cdf = np.cumsum(hist) / hist.sum()
    pct = {f"p{q}": int(np.searchsorted(cdf, q / 100)) for q in (1, 5, 25, 50, 75, 95, 99)}
    REF.write_text(json.dumps({
        "histogram": hist.tolist(),
        "n_images": len(pool),
        "derived_from": "split != eval (학습 풀 전량), tiles_v1_masked",
        "percentiles": pct,
        "mean": round(float((np.arange(256) * hist).sum() / hist.sum()), 3),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"기준 분포 백분위 {pct} · 평균 "
          f"{(np.arange(256)*hist).sum()/hist.sum():.2f} · {(time.time()-t0)/60:.1f}분")
    print(f"기록: {REF}")
    return 0


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
    if not REF.exists():
        print("!! 기준 분포가 없다. --stage calibrate 를 먼저 돌려라.")
        return 3
    ref = np.asarray(json.loads(REF.read_text(encoding="utf-8"))["histogram"], dtype=np.int64)
    ref_cdf = np.cumsum(ref) / ref.sum()

    done = load_progress()
    todo = [r for r in manifest.itertuples() if r.image_id not in done]
    print(f"정합 대상 {len(todo):,}장 (완료 {len(done):,}장)")
    if not todo:
        return 0
    t0 = time.time()
    non_monotone = 0
    with PROGRESS.open("a", encoding="utf-8", newline="\n") as fh:
        for k, r in enumerate(todo, 1):
            src = REPO_ROOT / r.rel_path
            with Image.open(src) as im:
                arr = np.asarray(im.convert("L"))
            lut = build_lut(np.bincount(arr.ravel(), minlength=256), ref_cdf)
            if np.any(np.diff(lut.astype(np.int32)) < 0):
                non_monotone += 1          # 원리상 0 이어야 한다. 세어서 보고한다.
            out = lut[arr]
            buf = BytesIO()
            Image.fromarray(out, mode="L").save(
                buf, format="JPEG", quality=quality, progressive=False, optimize=False)
            data = buf.getvalue()
            num = r.image_id.rsplit(":", 1)[-1]
            dst = OUT / r.modality / r.material / f"{num}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            fh.write(json.dumps({
                "image_id": r.image_id,
                "rel_path": dst.relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "width_px": int(arr.shape[1]), "height_px": int(arr.shape[0]),
            }, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                fh.flush()
            if k % 5000 == 0:
                rate = k / max(time.time() - t0, 1e-9)
                print(f"  {k:,}/{len(todo):,}  {rate:.0f}장/초  "
                      f"남은 시간 약 {(len(todo)-k)/max(rate,1e-9)/60:.0f}분", flush=True)
        fh.flush()
    print(f"완료 {len(todo):,}장 · {(time.time()-t0)/60:.1f}분 · "
          f"단조가 깨진 사상표 {non_monotone}건")
    return 0


def stage_manifest(manifest) -> int:
    """`rel_path`·`sha256` 만 갈아 끼운다. 분할·묶음·클래스는 손대지 않는다."""
    from data.invariants import check_invariants
    from data.label_map import load_label_map
    from data.manifest_io import FLOAT_FORMAT, read_annotations

    prog = load_progress()
    missing = [i for i in manifest["image_id"] if i not in prog]
    if missing:
        print(f"!! 정합이 안 된 이미지 {len(missing):,}장. --stage apply 를 끝내라.")
        return 3

    m = manifest.copy()
    frozen = ["split", "client", "group_id", "group_size", "eval_subset", "strata_key",
              "defect_types", "iso_codes", "has_defect", "material", "width_px", "height_px"]
    before = {c: m[c].astype(str).copy() for c in frozen}
    m["rel_path"] = m["image_id"].map(lambda i: prog[i]["rel_path"]).astype("string")
    m["sha256"] = m["image_id"].map(lambda i: prog[i]["sha256"]).astype("string")

    changed = [c for c in frozen if not before[c].eq(m[c].astype(str)).all()]
    if changed:
        print(f"!! 불변이어야 할 컬럼이 바뀌었다: {changed}")
        return 4
    print(f"불변 확인: {len(frozen)}개 컬럼 전부 동일 (split·client·group_id·클래스 포함)")

    lm = load_label_map()
    a = read_annotations(V1 / "annotations.csv")
    violations = check_invariants(m, a, lm, raw_root=REPO_ROOT)
    print(f"불변조건 위반 {len(violations)}건")
    for v in violations[:20]:
        print(f"   {v}")

    (V1 / "manifest_pre_histmatch.csv").write_bytes((V1 / "manifest.csv").read_bytes())
    out = m.sort_values("image_id", kind="stable").reset_index(drop=True)
    for col in out.columns:
        if str(out[col].dtype) == "boolean":
            out[col] = out[col].map({True: "True", False: "False"}).astype("string")
    with (V1 / "manifest.csv").open("w", encoding="utf-8", newline="\n") as fh:
        out.to_csv(fh, index=False, na_rep="", float_format=FLOAT_FORMAT,
                   lineterminator="\n")
    print(f"기록: {V1 / 'manifest.csv'}  ·  직전본은 manifest_pre_histmatch.csv")
    print("SNAPSHOT 은 잠그지 않았다.")
    return 7 if violations else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("calibrate", "apply", "manifest", "contrast"),
                    required=True)
    ap.add_argument("--quality", type=int, default=None)
    ap.add_argument("--per-pop", type=int, default=1200)
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data/interim/aihub_labels")
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
    if args.stage == "manifest":
        return stage_manifest(manifest)
    from scripts.measure_hist_match_contrast import run_contrast
    return run_contrast(args)


if __name__ == "__main__":
    raise SystemExit(main())
