"""타일링 본 실행 — 재인코딩과 매니페스트 v1 (실행 순서 ④⑤⑥).

`tiling.py` 가 규칙이고 이 모듈이 실행이다. 규칙은 픽셀을 읽지 않고, 실행은 규칙이 정한
상자를 그대로 따른다. 실행 쪽에서 상자를 다시 정하지 않는다.

**원본을 고치지 않는다.** zip 은 읽기만 하고 결과는 새 경로에만 쓴다(불변조건 1).
**잠그지 않는다.** SNAPSHOT 잠금은 프로브 판정 뒤 총괄이 연다.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import yaml

from data.convert.tiling import encode_tile
from data.frozen_guard import assert_writable

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 진행 기록을 디스크에 밀어내는 주기(장). 죽었을 때 잃는 최대치가 이 값이다.
FLUSH_EVERY = 200
PROGRESS_EVERY = 2000


def repo_rel(path: Path) -> str:
    """저장소 기준 상대 경로. **정션을 따라가지 않는다.**

    `data/interim` 은 본체와 공유하는 정션이라 `resolve()` 를 쓰면 워크트리 밖 실경로가
    나오고 상대 경로 계산이 깨진다. `abspath` 는 링크를 풀지 않고 정규화만 한다.

    저장소 밖 경로는 절대 경로 그대로 돌려준다. 여기서 예외를 던지면 6만 장짜리 작업이
    경로 하나 때문에 죽는다.
    """
    abs_path = Path(os.path.abspath(path))
    try:
        return abs_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return abs_path.as_posix()


def load_progress(path: Path) -> dict[str, dict]:
    """진행 기록을 읽는다. **기록에 있고 파일도 실제로 있는 것만** 끝난 것으로 친다.

    기록만 있고 파일이 없으면 쓰다가 죽은 것이므로 다시 뜬다.
    """
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if (REPO_ROOT / rec["rel_path"]).exists():
            done[rec["image_id"]] = rec
    return done


def encode_all(rows, plan_by_id, members, file_names, spec, args) -> int:
    """계획대로 잘라 새 경로에 다시 인코딩한다. 중간에 죽어도 재개된다.

    zip 은 한 번씩만 연다. 6만 장을 장마다 여닫으면 그 비용이 인코딩보다 커진다.
    """
    # --manifest-out 을 동결본으로 겨누면 정본이 사라진다. 진입에서 막는다 (80번 G11-1).
    assert_writable(args.manifest_out, what="--manifest-out 대상")
    args.manifest_out.mkdir(parents=True, exist_ok=True)
    prog_path = args.manifest_out / "encode_progress.jsonl"

    done = load_progress(prog_path)
    if done:
        print(f"\n재개: 이미 끝난 {len(done):,}장을 건너뛴다")

    todo, missing = [], []
    for row in rows:
        plan = plan_by_id[row.image_id]
        if not plan.keep or row.image_id in done:
            continue
        # 원천은 파일명으로 찾는다. id 로 찾으면 id≠파일명인 10장을 잃는다.
        fname = file_names.get(row.image_id)
        if not fname or fname not in members:
            missing.append(f"{row.image_id} (file_name={fname!r})")
            continue
        zp, name = members[fname]
        # 출력 이름은 매니페스트 키(id)를 쓴다. 원천 파일명은 `RT_ST_01_...` 처럼
        # 클래스를 이름에 담고 있어 그대로 옮기면 파일명 자체가 지름길이 된다.
        num = row.image_id.rsplit(":", 1)[-1]
        dst = args.out / row.modality / row.material / f"{num}.jpg"
        todo.append((zp, name, row, plan, dst))

    if missing:
        # 조용히 건너뛰면 폐기 회계가 틀렸는데도 맞는 것처럼 보인다.
        print(f"!! 원천 zip 에서 못 찾은 이미지 {len(missing):,}장. 중단한다.")
        for x in missing[:10]:
            print(f"     {x}")
        return 4

    print(f"재인코딩 대상 {len(todo):,}장 → {args.out}")
    by_zip: dict[Path, list] = defaultdict(list)
    for item in todo:
        by_zip[item[0]].append(item)

    n_ok, errors = 0, []
    t0 = time.time()
    with prog_path.open("a", encoding="utf-8", newline="\n") as fh:
        for zp in sorted(by_zip):
            with zipfile.ZipFile(zp) as z:
                for _, name, row, plan, dst in by_zip[zp]:
                    try:
                        sha, w, h = encode_tile(
                            BytesIO(z.read(name)), plan.box, dst,
                            mode=spec["mode"], quality=spec["quality"],
                            progressive=spec["progressive"], optimize=spec["optimize"],
                        )
                    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 세고 보고한다
                        errors.append(
                            f"{row.image_id} ({zp.name}/{name}): {type(exc).__name__} {exc}")
                        continue
                    fh.write(json.dumps({
                        "image_id": row.image_id,
                        "rel_path": repo_rel(dst),
                        "sha256": sha, "width_px": w, "height_px": h,
                        "reason": plan.reason, "box": list(plan.box),
                    }, ensure_ascii=False) + "\n")
                    n_ok += 1
                    if n_ok % FLUSH_EVERY == 0:
                        fh.flush()
                    if n_ok % PROGRESS_EVERY == 0:
                        rate = n_ok / max(time.time() - t0, 1e-9)
                        left = (len(todo) - n_ok) / max(rate, 1e-9) / 60
                        print(f"  {n_ok:,}/{len(todo):,}  {rate:.0f}장/초  "
                              f"남은 시간 약 {left:.0f}분", flush=True)
        fh.flush()

    print(f"재인코딩 완료 {n_ok:,}장 · 실패 {len(errors):,}장 · {(time.time()-t0)/60:.1f}분")
    if errors:
        print("!! 실패분이 있다. 매니페스트 v1 을 만들지 않고 중단한다.")
        for e in errors[:20]:
            print(f"     {e}")
        (args.manifest_out / "encode_errors.txt").write_text(
            "\n".join(errors) + "\n", encoding="utf-8", newline="\n")
        return 5
    return 0


def build_v1(args, spec, plan_by_id, report) -> int:
    # 매니페스트·분할 메타를 쓴다. 동결본을 겨누면 여기서 멈춘다 (80번 G11-1).
    assert_writable(args.manifest_out, what="--manifest-out 대상")
    """매니페스트 v1 + 최종 분할. **잠그지 않는다.**"""
    from data.invariants import check_invariants
    from data.label_map import load_label_map
    from data.manifest_io import FLOAT_FORMAT, read_annotations, read_manifest
    from data.split.pipeline import assign_splits, distribution_heatmap

    prog = load_progress(args.manifest_out / "encode_progress.jsonl")
    # 계약 로더로 읽는다. 전부 문자열로 읽으면 bbox 가 문자열이 되어 불변조건 검사가
    # 숫자 비교에서 깨진다. 컬럼 순서·dtype 은 계약이 정한다.
    m = read_manifest(args.v0 / "manifest.csv")
    a = read_annotations(args.v0 / "annotations.csv")
    n0, na0 = len(m), len(a)

    # 좌표 안전장치. 결함 인스턴스를 가진 이미지는 전부 전체 프레임 통과여야 한다.
    # 아니면 bbox 를 옮겨야 하는데 그 경로는 확정 규격에 없다. 만들지 않고 멈춘다.
    full = (0, 0, spec["tile_w"], spec["tile_h"])
    shifted = sorted({i for i in set(a["image_id"]) & set(prog)
                      if plan_by_id[i].box != full})
    if shifted:
        print(f"!! 결함 인스턴스가 있는데 잘린 이미지 {len(shifted):,}장. "
              "bbox 재계산 경로가 규격에 없다. 중단한다.")
        for x in shifted[:10]:
            print(f"     {x}")
        return 6

    m = m[m["image_id"].isin(prog)].reset_index(drop=True)
    a = a[a["image_id"].isin(prog)].reset_index(drop=True)
    for col, dtype in (("rel_path", "string"), ("sha256", "string"),
                       ("width_px", "Int64"), ("height_px", "Int64")):
        m[col] = m["image_id"].map(lambda i, k=col: prog[i][k]).astype(dtype)

    # 폐기로 묶음에서 빠진 장이 있으므로 크기를 다시 센다. **묶음 자체는 쪼개지 않는다.**
    # 연속 id 한가운데가 빠져도 같은 촬영 묶음인 것은 그대로다. 여기서 쪼개면 한 묶음이
    # 학습과 평가로 갈려 누수가 된다. 묶음은 합쳐질 뿐 쪼개지지 않는다.
    m["group_size"] = m.groupby("group_id")["image_id"].transform("size").astype("Int64")
    print(f"\n매니페스트 v1: 이미지 {n0:,} → {len(m):,} (폐기 {n0-len(m):,}) · "
          f"결함 인스턴스 {na0:,} → {len(a):,}")

    cfg = yaml.safe_load((REPO_ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    seed = int(cfg["split"]["seed"])
    lm = load_label_map()
    iso = {k: v.iso_code for k, v in lm.defect_types.items()}

    print(f"\n최종 분할 (동일 시드 {seed}, 1회)")
    m, meta = assign_splits(m, iso, seed)
    print(f"   split {m['split'].value_counts().to_dict()}")
    print(f"   client {m['client'].value_counts(dropna=True).to_dict()}")
    if meta.dirichlet:
        print(f"   dirichlet {meta.dirichlet}")
    for line in meta.band_report:
        print(f"   [밴드 보고] {line}")

    violations = check_invariants(m, a, lm, raw_root=REPO_ROOT)
    print(f"\n불변조건 위반 {len(violations)}건")
    for v in violations[:20]:
        print(f"   {v}")

    for df, name, sort_col in ((m, "manifest.csv", "image_id"),
                               (a, "annotations.csv", "ann_id")):
        out = df.sort_values(sort_col, kind="stable").reset_index(drop=True)
        for col in out.columns:
            if str(out[col].dtype) == "boolean":
                out[col] = out[col].map({True: "True", False: "False"}).astype("string")
        with (args.manifest_out / name).open("w", encoding="utf-8", newline="\n") as fh:
            out.to_csv(fh, index=False, na_rep="", float_format=FLOAT_FORMAT,
                       lineterminator="\n")

    with (args.manifest_out / "distribution.csv").open("w", encoding="utf-8", newline="\n") as fh:
        distribution_heatmap(m).to_csv(fh, index=True, lineterminator="\n")

    with (args.manifest_out / "split_meta.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump({
            "seed": meta.seed, "eval_folds": meta.eval_folds, "val_folds": meta.val_folds,
            "dirichlet": meta.dirichlet, "band_report": list(meta.band_report),
            "tile_rule_version": cfg["preprocess"]["tile"]["rule_version"],
            "encode": cfg["preprocess"]["tile"]["encode"],
            "images": len(m), "annotations": len(a),
            "discarded": n0 - len(m),
            "invariant_violations": [str(v) for v in violations],
            "accounting": report["cells"],
        }, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"\n기록: {args.manifest_out}")
    print("SNAPSHOT 은 잠그지 않았다. 프로브 판정 뒤 총괄이 연다.")
    return 7 if violations else 0
