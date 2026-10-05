"""mock 스냅샷 생성 — 학습·채점 코드의 선행 개발용 (열린질문 Q8, 검수 #4 승인).

계약 #2 스키마를 준수하는 매니페스트를 만든다. **실이미지는 없다.** 목적은
AI허브 승인(8/25)을 기다리지 않고 로더·채점기·학습 래퍼를 지금 짜는 것이다.

산출물은 불변식 IV1~IV11 을 전부 통과해야 한다(IV12 는 실파일이 없어 제외).

별도 자료 생성은 data/mock/README.md의 새 출력 폴더 확인 절차를 따르고 --out-root를 지정한다.
생성 전에 선택한 프로필의 출력 경로를 모두 확인한다. 하나라도 비어 있지 않은 디렉터리이거나
디렉터리 대신 파일이 놓여 있으면, 어느 프로필도 생성하지 않고 종료 코드 2로 끝난다.
없는 경로와 빈 디렉터리는 허용한다. 강제 덮어쓰기 옵션은 없다.
기본 출력 경로는 data/mock/이며, 현재 보존된 두 동결본은 거부 조건에 해당한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.convert.geometry import polygon_metrics, px_to_mm
from data.invariants import check_invariants
from data.label_map import load_label_map
from data.manifest_io import (
    ANNOTATION_COLUMNS,
    MANIFEST_COLUMNS,
    VerdictMode,
    write_snapshot,
)
from data.split.dirichlet import (
    DEFAULT_CONCENTRATION,
    partition_with_acceptance,
    rarest_first_order,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "configs" / "mock_profile.yaml"
OUT_ROOT = REPO_ROOT / "data" / "mock"
NORMAL = "__normal__"
KST = timezone(timedelta(hours=9))
#: mock 은 시각에 의존하지 않는다 — 재실행 시 바이트가 같아야 하므로 고정 타임스탬프를 쓴다.
FIXED_TIMESTAMP = datetime(2026, 8, 17, 0, 0, 0, tzinfo=KST).isoformat()


def _fake_sha256(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _fake_phash(seed_str: str) -> str:
    """256-bit pHash (hash_size=16) 자리 — 64 hex 문자."""
    return hashlib.sha256(("phash:" + seed_str).encode("utf-8")).hexdigest()


def _blob_polygon(rng: np.random.Generator, w: int, h: int) -> list[list[float]]:
    """이미지 안에 들어가는 볼록한 다각형 결함 하나."""
    r = float(rng.uniform(6, min(w, h) * 0.08))
    cx = float(rng.uniform(r + 2, w - r - 2))
    cy = float(rng.uniform(r + 2, h - r - 2))
    n = int(rng.integers(6, 12))
    angles = np.sort(rng.uniform(0, 2 * math.pi, size=n))
    radii = r * rng.uniform(0.7, 1.0, size=n)
    return [
        [round(float(cx + rad * math.cos(t)), 2), round(float(cy + rad * math.sin(t)), 2)]
        for t, rad in zip(angles, radii, strict=True)
    ]


def build_profile(
    name: str,
    profile: dict,
    grouping: dict,
    split_cfg: dict,
    seed: int,
    verdict_mode: VerdictMode,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    lm = load_label_map()
    rng = np.random.default_rng(seed)
    source = profile["source"]
    has_loc = bool(profile["has_localization"])

    # --- 1) 이미지 + 묶음 생성 -----------------------------------------------------
    images: list[dict] = []
    for material, spec in profile["materials"].items():
        weights = spec["class_weights"]
        keys = list(weights)
        probs = np.asarray([weights[k] for k in keys], dtype=float)
        probs = probs / probs.sum()

        remaining = int(spec["n_images"])
        group_seq = 0
        while remaining > 0:
            singleton = rng.random() < float(grouping["singleton_ratio"])
            size = 1 if singleton else int(rng.integers(2, int(grouping["max_group_size"]) + 1))
            size = min(size, remaining)
            remaining -= size

            # 같은 용접부이므로 묶음 안의 결함 구성은 동일하다
            picked = keys[int(rng.choice(len(keys), p=probs))]
            if picked == NORMAL:
                types: tuple[str, ...] = ()
            elif rng.random() < 0.15:
                other = [k for k in keys if k not in (NORMAL, picked)]
                types = tuple(sorted({picked, other[int(rng.integers(len(other)))]}))
            else:
                types = (picked,)

            group_key = f"{source}/{material}/w{group_seq:05d}"
            group_seq += 1
            for frame in range(size):
                rel = f"data/raw/{source}/{profile['modality']}/{material}/w{group_seq - 1:05d}_f{frame}.png"
                images.append(
                    {
                        "image_id": f"{source}:{rel}",
                        "source": source,
                        "rel_path": rel,
                        "sha256": _fake_sha256(rel),
                        "width_px": int(spec["width_px"]),
                        "height_px": int(spec["height_px"]),
                        "modality": profile["modality"],
                        "material": material,
                        "defect_type_list": types,
                        "_group_key": group_key,
                    }
                )

    m = pd.DataFrame(images)

    # group_id 는 내용에서만 유도한다 — 순회·삽입 순서와 무관 (§6-5).
    # 규칙은 data/dedup/phash.py 와 동일하다: grp_{재질}_{최소 sha256 앞 12자}
    gid = (
        m.groupby("_group_key")
        .apply(lambda g: f"grp_{g['material'].iloc[0]}_{g['sha256'].min()[:12]}",
               include_groups=False)
        .rename("group_id")
    )
    m = m.join(gid, on="_group_key")
    if m["group_id"].nunique() != m["_group_key"].nunique():
        raise AssertionError("group_id 충돌 — sha256 앞 16자가 겹쳤다")
    m["group_size"] = m.groupby("group_id")["image_id"].transform("size").astype(int)
    m["phash_hex"] = m["group_id"].map(_fake_phash)   # 같은 묶음은 같은 해시 근방

    # --- 2) 층화 키 — 재질 내 최희소 클래스, 빈도는 런타임 유도 (§6-6) ----------------
    iso = {k: v.iso_code for k, v in lm.defect_types.items()}
    exploded = m.explode("defect_type_list")
    freq_order: dict[str, tuple[str, ...]] = {}
    for material, part in exploded.groupby("material"):
        labels = part["defect_type_list"].fillna(NORMAL).tolist()
        freq_order[material] = rarest_first_order(labels, [1] * len(labels), iso)

    def representative(row) -> str:
        types = row["defect_type_list"]
        if not types:
            return NORMAL
        order = freq_order[row["material"]]
        return min(types, key=lambda k: order.index(k) if k in order else len(order))

    m["_repr"] = m.apply(representative, axis=1)
    m["strata_key"] = m["material"] + "|" + m["_repr"]

    # --- 3) 글로벌 평가셋 20% 선분리 (묶음 단위·층화) --------------------------------
    sgkf = StratifiedGroupKFold(
        n_splits=int(split_cfg["eval_folds"]), shuffle=True, random_state=seed + 2
    )
    folds = list(sgkf.split(m, y=m["strata_key"], groups=m["group_id"]))
    eval_idx = folds[0][1]
    m["split"] = "train"
    m.loc[m.index[eval_idx], "split"] = "eval"
    m["client"] = pd.NA

    # --- 4) 학습 풀 → 클라이언트. AL 은 자연 분할(C3), ST 는 Dirichlet(C1:C2) ---------
    pool = m.loc[m["split"] != "eval"]
    groups = (
        pool.groupby("group_id")
        .agg(material=("material", "first"), repr_=("_repr", "first"), size=("image_id", "size"))
        .sort_index()
    )
    dirichlet_meta: dict[str, object] = {}
    st = groups.loc[groups["material"] == "ST"]
    if len(st):
        order = rarest_first_order(
            st["repr_"].tolist(), st["size"].tolist(), iso
        )
        res = partition_with_acceptance(
            tuple(st.index),
            st["repr_"].to_numpy(dtype=object),
            st["size"].to_numpy(dtype=np.int64),
            concentration=DEFAULT_CONCENTRATION,
            seed=seed + 3,
            label_priority=order,
        )
        mapping = {g: ("C1" if a == 0 else "C2") for g, a in zip(st.index, res.assignment)}
        dirichlet_meta = {
            "concentration": list(DEFAULT_CONCENTRATION),
            "seed_used": res.seed_used,
            "attempts": res.attempts,
            "c1_share": round(res.c1_share, 4),
        }
    else:
        mapping = {}
    for g in groups.index[groups["material"] == "AL"]:
        mapping[g] = "C3"
    for g in groups.index[~groups["material"].isin(["ST", "AL"])]:
        # 재질 정보가 없는 백업 경로(RIAWELC) — 3-way Dirichlet 은 후속. mock 은 균등 순환.
        # 파이썬 hash() 는 프로세스마다 달라지므로 쓰지 않는다 (재현성).
        mapping[g] = ["C1", "C2", "C3"][int(hashlib.sha256(g.encode()).hexdigest(), 16) % 3]
    m.loc[m["split"] != "eval", "client"] = m.loc[m["split"] != "eval", "group_id"].map(mapping)

    # --- 5) 클라이언트 내 val 10% (묶음 단위) ---------------------------------------
    for client, part in m.loc[m["split"] != "eval"].groupby("client"):
        n_splits = min(int(split_cfg["val_folds"]), part["group_id"].nunique())
        if n_splits < 2:
            continue
        inner = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 4)
        try:
            _, val_idx = next(iter(inner.split(part, y=part["strata_key"], groups=part["group_id"])))
        except ValueError:
            continue
        m.loc[part.index[val_idx], "split"] = "val"

    # --- 6) 판정문 채점 표본 (분할 확정과 동시에 뽑는다 — 열린질문 Q4) ----------------
    eval_groups = sorted(m.loc[m["split"] == "eval", "group_id"].unique())
    k = max(1, int(len(eval_groups) * float(split_cfg["judgment_subset_ratio"])))
    picked = set(np.random.default_rng(seed + 5).choice(eval_groups, size=k, replace=False))
    m["eval_subset"] = pd.NA
    m.loc[m["group_id"].isin(picked) & (m["split"] == "eval"), "eval_subset"] = "judgment_2000"

    # --- 7) annotations -------------------------------------------------------------
    absolute = verdict_mode is VerdictMode.ABSOLUTE
    ann_rows: list[dict] = []
    thickness: list[float | None] = []
    scale: list[float | None] = []
    for row in m.itertuples():
        if absolute:
            t = round(float(rng.choice([8.0, 10.0, 12.0, 16.0, 20.0, 25.0])), 2)
            s = round(float(rng.uniform(4.0, 12.0)), 2)
        else:
            t, s = None, None
        thickness.append(t)
        scale.append(s)

        for seq, dtype in enumerate(row.defect_type_list):
            ann: dict[str, object] = {c: pd.NA for c in ANNOTATION_COLUMNS}
            ann.update(
                {
                    "ann_id": f"{row.image_id}#{seq}",
                    "image_id": row.image_id,
                    "src_label_raw": lm.defect_types[dtype].name_ko
                    if row.source == "aihub71761"
                    else next(k for k, v in lm.sources[row.source].mapping.items() if v == dtype),
                    "defect_type": dtype,
                    "iso_code": lm.defect_types[dtype].iso_code,
                    "geom_valid": True,
                    "geom_flags": "",
                }
            )
            if has_loc:
                poly = _blob_polygon(rng, row.width_px, row.height_px)
                g = polygon_metrics(poly, row.width_px, row.height_px)
                if not g.valid:
                    raise AssertionError(f"mock 폴리곤이 무효다: {poly}")
                x1, y1, x2, y2 = g.bbox_px
                ann.update(
                    {
                        "polygon_json": json.dumps(poly, separators=(",", ":")),
                        "bbox_x1_px": x1, "bbox_y1_px": y1, "bbox_x2_px": x2, "bbox_y2_px": y2,
                        "area_px": round(g.area_px, 2),
                        "major_axis_px": round(g.major_axis_px, 2),
                        "minor_axis_px": round(g.minor_axis_px, 2),
                        "equiv_diameter_px": round(g.equiv_diameter_px, 2),
                        "geom_flags": g.flags_str,
                    }
                )
                mm_major = px_to_mm(g.major_axis_px, s)
                mm_equiv = px_to_mm(g.equiv_diameter_px, s)
                if mm_major is not None:
                    ann["major_axis_mm"] = round(mm_major, 2)
                    ann["equiv_diameter_mm"] = round(mm_equiv, 2)
            ann_rows.append(ann)

    a = pd.DataFrame(ann_rows, columns=list(ANNOTATION_COLUMNS)) if ann_rows else pd.DataFrame(
        columns=list(ANNOTATION_COLUMNS)
    )

    # --- 8) manifest 마무리 ---------------------------------------------------------
    m["thickness_mm"] = thickness
    m["px_per_mm"] = scale
    m["thickness_source"] = "metadata" if absolute else "none"
    m["scale_source"] = "metadata" if absolute else "none"
    m["quality_level"] = "C" if absolute else pd.NA
    m["has_defect"] = m["defect_type_list"].map(bool)
    m["n_defects"] = m["defect_type_list"].map(len)
    m["defect_types"] = m["defect_type_list"].map(lambda t: ";".join(sorted(t)))
    m["iso_codes"] = m["defect_type_list"].map(
        lambda t: ";".join(sorted(lm.defect_types[k].iso_code for k in t))
    )
    m["src_labels_raw"] = a.groupby("image_id")["src_label_raw"].apply(
        lambda s: ";".join(sorted(s))
    ).reindex(m["image_id"]).fillna("").to_numpy()
    m["label_type"] = profile["label_type"]
    m["has_localization"] = has_loc
    m["ingest_version"] = profile["ingest_version"]
    m["label_map_version"] = lm.version
    m["notes"] = "MOCK — 실이미지 없음. 스키마 검증·선행 개발 전용"

    manifest = m.loc[:, list(MANIFEST_COLUMNS)].copy()

    caps = {
        "generated_at": FIXED_TIMESTAMP,
        "snapshot_id": name,
        "source": source,
        "is_mock": True,
        "counts": {
            "images_total": len(manifest),
            "with_thickness": int(manifest["thickness_mm"].notna().sum()),
            "with_pixel_scale": int(manifest["px_per_mm"].notna().sum()),
            "with_quality_level": int(manifest["quality_level"].notna().sum()),
        },
        "capabilities": {
            "localization": has_loc,
            "thickness_mm": absolute,
            "pixel_scale": absolute,
            "size_mm": absolute,
            "verdict_mode": verdict_mode.value,
        },
        "assumptions": {
            "thickness_mm": None, "px_per_mm": None, "quality_level": None, "rationale": None,
        },
        "split_meta": {
            "seed": seed,
            "eval_folds": int(split_cfg["eval_folds"]),
            "val_folds": int(split_cfg["val_folds"]),
            "dirichlet": dirichlet_meta or None,
        },
    }
    return manifest, a, caps


#: 출력 경로가 비어 있지 않아 거부할 때의 종료 코드. 불변식 위반(1)과 구별한다.
EXIT_OCCUPIED = 2

#: 거부 메시지에 보여 주는 항목 수. 그 디렉터리에 무엇이 있는지 알려 주되 목록으로 화면을 덮지 않는다.
SHOWN_ENTRIES = 5


def occupied_outputs(out_root: Path, names: list[str]) -> list[tuple[Path, list[str]]]:
    """쓰려는 스냅샷 디렉터리 가운데 **이미 있고 비어 있지 않은** 곳. `(경로, 그 안의 이름들)` 목록.

    기준은 "생성기가 쓰는 네 파일이 있는가" 가 아니다.
    알려진 산출물 외의 파일이나 하위 디렉터리가 있어도 기존 내용을 보존하기 위해 거부한다.
    없는 경로와 빈 디렉터리는 허용한다. 그 이름으로 **파일**이 놓여 있어도 거부한다.

    기본 `--out-root` 는 동결된 `data/mock/` 이다(개발규약 1-6 — 재생성 금지). 인자 없이 한 번 돌리면
    봉인본이 덮였고, `mock_riawelc_v1` 은 사상표 표기가 바뀐 뒤라 바이트까지 달라진다(65번 §8~§11).
    **강제 옵션은 없다** — 다시 만들 일이 있으면 새 경로에 만든다.
    """
    found = []
    for name in names:
        d = Path(out_root) / name
        if d.is_dir():
            inside = sorted(e.name for e in d.iterdir())
            if inside:
                found.append((d, inside))
        elif d.exists():
            found.append((d, ["(디렉터리가 아니라 파일이다)"]))
    return found


def main(argv: list[str] | None = None) -> int:
    cfg = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=[*cfg["profiles"], "all"], default="all")
    ap.add_argument("--verdict-mode", choices=[v.value for v in VerdictMode], default=None,
                    help="프로파일 기본값을 덮어쓴다. absolute 로 두면 두께·스케일이 채워진다")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args(argv)

    names = list(cfg["profiles"]) if args.profile == "all" else [args.profile]

    # 하나라도 걸리면 **아무것도 만들기 전에** 멈춘다 — 앞 프로필만 쓰인 반쯤 상태를 남기지 않는다.
    blocked = occupied_outputs(args.out_root, names)
    if blocked:
        print("비어 있지 않은 경로에는 쓰지 않는다 — 아무것도 쓰지 않았다:", file=sys.stderr)
        for d, inside in blocked:
            more = f" 외 {len(inside) - SHOWN_ENTRIES}개" if len(inside) > SHOWN_ENTRIES else ""
            print(f"  {d}  ({', '.join(inside[:SHOWN_ENTRIES])}{more})", file=sys.stderr)
        print("새 `--out-root` 를 지정하라. 강제 덮어쓰기 옵션은 없다.", file=sys.stderr)
        return EXIT_OCCUPIED
    lm = load_label_map()
    rc = 0
    for name in names:
        profile = cfg["profiles"][name]
        mode = VerdictMode(args.verdict_mode or profile["verdict_mode"])
        manifest, annotations, caps = build_profile(
            name, profile, cfg["grouping"], cfg["split"], int(cfg["seed"]), mode
        )
        violations = check_invariants(manifest, annotations, lm)
        if violations:
            print(f"[{name}] 불변식 위반 — 스냅샷을 쓰지 않는다:")
            for x in violations:
                print(f"  {x}")
            rc = 1
            continue
        out = Path(args.out_root) / name
        digest = write_snapshot(out, manifest, annotations, caps)
        by_split = manifest["split"].value_counts().to_dict()
        by_client = manifest["client"].value_counts(dropna=True).to_dict()
        print(f"[{name}] {len(manifest)}행 / 결함 {len(annotations)}건 → {out}")
        print(f"  snapshot_digest {digest}")
        print(f"  split {by_split} / client {by_client}")
        print(f"  verdict_mode {caps['capabilities']['verdict_mode']} "
              f"localization {caps['capabilities']['localization']}")
        if caps["split_meta"]["dirichlet"]:
            print(f"  dirichlet {caps['split_meta']['dirichlet']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
