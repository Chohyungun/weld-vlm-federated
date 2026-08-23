"""ingest → dedup → split → 잠금 전 구간 테스트. 스펙 §7-3 + §6-11 테스트 13·14·24.

실제 PNG 파일을 만들어 돌린다 — pHash·해시·불변식이 전부 실물 기준으로 검증된다.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from data.dedup.phash import build_groups, compute_phash
from data.ingest.base import records_to_frames, resolve_image_path
from data.ingest.riawelc import RiawelcAdapter
from data.invariants import check_invariants
from data.label_map import UnmappedLabelError, load_label_map
from data.manifest_io import (
    ANNOTATION_COLUMNS,
    MANIFEST_COLUMNS,
    VerdictMode,
    join_defects,
    load_snapshot,
    write_snapshot,
)
from data.split.pipeline import (
    JUDGMENT_SUBSET,
    SPLIT_EVAL,
    assign_clients,
    assign_splits,
    distribution_heatmap,
    sample_judgment_groups,
)

#: 실물 클래스 폴더명 (이탈리아어 Difetto). Difetto3 은 원본에 없다.
CLASSES = ("Difetto1", "Difetto2", "Difetto4", "NoDifetto")
#: 저자 제공 분할. 분할 정보로 쓰지 않는다 — 경로의 일부일 뿐이다.
AUTHOR_SPLITS = ("training", "testing")
SEED = 20260825


@pytest.fixture(scope="module")
def lm():
    return load_label_map()


@pytest.fixture(scope="module")
def raw_root(tmp_path_factory):
    """RIAWELC 실물 형상 — `dataset/DB - Copy/{저자분할}/{클래스}/*.png`.

    실물 그대로 **클래스 디렉터리가 저자 분할 아래**에 있고, 파일명은 타일 패턴
    `{모원본}_[행][열].png` 다. 모원본 1개가 여러 저자 분할에 걸치게 만들어,
    저자 분할을 따라가면 누수가 나는 상황을 재현한다.
    """
    root = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(SEED)
    for kind in CLASSES:
        base_img = {}
        for mother in range(8):
            base_img[mother] = rng.normal(130, 12, (227, 227))
        for mother in range(8):
            for tile in range(2):
                # 같은 모원본의 타일을 서로 다른 저자 분할에 흩뿌린다
                split = AUTHOR_SPLITS[tile % len(AUTHOR_SPLITS)]
                d = root / "riawelc" / "dataset" / "DB - Copy" / split / kind
                d.mkdir(parents=True, exist_ok=True)
                arr = np.clip(
                    base_img[mother] + rng.normal(0, 1.0, (227, 227)), 0, 255
                ).astype(np.uint8)
                name = f"bam{mother}_Img1_A80_S1_[{tile}][3].png"
                Image.fromarray(arr, mode="L").save(d / name)
    return root


@pytest.fixture(scope="module")
def ingested(raw_root, lm):
    adapter = RiawelcAdapter()
    records = [
        adapter.parse(item, lm)
        for item in adapter.discover(raw_root, rel_base=raw_root)
    ]
    return adapter, records


def test_discover_finds_all_and_reads_only(ingested, raw_root) -> None:
    _, records = ingested
    assert len(records) == len(CLASSES) * 8 * 2
    # 원본이 그대로인지 — ingest 는 읽기만 한다
    assert sum(1 for _ in (raw_root / "riawelc").rglob("*.png")) == len(records)


def test_riawelc_records_have_no_localization(ingested) -> None:
    _, records = ingested
    assert all(not r.has_localization for r in records)
    assert all(r.material == "UNK" for r in records)
    for r in records:
        for d in r.defects:
            assert d.bbox_px is None and d.polygon is None and d.area_px is None


def test_normal_class_yields_zero_defects(ingested, lm) -> None:
    """ND = 정상 → annotations 행 0개. N1(위치 없음)과 N2(결함 없음)가 갈린다."""
    _, records = ingested
    nd = [r for r in records if "/NoDifetto/" in r.rel_path]
    assert nd and all(not r.defects for r in nd)
    cr = [r for r in records if "/Difetto1/" in r.rel_path]
    assert cr and all(len(r.defects) == 1 for r in cr)


def test_unknown_class_dir_fails_loudly(raw_root, lm, tmp_path) -> None:
    import shutil

    alt = tmp_path / "raw2"
    shutil.copytree(raw_root, alt)
    bad = alt / "riawelc" / "dataset" / "DB - Copy" / "training" / "UNDERCUT"
    bad.mkdir()
    shutil.copy(next((alt / "riawelc").rglob("*.png")), bad / "x.png")
    adapter = RiawelcAdapter()
    with pytest.raises(UnmappedLabelError):
        for item in adapter.discover(alt, rel_base=alt):
            adapter.parse(item, lm)


def test_capabilities_default_is_clause_only(ingested) -> None:
    """두께·스케일이 없으면 안전 기본값 clause_only. ingest 가 스스로 올리지 않는다."""
    adapter, records = ingested
    caps = adapter.capabilities(records)
    assert caps.verdict_mode is VerdictMode.CLAUSE_ONLY
    assert caps.localization is False
    assert caps.counts["images_total"] == len(records)


def test_frames_to_columns(ingested, lm) -> None:
    _, records = ingested
    m, a = records_to_frames(records, lm)
    assert tuple(m.columns) == MANIFEST_COLUMNS
    assert tuple(a.columns) == ANNOTATION_COLUMNS
    # 묶음·분할 컬럼은 후속 단계가 채운다
    assert m["group_id"].isna().all() and m["split"].isna().all()


@pytest.fixture(scope="module")
def grouped(ingested, lm, raw_root):
    _, records = ingested
    m, a = records_to_frames(records, lm)
    hexes = [compute_phash(resolve_image_path(p, raw_root)) for p in m["rel_path"]]
    m["phash_hex"] = hexes
    res = build_groups(
        m["image_id"].tolist(), m["sha256"].tolist(), hexes, m["material"].tolist(), threshold=32
    )
    m["group_id"] = list(res.group_ids)
    m["group_size"] = m.groupby("group_id")["image_id"].transform("size").astype(int)
    return m, a, res


def test_split_requires_groups(ingested, lm) -> None:
    _, records = ingested
    m, _ = records_to_frames(records, lm)
    with pytest.raises(ValueError, match="dedup"):
        assign_splits(m, {}, SEED)


@pytest.fixture(scope="module")
def splitted(grouped, lm):
    m, a, _ = grouped
    iso = {k: v.iso_code for k, v in lm.defect_types.items()}
    return (*assign_splits(m, iso, SEED), a)


def test_all_invariants_pass_end_to_end(splitted, lm, raw_root) -> None:
    m, _meta, a = splitted
    violations = check_invariants(m, a, lm, raw_root=raw_root, hash_sample_frac=1.0)
    assert not violations, "\n".join(str(v) for v in violations)


def test_eval_taken_before_clients(splitted, lm) -> None:
    """평가셋 선분리가 먼저다 — 함수 합성으로 강제되는지 직접 확인."""
    m, _meta, _a = splitted
    iso = {k: v.iso_code for k, v in lm.defect_types.items()}
    with pytest.raises(ValueError, match="선분리가 먼저"):
        assign_clients(m, iso, SEED)          # 평가셋이 섞인 프레임을 넣으면 거부


def test_no_group_straddles_split_or_client(splitted) -> None:
    m, _meta, _a = splitted
    assert (m.groupby("group_id")["split"].nunique() == 1).all()
    assert (m.assign(c=m["client"].fillna("EV")).groupby("group_id")["c"].nunique() == 1).all()


def test_judgment_subset_is_stratified_two_axis(splitted) -> None:
    """게이트 #6 결정 D — 층화는 재질 × 클래스 2축, 합부 축 없음."""
    m, _meta, _a = splitted
    picked = sample_judgment_groups(m, 0.5, SEED + 5)
    ev = m.loc[m["split"] == SPLIT_EVAL]
    strata_in_eval = set(ev["strata_key"].astype(str))
    strata_picked = set(
        ev.loc[ev["group_id"].isin(picked), "strata_key"].astype(str)
    )
    # 층이 통째로 사라지지 않는다 (소수 클래스 보호)
    assert strata_picked == strata_in_eval
    # eval 밖으로 새지 않는다
    assert (m.loc[m["eval_subset"].notna(), "split"] == SPLIT_EVAL).all()
    assert set(m["eval_subset"].dropna().unique()) <= {JUDGMENT_SUBSET}


def test_deterministic_rerun(grouped, lm) -> None:
    m, _a, _res = grouped
    iso = {k: v.iso_code for k, v in lm.defect_types.items()}
    a1, _ = assign_splits(m, iso, SEED)
    a2, _ = assign_splits(m, iso, SEED)
    for col in ("split", "client", "eval_subset", "strata_key"):
        assert a1[col].astype("string").equals(a2[col].astype("string"))


def test_snapshot_roundtrip_and_join(splitted, tmp_path) -> None:
    m, meta, a = splitted
    caps = {
        "generated_at": "2026-08-17T00:00:00+09:00",
        "snapshot_id": "test_riawelc",
        "source": "riawelc",
        "is_mock": False,
        "counts": {"images_total": len(m), "with_thickness": 0,
                   "with_pixel_scale": 0, "with_quality_level": 0},
        "capabilities": {"localization": False, "thickness_mm": False, "pixel_scale": False,
                         "size_mm": False, "verdict_mode": "clause_only"},
        "assumptions": {"thickness_mm": None, "px_per_mm": None,
                        "quality_level": None, "rationale": None},
        "split_meta": {"seed": meta.seed, "eval_folds": meta.eval_folds,
                       "val_folds": meta.val_folds, "dirichlet": meta.dirichlet},
    }
    root = tmp_path / "snap"
    write_snapshot(root, m, a, caps)
    snap = load_snapshot(root)
    assert snap.can_score("map") is False        # N1 — 위치 지표 산출 불가
    joined = join_defects(snap)
    n_normal = int((~snap.manifest["has_defect"].astype(bool)).sum())
    assert len(joined) == len(snap.annotations) + n_normal


def test_heatmap_uses_original_classes(splitted) -> None:
    m, _meta, _a = splitted
    heat = distribution_heatmap(m)
    assert int(heat.to_numpy().sum()) >= len(m)
    types = {t for _, t in heat.index}
    assert "crack" in types and "__normal__" in types


# ---- 타일 묶음(E2) 과 저자 분할 무시 -------------------------------------------------


def test_mother_image_key_parsing() -> None:
    """파일명 접두사 = 모원본. ` - Copia` 접미를 흡수하지 않으면 30건이 독립 묶음이 된다."""
    from data.ingest.riawelc import mother_image_key

    assert mother_image_key("bam5_Img2_A80_S5_[3][10].png") == "bam5_Img2_A80_S5"
    assert mother_image_key("RRT-40R_Img3_A80_S1_[10][13] - Copia.png") == "RRT-40R_Img3_A80_S1"
    # 같은 모원본의 원본 타일과 사본 타일이 같은 키로 접힌다
    assert mother_image_key("RRT-40R_Img3_A80_S1_[1][2].png") == mother_image_key(
        "RRT-40R_Img3_A80_S1_[10][13] - Copia.png"
    )
    assert mother_image_key("nontile.png") is None


def test_adapter_emits_group_key(ingested) -> None:
    _, records = ingested
    assert all(r.group_key for r in records)
    # 모원본 8개 × 클래스 4종 = 32 (타일 2장씩 접힌다)
    assert len({(r.rel_path.split("/")[-2], r.group_key) for r in records}) == len(CLASSES) * 8


def test_tiles_of_same_mother_group_together(ingested, lm, raw_root) -> None:
    """같은 모원본의 타일은 저자 분할이 달라도 한 묶음이어야 한다 — 갈리면 누수다."""
    _, records = ingested
    m, _ = records_to_frames(records, lm)
    hexes = [compute_phash(resolve_image_path(p, raw_root)) for p in m["rel_path"]]
    res = build_groups(
        m["image_id"].tolist(), m["sha256"].tolist(), hexes, m["material"].tolist(),
        threshold=0,                                   # pHash 를 끄고 E2 만으로
        meta_keys=[r.group_key for r in records],
    )
    m = m.assign(group_id=list(res.group_ids))
    by_mother = m.assign(mother=[r.group_key for r in records]).groupby("mother")["group_id"]
    assert (by_mother.nunique() == 1).all(), "같은 모원본의 타일이 여러 묶음으로 갈렸다"
    assert res.edge_counts["E2"] > 0


def test_author_split_dirs_never_become_split_values(splitted) -> None:
    """저자 분할(training/testing)은 경로일 뿐 split 컬럼에 들어가면 안 된다."""
    m, _meta, _a = splitted
    assert set(m["split"].unique()) <= {"train", "val", "eval"}
    # 원본 경로에는 저자 분할이 남아 있다 — 그런데도 split 에는 안 들어갔다
    assert m["rel_path"].str.contains("training").any()


# ---- 바이트 동일 복제본 (RIAWELC testing = training 복제) -----------------------------


def test_drop_exact_duplicates_keeps_one(ingested, lm) -> None:
    """복제본은 하나만 남고, 남는 쪽은 rel_path 사전순 첫 번째로 결정론적이다."""
    from data.ingest.base import drop_exact_duplicates

    _, records = ingested
    dup = records[0].model_copy(update={
        "image_id": records[0].image_id + "_copy",
        "rel_path": records[0].rel_path.replace("/training/", "/testing/"),
    })
    kept, dropped = drop_exact_duplicates([*records, dup])
    assert len(kept) == len(records)
    assert len(dropped) == 1
    # 사전순으로 testing < training 이므로 testing 쪽이 남는다 — 순서가 아니라 규칙이 정한다
    assert {r.rel_path for r in kept} | {r.rel_path for r in dropped} == {
        r.rel_path for r in [*records, dup]
    }
    kept2, _ = drop_exact_duplicates([dup, *records])      # 입력 순서 뒤집어도 동일
    assert [r.rel_path for r in kept] == [r.rel_path for r in kept2]


def test_byte_identical_images_land_in_one_group(ingested, lm, raw_root) -> None:
    """복제본을 굳이 남기더라도 E1 엣지가 같은 묶음으로 묶어 누수를 막는다."""
    _, records = ingested
    m, _ = records_to_frames(records, lm)
    hexes = [compute_phash(resolve_image_path(p, raw_root)) for p in m["rel_path"]]
    ids = m["image_id"].tolist() + ["dup:x"]
    shas = m["sha256"].tolist() + [m["sha256"].iloc[0]]     # 첫 이미지와 바이트 동일
    hx = hexes + [hexes[0]]
    mats = m["material"].tolist() + ["UNK"]
    res = build_groups(ids, shas, hx, mats, threshold=0)
    gid = dict(zip(ids, res.group_ids))
    assert gid["dup:x"] == gid[m["image_id"].iloc[0]]
    assert res.edge_counts["E1"] >= 1
