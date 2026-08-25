"""P2·P3 실행부 테스트. `40_규격지름길_대응판정.md` §5-2.

GPU 없이 **판정 배관**을 검증한다. 학습 자체는 주입된 `score_fn` 으로 대체한다.
핵심은 두 가지다: 묶음 단위 홀드아웃이 실제로 누수를 막는가, 평가셋이 섞여 들어가지 않는가.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from evaluation.probes.source_probe import (
    BAND,
    CROP,
    TILE,
    ProbeRow,
    bootstrap_auc_ci,
    downscale,
    group_holdout,
    load_provenance,
    load_rows,
    patch_shuffle,
    roc_auc,
    run_source_probe,
    summarize_sources,
)

MANIFEST_HEADER = "image_id,rel_path,modality,split,group_id,iso_codes\n"


def write_manifest(tmp_path, rows):
    p = tmp_path / "manifest.csv"
    p.write_text(MANIFEST_HEADER + "".join(rows), encoding="utf-8")
    return p


def write_provenance(tmp_path, items):
    p = tmp_path / "encode_progress.jsonl"
    p.write_text(
        "".join(json.dumps({"image_id": i, "reason": r}) + "\n" for i, r in items),
        encoding="utf-8",
    )
    return p


def row(iid, group, source, codes=()):
    return ProbeRow(iid, f"p/{iid}.jpg", group, source, tuple(codes))


# --- 출처 라벨 ------------------------------------------------------------------

def test_provenance_maps_reasons_to_sources(tmp_path):
    p = write_provenance(tmp_path, [("a", "ok"), ("b", "tiled"),
                                    ("c", "oversized_band_cropped")])
    assert load_provenance(p) == {"a": CROP, "b": TILE, "c": BAND}


def test_unknown_reason_is_dropped_not_guessed(tmp_path):
    p = write_provenance(tmp_path, [("a", "ok"), ("x", "누락사유")])
    assert load_provenance(p) == {"a": CROP}


# --- 평가셋 격리 ----------------------------------------------------------------

def test_eval_split_never_enters_the_probe(tmp_path):
    """학습 풀 내부 홀드아웃에서만 학습·채점한다. 평가셋은 건드리지 않는다."""
    m = write_manifest(tmp_path, [
        "a,p/a.jpg,RT,train,g1,2011\n",
        "b,p/b.jpg,RT,eval,g2,\n",
        "c,p/c.jpg,RT,val,g3,\n",
    ])
    prov = {"a": CROP, "b": TILE, "c": TILE}
    rows, unmatched = load_rows(m, prov)
    assert {r.image_id for r in rows} == {"a", "c"}
    assert unmatched == []


def test_unmatched_images_are_counted_not_silently_dropped(tmp_path):
    m = write_manifest(tmp_path, ["a,p/a.jpg,RT,train,g1,\n", "z,p/z.jpg,RT,train,g2,\n"])
    rows, unmatched = load_rows(m, {"a": CROP})
    assert len(rows) == 1 and unmatched == ["z"]


def test_other_modality_excluded(tmp_path):
    m = write_manifest(tmp_path, ["a,p/a.jpg,VT,train,g1,\n"])
    rows, _ = load_rows(m, {"a": CROP})
    assert rows == []


# --- 묶음 단위 홀드아웃 -----------------------------------------------------------

def test_holdout_never_splits_a_group():
    """이미지 단위로 뜨면 같은 용접부가 양쪽에 들어가 AUC 가 부풀려진다.
    그러면 지름길이 없는데 있다고 보고하게 된다."""
    rows = [row(f"i{g}_{k}", f"g{g}", TILE) for g in range(20) for k in range(3)]
    train, test = group_holdout(rows)
    assert {r.group_id for r in train} & {r.group_id for r in test} == set()


def test_holdout_is_deterministic():
    rows = [row(f"i{g}", f"g{g}", CROP) for g in range(50)]
    a = {r.image_id for r in group_holdout(rows)[1]}
    b = {r.image_id for r in group_holdout(rows)[1]}
    assert a == b


def test_holdout_covers_every_row():
    rows = [row(f"i{g}", f"g{g}", CROP) for g in range(50)]
    train, test = group_holdout(rows)
    assert len(train) + len(test) == len(rows)


def test_holdout_never_empty():
    _train, test = group_holdout([row("i0", "g0", CROP), row("i1", "g1", TILE)])
    assert test


# --- AUC ------------------------------------------------------------------------

def test_auc_perfect_separation():
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_auc_reversed_separation():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_auc_all_tied_is_half():
    """전부 같은 점수면 판별력이 없다."""
    assert roc_auc([0.5] * 4, [0, 0, 1, 1]) == pytest.approx(0.5)


def test_auc_single_class_is_half_not_error():
    assert roc_auc([0.1, 0.9], [1, 1]) == pytest.approx(0.5)


# --- 클러스터 부트스트랩 CI -------------------------------------------------------

def test_group_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    labels = [i % 2 for i in range(60)]
    scores = [l + rng.normal(0, 0.5) for l in labels]
    groups = [f"g{i // 3}" for i in range(60)]
    lo, hi = bootstrap_auc_ci(scores, labels, groups, n_resamples=200)
    assert lo <= roc_auc(scores, labels) <= hi


def test_group_bootstrap_ci_is_wider_than_image_level():
    """이미지 단위로 재표집하면 CI 가 좁아지고, 좁은 CI 는 통과선을 쉽게 통과시킨다.

    묶음 안의 값이 같을 때(같은 용접부 연속 촬영이 그렇다) 이미지 단위 재표집은 묶음
    하나를 여러 독립 표본처럼 세어 분산을 과소평가한다.
    """
    rng = np.random.default_rng(1)
    labels, scores, groups = [], [], []
    for g in range(20):
        lab = g % 2
        shared = lab + rng.normal(0, 0.6)      # 묶음 안은 값이 같다(완전상관)
        for _ in range(5):
            labels.append(lab)
            scores.append(shared)
            groups.append(f"g{g}")
    g_lo, g_hi = bootstrap_auc_ci(scores, labels, groups, n_resamples=400)
    i_lo, i_hi = bootstrap_auc_ci(
        scores, labels, [str(i) for i in range(len(labels))], n_resamples=400
    )
    assert (g_hi - g_lo) > (i_hi - i_lo)


# --- 이미지 변환 (P3) ------------------------------------------------------------

def test_patch_shuffle_preserves_pixel_multiset():
    arr = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    out = patch_shuffle(arr, patch=16)
    assert sorted(out.ravel()) == sorted(arr.ravel())


def test_patch_shuffle_actually_moves_patches():
    arr = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    assert not np.array_equal(patch_shuffle(arr, patch=16), arr)


def test_patch_shuffle_is_deterministic():
    arr = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    assert np.array_equal(patch_shuffle(arr, 16), patch_shuffle(arr, 16))


def test_patch_shuffle_keeps_shape_for_divisible_input():
    arr = np.zeros((720, 1280), dtype=np.uint8)
    assert patch_shuffle(arr, 16).shape == (720, 1280)


def test_downscale_to_target_size():
    arr = np.zeros((720, 1280), dtype=np.uint8)
    assert downscale(arr, 32).shape == (32, 32)


def test_downscale_is_deterministic():
    arr = np.arange(720 * 1280, dtype=np.uint32).reshape(720, 1280)
    assert np.array_equal(downscale(arr, 32), downscale(arr, 32))


def test_downscale_destroys_fine_detail():
    """32×32 로 줄이면 전역 통계만 남아야 한다."""
    arr = np.zeros((720, 1280), dtype=np.uint8)
    arr[100:104, 100:104] = 255           # 4×4 미세 구조
    assert downscale(arr, 32).max() == 0


# --- 실행 배관 -----------------------------------------------------------------

def probe_rows(n_crop=30, n_tile=30):
    rows = [row(f"c{i}", f"gc{i}", CROP) for i in range(n_crop)]
    rows += [row(f"t{i}", f"gt{i}", TILE) for i in range(n_tile)]
    return rows


def test_probe_reports_perfect_discrimination():
    rows = probe_rows()
    train, test = group_holdout(rows)
    r = run_source_probe(train, test, lambda tr, te: [x.is_tile for x in te])
    assert r.auc == pytest.approx(1.0)
    assert r.n_test_groups == len({x.group_id for x in test})


def test_probe_reports_no_discrimination():
    rows = probe_rows()
    train, test = group_holdout(rows)
    r = run_source_probe(train, test, lambda tr, te: [0.5] * len(te))
    assert r.auc == pytest.approx(0.5)


def test_probe_rejects_length_mismatch():
    """점수 개수가 표본과 다르면 조용히 잘라 쓰지 않는다."""
    rows = probe_rows()
    train, test = group_holdout(rows)
    with pytest.raises(ValueError):
        run_source_probe(train, test, lambda tr, te: [0.5])


def test_probe_counts_sources_in_holdout():
    rows = probe_rows()
    train, test = group_holdout(rows)
    r = run_source_probe(train, test, lambda tr, te: [0.5] * len(te), with_ci=False)
    assert set(r.label_counts) <= {CROP, TILE}
    assert sum(r.label_counts.values()) == len(test)


def test_summarize_sources_counts_all_three_kinds():
    rows = [row("a", "g1", CROP), row("b", "g2", TILE), row("c", "g3", BAND)]
    assert summarize_sources(rows) == {CROP: 1, TILE: 1, BAND: 1}
