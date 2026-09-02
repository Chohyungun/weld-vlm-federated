"""평가 자산(D6) 격리 — 규칙을 문서에서 코드로 옮긴다. 77번 과제 5.

총괄 판정(2026-09-01)은 `gold_clauses.csv` 의 공개 노출을 레드라인 위반으로 보지
않는 대신 **실질 통제**를 요구했다. 그 통제가 이 파일이다.

**자기참조 검사가 되지 않게 짰다.** 감사 M9 가 B 의 부등식 방향 게이트를 "원리적으로
통과할 수밖에 없는 검사"로 판정했다. 같은 함정을 피하려고 여기에는 **양성 대조**가
먼저 있다 — 훅이 실제로 예외를 내는지 확인한 뒤에야 "학습 경로가 안 열었다"가 의미를
가진다. 정적 스캔도 마찬가지로, 일부러 심은 위반을 잡는지 먼저 확인한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation.isolation import (
    CORPUS_TRAIN_PATHS,
    D6_ASSET_NAMES,
    TRAIN_ROOTS,
    TRAIN_SCRIPTS,
    EvalAssetLeak,
    forbid_d6_assets,
    scan_training_paths,
)

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "corpus" / "derived" / "gold_clauses.csv"


# --------------------------------------------------------------------------------------
# 양성 대조 — 검사가 실패할 수 있는지 먼저 증명한다
# --------------------------------------------------------------------------------------

def test_runtime_hook_actually_trips(tmp_path: Path) -> None:
    """훅이 살아 있는가. **이 시험이 깨지면 아래 격리 시험은 전부 무의미하다.**"""
    decoy = tmp_path / "gold_clauses.csv"
    decoy.write_text("clause_id\n", encoding="utf-8")
    with pytest.raises(EvalAssetLeak), forbid_d6_assets():
        decoy.read_text(encoding="utf-8")


def test_runtime_hook_catches_pandas_path(tmp_path: Path) -> None:
    """csv 모듈뿐 아니라 pandas 로 우회해도 잡히는가."""
    decoy = tmp_path / "gold_clauses.csv"
    decoy.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(EvalAssetLeak), forbid_d6_assets():
        pd.read_csv(decoy)


def test_runtime_hook_is_inert_outside_the_block(tmp_path: Path) -> None:
    """블록 밖에서는 아무 일도 없어야 한다 — 채점기는 이 파일을 읽어야 하기 때문이다."""
    decoy = tmp_path / "gold_clauses.csv"
    decoy.write_text("a\n", encoding="utf-8")
    with forbid_d6_assets():
        pass
    assert decoy.read_text(encoding="utf-8") == "a\n"


def test_static_scan_catches_a_planted_reference(tmp_path: Path) -> None:
    """정적 스캔이 심어 둔 위반을 잡는가."""
    fake = tmp_path / "detection"
    fake.mkdir()
    (fake / "leak.py").write_text(
        'rows = read_csv("corpus/derived/gold_clauses.csv")\n', encoding="utf-8")
    hits = scan_training_paths(repo=tmp_path, roots=("detection",))
    assert [h.asset for h in hits] == ["gold_clauses.csv"]
    assert hits[0].path == "detection/leak.py"


def test_static_scan_sees_real_files() -> None:
    """스캔 대상이 실제로 존재하는가. 경로 오타로 0개를 훑고 통과하는 사고를 막는다."""
    from evaluation.isolation import _iter_sources

    seen = list(_iter_sources((*TRAIN_ROOTS, *CORPUS_TRAIN_PATHS, *TRAIN_SCRIPTS), REPO))
    assert len(seen) >= 10, f"학습 경로 소스가 {len(seen)}개뿐 — 경로 정의를 확인하라"
    roots_seen = {p.relative_to(REPO).parts[0] for p in seen}
    assert set(TRAIN_ROOTS) <= roots_seen, f"훑지 못한 학습 루트가 있다: {roots_seen}"


# --------------------------------------------------------------------------------------
# 본 검사
# --------------------------------------------------------------------------------------

def test_no_training_path_mentions_d6_assets() -> None:
    """학습 경로 어디에도 D6 자산 이름이 없다 (불변조건 1-4).

    `corpus/rules` 는 대상이 아니다 — `gold_clauses.csv` 를 **만드는** 쪽이다.
    """
    hits = scan_training_paths()
    assert not hits, "학습 경로가 평가 자산을 언급한다:\n" + "\n".join(map(str, hits))


@pytest.mark.skipif(not GOLD.exists(), reason="gold_clauses.csv 미실체화")
def test_gold_clauses_is_readable_by_the_scorer() -> None:
    """격리는 "아무도 못 읽는다"가 아니다. **채점기는 읽어야 한다.**

    이 시험이 없으면 통제를 과하게 걸어 채점을 막아 놓고도 통과할 수 있다.
    """
    from evaluation.gold import read_derived_csv

    rows = read_derived_csv(GOLD)
    assert rows, "정답 조항 목록이 비었다"


def test_detection_view_never_selects_eval_rows(tmp_path: Path, monkeypatch) -> None:
    """학습 뷰가 평가셋 행을 담지 않는다 — 이름으로 막을 수 없는 축의 행동 시험.

    이미지 실물이 없는 mock 스냅샷이라 하드링크만 무력화하고, **선택 로직은 실물 그대로**
    돌린다. 뷰에 들어간 image_id 와 eval 행의 교집합이 0 이어야 한다.
    """
    from data.manifest_io import load_snapshot
    from detection import dataset_view

    snap = load_snapshot(REPO / "data" / "mock" / "mock_aihub_v1")
    eval_ids = set(snap.manifest.loc[snap.manifest["split"] == "eval", "image_id"])
    assert eval_ids, "mock 스냅샷에 eval 행이 없다 — 시험이 공허해진다"

    monkeypatch.setattr(dataset_view, "_link_or_copy", lambda src, dst: None)
    seen: set[str] = set()
    real_split_view = dataset_view.split_view

    def spy(manifest, split, client=None):
        assert split != "eval", "학습 뷰가 eval split 을 요청했다"
        out = real_split_view(manifest, split, client=client)
        seen.update(out["image_id"])
        return out

    monkeypatch.setattr(dataset_view, "split_view", spy)
    with forbid_d6_assets():
        dataset_view.build_yolo_view(
            snap, out_dir=tmp_path / "view", train_client=None,
            label_map_path=REPO / "configs" / "label_map.yaml",
        )

    assert seen, "뷰가 아무 행도 고르지 않았다 — 시험이 공허해진다"
    assert not (seen & eval_ids), f"학습 뷰에 평가셋 행 {len(seen & eval_ids)}건이 들었다"


def test_d6_asset_list_is_not_empty() -> None:
    """자산 목록이 비면 모든 격리 시험이 조용히 통과한다."""
    assert D6_ASSET_NAMES
