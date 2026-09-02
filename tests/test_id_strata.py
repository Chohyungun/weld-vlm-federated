"""층화 채점의 층 정의 — 정합 시험. 총괄 판정 6.

가장 중요한 시험은 **절단점이 평가셋에서 오지 않는다**는 것이다. 층 정의가 평가셋 유도면
층화 채점 자체가 74번 A-2 와 같은 문제를 안고 시작한다.

두 번째로 중요한 것은 `scripts/recompute_baselines.py` 의 `idq` 축과 **같은 층**을
만드는가다. 갈리면 76번의 천장 수치와 채점기의 층이 다른 것을 가리키게 된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from data.id_strata import STRATUM_AXIS, id_number, load_cut_points, materialize, stratum_of

REPO_ROOT = Path(__file__).resolve().parents[1]
V1 = REPO_ROOT / "data/interim/manifest_v1"
pytestmark = pytest.mark.skipif(not (V1 / "manifest.csv").is_file(),
                                reason="동결 스냅샷이 워크트리에 없다")


@pytest.fixture(scope="module")
def manifest():
    """`load_snapshot` 은 계약 4파일의 sha256 을 전수 재검증한다 — 호출당 수십 MB 다.
    이 파일 시험 전체가 같은 동결본을 보므로 한 번만 읽는다. (머지 게이트가 전 시험을
    돌리므로 여기서 아끼는 시간이 매 머지마다 절약된다.)"""
    from data.manifest_io import load_snapshot

    return load_snapshot(V1).manifest


def test_id_number_는_접두사를_떼낸다():
    assert list(id_number(["aihub71761:14503000", "x:7"])) == [14503000, 7]


def test_축_이름이_고정돼_있다():
    assert STRATUM_AXIS == "idq"


@pytest.mark.parametrize("k", [2, 8, 64])
def test_절단점이_평가셋에서_오지_않는다(k, manifest):
    """이 시험이 이 모듈의 존재 이유다.

    train+val 만으로 뽑은 절단점과, 전량(eval 포함)으로 뽑은 절단점이 같으면 모듈이
    실수로 평가셋을 섞고 있다는 뜻이다 — 두 모집단의 분포가 실제로 다르므로 같을 수 없다.
    """
    m = manifest
    got = load_cut_points(k)
    allrows = np.quantile(id_number(m["image_id"]).astype(float),
                          np.linspace(0, 1, k + 1)[1:-1])
    assert got.shape[0] == k - 1
    if k > 2:
        assert not np.allclose(got, np.unique(allrows)), (
            "절단점이 전량 분위와 같다 — 평가셋이 섞였을 수 있다")


@pytest.mark.parametrize("k", [2, 4, 64])
def test_층_번호가_범위_안에_있다(k, manifest):
    m = manifest
    s = stratum_of(list(m["image_id"]), k)
    assert s.min() >= 0
    assert s.max() <= k - 1
    assert len(s) == len(m)


def test_K가_작으면_거부한다():
    with pytest.raises(ValueError):
        load_cut_points(1)


def test_기준선_스크립트와_같은_층을_만든다(manifest):
    """`recompute_baselines.py` 의 `idq` 축과 층이 일치해야 한다.

    갈리면 76번이 낸 content-free 천장(0.9149, 족 `idq512`)과 채점기의 층이 서로 다른
    것을 가리킨다 — 같은 이름으로 다른 것을 재는 상태다.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from recompute_baselines import Population, make_cuts

    m = manifest.assign(
        id_num=lambda d: id_number(d["image_id"]),
        group_size=lambda d: d["group_size"].astype("int64"),
        file_bytes=0,
        prov="N-crop",
    )
    tv = m[m["split"] != "eval"].reset_index(drop=True)
    ev = m[m["split"] == "eval"].reset_index(drop=True)
    cuts = make_cuts(tv)
    for k in (4, 64):
        pop = Population(ev, {i: [] for i in ev["image_id"]}, [], cuts)
        theirs = pop.code[("idq", k)]
        mine = stratum_of(list(ev["image_id"]), k)
        assert np.array_equal(theirs, mine), f"K={k} 에서 층이 갈린다"


def test_산출표를_쓸_수_있다(tmp_path: Path):
    out = materialize(8, out=tmp_path / "s.csv")
    import pandas as pd

    f = pd.read_csv(out)
    assert list(f.columns) == ["image_id", "split", "stratum"]
    assert f["stratum"].between(0, 7).all()
    assert f["image_id"].is_unique
