"""촬영 순서(`image_id`) 구간 정의 — 층화 채점의 층. 총괄 판정 6("을") 소비 지점.

## 왜 이 모듈이 있나

판정 6 이 헤드라인 채점을 **전역 Macro-F1 + id 구간 층화 채점 병기**로 확정했다.
층을 정의하는 코드가 `scripts/recompute_baselines.py` 안에만 있으면 D 가 채점기에서
같은 구간을 다시 만들어야 하고, **분위 절단점을 어느 모집단에서 뽑느냐**가 어긋나기 쉽다.
그 어긋남이 정확히 74번 감사 A-2 의 사고 유형이고, 76번 §1-5 에서 감사 자신도 걸렸다
(평가셋 분위로 자른 값 0.3877 대 train+val 분위 0.3541).

그래서 **절단점을 만드는 곳을 한 군데로 못박는다.** 층이 필요한 쪽은 전부 여기를 부른다.

    from data.id_strata import stratum_of, load_cut_points

    strata = stratum_of(image_ids, k=64)     # image_id -> 구간 번호 (0..k-1)

## 규칙 셋

1. **절단점은 train+val 에서만 뽑는다.** 평가셋 분위로 자르면 층 정의 자체가 평가셋
   유도가 된다 — 눈에 안 띄는 종류의 누출이다.
2. **`image_id` 의 숫자 부분만 쓴다.** 화소를 보지 않는다.
3. **K 는 호출자가 준다.** 이 모듈은 K 를 고르지 않는다 — 층이 잘아질수록 통제는
   세지지만 층당 표본이 줄어 분산이 커진다. 그 절충은 채점 설계(D)와 총괄의 판단이다.

## 산출 표

`materialize()` 가 `data/interim/manifest_v1/id_strata_k{K}.csv` 를 만든다
(`image_id,split,stratum`). 채점기가 매번 매니페스트를 열지 않아도 되고, 어느 판을 썼는지
파일 하나로 고정된다.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["STRATUM_AXIS", "id_number", "load_cut_points", "materialize", "stratum_of"]

#: 층 축의 이름. 보고·결과표에 이 문자열을 쓴다.
STRATUM_AXIS = "idq"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOT = _REPO_ROOT / "data/interim/manifest_v1"


def id_number(image_ids: Iterable[str]) -> np.ndarray:
    """`aihub71761:14503000` → `14503000`. 취득 순서의 대리 변수다."""
    return np.array([int(str(i).rsplit(":", 1)[-1]) for i in image_ids], dtype=np.int64)


@lru_cache(maxsize=4)
def _fit_id_numbers(snapshot: str) -> np.ndarray:
    """train+val 의 `image_id` 숫자부. **`load_snapshot` 은 해시를 전수 재검증하므로
    호출당 수십 MB 를 읽는다** — 층을 여러 K 로 뽑으면 그 비용이 K 배가 된다.
    스냅샷은 동결이라 결과가 변하지 않으니 프로세스 안에서 한 번만 읽는다.
    """
    from data.manifest_io import load_snapshot

    m = load_snapshot(Path(snapshot)).manifest
    return id_number(m.loc[m["split"] != "eval", "image_id"]).astype(float)


def load_cut_points(k: int, snapshot: Path | None = None) -> np.ndarray:
    """train+val 에서 뽑은 K 분위 절단점. **평가셋을 열지 않는다.**

    절단점이 중복되면 실제 칸 수가 K 보다 적어진다. 그대로 두면 층 번호의 최댓값이
    호출자 기대와 어긋나므로 뒤를 `+inf` 로 채워 자릿수를 유지한다(빈 층은 표본 0).
    """
    if k < 2:
        raise ValueError(f"K 는 2 이상이어야 한다 (받은 값 {k})")
    vals = _fit_id_numbers(str(snapshot or _SNAPSHOT))
    qs = np.unique(np.quantile(vals, np.linspace(0, 1, k + 1)[1:-1]))
    if qs.size < k - 1:
        qs = np.concatenate([qs, np.full(k - 1 - qs.size, np.inf)])
    return qs


def stratum_of(image_ids: Sequence[str], k: int, snapshot: Path | None = None) -> np.ndarray:
    """이미지별 층 번호 (0..K-1). 절단점은 항상 train+val 에서 온다."""
    cuts = load_cut_points(k, snapshot)
    return np.searchsorted(cuts, id_number(image_ids).astype(float), side="right")


def materialize(k: int, snapshot: Path | None = None, out: Path | None = None) -> Path:
    """`image_id,split,stratum` 표를 쓴다. 채점기가 이 파일 하나만 읽으면 된다."""
    from data.manifest_io import load_snapshot

    root = snapshot or _SNAPSHOT
    snap = load_snapshot(root)
    m = snap.manifest
    frame = pd.DataFrame({
        "image_id": m["image_id"].astype(str),
        "split": m["split"].astype(str),
        "stratum": stratum_of(list(m["image_id"]), k, root),
    }).sort_values("image_id", kind="stable")
    target = out or (root / f"id_strata_k{k}.csv")
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        frame.to_csv(fh, index=False, lineterminator="\n")
    return target
