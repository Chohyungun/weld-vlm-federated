"""분할 — 순서 고정. 스펙 §6-6 ~ §6-9.

    ① 묶음 생성(dedup) → ② 글로벌 평가셋 20% 선분리 → ③ C3 자연분할 + C1:C2 Dirichlet
    → ③′ 클라이언트 내 val → ④ manifest 확정·잠금 → ⑤ 히트맵

**② 가 ③ 보다 먼저**인 것은 함수 합성으로 강제한다 — `assign_clients()` 는 평가셋을 제외한
묶음 목록만 입력받으므로 코드 구조상 순서 역전이 불가능하다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from data.split.dirichlet import (
    DEFAULT_CONCENTRATION,
    partition_with_acceptance,
    rarest_first_order,
)

NORMAL_STRATUM = "__normal__"
SPLIT_EVAL, SPLIT_TRAIN, SPLIT_VAL = "eval", "train", "val"
JUDGMENT_SUBSET = "judgment_2000"

EVAL_FOLDS = 5      # fold 0 = 20%
VAL_FOLDS = 10      # fold 0 = 10%


@dataclass(frozen=True)
class SplitMeta:
    seed: int
    eval_folds: int
    val_folds: int
    dirichlet: dict[str, object] | None = None
    stratum_order: dict[str, tuple[str, ...]] = field(default_factory=dict)
    band_report: tuple[str, ...] = ()


def strata_keys(
    manifest: pd.DataFrame, iso_codes: Mapping[str, str]
) -> tuple[pd.Series, dict[str, tuple[str, ...]]]:
    """층화 키 = `{material}|{묶음 대표 클래스}`. 대표 = 재질 내 최희소 클래스.

    빈도 순서는 **런타임 실측에서 유도**한다. 공지 수치를 하드코딩하면 AL 의 융합불량 329 대
    균열 332 처럼 3장 차이에서 순서가 뒤집힌다 (§6-0 쟁점 8).
    """
    per_image = manifest["defect_types"].fillna("").map(
        lambda s: tuple(x for x in s.split(";") if x) or (NORMAL_STRATUM,)
    )
    order: dict[str, tuple[str, ...]] = {}
    for material, idx in manifest.groupby("material").groups.items():
        labels = [lab for i in idx for lab in per_image.loc[i]]
        order[str(material)] = rarest_first_order(labels, [1] * len(labels), iso_codes)

    # 묶음 대표: 구성원 라벨 합집합 중 최희소. 묶음 전체가 같은 값을 갖는다(복제).
    frame = pd.DataFrame(
        {"group_id": manifest["group_id"], "material": manifest["material"], "labels": per_image}
    )
    rep: dict[str, str] = {}
    for gid, part in frame.groupby("group_id"):
        material = str(part["material"].iloc[0])
        union = {lab for labs in part["labels"] for lab in labs}
        seq = order[material]
        rep[str(gid)] = min(union, key=lambda k: seq.index(k) if k in seq else len(seq))

    key = manifest["material"].astype(str) + "|" + manifest["group_id"].map(rep).astype(str)
    return key.astype("string"), order


def _first_fold(frame: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    """StratifiedGroupKFold fold 0 의 test 인덱스. 층이 부족하면 묶음 단위로 강등한다."""
    n_splits = min(n_splits, frame["group_id"].nunique())
    if n_splits < 2:
        return np.array([], dtype=int)
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        _, test = next(iter(sgkf.split(frame, y=frame["strata_key"], groups=frame["group_id"])))
        return test
    except ValueError:
        # 층화 불가(층 하나에 묶음 1개 등) → 층화 없이 묶음 단위로만 뗀다.
        # 층화를 포기해도 **묶음 원자성은 절대 포기하지 않는다.**
        groups = np.array(sorted(frame["group_id"].unique()))
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        take = set(groups[: max(1, len(groups) // n_splits)].tolist())
        return np.flatnonzero(frame["group_id"].isin(take).to_numpy())


def assign_splits(
    manifest: pd.DataFrame,
    iso_codes: Mapping[str, str],
    seed: int,
    *,
    eval_folds: int = EVAL_FOLDS,
    val_folds: int = VAL_FOLDS,
    judgment_ratio: float = 2000 / 14000,
    concentration: tuple[float, float] = DEFAULT_CONCENTRATION,
) -> tuple[pd.DataFrame, SplitMeta]:
    """§6-6~§6-8 을 순서대로 수행하고 split/client/eval_subset/strata_key 를 채운다."""
    m = manifest.copy()
    if m["group_id"].isna().any():
        raise ValueError("group_id 가 비어 있다 — dedup 단계를 먼저 돌려라 (순서 고정)")

    m["strata_key"], order = strata_keys(m, iso_codes)

    # ② 글로벌 평가셋 20% — 회사별 분할보다 먼저
    m["split"] = SPLIT_TRAIN
    eval_idx = _first_fold(m, eval_folds, seed + 2)
    m.loc[m.index[eval_idx], "split"] = SPLIT_EVAL
    m["client"] = pd.NA

    # ③ 학습 풀 → 클라이언트. 평가셋을 제외한 묶음만 넘긴다(순서 역전 불가)
    pool = m.loc[m["split"] != SPLIT_EVAL]
    mapping, dmeta = assign_clients(pool, iso_codes, seed + 3, concentration=concentration)
    m.loc[pool.index, "client"] = pool["group_id"].map(mapping).astype("string")
    if m.loc[m["split"] != SPLIT_EVAL, "client"].isna().any():
        raise AssertionError("학습 풀에 클라이언트가 배정되지 않은 묶음이 있다")

    # ③′ 클라이언트 내 val 10% — 곡선 로깅 전용, 학습에 개입하지 않는다
    for _, part in m.loc[m["split"] != SPLIT_EVAL].groupby("client"):
        val_idx = _first_fold(part, val_folds, seed + 4)
        if len(val_idx):
            m.loc[part.index[val_idx], "split"] = SPLIT_VAL

    # 판정문 채점 표본 — 분할 확정과 동시에 뽑는다(나중에 뽑으면 실행마다 달라진다)
    m["eval_subset"] = pd.NA
    picked = sample_judgment_groups(m, judgment_ratio, seed + 5)
    if picked:
        sel = (m["split"] == SPLIT_EVAL) & m["group_id"].isin(picked)
        m.loc[sel, "eval_subset"] = JUDGMENT_SUBSET

    return m, SplitMeta(
        seed=seed,
        eval_folds=eval_folds,
        val_folds=val_folds,
        dirichlet=dmeta,
        stratum_order={k: tuple(v) for k, v in order.items()},
        band_report=band_report(m),
    )


def sample_judgment_groups(
    manifest: pd.DataFrame, ratio: float, seed: int
) -> frozenset[str]:
    """판정문 채점 표본(2,000건 상당)의 묶음 집합. 스펙 §2-1 `eval_subset`.

    **층화는 재질 × 클래스 2축이다** (검수 #6 결정 D). `strata_key` 가 정확히 그
    2축이므로 층별 비례 배분한다. 합부(verdict) 축은 쓰지 않는다 — 합부는 판정 **결과**이고
    분할 시점(8/25)에 존재하지 않는 정보다.

    표본은 **묶음 단위**로 뽑는다. 이미지 단위로 뽑으면 같은 용접부의 프레임이 채점 표본과
    비표본으로 갈려 층화 비율이 실제 이미지 수에서 어긋난다.
    """
    ev = manifest.loc[manifest["split"] == SPLIT_EVAL]
    if ev.empty or ratio <= 0:
        return frozenset()

    groups = (
        ev.groupby("group_id")
        .agg(strata=("strata_key", "first"), size=("image_id", "size"))
        .sort_index()
    )
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    for stratum in sorted(groups["strata"].astype(str).unique()):
        pool = np.array(sorted(groups.index[groups["strata"].astype(str) == stratum]))
        # 층별 비례. 반올림으로 0이 되어도 층이 비지 않게 최소 1개는 남긴다 —
        # 소수 클래스(AL 슬래그 등)가 채점 표본에서 통째로 사라지는 것을 막는다.
        k = max(1, round(len(pool) * ratio))
        take = rng.choice(pool, size=min(k, len(pool)), replace=False)
        picked.extend(str(g) for g in take)
    return frozenset(picked)


def assign_clients(
    pool: pd.DataFrame,
    iso_codes: Mapping[str, str],
    seed: int,
    *,
    concentration: tuple[float, float] = DEFAULT_CONCENTRATION,
) -> tuple[dict[str, str], dict[str, object] | None]:
    """학습 풀 묶음 → 클라이언트. **입력에 평가셋 묶음이 없어야 한다.**

    - AL = C3 자연 분할 (무작위성 없음)
    - ST = C1:C2 Dirichlet
    - 그 외 재질(백업 경로의 UNK) = 균등 3분할. 실데이터에서는 §3-3 대로 Dirichlet 3-way 로
      바꾸는 것이 게이트 안건이다 — 지금은 재현 가능한 결정론적 배분으로 둔다.
    """
    if (pool["split"] == SPLIT_EVAL).any():
        raise ValueError("클라이언트 분할 입력에 평가셋이 섞였다 — 선분리가 먼저다")

    groups = (
        pool.groupby("group_id")
        .agg(material=("material", "first"), strata=("strata_key", "first"),
             size=("image_id", "size"))
        .sort_index()
    )
    groups["repr"] = groups["strata"].astype(str).str.split("|").str[-1]

    mapping: dict[str, str] = {}
    dmeta: dict[str, object] | None = None

    st = groups.loc[groups["material"] == "ST"]
    if len(st):
        res = partition_with_acceptance(
            tuple(st.index),
            st["repr"].to_numpy(dtype=object),
            st["size"].to_numpy(dtype=np.int64),
            concentration=concentration,
            seed=seed,
            label_priority=rarest_first_order(
                st["repr"].tolist(), st["size"].tolist(), iso_codes
            ),
        )
        mapping.update(
            {str(g): ("C1" if a == 0 else "C2") for g, a in zip(st.index, res.assignment)}
        )
        dmeta = {
            "concentration": list(concentration),
            "seed_used": res.seed_used,
            "attempts": res.attempts,
            "c1_share": round(res.c1_share, 4),
        }

    for g in groups.index[groups["material"] == "AL"]:
        mapping[str(g)] = "C3"

    other = groups.loc[~groups["material"].isin(["ST", "AL"])]
    for rank, g in enumerate(other.index):
        mapping[str(g)] = ("C1", "C2", "C3")[rank % 3]

    return mapping, dmeta


def band_report(manifest: pd.DataFrame) -> tuple[str, ...]:
    """층별 평가 비율 검사 — granularity 인지형 밴드. **위반은 기록·보고이고 정지가 아니다.**

    소수 층의 고정 ±2%p 는 묶음 원자 크기보다 좁아 정상 데이터에서도 원리적으로 실패한다.
    묶음 straddle 만 하드 실패이고(§6-6 표), 그 검사는 IV7/IV8 이 담당한다.
    """
    lines: list[str] = []
    total_eval = float((manifest["split"] == SPLIT_EVAL).sum())
    target = total_eval / max(len(manifest), 1)
    for stratum, part in manifest.groupby("strata_key"):
        n = len(part)
        ratio = float((part["split"] == SPLIT_EVAL).sum()) / n
        largest = int(part.groupby("group_id").size().max())
        band = max(0.02, largest / n)
        if abs(ratio - target) > band:
            lines.append(
                f"{stratum}: eval {ratio:.1%} (목표 {target:.1%}, 밴드 ±{band:.1%}, "
                f"n={n}, 최대묶음={largest})"
            )
    return tuple(lines)


def distribution_heatmap(manifest: pd.DataFrame) -> pd.DataFrame:
    """⑤ 재질×클래스×클라이언트 분포. **원본 클래스 기준**(축약 라벨이 아니다)."""
    rows: list[dict[str, object]] = []
    for row in manifest.itertuples():
        types = [t for t in str(row.defect_types or "").split(";") if t] or [NORMAL_STRATUM]
        for t in types:
            rows.append(
                {
                    "material": row.material,
                    "defect_type": t,
                    "client": row.client if pd.notna(row.client) else "eval",
                    "split": row.split,
                }
            )
    long = pd.DataFrame(rows)
    return (
        long.pivot_table(index=["material", "defect_type"], columns="client",
                         values="split", aggfunc="count", fill_value=0)
        .sort_index()
    )
