"""P2 출처 판별 · P3 셔플·저해상 프로브의 실행부. `40_규격지름길_대응판정.md` §5-2.

**P2 가 묻는 것**: "이 1280×720 은 원본 결함 크롭인가, 파노라마 타일인가."
맞힐 수 있으면 규격을 통일해도 **출처**가 새 지름길로 남아 있다는 뜻이다.

**P3 이 묻는 것**: 16×16 패치를 섞어 텍스처만 남기거나 32×32 로 줄여 전역 통계만 남겨도
출처나 결함이 판별되는가.

## 지켜야 하는 조건

- **학습 풀 내부 홀드아웃에서 학습·채점한다.** 평가셋은 건드리지 않는다.
- 홀드아웃은 **묶음 단위**로 뗀다. 이미지 단위로 뜨면 같은 용접부가 학습과 채점에 갈려
  들어가 AUC 가 부풀려지고, 그러면 "지름길이 있다"는 거짓 경보가 난다.
- **last 채점.** 3 epoch 을 끝까지 돌리고 마지막 상태로 잰다. best 선택은 암묵적 조기
  종료다.
- 시드를 고정한다. 프로브 값이 흔들리면 게이트가 흔들린다.

## 출처 라벨의 출처

타일링 산출물 `encode_progress.jsonl` 의 이미지별 `reason` 에서 온다.

| `reason` | 출처 | 뜻 |
|---|---|---|
| `ok` | `N-crop` | 원래부터 1280×720 이던 이미지 |
| `tiled` | `N-tile` | 파노라마에서 잘라낸 타일 |
| `oversized_band_cropped` | `N-band` | 밴드가 타일보다 커서 중심 크롭 |
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CROP = "N-crop"
TILE = "N-tile"
BAND = "N-band"

REASON_TO_SOURCE = {
    "ok": CROP,
    "tiled": TILE,
    "oversized_band_cropped": BAND,
}

SEED = 20260825
EPOCHS = 3
INPUT_SIZE = 224
PATCH_SIZE = 16
LOWRES_SIZE = 32


@dataclass(frozen=True)
class ProbeRow:
    """프로브 표본 한 건."""

    image_id: str
    rel_path: str
    group_id: str
    source: str
    iso_codes: tuple[str, ...] = ()

    @property
    def is_tile(self) -> int:
        """P2 의 이진 라벨. 타일이면 1."""
        return int(self.source == TILE)


def load_provenance(path: str | Path) -> dict[str, str]:
    """`encode_progress.jsonl` 에서 이미지별 출처를 읽는다.

    **매니페스트에 출처 컬럼이 없어 이 로그가 유일한 출처다.** 진행 로그이지 동결
    산출물이 아니므로, 스냅샷 안으로 승격하는 것이 옳다(§7-1 `tiles.csv`).
    """
    out: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            reason = str(rec.get("reason", ""))
            src = REASON_TO_SOURCE.get(reason)
            if src is not None:
                out[str(rec["image_id"])] = src
    return out


def load_rows(
    manifest_path: str | Path,
    provenance: dict[str, str],
    *,
    splits: Sequence[str] = ("train", "val"),
    modality: str = "RT",
) -> tuple[list[ProbeRow], list[str]]:
    """매니페스트와 출처를 조인한다.

    **`split == "eval"` 행은 넣지 않는다.** 학습 풀 내부에서만 학습·채점한다는 조건이
    여기서 강제된다. 조인 실패는 조용히 버리지 않고 세어서 돌려준다.
    """
    rows: list[ProbeRow] = []
    unmatched: list[str] = []
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if modality and r.get("modality") != modality:
                continue
            if r.get("split") not in splits:
                continue
            src = provenance.get(r["image_id"])
            if src is None:
                unmatched.append(r["image_id"])
                continue
            codes = tuple(c for c in (r.get("iso_codes") or "").split("|") if c)
            rows.append(
                ProbeRow(r["image_id"], r["rel_path"], r["group_id"], src, codes)
            )
    return rows, unmatched


def group_holdout(
    rows: Sequence[ProbeRow], *, holdout_frac: float = 0.2, seed: int = SEED
) -> tuple[list[ProbeRow], list[ProbeRow]]:
    """**묶음 단위** 홀드아웃. 같은 묶음은 한쪽에만 들어간다.

    이미지 단위로 뜨면 같은 용접부 연속 촬영이 학습과 채점에 갈려 들어가고, 분류기가
    출처가 아니라 그 용접부를 외운다. 그러면 AUC 가 부풀려져 **지름길이 없는데 있다고
    보고**하게 된다.
    """
    groups = sorted({r.group_id for r in rows})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    n_hold = max(1, round(len(groups) * holdout_frac))
    hold = {groups[i] for i in perm[:n_hold]}
    train = [r for r in rows if r.group_id not in hold]
    test = [r for r in rows if r.group_id in hold]
    return train, test


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """이진 AUC. 한쪽 클래스만 있으면 판별이 정의되지 않으므로 0.5 로 둔다."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # 동점 처리 — 평균 순위
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


# --- 이미지 변환 (P3) ----------------------------------------------------------

def patch_shuffle(arr: np.ndarray, patch: int = PATCH_SIZE, seed: int = SEED) -> np.ndarray:
    """16×16 패치를 무작위 재배열한다. **텍스처만 남고 배치 정보가 사라진다.**

    같은 시드에 같은 결과를 낸다. 이미지마다 다른 순열을 쓰면 프로브 값이 흔들린다.
    """
    h, w = arr.shape[:2]
    gh, gw = h // patch, w // patch
    if gh == 0 or gw == 0:
        return arr.copy()
    crop = arr[: gh * patch, : gw * patch]
    tiles = crop.reshape(gh, patch, gw, patch, *crop.shape[2:])
    tiles = tiles.swapaxes(1, 2).reshape(gh * gw, patch, patch, *crop.shape[2:])
    rng = np.random.default_rng(seed)
    tiles = tiles[rng.permutation(gh * gw)]
    out = tiles.reshape(gh, gw, patch, patch, *crop.shape[2:]).swapaxes(1, 2)
    return out.reshape(gh * patch, gw * patch, *crop.shape[2:])


def downscale(arr: np.ndarray, size: int = LOWRES_SIZE) -> np.ndarray:
    """32×32 로 줄인다. **전역 통계만 남는다.** 최근접 추출이라 결정론적이다."""
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return arr.copy()
    ys = (np.arange(size) * h // size).clip(0, h - 1)
    xs = (np.arange(size) * w // size).clip(0, w - 1)
    return arr[np.ix_(ys, xs)]


# --- 실행 결과 -----------------------------------------------------------------

@dataclass(frozen=True)
class SourceProbeResult:
    condition: str
    auc: float
    n_train: int
    n_test: int
    n_test_groups: int
    label_counts: dict[str, int] = field(default_factory=dict)
    ci_lo: float | None = None
    ci_hi: float | None = None

    def as_dict(self) -> dict:
        return {
            "condition": self.condition, "auc": self.auc,
            "n_train": self.n_train, "n_test": self.n_test,
            "n_test_groups": self.n_test_groups,
            "label_counts": self.label_counts,
            "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
        }


def bootstrap_auc_ci(
    scores: Sequence[float],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    n_resamples: int = 2000,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """묶음 단위 재표집으로 AUC 의 CI 를 낸다.

    이미지 단위로 재표집하면 CI 가 좁아지고, 좁은 CI 는 통과선 `0.65` 를 쉽게 통과시킨다.
    """
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    keys = sorted(by_group)
    if not keys:
        return 0.5, 0.5
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for k in range(n_resamples):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = [i for j in pick for i in by_group[keys[j]]]
        draws[k] = roc_auc(s[idx], y[idx])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run_source_probe(
    train: Sequence[ProbeRow],
    test: Sequence[ProbeRow],
    score_fn: Callable[[Sequence[ProbeRow], Sequence[ProbeRow]], Sequence[float]],
    *,
    condition: str = "raw",
    with_ci: bool = True,
) -> SourceProbeResult:
    """출처 판별을 실행하고 AUC 를 낸다.

    `score_fn` 을 주입받는 이유는 두 가지다. 학습 백엔드를 갈아 끼울 수 있고, 테스트가
    GPU 없이 판정 배관을 검증할 수 있다.
    """
    labels = [r.is_tile for r in test]
    scores = list(score_fn(train, test))
    if len(scores) != len(test):
        raise ValueError(f"점수 {len(scores)}개 대 표본 {len(test)}개 — 길이가 다르다")
    auc = roc_auc(scores, labels)
    lo = hi = None
    if with_ci and test:
        lo, hi = bootstrap_auc_ci(scores, labels, [r.group_id for r in test])
    counts: dict[str, int] = {}
    for r in test:
        counts[r.source] = counts.get(r.source, 0) + 1
    return SourceProbeResult(
        condition=condition, auc=auc,
        n_train=len(train), n_test=len(test),
        n_test_groups=len({r.group_id for r in test}),
        label_counts=counts, ci_lo=lo, ci_hi=hi,
    )


def summarize_sources(rows: Iterable[ProbeRow]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.source] = out.get(r.source, 0) + 1
    return out
