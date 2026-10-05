"""중복 묶음 — 256-bit pHash + 전수 popcount + union-find. 스펙 §6-2 ~ §6-5.

함정 구간 #8. 설계 근거는 스펙 §6-0 쟁점 표에 있다. 요점 셋만 다시 적는다.

- **256-bit (hash_size=16).** 64-bit 은 균일 배경 RT 에서 판별력이 부족하다.
- **전수 popcount.** 비둘기집 다중 인덱스(MIH)는 임계를 올리는 순간 회수 보장이 깨진다.
  근사·인덱스가 없으므로 회수율 100% 가 자명하고 임계를 바꿔도 재인덱싱이 없다.
- **성분을 절단하지 않는다.** 과분리(누수)와 과병합(층화 오차)의 손실이 비대칭이므로
  모든 이지선다에서 병합 쪽을 택한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

#: §6-2 확정값. configs/base.yaml 로 옮기기 전 단일 출처.
HASH_SIZE = 16
HIGHFREQ_FACTOR = 4
HASH_BITS = HASH_SIZE * HASH_SIZE          # 256
HASH_WORDS = HASH_BITS // 64               # 4 × uint64
BIT_DEPTH_CLIP = (0.5, 99.5)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE = (8, 8)
#: 출발점 t0 = 32/256 (관례 8/64 와 같은 비율 12.5%). 확정은 §6-4 눈 확인 절차로.
THRESHOLD_START = 32
#: 상한. 스펙 초기값 48(18.8%)은 **실측으로 너무 낮았다** — 픽스처 240장에서 과분리(누수)가
#: 0 이 되는 지점이 t=74(28.9%)였고 48 에서는 같은 용접부 쌍 30건이 여전히 갈려 있었다.
#: 96(37.5%)으로 올린다. 손실 비대칭(§6-1)상 상한을 낮게 잡는 것은 누수를 남기는 방향이고,
#: 과병합은 층화 밴드가 흡수한다.
THRESHOLD_CAP = 96

EDGE_SHA256 = "E1"
EDGE_META = "E2"
EDGE_PHASH = "E3"


# --------------------------------------------------------------------------------------
# 전처리 + 해시
# --------------------------------------------------------------------------------------


def preprocess(path: Path, *, use_clahe: bool = True) -> Image.Image:
    """비트깊이 정규화 → 그레이스케일 → CLAHE. **기하 변형 금지.**

    dedup 전처리는 학습 전처리와 완전히 분리된다 — 레터박스·크롭·EXIF transpose 없음.

    `use_clahe=False` 는 열린질문 Q14 의 비교 경로 전용이다(히스토그램 이중모드 형성 실패 시
    1회 비교). 본 실행에서 끄려면 감독 승인이 필요하다.
    """
    with Image.open(path) as im:
        arr = np.asarray(im.convert("I") if im.mode in ("I", "I;16", "I;16B") else im.convert("L"))

    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, BIT_DEPTH_CLIP)
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)

    if not use_clahe:
        return Image.fromarray(arr, mode="L")
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE)
    return Image.fromarray(clahe.apply(arr), mode="L")


def compute_phash(path: Path, *, use_clahe: bool = True) -> str:
    """256-bit pHash 를 64자 hex 로. manifest.phash_hex 에 그대로 들어간다."""
    h = imagehash.phash(
        preprocess(path, use_clahe=use_clahe),
        hash_size=HASH_SIZE,
        highfreq_factor=HIGHFREQ_FACTOR,
    )
    hex_str = str(h)
    if len(hex_str) != HASH_BITS // 4:
        raise ValueError(f"pHash 길이가 {len(hex_str)}자다. {HASH_BITS//4}자여야 한다")
    return hex_str


def pack_hashes(hex_hashes: Sequence[str]) -> np.ndarray:
    """hex 문자열 목록 → (N, 4) uint64. 전수 popcount 의 입력."""
    if not hex_hashes:
        return np.empty((0, HASH_WORDS), dtype=np.uint64)
    out = np.empty((len(hex_hashes), HASH_WORDS), dtype=np.uint64)
    for i, hx in enumerate(hex_hashes):
        if len(hx) != HASH_BITS // 4:
            raise ValueError(f"[{i}] pHash 길이 불일치: {len(hx)}")
        for w in range(HASH_WORDS):
            out[i, w] = int(hx[w * 16 : (w + 1) * 16], 16)
    return out


# --------------------------------------------------------------------------------------
# 전수 거리 계산
# --------------------------------------------------------------------------------------


def hamming_matrix(packed: np.ndarray) -> np.ndarray:
    """(N, N) 해밍 거리. 소규모(N ≤ 수천)에서만 쓴다 — 테스트·검증용."""
    xor = packed[:, None, :] ^ packed[None, :, :]
    return np.bitwise_count(xor).sum(axis=2).astype(np.int32)


def iter_close_pairs(
    packed: np.ndarray, threshold: int, tile: int = 1024
) -> list[tuple[int, int, int]]:
    """거리 ≤ threshold 인 (i, j, d) 쌍 전량. i < j.

    타일 단위 XOR + `np.bitwise_count` 로 전수 계산한다. 근사·인덱스가 없으므로
    회수율 100% 가 자명하다. 호출부는 packed 를 **sha256 오름차순으로 정렬해** 넘긴다 —
    스캔 순서가 고정되어 결과가 결정론적이다.
    """
    n = len(packed)
    pairs: list[tuple[int, int, int]] = []
    for a0 in range(0, n, tile):
        a1 = min(a0 + tile, n)
        block_a = packed[a0:a1]
        for b0 in range(a0, n, tile):
            b1 = min(b0 + tile, n)
            xor = block_a[:, None, :] ^ packed[b0:b1][None, :, :]
            dist = np.bitwise_count(xor).sum(axis=2).astype(np.int32)
            ii, jj = np.nonzero(dist <= threshold)
            for i_local, j_local in zip(ii, jj, strict=True):
                i, j = a0 + int(i_local), b0 + int(j_local)
                if i < j:
                    pairs.append((i, j, int(dist[i_local, j_local])))
    pairs.sort()
    return pairs


def distance_histogram(packed: np.ndarray, tile: int = 1024) -> np.ndarray:
    """거리 0..HASH_BITS 전수 히스토그램. §6-4 골짜기 탐색의 입력.

    인덱스 산물이 아니라 전수·무편향이므로 "가짜 골짜기"가 생기지 않는다.
    """
    n = len(packed)
    hist = np.zeros(HASH_BITS + 1, dtype=np.int64)
    for a0 in range(0, n, tile):
        a1 = min(a0 + tile, n)
        for b0 in range(a0, n, tile):
            b1 = min(b0 + tile, n)
            xor = packed[a0:a1][:, None, :] ^ packed[b0:b1][None, :, :]
            dist = np.bitwise_count(xor).sum(axis=2).astype(np.int32)
            mask = np.ones_like(dist, dtype=bool)
            if a0 == b0:
                mask = np.triu(mask, k=1)
            hist += np.bincount(dist[mask].ravel(), minlength=HASH_BITS + 1)
    return hist


# --------------------------------------------------------------------------------------
# union-find + 묶음
# --------------------------------------------------------------------------------------


class UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


@dataclass(frozen=True)
class GroupResult:
    group_ids: tuple[str, ...]              # 입력 순서와 같은 길이
    n_groups: int
    max_group_size: int
    edge_counts: dict[str, int]
    cross_material_pairs: tuple[tuple[str, str, int], ...]   # 해소 필수 게이트
    threshold: int


def build_groups(
    image_ids: Sequence[str],
    sha256s: Sequence[str],
    phash_hex: Sequence[str],
    materials: Sequence[str],
    threshold: int = THRESHOLD_START,
    *,
    meta_keys: Sequence[str | None] | None = None,
) -> GroupResult:
    """엣지 3종(E1 sha256 일치 / E2 메타 ID 일치 / E3 pHash ≤ t*)으로 묶음을 만든다.

    - **교차 재질은 union 하지 않는다.** 대신 근접쌍을 `cross_material_pairs` 로 보고한다 —
      로그만 남기고 통과하면 탐지하고도 누수가 지나간다.
    - `group_id = "grp_" + min(구성원 sha256)[:16]` — 내용에서만 유도되므로 순회·삽입·병렬
      순서와 무관하다. 단독 이미지도 같은 규칙이라 group_id 는 전 행 non-null 이다.
    """
    n = len(image_ids)
    if not (n == len(sha256s) == len(phash_hex) == len(materials)):
        raise ValueError("입력 길이가 다르다")
    if meta_keys is not None and len(meta_keys) != n:
        raise ValueError("meta_keys 길이가 다르다")

    order = sorted(range(n), key=lambda i: sha256s[i])   # 처리 순서 고정
    uf = UnionFind(n)
    counts = {EDGE_SHA256: 0, EDGE_META: 0, EDGE_PHASH: 0}
    cross: list[tuple[str, str, int]] = []

    def same_material(i: int, j: int) -> bool:
        return materials[i] == materials[j]

    # E1 — 파일 내용 완전 일치
    by_sha: dict[str, list[int]] = {}
    for i in order:
        by_sha.setdefault(sha256s[i], []).append(i)
    for group in by_sha.values():
        for j in group[1:]:
            if same_material(group[0], j):
                uf.union(group[0], j)
                counts[EDGE_SHA256] += 1
            else:
                cross.append((image_ids[group[0]], image_ids[j], 0))

    # E2 — 필름/용접부 ID 일치 (채택 게이트 통과 시에만 값이 들어온다)
    if meta_keys is not None:
        by_meta: dict[str, list[int]] = {}
        for i in order:
            key = meta_keys[i]
            if key:
                by_meta.setdefault(key, []).append(i)
        for group in by_meta.values():
            for j in group[1:]:
                if same_material(group[0], j):
                    uf.union(group[0], j)
                    counts[EDGE_META] += 1
                else:
                    cross.append((image_ids[group[0]], image_ids[j], 0))

    # E3 — pHash 근접
    packed = pack_hashes([phash_hex[i] for i in order])
    for a, b, d in iter_close_pairs(packed, threshold):
        i, j = order[a], order[b]
        if same_material(i, j):
            uf.union(i, j)
            counts[EDGE_PHASH] += 1
        else:
            cross.append((image_ids[i], image_ids[j], d))

    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(uf.find(i), []).append(i)

    # group_id = grp_{재질}_{구성원 최소 sha256 앞 12자}
    #
    # 내용에서만 유도되므로 순회·삽입·병렬 순서와 무관하다. 재질을 접두로 넣는 이유는
    # **유일성이 증명되기 때문**이다: 같은 재질 안에서 동일 sha256 은 E1 이 반드시 union
    # 하므로 한 묶음이고, 따라서 (재질, 최소 sha) 가 같은 서로 다른 묶음은 존재할 수 없다.
    # 재질을 빼면 같은 파일이 두 재질로 등장할 때(= 교차 재질 해소 게이트로 보내야 하는
    # 바로 그 경우) 두 묶음의 ID 가 충돌해 게이트 보고 대신 예외가 난다.
    gid = [""] * n
    for idxs in members.values():
        label = f"grp_{materials[idxs[0]]}_{min(sha256s[i] for i in idxs)[:12]}"
        for i in idxs:
            gid[i] = label

    if len(set(gid)) != len(members):
        raise AssertionError(
            "group_id 충돌 — (재질, 최소 sha256 앞 12자)가 겹쳤다. "
            "sha256 스텁을 쓰는 테스트가 아니라면 해시 충돌이다"
        )
    for idxs in members.values():
        if len({materials[i] for i in idxs}) != 1:
            raise AssertionError("혼합 재질 묶음이 생겼다")

    return GroupResult(
        group_ids=tuple(gid),
        n_groups=len(members),
        max_group_size=max((len(v) for v in members.values()), default=0),
        edge_counts=counts,
        cross_material_pairs=tuple(sorted(set(cross))),
        threshold=threshold,
    )
