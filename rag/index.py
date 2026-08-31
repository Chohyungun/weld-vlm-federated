"""색인 구축 진입점 — `chunk_meta.jsonl` → 검색 가능한 색인. 스펙 §5-2·§5-5.

청킹은 **조항 단위**다. 고정 길이 분할을 쓰지 않는다 — 조항 경계가 깨지면 인용 단위가
무너진다. 메타데이터는 B 의 `derive_chunk_meta` 산출물을 그대로 소비하며, D 가
`limits.csv` 를 직접 순회하지 않는다.

검색 설정은 `configs/rag.yaml` 이 단일 소스다(다섯 칸 공통 고정). 후보가 0~1개인 질의는
**임베딩 모델 없이** 끝난다 — 구조화 lookup 이 이미 답을 확정했기 때문이고, 그래서
임베딩 선정(GPU 실측)이 끝나기 전에도 색인·검색 경로 대부분이 작동한다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag.retrieve import Chunk, Query, RetrievalResult, chunk_from_meta, retrieve

CONFIG_PATH = Path("configs/rag.yaml")


@dataclass(frozen=True)
class RagConfig:
    top_k: int
    chunk_meta: str
    grade_map: str
    embedding_model: str | None
    trial_n_queries: int
    trial_seed: int
    no_hit_output: str


def load_rag_config(path: str | Path = CONFIG_PATH) -> RagConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    r, i, e = raw["retrieval"], raw["index"], raw["embedding"]
    return RagConfig(
        top_k=int(r["top_k"]),
        chunk_meta=str(i["chunk_meta"]),
        grade_map=str(i["grade_map"]),
        embedding_model=e.get("model"),
        trial_n_queries=int(e["trial_n_queries"]),
        trial_seed=int(e["trial_seed"]),
        no_hit_output=str(r["no_hit_output"]),
    )


def load_chunks(
    chunk_meta_path: str | Path, texts: Mapping[str, str] | None = None
) -> tuple[Chunk, ...]:
    """`chunk_meta.jsonl` 을 색인 청크로 만든다. 변환은 `chunk_from_meta` 한 지점이다.

    `texts` 는 조항 본문(파싱 문서 유래). 없으면 메타만으로 색인한다 — 1차 필터와
    후보 0~1 경로는 본문 없이도 성립한다.
    """
    chunks: list[Chunk] = []
    with Path(chunk_meta_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            text = (texts or {}).get(str(meta.get("clause_id", "")), "")
            chunks.append(chunk_from_meta(meta, text))
    ids = [c.chunk_id for c in chunks]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"청크 ID 중복 {dup[:5]} — 조항 단위 청킹이 깨졌다")
    return tuple(chunks)


def load_grade_map(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


class MissingEmbeddingModel(RuntimeError):
    """dense 정렬이 필요한데 임베딩 모델이 아직 선정되지 않았다."""


@dataclass(frozen=True)
class Index:
    """검색 색인. 청크 + 설정 + (선정 후) dense 랭커."""

    chunks: tuple[Chunk, ...]
    config: RagConfig
    grade_map: dict
    ranker: object | None = None

    def search(self, query: Query) -> RetrievalResult:
        """메타 필터 1차 → 후보 복수일 때만 dense.

        임베딩 모델이 미선정(`ranker is None`)인데 후보가 2개 이상이면 **조용히
        사전순으로 넘기지 않고 명시적으로 실패한다.** 사전순 폴백으로 낸 top-1 이
        실측처럼 읽히면 임베딩 선정 절차 자체가 무의미해진다.
        """
        if self.ranker is not None:
            return retrieve(
                self.chunks, query, self.grade_map,
                rank=self.ranker, top_k=self.config.top_k,
            )
        probe = retrieve(self.chunks, query, self.grade_map, top_k=self.config.top_k)
        if probe.used_dense:
            raise MissingEmbeddingModel(
                f"후보 {probe.n_candidates}개 — dense 정렬이 필요한데 임베딩 모델이 "
                "미선정이다 (configs/rag.yaml embedding.model). 실측 후 채운다"
            )
        return probe

    def snapshot_digest(self) -> str:
        """색인 내용의 결정론적 해시. 스냅샷 고정(불변조건 1-6)의 재료다."""
        h = hashlib.sha256()
        for c in sorted(self.chunks, key=lambda x: x.chunk_id):
            h.update(repr((
                c.chunk_id, c.inspection_methods, c.defect_codes,
                str(c.thickness_min), str(c.thickness_max),
                c.quality_scheme, c.quality_levels, c.scope, c.text,
            )).encode("utf-8"))
        return h.hexdigest()


def build_index(
    chunk_meta_path: str | Path | None = None,
    *,
    config_path: str | Path = CONFIG_PATH,
    texts: Mapping[str, str] | None = None,
    ranker: object | None = None,
) -> Index:
    """색인 구축 단일 진입점."""
    cfg = load_rag_config(config_path)
    chunks = load_chunks(chunk_meta_path or cfg.chunk_meta, texts)
    return Index(
        chunks=chunks, config=cfg,
        grade_map=load_grade_map(cfg.grade_map), ranker=ranker,
    )


def queries_from_gold_rows(
    gold_rows: Sequence[Mapping[str, object]],
    *,
    n: int,
    seed: int,
) -> tuple[tuple[Query, str], ...]:
    """임베딩 실측용 질의 100건 — 정답 조항 목록에서 **결정론적으로** 생성한다.

    각 gold 행의 구조화 키 조합에서 두께를 구간 안 격자로 뽑는다. 같은 시드는 언제나
    같은 질의를 만든다. 한국어 평가셋(599행)은 여기 관여하지 않는다 — D6 격리.
    """
    from decimal import Decimal

    import numpy as np

    if not gold_rows:
        return ()
    rng = np.random.default_rng(seed)
    out: list[tuple[Query, str]] = []
    rows = sorted(gold_rows, key=lambda r: str(r.get("rule_id", r.get("clause_id"))))
    for k in range(n):
        r = rows[int(rng.integers(0, len(rows)))]
        tmin = Decimal(str(r.get("thickness_min") or 0))
        tmax_raw = r.get("thickness_max")
        tmax = Decimal(str(tmax_raw)) if tmax_raw not in (None, "") else tmin + Decimal(40)
        # 구간 [min, max) 안의 격자점 — 경계 사례를 섞기 위해 20% 는 상한 직전을 찍는다
        frac = Decimal("0.99") if k % 5 == 0 else Decimal(str(round(float(rng.uniform(0.05, 0.9)), 3)))
        t = tmin + (tmax - tmin) * frac
        out.append((
            Query(
                inspection_method=str(r["inspection_method"]),
                defect_code=str(r["defect_code"]),
                thickness_mm=t,
                quality_scheme=str(r["quality_scheme"]),
                quality_level=str(r["quality_level"]),
            ),
            str(r["clause_id"]),
        ))
    return tuple(out)
