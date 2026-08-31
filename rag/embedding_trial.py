"""임베딩 모델 결정 하네스 — 질의 100건 top-1 실측. 스펙 §5-4, 의사결정로그 미정 항목.

후보는 **BGE-M3 대 Qwen3-Embedding-4B** 이고, 벤치마크 순위가 아니라 우리 정답 조항
질의 100건의 top-1 로 고른다. 사전 등록된 결정 절차다.

**정답 조항 목록 사용이 격리 위반이 아닌 근거**: `13_spec_D` §5-4 가 "gold_clauses.csv 에서
질의 100건 추출 → 후보별 top-1 측정 → 채택 후 고정"을 임베딩 선정 절차로 사전 등록했고,
의사결정로그 미정 항목("임베딩 모델 — 질의 100건 top-1 실측")이 같은 절차를 가리킨다.
격리가 금지하는 것은 평가 자산의 **학습 투입**이며, 이 사용은 학습이 아니라 검색기 구성
요소의 사전 등록된 선정 절차다. 선정 후 임베딩은 고정되고 재학습하지 않는다.

**GPU 실행은 총괄 신호 대기다.** 모델 로드는 주입식이라 하네스 시험은 스텁으로 돈다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from rag.retrieve import Chunk, EmbeddingTrial, Query, pick_embedding, retrieve

CANDIDATES = ("BAAI/bge-m3", "Qwen/Qwen3-Embedding-4B")

Embedder = Callable[[Sequence[str]], np.ndarray]
"""문자열 목록 → (n, d) 임베딩. 모델 로드는 호출자가 한다 — GPU 신호 대기."""


def make_dense_ranker(embed: Embedder):
    """임베더 하나로 `retrieve(rank=...)` 콜러블을 만든다.

    코사인 정렬. 청크 쪽 텍스트가 비어 있으면 그 청크는 뒤로 보낸다 — 본문 없는 청크가
    우연히 1위가 되는 것을 막는다.
    """
    def rank(query_text: str, cands: Sequence[Chunk]) -> Sequence[Chunk]:
        texts = [c.text for c in cands]
        has_text = [bool(t.strip()) for t in texts]
        vecs = embed([query_text] + [t if t.strip() else " " for t in texts])
        q = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-12)
        scores = []
        for i, c in enumerate(cands):
            v = vecs[i + 1]
            sim = float(q @ (v / (np.linalg.norm(v) + 1e-12)))
            scores.append((not has_text[i], -sim, c.chunk_id))
        order = sorted(range(len(cands)), key=lambda i: scores[i])
        return [cands[i] for i in order]

    return rank


@dataclass(frozen=True)
class TrialOutcome:
    trial: EmbeddingTrial
    n_dense: int
    n_lookup_only: int
    per_query: tuple[dict, ...]

    def as_dict(self) -> dict:
        return {
            "model": self.trial.model,
            "top1": self.trial.top1,
            "top3": self.trial.top3,
            "n_queries": self.trial.n_queries,
            "n_dense": self.n_dense,
            "n_lookup_only": self.n_lookup_only,
        }


def run_trial(
    model_name: str,
    queries_gold: Sequence[tuple[Query, str]],
    chunks: Sequence[Chunk],
    embed: Embedder,
    *,
    grade_map: dict | None = None,
    top_k: int = 3,
) -> TrialOutcome:
    """후보 모델 하나의 top-1/top-3 을 잰다.

    후보 0~1개 질의는 임베딩과 무관하게 결과가 같으므로 **따로 센다** — 이 몫이 크면
    임베딩 선택이 최종 성능에 미치는 영향 자체가 작다는 뜻이고, 그것도 보고 대상이다.
    """
    ranker = make_dense_ranker(embed)
    hit1 = hit3 = n_dense = n_lookup = 0
    per_query: list[dict] = []
    for q, gold in queries_gold:
        r = retrieve(chunks, q, grade_map, rank=ranker, top_k=top_k)
        if r.used_dense:
            n_dense += 1
        else:
            n_lookup += 1
        ok1 = bool(r.chunk_ids) and r.chunk_ids[0] == gold
        ok3 = gold in r.chunk_ids
        hit1 += ok1
        hit3 += ok3
        per_query.append({
            "query": q.to_text(), "gold": gold,
            "got": list(r.chunk_ids), "top1": ok1, "used_dense": r.used_dense,
        })
    n = len(queries_gold)
    return TrialOutcome(
        trial=EmbeddingTrial(
            model=model_name,
            top1=hit1 / n if n else 0.0,
            top3=hit3 / n if n else 0.0,
            n_queries=n,
        ),
        n_dense=n_dense,
        n_lookup_only=n_lookup,
        per_query=tuple(per_query),
    )


def decide(
    outcomes: Sequence[TrialOutcome], *, tie_break: str = "BAAI/bge-m3"
) -> dict:
    """실측 결과로 모델을 선정한다. 선정 후 고정 — 재학습·재선정 없음.

    `pick_embedding` 이 실측 없는 선정을 거부하는 것까지 포함해 규칙은 그쪽에 있다.
    """
    winner = pick_embedding([o.trial for o in outcomes], tie_break=tie_break)
    return {
        "winner": winner,
        "outcomes": [o.as_dict() for o in outcomes],
        "note": (
            "선정 후 configs/rag.yaml embedding.model 에 기록하고 색인 스냅샷을 "
            "재부여한다. 이후 재선정하지 않는다"
        ),
    }
