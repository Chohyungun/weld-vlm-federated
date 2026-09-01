"""임베딩 모델 결정 실측 — BGE-M3 대 Qwen3-Embedding-4B, 질의 100건 top-1. 66번 과제 2.

사전 등록 절차(13_spec_D §5-4, 의사결정로그 미정 항목) 그대로 돌린다. **변별이 되는지
먼저 재고**, 안 되면 모델을 고르지 않는다 — 억지로 고르면 그 선택이 실측처럼 읽힌다.

두 후보를 실제로 부르기 전에 **임베더가 호출되기는 하는가**를 먼저 잰다. 구조화 lookup
(`rag.retrieve.retrieve`)은 후보가 2개 이상일 때만 dense 정렬을 부르므로, 후보가 늘 1개면
두 후보 모델의 출력은 **정의상 동일**하고 비교는 성립하지 않는다. 그 경우 모델을 내려받는
것은 측정이 아니다.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.gold import read_derived_csv
from rag.embedding_trial import CANDIDATES, run_trial
from rag.index import load_chunks, load_rag_config, queries_from_gold_rows
from rag.retrieve import filter_chunks

OUT = Path("outputs/pilot_d")


class EmbedderCalled(RuntimeError):
    """임베더가 실제로 호출됐다 — 그러면 후보 모델 실측이 의미를 갖는다."""


def refusing_embedder(_texts):
    raise EmbedderCalled("dense 정렬 경로 진입")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_rag_config()
    chunks = load_chunks(cfg.chunk_meta)
    gold_rows = read_derived_csv("corpus/derived/gold_clauses.csv")
    grade_map = {}

    queries = queries_from_gold_rows(
        gold_rows, n=cfg.trial_n_queries, seed=cfg.trial_seed
    )

    # 1. 질의별 1차 필터 후보 수 — 변별 가능성의 상한이다
    cand_counts = Counter()
    per_query = []
    for q, gold in queries:
        cands = filter_chunks(chunks, q, grade_map)
        cand_counts[len(cands)] += 1
        per_query.append({
            "query": q.to_text(), "gold": gold, "n_candidates": len(cands),
            "candidates": [c.chunk_id for c in cands],
            "gold_rank": ([c.chunk_id for c in cands].index(gold) + 1
                          if gold in [c.chunk_id for c in cands] else None),
        })

    n_multi = sum(v for k, v in cand_counts.items() if k >= 2)

    # 2. 하네스를 그대로 돌린다. 임베더는 호출되면 예외를 던진다 — 호출 여부가 측정값이다.
    try:
        outcome = run_trial(
            "(호출 감지용 스텁)", queries, chunks, refusing_embedder,
            grade_map=grade_map, top_k=cfg.top_k,
        )
        embedder_called = False
        trial = outcome.as_dict()
    except EmbedderCalled:
        embedder_called = True
        trial = {}

    # 3. 본문 유무 — dense 정렬이 걸리더라도 정렬할 재료가 있는가
    text_lengths = {c.chunk_id: len(c.text) for c in chunks}
    n_empty_text = sum(1 for v in text_lengths.values() if v == 0)

    # 4. 보조 지표 — 정답 조항의 순위 분포와 동점 수
    rank_dist = Counter(str(p["gold_rank"]) for p in per_query)
    ties = sum(1 for p in per_query if p["n_candidates"] >= 2)

    discriminable = embedder_called and n_empty_text < len(chunks)
    verdict = (
        "변별 가능 — 후보 모델 실측을 진행한다"
        if discriminable else
        "**변별 불가** — 모델을 고르지 않는다. 본 corpus 확장 후로 미룬다"
    )

    payload = {
        "candidates": list(CANDIDATES),
        "config": {
            "chunk_meta": cfg.chunk_meta, "top_k": cfg.top_k,
            "n_queries": cfg.trial_n_queries, "seed": cfg.trial_seed,
        },
        "index": {
            "n_chunks": len(chunks),
            "clause_ids": sorted(c.chunk_id for c in chunks),
            "n_chunks_with_empty_text": n_empty_text,
            "text_lengths": text_lengths,
        },
        "queries": {
            "n": len(queries),
            "candidate_count_distribution": {str(k): v for k, v in sorted(cand_counts.items())},
            "n_queries_needing_dense": n_multi,
            "embedder_called": embedder_called,
            "gold_rank_distribution": dict(rank_dist),
            "n_ties": ties,
        },
        "trial": trial,
        "verdict": verdict,
        "reasons": [
            f"1차 메타 필터 후보가 2개 이상인 질의 {n_multi}/{len(queries)}건 — "
            "dense 정렬 경로에 진입하는 질의가 없으면 두 후보의 출력은 정의상 같다",
            f"색인 청크 {len(chunks)}개 중 본문이 빈 것 {n_empty_text}개 — "
            "조항 본문이 색인에 없으면 정렬할 재료 자체가 없다(B 파생물에 text 필드 부재)",
        ],
        "isolation_note": (
            "gold_clauses.csv 사용은 D6 격리 위반이 아니다. 13_spec_D §5-4 가 "
            "'gold_clauses.csv 에서 질의 100건 추출 → 후보별 top-1 측정 → 채택 후 고정'을 "
            "임베딩 선정 절차로 사전 등록했고, 의사결정로그 미정 항목이 같은 절차를 "
            "가리킨다. 격리가 금지하는 것은 평가 자산의 **학습 투입**이며 이 사용은 학습이 "
            "아니라 검색기 구성요소의 사전 등록된 선정 절차다. 한국어 평가셋 599행은 "
            "이 절차에 관여하지 않는다."
        ),
        "per_query": per_query,
    }
    dest = OUT / "embedding_decision_v1.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"청크 {len(chunks)}개 (본문 빈 것 {n_empty_text}개)")
    print(f"후보 수 분포 {dict(sorted(cand_counts.items()))} · dense 필요 질의 {n_multi}/{len(queries)}")
    print(f"임베더 호출 여부 {embedder_called} · 정답 순위 분포 {dict(rank_dist)} · 동점 {ties}")
    print(verdict)
    print(f"저장: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
