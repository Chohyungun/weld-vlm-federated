"""색인 지문은 청크의 출처 문서 · 쪽 · 조항을 담는다 — 07번 §40 의 4.

같은 본문이라도 다른 문서 · 쪽 · 조항에서 왔으면 다른 색인이다. 앞 판의 지문은 이 셋을 넣지 않아, 출처가 바뀐 색인이
같은 지문으로 지났다.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from rag.index import Index
from rag.retrieve import Chunk


def index(**over) -> Index:
    c = Chunk(chunk_id="c1", doc="doc_a", clause_id="k1", defect_codes=("2011",), text="본문", source_page=3)
    return Index(chunks=(replace(c, **over),), config=None, grade_map={})


@pytest.mark.parametrize("over", [{"doc": "doc_b"}, {"source_page": 4}, {"source_page": None}, {"clause_id": "k2"}])
def test_출처가_바뀌면_지문이_바뀐다(over: dict) -> None:
    assert index().snapshot_digest() != index(**over).snapshot_digest()


def test_같은_청크면_지문이_같다() -> None:
    assert index().snapshot_digest() == index().snapshot_digest()
