"""조항 검색 — 메타데이터 필터 1차 + dense 보조. 스펙 §5-1·§5-3.

**자연어 유사도 검색이 아니다.** 질의는 구조화 키
`{검사방식, 결함코드, 실측 치수, 모재 두께, 지정 품질수준}`이고, 조항 청크의 메타데이터와
정확·범위 매칭으로 후보를 좁힌 뒤 **후보가 복수일 때만** dense 유사도로 정렬한다.
자연어 질의를 생성하면 재현성이 떨어지고, 검색 실패의 원인이 질의 생성 탓인지 색인 탓인지
분리되지 않는다.

사실상 구조화 lookup이므로 **조항 top-1이 95% 이상 나와야 정상**이다. 미달 시 진단 순서는
고정돼 있다: ① 등급 어휘 정합성(`grade_map` 경유 매칭이 실제로 걸리는가) → ② 색인·메타
데이터 구축 오류 → ③ 두께 구간 경계. ①을 앞에 둔 이유는 §5-3에 적었다.

검색 설정은 5칸 공통 고정이고 칸별로 재학습하지 않는다 — RAG는 집계 대상이 아니다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

TOP_K = 3
NO_CLAUSE = "해당 조항 없음"
"""검색 결과 0건일 때의 출력. 환각으로 채우지 않도록 프롬프트에도 명시한다(§5-6)."""


@dataclass(frozen=True)
class Chunk:
    """조항 단위 청크. 고정 길이 분할을 쓰지 않는다 — 조항 경계가 깨진다.

    메타데이터는 `corpus/derived/chunk_meta.jsonl`(B의 `derive_chunk_meta` 산출물)에서
    온다. D가 `limits.csv`를 직접 순회해 만들지 않는다.
    """

    chunk_id: str
    doc: str
    clause_id: str
    inspection_methods: tuple[str, ...] = ()
    """`RT` / `VT` / `ALL` 의 합집합. 계약 #3 변경(게이트 #13 결정 L)이며 B의
    `derive_chunk_meta` 가 조항 단위로 집계해 내려보낸다(복수형인 이유).

    **같은 결함 코드라도 표면 기공과 내부 기공은 허용치가 다르다** — 축이 없으면 잘못된
    행을 집어도 형식상 정상으로 보이고, 판정 정합성이 틀린 채로 높게 나온다.
    빈 튜플은 축 정보가 없는 청크이며 필터를 통과시킨다(무근거 인용 판정 기준 집합 유지).
    """
    defect_codes: tuple[str, ...] = ()
    thickness_min: Decimal | None = None
    thickness_max: Decimal | None = None
    quality_scheme: str | None = None
    quality_levels: tuple[str, ...] = ()
    text: str = ""
    source_page: int | None = None
    scope: str = "active"
    """`excluded` 행도 색인에 남긴다 — 빼면 그 조항이 영구 미검색이 되어 무근거 인용
    1차 판정의 기준 집합이 좁아진다. 대신 채점에서는 제외한다."""
    materials: tuple[str, ...] = ()
    """적용 재질. ALL은 공통 조항이며, 빈 값은 재질 메타가 없는 레거시 청크다."""


@dataclass(frozen=True)
class Query:
    """구조화 질의 키. 자연어를 만들지 않는다."""

    inspection_method: str
    defect_code: str
    thickness_mm: Decimal | None = None
    quality_scheme: str | None = None
    quality_level: str | None = None
    size_mm: Decimal | None = None
    material: str | None = None

    def to_text(self) -> str:
        """dense 보조 검색용 **결정론적 직렬화**. 같은 키는 언제나 같은 문자열이다.

        이것은 "자연어 질의 생성"(모델이 문장을 만드는 것)이 아니다. 템플릿은 파일로
        고정하고 스냅샷 해시에 포함한다.
        """
        parts = [
            f"검사방식={self.inspection_method}",
            f"결함코드={self.defect_code}",
            f"재질={self.material or ''}",
            f"두께={'' if self.thickness_mm is None else self.thickness_mm}mm",
            f"품질규격={self.quality_scheme or ''}",
            f"품질수준={self.quality_level or ''}",
        ]
        return "|".join(parts)


@dataclass(frozen=True)
class RetrievalResult:
    chunk_ids: tuple[str, ...]
    used_dense: bool
    n_candidates: int
    reason: str = ""

    @property
    def found(self) -> bool:
        return bool(self.chunk_ids)


def _level_matches(
    chunk: Chunk, query: Query, grade_map: Mapping[str, Mapping[str, Sequence[str]]] | None
) -> bool:
    """등급 어휘 매칭. **정확 매칭을 그대로 쓰면 IACS 유래 청크가 전부 탈락한다.**

    B가 `quality_level`을 단일 `B/C/D` enum에서 스킴 스코프 값으로 바꿨다 — IACS Rec.47은
    `STD`/`LIM` 2단, KS는 류 등급, 무등급 단일 한계 조항은 `ALL`이다. 질의의 품질수준은
    iso5817 계열 가정값이므로 `grade_map[scheme][질의 수준] ∋ chunk 수준`을 경유한다.
    """
    if chunk.quality_scheme is not None and query.quality_scheme != chunk.quality_scheme:
        # 기존 grade_map의 입력 규격은 ISO 5817이다. 명시된 대응만 허용하고,
        # 우연히 같은 등급 문자열을 쓰는 다른 규격은 통과시키지 않는다.
        if query.quality_scheme != "iso5817" or not grade_map or query.quality_level is None:
            return False
        allowed = grade_map.get(chunk.quality_scheme, {}).get(query.quality_level, ())
        return any(lv in chunk.quality_levels for lv in allowed)
    if not chunk.quality_levels:
        return True                      # 등급 무관 조항
    if query.quality_level is None:
        return True
    if query.quality_level in chunk.quality_levels:
        return True
    if grade_map and chunk.quality_scheme:
        allowed = grade_map.get(chunk.quality_scheme, {}).get(query.quality_level, ())
        return any(lv in chunk.quality_levels for lv in allowed)
    return False


METHOD_ANY = "ALL"
"""검사 방법과 무관한 조항. RT·VT 어느 질의에도 응한다(B `InspectionMethod.ALL`)."""


def _method_matches(chunk: Chunk, query: Query) -> bool:
    """검사축 매칭 — **1차 필터의 첫 조건**이다(§5-3).

    주 실험은 RT 한정이라 실질적으로 `RT` 고정이고 동작이 바뀌지 않는다. 확장이지
    재정의가 아니다. VT 청크는 여기서 탈락하되 색인에는 남는다.
    """
    if not chunk.inspection_methods:
        return True
    return (
        query.inspection_method in chunk.inspection_methods
        or METHOD_ANY in chunk.inspection_methods
    )


def _thickness_matches(chunk: Chunk, query: Query) -> bool:
    """반구간 `[min, max)`. 양끝 포함으로 두면 경계 두께에서 두 조항이 동시에 걸려
    top-1이 흔들린다."""
    if chunk.thickness_min is None and chunk.thickness_max is None:
        return True
    if query.thickness_mm is None:
        return False
    t = query.thickness_mm
    if chunk.thickness_min is not None and t < chunk.thickness_min:
        return False
    return not (chunk.thickness_max is not None and t >= chunk.thickness_max)


def filter_chunks(
    chunks: Sequence[Chunk],
    query: Query,
    grade_map: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> tuple[Chunk, ...]:
    """1차 메타 필터. 조건 순서에서 **검사축이 맨 앞**이다(§5-3)."""
    out = []
    for c in chunks:
        if not _method_matches(c, query):
            continue
        if c.materials and "ALL" not in c.materials and query.material not in c.materials:
            continue
        if c.defect_codes and query.defect_code not in c.defect_codes:
            continue
        if not c.defect_codes:
            continue                     # 메타 null 청크는 필터에서 자동 탈락
        if not _thickness_matches(c, query):
            continue
        if not _level_matches(c, query, grade_map):
            continue
        out.append(c)
    return tuple(out)


def retrieve(
    chunks: Sequence[Chunk],
    query: Query,
    grade_map: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    *,
    rank: object | None = None,
    top_k: int = TOP_K,
) -> RetrievalResult:
    """구조화 lookup → (복수일 때만) dense 정렬.

    Args:
        rank: `(query_text, chunks) -> Sequence[Chunk]` 콜러블. 후보가 2건 이상일 때만
            호출된다. None 이면 `chunk_id` 사전순으로 정렬한다 — **임의 순서를 쓰지
            않는 이유는 재현성**이다.

    검색 결과가 0건이면 빈 결과를 돌려주고 판정부가 `해당 조항 없음`을 출력한다.
    환각으로 채우지 않는 것이 이 경로의 목적이다.
    """
    cands = filter_chunks(chunks, query, grade_map)
    if not cands:
        return RetrievalResult((), False, 0, NO_CLAUSE)
    if len(cands) == 1:
        # 임베딩 호출 없음 — 구조화 lookup 이 이미 답을 확정했다.
        return RetrievalResult((cands[0].chunk_id,), False, 1)
    if rank is None:
        ordered = sorted(cands, key=lambda c: c.chunk_id)
    else:
        ordered = list(rank(query.to_text(), cands))  # type: ignore[operator]
    return RetrievalResult(
        tuple(c.chunk_id for c in ordered[:top_k]), True, len(cands)
    )


@dataclass(frozen=True)
class EmbeddingTrial:
    """임베딩 후보 실측 결과. **벤치마크 순위가 아니라 우리 문서에서의 성능**으로 고른다."""

    model: str
    top1: float
    top3: float
    n_queries: int


def pick_embedding(trials: Sequence[EmbeddingTrial], *, tie_break: str = "BGE-M3") -> str:
    """정답 조항 질의 100건 top-1 실측으로 선정한다(§5-4).

    동률이면 라이선스가 명확하고 dense+sparse 를 함께 주는 쪽을 택한다. 선정 후 **고정**
    하고 재학습·재선정하지 않는다 — 색인 스냅샷을 얼리는 조건이다.
    """
    if not trials:
        raise ValueError("실측 없이 임베딩을 고르지 않는다 — 질의 100건을 먼저 돌린다")
    best = max(t.top1 for t in trials)
    top = [t.model for t in trials if t.top1 == best]
    return tie_break if tie_break in top else min(top)


@dataclass(frozen=True)
class IndexStats:
    n_chunks: int
    n_active: int
    n_excluded: int
    methods: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_chunks": self.n_chunks, "n_active": self.n_active,
            "n_excluded": self.n_excluded, "methods": self.methods,
        }


def chunk_from_meta(meta: Mapping[str, object], text: str = "") -> Chunk:
    """B의 `derive_chunk_meta()` 항목 하나를 청크로 옮긴다.

    **D가 `limits.csv` 를 직접 순회해 메타를 만들지 않는다**(B 스펙 §1-4 금지 규칙 ④).
    필드명이 어긋나면 축 필터가 조용히 통과해 버리므로, 여기가 유일한 변환 지점이다.
    """
    def _dec(key: str) -> Decimal | None:
        v = meta.get(key)
        return None if v is None else Decimal(str(v))

    return Chunk(
        chunk_id=str(meta["clause_id"]),
        doc=", ".join(str(d) for d in meta.get("source_docs", ())),
        clause_id=str(meta["clause_id"]),
        inspection_methods=tuple(str(m) for m in meta.get("inspection_methods", ())),
        defect_codes=tuple(str(c) for c in meta.get("defect_codes", ())),
        materials=tuple(str(m) for m in meta.get("materials", ())),
        thickness_min=_dec("thickness_min"),
        thickness_max=_dec("thickness_max"),
        quality_scheme=(
            str(meta["quality_schemes"][0])            # type: ignore[index]
            if meta.get("quality_schemes") else None
        ),
        quality_levels=tuple(str(v) for v in meta.get("quality_levels", ())),
        text=text,
        scope=str(meta.get("scope", "active")),
    )


def index_stats(chunks: Sequence[Chunk]) -> IndexStats:
    methods: dict[str, int] = {}
    for c in chunks:
        for key in (c.inspection_methods or ("(없음)",)):
            methods[key] = methods.get(key, 0) + 1
    return IndexStats(
        n_chunks=len(chunks),
        n_active=sum(1 for c in chunks if c.scope == "active"),
        n_excluded=sum(1 for c in chunks if c.scope != "active"),
        methods=methods,
    )
