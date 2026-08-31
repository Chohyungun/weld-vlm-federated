"""⑤ 판정부 생성 경로 — 검출 출력 + 검색 결과 → 프롬프트 → 생성 → 공통 스키마. 52번 §4-⑤.

파일럿 판정 축은 좁혀서 돌린다: **조항 검색 + 기준 서술**. 합부는 내지 않는다.

## 원칙 (코드가 강제한다)

- **greedy 1회. 재시도·재프롬프트 금지.** 스키마 위반 출력은 오답 처리하고 실패율을
  별도 집계한다 — P9 러너의 오탐 정의와 같은 원칙이다.
- **검색 0건이면 생성을 호출하지 않는다.** `해당 조항 없음` + `판정불가` 를 결정론적으로
  낸다. 프롬프트에도 같은 규칙이 적혀 있지만, 후보가 없는데 모델에게 물어보는 것 자체가
  환각을 청하는 일이다.
- **추출은 관대하게, 검증은 엄격하게** (13_spec_D §2-4). 코드펜스·산문은 벗기되
  필드값은 보정하지 않는다.
- 생성 모델은 통합형과 **같은 체크포인트**를 쓴다(개발규약). `configs/rag.yaml`
  `judge.checkpoint` 가 자리이고, C 의 ⑥ 확정을 기다린다 — null 이면 로드 시점에
  명시적으로 실패한다.

모델 로드·GPU 는 주입식(`generate_fn`)이라 배선 시험은 스텁으로 돈다.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rag.retrieve import NO_CLAUSE, Chunk, RetrievalResult

GenerateFn = Callable[[str], str]
"""프롬프트 → 생성 텍스트. greedy 설정·체크포인트 로드는 호출자(GPU 신호 후) 몫."""

UNDECIDABLE = "판정불가"
PROMPT_PATH = Path("rag/prompts/verdict_pilot_v1.txt")


def load_prompt_template(path: str | Path = PROMPT_PATH) -> tuple[str, str]:
    """프롬프트 원문과 sha256. 해시는 스냅샷·MLflow 태그(`prompt_sha256`)에 들어간다.

    다섯 칸에서 한 글자도 다르면 안 되므로(불변조건 3-3) 파일 하나가 정본이다.
    """
    raw = Path(path).read_text(encoding="utf-8")
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_prompt(
    template: str,
    defects: Sequence[Mapping[str, object]],
    clauses: Sequence[Chunk],
) -> str:
    """프롬프트 조립. 결함·조항을 결정론적 직렬화로 채운다."""
    defect_lines = "\n".join(
        f"- 결함코드 {d.get('iso_code')} / 크기(px) {d.get('size_px', '미상')}"
        for d in defects
    ) or "- (검출된 결함 없음)"
    clause_lines = "\n".join(
        f"- [{c.chunk_id}] {c.text or '(본문 미등재)'}" for c in clauses
    ) or f"- {NO_CLAUSE}"
    return template.replace("{defects}", defect_lines).replace("{clauses}", clause_lines)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_generation(text: str) -> tuple[dict | None, str | None]:
    """생성 텍스트에서 판정 JSON 을 꺼낸다. 성공 시 (dict, None), 실패 시 (None, 사유).

    코드펜스와 앞뒤 산문은 벗긴다(모델의 습관이지 판정 능력의 결함이 아니다).
    그 뒤의 검증은 엄격하다 — verdict enum, cited 목록형. **필드값 보정은 없다.**
    """
    m = _JSON_BLOCK.search(text)
    if not m:
        return None, "no_json"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "json_decode"
    if not isinstance(obj, dict):
        return None, "schema_violation"
    verdict = obj.get("verdict")
    cited = obj.get("cited_clauses")
    if verdict not in ("합격", "불합격", UNDECIDABLE):
        return None, "schema_violation"
    if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
        return None, "schema_violation"
    return obj, None


@dataclass(frozen=True)
class JudgeOutput:
    """이미지 1장의 판정부 출력 — 분리형 원시 출력 계약의 verdicts.jsonl 한 줄."""

    image_id: str
    retrieved: tuple[str, ...]
    verdict: str
    cited_clauses: tuple[str, ...]
    basis: str
    parse_ok: bool
    parse_error: str | None = None
    generated: bool = True
    raw_text: str = ""

    def as_row(self) -> dict:
        return {
            "image_id": self.image_id,
            "retrieved": list(self.retrieved),
            "verdict": self.verdict,
            "cited": list(self.cited_clauses),
            "basis": self.basis,
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
            "generated": self.generated,
        }


def judge_image(
    image_id: str,
    defects: Sequence[Mapping[str, object]],
    retrieval: RetrievalResult,
    clauses: Sequence[Chunk],
    generate_fn: GenerateFn,
    template: str,
) -> JudgeOutput:
    """이미지 1장의 판정. **생성 호출은 최대 1회다.**

    검색 0건이면 생성을 부르지 않고 결정론적으로 `판정불가` 를 낸다. 스키마 위반은
    그대로 오답으로 남긴다 — 재프롬프트로 살리면 실패율이 거짓말을 한다.
    """
    if not retrieval.found:
        return JudgeOutput(
            image_id=image_id, retrieved=(), verdict=UNDECIDABLE,
            cited_clauses=(), basis=NO_CLAUSE,
            parse_ok=True, generated=False,
        )
    prompt = build_prompt(template, defects, clauses)
    raw = generate_fn(prompt)                       # greedy 1회. 여기가 유일한 호출이다
    obj, err = parse_generation(raw)
    if obj is None:
        return JudgeOutput(
            image_id=image_id, retrieved=tuple(retrieval.chunk_ids),
            verdict=UNDECIDABLE, cited_clauses=(), basis="",
            parse_ok=False, parse_error=err, raw_text=raw,
        )
    return JudgeOutput(
        image_id=image_id, retrieved=tuple(retrieval.chunk_ids),
        verdict=str(obj["verdict"]),
        cited_clauses=tuple(obj["cited_clauses"]),
        basis=str(obj.get("basis", "")),
        parse_ok=True, raw_text=raw,
    )


@dataclass(frozen=True)
class JudgeRunReport:
    outputs: tuple[JudgeOutput, ...]
    prompt_sha256: str
    failure_counts: dict[str, int] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        n = len(self.outputs)
        return sum(1 for o in self.outputs if not o.parse_ok) / n if n else 0.0

    def as_dict(self) -> dict:
        return {
            "n_images": len(self.outputs),
            "n_generated": sum(1 for o in self.outputs if o.generated),
            "n_no_hit": sum(1 for o in self.outputs if not o.generated),
            "failure_rate": self.failure_rate,
            "failure_counts": self.failure_counts,
            "prompt_sha256": self.prompt_sha256,
        }


def run_judge(
    items: Sequence[tuple[str, Sequence[Mapping[str, object]], RetrievalResult, Sequence[Chunk]]],
    generate_fn: GenerateFn,
    *,
    prompt_path: str | Path = PROMPT_PATH,
) -> JudgeRunReport:
    """판정부 일괄 실행. 실패율을 사유별로 집계한다 — 별도 보고 지표다."""
    template, digest = load_prompt_template(prompt_path)
    outputs: list[JudgeOutput] = []
    failures: dict[str, int] = {}
    for image_id, defects, retrieval, clauses in items:
        out = judge_image(image_id, defects, retrieval, clauses, generate_fn, template)
        outputs.append(out)
        if not out.parse_ok and out.parse_error:
            failures[out.parse_error] = failures.get(out.parse_error, 0) + 1
    return JudgeRunReport(
        outputs=tuple(outputs), prompt_sha256=digest, failure_counts=failures
    )
