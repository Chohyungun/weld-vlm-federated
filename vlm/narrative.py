"""서술 추출 · 상태 — 학습 쪽(하네스 서술 칸)의 구현. 채점 쪽 추출기와 **코드는 따로, 규칙은 하나**다.

규칙은 게이트 판정 04 §1 의 공통 계약에 D 03 2판 §1-4b 의 R1 · P1 ~ P5 를 더하고, 펜스 표지 줄의 문법은 04 §7-4 로 좁힌 것이다.
두 구현이 같은 벡터 파일에서 같은 다섯 값을 내야 2 차 등록이 선다(04 §1 의 6 · D 03 2판 §5b). **벡터의 기대가 정본이다** — 어긋나면
기대를 구현에 맞추지 않고 원인을 찾는다.

    extract(text, gen_stop) → NarrativeExtract(state, narrative, front_text, stop_kept, upstream_reason)

상태는 이 순서로 정한다(04 §7-4): ⓪ 검증 오류 → ① 레코드 실패 → ② 미기재 → ③ 잘림 → ④ 완결.

- ⓪ 종료 사유가 칸 없음(인자를 생략) · `None` · `eos` · `length` 밖의 값이면 검증 오류다. 서술 유무와 파싱보다 먼저 본다(R1).
  생성문 `text` 가 문자열이 아니어도 검증 오류다 — 계약 밖 입력이다(D 와 같은 꼴, 지시 1022c 덧붙임 D).
- ① 구조 파싱 유효는 정본 파서 `vlm.gen_parse.parse_generation` 의 판정이다 — 해독 성공만이 아니라 `defects` 가 목록이어야 한다.
  실패면 레코드 실패이고 사유는 그 파서의 사유(`no_json` · `json_decode` · `truncated` · `schema_violation`)다. 뒤의 다른 JSON 을
  찾아 살리지 않는다. 개별 상자를 버리는 지금 정책은 그대로다 — 레코드 실패로 키우지 않는다.
- 첫 객체의 끝 위치는 정본 파서가 내지 않으므로 같은 규칙(`text.strip()` → 첫 `{` → `raw_decode`)으로 다시 잰다. 다시 읽은 객체가
  정본 파서가 읽은 객체와 다르면 규칙이 갈린 것이라 멈춘다(`AssertionError`).
- 앞 글 · 뒤 글은 첫 객체 밖의 앞 · 뒤다. 각각 `\\n` 으로 나눈 조각에서 표지 줄을 빼고 `\\n` 으로 다시 이은 뒤 `str.strip()` 으로
  벗긴다(P2 · P1). 표지 줄 = 벗긴 조각이 역따옴표 셋 이상으로 시작하고, 그 뒤가 비었거나 언어 표시(`[A-Za-z0-9_+-]*`)뿐인 줄이다.
  역따옴표 셋 뒤에 그 밖의 글이 붙은 줄은 서술이다. 물결표 펜스는 표지가 아니다.
- 서술 = 벗긴 앞 · 뒤가 둘 다 비지 않으면 공백 하나(U+0020)로 이은 것, 하나만 비지 않으면 그것(P3). 안쪽의 줄바꿈 · 공백 ·
  유니코드 꼴은 바꾸지 않는다(P5).
- JSON 앞에 글 = ② ~ ④ 에서 벗긴 앞 글이 비지 않음(P4) — 분모는 구조 파싱 유효(② ~ ④)다.

토치 · 모델을 부르지 않는다. 정본 파서만 부른다.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DRAFT_VECTOR_PATH",
    "DRAFT_VECTOR_SHA256",
    "EXTRACTOR_ID",
    "GEN_STOPS",
    "RELEASE_VECTOR_PATH",
    "STATES",
    "VECTOR_COMMIT",
    "VECTOR_PATH",
    "VECTOR_SHA256",
    "NarrativeExtract",
    "VectorRejected",
    "canonical_vector_bytes",
    "extract",
    "is_fence_marker",
    "load_vectors",
    "run_vectors",
    "source_sha256",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
#: 이 구현의 식별자 — 2 차 등록 블록의 학습 쪽 추출기 칸에 LF sha256(`source_sha256`)과 함께 싣는다.
EXTRACTOR_ID = "vlm_narrative_v1"
#: 상태 다섯(04 §1 의 3 · 04 §7-4 의 순서).
STATES = ("validation_error", "record_fail", "missing", "truncated", "complete")
#: 1.4 계약의 종료 사유 — 그 밖은 검증 오류다(R1).
GEN_STOPS = ("eos", "length")
#: 공통 벡터 정본(04 §7-5 ⑤) — D 가 0 단계로 고정한 03f-2 의 바이트(초안 03f 의 사례 그대로 + 새 사례, 04 §7-4 의 좁힌 펜스 문법).
#: 자리는 그 바이트를 고정한 D 커밋(`VECTOR_COMMIT`)의 경로다. D 가 같은 바이트를 `evaluation/vectors/narrative_extract.v1.json` 으로
#: 올리면 자리만 바꾼다 — sha256 은 그대로다. 읽을 때는 바이트로 읽어 sha256 을 맞댄다(`load_vectors`).
VECTOR_PATH = "docs/dev_log/2026-10-20-설계복구/03f-2_추출벡터_D.json"
VECTOR_SHA256: str | None = "e7c401fcb81b6fea85e80ff4c300fa6de1b0aa20b9e1267a1eeaf5a440b33964"
VECTOR_COMMIT = "e3e0ac8a150ef4e7fb7e9e1967c38d171f4ba9db"
#: 정본의 정규 배포 자리(04 §7-5 ⑤) — D 가 같은 바이트를 여기 올리면 `VECTOR_PATH` 를 이리로 바꾼다.
RELEASE_VECTOR_PATH = "evaluation/vectors/narrative_extract.v1.json"
#: 개발용 초안 — main 의 03f(36 사례, 2026-10-22T09:25:01Z 고정). 정본이 아니다 — 이것과 맞았다고 정본 대조를 대신하지 않는다.
DRAFT_VECTOR_PATH = "docs/dev_log/2026-10-20-설계복구/03f_추출벡터_초안_D.json"
DRAFT_VECTOR_SHA256 = "ba97dc03e1d23660194890505fd3646e40cd394514942095233ff2049e07d514"

_ABSENT = object()
_FENCE = re.compile(r"`{3,}[A-Za-z0-9_+-]*")
_FIELDS = ("state", "narrative", "front_text", "stop_kept", "upstream_reason")


@dataclass(frozen=True)
class NarrativeExtract:
    """추출 결과 다섯 값 — 벡터의 `expect` 와 같은 이름이다."""

    state: str
    narrative: str | None
    front_text: bool | None
    stop_kept: str | None
    upstream_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in _FIELDS}


class VectorRejected(ValueError):
    """벡터 파일이 없거나 sha256 이 고정값과 다르거나 꼴이 아니다 — 대조하지 않는다."""


def is_fence_marker(piece: str) -> bool:
    """줄 조각 하나가 코드 펜스 표지 줄인가(04 §7-4)."""
    return _FENCE.fullmatch(piece.strip()) is not None


def _without_markers(part: str) -> str:
    """표지 줄을 조각 목록에서 빼고(그 줄바꿈도 함께) 다시 이은 뒤 벗긴다(P2 → P1)."""
    return "\n".join(p for p in part.split("\n") if not is_fence_marker(p)).strip()


def _outside_first_object(text: str) -> tuple[str, str, Any]:
    """첫 JSON 객체 밖의 (앞 글, 뒤 글)과 그 객체 — 정본 파서와 같은 규칙으로 끝 위치를 다시 잰다."""
    s = text.strip()
    start = s.find("{")
    obj, end = json.JSONDecoder().raw_decode(s[start:])
    return s[:start], s[start + end:], obj


def extract(text: Any, gen_stop: Any = _ABSENT) -> NarrativeExtract:
    """생성문과 종료 사유 → 다섯 값. `gen_stop` 칸이 없는 줄은 인자를 생략해 넘긴다(빈 값으로 메우지 않는다)."""
    if gen_stop is _ABSENT or not isinstance(gen_stop, str) or gen_stop not in GEN_STOPS or not isinstance(text, str):
        return NarrativeExtract("validation_error", None, None, None, None)
    from vlm.gen_parse import _first_object, parse_generation

    parsed = parse_generation(text)
    if parsed.parse_error is not None:
        return NarrativeExtract("record_fail", None, None, None, parsed.parse_error)
    front, back, obj = _outside_first_object(text)
    if obj != _first_object(text)[0]:
        raise AssertionError("첫 객체를 다시 읽은 값이 정본 파서의 값과 다르다 — 경계 규칙이 갈렸다")
    f, b = _without_markers(front), _without_markers(back)
    narrative = f"{f} {b}" if f and b else (f or b)
    if not narrative:
        state = "missing"
    elif gen_stop == "length":
        state = "truncated"
    else:
        state = "complete"
    return NarrativeExtract(state, narrative, bool(f), gen_stop, None)


def source_sha256() -> str:
    """이 모듈 바이트의 sha256(CRLF → LF 정규화) — 등록의 구현 해시."""
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_vectors(path: str | Path, *, sha256: str) -> tuple[list[dict[str, Any]], str]:
    """벡터 파일을 **바이트로** 읽어 sha256 을 맞댄 뒤 사례 목록을 낸다(D 03 2판 §5b 의 1). (사례, sha256)."""
    p = Path(path)
    p = p if p.is_absolute() else REPO_ROOT / p
    if not p.is_file():
        raise VectorRejected(f"벡터 파일이 없다: {Path(path).as_posix()}")
    raw = p.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha256:
        raise VectorRejected(f"벡터 파일의 sha256 {got} ≠ 고정값 {sha256}")
    body = json.loads(raw.decode("utf-8"))
    vectors = body.get("vectors") if isinstance(body, Mapping) else None
    if not isinstance(vectors, list) or not vectors:
        raise VectorRejected("벡터 파일에 사례 목록(vectors)이 없다")
    for v in vectors:
        if not isinstance(v, Mapping) or "id" not in v or "text" not in v or not isinstance(v.get("expect"), Mapping):
            raise VectorRejected(f"사례의 꼴이 아니다: {v!r}"[:200])
        if set(v["expect"]) != set(_FIELDS):
            raise VectorRejected(f"사례 {v['id']} 의 기대 칸이 {list(_FIELDS)} 가 아니다")
    return [dict(v) for v in vectors], got


def _same(a: Any, b: Any) -> bool:
    """값과 자료형이 함께 같은가 — `0` · `1` 과 `False` · `True` 를 다르게 본다(지시 1022c 덧붙임 D)."""
    return type(a) is type(b) and a == b


def canonical_vector_bytes(repo_root: Path | None = None) -> tuple[bytes, str]:
    """정본 벡터의 바이트와 읽은 곳 — 정규 배포 자리(`RELEASE_VECTOR_PATH`) → 지금 자리(`VECTOR_PATH`) → 그 바이트를 고정한 D 커밋의
    blob(`VECTOR_COMMIT`, 같은 저장소의 객체) 순서. 어느 쪽도 sha256 이 고정값과 같아야 하고, 셋 다 없으면 `VectorRejected` 다 —
    정본 대조는 건너뛰지 않는다(지시 1022c 덧붙임 E). 시험 · 보고 · 드라이런이 쓰는 자리 찾기다(학습 · 평가 경로는 쓰지 않는다).
    """
    import subprocess

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    for rel, where in ((RELEASE_VECTOR_PATH, "release"), (VECTOR_PATH, "tree")):
        p = root / rel
        if p.is_file():
            raw = p.read_bytes()
            if hashlib.sha256(raw).hexdigest() != VECTOR_SHA256:
                raise VectorRejected(f"정본 자리 {rel} 의 sha256 이 고정값 {VECTOR_SHA256} 과 다르다")
            return raw, where
    r = subprocess.run(["git", "cat-file", "blob", f"{VECTOR_COMMIT}:{VECTOR_PATH}"], cwd=root, capture_output=True,
                       check=False)
    if r.returncode != 0 or not r.stdout:
        raise VectorRejected(f"정본 벡터가 {RELEASE_VECTOR_PATH} · {VECTOR_PATH} 에도, 커밋 {VECTOR_COMMIT[:12]} 의 blob 에도 없다")
    if hashlib.sha256(r.stdout).hexdigest() != VECTOR_SHA256:
        raise VectorRejected(f"커밋 {VECTOR_COMMIT[:12]} 의 blob 의 sha256 이 고정값 {VECTOR_SHA256} 과 다르다")
    return r.stdout, "commit"


def run_vectors(vectors: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """사례마다 다섯 값을 기대와 맞댄다(값과 자료형 — `_same`). `gen_stop` 열쇠가 없는 사례는 인자를 생략해 부른다."""
    rows, bad = [], []
    for v in vectors:
        got = (extract(v["text"], v["gen_stop"]) if "gen_stop" in v else extract(v["text"])).as_dict()
        diff = sorted(k for k in _FIELDS if not _same(got[k], v["expect"][k]))
        rows.append({"id": v["id"], **got,
                     "narrative_sha256": None if got["narrative"] is None
                     else hashlib.sha256(got["narrative"].encode("utf-8")).hexdigest()})
        if diff:
            bad.append({"id": v["id"], "fields": diff, "expect": {k: v["expect"][k] for k in diff},
                        "got": {k: got[k] for k in diff}})
    return {"n": len(rows), "agree": len(rows) - len(bad), "mismatches": bad, "rows": rows,
            "extractor_id": EXTRACTOR_ID, "extractor_sha256": source_sha256()}
