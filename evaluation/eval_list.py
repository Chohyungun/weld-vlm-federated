"""평가 목록 파일의 **정규형** — 내보내기 쪽과 채점 쪽이 같은 함수를 부른다. 07번 미니스펙 §14-1 자.

목록은 계약의 일부다. "모든 id 를 한 번씩, 목록 순서 그대로" 가 성립하려면 **그 목록이 무엇인지** 가 먼저 한 가지여야 한다.
쓰는 쪽과 읽는 쪽이 각자 읽으면 BOM·CRLF·끝 개행·공백 하나로 갈린다 — 그 차이가 해시를 바꾸고, 바뀐 해시가 정상 묶음을 거부한다.

## 정규형

한 줄에 id 하나. UTF-8, **LF**, BOM 없음, 끝 개행 있음, 빈 줄 없음, 앞뒤 공백 없음, 중복 없음.
**정렬은 요구하지 않는다** — 순서 자체가 계약이기 때문이다(배치 구획이 목록 순서를 따른다).

## 두 해시

| 이름 | 무엇 | 무엇을 묶나 |
|---|---|---|
| `file_sha256` | 파일 바이트 | **순서까지** — 배치 구획이 여기 달려 있다 |
| `set_sha256` | 코드포인트로 정렬한 id 를 LF 로 이은 것(끝 개행 없음) | 집합 신원 — 부표본이 전량의 부분집합인지 같은 잣대로 본다 |

둘 다 산출물과 곁 파일에 싣는다. 하나만 두면 "같은 집합을 다른 순서로 돌린 것" 과 "다른 집합" 을 구분하지 못한다.

## 검사 없이 만들 수 없다

`EvalList` 는 생성될 때 **자기 id 에서 두 해시를 다시 계산해** 받은 값과 맞댄다. 그래서 아무 해시나 적어
만든 목록 객체는 존재할 수 없고, 묶음 검증이 곁 파일의 해시와 맞대는 값은 언제나 id 에서 나온 값이다.
만드는 함수(`canonical_bytes`)와 검사하는 함수(`validate_eval_list`)는 줄 하나의 규칙을 같은 함수(`_line_problem`)에서 읽는다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from evaluation.reject_v14 import RejectCode, Rejection

MAX_ID_LEN = 512
"""id 한 줄의 길이 상한. 넘으면 목록 파일이 아니라 다른 것을 읽고 있는 것이다."""

_BOM = chr(0xFEFF)


def _line_problem(line: str) -> tuple[str, dict] | None:
    """id 한 줄이 정규형에 맞는가. **만드는 쪽과 검사하는 쪽이 이 함수 하나를 부른다.**

    Returns:
        맞으면 `None`, 아니면 `(사유, 덧붙일 값)`. 덧붙일 값에 id 를 넣지 않는다.
    """
    if line == "":
        return "빈 줄", {}
    if line != line.strip():
        return "앞뒤 공백", {}
    if len(line) > MAX_ID_LEN:
        return "id 가 너무 길다", {"length": len(line)}
    return None


def set_digest(ids) -> str:
    """집합 신원. 정렬해 LF 로 잇고 **끝 개행을 붙이지 않는다** — 파일 해시와 헷갈리지 않게."""
    return hashlib.sha256("\n".join(sorted(set(ids))).encode("utf-8")).hexdigest()


def canonical_bytes(ids) -> bytes:
    """정규형 바이트를 만든다. **순서는 준 그대로** 둔다 — 정렬하지 않는다.

    이 함수가 낸 바이트는 `validate_eval_list` 를 지난다. 검사기가 거부할 목록은 여기서도 만들지 않는다
    (빈 목록 · 상한을 넘는 id · BOM 으로 시작하는 첫 id 를 포함한다).

    Raises:
        ValueError: 정규형으로 적을 수 없는 목록이면. 문구에 id 값을 넣지 않고 **몇 번째인지**만 말한다.
    """
    out = list(ids)
    if not out:
        raise ValueError("정규형에 맞지 않는 목록: 빈 목록")
    seen: set[str] = set()
    for n, x in enumerate(out, 1):
        if not isinstance(x, str):
            raise ValueError(f"정규형에 맞지 않는 id({n}번째): 문자열이 아니다")
        if "\n" in x or "\r" in x:
            raise ValueError(f"정규형에 맞지 않는 id({n}번째): 줄바꿈이 들어 있다")
        problem = _line_problem(x)
        if problem is not None:
            raise ValueError(f"정규형에 맞지 않는 id({n}번째): {problem[0]}")
        if x in seen:
            raise ValueError(f"목록에 중복 id 가 있다({n}번째)")
        seen.add(x)
    if out[0].startswith(_BOM):
        raise ValueError("정규형에 맞지 않는 목록: 첫 id 가 BOM 으로 시작한다")
    return ("\n".join(out) + "\n").encode("utf-8")


@dataclass(frozen=True)
class EvalList:
    """검사를 지난 목록. **`validate_eval_list` 로 만든다** — 직접 만들면 같은 검사를 여기서 다시 받는다."""

    ids: tuple[str, ...]
    file_sha256: str
    set_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.ids, tuple):
            raise ValueError("ids 는 튜플이어야 한다 — 만든 뒤 바뀌지 않아야 해시가 뜻을 가진다")
        raw = canonical_bytes(self.ids)
        if self.file_sha256 != hashlib.sha256(raw).hexdigest():
            raise ValueError("file_sha256 이 ids 의 정규형 바이트에서 나온 값이 아니다")
        if self.set_sha256 != set_digest(self.ids):
            raise ValueError("set_sha256 이 ids 에서 나온 값이 아니다")

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def id_set(self) -> frozenset[str]:
        return frozenset(self.ids)


def validate_eval_list(raw: bytes, *, where: str = "eval_list") -> EvalList:
    """목록 파일의 바이트를 정규형으로 검사하고 읽는다.

    **바이트를 받는다** — 경로나 문자열을 받으면 호출부마다 다른 방식으로 디코드하게 되고, 그 차이가 해시를 가른다.
    첫 위반에서 멈춘다(사유는 하나다).

    Raises:
        BundleRejected: 정규형에서 벗어나면. 사유는 `LIST_NOT_CANONICAL` 하나이고 `detail.reason` 이 무엇인지 말한다.
    """
    from evaluation.reject_v14 import BundleRejected

    def bad(reason: str, **more):
        raise BundleRejected([Rejection(RejectCode.LIST_NOT_CANONICAL, where,
                                        {"reason": reason, **more})])

    if raw.startswith(b"\xef\xbb\xbf"):
        bad("BOM")
    if b"\r" in raw:
        bad("CR")
    if raw == b"":
        bad("빈 파일")
    if not raw.endswith(b"\n"):
        bad("끝 개행 없음")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        bad("UTF-8 아님", at=exc.start)

    lines = text[:-1].split("\n")
    seen: set[str] = set()
    for n, line in enumerate(lines, 1):
        problem = _line_problem(line)
        if problem is not None:
            bad(problem[0], line=n, **problem[1])
        if line in seen:
            bad("중복", line=n)
        seen.add(line)

    return EvalList(ids=tuple(lines),
                    file_sha256=hashlib.sha256(raw).hexdigest(),
                    set_sha256=set_digest(lines))
