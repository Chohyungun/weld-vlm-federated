"""2차 **자동** 판독의 기록 계약 — 근사 중복 규칙 재설계의 판독자 일치율용.

## 이것이 무엇이 아닌가

**사람 검수가 아니다.** 두 번째 자동 판독이다. 같은 이름을 헐겁게 쓰면 오늘 닫은 문이 다시 열린다 —
기록 경로만 있으면 사람 표본 검수를 거쳤다고 받아 주던 자리를 막았는데, 여기서 "검수" 라고 부르면
그 결정이 무의미해진다.

그래서 이 모듈은 **끌 수 없는 표시 둘**을 산출물에 박는다.

- `read_type = "automated_second_read"` — 생성자가 다른 값을 받지 않는다.
- `human_review = false` — 값이 아니라 **상수**다.

그리고 `limits` 에 `"사람 판독 미측정"` 이 **항상** 들어간다. 지우면 시험이 깨진다.

## 판독 기준은 여기 없다

어휘도 규칙도 이 모듈에 없다. **판독자가 기준을 정하면 독립이 아니다.** 기준은 1차 판독자가 주고,
이 모듈은 그 기준의 id 와 해시를 받아 기록만 한다. 기준 밖 어휘가 들어오면 거부한다.

## 표시는 **읽는 쪽에서 요구해야** 뜻이 있다

박아 두기만 하면 아무것도 막지 못한다. 쓰는 코드가 없거나, 있어도 표시를 안 보면 그냥 주석이다.
같은 구조를 오늘 세 번 봤다 — 기록 경로만 있으면 통과시키던 검사, 부르는 코드가 없던 봉인 검사기,
그리고 이 모듈의 첫 판.

그래서 **소비 경로를 여기 둔다.** `require_second_read` 가 받은 사전의 표시를 보고 아니면 거부한다.
`as_dict()` 를 거치지 않고 손으로 만든 사전은 `contract.sha256` 이 맞지 않아 거부된다 — 맞추려면
표시를 전부 올바르게 담아야 하고, 그러면 계약을 지킨 것이다.

**관문은 판독 내용까지 본다.** 계약 지문은 봉인 **값**과 판독 **수**만 담는다. 그래서 관문이 사전의 판독
목록에서 봉인을 다시 계산해 사전의 봉인 값과 맞댄다 — 앞 판은 이것을 되살리기(`rebuild`)에서만 해서,
관문만 부르는 소비자는 저장 뒤 고친 판독을 받았다.

## 순서를 봉인한다

1차 판정을 본 뒤에 내 판정을 고치면 일치율이 뜻을 잃는다. `seal()` 이 내 판독 전량의 지문을
돌려준다 — **1차 판정을 받기 전에** 그 지문을 남겨 두면, 나중에 고치지 않았다는 것이 지문으로 선다.
`agreement()` 는 봉인 지문이 그대로인지 먼저 본다.

그 증명은 봉인 값이 **언제·어디에** 남았는지에 달렸다. 그래서 `agreement()` 는 봉인 값만이 아니라 그 값을 남긴
**커밋과 경로**(`SealAnchor`)를 받아 일치율 산출에 싣는다. 봉인이 1차 판정보다 앞섰는지는 그 커밋과 1차 판정을
받은 기록을 맞대 읽는 사람이 확인한다 — 이 모듈은 그 순서를 스스로 증명하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

READ_TYPE = "automated_second_read"
"""**바꿀 수 없다.** 이 산출물이 사람 검수로 읽히는 것을 막는 표시다."""

HUMAN_REVIEW = False
"""상수다. 필드가 아니라 상수인 이유는 켤 수 있으면 켜지기 때문이다."""

REQUIRED_LIMIT = "사람 판독 미측정"
"""한계 목록에 **항상** 들어간다. 2차 자동 판독은 사람 판독을 대신하지 않는다."""


CONTRACT_VERSION = "second_read/1"
"""계약 판. 표시의 구성이 바뀌면 올린다 — 옛 산출물이 조용히 통과하지 않게."""

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_READER = re.compile(r"^(?:[A-F]|tool:[a-z0-9_.\-]{1,40})$")
"""판독자는 한 글자 표지(`A`~`F`)나 `tool:<id>` 뿐이다.

자유 문자열이면 `"사람 검수단"` 같은 값이 들어간다. 그러면 산출물이 사람 검수로 읽히는 길이
`read_type` 을 우회해 열린다 — 오늘 닫은 문과 같은 선이다.
"""


class SealBroken(RuntimeError):
    """봉인 뒤 판독이 바뀌었다 — 일치율을 내지 않는다."""


@dataclass(frozen=True)
class SealAnchor:
    """봉인 값과 그 값을 **남긴 자리**. 1차 판정을 받기 전에 이 커밋의 이 경로에 봉인 값이 적혀 있어야 한다."""

    sha256: str
    commit: str
    """봉인 값을 적은 커밋(소문자 40자리)."""
    path: str
    """그 커밋에서 봉인 값이 적힌 파일의 저장소 상대 경로."""

    def __post_init__(self) -> None:
        if not _HEX64.match(self.sha256 or ""):
            raise ValueError("봉인 값이 sha256(소문자 64자리)이 아니다")
        if not _HEX40.match(self.commit or ""):
            raise ValueError("봉인을 남긴 커밋이 SHA(소문자 40자리)가 아니다")
        if not self.path or self.path.startswith("/") or "\\" in self.path or ".." in self.path.split("/"):
            raise ValueError("봉인을 남긴 경로가 저장소 상대 경로가 아니다")

    def as_dict(self) -> dict:
        return {"sha256": self.sha256, "commit": self.commit, "path": self.path,
                "note": ("봉인 값을 남긴 자리다. 봉인이 1차 판정보다 앞섰는지는 이 커밋과 1차 판정을 받은 "
                         "기록을 맞대 확인한다 — 이 산출물이 그 순서를 스스로 증명하지 않는다")}


class NotSecondRead(RuntimeError):
    """받은 사전이 2차 자동 판독의 계약을 지키지 않는다. **사유를 전부** 들고 있다."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("2차 자동 판독 계약 불충족: " + " · ".join(self.reasons))


@dataclass(frozen=True)
class Criteria:
    """1차 판독자가 준 기준. **내용을 여기서 해석하지 않는다.**"""

    criteria_id: str
    criteria_sha256: str
    allowed_verdicts: tuple[str, ...]
    source: str
    """누가 줬나. 판독자가 스스로 만든 기준이면 독립이 아니다."""

    def __post_init__(self) -> None:
        if not self.criteria_id:
            raise ValueError("기준의 id 가 있어야 한다 — 어느 판으로 읽었는지가 결과의 일부다")
        if not _HEX64.match(self.criteria_sha256 or ""):
            # 한 글자짜리도 통과하면 "해시를 적었다" 가 아무 뜻이 없다.
            raise ValueError("기준 해시가 sha256(소문자 64자리)이 아니다")
        if len(self.allowed_verdicts) < 2:
            raise ValueError("어휘가 둘 미만이면 판독이 아니다")
        if len(set(self.allowed_verdicts)) != len(self.allowed_verdicts):
            raise ValueError("어휘에 중복이 있다")


@dataclass(frozen=True)
class Read:
    """쌍 하나의 판독. `note` 는 사유이고 판정에 쓰이지 않는다."""

    pair_id: str
    verdict: str
    note: str = ""


@dataclass
class SecondRead:
    """2차 자동 판독 한 벌."""

    criteria: Criteria
    reads: tuple[Read, ...]
    reader: str = "D"
    limits: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not _READER.match(self.reader or ""):
            raise ValueError(
                f"판독자는 한 글자 표지(A~F)나 'tool:<id>' 여야 한다 — 받은 값 {self.reader!r}. "
                "자유 문자열이면 사람 검수처럼 보이는 값이 들어간다")
        ids = [r.pair_id for r in self.reads]
        if len(set(ids)) != len(ids):
            raise ValueError("같은 쌍을 두 번 판독했다")
        bad = sorted({r.verdict for r in self.reads} - set(self.criteria.allowed_verdicts))
        if bad:
            raise ValueError(f"기준 밖 어휘: {bad} — 판독자가 어휘를 늘리지 않는다")
        if REQUIRED_LIMIT not in self.limits:
            self.limits = (REQUIRED_LIMIT, *self.limits)

    def seal(self) -> str:
        """판독 전량의 지문. **1차 판정을 받기 전에** 이 값을 남긴다."""
        return _seal_of(self.criteria.criteria_id, self.criteria.criteria_sha256,
                        ((r.pair_id, r.verdict) for r in self.reads))

    def as_dict(self) -> dict:
        body = self._payload()
        body["contract"] = {"version": CONTRACT_VERSION, "sha256": contract_digest(body)}
        return body

    def _payload(self) -> dict:
        return {
            "read_type": READ_TYPE,
            "human_review": HUMAN_REVIEW,
            "what_this_is_not": ("사람 검수가 아니다. 두 번째 자동 판독이다 — "
                                 "사람 판독을 대신하지 않는다"),
            "reader": self.reader,
            "criteria": {"id": self.criteria.criteria_id,
                         "sha256": self.criteria.criteria_sha256,
                         "allowed_verdicts": list(self.criteria.allowed_verdicts),
                         "source": self.criteria.source,
                         "note": "기준은 1차 판독자가 줬다. 2차 판독자가 정하지 않았다"},
            "n_reads": len(self.reads),
            "seal_sha256": self.seal(),
            "reads": [{"pair_id": r.pair_id, "verdict": r.verdict, "note": r.note}
                      for r in self.reads],
            "limits": list(self.limits),
        }


def _seal_of(criteria_id, criteria_sha256, pairs) -> str:
    """봉인 지문의 정의. 판독 객체와 관문이 **같은 함수**로 계산한다."""
    body = json.dumps(
        {"criteria": [criteria_id, criteria_sha256], "reads": sorted(pairs)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _seal_from_payload(payload: Mapping) -> str | None:
    """사전의 판독 목록에서 봉인을 다시 계산한다. 목록의 꼴이 틀리면 `None`."""
    c = payload.get("criteria") or {}
    reads = payload.get("reads")
    if not isinstance(reads, list) or not all(
            isinstance(r, Mapping) and isinstance(r.get("pair_id"), str)
            and isinstance(r.get("verdict"), str) for r in reads):
        return None
    return _seal_of(c.get("id"), c.get("sha256"), ((r["pair_id"], r["verdict"]) for r in reads))


def contract_digest(payload: Mapping) -> str:
    """표시 전량에서 계산한 지문. **손으로 만든 사전은 이 값을 맞출 수 없다** —
    맞추려면 표시를 전부 올바르게 담아야 하고, 그러면 계약을 지킨 것이다."""
    c = payload.get("criteria") or {}
    body = json.dumps({
        "v": CONTRACT_VERSION,
        "read_type": payload.get("read_type"),
        "human_review": payload.get("human_review"),
        "reader": payload.get("reader"),
        "criteria_id": c.get("id"),
        "criteria_sha256": c.get("sha256"),
        "allowed_verdicts": sorted(c.get("allowed_verdicts") or []),
        "source": c.get("source"),
        "n_reads": payload.get("n_reads"),
        "seal_sha256": payload.get("seal_sha256"),
        "limits": sorted(payload.get("limits") or []),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def require_second_read(payload: Mapping) -> dict:
    """**소비 경로의 관문.** 2차 자동 판독이 아니면 받지 않는다.

    보는 것 일곱. 하나라도 어긋나면 거부하고 **사유를 전부** 낸다.

    1. `read_type` 이 `automated_second_read` 다.
    2. `human_review` 가 **`False` 그 자체**다 — `0`·`""` 같은 거짓값을 받지 않는다.
    3. 한계 목록에 `사람 판독 미측정` 이 있다.
    4. `contract.version` 이 이 판이다.
    5. `contract.sha256` 이 사전의 표시에서 다시 계산한 값과 같다.
    6. **판독 목록에서 다시 계산한 봉인이 사전의 봉인 값과 같다** — 저장 뒤 고친 판독을 받지 않는다.
    7. 판독 수가 목록의 길이와 같다.

    Raises:
        NotSecondRead: 계약을 지키지 않으면.
    """
    reasons: list[str] = []
    if payload.get("read_type") != READ_TYPE:
        reasons.append(f"read_type 이 {payload.get('read_type')!r} 다 — {READ_TYPE!r} 여야 한다")
    if payload.get("human_review") is not False:
        reasons.append(f"human_review 가 {payload.get('human_review')!r} 다 — False 여야 한다")
    if REQUIRED_LIMIT not in (payload.get("limits") or ()):
        reasons.append(f"한계 목록에 {REQUIRED_LIMIT!r} 가 없다")
    contract = payload.get("contract") or {}
    if contract.get("version") != CONTRACT_VERSION:
        reasons.append(f"계약 판이 {contract.get('version')!r} 다 — {CONTRACT_VERSION!r} 여야 한다")
    elif contract.get("sha256") != contract_digest(payload):
        reasons.append("계약 지문이 사전의 표시와 맞지 않다 — as_dict() 를 거치지 않았거나 고쳐졌다")
    seal = _seal_from_payload(payload)
    if seal is None:
        reasons.append("판독 목록의 꼴이 틀렸다 — 쌍 id 와 판정이 문자열인 사전의 목록이어야 한다")
    else:
        if seal != payload.get("seal_sha256"):
            reasons.append("판독 목록에서 다시 계산한 봉인이 사전의 봉인 값과 다르다 — 저장 뒤 판독이 고쳐졌다")
        if payload.get("n_reads") != len(payload["reads"]):
            reasons.append("판독 수가 목록의 길이와 다르다")
    if reasons:
        raise NotSecondRead(reasons)
    return dict(payload)


def rebuild(payload: Mapping) -> SecondRead:
    """검증한 사전을 판독 객체로 되살린다. **봉인까지 맞아야 한다.**"""
    p = require_second_read(payload)
    c = p["criteria"]
    out = SecondRead(
        criteria=Criteria(c["id"], c["sha256"], tuple(c["allowed_verdicts"]), c["source"]),
        reads=tuple(Read(r["pair_id"], r["verdict"], r.get("note", "")) for r in p["reads"]),
        reader=p["reader"], limits=tuple(p["limits"]))
    if out.seal() != p["seal_sha256"]:
        raise SealBroken("사전의 봉인 지문이 판독 내용과 다르다 — 저장 뒤 고쳐졌다")
    return out


def agreement_from_payload(payload: Mapping, first: Mapping[str, str], *, sealed: SealAnchor,
                           first_source: str = "") -> dict:
    """저장된 산출물로 일치율을 낸다. **관문을 지나야 들어온다.**"""
    return agreement(rebuild(payload), first, sealed=sealed, first_source=first_source)


def agreement(second: SecondRead, first: Mapping[str, str], *, sealed: SealAnchor,
              first_source: str = "") -> dict:
    """1차 판정과 맞대 **일치율**을 낸다.

    Args:
        first: 1차 판독자의 `쌍 id → 판정`.
        sealed: 1차 판정을 받기 **전에** 남겨 둔 봉인 지문과 **그 값을 남긴 커밋·경로**.
            산출물이 이것을 싣는다 — 방금 계산한 봉인을 넘기면 순서의 증명이 되지 않는다.

    Raises:
        SealBroken: 봉인 뒤 판독이 바뀌었으면.
        ValueError: 두 판독의 쌍 집합이 다르거나, 1차가 기준 밖 어휘를 쓰면.

    일치율은 **두 자동 판독이 얼마나 같은가**이지 어느 쪽이 맞다는 근거가 아니다. 둘 다 틀릴 수 있고,
    같은 기준을 읽은 두 구현이라 오류가 **상관될** 수 있다. 그래서 정답률이라고 부르지 않는다.
    """
    if not isinstance(sealed, SealAnchor):
        raise TypeError("sealed 는 SealAnchor 다 — 봉인 값을 남긴 커밋과 경로가 함께 있어야 한다")
    if second.seal() != sealed.sha256:
        raise SealBroken("봉인 지문이 다르다 — 1차 판정을 본 뒤 판독이 바뀌었다")
    mine = {r.pair_id: r.verdict for r in second.reads}
    if set(mine) != set(first):
        only_mine = sorted(set(mine) - set(first))
        only_first = sorted(set(first) - set(mine))
        raise ValueError(f"쌍 집합이 다르다 — 내 쪽만 {len(only_mine)}개, 1차만 {len(only_first)}개")
    bad = sorted(set(first.values()) - set(second.criteria.allowed_verdicts))
    if bad:
        raise ValueError(f"1차 판정에 기준 밖 어휘가 있다: {bad}")

    n = len(mine)
    same = sum(1 for k in mine if mine[k] == first[k])
    confusion = Counter((first[k], mine[k]) for k in mine)
    by_verdict = {}
    for v in second.criteria.allowed_verdicts:
        n_v = sum(1 for k in mine if first[k] == v)
        hit = sum(1 for k in mine if first[k] == v and mine[k] == v)
        by_verdict[v] = {"n_first": n_v, "n_agreed": hit,
                         "rate": (hit / n_v) if n_v else None}
    return {
        "read_type": READ_TYPE,
        "human_review": HUMAN_REVIEW,
        "comparison": "두 자동 판독의 일치율. 정답률이 아니다",
        "caveat": ("같은 기준을 읽은 두 구현이라 오류가 상관될 수 있다. "
                   "일치가 높아도 둘 다 틀릴 수 있다"),
        "first_source": first_source,
        "seal": sealed.as_dict(),
        "n_pairs": n,
        "n_agreed": same,
        "agreement_rate": same / n if n else None,
        "by_first_verdict": by_verdict,
        "confusion": {f"{a}->{b}": c for (a, b), c in sorted(confusion.items())},
        "disagreed_pairs": sorted(k for k in mine if mine[k] != first[k]),
        "limits": list(second.limits),
    }


_TOKEN_SEP = re.compile(r"[\s,;|]+")
"""판정처럼 보이는 열을 찾을 때 쓰는 구분자. 공백만 보면 `p1,duplicate` 가 id 에 붙은 채 들어온다."""


def load_pairs(raw: str, *, allowed: Sequence[str]) -> tuple[str, ...]:
    """판독할 쌍 목록을 읽는다. **판정은 들어 있지 않아야 한다.**

    1차 판정이 같은 파일에 섞여 오면 독립이 깨진다. 판정처럼 보이는 열이 있으면 거부한다 —
    공백·쉼표·세미콜론·세로줄로 나눈 토큰 가운데 하나라도 판정 어휘면 거부다.
    줄은 **LF 로만** 나눈다(`str.splitlines()` 는 U+2028·U+0085 에서도 쪼갠다). 줄 끝의 CR 은 공백으로 벗긴다.
    """
    ids: list[str] = []
    for n, line in enumerate(raw.split("\n"), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if any(v in _TOKEN_SEP.split(s) for v in allowed):
            raise ValueError(f"{n}행에 판정처럼 보이는 값이 있다 — 쌍 목록만 받는다")
        ids.append(s)
    if len(set(ids)) != len(ids):
        raise ValueError("쌍 목록에 중복이 있다")
    return tuple(ids)
