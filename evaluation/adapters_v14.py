"""통합형 본실험 생성문 → 계약 #4 **1.4 레코드**. 07번 미니스펙 §12-4 · §13-3 · §14-1 · §16-2.

`evaluation/adapters.py` 는 고치지 않는다 — 사전실험 재채점 경로가 그 바이트에 기대고, v3 산출물이 파일별 해시를 싣는다.
새 결합 규칙은 여기 새 함수로 선다.

## 새 결합 규칙 `decoupled_v2`

옛 규칙(v1)은 판정·인용 필드가 깨지면 **레코드 전체**를 빈 예측으로 버렸다. 박스가 멀쩡해도 그 이미지는 전량 미검출이 됐다.

| 무엇이 깨졌나 | v1 | **v2** |
|---|---|---|
| JSON 이 없다·읽히지 않는다·잘렸다 | 레코드 실패 | 레코드 실패 (같다) |
| `defects` 가 목록이 아니다 | 레코드 실패 | 레코드 실패 (같다) |
| `verdict` 가 어휘 밖 | 레코드 실패 | **박스는 채점. `clause_error` 에 적고 조항 축에서만 실패** |
| `cited_clauses` 가 문자열 목록이 아니다 | 레코드 실패 | 〃 |
| 결함 항목 하나가 깨졌다 | 그 항목만 폐기 | 그 항목만 폐기 (같다 — 공통 정책) |

**`1.4 ⇔ decoupled_v2`.** 1.4 레코드는 이 규칙으로만 만들어진다. 규칙 id 는 **등록 객체에서 온다** —
이 함수에 기본값이 없다. 기본값을 두면 main 호출이 인자를 빠뜨렸을 때 조용히 옛 규칙으로 채점된다.

## 어떤 값도 지어내지 않는다

`verdict` 가 어휘 밖이면 `null` 이다. 1.3 의 `failed_record` 는 `판정불가` 를 채워 넣었는데, 그것은 모델이 하지 않은 말이다.
`cited_clauses` 가 깨지면 빈 목록이고, **키가 없는 것과 빈 목록을 낸 것을 가른다** — 상류가 키 누락을 `null` 로 넘기기로 했다(§14-1 바).

## 모델의 실패와 배관의 고장을 가른다

레코드 실패를 **판정하는 것은 상류(내보내기)뿐이다.** 이 어댑터는 상류가 적어 보낸 사유를 승계할 뿐 새로 판정하지 않는다.
실패율은 모델의 것만 세어야 하므로, 상류 줄이 **스스로 모순이면 세지 않고 멈춘다**(`UpstreamContractError`).

짝 규칙(07번 §13-3 다-10): `parse_error` 가 null 이면 `bbox_px_parsed` 는 `defects` 가 목록인 객체이고,
`parse_error` 가 null 이 아니면 `bbox_px_parsed` 는 null 이다. 사유는 비었는데 파싱 결과가 비거나 틀린 줄,
사유가 있는데 파싱 결과도 있는 줄은 모델이 낸 것이 아니라 **쓰는 쪽이 낸 것**이다.

`gen_stop` 이 `stop_undetermined` 인 줄도 같다. 그 표시가 있는 묶음은 묶음 검증이 먼저 거부한다(§16-2 m-2).
여기까지 왔다면 검증을 거치지 않은 것이므로, 종료 사유를 지어내 레코드를 만들지 않고 멈춘다.

## 파서는 모델 출력으로 죽지 않는다

깊은 중첩은 `RecursionError`, 수천 자리 정수는 `ValueError` 를 낸다. 둘 다 `JSONDecodeError` 가 아니다.
파서가 죽으면 그 이미지를 영원히 못 넘긴다 — greedy 는 다시 돌려도 같은 글을 낸다. 그래서 **모든 해독 실패를 분류로 떨어뜨린다**(§14-1 마).

## 참조 파서

`reference_parse` 는 **생성문 글자**에서 출발해 읽는 쪽의 판정까지 한 번에 간다(§14-1 아). 적합성 벡터
(`evaluation.conformance_v14`)가 이 함수를 통과해야 하고, 쓰는 쪽의 파서도 같은 벡터를 통과해야 한다.
좌표는 **모델이 낸 값 그대로**다 — 역변환 전이다. 규약별 역변환은 문자 일치 카나리아의 일이다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, get_args

from pydantic import ValidationError

from evaluation.policy import DEFECT_ITEM_POLICY, filter_defect_items
from evaluation.schema import Verdict
from evaluation.schema_v14 import (
    SCHEMA_VERSION_V14,
    PredictionRecordV14,
    failed_record_v14,
    tag_parts,
)

COUPLING_RULE_V2 = "decoupled_v2"
"""이 모듈이 구현하는 결합 규칙의 id. 등록 객체의 값이 이것과 다르면 1.4 레코드를 만들지 않는다."""

_VERDICTS = frozenset(get_args(Verdict))

STOP_UNDETERMINED = "stop_undetermined"
"""상류가 `gen_stop` 을 정할 수 없을 때 적는 표시. 상류는 중단하지 않고 적어 두며 **묶음 검증이 거부**한다(§16-2 m-2).
이 어댑터는 그 줄을 만나면 멈춘다 — 레코드 자료형에 없는 값이라 채우려면 지어내야 한다."""

PARSE_ERRORS: tuple[str, ...] = ("no_json", "json_decode", "truncated", "schema_violation")
"""레코드 실패 사유의 어휘(계약 #4 1.4). 상류가 적는 값과 참조 파서가 내는 값이 같은 어휘다."""

_MAX_JSON_BYTES = 4 << 20
"""한 줄에서 JSON 으로 읽어 볼 최대 길이. 넘으면 생성문이 아니라 다른 것이다 — 해독을 시도하지 않는다."""

OUTCOME_OK = "ok"
OUTCOME_RECORD_FAIL = "record_fail"


@dataclass
class AdaptReportV14:
    """어댑터 통과 결과. **버린 것과 고른 규칙을 건수로 증명한다.**

    `evaluation.adapters.AdaptReport` 를 재사용하지 않는다 — 그 dataclass 는 검출 export 와 파일럿이 함께 쓰고
    산출물에 키가 그대로 나간다. 필드를 더하면 그쪽 산출물이 달라진다.

    **어댑터가 새로 판정한 레코드 실패라는 칸은 없다.** 레코드 실패는 상류가 적은 것만 있고,
    상류 줄이 모순이면 세지 않고 멈춘다. 언제나 0 인 칸을 두면 "재서 0 이었다" 로 읽힌다.
    """

    records: list[PredictionRecordV14] = field(default_factory=list)
    coupling_rule: str = COUPLING_RULE_V2
    n_lines: int = 0
    upstream_failures: dict[str, int] = field(default_factory=dict)
    """상류(내보내기)가 실패로 적어 보낸 것. 재시도 없이 그대로 승계한다. **레코드 실패의 전부다.**"""
    clause_errors: dict[str, int] = field(default_factory=dict)
    """조항 축 위반. 레코드 실패와 **서로소**다 — 합산 위반율이 같은 줄을 두 번 세지 않게."""
    n_boxes: int = 0
    bounds_measured: bool = False
    """이미지 크기를 받았는가. 받지 않았으면 경계 이탈은 **재지 않은 것**이지 0 이 아니다."""
    n_boxes_out_of_bounds: int = 0
    n_boxes_size_unknown: int = 0
    """크기표를 받았으나 그 이미지가 표에 없어 이탈을 재지 못한 박스."""
    n_bad_items: int = 0
    n_unknown_code: int = 0
    n_out_of_scope: int = 0
    out_of_scope_codes: dict[str, int] = field(default_factory=dict)
    citations: dict[str, list[str]] = field(default_factory=dict)
    gen_stop_counts: dict[str, int] = field(default_factory=dict)
    parse_by_stop: dict[str, int] = field(default_factory=dict)
    """`gen_stop` × 레코드 실패 여부의 교차표 — 길이 종료와 JSON 절단이 다른 사실임을 보이는 재료."""

    @property
    def n_record_failures(self) -> int:
        return sum(self.upstream_failures.values())

    @property
    def n_clause_error_records(self) -> int:
        return sum(1 for r in self.records if r.clause_error)

    def as_dict(self) -> dict:
        n = len(self.records)
        return {
            "coupling_rule": self.coupling_rule,
            "schema_version": SCHEMA_VERSION_V14,
            "n_lines": self.n_lines,
            "n_records": n,
            "upstream_parse_failures": dict(sorted(self.upstream_failures.items())),
            "n_record_failures": self.n_record_failures,
            "record_failure_rate": self.n_record_failures / n if n else 0.0,
            "clause_errors": dict(sorted(self.clause_errors.items())),
            "n_clause_error_records": self.n_clause_error_records,
            "clause_error_rate": self.n_clause_error_records / n if n else 0.0,
            "combined_violation_rate": (
                (self.n_record_failures + self.n_clause_error_records) / n if n else 0.0),
            "n_boxes": self.n_boxes,
            "out_of_bounds_measured": self.bounds_measured,
            "n_boxes_out_of_bounds": self.n_boxes_out_of_bounds if self.bounds_measured else None,
            "n_boxes_size_unknown": self.n_boxes_size_unknown if self.bounds_measured else None,
            "discard_policy": DEFECT_ITEM_POLICY,
            "n_bad_items_dropped": self.n_bad_items,
            "n_unknown_code_dropped": self.n_unknown_code,
            "n_out_of_scope_dropped": self.n_out_of_scope,
            "out_of_scope_codes": dict(sorted(self.out_of_scope_codes.items())),
            "n_images_with_citation": sum(1 for v in self.citations.values() if v),
            "gen_stop_counts": dict(sorted(self.gen_stop_counts.items())),
            "parse_by_stop": dict(sorted(self.parse_by_stop.items())),
        }


class UpstreamContractError(ValueError):
    """상류 줄이 계약을 어겼다 — 모델의 실패가 아니라 **배관 고장**이다.

    모델이 무엇을 냈든 상류는 정해진 필드를 정해진 자료형으로 실어야 한다. 그것이 깨진 것을 레코드 실패로 적으면
    배관 고장이 모델의 점수로 기록된다.
    """


def decode_line(raw: str) -> tuple[Any, str | None]:
    """생성문에서 첫 JSON 객체를 꺼낸다. **어떤 입력에도 예외를 내지 않는다.**

    Returns:
        `(객체, None)` 또는 `(None, 실패 사유)`. 사유는 1.4 어휘(`no_json`·`json_decode`·`truncated`)다.
    """
    s = raw.strip()
    if len(s.encode("utf-8", "ignore")) > _MAX_JSON_BYTES:
        return None, "json_decode"
    i = s.find("{")
    if i < 0:
        return None, "no_json"
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[i:])
    except json.JSONDecodeError as exc:
        incomplete = exc.pos >= len(s[i:]) or exc.msg.startswith("Unterminated string")
        return None, "truncated" if incomplete else "json_decode"
    except RecursionError:
        return None, "json_decode"      # 깊은 중첩 — 죽지 않고 분류로 떨어뜨린다
    except ValueError:
        return None, "json_decode"      # 수천 자리 정수 등
    return obj, None


def _clause_fields(parsed: Mapping) -> tuple[str | None, list[str], list[str]]:
    """판정·인용을 읽고 **값을 고치지 않는다.**

    Returns:
        `(verdict, cited_clauses, clause_error)`.
    """
    errors: list[str] = []

    verdict = parsed.get("verdict")
    if not (isinstance(verdict, str) and verdict in _VERDICTS):
        verdict = None
        errors.append("verdict_invalid")

    cited = parsed.get("cited_clauses")
    if cited is None or not (isinstance(cited, list) and all(isinstance(c, str) for c in cited)):
        cited = []
        errors.append("citations_invalid")

    return verdict, cited, sorted(errors)


@dataclass(frozen=True)
class ReferenceParse:
    """생성문 하나를 읽은 결과 — 적합성 벡터가 맞대는 일곱 값."""

    outcome: str
    """`ok`(레코드가 선다) 또는 `record_fail`."""
    reason: str | None
    """`record_fail` 일 때의 사유(1.4 어휘). `ok` 면 `None`."""
    kept: tuple[tuple, ...] = ()
    """유지되는 결함 항목 `(iso_code, x1, y1, x2, y2)` — **원시 순서대로**, 모델이 낸 좌표 그대로."""
    dropped: int = 0
    """폐기한 항목 수 — 좌표가 깨져 버린 것과 공통 정책이 버린 것(미지 코드·채점 밖 코드)의 **합**."""
    verdict: str | None = None
    cited: tuple[str, ...] = ()
    clause_error: tuple[str, ...] = ()

    def as_tuple(self) -> tuple:
        return (self.outcome, self.kept, self.dropped, self.verdict, self.cited,
                self.clause_error, self.reason)


def reference_parse(text: str, *, known_iso_codes: Iterable[str],
                    scoring_iso_codes: Sequence[str]) -> ReferenceParse:
    """읽는 쪽의 **참조 파서**. 생성문 글자에서 결합 규칙 `decoupled_v2` 의 판정까지 간다.

    순서는 §14-2 의 4 다 — `defects`(레코드 실패) → 항목 폐기 → `verdict`·`cited_clauses`(조항 축).
    모델 출력의 어떤 자료형에도 예외를 내지 않는다.

    항목의 좌표는 `bbox_2d` 에서 읽는다. 길이 4 의 목록이 아니거나, 수로 읽히지 않거나, 유한하지 않거나,
    퇴화한 상자면 그 항목만 버린다. 남은 항목은 공통 폐기 정책(`evaluation.policy.filter_defect_items`)이
    코드로 한 번 더 거른다. 좌표를 바꾸지 않는다.
    """
    obj, why = decode_line(text)
    if why is not None:
        return ReferenceParse(OUTCOME_RECORD_FAIL, why)
    raw_defects = obj.get("defects") if isinstance(obj, Mapping) else None
    if not isinstance(raw_defects, list):
        return ReferenceParse(OUTCOME_RECORD_FAIL, "schema_violation")

    items: list[dict] = []
    n_bad = 0
    for d in raw_defects:
        box = d.get("bbox_2d") if isinstance(d, Mapping) else None
        if not isinstance(box, list) or len(box) != 4:
            n_bad += 1
            continue
        try:
            xyxy = [float(v) for v in box]
        except (TypeError, ValueError, OverflowError):
            n_bad += 1
            continue
        if not all(math.isfinite(v) for v in xyxy) or xyxy[0] >= xyxy[2] or xyxy[1] >= xyxy[3]:
            n_bad += 1
            continue
        items.append({"iso_code": d.get("iso_code", ""), "bbox_px": xyxy})

    filt = filter_defect_items(items, known_codes=set(known_iso_codes),
                               scoring_codes=list(scoring_iso_codes))
    verdict, cited, clause_error = _clause_fields(obj)
    kept = tuple((d["iso_code"], *d["bbox_px"]) for d in filt.kept)
    return ReferenceParse(OUTCOME_OK, None, kept, n_bad + filt.n_dropped,
                          verdict, tuple(cited), tuple(clause_error))


@dataclass(frozen=True)
class _Line:
    """줄 하나에서 읽어 낸 공통 값. **반복 변수를 닫아 두지 않으려고** 값으로 넘긴다."""

    where: str
    image_id: str
    cell: str
    client: str | None
    seed: int
    gen_stop: str
    n_new_tokens: int
    common: dict


def _build(where: str, make):
    """레코드 생성자를 부르고, 자료형 규칙 위반을 **줄 번호와 함께** 배관 고장으로 올린다."""
    try:
        return make()
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "레코드"
        raise UpstreamContractError(f"{where}: 레코드 규칙 위반 ({loc}) — {first.get('msg', '')}") from None


def _record_failure(rep: AdaptReportV14, line: _Line, reason: str) -> None:
    """상류가 적어 보낸 레코드 실패 하나를 승계한다."""
    rep.upstream_failures[reason] = rep.upstream_failures.get(reason, 0) + 1
    rep.records.append(_build(line.where, lambda: failed_record_v14(
        line.image_id, cell=line.cell, client=line.client, seed=line.seed, error=reason,
        gen_stop=line.gen_stop, n_new_tokens=line.n_new_tokens,
        coord_space=line.common["coord_space"], coord_cfg_hash=line.common["coord_cfg_hash"],
        latency_ms=line.common["latency_ms"], raw_output_ref=line.common["raw_output_ref"],
    )))
    key = f"{line.gen_stop}|fail"
    rep.parse_by_stop[key] = rep.parse_by_stop.get(key, 0) + 1


def adapt_unified_main(
    lines: Iterable[str],
    *,
    tag: str,
    seed: int,
    coupling_rule: str,
    known_iso_codes: Iterable[str],
    scoring_iso_codes: Sequence[str],
    image_size: Mapping[str, tuple[int, int]] | None = None,
) -> AdaptReportV14:
    """본실험 통합형의 원시 생성문 줄들을 1.4 레코드로 바꾼다.

    Args:
        lines: `generations.jsonl` 의 줄. 호출부가 **LF 로만** 나눠 넘긴다.
        tag: `uni_local_C1` … `uni_fed`. 칸과 참여자가 여기서 나온다.
        seed: 시드 **값**(번호가 아니다).
        coupling_rule: 등록 객체의 결합 규칙 id. **기본값이 없다.**
        known_iso_codes: 사상표 전체 코드.
        scoring_iso_codes: 채점 4클래스.
        image_size: 경계 이탈 **집계용**. 판정에 쓰지 않는다. 주지 않으면 이탈은 재지 않은 것으로 나간다.

    Raises:
        ValueError: 등록된 규칙이 이 모듈의 것이 아니면. 1.4 레코드는 `decoupled_v2` 로만 만들어진다.
        UpstreamContractError: 상류 줄이 계약을 어기면 — 짝 규칙 위반과 `stop_undetermined` 를 포함한다.
    """
    if coupling_rule != COUPLING_RULE_V2:
        raise ValueError(
            f"1.4 레코드는 {COUPLING_RULE_V2!r} 로만 만든다 — 등록된 규칙은 {coupling_rule!r} 이다"
        )
    cell, client = tag_parts(tag)
    known = set(known_iso_codes)
    scoring = list(scoring_iso_codes)
    rep = AdaptReportV14(coupling_rule=coupling_rule, bounds_measured=image_size is not None)

    for n, raw in enumerate(lines, 1):
        where = f"{tag}:{n}"
        if not raw.strip():
            raise UpstreamContractError(f"{where}: 빈 줄 — 줄 수가 계약의 일부다")
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise UpstreamContractError(f"{where}: 줄이 JSON 이 아니다 ({type(exc).__name__})") from None
        if not isinstance(row, dict):
            raise UpstreamContractError(f"{where}: 줄이 객체가 아니다")

        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise UpstreamContractError(f"{where}: image_id 가 없다")

        gen_stop = row.get("gen_stop")
        if gen_stop == STOP_UNDETERMINED:
            raise UpstreamContractError(
                f"{where}: gen_stop 이 {STOP_UNDETERMINED} 다 — 묶음 검증이 거부했어야 하는 줄이다. "
                "종료 사유를 지어내 레코드를 만들지 않는다")
        if gen_stop not in ("eos", "length"):
            raise UpstreamContractError(f"{where}: gen_stop 이 계약 밖이다 ({gen_stop!r})")

        n_new = row.get("n_new_tokens")
        if type(n_new) is not int or n_new < 0:
            raise UpstreamContractError(f"{where}: n_new_tokens 가 0 이상의 정수가 아니다")

        upstream_bad = row.get("n_bad_items_dropped", 0)
        if type(upstream_bad) is not int or upstream_bad < 0:
            raise UpstreamContractError(f"{where}: n_bad_items_dropped 가 0 이상의 정수가 아니다")

        upstream_err = row.get("parse_error")
        if upstream_err is not None and upstream_err not in PARSE_ERRORS:
            raise UpstreamContractError(f"{where}: 계약 밖 parse_error {upstream_err!r}")

        # ── 짝 규칙 — 사유와 파싱 결과가 서로 맞는가. 모순이면 모델이 아니라 쓰는 쪽이 낸 줄이다.
        parsed = row.get("bbox_px_parsed")
        if upstream_err is not None:
            if parsed is not None:
                raise UpstreamContractError(
                    f"{where}: parse_error 가 {upstream_err!r} 인데 bbox_px_parsed 가 null 이 아니다")
        else:
            if not isinstance(parsed, Mapping):
                raise UpstreamContractError(
                    f"{where}: parse_error 가 null 인데 bbox_px_parsed 가 객체가 아니다")
            if not isinstance(parsed.get("defects"), list):
                raise UpstreamContractError(
                    f"{where}: parse_error 가 null 인데 bbox_px_parsed.defects 가 목록이 아니다")

        # 여기까지 온 줄만 센다 — 멈춘 줄은 집계에 들어가지 않는다.
        rep.n_lines += 1
        rep.gen_stop_counts[gen_stop] = rep.gen_stop_counts.get(gen_stop, 0) + 1
        rep.n_bad_items += upstream_bad

        common = {
            "coord_space": row.get("coord_space"),
            "coord_cfg_hash": row.get("coord_cfg_hash"),
            "latency_ms": row.get("latency_ms"),
            "raw_output_ref": row.get("raw_output_ref"),
        }
        line = _Line(where=where, image_id=image_id, cell=cell, client=client, seed=seed,
                     gen_stop=gen_stop, n_new_tokens=n_new, common=common)

        if upstream_err is not None:
            _record_failure(rep, line, upstream_err)
            continue

        # 여기부터는 박스를 채점한다 — 판정·인용이 깨져도 레코드를 버리지 않는다.
        verdict, cited, clause_error = _clause_fields(parsed)
        for e in clause_error:
            rep.clause_errors[e] = rep.clause_errors.get(e, 0) + 1

        filt = filter_defect_items(parsed["defects"], known_codes=known, scoring_codes=scoring)
        rep.n_bad_items += filt.n_bad_item
        rep.n_unknown_code += filt.n_unknown_code
        rep.n_out_of_scope += filt.n_out_of_scope
        for c, k in filt.out_of_scope_codes.items():
            rep.out_of_scope_codes[c] = rep.out_of_scope_codes.get(c, 0) + k

        defects = [
            {
                "iso_code": d["iso_code"],
                "bbox_px": d["bbox_px"],
                "score": None,      # 생성 모델은 신뢰도를 내지 않는다. 지어내지 않는다
                "size_px": max(d["bbox_px"][2] - d["bbox_px"][0],
                               d["bbox_px"][3] - d["bbox_px"][1]),
                "size_basis": "major_axis",
                "retrieved": None,  # 통합형에는 검색을 붙이지 않는다
            }
            for d in filt.kept
        ]

        wh = None if image_size is None else image_size.get(image_id)
        for d in defects:
            rep.n_boxes += 1
            if image_size is None:
                continue
            if wh is None:
                rep.n_boxes_size_unknown += 1
                continue
            x1, y1, x2, y2 = d["bbox_px"]
            if x1 < 0 or y1 < 0 or x2 > wh[0] or y2 > wh[1]:
                rep.n_boxes_out_of_bounds += 1

        rep.citations[image_id] = list(cited)
        rep.records.append(_build(where, lambda: PredictionRecordV14(
            schema_version=SCHEMA_VERSION_V14, image_id=image_id,
            cell=cell, client=client, seed=seed,            # type: ignore[arg-type]
            defects=defects,                                # type: ignore[arg-type]
            verdict=verdict, cited_clauses=list(cited),     # type: ignore[arg-type]
            parse_ok=True, parse_error=None, clause_error=clause_error,  # type: ignore[arg-type]
            gen_stop=gen_stop, n_new_tokens=n_new,          # type: ignore[arg-type]
            **common,                                       # type: ignore[arg-type]
        )))
        key = f"{gen_stop}|ok"
        rep.parse_by_stop[key] = rep.parse_by_stop.get(key, 0) + 1

    return rep
