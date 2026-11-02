"""서술 추출기(채점 쪽, `evaluation.narrative_extract`) — 공통 벡터 정본 전량 · 벡터 바이트 고정 · 문자열 보존 · R1 순서 · 정본 파서 경계.

벡터의 기대는 구현 전에 손으로 고정했다(03f · 03f-2 — 2026-10-22 커밋 e3e0ac8a). 이 시험은 기대를 구현에 맞춰 고치지 않는다 —
어긋나면 원인(전달 · 계산 · 계약 미정)을 먼저 찾는다(03 2판 §5b).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import narrative_conformance as NC
from evaluation import narrative_extract as NE
from evaluation.adapters_v14 import COUPLING_RULE_V2, UpstreamContractError, adapt_unified_main
from evaluation.reject_v14 import RejectCode

REPO = Path(__file__).resolve().parents[1]
DRAFT_V2 = REPO / "docs/dev_log/2026-10-20-설계복구/03f-2_추출벡터_D.json"
DRAFT_V1 = REPO / "docs/dev_log/2026-10-20-설계복구/03f_추출벡터_초안_D.json"
VECTORS = NC.load_vectors()
J = '{"defects":[],"verdict":"판정불가","cited_clauses":[]}'


def test_벡터_정본은_핀한_바이트이고_dev_log_의_03f_2_와_같다() -> None:
    raw = (REPO / NC.VECTOR_REL).read_bytes()
    assert NC.sha256_bytes(raw) == NC.VECTOR_SHA256
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    if not DRAFT_V2.exists():
        pytest.skip("dev_log 의 03f-2 가 없는 체크아웃")
    assert DRAFT_V2.read_bytes() == raw


def test_03f_의_36_사례는_내용_그대로_옮겼다() -> None:
    if not DRAFT_V1.exists():
        pytest.skip("dev_log 의 03f 가 없는 체크아웃")
    old = json.loads(DRAFT_V1.read_bytes().decode("utf-8"))["vectors"]
    assert len(old) == 36 and VECTORS[:36] == old


@pytest.mark.parametrize("vector", VECTORS, ids=[v["id"] for v in VECTORS])
def test_벡터마다_다섯_값이_기대와_같다(vector: dict) -> None:
    assert NC.call(NE.extract, vector).as_dict() == {k: vector["expect"][k] for k in NC.FIELDS}


def test_문자열은_파일의_이스케이프에서_풀려_그대로_남는다() -> None:
    """파일 바이트의 해시와 따로 — 풀린 문자열이 CRLF · 전각 공백 · 폭 없는 공백을 품는다(195 M3)."""
    by_id = {v["id"]: v for v in VECTORS}
    assert by_id["C2"]["text"].count("\r\n") == 3 and by_id["C2"]["expect"]["narrative"].count("\r\n") == 1
    assert by_id["F6"]["expect"]["narrative"].count("\r\n") == 1
    assert by_id["C4"]["text"].count("\u3000") == 2 and "\t" in by_id["C4"]["text"]
    assert by_id["G2"]["text"][0] == "\u200b" and by_id["G2"]["expect"]["narrative"][0] == "\u200b"
    raw = (REPO / NC.VECTOR_REL).read_bytes()
    assert b"\\u3000" in raw and b"\\u200b" in raw and "\u3000".encode() not in raw and "\u200b".encode() not in raw


@pytest.mark.parametrize("stop", ["eos", "length", "stop_undetermined", None, "stop", "EOS", "", "<칸 없음>"])
def test_종료_사유_검사가_해독보다_먼저다_1_4_어댑터와_같은_값을_거부한다(stop) -> None:
    """R1 — 1.4 어댑터는 `gen_stop` 을 해독보다 먼저 본다. 어댑터가 거부하는 값 = 추출기의 검증 오류인 값."""
    line = {"image_id": "t"} if stop == "<칸 없음>" else {"image_id": "t", "gen_stop": stop}
    with pytest.raises(UpstreamContractError) as err:
        adapt_unified_main([json.dumps(line)], tag="uni_central", seed=1, coupling_rule=COUPLING_RULE_V2,
                           known_iso_codes=(), scoring_iso_codes=())
    valid = stop in ("eos", "length")
    assert ("gen_stop" in str(err.value)) == (not valid)
    assert ("n_new_tokens" in str(err.value)) == valid              # 계약 안의 값이면 다음 칸에서 멈춘다
    got = NE.extract(J + "\n가") if stop == "<칸 없음>" else NE.extract(J + "\n가", stop)
    assert (got.state == NE.VALIDATION_ERROR) == (not valid)


def test_묶음_검증의_거부_어휘가_상류_미정_표시와_같다() -> None:
    assert RejectCode.STOP_UNDETERMINED.value == "stop_undetermined"
    assert NE.extract(J + "\n가", "stop_undetermined").state == NE.VALIDATION_ERROR


def test_글_길이_상한은_정본_파서의_경계_그대로다() -> None:
    big = J + "\n" + "가" * ((4 << 20) // 3 + 10)                   # UTF-8 4 MiB 초과
    got = NE.extract(big, "eos")
    assert (got.state, got.upstream_reason) == (NE.RECORD_FAIL, "json_decode")


def test_칸_없음과_null_은_둘_다_검증_오류다() -> None:
    assert NE.extract(J + "\n가").state == NE.VALIDATION_ERROR
    assert NE.extract(J + "\n가", None).state == NE.VALIDATION_ERROR


def test_글자가_아닌_입력은_멈춘다() -> None:
    with pytest.raises(TypeError):
        NE.extract(b"{}", "eos")                                      # type: ignore[arg-type]


def test_첫_객체의_범위가_참조_파서와_갈리면_멈춘다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NE, "decode_line", lambda raw: ({"defects": [1]}, None))
    with pytest.raises(RuntimeError, match="참조 파서"):
        NE.extract(J + "\n가", "eos")


def test_추출은_정규화하지_않는다() -> None:
    """P5 — 전각 숫자 · ㎜ · 겹 공백은 추출문에 그대로 남는다(정규화는 검사기의 일)."""
    got = NE.extract(J + "\n１２ ㎜  가  나", "eos")
    assert got.narrative == "１２ ㎜  가  나" and got.state == NE.COMPLETE
