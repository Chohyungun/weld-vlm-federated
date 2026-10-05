"""통합형 채점 원장 — 07번 §12-5 · §31-4 · §32-8 · 진입점 미니스펙 3판 3-5 · 8-3.

지키는 것.

1. 줄은 정규 JSON 한 줄이고 앞 줄의 sha 로 잇는다. 중간을 지우거나 고치거나 순서를 바꾸면 읽기가 멈춘다.
2. 쓴 줄을 되읽어 확인한다. 예약 키(`kind` · `at` · `prev_sha256`)를 payload 가 덮지 못한다. naive 시각은 받지 않는다.
3. 판정 한 번 규칙은 **내용 해시**로 찾는다 — 원시 해시가 달라도 같은 내용이면 걸린다.
4. 쓰는 쪽의 시도 수는 그 목적 · 그 생성 지문의 `call` 줄만 센다 — 재생성 줄은 세지 않는다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from evaluation.scoring_ledger import (GENESIS, LedgerChainBroken, append_line, count_calls, find_verdicts,
                                       read_ledger)

KST = timezone(timedelta(hours=9))
T = datetime(2026, 10, 1, 12, 0, tzinfo=KST)


def call(purpose="frame_diag", gen="g" * 64):
    return {"purpose": purpose, "generation_sha256": gen, "registration_mode": "probe", "run_kind": purpose,
            "bundles": [{"tag": "uni_central", "seed_index": 1, "mode": "model",
                         "generations_sha256": "r" * 64, "generations_content_sha256": "c" * 64}]}


def verdict(content="c" * 64, raw="r" * 64):
    return {"call_sha256": "x" * 64, "generation_sha256": "g" * 64, "status": "pass",
            "bundles": [{"generations_sha256": raw, "generations_content_sha256": content}]}


def test_빈_원장은_빈_목록이다(tmp_path) -> None:
    assert read_ledger(tmp_path / "none.jsonl") == []


def test_줄은_앞_줄의_sha_로_잇는다(tmp_path) -> None:
    p = tmp_path / "ledger.jsonl"
    a = append_line(p, "call", call(), at=T)
    b = append_line(p, "frame_diag_verdict", verdict(), at=T)
    rows = read_ledger(p)
    assert [s for s, _ in rows] == [a, b]
    assert rows[0][1]["prev_sha256"] == GENESIS and rows[1][1]["prev_sha256"] == a


@pytest.mark.parametrize("damage", ["drop_first", "edit_first", "swap"])
def test_지우거나_고치거나_바꾸면_읽기가_멈춘다(tmp_path, damage: str) -> None:
    p = tmp_path / "ledger.jsonl"
    for _ in range(3):
        append_line(p, "call", call(), at=T)
    lines = p.read_bytes().split(b"\n")[:-1]
    if damage == "drop_first":
        lines = lines[1:]
    elif damage == "edit_first":
        lines[0] = lines[0].replace(b'"probe"', b'"strict"')
    else:
        lines[0], lines[1] = lines[1], lines[0]
    p.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(LedgerChainBroken):
        read_ledger(p)


def test_정규형이_아닌_줄은_받지_않는다(tmp_path) -> None:
    p = tmp_path / "ledger.jsonl"
    append_line(p, "call", call(), at=T)
    p.write_bytes(p.read_bytes().replace(b",", b", ", 1))
    with pytest.raises(LedgerChainBroken, match="정규형"):
        read_ledger(p)


def test_모르는_종류는_쓰지도_읽지도_않는다(tmp_path) -> None:
    with pytest.raises(ValueError):
        append_line(tmp_path / "l.jsonl", "note", {}, at=T)


def test_예약_키와_naive_시각은_받지_않는다(tmp_path) -> None:
    with pytest.raises(ValueError, match="예약 키"):
        append_line(tmp_path / "l.jsonl", "call", {"prev_sha256": GENESIS}, at=T)
    with pytest.raises(ValueError, match="naive"):
        append_line(tmp_path / "l.jsonl", "call", call(), at=datetime(2026, 10, 1))


def test_판정_한_번_규칙은_내용_해시로_찾는다(tmp_path) -> None:
    """원시 해시가 달라도(지연만 다르게 다시 내보냈다) 내용 해시가 같으면 걸린다(07번 §32-2)."""
    p = tmp_path / "ledger.jsonl"
    append_line(p, "call", call(), at=T)
    append_line(p, "frame_diag_verdict", verdict(content="c" * 64, raw="r" * 64), at=T)
    assert len(find_verdicts(p, ["c" * 64])) == 1
    assert find_verdicts(p, ["d" * 64]) == []


def test_시도_수는_그_목적과_생성_지문의_call_줄만_센다(tmp_path) -> None:
    p = tmp_path / "ledger.jsonl"
    append_line(p, "call", call(), at=T)
    append_line(p, "call", call(purpose="rehearsal"), at=T)
    append_line(p, "call", call(gen="h" * 64), at=T)
    append_line(p, "frame_diag_verdict", verdict(), at=T)
    append_line(p, "frame_diag_regenerated", {"verdict_sha256": "v" * 64, "artifact": "a.json",
                                              "artifact_sha256": "s" * 64}, at=T)
    assert count_calls(p, purpose="frame_diag", generation_sha256="g" * 64) == 1
