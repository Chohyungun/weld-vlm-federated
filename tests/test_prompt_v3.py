"""프롬프트 v3(2 차 레시피) — 02 2판 §4-1 의 탐침 계획 2 · §3-5 의 새 시험 8(지시 1022b).

- v2(1 차 등록의 프롬프트)는 바이트 그대로다 — 1 차 체제 진단(A1 · A4 · A5)이 그 해시로 돈다.
- v3 은 02d 의 (나) 초안 문자열 그대로다(문안 확정은 A2 탐침 뒤 — 고치면 이 고정값도 같이 바뀐다).
- v2 에서 바뀐 줄(첫 줄 · 넷째 줄) ∩ 금지 문자 = ∅ — 틀 표지 · 말줄임 · 판정 값 낱말 · 길이 단위가 없다. 이 검사는 지시문의 뜻이나 모델의
  복사를 검증하지 않는다(그것은 A2 의 실측이다).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm.pilot_vlm import load_prompt

V2 = ROOT / "vlm/prompts/unified_v2_absorig.txt"
V3 = ROOT / "vlm/prompts/unified_v3_basis.txt"
#: 1 차 등록(`main-20261017-1.json` 의 `generation.prompt_sha256`)의 값.
V2_SHA = "08d3bda5777bb79583377b37194ee7c5bff33834a3aa564900cd0eb74c894a2b"
#: 02d 의 (나) 초안 문자열 해시(`prompt.draft_na.sha256_text`).
V3_SHA = "8b66edd7df56537ede0c08433abf0ff875d87785901c1aa32c067343d43b7fba"
FORBIDDEN_CHARS = set('<>|[]{}"\\/…')
FORBIDDEN_WORDS = ("합격", "불합격", "mm")


def test_v2_는_1차_등록의_바이트_그대로다():
    assert load_prompt(V2)[1] == V2_SHA
    assert b"\r" not in V2.read_bytes()


def test_v3_은_초안_문자열_그대로이고_v2_와는_두_줄만_다르다():
    text, sha = load_prompt(V3)
    assert sha == V3_SHA and hashlib.sha256(V3.read_bytes()).hexdigest() == V3_SHA     # 줄끝이 LF 그대로다
    v2 = load_prompt(V2)[0].split("\n")
    lines = text.split("\n")
    changed = [i for i, ln in enumerate(lines) if ln.strip() and ln not in v2]
    assert changed == [0, 3]
    assert lines[1] == v2[1] and lines[2] == v2[2]                                     # JSON 틀 · 좌표 안내는 v2 와 같은 줄


def test_v2_에서_바뀐_줄에는_금지_문자_낱말이_없다():
    text, _ = load_prompt(V3)
    v2 = set(load_prompt(V2)[0].split("\n"))
    changed = [ln for ln in text.split("\n") if ln.strip() and ln not in v2]
    assert changed
    for ln in changed:
        assert not (set(ln) & FORBIDDEN_CHARS), ln
        assert not any(w in ln for w in FORBIDDEN_WORDS), ln
