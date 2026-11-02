"""프롬프트 v3.1(2 차 레시피의 새 판) — C 08 2 판 Q5 · 09 §7 결정 6(지시 1024f 과제 6).

- v3 초안(`unified_v3_basis.txt`)은 바이트 그대로 남는다 — 덮지 않는다(`tests/test_prompt_v3.py` 가 그 해시를 고정한다).
- v3.1 은 v3 의 넷째 줄 '적용 조항' 을 정답 서술 2 판의 낱말 '대응 조항' 으로 맞춘 것이다. 다른 줄 · 다른 글자는 같다.
- 학습 등록의 레시피 매핑(`vlm.registration.REGIME_PROMPTS` · `REGIME_PROMPT_SHA256`) · 하네스 04 · 결과 읽기 05a 가 같은 자리 · 해시를 읽는다.
  1 차 체제의 값(v2)은 그대로다.
- 바뀐 줄 ∩ 금지 문자 = ∅ — v3 시험과 같은 금지 목록이다. 이 검사는 지시문의 뜻이나 모델의 복사를 검증하지 않는다(A2 의 실측이다).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_prompt_v3 import FORBIDDEN_CHARS, FORBIDDEN_WORDS, V2_SHA, V3, V3_SHA
from vlm.pilot_vlm import load_prompt

V3_1 = ROOT / "vlm/prompts/unified_v3_1_basis.txt"
V3_1_SHA = "dad1769abda33cb766556ee48f0973db298d4468251d7a94f0484e305623eb68"


def _load(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem.replace("-", "_"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_v3_1_은_v3_의_한_낱말만_바꾼_새_판이다():
    text, sha = load_prompt(V3_1)
    assert sha == V3_1_SHA and hashlib.sha256(V3_1.read_bytes()).hexdigest() == V3_1_SHA
    assert b"\r" not in V3_1.read_bytes()
    v3 = load_prompt(V3)[0]
    assert load_prompt(V3)[1] == V3_SHA                                 # v3 는 덮지 않았다
    assert v3.count("적용 조항") == 1 and text == v3.replace("적용 조항", "대응 조항")
    changed = [i for i, (a, b) in enumerate(zip(text.split("\n"), v3.split("\n"))) if a != b]
    assert changed == [3] and len(text.split("\n")) == len(v3.split("\n"))
    assert "적용" not in text


def test_바뀐_줄에는_금지_문자_낱말이_없다():
    line = load_prompt(V3_1)[0].split("\n")[3]
    assert not (set(line) & FORBIDDEN_CHARS), line
    assert not any(w in line for w in FORBIDDEN_WORDS), line


def test_학습_등록_하네스_결과_읽기가_같은_판을_읽는다():
    from vlm import registration as R

    assert R.REGIME_PROMPTS["json_line_then_basis"] == V3_1.relative_to(ROOT).as_posix()
    assert R.REGIME_PROMPT_SHA256["json_line_then_basis"] == V3_1_SHA
    assert R.REGIME_PROMPTS["json_only"] == "vlm/prompts/unified_v2_absorig.txt"
    assert R.REGIME_PROMPT_SHA256["json_only"] == V2_SHA                 # 1 차 체제 값 그대로
    h04 = _load("docs/dev_log/2026-10-19-본실험복구/04_암기시험_C.py")
    assert h04.PROMPT_2 == R.REGIME_PROMPTS["json_line_then_basis"]
    r05a = _load("docs/dev_log/2026-10-20-설계복구/05a_결과읽기_C.py")
    assert (r05a.PROMPT_2, r05a.PROMPT_2_SHA256) == (R.REGIME_PROMPTS["json_line_then_basis"], V3_1_SHA)
