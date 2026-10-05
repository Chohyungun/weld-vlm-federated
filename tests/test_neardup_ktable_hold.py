"""k 선택기의 **실행 보류와 구판 표시**를 고정한다.

무엇을 막는 시험인가. 선택 로직은 철회된 구판인데 독스트링만 새 정책으로 바뀐 적이 있다.
그러면 **철회된 규칙이 주석이 아니라 데이터로 배포된다** — 산출물을 받는 쪽은 그것을 현행 규칙으로 읽는다.
그래서 네 가지를 시험으로 박는다.

  1. 산출물의 `고르는_규칙` 이 철회된 규칙을 **현행처럼** 말하지 않는다
  2. `고른_k` 에 구판 값이 실리지 않는다. 구판 값은 이름 자체가 경고인 칸에만 있다
  3. 인정 인자 없이 돌리면 **산출물을 쓰지 않고** 까닭을 적고 **종료 코드 2** 로 끝난다
  4. 인정 인자를 준 경로도 같은 표시를 달고 나온다

**새 정책을 검증하는 시험이 아니다.** 새 정책은 구현되지 않았고 이 시험은 그 사실이 산출물에
드러나는지만 본다. 구현되면 이 시험을 고쳐야 한다 — 고칠 때 무엇이 바뀌는지 드러나게 하는 것이 목적이다.

**실물을 열지 않는다.** 매니페스트·간선·산출 경로를 전부 합성과 임시 경로로 갈아끼운 뒤 부른다.
실물 산출 경로를 보게 두면, 보류가 깨졌을 때 **덮어쓴 뒤에** 실패한다 — 시험이 사고를 한 번 더 낸다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "neardup" / "nd_stage14_ktable.py"
OUT_NAME = "stage14_ktable.json"

pytestmark = pytest.mark.skipif(not SRC.exists(), reason="k 표 스크립트가 없다")


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(SRC.parent))
    import nd_stage14_ktable as m
    return m


def _man(n: int = 200) -> dict:
    """합성 매니페스트. 재질×결함유무 네 층이 **각 50장씩** 고르게 있고 묶음은 전부 다르다.

    수를 200 으로 잡은 까닭이 있다. 더 작게 잡으면 어느 k 에서도 층화가 서지 않아
    구판 로직도 `min(ok)` 이 아니라 `None` 을 낸다 — 그러면 `고른_k` 시험이 **고치기 전 코드에서도
    통과해** 아무것도 고정하지 못한다. 층화가 실제로 서는 크기여야 그 시험이 구판과 현행을 가른다.
    각 층 50장이면 20% 목표가 정확히 10장이라 탐욕법이 편차 0 으로 맞춘다.
    """
    out = {}
    for i in range(n):
        out[f"x{i}"] = {
            "image_id": f"x{i}",
            "group_id": f"g{i}",
            "material": "ST" if i % 2 else "AL",
            "has_defect": "1" if (i // 2) % 2 else "0",
        }
    return out


@pytest.fixture
def 합성_실행(mod, monkeypatch, tmp_path):
    """실물을 전혀 열지 않고 `main()` 을 부르는 자리를 만든다.

    산출 경로(`W`)를 임시 폴더로 갈아끼우고 매니페스트·간선을 합성으로 바꾼다.
    보류가 깨지더라도 **실물 산출물을 덮어쓰지 않고 실물 매니페스트도 열지 않는다.**
    """
    monkeypatch.setattr(mod, "W", tmp_path)
    monkeypatch.setattr(mod, "load_manifest", lambda: _man())
    monkeypatch.setattr(mod, "load_edges", lambda rule: ([], {"쌍_전체": 0, "간선": 0,
                                                             "비간선": 0, "규칙_적용불가": 0, "묶음별": {}}))

    def run(*argv):
        monkeypatch.setattr(sys, "argv", [str(SRC), *argv])
        return mod.main()
    return run


def test_고르는_규칙이_철회된_규칙을_현행처럼_말하지_않는다(mod):
    out = mod.table(_man(), [], "시험")
    rule = out["고르는_규칙"]
    # 문자열 하나로 실리면 그대로 현행 규칙처럼 읽힌다. 상태가 붙은 구조여야 한다.
    assert isinstance(rule, dict), "철회 상태를 담을 수 없는 형태다"
    assert rule["새_정책_구현됨"] is False
    blob = str(rule)
    assert "구판" in blob and "철회" in blob, "산출물만 보고 구판임을 알 수 없다"
    assert "보류" in rule["상태"]


def test_고른_k_에_구판_값이_실리지_않는다(mod):
    out = mod.table(_man(), [], "시험")
    assert out["고른_k"] is None, "구판 로직의 값이 현행 선택으로 읽히는 이름에 실렸다"
    assert out["고른_k가_비어_있는_까닭"]
    # 값을 버리지는 않는다 — 이름 자체가 경고인 칸에 둔다.
    assert "구판_로직의_값_현행_선택_아님" in out


def test_인정_인자가_없으면_산출물을_쓰지_않고_2로_끝낸다(합성_실행, capsys, tmp_path):
    code = 합성_실행()
    # **0 이 아님이 아니라 2 로 박는다.** 사고로 1 이 나도 통과하면 시험이 아무것도 고정하지 못한다.
    assert code == 2, "문서와 소스가 약속한 종료 코드는 2 다"
    assert "보류" in capsys.readouterr().out, "까닭이 적히지 않았다"
    assert list(tmp_path.iterdir()) == [], "보류인데 산출물이 쓰였다"


def test_인정_인자를_주면_돌되_같은_표시를_달고_나온다(합성_실행, tmp_path):
    code = 합성_실행(mod_ack := "--구판-선택기-임을-인정함")
    assert code == 0
    written = tmp_path / OUT_NAME
    assert written.exists(), "인정했는데도 산출물이 없다"
    rep = json.loads(written.read_text(encoding="utf-8"))
    v1 = rep["v1"]
    # 인정 경로로 나온 산출물도 구판임을 달고 나와야 한다. 여기가 실제로 배포되는 쪽이다.
    assert v1["고른_k"] is None
    assert "구판_로직의_값_현행_선택_아님" in v1, "구판 값이 어디에도 없으면 새 정책과 대조할 수 없다"
    assert v1["고르는_규칙"]["새_정책_구현됨"] is False
    assert "구판" in v1["고르는_규칙"]["상태"]
    assert mod_ack  # 인자 이름을 그대로 쓴다


def test_인정_인자의_이름이_뜻을_담고_있다(mod):
    # 짧은 깃발은 뜻을 모르고도 붙는다. 이름만 보고 무엇을 인정하는지 알 수 있어야 한다.
    assert "구판" in mod.ACK
