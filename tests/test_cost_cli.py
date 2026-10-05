"""비용 계측 CLI 의 출력부와 산출 경로 — `train`·`gen`·`both` 세 경로.

## 무엇을 막는가

**(1) 없는 키를 찍다가 죽는 것.** `main()` 의 마지막 요약이 `환산_전량평가` 를 찾았는데
`measure_gen` 은 그 키를 내지 않는다(`환산_곡선기준`·`환산_자유길이_상한` 을 낸다).
JSON 저장은 그 앞에서 끝나므로 **측정값을 잃는 오류가 아니라 정상 종료를 막는 오류**였다.
그래도 반환값이 0 이 아니면 호출한 쪽은 실패로 읽는다.

**(2) 이전 계측을 덮는 것.** 산출 경로가 고정 파일 하나였다. `--part gen` 실행이 직전
`--part train` 산출물을 덮었고 실제로 두 번 그렇게 됐다. 이제 `--run-id` 로 갈리고
같은 경로가 있으면 **거부한다**(측정은 GPU 시간을 쓴 결과라 되돌릴 수 없다).

## 어떻게 재는가

측정 함수를 가짜 반환값으로 바꿔 **CPU 로만** 돈다. GPU 를 다시 돌려 검증하지 않는다.
가짜 반환에는 실제 반환 구조의 키만 넣는다 — `환산_전량평가` 를 **넣지 않으므로**, 세 경로가
0 을 반환하는 것이 곧 그 키를 더는 참조하지 않는다는 뜻이다.

이 시험이 보증하지 않는 것: 실제 측정값·성능·VRAM·재개·모델 품질. 출력부와 경로만 본다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("torch", reason="계측 모듈이 최상위에서 torch 를 가져온다")

from scripts.probe import cost_4b  # noqa: E402

FAKE_TRAIN = {
    "정상상태": {"n_steps": 5, "샘플당_s_평균": 3.64, "샘플당_s_중앙값": 3.62,
               "peak_vram_gb": 5.675},
    "환산": {"epoch당_시간_h_평균": 45.38},
    "최장타깃_단발": {"n_defects": 14, "감독토큰": 444, "wall_s": 5.85, "peak_vram_gb": 5.65},
}
#: **`환산_전량평가` 가 없다.** 이것이 이 시험의 요점이다.
FAKE_GEN = {
    "생성곡선": {"측정": [{"고정_길이": 64, "지연_s_중앙값": 9.92}], "적합": {"초당_토큰": 6.86},
             "로그확률_부담": None},
    "환산_곡선기준": {"n_eval": 12461, "장당_초_예시": {"64": 9.92}},
    "실행별_자유길이": [],
    "환산_자유길이_상한": {"n_eval": 12461, "한도512_5모델_h": 679.8},
}


@pytest.fixture
def stub(monkeypatch):
    """측정 함수를 가짜로 바꾸고 호출 인자를 기록한다."""
    calls: list[tuple[str, dict]] = []

    def fake_train(**kw):
        calls.append(("train", kw))
        return dict(FAKE_TRAIN)

    def fake_gen(**kw):
        calls.append(("gen", kw))
        return dict(FAKE_GEN)

    monkeypatch.setattr(cost_4b, "measure_train", fake_train)
    monkeypatch.setattr(cost_4b, "measure_gen", fake_gen)
    # 커밋 해시를 읽는 `os.popen` 도 막는다 — 시험이 저장소 상태에 의존하지 않게.
    monkeypatch.setattr(cost_4b.os, "popen", lambda _c: _Fake("stub"))
    return calls


class _Fake:
    def __init__(self, text: str) -> None:
        self._t = text

    def read(self) -> str:
        return self._t


def _run(monkeypatch, part: str, out: Path) -> int:
    monkeypatch.setattr(sys, "argv", [
        "cost_4b", "--part", part, "--run-id", "t", "--out", str(out)])
    return cost_4b.main()


@pytest.mark.parametrize(("part", "want"), [
    ("train", {"학습"}),
    ("gen", {"생성"}),
    ("both", {"학습", "생성"}),
])
def test_세_경로가_정상_종료하고_저장한다(monkeypatch, tmp_path, stub, part, want):
    out = tmp_path / f"{part}.json"
    assert _run(monkeypatch, part, out) == 0, f"{part}: 요약 출력에서 죽는다"
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert want <= set(rep), f"{part}: 저장된 절이 빠졌다"
    assert rep["part"] == part and rep["run_id"] == "t"
    assert {c[0] for c in stub} == {"train" if p == "학습" else "gen" for p in want}


def test_생성_요약이_없는_키를_찾지_않는다(monkeypatch, tmp_path, stub, capsys):
    """가짜 반환에 `환산_전량평가` 가 없는데 0 을 반환하면 참조가 사라진 것이다."""
    out = tmp_path / "gen.json"
    assert "환산_전량평가" not in FAKE_GEN
    assert _run(monkeypatch, "gen", out) == 0
    text = capsys.readouterr().out
    assert "곡선 기준" in text and "자유 길이" in text
    assert "상한" in text, "자유 길이 값이 상한이라는 표시가 요약에서 빠졌다"


def test_기존_산출물이_있으면_거부한다(monkeypatch, tmp_path, stub):
    out = tmp_path / "gen.json"
    out.write_text('{"먼저": "있던 계측"}', encoding="utf-8")
    assert _run(monkeypatch, "gen", out) == 2, "덮어쓰기를 막지 않았다"
    assert json.loads(out.read_text(encoding="utf-8")) == {"먼저": "있던 계측"}
    assert not stub, "거부하기 전에 측정을 시작했다 — GPU 시간을 쓰고 나서 거부하면 늦다"


def test_run_id_가_없으면_실행되지_않는다(monkeypatch, tmp_path, stub):
    """고정 경로로 조용히 쓰던 경로를 되살리지 않는다."""
    monkeypatch.setattr(sys, "argv", ["cost_4b", "--part", "gen"])
    with pytest.raises(SystemExit):
        cost_4b.main()
    assert not stub


def test_기존_계측_경로가_상수로_남아_있다():
    """09-21 산출물을 가리키는 경로를 지우지 않았다는 것을 고정한다."""
    assert cost_4b.LEGACY_OUT.name == "cost_4b.json"
    assert cost_4b.OUT_DIR.name == "cost_4b"
    assert cost_4b.LEGACY_OUT != cost_4b.OUT_DIR
