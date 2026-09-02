"""인용 금지 표식 — 이빨 시험. 80번 G11-4 · G2.

이 검사의 위험은 **과차단**이다. `sep_fed` 는 논문 본체인 분리형 칸이고 금지 대상은
회계 두 파일뿐이다. 디렉터리를 통째로 막으면 써도 되는 가중치까지 막혀서 다음 사람이
표식을 지운다 — 그러면 통제가 통째로 사라진다.

그래서 시험은 양방향으로 건다. **막아야 할 것을 막는가**와 **막으면 안 되는 것을
통과시키는가**를 같은 무게로 본다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.citation_ban import BANS, MARKER, is_banned, parse_marker, render

REPO_ROOT = Path(__file__).resolve().parents[1]


def _place(tmp_path: Path, ban) -> Path:
    d = tmp_path / Path(ban.directory).name
    d.mkdir(parents=True, exist_ok=True)
    (d / MARKER).write_text(render(ban), encoding="utf-8", newline="\n")
    return d


def test_스펙이_비어있지_않다():
    assert BANS, "금지 목록이 비었다 — 80번 G11-4 가 넷을 지정한다"
    assert {Path(b.directory).name for b in BANS} == {
        "uni_central", "uni_fed", "predictions", "sep_fed"}


@pytest.mark.parametrize("ban", BANS, ids=lambda b: Path(b.directory).name)
def test_표식이_기계_판독_가능하다(tmp_path: Path, ban):
    d = _place(tmp_path, ban)
    spec = parse_marker(d / MARKER)
    assert spec is not None, "JSON 블록을 못 읽는다"
    assert spec["spec"] == "do-not-cite/1"
    assert list(spec["banned"]) == list(ban.banned)
    assert list(spec["allowed"]) == list(ban.allowed)


@pytest.mark.parametrize("ban", BANS, ids=lambda b: Path(b.directory).name)
def test_금지_대상은_실제로_막힌다(tmp_path: Path, ban):
    d = _place(tmp_path, ban)
    probe = ban.banned[0] if ban.banned[0] != "*" else "아무파일.npz"
    banned, why = is_banned(d / probe)
    assert banned, f"{probe} 가 통과했다 — 표식이 물지 않는다"
    assert ban.reason_id in why


@pytest.mark.parametrize("ban", [b for b in BANS if b.allowed],
                         ids=lambda b: Path(b.directory).name)
def test_허용_대상은_통과한다(tmp_path: Path, ban):
    """과차단 시험. sep_fed 가중치·sep_* 검출 예측이 여기서 막히면 안 된다."""
    d = _place(tmp_path, ban)
    for name in ban.allowed:
        assert not is_banned(d / name)[0], f"{name} 이 막혔다 — 과차단이다"


def test_표식이_없으면_통과한다(tmp_path: Path):
    (tmp_path / "x.npz").write_text("x", encoding="utf-8")
    assert not is_banned(tmp_path / "x.npz")[0]


def test_표식이_깨졌으면_안전하게_금지로_본다(tmp_path: Path):
    """읽을 수 없는 표식을 '금지 없음'으로 읽으면 표식을 망가뜨리는 것이 우회로가 된다."""
    (tmp_path / MARKER).write_text("# 표식인데 JSON 블록이 없다\n", encoding="utf-8")
    banned, why = is_banned(tmp_path / "무엇이든.csv")
    assert banned
    assert "읽을 수 없다" in why


def test_하위_디렉터리까지_상속된다(tmp_path: Path):
    ban = next(b for b in BANS if b.banned == ("*",))
    d = _place(tmp_path, ban)
    sub = d / "runs" / "r000_c0"
    sub.mkdir(parents=True)
    assert is_banned(sub / "weights.pt")[0]


def test_sep_fed_범위가_80번과_일치한다():
    """가중치·예측을 막으면 안 되고 회계는 막아야 한다 — 이 경계가 80번의 핵심이다."""
    ban = next(b for b in BANS if b.directory.endswith("sep_fed"))
    assert set(ban.banned) == {"accounting.csv", "audit.json"}
    for name in ("global_r003.npz", "latest.npz", "atomic_log.csv"):
        assert name in ban.allowed, f"{name} 이 허용 목록에 없다"


def test_predictions_범위가_80번과_일치한다():
    ban = next(b for b in BANS if b.directory.endswith("predictions"))
    assert set(ban.banned) == {"uni_central.generations.jsonl", "uni_fed.generations.jsonl"}
    assert all(n.startswith("sep_") for n in ban.allowed)


# ---------------------------------------------------------------------------------
# 실물 산출물 — 워크트리에 있을 때만
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("ban", BANS, ids=lambda b: Path(b.directory).name)
def test_실물_표식이_제자리에_있다(ban):
    d = REPO_ROOT / ban.directory
    if not d.is_dir():
        pytest.skip(f"{ban.directory} 가 워크트리에 없다")
    marker = d / MARKER
    assert marker.is_file(), (
        f"{ban.directory}/{MARKER} 이 없다 — `uv run python scripts/citation_ban.py --write`")
    spec = json.loads(json.dumps(parse_marker(marker)))
    assert list(spec["banned"]) == list(ban.banned)
