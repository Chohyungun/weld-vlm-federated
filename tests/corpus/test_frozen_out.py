"""봉인 산출 디렉터리 가드 — 이빨 시험. 09-08 총괄 지시(dispatch_B_동결가드).

`run_cycle_corpus --out` 기본값이 이미 봉인된 `cycle_pilot` 이었다. 인자 없는 실행 한 번이
논문에 실을 해시의 실물을 지운다(규약 1-6). `make_pairs_pilot` 은 더 나빴다 — 가드가 있긴
했는데 **경로가 v1 인지만 봤고**, 그 사이 봉인된 v2 가 기본값이라 있는 가드가 실제 위험을
통과시켰다.

그래서 여기 시험은 두 가지를 따로 건다.

* **막는가** — 계약(`SNAPSHOT.sha256`)이 있는 디렉터리를 겨눈 실행이 실제로 멈추는가.
* **그게 가드 때문인가** — 가드를 무력화하면 같은 실행이 가드 지점을 지나가는가.
  80번 G2 가 "실패할 수 없는 검사는 없는 것보다 나쁘다"고 적었다. 앞의 것만 걸면 다른
  이유로 멈춰도 통과한다.

GPU 는 쓰지 않는다. 무력화 쪽은 가드 바로 뒤 호출을 감시 예외로 갈아 끼워 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.generate import make_pairs_pilot as P
from corpus.generate import run_cycle_corpus as R
from corpus.generate import snapshot_cycle_pilot as S
from corpus.generate.frozen_out import (
    CONTRACT_NAME,
    FrozenDirectoryError,
    assert_not_frozen,
    is_frozen,
    snapshot_summary,
)
from corpus.validate import compare_judges as C

REPO = Path(__file__).resolve().parents[2]


class _Sentinel(RuntimeError):
    """가드를 지나갔다는 표시. 이 예외가 오르면 멈춘 이유가 가드가 아니었다."""


def _seal(d: Path, *, names=("pairs.jsonl", "counts.json"),
          digest="0" * 64) -> Path:
    """계약 파일만 둔 가짜 봉인 디렉터리. 실물 봉인본은 읽기만 한다."""
    d.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{'a' * 64}  {n}\n" for n in names)
    (d / CONTRACT_NAME).write_text(body + f"# snapshot_digest {digest}\n",
                                   encoding="utf-8")
    return d


# --------------------------------------------------------------------- 계약 판정

def test_계약이_없으면_통과한다(tmp_path: Path):
    assert not is_frozen(tmp_path)
    assert_not_frozen(tmp_path)            # 예외 없이 지나가야 한다


def test_계약이_있으면_막고_무엇을_해야_하는지_말한다(tmp_path: Path):
    _seal(tmp_path / "sealed")
    with pytest.raises(FrozenDirectoryError) as e:
        assert_not_frozen(tmp_path / "sealed", what="--out 대상", flag="--out")
    msg = str(e.value)
    assert "--out 대상" in msg
    # 권한 오류만 던지면 다음 사람이 속성을 풀고 다시 돌린다 — 그게 막으려는 사고다.
    assert "새 경로" in msg and "1-6" in msg
    assert "0" * 64 in msg or "snapshot_digest" in msg


def test_존재하지_않는_경로는_봉인이_아니다(tmp_path: Path):
    assert_not_frozen(tmp_path / "없는디렉터리")


def test_계약_요약을_읽는다(tmp_path: Path):
    d = _seal(tmp_path / "s", names=("a.jsonl", "b.json"), digest="f" * 64)
    names, digest = snapshot_summary(d)
    assert names == ["a.jsonl", "b.json"]
    assert digest == "f" * 64


def test_계약이_깨져도_가드는_죽지_않는다(tmp_path: Path):
    """읽기 실패로 가드가 예외를 내면 호출자는 가드를 빼 버린다. 요약만 비운다."""
    d = tmp_path / "s"
    d.mkdir()
    (d / CONTRACT_NAME).write_bytes(b"\xff\xfe not utf-8 \xff")
    assert snapshot_summary(d) == ([], None)
    with pytest.raises(FrozenDirectoryError):
        assert_not_frozen(d)


# ------------------------------------------------------- run_cycle_corpus 진입점

def test_사이클은_out_없이_돌지_않는다(monkeypatch):
    """기본값이 봉인본을 겨누던 결함의 구조적 수리 — 기본값 자체를 없앴다."""
    monkeypatch.setattr("sys.argv", ["run_cycle_corpus"])
    with pytest.raises(SystemExit) as e:
        R.main()
    assert e.value.code == 2           # argparse: 필수 인자 누락


def test_사이클은_봉인_디렉터리를_겨누면_멈춘다(tmp_path, monkeypatch):
    out = _seal(tmp_path / "cycle_pilot_v9")
    monkeypatch.setattr("sys.argv",
                        ["run_cycle_corpus", "--out", str(out), "--stage", "generate"])
    monkeypatch.setattr(R, "build_clause_records",
                        lambda *a, **k: (_ for _ in ()).throw(_Sentinel()))
    with pytest.raises(FrozenDirectoryError):
        R.main()
    # 막힌 뒤에도 아무것도 안 만들었어야 한다. 중간 산출물이 봉인본 위에 떨어지면
    # 막은 의미가 없다 (09-02 `_raw_generated.json` 소실이 그 경로였다).
    assert sorted(p.name for p in out.iterdir()) == [CONTRACT_NAME]


def test_사이클_가드를_빼면_지나간다(tmp_path, monkeypatch):
    """이빨 시험. 위 시험이 가드 때문에 통과하는지 확인한다."""
    out = _seal(tmp_path / "cycle_pilot_v9")
    monkeypatch.setattr("sys.argv",
                        ["run_cycle_corpus", "--out", str(out), "--stage", "generate"])
    monkeypatch.setattr(R, "assert_not_frozen", lambda *a, **k: None)
    monkeypatch.setattr(R, "build_clause_records",
                        lambda *a, **k: (_ for _ in ()).throw(_Sentinel()))
    with pytest.raises(_Sentinel):
        R.main()


# ------------------------------------------------------- make_pairs_pilot 진입점

def test_페어는_out_없이_돌지_않는다(monkeypatch):
    monkeypatch.setattr("sys.argv", ["make_pairs_pilot"])
    with pytest.raises(SystemExit) as e:
        P.main()
    assert e.value.code == 2


def test_페어는_봉인_디렉터리를_겨누면_멈춘다(tmp_path, monkeypatch):
    out = _seal(tmp_path / "pairs_pilot_v9")
    monkeypatch.setattr("sys.argv", ["make_pairs_pilot", "--out", str(out)])
    monkeypatch.setattr("data.manifest_io.load_snapshot",
                        lambda *a, **k: (_ for _ in ()).throw(_Sentinel()))
    with pytest.raises(FrozenDirectoryError):
        P.main()
    assert sorted(p.name for p in out.iterdir()) == [CONTRACT_NAME]


def test_페어_가드를_빼면_지나간다(tmp_path, monkeypatch):
    out = _seal(tmp_path / "pairs_pilot_v9")
    monkeypatch.setattr("sys.argv", ["make_pairs_pilot", "--out", str(out)])
    monkeypatch.setattr(P, "assert_not_frozen", lambda *a, **k: None)
    monkeypatch.setattr("data.manifest_io.load_snapshot",
                        lambda *a, **k: (_ for _ in ()).throw(_Sentinel()))
    with pytest.raises(_Sentinel):
        P.main()


def test_페어_가드는_이름이_아니라_계약을_본다(tmp_path, monkeypatch):
    """옛 가드는 `v1` 인지만 봤다. 그래서 봉인된 v2(=당시 기본값)를 통과시켰다.

    이름을 열거하는 가드는 자산이 느는 속도를 못 따라간다. 여기서는 v1·v2 어느 이름도
    아닌 디렉터리를 계약만 붙여 놓고 막히는지 본다.
    """
    out = _seal(tmp_path / "완전히_다른_이름")
    monkeypatch.setattr("sys.argv", ["make_pairs_pilot", "--out", str(out)])
    with pytest.raises(FrozenDirectoryError):
        P.main()


# --------------------------------------------------- snapshot_cycle_pilot 진입점

def test_스냅샷은_dir_없이_돌지_않는다(monkeypatch):
    monkeypatch.setattr("sys.argv", ["snapshot_cycle_pilot"])
    with pytest.raises(SystemExit) as e:
        S.main()
    assert e.value.code == 2


def test_스냅샷은_이미_봉인된_곳을_다시_쓰지_않는다(tmp_path, monkeypatch, capsys):
    """09-02 의 `2cedbe01… → d033c1c6… → fb316682…` 사슬이 이 경로로 생겼다."""
    out = _seal(tmp_path / "cycle_pilot_v9")
    monkeypatch.setattr("sys.argv", ["snapshot_cycle_pilot", "--dir", str(out)])
    assert S.main() == 4
    err = capsys.readouterr().err
    assert "--check" in err and "--reseal" in err
    assert sorted(p.name for p in out.iterdir()) == [CONTRACT_NAME]


def test_스냅샷_가드를_빼면_다음_단계로_간다(tmp_path, monkeypatch):
    """이빨 시험 — 4 가 아니라 2(대상 없음)로 떨어져야 가드가 세운 것이 맞다."""
    out = _seal(tmp_path / "cycle_pilot_v9")
    monkeypatch.setattr("sys.argv", ["snapshot_cycle_pilot", "--dir", str(out)])
    monkeypatch.setattr(S, "is_frozen", lambda *a, **k: False)
    assert S.main() == 2


def test_reseal_은_봉인이_없으면_거부한다(tmp_path, monkeypatch):
    d = tmp_path / "빈곳"
    d.mkdir()
    monkeypatch.setattr("sys.argv",
                        ["snapshot_cycle_pilot", "--dir", str(d), "--reseal"])
    assert S.main() == 4


def test_reseal_은_비대화식에서_거부한다(tmp_path, monkeypatch, capsys):
    """확인을 받을 수 없는 자리에서 통과시키면 그 플래그는 절차가 아니라 스위치다."""
    out = _seal(tmp_path / "cycle_pilot_v9")
    monkeypatch.setattr("sys.argv",
                        ["snapshot_cycle_pilot", "--dir", str(out), "--reseal"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit) as e:
        S.main()
    assert "확인" in str(e.value)
    # 무엇을 지우는지 화면에 찍었어야 한다 (총괄 지시 과제 1).
    out_txt = capsys.readouterr().out
    assert "snapshot_digest" in out_txt and "pairs.jsonl" in out_txt


def test_check_는_봉인된_곳에서도_돈다(tmp_path, monkeypatch):
    """대조는 아무것도 쓰지 않는다. 막으면 봉인본을 검증할 방법이 사라진다."""
    out = _seal(tmp_path / "cycle_pilot_v9")
    monkeypatch.setattr("sys.argv",
                        ["snapshot_cycle_pilot", "--dir", str(out), "--check"])
    # 대상 파일이 없으니 2 로 떨어진다 — 중요한 건 4(봉인 거부)가 아니라는 것이다.
    assert S.main() == 2


# ------------------------------------------------------- compare_judges 진입점

def test_대조기는_dir_없이_돌지_않는다(monkeypatch):
    """인자가 아예 없어서 항상 봉인된 cycle_pilot 에 썼다. 그 자리가 기본값이었다."""
    monkeypatch.setattr("sys.argv", ["compare_judges"])
    with pytest.raises(SystemExit) as e:
        C.main()
    assert e.value.code == 2


def test_대조기는_봉인된_곳에_쓰지_않는다(tmp_path, monkeypatch):
    """`judge_agreement.json` 은 v1 SNAPSHOT 구성원이다 — 덮어쓰면 봉인이 깨진다."""
    out = _seal(tmp_path / "cycle_pilot_v9", names=("judge_agreement.json",))
    monkeypatch.setattr("sys.argv", ["compare_judges", "--dir", str(out)])
    with pytest.raises(FrozenDirectoryError):
        C.main()
    assert sorted(p.name for p in out.iterdir()) == [CONTRACT_NAME]


def test_대조기는_out_을_따로_주면_그리로_쓴다(tmp_path, monkeypatch):
    """봉인본을 **읽는** 것은 막지 않는다. 막으면 봉인된 corpus 를 분석할 수 없다."""
    src = _seal(tmp_path / "cycle_pilot_v9")
    dst = tmp_path / "새자리"
    monkeypatch.setattr("sys.argv",
                        ["compare_judges", "--dir", str(src), "--out", str(dst)])
    monkeypatch.setattr(C, "verdicts",
                        lambda *a, **k: (_ for _ in ()).throw(_Sentinel()))
    with pytest.raises(_Sentinel):        # 가드를 지나 본체까지 갔다
        C.main()


# ------------------------------------------------------------------- 실물 봉인본

FROZEN_REAL = (
    REPO / "corpus/generate/cycle_pilot",
    REPO / "corpus/generate/cycle_pilot_v2",
    REPO / "data/processed/pairs_pilot_v1",
    REPO / "data/processed/pairs_pilot_v2",
)


@pytest.mark.parametrize("d", FROZEN_REAL, ids=lambda p: p.name)
def test_실물_봉인본이_아직_잠겨_있다(d: Path):
    if not d.is_dir():
        pytest.skip("워크트리에 없다")
    assert is_frozen(d), f"{CONTRACT_NAME} 이 없다 — 봉인이 풀렸다"
    with pytest.raises(FrozenDirectoryError):
        assert_not_frozen(d)


@pytest.mark.parametrize("mod", [R, P], ids=["run_cycle_corpus", "make_pairs_pilot"])
def test_옛_기본값_경로는_지금이라면_전부_거부된다(mod):
    """보고된 결함 그 자체에 대한 회귀 시험.

    `LEGACY_OUT_DIRS` 는 한때 `--out` 기본값이었던 자리다. 그 경로들이 지금 실행되면
    반드시 막혀야 한다 — 막히지 않는다면 기본값을 되돌려도 안전하다는 뜻이 되고,
    그건 사실이 아니다.
    """
    for d in mod.LEGACY_OUT_DIRS:
        if not d.is_dir():
            continue
        with pytest.raises(FrozenDirectoryError):
            assert_not_frozen(d)


def test_봉인_구성원이_실물과_이름이_맞는다():
    """계약 파일이 유령 목록이 되지 않았는지. 이름이 어긋나면 대조가 불가능하다."""
    for d in FROZEN_REAL:
        if not d.is_dir():
            continue
        names, digest = snapshot_summary(d)
        assert names, f"{d} 의 계약 파일이 비었다"
        assert digest and len(digest) == 64, f"{d} 에 snapshot_digest 가 없다"
        missing = [n for n in names if not (d / n).is_file()]
        assert not missing, f"{d}: 계약에 있는데 실물이 없다 — {missing}"


def test_corpus_는_계약_이름을_다시_박지_않는다():
    """이름을 두 벌로 두면 한쪽만 바뀐다. 정본은 `data.frozen_guard` 하나다 (P6)."""
    from data.frozen_guard import CONTRACT_NAME as CANON

    assert CONTRACT_NAME is CANON
    assert S.SNAPSHOT is CANON
    for mod in ("run_cycle_corpus.py", "make_pairs_pilot.py", "frozen_out.py",
                "snapshot_cycle_pilot.py"):
        src = (REPO / "corpus/generate" / mod).read_text(encoding="utf-8")
        assert '"SNAPSHOT.sha256"' not in src, f"{mod} 이 계약 이름을 다시 박았다"


def test_봉인_대상을_쓰는_진입점이_전부_가드를_부른다():
    """새 진입점이 늘 때 가드를 빼먹는 것을 막는다 (80번 G1-2 배선 시험)."""
    import ast

    for mod in ("run_cycle_corpus.py", "make_pairs_pilot.py"):
        src = (REPO / "corpus/generate" / mod).read_text(encoding="utf-8")
        tree = ast.parse(src)
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "assert_not_frozen" in called, f"{mod} 이 가드를 부르지 않는다"
    src = (REPO / "corpus/validate/compare_judges.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "assert_not_frozen" in called, "compare_judges 가 가드를 부르지 않는다"
    src = (REPO / "corpus/generate/snapshot_cycle_pilot.py").read_text(encoding="utf-8")
    assert "is_frozen(out_dir)" in src, "snapshot_cycle_pilot 이 봉인을 확인하지 않는다"


def test_사이클_보고서가_봉인본을_가리키면_계약이_그것을_담고_있다():
    """봉인본 안의 보고서가 실제로 계약 구성원인지. 목록이 실물과 갈리면 대조가 깨진다."""
    for d in (REPO / "corpus/generate/cycle_pilot",
              REPO / "corpus/generate/cycle_pilot_v2"):
        if not d.is_dir():
            continue
        names, _ = snapshot_summary(d)
        assert "cycle_corpus_report.json" in names, f"{d}: 보고서가 계약에 없다"
        json.loads((d / "cycle_corpus_report.json").read_text(encoding="utf-8"))
