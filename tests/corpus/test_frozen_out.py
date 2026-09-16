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

import io
import json
import warnings
from pathlib import Path

import pytest

from corpus.generate import frozen_out as FO
from corpus.generate import make_pairs_pilot as P
from corpus.generate import run_cycle_corpus as R
from corpus.generate import snapshot_cycle_pilot as S
from corpus.generate.frozen_out import (
    CONTRACT_NAME,
    FrozenDirectoryError,
    assert_not_frozen,
    is_frozen,
    snapshot_summary,
    verify_contract,
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


class _Tty(io.StringIO):
    """대화식 stdin 흉내. `isatty` 를 True 로 답한다."""

    def isatty(self) -> bool:
        return True


def test_확인은_이름을_정확히_입력해야_통과한다(tmp_path, monkeypatch):
    """대화식이어도 아무 키나 눌러서는 안 통과한다 — 그러면 확인이 아니라 관성이다.

    반대쪽도 건다. **통과할 수 없는 확인은 확인이 아니라 차단이다** — 사람이 절차를
    밟았는데도 막히면 다음 사람은 절차 대신 가드를 지운다.
    """
    from corpus.generate.frozen_out import confirm_overwrite

    out = _seal(tmp_path / "cycle_pilot_v9")

    monkeypatch.setattr("sys.stdin", _Tty("y\n"))
    with pytest.raises(SystemExit) as e:
        confirm_overwrite(out, action="다시 쓴다", flag="--reseal")
    assert "중단" in str(e.value)

    monkeypatch.setattr("sys.stdin", _Tty("cycle_pilot_v9\n"))
    confirm_overwrite(out, action="다시 쓴다", flag="--reseal")   # 예외 없이 통과


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
    entry = FO.EXPECTED_SEALED.get(d.relative_to(REPO).as_posix(), {})
    if not d.is_dir():
        if entry.get("status") == "lost":
            pytest.skip(f"{d.name}: 소실 기록 — {entry.get('record')}")
        pytest.skip(f"{d.name}: 이 트리에 없다 — 있어야 하는지는 계약 대조 시험이 판정한다")
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


class IncompleteTreeWarning(UserWarning):
    """이 트리에 원본이 없어 계약 대조를 다 못 했다. 통과했다는 뜻이 아니다."""


@pytest.mark.parametrize("d", FROZEN_REAL, ids=lambda p: p.name)
def test_봉인_구성원이_실물과_이름이_맞는다(d: Path):
    """계약 파일이 유령 목록이 되지 않았는지. 이름이 어긋나면 대조가 불가능하다.

    **트리마다 답이 다르다.** 계약 구성원 일부가 `.gitignore` 로 추적 밖이라
    (`corpus/generate/cycle_pilot/*.jsonl`) 만든 워크트리에만 있고 main 체크아웃에는
    없다. 단순 존재 검사로 두면 만든 자리에서만 통과하고 게이트에서 깨진다 — 09-08 에
    실제로 그랬다.

    그래서 **git 이 그 이름을 추적하는가**로 가른다. 추적분이 없으면 훼손이라 실패하고,
    미추적분이 없으면 이 트리에 원본이 없을 뿐이라 건너뛴다. 건너뛸 때는 **경고와
    skip 사유에 이름을 남긴다** — 조용히 통과하면 다음 사람은 다 본 줄 안다.
    """
    rel = d.relative_to(REPO).as_posix()
    entry = FO.EXPECTED_SEALED.get(rel)
    assert entry is not None, (f"{rel}: 봉인처 명부에 없다 — data/frozen_guard.py 의 "
                               "EXPECTED_SEALED(A 소관)에 올리도록 총괄에 보고하라")
    # 옛 판정은 디렉터리가 없으면 그냥 건너뛰었다 — 09-11 소실이 그렇게 초록으로 지나갔다.
    r = verify_contract(d, expectation=entry)

    if r["verdict"] in ("lost_recorded", "absent_tree"):
        msg = f"{d.name}: [{r['verdict']}] {r.get('reason')}"
        warnings.warn(msg, IncompleteTreeWarning, stacklevel=2)
        pytest.skip(msg)

    assert r["verdict"] not in FO.FAILING, (
        f"{d}: [{r['verdict']}] {r.get('reason')} — {r.get('missing_tracked')}")

    if r["unverified"]:
        msg = (f"{d.name}: 계약 구성원 {r['n_members']}개 중 "
               f"{len(r['unverified'])}개를 이 트리에서 대조하지 못했다 "
               f"({r['verdict']}) — {r['unverified']}. "
               "추적 밖 자산이라 만든 워크트리에만 있다.")
        warnings.warn(msg, IncompleteTreeWarning, stacklevel=2)
        pytest.skip(msg)

    # 계약과 실물이 둘 다 있는 트리에서는 엄격히 본다.
    assert r["verdict"] == "ok", r
    assert r["digest"] and len(r["digest"]) == 64, f"{d} 에 snapshot_digest 가 없다"
    assert r["n_members"], f"{d} 의 계약 파일이 비었다"


def test_대조_판정은_훼손과_트리_차이를_가른다(tmp_path):
    """이빨 시험 — 위 시험이 **넘어가면 안 될 것**까지 넘기지 않는지.

    같은 "실물 없음"이라도 추적분이 사라진 것은 훼손이고, 미추적분이 없는 것은 트리
    차이다. 둘을 같게 다루면 한쪽은 매번 깨지고 다른 한쪽은 영영 안 걸린다.
    """
    d = _seal(tmp_path / "s", names=("tracked.json", "ignored.jsonl"))
    tracked = frozenset({CONTRACT_NAME, "tracked.json"})

    # 둘 다 있다 → 엄격 통과
    (d / "tracked.json").write_text("{}", encoding="utf-8")
    (d / "ignored.jsonl").write_text("{}\n", encoding="utf-8")
    assert verify_contract(d, tracked=tracked)["verdict"] == "ok"

    # 미추적분만 없다 → 트리 차이. 이름이 판정에 남아야 한다
    (d / "ignored.jsonl").unlink()
    r = verify_contract(d, tracked=tracked)
    assert r["verdict"] == "incomplete_tree"
    assert r["unverified"] == ["ignored.jsonl"] and not r["missing_tracked"]

    # 추적분이 없다 → 훼손. 미추적분 상태와 무관하게 broken 이다
    (d / "tracked.json").unlink()
    r = verify_contract(d, tracked=tracked)
    assert r["verdict"] == "broken"
    assert r["missing_tracked"] == ["tracked.json"]


def test_추적_여부를_모르면_괜찮다고_하지_않는다(tmp_path):
    """git 을 못 물었을 때 통과로 떨어지면 가드가 아니라 장식이다."""
    d = _seal(tmp_path / "s", names=("a.jsonl",))
    monkey = verify_contract(d, tracked=None)   # tmp 는 저장소 밖 → git 판단 불가
    assert monkey["verdict"] in ("unverifiable", "broken", "incomplete_tree")
    assert monkey["unverified"] == ["a.jsonl"] or monkey["missing_tracked"] == ["a.jsonl"]


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


# ---------------------------------------------------------- 봉인처 명부 (30번 §8-4 나)
#
# 계약서가 있는 디렉터리만 찾으면, 디렉터리째 사라진 봉인본은 목록에서 같이 사라진다.
# 09-11 에 data/processed 가 비었을 때 frozen_out 이 "깨짐 0" 을 낸 이유다.


def _registry_tree(tmp_path: Path, *, with_root: bool) -> Path:
    """가짜 저장소 뿌리. `with_root` 면 저장 뿌리(data/processed)만 있고 봉인본은 없다."""
    root = tmp_path / "repo"
    (root / "corpus/generate").mkdir(parents=True)
    if with_root:
        (root / "data/processed").mkdir(parents=True)
    return root


def test_명부가_실물_봉인처를_전부_덮는다():
    """명부에 없는 봉인처는 사라져도 아무도 모른다."""
    for d in FROZEN_REAL:
        assert d.relative_to(REPO).as_posix() in FO.EXPECTED_SEALED, d


def test_있어야_할_봉인처가_통째로_없으면_실패한다(tmp_path, monkeypatch, capsys):
    """09-11 사고 그대로 — 저장 뿌리는 있는데 봉인 디렉터리가 계약서째 없다."""
    root = _registry_tree(tmp_path, with_root=True)
    monkeypatch.setattr(FO, "_REPO", root)
    monkeypatch.setattr(FO, "EXPECTED_SEALED",
                        {"data/processed/pairs_x": {"status": "expected", "owner": "B"}})
    assert FO.main([]) == 1
    out = capsys.readouterr().out
    assert "missing_contract" in out and "pairs_x" in out


def test_명부가_없으면_옛_동작처럼_조용히_지나간다(tmp_path, monkeypatch, capsys):
    """이빨 시험 — 위 실패가 **명부 때문에** 나는지. 명부를 비우면 같은 트리가 초록이다."""
    root = _registry_tree(tmp_path, with_root=True)
    monkeypatch.setattr(FO, "_REPO", root)
    monkeypatch.setattr(FO, "EXPECTED_SEALED", {})
    assert FO.main([]) == 0
    assert "깨짐 0" in capsys.readouterr().out


def test_저장_뿌리가_없는_트리는_실패가_아니라_알린다(tmp_path, monkeypatch, capsys):
    """정션을 안 붙인 새 clone — 여기서는 볼 수 없을 뿐 소실이라고 단정할 수 없다."""
    root = _registry_tree(tmp_path, with_root=False)
    monkeypatch.setattr(FO, "_REPO", root)
    monkeypatch.setattr(FO, "EXPECTED_SEALED",
                        {"data/processed/pairs_x": {"status": "expected", "owner": "B"}})
    assert FO.main([]) == 0
    out = capsys.readouterr().out
    assert "absent_tree" in out and "이 트리에 없음 1" in out


def test_추적된_계약서가_없으면_뿌리가_없어도_실패한다(tmp_path):
    """git 이 계약서를 들고 있는데 파일이 없다면 트리 차이가 아니라 훼손이다."""
    d = tmp_path / "없는뿌리" / "sealed"
    r = verify_contract(d, tracked=frozenset({CONTRACT_NAME}),
                        expectation={"status": "expected"})
    assert r["verdict"] == "missing_contract"


def test_소실_기록은_알리되_실패시키지_않는다(tmp_path, monkeypatch, capsys):
    """기록된 소실까지 매번 실패로 만들면 사람들이 이 검사를 끈다. 대신 매번 말한다."""
    root = _registry_tree(tmp_path, with_root=True)
    monkeypatch.setattr(FO, "_REPO", root)
    monkeypatch.setattr(FO, "EXPECTED_SEALED", {
        "data/processed/pairs_gone": {"status": "lost", "owner": "B",
                                      "record": "소실 확정 기록 X"}})
    assert FO.main([]) == 0
    out = capsys.readouterr().out
    assert "lost_recorded" in out and "소실 확정 기록 X" in out and "소실 기록 1" in out


def test_소실_기록된_이름이_다시_나타나면_실패한다(tmp_path, monkeypatch):
    """같은 이름으로 되살아났다면 명부가 낡았거나 재생성이다 — 후자는 규약 1-6 위반."""
    root = _registry_tree(tmp_path, with_root=True)
    _seal(root / "data/processed/pairs_gone")
    monkeypatch.setattr(FO, "_REPO", root)
    monkeypatch.setattr(FO, "EXPECTED_SEALED", {
        "data/processed/pairs_gone": {"status": "lost", "owner": "B", "record": "기록"}})
    assert [x["verdict"] for x in FO.check_registry()] == ["lost_but_present"]
    assert FO.main([]) == 1


def test_실제_트리에서_소실이_조용히_사라지지_않는다(capsys):
    """회귀 — pairs_pilot_v2 는 없어졌다. 출력에 그 사실이 **매번** 나와야 한다."""
    FO.main([])
    out = capsys.readouterr().out
    assert "pairs_pilot_v2" in out and "lost_recorded" in out


# ------------------------------------------ 명부 단일화 (09-16, A 62f660b 뒤의 B 전환)


def test_명부는_data_frozen_guard_의_단일_명부다():
    """두 곳에 살면 한쪽만 고쳐진다. 이 모듈은 A 의 명부를 **같은 객체로** 내보내기만 한다."""
    import ast

    from corpus.validate import verify_backup as VB
    from data import frozen_guard as FG

    assert FO.EXPECTED_SEALED is FG.EXPECTED_SEALED
    assert VB.EXPECTED_SEALED is FG.EXPECTED_SEALED
    for mod in (FO, VB):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for node in tree.body:
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, ast.AnnAssign) else [])
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            assert "EXPECTED_SEALED" not in names, f"{mod.__name__} 가 명부를 다시 정의한다"


def test_상태_어휘를_빠짐없이_판정한다():
    """A 가 어휘에 단어를 더하면 여기서 멈춘다 — 그 단어가 "있어야 하는지" 를 B 가 정해야
    대조기와 백업 목록이 함께 선다. 모르는 단어를 조용히 expected 로 읽지 않는다."""
    from data.frozen_guard import SEALED_STATUSES

    assert FO.PRESENT_STATUSES | FO.ABSENT_STATUSES == set(SEALED_STATUSES)
    assert not (FO.PRESENT_STATUSES & FO.ABSENT_STATUSES)


def _real_seal(d: Path) -> Path:
    """해시가 맞는 작은 봉인본 — 대조기가 `ok` 를 낼 수 있게."""
    import hashlib

    d.mkdir(parents=True, exist_ok=True)
    (d / "pairs.jsonl").write_bytes(b'{"a": 1}\n')
    h = hashlib.sha256((d / "pairs.jsonl").read_bytes()).hexdigest()
    line = f"{h}  pairs.jsonl"
    digest = hashlib.sha256((line + "\n").encode()).hexdigest()
    (d / CONTRACT_NAME).write_text(f"{line}\n# snapshot_digest {digest}\n", encoding="utf-8")
    return d


def test_restored_는_expected_와_같게_판정하고_근거를_옮긴다(tmp_path, monkeypatch, capsys):
    """복원 자리가 비면 expected 와 똑같이 실패, 차 있으면 통과 — 근거 등급은 결과·출력에 남는다."""
    root = _registry_tree(tmp_path, with_root=True)
    reg = {"data/processed/back": {"status": "restored", "owner": "A",
                                   "record": "소실 뒤 복원", "evidence": "64자 digest 전체 일치"}}
    monkeypatch.setattr(FO, "_REPO", root)
    monkeypatch.setattr(FO, "EXPECTED_SEALED", reg)

    [r] = FO.check_registry()
    assert r["verdict"] == "missing_contract" and r["verdict"] in FO.FAILING
    assert FO.main([]) == 1
    capsys.readouterr()

    _real_seal(root / "data/processed/back")
    [r] = FO.check_registry()
    assert r["verdict"] == "ok", r
    assert r["expected_status"] == "restored" and r["evidence"] == "64자 digest 전체 일치"
    assert FO.main([]) == 0
    assert "복원 근거: 64자 digest 전체 일치" in capsys.readouterr().out


def test_실물_트리에서_명부_전체_대조가_실패_0이다():
    """명부 **전체**를 실물에 대조한다 — A 소유 자리(`manifest_v1`·복원된 pilot3000 계열)까지.

    `FROZEN_REAL` 은 B 네 곳만 채점하고, A 의 `test_restored_자리에_실물과_계약서가_있다` 는
    세 자리 중 **하나라도 없으면 skip** 한다. 둘을 합쳐도 복원 자리 하나가 사라지면 스위트는
    초록이다 — 09-11 과 같은 모양이다(45번 I-1). 여기서는 저장 뿌리가 있는데 봉인본이 없으면
    `missing_contract` 로 실패한다. 뿌리 자체가 없는 트리(정션 없는 clone)는 `absent_tree` 라
    실패가 아니고, 그 사실은 경고로 남긴다.
    """
    results = FO.check_registry()
    bad = [(r["rel"], r["verdict"], r.get("reason")) for r in results if r["verdict"] in FO.FAILING]
    assert not bad, bad
    absent = [r["rel"] for r in results if r["verdict"] == "absent_tree"]
    if absent:
        warnings.warn(f"이 트리에서 볼 수 없는 봉인처 {len(absent)}곳: {absent}",
                      IncompleteTreeWarning, stacklevel=2)


def test_어휘_밖_상태는_거부한다(tmp_path):
    """명부의 오타("expectd")를 expected 로 읽어 넘기면 판정이 조용히 바뀐다."""
    with pytest.raises(ValueError, match="어휘"):
        verify_contract(tmp_path / "x", expectation={"status": "expectd", "owner": "B"})
