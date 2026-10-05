"""공개 검수기·백업 대조기 오류 검출 시험. 09-08 판정 (24번 A-4 ①②).

둘 다 **통과가 곧 승인**이 되는 도구다. 공개 검수기가 놓치면 AI허브 파생물이나 규정
원문이 공개 저장소로 나가고, 백업 대조기가 놓치면 깨진 사본을 정본으로 믿는다. 그래서
"잡는가" 만이 아니라 **"안 잡으면 어떻게 되는가"** 를 같이 건다.

실물 봉인본은 읽기만 한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from corpus.generate.frozen_out import CONTRACT_NAME
from corpus.validate import screen_public_release as S
from corpus.validate import verify_backup as B

REPO = Path(__file__).resolve().parents[2]


def _seal(d: Path, files: dict[str, str]) -> Path:
    """실물 + 계약서를 갖춘 가짜 봉인 디렉터리."""
    import hashlib

    d.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, body in files.items():
        p = d / name
        p.write_text(body, encoding="utf-8", newline="")
        entries.append((hashlib.sha256(p.read_bytes()).hexdigest(), name))
    digest = hashlib.sha256("".join(h for h, _ in entries).encode()).hexdigest()
    (d / CONTRACT_NAME).write_text(
        "\n".join(f"{h}  {n}" for h, n in entries) + f"\n# snapshot_digest {digest}\n",
        encoding="utf-8", newline="")
    return d


# ------------------------------------------------------------- 공개 검수기

def test_유료표준_식별자를_잡는다():
    hits = S.screen_text("이 판정은 ISO 5817 C 등급을 따른다", set())
    assert [h[0] for h in hits] == ["paid_std"]


def test_aihub_식별자를_잡는다():
    """AI허브 파생 식별자가 공개 저장소로 나가면 레드라인 1 위반이다.

    취득 id 는 **합성 값**이다(실물 레코드가 아니다) — 규칙은 접두 형식만 보므로 시험의 뜻은 같다.
    실물 식별자를 픽스처로 박으면 탐지기를 시험하려다 레드라인 1 을 스스로 어긴다(09-18 추기).
    """
    hits = S.screen_text('{"image_id": "aihub00000:00000000"}', set())
    assert "aihub_id" in [h[0] for h in hits]


def test_로컬_절대경로를_잡는다():
    """입력은 **존재하지 않는 가짜 경로**다 — 드라이브(Q:)도 폴더 이름도 이 저장소·사용자와 무관하다.

    실제 위치 형태를 픽스처로 박으면 탐지기를 시험하려다 규약 2-6 을 스스로 어긴다(46번 §4).
    옛 규칙(첫 폴더 이름 목록)에서도 잡히던 `Users` 형태다. 규칙을 넓힌 뒤 새로 잡는 형태는
    아래 시험이 따로 건다(50번 §6).
    """
    hits = S.screen_text(r"Q:\Users\example_user\sample_project\corpus 에서 읽었다", set())
    assert "local_path" in [h[0] for h in hits]


# 아래 입력도 전부 존재하지 않는 가짜 경로다(Q: 드라이브·example 이름).
@pytest.mark.parametrize("text", [
    # 문자열 안에 JSON 이 한 번 더 들어 있으면 json.loads 뒤에도 역슬래시가 2개로 남는다
    r"raw 출력의 경로 Q:\\sample_project\\notes.md",
    # 위 입력은 UNC 갈래(\\sample_project\\notes.md)로도 읽혀서, 드라이브 갈래의 구분자 {1,2} 를
    # {1} 로 줄여도 통과한다(51번 m-3). 뒤에 구분자가 더 없어 UNC 로 못 읽는 입력을 따로 둔다.
    r"설정값 Q:\\sample_project 를 읽었다",
    r"출처는 Q:\sample_project\notes.md 이다",           # 첫 폴더가 옛 목록(Users 등) 밖
    r"적재 위치 Q:\공유 드라이브\sample 확인",              # 공유 드라이브 이름
    r"원본은 \\example-host\sample_share\x 에 있다",       # UNC
], ids=["JSON_이스케이프", "JSON_이스케이프_드라이브만", "목록_밖_첫_폴더", "공유_드라이브", "UNC"])
def test_폴더_이름_목록과_무관하게_로컬_경로를_잡는다(text):
    """옛 규칙은 첫 폴더가 `Users`·`Program Files`·저장소 이름일 때만 잡았다 (추기)."""
    assert "local_path" in [h[0] for h in S.screen_text(text, set())]


@pytest.mark.parametrize("text", [
    "자료는 https://example.com/docs/page 에 있다",
    "회의는 12:30 에 시작했다 (2026-09-16T12:30:00)",
    "ISO 5817:2014 의 등급 표기",
    r"documents:\n  - doc_id: X",                          # YAML 문자열의 줄바꿈 이스케이프
    r"패턴은 \\d+\\s* 이다",                                # 이스케이프된 정규식 조각
], ids=["URL", "시각", "규격_표기", "YAML_줄바꿈", "정규식_조각"])
def test_경로가_아닌_콜론_표기는_잡지_않는다(text):
    """넓힌 만큼 오탐이 늘면 적발 목록이 소음이 된다. 다른 사유(유료 표준 등)는 여기서 보지 않는다."""
    assert "local_path" not in [h[0] for h in S.screen_text(text, set())]


def test_경로_규칙은_폴더_이름_목록을_들고_있지_않다():
    """목록에 기대면 목록 밖 경로가 빠진다 — 드라이브 뒤에 폴더 이름 대안 묶음이 없어야 한다.

    옛 규칙은 드라이브 구분자 바로 뒤에 `(?:Users|…)` 대안을 두었다(저장소 이름 포함).
    POSIX 갈래의 `(?:home|Users)` 는 지시대로 남긴 것이라 여기서 보지 않는다.
    """
    p = S.LOCAL_PATH.pattern
    assert "(?:Users|" not in p
    assert "Program Files" not in p
    assert re.search(r"\](?:\{1,2\})?\(\?:", p) is None, "드라이브 구분자 뒤에 대안 묶음이 있다"


def test_원문_연속_일치를_잡는다():
    """규정 원문을 길게 그대로 실으면 재배포다 (.gitignore:54)."""
    src = "가" * 30 + "이 조항은 용접부의 표면 결함을 다루며 검사원이 확인한다" + "나" * 30
    sh = {src[i:i + S.SHINGLE] for i in range(len(src) - S.SHINGLE + 1)}
    assert S.screen_text(src[20:20 + S.SHINGLE + 10], sh)[0][0] == "verbatim_source"


def test_짧은_겹침은_잡지_않는다():
    """한계 미만까지 잡으면 흔한 표현마다 걸려 검수가 무의미해진다."""
    src = "용접부의 표면 결함을 다룬다 " * 20
    sh = {src[i:i + S.SHINGLE] for i in range(len(src) - S.SHINGLE + 1)}
    assert S.screen_text("용접부의 표면", sh) == []


def test_원천이_없으면_통과로_적지_않는다(monkeypatch, tmp_path):
    """원천 문서가 없으면 재배포 판정을 **못 한** 것이다. 조용히 clear 로 떨어지면 안 된다."""
    monkeypatch.setattr(S, "SOURCE_DOCS", (tmp_path / "없음.md",))
    monkeypatch.setattr(S, "TARGET_DIRS", ())
    r = S.screen()
    assert r["n_source_shingles"] == 0
    assert "warning" in r and "근거가 아니다" in r["warning"]


def test_검수는_적발되면_차단으로_끝난다(monkeypatch, tmp_path):
    """이빨 시험 — 적발이 종료코드에 반영되지 않으면 게이트가 아니다."""
    d = _seal(tmp_path / "cycle_x", {"a.jsonl": json.dumps(
        {"text": "ISO 5817 을 인용한다"}, ensure_ascii=False) + "\n"})
    monkeypatch.setattr(S, "TARGET_DIRS", (d,))
    monkeypatch.setattr(S, "untracked_members", lambda _d: ["a.jsonl"])
    _fake_source(tmp_path, monkeypatch)   # 원천도 픽스처로 — 미추적 실물에 기대지 않는다
    assert S.main([]) == 1

    # 깨끗하면 0 이어야 한다. 늘 1이면 그것도 게이트가 아니다.
    (d / "a.jsonl").write_text(
        json.dumps({"text": "안전한 문장"}, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="")
    assert S.main([]) == 0


# ------------------------------------- 검사 결과 구분 (66번, 09-18 승인)
#
# 옛 판은 대상 디렉터리가 없으면 조용히 건너뛰어, 한 글자도 안 본 실행이 적발 0·clear 로 나왔다.
# `clear` 는 실제로 검사한 문자열이 있을 때만이고, 못 본 경우는 `unverified`(종료 2)다.


def _fake_source(tmp_path: Path, monkeypatch) -> None:
    """원천 문서도 픽스처로 둔다.

    실물 원천(`corpus/parse/survey/**`)은 **미추적**이라 새 clone·스크래치 사본에는 없다. 그 트리에서
    원천 부재는 이제 `unverified` 이므로, 픽스처로 고정하지 않으면 시험 결과가 트리에 따라 갈린다.
    """
    src = tmp_path / "원천_픽스처.md"
    src.write_text("용접부의 표면 결함을 다룬다 " * 20, encoding="utf-8")
    monkeypatch.setattr(S, "SOURCE_DOCS", (src,))


def _one_target(tmp_path: Path, monkeypatch, body: str = "안전한 문장",
                names=("a.jsonl",)) -> Path:
    """봉인 계약서 + 구성원을 갖춘 가짜 대상 하나. 실물 봉인 자산은 건드리지 않는다."""
    files = {n: json.dumps({"text": body}, ensure_ascii=False) + "\n" for n in names}
    d = _seal(tmp_path / "cycle_x", files)
    monkeypatch.setattr(S, "TARGET_DIRS", (d,))
    monkeypatch.setattr(S, "untracked_members", lambda _d: list(names))
    _fake_source(tmp_path, monkeypatch)
    return d


def test_대상이_없으면_통과가_아니라_미검사다(tmp_path, monkeypatch, capsys):
    """09-20 회신 §20-5 의 경로 — 검사한 문자열이 0개인데 clear 가 나왔다."""
    monkeypatch.setattr(S, "TARGET_DIRS", (tmp_path / "없는대상",))
    r = S.screen()
    assert r["verdict"] == "unverified" and r["n_texts_examined"] == 0
    assert "no_target_dir" in [u["reason"] for u in r["unverified_reasons"]]
    assert r["n_targets_missing"] == 1
    assert S.main([]) == 2
    err = capsys.readouterr().err
    assert "위험이 없다는 뜻이 아니다" in err


def test_구성원이_하나라도_없으면_미검사다(tmp_path, monkeypatch):
    d = _one_target(tmp_path, monkeypatch, names=("a.jsonl", "b.jsonl"))
    (d / "b.jsonl").unlink()
    r = S.screen()
    assert r["verdict"] == "unverified"
    assert [t["missing_members"] for t in r["targets"]] == [["b.jsonl"]]
    assert r["n_texts_examined"] == 1            # 남은 한 파일은 실제로 봤다
    assert S.main([]) == 2


def test_원천_문서가_일부라도_없으면_미검사다(tmp_path, monkeypatch):
    """재배포 판정을 못 한 것이다. 적발이 있으면 차단이 먼저."""
    _one_target(tmp_path, monkeypatch)
    src = tmp_path / "원천.md"
    src.write_text("용접부의 표면 결함을 다룬다 " * 20, encoding="utf-8")
    monkeypatch.setattr(S, "SOURCE_DOCS", (src, tmp_path / "없는원천.md"))
    r = S.screen()
    assert r["verdict"] == "unverified" and r["n_texts_examined"] == 1
    assert "no_source_docs" in [u["reason"] for u in r["unverified_reasons"]]
    assert S.main([]) == 2

    # 적발과 미검사 사유가 **함께** 성립하면 차단이 이긴다(종료 1). `_one_target` 이 원천 픽스처를
    # 정상으로 되돌리므로 부재 상태를 다시 건다 — 안 걸면 이 시점에 미검사 사유가 없어서
    # 우선순위를 뒤집어도 시험이 통과한다(67번 Minor 1).
    _one_target(tmp_path, monkeypatch, body="ISO 5817 을 인용한다")
    monkeypatch.setattr(S, "SOURCE_DOCS", (src, tmp_path / "없는원천.md"))
    both = S.screen()
    assert "no_source_docs" in [u["reason"] for u in both["unverified_reasons"]]
    assert both["verdict"] == "blocked" and both["n_blocking"] == 1
    assert S.main([]) == 1


# ------------------------------ 원천별 내용 검사 (검토 §23-3, 09-21 지시)
#
# 원천 파일이 **있어도** 비교 조각을 하나도 못 만들면 그 원천으로는 재배포 판정을 못 한 것이다.
# 합집합의 조각 수만 보면 다른 원천의 조각에 가려 통과로 읽힌다.

_NORMAL_SOURCE = "용접부의 표면 결함을 다룬다 " * 20


def _two_sources(tmp_path: Path, monkeypatch, first: str | None, second: str | None):
    """원천 둘을 픽스처로 둔다. `None` 은 파일 없음, 문자열은 그 내용의 파일이다."""
    paths = []
    for name, body in (("원천_하나.md", first), ("원천_둘.md", second)):
        p = tmp_path / name
        if body is not None:
            p.write_text(body, encoding="utf-8")
        paths.append(p)
    monkeypatch.setattr(S, "SOURCE_DOCS", tuple(paths))
    return paths


@pytest.mark.parametrize("first, second, verdict, code, n_bad", [
    ("", _NORMAL_SOURCE, "unverified", 2, 1),                       # 빈 파일
    (" \t\r\n  \n", _NORMAL_SOURCE, "unverified", 2, 1),            # 공백·개행뿐
    ("가" * (S.SHINGLE - 1), _NORMAL_SOURCE, "unverified", 2, 1),   # 정규화 뒤 39자
    ("가" * S.SHINGLE, _NORMAL_SOURCE, "clear", 0, 0),              # 40자 — 조각 1개, 정상
    (_NORMAL_SOURCE, "", "unverified", 2, 1),                       # 둘 중 하나만(뒤쪽) 빈 경우
    ("", "", "unverified", 2, 2),                                   # 둘 다 빈 경우
    ("가" + " " * (S.SHINGLE * 2) + "나", _NORMAL_SOURCE, "unverified", 2, 1),  # 공백으로 부풀린 3자
], ids=["빈_파일", "공백뿐", "39자", "40자", "뒤쪽만_빔", "둘_다_빔", "공백으로_부풀림"])
def test_원천이_비교_조각을_못_만들면_미검사다(tmp_path, monkeypatch, capsys,
                                first, second, verdict, code, n_bad):
    _one_target(tmp_path, monkeypatch)
    paths = _two_sources(tmp_path, monkeypatch, first, second)
    r = S.screen()
    assert len(r["source_docs_present"]) == 2           # 파일은 둘 다 **있다**
    assert r["n_files_examined"] == 1 and r["n_texts_examined"] == 1
    bad = [u["what"] for u in r["unverified_reasons"] if u["reason"] == "source_without_shingles"]
    assert len(bad) == n_bad and r["verdict"] == verdict
    assert [s["n_shingles"] == 0 for s in r["sources"]] == [
        len(S.normalize(body)) < S.SHINGLE for body in (first, second)]
    if n_bad == 2:
        assert r["n_source_shingles"] == 0 and "warning" in r
    elif n_bad == 1:
        assert r["n_source_shingles"] > 0               # 다른 원천의 조각에 가려지면 안 된다
        assert any(Path(w).name == p.name for w in bad for p in paths)

    assert S.main([]) == code
    out = capsys.readouterr().out
    assert ("비교 조각 0개" in out) == (n_bad > 0)
    assert S.main(["--json"]) == code
    assert json.loads(capsys.readouterr().out)["verdict"] == verdict


def test_빈_원천이_적발을_가리지도_적발이_빈_원천을_지우지도_않는다(tmp_path, monkeypatch, capsys):
    """차단이 이기되(종료 1) 미검사 사유는 결과에 남는다 — 둘 다 사람이 봐야 한다."""
    _one_target(tmp_path, monkeypatch, body="ISO 5817 을 인용한다")
    _two_sources(tmp_path, monkeypatch, "", _NORMAL_SOURCE)
    assert S.main(["--json"]) == 1
    r = json.loads(capsys.readouterr().out)
    assert r["verdict"] == "blocked" and r["n_blocking"] == 1
    assert "source_without_shingles" in [u["reason"] for u in r["unverified_reasons"]]


def test_파일은_있는데_문자열이_하나도_없으면_미검사다(tmp_path, monkeypatch):
    """구성원이 실재해도 검사한 문자열이 0 이면 본 것이 없다. 파일 수와 문자열 수를 따로 센다."""
    d = _seal(tmp_path / "cycle_x", {"a.jsonl": json.dumps({}) + "\n"})
    monkeypatch.setattr(S, "TARGET_DIRS", (d,))
    monkeypatch.setattr(S, "untracked_members", lambda _d: ["a.jsonl"])
    _fake_source(tmp_path, monkeypatch)
    r = S.screen()
    assert r["n_files_examined"] == 1 and r["n_texts_examined"] == 0
    assert r["verdict"] == "unverified"
    assert [u["reason"] for u in r["unverified_reasons"]] == ["nothing_examined"]
    assert S.main([]) == 2


def test_정상_입력은_clear_이고_검사량이_남는다(tmp_path, monkeypatch):
    _one_target(tmp_path, monkeypatch)
    r = S.screen()
    assert r["verdict"] == "clear" and r["unverified_reasons"] == []
    assert r["n_files_examined"] == 1 and r["n_texts_examined"] == 1
    assert "위험" not in r["verdict_note"]
    assert S.main([]) == 0


def test_차단_적발은_그대로_1이다(tmp_path, monkeypatch):
    _one_target(tmp_path, monkeypatch, body="ISO 5817 을 인용한다")
    r = S.screen()
    assert r["verdict"] == "blocked" and r["n_blocking"] == 1
    assert S.main([]) == 1


def test_clear_는_검사한_문자열이_있을_때만이다(tmp_path, monkeypatch):
    """이빨 — 빈 검사가 clear 로 나오면 이 시험이 실패해야 한다."""
    for dirs in ((), (tmp_path / "없는대상",)):
        monkeypatch.setattr(S, "TARGET_DIRS", dirs)
        r = S.screen()
        assert r["n_texts_examined"] == 0 and r["verdict"] != "clear"


def test_명시적_제외는_사유와_함께_통과한다(tmp_path, monkeypatch, capsys):
    """플래그로 뺀 대상은 "대상 부재" 를 일으키지 않는다 — 사유가 결과에 남는다.

    다만 **제외가 검사를 대신하지는 않는다**: 뺀 것 말고 실제로 본 문자열이 있어야 clear 다.
    전부 빼면 `nothing_examined` 로 미검사다(아래 시험) — 그러지 않으면 플래그 하나로 빈 검사가
    다시 통과한다.
    """
    kept = _one_target(tmp_path, monkeypatch)                 # 정상 대상 하나
    gone = tmp_path / "없는대상"
    monkeypatch.setattr(S, "TARGET_DIRS", (gone, kept))
    reason = "사유: 다른 저장소로 옮겼다"
    r = S.screen(excludes={"없는대상": reason})
    assert r["verdict"] == "clear" and r["n_targets_excluded"] == 1
    assert r["targets"][0]["exclude_reason"] == reason
    assert r["unverified_reasons"] == [] and r["n_texts_examined"] == 1
    assert S.main(["--exclude", f"없는대상={reason}"]) == 0
    assert f"제외 — {reason}" in capsys.readouterr().out

    # 제외만 남으면 본 것이 없다 → 미검사
    monkeypatch.setattr(S, "TARGET_DIRS", (gone,))
    r2 = S.screen(excludes={"없는대상": reason})
    assert r2["verdict"] == "unverified"
    assert [u["reason"] for u in r2["unverified_reasons"]] == ["nothing_examined"]


@pytest.mark.parametrize("case", ["clear", "blocked", "unverified", "excluded_only"])
def test_두_CLI_가_같은_판정과_종료_코드를_낸다(tmp_path, monkeypatch, capsys, case):
    """일반 출력과 `--json` 이 갈리면 자동화(종료 코드)와 사람(화면)이 다른 것을 본다.

    제외 사유는 JSON 에 남고 `verdict_note` 로 "검사 완료" 와 구분된다(검토 §22-2).
    """
    argv: list[str] = []
    if case == "unverified":
        monkeypatch.setattr(S, "TARGET_DIRS", (tmp_path / "없는대상",))
        _fake_source(tmp_path, monkeypatch)
        want = ("unverified", 2)
    elif case == "excluded_only":
        monkeypatch.setattr(S, "TARGET_DIRS", (tmp_path / "없는대상",))
        _fake_source(tmp_path, monkeypatch)
        argv = ["--exclude", "없는대상=사유: 다른 저장소로 옮겼다"]
        want = ("unverified", 2)          # 전부 제외해도 본 것이 없으면 미검사다
    else:
        _one_target(tmp_path, monkeypatch,
                    body="ISO 5817 을 인용한다" if case == "blocked" else "안전한 문장")
        want = ("blocked", 1) if case == "blocked" else ("clear", 0)

    assert S.main(argv) == want[1]
    plain = capsys.readouterr()
    assert f"판정: {want[0]}" in plain.out

    assert S.main([*argv, "--json"]) == want[1]
    r = json.loads(capsys.readouterr().out)
    assert r["verdict"] == want[0] and r["verdict_note"] == S.VERDICT_NOTE[want[0]]
    if case == "excluded_only":
        assert r["targets"][0]["exclude_reason"] == "사유: 다른 저장소로 옮겼다"
        assert [u["reason"] for u in r["unverified_reasons"]] == ["nothing_examined"]
        assert r["n_texts_examined"] == 0 and "검사 완료" not in r["verdict_note"]


_MOCK_BODY = json.dumps({"image_id": "aihub00000:00000000"}, ensure_ascii=False) + "\n"


def _mock_tree(tmp_path: Path, monkeypatch) -> Path:
    """가짜 저장소 뿌리. 정상 대상 하나(실제로 검사할 것)를 돌려준다."""
    kept = _one_target(tmp_path, monkeypatch)
    monkeypatch.setattr(S, "REPO", tmp_path)
    monkeypatch.setattr(S, "untracked_members", lambda _d: ["a.jsonl"])
    return kept


@pytest.mark.parametrize("rel", [
    "data/mock/mock_aihub_v1", "data/mock/mock_riawelc_v1",
    "data/mock/mock_aihub_v1/sub",                       # 스냅샷 **안**도 같은 판정 범위다
], ids=["aihub_스냅샷", "riawelc_스냅샷", "스냅샷_안"])
def test_판정받은_두_mock_스냅샷만_기본_제외다(tmp_path, monkeypatch, rel):
    """09-18 에 합성으로 판정한 것은 두 스냅샷뿐이다. 사유에 판정 근거와 범위가 남는다."""
    kept = _mock_tree(tmp_path, monkeypatch)
    mock = _seal(tmp_path / rel, {"a.jsonl": _MOCK_BODY})
    monkeypatch.setattr(S, "TARGET_DIRS", (mock, kept))
    r = S.screen()
    t = r["targets"][0]
    assert t["excluded"] and "합성" in t["exclude_reason"]
    assert "09-18 판정" in t["exclude_reason"] and "생성기 산물" in t["exclude_reason"]
    assert "mock_aihub_v1" in t["exclude_reason"] and "mock_riawelc_v1" in t["exclude_reason"]
    assert r["verdict"] == "clear" and r["n_blocking"] == 0 and S.main([]) == 0
    assert r["n_files_examined"] == 1 and r["n_texts_examined"] == 1   # 본 것은 정상 대상 하나뿐


@pytest.mark.parametrize("rel", [
    "data/mock/새폴더/x",                # mock 아래 새로 생긴 자료 — 판정 밖
    "data/mock/mock_aihub_v2",           # 이름이 비슷한 새 스냅샷 — 판정 밖
    "data/other/mock_x",                 # 같은 내용, 다른 경로
    "other/data/mock/x",                 # 경로 안에 mock 을 품었다 (67번 Minor 2)
    "other/data/mock/mock_aihub_v1/x",   # 경로 안에 제외 접두 **전체**를 품었다 — 부분 일치면 새어 나간다
], ids=["mock_아래_새폴더", "비슷한_이름", "다른_경로", "경로_안의_mock", "경로_안의_접두_전체"])
def test_판정_밖의_경로는_mock_이어도_검사한다(tmp_path, monkeypatch, rel):
    """경로 이름이 mock 이라는 이유만으로 새 자료까지 자동 승인하지 않는다(09-21 추기).
    접두 제외는 경로 **머리**에서, 그것도 판정받은 디렉터리 경계에서만 먹는다."""
    kept = _mock_tree(tmp_path, monkeypatch)
    target = _seal(tmp_path / rel, {"a.jsonl": _MOCK_BODY})
    monkeypatch.setattr(S, "TARGET_DIRS", (target, kept))
    r = S.screen()
    assert not r["targets"][0]["excluded"]
    assert r["verdict"] == "blocked" and r["n_blocking"] == 1 and S.main([]) == 1


def test_mock_뿌리와_안내문은_기본_제외가_아니다():
    """`data/mock` 자체와 `data/mock/README.md` 는 제외 범위 밖이다 — 검사 대상이다."""
    for rel in ("data/mock", "data/mock/README.md", "data/mock/mock_aihub_v1.bak"):
        assert not S._excluded_by_prefix(S.REPO / rel), rel
    for rel in S.EXCLUDED_PREFIXES:
        assert S._excluded_by_prefix(S.REPO / rel), rel
    assert set(S.EXCLUDED_PREFIXES) == {"data/mock/mock_aihub_v1", "data/mock/mock_riawelc_v1"}


def test_mock_안내문은_검사되고_지금_내용에_차단_사유가_없다():
    """안내문은 합성 판정의 대상이 아니라 **사람이 쓴 문서**다. 실제로 규칙에 건다.

    원천 조각은 실물이 있는 트리에서만 쓴다(미추적) — 없으면 규칙 셋(유료 표준·AI허브 식별자·
    로컬 경로)만 본다. 파일이 없는 트리에서는 사유와 함께 건너뛴다.
    """
    readme = REPO / "data/mock/README.md"
    if not readme.is_file():
        pytest.skip("data/mock/README.md 가 이 트리에 없다 — 안내문을 검사하지 못했다")
    text = readme.read_text(encoding="utf-8")
    shingles = S.source_shingles()                # 실물 원천이 없으면 빈 집합
    assert len(text.strip()) > 0
    assert S.screen_text(text, shingles) == []
    for i, line in enumerate(text.splitlines(), 1):
        assert S.screen_text(line, shingles) == [], f"{i}행"


# ------------------------------------------------------------- 백업 대조기

def test_목록은_계약서_자신도_담는다(tmp_path):
    """계약서를 빼면 사본만으로는 무엇이 있어야 하는지 알 수 없다 — 사본이 자립해야 한다."""
    d = _seal(tmp_path / "pairs_x", {"pairs.jsonl": "{}\n", "counts.json": "{}\n"})
    plan = B.build_plan([d])
    assert {i["name"] for i in plan["items"]} == {"pairs.jsonl", "counts.json",
                                                 CONTRACT_NAME}
    assert plan["total_bytes"] > 0 and not plan["problems"]
    # 레드라인 문구가 목록에 붙어 있어야 사람이 목적지를 고를 때 본다.
    assert "국외" in plan["redline"]


def test_계약에_있는데_실물이_없으면_목록이_문제로_적는다(tmp_path):
    d = _seal(tmp_path / "pairs_x", {"pairs.jsonl": "{}\n"})
    (d / "pairs.jsonl").unlink()
    plan = B.build_plan([d])
    assert plan["problems"] and "실물이 없다" in plan["problems"][0]


def test_사본이_같으면_통과한다(tmp_path):
    d = _seal(tmp_path / "pairs_x", {"pairs.jsonl": "{}\n"})
    plan = B.build_plan([d])
    dest = tmp_path / "drive"
    (dest / d.name).mkdir(parents=True)
    for it in plan["items"]:
        (dest / it["rel"]).write_bytes(Path(it["src"]).read_bytes())
    r = B.verify(plan, dest)
    assert r["verdict"] == "ok" and r["n_bad"] == 0


@pytest.mark.parametrize("damage", ["mismatch", "missing"])
def test_사본이_다르면_잡는다(tmp_path, damage):
    """이빨 시험 — 손상과 누락을 둘 다 잡아야 한다. 하나만 잡으면 나머지로 샌다."""
    d = _seal(tmp_path / "pairs_x", {"pairs.jsonl": "{}\n", "counts.json": "{}\n"})
    plan = B.build_plan([d])
    dest = tmp_path / "drive"
    (dest / d.name).mkdir(parents=True)
    for it in plan["items"]:
        (dest / it["rel"]).write_bytes(Path(it["src"]).read_bytes())

    target = dest / d.name / "pairs.jsonl"
    if damage == "mismatch":
        target.write_bytes(b"{} \n")
    else:
        target.unlink()

    r = B.verify(plan, dest)
    assert r["verdict"] == "failed"
    assert [x["verdict"] for x in r["rows"] if x["rel"].endswith("pairs.jsonl")] == [damage]


def test_원본이_계약과_어긋나면_사본도_통과하지_않는다(tmp_path):
    """원본↔사본만 보면 원본이 이미 틀어져 있어도 같이 틀어진 채 통과한다."""
    d = _seal(tmp_path / "pairs_x", {"pairs.jsonl": "{}\n"})
    (d / "pairs.jsonl").write_text("변조됨\n", encoding="utf-8", newline="")
    plan = B.build_plan([d])                      # 계약서는 옛 해시를 들고 있다
    dest = tmp_path / "drive"
    (dest / d.name).mkdir(parents=True)
    for it in plan["items"]:
        (dest / it["rel"]).write_bytes(Path(it["src"]).read_bytes())
    r = B.verify(plan, dest)
    assert r["verdict"] == "failed"
    bad = [x for x in r["rows"] if x["verdict"] == "contract_mismatch"]
    assert bad and bad[0]["matches_source"] is True   # 사본은 원본과 같은데도 막혔다


def test_verify_는_dest_없이_돌지_않는다():
    assert B.main(["verify"]) == 2


# ------------------------------------------------------------- 실물 (읽기만)

def test_실물_백업_목록이_계약과_맞는다():
    """사람이 옮길 실물. 목록이 계약과 어긋나면 잘못된 것을 복사한다.

    09-08 판정으로 cycle_pilot·cycle_pilot_v2 가 한 벌에 들어왔다(추적 전환 기각).
    09-15 부터 목록은 봉인처 명부에서 파생한다 — 소실 기록(v2)은 빠지고, 봉인 아닌
    `judge_labels/` 가 들어온다. 그래서 파일 수를 상수로 박지 않고 구성으로 센다.
    """
    plan = B.build_plan(B.DEFAULT_DIRS)
    if plan["problems"]:
        pytest.skip(f"이 트리에서 대상이 온전하지 않다: {plan['problems']}")
    assert {Path(d).name for d in B.DEFAULT_DIRS} == set(plan["dirs"])
    expected = 0
    for d in B.SEALED_DIRS:
        names, _ = B.snapshot_summary(d)
        expected += len(names) + 1                      # 구성원 + 계약서
    for d in B.UNSEALED_DIRS:
        expected += sum(1 for p in d.iterdir() if p.is_file())
    assert plan["n_files"] == expected, (plan["n_files"], expected)
    for it in plan["items"]:
        if it["contract_sha256"] is not None:
            assert it["sha256"] == it["contract_sha256"], it["rel"]


def test_백업_목록은_봉인처_명부에서_파생된다():
    """두 목록이 따로 살면 한쪽만 고쳐진다 — 09-13 에 이 목록이 소실된 v2 를 계속 들고 있었다."""
    from corpus.generate.frozen_out import EXPECTED_SEALED

    expected = {B.REPO / rel for rel, e in EXPECTED_SEALED.items()
                if e["status"] in B.BACKED_UP_STATUSES}
    lost = {B.REPO / rel for rel, e in EXPECTED_SEALED.items() if e["status"] == "lost"}
    assert set(B.SEALED_DIRS) == expected
    assert not (set(B.DEFAULT_DIRS) & lost), "소실 기록된 곳은 옮길 실물이 없다"
    assert B.REPO / "corpus/validate/judge_labels" in B.DEFAULT_DIRS
    # 대조기와 같은 집합이어야 "있어야 한다" 와 "옮긴다" 가 갈리지 않는다.
    from corpus.generate.frozen_out import PRESENT_STATUSES
    assert B.BACKED_UP_STATUSES is PRESENT_STATUSES
    # 09-16 수용 — 본실험 매니페스트 계약도 한 벌에 든다(39번 I-1 · 62f660b).
    assert B.REPO / "data/interim/manifest_v1" in B.SEALED_DIRS


def test_복원된_봉인처도_백업_목록에_든다():
    """`restored` 는 소실 뒤 동일 바이트로 돌아온 자리다 — git 밖 단일 사본이 다시 생긴
    것이라 백업이 급하다. 상태 어휘를 늘린 쪽(A)과 목록을 파생하는 쪽(B)이 따로 살면
    목록이 조용히 줄어든다 — 09-13 에 반대 방향으로 같은 일이 있었다 (41번 I-1)."""
    reg = {"a/expected": {"status": "expected", "owner": "B"},
           "b/restored": {"status": "restored", "owner": "A", "record": "복원", "evidence": "64자 일치"},
           "c/lost": {"status": "lost", "owner": "B", "record": "소실"}}
    got = {p.as_posix().rsplit("/", 2)[-2] + "/" + p.name for p in B.sealed_dirs(reg, Path("/r"))}
    assert got == {"a/expected", "b/restored"}


def test_사람_라벨_폴더가_백업_목록에_있고_미추적분이_위험분으로_잡힌다(tmp_path, monkeypatch):
    """labels_*.jsonl 은 미추적이다. 백업 목록에 없으면 09-11 과 같은 단일 사본 구조다."""
    d = tmp_path / "judge_labels"
    d.mkdir()
    (d / "sheet_v1.jsonl").write_text("{}\n", encoding="utf-8", newline="")
    (d / "labels_v1.jsonl").write_text('{"labeler": "x"}\n', encoding="utf-8", newline="")
    monkeypatch.setattr(B, "UNSEALED_DIRS", (d,))
    monkeypatch.setattr(B, "tracked_names", lambda _d: frozenset({"sheet_v1.jsonl"}))
    plan = B.build_plan([d])
    assert not plan["problems"]
    by = {it["name"]: it for it in plan["items"]}
    assert by["labels_v1.jsonl"]["tracked"] is False and by["labels_v1.jsonl"]["sealed"] is False
    assert by["sheet_v1.jsonl"]["tracked"] is True
    assert plan["n_at_risk"] == 1
    # 봉인 아닌 항목은 계약 대조가 없다 — 그 사실이 항목에 남아야 verify 가 속지 않는다
    assert all(it["contract_sha256"] is None for it in plan["items"])


def test_봉인도_아니고_목록에도_없는_디렉터리는_여전히_문제로_적는다(tmp_path):
    """아무 디렉터리나 넘기면 전부 담아 주는 것이 아니다 — 명시된 곳만."""
    d = tmp_path / "아무거나"
    d.mkdir()
    (d / "x.txt").write_text("x", encoding="utf-8")
    plan = B.build_plan([d])
    assert plan["problems"] and "봉인본이 아니다" in plan["problems"][0]


def test_위험분과_안전분을_목록이_가른다():
    """★(git 밖)과 ㆍ(git 에도 있음)을 안 가르면 사람이 무엇이 유일본인지 모른다."""
    plan = B.build_plan(B.DEFAULT_DIRS)
    if plan["problems"]:
        pytest.skip("이 트리에서 대상이 온전하지 않다")
    assert 0 < plan["n_at_risk"] < plan["n_files"]
    assert plan["at_risk_bytes"] < plan["total_bytes"]
    # 추적분은 git 이 들고 있다는 뜻이므로 실제로 추적 중이어야 한다.
    from corpus.generate.frozen_out import tracked_names
    for it in plan["items"]:
        d = Path(it["src"]).parent
        assert it["tracked"] == (it["name"] in (tracked_names(d) or frozenset())), it["rel"]
