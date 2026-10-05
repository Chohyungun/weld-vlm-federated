"""봉인 관문(`seal_pairs_main`) — 종료 코드 하나로 통과를 선언하지 않는가.

검사기를 실제로 띄우지 않고 **돌려받는 보고를 대신 넣어** 판정 논리만 본다. 여기서 잠그는 것은 둘이다.
① 두 검사기 중 하나라도 통과가 아니면 봉인을 거부한다. ② 종료 코드가 0 이어도 보고의 내용이 통과가
아니면 거부한다 — 독립 대조기는 `--pairs` 를 빼면 대조 없이 0 을, `--allow` 를 주면 불일치를 면제하고도 0 을 낸다.

**둘 다 통과하는 경로**는 여기서 덮지 않는다. 실제 두 검사기를 실물 빌드에 걸어 본 뒤에 닫힌다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.validate import seal_pairs_main as S

OK_VERIFY = {"n_records": 4, "n_discarded": 1, "coverage": "full", "verdict": "일치",
             "not_checked": [], "failures": {}, "ok": True, "n_image_paths_sampled": 4}
#: 독립 대조기의 보고. **그 소스는 어느 경로로 끝나든 `sealable`·`coverage`·`verdict`·`not_compared` 를
#: 늘 싣는다.** 아래 여섯은 그 소스(`scripts/probe/pairs_independent.py`, 판본 `a17502b3…`)가 실제로
#: 내는 모양이다. 출처는 **상수·함수 이름으로만** 가리킨다 — 줄 번호는 판본마다 밀리고, 여기 적었던
#: 다섯 개가 실제로 열 줄쯤 어긋나 있었다(교차 검수 I-5). 실물이 낼 수 없는 보고로 시험하면
#: 통과해도 아무것도 확인하지 않는다.
OK_INDEP = {"mismatch": {}, "blocking": {}, "not_compared": [], "coverage": "full",
            "verdict": "일치", "sealable": True, "meta_note": None, "counts_note": None,
            "n_expected_keep": 4, "n_expected_discard": 1, "n_got": 4}
"""`main()` 의 판정부 마지막 갈래 — blocking·mismatch·not_compared 가 다 비면 `일치` · `EXIT_MATCH`(0)."""

MISMATCH_INDEP = {**OK_INDEP, "mismatch": {"clauses": 3}, "blocking": {"clauses": 3},
                  "verdict": "불일치", "sealable": False}
"""판정부 첫 갈래 — `blocking` 이 남으면 `불일치` · `EXIT_MISMATCH`(1)."""

ALLOWED_INDEP = {**OK_INDEP, "mismatch": {"clauses": 3}, "blocking": {},
                 "verdict": "허용 범주만 남음", "sealable": False}
"""판정부 둘째 갈래 — `--allow` 로 `blocking` 을 비웠다 · `EXIT_ALLOWED`(3).
**허용으로 비운 불일치도 불일치다.**"""

LIMITED_INDEP = {**OK_INDEP, "not_compared": ["meta.counts_by_split"], "coverage": "partial",
                 "verdict": "대조 범위 제한 — 본 것은 전부 같았다", "sealable": False,
                 "meta_note": 'PAIRS_META.json 의 counts_by_split 구조를 모른다 — 대조하지 않았다'}
"""판정부 셋째 갈래 — mismatch 는 비고 `not_compared` 가 남으면 `EXIT_PARTIAL`(2).

`PAIRS_META.json` 은 **있는데** `counts_by_split` 의 구조를 모를 때의 보고다. 그 경로에서만 축이
`meta.counts_by_split` **하나**이고 `meta_note` 가 위 문장이다(`compare()` 의 메타 블록). 파일이 **없으면**
축이 일곱이다 — 그 경우는 `MISMATCH_LIMITED_INDEP` 가 싣는다. 앞 판은 축 하나에 "파일이 없다" 를 붙여
두 경로를 섞었다(교차 검수 m-9). 축 표기는 **점**이다(m-5). `meta_note` 는 관문이 읽지 않는다 —
분류는 `not_compared` 가 비었는지로 한다."""

MISMATCH_LIMITED_INDEP = {**MISMATCH_INDEP,
                          "not_compared": [
                            'meta.counts_by_split',
                            'meta.image_paths_checked',
                            'meta.input_snapshot.snapshot_digest',
                            'meta.limits_table.sha256',
                            'meta.n_annotations_skipped_geom_invalid',
                            'meta.n_annotations_skipped_in_records',
                            'meta.validated_by',
                          ],
                          "coverage": "partial",
                          "meta_note": 'PAIRS_META.json 이 없다 — 분할 축 회계와 기하 무효 총계를 대조하지 않았다(파일럿에는 없다)'}
"""**둘이 겹친 보고.** `mismatch` 와 `not_compared` 는 서로 독립이라 `PAIRS_META.json` 이 없는 빌드에
레코드 불일치가 하나 있으면 함께 찬다. 판정부는 `blocking` 을 먼저 보므로 `EXIT_MISMATCH`(1) 다.

파일이 없으면 `compare()` 가 메타 블록을 통째로 건너뛰고 `skipped.extend(...)` 로 **축 일곱**을 넣는다.
위 목록과 `meta_note` 는 그 소스의 구문 트리에서 상수를 풀어 뽑은 값이고, `sorted(set(skipped))` 의
순서를 따른다 — 손으로 옮기지 않았다(교차 검수 m-9)."""

NOT_COMPARED_INDEP = {"mismatch": {}, "blocking": {}, "not_compared": ["pairs.*"], "coverage": "none",
                      "verdict": "대조하지 않았다 — 기대 레코드만 냈다", "sealable": False}
"""`--pairs` 를 주지 않은 분기 — `not_compared` 는 `NOT_COMPARED_ALL`(`"pairs.*"`) 하나이고
`EXIT_PARTIAL`(2) 을 낸다. 이 분기는 `compare()` 를 부르지 않으므로 `n_expected_keep`·`n_got`·
`meta_note`·`counts_note` 가 **아예 없다** — 그래서 다른 픽스처를 물려받지 않는다(m-6).
관문은 늘 `--pairs` 를 주므로 **나오면 안 되는 보고다.**"""


@pytest.fixture()
def gate(tmp_path, monkeypatch):
    """`_run` 을 가로채 보고를 대신 넣는다. 호출 인자를 남겨 두어 무엇을 주고 무엇을 안 줬는지 본다."""
    calls: list[list[str]] = []
    replies: list[tuple[int, dict | None, str]] = []

    def fake_run(argv):
        calls.append(argv)
        return replies[len(calls) - 1]

    monkeypatch.setattr(S, "_run", fake_run)
    monkeypatch.setattr(S, "INDEPENDENT", tmp_path / "pairs_independent.py")
    (tmp_path / "pairs_independent.py").write_text("# 자리만 있으면 된다", encoding="utf-8")

    def run(*answers, out=None):
        replies.clear()
        replies.extend(answers)
        calls.clear()          # 한 시험에서 여러 번 부를 수 있게 회차마다 비운다
        return S.check(tmp_path / "build", tmp_path / "snap", tmp_path, out or (tmp_path / "out"))

    run.calls = calls          # type: ignore[attr-defined]
    return run


@pytest.fixture()
def gate_exit(tmp_path, monkeypatch):
    """`main()` 의 **종료 코드**까지 본다 — 합성 판정이 코드로 번역되는 자리가 거기다."""
    replies: list[tuple[int, dict | None, str]] = []
    calls: list[list[str]] = []

    def fake_run(argv):
        calls.append(argv)
        return replies[len(calls) - 1]

    monkeypatch.setattr(S, "_run", fake_run)
    monkeypatch.setattr(S, "INDEPENDENT", tmp_path / "pairs_independent.py")
    (tmp_path / "pairs_independent.py").write_text("# 자리만 있으면 된다", encoding="utf-8")

    def run(*answers):
        replies.clear()
        replies.extend(answers)
        calls.clear()
        return S.main(["--build", str(tmp_path / "build"), "--snapshot", str(tmp_path / "snap"),
                       "--root", str(tmp_path), "--out", str(tmp_path / "out")])
    return run


def test_둘_다_통과하면_봉인_가능이다(gate):
    rep = gate((0, OK_VERIFY, ""), (0, OK_INDEP, ""))
    assert rep["봉인_가능"] and rep["판정"] == "통과"
    assert [c["사유"] for c in rep["검사"]] == [[], []]


def test_독립_대조에는_pairs_를_주고_allow_를_주지_않는다(gate):
    """`--pairs` 가 없으면 대조 없이 0 이 나오고, `--allow` 는 불일치를 면제한다. 둘 다 관문을 무력화한다."""
    gate((0, OK_VERIFY, ""), (0, OK_INDEP, ""))
    argv = gate.calls[1]                      # type: ignore[attr-defined]
    assert "--pairs" in argv and "--allow" not in argv


@pytest.mark.parametrize("label, verify, indep", [
    ("검사기 1 이 적발", (1, {**OK_VERIFY, "failures": {"F_discard_unjustified": 1}, "ok": False,
                         "verdict": "불일치"}, ""), (0, OK_INDEP, "")),
    ("검사기 2 가 불일치", (0, OK_VERIFY, ""), (1, MISMATCH_INDEP, "")),
    ("불일치를 허용 범주로 비웠다", (0, OK_VERIFY, ""), (3, ALLOWED_INDEP, "")),
    ("계약 필드가 빠진 보고", (0, OK_VERIFY, ""), (0, {"n_expected_keep": 4}, "")),
    ("보고를 읽지 못했다", (0, OK_VERIFY, ""), (0, None, "traceback")),
])
def test_하나라도_통과가_아니면_거부한다(gate, label, verify, indep):
    rep = gate(verify, indep)
    assert not rep["봉인_가능"], label
    assert any(c["사유"] for c in rep["검사"]), label


def test_종료_코드가_0_이어도_대조_범위가_좁으면_거부한다(gate):
    """적발이 없는 것과 전부 본 것은 다른 사실이다. 범위 제한은 2 로 낸다 — 봉인 조건은 0 이다."""
    rep = gate((2, {**OK_VERIFY, "coverage": "partial", "not_checked": ["meta.image_paths_checked"],
                    "verdict": "대조 범위 제한 — 본 것은 전부 같았다"}, ""), (0, OK_INDEP, ""))
    assert not rep["봉인_가능"] and rep["판정"] == "대조 범위 제한"


def test_메타_대조를_건너뛰면_범위_제한이다(gate):
    """실물은 `meta_note` 를 붙일 때 `not_compared` 에도 축을 넣어 종료 2 를 낸다 — 그 모양으로 건다."""
    rep = gate((0, OK_VERIFY, ""), (2, LIMITED_INDEP, ""))
    assert not rep["봉인_가능"] and rep["판정"] == "대조 범위 제한"


def test_독립_대조기가_없으면_거부한다(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "INDEPENDENT", tmp_path / "없는파일.py")
    rep = gate((0, OK_VERIFY, ""))
    assert not rep["봉인_가능"] and rep["검사"][1]["종료코드"] is None


def test_산출_경로가_이미_있으면_지우지_않고_거부한다(gate, tmp_path):
    (tmp_path / "이미있음").mkdir()
    rep = gate((0, OK_VERIFY, ""), out=tmp_path / "이미있음")
    assert not rep["봉인_가능"]
    assert (tmp_path / "이미있음").is_dir()          # 지우지 않는다


def test_덮지_않는_것을_보고가_스스로_말한다(gate):
    """0 을 '봉인 절차가 끝났다' 로 읽지 않게, 이 관문 밖의 조건을 산출물이 들고 다닌다."""
    rep = gate((0, OK_VERIFY, ""), (0, OK_INDEP, ""))
    assert rep["덮지_않는_것"]
    assert json.dumps(rep, ensure_ascii=False)      # 표준 JSON 으로 직렬화된다


# ------------------------------------------------------------------ 섞인 상태 (외부 검토 §2)

FINDING = {**OK_VERIFY, "failures": {"F_discard_unjustified": 1}, "ok": False, "verdict": "불일치"}
LIMITED_VERIFY = {**OK_VERIFY, "coverage": "partial", "not_checked": ["meta.image_paths_checked"],
                  "verdict": "대조 범위 제한 — 본 것은 전부 같았다"}
def test_적발과_범위_제한이_섞이면_적발이_이긴다(gate):
    """`all`·`any` 로 합치면 적발이 범위 제한으로 내려간다 — 최종 상태만 읽는 쪽이 잘못 분류한다."""
    rep = gate((1, FINDING, ""), (2, LIMITED_INDEP, ""))
    assert rep["판정"] == "불통과", rep["판정"]
    assert not rep["봉인_가능"]


def test_적발로_판정해도_미대조_축은_지우지_않는다(gate):
    """낮은 단계로 요약하지 않는 것과 사실을 버리는 것은 다르다. 둘 다 남아야 한다."""
    rep = gate((1, FINDING, ""), (2, LIMITED_INDEP, ""))
    assert rep["범위제한_있음"] is True
    assert rep["범위제한_검사"] == ["pairs_independent"]
    # 실물 사유는 `not_compared` 의 축 이름을 댄다. 축이 남았다는 사실이 사유에 살아 있어야 한다.
    assert any("meta.counts_by_split" in x for x in rep["검사"][1]["사유"]), rep["검사"][1]["사유"]


@pytest.mark.parametrize("label, verify, indep", [
    ("적발 + 범위 제한", (1, FINDING, ""), (2, LIMITED_INDEP, "")),
    ("범위 제한 + 실행 실패", (2, LIMITED_VERIFY, ""), (0, None, "traceback")),
])
def test_섞인_상태의_종료_코드는_1_이다(gate_exit, label, verify, indep):
    """규약은 `1 = 적발, 2 = 대조 범위 제한` 이다. 적발이 있었는데 2 가 나가면 규약이 거짓말을 한다."""
    assert gate_exit(verify, indep) == 1, label


def test_범위_제한만_있으면_2_다(gate_exit):
    """섞인 상태를 1 로 올리면서 단독 범위 제한까지 1 로 올려서는 안 된다."""
    assert gate_exit((2, LIMITED_VERIFY, ""), (0, OK_INDEP, "")) == 2


def test_둘_다_통과하면_0_이다(gate_exit):
    assert gate_exit((0, OK_VERIFY, ""), (0, OK_INDEP, "")) == 0


def test_대조기_부재는_단순_미대조와_다르다(gate, tmp_path, monkeypatch):
    """부재·실행 실패는 '못 봤다' 가 아니라 '돌지 못했다' 다. 같은 칸에 넣으면 구별이 사라진다."""
    monkeypatch.setattr(S, "INDEPENDENT", tmp_path / "없는파일.py")
    rep = gate((2, LIMITED_VERIFY, ""))
    assert rep["판정"] == "불통과"
    assert rep["검사"][0]["상태"] == "범위제한"
    assert rep["검사"][1]["상태"] == "실행실패"


@pytest.mark.parametrize("label, reply, want", [
    ("깨끗하다", (0, OK_VERIFY, ""), "통과"),
    ("적발", (1, FINDING, ""), "적발"),
    ("범위 제한", (2, LIMITED_VERIFY, ""), "범위제한"),
    ("보고를 읽지 못했다", (0, None, "traceback"), "실행실패"),
    ("보고와 종료 코드가 어긋난다", (0, FINDING, ""), "실행실패"),
])
def test_검사마다_상태가_넷을_가른다(gate, label, reply, want):
    """세 갈래를 한 불리언 둘로 표현하던 자리다. 상태를 이름으로 두면 합성 규칙이 드러난다."""
    rep = gate(reply, (0, OK_INDEP, ""))
    assert rep["검사"][0]["상태"] == want, (label, rep["검사"][0])


# ------------------------------------------------------------- 독립 대조기의 보고를 필드로 분류한다 (검수 I-1)


def test_독립_대조기의_범위_제한은_범위_제한이다(gate):
    """`verdict` 문장으로 분류하면 이 보고가 적발이 된다 — 그 대조기는 범위 제한일 때 `일치` 라고 쓰지 않는다.

    사람이 읽는 문장이 아니라 기계가 읽는 필드(`mismatch`·`coverage`)로 가른다.
    """
    rep = gate((0, OK_VERIFY, ""), (2, LIMITED_INDEP, ""))
    assert rep["검사"][1]["상태"] == "범위제한", rep["검사"][1]
    assert rep["판정"] == "대조 범위 제한"
    assert rep["범위제한_있음"] is True and rep["범위제한_검사"] == ["pairs_independent"]
    assert rep["불통과_검사"] == []


def test_독립_대조기의_범위_제한만_있으면_종료_2_다(gate_exit):
    assert gate_exit((0, OK_VERIFY, ""), (2, LIMITED_INDEP, "")) == 2


def test_허용_범주로_비운_불일치는_적발이다(gate):
    """종료 3 이다. `blocking` 이 비었어도 `mismatch` 가 남았으면 불일치다 — 기존 입장 그대로."""
    rep = gate((0, OK_VERIFY, ""), (3, ALLOWED_INDEP, ""))
    assert rep["검사"][1]["상태"] == "적발" and rep["판정"] == "불통과"


def test_대조를_하나도_하지_않은_보고는_막는다(gate):
    """`coverage: none` 은 `--pairs` 를 주지 않은 실행이다. 관문은 늘 주므로 이 보고가 오면 실행이 잘못됐다.

    범위가 좁았던 것이 아니라 **대조를 하지 않은 것**이라 범위 제한으로 내리지 않는다.
    """
    rep = gate((0, OK_VERIFY, ""), (2, NOT_COMPARED_INDEP, ""))
    assert rep["검사"][1]["상태"] == "실행실패", rep["검사"][1]
    assert rep["판정"] == "불통과"


@pytest.mark.parametrize("label, code, rep_in", [
    ("종료 0 인데 범위가 좁다", 0, LIMITED_INDEP),
    ("종료 2 인데 전 축을 봤다", 2, OK_INDEP),
    ("봉인 가능이라는데 불일치가 있다", 1, {**MISMATCH_INDEP, "sealable": True}),
])
def test_독립_대조의_필드가_서로_어긋나면_막는다(gate, label, code, rep_in):
    """검사 1 에 넣은 원칙과 같다 — 보고와 종료 코드가 어긋나면 어느 쪽도 믿지 않는다."""
    rep = gate((0, OK_VERIFY, ""), (code, rep_in, ""))
    assert rep["검사"][1]["상태"] == "실행실패", (label, rep["검사"][1])


def test_실행실패_검사가_최종_보고에_따로_있다(gate):
    """`불통과_검사` 만 읽으면 '어긋났다' 와 '돌지 못했다' 가 같은 칸에 들어간다 (검수 I-3)."""
    rep = gate((1, FINDING, ""), (0, None, "traceback"))
    assert rep["실행실패_검사"] == ["pairs_independent"]
    assert rep["불통과_검사"] == ["verify_pairs_main", "pairs_independent"]
    assert rep["판정"] == "불통과"


def test_같은_불일치가_사유에_두_번_붙지_않는다(gate):
    """`mismatch` 와 `종료 코드 1` 이 겹쳐 적혔다 (검수 m-2)."""
    사유 = gate((0, OK_VERIFY, ""), (1, MISMATCH_INDEP, ""))["검사"][1]["사유"]
    assert len(사유) == 1, 사유


def test_표준오류의_절대경로를_보고에_싣지_않는다(gate, tmp_path):
    """`_rel()` 로 경로를 감추면서 표준오류 꼬리는 그대로 실었다 (검수 m-3)."""
    사유 = gate((0, OK_VERIFY, ""), (0, None, r"Traceback: C:\Users\someone\secret\x.py 에서"))["검사"][1]["사유"]
    붙인말 = " ".join(사유)
    assert "someone" not in 붙인말 and "C:" not in 붙인말, 붙인말


# ------------------------------------------------- 한 검사 안에서 겹친 경우 (검수 I-4) · 경로 감추기 (m-7·m-8)


def test_한_검사_안에서_불일치와_범위_제한이_겹쳐도_둘_다_남는다(gate):
    """`elif` 였던 탓에 불일치가 있으면 범위 제한을 보지 않았다. 상태는 적발이 맞고, 사실이 지워지면 안 된다."""
    rep = gate((0, OK_VERIFY, ""), (1, MISMATCH_LIMITED_INDEP, ""))
    검사 = rep["검사"][1]
    assert 검사["상태"] == "적발", 검사
    assert 검사["미대조_있음"] is True
    assert any("불일치" in x for x in 검사["사유"]) and any("대조 못 한 축" in x for x in 검사["사유"]), 검사["사유"]
    assert rep["범위제한_있음"] is True and rep["범위제한_검사"] == ["pairs_independent"]
    assert rep["판정"] == "불통과"


def test_산출물_검사_안에서_겹친_경우도_같다(gate):
    """같은 형태가 검사 1 에도 있다 — 적발과 미대조 축이 한 보고에 함께 올 수 있다."""
    both = {**OK_VERIFY, "failures": {"F_discard_unjustified": 1}, "ok": False, "verdict": "불일치",
            "coverage": "partial", "not_checked": ["meta.image_paths_checked"]}
    rep = gate((1, both, ""), (0, OK_INDEP, ""))
    assert rep["검사"][0]["상태"] == "적발" and rep["검사"][0]["미대조_있음"] is True
    assert rep["범위제한_있음"] is True and rep["범위제한_검사"] == ["verify_pairs_main"]


def test_겹친_경우의_종료_코드는_1_이다(gate_exit):
    assert gate_exit((0, OK_VERIFY, ""), (1, MISMATCH_LIMITED_INDEP, "")) == 1


def test_최상위_경로도_감춘다(gate, tmp_path):
    """`--snapshot` 기본값이 절대경로라 보고에 그대로 실렸다 (검수 m-7)."""
    rep = gate((0, OK_VERIFY, ""), (0, OK_INDEP, ""))
    붙인말 = f"{rep['build']} {rep['snapshot']}"
    assert str(tmp_path) not in 붙인말 and ":" not in 붙인말, 붙인말


@pytest.mark.parametrize("label, raw, 금지", [
    ("POSIX 절대경로", "Traceback: /home/someone/secret/x.py 에서", "/home/someone"),
    ("공백이 든 집 경로", str(Path.home() / "My Papers" / "secret.py"), str(Path.home())),
    ("저장소 안의 공백 경로", str(S.REPO / "some dir" / "x.py"), str(S.REPO)),
])
def test_공백과_POSIX_경로도_감춘다(label, raw, 금지):
    """정규식이 드라이브 경로만 봤다 (검수 m-8). POSIX 경로와 공백이 든 경로의 **뿌리**가 사라져야 한다.

    공백이 든 경로는 첫 공백에서 멈추므로 꼬리가 남는다 — 남는 것은 절대경로가 아니고,
    거기까지가 `_scrub` 의 약속이다(그 docstring).
    """
    got = S._scrub(raw)
    assert 금지 not in got, (label, got)
    assert "<경로>" in got or got.startswith("some dir"), (label, got)


def test_사유를_잇는_구분자를_경로로_읽지_않는다():
    """`/` 를 넓게 잡으면 " / " 로 이은 사유나 낱말의 사선이 경로로 지워진다."""
    assert S._scrub("첫째 / 둘째") == "첫째 / 둘째"
    assert S._scrub("검출/판정 구조") == "검출/판정 구조"
