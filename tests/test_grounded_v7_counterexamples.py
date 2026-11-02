"""판정 v7 결정적 grounded 의 반례 — 검토 §137 I-02(결함 부재절의 범위) · I-03(속성 판독 불가는 부재가 아니다), 보강 7.

반례(`evaluation/vectors/grounded_v7_counterexamples.json`)는 개발 기록의 서른한 묶음 사본이다. 35a · 37a 는 고치기 전에 의미로만
정해 적었고, 37b ~ 37g 는 원저자가 아닌 적대 검토(1차 ~ 6차)가, 37h · 37i 는 원저자가 단독으로(7차 · 8차), 37k 는 원저자 아닌 독립
검수가 돌리기 전에 의도를 정해 적었다. 판단으로 바꾼 의도는 사본의 `overrides` 에 앞 값 · 까닭과 함께 싣는다(원본은 그대로). 기대값은 잠긴 사실과 글의 참거짓이고 어떤 구현의 출력에서 베끼지 않았다. 의도 null → grounded_codes 가 빈 목록, 코드 목록 → 그 코드가 모두 나온다, 빈 목록 →
코드와 무관하게 기각. 남은 위험(`residual`)은 지금도 의도와 어긋나는 사례다 — 시험은 그것들이 아직 틀린다는 것을 고정한다.
사례의 열쇠는 묶음/번호다(번호는 묶음 사이에 겹칠 수 있다).
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import grounded_v7_ref as R  # noqa: E402
from scripts.probe.grounded_v7_structural_cases import L, task_of  # noqa: E402

CX = ROOT / "evaluation/vectors/grounded_v7_counterexamples.json"
SOURCES = ROOT / "docs/dev_log/2026-10-07-본실험코퍼스"
DOC = json.loads(CX.read_text(encoding="utf-8"))
CASES = DOC["cases"]
RESIDUAL = {(r["source"], r["id"]): r for r in DOC["residual"]}
TAGS = ["35a", "37a", "37b-A", "37b-B", "37b-C", "37b-C2", "37c-D", "37c-E", "37c-F", "37c-F2", "37c-F3", "37c-F4", "37d-H", "37d-I1",
        "37d-I2", "37d-I3", "37e-J", "37e-J2", "37e-K", "37f-L", "37f-M", "37g-N", "37g-N2", "37g-O", "37h-Q", "37i-R", "37k-C",
        "37k-CA", "37k-CA2", "37k-CB", "37k-CB2"]


def _ok(intent, got) -> bool:
    if intent is None:
        return got == []
    return bool(got) and set(intent) <= set(got)


def test_반례_사본은_원본과_같다() -> None:
    """원본(개발 기록)이 있는 트리에서는 바이트 해시와 사례가 사본과 같아야 한다 — 손으로 옮기지 않았다. 해시는 저장소 바이트(LF)로
    본다 — 작업 트리가 CRLF 로 풀려 있어도 같다."""
    assert [s["tag"] for s in DOC["sources"]] == TAGS
    assert sum(s["n"] for s in DOC["sources"]) == len(CASES) == len({(c["source"], c["id"]) for c in CASES})
    present = [s for s in DOC["sources"] if (SOURCES / s["name"]).exists()]
    if not present:
        pytest.skip("반례 원본(개발 기록)이 없는 트리")
    ov = {(o["key"], o["field"]): o for o in DOC["overrides"]}
    for s in present:
        raw = (SOURCES / s["name"]).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest() == s["sha256"], s["name"]
        want = []
        for c in json.loads(raw)["cases"]:
            row = {"source": s["tag"], "id": c["id"], "tier": c.get("tier", "core"),
                   "target": c.get("target", c.get("rule", c.get("probe", ""))), "defects": c["defects"], "text": c["text"],
                   "intent": c["intent"]}
            for k in ("tier", "target", "intent"):
                o = ov.get((f"{s['tag']}/{c['id']}", k))
                if o:
                    assert o["from"] == row[k], o
                    row[k] = o["to"]
            want.append(row)
        assert want == [c for c in CASES if c["source"] == s["tag"]]
    # 판단으로 바꾼 의도는 판단 2(지시 1015c)의 여덟뿐이고 모두 기각이다
    assert sorted({o["key"] for o in DOC["overrides"] if o["field"] == "intent"}) == [
        "37e-J/J-01", "37e-J/J-02", "37e-J/J-03", "37e-J/J-47", "37e-J2/J2-07", "37e-K/K-17", "37h-Q/Q-13", "37i-R/R-17"]
    assert all(o["to"] == [] for o in DOC["overrides"] if o["field"] == "intent")


def test_반례_수와_남은_위험() -> None:
    """남은 위험은 고치기 전 정본도 틀렸거나(exposed 없음), 고치기 전 정본이 같은 글의 속성 판독 불가 문장 · 절을 부재로 잘못 읽은
    덕에 맞았던 것(exposed — 그 부분을 빼면 고치기 전 정본도 지금과 같은 결과를 낸다)이다. 35a 는 남은 위험이 없다."""
    n = Counter(c["source"] for c in CASES)
    assert n == {"35a": 18, "37a": 32, "37b-A": 42, "37b-B": 52, "37b-C": 36, "37b-C2": 5, "37c-D": 48, "37c-E": 76, "37c-F": 26,
                 "37c-F2": 10, "37c-F3": 6, "37c-F4": 2, "37d-H": 66, "37d-I1": 26, "37d-I2": 10, "37d-I3": 5, "37e-J": 48,
                 "37e-J2": 12, "37e-K": 63, "37f-L": 45, "37f-M": 78, "37g-N": 55, "37g-N2": 6, "37g-O": 77, "37h-Q": 42, "37i-R": 24,
                 "37k-C": 31, "37k-CA": 36, "37k-CA2": 6, "37k-CB": 36, "37k-CB2": 9}
    for r in RESIDUAL.values():
        if r["old_ok"]:
            assert r["exposed"]["old_without"] == r["new"] == r["exposed"]["new_without"], r["key"]
        else:
            assert "exposed" not in r, r["key"]
    assert sorted(r["key"] for r in RESIDUAL.values() if r["old_ok"]) == [
        "37c-E/E-57", "37c-E/E-66", "37d-H/H-56", "37d-H/H-66", "37e-J/J-33", "37e-J/J-34", "37e-K/K-43", "37e-K/K-44", "37e-K/K-45"]
    assert not any(r["source"] == "35a" for r in RESIDUAL.values())
    # 긍정 서술 뒤 이음(있고 · 보이며 · 있으면서)의 부재 범위는 고치기 전처럼 앞 칸까지 번지는 넷과 확장 N-29(주제 구역이 다른 이름에 걸침)
    assert [r["id"] for r in RESIDUAL.values() if r["source"] == "37a"] == ["N-04", "N-22", "N-23", "N-31", "N-29"]


@pytest.mark.parametrize("case", CASES, ids=[f"{c['source']}/{c['id']}" for c in CASES])
def test_반례의_의도(case) -> None:
    task = task_of([tuple(x) for x in case["defects"]])
    got = R.grounded_codes(task, case["text"])
    key = (case["source"], case["id"])
    if key in RESIDUAL:
        assert not _ok(case["intent"], got), (key, "남은 위험이 닫혔다 — residual 을 고친다", got)
        assert got == RESIDUAL[key]["new"], key
    else:
        assert _ok(case["intent"], got), (case["id"], case["intent"], got)
    if case["source"] in ("35a", "37a"):
        assert R.contract_codes(task, case["text"]) == [], case["id"]


def test_모듈_꼭대기의_정의는_겹치지_않는다() -> None:
    """같은 이름의 함수가 둘이면 뒤의 것이 앞의 것을 덮는다 — 결함 행의 절 분리가 정상 행의 규칙으로 덮인 I-02 의 뿌리다."""
    tree = ast.parse((ROOT / "evaluation/grounded_v7_ref.py").read_text(encoding="utf-8"))
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    names += [t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)]
    names += [n.target.id for n in tree.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
    assert [k for k, v in Counter(names).items() if v > 1] == []


def test_결함_행과_정상_행의_절_분리는_다른_함수다() -> None:
    assert not hasattr(R, "_clauses")
    assert "_defect_clause_spans" in R.defect_grounded.__code__.co_names
    assert "_normal_clauses" not in R.defect_grounded.__code__.co_names
    assert "_normal_clauses" in R.normal_negative.__code__.co_names
    assert "_defect_clause_spans" not in R.normal_negative.__code__.co_names
    s = "균열은 없다, 아래쪽 오른쪽에 기공 1개가 표시되어 있다"
    assert [s[a:b] for a, b in R._defect_clause_spans(s)] == ["균열은 없다", "아래쪽 오른쪽에 기공 1개가 표시되어 있다"]
    assert R._normal_clauses(s) == [s]                              # 정상 행의 절은 쉼표에서 나누지 않는다(그대로)


def test_바탕_경계는_고치기_전_결함_행이_쓰던_경계다() -> None:
    """고치기 전에는 정상 행 규칙이 결함 행 규칙을 덮어, 결함 행도 실제로는 `_CLAUSE_SPLIT` 으로 나뉘었다. 그 경계는 늘 나눈다."""
    assert R._BASE_CUT.pattern == R._CLAUSE_SPLIT.pattern


@pytest.mark.parametrize("text, pieces", [
    ("위쪽 왼쪽, 아래쪽 오른쪽에는 균열이 없다", 1),                    # 낱말 나열의 쉼표
    ("균열, 기공은 없다", 1),
    ("균열이 있다고 볼 근거는 없다", 1),                                # 인용
    ("균열하고 기공은 없다", 1),
    ("균열이며 기공이며 모두 없다", 1),
    ("균열이며, 기공이며 모두 보이지 않는다", 1),
    ("결함은 기공이고, 균열은 없다", 1),                                # 계사에서는 나누지 않는다
    ("균열은 위쪽 왼쪽에 있고 아래쪽 오른쪽에는 없다", 1),              # 긍정 서술 뒤 이음에서는 나누지 않는다
    ("균열 1개가 위쪽 왼쪽에 있고 아래쪽 오른쪽에는 없다", 1),
    ("기공 1개가 보이며 균열은 없다", 1),
    ("기공 1개가 표시되어 있으므로 균열은 없다", 1),
    ("기공 1개가 보이나 균열은 보이지 않는다", 1),
    ("기공 1개가 있는 반면, 균열은 없다", 1),
    ("균열은 위쪽 왼쪽에 있고, 모양과 크기는 주석만으로 알 수 없다", 1),
    ("균열은 관찰되고 있지 않다", 1),
    ("균열 1개가 있다, 기공과 이어져 있는지는 알 수 없다", 1),          # 오른쪽 이름이 주어가 아니다
    ("균열 없이 기공 1개만 위쪽 왼쪽 칸에 있다", 1),                    # 쉼표 없는 '없이' — 주절의 자리가 범위다
    ("균열이 없는 대신 기공 1개가 위쪽 왼쪽 칸에 있다", 1),              # 쉼표 없는 '대신'
    ("결함은 균열이 아니고 기공 1개이다", 1),                           # 정체 부정은 경계가 아니다
    ("균열은 없다, 크기는 알 수 없다", 1),                              # 오른쪽에 이름이 없다
    ("균열은 없고 기공 1개가 있다", 2),                                 # 바탕 경계
    ("균열은 없다, 아래쪽 오른쪽에 기공 1개가 표시되어 있다", 2),       # 완결된 '다,' · 오른쪽에 주어 이름
    ("아래쪽 오른쪽에 기공 1개가 표시되어 있다, 균열은 없다", 2),
    ("위쪽 왼쪽에는 기공 1개가 있다, 균열은 없다", 2),                  # 나누되, 처소 주제는 범위로 이어진다(아래 범위 시험)
    ("기공 1개가 위쪽 왼쪽에 있다, 같은 칸에는 균열이 없다", 2),
    ("균열은 없으며, 기공 1개가 있다", 2),                              # 부정 서술 뒤 이음
    ("균열은 보이지 않았으며, 기공 1개가 있다", 2),
    ("균열 없이, 기공 1개가 있다", 2),
    ("균열은 보이지 않으므로, 아래쪽 오른쪽의 기공 1개만 표시되어 있다", 2),  # 까닭 이음
    ("균열은 보이지 않는 반면, 기공 1개가 있다", 2),
], ids=["list-zones", "list-names", "quote-dago", "list-hago", "name-imyeo", "name-imyeo-comma", "copula", "pos-go-bare",
        "pos-go-counted", "pos-boimyeo", "pos-eumeuro", "pos-boina", "pos-banmyeon", "pos-attr-rest", "progressive",
        "right-not-subject", "eopi-no-comma", "daesin-no-comma", "identity", "right-nameless", "base-eopgo", "neg-da", "pos-da",
        "topic-da", "anaphor-da", "neg-eumyeo-comma", "neg-past", "neg-eopi", "neg-eumeuro", "neg-banmyeon"])
def test_결함_행의_절_경계(text, pieces) -> None:
    assert len(R._defect_clause_spans(text)) == pieces, [text[a:b] for a, b in R._defect_clause_spans(text)]


@pytest.mark.parametrize("tail, unknown", [
    ("의 모양은 주석만으로 알 수 없다", True),
    ("의 길이와 폭은 영상만으로 알 수 없다", True),
    ("의 모양, 크기, 길이는 알 수 없다", True),
    ("은 모양과 크기를 알 수 없다", True),
    ("이 얼마나 깊은지는 주석에 나와 있지 않다", True),
    ("이 얼마나 넓게 벌어져 있는지는 주석만으로 알 수 없다", True),
    ("의 크기는 확인되지 않는다", True),
    ("의 크기는 주석에 표시되어 있지 않다", True),
    ("의 크기 정보는 주석에 포함되어 있지 않다", True),
    ("에 대한 크기 정보는 주석에 없다", True),
    ("과 균열의 모양과 크기는 주석만으로 알 수 없다", True),
    ("의 길이와 기공의 크기는 알 수 없다", True),
    (" 사이의 간격은 알 수 없다", True),
    ("이 어떤 모양인지는 주석만으로 알 수 없다", True),
    ("이 어느 방향으로 뻗어 있는지는 알 수 없다", True),
    ("은 그 크기를 주석만으로 알 수 없다", True),
    ("은 확인할 수 없다", False),                                       # 존재를 모른다 — 부재 주장
    ("이 있는지는 알 수 없다", False),
    ("의 위치는 알 수 없다", False),
    ("이 어느 쪽에 있는지는 알 수 없다", False),
    ("의 개수는 알 수 없다", False),
    ("의 분포는 주석만으로 알 수 없다", False),
    ("의 모양과 위치는 주석만으로 알 수 없다", False),                  # 나열 끝이 속성이 아니다
    ("의 크기는 물론 존재 여부도 확인할 수 없다", False),               # 닫힌 목록 밖의 말이 낀다
    ("은 크기와 상관없이 확인할 수 없다", False),
    ("의 모양만으로는 진짜인지 알 수 없다", False),
    ("은 끝내 확인되지 않는다", False),                                 # '끝내' 는 '끝' 이 아니다
    ("의 모습은 영상에서 드러나지 않는다", False),
    ("의 모양을 알 수 없을 뿐 기공은 없다", False),                     # 절 끝까지가 아니다
    ("의 흔적은 없다", False),
    ("의 모양은 없다", False),                                          # 앎의 부정이 아니다
], ids=["shape", "list-wa", "list-comma", "topic", "eolmana", "eolmana-wide", "hwagin-doeji", "pyosi-anh",
        "info-pohamdoe", "e-daehan", "name-list", "name-attr-pairs", "sai", "eotteon", "eoneu-banghyang", "geu",
        "exist", "whether", "location", "eoneu-jjok", "count", "distribution", "list-ends-location", "mullon",
        "sanggwaneopsi", "jinjja", "kkeutnae", "moseup", "not-clause-end", "trace", "plain-eopda"])
def test_속성_판독_불가의_경계(tail, unknown) -> None:
    assert bool(R.ATTR_UNKNOWN.match(tail)) is unknown, tail


def test_속성_판독_불가는_이름_없는_부정이나_띠_소견이_같은_문장에_있으면_받지_않는다() -> None:
    """속성 판독 불가를 받은 문장의 다른 절에 이름 없는 부정이나 띠 · 높이 소견이 있으면, 문장 전체를 고치기 전 절로 읽는다 —
    뒤 절의 주어가 앞 이름(생략)일 수 있고, 띠 대조는 '없' 이 든 문장을 묻지 않는다. 같은 문장의 다른 절이 속성 판독 불가뿐이면 받는다."""
    task = task_of([("균열", "위 왼쪽")])
    head = "위쪽 왼쪽에 균열 1개가 표시되어 있다. "
    assert "absence_contradicts_locked" in R.grounded_codes(task, head + "균열은 크기를 알 수 없고 위치도 확인할 수 없다. " + L)
    assert R.grounded_codes(task, head + "균열은 크기를 알 수 없고 길이도 알 수 없다. " + L) == []
    # 띠 소견이 있으면 고치기 전 절(한 절)로 읽는다 — 부재 범위가 그 띠라서 잠긴 칸과 겹치지 않으면 지난다(고치기 전과 같다)
    two = task_of([("균열", "위 왼쪽"), ("기공", "아래 가운데")])
    text = "위쪽 왼쪽에 균열 1개, 아래쪽 가운데에 기공 1개가 표시되어 있다. 균열은 크기를 알 수 없으며 기공은 아래쪽 띠에 있다. " + L
    assert R.grounded_codes(two, text) == []
    # 띠 소견이 거짓이면 고치기 전처럼 기각한다(속성 판독 불가를 받으면 '없' 때문에 띠 대조가 건너뛰어져 지나간다)
    low = task_of([("균열", "아래 오른쪽")])
    false_band = "아래쪽 오른쪽에 균열 1개가 표시되어 있다. 균열의 모양은 알 수 없으나 균열은 위쪽 띠를 따라 이어져 있다. " + L
    assert R.grounded_codes(low, false_band) != []


@pytest.mark.parametrize("locked, text, want", [
    # 닫힌 부재 술어 뒤, 제 주어를 가진 새 절의 구역은 앞 부재의 범위가 아니다
    ([("균열", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "균열은 없다, 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 위쪽 왼쪽에 균열 1개가 표시되어 있다.", ["absence_contradicts_locked"]),
    ([("균열", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "균열은 보이지 않았으며, 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 위쪽 왼쪽에 균열 1개가 표시되어 있다.", ["absence_contradicts_locked"]),
    # 정형 부재 주장은 앞 절의 구역을 물려받지 않는다 — 앞 구역이 긍정 서술의 자리(bare) · 전역어(global) · 제 구역(zoned)
    ([("균열", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "아래쪽 오른쪽에 기공 1개가 표시되어 있다, 균열은 없다. 위쪽 왼쪽에 균열 1개가 표시되어 있다.", ["absence_contradicts_locked"]),
    ([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")],
     "위쪽 왼쪽에는 기공 1개가 있다, 균열은 어디에도 없다. 아래쪽 오른쪽에 균열 1개가 표시되어 있다.", ["absence_contradicts_locked"]),
    ([("균열", "위 왼쪽")], "위쪽 왼쪽에 균열 1개가 표시되어 있다, 아래쪽 오른쪽에는 균열이 없다.", None),
    ([("기공", "위 왼쪽")], "기공 1개가 위쪽 왼쪽에 표시되어 있다, 아래쪽 오른쪽에는 균열이 표시되어 있지 않다.", None),
    # 처소 주제 꼴('…엔 · 칸에서')의 긍정 서술 뒤 쉼표로 이은 맨 부재도 영상 전체의 부재다(판단 2 — 거짓 통과가 거짓 기각보다 비싸다)
    ([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")],
     "위쪽 왼쪽에 기공 1개, 아래쪽 오른쪽에 균열 1개가 표시되어 있다. 위쪽 왼쪽엔 기공 1개가 보인다, 균열은 보이지 않는다.",
     ["absence_contradicts_locked"]),
    ([("기공", "위 왼쪽"), ("기공", "위 왼쪽"), ("균열", "아래 오른쪽")],
     "위쪽 왼쪽에 기공 2개, 아래쪽 오른쪽에 균열 1개가 표시되어 있다. 위쪽 왼쪽 칸에서 기공 2개가 확인된다, 균열은 확인되지 않는다.",
     ["absence_contradicts_locked"]),
    # 그 밖은 고치기 전처럼 앞 구역을 물려받는다 — '도' · 되짚는 말 · 정체 부정 · 덧붙임 '에도'(첫 문장이 잠긴 칸을 서술해 칸 배정과
    # 섞이지 않게 한다)
    ([("균열", "아래 오른쪽"), ("기공", "아래 오른쪽")],
     "위쪽 왼쪽에 균열은 없으며, 기공도 없다. 아래쪽 오른쪽에 균열 1개와 기공 1개가 표시되어 있다.", None),
    ([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")],
     "위쪽 왼쪽에 기공 1개가 보인다, 거기에 균열은 없다. 아래쪽 오른쪽에 균열 1개가 표시되어 있다.", None),
    ([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")],
     "위쪽 왼쪽에 기공 1개, 아래쪽 오른쪽에 균열 1개가 표시되어 있다. 위쪽 왼쪽에 표시된 것은 기공 1개다, 균열이 아니다.", None),
    ([("기공", "위 왼쪽"), ("균열", "위 왼쪽")],
     "위쪽 왼쪽에 기공 1개가 있다, 균열은 해당 칸에도 아래쪽 오른쪽에도 없다. 위쪽 왼쪽에 균열 1개가 표시되어 있다.",
     ["absence_contradicts_locked"]),
    # 칸 떼기는 정형 부재 토막만 — 관형절이 꾸미는 칸 · 의심 술어의 칸은 고치기 전처럼 앞 이름에 붙는다
    ([("기공", "위 왼쪽")], "기공 1개가 표시되어 있다. 기공 1개가 있는 위쪽 왼쪽 칸에는 균열이 없다.", None),
    ([("기공", "위 왼쪽"), ("기공", "위 왼쪽")],
     "기공 2개는 위쪽 왼쪽 칸에 있지만, 위쪽 가운데 칸에는 기공이 더 있는지 확실하지 않다.", []),
    # 부재 자리도 앞 이름 자리로 센다 — 생략된 주어의 칸 · 수는 그 이름에(고치기 전과 같다)
    ([("균열", "아래 오른쪽"), ("기공", "아래 왼쪽")],
     "균열 1개와 기공 1개가 표시되어 있는데, 균열은 위쪽에는 없고 아래쪽 오른쪽에 있다. 기공 1개는 아래쪽 왼쪽에 있다.", None),
    # 인용 · 의문 앞의 '1개도 없' 은 부재가 아니다
    ([("균열", "위 왼쪽")], "위쪽 왼쪽에 균열 1개가 표시되어 있다. 균열이 1개도 없다고 단정할 수는 없다.", None),
    # 부재라고 밝힌 이름의 속성 판독 불가는 그 이름의 언급이 아니다 — 밝히지 않은 이름이면 언급이다(이름 불일치)
    ([("기공", "위 왼쪽")], "위쪽 왼쪽에 기공 1개가 표시되어 있다. 균열은 표시되어 있지 않다. 따라서 균열의 모양도 알 수 없다.", None),
    ([("기공", "위 왼쪽")], "위쪽 왼쪽에 기공 1개가 표시되어 있다. 균열의 모양은 주석만으로 알 수 없다.", ["names_mismatch"]),
    # 다음 토막이 앞 부재를 닫지 않으면 범위를 잇는다 — 관형절 안의 주어 · 부정이 든 토막 · 범위를 뒤로 미룬 꼴 · 되받는 말 · 예외어
    ([("기공", "아래 오른쪽"), ("균열", "아래 오른쪽")],
     "아래쪽 오른쪽 칸에 기공 1개와 균열 1개가 표시되어 있다. 균열은 위쪽 왼쪽 칸에 보이지 않으며 기공 1개가 있는 아래쪽 오른쪽 칸에서도 "
     "보이지 않는다.", ["absence_contradicts_locked"]),
    ([("기공", "아래 오른쪽"), ("균열", "아래 오른쪽")],
     "아래쪽 오른쪽 칸에 기공 1개와 균열 1개가 표시되어 있다. 균열은 위쪽 왼쪽 칸에 없다, 기공 1개가 표시된 아래쪽 오른쪽 칸도 마찬가지다.",
     ["absence_contradicts_locked"]),
    ([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")],
     "위쪽 왼쪽에 기공 1개가 있고 아래쪽 오른쪽에 균열 1개가 있다. 균열은 보이지 않는다, 기공 1개가 있는 위쪽 왼쪽 칸에는.", None),
    ([("기공", "위 왼쪽"), ("균열", "위 왼쪽")],
     "위쪽 왼쪽 칸에 기공 1개와 균열 1개가 표시되어 있다. 균열은 보이지 않는다, 기공 1개가 함께 있는 위쪽 왼쪽 칸을 제외하면.", None),
    # 틀을 꾸미는 'ZONE에 표시된 …' 은 긍정 서술의 자리가 아니다 — 'ZONE에 NAME이 표시되어 있다' 는 자리다
    ([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")],
     "위쪽 왼쪽에 기공 1개가 있고 아래쪽 오른쪽에 균열 1개가 있다. 위쪽 왼쪽에 표시된 결함만 보면 기공 1개가 있다, 균열은 없다.", None),
    ([("기공", "위 오른쪽"), ("균열", "가운데")],
     "기공 1개와 균열 1개가 표시되어 있다. 기공 1개는 위쪽 오른쪽에, 균열 1개는 영상 가운데에 있다. 위쪽 오른쪽에 기공이 표시되어 있다, "
     "균열은 표시되어 있지 않다.", ["absence_contradicts_locked"]),
    # 주어로 쓴 이름은 제 토막에서, 비교 · 오인 언급은 바탕 절에서 뒤의 부정을 본다
    ([("기공", "위 왼쪽")], "위쪽 왼쪽 칸에 기공 1개가 표시되어 있다. 겉보기에는 슬래그혼입처럼 보인다, 실제로는 기공 1개이고 슬래그혼입은 없다.",
     None),
    ([("용입불량", "가운데 왼쪽"), ("용입불량", "가운데 오른쪽")],
     "용입불량 2개가 표시되어 있다. 용입불량은 가운데 왼쪽에 1개, 가운데 오른쪽에 1개가 있다, 영상 가운데에는 용입불량이 표시되어 있지 않다.",
     None),
    # '1개도 없' 은 평서로 끝날 때만 부재 — 이중 부정은 아니다. '1개도 표시되지 않았다' 는 부재다
    ([("균열", "위 왼쪽")], "위쪽 왼쪽 칸에 균열 1개가 표시되어 있다. 주석상 균열이 1개도 없는 것은 아니다, 위쪽 왼쪽 칸에 1개가 있다.", None),
    ([("균열", "아래 오른쪽")], "아래쪽 오른쪽 칸에 균열 1개가 표시되어 있다. 그러나 영상 전체에서 균열은 1개도 표시되지 않았다.",
     ["absence_contradicts_locked"]),
    # 칸을 한정한 부재는 그 이름이 없다는 말이 아니다 — 주석 밖 이름의 속성 판독 불가는 그대로 언급이다
    ([("기공", "위 왼쪽")], "위쪽 왼쪽 칸에 기공 1개가 표시되어 있다. 아래쪽 오른쪽 칸에는 균열이 없다. 균열의 길이는 주석만으로 알 수 없다.",
     ["names_mismatch"]),
    # '…칸에는 기공 1개만 있다' 는 그 칸에 다른 결함이 없다는 말이다 — 칸을 한정한 부재의 범위를 잇는다. 맨 부재는 닫는다
    ([("기공", "아래 오른쪽"), ("균열", "아래 오른쪽")],
     "위쪽 왼쪽 칸에는 균열이 없다, 아래쪽 오른쪽 칸에는 기공 1개만 있다. 아래쪽 오른쪽에 균열 1개가 있다.", ["absence_contradicts_locked"]),
    ([("기공", "아래 오른쪽"), ("균열", "위 왼쪽")],
     "균열은 없다, 아래쪽 오른쪽 칸에는 기공 1개만 있다. 위쪽 왼쪽에 균열 1개가 있다.", ["absence_contradicts_locked"]),
    # '보이지는 않는다' 는 부재 술어다(판단 2 · 검수 K-5) — 전칭어가 걸린 부분 부정('모든 칸에서 보이지는 않는다')만 정형 부재가 아니다
    ([("균열", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "아래쪽 오른쪽에 기공 1개가 표시되어 있다, 균열은 보이지는 않는다. 위쪽 왼쪽에 균열 1개가 표시되어 있다.", ["absence_contradicts_locked"]),
    ([("균열", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "균열은 모든 칸에서 보이지는 않는다, 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 위쪽 왼쪽에 균열 1개가 표시되어 있다.", None),
    # 속성 판독 불가로만 언급한 이름은 앞 칸의 이름 목록 가운데에 끼지 않는다 — 칸 바로 뒤 · 칸이 뒤에 오는 꼴은 그 이름의 칸이다
    ([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")],
     "위쪽 왼쪽 칸에는 기공 1개가 있고 균열의 길이는 알 수 없다. 아래쪽 오른쪽 칸에 균열 1개가 있다.", None),
    ([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")],
     "위쪽 왼쪽 칸의 균열은 길이를 알 수 없다. 위쪽 왼쪽 칸에 기공 1개, 아래쪽 오른쪽 칸에 균열 1개가 있다.", ["zone_outside_locked"]),
    ([("균열", "위 왼쪽")], "균열 1개가 표시되어 있다. 균열의 모양은 알 수 없지만 위쪽 왼쪽에 있다.", None),
    # 부재라고 밝힌 이름의 속성 판독 불가는 다음 문장의 앞 이름이 아니다 — 밝히지 않은 이름이면 앞 이름이다
    ([("기공", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "기공 2개가 표시되어 있다. 균열은 없어서 균열의 길이도 알 수 없다. 위쪽 왼쪽에 1개, 아래쪽 오른쪽에 1개가 있다.", None),
    ([("균열", "위 왼쪽")], "균열의 크기는 알 수 없다. 위쪽 왼쪽에 1개가 있다.", None),
    # 닫힌 뒤 토막의 예외어('…를 제외하면')는 앞 부재를 면제하지 않는다 — 예외어는 범위 끝까지에서 본다
    ([("균열", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "균열은 없다, 아래쪽 오른쪽에 기공 1개가 있다, 기공 1개를 제외하면 결함은 없다. 위쪽 왼쪽에 균열 1개가 있다.",
     ["absence_contradicts_locked"]),
    # 의심 · 정체 부정 · 덧붙임 '도' 는 정형 부재 주장이 아니다 — 고치기 전처럼 앞 구역을 물려받는다
    ([("균열", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "아래쪽 오른쪽에 기공 1개가 표시되어 있다, 균열은 알 수 없다. 위쪽 왼쪽에 균열 1개가 표시되어 있다.", None),
    ([("균열", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "아래쪽 오른쪽에 기공 1개가 표시되어 있다, 균열은 아니다. 위쪽 왼쪽에 균열 1개가 표시되어 있다.", None),
    ([("기공", "위 왼쪽"), ("기공", "아래 오른쪽"), ("균열", "위 가운데")],
     "위쪽 왼쪽 칸에 기공 1개가 있고 균열이 없다. 아래쪽 오른쪽 칸에 기공 1개가 있다, 균열도 없다. 위쪽 가운데에 균열 1개가 있다.", None),
    # 비교 언급('슬래그혼입처럼')은 뒤 토막의 정형 부재까지 바탕 절에서 본다 — 제 토막만 보면 긍정 언급이 된다
    ([("균열", "위 왼쪽")], "위쪽 왼쪽에 균열 1개가 표시되어 있다. 겉보기에는 슬래그혼입처럼 보인다, 기공은 없다.", None),
    # 새 경계의 주어 표지 '도'(검수 K-6) — '…, 기공도 1개 …에 있다' 의 오른쪽 토막은 제 주어를 가진다
    ([("기공", "위 왼쪽")], "균열의 길이는 알 수 없다, 기공도 1개 위쪽 왼쪽에 있다.", ["names_mismatch"]),
    ([("균열", "아래 오른쪽")], "균열의 길이는 알 수 없다, 균열도 아래쪽 오른쪽에 1개 있다.", None),
    # 맨 부재는 뒤의 새 절('…칸에는 슬래그혼입 1개뿐이다')의 칸으로 좁아지지 않는다 — 한정하는 말이면 좁아진다(검수 K-1)
    ([("기공", "위 왼쪽"), ("슬래그혼입", "아래 오른쪽"), ("균열", "위 가운데")],
     "위쪽 왼쪽 칸에는 기공 1개가 있다, 균열은 없다, 아래쪽 오른쪽 칸에는 슬래그혼입 1개뿐이다. 위쪽 가운데에 균열 1개가 있다.",
     ["absence_contradicts_locked"]),
    ([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")],
     "위쪽 왼쪽에 기공 1개가 있고 아래쪽 오른쪽에 균열 1개가 있다. 균열은 없다, 기공 1개가 있는 위쪽 왼쪽 칸에 한한 이야기다.", None),
    # '…칸에는 균열 1개만 있다' 가 앞 부재의 이름을 긍정하면 범위를 그 칸까지 잇지 않는다(검수 I-2)
    ([("균열", "아래 오른쪽")], "위쪽 왼쪽 칸에는 균열이 없다, 아래쪽 오른쪽 칸에는 균열 1개만 있다.", None),
    # 엄격한 긍정 서술 틀의 앞 이음('또한') · 술어 앞 부사 · 과거형 — 닫음과 앞 구역 판정이 모두 받는다
    ([("균열", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "균열은 없다, 또한 아래쪽 오른쪽에 기공 1개가 있다. 위쪽 왼쪽에 균열 1개가 있다.", ["absence_contradicts_locked"]),
    ([("균열", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "아래쪽 오른쪽에 기공 1개가 따로 있다, 균열은 없다. 위쪽 왼쪽에 균열 1개가 있다.", ["absence_contradicts_locked"]),
    ([("균열", "위 왼쪽"), ("기공", "아래 오른쪽")],
     "아래쪽 오른쪽에 기공 1개가 있었다, 균열은 없었다. 위쪽 왼쪽에 균열 1개가 있었다.", ["absence_contradicts_locked"]),
], ids=["back-da", "back-past", "fwd-bare-plain", "fwd-global", "fwd-zoned", "fwd-zoned-pyosi", "inherit-en", "inherit-eseo",
        "inherit-do", "inherit-geogi", "inherit-identity", "inherit-edo", "relative-zone", "doubt-zone", "elided-subject",
        "quote-none", "declared-absent-attr", "presupposition", "open-relative-neg", "open-echo", "open-dislocated",
        "open-exempt", "frame-not-plain", "plain-uncounted", "compare-mention", "asserted-piece", "none-double-neg",
        "none-pyosi", "zoned-not-declared", "only-zoned-continues", "only-bare-closes", "partial-plain", "partial-universal",
        "attr-zone-list-middle", "attr-zone-head", "attr-zone-after", "declared-absent-not-antecedent", "attr-antecedent",
        "exempt-closed-piece", "doubt-not-canonical", "identity-not-canonical", "inherit-do-parallel", "compare-own-piece",
        "subject-do-mention", "subject-do-pass", "bare-not-narrowed", "bare-limited", "only-same-name", "close-connector",
        "plain-adverb", "plain-past"])
def test_부재_주장의_범위(locked, text, want) -> None:
    """의도 None → 통과, 코드 목록 → 그 코드가 모두, 빈 목록 → 기각."""
    assert _ok(want, R.grounded_codes(task_of(locked), text + " " + L)), (text, R.grounded_codes(task_of(locked), text + " " + L))


@pytest.mark.parametrize("locked, text, want", [
    # 앞 칸의 수('위쪽 왼쪽에 기공 2개가 있고')는 총수가 아니라 그 칸의 수다 — 배분 수는 바로 앞 칸들로만 펼친다
    ([("기공", "위 왼쪽")] * 3 + [("기공", "아래 오른쪽"), ("기공", "가운데")],
     "기공 5개가 표시되어 있다. 위쪽 왼쪽에 기공 2개가 있고, 아래쪽 오른쪽과 영상 가운데에는 기공이 각각 1개씩 있다.", ["zone_count_mismatch"]),
    ([("기공", "위 왼쪽")] * 2 + [("기공", "아래 오른쪽"), ("기공", "가운데")],
     "기공 4개가 표시되어 있다. 위쪽 왼쪽에 기공 2개가 있고, 아래쪽 오른쪽과 영상 가운데에는 기공이 각각 1개씩 있다.", None),
    # '…칸은 비어 있다' 의 칸에는 배분 수를 펼치지 않는다
    ([("기공", "위 왼쪽"), ("기공", "아래 오른쪽"), ("기공", "가운데")],
     "기공 3개가 표시되어 있다. 위쪽 왼쪽과 아래쪽 오른쪽에 각각 1개씩 있고, 영상 가운데 칸은 비어 있다.", []),
    # 배분 수가 둘인 꼴은 펼치지 않는다 — 앞 배분 수를 '칸 수와 같은 수' 로 하나씩 읽어 거짓 글이 지나가지 않게
    ([("기공", "위 왼쪽"), ("기공", "위 오른쪽"), ("기공", "아래 왼쪽"), ("기공", "아래 오른쪽")],
     "위쪽 왼쪽과 위쪽 오른쪽에 기공이 각각 2개씩, 아래쪽 왼쪽과 아래쪽 오른쪽에 기공이 각각 1개씩 있다.", []),
], ids=["zone-count-before-each", "zone-count-before-each-ok", "empty-cell", "two-each-counts"])
def test_배분_수는_바로_앞_칸들로만_펼친다(locked, text, want) -> None:
    """검수 K-2 — 펼친 수는 묶음 합으로 떨어지지 않는다."""
    assert _ok(want, R.grounded_codes(task_of(locked), text + " " + L)), R.grounded_codes(task_of(locked), text + " " + L)


def test_공개_안전_실물_식별자가_없다() -> None:
    """반례 사본 · 개발 기록 원본 · 벡터 · 배분 사례에 데이터셋 유래 식별자가 없다(검수 M-2) — 'aihub' 부분 문자열, 홀로 선 6 ~ 10
    자리 수(날짜 꼴 20xxxxxx 제외), 그리고 회수본 '각각' 행 목록(있는 트리에서) 식별자의 숫자 부분."""
    import re

    files = [CX, ROOT / "evaluation/vectors/grounded_v7_vectors.jsonl", ROOT / "evaluation/vectors/grounded_v7_each_cases.json"]
    files += [SOURCES / s["name"] for s in DOC["sources"] if (SOURCES / s["name"]).exists()]
    texts = {f.name: f.read_text(encoding="utf-8") for f in files}
    for name, t in texts.items():
        assert "aihub" not in t.lower(), name
        loose = [x for x in re.findall(r"(?<![0-9A-Za-z])\d{6,10}(?![0-9A-Za-z])", t) if not re.fullmatch(r"20\d{6}", x)]
        assert loose == [], (name, loose[:3])
    rows = SOURCES / "40a_각각N개씩_행목록_B.jsonl"
    if rows.exists():
        # 데이터셋 유래 식별자(`id`)가 있는 판만 맞댄다 — 샤드 · 행 키(`row_key`)로 바꾼 판에는 맞댈 식별자가 없다
        ids = {d for line in rows.read_text(encoding="utf-8").splitlines() if line.strip()
               for d in re.findall(r"\d{5,}", json.loads(line).get("id", ""))}
        for name, t in texts.items():
            assert not [d for d in ids if d in t], name


def test_배분은_마지막_수만_칸마다로_펼친다() -> None:
    """'A 와 B 칸에 기공 1개씩 있고, B에는 1개가 더 있다' 의 '1개가 더' 는 배분 수가 아니다 — 배분 표지가 붙은 수가 마지막이 아니면
    펼치지 않는다(펼치면 칸 묶음의 합 3이 잠긴 위쪽 왼쪽 2 · 아래쪽 오른쪽 1과 우연히 맞아 거짓 글이 지나간다)."""
    task = task_of([("기공", "위 왼쪽"), ("기공", "위 왼쪽"), ("기공", "아래 오른쪽")])
    text = "위쪽 왼쪽과 아래쪽 오른쪽 칸에 기공 1개씩 있고, 아래쪽 오른쪽에는 1개가 더 있다. " + L
    assert "zone_count_mismatch" in R.grounded_codes(task, text)


def test_칸_앞에_부정이_있으면_부재의_범위로_보지_않는다() -> None:
    """'균열과 멀지 않은 위쪽 가운데' 의 칸은 앞의 기공에 붙는다 — 부정('않')이 이름과 칸 사이에 있으면 'NAME은 ZONE에 없다' 꼴이 아니다."""
    task = task_of([("기공", "위 가운데")])
    assert R.grounded_codes(task, "기공 1개는 균열과 멀지 않은 위쪽 가운데에 있고 슬래그혼입은 없다. " + L) == []


def test_예외어_절은_속성_판독_불가를_되돌리지_않는다() -> None:
    """'다른 결함은 없다' 는 잠긴 이름 밖의 부재라서, 같은 문장의 속성 판독 불가를 고치기 전 절로 되돌리는 까닭이 아니다."""
    task = task_of([("균열", "위 왼쪽")])
    head = "위쪽 왼쪽에 균열 1개가 표시되어 있다. "
    assert R.grounded_codes(task, head + "균열은 크기를 알 수 없고 다른 결함은 없다. " + L) == []


def test_노출의_절제본은_지금_정본이_다시_낸다() -> None:
    """exposed 의 근거 가운데 지금 정본 몫(속성 판독 불가 부분을 뺀 글의 결과)은 여기서 다시 돌린다 — 고치기 전 정본 몫(old_without)은
    생성기가 고치기 전 바이트(07ce544f)로 돌린 기록이다."""
    cases = {(c["source"], c["id"]): c for c in CASES}
    exposed = [r for r in RESIDUAL.values() if r["old_ok"]]
    assert exposed
    for r in exposed:
        c, e = cases[(r["source"], r["id"])], r["exposed"]
        assert c["text"].count(e["dropped"]) == 1, r["key"]
        rest = c["text"].replace(e["dropped"], e["replacement"])
        assert R.grounded_codes(task_of([tuple(x) for x in c["defects"]]), rest) == e["new_without"] == r["new"], r["key"]


def test_이식본은_정본과_같은_바이트다() -> None:
    """이식본(`corpus/production/judge_grounded_v7.py`)은 정본의 바이트 사본이다 — 반례의 의도도 이식본에서 그대로 선다."""
    from corpus.production import judge_grounded_v7 as D

    assert Path(D.__file__).read_bytes() == (ROOT / "evaluation/grounded_v7_ref.py").read_bytes()
    for case in CASES:
        task = task_of([tuple(x) for x in case["defects"]])
        key = f"{case['source']}/{case['id']}"
        assert D.grounded_codes(task, case["text"]) == R.grounded_codes(task, case["text"]), key
