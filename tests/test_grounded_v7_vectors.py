"""판정 v7 결정적 grounded 의 시험 벡터와 동치 하니스(`scripts/probe/check_grounded_v7_equivalence.py`).

벡터는 G150 합성 사례와 구조 사례(구역별 수 · 합 — `scripts/probe/grounded_v7_structural_cases.py`, 실물 id 없음)에서 평가 쪽 참조
구현(`evaluation/grounded_v7_ref.py`)으로 만든 것이다. 정식 구현이 같은 벡터를 다시 내야 한다(총괄 계약 5).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import grounded_v7_ref as R  # noqa: E402
from scripts.probe import check_grounded_v7_equivalence as H  # noqa: E402
from scripts.probe import grounded_v7_structural_cases as S  # noqa: E402

VEC = ROOT / "evaluation/vectors/grounded_v7_vectors.jsonl"
STRUCT = ROOT / "evaluation/vectors/grounded_v7_structural_cases.json"
N_ROWS = 150 + len(S.CASES) + len(S.NORMAL_CASES)


def test_벡터_파일의_머리와_행() -> None:
    meta, rows = H.read_vectors(VEC)
    assert meta["schema"] == H.SCHEMA and meta["n_rows"] == len(rows) == N_ROWS
    assert meta["fields"] == list(H.FIELDS)
    assert {r["group"] for r in rows} == {"N+", "D+", "ADV1", "ADV2", "STRUCT+", "STRUCT-", "NORM+", "NORM-"}
    assert len({r["id"] for r in rows}) == len(rows)
    assert all("aihub" not in json.dumps(r, ensure_ascii=False) for r in rows), "실물 id 를 싣지 않는다"


def test_참조_구현이_벡터를_다시_낸다() -> None:
    res = H.check(VEC, None, None, {})
    assert res["equal"] and res["n_rows"] == N_ROWS and res["mismatches"] == []


def test_다른_출력을_내는_후보는_행_단위로_잡힌다(monkeypatch) -> None:
    fake = types.ModuleType("fake_grounded")
    fake.__file__ = str(ROOT / "evaluation/grounded_v7_ref.py")
    fake.grounded_codes = lambda task, text: []
    fake.contract_codes = R.contract_codes
    fake.attribute_codes = R.attribute_codes
    fake.normal_codes = R.normal_codes
    monkeypatch.setitem(sys.modules, "fake_grounded", fake)
    res = H.check(VEC, "fake_grounded", None, {})
    assert not res["equal"]
    assert {m["field"] for m in res["mismatches"]} == {"grounded_codes"}
    assert all(m["expected"] and m["got"] == [] for m in res["mismatches"])


def test_함수_이름이_다르면_대응표로_맞댄다(monkeypatch) -> None:
    fake = types.ModuleType("fake_named")
    fake.__file__ = str(ROOT / "evaluation/grounded_v7_ref.py")
    fake.deterministic_grounded = R.grounded_codes
    fake.contract_codes = R.contract_codes
    fake.attribute_codes = R.attribute_codes
    fake.normal_codes = R.normal_codes
    monkeypatch.setitem(sys.modules, "fake_named", fake)
    with pytest.raises(SystemExit):
        H.check(VEC, "fake_named", None, {})
    assert H.check(VEC, "fake_named", None, {"grounded_codes": "deterministic_grounded"})["equal"]


def test_머리가_다른_파일은_읽지_않는다(tmp_path) -> None:
    bad = tmp_path / "v.jsonl"
    bad.write_text(json.dumps({"meta": {"schema": "other"}}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        H.read_vectors(bad)


def test_결정적_grounded_의_대표_꼴() -> None:
    """G150 에서 고친 추출 꼴 — 고유어 수사 · 쉼표 구역 목록 · 열이 앞서는 가운데 줄 · 속격 · 이어진 구역 묶음 · 띠 소견."""
    _, rows = H.read_vectors(VEC)
    by = {r["id"]: r for r in rows}
    for i in ("G1-DP-06", "G1-DP-11", "G1-DP-13", "G1-DP-04", "G1-DP-49", "G1-DP-31"):
        r = by[i]
        assert R.grounded_codes(r["task"], r["candidate"]) == [], i
    assert R.grounded_codes(by["G2-AN-22"]["task"], by["G2-AN-22"]["candidate"]) != []   # 띠 소견이 잠긴 칸과 어긋남


def test_구조_사례_파일은_생성기의_출력과_같다() -> None:
    cases, bad = S.build()
    assert bad == []
    tracked = json.loads(STRUCT.read_text(encoding="utf-8"))
    assert tracked == json.loads(json.dumps({"kind": "grounded_v7_structural_cases", "cases": cases}, ensure_ascii=False))
    assert len(cases) >= 12 and sum(c["intent"] is None for c in cases) >= 4
    assert sum(1 for c in cases if c.get("intent_field") == "normal_codes") >= 12


@pytest.mark.parametrize("case", S.CASES, ids=[c[0] for c in S.CASES])
def test_구조_사례의_의도(case) -> None:
    """구역별 수는 '이름 × 구역 × 수' 다중집합으로 맞대고, 구역별 합은 총합과 같아야 한다. 수 생략 꼴은 구역 집합 · 총수만 본다."""
    cid, defects, text, intent = case
    task = S.task_of(defects)
    got = R.grounded_codes(task, text)
    if intent is None:
        assert got == [], cid
    else:
        assert intent <= set(got), (cid, got)
    assert R.contract_codes(task, text) == [] and R.attribute_codes(task, text) == []


def test_외부_검토_반례_둘이_닫힌다() -> None:
    by = {c[0]: c for c in S.CASES}
    swapped, hidden = by["S-02"], by["S-04"]
    assert "zone_count_mismatch" in R.grounded_codes(S.task_of(swapped[1]), swapped[2])
    assert {"zone_count_mismatch", "zone_sum_mismatch"} <= set(R.grounded_codes(S.task_of(hidden[1]), hidden[2]))


def test_G150_양성의_결정적_기각은_0() -> None:
    _, rows = H.read_vectors(VEC)
    pos = [r for r in rows if r["group"] in ("N+", "D+")]
    assert len(pos) == 100 and all(r["expected"]["grounded_codes"] == [] for r in pos)


def test_뒤가_앞을_덮지_않는다() -> None:
    """외부 검토 보강 3 의 반례 둘(C-01 · C-05)과 대조(C-07). 명시된 구역별 수는 뒤의 수 생략 문장과 무관하게 대조하고,
    같은 항목의 다른 값은 순서와 무관하게 모순이다."""
    by = {c[0]: c for c in S.CASES}
    for cid, must in (("C-01", {"zone_count_mismatch"}), ("C-05", {"statement_conflict"}), ("C-06", {"statement_conflict"})):
        _, defects, text, _ = by[cid]
        assert must <= set(R.grounded_codes(S.task_of(defects), text)), cid
    _, defects, text, _ = by["C-07"]
    assert R.grounded_codes(S.task_of(defects), text) == []


TWELVE = tuple(f"G1-AN-{i:02d}" for i in (5, 7, 9, 13, 14, 15, 16, 19, 20, 23, 24, 25))
"""외부 검토 §112 의 정상 음성 미검출 12 건(7판 보강 4)."""


def test_정상_음성_열둘을_정상_패턴이_잡는다() -> None:
    _, rows = H.read_vectors(VEC)
    by = {r["id"]: r for r in rows}
    for i in TWELVE:
        assert R.normal_codes(by[i]["task"], by[i]["candidate"]), i


def test_정상_패턴은_양성과_결함_행을_기각하지_않는다() -> None:
    _, rows = H.read_vectors(VEC)
    for r in rows:
        if r["group"] in ("N+", "D+", "NORM+") or r["row_class"] == "defect":
            assert R.normal_codes(r["task"], r["candidate"]) == [], r["id"]


def test_생성_계약의_정상_문장은_정상_패턴을_지난다() -> None:
    from corpus.production import generation_policy as G
    task = S.task_of([])
    for a in G.NORMAL_ABSENCE_FORMS:
        for lim in G.NORMAL_LIMITATION_FORMS:
            assert R.normal_codes(task, a + " " + lim) == [] and R.contract_codes(task, a + " " + lim) == []


@pytest.mark.parametrize("case", S.NORMAL_CASES, ids=[c[0] for c in S.NORMAL_CASES])
def test_정상_변형의_의도(case) -> None:
    cid, text, intent = case
    got = R.normal_codes(S.task_of([]), text)
    if intent is None:
        assert got == [], cid
    else:
        assert intent <= set(got), (cid, got)


SEED_PROBE = (
    "import hashlib, json, sys; sys.path.insert(0, sys.argv[1]); "
    "from evaluation import grounded_v7_ref as R; "
    "from scripts.probe import check_grounded_v7_equivalence as H; "
    "_, rows = H.read_vectors(sys.argv[2]); "
    "out = [[r['id'], H.outputs(R, {}, r['task'], r['candidate'])] for r in rows]; "
    "print(json.dumps({'names': list(R.NAMES), 'out': hashlib.sha256(json.dumps(out, ensure_ascii=False, "
    "sort_keys=True).encode()).hexdigest()}, ensure_ascii=False))"
)
"""하위 프로세스에서 NAMES 와 벡터 전 행의 출력 지문을 낸다 — 한 줄짜리라 줄바꿈 이스케이프가 없다."""


def test_정본은_해시_시드와_무관하다() -> None:
    """PYTHONHASHSEED 0 · 1 · 2 세 프로세스에서 NAMES 튜플과 벡터 전 행의 출력이 같다(7판 보강 5).

    길이만으로 정렬하던 앞 판은 같은 길이 이름의 순서가 시드를 따라 정책 지문이 갈렸다."""
    import os
    import subprocess

    seen = []
    for seed in ("0", "1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"}
        res = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", SEED_PROBE, str(ROOT), str(VEC)],
                             env=env, cwd=str(ROOT), capture_output=True, check=True)
        seen.append(json.loads(res.stdout.decode("utf-8")))
    assert seen[0] == seen[1] == seen[2], seen
    assert seen[0]["names"] == sorted(seen[0]["names"], key=lambda n: (-len(n), n))
    _, rows = H.read_vectors(VEC)
    assert all(H.outputs(R, {}, r["task"], r["candidate"]) == {f: sorted(r["expected"][f]) for f in H.FIELDS}
               for r in rows)


def test_속성_어휘는_낱말_경계를_본다() -> None:
    """'생긴 · 시작은' 은 시각 속성이 아니고 '큰 · 작은 · 긴' 은 낱말 단독일 때만 잡는다(7판 보강 6). G2-AN-15 의 속성 기대값이 [] 가 됐다."""
    task = S.task_of([("기공", "위 가운데")])
    assert R.attribute_codes(task, "위쪽 가운데에 기공 1개가 표시되어 있다. 이 기공은 가스 차폐 불량으로 생긴 것이다. " + S.L) == []
    assert R.attribute_codes(task, "위쪽 가운데에 기공 1개가 표시되어 있다. 시작은 위쪽이다. " + S.L) == []
    assert R.attribute_codes(task, "위쪽 가운데에 큰 기공 1개가 표시되어 있다. " + S.L) == ["unannotated_attribute"]
    assert R.attribute_codes(task, "위쪽 가운데에 기공 1개가 표시되어 있다. 크고 둥근 기공이다. " + S.L) == ["unannotated_attribute"]
    _, rows = H.read_vectors(VEC)
    assert {r["id"]: r for r in rows}["G2-AN-15"]["expected"]["attribute_codes"] == []


def _rewrite_meta(tmp_path, **over) -> Path:
    meta, rows = H.read_vectors(VEC)
    body = [json.dumps({"meta": {**meta, **over}}, ensure_ascii=False, sort_keys=True)]
    body += [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]
    p = tmp_path / "v.jsonl"
    p.write_bytes(("\n".join(body) + "\n").encode("utf-8"))
    return p


def test_하니스는_정본이_바뀐_벡터를_알린다(tmp_path) -> None:
    """핀은 정본 바이트 하나 — 정본이 바뀌면 `stale` 이다. 생성 정책 · 판정 정책 지문은 핀이 아니다(게이트 72)."""
    assert H.PINNED == ("reference_sha256",)
    assert H.check(VEC, None, None, {})["stale"] == []
    res = H.check(_rewrite_meta(tmp_path, reference_sha256="0" * 64), None, None, {})
    assert res["equal"] and res["stale"] == ["reference_sha256"]


def test_의존_정책이_바뀌어도_벡터는_낡지_않는다(tmp_path) -> None:
    """생성 정책이 바뀔 때마다 벡터를 다시 봉인하지 않는다 — 의존 지문은 `deps_changed` 에 기록만 하고 종료 코드를 바꾸지 않는다.
    의존이 판정 동작을 바꾸면 벡터 대조의 불일치로 드러난다."""
    meta, _ = H.read_vectors(VEC)
    assert set(meta["deps_at_build"]) == set(H.DEPS) and meta["pinned"] == list(H.PINNED)
    deps = {k: ("1" * 64 if k == "generation_policy_sha256" else v) for k, v in meta["deps_at_build"].items()}
    res = H.check(_rewrite_meta(tmp_path, deps_at_build=deps), None, None, {})
    # 결합 트리에서는 다른 의존(판정 정책)도 바뀌어 있을 수 있다 — 목록의 길이를 묻지 않는다(게이트 73). 묻는 것은 셋이다:
    # 정본 핀은 바뀐 의존이 아니고, 대조는 같고, 낡지 않았다. 일부러 바꾼 생성 정책 지문은 기록에 보인다
    assert res["equal"] and res["stale"] == []
    assert "reference_sha256" not in res["deps_changed"] and set(res["deps_changed"]) <= set(H.DEPS)
    assert "generation_policy_sha256" in res["deps_changed"]
