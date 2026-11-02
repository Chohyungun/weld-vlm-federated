"""판정 v7 결정적 grounded 의 시험 벡터와 동치 하니스(`scripts/probe/check_grounded_v7_equivalence.py`).

벡터는 G150 합성 사례와 구조 사례(구역별 수 · 합 — `scripts/probe/grounded_v7_structural_cases.py`, 실물 id 없음)에서 평가 쪽 참조
구현(`evaluation/grounded_v7_ref.py`)으로 만든 것이다. 정식 구현이 같은 벡터를 다시 내야 한다(총괄 계약 5).
보강 7 에서 배분 서술 사례(`evaluation/vectors/grounded_v7_each_cases.json` — 합성 EA-xx 와 생성 v2 회수본 감사 행에서 글 · 주석만
옮긴 GV-xx, 실물 id 없음)를 `extend` 로 더했다.
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
EACH_FILE = ROOT / "evaluation/vectors/grounded_v7_each_cases.json"
EACH = json.loads(EACH_FILE.read_text(encoding="utf-8"))["cases"]
GAP_FILE = ROOT / "evaluation/vectors/grounded_v7_gap46_cases.json"
GAP = json.loads(GAP_FILE.read_text(encoding="utf-8"))["cases"]
GAP49_FILE = ROOT / "evaluation/vectors/grounded_v7_gap49_cases.json"
GAP49 = json.loads(GAP49_FILE.read_text(encoding="utf-8"))["cases"]
GAP50_FILE = ROOT / "evaluation/vectors/grounded_v7_gap50_cases.json"
GAP50 = json.loads(GAP50_FILE.read_text(encoding="utf-8"))["cases"]
GAP52_FILE = ROOT / "evaluation/vectors/grounded_v7_gap52_cases.json"
GAP52 = json.loads(GAP52_FILE.read_text(encoding="utf-8"))["cases"]
N_ROWS = 150 + len(S.CASES) + len(S.NORMAL_CASES) + len(EACH) + len(GAP) + len(GAP49) + len(GAP50) + len(GAP52)


def test_벡터_파일의_머리와_행() -> None:
    meta, rows = H.read_vectors(VEC)
    assert meta["schema"] == H.SCHEMA and meta["n_rows"] == len(rows) == N_ROWS
    assert meta["fields"] == list(H.FIELDS)
    assert {r["group"] for r in rows} == {"N+", "D+", "ADV1", "ADV2", "STRUCT+", "STRUCT-", "NORM+", "NORM-", "EACH+", "EACH-",
                                          *(c["group"] for c in GAP), *(c["group"] for c in GAP49), *(c["group"] for c in GAP50),
                                          *(c["group"] for c in GAP52)}
    assert all(c["group"].startswith("GAP46-") for c in GAP) and all(c["group"].startswith("GAP49-") for c in GAP49)
    assert all(c["group"].startswith("GAP50-") for c in GAP50) and all(c["group"].startswith("GAP52-") for c in GAP52)
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


SUPPLEMENT6_REFERENCE_SHA256 = "07ce544f974e95692800314413a1c87f4ad6d741a54cf239404e979713e00962"
"""보강 6(256c40b)의 정본 바이트 — 판 4 벡터의 핀이었다(당시 값). 지금 값은 meta 의 reference_sha256 이다."""
SUPPLEMENT6_VECTORS_SHA256 = "48523cc37b02a8196f3ce3dade0919f3ef44b2b651a6a83f8208467d8a84af08"
SUPPLEMENT7_FIRST_REFERENCE_SHA256 = "7f80f90a0f9cb6e07dcd2c6e9d4f980346311e8d72ff8937b7b70b606e664518"
"""보강 7 의 첫 핀(6판, 커밋 6b11185e)의 정본 바이트 — 독립 검수를 반영하며 다시 걸었다(당시 값)."""
SUPPLEMENT7_FIRST_VECTORS_SHA256 = "fedc73171c1cd467c06dc47ff053d1f2a5f16a03dbb021885e1a77e35e3c1796"
SUPPLEMENT7_REVIEWED_REFERENCE_SHA256 = "9aa88c939a341036031330f5354d8b927cb12208f8b3c230cd33a727a76f08ad"
"""독립 검수를 반영한 7판(커밋 223b31b6)의 정본 바이트 — v8 결정적 층의 틈 정정(47)에서 다시 걸었다(당시 값)."""
SUPPLEMENT7_REVIEWED_VECTORS_SHA256 = "aa02edddc4a405fa565984bf2f4eac7143729dacedc59eb65e7367d866abdde9"
GAP46_REFERENCE_SHA256 = "ba291a24854699352f4e943cc99227a0b25734417a1b5c798551ec32ab7395ed"
"""v8 결정적 층의 틈 정정(47, 커밋 8696585a)의 정본 바이트 — 49 독립 검수의 재정정에서 다시 걸었다(당시 값)."""
GAP46_VECTORS_SHA256 = "093b6ed086e2b5777520d2cc4499a9c6411ffe2ceddddea34cbd936c8e1d7105"
GAP49_REFERENCE_SHA256 = "7518fe04355c3615adcca0664d1da37a47705f087ad117d746e2cb5e73195da1"
"""49 독립 검수의 재정정(커밋 b84a9901)의 정본 바이트 — 50 독립 검수의 '너무 넓음' 정정에서 다시 걸었다(당시 값)."""
GAP49_VECTORS_SHA256 = "4590b60b290c156ebd97f7417fddb0ecb9d9256ef45ddd6c9c6e5505d6b0ecc0"
GAP50_REFERENCE_SHA256 = "7e5e769f17c856ead270ac00d561300cef9a409d8b639af8218815462093cd1c"
"""50 독립 검수의 '너무 넓음' 정정(커밋 6db5bbf1)의 정본 바이트 — 봉인 집합의 두 결함 음성 뒤 정정에서 다시 걸었다(당시 값)."""
GAP50_VECTORS_SHA256 = "0314a3787e70598721b6400322c03439155bf76f7b6ade172b758c83ec322d5c"


def test_핀의_당시_값과_지금_값을_함께_싣는다() -> None:
    """보강 7(§137 I-02 · I-03 · 배분 서술)에서 핀을 다시 걸고 행을 더했다 — 보강 6 핀에서 6판으로 다시 건 핀의 바뀐 기대값은 0 행,
    독립 검수를 반영해 7판으로 다시 건 핀은 배분 사례 세 행의 코드가 바뀌었다. 앞 핀들과 그때 벡터 파일의 해시가 meta 에 차례로
    남는다. 더한 행은 배분 서술 사례 파일의 행이고, 그 파일의 해시와 까닭이 함께 남는다. v8 결정적 층의 틈 정정(47)으로 다시 건
    핀의 바뀐 기대값은 0 행이고, 그 합성 사례(46 탐침 · 변형)를 둘째로 더했다. 49 독립 검수의 재정정으로 다시 건 핀의 바뀐 기대값도
    0 행이고, 정본이 가르는 그 합성 사례(띠 예외어 · 부재 예외 · 이름 나열)를 셋째로 더했다. 50 독립 검수의 '너무 넓음' 정정으로
    다시 건 핀의 바뀐 기대값도 0 행이고, 그 합성 사례(띠 예외어 · 이름만 뺀 부재 · 이웃 꼴)를 넷째로 더했다. 봉인 집합의 두 결함
    음성 뒤 정정(부재 예외 나열 · 속성 이름씨)으로 다시 건 핀의 바뀐 기대값도 0 행이고, 그 합성 사례(이름 + 들 · 결함, 칸 + 영역 ·
    부분, 구분자)를 다섯째로 더했다."""
    meta, rows = H.read_vectors(VEC)
    assert meta["version"] == H.VERSION == 6
    assert meta["reference_sha256"] == H.lf_sha256(ROOT / "evaluation/grounded_v7_ref.py") not in (
        SUPPLEMENT6_REFERENCE_SHA256, SUPPLEMENT7_FIRST_REFERENCE_SHA256, SUPPLEMENT7_REVIEWED_REFERENCE_SHA256,
        GAP46_REFERENCE_SHA256, GAP49_REFERENCE_SHA256, GAP50_REFERENCE_SHA256)
    first, past, reviewed, gap46, gap49, gap50 = meta["reference_history"]
    assert (first["version"], first["reference_sha256"], first["vectors_sha256"]) == (
        4, SUPPLEMENT6_REFERENCE_SHA256, SUPPLEMENT6_VECTORS_SHA256)
    assert first["changed"] == [] and "보강 7" in first["note"]
    assert (past["version"], past["reference_sha256"], past["vectors_sha256"]) == (
        6, SUPPLEMENT7_FIRST_REFERENCE_SHA256, SUPPLEMENT7_FIRST_VECTORS_SHA256)
    assert sorted(c["id"] for c in past["changed"]) == ["EA-13", "EA-14", "EA-18"] and "독립 검수" in past["note"]
    assert (reviewed["version"], reviewed["reference_sha256"], reviewed["vectors_sha256"]) == (
        6, SUPPLEMENT7_REVIEWED_REFERENCE_SHA256, SUPPLEMENT7_REVIEWED_VECTORS_SHA256)
    assert reviewed["changed"] == [] and "47" in reviewed["note"]
    assert (gap46["version"], gap46["reference_sha256"], gap46["vectors_sha256"]) == (
        6, GAP46_REFERENCE_SHA256, GAP46_VECTORS_SHA256)
    assert gap46["changed"] == [] and "49" in gap46["note"]
    assert (gap49["version"], gap49["reference_sha256"], gap49["vectors_sha256"]) == (
        6, GAP49_REFERENCE_SHA256, GAP49_VECTORS_SHA256)
    assert gap49["changed"] == [] and "50" in gap49["note"]
    assert (gap50["version"], gap50["reference_sha256"], gap50["vectors_sha256"]) == (
        6, GAP50_REFERENCE_SHA256, GAP50_VECTORS_SHA256)
    assert gap50["changed"] == [] and "봉인 집합" in gap50["note"]
    ext = meta["extensions"]
    assert [(e["cases"], e["previous_sha256"]) for e in ext] == [("grounded_v7_each_cases.json", None),
                                                                ("grounded_v7_gap46_cases.json", None),
                                                                ("grounded_v7_gap49_cases.json", None),
                                                                ("grounded_v7_gap50_cases.json", None),
                                                                ("grounded_v7_gap52_cases.json", None)]
    assert [e["sha256"] for e in ext] == [H.lf_sha256(EACH_FILE), H.lf_sha256(GAP_FILE), H.lf_sha256(GAP49_FILE),
                                          H.lf_sha256(GAP50_FILE), H.lf_sha256(GAP52_FILE)]
    n46, n49, n50, n52 = len(GAP), len(GAP49), len(GAP50), len(GAP52)
    tail = n52 + n50 + n49 + n46
    assert ext[0]["added"] == [c["id"] for c in EACH] == [r["id"] for r in rows[-tail - len(EACH):-tail]]
    assert "배분" in ext[0]["note"]
    assert ext[1]["added"] == [c["id"] for c in GAP] == [r["id"] for r in rows[-tail:-n52 - n50 - n49]] and "47" in ext[1]["note"]
    assert ext[2]["added"] == [c["id"] for c in GAP49] == [r["id"] for r in rows[-n52 - n50 - n49:-n52 - n50]] and "49" in ext[2]["note"]
    assert ext[3]["added"] == [c["id"] for c in GAP50] == [r["id"] for r in rows[-n52 - n50:-n52]] and "50" in ext[3]["note"]
    assert ext[4]["added"] == [c["id"] for c in GAP52] == [r["id"] for r in rows[-n52:]] and "봉인 집합" in ext[4]["note"]


def test_rebase_는_앞_핀과_바뀐_기대값을_남긴다(tmp_path) -> None:
    meta, rows = H.read_vectors(VEC)
    first = rows[0]["id"]
    rows[0] = {**rows[0], "expected": {**rows[0]["expected"], "grounded_codes": ["made_up_code"]}}
    src = tmp_path / "old.jsonl"
    H._write(src, {**meta, "reference_sha256": "0" * 64}, rows)
    out = tmp_path / "new.jsonl"
    res = H.rebase(src, out, "시험")
    new_meta, new_rows = H.read_vectors(out)
    assert res["previous_reference_sha256"] == "0" * 64 and new_meta["reference_sha256"] == meta["reference_sha256"]
    assert new_meta["reference_history"][:-1] == meta["reference_history"]
    last = new_meta["reference_history"][-1]
    assert last["reference_sha256"] == "0" * 64 and last["note"] == "시험"
    assert last["changed"] == [{"id": first, "field": "grounded_codes", "before": ["made_up_code"],
                                "after": H.read_vectors(VEC)[1][0]["expected"]["grounded_codes"]}]
    assert new_rows == H.read_vectors(VEC)[1]                       # 행의 입력은 그대로, 기대값은 지금 정본의 출력
    assert H.check(out, None, None, {})["equal"] and H.check(out, None, None, {})["stale"] == []


def test_rebase_는_정본이_그대로거나_까닭이_없으면_거부한다(tmp_path) -> None:
    with pytest.raises(SystemExit):
        H.rebase(VEC, tmp_path / "x.jsonl", "같은 정본")
    meta, rows = H.read_vectors(VEC)
    src = tmp_path / "old.jsonl"
    H._write(src, {**meta, "reference_sha256": "0" * 64}, rows)
    with pytest.raises(SystemExit):
        H.rebase(src, tmp_path / "y.jsonl", "  ")
    assert not (tmp_path / "x.jsonl").exists() and not (tmp_path / "y.jsonl").exists()


@pytest.mark.parametrize("case", EACH, ids=[c["id"] for c in EACH])
def test_배분_서술_사례의_의도(case) -> None:
    """'A 와 B 에 각각 N개씩' 은 칸마다 N 이다 — 맞는 글은 지나고(EACH+), '각각' 이 들었지만 주석과 안 맞는 글은 기각된다(EACH-)."""
    got = R.grounded_codes(case["task"], case["candidate"])
    if case["intent"] is None:
        assert got == [], (case["id"], got)
    else:
        assert got and set(case["intent"]) <= set(got), (case["id"], got)
    assert R.contract_codes(case["task"], case["candidate"]) == [] and R.attribute_codes(case["task"], case["candidate"]) == []


def test_배분_서술_사례는_회수본_행을_고루_싣는다() -> None:
    """회수본 감사 행(GV-xx)은 맞는 행 10 개 이상(칸 둘 · 칸 셋), '각각' 이 들었지만 주석과 안 맞는 행 5 개 이상(이름 둘 이상 포함)이다.
    실물 id 는 싣지 않는다."""
    gv = [c for c in EACH if c["id"].startswith("GV-")]
    good, bad = [c for c in gv if c["intent"] is None], [c for c in gv if c["intent"] == []]
    zones = lambda c: len({(p["where"]) for p in c["task"]["evidence"]["defect_positions"]})
    names = lambda c: len(c["task"]["evidence"]["defect_counts_by_name"])
    assert len(good) >= 10 and {zones(c) for c in good} >= {2, 3} and len(bad) >= 5 and any(names(c) >= 2 for c in bad)
    assert all("각각" in c["candidate"] for c in gv)
    assert "aihub" not in EACH_FILE.read_text(encoding="utf-8")


def test_extend_는_새_id_만_덧붙이고_기존_행은_그대로다(tmp_path) -> None:
    meta, rows = H.read_vectors(VEC)
    src = tmp_path / "cut.jsonl"
    H._write(src, {**meta, "n_rows": len(rows) - 2}, rows[:-2])
    out = tmp_path / "full.jsonl"
    res = H.extend(src, GAP52_FILE, out, "시험")
    new_meta, new_rows = H.read_vectors(out)
    assert res["added"] == [r["id"] for r in rows[-2:]] and new_rows == rows
    assert new_meta["extensions"][-1]["added"] == res["added"] and new_meta["extensions"][-1]["note"] == "시험"
    assert H.check(out, None, None, {})["equal"]


def test_extend_는_핀이_다르거나_입력이_다르거나_까닭이_없거나_더할_행이_없으면_거부한다(tmp_path) -> None:
    meta, rows = H.read_vectors(VEC)
    stale = tmp_path / "stale.jsonl"
    H._write(stale, {**meta, "n_rows": len(rows) - 1, "reference_sha256": "0" * 64}, rows[:-1])
    changed = tmp_path / "changed.jsonl"
    bad = [dict(r) for r in rows[:-1]]
    bad[-1] = {**bad[-1], "candidate": bad[-1]["candidate"] + " "}            # 틈 사례의 같은 id, 다른 입력
    H._write(changed, {**meta, "n_rows": len(bad)}, bad)
    cut = tmp_path / "cut.jsonl"
    H._write(cut, {**meta, "n_rows": len(rows) - 1}, rows[:-1])
    for src, note in ((stale, "시험"), (changed, "시험"), (cut, "  "), (VEC, "더할 행 없음")):
        with pytest.raises(SystemExit):
            H.extend(src, GAP50_FILE, tmp_path / "out.jsonl", note)
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize("case", GAP49, ids=[c["id"] for c in GAP49])
def test_재정정_사례의_의도(case) -> None:
    """49 독립 검수의 재정정에서 정본이 가르는 사례 — 띠 예외어는 띠 주어 바로 앞에 붙은 것만(H), 예외어가 든 부재는 예외로 든 칸 ·
    이름을 뺀 나머지로 대조(A), 칸 바로 뒤의 이름은 다른 칸의 나열에 끌려가지 않는다(I). 의도 null 은 통과, 코드 목록은 그 기각."""
    got = R.grounded_codes(case["task"], case["candidate"])
    if case["intent"] is None:
        assert got == [], (case["id"], got)
    else:
        assert got and set(case["intent"]) <= set(got), (case["id"], got)


def test_예외어는_붙은_자리와_뺀_것을_읽는다() -> None:
    """넓힌 예외어가 너무 넓지 않은가 — 덧붙임('외에도 · 말고도' · '…외에 용입불량도 있으며, 결함은') · 띠 소견 뒤의 예외 마디는
    띠 예외가 아니고, 무엇을 뺐는지 읽지 못하는 부재('다른 칸에는')는 고치기 전처럼 대조하지 않는다. 뺀 것을 읽으면 여집합으로
    대조한다('위쪽 왼쪽 칸을 뺀 칸에는 결함이 없다' — 아래 오른쪽에 균열이 있다)."""
    assert R._BAND_EXCEPT.search("용입불량을 뺀 결함은 모두 위쪽 높이의 칸에 있다")
    assert R._BAND_EXCEPT.search("기공 말고 다른 결함은 모두 위쪽 높이의 칸에 있다")
    for s in ("기공 외에 용입불량도 있으며, 결함은 모두 아래쪽 높이의 칸에 있다", "용입불량 외에도 결함은 모두 위쪽 높이의 칸에 있다",
              "용입불량 말고도 결함은 모두 위쪽 높이의 칸에 있다", "결함은 모두 아래쪽 높이의 칸에 있고 기공을 뺀 나머지도 그렇다"):
        assert not R._BAND_EXCEPT.search(s), s
    names, zones, _ = R._excepted("위쪽 높이와 아래 가운데를 뺀 칸에는 표시된 결함이 없다")
    assert names == set() and zones == {("위", c) for c in ("왼쪽", "가운데", "오른쪽")} | {("아래", "가운데")}
    assert R._excepted("기공을 뺀 위 왼쪽 칸에는 결함이 없다")[:2] == ({"기공"}, set())
    assert R._excepted("다른 칸에는 결함이 없다") is None and R._excepted("기공 외에도 균열이 없다") is None
    task = S.task_of([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")])
    head = "기공 1개가 위쪽 왼쪽 칸에, 균열 1개가 아래쪽 오른쪽 칸에 표시되어 있다. "
    assert R.grounded_codes(task, head + S.L) == []
    assert "absence_contradicts_locked" in R.grounded_codes(task, head + "위쪽 왼쪽 칸을 뺀 칸에는 결함이 없다. " + S.L)
    assert "absence_contradicts_locked" in R.grounded_codes(task, head + "위쪽 왼쪽 칸을 제외한 칸에는 결함이 없다. " + S.L)
    assert R.grounded_codes(task, head + "위쪽 왼쪽 칸과 아래쪽 오른쪽 칸을 뺀 칸에는 결함이 없다. " + S.L) == []
    assert R.grounded_codes(task, head + "다른 칸에는 결함이 없다. " + S.L) == []


@pytest.mark.parametrize("case", GAP50, ids=[c["id"] for c in GAP50])
def test_50_사례의_의도(case) -> None:
    """50 독립 검수의 '너무 넓음' 정정에서 정본이 가르는 사례 — 띠 예외어(B)와 부재 예외(X: 이름만 뺀 부재는 영상 전체로 대조하고, 뺀
    것을 칸 · 이름으로 읽지 못하면 대조하지 않는다)와 그 이웃 꼴. 의도 null 은 통과, 코드 목록은 그 기각."""
    got = R.grounded_codes(case["task"], case["candidate"])
    if case["intent"] is None:
        assert got == [], (case["id"], got)
    else:
        assert got and set(case["intent"]) <= set(got), (case["id"], got)


def test_이름만_뺀_부재는_영상_전체로_대조한다() -> None:
    """'기공을 뺀 결함은 없다' — 남은 칸이 비면 뺀 칸 밖 모두, 뺀 칸도 없으면 영상 전체로 대조한다(아래 오른쪽에 균열이 잠김). 예외로
    든 이름이 잠긴 이름을 다 덮으면 참이고, 뺀 것이 칸도 이름도 아니면('영상 밖에는') 무엇을 뺐는지 모르니 대조하지 않는다."""
    task = S.task_of([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")])
    head = "기공 1개가 위쪽 왼쪽 칸에, 균열 1개가 아래쪽 오른쪽 칸에 표시되어 있다. "
    for s in ("기공을 뺀 결함은 없다.", "기공 외에는 결함이 없다.", "기공 말고 다른 결함은 없다.", "균열을 뺀 결함은 없다.",
              "기공을 빼면 결함은 없다."):
        assert "absence_contradicts_locked" in R.grounded_codes(task, head + s + " " + S.L), s
    for s in ("기공과 균열을 뺀 결함은 없다.", "기공과 균열 외에는 결함이 없다.", "영상 밖에는 결함이 없다.",
              "기공 외에는 아래 왼쪽 칸에 결함이 없다."):
        assert R.grounded_codes(task, head + s + " " + S.L) == [], s
    assert R._excepted("영상 밖에는 결함이 없다")[:2] == (set(), set())
    assert not R._excepted_absence(set(), {"균열": {("아래", "오른쪽")}}, (set(), set(), [(0, 2)]))
    assert R._excepted_absence(set(), {"균열": {("아래", "오른쪽")}}, ({"기공"}, set(), [(0, 2)]))


@pytest.mark.parametrize("case", GAP52, ids=[c["id"] for c in GAP52])
def test_52_사례의_의도(case) -> None:
    """부재 예외 나열의 정정에서 정본이 가르는 사례 — 이름 + 들 · 이름 + 결함, 칸 + 영역 · 부분, 구분자 · ㆍ / 이나와 이웃 꼴.
    의도 null 은 통과, 코드 목록은 그 기각."""
    got = R.grounded_codes(case["task"], case["candidate"])
    if case["intent"] is None:
        assert got == [], (case["id"], got)
    else:
        assert got and set(case["intent"]) <= set(got), (case["id"], got)


def test_부재_예외_나열은_이름_들_결함_칸_영역_부분_구분자를_읽는다() -> None:
    """'기공들을 · 기공 결함을 뺀', '위쪽 왼쪽 영역 외에는', '기공·균열 · 기공/균열 · 기공이나 균열 외에는' — 나열을 읽어 뺀 것 밖으로
    대조한다(뺀 이름이 잠긴 이름을 다 덮으면 참). '영역 · 부분' 앞이 칸이 아니면('표시된 부분 외에는') 무엇을 뺐는지 모르니 대조하지 않는다."""
    assert R._excepted("기공들을 뺀 결함은 없다")[:2] == ({"기공"}, set())
    assert R._excepted("기공 결함을 뺀 결함은 없다")[:2] == ({"기공"}, set())
    for s in ("기공·균열 외에는 결함이 없다", "기공ㆍ균열 외에는 결함이 없다", "기공/균열 외에는 결함이 없다", "기공이나 균열 외에는 결함이 없다",
              "기공들·균열 외에는 결함이 없다", "기공 결함·균열 결함 외에는 결함이 없다"):
        assert R._excepted(s)[:2] == ({"기공", "균열"}, set()), s
    assert R._excepted("위쪽 왼쪽 영역 외에는 결함이 없다")[:2] == (set(), {("위", "왼쪽")})
    assert R._excepted("표시된 부분 외에는 결함이 없다")[:2] == (set(), set())
    task = S.task_of([("기공", "위 왼쪽"), ("균열", "아래 오른쪽")])
    head = "기공 1개가 위쪽 왼쪽 칸에, 균열 1개가 아래쪽 오른쪽 칸에 표시되어 있다. "
    for s in ("기공들을 뺀 결함은 없다.", "기공 결함 외에는 결함이 없다.", "위쪽 왼쪽 영역 외에는 결함이 없다.",
              "아래쪽 오른쪽 부분을 뺀 곳에는 결함이 없다.", "기공·용입불량 외에는 결함이 없다."):
        assert "absence_contradicts_locked" in R.grounded_codes(task, head + s + " " + S.L), s
    for s in ("기공·균열 외에는 결함이 없다.", "기공/균열 외에는 결함이 없다.", "기공이나 균열 외에는 결함이 없다.",
              "기공들과 균열 외에는 결함이 없다.", "위쪽 왼쪽 영역과 아래쪽 오른쪽 영역 외에는 결함이 없다.", "표시된 부분 외에는 결함이 없다."):
        assert R.grounded_codes(task, head + s + " " + S.L) == [], s
