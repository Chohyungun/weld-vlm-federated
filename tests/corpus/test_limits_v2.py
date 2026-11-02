"""허용치 표 2 판(부분 B) — 표 · 스키마 · 로더 · 조항 본문 · 정답 파생 · 역할(C 08 2 판 §1 · §2 · §7 의 시험 1 ~ 6 · 12(가) · 17).

1 판 표 · 1 판 파생은 바이트 그대로여야 한다. 2 판 표는 1 판 31 칸 뒤에 칸 넷을 더하고, 행 · 값 · 구간 · 조항은 바꾸지 않는다.
거부 사례는 실물 2 판 표의 사본에서 한 칸씩 바꿔 만든다(값을 이 파일에 옮기지 않는다).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from corpus.generate.run_cycle_corpus import defect_names
from corpus.rules import clause_text as CT
from corpus.rules import limit_eval
from corpus.rules.limits_loader import (
    EXPECTED_COLUMNS,
    EXPECTED_COLUMNS_V2,
    NEW_COLUMNS_V2,
    LimitsValidationError,
    load_limits,
)

REPO = Path(__file__).resolve().parents[2]
V1 = REPO / "corpus/rules/limits_v0_pilot.csv"
V2 = REPO / "corpus/rules/limits_v2_pilot.csv"
#: 1 판 표의 원시 sha256 — 바뀌면 1 판 회귀가 깨진 것이다(D 07 §1-4 의 고정 입력).
V1_SHA = "334c78811677accb86a2ec14dad24716e16add3b6a14b47abe33ca6fe9e0cccb"
#: 구판 회귀의 기준 트리 — 이 커밋의 코드로 낸 1 판 정답 파생과 지금 코드의 것이 같아야 한다(D 07 §1-4).
BASELINE_COMMIT = "b16e2306"


def _rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _write(path: Path, head: list[str], rows: list[dict[str, str]]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def _edit(tmp_path: Path, edits: dict[str, dict[str, str]], *, drop: tuple[str, ...] = (),
          head: list[str] | None = None) -> Path:
    """실물 2 판 표의 사본 — `rule_id` 마다 칸을 바꾸고(`edits`), 행을 빼고(`drop`), 머리를 갈아 끼운다."""
    h, rows = _rows(V2)
    out = []
    for r in rows:
        if r["rule_id"] in drop:
            continue
        out.append({**r, **edits.get(r["rule_id"], {})})
    head = head or h
    return _write(tmp_path / "limits_v2_edit.csv", head, [{k: r.get(k, "") for k in head} for r in out])


def _checks(path: Path) -> set[str]:
    with pytest.raises(LimitsValidationError) as e:
        load_limits(path, pilot=True)
    return {v.check for v in e.value.violations}


# ---------------------------------------------------------------- 시험 17 — 값 바이트 불변

def v1_cells_problems(v1: Path, v2: Path) -> list[str]:
    """2 판의 앞 31 칸이 `rule_id` 마다 1 판과 **같은 원문 문자열**인가(형 변환 · 공백 정리 없이 — C 08 2 판 §1-5). 행 수 · `rule_id`
    집합 · 머리의 앞 31 칸(순서까지)도 같아야 한다."""
    h1, r1 = _rows(v1)
    h2, r2 = _rows(v2)
    out = []
    if h2[:len(h1)] != h1 or tuple(h2[len(h1):]) != NEW_COLUMNS_V2:
        out.append("머리: 앞 칸이 1 판과 다르거나 뒤 칸이 새 칸 넷이 아니다")
    if [r["rule_id"] for r in r1] != [r["rule_id"] for r in r2]:
        out.append("행 수 · rule_id 가 다르다")
    by = {r["rule_id"]: r for r in r2}
    for r in r1:
        other = by.get(r["rule_id"])
        bad = [c for c in h1 if other is None or other.get(c) != r[c]]
        if bad:
            out.append(f"{r['rule_id']}: 원문 문자열이 다른 칸 {bad}")
    return out


def test_2판_표의_앞_31칸은_rule_id_마다_1판과_원문이_같다():
    assert hashlib.sha256(V1.read_bytes()).hexdigest() == V1_SHA
    assert v1_cells_problems(V1, V2) == []
    raw = V2.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf") and raw.endswith(b"\n")


def test_표기만_바꾼_값은_형_변환_뒤에_같아도_거부한다(tmp_path):
    """끝자리 0 · 앞뒤 공백 — 로더는 같은 수 · 같은 값으로 읽지만 원문 대조는 다르다고 해야 한다."""
    h, rows = _rows(V2)
    value_col = next(c for c in ("limit_value", "thickness_min") if rows[0][c])
    for change in (lambda v: v + "0" if "." in v else v + ".0", lambda v: " " + v):
        edited = [dict(r) for r in rows]
        edited[0][value_col] = change(edited[0][value_col])
        p = _write(tmp_path / "x.csv", h, edited)
        load_limits(p, pilot=True)                          # 정규화한 값은 같아 로더는 받는다
        assert v1_cells_problems(V1, p), "원문 대조가 표기 변경을 놓쳤다"


# ---------------------------------------------------------------- 시험 1 — 머리 · 판 판별

def test_1판_표는_지금_머리_그대로_1판으로_읽힌다():
    t = load_limits(V1, pilot=True)
    assert t.edition == "v1" and len(t.rows) == 12
    assert all(r.measure is None and r.defect_class is None and r.conditions == () for r in t.rows)
    assert not any("V12" in s for s in t.checks_skipped)


def test_2판_표는_2판으로_읽히고_VT_행의_생략을_적는다():
    t = load_limits(V2, pilot=True)
    assert t.edition == "v2" and len(t.rows) == 12
    assert tuple(_rows(V2)[0]) == EXPECTED_COLUMNS_V2 and EXPECTED_COLUMNS_V2[:31] == EXPECTED_COLUMNS
    vt = [s for s in t.checks_skipped if s.startswith("V12-V13")]
    assert len(vt) == 1 and "KRA27-S-01" in vt[0]
    # 1 판과 행 값이 같다 — 새 칸 넷만 다르다(행 · 값 · 구간 · 조항 불변)
    t1 = load_limits(V1, pilot=True)
    drop = set(NEW_COLUMNS_V2)
    assert [r.model_dump(exclude=drop) for r in t.rows] == [r.model_dump(exclude=drop) for r in t1.rows]


@pytest.mark.parametrize("keep", [("measure",), ("measure", "aggregation", "defect_class")])
def test_새_칸이_일부만_있으면_거부한다(tmp_path, keep):
    head = list(EXPECTED_COLUMNS) + list(keep)
    assert _checks(_edit(tmp_path, {}, head=head)) == {"V1"}


def test_미정의_칸은_2판에서도_거부한다(tmp_path):
    head = list(EXPECTED_COLUMNS_V2) + ["role"]
    assert _checks(_edit(tmp_path, {}, head=head)) == {"V1"}


# ---------------------------------------------------------------- 시험 2 — V12 (거부 일곱)

V12_CASES = {
    "값 한계 행에 measure 없음": {"KRA27-T15-01": {"measure": "", "aggregation": ""}},
    "불허 행에 measure": {"KRA27-3D-01": {"measure": "최대결함크기", "aggregation": "개별"}},
    "합계길이 + 개별": {f"KRA27-T16-0{i}": {"aggregation": "개별"} for i in range(1, 7)},
    "군집간격 + 개별": {f"KRA27-T15-0{i}": {"conditions": "구별곤란:제2종|군집간격|무등급전제|혼재:제2종"} for i in range(1, 5)},
    "모르는 토큰": {"KRA27-T15-01": {"conditions": "무등급전제|모르는조건"}},
    "토큰 겹침": {"KRA27-T15-01": {"conditions": "무등급전제|무등급전제"}},
    "무등급전제 + 품질 체계": {"KRA27-T15-01": {"quality_scheme": "iso5817", "quality_level": "C"}},
}


@pytest.mark.parametrize("case", sorted(V12_CASES))
def test_V12_는_행의_필수와_짝을_거부한다(tmp_path, case):
    assert "V12" in _checks(_edit(tmp_path, V12_CASES[case])), case


def test_조건_토큰이_정렬_순이_아니면_거부한다(tmp_path):
    rows = {f"KRA27-T15-0{i}": {"conditions": "혼재:제2종|구별곤란:제2종|무등급전제"} for i in range(1, 5)}
    assert "V12" in _checks(_edit(tmp_path, rows))


# ---------------------------------------------------------------- 시험 3 — V13 (거부 넷 · 수락 하나)

def test_V13_짝_종별의_행이_없으면_거부한다(tmp_path):
    rows = {f"KRA27-T15-0{i}": {"conditions": "구별곤란:제2종|구별곤란:제3종|무등급전제|혼재:제2종"} for i in range(1, 5)}
    assert "V13" in _checks(_edit(tmp_path, rows, drop=("KRA27-3D-01",)))


def test_V13_짝이_대칭이_아니면_거부한다(tmp_path):
    rows = {f"KRA27-T15-0{i}": {"conditions": "구별곤란:제2종|구별곤란:제3종|무등급전제|혼재:제2종"} for i in range(1, 5)}
    assert "V13" in _checks(_edit(tmp_path, rows))                 # 3D 행이 거꾸로 '구별곤란:제1종' 을 들지 않는다


def test_V13_한_코드에_종별_둘이면_거부한다(tmp_path):
    assert "V13" in _checks(_edit(tmp_path, {"KRA27-T16-04": {"defect_class": "제1종"}}))


def test_V13_같은_조항_안의_칸이_다르면_거부한다(tmp_path):
    assert "V13" in _checks(_edit(tmp_path, {"KRA27-T16-01": {"conditions": "구별곤란:제1종|무등급전제|혼재:제1종"}}))


def test_V13_은_VT_행의_빈칸을_받는다():
    t = load_limits(V2, pilot=True)
    vt = next(r for r in t.rows if r.inspection_method.value == "VT")
    assert (vt.measure, vt.aggregation, vt.defect_class, vt.conditions) == (None, None, None, ())


@pytest.mark.parametrize("cell", [{"measure": "최대결함크기"}, {"aggregation": "개별"}, {"defect_class": "제1종"},
                                  {"conditions": "무등급전제"}])
def test_V12_VT_행은_새_칸_넷_가운데_하나라도_차면_거부한다(tmp_path, cell):
    """09 §10 M1 — 부분 B 범위 밖인 VT 행은 넷 모두 빈칸이다. 하나만 채워도 V12 의 VT 문구로 거부한다(종별 · 조건처럼 다른 짝
    검사가 잡지 않는 칸도)."""
    _h, rows = _rows(V2)
    vt = next(r["rule_id"] for r in rows if r["inspection_method"] == "VT")
    with pytest.raises(LimitsValidationError) as e:
        load_limits(_edit(tmp_path, {vt: cell}), pilot=True)
    msgs = [v.message for v in e.value.violations if v.check == "V12" and vt in v.message and "VT 행에 새 칸" in v.message]
    assert msgs and next(iter(cell)) in msgs[0], [v.message for v in e.value.violations]


# ---------------------------------------------------------------- 시험 4 — 정답 파생 기준선

def _baseline_gold_json(tmp_path: Path) -> str:
    """기준 트리(`b16e2306`)의 허용치 규칙 모듈로 1 판 표의 `derive_gold_clauses` 를 낸 정렬 JSON(D 07 §1-4 의 기준선)."""
    out = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", BASELINE_COMMIT,
                          "corpus/__init__.py", "corpus/rules", "configs/label_map.yaml"], capture_output=True, check=False)
    if out.returncode != 0:
        pytest.skip(f"기준 트리 {BASELINE_COMMIT} 를 꺼내지 못했다")
    root = tmp_path / "baseline"
    with tarfile.open(fileobj=io.BytesIO(out.stdout)) as tar:
        tar.extractall(root, filter="data")
    code = ("import json, logging, sys; logging.disable(logging.WARNING); sys.path.insert(0, sys.argv[1]);"
            "from corpus.rules.limits_loader import load_limits; from corpus.rules.limit_eval import derive_gold_clauses;"
            "t = load_limits(sys.argv[2], pilot=True);"
            "sys.stdout.buffer.write(json.dumps(list(derive_gold_clauses(t)), sort_keys=True, ensure_ascii=False).encode('utf-8'))")
    res = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code, str(root), str(V1)], capture_output=True,
                         check=True, cwd=str(root))
    return res.stdout.decode("utf-8")


def test_신판_코드로_1판_표의_정답_파생을_다시_내도_기준_트리와_같은_바이트다(tmp_path):
    now = json.dumps(list(limit_eval.derive_gold_clauses(load_limits(V1, pilot=True))), sort_keys=True, ensure_ascii=False)
    assert now == _baseline_gold_json(tmp_path)


def test_2판_표의_정답_파생은_1판과_같다():
    """정답 조항 목록에는 새 칸을 싣지 않는다 — 행 · 값 · 구간 · 조항이 같으므로 출력도 같다."""
    assert limit_eval.derive_gold_clauses(load_limits(V2, pilot=True)) == \
        limit_eval.derive_gold_clauses(load_limits(V1, pilot=True))


# ---------------------------------------------------------------- 시험 5 — 역할(주 · 보조)

def test_1판_역할은_보조가_없고_주가_1판_조항이다():
    roles = limit_eval.derive_clause_roles(load_limits(V1, pilot=True))
    assert all(v["auxiliary"] == () and v["defect_class"] is None and v["conditions"] == () for v in roles.values())
    st = {code: v["main"] for (mat, code), v in roles.items() if mat == "ST"}
    assert st == {"100": "KRA27-3D", "2011": "KRA27-T15", "301": "KRA27-T16", "401": "KRA27-T16"}
    assert all(v["main"] is None for (mat, _), v in roles.items() if mat == "AL")


def test_2판_역할은_구별곤란_토큰에서_보조를_낸다():
    roles = limit_eval.derive_clause_roles(load_limits(V2, pilot=True))
    aux = {code: v["auxiliary"] for (mat, code), v in roles.items() if mat == "ST"}
    assert aux == {"2011": (("KRA27-T16", "구별곤란"),), "301": (("KRA27-T15", "구별곤란"),),
                   "401": (("KRA27-T15", "구별곤란"),), "100": ()}
    assert all(v["main"] is None and v["auxiliary"] == () for (mat, _), v in roles.items() if mat == "AL")


def test_주_조항이_둘이면_멈춘다(tmp_path):
    p = _edit(tmp_path, {"KRA27-T16-04": {"clause_id": "KRA27-T16X"}})        # 401 의 행 하나만 다른 조항으로
    table = load_limits(p, pilot=True)                                         # 표는 받는다 — 멈추는 것은 역할 파생이다
    with pytest.raises(ValueError, match="주 조항이 둘 이상"):
        limit_eval.derive_clause_roles(table)


def _vector(name: str, sha: str) -> dict | None:
    """평가 쪽 기대 사례 — 작업 트리에 있으면 그것을, 없으면 평가 쪽 1 단계 커밋의 blob 을 읽는다(사본을 두지 않는다). sha 를 맞댄다."""
    rel = f"evaluation/vectors/{name}"
    p = REPO / rel
    raw = p.read_bytes() if p.is_file() else None
    if raw is None:
        out = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"91e83979:{rel}"], capture_output=True, check=False)
        raw = out.stdout if out.returncode == 0 else None
    if raw is None:
        return None
    assert hashlib.sha256(raw).hexdigest() == sha, f"{name} 의 바이트가 고정값과 다르다"
    return json.loads(raw)


CLAUSE_AXIS_ROLES = ("clause_axis_roles.v1.json", "78d297adf7f61f2c1e2eca25a51d8c6b86acf018a098c3bfe10a0786a4a12a8b")


def test_2판_역할은_평가_쪽_사례의_신판_세계와_같다():
    """C 시험 5 ↔ D 07 §13-3 의 입력 쪽 — 평가 쪽 사례 파일의 `worlds.edition2` 와 같은 (재질, 코드) · 주 · 보조. 분해 함수와의 결합은
    평가 쪽 병합 뒤의 통합 커밋이다."""
    doc = _vector(*CLAUSE_AXIS_ROLES)
    if doc is None:
        pytest.skip("평가 쪽 사례 파일을 읽지 못했다(작업 트리 · 1 단계 커밋 모두 없음)")
    roles = limit_eval.derive_clause_roles(load_limits(V2, pilot=True))
    mine = {(m, c): (v["main"], [list(a) for a in v["auxiliary"]]) for (m, c), v in roles.items() if v["main"] is not None}
    world = {(r["material"], r["code"]): (r["main"], r["auxiliary"]) for r in doc["worlds"]["edition2"]["roles"]}
    assert mine == world
    assert sorted({v["main"] for v in roles.values() if v["main"]} | {"KRA27-S"}) == doc["worlds"]["edition2"]["index"]


# ---------------------------------------------------------------- 시험 6 — 조항 본문

def test_1판_조항_본문은_파생_파일의_바이트_그대로다():
    names = defect_names()
    texts = CT.derive_clause_texts(load_limits(V1, pilot=True).rows, names)
    doc = json.loads((REPO / "corpus/derived/clause_texts.json").read_text(encoding="utf-8"))
    assert doc["clauses"] == texts and doc["_meta"]["source_sha256"] == V1_SHA


def test_2판_조항_본문은_측정량_종별_조건_줄을_더한다():
    names = defect_names()
    v1 = CT.derive_clause_texts(load_limits(V1, pilot=True).rows, names)
    v2 = CT.derive_clause_texts(load_limits(V2, pilot=True).rows, names)
    assert "측정량: 최대결함크기." in v2["KRA27-T15"] and "종별: 제1종." in v2["KRA27-T15"]
    assert "조건: 구별곤란:제1종, 군집간격, 무등급전제, 혼재:제1종." in v2["KRA27-T16"]
    assert "측정량" not in v2["KRA27-3D"] and "조건: 무등급전제." in v2["KRA27-3D"]
    assert v2["KRA27-S"] == v1["KRA27-S"]                                    # VT 행은 칸이 비어 1 판과 같다
    for cid in v1:
        assert all(line in v2[cid] for line in v1[cid].split(". ")[:6])       # 1 판 줄은 그대로 남는다


# ---------------------------------------------------------------- 시험 12 (가) — 묶음 함수 두 구현

def test_로더_묶음은_평가_쪽_묶음_함수와_같은_값이고_목록이_같다():
    from corpus.generate import make_pairs_main as MM
    from evaluation.narrative_registration import limits_loader_bundle_sha256
    from evaluation.provenance import LIMITS_RULE_FILES

    assert MM.LIMITS_LOADER_FILES == LIMITS_RULE_FILES
    assert MM.lf_bundle(MM.LIMITS_LOADER_FILES)["sha256"] == limits_loader_bundle_sha256(REPO)
    assert MM.lf_bundle(MM.GENERATOR_FILES)["sha256"] == limits_loader_bundle_sha256(REPO, MM.GENERATOR_FILES)


def test_묶음_목록은_평가_쪽_사례_파일의_목록과_같다():
    doc = _vector("narrative_inputs_cases.v1.json", "4d06a076fc52fed0f96408d0b7923eed4dfe12e3ac3d46bed18af15b09f764b5")
    if doc is None:
        pytest.skip("평가 쪽 사례 파일을 읽지 못했다")
    from corpus.generate import make_pairs_main as MM

    assert list(MM.LIMITS_LOADER_FILES) == doc["loader_files"]
    assert list(MM.GENERATOR_FILES) == doc["generator_files"]


# ---------------------------------------------------------------- 시험 5 ↔ 평가 쪽 §13-3 — 결합(대역 없음, 지시 1024i)

#: 평가 쪽 고정 사례 — 읽기만 한다(바이트 불변). 기대는 이 파일에서만 온다.
CLAUSE_AXIS_CASES = "evaluation/vectors/clause_axis_roles.v1.json"
CLAUSE_AXIS_CASES_SHA256 = "78d297adf7f61f2c1e2eca25a51d8c6b86acf018a098c3bfe10a0786a4a12a8b"


def test_신판_표의_역할을_평가_쪽_함수로_세면_고정_사례의_수와_같다():
    """생성 쪽 신판 로더 → `derive_clause_roles` → 평가 쪽 `roles_from_derived` → 정답 쌍(주) · 보조 차집합 · 보고(평가 쪽 함수)를 평가 쪽
    고정 사례의 신판 세계와 맞댄다. 사례의 신판 역할(대역) · 색인과도 같아야 한다. 현 표에 없는 '주 없음 · 보조 있음' 은 평가 쪽 합성
    시험의 몫이다 — 신판 세계가 없는 사례는 건너뛰고, 건너뛴 수를 단언한다."""
    from evaluation import clause_axis as CA

    raw = (REPO / CLAUSE_AXIS_CASES).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CLAUSE_AXIS_CASES_SHA256
    cases = json.loads(raw.decode("utf-8"))
    world = cases["worlds"]["edition2"]
    table = load_limits(V2, pilot=True)
    roles = CA.roles_from_derived(limit_eval.derive_clause_roles(table, method="RT"))
    stand_in = CA.roles_from_derived({(r["material"], r["code"]): {"main": r["main"], "auxiliary": [tuple(a) for a in r["auxiliary"]],
                                                                   "defect_class": None, "conditions": ()}
                                      for r in world["roles"]})
    assert {k: v for k, v in roles.items() if k in stand_in} == stand_in
    # 생성 쪽은 표에 조항이 없는 재질(AL)도 '주 없음 · 보조 없음' 으로 적는다. 사례는 그 열쇠를 두지 않는다 — 평가 쪽 함수에서 둘은
    # 같은 뜻이다(역할이 없어도 · 주가 없어도 '정답 조항 없음'). 남는 열쇠가 그것뿐인지 본 뒤, 아래 수는 생성 쪽 역할 전체로 센다.
    extra = {k: v for k, v in roles.items() if k not in stand_in}
    assert extra and {m for m, _ in extra} == {"AL"} and all(v.main is None and not v.auxiliary for v in extra.values())
    assert sorted({r.clause_id for r in table.rows}) == world["index"]
    cells = [c for c in cases["cases"] if c["images"]["edition2"] is not None]
    assert len(cells) == len(cases["cases"]) - 1          # 신판 세계가 없는 사례는 '주 없음 · 보조 있음' 하나다
    for case in cells:
        images = case["images"]["edition2"]
        material_of = {im["image"]: im["material"] for im in images}
        gt = {im["image"]: im["gt_codes"] for im in images}
        pairs, skipped = CA.gold_pairs_from_roles(roles, material_of, gt)
        aux = CA.auxiliary_only_by_image(roles, material_of, gt)
        rep = CA.clause_axis_report({im["image"]: im["cited"] for im in images}, pairs, world["index"], auxiliary_only=aux)
        got = {"n_gold_pairs": rep["citation_match_rate"]["denominator"], "match_hits": rep["citation_match_rate"]["numerator"],
               "skipped": skipped, "n_citations": rep["n_citations"], "correct": rep["citation_precision"]["numerator"],
               "nonexistent": rep["ungrounded_rate"]["nonexistent"], "irrelevant": rep["ungrounded_rate"]["irrelevant"],
               "auxiliary_only": rep["ungrounded_rate"]["auxiliary_only"]}
        assert got == case["expect"], case["id"]
