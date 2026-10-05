"""`scripts/probe/pairs_independent.py` — 본실험 페어의 독립 재구현 대조(06번 §5 관점 C).

기대값은 전부 **손으로 적은 값**이다. 빌더도, 빌더가 만든 산출도 쓰지 않는다 — 봉인 자산 없이 돈다.
허용치 표와 사상표는 추적되는 실물(`corpus/rules/limits_v0_pilot.csv`·`configs/label_map.yaml`)을 읽는다.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.probe import pairs_independent as pi

REPO = Path(__file__).resolve().parents[1]
LIMITS = REPO / "corpus" / "rules" / "limits_v0_pilot.csv"
LABEL_MAP = REPO / "configs" / "label_map.yaml"

M_COLS = ["image_id", "rel_path", "sha256", "width_px", "height_px", "material", "has_defect", "n_defects",
          "iso_codes", "group_id", "split", "client"]
A_COLS = ["ann_id", "image_id", "iso_code", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px", "major_axis_px",
          "equiv_diameter_px", "geom_valid"]


def _img(image_id, split="train", client="C1", material="ST", codes="", n=0, group=None, path=None, sha=None):
    return {"image_id": image_id, "rel_path": path or f"tiles/RT/{material}/{image_id}.jpg",
            "sha256": sha or f"{abs(hash(image_id)):064x}"[:64], "width_px": "1280",
            "height_px": "720", "material": material, "has_defect": "True" if n else "False", "n_defects": str(n),
            "iso_codes": codes, "group_id": group or f"g-{image_id}", "split": split, "client": client}


def _ann(image_id, k, code, box=(10, 20, 110, 220), valid=True):
    corners = [str(v) for v in box] if valid else ["", "", "", ""]
    return {"ann_id": f"{image_id}#{k}", "image_id": image_id, "iso_code": code, "bbox_x1_px": corners[0],
            "bbox_y1_px": corners[1], "bbox_x2_px": corners[2], "bbox_y2_px": corners[3],
            "major_axis_px": "12.50" if valid else "", "equiv_diameter_px": "8.10" if valid else "",
            "geom_valid": "True" if valid else "False"}


def _write_csv(path: Path, cols, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _eval_row(image_id="z_pad"):
    return {**_img(image_id, split="eval", codes="", n=0), "client": ""}


def _digest(lines: list[str]) -> str:
    """주어진 줄을 **주어진 순서로** 이어 sha256 을 낸다. 모듈의 함수를 부르지 않는다.

    정해진 순서로 적힌 계약서에서는 이것이 지문이다. 순서를 바꾼 줄에 쓰면 **적힌 순서로 맞춘 오기**가
    된다 — 갈림 입력을 만드는 데 그렇게 쓴다. 규칙의 옳고 그름은 이 함수가 아니라 동결본 기록값과
    갈림 입력의 기대 판정이 가른다."""
    return hashlib.sha256("".join(x + "\n" for x in lines).encode("utf-8")).hexdigest()


#: 실물 동결본의 구성원 넷과 그 순서. 합성 계약서도 이 꼴로 쓴다.
_MEMBERS = ("manifest.csv", "annotations.csv", "data_capabilities.yaml", "tiles.csv")


def _write_contract(root: Path) -> list[str]:
    """실물 꼴의 계약서를 쓴다 — 구성원 넷을 정해진 순서로, 실제로 계산한 지문과 함께. 줄을 돌려준다."""
    if not (root / "data_capabilities.yaml").exists():
        (root / "data_capabilities.yaml").write_text("fixture: true\n", encoding="utf-8", newline="\n")
    if not (root / "tiles.csv").exists():
        (root / "tiles.csv").write_text("image_id,source\n", encoding="utf-8", newline="\n")
    lines = [f"{hashlib.sha256((root / m).read_bytes()).hexdigest()}  {m}" for m in _MEMBERS]
    (root / "SNAPSHOT.sha256").write_text("\n".join(lines) + f"\n# snapshot_digest {_digest(lines)}\n",
                                           encoding="utf-8", newline="\n")
    return lines


def _snapshot(root: Path, images, anns) -> Path:
    """eval 행이 없으면 하나 채운다 — 적재기가 'eval 0건' 을 거부하기 때문이다(격리 검사가 공허해진다)."""
    if not any(r["split"] == "eval" for r in images):
        images = [*images, _eval_row()]
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "manifest.csv", M_COLS, images)
    _write_csv(root / "annotations.csv", A_COLS, anns)
    # 정상 픽스처에는 **실제로 계산한 지문**을 싣는다. 자리표 문자열이면 지문 검산이 시험에 들어오지 않는다.
    _write_contract(root)
    return root


@pytest.fixture
def world(tmp_path):
    """이미지 일곱 장: 정상 · 기공 11개 · 균열+기공 · 슬래그+융합불량 · 무효 기하 섞임 · AL 기공 · 폐기 대상. eval 한 장."""
    images = [
        _img("a_normal"),
        _img("b_many", codes="2011", n=11),
        _img("c_crack_poro", client="C2", codes="100;2011", n=2),
        _img("d_slag_lof", split="val", codes="301;401", n=2),
        _img("e_mixed_geom", codes="2011", n=2),
        _img("f_al", client="C3", material="AL", codes="2011", n=1),
        _img("g_discard", codes="100", n=1),
        {**_img("z_eval", split="eval", codes="2011", n=1, group="g-eval"), "client": ""},
    ]
    anns = [_ann("b_many", k, "2011", (k, k, k + 50, k + 60)) for k in range(1, 12)]
    anns += [_ann("c_crack_poro", 1, "2011"), _ann("c_crack_poro", 2, "100", (0, 0, 1280, 720))]
    anns += [_ann("d_slag_lof", 1, "401"), _ann("d_slag_lof", 2, "301")]
    anns += [_ann("e_mixed_geom", 1, "2011", valid=False), _ann("e_mixed_geom", 2, "2011")]
    anns += [_ann("f_al", 1, "2011"), _ann("g_discard", 1, "100", valid=False), _ann("z_eval", 1, "2011")]
    return _snapshot(tmp_path / "snap", images, anns)


def _expected(root: Path, limits: Path = LIMITS) -> dict[str, dict]:
    return {r["image_id"]: r for r in pi.build_expected(root, limits, LABEL_MAP)["records"]}


# ---------------------------------------------------------------- 기대 레코드


def test_ann_id_는_문자열로_정렬한다(world):
    """06번 §2-나 '정렬은 학습 계약이다' — "#10" < "#2". 번호순으로 짜면 11개 이상인 페어에서 순서가 갈린다."""
    rec = _expected(world)["b_many"]
    firsts = [d["bbox_px"][0] for d in rec["defects"]]
    assert firsts == [1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9]


def test_기하_무효는_결함_목록에서_빠지고_개수로_남는다(world):
    rec = _expected(world)["e_mixed_geom"]
    assert [d["bbox_px"] for d in rec["defects"]] == [[10, 20, 110, 220]]
    assert rec[pi.SKIP_COUNT_FIELD] == 1 and rec["discard"] == []


def test_결함_라벨인데_유효_기하가_없으면_폐기한다(world):
    assert _expected(world)["g_discard"]["discard"] == ["defect_label_no_valid_geometry"]
    assert _expected(world)["a_normal"]["discard"] == [] and _expected(world)["a_normal"]["defects"] == []


def test_재질이_다르면_조항이_없다(world):
    """안 B(09-21 판정) — 허용치 표는 강재만 덮는다. AL 결함 페어는 `clauses=[]` 이고 미특정 코드를 적는다."""
    rec = _expected(world)["f_al"]
    assert rec["clauses"] == [] and rec["candidate_rules"] == []
    assert rec["uncited_codes"] == [{"code": "2011", "reason": pi.UNCITED_MATERIAL}]
    assert rec["coord_space"] == "ABS_ORIG"


def test_VT_행은_고르지_않는다(world):
    """표면 기공 행(KRA27-S)은 코드가 같고 두께 구간도 겹친다. 가르는 것은 검사 방식 축 하나다."""
    rec = _expected(world)["e_mixed_geom"]
    assert rec["clauses"] == ["KRA27-T15"]
    assert [r["rule_id"] for r in rec["candidate_rules"]] == [f"KRA27-T15-0{k}" for k in (1, 2, 3, 4)]


def test_두_코드가_한_조항을_가리키면_조항은_하나_행은_여섯(world):
    rec = _expected(world)["d_slag_lof"]
    assert rec["clauses"] == ["KRA27-T16"]
    assert [r["rule_id"] for r in rec["candidate_rules"]] == [f"KRA27-T16-0{k}" for k in range(1, 7)]
    assert [d["type"] for d in rec["defects"]] == ["401", "301"]  # ann_id 순 — 코드순이 아니다


def test_조항은_문자열_오름차순_행은_표의_순서(world):
    rec = _expected(world)["c_crack_poro"]
    assert rec["clauses"] == ["KRA27-3D", "KRA27-T15"]
    assert [r["rule_id"] for r in rec["candidate_rules"]] == \
        ["KRA27-3D-01", *[f"KRA27-T15-0{k}" for k in (1, 2, 3, 4)]]  # rule_id 순 — 표의 행 순서가 아니다
    assert rec["defects"][1]["bbox_px"] == [0, 0, 1280, 720]  # 경계에 닿는 상자는 유효하다(0 ≤ x1 < x2 ≤ W)


def test_서술_값은_원문_폐구간으로_되돌린다(world):
    values = {v["rule_id"]: v for v in _expected(world)["c_crack_poro"]["narrative_values"]}
    assert (values["KRA27-T15-01"]["thickness_gt"], values["KRA27-T15-01"]["thickness_le"]) == (None, "10")
    assert (values["KRA27-T15-02"]["thickness_gt"], values["KRA27-T15-02"]["thickness_le"]) == ("10", "25")
    assert values["KRA27-T15-02"]["limit_value"] == "5" and values["KRA27-T15-02"]["limit_op"] == "le"
    assert values["KRA27-T15-03"]["limit_rule"] == "prop_t" and values["KRA27-T15-03"]["limit_factor"] == "0.2"
    crack = values["KRA27-3D-01"]
    assert (crack["thickness_gt"], crack["thickness_le"], crack["limit_rule"]) == (None, None, "none_permitted")


def test_상한_공란은_무한대다(world):
    values = {v["rule_id"]: v for v in _expected(world)["d_slag_lof"]["narrative_values"]}
    assert (values["KRA27-T16-03"]["thickness_gt"], values["KRA27-T16-03"]["thickness_le"]) == ("50", None)


def test_candidate_rules_의_항목과_두께_표기(world):
    """B 10번 §3 — 키 다섯, 두께는 소수 둘째 자리 문자열, 상한 공란은 null."""
    rules = {r["rule_id"]: r for r in _expected(world)["d_slag_lof"]["candidate_rules"]}
    assert set(rules["KRA27-T16-01"]) == {"rule_id", "clause_id", "defect_code", "thickness_min", "thickness_max"}
    assert (rules["KRA27-T16-01"]["thickness_min"], rules["KRA27-T16-01"]["thickness_max"]) == ("0.00", "12.01")
    assert (rules["KRA27-T16-03"]["thickness_min"], rules["KRA27-T16-03"]["thickness_max"]) == ("50.01", None)
    assert {rules[k]["defect_code"] for k in rules} == {"301", "401"}
    assert {rules[k]["clause_id"] for k in rules} == {"KRA27-T16"}


# ---------------------------------------------------------------- 입력 규약


def _mutated_limits(tmp_path: Path, edit) -> Path:
    with open(LIMITS, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols, rows = reader.fieldnames, list(reader)
    edit(rows)
    out = tmp_path / "limits_mut.csv"
    _write_csv(out, cols, rows)
    return out


def test_모르는_부등호는_예외다(world, tmp_path):
    with pytest.raises(pi.InputContractError, match="모르는 부등호"):
        _expected(world, _mutated_limits(tmp_path, lambda rows: rows[0].update(limit_op="ge")))


def test_scope_는_필터이고_오류가_아니다(world, tmp_path):
    """B 10번 §5 — scope=active 가 아닌 행은 선택에서 빠진다. 그 코드의 조항이 없어지면 미특정으로 남는다."""
    limits = _mutated_limits(tmp_path, lambda rows: [r.update(scope="excluded") for r in rows
                                                     if r["defect_code"] == "2011"])
    rec = _expected(world, limits)["e_mixed_geom"]
    assert rec["clauses"] == [] and rec["candidate_rules"] == []
    assert rec["uncited_codes"] == [{"code": "2011", "reason": pi.UNCITED_CODE}]


def test_선택된_행에_canonical_false_나_quality_축이_있으면_멈춘다(world, tmp_path):
    """선택 축은 아니지만 선택되면 빌드가 멈춘다(B 10번 §5)."""
    with pytest.raises(pi.AmbiguousContract, match="canonical"):
        _expected(world, _mutated_limits(tmp_path, lambda rows: rows[0].update(canonical="false")))
    with pytest.raises(pi.AmbiguousContract, match="quality_scheme"):
        _expected(world, _mutated_limits(tmp_path, lambda rows: rows[0].update(quality_scheme="iso5817")))


def test_고르지_않는_행의_canonical_은_보지_않는다(world, tmp_path):
    """VT 행(KRA27-S)은 선택되지 않으므로 그 행의 canonical·quality 축은 판정에 들지 않는다."""
    limits = _mutated_limits(tmp_path, lambda rows: rows[-1].update(canonical="false", quality_scheme="iso5817"))
    assert _expected(world, limits)["e_mixed_geom"]["clauses"] == ["KRA27-T15"]


def test_재질_ALL_행은_모든_재질에_걸린다(world, tmp_path):
    """B 10번 §5 — 재질 축은 {레코드 재질, ALL}. 지금 표에 ALL 행은 없다."""
    limits = _mutated_limits(tmp_path, lambda rows: [r.update(material="ALL") for r in rows
                                                     if r["defect_code"] == "2011" and r["inspection_method"] == "RT"])
    rec = _expected(world, limits)["f_al"]  # 재질 AL
    assert rec["clauses"] == ["KRA27-T15"] and rec["uncited_codes"] == []


def test_한_코드가_두_조항에_걸리면_멈춘다(world, tmp_path):
    with pytest.raises(pi.AmbiguousContract, match="두 조항"):
        _expected(world, _mutated_limits(tmp_path, lambda rows: rows[-1].update(inspection_method="RT")))


def test_그리드_밖_두께_구간은_멈춘다(world, tmp_path):
    """상한만 그리드 밖으로 옮긴다 — 구간을 겹치게 하면 겹침 검사가 먼저 걸려 이 검사를 못 본다."""
    with pytest.raises(pi.AmbiguousContract, match="0.01 그리드"):
        _expected(world, _mutated_limits(tmp_path, lambda rows: rows[3].update(thickness_max="100.015")))


@pytest.mark.parametrize(("column", "value", "message"), [
    ("has_defect", "true", "불리언"), ("material", "ST ", "앞뒤 공백"), ("material", "", "빈 값"),
    ("iso_codes", "2011.0", "결함 코드"), ("width_px", "1280.0", "정수"), ("split", "test", "split"),
])
def test_매니페스트_표기_이탈은_예외다(tmp_path, column, value, message):
    image = _img("x", codes="2011", n=1)
    image[column] = value
    root = _snapshot(tmp_path / "s", [image], [_ann("x", 1, "2011")])
    with pytest.raises(pi.InputContractError, match=message):
        _expected(root)


@pytest.mark.parametrize(("edit", "error"), [
    ({"iso_code": "2011.0"}, pi.InputContractError), ({"iso_code": "999"}, pi.InputContractError),
    ({"geom_valid": "1"}, pi.InputContractError), ({"bbox_x1_px": "10.0"}, pi.InputContractError),
    ({"ann_id": "y#1"}, pi.InputContractError),
])
def test_어노테이션_표기_이탈은_예외다(tmp_path, edit, error):
    root = _snapshot(tmp_path / "s", [_img("x", codes="2011", n=1)], [{**_ann("x", 1, "2011"), **edit}])
    with pytest.raises(error):
        _expected(root)


def test_유효와_무효의_합이_n_defects_와_다르면_멈춘다(tmp_path):
    """B 10번 §6 — 폐기가 아니라 빌드 중단이다. 본 매니페스트 49,847장은 전부 성립한다."""
    root = _snapshot(tmp_path / "s", [_img("x", codes="2011", n=2)], [_ann("x", 1, "2011")])
    with pytest.raises(pi.InputContractError, match="n_defects"):
        _expected(root)


def test_기하_무효인데_상자가_있으면_멈춘다(tmp_path):
    bad = {**_ann("x", 1, "2011"), "geom_valid": "False"}
    root = _snapshot(tmp_path / "s", [_img("x", codes="2011", n=1)], [bad])
    with pytest.raises(pi.AmbiguousContract):
        _expected(root)


def test_참여자_어휘를_닫는다(tmp_path):
    root = _snapshot(tmp_path / "s", [_img("x", client="C9")], [])
    with pytest.raises(pi.InputContractError, match="참여자 어휘"):
        _expected(root)


@pytest.mark.parametrize("path", ["/abs/a.jpg", "C:/abs/a.jpg", "tiles/../a.jpg"])
def test_상대_경로만_받는다(tmp_path, path):
    root = _snapshot(tmp_path / "s", [_img("x", path=path)], [])
    with pytest.raises(pi.InputContractError, match="경로"):
        _expected(root)


def test_has_defect_와_n_defects_가_어긋나면_멈춘다(tmp_path):
    bad = {**_img("x"), "has_defect": "True", "n_defects": "0"}
    root = _snapshot(tmp_path / "s", [bad], [])
    with pytest.raises(pi.InputContractError, match="has_defect"):
        _expected(root)


def test_eval_행이_0건이면_거부한다(tmp_path):
    """격리 검사가 공허해진다 — 겹칠 것이 없으면 겹침 0 은 아무것도 말하지 않는다."""
    root = tmp_path / "noeval"
    root.mkdir()
    _write_csv(root / "manifest.csv", M_COLS, [_img("x")])
    _write_csv(root / "annotations.csv", A_COLS, [])
    _write_contract(root)
    with pytest.raises(pi.InputContractError, match="eval 행이 0건"):
        _expected(root)


def test_두께_구간이_겹치면_멈춘다(world, tmp_path):
    """한 두께에 기준이 둘이면 서술이 모순된 기준을 나열한다."""
    limits = _mutated_limits(tmp_path, lambda rows: rows[1].update(thickness_min="0"))
    with pytest.raises(pi.AmbiguousContract, match="두께 구간이 겹친다"):
        _expected(world, limits)


def test_계약_해시가_다르면_읽지_않는다(world):
    with open(world / "manifest.csv", "a", encoding="utf-8", newline="") as f:
        f.write("\n")
    with pytest.raises(pi.InputContractError, match="SNAPSHOT"):
        _expected(world)


def test_eval_은_식별자만_남고_색인에_들지_않는다(world):
    images, eval_keys = pi.load_manifest(world)
    assert "z_eval" not in images
    assert eval_keys["image_id"] == {"z_eval"} and eval_keys["group_id"] == {"g-eval"}
    assert set(eval_keys) == set(pi.ISOLATION_AXES)  # 네 축 전부 남긴다
    index, dropped = pi.load_annotations(world, images, eval_keys, pi.load_label_codes(LABEL_MAP))
    assert dropped == 1 and "z_eval" not in index
    assert "z_eval" not in _expected(world)


def test_eval_행에_참여자가_붙어_있으면_예외다(tmp_path):
    root = _snapshot(tmp_path / "s", [_img("z", split="eval", client="C1")], [])
    with pytest.raises(pi.InputContractError, match="eval 행에 client"):
        _expected(root)


def test_사상표의_코드를_모듈에_적어_두지_않는다():
    """불변조건 1-8. 코드 집합은 사상표에서 읽는다 — 모듈 본문에 네 코드의 문자열 상수가 없어야 한다."""
    consts = {n.value for n in ast.walk(ast.parse(Path(pi.__file__).read_text(encoding="utf-8")))
              if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not consts & pi.load_label_codes(LABEL_MAP)


def test_빌더와_프로젝트_적재기를_import_하지_않는다():
    """독립성의 전제. 표준 라이브러리와 PyYAML 만 쓴다."""
    tree = ast.parse(Path(pi.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    assert roots <= {"__future__", "argparse", "csv", "hashlib", "itertools", "json", "operator", "re", "sys",
                     "collections", "dataclasses", "decimal", "pathlib", "yaml"}
    # 정적 import 만 보면 동적 적재가 지나간다(검토 12번 m-1). 이름과 속성에서 그 경로도 막는다.
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not used & {"__import__", "import_module", "exec_module", "load_module", "SourceFileLoader", "eval", "exec"}


def _row(image_id="a", group="g-a", path="p/a.jpg", sha="a" * 64):
    return pi.ImageRow(image_id=image_id, split="train", client="C1", material="ST", rel_path=path,
                       path_key=pi._path_key(path), sha256=sha, width_px=1280, height_px=720,
                       has_defect=False, n_defects=0, iso_codes=(), group_id=group)


def _keys(image_id="z", group="g-z", path="p/z.jpg", sha="z" * 64):
    return {"image_id": frozenset({image_id}), "group_id": frozenset({group}),
            "rel_path": frozenset({path}), "sha256": frozenset({sha})}


def test_eval_격리_단언이_실제로_물_수_있다():
    """같은 함수에서 걸러낸 것을 다시 세면 실패할 수 없는 검사가 된다(검토 12번 I-2). 함수 경계를 넘겨 두었다."""
    pi.assert_isolated([_row()], _keys())  # 겹치지 않으면 조용하다
    with pytest.raises(AssertionError, match="image_id 축에서 1건"):
        pi.assert_isolated([_row(image_id="z")], _keys())


@pytest.mark.parametrize(("axis", "row"), [
    ("group_id", _row(group="g-z")), ("rel_path", _row(path="p/z.jpg")), ("sha256", _row(sha="z" * 64)),
])
def test_id_가_달라도_다른_축에서_겹치면_잡는다(axis, row):
    """같은 파일이 다른 id 로 양쪽에 있으면 id 축만으로는 겹침 0 이 나온다."""
    with pytest.raises(AssertionError, match=f"{axis} 축에서 1건"):
        pi.assert_isolated([row], _keys())


# ---------------------------------------------------------------- 대조기


def _criterion(v: dict) -> str:
    head = "모든 두께" if v["thickness_gt"] is None and v["thickness_le"] is None else " ".join(
        x for x in (f"두께 {v['thickness_gt']} mm 초과" if v["thickness_gt"] else "",
                    f"{v['thickness_le']} mm 이하" if v["thickness_le"] else "") if x)
    if v["limit_rule"] == "none_permitted":
        return f"{head}: 허용하지 않는다"
    body = f"{v['limit_value']} mm" if v["limit_rule"] == "const" else f"모재 두께의 {v['limit_factor']} 배"
    return f"{head}: {body} {'이하' if v['limit_op'] == 'le' else '미만'}"


def _pair(e: dict) -> dict:
    """시험이 직접 짠 레코드 작성기 — 06번이 적은 본실험 레코드 꼴. 빌더의 코드가 아니다."""
    seen, criteria = set(), []
    for v in e["narrative_values"]:
        text = _criterion(v)
        if text not in seen:
            seen.add(text)
            criteria.append(text)
    text = f"조항 {' '.join(e['clauses'])} 의 기준은 {' / '.join(criteria)} 이다." if criteria else "인용할 조항이 없다."
    rec = {"image_id": e["image_id"], "image_path": e["image_path"], "client": e["client"], "split": e["split"],
           "material": e["material"], "width_px": e["width_px"], "height_px": e["height_px"],
           "coord_space": e["coord_space"],
           "skeleton": {"defects": [{**d, "size_mm": None} for d in e["defects"]], "verdict": None,
                        "verdict_mode": "clause_only", "clauses": e["clauses"],
                        "candidate_rules": [dict(r) for r in e["candidate_rules"]],
                        "uncited_codes": [dict(x) for x in e["uncited_codes"]]},
           "target_text": text}
    if e[pi.SKIP_COUNT_FIELD]:  # 0 이면 키가 없다(B 10번 §1)
        rec[pi.SKIP_COUNT_FIELD] = e[pi.SKIP_COUNT_FIELD]
    return rec


def _build(world: Path, out: Path, mutate=None, *, newline="\n", discarded=True, meta=None) -> Path:
    expected = pi.build_expected(world, LIMITS, LABEL_MAP)["records"]
    pairs = [_pair(e) for e in expected if not e["discard"]]
    drops = [{"image_id": e["image_id"], "reasons": e["discard"]} for e in expected if e["discard"]]
    if mutate:
        pairs, drops = mutate(pairs, drops) or (pairs, drops)
    out.mkdir(parents=True)
    (out / "pairs.jsonl").write_bytes("".join(json.dumps(p, ensure_ascii=False) + newline for p in pairs).encode())
    if discarded:
        (out / "discarded.jsonl").write_bytes("".join(json.dumps(d) + "\n" for d in drops).encode())
    if meta is not None:
        (out / "PAIRS_META.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return out


def _meta(world: Path, **over) -> dict:
    """빌더가 낼 메타를 시험이 직접 짠다. 총계는 폐기된 이미지 몫을 포함한다."""
    recs = pi.build_expected(world, LIMITS, LABEL_MAP)["records"]
    kept = [r for r in recs if not r["discard"]]
    by_split: dict[str, dict[str, dict[str, int]]] = {}
    for r in kept:
        cell = by_split.setdefault(r["split"], {}).setdefault(
            r["client"], {"n_total": 0, "n_defect": 0, "n_normal": 0})
        cell["n_total"] += 1
        cell["n_defect" if r["defects"] else "n_normal"] += 1
    body = [x for x in (world / "SNAPSHOT.sha256").read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.startswith("#")]
    return {"counts_by_split": by_split,
            pi.META_SNAPSHOT: {"path": world.name, "snapshot_digest": _digest(body),
                               "splits_used": ["train", "val"]},
            pi.META_LIMITS: {"path": str(LIMITS), "sha256": hashlib.sha256(LIMITS.read_bytes()).hexdigest()},
            pi.META_VALIDATED_BY: "rule",
            pi.SKIP_COUNT_FIELD: sum(r[pi.SKIP_COUNT_FIELD] for r in recs),
            pi.META_SKIP_IN_RECORDS: sum(r[pi.SKIP_COUNT_FIELD] for r in kept),
            pi.META_PATHS_CHECKED: {"n": len(recs), "of": len(recs), "basis": "저장소 루트 기준"}, **over}


def _counts(world: Path) -> dict:
    recs = pi.build_expected(world, LIMITS, LABEL_MAP)["records"]
    kept = [r for r in recs if not r["discard"]]
    per: dict[str, dict[str, int]] = {}
    for r in kept:
        cell = per.setdefault(r["client"], {"n_total": 0, "n_defect": 0, "n_normal": 0})
        cell["n_total"] += 1
        cell["n_defect" if r["defects"] else "n_normal"] += 1
    return {"clients": per, "discarded": {"quarantine": len(recs) - len(kept)}}


def _full(world: Path, out: Path, mutate=None, *, meta=None, **kw) -> Path:
    """`counts.json` 까지 갖춘 산출 — 전 범위 대조가 되는 상태(봉인 소비 경로가 기대하는 꼴)."""
    pairs = _build(world, out, mutate, meta=_meta(world) if meta is None else meta, **kw)
    (pairs / "counts.json").write_text(json.dumps(_counts(world), ensure_ascii=False), encoding="utf-8")
    return pairs


def _kinds(world: Path, pairs_dir: Path) -> dict[str, int]:
    return pi.compare(pi.build_expected(world, LIMITS, LABEL_MAP), pairs_dir)["mismatch"]


def _by_id(pairs, image_id):
    return next(p for p in pairs if p["image_id"] == image_id)


def test_같은_규약으로_짠_산출은_전_레코드가_일치한다(world, tmp_path):
    out = tmp_path / "report"
    pairs = _full(world, tmp_path / "pairs")  # 메타와 회계까지 갖춘 산출이라야 "일치" 다
    code = pi.main(["--manifest-dir", str(world), "--limits", str(LIMITS), "--label-map", str(LABEL_MAP),
                    "--pairs", str(pairs), "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert code == 0 and summary["mismatch"] == {} and summary["verdict"] == "일치"
    assert summary["coverage"] == "full" and summary["not_compared"] == []
    assert summary["n_expected_keep"] == 6 and summary["n_expected_discard"] == 1 and summary["n_got"] == 6
    assert summary["expected_table"] == summary["got_table"]
    assert b"\r" not in (out / "expected.jsonl").read_bytes()


def _m_bbox(pairs, drops):
    _by_id(pairs, "e_mixed_geom")["skeleton"]["defects"][0]["bbox_px"][2] += 1


def _m_order(pairs, drops):
    _by_id(pairs, "b_many")["skeleton"]["defects"].sort(key=lambda d: d["bbox_px"][0])  # 번호순으로 '고친' 빌더


def _m_type(pairs, drops):
    _by_id(pairs, "d_slag_lof")["skeleton"]["defects"][0]["type"] = "301"


def _m_type_int(pairs, drops):
    for d in _by_id(pairs, "b_many")["skeleton"]["defects"]:
        d["type"] = int(d["type"])


def _m_clause_order(pairs, drops):
    _by_id(pairs, "c_crack_poro")["skeleton"]["clauses"].reverse()


def _m_al_steel_clause(pairs, drops):
    _by_id(pairs, "f_al")["skeleton"]["clauses"] = ["KRA27-T15"]  # 파일럿 v1 의 219건 양식


def _m_missing(pairs, drops):
    return [p for p in pairs if p["image_id"] != "a_normal"], drops


def _m_eval_leak(pairs, drops):
    pairs.append({**_by_id(pairs, "a_normal"), "image_id": "z_eval"})


def _m_client(pairs, drops):
    _by_id(pairs, "c_crack_poro")["client"] = "C1"


def _m_split(pairs, drops):
    _by_id(pairs, "d_slag_lof")["split"] = "train"


def _m_material(pairs, drops):
    _by_id(pairs, "f_al")["material"] = "ST"


def _m_width(pairs, drops):
    _by_id(pairs, "a_normal")["width_px"] = 1024


def _m_unsorted(pairs, drops):
    pairs.reverse()


def _m_duplicate(pairs, drops):
    pairs.insert(1, dict(pairs[0]))


def _m_kept_discard(pairs, drops):
    pairs.append({**_by_id(pairs, "a_normal"), "image_id": "g_discard"})
    return pairs, []


def _m_rule_value(pairs, drops):
    _by_id(pairs, "e_mixed_geom")["skeleton"]["candidate_rules"][1]["thickness_max"] = "25.00"


def _m_rule_order(pairs, drops):
    _by_id(pairs, "c_crack_poro")["skeleton"]["candidate_rules"].reverse()


def _m_verdict(pairs, drops):
    _by_id(pairs, "c_crack_poro")["skeleton"]["verdict"] = "reject"


def _m_skip_count(pairs, drops):
    _by_id(pairs, "e_mixed_geom")[pi.SKIP_COUNT_FIELD] = 0


def _m_text_value(pairs, drops):
    p = _by_id(pairs, "e_mixed_geom")
    p["target_text"] = p["target_text"].replace(": 4 mm 이하", ": 5 mm 이하")


def _m_text_criteria_deleted(pairs, drops):
    """06번 §2-나의 반례 — 기준 문장을 지우고 두께 구간 머리만 남긴 서술. 파일럿 게이트 ③은 이것을 통과시켰다."""
    _by_id(pairs, "e_mixed_geom")["target_text"] = (
        "조항 KRA27-T15 의 기준은 두께 10 mm 이하 / 두께 10 mm 초과 25 mm 이하 / 두께 25 mm 초과 50 mm 이하 / "
        "두께 50 mm 초과 100 mm 이하 이다.")


def _m_text_op(pairs, drops):
    p = _by_id(pairs, "e_mixed_geom")
    p["target_text"] = p["target_text"].replace(": 4 mm 이하", ": 4 mm 미만")


def _m_skip_zero_key(pairs, drops):
    _by_id(pairs, "a_normal")[pi.SKIP_COUNT_FIELD] = 0  # 0 이면 키가 없어야 한다


def _m_thickness_number(pairs, drops):
    for r in _by_id(pairs, "e_mixed_geom")["skeleton"]["candidate_rules"]:
        r["thickness_min"] = float(r["thickness_min"])  # 값은 같고 표기만 수다


def _m_uncited_reason(pairs, drops):
    _by_id(pairs, "f_al")["skeleton"]["uncited_codes"] = [{"code": "2011", "reason": pi.UNCITED_CODE}]


def _m_field_place(pairs, drops):
    p = _by_id(pairs, "a_normal")
    p["skeleton"]["material"] = p.pop("material")  # 자리를 옮긴다


def _m_key_order(pairs, drops):
    p = _by_id(pairs, "a_normal")
    moved = {"skeleton": p.pop("skeleton"), "material": p.pop("material")}
    p.update(moved)  # skeleton 이 material 앞으로 온다


def _m_text_null(pairs, drops):
    for p in pairs:
        p["target_text"] = None  # 서술을 하나도 내지 않은 빌드


def _m_text_missing(pairs, drops):
    for p in pairs:
        p.pop("target_text")


def _m_defect_keys(pairs, drops):
    d = _by_id(pairs, "e_mixed_geom")["skeleton"]["defects"][0]
    d["size_mm"] = d.pop("size_mm")  # 키를 끝으로 다시 넣어도 순서는 같다 — 실제로는 size_px 를 뒤로 보낸다
    d["size_px"] = d.pop("size_px")


def _m_size_mm(pairs, drops):
    _by_id(pairs, "e_mixed_geom")["skeleton"]["defects"][0]["size_mm"] = {"major_axis": 1.0}


def _m_coord_space(pairs, drops):
    _by_id(pairs, "a_normal")["coord_space"] = "NORM_1000"


@pytest.mark.parametrize(("mutate", "kind"), [
    (_m_bbox, "defect_bbox"), (_m_order, "defect_order"), (_m_type, "defect_type"), (_m_type_int, "defect_value_type"),
    (_m_clause_order, "clauses_order"), (_m_al_steel_clause, "clauses_set"), (_m_missing, "missing_record"),
    (_m_eval_leak, "eval_image_in_pairs"), (_m_client, "client"), (_m_split, "split"), (_m_material, "material"),
    (_m_width, "width_px"), (_m_unsorted, "file_order"), (_m_duplicate, "duplicate_image_id"),
    (_m_kept_discard, "unexpected_record"), (_m_rule_value, "candidate_rules_value"),
    (_m_rule_order, "candidate_rules_order"), (_m_verdict, "verdict_not_null"), (_m_skip_count, "skipped_count"),
    (_m_text_value, "narrative_value_missing"), (_m_text_criteria_deleted, "narrative_value_missing"),
    (_m_text_op, "narrative_op_lt_count"), (_m_skip_zero_key, "skipped_count_zero_key_present"),
    (_m_thickness_number, "candidate_rules_thickness_type"), (_m_uncited_reason, "uncited_codes_set"),
    (_m_field_place, "field_place:material"), (_m_key_order, "key_order"), (_m_coord_space, "coord_space"),
    (_m_defect_keys, "defect_item_keys"), (_m_size_mm, "defect_size_mm_not_null"),
    (_m_text_null, "field_absent:target_text"), (_m_text_missing, "field_absent:target_text"),
])
def test_깨뜨린_산출은_그_범주로_잡힌다(world, tmp_path, mutate, kind):
    assert kind in _kinds(world, _build(world, tmp_path / "pairs", mutate))


def test_폐기해야_할_이미지를_남기면_폐기_누락도_함께_센다(world, tmp_path):
    kinds = _kinds(world, _build(world, tmp_path / "pairs", _m_kept_discard))
    assert kinds["unexpected_record"] == 1 and kinds["discard_missing"] == 1


def test_eval_묶음이_섞이면_이미지와_별도로_센다(world, tmp_path):
    """같은 묶음의 다른 이미지가 train 에 있는 경우 — 이미지 교차는 0 인데 묶음 교차가 1 이다."""
    rows = _read_csv(world / "manifest.csv")
    for r in rows:
        if r["image_id"] == "a_normal":
            r["group_id"] = "g-eval"
    anns = _read_csv(world / "annotations.csv")
    root = _snapshot(world.parent / "snap2", rows, anns)
    with pytest.raises(AssertionError, match="group_id 축에서 1건"):
        _expected(root)  # 적재 단계에서 이미 걸린다 — 산출을 만들 필요가 없다


def test_CRLF_로_쓴_산출을_잡는다(world, tmp_path):
    assert _kinds(world, _build(world, tmp_path / "pairs", newline="\r\n")) == {"crlf_in_pairs": 1}


def test_폐기_목록이_없으면_그것도_불일치다(world, tmp_path):
    kinds = _kinds(world, _build(world, tmp_path / "pairs", discarded=False))
    assert kinds == {"discarded_file_absent": 1, "discard_missing": 1}


def test_파일럿_꼴의_레코드는_없는_필드로_센다(world, tmp_path):
    """봉인본 `pairs_pilot_v1` 에는 06번이 더하기로 한 네 필드와 무효 기하의 개수가 없다 — 예상 불일치."""
    def strip(pairs, drops):
        for p in pairs:
            for name in ("material", "width_px", "height_px", "coord_space", pi.SKIP_COUNT_FIELD):
                p.pop(name, None)
            for name in ("candidate_rules", "uncited_codes"):
                p["skeleton"].pop(name)

    kinds = _kinds(world, _build(world, tmp_path / "pairs", strip))
    assert kinds == {"field_absent:candidate_rules": 6, "field_absent:coord_space": 6,
                     "field_absent:height_px": 6, "field_absent:material": 6, "field_absent:uncited_codes": 6,
                     "field_absent:width_px": 6, "skipped_count_unrecorded": 1}


def test_메타의_기하_무효_두_수를_대조한다(world, tmp_path):
    """메타의 같은 이름은 빌드 총계이고 레코드의 것은 이미지 하나의 수다 — 층을 섞지 않는다."""
    meta = _meta(world)
    assert meta[pi.SKIP_COUNT_FIELD] == 2 and meta[pi.META_SKIP_IN_RECORDS] == 1  # 폐기 이미지 몫 1
    assert _kinds(world, _full(world, tmp_path / "ok", meta=meta)) == {}


@pytest.mark.parametrize("key", [pi.SKIP_COUNT_FIELD, pi.META_SKIP_IN_RECORDS])
def test_메타의_수가_틀리면_잡는다(world, tmp_path, key):
    meta = _meta(world)
    meta[key] += 1
    assert f"meta:{key}" in _kinds(world, _build(world, tmp_path / f"bad_{key}", meta=meta))


def test_경로_검사_주장의_건수를_내_모집단과_맞댄다(world, tmp_path):
    """파일시스템을 다시 보지 않는다. 빌드가 \"몇 개 보았다\" 고 적은 수를 내가 따로 센 수와 맞댄다."""
    meta = _meta(world)
    assert meta[pi.META_PATHS_CHECKED] == {"n": 7, "of": 7, "basis": "저장소 루트 기준"}  # 남는 6 + 폐기 1
    assert _kinds(world, _full(world, tmp_path / "ok", meta=meta)) == {}


@pytest.mark.parametrize("claim", [
    {"n": 0, "of": 7, "basis": "x"},      # 검사를 끄고 빌드했다
    {"n": 7, "of": 6, "basis": "x"},      # 모집단이 내 셈과 다르다
    {"n": 3, "of": 3, "basis": "x"},      # 일부만 보았다
])
def test_경로_검사_주장이_어긋나면_잡는다(world, tmp_path, claim):
    meta = _meta(world, **{pi.META_PATHS_CHECKED: claim})
    kinds = _kinds(world, _build(world, tmp_path / f"bad{claim['n']}{claim['of']}", meta=meta))
    assert f"meta:{pi.META_PATHS_CHECKED}" in kinds


def test_참거짓만_싣던_옛_메타는_불일치가_아니라_주석이다(world, tmp_path):
    """건수를 싣기 전의 빌드와도 돌아야 한다 — 다만 그 주장은 대조하지 않았다고 적는다."""
    meta = _meta(world, **{pi.META_PATHS_CHECKED: True})
    result = pi.compare(pi.build_expected(world, LIMITS, LABEL_MAP), _full(world, tmp_path / "old", meta=meta))
    assert result["mismatch"] == {} and "건수를 싣지 않는다" in result["meta_note"]
    assert result["not_compared"] == [f"meta.{pi.META_PATHS_CHECKED}"]  # 산문 말고 목록에도 남는다


def test_대조하지_못한_축은_기계가_읽는_목록에_남는다(world, tmp_path):
    """못 본 것을 통과로 세지 않는다. 다만 실패로도 세지 않는다 — 둘은 다르다."""
    full = pi.compare(pi.build_expected(world, LIMITS, LABEL_MAP), _full(world, tmp_path / "full"))
    assert full["not_compared"] == []
    none = pi.compare(pi.build_expected(world, LIMITS, LABEL_MAP), _build(world, tmp_path / "none"))
    assert none["mismatch"] == {} and set(none["not_compared"]) == {
        "counts.clients", "counts.discarded.quarantine",
        "meta.counts_by_split", f"meta.{pi.SKIP_COUNT_FIELD}", f"meta.{pi.META_SKIP_IN_RECORDS}",
        f"meta.{pi.META_PATHS_CHECKED}", f"meta.{pi.META_SNAPSHOT}.snapshot_digest",
        f"meta.{pi.META_LIMITS}.sha256", f"meta.{pi.META_VALIDATED_BY}"}


def test_메타가_없으면_대조하지_않았다고_적는다(world, tmp_path):
    """산문만 보면 못 본 범위를 세지 못한다 — 메타 부재는 **일곱 축**이다(교차 검수 I-4)."""
    result = pi.compare(pi.build_expected(world, LIMITS, LABEL_MAP), _build(world, tmp_path / "nometa"))
    assert result["mismatch"] == {} and "PAIRS_META.json 이 없다" in result["meta_note"]
    meta_axes = [x for x in result["not_compared"] if x.startswith("meta.")]
    assert len(meta_axes) == 7, meta_axes  # 메타 블록이 대조하는 축의 수와 같아야 한다


def test_allow_는_적어_준_범주만_넘긴다(world, tmp_path):
    """허용해도 **0 은 주지 않는다** — 봉인 여부는 종료 코드 하나로 읽혀야 한다."""
    pairs = _full(world, tmp_path / "pairs", _m_al_steel_clause)
    base = ["--manifest-dir", str(world), "--limits", str(LIMITS), "--label-map", str(LABEL_MAP), "--pairs", str(pairs)]
    assert pi.main([*base, "--out", str(tmp_path / "r1")]) == pi.EXIT_MISMATCH
    assert pi.main([*base, "--out", str(tmp_path / "r2"), "--allow", "clauses_set"]) == pi.EXIT_ALLOWED
    assert pi.main([*base, "--out", str(tmp_path / "r3"), "--allow", "defect_bbox"]) == pi.EXIT_MISMATCH


def test_본_것만_같았을_때_일치라고_부르지_않는다(world, tmp_path):
    """`verdict == "일치"` 는 봉인 조건이다. 범위가 좁으면 그 이름을 주지 않는다."""
    base = ["--manifest-dir", str(world), "--limits", str(LIMITS), "--label-map", str(LABEL_MAP)]
    full = tmp_path / "r_full"
    assert pi.main([*base, "--pairs", str(_full(world, tmp_path / "p1")), "--out", str(full)]) == pi.EXIT_MATCH
    got = json.loads((full / "summary.json").read_text(encoding="utf-8"))
    assert got["verdict"] == "일치" and got["coverage"] == "full"

    partial = tmp_path / "r_partial"
    assert pi.main([*base, "--pairs", str(_build(world, tmp_path / "p2")),
                    "--out", str(partial)]) == pi.EXIT_PARTIAL
    got = json.loads((partial / "summary.json").read_text(encoding="utf-8"))
    assert got["mismatch"] == {} and got["coverage"] == "partial"
    assert got["verdict"] != "일치" and "대조 범위 제한" in got["verdict"]


def _run(world: Path, pairs: Path, out: Path, allow: str = "") -> tuple[int, dict]:
    argv = ["--manifest-dir", str(world), "--limits", str(LIMITS), "--label-map", str(LABEL_MAP),
            "--pairs", str(pairs), "--out", str(out)]
    if allow:
        argv += ["--allow", allow]
    code = pi.main(argv)
    return code, json.loads((out / "summary.json").read_text(encoding="utf-8"))


def test_봉인_소비_경로는_종료_코드로_상태를_가른다(world, tmp_path):
    """`0` 은 봉인해도 되는 상태 하나뿐이다. 부분 대조와 허용 불일치에 0 을 주지 않는다."""
    code, got = _run(world, _full(world, tmp_path / "p_full"), tmp_path / "r_full")
    assert (code, got["verdict"], got["sealable"]) == (pi.EXIT_MATCH, "일치", True)

    code, got = _run(world, _build(world, tmp_path / "p_part"), tmp_path / "r_part")
    assert code == pi.EXIT_PARTIAL and got["sealable"] is False and got["mismatch"] == {}

    pairs = _full(world, tmp_path / "p_allow", _m_al_steel_clause)
    code, got = _run(world, pairs, tmp_path / "r_allow", allow="clauses_set")
    assert code == pi.EXIT_ALLOWED and got["sealable"] is False and got["blocking"] == {}

    pairs = _full(world, tmp_path / "p_bad", _m_bbox)
    code, got = _run(world, pairs, tmp_path / "r_bad")
    assert code == pi.EXIT_MISMATCH and got["sealable"] is False


def test_필수_파일이_없으면_미대조_목록에_넣는다(world, tmp_path):
    """없는 것을 조용히 넘기면 \"봤는데 같았다\" 와 구별되지 않는다."""
    pairs = _build(world, tmp_path / "p_nocounts", meta=_meta(world))  # counts.json 을 쓰지 않는다
    code, got = _run(world, pairs, tmp_path / "r_nocounts")  # counts.json 을 쓰지 않는 도우미다
    assert code == pi.EXIT_PARTIAL
    assert set(got["not_compared"]) == {"counts.clients", "counts.discarded.quarantine"}
    assert got["mismatch"] == {} and "counts.json 이 없다" in got["counts_note"]


# ---------------------------------------------------------------- 고정 소스 회귀 (검토 §27-19)


def test_정상_빈_폐기_목록은_통과시킨다(world, tmp_path):
    """검토 §27-19 — 폐기가 0건인 빌드에서 빈 파일을 거부하면 안 된다. 그때 E 는 통과시켰고 그 판정을 고정한다."""
    rows = _read_csv(world / "manifest.csv")
    rows = [r for r in rows if r["image_id"] != "g_discard"]  # 폐기 대상이 없는 세계
    anns = [a for a in _read_csv(world / "annotations.csv") if a["image_id"] != "g_discard"]
    clean = _snapshot(tmp_path / "clean", rows, anns)
    pairs = _full(clean, tmp_path / "p_empty")
    (pairs / "discarded.jsonl").write_bytes(b"")  # 줄바꿈 없는 빈 파일
    code, got = _run(clean, pairs, tmp_path / "r_empty")
    assert (code, got["verdict"]) == (pi.EXIT_MATCH, "일치")


def test_근거_없는_폐기는_잡는다(world, tmp_path):
    """검토 §27-19 — 정상 1건을 폐기하고 회계를 맞춰도 기대와 어긋난다."""
    def drop_one(pairs, drops):
        victim = _by_id(pairs, "a_normal")
        return [p for p in pairs if p is not victim], [*drops, {"image_id": "a_normal", "reasons": ["made_up"]}]

    code, got = _run(world, _full(world, tmp_path / "p_drop", drop_one), tmp_path / "r_drop")
    assert code == pi.EXIT_MISMATCH
    assert {"missing_record", "discard_unexpected"} <= set(got["mismatch"])


@pytest.mark.parametrize("case", ["digest", "limits_hash_missing", "validated_by"])
def test_메타_선언_세_변이를_원천과_맞댄다(world, tmp_path, case):
    """검토 §27-19 — 이 셋을 그때 내 대조기가 **전부 통과시켰다.** 판정을 뒤집고 고정한다.

    `validated_by` 는 거부만 한다. 사람 검수가 없었음을 증명하는 것이 아니라, 빌드 산출물만으로는
    그 주장을 확인할 수 없으니 봉인을 멈춘다는 뜻이다.
    """
    meta = _meta(world)
    if case == "digest":
        meta[pi.META_SNAPSHOT] = {**meta[pi.META_SNAPSHOT], "snapshot_digest": "0" * 64}
        want = f"meta:{pi.META_SNAPSHOT}.snapshot_digest"
    elif case == "limits_hash_missing":
        meta[pi.META_LIMITS] = {k: v for k, v in meta[pi.META_LIMITS].items() if k != "sha256"}
        want = f"meta:{pi.META_LIMITS}.sha256"
    else:
        meta[pi.META_VALIDATED_BY] = "human_double_checked"
        want = f"meta:{pi.META_VALIDATED_BY}"
    code, got = _run(world, _full(world, tmp_path / f"p_{case}", meta=meta), tmp_path / f"r_{case}")
    assert code == pi.EXIT_MISMATCH and got["sealable"] is False
    assert list(got["mismatch"]) == [want]  # 한 변이가 다른 범주까지 켜지 않는다


def test_원천에_digest_줄이_없으면_미대조로_남긴다(world, tmp_path):
    """맞댈 원천이 없는 것과 맞대 봤더니 같은 것을 가른다."""
    snap = world / "SNAPSHOT.sha256"
    kept = [x for x in snap.read_text(encoding="utf-8").splitlines() if not x.startswith("#")]
    meta = _meta(world)
    snap.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
    code, got = _run(world, _full(world, tmp_path / "p_nodigest", meta=meta), tmp_path / "r_nodigest")
    assert code == pi.EXIT_PARTIAL and got["mismatch"] == {}
    assert f"meta.{pi.META_SNAPSHOT}.snapshot_digest" in got["not_compared"]


# ---------------------------------------------------------------- 입력 지문 (검토 E-1)

#: 동결본 두 곳의 계약서 줄과 기록된 지문 — 파일에서 옮긴 텍스트다. 이 시험은 봉인 자산을 읽지 않는다.
_FROZEN_CONTRACTS = [
    ([("436f4cbea46f954cb4a81651ffd3e854aed9f4df7b2e9da0b499abf88d05ddbe", "manifest.csv"),
      ("1b56b14658018c317844c948fb725105bc4f448a0ab9c8f21caaa4db31538cb1", "annotations.csv"),
      ("360319f2c30ecf3640b9094f1c90fa97032d3535ebcd1e1ef72030fdc01f7e02", "data_capabilities.yaml"),
      ("35f80ea494f40126de03b9d90ca43d1e48f262e23892525091655b7ce6ac0115", "tiles.csv")],
     "1f80e98b151372e3fa44c754d0c5ab2cf06bd731d3dfdb8f2515d05383834899"),
    ([("ac2ca81f5d2f98452c50b80f37ff87ecd41ad6fe0c938e4baa783028496779a0", "manifest.csv"),
      ("b4382ce53c80ec4cc4b9cc122ce7438a5ea2c43b5367601b0a7b4888014f9eff", "annotations.csv"),
      ("124f854940d5bb4c700bf64e33c056b165bfe3ddbeb113dc30c1be4ae1e35eda", "data_capabilities.yaml"),
      ("a8d8db063c928fc7b91bad2984e5300da4f5eb227efdde271ed30ddfd3a698f9", "tiles.csv")],
     "c0a254a83020a28caef2d40f70684a775e06e9091042820d3487e1cf1b52f1e1"),
]


@pytest.mark.parametrize("rows, recorded", _FROZEN_CONTRACTS, ids=["manifest_v1", "pilot3000"])
def test_지문을_다시_계산하면_동결본의_기록과_같다(rows, recorded):
    """규칙을 문서가 적지 않아 계약서 줄과 기록값으로 정했다. 두 동결본에서 같은 규칙이 기록값을 낸다."""
    assert pi.recompute_snapshot_digest({name: h for h, name in rows}) == recorded


def test_계약서에_적힌_순서는_지문에_들지_않는다():
    """지문은 **정해진 순서**로 계산한다. 이름순으로 넣어도 기록값이 나온다(검수 I-1).

    앞 판에는 반대로 '적힌 순서가 지문에 든다' 를 확인하는 시험이 있었다 — 동결본 두 곳이 모두 정해진
    순서로 적혀 있어 두 이해가 구별되지 않았다."""
    rows, recorded = _FROZEN_CONTRACTS[0]
    assert pi.recompute_snapshot_digest({name: h for h, name in sorted(rows, key=lambda r: r[1])}) == recorded


def _rewrite_digest(world: Path, value: str) -> None:
    snap = world / "SNAPSHOT.sha256"
    body = [x for x in snap.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    snap.write_text("\n".join(body) + f"\n# snapshot_digest {value}\n", encoding="utf-8", newline="\n")


def test_계약서의_지문만_바꾸면_읽지_않는다(world):
    """파일별 해시는 그대로이고 적힌 지문만 틀렸다 — 계약서가 스스로 어긋난다."""
    _rewrite_digest(world, "0" * 64)
    with pytest.raises(pi.InputContractError, match="다시 계산한"):
        _expected(world)


def test_계약서와_메타를_같은_오기로_바꿔도_잡는다(world, tmp_path):
    """검토 E-1 의 반례. 적힌 문자열끼리 맞대면 둘이 같아 통과한다 — 다시 계산해야 걸린다."""
    meta = _meta(world)
    wrong = "f" * 64
    meta[pi.META_SNAPSHOT] = {**meta[pi.META_SNAPSHOT], "snapshot_digest": wrong}
    pairs = _full(world, tmp_path / "p_same_wrong", meta=meta)
    _rewrite_digest(world, wrong)
    with pytest.raises(pi.InputContractError, match="다시 계산한"):
        pi.main(["--manifest-dir", str(world), "--limits", str(LIMITS), "--label-map", str(LABEL_MAP),
                 "--pairs", str(pairs), "--out", str(tmp_path / "r_same_wrong")])


def test_메타의_지문만_바꾸면_불일치다(world, tmp_path):
    """계약서는 스스로 맞고 빌드의 선언만 틀렸다 — 입력은 믿을 수 있고 빌드가 어긋난 것이다."""
    meta = _meta(world)
    meta[pi.META_SNAPSHOT] = {**meta[pi.META_SNAPSHOT], "snapshot_digest": "0" * 64}
    code, got = _run(world, _full(world, tmp_path / "p_meta_only", meta=meta), tmp_path / "r_meta_only")
    assert code == pi.EXIT_MISMATCH
    assert list(got["mismatch"]) == [f"meta:{pi.META_SNAPSHOT}.snapshot_digest"]


def test_필수_구성원이_계약서에_없으면_읽지_않는다(world):
    """계약서가 검증할 파일을 스스로 고르면, 빠진 입력은 파일별 해시 대조에서도 빠진다."""
    snap = world / "SNAPSHOT.sha256"
    body = [x for x in snap.read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.startswith("#") and not x.endswith("annotations.csv")]
    snap.write_text("\n".join(body) + f"\n# snapshot_digest {_digest(body)}\n", encoding="utf-8", newline="\n")
    with pytest.raises(pi.InputContractError, match="필수 구성원"):
        _expected(world)


def _contract_lines(world: Path) -> list[str]:
    snap = world / "SNAPSHOT.sha256"
    return [x for x in snap.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]


def _write_lines(world: Path, body: list[str], tail: list[str]) -> None:
    (world / "SNAPSHOT.sha256").write_text("\n".join(body + tail) + "\n", encoding="utf-8", newline="\n")


def _digest_line(world: Path) -> str:
    return next(x for x in (world / "SNAPSHOT.sha256").read_text(encoding="utf-8").splitlines()
                if x.startswith("#"))


def test_줄_순서만_바꾼_계약서는_받는다(world):
    """적재 쪽 판정 **통과** — 줄 순서와 무관하게 정해진 순서로 다시 계산하면 기록과 같다."""
    body, line = _contract_lines(world), _digest_line(world)
    _write_lines(world, list(reversed(body)), [line])
    _expected(world)  # 멈추지 않는다


def test_줄_순서를_바꾸고_지문을_적힌_순서로_맞추면_멈춘다(world):
    """적재 쪽 판정 **거부** — 적힌 순서로 맞춘 지문은 정해진 순서로 다시 계산한 값과 다르다."""
    body = list(reversed(_contract_lines(world)))
    _write_lines(world, body, [f"# snapshot_digest {_digest(body)}"])
    with pytest.raises(pi.InputContractError, match="다시 계산한"):
        _expected(world)


def test_필수_구성원이_빠지고_지문을_맞춰도_멈춘다(world):
    """적재 쪽 판정 **거부** — `data_capabilities.yaml` 이 빠지면 남은 줄로 지문을 맞춰도 받지 않는다."""
    body = [x for x in _contract_lines(world) if not x.endswith("data_capabilities.yaml")]
    _write_lines(world, body, [f"# snapshot_digest {_digest(body)}"])
    with pytest.raises(pi.InputContractError, match="필수 구성원"):
        _expected(world)


def test_모르는_구성원을_더하고_지문을_맞춰도_멈춘다(world):
    """적재 쪽 판정 **거부** — 받는 구성원 밖의 줄은 해시가 맞아도 받지 않는다."""
    (world / "extra.csv").write_text("x\n", encoding="utf-8", newline="\n")
    body = [*_contract_lines(world),
            f"{hashlib.sha256((world / 'extra.csv').read_bytes()).hexdigest()}  extra.csv"]
    _write_lines(world, body, [f"# snapshot_digest {_digest(body)}"])
    with pytest.raises(pi.InputContractError, match="받는 구성원이 아니다"):
        _expected(world)


def test_지문_줄_뒤에_주석_한_줄이_붙으면_멈춘다(world):
    """적재 쪽 판정 **거부**. 앞 판은 마지막 `#` 줄을 지문 줄로 읽어 지문 대조를 건너뛰었다(검수 I-3)."""
    body, line = _contract_lines(world), _digest_line(world)
    _write_lines(world, body, [line, "# 아무 말"])
    with pytest.raises(pi.InputContractError, match="주석 줄"):
        _expected(world)


def test_주석을_더한_같은_오기도_범위_제한으로_내려가지_않는다(world, tmp_path):
    """검토 E-1 의 반례에 주석 한 줄을 더한 입력. 앞 판은 멈추지 않고 종료 2 로 내려갔다(검수 I-3)."""
    meta = _meta(world)
    wrong = "f" * 64
    meta[pi.META_SNAPSHOT] = {**meta[pi.META_SNAPSHOT], "snapshot_digest": wrong}
    pairs = _full(world, tmp_path / "p_comment", meta=meta)
    _write_lines(world, _contract_lines(world), [f"# snapshot_digest {wrong}", "# 아무 말"])
    with pytest.raises(pi.InputContractError):
        pi.main(["--manifest-dir", str(world), "--limits", str(LIMITS), "--label-map", str(LABEL_MAP),
                 "--pairs", str(pairs), "--out", str(tmp_path / "r_comment")])


@pytest.mark.parametrize("line", ["# snapshot_digest", "# snapshot_digest 1234", "# snapshot_digest " + "F" * 64])
def test_적힌_지문을_읽지_못하면_지문이_없는_것과_다르다(world, line):
    """머리는 있는데 값을 읽지 못하면 멈춘다. 지문 줄이 **없는** 것은 받아서 미대조로 남긴다(별도 시험)."""
    _write_lines(world, _contract_lines(world), [line])
    with pytest.raises(pi.InputContractError, match="읽지 못했다"):
        _expected(world)


def test_선택_구성원이_없어도_받는다(world):
    """적재 쪽이 받는 입력이다 — `tiles.csv` 는 선택이다. 있는 셋으로 정해진 순서의 지문을 낸다."""
    body = [x for x in _contract_lines(world) if not x.endswith("tiles.csv")]
    _write_lines(world, body, [f"# snapshot_digest {_digest(body)}"])
    _expected(world)  # 멈추지 않는다


def test_구성원_줄의_꼴이_다르면_계약_위반이다(world):
    """이름에 공백이 든 줄은 앞 판에서 `ValueError` 로 죽었다 — 계약 위반의 오류 종류로 멈춘다(검수 m-1)."""
    body = [*_contract_lines(world), f"{'0' * 64}  some name.csv"]
    _write_lines(world, body, [])
    with pytest.raises(pi.InputContractError, match="꼴"):
        _expected(world)


def test_보고에_다시_계산한_지문이_실린다(world, tmp_path):
    """적힌 값과 다시 계산한 값을 둘 다 싣는다 — 읽는 쪽이 무엇을 확인했는지 보고에서 알 수 있다."""
    code, got = _run(world, _full(world, tmp_path / "p_digest"), tmp_path / "r_digest")
    body = [x for x in (world / "SNAPSHOT.sha256").read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.startswith("#")]
    assert code == pi.EXIT_MATCH
    assert got["input_digest"]["computed"] == got["input_digest"]["declared"] == _digest(body)
    assert got["input_digest"]["rule"] == pi.DIGEST_RULE


@pytest.mark.parametrize("key", ["counts_by_split", pi.SKIP_COUNT_FIELD, pi.META_PATHS_CHECKED])
def test_메타_변이는_거부한다(world, tmp_path, key):
    """검토 §27-19 의 메타 계열 변이 — 값이 어긋나면 봉인 경로를 막는다."""
    meta = _meta(world)
    meta[key] = {"n": 1, "of": 1, "basis": "x"} if key == pi.META_PATHS_CHECKED else (
        {} if key == "counts_by_split" else 999)
    code, got = _run(world, _full(world, tmp_path / f"p_{key}", meta=meta), tmp_path / f"r_{key}")
    assert code != pi.EXIT_MATCH and got["sealable"] is False


def test_pairs_없이_돌면_0_을_주지_않는다(world, tmp_path):
    """대조를 하나도 하지 않은 실행이다. 봉인 소비 경로가 읽는 네 필드를 빼먹지도 않는다(교차 검수 I-2)."""
    out = tmp_path / "expected_only"
    code = pi.main(["--manifest-dir", str(world), "--limits", str(LIMITS),
                    "--label-map", str(LABEL_MAP), "--out", str(out)])
    got = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert code == pi.EXIT_PARTIAL
    assert got["sealable"] is False and got["coverage"] == "none"
    assert got["not_compared"] == [pi.NOT_COMPARED_ALL] and got["mismatch"] == {}
    assert "verdict" in got and (out / "expected.jsonl").exists()


def test_out_은_새_경로여야_하고_입력_안에_둘_수_없다(world, tmp_path):
    pairs = _full(world, tmp_path / "pairs")
    base = ["--manifest-dir", str(world), "--limits", str(LIMITS), "--label-map", str(LABEL_MAP), "--pairs", str(pairs)]
    with pytest.raises(SystemExit, match="입력 디렉터리"):
        pi.main([*base, "--out", str(world / "report")])
    with pytest.raises(SystemExit, match="입력 디렉터리"):
        pi.main([*base, "--out", str(pairs)])
    assert pi.main([*base, "--out", str(tmp_path / "ok")]) == pi.EXIT_MATCH
    with pytest.raises(SystemExit, match="비어 있지 않다"):
        pi.main([*base, "--out", str(tmp_path / "ok")])
