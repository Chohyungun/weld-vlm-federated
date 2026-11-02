"""학습 레시피와 타깃 계약 — 2026-10-20 설계 복구 02 2판 §3-5 의 새 시험 1 ~ 4 (지시 1022b).

- 1 차 레시피 `json_only` 는 지금과 같은 바이트다(타깃 · 계약 해시 = 1 차 등록 값).
- 2 차 레시피 `json_line_then_basis` 는 JSON 한 줄 + 줄바꿈 + `target_text`(가공 없음)이고, 서술의 꼴이 어긋나면 고치지 않고 거부한다.
- 지금 파서는 새 타깃에서 1 차와 같은 항목을 읽는다. 상류 정책(첫 객체 · 스키마 위반 · 상자 하나 버림)은 그대로다.
- 학습은 받은 레시피 **한 값**에서 타깃과 계약 해시를 함께 낸다 — 지표만 새 계약을 말하는 길이 없다.

모델 없이 돈다(`tests/uni_fakes.py` 의 대역 · CPU).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.uni_fakes import CountingLoader, make_rows
from vlm import pilot_vlm as PV
from vlm.coords import CoordCfg, ImageGeom, coord_cfg_hash
from vlm.gen_parse import parse_generation

#: 1 차 등록(`configs/registration/main-20261017-1.train_config.json`)의 계약 해시 — 1 차 레시피의 값이어야 한다.
V1_SHA = "42bd30847fe9266eb9b0d7e5a6cfb1573a6b44fc19318e5650276f09aa105316"
#: 02 2판 표 4-2 의 제안 해시(02d `contract_na_sha256`) — 2 차 레시피의 값이어야 한다.
V2_SHA = "ef102bd1fddbc66ede553fb4625a6d6c6d25612ced3dd857eedf40aee42cbc78"
GEOM = ImageGeom(orig_w=64, orig_h=32)


def _row(**over) -> dict:
    row = {"image_id": "t1", "skeleton": {"defects": [{"type": "2011", "bbox_px": [1, 2, 10, 12]}],
                                          "verdict": None, "clauses": ["가상-1"]},
           "target_text": "방사선투과 영상에서 시험 결함 1개가 관찰된다. 시험 조항의 기준은 다음과 같다."}
    row.update(over)
    return row


# ================================================================ 시험 1 · 2 — 계약 · 타깃 바이트
def test_레시피_집합과_계약_해시가_등록과_제안값과_같다():
    from vlm.registration import TRAIN_RECIPES

    assert tuple(PV.TARGET_CONTRACTS) == TRAIN_RECIPES == ("json_only", "json_line_then_basis")
    assert PV.target_contract_sha256() == PV.target_contract_sha256("json_only") == PV.TARGET_CONTRACT_SHA256 == V1_SHA
    assert PV.target_contract_sha256("json_line_then_basis") == V2_SHA
    tc = json.loads((ROOT / "configs/registration/main-20261017-1.train_config.json").read_text(encoding="utf-8"))
    assert tc["target_contract_sha256"] == V1_SHA
    assert PV.recipe_of_contract(V1_SHA) == "json_only" and PV.recipe_of_contract(V2_SHA) == "json_line_then_basis"
    assert (PV.NARRATIVE_LAYOUT, PV.NARRATIVE_SEPARATOR) == ("json_line + separator + basis", "\n")
    with pytest.raises(ValueError):
        PV.target_contract_sha256("two_task_mix")
    with pytest.raises(ValueError):
        PV.recipe_of_contract("0" * 64)


def test_1차_레시피의_타깃은_지금과_같은_바이트다():
    row = _row()
    legacy = json.dumps({"defects": [{"iso_code": "2011", "bbox_2d": [1, 2, 10, 12]}], "verdict": "판정불가",
                         "cited_clauses": ["가상-1"]}, ensure_ascii=False, separators=(",", ":"))
    assert PV.build_target(row, GEOM) == PV.build_target(row, GEOM, recipe="json_only") == legacy
    no_text = {k: v for k, v in row.items() if k != "target_text"}
    assert PV.build_target(no_text, GEOM) == legacy                  # 1 차 레시피는 서술을 읽지 않는다


def test_2차_레시피의_타깃은_JSON_한_줄_줄바꿈_target_text_그대로다():
    row = _row(target_text="공백  두 칸 · 기호 (괄호) 와 숫자 12 mm 그대로.")
    t = PV.build_target(row, GEOM, recipe="json_line_then_basis")
    assert t == PV.build_target(row, GEOM) + "\n" + row["target_text"]
    _line, sep, basis = t.partition("\n")
    assert sep == "\n" and basis == row["target_text"] and "\n" not in basis


@pytest.mark.parametrize("text", [None, 3, "", "   ", " 앞 공백", "뒤 공백 ", "가운데\n줄바꿈", "캐리지\r리턴",
                                  "줄 구분 문자", "끝 줄바꿈\n"])
def test_2차_레시피는_서술의_꼴이_어긋나면_고치지_않고_거부한다(text):
    row = _row(target_text=text)
    with pytest.raises(ValueError):
        PV.build_target(row, GEOM, recipe="json_line_then_basis")
    no_text = {k: v for k, v in _row().items() if k != "target_text"}
    with pytest.raises(ValueError):
        PV.build_target(no_text, GEOM, recipe="json_line_then_basis")
    with pytest.raises(ValueError):
        PV.build_target(_row(), GEOM, recipe="two_task_mix")


# ================================================================ 시험 4 — 지금 파서 · 상류 정책
def test_새_타깃을_지금_파서에_넣으면_1차와_같은_항목이고_상류_정책이_그대로다():
    row = _row()
    v1 = PV.build_target(row, GEOM)
    v2 = PV.build_target(row, GEOM, recipe="json_line_then_basis")
    a, b = parse_generation(v1), parse_generation(v2)
    assert b.parse_error is None and (a.items, a.n_bad, a.verdict, a.cited) == (b.items, b.n_bad, b.verdict, b.cited)
    # 앞 산문 · 뒤의 두 번째 객체는 첫 객체의 값을 바꾸지 않는다
    assert parse_generation("앞 산문이다.\n" + v2).items == a.items
    assert parse_generation(v2 + '\n{"defects":[],"verdict":"합격","cited_clauses":[]}').items == a.items
    # 앞 산문의 중괄호는 해독 실패 — 뒤의 JSON 을 찾아 살리지 않는다
    assert parse_generation("앞 {가상} 산문.\n" + v2).parse_error == "json_decode"
    # 해독은 되지만 구조 필드가 아니면 스키마 위반(첫 객체 `{}` 뒤 설명)
    assert parse_generation("{} 뒤 설명\n" + v2).parse_error == "schema_violation"
    # 개별 상자가 깨지면 그 상자만 버린다 — 레코드 실패로 키우지 않는다
    bad = parse_generation('{"defects":[{"iso_code":"2011","bbox_2d":[1,2]},{"iso_code":"2011","bbox_2d":[1,2,10,12]}],'
                           '"verdict":"판정불가","cited_clauses":[]}\n서술이다.')
    assert bad.parse_error is None and bad.n_bad == 1 and len(bad.items) == 1


# ================================================================ 시험 3 — 학습이 받은 레시피 한 값에서 타깃과 계약이 나온다
@pytest.fixture
def _no_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _rows_with_text(tmp: Path, n: int = 3) -> list[dict]:
    rows = make_rows(tmp / "img", "C1", n)
    for r in rows:
        r["target_text"] = f"추출 시험 {r['image_id']} 의 서술이다."
    return rows


@pytest.mark.parametrize("recipe", ["json_only", "json_line_then_basis"])
def test_train_rounds_는_받은_레시피로_타깃을_만들고_그_계약을_지표에_싣는다(tmp_path, monkeypatch, _no_cuda, recipe):
    seen: list[tuple[str, str]] = []
    real = PV.build_target

    def spy(row, geom, coord_cfg=None, *, recipe="json_only"):
        t = real(row, geom, coord_cfg, recipe=recipe)
        seen.append((recipe, t))
        return t

    monkeypatch.setattr(PV, "build_target", spy)
    rows = _rows_with_text(tmp_path)
    _a, _k, m, _ = PV.train_rounds(rows=rows, epochs=1, round_idx=0, client_idx=0, base_seed=7, model_id="tiny",
                                   model_loader=CountingLoader(), purpose="rehearsal", standin_allowed=True,
                                   recipe=recipe)
    assert seen and {r for r, _t in seen} == {recipe}                     # 실제 타깃이 받은 레시피로 만들어졌다
    assert all(("\n" in t) == (recipe == "json_line_then_basis") for _r, t in seen)
    assert m["target_contract_sha256"] == PV.target_contract_sha256(recipe)
    assert m["resume_identity"]["target_contract_sha256"] == PV.target_contract_sha256(recipe)
    assert ("target_recipe" in m) == (recipe != "json_only")             # 1 차 레시피의 지표는 지금과 같은 꼴


def test_train_rounds_는_모르는_레시피를_모델을_올리기_전에_거부한다(tmp_path, _no_cuda):
    loader = CountingLoader()
    with pytest.raises(ValueError):
        PV.train_rounds(rows=_rows_with_text(tmp_path), epochs=1, round_idx=0, client_idx=0, base_seed=7,
                        model_id="tiny", model_loader=loader, purpose="rehearsal", standin_allowed=True,
                        recipe="two_task_mix")
    assert loader.n == 0


def test_학습_설정_원문은_지표의_계약이_설정_레시피의_계약과_다르면_서지_않는다():
    from vlm.train_cell import train_config_object

    blocks = {k: {} for k in ("lora", "quant", "optimizer", "loss", "lr_schedule")}
    metrics = {"processor_config_sha256": "p" * 64, "config_blocks": blocks, "prompt_sha256": "q" * 64,
               "template_mode": PV.template_mode_of({"enable_thinking": False}),
               "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG")),
               "target_contract_sha256": V1_SHA}
    kw = {"model_id": "m", "model_revision": "r", "pairs_digest": "d", "chat_template_kwargs": {"enable_thinking": False},
          "coord_space": "ABS_ORIG", "num_rounds": 1, "local_epochs": 1, "total_epochs": 1, "prompt_sha256": "q" * 64,
          "init_digest": "i"}
    assert train_config_object(**kw, metrics=metrics)["target_contract_sha256"] == V1_SHA
    with pytest.raises(ValueError, match="target_contract_sha256"):      # 지표는 옛 계약 · 설정은 새 레시피
        train_config_object(**kw, metrics=metrics, recipe="json_line_then_basis")
    with pytest.raises(ValueError, match="target_contract_sha256"):      # 지표는 새 계약 · 설정은 옛 레시피
        train_config_object(**kw, metrics={**metrics, "target_contract_sha256": V2_SHA})


# ================================================================ 실행 설정 키(외부 검토 194 M5) — 값 누락을 1 차 레시피로 읽지 않는다
def _raw(**over) -> dict:
    from fl.uni_run_config import UNI_RUN_DEFAULTS

    return {**UNI_RUN_DEFAULTS, "purpose": "rehearsal", **over}


def test_서버는_레시피_키를_늘_명시해_내려보낸다():
    from fl.uni_run_config import UNI_RUN_KEYS, client_run_cfg, down_config

    assert "uni-target-recipe" in UNI_RUN_KEYS                             # 클라이언트의 필수 키
    sent = down_config(lambda k, d: {"purpose": "rehearsal"}.get(k, d))   # 실행 설정에 없다(1 차 체제)
    assert sent["uni-target-recipe"] == "" and client_run_cfg(sent)["recipe"] == "json_only"
    sent = down_config(lambda k, d: {"purpose": "rehearsal", "uni-target-recipe": "json_line_then_basis"}.get(k, d))
    assert client_run_cfg(sent)["recipe"] == "json_line_then_basis"


def test_클라이언트는_레시피_키가_빠지면_1차_레시피로_읽지_않고_거부한다():
    from fl.uni_run_config import client_run_cfg

    raw = _raw()
    del raw["uni-target-recipe"]
    with pytest.raises(ValueError, match="uni-target-recipe"):
        client_run_cfg(raw)


@pytest.mark.parametrize("value", ["json_only", "two_task_mix", " json_line_then_basis", "JSON_LINE_THEN_BASIS"])
def test_레시피_키의_값은_비우거나_새_레시피_이름_그대로다(value):
    """1 차 레시피는 빈 문자열 한 가지로만 적는다(같은 뜻의 두 표기 금지). 모르는 이름 · 공백 · 대소문자 바꿈은 거부."""
    from fl.uni_run_config import client_run_cfg, down_config

    with pytest.raises(ValueError, match="uni-target-recipe"):
        client_run_cfg(_raw(**{"uni-target-recipe": value}))
    with pytest.raises(ValueError, match="uni-target-recipe"):
        down_config(lambda k, d: {"purpose": "rehearsal", "uni-target-recipe": value}.get(k, d))


# ================================================================ 손실 분해 05 의 basis 사상(지시 1022c 덧붙임 F)
def _load_script(name: str, rel: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_손실_분해의_basis_사상은_02a_의_나_사상과_글자마다_같다():
    """구분자 줄바꿈 = structure, 그 뒤 서술 = basis — 02 §2-5 의 손실 몫(02a `char_labels`)과 견줄 수 있어야 한다."""
    L05 = _load_script("h05_recipe_test", "docs/dev_log/2026-10-19-본실험복구/05_손실분해_C.py")
    A02 = _load_script("a02_recipe_test", "docs/dev_log/2026-10-20-설계복구/02a_타깃길이_C.py")
    rows = (_row(), _row(skeleton={"defects": [], "verdict": "합격", "clauses": []}, target_text="결함이 관찰되지 않는다."))
    for row in rows:
        v1 = PV.build_target(row, GEOM)
        s = PV.build_target(row, GEOM, recipe="json_line_then_basis")
        lab = L05.semantic_spans(s)
        assert A02.char_labels("na", s, v1, L05) == ["class" if c in ("class_first", "class_rest") else c for c in lab]
        assert lab[len(v1)] == "structure" and set(lab[len(v1) + 1:]) == {"basis"}
        assert lab[:len(v1)] == L05.semantic_spans(v1)                    # JSON 구간은 1 차 타깃과 같은 범주
