"""M-03 — 2 차 체제의 같은 임시 세계 사슬(지시 1023c · 1023d · 외부 검토 200 M-03 · 202 '남는 범위' · 205 · 208 · 판 6 연결 2 판 §7 묶음 2).
모델 · GPU 없이.

한 임시 세계(`tests.test_uni_joint_entry.joint` — 본체 체크아웃 자리의 git 저장소 · 합성 동결 스냅샷 · 평가 한 장)에서 **같은 신원**으로
끝까지 간다 — 2 차 학습 등록 → 착수 고정 → 학습 산출 메타 → 평가 등록 판 6(여섯 파일) → 커밋 후 검수 · 영수증 → export → 보조 채점 입력.

| 단계 | 대역(바꿔 끼운 것) | 실제(파일 · 함수로 대조한 것) |
|---|---|---|
| ① 2 차 학습 등록 | 토크나이저 신원(합성 스냅샷 `fake_tokenizer_env`) · 행별 토큰 표의 셈과 자리(`use_v2_token_table`) | 등록 판독 · 체제 짝(레시피 × 프롬프트 3판 × 한도) · 표 출처, `registration_probe` |
| ② 착수 고정 · 시작 문 | 코드 · 환경 칸(`CODE_ENV`) · 공통 칸(실물 `common_probe` 의 값에 학습 전 계약 칸을 더한 것 — 실물 대조는 봉인 페어가 필요하다) · 초기 어댑터의 적재기 | 착수 고정 쓰기 · 평가 쪽 허가 검증 · 본 칸의 시작 문(면제 없음) |
| ③ 학습 산출 메타 | 학습 자체 — 칸 다섯의 meta · 어댑터 · 원장은 합성(`uni_export_world.write_adapter`) | 학습 설정 원문은 학습 쪽 생산자의 바이트, 그 sha256 · 계약 2판 · 프롬프트 3판 · `start_gate.pin_sha256` |
| ④ 평가 등록 판 6 | 1 차 평가 설정 · 에코 목록은 합성 바이트(보존 지문 상수 둘을 그 바이트로 바꿔 끼운다) · 채점 쪽 입력(`filled`) | 본 등록 짓기(`build_main_registration` — 새 체제 입력 넷 · 계약 실물 · 착수 고정 결속 · D6 · 판 6 신원 · 계약 ↔ PIN · 프로필) · 2 차 설정 도출(`build_regime2_eval`) · 여섯 파일 쓰기 · 커밋 후 검수(`validate_committed_main_registration` — `load_bound_eval` · 출처 운반) · 영수증 |
| ⑤ export | 실측 · 생성기 적재(호출을 세는 대역) · 생성 글(대역) | export 사전 점검(영수증 · 등록 · 닻 · 결정적 평가 설정 · 표본 곁 파일 · 실측값 · 학습 행 · 계약 → 레시피 · 등록 판 짝 · 판 6 신원 · 계약 ↔ PIN 실제 호출) · 묶음 쓰기 |
| ⑥ 보조 채점 입력 | 진입 관문의 기지답 · 프레임 · 대조선 · 카나리아와 채점 계산 · 정답 코드 · 묶음 검증의 구현 식별자 검사 | 목록 결속 · 묶음 검증 · 프로필 선택(`profile_for`) · 판 6 신원 · 실제 검사기 선택(`load_profile_checker`) · 서술 추출 |

불일치 — 같은 세계에서 칸 하나씩 바꾼다. (가) 학습 레시피 ≠ 평가 등록의 레시피 → 본 등록 짓기(`train_config_vs_config`) ·
(가2) 등록 뒤 학습 원문이 1 차 계약으로 → export(`registration_version`) · (나) 계약 sha(모르는 계약) → export(`train_config_contract`) ·
(다) 판 5 등록에 2 차 산출 → export(`registration_version`) · (라) 프롬프트 sha → 본 등록 짓기(`train_config_vs_config`). 모두 생성기
적재 · 이미지 열기 0 이다. 기록 하나 — 등록 판 짝을 대역으로 꺼도 판 5 등록은 결정적 평가 설정 결속(`eval_binding`)에서 멈춘다.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.probe.score_unified as SU
import vlm.export_run as XR
from evaluation import eval_registration as ER
from evaluation import narrative_registration as NR
from evaluation import prereg_unified as P
from evaluation import strict_narrative as SN
from evaluation.entry_gate import load_registration_for, read_manifest_meta
from evaluation.eval_list import canonical_bytes, validate_eval_list
from evaluation.narrative_conformance import VECTOR_REL
from scripts import main_uni as M
from scripts.prepare_eval_regime2 import CANDIDATE_REL, candidate_bytes
from tests import test_ecc_train as ET
from tests import test_uni_joint_entry as J
from tests import uni_export_world as W
from tests import v6_link_world as VW
from tests.test_prereg_unified import filled
from tests.uni_fakes import CountingLoader, use_v2_token_table, write_pairs
from vlm import ecc as E
from vlm import export_preflight as XP
from vlm import main_registration as MR
from vlm import narrative as VN
from vlm import pilot_vlm as PV
from vlm import registration as R
from vlm.export_writer import ExportRefused
from vlm.init_adapter import build_initial_adapter
from vlm.pin import build_pin, common_probe, init_probe, read_pin, write_pin
from vlm.train_cell import main_start_gate

V1, V2 = PV.RECIPE_V1, PV.RECIPE_V2
PROMPT_V3 = R.REGIME_PROMPTS[V2]
#: 2 차 한도는 사용자 결정 전 — 시험 값(1 차 512 와 다르기만 하면 된다).
LIMIT = 1024
SEEDS = [7, 8, 9]
#: 학습량 — 합성 연합 원장(`uni_export_world.write_adapter`, 라운드 = 그 세계 등록의 R = 2)과 같다.
BUDGET = {"num_rounds": 2, "local_epochs": 1, "total_epochs": 2}
MODEL = ("Qwen/Qwen3.5-4B", "a" * 40)
PAIRS_DIGEST = "a02bb3e88de48980f87265dc515e7843559ac60bc83d4003dee546136db90209"
STEM = "main-20261023-1"
REG_REL = f"configs/registration/{STEM}.json"
SIDE_HEAD = "image_id,stratum,G_h,n_h"
JSON_LINE = '{"defects":[{"iso_code":"2011","bbox_2d":[10,20,30,40]}],"verdict":"불합격","cited_clauses":[]}'
NARRATIVE = "두께 정보가 없어 합부 판정은 내리지 않는다."
AUX_NAMES = (*SN.NARRATIVE_NAMES, *P.CLAUSE_AXIS_METRICS)
#: 코드 · 환경 측정만 대역 — 작업 트리는 커밋 전이라 실물 코드 측정은 더럽다고 낸다. 초기 어댑터는 실물로 잰다.
CODE_ENV = {"code": lambda: {"commit": "c" * 40, "dirty": []},
            "env": lambda: {"torch": "t", "cuda": None, "gpu_name": None}}
MEASURED7 = {k: W.MEASURED[k] for k in ("processor_config_sha256", "processor_min_pixels", "processor_max_pixels",
                                        "patch_size", "merge_size", "gen_prefix_sha256", "transformers_version")}
#: 채점 진입점의 판 6 검사 인자(`--aux-train-extractor` · `--aux-prompt` · `--aux-vectors`)에 주는 실제 파일 — 학습 쪽 추출기 ·
#: 2 차 프롬프트 · 정본 추출 벡터.
AUX_FILES = {"train_extractor": ROOT / "vlm/narrative.py", "prompt": ROOT / PROMPT_V3, "vectors": ROOT / VECTOR_REL}
ALL_CHECKED = {"code_tree": "checked", "train_extractor": "checked", "prompt": "checked", "vectors": "checked"}


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


class Spies2(J.Spies):
    """쓰는 쪽의 호출 기록(`J.Spies`) — 생성기만 2 차 꼴의 글(JSON 한 줄 + 줄바꿈 + 서술)을 낸다."""

    def _loader(self, cfg):
        self.calls["loader"] += 1
        return W.gen(text_of=lambda image_id: JSON_LINE + PV.NARRATIVE_SEPARATOR + NARRATIVE)


@pytest.fixture
def approved(monkeypatch):
    """승인 목록을 호출을 세는 대역으로 바꿔 끼운다(`test_uni_joint_entry` 머리말) — 본실험 목적이 내용 검사에 닿게."""
    s = Spies2()
    monkeypatch.setattr(XR, "APPROVED_MEASURES", (s.measure,))
    monkeypatch.setattr(XR, "APPROVED_GENERATOR_LOADERS", {"model": (s.loader,), "echo": (s.loader,)})
    return s


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _rows() -> list[dict]:
    """학습 행 — 세 참여자, 2 차 타깃(`target_text`)을 짓는 칸까지."""
    return [{"image_id": f"{c}-t{k}", "image_path": "x.png", "client": c, "split": "train", "width_px": 1280,
             "height_px": 720, "target_text": f"{c} 학습 행 {k} 의 서술이다.",
             "skeleton": {"defects": [{"type": "2011", "bbox_px": [10, 20, 30, 40]}], "verdict": "불합격", "clauses": []}}
            for c, n in (("C1", 3), ("C2", 2), ("C3", 1)) for k in range(n)]


def train_config_text(*, init: str, recipe: str = V2, prompt_sha: str | None = None, contract: str | None = None) -> str:
    """학습 설정 원문 — 학습 쪽 생산자(`train_config_object` · `canonical_json`)가 이 세계의 값으로 낸 바이트(손으로 적지 않는다).

    `contract` 를 주면 생산자가 낸 원문의 계약 칸만 **나중에 고친다** — 생산자는 레시피의 계약과 다른 계약의 원문을 내지 않는다
    (`train_config_object` 가 멈춘다). 불일치 시험의 '다른 곳에서 온 · 고쳐진 원문' 이다."""
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.pilot_vlm import canonical_json, load_prompt, target_contract_sha256
    from vlm.train_cell import train_config_object

    psha = prompt_sha or load_prompt(ROOT / PROMPT_V3)[1]
    metrics = {"processor_config_sha256": W.MEASURED["processor_config_sha256"],
               "config_blocks": {k: {} for k in ("lora", "quant", "optimizer", "loss", "lr_schedule")},
               "prompt_sha256": psha, "template_mode": "enable_thinking=False",
               "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG")),
               "target_contract_sha256": target_contract_sha256(recipe)}
    obj = train_config_object(
        model_id=MODEL[0], model_revision=MODEL[1], pairs_digest=PAIRS_DIGEST, chat_template_kwargs={"enable_thinking": False},
        coord_space="ABS_ORIG", prompt_sha256=psha, init_digest=init, metrics=metrics, recipe=recipe, **BUDGET)
    return canonical_json(obj if contract is None else {**obj, "target_contract_sha256": contract})


def _train_registration(rule: Path, *, contract_sha: str | None) -> dict:
    body = {"kind": R.KIND, "version": 1, "seed_indices": [1], **BUDGET, "recipe": V2, "eval_gen_limit": LIMIT,
            "quantization": dict(PV.QUANT_4BIT), "micro_batch": PV.MICRO_BATCH, "grad_accum": PV.GRAD_ACCUM, "lr": PV.LR,
            "early_coord_check_rules": {"version": yaml.safe_load(rule.read_bytes())["version"], "sha256": _sha(rule.read_bytes())},
            "init_adapter_digest": None}
    if contract_sha is not None:
        body["narrative_inputs"] = {"version": NR.INPUTS_CONTRACT_VERSION, "sha256": contract_sha}
    return body


@pytest.fixture
def e2e(tmp_path, monkeypatch, approved):
    """①②③ — 2 차 학습 등록 · 착수 고정 · 시작 문 · 칸 다섯의 학습 산출 메타. ④ 의 입력(1 차 설정 · 고정 후보 · 목록 · 곁 파일)도 둔다."""
    from evaluation.actuals import impl_key
    from vlm.pilot_vlm import load_pairs, train_rows_digest
    from vlm.seams import impl_id

    w = J.joint(tmp_path)
    VW.copy_pinned(w.repo)                                    # 판 6 신원 · 계약 실물 · 출처 아홉 행 · 프롬프트 3판 — 명시 목록
    contract_raw = (w.repo / NR.INPUTS_CONTRACT_REL).read_bytes()
    rule = ET._rule_file(tmp_path)
    monkeypatch.setattr(E, "RULE_PATH", rule)
    rows = _rows()
    pairs = write_pairs(tmp_path / "main_pairs" / "pairs.jsonl", rows)
    table = use_v2_token_table(monkeypatch, tmp_path, pairs_digest=PAIRS_DIGEST, model_id=MODEL[0], revision=MODEL[1],
                               rows=rows)
    prov = json.loads(table.with_name(table.name + ".provenance.json").read_text(encoding="utf-8"))
    sums = {c: v["supervised_tokens"] for c, v in prov["per_client"].items()}
    # 설정 · 학습 등록 · 1 차 평가 설정(합성 바이트 — 보존 지문을 그 바이트로) · 고정 후보(도출) — 한 커밋(export 는 영수증 커밋의 바이트를 읽는다)
    fixed = {"snapshot_id": "synthetic_v1", "snapshot_digest": w.digest, "uni_processor_kwargs": {"max_pixels": W.MAX_PIXELS},
             "uni_model": {"id": MODEL[0], "revision": MODEL[1]}, "uni_prompt_path": PROMPT_V3, "uni_prompt_sha256": None,
             "uni_coord_space": "ABS_ORIG", "uni_chat_template_kwargs": {"enable_thinking": False}, "uni_batch_size": 1,
             "uni_max_new_tokens": LIMIT, "uni_pairs": {"path": pairs.as_posix(), "digest": PAIRS_DIGEST},
             "uni_train_budget": dict(BUDGET), "uni_expected_supervised_tokens": sums}
    cfg_path = w.repo / "configs" / "base.yaml"
    cfg_path.write_text(yaml.safe_dump({"fixed_before_main_runs": fixed, "experiment": {"seeds": SEEDS}}, allow_unicode=True),
                        encoding="utf-8")
    reg_path = w.repo / "configs" / "train_registration.yaml"
    reg_path.write_text(yaml.safe_dump(_train_registration(rule, contract_sha=None), allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    gen_raw, echo_raw = canonical_bytes([W.EVAL_ID]), canonical_bytes(list(W.IDS))
    gen = validate_eval_list(gen_raw)
    side_raw = f"{SIDE_HEAD}\n{W.EVAL_ID},ST|b0,1,1\n".encode()
    source = copy.deepcopy(yaml.safe_load((ROOT / ER.CONFIG_REL).read_text(encoding="utf-8")))
    sealed = {"images": 1, "groups": 1, "file_sha256": gen.file_sha256, "set_sha256": gen.set_sha256,
              "side_sha256": _sha(side_raw)}
    source["population"]["snapshot_digest"] = w.digest
    source["population"]["sample"]["main_size"] = 3000                 # 봉인 키 집합(1000 · 2000 · 3000 · 4000)은 1 차 그대로
    source["population"]["sample"]["sealed"]["3000"] = sealed
    source_raw = yaml.safe_dump(source, allow_unicode=True, sort_keys=False).encode("utf-8")
    (w.repo / ER.CONFIG_REL).write_bytes(source_raw)
    monkeypatch.setattr(ER, "SOURCE_EVAL_SHA256_V1", _sha(source_raw))             # 1 차 보존 지문 — 이 세계의 1 차 바이트로
    monkeypatch.setattr(MR, "ECHO_LIST_RAW_SHA256_V1", _sha(echo_raw))             # D6 — 이 세계의 에코 원본 raw 로
    monkeypatch.setattr(MR, "ECHO_LIST_SET_SHA256_V1", validate_eval_list(echo_raw).set_sha256)
    # ② 착수 고정 · 착수 허가 · 시작 문 — 공통 칸은 실물 측정값에 학습 전 계약 칸을 더한 이음새(실물 대조는 봉인 페어가 있어야 선다)
    monkeypatch.setattr(M, "run_profile", lambda out: {"train_registration": reg_path, "run_name": "r2"})
    cfg = M.load_main_config(cfg_path)
    spec = M.spec_for(cfg, 1, out=tmp_path / "out", config_path=cfg_path)
    common = common_probe(cfg_path, registration_path=reg_path)
    common["registration"]["narrative_inputs"] = {"version": NR.INPUTS_CONTRACT_VERSION, "sha256": _sha(contract_raw)}
    probes = {**CODE_ENV, "common": lambda: copy.deepcopy(common)}
    reg_path.write_text(yaml.safe_dump(_train_registration(rule, contract_sha=_sha(contract_raw)), allow_unicode=True,
                                       sort_keys=False), encoding="utf-8")
    train_doc = R.registration_from_bytes(reg_path.read_bytes())
    cand_raw = candidate_bytes(ER.build_regime2_eval(ER.load_eval_registration(source_raw), base_doc=ER.load_base(cfg_path.read_bytes()),
                                                     train_doc=train_doc, prompt_raw=(w.repo / PROMPT_V3).read_bytes()))
    (w.repo / CANDIDATE_REL).parent.mkdir(parents=True, exist_ok=True)
    (w.repo / CANDIDATE_REL).write_bytes(cand_raw)
    doc = ER.load_eval_registration(cand_raw)
    W.git(w.repo, "add", "-A")
    W.git(w.repo, "commit", "-q", "-m", "2 차 설정 · 학습 등록 · 1 차 평가 설정 · 고정 후보 · 판 6 실물")
    build_initial_adapter(model_id=MODEL[0], seed=SEEDS[0], cache_path=spec.init_adapter_path, revision=MODEL[1],
                          purpose="rehearsal", model_loader=CountingLoader())
    pin_sha = write_pin(spec.pin_path, build_pin(init_adapter_path=spec.init_adapter_path, config_path=cfg_path,
                                                 registration_path=reg_path, probes=probes))
    init = init_probe(spec.init_adapter_path)["init_adapter_digest"]
    rules = E.load_check_rules(rule)["rules"]
    expected = {"start_adapter_digest": init, "pin_sha256": pin_sha, "check_run_id": "uni_central-ecc", "seed": 1}
    src = tmp_path / "early_coord_check.json"
    src.write_bytes(ET._permit_bytes(tmp_path, rules, expected))
    E.accept_permit(src, rules=rules, expected=expected, seed_dir=spec.ecc_permit_path.parent,
                    permit_name=spec.ecc_permit_path.name, attempts_parent=tmp_path / "ecc_parent",
                    attempt_row={"run_id": "uni_central-ecc", "pin_sha256": pin_sha, "last_step": 100, "hours": 0.5,
                                 "won": 1500}, extra={"attempt": 1, "ecc_root": ET.ECC_ROOT})
    gate = main_start_gate(spec, init_digest=init, pin_probes=probes)
    # ③ 칸 다섯의 학습 산출 메타 — 같은 원문, C1 은 참여자 행 전체의 digest(비율 1), start_gate 의 착수 고정 sha
    tc = train_config_text(init=init)
    train_root = w.repo / "main_in" / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    (train_root / M.PIN_FILE).write_bytes(spec.pin_path.read_bytes())             # export 는 어댑터 폴더의 부모에서 착수 고정을 읽는다
    rows_digest = {c: train_rows_digest(load_pairs("train", c, pairs_path=pairs)) for c in ("C1", "C2", "C3")}

    def write_metas(text: str, *, start_gate=None) -> dict:
        metas = {}
        for tag in MR.CELLS:
            ident = {"purpose": "main", "tag": tag, "seed_index": 1, "seed_value": SEEDS[0]}
            dig = rows_digest[tag[-2:]] if tag.startswith("uni_local_") else dict(rows_digest)
            d = W.write_adapter(w, tag=tag, purpose="main", base=train_root,
                                meta_over={"identity": ident, "train_config": text, "init_adapter_digest": init,
                                           "train_rows_digest": dig, "start_gate": dict(gate if start_gate is None else start_gate)})
            metas[tag] = json.loads((d / "adapter_last.meta.json").read_text(encoding="utf-8"))
        return metas

    keys = tuple(sorted(impl_key(impl_id(fn, seam=seam, approved=True))
                        for fn, seam in ((approved.measure, "export_measure"), (approved.loader, "export_generator"))))
    lst, echo, side = (tmp_path / "main_in" / n for n in ("gen.txt", "echo.txt", "side.csv"))
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_bytes(gen_raw)
    echo.write_bytes(echo_raw)
    side.write_bytes(side_raw)
    e = SimpleNamespace(w=w, s=approved, cfg=cfg, cfg_path=cfg_path, reg_path=reg_path, spec=spec, gate=gate,
                        pin=read_pin(spec.pin_path)[0], pin_raw=spec.pin_path.read_bytes(), pin_sha=pin_sha, init=init, tc=tc,
                        train_root=train_root, write_metas=write_metas, metas=write_metas(tc), impl_ids=keys, gen_raw=gen_raw,
                        echo_raw=echo_raw, doc=doc, cand_raw=cand_raw, source_raw=source_raw, contract_raw=contract_raw,
                        sealed=sealed, side_raw=side_raw, lst=lst, echo=echo, side=side, meta=read_manifest_meta(w.snap),
                        pairs=pairs, tmp=tmp_path)
    e.scoring = _scoring(e)
    return e


def _build(e, **over):
    kw = {"cfg": e.cfg, "seed_index": 1, "gen_raw": e.gen_raw, "echo_raw": e.echo_raw, "eval_doc": e.doc,
          "manifest_meta": e.meta, "snapshot_digest": e.w.digest, "measured": dict(MEASURED7), "init_digest": e.init,
          "pin": e.pin, "metas": e.metas, "scoring": getattr(e, "scoring", {}), "impl_ids": e.impl_ids,
          "prompt_path": e.w.repo / PROMPT_V3, "now": J.NOW, "repo": e.w.repo, "pin_raw": e.pin_raw,
          "inputs_contract_raw": e.contract_raw, "profile": NR.PROFILE_REGIME2}
    kw.update(over)
    return MR.build_main_registration(**kw)


def _scoring(e) -> dict:
    """평가 쪽이 값을 낼 칸만 — 본 등록 짓기가 빈 칸으로 알린 이름에 시험의 유효 값(`filled`)을 넣는다(`test_main_registration` 과
    같은 꼴, 채점 쪽 입력의 대역)."""
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        _build(e, scoring={})
    assert exc.value.code == "registration_incomplete"
    out: dict = {}
    for name in exc.value.detail:
        area, _, key = name.partition(".")
        if key:
            out.setdefault(area, {})[key] = getattr(getattr(filled(), area), key)
        else:
            out[area] = getattr(filled(), area)
    out["planned_seed_count"] = 1
    return json.loads(json.dumps(out, default=list))


def _register(e, **over):
    """④ — 본 등록 짓기(실제) — 새 체제는 작성기가 판 6 블록을 실제 파일에서 짓는다. (등록 객체, 등록 바이트, 학습 설정 원문)."""
    return _build(e, **over)


def _as_v5(reg, *, limit: int | None = None):
    """같은 등록을 판 5 꼴로 — 판 5 등록에 2 차 산출을 들이미는 불일치 시험의 입력(본 등록 짓기는 2 차 레시피에 판 5 를 짓지 않는다).
    `limit` 을 주면 생성 한도를 그 값(1 차 설정의 512)으로 — 옛 경로의 결속을 지나 그 뒤 검사(등록 판 짝)에 닿게."""
    out = replace(reg, version=P.REGISTRATION_VERSION, narrative=None, clause_axis=None)
    return out if limit is None else replace(out, generation=replace(out.generation, max_new_tokens=limit))


def _commit(e, reg, raw: bytes, tc_text: str, *, write_receipt: bool = True) -> Path:
    """④ — 여섯 파일(판 6) 또는 넷(판 5)을 실제 작성기로 쓰고 본줄기에 커밋, 실제 영수증 작성기(커밋 후 검수를 먼저 지난다)로 영수증.
    `write_receipt=False` 면 영수증을 손으로 쓴다(커밋 후 검수가 거부할 입력의 불일치 시험)."""
    d = e.w.repo / "configs" / "registration"
    v6 = reg.version == P.REGISTRATION_VERSION_V6
    files = MR.write_main_registration(d, stem=STEM, raw=raw, tc_text=tc_text, gen_raw=e.gen_raw, echo_raw=e.echo_raw,
                                       eval_raw=e.cand_raw if v6 else None,
                                       sources_raw=NR.build_v6_sources(registration_raw=raw, eval_raw=e.cand_raw, repo=e.w.repo)
                                       if v6 else None)
    W.git(e.w.repo, "add", *[p.relative_to(e.w.repo).as_posix() for p in files.values()])
    W.git(e.w.repo, "commit", "-q", "-m", "본실험 등록(2 차)")
    head = W.git(e.w.repo, "rev-parse", "HEAD")
    if write_receipt:
        return MR.write_main_receipt(registration_rel=REG_REL, commit=head, checkout=e.w.repo, now=J.NOW)
    receipt = {"generation_sha256": reg.generation_sha256(), "scoring_sha256": reg.scoring_sha256(),
               "registered_at": "2026-09-21T09:00:00+09:00", "main_commit": head, "kind": "main",
               "registration_path": REG_REL, "registration_file_sha256": _sha(raw)}
    rp = e.w.repo / "main_in" / f"{STEM}.receipt.json"
    rp.write_text(json.dumps(receipt), encoding="utf-8")
    return rp


def _export(e, rp: Path, folder: Path, *, mode: str = "model") -> str | None:
    """⑤ — 본실험 C1 모델 묶음(표본 모드 — 봉인한 곁 파일) 또는 에코 묶음. 거부 사유 코드, 지나면 None."""
    kw = {"adapter_dir": e.train_root / "uni_local_C1_s1", "side_path": e.side} if mode == "model" else {"tag": "uni_central"}
    try:
        W.run(e.w, purpose="main", mode=mode, receipt_path=rp, list_path=e.lst if mode == "model" else e.echo, run_root=None,
              plan_path=None, folder=folder, loader=e.s.loader, measure_fn=e.s.measure, image_path_of=e.s.paths(e.w), **kw)
    except ExportRefused as exc:
        return exc.code
    return None


def _exported(e, tmp_path: Path):
    """④ ⑤ — 판 6 등록 · 커밋 · 영수증 뒤 C1 모델 묶음 export(지나야 한다). (등록, 영수증, 묶음 폴더)."""
    reg, raw, tc_text = _register(e)
    rp = _commit(e, reg, raw, tc_text)
    folder = tmp_path / "main_out" / "export"
    assert _export(e, rp, folder) is None
    return reg, rp, folder


def _aux(e, rp: Path, folder: Path, monkeypatch, *, train_extractor: Path | None = None, prompt: Path | None = None,
         vectors: Path | None = None):
    """⑥ — 채점 진입점의 목록 결속 · 묶음 검증(실제) 뒤 보조 채점 입력(`_aux_scoring`). 진입 관문의 기지답 · 프레임 · 대조선 ·
    카나리아와 채점 계산은 돌지 않는다(대역). 묶음 검증의 구현 식별자 검사는 시험의 대역 구현이 커밋된 코드가 아니라 끈다
    (`test_uni_joint_entry` 의 가-8 과 같은 이음새). 판 6 검사 인자 셋(학습 쪽 추출기 · 프롬프트 · 추출 벡터 파일)은 명령줄의
    `--aux-train-extractor` · `--aux-prompt` · `--aux-vectors` 와 같은 칸으로 준다 — 주지 않으면 진입점이 `not_given` 으로 적는다."""
    import evaluation.bundle_v14 as BV
    from evaluation.prereg_unified import read_receipt

    monkeypatch.setattr(BV, "impl_problems", lambda v: [])
    rc = read_receipt(rp)
    args = argparse.Namespace(registration="strict", purpose="main", generation_list=e.lst, scoring_list=e.lst,
                              echo_list=e.echo, snapshot=e.w.snap, bundles=folder, train_root=e.train_root,
                              aux_train_extractor=train_extractor, aux_prompt=prompt, aux_vectors=vectors)
    ctx = SU.Ctx(args=args, checkout=e.w.repo, repo=e.w.repo, receipt=rc, reg=load_registration_for(rc, repo=e.w.repo),
                 mode=SU.MODE_SAMPLE, eval_doc=e.doc, sealed=dict(e.sealed), side_raw=e.side_raw)
    gen, pop, echo, meta = SU._read_lists(ctx)
    verified, rejected = SU._verify_all(ctx, SU._bundle_files(folder), gen, pop, echo, meta)
    classes = [str(c) for c in ctx.reg.metric.class_codes]
    aux = SU._aux_scoring(ctx, verified, {(vb.tag, vb.seed_index): None for vb in verified}, list(pop.ids),
                          {i: ("2011",) for i in pop.ids}, {}, list(SU._known_codes()), classes)
    return aux, verified, rejected


def _loaded(e) -> bool:
    return e.s.calls["loader"] > 0 or e.s.calls["image"] > 0


# ================================================================ 양성 — 끝까지
def test_양성_대역은_토크나이저_학습_생성기_실제는_등록_착수고정_평가등록_export_보조입력_같은_신원으로_끝까지_지난다(e2e, tmp_path,
                                                                                       monkeypatch):
    """①~⑥ 을 같은 신원으로 — 레시피 · 계약 · 프롬프트 sha 가 학습 등록 · 착수 고정 · 학습 설정 원문 · 평가 등록 · export 의 생성기
    설정 · 곁 값에서 같다. 대역 · 실제의 나눔은 머리말 표(①~⑥ 행)."""
    e = e2e
    v3_sha = R.regime_prompt_sha256(V2)
    seen: list = []
    real = XP.export_preflight

    def spy(**kw):
        res = real(**kw)
        seen.append(res)
        return res

    monkeypatch.setattr(XP, "export_preflight", spy)
    # ② 착수 고정의 공통 칸이 그 등록(학습 전 계약 포함)을 싣고, 시작 문은 새 착수 허가로 섰다(면제 없음)
    reg_in_pin = e.pin["common"]["registration"]
    assert reg_in_pin["recipe"] == V2 and reg_in_pin["eval_gen_limit"] == LIMIT and e.pin["common"]["prompt_sha256"] == v3_sha
    assert reg_in_pin["narrative_inputs"] == {"version": 1, "sha256": _sha(e.contract_raw)}
    assert e.gate["pin_sha256"] == e.pin_sha and "ecc_permit_sha256" in e.gate and "ecc_permit_waiver_sha256" not in e.gate
    # ③ 메타의 원문 = 학습 쪽 생산자의 바이트 — 계약 2판 · 프롬프트 3판, start_gate 의 착수 고정 sha
    tc = json.loads(e.tc)
    assert (tc["target_contract_sha256"], tc["prompt_sha256"]) == (PV.target_contract_sha256(V2), v3_sha)
    assert all(m["train_config"] == e.tc and m["start_gate"]["pin_sha256"] == e.pin_sha for m in e.metas.values())
    # ④ 평가 등록 판 6 — 짓기(실제)가 작성기로 블록을 짓는다. 2 차 설정은 1 차 바이트에서 도출한 고정 후보다
    reg, raw, tc_text = _register(e)
    assert reg.version == P.REGISTRATION_VERSION_V6 and tc_text == e.tc
    assert reg.generation.prompt_sha256 == v3_sha and reg.generation.max_new_tokens == LIMIT
    assert reg.generation.train_config_sha256 == _sha(e.tc.encode("utf-8"))
    reg.require_complete(side="all")
    n = reg.narrative
    assert (tuple(n.open_holes), tuple(n.exploratory_kinds)) == (NR.OPEN_HOLES_V4, NR.EXPLORATORY_KINDS_V4)
    assert (n.checker_id, n.checker_vectors_sha256, n.confirmation_cases_sha256) == (
        NR.CHECKER_PINS[1].checker_id, NR.V6_EVIDENCE_V4[0].sha256, NR.V6_EVIDENCE_V4[1].sha256)
    assert n.inputs_contract_sha256 == _sha(e.contract_raw) and n.generator_sha256 == NR.generator_bundle_identity(ROOT)["sha256"]
    assert NR.identity_problems(reg, repo=ROOT, profile=NR.PROFILE_REGIME2, train_extractor_path=AUX_FILES["train_extractor"],
                                prompt_path=AUX_FILES["prompt"], vectors_path=AUX_FILES["vectors"]) == []
    assert NR.profile_for(reg, train_entry=reg_in_pin).profile_id == NR.PROFILE_REGIME2
    assert ER.narrative_layout_problems(reg, train_contract=PV.TARGET_CONTRACTS[V2]) == []
    assert n.train_extractor_id == VN.EXTRACTOR_ID
    assert e.doc["auxiliary"][-5:] == list(ER.AUXILIARY_V6) and e.doc["generation"]["max_new_tokens"] == LIMIT
    # ④′ 여섯 파일 → 커밋 → 커밋 후 검수(첫 정식 소비자) → 영수증(실제 작성기)
    rp = _commit(e, reg, raw, tc_text)
    reg_dir = e.w.repo / "configs" / "registration"
    assert sorted(p.name for p in reg_dir.iterdir() if p.is_file()) == sorted([f"{STEM}{s}" for s in
                                                              (".json", ".train_config.json", ".generation.list", ".echo.list",
                                                               ".eval.yaml", ".v6_sources.json", ".receipt.json")])
    assert (reg_dir / f"{STEM}.eval.yaml").read_bytes() == e.cand_raw == (e.w.repo / CANDIDATE_REL).read_bytes()
    head = W.git(e.w.repo, "rev-parse", "HEAD")
    bound = MR.validate_committed_main_registration(registration_rel=REG_REL, commit=head, checkout=e.w.repo)
    assert (bound.profile_id, bound.relative_path, bound.sha256) == (NR.PROFILE_REGIME2, f"configs/registration/{STEM}.eval.yaml",
                                                                    _sha(e.cand_raw))
    assert json.loads(rp.read_text(encoding="utf-8"))["main_commit"] == head
    # ⑤ export — 사전 점검이 결정적 평가 설정 · 계약 ↔ PIN · 판 6 신원을 지나고 묶음을 쓴다(모델 · 에코)
    folder = tmp_path / "main_out" / "export"
    assert _export(e, rp, folder) is None
    assert (folder / "uni_local_C1_s1.export_meta.json").is_file() and e.s.calls["loader"] == 1
    (pre,) = seen
    assert (pre.gen_cfg.target_recipe, pre.gen_cfg.max_new_tokens, pre.registration.version) == (V2, LIMIT, "6")
    assert (pre.sidecar_values["prompt_sha256"], pre.sidecar_values["train_config_sha256"]) == (v3_sha, _sha(e.tc.encode()))
    assert _export(e, rp, tmp_path / "main_out" / "export_echo", mode="echo") is None
    # ⑥ 채점 진입점 — 묶음 검증(평가 쪽 검증기) 뒤 판 6 의 보조 입력: 프로필 regime2_v4 · 실제 검사기 v4 · 검사 인자 셋 checked
    aux, verified, rejected = _aux(e, rp, folder, monkeypatch, **AUX_FILES)
    assert rejected == {} and [(vb.tag, vb.seed_index) for vb in verified] == [("uni_local_C1", 1)]
    assert isinstance(aux, SN.AuxScoring) and dict(aux.identity_checks) == ALL_CHECKED
    assert set(aux.models) == {("uni_local_C1", 1)} and aux.wanted == frozenset(AUX_NAMES)
    assert tuple(aux.exploratory_kinds) == NR.EXPLORATORY_KINDS_V4
    assert aux.checker_identity["checker_id"] == NR.CHECKER_PINS[1].checker_id
    assert type(aux.checker).__module__ == NR.CHECKER_MODULES[NR.PROFILE_REGIME2]
    item = aux.models[("uni_local_C1", 1)].items[0]
    assert (item.image_id, item.material, item.structured_codes) == (W.EVAL_ID, "ST", ("2011",))
    assert (item.extract.state, item.extract.narrative) == ("complete", NARRATIVE)


def test_진입점_검사_인자를_생략하면_실제_보조_입력은_내되_세_칸을_not_given_으로_적는다(e2e, tmp_path, monkeypatch):
    """양성과 같은 사슬에서 채점 진입점에 판 6 검사 인자를 주지 않는다 — 보조 입력은 나오되 진입점이 무엇을 맞대지 않았는지 적는다
    (`identity_checks` — 코드 트리만 checked)."""
    e = e2e
    _reg, rp, folder = _exported(e, tmp_path)
    aux, _verified, rejected = _aux(e, rp, folder, monkeypatch)
    assert rejected == {} and isinstance(aux, SN.AuxScoring) and set(aux.models) == {("uni_local_C1", 1)}
    assert dict(aux.identity_checks) == {"code_tree": "checked", "train_extractor": "not_given", "prompt": "not_given",
                                         "vectors": "not_given"}


def test_진입점에_등록과_다른_학습_쪽_추출기를_주면_실제_진입점이_보조_지표_대신_사유를_남긴다(e2e, tmp_path, monkeypatch):
    """등록(판 6)은 트리의 학습 쪽 추출기로 지었는데 채점 진입점에는 바이트가 다른 추출기(식별자는 같다)를 준다 — 진입점은 보조 지표를
    내지 않고 사유를 남긴다(`AuxRefused`). 묶음 검증은 그대로 지난다(M1 쪽은 그대로)."""
    e = e2e
    altered = e.tmp / "narrative_altered.py"
    _reg, rp, folder = _exported(e, tmp_path)
    altered.write_bytes(AUX_FILES["train_extractor"].read_bytes() + "\n# 시험용 바꾼 사본 — 식별자는 같고 바이트만 다르다\n".encode())
    aux, verified, rejected = _aux(e, rp, folder, monkeypatch, **{**AUX_FILES, "train_extractor": altered})
    assert rejected == {} and [(vb.tag, vb.seed_index) for vb in verified] == [("uni_local_C1", 1)]
    assert isinstance(aux, SN.AuxRefused) and dict(aux.identity_checks) == ALL_CHECKED
    assert [p.split(":")[0] for p in aux.problems] == ["narrative.train_extractor_sha256"]
    assert aux.computed_names() == [] and aux.not_computed_reasons() == {name: SN.REFUSED for name in AUX_NAMES}
    rep = aux.report()["aux_identity"]
    assert (rep["status"], rep["names"], rep["problems"], rep["checks"]) == ("refused", list(AUX_NAMES), list(aux.problems),
                                                                            ALL_CHECKED)


def test_기록_등록_판_짝을_대역으로_꺼도_판_5_등록은_결정적_평가_설정_결속에서_멈춘다(e2e, tmp_path, monkeypatch):
    """지시 1023c 의 export 검사(등록 판 짝)를 대역으로 꺼도, 판 5 등록은 영수증 커밋의 옛 평가 설정(1 차 · 한도 512)과 2 차 한도가 결속
    되지 않아(`eval_binding`) 생성기 전에 멈춘다 — 판 6 연결 전에는 보조 입력 없이 채점까지 갔다(기록). 영수증은 손으로 쓴다(커밋 후
    검수가 거부하는 입력이다)."""
    e = e2e
    reg6, _raw6, tc_text = _register(e)
    reg = _as_v5(reg6)
    rp = _commit(e, reg, P.dump_registration(reg), tc_text, write_receipt=False)
    monkeypatch.setattr(XP, "registration_version_of_recipe", lambda recipe: reg.version)
    folder = tmp_path / "main_out" / "export"
    assert _export(e, rp, folder) == "eval_binding" and not _loaded(e)
    head = W.git(e.w.repo, "rev-parse", "HEAD")
    with pytest.raises(MR.MainRegistrationRefused) as exc:                     # 커밋 후 검수도 같은 자리에서 거부한다
        MR.validate_committed_main_registration(registration_rel=REG_REL, commit=head, checkout=e.w.repo)
    assert exc.value.code == "eval_binding"


def test_등록_판_짝_함수는_판을_정한_레시피만_받는다():
    """실제 함수 — 1 차 레시피는 판 5, 새 체제는 판 6, 판을 정하지 않은 레시피는 거부한다(레시피가 늘면 export 가 닫힌다)."""
    assert XP.registration_version_of_recipe(V1) == P.REGISTRATION_VERSION
    assert XP.registration_version_of_recipe(V2) == P.REGISTRATION_VERSION_V6
    with pytest.raises(ValueError, match="판을 정하지 않은"):
        XP.registration_version_of_recipe("json_then_other")
    assert set(PV.TARGET_CONTRACTS) == {V1, V2}                      # 레시피가 늘면 짝 함수와 이 시험을 함께 고친다


# ================================================================ 불일치 — 칸 하나씩
def test_불일치_가_학습_레시피가_평가_등록과_다르면_실제_본_등록_짓기가_멈춘다(e2e):
    """학습 등록 · 착수 고정은 2 차인데 학습 산출의 원문은 1 차 계약(프롬프트는 3판 그대로) — 평가 등록의 레시피(착수 고정에 묶인
    학습 등록의 레시피)와 다르다. 실제: 본 등록 짓기의 원문 ↔ 설정 대조(`train_config_vs_config`)."""
    e = e2e
    metas = e.write_metas(train_config_text(init=e.init, recipe=V1))
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        _register(e, metas=metas)
    assert exc.value.code == "train_config_vs_config" and "target_contract_sha256" in str(exc.value) and V2 in str(exc.value)
    assert not (e.w.repo / REG_REL).exists() and not _loaded(e)


def test_불일치_가2_등록_뒤_학습_원문이_1차_계약으로_바뀌면_실제_export_등록_판_짝이_생성기_대역_전에_멈춘다(e2e, tmp_path):
    """판 6 등록까지 2 차로 선 뒤 학습 산출의 원문만 1 차 계약(아는 계약)으로 바뀐다 — 계약에서 되찾은 레시피(1 차)의 판(5)이 등록의
    판(6)과 다르다. 실제: export 사전 점검(계약 → 레시피 → 등록 판 짝)."""
    e = e2e
    reg, raw, tc_text = _register(e)
    rp = _commit(e, reg, raw, tc_text)
    e.write_metas(train_config_text(init=e.init, recipe=V1))
    folder = tmp_path / "main_out" / "export"
    assert _export(e, rp, folder) == "registration_version"
    assert not _loaded(e) and not folder.exists()


def test_불일치_나_계약_sha_가_다르면_실제_export_사전_점검이_생성기_대역_전에_멈춘다(e2e, tmp_path):
    """평가 등록(판 6)까지 같은 신원으로 선 뒤, 학습 산출 meta 의 원문만 모르는 계약으로 바뀐다. 실제: export 사전 점검
    (`recipe_of_contract` → `train_config_contract`)."""
    e = e2e
    reg, raw, tc_text = _register(e)
    rp = _commit(e, reg, raw, tc_text)
    e.write_metas(train_config_text(init=e.init, contract="0" * 64))
    folder = tmp_path / "main_out" / "export"
    assert _export(e, rp, folder) == "train_config_contract"
    assert not _loaded(e) and not folder.exists()


def test_불일치_다_판_5_등록에_2차_산출이면_실제_export_등록_판_짝이_생성기_대역_전에_멈춘다(e2e, tmp_path):
    """본 등록 짓기는 2 차 레시피에 판 5 를 짓지 않는다(판 6 연결 2 판 — 새 체제 입력 누락은 `v6_form`). 판 5 꼴로 고친 등록으로 2 차
    산출을 export 하면 등록 판 짝(지시 1023c 의 export 검사)에서 멈춘다. 실제: 본 등록 짓기의 거부 · export 사전 점검(등록 판 짝)."""
    e = e2e
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        _register(e, profile=None, pin_raw=None, inputs_contract_raw=None, repo=None)
    assert exc.value.code == "v6_form"                                 # 새 체제에 구판 인자 경로 — 판 5 를 짓지 않는다
    reg6, _raw6, tc_text = _register(e)
    reg = _as_v5(reg6, limit=512)                                      # 옛 경로의 결속(1 차 한도 · 1 차 base)은 지나고 등록 판 짝에서 멈춘다
    assert reg.version == P.REGISTRATION_VERSION and reg.narrative is None
    base = e.cfg_path.read_text(encoding="utf-8").replace(f"uni_max_new_tokens: {LIMIT}", "uni_max_new_tokens: 512")
    assert base != e.cfg_path.read_text(encoding="utf-8")
    e.cfg_path.write_text(base, encoding="utf-8")                      # 그 anchor 의 base 도 1 차 한도 — 같은 커밋에
    W.git(e.w.repo, "add", "configs/base.yaml")
    rp = _commit(e, reg, P.dump_registration(reg), tc_text, write_receipt=False)
    folder = tmp_path / "main_out" / "export"
    assert _export(e, rp, folder) == "registration_version"
    assert not _loaded(e) and not folder.exists()


def test_불일치_라_프롬프트_sha_가_다르면_실제_본_등록_짓기가_멈춘다(e2e):
    """학습 산출의 원문이 2 차 계약에 1 차 프롬프트(v2)의 sha — 설정의 프롬프트(3판)와 다르다. 실제: 본 등록 짓기의 원문 ↔ 설정 대조
    (`train_config_vs_config`)."""
    e = e2e
    metas = e.write_metas(train_config_text(init=e.init, prompt_sha=R.regime_prompt_sha256(V1)))
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        _register(e, metas=metas)
    assert exc.value.code == "train_config_vs_config" and "prompt_sha256" in str(exc.value)
    assert not (e.w.repo / REG_REL).exists() and not _loaded(e)
