"""본실험 등록의 채점 쪽 입력 JSON — 값이 등록 칸의 조건 · 관계식 · 규칙 ⑤ 의 입력을 지나고, 키 집합이 등록 짓기
(`vlm.main_registration`)가 '채점 쪽 입력이 채울 칸' 으로 알리는 빈 칸과 같다. 값이 없는 칸은 null 로 두었고, 등록 짓기는 그 이름으로
거부한다 — 값을 꾸며 넣지 않는다."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

J = ROOT / "docs/dev_log/2026-10-16-본실험실행/eval_lists_D/scoring_input_main_v1.json"
SCRIPT = J.parent / "make_scoring_input.py"
AREAS = ("metric", "baseline", "ci", "population", "canary")
OPEN = ["canary.frame_diag_sha256", "canary.known_answer_sha256"]
"""값이 없는 칸 — 통과한 진단 산출물이 없고, 커밋한 M1 기지답 픽스처가 없다."""
pytestmark = pytest.mark.skipif(not J.is_file(), reason="채점 쪽 입력 JSON 이 이 트리에 없다")


def _doc() -> dict:
    return json.loads(J.read_text(encoding="utf-8"))


def _flat(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k in AREAS:
            out.update({f"{k}.{kk}": vv for kk, vv in v.items()})
        else:
            out[k] = v
    return out


def test_파일은_LF_정규_꼴이고_값이_없는_칸은_둘이다() -> None:
    raw = J.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert raw == (json.dumps(_doc(), ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
    assert sorted(n for n, v in _flat(_doc()).items() if v is None) == OPEN


def test_값은_등록_칸의_조건과_관계식과_규칙5_입력을_지난다() -> None:
    from evaluation.prereg_unified import UnifiedRegistration, _kinds
    from evaluation.rule5 import registered_inputs

    reg = UnifiedRegistration()
    reg.generation.purpose = "main"
    reg.seeds = {"1": 20260828}                                   # 시드표는 등록 짓기가 설정에서 채운다 — 규칙 ⑤ 의 시드 수 대조용
    kinds = _kinds()
    flat = _flat(_doc())
    for name, v in flat.items():
        area, _, key = name.partition(".")
        if not key:
            setattr(reg, area, v)
            continue
        setattr(getattr(reg, area), key, tuple(v) if kinds[name] == "str_tuple" and isinstance(v, list) else v)
    assert reg.invalid(side="scoring") == []
    got = registered_inputs(reg)
    assert (got.n_resamples, got.min_defined_draws, sorted(got.seeds)) == (2000, 1900, [1])
    gaps = set(reg.missing(side="scoring"))
    assert set(OPEN) <= gaps and not gaps & {n for n, v in flat.items() if v is not None}


def test_키_집합은_등록_짓기가_알리는_빈_칸과_같고_v2_는_빈_칸_없이_결속을_지난다(tmp_path) -> None:
    """등록 짓기의 시험과 같은 합성 생성 쪽 입력(`tests.uni_export_world`)에 v1 과, 스크립트가 시험용 대역 허가 파일로 낸 v2 를 넣는다.
    등록 짓기가 같은 트리에 있을 때만 돈다."""
    MR = pytest.importorskip("vlm.main_registration")
    import yaml

    from evaluation.entry_gate import read_manifest_meta
    from evaluation.eval_list import canonical_bytes, validate_eval_list
    from evaluation.eval_registration import strict_binding_problems
    from evaluation.recovery_interval import registered_mismatches
    from evaluation.rule5 import registered_inputs
    from tests import uni_export_world as W
    from vlm import pilot_vlm as PV
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.pilot_vlm import load_prompt

    w = W.build(tmp_path / "w")
    gen_raw, echo_raw = canonical_bytes([W.EVAL_ID]), canonical_bytes(list(W.IDS))
    gen = validate_eval_list(gen_raw)
    doc = yaml.safe_load((ROOT / "configs/eval_registration.yaml").read_text(encoding="utf-8"))
    doc["population"]["sample"]["main_size"] = 1000               # 합성 세계의 평가 한 장을 크기 1000 의 봉인으로 둔다
    doc["population"]["sample"]["sealed"]["1000"] = {"images": 1, "groups": 1, "file_sha256": gen.file_sha256,
                                                     "set_sha256": gen.set_sha256, "side_sha256": "0" * 64}
    init = hashlib.sha256(b"init").hexdigest()
    cfg = SimpleNamespace(seeds=(20260828, 20260829, 20260830), snapshot_digest=w.digest, model_id="Qwen/Qwen3.5-4B",
                          model_revision="a" * 40, prompt_path=W.PROMPT_REL, chat_template_kwargs={"enable_thinking": False},
                          coord_space="ABS_ORIG", batch_size=1, max_new_tokens=512, total_epochs=1, num_rounds=1,
                          local_epochs=1, processor_kwargs={"max_pixels": W.MAX_PIXELS}, pairs_digest="p" * 64)
    measured = {k: W.MEASURED[k] for k in ("processor_config_sha256", "processor_min_pixels", "processor_max_pixels",
                                           "patch_size", "merge_size", "gen_prefix_sha256", "transformers_version")}
    recipe = "json_only"                                          # 학습 레시피 — 1 차 기본
    contract_of = getattr(PV, "target_contract_sha256", None)    # 레시피별 계약 함수가 있는 판은 그 값, 앞 판은 계약 상수
    tc = json.dumps({"model_id": cfg.model_id, "model_revision": cfg.model_revision,
                     "prompt_sha256": load_prompt(ROOT / W.PROMPT_REL)[1],
                     "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG")),
                     "processor_config_sha256": measured["processor_config_sha256"], "init_adapter_digest": init,
                     "target_contract_sha256": contract_of(recipe) if contract_of else PV.TARGET_CONTRACT_SHA256,
                     "num_rounds": 1, "local_epochs": 1, "total_epochs": 1}, sort_keys=True, separators=(",", ":"))
    metas = {t: {"identity": {"purpose": "main", "tag": t, "seed_index": 1, "seed_value": cfg.seeds[0]},
                 "init_adapter_digest": init, "train_config": tc} for t in MR.CELLS}
    kw = {"cfg": cfg, "seed_index": 1, "gen_raw": gen_raw, "echo_raw": echo_raw, "eval_doc": doc,
          "manifest_meta": read_manifest_meta(w.snap), "snapshot_digest": w.digest, "measured": measured,
          "init_digest": init,
          "pin": {"init_adapter": {"init_adapter_digest": init}, "common": {"registration": {"recipe": recipe}}},
          "metas": metas,
          "impl_ids": ("export_measure=vlm.export_measure:measure_env@vlm/export_measure.py",),
          "prompt_path": ROOT / W.PROMPT_REL, "now": datetime(2026, 10, 17, 9, 0, tzinfo=UTC)}

    def refused(scoring) -> tuple[str, object]:
        with pytest.raises(MR.MainRegistrationRefused) as exc:
            MR.build_main_registration(**kw, scoring=scoring)
        return exc.value.code, exc.value.detail

    code, gaps = refused({})
    assert code == "registration_incomplete" and set(gaps) == set(_flat(_doc()))          # 키 집합 = 등록 짓기의 입력 집합
    assert refused(_doc()) == ("registration_incomplete", OPEN)                              # v1 — 모르는 칸 · 충돌 없이 빈 칸 둘만
    from evaluation import known_answer as KA
    from tests.test_score_unified_strict import _ecc_rules_raw, _permit_raw

    rules_raw = _ecc_rules_raw()                                     # 시험용 대역 — 아침에는 시드 폴더의 ecc_permit.json 과 저장소 규칙
    permit = _permit_raw(rules_raw)
    (tmp_path / "rules.yaml").write_bytes(rules_raw)
    (tmp_path / "permit.json").write_bytes(permit)
    c_raw = (ROOT / "vlm/main_registration.py").read_bytes()
    blob = hashlib.sha1(b"blob %d\x00" % len(c_raw) + c_raw).hexdigest()
    subprocess.run([sys.executable, "-X", "utf8", "-B", str(SCRIPT), str(ROOT / "vlm/main_registration.py"), blob,
                    "--ecc-permit", str(tmp_path / "permit.json"), "--ecc-rules", str(tmp_path / "rules.yaml"),
                    "--out-dir", str(tmp_path)], check=True, capture_output=True)
    v2 = json.loads((tmp_path / "scoring_input_main_v2.json").read_text(encoding="utf-8"))
    assert set(_flat(v2)) == set(_flat(_doc())) and [n for n, v in _flat(v2).items() if v is None] == []
    assert {n for n in _flat(v2) if _flat(v2)[n] != _flat(_doc())[n]} == set(OPEN)          # v1 과 다른 칸은 두 칸뿐
    assert v2["canary"]["frame_diag_sha256"] == hashlib.sha256(permit).hexdigest()
    assert v2["canary"]["known_answer_sha256"] == KA.known_answer_sha256((ROOT / KA.FIXTURE_REL).read_bytes())
    reg, _raw, _tc = MR.build_main_registration(**kw, scoring=v2)                         # 빈 칸 0 — 등록이 지어진다
    from evaluation.early_coord_check import load_rules
    from evaluation.ecc_frame_check import model_free_bundle_sha256
    from tests.test_ecc_frame_check import LOG, _admission, _waiver

    adm = _admission(load_rules(rules_raw))                          # 허가 면제 갈래 — 통과 산출 · 면제 파일 · 의사결정로그(대역)
    wv = _waiver(adm)
    (tmp_path / "admission.json").write_bytes(adm)
    (tmp_path / "waiver.json").write_bytes(wv)
    (tmp_path / "log.md").write_text(LOG, encoding="utf-8")
    (tmp_path / "waived").mkdir()
    subprocess.run([sys.executable, "-X", "utf8", "-B", str(SCRIPT), str(ROOT / "vlm/main_registration.py"), blob,
                    "--frame-admission", str(tmp_path / "admission.json"), "--ecc-waiver", str(tmp_path / "waiver.json"),
                    "--decision-log", str(tmp_path / "log.md"), "--ecc-rules", str(tmp_path / "rules.yaml"),
                    "--out-dir", str(tmp_path / "waived")], check=True, capture_output=True)
    v2w = json.loads((tmp_path / "waived/scoring_input_main_v2.json").read_text(encoding="utf-8"))
    assert {n for n in _flat(v2w) if _flat(v2w)[n] != _flat(v2)[n]} == {"canary.frame_diag_sha256"}
    assert v2w["canary"]["frame_diag_sha256"] == model_free_bundle_sha256(adm, wv)
    regw, _raw_w, _tc_w = MR.build_main_registration(**kw, scoring=v2w)                 # 면제 갈래도 빈 칸 0
    assert regw.missing(side="all") == [] and strict_binding_problems(regw, doc) == []
    both = subprocess.run([sys.executable, "-X", "utf8", "-B", str(SCRIPT), str(ROOT / "vlm/main_registration.py"), blob,
                           "--ecc-permit", str(tmp_path / "permit.json"), "--frame-admission", str(tmp_path / "admission.json"),
                           "--ecc-waiver", str(tmp_path / "waiver.json"), "--out-dir", str(tmp_path / "waived")],
                          capture_output=True, check=False)
    assert both.returncode != 0                                       # 허가와 면제를 함께 주면 내지 않는다
    assert reg.missing(side="all") == [] and reg.invalid(side="all") == []
    assert strict_binding_problems(reg, doc) == [] and registered_mismatches(reg.ci) == []
    assert registered_inputs(reg).statistic == _doc()["ci"]["rule5_statistic"]
