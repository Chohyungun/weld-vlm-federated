"""본실험 등록(④)을 짓는 길 — `vlm.main_registration` · `scripts/main_uni.py register · receipt`(지시 1016l, D 04 §2). 모델 · GPU 없이.

합성 산출 파일(시드 폴더의 착수 고정 · 초기 어댑터 · 칸 다섯의 어댑터 meta, 실측 파일, 합성 동결 스냅샷)로 등록이 지어지는지,
학습 설정 해시가 어댑터 meta 원문과 같은지, 빈 칸 · 어긋난 입력을 거부하는지, 영수증이 등록이 병합된 **다음** 커밋이라는 README 규칙과
부딪치지 않는지 본다. 에코 목록의 정의 안(`echo_list_ids`)은 ECC 대상 규칙과 같은 값을 train 에서 낸다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.eval_list import canonical_bytes, validate_eval_list
from scripts import main_uni as M
from tests import uni_export_world as W
from tests.test_prereg_unified import filled
from vlm import main_registration as MR

SEEDS = (20260828, 20260829, 20260830)
EVAL_DOC = yaml.safe_load((ROOT / "configs/eval_registration.yaml").read_text(encoding="utf-8"))
NOW = __import__("datetime").datetime(2026, 10, 17, 9, 0, tzinfo=__import__("datetime").timezone.utc)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """합성 동결 스냅샷(평가 한 장 · val 넷)과 그 위의 본실험 입력 — 생성 목록은 평가 한 장(크기 1000 의 봉인으로 둔다), 에코는 val."""
    from evaluation.entry_gate import read_manifest_meta
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.pilot_vlm import TARGET_CONTRACT_SHA256, load_prompt

    w = W.build(tmp_path_factory.mktemp("mainreg"))
    gen_raw, echo_raw = canonical_bytes([W.EVAL_ID]), canonical_bytes(list(W.IDS))
    gen = validate_eval_list(gen_raw)
    doc = copy.deepcopy(EVAL_DOC)
    doc["population"]["sample"]["main_size"] = 1000
    doc["population"]["sample"]["sealed"]["1000"] = {"images": 1, "groups": 1, "file_sha256": gen.file_sha256,
                                                     "set_sha256": gen.set_sha256, "side_sha256": "0" * 64}
    init = _sha(b"init")
    cfg = SimpleNamespace(seeds=SEEDS, snapshot_digest=w.digest, model_id="Qwen/Qwen3.5-4B", model_revision="a" * 40,
                          prompt_path=W.PROMPT_REL, chat_template_kwargs={"enable_thinking": False}, coord_space="ABS_ORIG",
                          batch_size=1, max_new_tokens=512, total_epochs=1, num_rounds=1, local_epochs=1,
                          processor_kwargs={"max_pixels": W.MAX_PIXELS}, pairs_digest="p" * 64)
    measured = {k: W.MEASURED[k] for k in ("processor_config_sha256", "processor_min_pixels", "processor_max_pixels",
                                           "patch_size", "merge_size", "gen_prefix_sha256", "transformers_version")}
    tc = json.dumps({"model_id": cfg.model_id, "model_revision": cfg.model_revision,
                     "prompt_sha256": load_prompt(ROOT / W.PROMPT_REL)[1],
                     "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG")),
                     "processor_config_sha256": measured["processor_config_sha256"], "init_adapter_digest": init,
                     "target_contract_sha256": TARGET_CONTRACT_SHA256,
                     "num_rounds": 1, "local_epochs": 1, "total_epochs": 1}, sort_keys=True, separators=(",", ":"))
    return SimpleNamespace(w=w, gen_raw=gen_raw, echo_raw=echo_raw, doc=doc, init=init, cfg=cfg, measured=measured, tc=tc,
                           meta=read_manifest_meta(w.snap))


def _contract_v2() -> str:
    from vlm.pilot_vlm import target_contract_sha256

    return target_contract_sha256("json_line_then_basis")


def _metas(world, *, tc=None, init=None, seed=SEEDS[0], drop=()):
    return {t: {"identity": {"purpose": "main", "tag": t, "seed_index": 1, "seed_value": seed},
                "init_adapter_digest": init or world.init, "train_config": tc or world.tc} for t in MR.CELLS if t not in drop}


def _kw(world, **over):
    kw = {"cfg": world.cfg, "seed_index": 1, "gen_raw": world.gen_raw, "echo_raw": world.echo_raw, "eval_doc": world.doc,
          "manifest_meta": world.meta, "snapshot_digest": world.w.digest, "measured": world.measured,
          "init_digest": world.init, "pin": {"init_adapter": {"init_adapter_digest": world.init},
                                             "common": {"registration": {"recipe": "json_only"}}},
          "metas": _metas(world), "scoring": {}, "impl_ids": ("export_measure=vlm.export_measure:measure_env@vlm/export_measure.py",),
          "prompt_path": ROOT / W.PROMPT_REL, "now": NOW}
    kw.update(over)
    return kw


def _scoring(world) -> dict:
    """평가 쪽이 값을 낼 칸만 — 등록 짓기가 빈 칸으로 알린 이름에 시험의 유효 값(`filled`)을 넣는다. JSON 을 한 번 지난다."""
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        MR.build_main_registration(**_kw(world))
    assert exc.value.code == "registration_incomplete"
    gaps, F = exc.value.detail, filled()
    out: dict = {}
    for name in gaps:
        area, _, key = name.partition(".")
        if key:
            out.setdefault(area, {})[key] = getattr(getattr(F, area), key)
        else:
            out[area] = getattr(F, area)
    out["planned_seed_count"] = 1                                          # 이 등록의 시드는 하나다(규칙 ⑤)
    return json.loads(json.dumps(out, default=list))


def test_합성_산출로_본실험_등록이_지어지고_학습_설정_해시가_어댑터_meta_원문과_같다(world, tmp_path):
    from evaluation.eval_registration import strict_binding_problems
    from evaluation.prereg_unified import load_registration
    from evaluation.recovery_interval import REGISTERED_CI

    scoring = _scoring(world)
    assert set(scoring) - {"planned_seed_count", "adoption_anchor"} <= set(MR.SCORING_AREAS)
    assert "ci" in scoring and "n_resamples" in scoring["ci"] and "prediction_sha256" in scoring["baseline"]
    assert not set(scoring.get("ci", {})) & set(REGISTERED_CI)               # 구간 규약은 구현에서 — 입력으로 받지 않는다
    reg, raw, tc_text = MR.build_main_registration(**_kw(world, scoring=scoring))
    g = reg.generation
    assert tc_text == world.tc and g.train_config_sha256 == _sha(world.tc.encode("utf-8"))
    assert (g.purpose, g.list_split, g.snapshot_digest, g.init_adapter_digest) == ("main", "eval", world.w.digest, world.init)
    assert g.eval_list_file_sha256 == validate_eval_list(world.gen_raw).file_sha256
    assert g.echo_list_file_sha256 == validate_eval_list(world.echo_raw).file_sha256
    assert {k: getattr(g, k) for k in world.measured} == world.measured and g.max_new_tokens == 512
    assert (g.budget_n, g.budget_r, g.budget_e) == (1, 1, 1) and g.train_rows_digest is None and g.plan_sha256 is None
    assert reg.seeds == {"1": SEEDS[0]} and reg.population.n_images == 1 and reg.population.n_clusters == 1
    assert reg.missing(side="all") == [] and strict_binding_problems(reg, world.doc) == []
    assert load_registration(raw).generation_sha256() == reg.generation_sha256()           # 정규형 바이트
    files = MR.write_main_registration(tmp_path / "out", stem="main-20261017-1", raw=raw, tc_text=tc_text,
                                       gen_raw=world.gen_raw, echo_raw=world.echo_raw)
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == ["main-20261017-1.echo.list", "main-20261017-1.generation.list", "main-20261017-1.json",
                     "main-20261017-1.train_config.json"]                                   # 영수증은 없다(다음 커밋)
    assert files["train_config"].read_bytes() == world.tc.encode("utf-8")                  # 끝 LF 없음
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        MR.write_main_registration(tmp_path / "out", stem="main-20261017-1", raw=raw, tc_text=tc_text,
                                   gen_raw=world.gen_raw, echo_raw=world.echo_raw)
    assert exc.value.code == "name_taken"


def _base_for(world) -> bytes:
    """영수증 커밋의 base 설정 — 판독(`load_bound_eval`)이 1 차 설정의 닻 · 생성 한도와 맞댄다."""
    fixed = {"snapshot_id": "synthetic_v1", "snapshot_digest": world.doc["population"]["snapshot_digest"],
             "uni_max_new_tokens": world.doc["generation"]["max_new_tokens"]}
    return yaml.safe_dump({"fixed_before_main_runs": fixed}, allow_unicode=True).encode("utf-8")


def _code(world, **over) -> str:
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        MR.build_main_registration(**_kw(world, **over))
    return exc.value.code


@pytest.mark.parametrize(("over", "code"), [
    (lambda w: {"metas": _metas(w, drop=("uni_fed",))}, "cells"),
    (lambda w: {"metas": {**_metas(w), "uni_fed": {**_metas(w)["uni_fed"], "train_config": w.tc + " "}}},
     "train_config_differs"),
    (lambda w: {"metas": _metas(w, init=_sha(b"other"))}, "adapter_init"),
    (lambda w: {"metas": _metas(w, seed=SEEDS[1])}, "adapter_seed"),
    (lambda w: {"metas": {**_metas(w), "uni_central": {**_metas(w)["uni_central"], "train_config": ""}}},
     "train_config_missing"),
    (lambda w: {"pin": {"init_adapter": {"init_adapter_digest": _sha(b"other")}}}, "init_vs_pin"),
    # 레시피 — 착수 고정에 묶인 등록의 레시피가 없거나 모르면 거부, 레시피의 계약이 학습 설정 원문의 계약과 다르면 거부
    (lambda w: {"pin": {"init_adapter": {"init_adapter_digest": w.init}}}, "train_recipe"),
    (lambda w: {"pin": {"init_adapter": {"init_adapter_digest": w.init},
                        "common": {"registration": {"recipe": "two_task_mix"}}}}, "train_recipe"),
    (lambda w: {"pin": {"init_adapter": {"init_adapter_digest": w.init},
                        "common": {"registration": {"recipe": "json_line_then_basis"}}}}, "train_config_vs_config"),
    # 체제 짝(지시 1022c 덧붙임 A) — 등록 레시피 · 원문의 계약은 2 차인데 학습한 프롬프트는 v2(1 차 프롬프트)
    (lambda w: {"pin": {"init_adapter": {"init_adapter_digest": w.init},
                        "common": {"registration": {"recipe": "json_line_then_basis"}}},
                "metas": _metas(w, tc=json.dumps({**json.loads(w.tc), "target_contract_sha256": _contract_v2()},
                                                 sort_keys=True, separators=(",", ":")))}, "train_regime"),
    (lambda w: {"gen_raw": canonical_bytes([W.EVAL_ID, W.IDS[0]])}, "generation_list_not_sealed"),
    (lambda w: {"echo_raw": canonical_bytes([W.EVAL_ID])}, "list_split"),                # 에코는 val 이어야 한다
    (lambda w: {"snapshot_digest": "f" * 64}, "snapshot"),
    (lambda w: {"seed_index": 4}, "seed_index"),
    (lambda w: {"measured": {**w.measured, "processor_config_sha256": _sha(b"x")}}, "train_config_vs_config"),
    (lambda w: {"cfg": SimpleNamespace(**{**vars(w.cfg), "total_epochs": 2})}, "budget"),
    (lambda w: {"cfg": SimpleNamespace(**{**vars(w.cfg), "max_new_tokens": None})}, "config_incomplete"),
])
def test_어긋나거나_빈_입력은_등록을_짓지_않는다(world, over, code):
    assert _code(world, scoring=_scoring(world), **over(world)) == code


def test_채점_쪽_입력은_빈_칸_없이_모르는_칸_없이_구현과_같아야_한다(world):
    s = _scoring(world)
    drop = copy.deepcopy(s)
    area = next(a for a in ("ci", "baseline") if a in drop)
    key = min(drop[area])
    del drop[area][key]
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        MR.build_main_registration(**_kw(world, scoring=drop))
    assert exc.value.code == "registration_incomplete" and exc.value.detail == [f"{area}.{key}"]
    assert _code(world, scoring={**s, "ci": {**s["ci"], "alpha": 0.1}}) == "scoring_conflict"          # 구현의 α 와 다르다
    assert _code(world, scoring={**s, "population": {**s.get("population", {}), "n_images": 2}}) == "scoring_conflict"
    assert _code(world, scoring={**s, "seeds": {"1": 1}}) == "scoring_keys"                              # 시드표는 설정에서
    assert _code(world, scoring={**s, "ci": {**s["ci"], "nope": 1}}) == "scoring_keys"
    assert _code(world, scoring={**s, "canary": {**s.get("canary", {}), "coord_cfg": {}}}) == "scoring_keys"
    assert _code(world, scoring={**s, "planned_seed_count": 3}) == "eval_binding"                        # 규칙 ⑤ 의 시드 수


def test_실측_파일은_서버_실측_꼴과_실측_함수의_출력을_받고_일곱_칸이어야_한다(world):
    prov = {"measured_again": world.measured, "source_registration_sha256": "0" * 64, "training_rows": 3}
    assert MR.read_measurement(json.dumps(prov).encode()) == world.measured
    assert MR.read_measurement(json.dumps(world.measured).encode()) == world.measured
    for bad, code in (({**world.measured, "patch_size": True}, "measurement_value"),
                      ({**world.measured, "gen_prefix_sha256": "x"}, "measurement_value"),
                      ({k: v for k, v in world.measured.items() if k != "merge_size"}, "measurement_keys"),
                      ({**world.measured, "extra": 1}, "measurement_keys")):
        with pytest.raises(MR.MainRegistrationRefused) as exc:
            MR.read_measurement(json.dumps({"measured_again": bad}).encode())
        assert exc.value.code == code


def test_에코_목록의_정의_안은_ECC_대상_규칙을_val_에_건_것이다():
    from evaluation.early_coord_check import target_ids

    rows = [{"image_id": f"{s}{i}", "split": s, "skeleton": {"defects": [{"type": "2011"}] if i % 2 else []}}
            for s in ("train", "val", "eval") for i in range(8)]
    removed = {"train1", "val3", "eval1"}
    assert MR.echo_list_ids(rows, removed, split="train") == target_ids(rows, removed)        # 같은 규칙
    assert MR.echo_list_ids(rows, removed) == ["val1", "val5", "val7"]                         # 결함 장 · R0 뒤 · 정렬


def test_구현_식별자는_export_의_실제_구현이고_대역이_없다():
    ids = MR.export_impl_ids("main")
    assert ids == MR.export_impl_ids("main") and len(ids) >= 2
    assert any(i.startswith("export_measure=") for i in ids) and any(i.startswith("export_generator=") for i in ids)


# ---------------------------------------------------------------- 명령줄 — 시드 폴더에서 짓는다
def _seed_folder(world, out: Path) -> None:
    from vlm.pin import PIN_SCHEMA, init_probe

    sd = out / "seed1"
    (sd / "fl" / "uni_fed").mkdir(parents=True)
    np.savez(sd / "fl" / "uni_fed" / "initial.npz", a=np.full((2, 2), 3.0, dtype=np.float32))
    init = init_probe(sd / "fl" / "uni_fed" / "initial.npz")["init_adapter_digest"]
    (sd / "PIN.json").write_text(json.dumps({"schema": PIN_SCHEMA, "init_adapter": {"init_adapter_digest": init},
                                             "common": {"registration": {"recipe": "json_only"}}}), encoding="utf-8")
    tc = world.tc.replace(world.init, init)
    for tag, meta in _metas(world, tc=tc, init=init).items():
        d = sd / f"{tag}_s1"
        d.mkdir()
        (d / "adapter_last.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_등록_이름은_저장소_등록_폴더와_출력_폴더의_같은_날_번호_다음이다(tmp_path):
    """이름 짓기는 실제 저장소의 `configs/registration/` 을 읽는다 — 이미 병합된 이름을 다시 쓰지 않는다(게이트 96 에서 `-1` 이
    병합되자 다음 이름이 `-2` 가 됐다). 다른 날의 번호 · 숫자 아닌 꼬리는 세지 않는다."""
    repo_reg, out = tmp_path / "repo_registration", tmp_path / "reg"
    repo_reg.mkdir()
    out.mkdir()
    assert M._next_reg_name(NOW, repo_reg, out) == "main-20261017-1"
    for name in ("main-20261017-1.json", "main-20261017-1.receipt.json", "main-20261016-7.json", "main-20261017-x.json"):
        (repo_reg / name).write_text("{}", encoding="utf-8")
    assert M._next_reg_name(NOW, repo_reg, out) == "main-20261017-2"
    (out / "main-20261017-5.json").write_text("{}", encoding="utf-8")
    assert M._next_reg_name(NOW, repo_reg, out) == "main-20261017-6"


def test_명령줄_register_는_시드_폴더와_파일로_등록_넷을_쓰고_receipt_는_병합_다음_커밋에서_쓴다(world, tmp_path, capsys,
                                                                             monkeypatch):
    # 이름 짓기는 실제 저장소의 등록 폴더를 읽는다(코드의 뜻 — 위 시험). 이 시험은 저장소에 무엇이 병합됐는지와 무관해야 하므로
    # 그 자리만 빈 임시 폴더로 바꾸고, 실행기가 실제 저장소 폴더를 넘기는지는 따로 본다.
    real_next, seen_dirs, repo_reg = M._next_reg_name, [], tmp_path / "repo_registration"
    repo_reg.mkdir()

    def isolated_next(now, repo_dir, *rest):
        seen_dirs.append(Path(repo_dir))
        return real_next(now, repo_reg, *rest)

    monkeypatch.setattr(M, "_next_reg_name", isolated_next)
    out = tmp_path / "main_u"
    _seed_folder(world, out)
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "gen.list").write_bytes(world.gen_raw)
    (inputs / "echo.list").write_bytes(world.echo_raw)
    (inputs / "scoring.json").write_text(json.dumps(_scoring(world)), encoding="utf-8")
    (inputs / "MEASUREMENT_PROVENANCE.json").write_text(json.dumps({"measured_again": world.measured}), encoding="utf-8")
    (inputs / "eval_registration.yaml").write_text(yaml.safe_dump(world.doc, allow_unicode=True), encoding="utf-8")
    kw = {"generation_list": inputs / "gen.list", "echo_list": inputs / "echo.list", "scoring": inputs / "scoring.json",
          "snapshot": world.w.snap, "reg_out": tmp_path / "reg", "measurement": inputs / "MEASUREMENT_PROVENANCE.json",
          "out": out, "now": NOW, "eval_registration": inputs / "eval_registration.yaml"}
    files = M.stage_register_main(world.cfg, 1, **kw)
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["name"] == "main-20261017-1" and files["registration"].name == "main-20261017-1.json"
    assert seen_dirs[0] == M.REPO_ROOT / "configs" / "registration"
    assert {v["sha256"] for v in report["files"].values()} == {_sha(p.read_bytes()) for p in files.values()}
    with pytest.raises(SystemExit):                                       # 같은 이름은 다시 쓰지 않는다
        M.stage_register_main(world.cfg, 1, name="main-20261017-1", **kw)
    assert "[name_taken]" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        M.stage_register_main(world.cfg, 1, **{**kw, "scoring": inputs / "none.json"})
    assert "[register_inputs]" in capsys.readouterr().err
    (out / "seed1" / "uni_fed_s1" / "adapter_last.meta.json").unlink()
    with pytest.raises(SystemExit):
        M.stage_register_main(world.cfg, 1, **kw)
    assert "[register_meta]" in capsys.readouterr().err

    # 영수증 — 본줄기에 등록 넷 · 평가 등록 설정을 병합한 커밋에서
    repo = tmp_path / "repo"
    (repo / "configs" / "registration").mkdir(parents=True)
    W.git(repo, "init", "-q", "-b", "main")
    (repo / "configs" / "eval_registration.yaml").write_bytes((inputs / "eval_registration.yaml").read_bytes())
    (repo / "configs" / "base.yaml").write_bytes(_base_for(world))              # 판독은 그 커밋의 base 도 맞댄다
    for p in files.values():
        (repo / "configs" / "registration" / p.name).write_bytes(p.read_bytes())
    W.git(repo, "add", "-A")
    W.git(repo, "commit", "-q", "-m", "등록")
    head = W.git(repo, "rev-parse", "HEAD")
    rel = "configs/registration/main-20261017-1.json"
    rp = MR.write_main_receipt(registration_rel=rel, commit=head, checkout=repo, now=NOW)
    rc = json.loads(rp.read_text(encoding="utf-8"))
    assert (rc["kind"], rc["main_commit"], rc["registration_path"]) == ("main", head, rel)
    assert rc["registration_file_sha256"] == _sha(files["registration"].read_bytes())
    from evaluation.prereg_unified import read_receipt

    assert read_receipt(rp).kind == "main"
    with pytest.raises(FileExistsError):                                 # 배타 생성
        MR.write_main_receipt(registration_rel=rel, commit=head, checkout=repo, now=NOW)
    W.git(repo, "add", "-A")
    W.git(repo, "commit", "-q", "-m", "영수증")
    head2 = W.git(repo, "rev-parse", "HEAD")
    rp.unlink()
    with pytest.raises(MR.MainRegistrationRefused) as exc:              # 영수증이 든 커밋을 등록 커밋으로 주면 거부
        MR.write_main_receipt(registration_rel=rel, commit=head2, checkout=repo, now=NOW)
    assert exc.value.code == "receipt_same_commit"


def test_영수증은_본줄기_조상이고_커밋의_바이트가_등록과_같아야_쓴다(world, tmp_path):
    _reg, raw, tc_text = MR.build_main_registration(**_kw(world, scoring=_scoring(world)))
    repo = tmp_path / "repo"
    d = repo / "configs" / "registration"
    d.mkdir(parents=True)
    W.git(repo, "init", "-q", "-b", "main")
    (repo / "configs" / "eval_registration.yaml").write_text(yaml.safe_dump(world.doc, allow_unicode=True), encoding="utf-8")
    (repo / "configs" / "base.yaml").write_bytes(_base_for(world))
    MR.write_main_registration(d, stem="main-20261017-2", raw=raw, tc_text=tc_text, gen_raw=world.gen_raw,
                               echo_raw=world.echo_raw)
    (d / "main-20261017-2.echo.list").write_bytes(canonical_bytes(list(W.IDS[:2])))         # 다른 에코 목록을 병합했다
    W.git(repo, "add", "-A")
    W.git(repo, "commit", "-q", "-m", "등록")
    rel = "configs/registration/main-20261017-2.json"
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        MR.write_main_receipt(registration_rel=rel, commit="HEAD", checkout=repo, now=NOW)
    assert exc.value.code == "list_vs_registration"
    W.git(repo, "checkout", "-q", "-b", "side")
    (d / "main-20261017-2.echo.list").write_bytes(world.echo_raw)
    W.git(repo, "commit", "-q", "-am", "곁가지")
    with pytest.raises(MR.MainRegistrationRefused) as exc:              # 본줄기의 조상이 아니다
        MR.write_main_receipt(registration_rel=rel, commit="HEAD", checkout=repo, now=NOW)
    assert exc.value.code == "receipt_not_ancestor"
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        MR.write_main_receipt(registration_rel="configs/other/main-x.json", commit="HEAD", checkout=repo, now=NOW)
    assert exc.value.code == "receipt_name"
    assert not list(d.glob("*.receipt.json"))


def test_명령줄의_등록_인자는_그_명령에서만_받는다(capsys, tmp_path):
    assert M.main(["pin", "--seed", "1", "--echo-list", "x"]) == 2
    assert "[register_args]" in capsys.readouterr().err
    assert M.main(["receipt", "--registration", "configs/registration/main-x.json"]) == 2
    assert "[receipt_args]" in capsys.readouterr().err
    assert M.main(["register", "--seed", "1"]) == 2                     # 새 레시피 — 뿌리부터(판 6 연결 2 판 §4-2)
    assert "[main_root_required]" in capsys.readouterr().err
    assert M.main(["register", "--seed", "1", "--main-out-root", str(tmp_path / "r")]) == 2
    assert "[register_inputs]" in capsys.readouterr().err


def test_에코_목록_빌더는_스냅샷_닻과_R0_지문을_맞대고_val_만_낸다(world, tmp_path):
    from scripts.probe import main_echo_list as EL

    rows = [{"image_id": i, "split": "val", "skeleton": {"defects": [{"type": "2011"}] if k != 1 else []}}
            for k, i in enumerate(W.IDS)] + [{"image_id": "t1", "split": "train", "skeleton": {"defects": [{"type": "2011"}]}}]
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    removed = tmp_path / "removed.txt"
    removed.write_text(f"{W.IDS[3]}\nt1\n", encoding="utf-8")
    sha = _sha(removed.read_bytes())
    raw, rep = EL.build(snapshot=world.w.snap, pairs=pairs, removed=removed, removed_sha256=sha, anchor_digest=world.w.digest)
    assert raw == canonical_bytes([W.IDS[0], W.IDS[2]]) and rep["n"] == 2 and W.IDS[0] not in json.dumps(rep)
    for kw, word in (({"removed_sha256": "0" * 64}, "R0"), ({"anchor_digest": "0" * 64}, "닻")):
        with pytest.raises(SystemExit, match=word):
            EL.build(**{"snapshot": world.w.snap, "pairs": pairs, "removed": removed, "removed_sha256": sha,
                        "anchor_digest": world.w.digest, **kw})


@pytest.mark.parametrize("where", ["measure", "loader", "inner_model"])
def test_구현_식별자는_대역이면_바깥이든_안쪽이든_등록에_싣지_않는다(monkeypatch, where):
    """G 05 m-5 — 바깥(실측 · 생성기 적재) 대역의 열쇠가 등록에 실려 이름을 버리는 길을 막는다. 안쪽 대역도 거부한다."""
    import vlm.export_generator as EG
    import vlm.export_measure as EM
    import vlm.export_model as EMo

    def stand_in(*a, **k):
        return None

    target = {"measure": (EM, "measure_env"), "loader": (EG, "load_generator"),
              "inner_model": (EMo, "load_model_generator")}[where]
    monkeypatch.setattr(*target, stand_in)
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        MR.export_impl_ids("main")
    assert exc.value.code == "standin_impl"


# ---------------------------------------------------------------- 판 6 연결 2 판 — 작성기(§4-1 · R1 · R2 · R3 · R16 · R20)
from dataclasses import replace

from evaluation import narrative_registration as NR
from tests.test_regime2_end_to_end import (
    REG_REL,
    STEM,
    _commit,
    _no_cuda,
    _register,
    approved,
    e2e,
)

FIXTURES = (e2e, approved, _no_cuda)
"""가져온 fixture — pytest 는 모듈 이름 공간의 fixture 를 이름으로 모은다."""


@pytest.mark.private_assets
def test_v6_link_recipe_version_mismatch(world, e2e):
    """R1 — 옛 json_only 에 새 판 6 입력(프로필 · PIN 바이트 · 계약)을 주거나, 새 레시피에 구판 프로필을 주면 `regime_version`. 파일 쓰기 0."""
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        MR.build_main_registration(**_kw(world, scoring=_scoring(world)), profile=NR.PROFILE_REGIME2)
    assert exc.value.code == "regime_version"
    with pytest.raises(MR.MainRegistrationRefused) as exc2:
        MR.build_main_registration(**_kw(world, scoring=_scoring(world)), pin_raw=b"{}")
    assert exc2.value.code == "regime_version"
    with pytest.raises(MR.MainRegistrationRefused) as exc3:
        _register(e2e, profile=NR.PROFILE_LEGACY)
    assert exc3.value.code == "regime_version"
    assert not (e2e.w.repo / REG_REL).exists()


@pytest.mark.private_assets
def test_v6_link_required_blocks_closed(e2e, tmp_path):
    """R2 — 신규의 계약 · 프로필 · PIN 바이트 누락은 `v6_form`, 블록을 입력으로 덮으려 하면 `scoring_conflict`, 모르는 키는 `scoring_keys`,
    판 6 에 곁 파일 없는 쓰기는 `v6_form`. 작성기의 빈 반환을 통과로 쓰지 않는다."""
    e = e2e
    for over in ({"inputs_contract_raw": None}, {"profile": None}, {"pin_raw": None}, {"repo": None}):
        with pytest.raises(MR.MainRegistrationRefused) as exc:
            _register(e, **over)
        assert exc.value.code == "v6_form", over
    reg, raw, tc_text = _register(e)
    from dataclasses import asdict

    same = json.loads(json.dumps({**e.scoring, "narrative": asdict(reg.narrative)}, default=list))
    assert _register(e, scoring=same)[1] == raw                               # 같은 블록은 받는다(덮어쓰기 아님)
    bad = json.loads(json.dumps({**e.scoring, "narrative": {**asdict(reg.narrative), "open_holes": list(NR.OPEN_HOLES)}},
                                default=list))
    with pytest.raises(MR.MainRegistrationRefused) as exc2:
        _register(e, scoring=bad)
    assert exc2.value.code == "scoring_conflict"
    with pytest.raises(MR.MainRegistrationRefused) as exc3:
        _register(e, scoring={**e.scoring, "narrative_extra": {}})
    assert exc3.value.code == "scoring_keys"
    with pytest.raises(MR.MainRegistrationRefused) as exc4:
        MR.write_main_registration(tmp_path / "six", stem=STEM, raw=raw, tc_text=tc_text, gen_raw=e.gen_raw, echo_raw=e.echo_raw)
    assert exc4.value.code == "v6_form" and not (tmp_path / "six").exists()
    with pytest.raises(MR.MainRegistrationRefused) as exc5:                  # 판 5 에 곁 파일 둘
        MR.write_main_registration(tmp_path / "five", stem=STEM, raw=_kw_v5_raw(e, reg), tc_text=tc_text, gen_raw=e.gen_raw,
                                   echo_raw=e.echo_raw, eval_raw=e.cand_raw, sources_raw=b"{}\n")
    assert exc5.value.code == "regime_version"


def _kw_v5_raw(e, reg) -> bytes:
    from evaluation.prereg_unified import REGISTRATION_VERSION, dump_registration

    return dump_registration(replace(reg, version=REGISTRATION_VERSION, narrative=None, clause_axis=None))


@pytest.mark.private_assets
def test_v6_link_five_cells_identity(e2e):
    """R3 — 한 태그 누락 · 같은 meta 재사용(원문 차이) · 다른 목적 · 다른 시드는 기존 `cells` · `adapter_identity` · `adapter_seed` ·
    `train_config_differs` 로 멈추고 여섯 파일은 0 이다."""
    e = e2e
    cases = [({k: v for k, v in e.metas.items() if k != "uni_fed"}, "cells"),
             ({**e.metas, "uni_central": {**e.metas["uni_central"], "train_config": e.tc + " "}}, "train_config_differs"),
             ({**e.metas, "uni_local_C2": {**e.metas["uni_local_C2"], "identity": {**e.metas["uni_local_C2"]["identity"],
                                                                                   "purpose": "a3"}}}, "adapter_identity"),
             ({**e.metas, "uni_local_C3": {**e.metas["uni_local_C3"], "identity": {**e.metas["uni_local_C3"]["identity"],
                                                                                   "seed_value": 8}}}, "adapter_seed")]
    for metas, code in cases:
        with pytest.raises(MR.MainRegistrationRefused) as exc:
            _register(e, metas=metas)
        assert exc.value.code == code, code
    assert not list((e.w.repo / "configs" / "registration").glob(f"{STEM}*"))


@pytest.mark.private_assets
def test_v6_link_meta_pin_bytes(e2e):
    """R16 — 한 메타의 `start_gate.pin_sha256` 이 없거나 다르면 `pin_bytes`; 넘긴 PIN 바이트가 PIN 객체와 다르면 `pin_bytes`. 발행은
    다섯 메타의 sha 가 모두 착수 고정 바이트의 sha 일 때만."""
    e = e2e
    missing = {**e.metas, "uni_fed": {k: v for k, v in e.metas["uni_fed"].items() if k != "start_gate"}}
    differs = {**e.metas, "uni_local_C1": {**e.metas["uni_local_C1"], "start_gate": {"pin_sha256": "0" * 64}}}
    for metas in (missing, differs):
        with pytest.raises(MR.MainRegistrationRefused) as exc:
            _register(e, metas=metas)
        assert exc.value.code == "pin_bytes"
    with pytest.raises(MR.MainRegistrationRefused) as exc2:
        _register(e, pin_raw=e.pin_raw + b"\n")
    assert exc2.value.code == "pin_bytes"


@pytest.mark.private_assets
def test_v6_link_write_once_and_receipt_order(e2e, tmp_path):
    """R20 — 여섯 가운데 하나가 미리 있으면 `name_taken`(덮어쓰기 0), 영수증이 든 커밋은 `receipt_same_commit`, 본줄기 밖 커밋은
    `receipt_not_ancestor`. 검사 실패 때 여섯 파일 · 영수증 새 생성 0."""
    e = e2e
    reg, raw, tc_text = _register(e)
    sources = NR.build_v6_sources(registration_raw=raw, eval_raw=e.cand_raw, repo=e.w.repo)
    d = tmp_path / "reg"
    d.mkdir()
    (d / f"{STEM}.v6_sources.json").write_bytes(b"old\n")
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        MR.write_main_registration(d, stem=STEM, raw=raw, tc_text=tc_text, gen_raw=e.gen_raw, echo_raw=e.echo_raw,
                                   eval_raw=e.cand_raw, sources_raw=sources)
    assert exc.value.code == "name_taken" and sorted(p.name for p in d.iterdir()) == [f"{STEM}.v6_sources.json"]
    assert (d / f"{STEM}.v6_sources.json").read_bytes() == b"old\n"
    with pytest.raises(MR.MainRegistrationRefused) as exc2:                  # 출처가 이 바이트를 묶지 않았다
        MR.write_main_registration(tmp_path / "reg2", stem=STEM, raw=raw, tc_text=tc_text, gen_raw=e.gen_raw,
                                   echo_raw=e.echo_raw, eval_raw=e.cand_raw + b"# x\n", sources_raw=sources)
    assert exc2.value.code == "v6_sources_mismatch" and not (tmp_path / "reg2").exists()
    rp = _commit(e, reg, raw, tc_text)
    head = W.git(e.w.repo, "rev-parse", "HEAD")
    W.git(e.w.repo, "add", "-A")
    W.git(e.w.repo, "commit", "-q", "-m", "영수증 커밋")
    with pytest.raises(MR.MainRegistrationRefused) as exc3:
        MR.write_main_receipt(registration_rel=REG_REL, commit="HEAD", checkout=e.w.repo, now=NOW)
    assert exc3.value.code == "receipt_same_commit"
    W.git(e.w.repo, "checkout", "-q", "-b", "side", head)
    (e.w.repo / "x.txt").write_text("x\n", encoding="utf-8")
    W.git(e.w.repo, "add", "x.txt")
    W.git(e.w.repo, "commit", "-q", "-m", "곁가지")
    side = W.git(e.w.repo, "rev-parse", "HEAD")
    W.git(e.w.repo, "checkout", "-q", "main")
    with pytest.raises(MR.MainRegistrationRefused) as exc4:
        MR.write_main_receipt(registration_rel=REG_REL, commit=side, checkout=e.w.repo, now=NOW)
    assert exc4.value.code == "receipt_not_ancestor"
    assert rp.is_file()
