"""2차 후보는 결정값을 계승하되 체제·페어만 새 입력과 묶는다. 발행이나 실측이 아니다."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from evaluation import eval_registration as ER
from evaluation.prereg_unified import REGISTRATION_VERSION_V6
from scripts import prepare_eval_regime2 as C

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    source = ER.load_eval_registration((ROOT / C.SOURCE_REL).read_bytes())
    base = ER.load_base((ROOT / "configs/base.yaml").read_bytes())
    train = yaml.safe_load((ROOT / "configs/train_registration.yaml").read_bytes())
    prompt = (ROOT / base["fixed_before_main_runs"]["uni_prompt_path"]).read_bytes()
    return {"source": source, "base": base, "train": train, "prompt_raw": prompt}


def test_2차_후보가_현재_base와_맞고_1차_결정값은_그대로다():
    args = inputs()
    source = copy.deepcopy(args["source"])
    got = C.build_candidate(**args)
    fixed = args["base"]["fixed_before_main_runs"]
    assert C.candidate_bytes(got) == (ROOT / C.CANDIDATE_REL).read_bytes()
    assert ER.registration_problems(got, base_doc=args["base"]) == []
    assert C.candidate_problems(got, **args) == []
    assert args["source"] == source
    assert {k: v for k, v in got.items() if k not in ("generation", "auxiliary")} == {
        k: v for k, v in source.items() if k not in ("generation", "auxiliary")
    }
    assert got["auxiliary"] == [*source["auxiliary"], *ER.AUXILIARY_V6]        # 2차 보조 다섯 이름은 기존 뒤에 그 순서로
    assert got["generation"] == {
        "max_new_tokens": args["train"]["eval_gen_limit"],
        "recipe": args["train"]["recipe"],
        "prompt_file": fixed["uni_prompt_path"],
        "prompt_sha256": hashlib.sha256(args["prompt_raw"]).hexdigest(),
        "pairs_digest": fixed["uni_pairs"]["digest"],
        "registration_version": REGISTRATION_VERSION_V6,
    }
    # 모집단의 동결 분할과 생성 모델의 페어 스냅샷은 다른 식별자다. 전자를 덮지 않는다.
    assert got["population"]["snapshot_digest"] == source["population"]["snapshot_digest"]
    assert got["population"]["snapshot_digest"] != got["generation"]["pairs_digest"]


@pytest.mark.parametrize("field,value", [
    ("generation.recipe", "json_only"),
    ("generation.max_new_tokens", 512),
    ("generation.prompt_file", "vlm/prompts/unified_v2_absorig.txt"),
    ("generation.prompt_sha256", "0" * 64),
    ("generation.pairs_digest", "0" * 64),
    ("generation.registration_version", "5"),
    ("representative.metric_id", "other"),
    ("population.sample.main_size", 2000),
    ("population.sample.sealed.1000.file_sha256", "0" * 64),
    ("seeds", [1, 2]),
])
def test_허용한_후보_값과_계승한_결정이_바뀌면_거부한다(field, value):
    args = inputs()
    got = C.build_candidate(**args)
    keys = field.split(".")
    cur = got
    for key in keys[:-1]:
        cur = cur[key]
    cur[keys[-1]] = value
    assert any(field in p for p in C.candidate_problems(got, **args))


@pytest.mark.parametrize("edit", ["recipe", "prompt", "limit", "pairs", "population"])
def test_입력_체제가_갈리거나_범위_밖_결정이_필요하면_후보를_만들지_않는다(edit):
    args = inputs()
    if edit == "recipe":
        args["train"]["recipe"] = "json_only"
    elif edit == "prompt":
        args["prompt_raw"] += b"changed"
    elif edit == "limit":
        args["train"]["eval_gen_limit"] += 1
    elif edit == "pairs":
        args["base"]["fixed_before_main_runs"]["uni_pairs"]["digest"] = None
    else:
        args["base"]["fixed_before_main_runs"]["snapshot_digest"] = "0" * 64
    with pytest.raises(ValueError):
        C.build_candidate(**args)


def test_명령은_후보만_한번_쓰고_기존_파일은_덮어쓰지_않는다(tmp_path, monkeypatch, capsys):
    # 작은 합성 뿌리에 필요한 입력만 복사한다. 데이터 폴더를 걷지 않는다.
    args = inputs()
    names = [C.SOURCE_REL, "configs/base.yaml", "configs/train_registration.yaml",
             args["base"]["fixed_before_main_runs"]["uni_prompt_path"]]
    before = {}
    for name in names:
        raw = (ROOT / name).read_bytes()
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        before[name] = raw
    monkeypatch.setattr(C, "ROOT", tmp_path)
    assert C.main(["--write"]) == 0
    assert C.main([]) == 0
    candidate = tmp_path / C.CANDIDATE_REL
    saved = candidate.read_bytes()
    assert C.main(["--write"]) == 1
    assert candidate.read_bytes() == saved
    assert all((tmp_path / name).read_bytes() == raw for name, raw in before.items())
    assert not list((tmp_path / "configs/registration").glob("*.receipt.json"))
    assert '"issued": false' in capsys.readouterr().out


# ---------------------------------------------------------------- 판 6 연결 2 판 — 체제 섞임(§2-3 · R5) · 후보 보존

CANDIDATE_V1_SHA256 = "1e5c6aabfc870f0261ae1b7614307a1a1e95dec36f8ae4729e2421cd90b4340c"
"""앞 판 후보(2026-10-27, 보조 다섯 없음)의 바이트 — 그대로 보존한다."""


def _codes(problems: list[str]) -> set[str]:
    assert all(": " in p for p in problems), problems
    return {p.split(":", 1)[0] for p in problems}


def test_v6_link_eval_regime_mixing():
    """R5 — 새 체제 대조에 1차 설정을, 보조 이름의 치환 · 누락 · 중복 · 순서 · 앞뒤 바꿈을, 시드 · 표본의 변경을 들이밀면 각각
    `eval_binding` 또는 `aux_names` 다. M1 규약을 맞춰서 구제하지 않는다."""
    args = inputs()
    got = C.build_candidate(**args)
    assert _codes(C.candidate_problems(args["source"], **args)) == {"eval_binding", "aux_names"}
    for aux in ([*got["auxiliary"][:-1], "ungrounded_rate_x"], got["auxiliary"][:-1], [*got["auxiliary"], "ungrounded_rate"],
                [*got["auxiliary"][:6], *reversed(got["auxiliary"][6:])], [*ER.AUXILIARY_V6, *got["auxiliary"][:6]],
                got["auxiliary"][6:]):
        bad = copy.deepcopy(got)
        bad["auxiliary"] = aux
        assert _codes(C.candidate_problems(bad, **args)) == {"aux_names"}, aux
    for field, value in (("seeds", [1, 2]), ("population.sample.main_size", 2000),
                         ("population.sample.sealed.1000.set_sha256", "0" * 64), ("generation.max_new_tokens", 512),
                         ("generation.recipe", "json_only"), ("representative.metric_id", "m1_other")):
        bad = copy.deepcopy(got)
        cur = bad
        keys = field.split(".")
        for key in keys[:-1]:
            cur = cur[key]
        cur[keys[-1]] = value
        assert _codes(C.candidate_problems(bad, **args)) == {"eval_binding"}, field
    src2 = copy.deepcopy(args["source"])
    src2["auxiliary"].append("ungrounded_rate")
    with pytest.raises(ValueError, match="aux_names"):
        C.build_candidate(src2, args["base"], args["train"], args["prompt_raw"])
    assert ER.AUXILIARY_V6 == ("narrative_forbidden_rate", "narrative_reservation_rate", "citation_match_rate",
                               "citation_precision", "ungrounded_rate")


def test_v6_link_candidate_v1_preserved_and_new_candidate_adds_names():
    assert hashlib.sha256((ROOT / C.CANDIDATE_REL_V1).read_bytes()).hexdigest() == CANDIDATE_V1_SHA256
    assert C.CANDIDATE_REL != C.CANDIDATE_REL_V1 and C.CANDIDATE_REL.endswith("/main-20261028-1.eval.yaml")
    new = ER.load_eval_registration((ROOT / C.CANDIDATE_REL).read_bytes())
    old = ER.load_eval_registration((ROOT / C.CANDIDATE_REL_V1).read_bytes())
    assert new["auxiliary"] == [*old["auxiliary"], *ER.AUXILIARY_V6]
    assert {k: v for k, v in new.items() if k != "auxiliary"} == {k: v for k, v in old.items() if k != "auxiliary"}
    assert ER.registration_problems(new, base_doc=inputs()["base"]) == []
