"""착수 고정의 실물 측정 — 임시 git 저장소에서 `code_probe` · `common_probe` 를 대역 없이 한 번 돈다(지시 1013i 의 3-c)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import early_coord_check as D
from tests.test_main_uni import _cfg_yaml
from vlm import ecc as E
from vlm import pilot_vlm, registration
from vlm.pin import PinMismatch, build_pin, code_probe, common_probe, write_pin

PROMPT = "vlm/prompts/unified_v2_absorig.txt"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "vlm" / "prompts").mkdir(parents=True)
    shutil.copyfile(ROOT / PROMPT, r / PROMPT)
    (r / "vlm" / "a.py").write_bytes(b"X = 1\n")                        # LF 그대로(write_text 는 CRLF 로 바꾼다)
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "init")
    return r


def test_code_probe_는_커밋과_작업_트리의_바이트를_맞댄다(repo: Path):
    head = _git(repo, "rev-parse", "HEAD")
    assert code_probe(repo) == {"commit": head, "dirty": []}
    (repo / "vlm" / "a.py").write_bytes(b"X = 2\n")                          # 추적 파일이 커밋과 다르다
    (repo / "vlm" / "b.py").write_text("", encoding="utf-8")                 # 추적 밖 .py
    dirty = code_probe(repo)["dirty"]
    assert any("커밋과 다르다: vlm/a.py" in d for d in dirty) and any("추적 밖: vlm/b.py" in d for d in dirty)
    pin = build_pin(init_adapter_path=repo / "none.npz", repo=repo,
                    probes={"env": dict, "common": dict, "init_adapter": dict})
    with pytest.raises(PinMismatch):                                         # 더러운 코드로는 착수 고정을 쓰지 않는다
        write_pin(repo / "PIN.json", pin)


def test_common_probe_는_설정_프롬프트_등록_파일을_실물로_잰다(repo: Path, tmp_path: Path, monkeypatch):
    disc = tmp_path / "disc.json"
    disc.write_text("{}", encoding="utf-8")
    b = yaml.safe_load(D.RULES_PATH.read_bytes().decode("utf-8"))
    b.update({"sequential_rule": "single", "max_attempts": 1,
              "discrimination": {"path": str(disc), "sha256": hashlib.sha256(disc.read_bytes()).hexdigest()}})
    b["limits"] = {**b["limits"], "max_hours": 1, "max_cost_krw": 3000}
    rule = tmp_path / "early_coord_check.yaml"
    rule.write_bytes(yaml.safe_dump(b, allow_unicode=True, sort_keys=False).encode("utf-8"))
    monkeypatch.setattr(E, "RULE_PATH", rule)
    cfg = _cfg_yaml(tmp_path, uni_train_budget={"num_rounds": 1, "local_epochs": 1, "total_epochs": 1}, uni_max_new_tokens=512)
    reg = {"kind": registration.KIND, "version": 1, "seed_indices": [1], "num_rounds": 1, "local_epochs": 1, "total_epochs": 1,
           "recipe": "json_only", "eval_gen_limit": 512, "quantization": dict(pilot_vlm.QUANT_4BIT),
           "micro_batch": pilot_vlm.MICRO_BATCH, "grad_accum": pilot_vlm.GRAD_ACCUM, "lr": pilot_vlm.LR,
           "early_coord_check_rules": {"version": b["version"], "sha256": hashlib.sha256(rule.read_bytes()).hexdigest()},
           "init_adapter_digest": None}
    reg_path = tmp_path / "train_registration.yaml"
    reg_path.write_text(yaml.safe_dump(reg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    got = common_probe(cfg, repo=repo, registration_path=reg_path)
    prompt_text = (repo / PROMPT).read_bytes()
    assert got["prompt_sha256"] == hashlib.sha256(prompt_text).hexdigest()
    assert got["template_mode"] and len(got["coord_cfg_hash"]) == 64 and len(got["uni_block_sha256"]) == 64
    assert got["registration_sha256"] == hashlib.sha256(reg_path.read_bytes()).hexdigest()
    assert got["registration"]["eval_gen_limit"] == 512 and got["config_file_sha256"] == hashlib.sha256(cfg.read_bytes()).hexdigest()
    again = common_probe(cfg, repo=repo, registration_path=reg_path)
    assert json.dumps(again, sort_keys=True) == json.dumps(got, sort_keys=True)   # 같은 입력 → 같은 값
    (repo / PROMPT).write_bytes(prompt_text + b"\n")
    assert common_probe(cfg, repo=repo, registration_path=reg_path)["prompt_sha256"] != got["prompt_sha256"]
