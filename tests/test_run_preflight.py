"""실자료·GPU 없이 합성 저장소/압축본/승인/세션을 맞댄다. 실제 봉인 연결 시험은 별도다."""

from __future__ import annotations

import copy
import hashlib
import importlib.abc
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from data import frozen_guard
from evaluation import narrative_registration as NR
from vlm import (  # noqa: F401 — 합성 체크아웃으로 코드 출처를 대조할 모듈
    budget,
    init_adapter,
    narrative_inputs,
    probe_inputs,
    registration,
    token_table,
    uni_config,
)
from vlm import preflight_contract as S
from vlm import run_preflight as P

ROOT = Path(__file__).resolve().parents[1]
# 82abeee6의 대상 추적 .py는 403개. 자료 폴더 유입·코드 급증 시 복사 전에 멈춘다.
MAX_WORLD_SOURCE_FILES = 512


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8", newline="\n")
    return P.sha(path)


def git(root, *args):
    r = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args], capture_output=True, check=False)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    return r.stdout.decode("utf-8").strip()


def copy_tracked_sources(root, destination):
    # data/의 실물·정션을 걷지 않는다. 미추적 .py도 합성 코드에 섞지 않는다.
    tracked = set()
    for group in P.REPO_PACKAGES:
        tracked.update(rel for rel in git(root, "ls-files", "-z", "--", group).split("\0")
                       if rel.endswith(".py"))
    copy_limit = len(tracked)
    assert 0 < copy_limit <= MAX_WORLD_SOURCE_FILES, f"대역 코드 복사 상한 초과/빈 목록: {copy_limit}"
    for copied, rel in enumerate(sorted(tracked), 1):
        assert copied <= copy_limit, "추적 .py 수보다 많은 파일을 복사하려 한다"
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, dst)
    return copy_limit


class NoModels(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.calls = []

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in ("torch", "transformers"):
            self.calls.append(fullname)
            raise AssertionError(f"모델 모듈을 불렀다: {fullname}")


@pytest.fixture(autouse=True)
def no_model_calls():
    blocker = NoModels()
    sys.meta_path.insert(0, blocker)
    # 이미 다른 지정 시험이 적재한 경우에도 생성/학습 호출 진입을 잡는다.
    calls = []
    old_profile = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name in ("from_pretrained", "generate", "train_adapter", "train_local"):
            calls.append(frame.f_code.co_name)
            raise AssertionError("사전 점검에서 생성/학습 호출")

    sys.setprofile(profile)
    yield
    sys.setprofile(old_profile)
    sys.meta_path.remove(blocker)
    assert calls == [] and blocker.calls == []


class World:
    def ref(self, path):
        return {"path": path.relative_to(self.repo).as_posix(), "sha256": P.sha(path)}

    def save(self):
        put(self.contract, self.c)

    def inspect(self):
        self.save()
        return P.inspect(self.contract, **self.context)

    def archive(self, extra=None):
        with tarfile.open(self.inp / S.PATHS["archive"], "w") as tf:
            for row in self.members:
                tf.add(self.repo / row["path"], arcname=row["path"], recursive=False)
            if extra:
                info = tarfile.TarInfo(extra[0])
                if extra[1] == "symlink":
                    info.type, info.linkname = tarfile.SYMTYPE, "outside"
                    tf.addfile(info)
                elif extra[1] == "hardlink":
                    info.type, info.linkname = tarfile.LNKTYPE, self.members[0]["path"]
                    tf.addfile(info)
                else:
                    info.size = 1
                    tf.addfile(info, io.BytesIO(b"x"))
        self.c["inputs"]["tar_sha256"] = P.sha(self.inp / S.PATHS["archive"])


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World()
    w.repo, w.inp, w.cache = (tmp_path / p for p in ("repo", "input", "cache"))
    for folder in (w.repo, w.inp, w.cache):
        folder.mkdir()
    # 검사 구현은 정본 그대로. 자료·모델은 작은 합성 바이트뿐이다.
    copy_tracked_sources(ROOT, w.repo)
    cases = json.loads((ROOT / "evaluation/vectors/narrative_inputs_cases.v1.json").read_bytes())["world"]
    for rel, content in cases["files"].items():
        dst = w.repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content.encode())
    folder = "data/processed/synthetic_sealed"
    w.pairs = w.repo / folder
    w.pairs.mkdir(parents=True)
    rows = [{"image_id": "train-a", "client": "C1", "split": "train"},
            {"image_id": "val-a", "client": "C1", "split": "val"}]
    (w.pairs / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    limits = {"path": cases["limits_rel"], "sha256": P.sha(w.repo / cases["limits_rel"])}
    meta = {"text_contract": NR.TEXT_CONTRACT_V2, "limits_table": {**limits, "status": cases["limits_status"]},
            "identity": {"contract_version": NR.INPUTS_CONTRACT_VERSION, "limits": limits,
                         "limits_loader": NR.bundle_block(w.repo, NR.LIMITS_LOADER_FILES),
                         "generator": NR.bundle_block(w.repo, NR.GENERATOR_FILES_V1), "conditions": {"in_table": True}}}
    put(w.pairs / "PAIRS_META.json", meta)
    put(w.pairs / "counts.json", {})
    (w.pairs / "discarded.jsonl").write_bytes(b"")
    entries = [(P.sha(w.pairs / n), n) for n in ("pairs.jsonl", "discarded.jsonl", "counts.json", "PAIRS_META.json")]
    digest = hashlib.sha256("".join(h for h, _ in entries).encode()).hexdigest()
    (w.pairs / "SNAPSHOT.sha256").write_bytes(("".join(f"{h}  {n}\n" for h, n in entries) + f"\n# snapshot_digest {digest}\n").encode())
    monkeypatch.setitem(frozen_guard.EXPECTED_SEALED, folder, {"status": "expected"})
    nc = NR.narrative_inputs_block(w.repo, limits_rel=cases["limits_rel"], pairs_path=folder,
                                   snapshot_digest=digest, pairs_meta_raw=(w.pairs / "PAIRS_META.json").read_bytes())
    ncpath = w.repo / "configs/narrative_inputs.v1.json"
    ncpath.parent.mkdir(exist_ok=True)
    ncpath.write_bytes(NR.dump_narrative_inputs(nc))
    (w.repo / S.PATHS["registration"]).write_text(yaml.safe_dump({"narrative_inputs": {"version": 1, "sha256": P.sha(ncpath)}}), encoding="utf-8")
    render = {"model_id": "synthetic", "revision": "synthetic-revision", "processor_kwargs": {"min_pixels": 1, "max_pixels": 2},
              "chat_template_kwargs": {"enable_thinking": False}}
    (w.repo / S.PATHS["config"]).write_text(yaml.safe_dump({"fixed_before_main_runs": {
        "uni_processor_kwargs": render["processor_kwargs"], "uni_chat_template_kwargs": render["chat_template_kwargs"],
        "uni_model": {"id": render["model_id"], "revision": render["revision"]}}}), encoding="utf-8")
    (w.repo / ".gitignore").write_text("outputs/\ndata/processed/\ndata/interim/\ndata/tiles_v1_histmatch/\n", encoding="utf-8")
    versions = {n: "1.0" for n in S.PACKAGES}
    monkeypatch.setattr(P.importlib.metadata, "version", lambda name: versions[name])
    (w.repo / "uv.lock").write_text("".join(f'[[package]]\nname = "{n}"\nversion = "{v}"\n' for n, v in versions.items()), encoding="utf-8")
    (w.repo / "pyproject.toml").write_text('[project]\nname = "synthetic"\n', encoding="utf-8")
    model_files = []
    for name in ("tokenizer_config.json", "synthetic.weights"):
        p = w.cache / name
        p.write_bytes(b"synthetic-cache")
        model_files.append({"name": name, "bytes": p.stat().st_size, "sha256": P.sha(p)})
    manifest_model = w.repo / S.PATHS["model_manifest"]
    put(manifest_model, {"ignored_transfer_metadata": True, "model": {"id": render["model_id"], "revision": render["revision"], "files": model_files}})
    arrays = [np.array([1], dtype=np.float32), np.zeros(1, dtype=np.float32)]
    np.savez(w.inp / S.PATHS["adapter"], lora_A=arrays[0], lora_B=arrays[1])
    init_digest = hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()
    put(w.inp / S.PATHS["proof"], {"init_adapter_digest": init_digest})
    put(w.repo / S.PATHS["train_config"], {"init_adapter_digest": init_digest})
    (w.inp / S.PATHS["a2"]).write_bytes(b"train-a\n")
    recipe = "json_line_then_basis"
    prompt = w.repo / registration.REGIME_PROMPTS[recipe]
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_bytes((ROOT / registration.REGIME_PROMPTS[recipe]).read_bytes())
    vectors = w.repo / S.PATHS["vectors"]
    vectors.parent.mkdir(parents=True, exist_ok=True)
    vectors.write_bytes(b"[]\n")
    w.table = w.repo / "outputs/synthetic/per_row.csv"
    w.table.parent.mkdir(parents=True)
    w.table.write_bytes(b"image_id,client,supervised_tokens\ntrain-a,C1,3\n")
    target = P._literal(w.repo / "vlm/pilot_vlm.py", "TARGET_CONTRACTS")[recipe]
    target_sha = P.sha_bytes(json.dumps(target, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
    prov = {"kind": "per_row_supervised_tokens", "schema": 2, "recipe": recipe, "contract_sha256": target_sha,
            "pairs_digest": digest, "pairs": {"path": folder + "/pairs.jsonl", "sha256": P.sha(w.pairs / "pairs.jsonl")},
            "table_sha256": P.sha(w.table), "counter": {"module": "vlm.token_table", "sha256_lf": token_table.source_sha256()},
            "tokenizer": {"model_id": "synthetic", "revision": "synthetic-revision", "files": {"tokenizer_config.json": P.sha(w.cache / "tokenizer_config.json")},
                          "transformers": "1.0", "tokenizers": "1.0"}, "n_rows": 1}
    pp = w.table.with_name(w.table.name + budget.PROVENANCE_SUFFIX)
    put(pp, prov)
    monkeypatch.setitem(budget.TOKEN_TABLES, recipe, (w.table.relative_to(w.repo), P.sha(w.table)))
    w.measure = w.repo / "outputs/synthetic/measure.json"
    measure = {"inputs": {key: {"path": folder + "/" + name, "sha256": P.sha(w.pairs / name)}
                          for key, name in (("pairs", "pairs.jsonl"), ("pairs_meta", "PAIRS_META.json"), ("pairs_snapshot", "SNAPSHOT.sha256"))},
               "input_contract": {"pairs_digest": digest, "identity_failures": [], "snapshot_problems": [], "render_config": render,
                                  "render_check": {"encode_C_all_match": True, "template_B_all_match": True,
                                                   "prompt_sha256_text": P.sha_bytes(prompt.read_text(encoding="utf-8").encode())}},
               "checks": dict.fromkeys(S.MEASURE_CHECKS, True), "recommended_limit": {"value": 7}}
    measure["inputs"]["prompt"] = {"path": registration.REGIME_PROMPTS[recipe], "sha256": P.sha(prompt)}
    put(w.measure, measure)
    manifest = w.repo / S.PATHS["manifest"]
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"rel_path,split\ndata/tiles_v1_histmatch/synthetic.bin,train\n")
    tile = w.repo / "data/tiles_v1_histmatch/synthetic.bin"
    tile.parent.mkdir(parents=True)
    tile.write_bytes(b"synthetic image bytes, never decoded")
    w.members = [{"path": p.relative_to(w.repo).as_posix(), "bytes": p.stat().st_size, "sha256": P.sha(p)}
                 for p in [manifest, tile, *sorted(w.pairs.iterdir())]]
    put(w.inp / S.PATHS["members"], w.members)
    git(w.repo, "init", "-b", "run")
    git(w.repo, "config", "user.name", "synthetic")
    git(w.repo, "config", "user.email", "synthetic@example.invalid")
    git(w.repo, "config", "core.autocrlf", "false")
    git(w.repo, "add", ".")
    git(w.repo, "commit", "-m", "합성 실행 코드")
    base = git(w.repo, "rev-parse", "HEAD")
    w.c = {"sha": base, "bundle": {"name": "code.bundle", "ref": "run", "sha256": "0" * 64},
           "inputs": {"tar_sha256": "0" * 64, "members_sha256": P.sha(w.inp / S.PATHS["members"]), "count": len(w.members)},
           "pairs_v2": {"folder": folder, "digest": digest}, "init_adapter": {"npz_sha256": P.sha(w.inp / S.PATHS["adapter"]),
                       "proof_sha256": P.sha(w.inp / S.PATHS["proof"]), "digest": init_digest}, "a2_rows": P.sha(w.inp / S.PATHS["a2"]),
           "regime": {"recipe": recipe, "prompt_file": registration.REGIME_PROMPTS[recipe], "prompt_sha256": P.sha(prompt),
                      "max_new_tokens": S.MAX_NEW_TOKENS, "vectors_sha256": P.sha(vectors)},
           "c08": {"measure": w.ref(w.measure), "token_table": w.ref(w.table), "token_provenance": w.ref(pp), "narrative_inputs": w.ref(ncpath)},
           "model_manifest": w.ref(manifest_model), "a3": {"rows_sha256": "1" * 64, "threshold": .5, "unexposed_path": "다", "heldout_list": None},
           "caps_s": {"a2": 1, "a3_05": 2, "recover": 1}, "caps_cu": None, "gpu_expect": "synthetic",
           "env_expect": {"uv_lock_sha256": P.sha(w.repo / "uv.lock"), "pyproject_sha256": P.sha(w.repo / "pyproject.toml"),
                          "torch": "1.0", "transformers": "1.0", "bitsandbytes": "1.0", "gpu_name_has": "synthetic"}}
    w.contract = w.inp / "RUN_CONTRACT.json"
    w.environment = tmp_path / "ENV_RECEIPT.json"
    put(w.environment, {"session": "session-a", "cuda_ready": True, "gpu_name": "synthetic GPU", **w.c["env_expect"]})
    monkeypatch.setattr(P, "current_session", lambda: "session-a")
    w.context = {"inputs": str(w.inp), "checkout": str(w.repo), "model_cache": str(w.cache), "environment": str(w.environment)}
    w.archive()

    def approve():
        approval = {"original": "합성 승인, 실행하지 않는다.", "at": "2026-01-01T00:00:00+09:00",
                    "target": {k: copy.deepcopy(w.c[k]) for k in S.APPROVAL_TARGET}}
        ash = put(w.inp / "APPROVAL.json", approval)
        heading = "## 합성 승인"
        section = heading + "\n```json\n" + json.dumps({"record_sha256": ash, "approval": approval}, ensure_ascii=False) + "\n```\n"
        log = w.repo / "approval.md"
        log.write_text(section + "## 다음 절\n범위 밖\n", encoding="utf-8", newline="\n")
        git(w.repo, "checkout", "-B", "approval", base)
        put(w.repo / "approvals/APPROVAL.json", approval)
        git(w.repo, "add", "approval.md", "approvals/APPROVAL.json")
        git(w.repo, "commit", "-m", "합성 승인 기록")
        logcommit = git(w.repo, "rev-parse", "HEAD")
        git(w.repo, "checkout", "run")
        w.c["approved_by"] = {"record": {"path": "APPROVAL.json", "sha256": ash},
                              "decision_log": {"commit": logcommit, "path": "approval.md", "section": heading,
                                               "section_sha256": P.sha_bytes(section.encode())}}
        bundle = w.inp / "code.bundle"
        bundle.unlink(missing_ok=True)
        approval_ref = "approval-" + logcommit[:8]
        git(w.repo, "update-ref", "refs/heads/" + approval_ref, logcommit)
        git(w.repo, "bundle", "create", str(bundle), "run", approval_ref)
        heads = git(w.repo, "bundle", "list-heads", str(bundle)).splitlines()
        assert set(heads) == {base + " refs/heads/run", logcommit + " refs/heads/" + approval_ref}
        w.c["bundle"]["sha256"] = P.sha(bundle)
        w.save()

    w.approve = approve
    approve()
    # 실제 절차처럼 실행 ref와 승인 ref 둘만 담은 꾸러미를 복제한다.
    source, checkout = w.repo, tmp_path / "checkout"
    git(source, "clone", "-c", "core.autocrlf=false", "--branch", "run", str(w.inp / "code.bundle"), str(checkout))
    git(checkout, "config", "user.name", "synthetic")
    git(checkout, "config", "user.email", "synthetic@example.invalid")
    git(checkout, "config", "core.autocrlf", "false")
    for rel in ("data/processed", "data/interim", "data/tiles_v1_histmatch", "outputs"):
        shutil.copytree(source / rel, checkout / rel, dirs_exist_ok=True)
    for attr in ("pairs", "table", "measure"):
        setattr(w, attr, checkout / getattr(w, attr).relative_to(source))
    w.repo = checkout
    w.context["checkout"] = str(checkout)
    assert git(checkout, "rev-parse", "HEAD") == base
    approval_commit = w.c["approved_by"]["decision_log"]["commit"]
    assert git(checkout, "show", approval_commit + ":approval.md")
    assert json.loads(git(checkout, "show", approval_commit + ":approvals/APPROVAL.json")) == P.read(w.inp / "APPROVAL.json")
    # 복사한 코드는 원본과, 명시적으로 덮어쓴 코드는 합성 명부와 맞댄다.
    # 앞 시험의 import 여부와 무관하게 모든 적재 모듈의 경로는 합성 사본으로 돌린다.
    monkeypatch.setattr(P, "ROOT", checkout)
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.split(".")[0] in P.REPO_PACKAGES and path and Path(path).resolve().is_relative_to(ROOT):
            rel = Path(path).resolve().relative_to(ROOT).as_posix()
            synthetic = checkout / rel
            expected = cases["files"][rel].encode() if rel in cases["files"] else Path(path).read_bytes()
            assert synthetic.read_bytes() == expected, rel
            monkeypatch.setattr(module, "__file__", str(synthetic))
    return w


@pytest.mark.parametrize("branch", ["나", "다"])
def test_complete_synthetic_bundle(world, branch):
    w = world
    if branch == "나":
        tune = w.inp / S.PATHS["heldout"]
        tune.write_bytes(b"val-a\n")
        w.c["a3"].update(unexposed_path=branch, heldout_list=P.sha(tune))
        w.approve()
    r = w.inspect()
    assert r["all_ok"], r["checks"]
    assert set(r) == S.RECEIPT_KEYS
    path = w.inp / "PREFLIGHT.json"
    P.write_receipt(path, r)
    assert P.verify(path, w.contract) == r


@pytest.mark.parametrize("case,check", [
    ("bytes", "inputs"), ("model_rows", "tables"), ("empty", "nonempty"), ("env", "environment"),
    ("approval", "approval"), ("heldout", "heldout"), ("init", "init"), ("prompt", "prompt"), ("seal", "pairs"),
])
def test_nine_independent_refusals(world, case, check, monkeypatch):
    w = world
    if case == "bytes":
        (w.inp / "code.bundle").write_bytes(b"changed")
    elif case == "model_rows":
        p = w.repo / w.c["model_manifest"]["path"]
        r = P.read(p)
        r["model"]["files"].append(copy.deepcopy(r["model"]["files"][0]))
        w.c["model_manifest"]["sha256"] = put(p, r)
    elif case == "empty":
        (w.inp / S.PATHS["a2"]).write_bytes(b"")
        w.c["a2_rows"] = P.sha(w.inp / S.PATHS["a2"])
    elif case == "env":
        r = P.read(w.environment)
        r["cuda_ready"] = False
        put(w.environment, r)
    elif case == "approval":
        r = P.read(w.inp / "APPROVAL.json")
        r["target"]["sha"] = "a" * 40
        w.c["approved_by"]["record"]["sha256"] = put(w.inp / "APPROVAL.json", r)
    elif case == "heldout":
        w.c["a3"]["unexposed_path"] = "나"
    elif case == "init":
        np.savez(w.inp / S.PATHS["adapter"], lora_A=np.ones(1), lora_B=np.ones(1))
        w.c["init_adapter"]["npz_sha256"] = P.sha(w.inp / S.PATHS["adapter"])
    elif case == "prompt":
        r = P.read(w.measure)
        r["input_contract"]["render_config"]["chat_template_kwargs"]["enable_thinking"] = True
        w.c["c08"]["measure"]["sha256"] = put(w.measure, r)
    else:
        monkeypatch.setitem(frozen_guard.EXPECTED_SEALED, w.c["pairs_v2"]["folder"], {"status": "lost"})
    r = w.inspect()
    assert not r["all_ok"] and not r["checks"][check]["ok"]
    assert len(r["checks"]) == 9
    assert all(code in set(S.Code) for result in r["checks"].values() for code in result["codes"])
    # 앞 항목 거부가 뒤 항목의 실행을 삼키지 않는다.
    if case == "bytes":
        assert r["checks"]["pairs"]["ok"]


@pytest.mark.parametrize("path,kind", [("../outside", "file"), ("/absolute", "file"), ("C:/absolute", "file"),
                                      ("link", "symlink"), ("hard", "hardlink"), ("data/interim/manifest_v1/manifest.csv", "file")])
def test_archive_paths_and_duplicate_refused(world, path, kind):
    world.archive((path, kind))
    r = world.inspect()
    assert not r["all_ok"] and P.Code.PATH_UNSAFE in r["checks"]["inputs"]["codes"]


@pytest.mark.parametrize("case", ["code", "hidden_code", "wrong_ref", "model_config", "temp_digest", "fake_pass", "missing_check", "failure", "other_limit", "extra_file", "test_split"])
def test_additional_bindings(world, case):
    w = world
    if case in ("code", "hidden_code"):
        if case == "hidden_code":
            git(w.repo, "update-index", "--assume-unchanged", "vlm/probe_inputs.py")
        (w.repo / "vlm/probe_inputs.py").write_text("# changed\n")
    elif case == "wrong_ref":
        w.c["bundle"]["ref"] = "not-the-run"
    elif case == "model_config":
        path = w.repo / S.PATHS["config"]
        r = yaml.safe_load(path.read_bytes())
        r["fixed_before_main_runs"]["uni_model"]["revision"] = "other-revision"
        path.write_text(yaml.safe_dump(r), encoding="utf-8")
    elif case == "temp_digest":
        w.c["pairs_v2"]["digest"] = "e" * 64
    elif case == "extra_file":
        (w.pairs / "extra.json").write_bytes(b"{}")
    elif case == "test_split":
        p = w.repo / S.PATHS["manifest"]
        p.write_bytes(p.read_bytes().replace(b",train", b",test"))
        for row in w.members:
            if row["path"] == S.PATHS["manifest"]:
                row.update(sha256=P.sha(p), bytes=p.stat().st_size)
        w.c["inputs"]["members_sha256"] = put(w.inp / S.PATHS["members"], w.members)
        w.archive()
    else:
        r = P.read(w.measure)
        if case == "fake_pass":
            r["checks"] = {"limit_ok": True}
        elif case == "missing_check":
            r["checks"].pop("rows_equal_pairs_meta")
        elif case == "failure":
            r["input_contract"]["identity_failures"] = ["synthetic failure"]
        else:
            r["recommended_limit"]["value"] = S.MAX_NEW_TOKENS + 1
        w.c["c08"]["measure"]["sha256"] = put(w.measure, r)
    result = w.inspect()
    assert not result["all_ok"]
    if case == "model_config":
        assert not result["checks"]["prompt"]["ok"]


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "bool", "nan", "negative", "too_large", "absolute", "nonobject"])
def test_contract_closed_schema_and_threshold(world, case):
    w = world
    if case == "missing":
        del w.c["regime"]["recipe"]
    elif case == "extra":
        w.c["regime"]["limit_ok"] = True
    elif case in ("bool", "nan", "negative", "too_large"):
        w.c["a3"]["threshold"] = {"bool": True, "nan": float("nan"), "negative": -.01, "too_large": 1.01}[case]
    elif case == "absolute":
        w.c["c08"]["measure"]["path"] = "C:/outside.json"
    w.save()
    if case == "duplicate":
        raw = w.contract.read_text(encoding="utf-8")
        w.contract.write_text(raw.replace('"threshold": 0.5', '"threshold": 0.5, "threshold": 0.5'), encoding="utf-8")
    elif case == "nonobject":
        w.contract.write_bytes(b"[]")
    r = P.inspect(w.contract, **w.context)
    assert not r["all_ok"] and len(r["checks"]) == 9


@pytest.mark.parametrize("case", ["extra", "missing", "check_extra", "check_missing", "input_extra", "input_missing", "version", "bytes", "session", "code", "environment"])
def test_verify_rejects_stale_or_noncanonical_receipts(world, case, monkeypatch):
    w = world
    r = w.inspect()
    assert r["all_ok"], r["checks"]
    if case == "extra":
        r["extra"] = True
    elif case == "missing":
        del r["at"]
    elif case == "check_extra":
        r["checks"]["inputs"]["extra"] = True
    elif case == "check_missing":
        del r["checks"]["pairs"]
    elif case == "input_extra":
        r["inputs"]["extra"] = {"contract": "a" * 64, "measured": "a" * 64}
    elif case == "input_missing":
        del r["inputs"]["bundle"]
    elif case == "version":
        r["checker_version"] = "old"
    elif case == "bytes":
        (w.cache / "synthetic.weights").write_bytes(b"changed")
    elif case == "session":
        monkeypatch.setattr(P, "current_session", lambda: "session-b")
    elif case == "code":
        (w.repo / "vlm/probe_inputs.py").write_bytes(b"# changed")
    else:
        put(w.environment, {**P.read(w.environment), "session": "session-b"})
    p = w.inp / "PREFLIGHT.json"
    P.write_receipt(p, r)
    with pytest.raises(P.Rejected):
        P.verify(p, w.contract)


def test_write_failure_existing_output_and_cli_failure(world, monkeypatch):
    w = world
    r = w.inspect()
    p = w.inp / "PREFLIGHT.json"
    p.write_bytes(b"keep")
    with pytest.raises(P.Rejected):
        P.write_receipt(p, r)
    assert p.read_bytes() == b"keep"
    monkeypatch.setattr(P.os, "link", lambda *a: (_ for _ in ()).throw(OSError("synthetic write failure")))
    args = ["--contract", str(w.contract), "--inputs", str(w.inp), "--checkout", str(w.repo),
            "--model-cache", str(w.cache), "--environment", str(w.environment), "--out", str(w.inp / "new.json")]
    assert P.main(args) != 0
    assert not (w.inp / "new.json").exists() and not list(w.inp.glob(".preflight-*"))


def test_import_never_loads_model_libraries():
    code = "import sys; import vlm.run_preflight; from vlm.init_adapter import init_adapter_digest; import numpy as np; init_adapter_digest([np.zeros(1)]); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")


def test_md_section_includes_children_excludes_next_same_level():
    raw = "# 문서\r\n## 승인\r\n### 자식\r\n내용\r\n## 다음\r\n제외\r\n".encode()
    assert P._section(raw, "## 승인") == "## 승인\n### 자식\n내용\n"


@pytest.mark.parametrize("threshold", [0, 1])
def test_approved_threshold_closed_interval(world, threshold):
    world.c["a3"]["threshold"] = threshold
    world.approve()
    r = world.inspect()
    assert r["all_ok"], r["checks"]


def test_atomic_publish_race_does_not_overwrite(tmp_path, monkeypatch):
    dest = tmp_path / "PREFLIGHT.json"
    original = P.os.link

    def competitor(source, target):
        Path(target).write_bytes(b"competitor")
        return original(source, target)

    monkeypatch.setattr(P.os, "link", competitor)
    with pytest.raises(FileExistsError):
        P.write_receipt(dest, {"synthetic": True})
    assert dest.read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".preflight-*"))


def test_wire_names_change_in_schema_only(world, monkeypatch):
    w = world
    original = P.expected_inputs(w.c)
    monkeypatch.setitem(S.WIRE_NAMES, ("c08", "measure"), "sealed_measure")
    monkeypatch.setitem(S.WIRE_NAMES, ("model_manifest",), "cache_manifest")
    w.c["c08"]["sealed_measure"] = w.c["c08"].pop("measure")
    w.c["cache_manifest"] = w.c.pop("model_manifest")
    assert P.expected_inputs(w.c) == original
    r = w.inspect()
    assert r["all_ok"], r["checks"]
    p = w.inp / "PREFLIGHT.json"
    P.write_receipt(p, r)
    assert P.verify(p, w.contract) == r


def test_public_cell_api_and_canonical_contract_names(world):
    assert {"contract_view", "read", "inside", "validate", "sha"} <= set(P.__all__)
    assert all(callable(getattr(P, name)) for name in P.__all__)
    assert S.WIRE_NAMES == {}
    assert set(world.c["c08"]) == {"measure", "token_table", "token_provenance", "narrative_inputs"}
    assert "narrative_inputs" in P.expected_inputs(world.c)
    assert "inputs_contract" not in P.expected_inputs(world.c)
    world.c["c08"]["inputs_contract"] = world.c["c08"].pop("narrative_inputs")
    result = world.inspect()
    assert not result["all_ok"] and P.Code.CONTRACT_FORM in result["checks"]["tables"]["codes"]


@pytest.mark.parametrize("answer", [7, S.MAX_NEW_TOKENS])
def test_recommended_answer_at_or_below_fixed_limit(world, answer):
    measure = P.read(world.measure)
    measure["recommended_limit"]["value"] = answer
    world.c["c08"]["measure"]["sha256"] = put(world.measure, measure)
    result = world.inspect()
    assert result["all_ok"], result["checks"]


@pytest.mark.parametrize("limit", [S.MAX_NEW_TOKENS - 1, S.MAX_NEW_TOKENS + 1])
def test_generation_limit_must_equal_approved_value(world, limit):
    world.c["regime"]["max_new_tokens"] = limit
    world.approve()
    result = world.inspect()
    assert not result["all_ok"] and P.Code.C08_BINDING in result["checks"]["pairs"]["codes"]


def test_checker_loaded_from_other_root_is_rejected(world, monkeypatch):
    monkeypatch.setattr(P, "ROOT", ROOT)
    result = world.inspect()
    assert not result["all_ok"] and P.Code.CODE_MISMATCH in result["checks"]["inputs"]["codes"]


@pytest.mark.parametrize("package", ["vlm", "evaluation", "data", "scripts", "fl", "rag", "tracking", "corpus", "detection"])
def test_loaded_repository_packages_cannot_come_from_other_root(world, monkeypatch, package):
    monkeypatch.setitem(sys.modules, package + ".foreign_synthetic", SimpleNamespace(__file__=str(ROOT / package / "foreign_synthetic.py")))
    result = world.inspect()
    assert not result["all_ok"] and P.Code.CODE_MISMATCH in result["checks"]["inputs"]["codes"]


def test_init_digest_uses_canonical_function_on_synthetic_npz(world, monkeypatch):
    with np.load(world.inp / S.PATHS["adapter"], allow_pickle=False) as archive:
        arrays = [archive[key] for key in archive.files]
    expected = hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()
    canonical = init_adapter.init_adapter_digest
    assert canonical(arrays) == expected == world.c["init_adapter"]["digest"]
    calls = []

    def record(values):
        calls.append(canonical(values))
        return calls[-1]

    monkeypatch.setattr(init_adapter, "init_adapter_digest", record)
    result = world.inspect()
    assert result["all_ok"], result["checks"]
    assert calls == [expected]


@pytest.fixture
def source_repo(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-b", "source")
    files = {"vlm/module.py": b"before", "data/helpers/a b.py": b"helper"}
    for rel, content in {**files, "data/readme.txt": b"tracked non-python",
                         "data/raw/untracked.py": b"must not be copied"}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    git(root, "add", "--", *files, "data/readme.txt")
    # 경로는 인덱스에서, 바이트는 검토 중인 워크트리에서 가져온다.
    (root / "vlm/module.py").write_bytes(b"working tree bytes")
    return root


def test_world_copies_only_tracked_python_without_walking(source_repo, tmp_path, monkeypatch):
    destination = tmp_path / "copy"

    def no_walk(*args, **kwargs):
        raise AssertionError("소스 저장소를 파일 시스템으로 걷지 않는다")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "rglob", no_walk)
        guard.setattr(Path, "iterdir", no_walk)
        guard.setattr(P.os, "walk", no_walk)
        assert copy_tracked_sources(source_repo, destination) == 2
    assert (destination / "vlm/module.py").read_bytes() == b"working tree bytes"
    assert (destination / "data/helpers/a b.py").read_bytes() == b"helper"
    assert not (destination / "data/readme.txt").exists()
    assert not (destination / "data/raw").exists()


def test_world_source_limit_rejects_before_any_copy(source_repo, tmp_path, monkeypatch):
    destination = tmp_path / "not_created"
    monkeypatch.setattr(sys.modules[__name__], "MAX_WORLD_SOURCE_FILES", 1)

    def no_copy(*args, **kwargs):
        raise AssertionError("상한 초과를 복사 전에 거부해야 한다")

    monkeypatch.setattr(shutil, "copyfile", no_copy)
    with pytest.raises(AssertionError, match="대역 코드 복사 상한"):
        copy_tracked_sources(source_repo, destination)
    assert not destination.exists()



def test_world_with_preloaded_overwritten_module(request):
    # world를 요청하기 전에 실제 모듈을 적재해 전량 수집 순서의 조건을 만든다.
    module = importlib.import_module("corpus.rules.clause_text")
    rel = "corpus/rules/clause_text.py"
    original = Path(module.__file__).read_bytes()
    w = request.getfixturevalue("world")
    synthetic = w.repo / rel
    assert Path(module.__file__) == synthetic
    assert synthetic.read_bytes() != original
    # 검사기의 적재 코드 대조를 생략하지 않고 합성 체크아웃으로 통과해야 한다.
    result = w.inspect()
    assert result["all_ok"], result["checks"]
    assert result["checks"]["inputs"]["ok"]
