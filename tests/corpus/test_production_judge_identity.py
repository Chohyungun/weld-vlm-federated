"""Tiny CPU-only separate judge contracts. Fake results are not approvals."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from corpus.production import infer as I
from corpus.production import judge_identity as J
from corpus.production import judge_policy as P
from corpus.production import judge_probe as K
from corpus.production import judge_fingerprints as F
from corpus.production import quality as Q


def spec(family="qwen"):
    return {"model_id": f"local/{family}", "revision": "a" * 40, "family": family,
            "local_files_only": True, "quantization": "bf16", "batch_size": 2,
            "max_new_tokens": 64, "max_input_tokens": 4096, "seed": 1,
            "wall_seconds": 1800, "enable_thinking": False}


def make_run(root, n=5, *, source_qa=False):
    root.mkdir()
    cfg = {"generation": spec(), "judge": spec("llama"), "shard_size": 2}
    tasks = [{"id": f"id-{i}", "kind": "image_report", "split": "train", "client": "C1",
              "lineage_group_id": f"lineage-{i}", "image_path": f"not-read/{i}.png",
              "source_ids": ["annotation"], "evidence": {"annotations": [],
                  "defect_counts_by_name": {}, "defect_positions": [], "size_mm_available": False},
              "locked_target": {"defects": []}, "prompt": "주석을 설명하라."} for i in range(n)]
    if source_qa:
        tasks[0].update(kind="source_qa", client="public", image_path=None,
                        locked_target=None, evidence={"passage": "허용치는 2 mm이다."})
    raw = b"".join(I._canonical(t) + b"\n" for t in tasks)
    (root / "tasks.jsonl").write_bytes(raw)
    run = {"schema_version": 1, "identity_sha256": "b" * 64,
           "config_sha256": I._digest(cfg), "tasks_sha256": hashlib.sha256(raw).hexdigest(),
           "config": cfg, "counts": {"total": n}}
    (root / "RUN.json").write_bytes(I._canonical(run))
    return cfg


class Fake:
    made = []
    calls = []
    closed = []
    failure_at = None
    failure = "oom"

    def __init__(self, cfg, stage):
        self.identity = {"backend": "fake", "model_id": cfg["model_id"],
                         "revision": cfg["revision"], "family": cfg["family"],
                         "model_type": cfg["family"], "quantization": cfg["quantization"],
                         "tokenizer_sha256": "c" * 64}
        self.stage = stage
        Fake.made.append(stage)

    def generate_batch(self, prompts, system, deadline):
        Fake.calls.append((self.stage, prompts, system))
        if len(Fake.calls) == Fake.failure_at:
            if Fake.failure == "time":
                raise I.StageTimeLimit()
            raise RuntimeError("CUDA out of memory")
        text = ("표시된 결함이 없다." if self.stage == "generate" else
                '{"grounded":true,"scope_ok":true,"language_ok":true,"reason":"근거와 같다."}')
        return [{"text": text, "finish_reason": "eos", "input_tokens": 20,
                 "output_tokens": 15} for _ in prompts]

    def close(self):
        Fake.closed.append(self.stage)


def reset_fake():
    Fake.made, Fake.calls, Fake.closed = [], [], []
    Fake.failure_at, Fake.failure = None, "oom"


@pytest.fixture(autouse=True)
def clear_fake(monkeypatch):
    reset_fake()
    # Synthetic v6 wiring for writer plumbing only. It is not the B parser,
    # a policy test, or real probe/production approval evidence.
    identity = {"selected_schema": "judge_structured_v6",
                "capabilities": {"judge_structured_v6": True},
                "closure_sha256": "7" * 64, "test_only": True}
    base = {"output_schema": "judge_structured_v6", "structured_parser_ready": True,
            "parser_implementation": identity, "test_only": True}
    monkeypatch.setattr(P, "OUTPUT_SCHEMA_VERSION", "judge_structured_v6", raising=False)
    monkeypatch.setattr(Q, "JUDGE_V6_SCHEMA_VERSION", "judge_structured_v6", raising=False)
    monkeypatch.setattr(Q, "parse_judge_v6", lambda task, row: {}, raising=False)
    monkeypatch.setattr(F, "require_v6_parser", lambda: identity, raising=False)
    monkeypatch.setattr(P, "policy_identity", lambda: dict(base))


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "run"
    cfg = make_run(root)
    I.run_stage(root, "generate", cfg, backend_factory=Fake)
    reset_fake()
    return root, cfg, tmp_path / "new-judge"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite_shard(path, mutate):
    shard = read(path)
    mutate(shard)
    shard["payload_sha256"] = I._digest({k: v for k, v in shard.items() if k != "payload_sha256"})
    path.write_bytes(I._canonical(shard) + b"\n")


def immutable_bytes(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_partial_resume_source_contract_and_legacy_bytes(prepared):
    root, cfg, target = prepared
    I.run_stage(root, "judge", cfg, backend_factory=Fake)
    original = immutable_bytes(root)
    reset_fake()
    first = I.run_judge(root, cfg, target, max_shards=1, backend_factory=Fake)
    assert first["exit_code"] == 4 and first["completed_tasks"] == 2
    source = I.read_judge_source(root, cfg, target)
    assert len(source["rows"]) == 2  # No fallback to the five legacy rows.
    assert source["judge_dir"] == str(target)
    basis = source["identity_basis"]
    assert source["identity_sha256"] == I._digest(basis)
    assert basis["run_identity_sha256"] == "b" * 64
    assert basis["judge_model"] == {"model_id": "local/llama", "family": "llama"}
    assert basis["generate_shard_sha256"] == sorted(
        read(p)["payload_sha256"] for p in (root / "generated").glob("*.json"))
    for row in source["rows"]:
        assert row["_judge_identity_sha256"] == source["identity_sha256"]
        assert row["_stage"] == "judge" and row["_backend"] == "fake"
    saved = (target / "judged/000000.json").read_bytes()
    assert I.run_judge(root, cfg, target, backend_factory=Fake)["exit_code"] == 0
    assert (target / "judged/000000.json").read_bytes() == saved
    assert immutable_bytes(root) == original
    before = len(Fake.made)
    assert I.run_judge(root, cfg, target, backend_factory=Fake)["new_shards"] == 0
    assert len(I.read_judge_source(root, cfg, target)["rows"]) == 5
    assert len(Fake.made) == before


def test_zero_max_does_not_create_output_or_load(prepared):
    root, cfg, target = prepared
    result = I.run_judge(root, cfg, target, max_shards=0, backend_factory=Fake)
    assert result["exit_code"] == 4 and result["judge_identity_sha256"] is None
    assert not target.exists() and not Fake.made
    assert not (root / ".inference.lock").exists()


@pytest.mark.parametrize("broken", ["missing", "marker", "policy_marker", "closure", "unbound",
                                    "not_ready", "wrong_policy_schema"])
def test_structured_parser_capability_required_before_writer_output(prepared, monkeypatch, broken):
    root, cfg, target = prepared
    if broken == "missing":
        monkeypatch.delattr(Q, "parse_judge_v6")
    elif broken == "marker":
        monkeypatch.setattr(Q, "JUDGE_V6_SCHEMA_VERSION", "judge_structured_v5")
    elif broken == "policy_marker":
        monkeypatch.setattr(P, "OUTPUT_SCHEMA_VERSION", "judge_structured_v5")
    elif broken == "closure":
        monkeypatch.setattr(F, "require_v6_parser", lambda: {"capabilities": {"judge_structured_v6": False}})
    elif broken == "unbound":
        monkeypatch.setattr(P, "policy_identity", lambda: {"parser_implementation": {"wrong": True}})
    elif broken == "not_ready":
        bound = P.policy_identity()
        monkeypatch.setattr(P, "policy_identity", lambda: {**bound, "structured_parser_ready": False})
    else:
        bound = P.policy_identity()
        monkeypatch.setattr(P, "policy_identity", lambda: {**bound, "output_schema": "judge_structured_v5"})
    with pytest.raises(I.StageRefused, match="parser_missing"):
        I.run_judge(root, cfg, target, backend_factory=Fake)
    assert not target.exists() and not Fake.made and not (root / ".inference.lock").exists()


def test_v6_source_cannot_be_read_under_v5_policy(prepared, monkeypatch):
    root, cfg, target = prepared
    I.run_judge(root, cfg, target, max_shards=1, backend_factory=Fake)
    saved = immutable_bytes(target)
    basis = read(target / "JUDGE.json")["identity_basis"]
    assert basis["policy_identity"]["output_schema"] == "judge_structured_v6"
    assert basis["policy_identity"]["parser_implementation"]["selected_schema"] == "judge_structured_v6"
    monkeypatch.setattr(P, "OUTPUT_SCHEMA_VERSION", "judge_structured_v5")
    with pytest.raises(I.StageRefused, match="parser_missing"):
        I.read_judge_source(root, cfg, target)
    assert immutable_bytes(target) == saved


def test_new_model_config_preserves_run_and_is_used_on_resume(prepared):
    root, cfg, target = prepared
    prior = immutable_bytes(root)
    changed = {**spec("phi"), "max_new_tokens": 96}
    I.run_judge(root, cfg, target, judge_config=changed, max_shards=1, backend_factory=Fake)
    assert I.run_judge(root, cfg, target, backend_factory=Fake)["exit_code"] == 0
    assert I.read_judge_source(root, cfg, target)["identity_basis"]["judge_model_config"] == changed
    with pytest.raises(I.StageRefused, match="judge_config_changed"):
        I.read_judge_source(root, cfg, target, judge_config=cfg["judge"])
    assert immutable_bytes(root) == prior


@pytest.mark.parametrize("kind", ["model", "tokens", "tokenizer", "system", "renderer", "parser"])
def test_identity_distinguishes_effective_inputs(prepared, monkeypatch, kind):
    root, cfg, first = prepared
    I.run_judge(root, cfg, first, max_shards=1, backend_factory=Fake)
    old = read(first / "JUDGE.json")["identity_sha256"]
    selected = dict(cfg["judge"])
    factory = Fake
    if kind == "model":
        selected = spec("phi")
    elif kind == "tokens":
        selected["max_new_tokens"] += 1
    elif kind == "tokenizer":
        def factory(cfg, stage):
            instance = Fake(cfg, stage)
            instance.identity["tokenizer_sha256"] = "d" * 64
            return instance
    elif kind == "system":
        monkeypatch.setattr(P, "JUDGE_SYSTEM_V2", P.JUDGE_SYSTEM_V2 + " 다른 지시.")
    elif kind == "renderer":
        renderer = P.render_judge
        monkeypatch.setattr(P, "render_judge", lambda task, generated: renderer(task, generated) + " ")
    else:
        identity = P.policy_identity
        monkeypatch.setattr(P, "policy_identity", lambda: {**identity(), "parser_file_sha256": "d" * 64})
    second = first.with_name("other-judge")
    I.run_judge(root, cfg, second, judge_config=selected, max_shards=1, backend_factory=factory)
    assert read(second / "JUDGE.json")["identity_sha256"] != old
    if kind in ("system", "renderer", "parser"):
        with pytest.raises(I.StageRefused, match="judge_identity_changed"):
            I.read_judge_source(root, cfg, first)


@pytest.mark.parametrize("where", ["source", "judge"])
def test_valid_rehashed_edits_are_not_adopted(prepared, where):
    root, cfg, target = prepared
    I.run_judge(root, cfg, target, max_shards=1, backend_factory=Fake)
    before = len(Fake.made)
    if where == "source":
        path = root / "generated/000000.json"
        rewrite_shard(path, lambda s: s["rows"][0].update(text="새로운 생성문"))
        code = "judge_identity_changed"
    else:
        path = target / "judged/000000.json"
        rewrite_shard(path, lambda s: s.update(judge_identity_sha256="d" * 64))
        code = "judge_shard_identity"
    with pytest.raises(I.StageRefused, match=code):
        I.run_judge(root, cfg, target, backend_factory=Fake)
    assert len(Fake.made) == before


def test_generated_byte_reformat_is_rejected(prepared):
    root, cfg, target = prepared
    I.run_judge(root, cfg, target, max_shards=1, backend_factory=Fake)
    path = root / "generated/000000.json"
    path.write_text(json.dumps(read(path), indent=2), encoding="utf-8")
    with pytest.raises(I.StageRefused, match="judge_identity_changed"):
        I.read_judge_source(root, cfg, target)


def test_legacy_shard_cannot_enter_separate_prefix(prepared):
    root, cfg, target = prepared
    I.run_stage(root, "judge", cfg, max_shards=1, backend_factory=Fake)
    I.run_judge(root, cfg, target, max_shards=1, backend_factory=Fake)
    (target / "judged/000000.json").write_bytes((root / "judged/000000.json").read_bytes())
    with pytest.raises(I.StageRefused, match="judge_shard_identity"):
        I.read_judge_source(root, cfg, target)


@pytest.mark.parametrize("where", ["root", "judge"])
def test_existing_locks_preserved_before_backend(prepared, where):
    root, cfg, target = prepared
    target.mkdir()
    lock = (root if where == "root" else target) / ".inference.lock"
    lock.write_text("another owner", encoding="utf-8")
    with pytest.raises(I.StageRefused, match="writer_locked"):
        I.run_judge(root, cfg, target, backend_factory=Fake)
    assert lock.read_text(encoding="utf-8") == "another owner" and not Fake.made
    with pytest.raises(I.StageRefused, match="writer_locked"):
        I.read_judge_source(root, cfg, target)


def test_both_locks_held_during_load_and_generation(prepared):
    root, cfg, target = prepared
    class Locked(Fake):
        def __init__(self, cfg, stage):
            assert (root / ".inference.lock").is_file()
            assert (target / ".inference.lock").is_file()
            super().__init__(cfg, stage)
        def generate_batch(self, *args):
            assert (root / ".inference.lock").is_file()
            assert (target / ".inference.lock").is_file()
            return super().generate_batch(*args)
    I.run_judge(root, cfg, target, max_shards=1, backend_factory=Locked)
    assert not (root / ".inference.lock").exists()
    assert not (target / ".inference.lock").exists()


@pytest.mark.parametrize("where,action", [("root", "remove"), ("root", "replace"),
                                        ("judge", "remove"), ("judge", "replace")])
def test_lost_lock_ownership_prevents_commit(prepared, where, action):
    root, cfg, target = prepared
    lock = (root if where == "root" else target) / ".inference.lock"
    class LosingLock(Fake):
        def generate_batch(self, *args):
            output = super().generate_batch(*args)
            if action == "remove":
                lock.unlink()
            else:
                lock.write_text("replacement owner", encoding="utf-8")
            return output
    with pytest.raises(I.StageRefused, match="writer_lock_lost"):
        I.run_judge(root, cfg, target, backend_factory=LosingLock)
    assert not (target / "judged/000000.json").exists()
    if action == "replace":
        assert lock.read_text(encoding="utf-8") == "replacement owner"


def test_genesis_write_does_not_adopt_concurrent_unknown_output(prepared, monkeypatch):
    root, cfg, target = prepared
    original = J._write_genesis
    def write_and_inject(folder, basis):
        record = original(folder, basis)
        (folder / "judged").mkdir()
        (folder / "judged/000000.json").write_text("{}", encoding="utf-8")
        return record
    monkeypatch.setattr(J, "_write_genesis", write_and_inject)
    with pytest.raises(I.StageRefused, match="judge_snapshot_changed"):
        I.run_judge(root, cfg, target, backend_factory=Fake)
    assert not Fake.calls


@pytest.mark.parametrize("name", ["JUDGE.json.tmp", "unknown.txt", "judged/000000.json.tmp"])
def test_partial_or_unknown_paths_are_preserved(prepared, name):
    root, cfg, target = prepared
    path = target / name
    path.parent.mkdir(parents=True)
    path.write_text("partial", encoding="utf-8")
    with pytest.raises(I.StageRefused, match="partial_or_unknown_output"):
        I.run_judge(root, cfg, target, backend_factory=Fake)
    assert path.read_text(encoding="utf-8") == "partial" and not Fake.made


def test_prefix_gap_and_unknown_row_id_refused(prepared):
    root, cfg, target = prepared
    I.run_judge(root, cfg, target, backend_factory=Fake)
    path = target / "judged/000000.json"
    saved = path.read_bytes()
    path.unlink()
    with pytest.raises(I.StageRefused, match="shard_gap"):
        I.read_judge_source(root, cfg, target)
    path.write_bytes(saved)
    rewrite_shard(path, lambda s: s["rows"][0].update(id="unknown"))
    with pytest.raises(I.StageRefused, match="row_order"):
        I.read_judge_source(root, cfg, target)


@pytest.mark.parametrize("failure,code", [("oom", 5), ("time", 3)])
def test_partial_failure_then_explicit_same_identity_resume(prepared, failure, code):
    root, cfg, target = prepared
    Fake.failure_at, Fake.failure = 2, failure
    result = I.run_judge(root, cfg, target, backend_factory=Fake)
    assert result["exit_code"] == code and result["completed_shards"] == 1
    assert not (target / "judged/000001.json").exists()
    assert Fake.closed == ["judge"]
    old = read(target / "JUDGE.json")["identity_sha256"]
    Fake.failure_at = None
    assert I.run_judge(root, cfg, target, backend_factory=Fake)["exit_code"] == 0
    assert read(target / "JUDGE.json")["identity_sha256"] == old


def test_genesis_without_rows_can_only_resume_same_runtime(prepared):
    root, cfg, target = prepared
    Fake.failure_at = 1
    assert I.run_judge(root, cfg, target, backend_factory=Fake)["exit_code"] == 5
    assert I.read_judge_source(root, cfg, target)["rows"] == []
    Fake.failure_at = None
    def changed(cfg, stage):
        instance = Fake(cfg, stage)
        instance.identity["tokenizer_sha256"] = "e" * 64
        return instance
    with pytest.raises(I.StageRefused, match="runtime_changed"):
        I.run_judge(root, cfg, target, backend_factory=changed)
    assert not (target / "judged").exists()


@pytest.mark.parametrize("failure,code", [("oom", 5), ("time", 3)])
def test_runtime_binding_failure_before_genesis_closes_model(prepared, monkeypatch, failure, code):
    root, cfg, target = prepared
    def fail_binding(model, spec):
        if failure == "time":
            raise I.StageTimeLimit()
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(J, "judge_runtime_identity", fail_binding)
    result = I.run_judge(root, cfg, target, backend_factory=Fake)
    assert result["exit_code"] == code and result["runtime_identity_available"] is False
    assert not (target / "JUDGE.json").exists() and Fake.closed == ["judge"]


@pytest.mark.parametrize("when", ["load", "batch"])
def test_source_change_during_runtime_prevents_commit(prepared, when):
    root, cfg, target = prepared
    path = root / "generated/000000.json"
    class Mutating(Fake):
        def __init__(self, cfg, stage):
            super().__init__(cfg, stage)
            if when == "load":
                rewrite_shard(path, lambda s: s["rows"][0].update(text="changed"))
        def generate_batch(self, *args):
            output = super().generate_batch(*args)
            if when == "batch":
                rewrite_shard(path, lambda s: s["rows"][0].update(text="changed"))
            return output
    with pytest.raises(I.StageRefused, match="judge_snapshot_changed"):
        I.run_judge(root, cfg, target, backend_factory=Mutating)
    assert not (target / "judged/000000.json").exists()
    assert Fake.closed == ["judge"]


def test_runtime_change_during_batch_prevents_commit(prepared):
    root, cfg, target = prepared
    class Mutating(Fake):
        def generate_batch(self, *args):
            output = super().generate_batch(*args)
            self.identity["tokenizer_sha256"] = "e" * 64
            return output
    with pytest.raises(I.StageRefused, match="runtime_changed"):
        I.run_judge(root, cfg, target, backend_factory=Mutating)
    assert not (target / "judged/000000.json").exists()


def test_loaded_eos_pad_and_defaults_change_runtime_identity():
    cfg = spec("phi")
    fake = Fake(cfg, "judge")
    fake.identity["backend"] = "transformers"
    defaults = {"eos_token_id": [1, 2], "pad_token_id": 1, "repetition_penalty": 1.0}
    fake.model = SimpleNamespace(generation_config=SimpleNamespace(
        to_dict=lambda: dict(defaults), eos_token_id=[1, 2]))
    fake.tokenizer = SimpleNamespace(pad_token_id=1, eos_token_id=1, chat_template="template")
    first = J._runtime(fake, cfg)
    J._check_runtime(first, cfg)
    defaults["repetition_penalty"] = 1.1
    assert J._runtime(fake, cfg) != first
    defaults["repetition_penalty"] = 1.0
    fake.tokenizer.pad_token_id = 2
    assert J._runtime(fake, cfg) != first
    fake.tokenizer.pad_token_id = 1
    fake.model.generation_config.eos_token_id = [3]
    defaults["eos_token_id"] = [3]
    assert J._runtime(fake, cfg) != first


@pytest.mark.parametrize("relative", [".", "generated", "generated/nested", "judged", "judged/nested"])
def test_reserved_legacy_paths_rejected(prepared, relative):
    root, cfg, _ = prepared
    with pytest.raises(I.StageRefused, match="judge_dir_overlap"):
        I.run_judge(root, cfg, root / relative, max_shards=0, backend_factory=Fake)
    assert not Fake.made


def test_ancestor_directory_rejected(prepared):
    root, cfg, _ = prepared
    with pytest.raises(I.StageRefused, match="judge_dir_overlap"):
        I.run_judge(root, cfg, root.parent, max_shards=0, backend_factory=Fake)


def test_symlink_alias_rejected_without_backend(prepared):
    root, cfg, target = prepared
    alias = target.with_name("alias")
    try:
        alias.symlink_to(root / "judged", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("host does not permit creating test symlinks")
    with pytest.raises(I.StageRefused, match="path_alias"):
        I.run_judge(root, cfg, alias, max_shards=0, backend_factory=Fake)
    assert not Fake.made


def test_source_qa_user_prompt_remains_legacy_and_model_free_read(tmp_path, monkeypatch):
    root, target = tmp_path / "run", tmp_path / "judge"
    cfg = make_run(root, n=1, source_qa=True)
    I.run_stage(root, "generate", cfg, backend_factory=Fake)
    reset_fake()
    I.run_judge(root, cfg, target, backend_factory=Fake)
    task = I._json((root / "tasks.jsonl").read_bytes())
    generated = read(root / "generated/000000.json")["rows"][0]
    assert Fake.calls[0][1][0] == I._judge_prompt(task, generated)
    assert "row_class" not in json.loads(Fake.calls[0][1][0])
    def forbidden(*args):
        raise AssertionError("reader loaded a backend")
    monkeypatch.setattr(I, "TransformersBackend", forbidden)
    assert len(I.read_judge_source(root, cfg, target)["rows"]) == 1


def test_generation_must_be_complete_before_output_or_load(tmp_path):
    root, target = tmp_path / "run", tmp_path / "judge"
    cfg = make_run(root)
    I.run_stage(root, "generate", cfg, max_shards=1, backend_factory=Fake)
    reset_fake()
    with pytest.raises(I.StageRefused, match="generation_incomplete"):
        I.run_judge(root, cfg, target, backend_factory=Fake)
    assert not target.exists() and not Fake.made


def test_configured_self_judging_refused(prepared):
    root, cfg, target = prepared
    with pytest.raises(I.StageRefused, match="self_judging"):
        I.run_judge(root, cfg, target, judge_config=cfg["generation"], backend_factory=Fake)
    assert not target.exists() and not Fake.made


def test_raw_judge_json_is_not_corrected_or_approved(prepared):
    root, cfg, target = prepared
    malformed = '{"grounded":true,"extra":"must remain raw"}'
    class Raw(Fake):
        def generate_batch(self, *args):
            return [{**r, "text": malformed, "finish_reason": "length"}
                    for r in super().generate_batch(*args)]
    I.run_judge(root, cfg, target, max_shards=1, backend_factory=Raw)
    source = I.read_judge_source(root, cfg, target)
    assert all(r["text"] == malformed and r["finish_reason"] == "length" for r in source["rows"])
    assert all(r["_backend"] == "fake" for r in source["rows"])


@pytest.mark.parametrize("value", [-1, True, "1"])
def test_invalid_shard_limit_refused(prepared, value):
    root, cfg, target = prepared
    with pytest.raises(I.StageRefused, match="invalid_max_shards"):
        I.run_judge(root, cfg, target, max_shards=value, backend_factory=Fake)
    assert not target.exists() and not Fake.made


class ProductionLookingFake(Fake):
    """Emulated runtime for writer contract tests, not GPU approval evidence."""
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)
        self.identity["backend"] = "transformers"
        defaults = {"eos_token_id": [2], "pad_token_id": 0}
        self.model = SimpleNamespace(generation_config=SimpleNamespace(
            to_dict=lambda: dict(defaults), eos_token_id=[2]))
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, chat_template="synthetic template")


@pytest.fixture
def approved_writer(prepared, monkeypatch):
    # Isolate writer linkage here; the real seal/read verifier is covered by the
    # probe tests. No actual probe/production approval is created by this fixture.
    root, cfg, target = prepared
    source = root.parent / "mock-approval"
    source.mkdir()
    raw = b"synthetic approval boundary"
    (source / "APPROVAL.json").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    runtime = J.judge_runtime_identity(ProductionLookingFake(cfg["judge"], "judge"), cfg["judge"])
    reset_fake()
    basis = {"judge_config": cfg["judge"], "input_basis": {
        "policy_identity": P.policy_identity(), "cases_file_sha256": "f" * 64,
        "implementation_sha256": K._code_pins()},
        "common_implementation_sha256": J._source_pins(root)["implementation"],
        "runtime_model": runtime, "probe_identity_sha256": ["1" * 64, "2" * 64],
        "probe_call_ids": ["1" * 32, "2" * 32]}
    approval = {"receipt_sha256": sha, "identity_sha256": I._digest(basis),
                "identity_basis": basis, "files_sha256": {"APPROVAL.json": sha}}
    def files(folder):
        return {"APPROVAL.json": I._file_sha(Path(folder) / "APPROVAL.json")}
    def verify(folder, expected):
        if expected != sha or files(folder) != approval["files_sha256"]:
            raise I.StageRefused("probe_approval_pin", "synthetic boundary differs")
        return J._clone(approval)
    def copy(source_dir, target_dir, expected):
        result = verify(source_dir, expected)
        Path(target_dir).mkdir()
        (Path(target_dir) / "APPROVAL.json").write_bytes((Path(source_dir) / "APPROVAL.json").read_bytes())
        return result
    monkeypatch.setattr(K, "approval_files", files)
    monkeypatch.setattr(K, "read_probe_approval", verify)
    monkeypatch.setattr(K, "copy_probe_approval", copy)
    monkeypatch.setattr(I, "TransformersBackend", ProductionLookingFake)
    return prepared, source, sha, approval


def test_default_backend_requires_approval_before_output_or_load(prepared, monkeypatch):
    root, cfg, target = prepared
    def forbidden(*args):
        pytest.fail("missing approval must be rejected before model load")
    monkeypatch.setattr(I, "TransformersBackend", forbidden)
    with pytest.raises(I.StageRefused, match="probe_approval_required"):
        I.run_judge(root, cfg, target)
    assert not target.exists() and not Fake.made
    assert not (root / ".inference.lock").exists()


def test_injected_production_looking_runtime_without_approval_refused(prepared):
    root, cfg, target = prepared
    with pytest.raises(I.StageRefused, match="probe_approval_backend"):
        I.run_judge(root, cfg, target, backend_factory=ProductionLookingFake)
    assert not (target / "JUDGE.json").exists() and not Fake.calls
    assert Fake.closed == ["judge"]


def test_writer_binds_portable_approval_and_requires_explicit_pin_on_resume(approved_writer):
    (root, cfg, target), approval_dir, sha, approval = approved_writer
    first = I.run_judge(root, cfg, target, max_shards=1,
                        probe_approval=approval_dir, probe_approval_sha256=sha)
    assert first["probe_approval_sha256"] == sha and first["gpu_model_verified"] is True
    source = I.read_judge_source(root, cfg, target, expected_judge_identity_sha256=first["judge_identity_sha256"])
    binding = source["identity_basis"]["probe_approval"]
    assert binding["receipt_sha256"] == sha
    assert binding["identity_sha256"] == approval["identity_sha256"]
    assert binding["probe_call_ids"] == ["1" * 32, "2" * 32]
    assert binding["writer_run_identity_sha256"] == source["identity_basis"]["run_identity_sha256"]
    assert binding["writer_role"] == "separate-judge-not-a-probe"
    with pytest.raises(I.StageRefused, match="probe_approval_pin"):
        I.run_judge(root, cfg, target)
    # The original external directory is no longer needed by the read path.
    approval_dir.rename(approval_dir.with_name("original-unavailable"))
    assert I.read_judge_source(root, cfg, target,
                              **source["caller_expected_pins"])["identity_sha256"] == source["identity_sha256"]
    result = I.run_judge(root, cfg, target, probe_approval=target / "probe_approval",
                         probe_approval_sha256=sha)
    assert result["exit_code"] == 0 and result["completed_tasks"] == 5


@pytest.mark.parametrize("field", ["judge_config", "policy", "implementation", "runtime"])
def test_approval_writer_identity_mismatch_refused(approved_writer, field):
    (root, cfg, target), folder, sha, approval = approved_writer
    basis = approval["identity_basis"]
    if field == "judge_config":
        basis["judge_config"] = {**cfg["judge"], "max_new_tokens": 65}
    elif field == "policy":
        basis["input_basis"]["policy_identity"] = {"wrong": True}
    elif field == "implementation":
        basis["common_implementation_sha256"]["infer.py"] = "e" * 64
    else:
        basis["runtime_model"]["tokenizer_sha256"] = "e" * 64
    with pytest.raises(I.StageRefused, match="probe_approval_runtime" if field == "runtime" else "probe_approval_identity"):
        I.run_judge(root, cfg, target, probe_approval=folder, probe_approval_sha256=sha)
    assert not (target / "JUDGE.json").exists() and not Fake.calls
    if field != "runtime":
        assert not Fake.made and not target.exists()


def test_injected_runtime_cannot_consume_even_matching_approval(approved_writer):
    (root, cfg, target), folder, sha, _ = approved_writer
    with pytest.raises(I.StageRefused, match="probe_approval_backend"):
        I.run_judge(root, cfg, target, backend_factory=ProductionLookingFake,
                    probe_approval=folder, probe_approval_sha256=sha)
    assert not (target / "JUDGE.json").exists() and not Fake.calls


@pytest.mark.parametrize("when", ["read", "resume", "batch"])
def test_approval_tamper_prevents_read_resume_and_commit(approved_writer, monkeypatch, when):
    (root, cfg, target), folder, sha, _ = approved_writer
    selected = target / "probe_approval/APPROVAL.json"
    if when == "batch":
        class Mutating(ProductionLookingFake):
            def generate_batch(self, *args):
                result = super().generate_batch(*args)
                selected.write_bytes(b"changed")
                return result
        monkeypatch.setattr(I, "TransformersBackend", Mutating)
        with pytest.raises(I.StageRefused, match="judge_snapshot_changed|probe_approval_changed"):
            I.run_judge(root, cfg, target, probe_approval=folder, probe_approval_sha256=sha)
        assert not (target / "judged/000000.json").exists()
    else:
        I.run_judge(root, cfg, target, max_shards=1, probe_approval=folder, probe_approval_sha256=sha)
        selected.write_bytes(b"changed")
        loaded = len(Fake.made)
        with pytest.raises(I.StageRefused, match="probe_approval_pin"):
            if when == "read":
                I.read_judge_source(root, cfg, target)
            else:
                I.run_judge(root, cfg, target, probe_approval=folder, probe_approval_sha256=sha)
        assert len(Fake.made) == loaded


def test_fake_identity_is_labelled_and_cannot_be_upgraded_to_real(prepared, monkeypatch):
    root, cfg, target = prepared
    result = I.run_judge(root, cfg, target, max_shards=1, backend_factory=Fake)
    assert result["probe_approval_sha256"] is None and result["gpu_model_verified"] is False
    source = I.read_judge_source(root, cfg, target)
    assert source["identity_basis"]["probe_approval"] is None and not source["gpu_model_verified"]
    assert source["pin_provenance"] == "unapproved-fake"
    assert source["caller_expected_pins"] == {
        "expected_judge_identity_sha256": None, "expected_probe_approval_sha256": None}
    monkeypatch.setattr(I, "TransformersBackend", ProductionLookingFake)
    loaded = len(Fake.made)
    with pytest.raises(I.StageRefused, match="probe_approval_required"):
        I.run_judge(root, cfg, target)
    assert len(Fake.made) == loaded


@pytest.mark.parametrize("choice", ["judge", "approval", "both"])
def test_production_reader_requires_and_preserves_caller_pins(approved_writer, choice):
    (root, cfg, target), folder, sha, _ = approved_writer
    result = I.run_judge(root, cfg, target, max_shards=1, probe_approval=folder, probe_approval_sha256=sha)
    pins = {"expected_judge_identity_sha256": result["judge_identity_sha256"] if choice != "approval" else None,
            "expected_probe_approval_sha256": sha if choice != "judge" else None}
    loaded = len(Fake.made)
    with pytest.raises(I.StageRefused, match="judge_source_pin_required"):
        I.read_judge_source(root, cfg, target)
    source = I.read_judge_source(root, cfg, target, **pins)
    assert source["caller_expected_pins"] == pins and source["pin_provenance"] == "explicit-caller"
    assert I.read_judge_source(root, cfg, target, **source["caller_expected_pins"]) == source
    assert len(Fake.made) == loaded


@pytest.mark.parametrize("field", ["expected_judge_identity_sha256", "expected_probe_approval_sha256"])
def test_otherwise_valid_source_cannot_replace_callers_selected_identity(approved_writer, field):
    (root, cfg, target), folder, sha, _ = approved_writer
    result = I.run_judge(root, cfg, target, max_shards=1, probe_approval=folder, probe_approval_sha256=sha)
    pins = {"expected_judge_identity_sha256": result["judge_identity_sha256"],
            "expected_probe_approval_sha256": sha}
    pins[field] = "e" * 64
    with pytest.raises(I.StageRefused, match="judge_source_pin"):
        I.read_judge_source(root, cfg, target, **pins)


@pytest.mark.parametrize("value", [True, "", "not-a-sha", "e" * 63])
def test_reader_rejects_malformed_external_pin_even_for_fake(prepared, value):
    root, cfg, target = prepared
    I.run_judge(root, cfg, target, max_shards=1, backend_factory=Fake)
    with pytest.raises(I.StageRefused, match="judge_source_pin"):
        I.read_judge_source(root, cfg, target, expected_judge_identity_sha256=value)


def test_fake_reader_does_not_ignore_explicit_wrong_pin(prepared):
    root, cfg, target = prepared
    result = I.run_judge(root, cfg, target, max_shards=1, backend_factory=Fake)
    source = I.read_judge_source(root, cfg, target,
                                 expected_judge_identity_sha256=result["judge_identity_sha256"])
    assert source["pin_provenance"] == "explicit-caller" and not source["gpu_model_verified"]
    with pytest.raises(I.StageRefused, match="judge_source_pin"):
        I.read_judge_source(root, cfg, target, expected_probe_approval_sha256="a" * 64)
