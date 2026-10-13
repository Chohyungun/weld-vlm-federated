"""CPU-only contracts for the production inference runner; no model imports."""

import hashlib
import json
from typing import ClassVar

import pytest

from corpus.production import infer as I


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def spec(family="qwen"):
    return {"model_id": f"local/{family}", "revision": "a" * 40,
            "family": family, "local_files_only": True, "quantization": "bf16",
            "batch_size": 2, "max_new_tokens": 32, "max_input_tokens": 4096,
            "seed": 20261007, "wall_seconds": 1800, "enable_thinking": False}


def make_run(tmp_path, n=5):
    cfg = {"generation": spec(), "judge": spec("llama"), "shard_size": 2}
    tasks = [{"id": f"i-{i:03d}", "kind": "image_report", "split": "train", "client": "C1",
              "lineage_group_id": f"g-{i}", "image_path": f"images/{i}.png", "source_ids": ["ann"],
              "evidence": {"annotations": ["porosity"]}, "locked_target": {"defects": []},
              "prompt": f"기공 주석을 설명하라 {i}"} for i in range(n)]
    raw = b"".join(canonical(t) + b"\n" for t in tasks)
    (tmp_path / "tasks.jsonl").write_bytes(raw)
    run = {"schema_version": 1, "identity_sha256": "b" * 64, "config_sha256": digest(cfg),
           "tasks_sha256": hashlib.sha256(raw).hexdigest(), "counts": {"total": n}, "config": cfg}
    (tmp_path / "RUN.json").write_bytes(canonical(run))
    return cfg


class Fake:
    made: ClassVar[list] = []
    calls: ClassVar[list] = []
    closed: ClassVar[list] = []
    failure_call = None
    failure_type = "oom"

    def __init__(self, cfg, stage):
        self.stage = stage
        self.identity = {"backend": "fake", "model_id": cfg["model_id"], "revision": cfg["revision"],
                         "family": cfg["family"], "model_type": cfg["family"],
                         "quantization": cfg["quantization"], "tokenizer_sha256": "c" * 64}
        self.made.append(stage)

    def generate_batch(self, prompts, system, deadline):
        self.calls.append((self.stage, prompts, system))
        if len(self.calls) == self.failure_call:
            if self.failure_type == "time":
                raise I.StageTimeLimit()
            raise RuntimeError("CUDA out of memory")
        text = ("기공이 주석에 표시되어 있습니다." if self.stage == "generate" else
                '{"grounded":true,"scope_ok":true,"language_ok":true,"reason":"주석과 일치"}')
        return [{"text": text, "finish_reason": "eos", "input_tokens": 20, "output_tokens": 15} for _ in prompts]

    def close(self):
        self.closed.append(self.stage)


@pytest.fixture(autouse=True)
def clear_fake():
    Fake.made, Fake.calls, Fake.closed = [], [], []
    Fake.failure_call, Fake.failure_type = None, "oom"


def test_generate_resume_and_judge_are_complete(tmp_path):
    cfg = make_run(tmp_path)
    result = I.run_stage(tmp_path, "generate", cfg, max_shards=1, backend_factory=Fake)
    assert result["exit_code"] == 4 and result["completed_tasks"] == 2
    first = (tmp_path / "generated/000000.json").read_bytes()
    result = I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)
    assert result["exit_code"] == 0 and result["completed_tasks"] == 5
    assert (tmp_path / "generated/000000.json").read_bytes() == first
    assert len(Fake.calls) == 3
    assert I.run_stage(tmp_path, "judge", cfg, backend_factory=Fake)["exit_code"] == 0
    rows = I.read_stage_rows(tmp_path, "judge", cfg)
    assert [x["id"] for x in rows] == [f"i-{i:03d}" for i in range(5)]
    assert all(x["_stage"] == "judge" and x["_backend"] == "fake" for x in rows)
    assert all(json.loads(x["text"])["grounded"] is True for x in rows)
    judge_prompt = json.loads(next(c for c in Fake.calls if c[0] == "judge")[1][0])
    assert judge_prompt["candidate"] == "기공이 주석에 표시되어 있습니다."
    assert judge_prompt["evidence"] == {"annotations": ["porosity"]}
    before = len(Fake.made)
    assert I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)["new_shards"] == 0
    assert len(Fake.made) == before


def test_zero_max_shards_never_loads_backend(tmp_path):
    cfg = make_run(tmp_path)
    assert I.run_stage(tmp_path, "generate", cfg, max_shards=0, backend_factory=Fake)["exit_code"] == 4
    assert Fake.made == []
    assert not (tmp_path / ".inference.lock").exists()


@pytest.mark.parametrize("client,accepted", [("public", True), ("public_text", False)])
def test_source_qa_uses_public_client_not_training_scope_name(tmp_path, client, accepted):
    cfg = make_run(tmp_path, n=1)
    task = {"id": "q-001", "kind": "source_qa", "split": "train", "client": client,
            "lineage_group_id": "source-001", "image_path": None, "source_ids": ["source-001"],
            "evidence": {"passage": "기공은 용접금속에 갇힌 가스가 남긴 공동이다."},
            "locked_target": None, "prompt": "주어진 원문으로 기공의 정의를 설명하라."}
    raw = canonical(task) + b"\n"
    (tmp_path / "tasks.jsonl").write_bytes(raw)
    run = json.loads((tmp_path / "RUN.json").read_text(encoding="utf-8"))
    run["tasks_sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "RUN.json").write_bytes(canonical(run))
    if accepted:
        assert I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)["exit_code"] == 0
        assert I.read_stage_rows(tmp_path, "generate", cfg)[0]["id"] == "q-001"
    else:
        with pytest.raises(I.StageRefused, match="source_task_contract"):
            I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)
        assert Fake.made == []


@pytest.mark.parametrize("change,code", [("config", "config_changed"), ("task", "tasks_changed")])
def test_input_changes_rejected_before_loading(tmp_path, change, code):
    cfg = make_run(tmp_path)
    if change == "config":
        cfg["generation"]["batch_size"] = 1
    else:
        with (tmp_path / "tasks.jsonl").open("ab") as f:
            f.write(b"\n")
    with pytest.raises(I.StageRefused, match=code):
        I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)
    assert not Fake.made


def test_tampered_shard_and_rehashed_wrong_ids_are_rejected(tmp_path):
    cfg = make_run(tmp_path)
    I.run_stage(tmp_path, "generate", cfg, max_shards=1, backend_factory=Fake)
    path = tmp_path / "generated/000000.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["rows"][0]["text"] = "다른 내용"
    path.write_bytes(canonical(data))
    with pytest.raises(I.StageRefused, match="shard_digest"):
        I.read_stage_rows(tmp_path, "generate", cfg)
    data["rows"][0]["id"] = "wrong"
    data["payload_sha256"] = digest({k: v for k, v in data.items() if k != "payload_sha256"})
    path.write_bytes(canonical(data))
    with pytest.raises(I.StageRefused, match="row_order"):
        I.read_stage_rows(tmp_path, "generate", cfg)


def test_judge_requires_complete_generation(tmp_path):
    cfg = make_run(tmp_path)
    I.run_stage(tmp_path, "generate", cfg, max_shards=1, backend_factory=Fake)
    with pytest.raises(I.StageRefused, match="generation_incomplete"):
        I.run_stage(tmp_path, "judge", cfg, backend_factory=Fake)
    assert Fake.made == ["generate"]


@pytest.mark.parametrize("failure,exit_code", [("oom", 5), ("time", 3)])
def test_interruption_preserves_completed_shards_and_resumes(tmp_path, failure, exit_code):
    cfg = make_run(tmp_path)
    Fake.failure_call, Fake.failure_type = 2, failure
    result = I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)
    assert result["exit_code"] == exit_code
    assert result["completed_tasks"] == 2 and result["pending_tasks"] == 3
    assert not (tmp_path / "generated/000001.json").exists()
    assert not (tmp_path / ".inference.lock").exists()
    assert Fake.closed == ["generate"]
    Fake.failure_call = None
    assert I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)["exit_code"] == 0


def test_exclusive_lock_and_partial_file_are_preserved(tmp_path):
    cfg = make_run(tmp_path)
    lock = tmp_path / ".inference.lock"
    lock.write_text("existing writer")
    with pytest.raises(I.StageRefused, match="writer_locked"):
        I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)
    assert lock.read_text() == "existing writer"
    lock.unlink()
    (tmp_path / "generated").mkdir()
    partial = tmp_path / "generated/000000.json.tmp"
    partial.write_text("interrupted")
    with pytest.raises(I.StageRefused, match="partial_or_unknown_output"):
        I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)
    assert partial.read_text() == "interrupted"
    assert not Fake.made


def test_hole_in_completed_shards_rejected(tmp_path):
    cfg = make_run(tmp_path)
    I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)
    (tmp_path / "generated/000001.json").unlink()
    with pytest.raises(I.StageRefused, match="shard_gap"):
        I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)


def test_judge_is_bound_to_exact_generated_payload(tmp_path):
    cfg = make_run(tmp_path)
    I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)
    I.run_stage(tmp_path, "judge", cfg, backend_factory=Fake)
    path = tmp_path / "generated/000000.json"
    shard = json.loads(path.read_text(encoding="utf-8"))
    shard["rows"][0]["text"] = "기공이 다르게 표시되어 있습니다."
    shard["payload_sha256"] = digest({k: v for k, v in shard.items() if k != "payload_sha256"})
    path.write_bytes(canonical(shard))
    with pytest.raises(I.StageRefused, match="shard_inputs"):
        I.read_stage_rows(tmp_path, "judge", cfg)


def test_actual_family_mismatch_and_resume_model_change_rejected(tmp_path):
    cfg = make_run(tmp_path)

    def wrong_family(cfg, stage):
        model = Fake(cfg, stage)
        model.identity["model_type"] = "llama"
        return model

    with pytest.raises(I.StageRefused, match="model_family"):
        I.run_stage(tmp_path, "generate", cfg, backend_factory=wrong_family)
    I.run_stage(tmp_path, "generate", cfg, max_shards=1, backend_factory=Fake)

    def different_tokenizer(cfg, stage):
        model = Fake(cfg, stage)
        model.identity["tokenizer_sha256"] = "d" * 64
        return model

    with pytest.raises(I.StageRefused, match="runtime_changed"):
        I.run_stage(tmp_path, "generate", cfg, backend_factory=different_tokenizer)


def test_cross_family_requirement(tmp_path):
    cfg = make_run(tmp_path)
    cfg["judge"] = spec("qwen")
    run = json.loads((tmp_path / "RUN.json").read_text())
    run.update(config=cfg, config_sha256=digest(cfg))
    (tmp_path / "RUN.json").write_bytes(canonical(run))
    with pytest.raises(I.StageRefused, match="self_judging"):
        I.run_stage(tmp_path, "judge", cfg, backend_factory=Fake)


def test_invalid_token_count_prevents_commit(tmp_path):
    cfg = make_run(tmp_path)

    class Invalid(Fake):
        def generate_batch(self, prompts, system, deadline):
            rows = super().generate_batch(prompts, system, deadline)
            rows[0]["output_tokens"] = 33
            return rows

    with pytest.raises(I.StageRefused, match="token_count"):
        I.run_stage(tmp_path, "generate", cfg, backend_factory=Invalid)
    assert not (tmp_path / "generated/000000.json").exists()


def test_unpinned_revision_and_network_setting_rejected(tmp_path):
    cfg = make_run(tmp_path)
    cfg["generation"]["revision"] = "main"
    run = json.loads((tmp_path / "RUN.json").read_text())
    run.update(config=cfg, config_sha256=digest(cfg))
    (tmp_path / "RUN.json").write_bytes(canonical(run))
    with pytest.raises(I.StageRefused, match="unpinned_revision"):
        I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)


def test_local_model_rejects_commit_name_without_importing_model_runtime(tmp_path):
    model = tmp_path / "snapshots" / ("a" * 40)
    model.mkdir(parents=True)
    (model / "config.json").write_bytes(b'{"_commit_hash":"' + b"a" * 40 + b'"}')
    cfg = spec()
    cfg["model_id"] = str(model)
    with pytest.raises(I.StageRefused, match="local_revision"):
        I._model_config({"generation": cfg}, "generate")
    with pytest.raises(I.StageRefused, match="local_revision"):
        I.TransformersBackend(cfg, "generate")


def test_local_tree_hash_detects_weights_changed_under_same_snapshot_name(tmp_path):
    model = tmp_path / ("a" * 40)
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"original weights")
    revision = "sha256:" + digest([["model.safetensors", hashlib.sha256(weights.read_bytes()).hexdigest()]])
    assert I._model_location(str(model), revision) == model
    I._verify_local_tree(model, revision)
    weights.write_bytes(b"modified weights")
    with pytest.raises(I.StageRefused, match="local_model_changed"):
        I._verify_local_tree(model, revision)
    with pytest.raises(I.StageRefused, match="local_revision"):
        I._model_location(str(tmp_path / "absent"), revision)


def test_eval_and_lineage_leakage_rejected_even_with_updated_digest(tmp_path):
    cfg = make_run(tmp_path)
    tasks = [json.loads(line) for line in (tmp_path / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    tasks[1]["split"] = "val"
    tasks[1]["lineage_group_id"] = tasks[0]["lineage_group_id"]
    raw = b"".join(canonical(t) + b"\n" for t in tasks)
    (tmp_path / "tasks.jsonl").write_bytes(raw)
    run = json.loads((tmp_path / "RUN.json").read_text())
    run["tasks_sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "RUN.json").write_bytes(canonical(run))
    with pytest.raises(I.StageRefused, match="lineage_split_overlap"):
        I.run_stage(tmp_path, "generate", cfg, backend_factory=Fake)
