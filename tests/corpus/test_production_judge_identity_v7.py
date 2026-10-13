"""CPU-only v7 writer plumbing, with synthetic policy/parser and fake backend.

These tests establish storage, no-call and identity boundaries, not policy
discrimination, actual parser compatibility, or GPU probe approval.
"""

import hashlib
import json

import pytest

from corpus.production import infer as I
from corpus.production import judge_fingerprints as F
from corpus.production import judge_identity as J
from corpus.production import judge_policy as P
from corpus.production import quality as Q


def model_spec(family="qwen", tokens=64):
    return {"model_id": f"local/{family}", "revision": "a" * 40, "family": family,
            "local_files_only": True, "quantization": "bf16", "batch_size": 2,
            "max_new_tokens": tokens, "max_input_tokens": 4096, "seed": 1,
            "wall_seconds": 1800, "enable_thinking": False}


def prepare(root, defect_ids=(0, 4), n=5, judge_tokens=160):
    root.mkdir()
    cfg = {"generation": model_spec(), "judge": model_spec("llama", judge_tokens), "shard_size": 2}
    tasks = [{"id": f"id-{i}", "kind": "image_report", "split": "train", "client": "C1",
              "lineage_group_id": f"lineage-{i}", "image_path": f"not-read/{i}.png",
              "source_ids": ["annotation"], "evidence": {"annotations": [],
                  "defect_counts_by_name": {}, "defect_positions": [], "size_mm_available": False},
              "locked_target": {"defects": [{"test_only": True}] if i in defect_ids else []},
              "prompt": "Describe the supplied annotation."} for i in range(n)]
    raw = b"".join(I._canonical(task) + b"\n" for task in tasks)
    (root / "tasks.jsonl").write_bytes(raw)
    (root / "RUN.json").write_bytes(I._canonical({"schema_version": 1,
        "identity_sha256": "b" * 64, "config_sha256": I._digest(cfg),
        "tasks_sha256": hashlib.sha256(raw).hexdigest(), "config": cfg, "counts": {"total": n}}))
    I.run_stage(root, "generate", cfg, backend_factory=Fake)
    Fake.reset()
    return cfg


class Fake:
    made = []
    calls = []
    closed = []

    @classmethod
    def reset(cls):
        cls.made, cls.calls, cls.closed = [], [], []

    def __init__(self, cfg, stage):
        self.identity = {"backend": "fake", "model_id": cfg["model_id"],
                         "revision": cfg["revision"], "family": cfg["family"],
                         "model_type": cfg["family"], "quantization": cfg["quantization"],
                         "tokenizer_sha256": "c" * 64}
        self.stage, self.enabled = stage, False
        Fake.made.append(stage)

    def enable_json_object_stopping(self):
        assert self.stage == "judge"
        self.enabled = True

    def generate_batch(self, prompts, system, deadline):
        Fake.calls.append((self.stage, list(prompts), system))
        if self.stage == "generate":
            text, finish = "Synthetic candidate.", "eos"
        else:
            assert self.enabled
            text, finish = '{"test_only":true}', "json_closed"
        return [{"text": text, "finish_reason": finish, "input_tokens": 20,
                 "output_tokens": 15} for _ in prompts]

    def close(self):
        Fake.closed.append(self.stage)


@pytest.fixture(autouse=True)
def synthetic_v7(monkeypatch):
    Fake.reset()
    parser_identity = {"selected_schema": "judge_structured_v7",
                       "capabilities": {"judge_structured_v7": True},
                       "closure_sha256": "7" * 64, "test_only": True}
    policy_identity = {"output_schema": "judge_structured_v7", "structured_parser_ready": True,
                       "parser_implementation": parser_identity, "test_only": True}
    monkeypatch.setattr(Q, "JUDGE_V7_SCHEMA_VERSION", "judge_structured_v7", raising=False)
    def forbidden_parser(*args, **kwargs):
        raise AssertionError("writer/reader must preserve raw output, not parse model rows")
    monkeypatch.setattr(Q, "parse_judge_v7", forbidden_parser, raising=False)
    monkeypatch.setattr(F, "require_v7_parser", lambda: parser_identity, raising=False)
    monkeypatch.setattr(P, "policy_identity_v7", lambda: dict(policy_identity))

    def evaluate(task, text, *, generation_contract):
        assert generation_contract is True
        applicable = bool(task["locked_target"]["defects"])
        return {"row_class": "defect" if applicable else "normal", "model_applicable": applicable,
                "deterministic_pass": True, "deterministic_reasons": [], "flags": [], "test_only": True}

    def render(task, generated):
        assert task["locked_target"]["defects"], "normal row must not be rendered for a model"
        return json.dumps({"id": task["id"], "candidate": generated["text"]})

    monkeypatch.setattr(P, "evaluate_v7", evaluate)
    monkeypatch.setattr(P, "render_judge_v7", render)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "run"
    return root, prepare(root), tmp_path / "judge-v7"


def run(root, cfg, target, **kwargs):
    return I.run_judge(root, cfg, target, policy_version="v7", backend_factory=Fake, **kwargs)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite_shard(path, mutate):
    shard = read(path)
    mutate(shard)
    shard["payload_sha256"] = I._digest({key: val for key, val in shard.items() if key != "payload_sha256"})
    path.write_bytes(I._canonical(shard) + b"\n")


def test_mixed_and_normal_only_shards_count_all_rows_but_call_only_defects(prepared):
    root, cfg, target = prepared
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = run(root, cfg, target)
    assert result["status"] == "complete" and result["completed_tasks"] == 5
    assert result["new_shards"] == 3 and result["gpu_model_verified"] is False
    assert Fake.made == ["judge"] and Fake.closed == ["judge"]
    assert [json.loads(prompt)["id"] for _, prompts, _ in Fake.calls for prompt in prompts] == ["id-0", "id-4"]
    source = I.read_judge_source(root, cfg, target)  # schema-2 autodetection
    assert source["policy_version"] == "v7"
    assert source["all_task_ids"] == [f"id-{i}" for i in range(5)]
    assert source["model_task_ids"] == ["id-0", "id-4"]
    assert [row["id"] for row in source["rows"]] == ["id-0", "id-4"]
    assert len(source["deterministic_receipts"]) == 5
    assert all(row["candidate_finish_reason"] == "eos" for row in source["deterministic_receipts"])
    assert all(row["finish_reason"] == "json_closed" for row in source["rows"])
    middle = read(target / "judged/000001.json")
    assert middle["task_ids"] == ["id-2", "id-3"] and middle["model_task_ids"] == []
    assert middle["rows"] == [] and len(middle["deterministic_receipts"]) == 2
    assert middle["metrics"]["input_tokens"] == middle["metrics"]["output_tokens"] == 0
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_resume_normal_only_shard_does_not_construct_backend(prepared):
    root, cfg, target = prepared
    assert run(root, cfg, target, max_shards=1)["completed_tasks"] == 2
    first = (target / "judged/000000.json").read_bytes()
    Fake.reset()
    result = run(root, cfg, target, max_shards=1)
    assert result["completed_tasks"] == 4 and result["new_shards"] == 1
    assert not Fake.made and not Fake.calls and not Fake.closed
    assert (target / "judged/000000.json").read_bytes() == first
    result = run(root, cfg, target)
    assert result["completed_tasks"] == 5 and len(Fake.calls) == 1
    Fake.reset()
    assert run(root, cfg, target)["new_shards"] == 0
    assert not Fake.made


@pytest.mark.parametrize("defect_ids", [(), (4,)])
def test_unapproved_normal_prefix_refuses_without_backend_or_output(tmp_path, defect_ids):
    root, target = tmp_path / "run", tmp_path / "judge"
    cfg = prepare(root, defect_ids=defect_ids)
    with pytest.raises(I.StageRefused, match="unapproved_normal_prefix"):
        run(root, cfg, target)
    assert not Fake.made and not target.exists() and not (root / ".inference.lock").exists()


def test_zero_limit_does_not_construct_runtime_or_output(prepared):
    root, cfg, target = prepared
    result = run(root, cfg, target, max_shards=0)
    assert result["completed_tasks"] == 0 and result["runtime_identity_available"] is False
    assert not Fake.made and not target.exists()


@pytest.mark.parametrize("tokens", [64, 159, 161])
def test_v7_cap_is_exactly_160_before_loading_or_writing(tmp_path, tokens):
    root, target = tmp_path / "run", tmp_path / "judge"
    cfg = prepare(root, judge_tokens=tokens)
    with pytest.raises(I.StageRefused, match="exactly 160"):
        run(root, cfg, target)
    assert not Fake.made and not target.exists()


@pytest.mark.parametrize("mutation,reason", [
    (lambda s: s["model_task_ids"].append("id-1"), "judge_model_task_ids"),
    (lambda s: s["deterministic_receipts"][0].update(deterministic_pass=False), "judge_deterministic_receipts"),
    (lambda s: s["deterministic_receipts"].pop(), "judge_deterministic_receipts"),
    (lambda s: s["task_ids"].reverse(), "shard_inputs"),
    (lambda s: s["rows"].append({**s["rows"][0], "id": "id-1"}), "row_count"),
])
def test_rehashed_shard_cannot_change_receipts_model_subset_or_task_order(prepared, mutation, reason):
    root, cfg, target = prepared
    run(root, cfg, target, max_shards=1)
    rewrite_shard(target / "judged/000000.json", mutation)
    Fake.reset()
    with pytest.raises(I.StageRefused, match=reason):
        I.read_judge_source(root, cfg, target)
    assert not Fake.made


def test_explicit_v6_cannot_consume_v7_shards(prepared):
    root, cfg, target = prepared
    run(root, cfg, target)
    with pytest.raises(I.StageRefused):
        I.read_judge_source(root, cfg, target, policy_version="v6")


def test_v7_identity_pins_policy_and_stopper(prepared, monkeypatch):
    root, cfg, target = prepared
    run(root, cfg, target)
    source = I.read_judge_source(root, cfg, target)
    runtime = source["identity_basis"]["runtime_model"]
    assert runtime["judge_runtime"]["policy_version"] == "v7"
    runtime["judge_runtime"]["json_stop_implementation_sha256"] = "0" * 64
    with pytest.raises(I.StageRefused, match="judge_runtime"):
        J._check_runtime(runtime, cfg["judge"])
    original = P.policy_identity_v7()
    monkeypatch.setattr(P, "policy_identity_v7", lambda: {**original, "changed": True})
    with pytest.raises(I.StageRefused, match="judge_identity_changed"):
        I.read_judge_source(root, cfg, target)


def test_missing_parser_refuses_before_writer_side_effects(prepared, monkeypatch):
    root, cfg, target = prepared
    monkeypatch.delattr(Q, "parse_judge_v7")
    with pytest.raises(I.StageRefused, match="parser_missing"):
        run(root, cfg, target)
    assert not Fake.made and not target.exists()


def test_unknown_version_is_not_silently_downgraded(prepared):
    root, cfg, target = prepared
    with pytest.raises(I.StageRefused, match="judge_policy"):
        I.run_judge(root, cfg, target, policy_version="v8", backend_factory=Fake)
    assert not Fake.made and not target.exists()


def test_backend_cannot_relabel_a_different_task(prepared, monkeypatch):
    root, cfg, target = prepared
    original = Fake.generate_batch
    def wrong_task(self, prompts, system, deadline):
        rows = original(self, prompts, system, deadline)
        rows[0]["id"] = "id-from-another-task"
        return rows
    monkeypatch.setattr(Fake, "generate_batch", wrong_task)
    with pytest.raises(I.StageRefused, match="backend_task_id"):
        run(root, cfg, target)
    assert not (target / "judged").exists()
    assert Fake.closed == ["judge"]
