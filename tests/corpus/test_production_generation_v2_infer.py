"""Independent CPU contracts for generation-v2 persistence and lazy inference.

The backend is synthetic; preparation, rendering, writing, reading and resume
validation are real. No image, model, judge or sealed case is loaded.
"""

import hashlib
import json
import pytest

from corpus.production import generation_policy as G
from corpus.production import infer as I
from data.label_map import load_label_map


LABEL = load_label_map().defect_types["porosity"]
NORMAL_TEXT = "표시된 결함이 없다. 결함이 표시되지 않았다는 것이 합격이나 안전을 뜻하지는 않는다."
LIMITATION = "모재 두께와 화소당 실치수 정보가 없어 치수 기준의 합부 판정은 내리지 않는다."
OBSERVATION = (f"{LABEL.name_ko} 1개가 영상 가운데에 표시되어 있다. "
               "주석만으로는 결함의 모양과 크기를 알 수 없다.")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_bytes(canonical(value) + b"\n")


def model_spec(family="qwen"):
    return {"model_id": f"fixture/{family}", "revision": "a" * 40, "family": family,
            "quantization": "bf16", "local_files_only": True, "enable_thinking": False,
            "batch_size": 2, "max_input_tokens": 4096, "max_new_tokens": 384,
            "wall_seconds": 1800, "seed": 17}


def task(index, defect):
    return {"id": f"case-{index:03d}", "kind": "image_report", "split": "train", "client": "C1",
            "lineage_group_id": f"lineage-{index}", "source_ids": [f"annotation-{index}"],
            "image_path": f"never-opened/{index}.png", "image_sha256": digest(["image", index]),
            "train_user_text": "이 영상의 주석을 설명하라.", "prompt": "부모의 이전 설명 지시",
            "locked_target": {"defects": ([{"type": LABEL.iso_code, "bbox_px": [1, 2, 3, 4]}]
                                           if defect else []),
                              "defect_counts": {LABEL.iso_code: 1} if defect else {},
                              "clauses": [], "verdict": None},
            "evidence": {"material": "강재", "defect_counts_by_name": {LABEL.name_ko: 1} if defect else {},
                         "defect_positions": ([{"name": LABEL.name_ko, "where": "가운데-가운데"}]
                                              if defect else []),
                         "has_applicable_clause": False, "size_mm_available": False,
                         "limitation": LIMITATION}}


def make_parent(root, normal_count, defect_count, shard_size):
    root.mkdir(parents=True)
    (root / ".git").mkdir()
    parent = root / "parent"
    parent.mkdir()
    tasks = [task(i, i >= normal_count) for i in range(normal_count + defect_count)]
    config = {"schema_version": 1, "recipe": "parent_fixture", "shard_size": shard_size,
              "generation": model_spec(), "judge": model_spec("phi"),
              "inputs": {"expected_image_train": len(tasks), "expected_image_val": 0}}
    raw = b"".join(canonical(row) + b"\n" for row in tasks)
    (parent / "tasks.jsonl").write_bytes(raw)
    basis = {"schema_version": 1, "config_sha256": digest(config),
             "tasks_sha256": hashlib.sha256(raw).hexdigest(),
             "pairs_sha256": "a" * 64, "pairs_snapshot_digest": "b" * 64,
             "manifest_snapshot_digest": "c" * 64, "source_bank_sha256": "d" * 64,
             "prepare_code_sha256": "e" * 64}
    run = {"schema_version": 1, "identity_basis": basis, "identity_sha256": digest(basis),
           "config": config, "config_sha256": basis["config_sha256"],
           "tasks_sha256": basis["tasks_sha256"],
           "counts": {"tasks": len(tasks), "image_report": {"train": len(tasks), "val": 0},
                      "source_qa": {"train": 0, "val": 0},
                      "by_kind_split_client": {"image_report|train|C1": len(tasks)}}}
    write(parent / "RUN.json", run)
    write(parent / "config.json", config)
    return parent, tasks, config


def make_bundle(tmp_path, normal_count=2, defect_count=2, shard_size=2):
    root = tmp_path / "repository"
    parent, _, config = make_parent(root, normal_count, defect_count, shard_size)
    before = {p.name: p.read_bytes() for p in parent.iterdir()}
    destination = root / "generation-v2"
    G.prepare_regeneration(parent, destination, config, repo_root=root)
    assert before == {p.name: p.read_bytes() for p in parent.iterdir()}
    tasks = [json.loads(line) for line in (destination / "tasks.jsonl").read_bytes().splitlines()]
    return destination, read(destination / "config.json"), tasks


def fake_factory(raw_text=OBSERVATION, *, tokenizer="c" * 64, finish="eos", output_tokens=23):
    events = {"loads": [], "batches": [], "closes": []}

    class Backend:
        def __init__(self, spec, stage):
            events["loads"].append(stage)
            self.identity = {"backend": "fake", "model_id": spec["model_id"],
                             "revision": spec["revision"], "family": spec["family"],
                             "model_type": spec["family"], "quantization": spec["quantization"],
                             "tokenizer_sha256": tokenizer}
            self.last_metrics = {"peak_allocated_bytes": 101, "peak_reserved_bytes": 202}

        def generate_batch(self, prompts, system, deadline):
            events["batches"].append(list(prompts))
            return [{"text": raw_text, "finish_reason": finish, "input_tokens": 71,
                     "output_tokens": output_tokens} for _ in prompts]

        def close(self):
            events["closes"].append(True)

    return Backend, events


def rewrite_shard(path, change):
    shard = read(path)
    change(shard)
    shard["payload_sha256"] = digest({k: v for k, v in shard.items() if k != "payload_sha256"})
    write(path, shard)


def reseal_tasks(root, change):
    tasks = [json.loads(line) for line in (root / "tasks.jsonl").read_bytes().splitlines()]
    change(tasks)
    raw = b"".join(canonical(row) + b"\n" for row in tasks)
    (root / "tasks.jsonl").write_bytes(raw)
    run = read(root / "RUN.json")
    run["tasks_sha256"] = hashlib.sha256(raw).hexdigest()
    run["identity_basis"]["tasks_sha256"] = run["tasks_sha256"]
    run["identity_sha256"] = digest(run["identity_basis"])
    write(root / "RUN.json", run)


def test_template_only_run_never_constructs_model_and_has_null_provenance(tmp_path):
    root, config, tasks = make_bundle(tmp_path, normal_count=5, defect_count=0)
    backend, events = fake_factory()
    result = I.run_stage(root, "generate", config, backend_factory=backend)
    assert result["exit_code"] == 0
    assert result["template_rows"] == 5 and result["model_rows"] == 0
    assert events == {"loads": [], "batches": [], "closes": []}
    rows = I.read_stage_rows(root, "generate", config)
    assert [row["id"] for row in rows] == [t["id"] for t in tasks]
    for row in rows:
        assert row["text"] == NORMAL_TEXT
        assert row["raw_text"] is None and row["generator"] == row["_backend"] == "template"
        assert row["finish_reason"] == "template"
        assert type(row["input_tokens"]) is int and row["input_tokens"] == 0
        assert type(row["output_tokens"]) is int and row["output_tokens"] == 0
        assert row["format_normalized"] is False and row["language_ok"] is True
        assert row["observation_problems"] == []
    for path in (root / "generated").glob("*.json"):
        shard = read(path)
        assert shard["model"] is None
        assert shard["metrics"]["input_tokens"] == shard["metrics"]["output_tokens"] == 0
        assert shard["metrics"]["model_load_seconds"] == 0


def test_mixed_shards_filter_normals_before_model_and_restore_manifest_order(tmp_path):
    root, config, tasks = make_bundle(tmp_path, normal_count=3, defect_count=3, shard_size=6)
    backend, events = fake_factory()
    I.run_stage(root, "generate", config, backend_factory=backend)
    assert events["loads"] == ["generate"] and events["closes"] == [True]
    expected_prompts = [t["prompt"] for t in tasks if t["locked_target"]["defects"]]
    assert events["batches"] == [expected_prompts[:2], expected_prompts[2:]]
    rows = I.read_stage_rows(root, "generate", config)
    assert [row["id"] for row in rows] == [t["id"] for t in tasks]
    assert [row["generator"] for row in rows] == ["template", "model"] * 3
    for row in rows[1::2]:
        assert row["raw_text"] == OBSERVATION
        assert row["text"] == OBSERVATION + " " + LIMITATION
        assert row["finish_reason"] == "eos" and row["_backend"] == "fake"
        assert (row["input_tokens"], row["output_tokens"]) == (71, 23)
        assert row["observation_problems"] == [] and row["format_normalized"] is False
    shard = read(root / "generated/000000.json")
    assert shard["metrics"]["input_tokens"] == 71 * 3
    assert shard["metrics"]["output_tokens"] == 23 * 3


def test_resume_after_normal_prefix_pins_first_nonnull_runtime(tmp_path):
    root, config, _ = make_bundle(tmp_path, normal_count=2, defect_count=2, shard_size=1)
    backend, events = fake_factory()
    assert I.run_stage(root, "generate", config, max_shards=1, backend_factory=backend)["exit_code"] == 4
    first = (root / "generated/000000.json").read_bytes()
    assert events["loads"] == []
    assert I.run_stage(root, "generate", config, max_shards=1, backend_factory=backend)["exit_code"] == 4
    assert read(root / "generated/000000.json")["model"] is None
    assert read(root / "generated/000001.json")["model"]["tokenizer_sha256"] == "c" * 64
    wrong, wrong_events = fake_factory(tokenizer="d" * 64)
    with pytest.raises(I.StageRefused, match="runtime_changed"):
        I.run_stage(root, "generate", config, backend_factory=wrong)
    assert wrong_events["batches"] == []
    assert (root / "generated/000000.json").read_bytes() == first
    assert read(root / "generated/000002.json")["model"] is None
    assert not (root / "generated/000003.json").exists()
    assert I.run_stage(root, "generate", config, backend_factory=backend)["exit_code"] == 0
    complete = {p.name: p.read_bytes() for p in (root / "generated").glob("*.json")}
    assert I.run_stage(root, "generate", config, backend_factory=wrong)["new_shards"] == 0
    assert complete == {p.name: p.read_bytes() for p in (root / "generated").glob("*.json")}


def test_template_shard_after_model_is_still_model_null(tmp_path):
    root, config, _ = make_bundle(tmp_path, normal_count=2, defect_count=1, shard_size=1)
    backend, events = fake_factory()
    I.run_stage(root, "generate", config, backend_factory=backend)
    assert events["loads"] == ["generate"]
    assert [read(root / f"generated/{i:06d}.json")["model"] is None for i in range(3)] == [True, False, True]
    last = read(root / "generated/000002.json")
    assert last["metrics"]["peak_allocated_bytes"] == 0
    assert last["metrics"]["peak_reserved_bytes"] == 0
    assert len(I.read_stage_rows(root, "generate", config)) == 3


@pytest.mark.parametrize("raw_text,normalized,language_ok", [
    ("  " + OBSERVATION.replace(". ", ".\r\n\n") + "  ", True, True),
    (OBSERVATION + " 漢。", False, False),
    ("", False, True),
    ("형식 오류", False, True),
])
def test_invalid_observation_is_durable_not_retried_and_raw_is_preserved(tmp_path, raw_text, normalized, language_ok):
    root, config, _ = make_bundle(tmp_path, normal_count=0, defect_count=1)
    backend, events = fake_factory(raw_text=raw_text, finish="length")
    assert I.run_stage(root, "generate", config, backend_factory=backend)["exit_code"] == 0
    row = I.read_stage_rows(root, "generate", config)[0]
    assert row["raw_text"] == raw_text and row["finish_reason"] == "length"
    assert row["format_normalized"] is normalized and row["language_ok"] is language_ok
    assert row["text"].endswith(" " + LIMITATION)
    if normalized:
        assert row["text"] == OBSERVATION + " " + LIMITATION
    else:
        assert row["observation_problems"]
    if not language_ok:
        assert "obs_language" in row["observation_problems"] and "漢。" in row["text"]
    before = (root / "generated/000000.json").read_bytes()
    assert I.run_stage(root, "generate", config, backend_factory=backend)["new_shards"] == 0
    assert len(events["batches"]) == 1
    assert (root / "generated/000000.json").read_bytes() == before


@pytest.mark.parametrize("field,value", [
    ("text", "바꾼 완성 문장"), ("raw_text", "바꾼 관찰 문장"),
    ("format_normalized", True), ("language_ok", False), ("observation_problems", ["obs_empty"]),
    ("generator", "template"), ("input_tokens", True), ("output_tokens", False),
])
def test_rehashed_defect_row_tampering_is_rejected(tmp_path, field, value):
    root, config, _ = make_bundle(tmp_path, normal_count=0, defect_count=1)
    backend, _ = fake_factory()
    I.run_stage(root, "generate", config, backend_factory=backend)
    path = root / "generated/000000.json"
    rewrite_shard(path, lambda shard: shard["rows"][0].__setitem__(field, value))
    with pytest.raises(I.StageRefused, match="generation_render|token_count"):
        I.read_stage_rows(root, "generate", config)


@pytest.mark.parametrize("field,value", [
    ("raw_text", ""), ("generator", "model"), ("input_tokens", False),
    ("output_tokens", True), ("finish_reason", "eos"), ("text", NORMAL_TEXT + " "),
])
def test_rehashed_template_row_tampering_is_rejected(tmp_path, field, value):
    root, config, _ = make_bundle(tmp_path, normal_count=1, defect_count=0)
    backend, _ = fake_factory()
    I.run_stage(root, "generate", config, backend_factory=backend)
    rewrite_shard(root / "generated/000000.json", lambda shard: shard["rows"][0].__setitem__(field, value))
    with pytest.raises(I.StageRefused, match="generation_render"):
        I.read_stage_rows(root, "generate", config)


@pytest.mark.parametrize("defect", [False, True])
def test_shard_model_provenance_cannot_be_swapped_even_after_rehash(tmp_path, defect):
    root, config, _ = make_bundle(tmp_path, normal_count=int(not defect), defect_count=int(defect))
    backend, _ = fake_factory()
    I.run_stage(root, "generate", config, backend_factory=backend)
    value = None if defect else {"backend": "fake"}
    rewrite_shard(root / "generated/000000.json", lambda shard: shard.__setitem__("model", value))
    with pytest.raises(I.StageRefused, match="model_identity"):
        I.read_stage_rows(root, "generate", config)


@pytest.mark.parametrize("defect", [False, True])
def test_prompt_tampering_is_rejected_even_after_resealing_input(tmp_path, defect):
    root, config, _ = make_bundle(tmp_path, normal_count=int(not defect), defect_count=int(defect))
    reseal_tasks(root, lambda rows: rows[0].__setitem__("prompt", "변경된 지시"))
    backend, events = fake_factory()
    with pytest.raises(I.StageRefused, match="generation_task"):
        I.run_stage(root, "generate", config, backend_factory=backend)
    assert events["loads"] == []


@pytest.mark.parametrize("operation", ["read", "run"])
def test_v2_legacy_judge_entry_is_refused(tmp_path, operation):
    root, config, _ = make_bundle(tmp_path, normal_count=1, defect_count=0)
    backend, events = fake_factory()
    with pytest.raises(I.StageRefused, match="separate_judge_required"):
        if operation == "read":
            I.read_stage_rows(root, "judge", config)
        else:
            I.run_stage(root, "judge", config, backend_factory=backend)
    assert events["loads"] == []


def test_legacy_v3_row_format_and_reader_are_unchanged(tmp_path):
    root = tmp_path / "legacy"
    parent, tasks, config = make_parent(root, normal_count=1, defect_count=1, shard_size=2)
    config["recipe"] = "image_report_regeneration_v3"
    run = read(parent / "RUN.json")
    run["config"] = config
    run["config_sha256"] = digest(config)
    run["identity_basis"]["config_sha256"] = run["config_sha256"]
    run["identity_sha256"] = digest(run["identity_basis"])
    write(parent / "RUN.json", run)
    backend, events = fake_factory()
    assert I.run_stage(parent, "generate", config, backend_factory=backend)["exit_code"] == 0
    rows = I.read_stage_rows(parent, "generate", config)
    assert [row["id"] for row in rows] == [t["id"] for t in tasks]
    assert events["batches"] == [[t["prompt"] for t in tasks]]
    assert all(row["text"] == OBSERVATION and row["_backend"] == "fake" for row in rows)
    assert all("generator" not in row and "raw_text" not in row for row in rows)


def test_zero_shards_never_loads_or_writes(tmp_path):
    root, config, _ = make_bundle(tmp_path, normal_count=0, defect_count=1)
    backend, events = fake_factory()
    result = I.run_stage(root, "generate", config, max_shards=0, backend_factory=backend)
    assert result["exit_code"] == 4 and result["completed_tasks"] == 0
    assert not (root / "generated").exists() and events["loads"] == []


def test_bool_model_token_count_cannot_be_committed(tmp_path):
    root, config, _ = make_bundle(tmp_path, normal_count=0, defect_count=1)
    backend, _ = fake_factory(output_tokens=True)
    with pytest.raises(I.StageRefused, match="token_count"):
        I.run_stage(root, "generate", config, backend_factory=backend)
    assert not (root / "generated/000000.json").exists()


@pytest.mark.parametrize("failure,expected_rc", [("oom", 5), ("time", 3)])
def test_model_failure_preserves_completed_template_prefix_without_retry(tmp_path, failure, expected_rc):
    root, config, _ = make_bundle(tmp_path, normal_count=1, defect_count=1, shard_size=1)
    backend, events = fake_factory()

    class Failure(backend):
        def generate_batch(self, prompts, system, deadline):
            events["batches"].append(list(prompts))
            if failure == "oom":
                raise RuntimeError("CUDA out of memory")
            raise I.StageTimeLimit()

    result = I.run_stage(root, "generate", config, backend_factory=Failure)
    assert result["exit_code"] == expected_rc and result["completed_tasks"] == 1
    assert events["loads"] == ["generate"] and len(events["batches"]) == 1
    assert events["closes"] == [True]
    assert read(root / "generated/000000.json")["rows"][0]["text"] == NORMAL_TEXT
    assert not (root / "generated/000001.json").exists()
    assert not (root / ".inference.lock").exists()
    if failure == "oom":
        assert result["retry_policy"] == "no automatic retry"


def test_backend_cannot_reassign_model_output_to_another_task(tmp_path):
    root, config, _ = make_bundle(tmp_path, normal_count=0, defect_count=1)
    backend, _ = fake_factory()

    class WrongTask(backend):
        def generate_batch(self, prompts, system, deadline):
            rows = super().generate_batch(prompts, system, deadline)
            rows[0]["id"] = "another-task"
            return rows

    with pytest.raises(I.StageRefused, match="backend_task_id"):
        I.run_stage(root, "generate", config, backend_factory=WrongTask)
    assert not (root / "generated/000000.json").exists()
