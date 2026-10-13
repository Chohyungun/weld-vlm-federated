"""CPU linkage over real v4 template shards and the real v7 policy/parser.

Generation, shard validation, judging identities, writes, reads and resumes are
real. Model execution and portable approval I/O are explicitly synthetic. These
tests do not establish GPU discrimination, a real sealed approval or v4 release.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from corpus.production import generation_policy as G
from corpus.production import infer as I
from corpus.production import judge_identity as J
from corpus.production import judge_policy as P
from corpus.production import judge_probe as K
from corpus.production import judge_stopping as S
from corpus.production import quality as Q
from data.label_map import load_label_map


LABEL = load_label_map().defect_types["porosity"]
OBSERVATION = (f"{LABEL.name_ko} 1개가 영상 가운데에 표시되어 있다. "
               "주석만으로는 결함의 모양과 크기를 알 수 없다.")


def write(path, value):
    path.write_bytes(I._canonical(value) + b"\n")


def read(path):
    return I._json(path.read_bytes())


def spec(family, tokens):
    return {"model_id": f"fixture/{family}", "revision": "a" * 40, "family": family,
            "quantization": "bf16", "local_files_only": True, "enable_thinking": False,
            "batch_size": 2, "max_input_tokens": 4096, "max_new_tokens": tokens,
            "wall_seconds": 1800, "seed": 17}


class CpuBackend:
    """Model-shaped CPU double; never loaded from transformers or a cache."""
    events = []

    def __init__(self, config, stage):
        self.stage = stage
        self.events.append(("load", stage))
        self.identity = {"backend": "fake" if stage == "generate" else "transformers",
                         "model_id": config["model_id"], "revision": config["revision"],
                         "family": config["family"], "model_type": config["family"],
                         "quantization": config["quantization"], "tokenizer_sha256": "c" * 64}
        defaults = {"eos_token_id": [2], "pad_token_id": 0}
        self.model = SimpleNamespace(generation_config=SimpleNamespace(
            to_dict=lambda: dict(defaults), eos_token_id=[2]))
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2,
                                         chat_template="CPU fixture only")
        self.last_metrics = {}
        self.json_object_stopping = False

    def enable_json_object_stopping(self):
        self.json_object_stopping = True
        self.identity.update(json_stop_protocol=S.JSON_STOP_PROTOCOL,
                             json_stop_implementation_sha256=I._file_sha(Path(S.__file__)))

    def generate_batch(self, prompts, system, deadline):
        self.events.append(("batch", self.stage, len(prompts)))
        if self.stage == "generate":
            text, finish = OBSERVATION, "eos"
        else:
            assert self.json_object_stopping
            text = ('{"has_limitation":true,"language_ok":true,'
                    '"no_unannotated_attributes":true,"no_unsupported_relation":true,'
                    '"reason":"표시된 주석만 설명한다."}')
            finish = "json_closed"
        return [{"text": text, "finish_reason": finish, "input_tokens": 80,
                 "output_tokens": 40} for _ in prompts]

    def close(self):
        self.events.append(("close", self.stage))


def prepare(tmp_path, *, normals=2, defects=2, normal_first=True):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()
    parent, root = repository / "parent", repository / "v4"
    parent.mkdir()
    tasks = []
    for i in range(normals + defects):
        defect = i >= normals
        # Separate clients allow both real round-robin prefix orders, without
        # modifying an already prepared RUN or moving completed shards.
        client = "C1" if (not defect) == normal_first else "C2"
        tasks.append({"id": f"case-{i}", "kind": "image_report", "split": "train",
            "client": client, "lineage_group_id": f"lineage-{i}",
            "source_ids": [f"annotation-{i}"], "image_path": f"never-opened/{i}.png",
            "image_sha256": I._digest(["synthetic-image", i]),
            "train_user_text": "이 영상의 주석을 설명하라.",
            "prompt": "부모 지시", "locked_target": {
                "defects": [{"type": LABEL.iso_code, "bbox_px": [1, 2, 3, 4]}] if defect else [],
                "defect_counts": {LABEL.iso_code: 1} if defect else {}, "clauses": [], "verdict": None},
            "evidence": {"material": "강재", "size_mm_available": False, "has_applicable_clause": False,
                "limitation": "모재 두께와 화소당 실치수 정보가 없어 치수 기준의 합부 판정은 내리지 않는다.",
                "defect_counts_by_name": {LABEL.name_ko: 1} if defect else {},
                "defect_positions": [{"name": LABEL.name_ko, "where": "가운데-가운데"}] if defect else []}})
    config = {"schema_version": 1, "recipe": "parent_fixture", "shard_size": 1,
              "generation": spec("qwen", 384), "judge": spec("phi", 160),
              "inputs": {"expected_image_train": len(tasks), "expected_image_val": 0}}
    raw = b"".join(I._canonical(t) + b"\n" for t in tasks)
    (parent / "tasks.jsonl").write_bytes(raw)
    basis = {"schema_version": 1, "config_sha256": I._digest(config),
             "tasks_sha256": hashlib.sha256(raw).hexdigest()}
    counts = {f"image_report|train|{c}": sum(t["client"] == c for t in tasks)
              for c in {t["client"] for t in tasks}}
    write(parent / "RUN.json", {"schema_version": 1, "identity_basis": basis,
        "identity_sha256": I._digest(basis), "config": config,
        "config_sha256": basis["config_sha256"], "tasks_sha256": basis["tasks_sha256"],
        "counts": {"tasks": len(tasks), "image_report": {"train": len(tasks), "val": 0},
                   "source_qa": {"train": 0, "val": 0}, "by_kind_split_client": counts}})
    G.prepare_regeneration(parent, root, config, repo_root=repository)
    config = read(root / "config.json")
    CpuBackend.events = []
    result = I.run_stage(root, "generate", config, backend_factory=CpuBackend)
    assert result["model_rows"] == defects and result["template_rows"] == normals
    if defects == 0:
        assert CpuBackend.events == []
    CpuBackend.events = []
    return root, config, tmp_path / "judge"


def install_approval(monkeypatch, root, config):
    """Stub only sealed-approval transport; do not bypass identity/runtime checks."""
    folder = root.parent / "synthetic-approval"
    folder.mkdir()
    raw = b"synthetic approval I/O boundary; not GPU evidence"
    (folder / "APPROVAL.json").write_bytes(raw)
    pin = hashlib.sha256(raw).hexdigest()
    model = CpuBackend(config["judge"], "judge")
    model.enable_json_object_stopping()
    runtime = J.judge_runtime_identity(model, config["judge"], policy_version="v7")
    CpuBackend.events = []
    protocol = "judge_v7_g150_b_v2"
    basis = {"protocol": protocol, "approval_scope": "final_blinded", "final_blinded_approval": True,
        "judge_config": config["judge"], "runtime_model": runtime,
        "common_implementation_sha256": J._source_pins(root)["implementation"],
        "probe_identity_sha256": ["1" * 64, "2" * 64], "probe_call_ids": ["first", "second"],
        "input_basis": {"protocol": protocol, "policy_identity": P.policy_identity_v7(),
            "cases_file_sha256": "f" * 64, "implementation_sha256": K._code_pins(),
            "registration": {"expected_protocol": protocol, "expected_approval_scope": "final_blinded",
                             "expected_manifest_sha256": "3" * 64}}}
    receipt = {"receipt_sha256": pin, "identity_sha256": I._digest(basis),
               "identity_basis": basis, "files_sha256": {"APPROVAL.json": pin}}

    def files(path):
        return {"APPROVAL.json": I._file_sha(Path(path) / "APPROVAL.json")}

    def verify(path, expected):
        if expected != pin or files(path) != receipt["files_sha256"]:
            raise I.StageRefused("probe_approval_pin", "synthetic approval bytes differ")
        return J._clone(receipt)

    def copy(source, destination, expected):
        value = verify(source, expected)
        Path(destination).mkdir()
        (Path(destination) / "APPROVAL.json").write_bytes((Path(source) / "APPROVAL.json").read_bytes())
        return value

    monkeypatch.setattr(K, "approval_files", files)
    monkeypatch.setattr(K, "read_probe_approval", verify)
    monkeypatch.setattr(K, "copy_probe_approval", copy)
    monkeypatch.setattr(I, "TransformersBackend", CpuBackend)
    return {"probe_approval": folder, "probe_approval_sha256": pin}, receipt


@pytest.mark.parametrize("normals,defects,normal_first", [(2, 2, True), (2, 2, False), (3, 0, True)])
def test_real_v4_writer_v7_identity_read_and_resume(tmp_path, monkeypatch, normals, defects, normal_first):
    root, cfg, target = prepare(tmp_path, normals=normals, defects=defects, normal_first=normal_first)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    args, _ = install_approval(monkeypatch, root, cfg)
    first = I.run_judge(root, cfg, target, policy_version="v7", max_shards=1, **args)
    assert first["exit_code"] == 4 and first["completed_tasks"] == 1
    if normal_first:
        assert CpuBackend.events == []
        assert read(root / "generated/000000.json")["model"] is None
    source = I.read_judge_source(root, cfg, target,
                                 expected_judge_identity_sha256=first["judge_identity_sha256"])
    assert len(source["deterministic_receipts"]) == 1
    assert source["deterministic_receipts"][0]["model_applicable"] is (not normal_first)
    saved = (target / "judged/000000.json").read_bytes()
    result = I.run_judge(root, cfg, target, policy_version="v7", **args)
    assert result["exit_code"] == 0 and result["completed_tasks"] == normals + defects
    source = I.read_judge_source(root, cfg, target,
                                 expected_judge_identity_sha256=result["judge_identity_sha256"])
    assert len(source["model_task_ids"]) == len(source["rows"]) == defects
    _, tasks = I._read_inputs(root, cfg)
    by_id = {t["id"]: t for t in tasks}
    for row in source["rows"]:
        parsed = Q.parse_judge_v7(by_id[row["id"]], row)
        assert parsed["schema"] == "judge_structured_v7"
        assert parsed["errors"] == parsed["rejected"] == []
    assert sum(r["model_applicable"] for r in source["deterministic_receipts"]) == defects
    assert all(r["deterministic_pass"] for r in source["deterministic_receipts"])
    assert all(r["candidate_finish_reason"] == "template" for r in source["deterministic_receipts"]
               if not r["model_applicable"])
    assert saved == (target / "judged/000000.json").read_bytes()
    if not defects:
        assert CpuBackend.events == []
    CpuBackend.events = []
    assert I.run_judge(root, cfg, target, policy_version="v7", **args)["new_shards"] == 0
    assert CpuBackend.events == []
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("defects", [0, 2])
def test_normal_prefix_still_requires_approval(tmp_path, defects):
    root, cfg, target = prepare(tmp_path, defects=defects)
    with pytest.raises(I.StageRefused, match="unapproved_normal_prefix"):
        I.run_judge(root, cfg, target, policy_version="v7", backend_factory=CpuBackend)
    assert CpuBackend.events == [] and not target.exists()


@pytest.mark.parametrize("defects", [0, 2])
def test_configured_same_family_still_refused(tmp_path, defects):
    root, cfg, target = prepare(tmp_path, defects=defects)
    judge = {**cfg["generation"], "max_new_tokens": 160}
    with pytest.raises(I.StageRefused, match="self_judging"):
        I.run_judge(root, cfg, target, policy_version="v7", judge_config=judge, backend_factory=CpuBackend)
    assert CpuBackend.events == [] and not target.exists()


def test_actual_family_check_uses_later_runtime(tmp_path):
    root, cfg, _ = prepare(tmp_path)
    run, tasks = I._read_inputs(root, cfg)
    shards = I._read_shards(root, "generate", run, tasks, cfg)
    assert shards[0]["model"] is None and shards[1]["model"]["model_type"] == "qwen"
    with pytest.raises(I.StageRefused, match="self_judging"):
        J._check_generator_separation({"model_type": "qwen2"}, shards)
    J._check_generator_separation({"model_type": "phi3"}, shards)


@pytest.mark.parametrize("mutation,reason", [
    ("template_model", "model_identity"), ("template_text", "generation_render"),
    ("template_tokens", "generation_render"), ("defect_null", "model_identity"),
    ("defect_runtime", "model_changed_between_shards"),
])
def test_resealed_generation_tampering_refused_before_judge_load(tmp_path, mutation, reason):
    root, cfg, target = prepare(tmp_path)
    index = 0 if mutation.startswith("template_") else (1 if mutation == "defect_null" else 3)
    path = root / f"generated/{index:06d}.json"
    shard = read(path)
    if mutation == "template_model":
        shard["model"] = read(root / "generated/000001.json")["model"]
    elif mutation == "template_text":
        shard["rows"][0]["text"] += " 변조."
    elif mutation == "template_tokens":
        shard["rows"][0]["input_tokens"] = 1
    elif mutation == "defect_null":
        shard["model"] = None
    else:
        shard["model"]["tokenizer_sha256"] = "d" * 64
    shard["payload_sha256"] = I._digest({k: v for k, v in shard.items() if k != "payload_sha256"})
    write(path, shard)
    with pytest.raises(I.StageRefused, match=reason):
        I.run_judge(root, cfg, target, policy_version="v7", backend_factory=CpuBackend)
    assert CpuBackend.events == [] and not target.exists()


def test_approved_runtime_is_still_checked_for_all_template_source(tmp_path, monkeypatch):
    root, cfg, target = prepare(tmp_path, defects=0)
    args, receipt = install_approval(monkeypatch, root, cfg)
    receipt["identity_basis"]["runtime_model"]["tokenizer_sha256"] = "invalid"
    with pytest.raises(I.StageRefused, match="tokenizer_identity"):
        I.run_judge(root, cfg, target, policy_version="v7", **args)
    assert CpuBackend.events == [] and not (target / "JUDGE.json").exists()


def test_real_v4_v6_identity_no_longer_requires_first_model(tmp_path):
    root, cfg, target = prepare(tmp_path)

    class FakeJudge(CpuBackend):
        def __init__(self, config, stage):
            super().__init__(config, stage)
            self.identity["backend"] = "fake"

        def generate_batch(self, prompts, system, deadline):
            return [{"text": "CPU-only unparsed v6 output", "finish_reason": "eos",
                     "input_tokens": 20, "output_tokens": 10} for _ in prompts]

    result = I.run_judge(root, cfg, target, policy_version="v6", backend_factory=FakeJudge)
    assert result["completed_tasks"] == 4
    source = I.read_judge_source(root, cfg, target, policy_version="v6")
    assert len(source["rows"]) == 4
    assert I.run_judge(root, cfg, target, policy_version="v6", backend_factory=FakeJudge)["new_shards"] == 0
