"""새 SFT 경로의 CPU 계약 검사. 모델·CUDA·가중치 다운로드를 호출하지 않는다."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from corpus.production import train


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def hashing(monkeypatch):
    # common 검증 자체는 root의 release 검사에서 다룬다. 이 파일은 학습 소비 계약만 분리해 검사한다.
    monkeypatch.setattr(train, "_canonical", _canonical)
    monkeypatch.setattr(train, "_sha", _hash)


def _config(root):
    return {
        "training": {
            "model_id": "fixture/model",
            "revision": "a" * 40,
            "model_kind": "image_text",
            "local_files_only": True,
            "quantization": "4bit",
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 0.0001,
            "max_length": 4096,
            "save_steps": 100,
            "seed": 20261007,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj"],
            "enable_thinking": False,
            "image_root": str(root),
        }
    }


def _row(uid, *, client="public", split="train", image=None, lineage=None):
    content = [{"type": "text", "text": "관측과 확인 사항을 설명하세요."}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    return {
        "id": uid,
        "kind": "source_qa" if image is None else "image_report",
        "split": split,
        "client": client,
        "lineage_group_id": lineage or uid,
        "image_path": image,
        "image_sha256": (
            hashlib.sha256(b"test-only-image-metadata").hexdigest() if image is not None else None
        ),
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "근거에 따른 설명입니다."}]},
        ],
        "source_ids": ["source-1"] if image is None else [],
    }


@pytest.fixture
def release(tmp_path, monkeypatch):
    source = tmp_path / "images"
    source.mkdir()
    (source / "image.png").write_bytes(b"test-only-image-metadata")
    rows = [
        _row("a", client="C1", image="image.png"),
        _row("b", client="C2", image="image.png"),
        _row("c", client="C3", image="image.png"),
        _row("public"),
        _row("val", split="val"),
    ]
    directory = tmp_path / "release"
    directory.mkdir()
    dataset = directory / "dataset.jsonl"
    dataset.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))
    receipt = {
        "schema_version": 1,
        "ready_for_training": True,
        "dataset_sha256": _hash(dataset),
        "run_identity_sha256": "c" * 64,
        "counts": {"total": len(rows)},
    }
    (directory / "RELEASE.json").write_bytes(_canonical(receipt))
    common = types.ModuleType("corpus.production.common")
    common.validate_release = lambda path: None
    common.iter_jsonl = lambda path: (
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
    )
    monkeypatch.setitem(sys.modules, "corpus.production.common", common)
    return directory, _config(source), rows, tmp_path / "training"


def test_mask_preserves_assistant_eos_and_masks_padding():
    # EOS와 pad가 같은 ID여도 attention_mask를 기준으로 실제 EOS만 감독한다.
    assert train.assistant_labels([8, 8, 1, 2, 3, 8], [1, 2], [0, 0, 1, 1, 1, 1], 4) == [
        -100,
        -100,
        -100,
        -100,
        3,
        8,
    ]
    assert train.assistant_labels([1, 2, 3, 8, 8], [1, 2], [1, 1, 1, 1, 0], 4) == [
        -100,
        -100,
        3,
        8,
        -100,
    ]


@pytest.mark.parametrize(
    "ids,prompt,mask,limit",
    [
        ([1, 2, 3], [1, 4], [1, 1, 1], 10),
        ([1, 2], [1, 2], [1, 1], 10),
        ([1, 2, 3], [1], [1, 0, 1], 10),
        ([1, 2, 3], [1], [1, 1, 1], 2),
        ([1, 2, 3], [1], [1, 1], 10),
        ([1, 2, 3], [1], [0, 0, 0], 10),
    ],
)
def test_bad_mask_contract_is_rejected(ids, prompt, mask, limit):
    with pytest.raises(train.TrainingRejected):
        train.assistant_labels(ids, prompt, mask, limit)


@pytest.mark.parametrize(
    "scope,expected",
    [
        ("central", ["a", "b", "c", "public"]),
        ("C1", ["a"]),
        ("C2", ["b"]),
        ("C3", ["c"]),
        ("public_text", ["public"]),
    ],
)
def test_scope_and_val_isolation(release, scope, expected):
    _directory, config, rows, _output = release
    got = train.select_rows(rows, scope, image_root=Path(config["training"]["image_root"]))
    assert [r["id"] for r in got] == expected


@pytest.mark.parametrize("scope,expected", [("C1", ["a"]), ("public_text", ["public"])])
def test_other_scope_images_are_not_accessed(tmp_path, scope, expected, monkeypatch):
    (tmp_path / "c1.png").write_bytes(b"local-client-image")
    rows = [
        _row("a", client="C1", image="c1.png"),
        _row("b", client="C2", image="unavailable-c2.png"),
        _row("c", client="C3", image="unavailable-c3.png"),
        _row("val", client="C1", image="unavailable-val.png", split="val"),
        _row("public"),
    ]
    rows[0]["image_sha256"] = _hash(tmp_path / "c1.png")
    if scope == "public_text":
        (tmp_path / "c1.png").unlink()
    original = train.image_file
    accessed = []

    def access(value, image_root):
        accessed.append(value)
        return original(value, image_root)

    monkeypatch.setattr(train, "image_file", access)
    selected = train.select_rows(rows, scope, image_root=tmp_path)
    assert [row["id"] for row in selected] == expected
    assert accessed == (["c1.png"] if scope == "C1" else [])


def test_selected_scope_requires_its_image_and_digest(tmp_path):
    row = _row("local", client="C1", image="c1.png")
    with pytest.raises(train.TrainingRejected, match="missing"):
        train.select_rows([row], "C1", image_root=tmp_path)
    (tmp_path / "c1.png").write_bytes(b"image")
    row["image_sha256"] = "a" * 64
    with pytest.raises(train.TrainingRejected, match="content changed"):
        train.select_rows([row], "C1", image_root=tmp_path)


@pytest.mark.parametrize("digest", [None, "", "a" * 63, "A" * 64])
def test_all_image_rows_require_pinned_sha_even_when_unselected(tmp_path, digest):
    row = _row("c2", client="C2", image="not-local.png")
    row["image_sha256"] = digest
    with pytest.raises(train.TrainingRejected, match="image digest"):
        train.select_rows([_row("public"), row], "public_text", image_root=tmp_path)


def test_unselected_scope_still_validates_global_lineage(tmp_path):
    rows = [
        _row("public"),
        _row("b", client="C2", image="absent.png", lineage="shared"),
        _row("val", client="C2", image="absent.png", split="val", lineage="shared"),
    ]
    with pytest.raises(train.TrainingRejected, match="lineage crosses"):
        train.select_rows(rows, "public_text", image_root=tmp_path)


@pytest.mark.parametrize(
    "mutation", ["eval", "public_image", "image_mismatch", "duplicate", "lineage"]
)
def test_bad_release_rows_rejected(release, mutation):
    _, config, rows, _ = release
    rows = copy.deepcopy(rows)
    if mutation == "eval":
        rows[0]["split"] = "eval"
    elif mutation == "public_image":
        rows[0]["client"] = "public"
    elif mutation == "image_mismatch":
        rows[0]["messages"][0]["content"][0]["image"] = "other.png"
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        rows[-1]["lineage_group_id"] = rows[0]["lineage_group_id"]
    with pytest.raises(train.TrainingRejected):
        train.select_rows(rows, "central", image_root=Path(config["training"]["image_root"]))


@pytest.mark.parametrize(
    "value", ["../x", "C:/secret.png", "https://example.org/img", "/root/image", "a\\x"]
)
def test_image_path_cannot_escape(tmp_path, value):
    with pytest.raises(train.TrainingRejected):
        train.image_file(value, tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("revision", "main"),
        ("local_files_only", False),
        ("epochs", 0),
        ("batch_size", 2),
        ("seed", True),
        ("enable_thinking", True),
        ("quantization", "8bit"),
        ("lora_dropout", 1),
        ("learning_rate", float("nan")),
    ],
)
def test_training_config_does_not_silently_fill_invalid_values(tmp_path, key, value):
    config = _config(tmp_path)
    config["training"][key] = value
    with pytest.raises(train.TrainingRejected):
        train._training_config(config)


def test_local_path_cannot_shadow_a_pinned_hub_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fixture" / "model").mkdir(parents=True)
    with pytest.raises(train.TrainingRejected, match="shadowed"):
        train._training_config(_config(tmp_path))


def test_dry_run_no_model_import_or_output(release, monkeypatch):
    directory, config, _, output = release
    import builtins

    old = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "peft", "PIL"}:
            pytest.fail(f"dry_run imported {name}")
        return old(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = train.train_from_release(directory, config, "central", output, dry_run=True)
    assert result["counts"] == {
        "selected_train_rows": 4,
        "excluded_val_rows": 1,
        "image_report": 3,
        "source_qa": 1,
    }
    assert result["gpu_executed"] is False and result["tokenization_checked"] is False
    assert not output.exists()


def test_text_only_model_rejects_image_scope(release):
    directory, config, _, output = release
    config["training"]["model_kind"] = "causal_lm"
    with pytest.raises(train.TrainingRejected, match="text-only"):
        train.train_from_release(directory, config, "central", output, dry_run=True)
    result = train.train_from_release(directory, config, "public_text", output, dry_run=True)
    assert result["counts"]["source_qa"] == 1


def test_dataset_changed_after_release_fails(release):
    directory, config, _, output = release
    with (directory / "dataset.jsonl").open("ab") as fh:
        fh.write(b"\n")
    with pytest.raises(train.TrainingRejected, match="checksum"):
        train.train_from_release(directory, config, "central", output, dry_run=True)


def _checkpoint(output, identity, step=3):
    path = output / f"checkpoint-{step}"
    path.mkdir(parents=True)
    for name in (
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_model.safetensors",
        "adapter_config.json",
    ):
        (path / name).write_bytes(b"fixture " + name.encode())
    (path / "trainer_state.json").write_bytes(_canonical({"global_step": step, "epoch": 1.0}))
    train.record_checkpoint(path, identity)
    return path


def test_resume_rechecks_state_and_ignores_unfinished_checkpoint(tmp_path):
    identity = {"release": "r", "config": "c", "scope": "C1"}
    path = _checkpoint(tmp_path, identity)
    (tmp_path / "checkpoint-4").mkdir()
    assert train.resume_checkpoint(tmp_path, identity) == path
    (path / "optimizer.pt").write_bytes(b"changed")
    with pytest.raises(train.TrainingRejected, match="content changed"):
        train.resume_checkpoint(tmp_path, identity)


def test_resume_scope_and_missing_rng_rejected(tmp_path):
    path = _checkpoint(tmp_path, {"scope": "C1"})
    with pytest.raises(train.TrainingRejected, match="identity mismatch"):
        train.resume_checkpoint(tmp_path, {"scope": "central"})
    (path / "rng_state.pth").unlink()
    with pytest.raises(train.TrainingRejected, match="incomplete checkpoint"):
        train.resume_checkpoint(tmp_path, {"scope": "C1"})


class _Tensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values

    def new_tensor(self, values):
        return _Tensor(values)


class _Processor:
    def __init__(self):
        self.calls = []
        self.pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        ids = [1, 2] if kwargs["add_generation_prompt"] else [1, 2, 9, 3]
        return {"input_ids": _Tensor([ids]), "attention_mask": _Tensor([[1] * len(ids)])}

    def save_pretrained(self, path):
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_collator_consumes_actual_answer_and_disables_truncation(tmp_path):
    proc = _Processor()
    cfg = _config(tmp_path)["training"]
    cfg["model_kind"] = "causal_lm"
    row = _row("text")
    output = train.AssistantCollator(proc, cfg, tmp_path)([row])
    assert output["labels"].tolist() == [[-100, -100, 9, 3]]
    assert proc.calls[0][0][-1]["content"] == "근거에 따른 설명입니다."
    assert all(
        call[1]["truncation"] is False and call[1]["padding"] is False for call in proc.calls
    )
    assert all(call[1]["enable_thinking"] is False for call in proc.calls)


def test_collator_does_not_zip_away_an_incomplete_prompt_mask(tmp_path):
    class InvalidMask(_Processor):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            if kwargs["add_generation_prompt"]:
                result["attention_mask"] = _Tensor([[1]])
            return result

    cfg = _config(tmp_path)["training"]
    cfg["model_kind"] = "causal_lm"
    with pytest.raises(train.TrainingRejected, match="generation attention mask"):
        train.AssistantCollator(InvalidMask(), cfg, tmp_path)([_row("text")])


def test_collator_image_branch_passes_local_pixels(tmp_path, monkeypatch):
    (tmp_path / "a.png").write_bytes(b"fixture")

    class Picture:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def convert(self, mode):
            assert mode == "RGB"
            return self

        def close(self):
            self.closed = True

    picture = Picture()
    fake = types.ModuleType("PIL")
    fake.Image = types.SimpleNamespace(open=lambda path: picture)
    monkeypatch.setitem(sys.modules, "PIL", fake)
    proc = _Processor()
    row = _row("img", image="a.png", client="C1")
    row["image_sha256"] = _hash(tmp_path / "a.png")
    output = train.AssistantCollator(proc, _config(tmp_path)["training"], tmp_path)([row])
    assert proc.calls[0][0][0]["content"][0]["image"] is picture
    assert picture.closed and output["labels"].tolist() == [[-100, -100, 9, 3]]


def test_collator_rejects_image_changed_after_initial_selection(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    path.write_bytes(b"test-only-image-metadata")
    row = _row("local", client="C1", image="image.png")
    train.select_rows([row], "C1", image_root=tmp_path)
    path.write_bytes(b"replaced image bytes")
    fake = types.ModuleType("PIL")
    fake.Image = types.SimpleNamespace(open=lambda _path: pytest.fail("unverified image opened"))
    monkeypatch.setitem(sys.modules, "PIL", fake)
    with pytest.raises(train.TrainingRejected, match="content changed"):
        train.AssistantCollator(_Processor(), _config(tmp_path)["training"], tmp_path)([row])


def test_writer_lock_released_for_resume(tmp_path):
    with (
        train._writer_lock(tmp_path),
        pytest.raises(train.TrainingRejected, match="another trainer"),
        train._writer_lock(tmp_path),
    ):
        pass
    with train._writer_lock(tmp_path):
        pass


def test_last_export_hash_protects_idempotent_completion(tmp_path):
    class Model:
        def save_pretrained(self, path, **kwargs):
            assert kwargs == {"safe_serialization": True}
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (path / "adapter_model.safetensors").write_bytes(b"safetensors fixture")

    identity = {"identity_sha256": "f" * 64, "training": _config(tmp_path)["training"]}
    result = {"identity_sha256": "f" * 64, "global_step": 1}
    path = train._save_last(Model(), _Processor(), tmp_path, identity, 1, {}, result)
    assert train.verify_artifact(path, "f" * 64)["files"]["adapter_model.safetensors"]
    (path / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(train.TrainingRejected, match="content has changed"):
        train.verify_artifact(path, "f" * 64)


def test_actual_trainer_wiring_with_cpu_fakes(release, monkeypatch):
    directory, config, _, output = release
    config["training"]["model_kind"] = "causal_lm"
    calls = {}
    proc = _Processor()

    class Model:
        config = types.SimpleNamespace(use_cache=True)

        def named_modules(self):
            return [("language.q_proj", object()), ("visual.q_proj", object())]

        def named_parameters(self):
            return [("language.lora", types.SimpleNamespace(requires_grad=True))]

        def save_pretrained(self, path, **kwargs):
            (Path(path) / "adapter_config.json").write_text("{}", encoding="utf-8")
            (Path(path) / "adapter_model.safetensors").write_bytes(b"fake adapter")

    class Loader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.setdefault("loads", []).append(kwargs)
            assert kwargs["local_files_only"] is True and kwargs["revision"] == "a" * 40
            return Model()

    class ProcLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return proc

    class Arguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            calls["args"] = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.state = types.SimpleNamespace(global_step=1, epoch=1.0)
            calls["row_ids"] = [r["id"] for r in kwargs["train_dataset"]]

        def train(self, resume_from_checkpoint=None):
            calls["resume"] = resume_from_checkpoint
            encoded = self.kwargs["data_collator"]([self.kwargs["train_dataset"][0]])
            assert encoded["labels"].tolist() == [[-100, -100, 9, 3]]
            path = output / "checkpoint-1"
            path.mkdir()
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (path / name).write_bytes(b"fixture")
            self.kwargs["model"].save_pretrained(path)
            (path / "trainer_state.json").write_bytes(_canonical({"global_step": 1, "epoch": 1.0}))
            control = types.SimpleNamespace(should_save=False)
            cb = self.kwargs["callbacks"][0]
            assert cb.on_epoch_end(self.kwargs["args"], self.state, control).should_save is True
            cb.on_save(self.kwargs["args"], self.state, control)
            return types.SimpleNamespace(metrics={"train_loss": 1.0})

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, is_bf16_supported=lambda: True, empty_cache=lambda: None
    )
    peft = types.ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: calls.setdefault("lora", kwargs)
    peft.get_peft_model = lambda model, cfg: model
    peft.prepare_model_for_kbit_training = lambda model, **kwargs: model
    transformers = types.ModuleType("transformers")
    for name, value in {
        "AutoModelForCausalLM": Loader,
        "AutoModelForImageTextToText": Loader,
        "AutoProcessor": ProcLoader,
        "AutoTokenizer": ProcLoader,
        "BitsAndBytesConfig": lambda **kwargs: kwargs,
        "Trainer": FakeTrainer,
        "TrainerCallback": object,
        "TrainingArguments": Arguments,
        "set_seed": lambda seed: None,
    }.items():
        setattr(transformers, name, value)
    for name, value in (("torch", torch), ("peft", peft), ("transformers", transformers)):
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(train.importlib.metadata, "version", lambda name: "fixture")
    result = train.train_from_release(directory, config, "public_text", output)
    assert result["status"] == "trained" and result["global_step"] == 1
    assert calls["row_ids"] == ["public"]
    assert calls["lora"]["target_modules"] == ["language.q_proj"]
    assert (
        calls["args"]["save_only_model"] is False
        and calls["args"]["load_best_model_at_end"] is False
    )
    assert calls["args"]["ignore_data_skip"] is False and calls["args"]["report_to"] == "none"
    assert (output / "last" / "adapter_model.safetensors").is_file()
    assert (
        train.train_from_release(directory, config, "public_text", output, resume=True)["status"]
        == "already_complete"
    )
    changed = copy.deepcopy(config)
    changed["training"]["learning_rate"] = 0.001
    with pytest.raises(train.TrainingRejected, match="different release/config/scope"):
        train.train_from_release(directory, changed, "public_text", output, resume=True)
