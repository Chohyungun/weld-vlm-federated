"""검수된 설명형 릴리스의 SFT와 표준 PEFT 저장.

dry_run은 표준 라이브러리와 릴리스 검사만 사용한다. 실제 실행에서만
Transformers/PyTorch/PEFT를 import한다. 기존 JSON-only 연합 실험과 별도 레시피다.
API 근거: https://huggingface.co/docs/transformers/main_classes/trainer
https://huggingface.co/docs/transformers/chat_templating_multimodal
https://huggingface.co/docs/peft/developer_guides/quantization
https://huggingface.co/docs/peft/developer_guides/checkpoint
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPES = {"central", "C1", "C2", "C3", "public_text"}
TRAINING_CONTRACT_VERSION = 1


class TrainingRejected(ValueError):
    """모델을 적재하거나 기존 산출물에 덮어쓰기 전에 계약 위반을 알린다."""


def _canonical(value: Any) -> bytes:
    from .common import canonical_bytes

    return canonical_bytes(value)


def _sha(path: Path) -> str:
    from .common import sha256_file

    return sha256_file(path)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TrainingRejected(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("xb") as stream:
        stream.write(_canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _image_reference(value: Any) -> None:
    """다른 클라이언트의 파일에 접근하지 않고 참조의 문법만 검사한다."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise TrainingRejected("image_path must be a nonempty POSIX relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise TrainingRejected(f"unsafe image_path: {value}")


def image_file(value: Any, image_root: Path) -> Path:
    """선택한 train 이미지의 존재와 심볼릭 경로 탈출을 검사한다."""
    _image_reference(value)
    path = (image_root / value).resolve()
    if not _inside(path, image_root) or not path.is_file():
        raise TrainingRejected(f"image is missing or outside image_root: {value}")
    return path


def _validate_row(row: Any) -> None:
    if not isinstance(row, dict):
        raise TrainingRejected("dataset row must be an object")
    for key in ("id", "lineage_group_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise TrainingRejected(f"missing row {key}")
    if row.get("split") not in {"train", "val"}:
        raise TrainingRejected(f"eval/unknown split must not enter SFT: {row['id']}")
    if row.get("kind") not in {"image_report", "source_qa"}:
        raise TrainingRejected(f"unknown row kind: {row['id']}")
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or [m.get("role") for m in messages if isinstance(m, dict)] != ["user", "assistant"]
    ):
        raise TrainingRejected(f"exactly one user/assistant turn required: {row['id']}")
    image_parts = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list) or not content:
            raise TrainingRejected(f"content blocks missing: {row['id']}")
        texts = []
        for block in content:
            if not isinstance(block, dict):
                raise TrainingRejected("content block must be an object")
            if block.get("type") == "image" and message["role"] == "user":
                if set(block) != {"type", "image"}:
                    raise TrainingRejected("only a local image reference is permitted")
                image_parts.append(block["image"])
            elif block.get("type") == "text":
                if (
                    set(block) != {"type", "text"}
                    or not isinstance(block["text"], str)
                    or not block["text"].strip()
                ):
                    raise TrainingRejected("text content must be nonempty")
                texts.append(block["text"])
            else:
                raise TrainingRejected("unsupported message content")
        if not texts:
            raise TrainingRejected("both turns require text content")
    if row["kind"] == "source_qa":
        if row.get("client") != "public" or row.get("image_path") is not None or image_parts:
            raise TrainingRejected("source_qa must be public text without image-derived messages")
        if not isinstance(row.get("source_ids"), list) or not row["source_ids"]:
            raise TrainingRejected("source_qa requires source provenance")
    else:
        if row.get("client") not in {"C1", "C2", "C3"}:
            raise TrainingRejected("image_report must retain its original client")
        if image_parts != [row.get("image_path")]:
            raise TrainingRejected("message image and row image_path must be identical")
        _image_reference(row["image_path"])
        digest = row.get("image_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise TrainingRejected(f"invalid image digest: {row['id']}")


def select_rows(rows: Sequence[dict], scope: str, *, image_root: Path) -> list[dict]:
    """전역 메타데이터를 검사하되 이미지 실물은 선택한 train 범위에서만 읽는다."""
    if scope not in SCOPES:
        raise TrainingRejected(f"unknown training scope: {scope}")
    ids: set[str] = set()
    lineages: dict[str, str] = {}
    selected = []
    for row in rows:
        _validate_row(row)
        if row["id"] in ids:
            raise TrainingRejected(f"duplicate row id: {row['id']}")
        ids.add(row["id"])
        prior = lineages.setdefault(row["lineage_group_id"], row["split"])
        if prior != row["split"]:
            raise TrainingRejected(f"lineage crosses train/val: {row['lineage_group_id']}")
        if row["split"] != "train":
            continue
        if (
            scope == "central"
            or (scope == "public_text" and row["kind"] == "source_qa")
            or (scope in {"C1", "C2", "C3"} and row["client"] == scope)
        ):
            selected.append(row)
    if not selected:
        raise TrainingRejected(f"no train rows for scope={scope}")
    for row in selected:
        if row["kind"] == "image_report":
            path = image_file(row["image_path"], image_root)
            if _sha(path) != row["image_sha256"]:
                raise TrainingRejected(f"image content changed: {row['id']}")
    return selected


def assistant_labels(
    input_ids: Sequence[int],
    prompt_ids: Sequence[int],
    attention_mask: Sequence[int],
    max_length: int,
) -> list[int]:
    """토크나이저 독립 감독 계약. 절단·접두 불일치·빈 assistant·내부 패딩을 모두 거부한다."""
    ids, prompt, mask = list(input_ids), list(prompt_ids), list(attention_mask)
    if (
        not ids
        or len(ids) != len(mask)
        or not prompt
        or any(type(v) is not int for v in ids + prompt)
        or any(type(v) is not int or v not in (0, 1) for v in mask)
    ):
        raise TrainingRejected("invalid token IDs or attention mask")
    active = [i for i, value in enumerate(mask) if value]
    if not active or active != list(range(active[0], active[-1] + 1)):
        raise TrainingRejected("attention mask contains internal padding")
    sequence = [ids[i] for i in active]
    if len(sequence) > max_length:
        raise TrainingRejected(
            f"sequence has {len(sequence)} tokens > max_length={max_length}; truncation is forbidden"
        )
    if sequence[: len(prompt)] != prompt:
        raise TrainingRejected(
            "generation prompt is not the exact prefix of the assistant training sequence"
        )
    if len(sequence) <= len(prompt):
        raise TrainingRejected("no supervised assistant tokens")
    labels = [-100] * len(ids)
    for i in active[len(prompt) :]:
        labels[i] = ids[i]
    return labels


def _training_config(config: Mapping[str, Any]) -> dict:
    raw = config.get("training")
    if not isinstance(raw, dict):
        raise TrainingRejected("config.training is required")
    cfg = copy.deepcopy(raw)
    required = (
        "model_id",
        "revision",
        "model_kind",
        "local_files_only",
        "quantization",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_length",
        "save_steps",
        "seed",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "enable_thinking",
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise TrainingRejected(f"missing training configuration: {missing}")
    if not isinstance(cfg["model_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", cfg["model_id"]
    ):
        raise TrainingRejected(
            "model_id must be a Hub repository ID resolved from the pinned local cache"
        )
    if Path(cfg["model_id"]).exists():
        raise TrainingRejected(
            "model_id is shadowed by a local path; use the commit-pinned Hub cache"
        )
    if not isinstance(cfg["revision"], str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", cfg["revision"]
    ):
        raise TrainingRejected("revision must pin a full 40-hex model commit")
    if cfg["model_kind"] not in {"image_text", "causal_lm"}:
        raise TrainingRejected("model_kind must be image_text or causal_lm")
    if cfg["local_files_only"] is not True or cfg["enable_thinking"] is not False:
        raise TrainingRejected("local_files_only=true and enable_thinking=false are required")
    if cfg["quantization"] not in {"4bit", "bf16"}:
        raise TrainingRejected("quantization must be 4bit or bf16")
    for key in (
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "save_steps",
        "lora_r",
        "lora_alpha",
    ):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise TrainingRejected(f"{key} must be a positive integer")
    if cfg["batch_size"] != 1:
        raise TrainingRejected(
            "this mixed text/image recipe supports microbatch=1; use gradient accumulation"
        )
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise TrainingRejected("seed must be a nonnegative integer")
    for key in ("learning_rate", "lora_dropout"):
        if (
            isinstance(cfg[key], bool)
            or not isinstance(cfg[key], (int, float))
            or not math.isfinite(cfg[key])
        ):
            raise TrainingRejected(f"invalid numeric configuration: {key}")
    if cfg["learning_rate"] <= 0 or not 0 <= cfg["lora_dropout"] < 1:
        raise TrainingRejected("invalid learning_rate or lora_dropout")
    if (
        not isinstance(cfg["target_modules"], list)
        or not cfg["target_modules"]
        or any(not isinstance(v, str) or not v for v in cfg["target_modules"])
    ):
        raise TrainingRejected("target_modules must explicitly name LoRA module suffixes")
    if type(cfg.get("export_merged_bf16", False)) is not bool:
        raise TrainingRejected("export_merged_bf16 must be a boolean")
    return cfg


def _prepare(release_dir: Path, config: Mapping[str, Any], scope: str, output_dir: Path) -> tuple:
    from .common import iter_jsonl, validate_release

    validate_release(release_dir)
    receipt_path = release_dir / "RELEASE.json"
    receipt = _read_json(receipt_path)
    dataset = release_dir / "dataset.jsonl"
    if receipt.get("ready_for_training") is not True or receipt.get("schema_version") != 1:
        raise TrainingRejected("release is not approved for training")
    if receipt.get("dataset_sha256") != _sha(dataset):
        raise TrainingRejected("release dataset checksum mismatch")
    cfg = _training_config(config)
    root = Path(cfg.get("image_root", REPO_ROOT)).resolve()
    if not root.is_dir():
        raise TrainingRejected("image_root is not a directory")
    rows = list(iter_jsonl(dataset))
    selected = select_rows(rows, scope, image_root=root)
    if cfg["model_kind"] == "causal_lm" and any(r["kind"] == "image_report" for r in selected):
        raise TrainingRejected("a text-only model cannot consume image_report rows")
    if (
        _inside(output_dir, release_dir)
        or _inside(release_dir, output_dir)
        or _inside(output_dir, REPO_ROOT / "data")
        or output_dir.resolve() == REPO_ROOT
    ):
        raise TrainingRejected(
            "training output must be separate from release, source data and repository root"
        )
    counts = {
        "selected_train_rows": len(selected),
        "excluded_val_rows": sum(r["split"] == "val" for r in rows),
        "image_report": sum(r["kind"] == "image_report" for r in selected),
        "source_qa": sum(r["kind"] == "source_qa" for r in selected),
    }
    identity = {
        "schema_version": TRAINING_CONTRACT_VERSION,
        "release_sha256": _sha(receipt_path),
        "dataset_sha256": receipt["dataset_sha256"],
        "run_identity_sha256": receipt.get("run_identity_sha256"),
        "config_sha256": _digest(config),
        "scope": scope,
        "selected_ids_sha256": _digest([r["id"] for r in selected]),
        "counts": counts,
        "training": cfg,
        "image_root": root.as_posix(),
    }
    identity["identity_sha256"] = _digest(identity)
    return cfg, selected, root, identity


@contextmanager
def _writer_lock(output: Path):
    """프로세스 종료 시 OS가 해제하는 잠금. 남은 잠금 파일은 재개를 막지 않는다."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise TrainingRejected("another trainer owns this output directory") from exc
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _checkpoint_files(path: Path) -> dict[str, str]:
    required = [
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    hashes = {}
    for name in required:
        member = path / name
        if not member.is_file() or member.stat().st_size == 0:
            raise TrainingRejected(f"incomplete checkpoint: {member}")
        hashes[name] = _sha(member)
    return hashes


def record_checkpoint(path: Path, identity: Mapping[str, Any]) -> None:
    hashes = _checkpoint_files(path)
    state = _read_json(path / "trainer_state.json")
    step = state.get("global_step")
    if type(step) is not int or step <= 0 or path.name != f"checkpoint-{step}":
        raise TrainingRejected("checkpoint global_step does not match its directory")
    _write_json(
        path / "TRAINING_IDENTITY.json",
        {"identity": dict(identity), "global_step": step, "files": hashes},
    )


def resume_checkpoint(output: Path, identity: Mapping[str, Any]) -> Path:
    paths = sorted(
        (
            p
            for p in output.glob("checkpoint-*")
            if p.is_dir() and re.fullmatch(r"checkpoint-[0-9]+", p.name)
        ),
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,
    )
    for path in paths:
        marker = path / "TRAINING_IDENTITY.json"
        if not marker.is_file():
            # 저장 도중 중단된 폴더에는 완료 영수증이 없다. 마지막 완전 체크포인트만 쓴다.
            continue
        record = _read_json(marker)
        if record.get("identity") != dict(identity):
            raise TrainingRejected("checkpoint release/config/scope identity mismatch")
        if record.get("files") != _checkpoint_files(path):
            raise TrainingRejected("checkpoint content changed after it was saved")
        state = _read_json(path / "trainer_state.json")
        if (
            record.get("global_step") != state.get("global_step")
            or path.name != f"checkpoint-{state.get('global_step')}"
        ):
            raise TrainingRejected("checkpoint step identity mismatch")
        return path
    raise TrainingRejected(
        "resume requested but no complete optimizer/scheduler/RNG checkpoint exists"
    )


class AssistantCollator:
    """하나의 text 또는 image/text 예제를 직접 인코딩하여 시각 입력을 그대로 전달한다."""

    def __init__(self, processor: Any, config: Mapping[str, Any], image_root: Path):
        self.processor = processor
        self.config = dict(config)
        self.image_root = image_root

    def __call__(self, features: list[dict]) -> dict:
        if len(features) != 1:
            raise TrainingRejected("mixed-modal collator requires microbatch=1")
        row = features[0]
        messages = copy.deepcopy(row["messages"])
        opened = []
        try:
            if self.config["model_kind"] == "causal_lm":
                if row["kind"] != "source_qa":
                    raise TrainingRejected("text-only model received an image")
                messages = [
                    {"role": m["role"], "content": "\n".join(b["text"] for b in m["content"])}
                    for m in messages
                ]
            else:
                for message in messages:
                    for block in message["content"]:
                        if block["type"] == "image":
                            from PIL import Image

                            image_bytes = image_file(block["image"], self.image_root).read_bytes()
                            if hashlib.sha256(image_bytes).hexdigest() != row["image_sha256"]:
                                raise TrainingRejected(f"image content changed: {row['id']}")
                            with Image.open(io.BytesIO(image_bytes)) as original:
                                img = original.convert("RGB")
                            opened.append(img)
                            block["image"] = img
            kwargs = {
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "padding": False,
                "truncation": False,
                "enable_thinking": False,
            }
            full = dict(
                self.processor.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
            )
            prefix = self.processor.apply_chat_template(
                messages[:1], add_generation_prompt=True, **kwargs
            )
            ids = full["input_ids"].tolist()
            pids = prefix["input_ids"].tolist()
            if len(ids) != 1 or len(pids) != 1 or "attention_mask" not in full:
                raise TrainingRejected(
                    "processor returned an unexpected batch or no attention mask"
                )
            prompt_mask = prefix.get("attention_mask")
            if prompt_mask is not None:
                mask_rows = prompt_mask.tolist()
                if (
                    len(mask_rows) != 1
                    or len(mask_rows[0]) != len(pids[0])
                    or any(type(m) is not int or m not in (0, 1) for m in mask_rows[0])
                ):
                    raise TrainingRejected(
                        "processor returned an invalid generation attention mask"
                    )
            prefix_ids = (
                pids[0] if prompt_mask is None else [v for v, m in zip(pids[0], mask_rows[0]) if m]
            )
            labels = assistant_labels(
                ids[0], prefix_ids, full["attention_mask"].tolist()[0], self.config["max_length"]
            )
            full["labels"] = full["input_ids"].new_tensor([labels])
            return full
        finally:
            for image in opened:
                image.close()


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        p.relative_to(directory).as_posix(): _sha(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def verify_artifact(directory: Path, expected_identity: str | None = None) -> dict:
    record = _read_json(directory / "ARTIFACT.json")
    if expected_identity is not None and record.get("identity_sha256") != expected_identity:
        raise TrainingRejected("saved model identity mismatch")
    hashes = _artifact_hashes(directory)
    hashes.pop("ARTIFACT.json", None)
    if record.get("files") != hashes:
        raise TrainingRejected("saved model artifact content has changed")
    return record


def _save_last(
    model: Any, processor: Any, output: Path, identity: dict, step: int, runtime: dict, result: dict
) -> Path:
    target = output / "last"
    if target.exists():
        raise TrainingRejected("last adapter already exists; never overwrite it")
    staging = output / (".last-" + uuid.uuid4().hex)
    staging.mkdir()
    model.save_pretrained(staging, safe_serialization=True)
    processor.save_pretrained(staging)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(staging)
    for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer_config.json"):
        if not (staging / name).is_file():
            raise TrainingRejected(f"PEFT export missing {name}")
    _write_json(
        staging / "base_model.json",
        {
            "model_id": identity["training"]["model_id"],
            "revision": identity["training"]["revision"],
            "model_kind": identity["training"]["model_kind"],
            "quantization": identity["training"]["quantization"],
            "identity_sha256": identity["identity_sha256"],
            "global_step": step,
            "runtime": runtime,
        },
    )
    _write_json(staging / "TRAINING_RESULT.json", result)
    hashes = _artifact_hashes(staging)
    _write_json(
        staging / "ARTIFACT.json", {"identity_sha256": identity["identity_sha256"], "files": hashes}
    )
    os.replace(staging, target)
    return target


def export_merged_bf16(
    adapter_dir: str | Path, config: Mapping[str, Any], output_dir: str | Path
) -> dict:
    """선택적 BF16 통합 모델. QLoRA 모델에 직접 merge하지 않고 동일 판 BF16 원본에 적용한다."""
    cfg = _training_config(config)
    adapter, output = Path(adapter_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or _inside(output, adapter) or _inside(adapter, output):
        raise TrainingRejected("merged export needs a new, separate output directory")
    record = verify_artifact(adapter)
    base_receipt = _read_json(adapter / "base_model.json")
    if any(base_receipt.get(key) != cfg[key] for key in ("model_id", "revision", "model_kind")):
        raise TrainingRejected("merged export base model does not match the trained adapter")
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    cls = AutoModelForImageTextToText if cfg["model_kind"] == "image_text" else AutoModelForCausalLM
    base = cls.from_pretrained(
        cfg["model_id"],
        revision=cfg["revision"],
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )
    peft_model = PeftModel.from_pretrained(base, adapter, local_files_only=True, is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    staging = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    merged.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
    proc_cls = AutoProcessor if cfg["model_kind"] == "image_text" else AutoTokenizer
    proc = proc_cls.from_pretrained(adapter, local_files_only=True, trust_remote_code=False)
    proc.save_pretrained(staging)
    _write_json(
        staging / "base_model.json", {**base_receipt, "export_dtype": "bfloat16", "merged": True}
    )
    result = {"identity_sha256": record["identity_sha256"], "files": _artifact_hashes(staging)}
    _write_json(staging / "ARTIFACT.json", result)
    os.replace(staging, output)
    return {
        "status": "merged",
        "path": output.as_posix(),
        "identity_sha256": record["identity_sha256"],
    }


def _execute(
    cfg: dict, rows: list[dict], root: Path, identity: dict, output: Path, checkpoint: Path | None
) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise TrainingRejected(
            "this recipe is single-process; distributed execution is not registered"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise TrainingRejected("actual SFT requires a CUDA device supporting bf16")
    set_seed(cfg["seed"])
    runtime = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
    }
    runtime_path = output / "RUNTIME.json"
    if runtime_path.exists() and _read_json(runtime_path) != runtime:
        raise TrainingRejected("resume runtime package versions differ from the original run")
    if not runtime_path.exists():
        _write_json(runtime_path, runtime)
    common = {"revision": cfg["revision"], "local_files_only": True, "trust_remote_code": False}
    cls = AutoModelForImageTextToText if cfg["model_kind"] == "image_text" else AutoModelForCausalLM
    proc_cls = AutoProcessor if cfg["model_kind"] == "image_text" else AutoTokenizer
    processor = proc_cls.from_pretrained(
        cfg["model_id"], **common, **cfg.get("processor_kwargs", {})
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise TrainingRejected("tokenizer has neither padding nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_kwargs = {**common, "dtype": torch.bfloat16, "device_map": {"": 0}}
    if cfg["quantization"] == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = cls.from_pretrained(cfg["model_id"], **model_kwargs)
    if cfg["quantization"] == "4bit":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    # 시각 인코더는 동결하고 언어 모듈만 명시적으로 선택한다. suffix만 주면 vision의 q_proj까지 붙을 수 있다.
    targets = [
        name
        for name, _ in model.named_modules()
        if any(name.endswith("." + suffix) or name == suffix for suffix in cfg["target_modules"])
        and not any(part in name.lower() for part in ("visual", "vision", "image_tower"))
    ]
    if not targets:
        raise TrainingRejected("no language-side LoRA target modules found in the selected model")
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    model.config.use_cache = False
    if any(
        p.requires_grad and any(v in name.lower() for v in ("visual", "vision", "image_tower"))
        for name, p in model.named_parameters()
    ):
        raise TrainingRejected("a trainable vision parameter escaped the frozen-encoder contract")

    class CheckpointIdentity(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            record_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}", identity)
            return control

    args_kwargs = {
        "output_dir": str(output),
        "num_train_epochs": cfg["epochs"],
        "max_steps": -1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "learning_rate": cfg["learning_rate"],
        "lr_scheduler_type": "cosine",
        "warmup_steps": 0,
        "optim": "adamw_torch",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "seed": cfg["seed"],
        "data_seed": cfg["seed"],
        "eval_strategy": "no",
        "save_strategy": "steps",
        "save_steps": cfg["save_steps"],
        "save_total_limit": 3,
        "save_only_model": False,
        "load_best_model_at_end": False,
        "ignore_data_skip": False,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "report_to": "none",
        "push_to_hub": False,
        "dataloader_num_workers": 0,
        "dataloader_drop_last": False,
        "auto_find_batch_size": False,
        "logging_steps": min(10, cfg["save_steps"]),
    }
    if "save_safetensors" in inspect.signature(TrainingArguments).parameters:
        args_kwargs["save_safetensors"] = True
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**args_kwargs),
        train_dataset=rows,
        data_collator=AssistantCollator(processor, cfg, root),
        processing_class=processor,
        callbacks=[CheckpointIdentity()],
    )
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    step = int(trainer.state.global_step)
    if (
        step <= 0
        or trainer.state.epoch is None
        or float(trainer.state.epoch) + 1e-8 < cfg["epochs"]
    ):
        raise TrainingRejected("training stopped before all fixed epochs completed")
    last_checkpoint = resume_checkpoint(output, identity)
    if _read_json(last_checkpoint / "trainer_state.json")["global_step"] != step:
        raise TrainingRejected("the final step has no complete resumable checkpoint")
    final = {
        "status": "trained",
        "identity_sha256": identity["identity_sha256"],
        "scope": identity["scope"],
        "counts": identity["counts"],
        "global_step": step,
        "epochs": float(trainer.state.epoch),
        "adapter_dir": (output / "last").as_posix(),
        "checkpoint": last_checkpoint.as_posix(),
        "metrics": result.metrics,
        "runtime": runtime,
    }
    _save_last(model, processor, output, identity, step, runtime, final)
    # 8B BF16 재적재와 Trainer가 동시에 VRAM을 잡지 않게 참조를 먼저 해제한다.
    del trainer, model, processor
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return final


def train_from_release(
    release_dir: str | Path,
    config: Mapping[str, Any],
    scope: str,
    output_dir: str | Path,
    resume: bool = False,
    dry_run: bool = False,
) -> dict:
    """검수 릴리스 → 범위 제한 SFT → 마지막 PEFT 어댑터. dry_run은 모델·파일을 쓰지 않는다."""
    release, output = Path(release_dir).resolve(), Path(output_dir).resolve()
    cfg, rows, root, identity = _prepare(release, config, scope, output)
    summary = {
        "status": "dry_run",
        "identity_sha256": identity["identity_sha256"],
        "scope": scope,
        "counts": identity["counts"],
        "model_id": cfg["model_id"],
        "revision": cfg["revision"],
        "output_dir": output.as_posix(),
        "max_length": cfg["max_length"],
        "tokenization_checked": False,
        "gpu_executed": False,
        "planned_optimizer_steps": cfg["epochs"]
        * math.ceil(len(rows) / cfg["gradient_accumulation_steps"]),
    }
    if dry_run:
        if resume:
            summary["resume_checkpoint"] = resume_checkpoint(output, identity).as_posix()
        return summary
    with _writer_lock(output):
        identity_path = output / "TRAINING_IDENTITY.json"
        if identity_path.exists():
            if _read_json(identity_path) != identity:
                raise TrainingRejected("output belongs to a different release/config/scope")
            if not resume:
                raise TrainingRejected("existing run requires explicit resume=True")
        else:
            if resume or any(p.name != ".training.lock" for p in output.iterdir()):
                raise TrainingRejected(
                    "new training output must be empty; resume needs an identity"
                )
            _write_json(identity_path, identity)
        complete_path = output / "TRAINING_COMPLETE.json"
        if complete_path.exists():
            complete = _read_json(complete_path)
            if complete.get("identity_sha256") != identity["identity_sha256"]:
                raise TrainingRejected("completed run identity mismatch")
            verify_artifact(output / "last", identity["identity_sha256"])
            if cfg.get("export_merged_bf16", False):
                verify_artifact(output / "merged_bf16", identity["identity_sha256"])
            return {**complete, "status": "already_complete"}
        if resume and (output / "last").is_dir():
            verify_artifact(output / "last", identity["identity_sha256"])
            final = _read_json(output / "last" / "TRAINING_RESULT.json")
            if final.get("identity_sha256") != identity["identity_sha256"]:
                raise TrainingRejected("last training result identity mismatch")
        else:
            checkpoint = resume_checkpoint(output, identity) if resume else None
            final = _execute(cfg, rows, root, identity, output, checkpoint)
        if cfg.get("export_merged_bf16", False):
            merged = output / "merged_bf16"
            if merged.is_dir():
                verify_artifact(merged, identity["identity_sha256"])
                final["merged_export"] = {
                    "status": "merged",
                    "path": merged.as_posix(),
                    "identity_sha256": identity["identity_sha256"],
                }
            else:
                final["merged_export"] = export_merged_bf16(final["adapter_dir"], config, merged)
        _write_json(complete_path, final)
        return final
