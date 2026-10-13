"""Separate, immutable judge identities over an unchanged generation run.

This module does not replace the legacy judge reader/writer. A caller must select
one judge directory explicitly; there is no discovery, newest-folder selection,
or fallback to run/judged. Raw judge output is preserved for the strict quality
parser. Fake backends remain labelled and are never production approval.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import time
from datetime import UTC, datetime
from pathlib import Path

from . import infer as I

_SCHEMA = 1
_SHARD = re.compile(r"[0-9]{6}\.json")


def _clone(value):
    return I._json(I._canonical(value))


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_path(path: Path) -> Path:
    """Reject symlinks and Windows junction/reparse aliases at every level."""
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise I.StageRefused("path_alias", str(part))
        if part.exists():
            attrs = getattr(part.lstat(), "st_file_attributes", 0)
            if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise I.StageRefused("path_alias", str(part))
    return path


def _paths(run_dir, judge_dir):
    root, target = _safe_path(Path(run_dir)), _safe_path(Path(judge_dir))
    if target == root or target in root.parents:
        raise I.StageRefused("judge_dir_overlap", "judge directory is the run or its ancestor")
    for reserved in (root / "generated", root / "judged"):
        if target == reserved or reserved in target.parents:
            raise I.StageRefused("judge_dir_overlap", "legacy output paths are immutable")
    if target.exists() and not target.is_dir():
        raise I.StageRefused("invalid_judge_dir", str(target))
    return root, target


def _unlocked(root: Path, target: Path):
    for folder in (root, target):
        lock = folder / ".inference.lock"
        if lock.exists() or lock.is_symlink():
            raise I.StageRefused("writer_locked", "reader requires both writers to be absent")


def _files(folder: Path, *, absent_ok=False) -> dict:
    _safe_path(folder)
    if not folder.exists():
        if absent_ok:
            return {}
        raise I.StageRefused("generation_incomplete", "generated folder is absent")
    if not folder.is_dir():
        raise I.StageRefused("invalid_output_dir", str(folder))
    result = {}
    for path in sorted(folder.iterdir()):
        if not _SHARD.fullmatch(path.name):
            raise I.StageRefused("partial_or_unknown_output", f"preserve and inspect {path}")
        _safe_path(path)
        I._plain_file(path)
        result[path.name] = I._file_sha(path)
    return result


def _source_pins(root: Path) -> dict:
    _safe_path(root)
    result = {}
    for name in ("RUN.json", "tasks.jsonl"):
        path = root / name
        _safe_path(path)
        I._plain_file(path)
        result[name] = I._file_sha(path)
    result["generated"] = _files(root / "generated")
    result["implementation"] = {
        "infer.py": I._file_sha(Path(I.__file__)),
        "judge_identity.py": I._file_sha(Path(__file__)),
    }
    return result


def _output_pins(target: Path) -> dict:
    _safe_path(target)
    if not target.exists():
        return {"JUDGE.json": None, "judged": {}, "probe_approval": None}
    if not target.is_dir():
        raise I.StageRefused("invalid_judge_dir", str(target))
    for path in target.iterdir():
        if path.name not in ("JUDGE.json", "judged", ".inference.lock", "probe_approval"):
            raise I.StageRefused("partial_or_unknown_output", f"preserve and inspect {path}")
        _safe_path(path)
        if path.name not in ("judged", "probe_approval"):
            I._plain_file(path)
    genesis = target / "JUDGE.json"
    approval = target / "probe_approval"
    from .judge_probe import approval_files
    return {"JUDGE.json": I._file_sha(genesis) if genesis.exists() else None,
            "judged": _files(target / "judged", absent_ok=True),
            "probe_approval": approval_files(approval) if approval.exists() else None}


def _policy():
    # Read-only policy metadata is also used by CPU-only rule checks before B's
    # parser arrives. Active writer/read boundaries enforce v6 separately.
    from . import judge_policy as policy
    system = policy.JUDGE_SYSTEM_V2
    identity = _clone(policy.policy_identity())
    if not isinstance(system, str) or not system or not isinstance(identity, dict) or not identity:
        raise I.StageRefused("judge_policy", "missing system or policy identity")
    return policy, system, identity


def require_v5_parser():
    """Archived v5 parser capability; not the active writer/reader approval.

    This check never parses a row or loads a model. The policy identity separately
    binds the selected parser implementation and its dependency closure.
    """
    from . import judge_policy as policy, judge_fingerprints as fingerprints, quality
    parser = getattr(quality, "parse_judge_v5", None)
    if (getattr(quality, "JUDGE_V5_SCHEMA_VERSION", None) != "judge_structured_v5" or
            getattr(policy, "OUTPUT_SCHEMA_VERSION", None) != "judge_structured_v5" or
            not callable(parser)):
        raise I.StageRefused("parser_missing", "B structured-v5 parser and policy schema must be explicitly connected")
    try:
        provider = getattr(fingerprints, "require_v5_parser", None)
        if not callable(provider):
            raise ValueError("structured parser fingerprinter is unavailable")
        identity = provider()
        if (not isinstance(identity, dict) or
                identity.get("capabilities", {}).get("judge_structured_v5") is not True or
                policy.policy_identity().get("parser_implementation") != identity):
            raise ValueError("policy does not bind the actual structured parser closure")
    except (ValueError, fingerprints.FingerprintRefused) as exc:
        raise I.StageRefused("parser_missing", str(exc)) from exc
    return parser


def require_v6_parser():
    """Require the exact active v6 parser and its selected implementation closure.

    This checks no rows and loads no model. Missing B code, a stale v5 policy, or
    a parser/normalization dependency not bound by policy all fail before writing.
    """
    from . import judge_policy as policy, judge_fingerprints as fingerprints, quality
    parser = getattr(quality, "parse_judge_v6", None)
    if (getattr(quality, "JUDGE_V6_SCHEMA_VERSION", None) != "judge_structured_v6" or
            getattr(policy, "OUTPUT_SCHEMA_VERSION", None) != "judge_structured_v6" or
            not callable(parser)):
        raise I.StageRefused("parser_missing", "B structured-v6 parser and policy schema must be explicitly connected")
    try:
        provider = getattr(fingerprints, "require_v6_parser", None)
        if not callable(provider):
            raise ValueError("structured-v6 parser fingerprinter is unavailable")
        identity = provider()
        bound = policy.policy_identity()
        if (not isinstance(identity, dict) or
                identity.get("selected_schema") != "judge_structured_v6" or
                identity.get("capabilities", {}).get("judge_structured_v6") is not True or
                not isinstance(bound, dict) or
                bound.get("output_schema") != "judge_structured_v6" or
                bound.get("structured_parser_ready") is not True or
                bound.get("parser_implementation") != identity):
            raise ValueError("policy does not bind the active structured-v6 parser closure")
    except Exception as exc:  # fail closed before output/model load, including an incomplete B merge
        raise I.StageRefused("parser_missing", str(exc)) from exc
    return parser


def _decoding(spec: dict) -> dict:
    return {"do_sample": False, "num_beams": 1, "temperature": None,
            "top_p": None, "top_k": None, "forced_eos_token_id": None,
            "max_new_tokens": spec["max_new_tokens"], "max_input_tokens": spec["max_input_tokens"],
            "seed": spec["seed"], "enable_thinking": False,
            "truncation": False, "add_special_tokens": False}


def _runtime(model, spec: dict) -> dict:
    """Bind loaded defaults and effective EOS/PAD without altering legacy identity."""
    result = _clone(model.identity)
    I._check_model(result, spec)
    if result["backend"] == "fake":
        result["judge_runtime"] = {"test_only": True, "decoding": _decoding(spec)}
    else:
        try:
            defaults = _clone(model.model.generation_config.to_dict())
            eos = _clone(model.model.generation_config.eos_token_id)
            pad = _clone(model.tokenizer.pad_token_id)
            token_eos = _clone(model.tokenizer.eos_token_id)
            template = model.tokenizer.chat_template
        except (AttributeError, TypeError, ValueError) as exc:
            raise I.StageRefused("judge_runtime", "loaded decoding defaults are unavailable") from exc
        if not isinstance(defaults, dict) or not isinstance(template, str) or not template:
            raise I.StageRefused("judge_runtime", "invalid generation defaults/chat template")
        result["judge_runtime"] = {
            "test_only": False, "generation_config": defaults,
            "generation_config_sha256": I._digest(defaults),
            "effective_eos_token_id": eos, "effective_pad_token_id": pad,
            "tokenizer_eos_token_id": token_eos,
            "actual_chat_template_sha256": _sha_text(template),
            "decoding": _decoding(spec),
        }
    return result


def _check_runtime(runtime: dict, spec: dict):
    I._check_model(runtime, spec)
    details = runtime.get("judge_runtime")
    if not isinstance(details, dict) or details.get("decoding") != _decoding(spec):
        raise I.StageRefused("judge_runtime", "missing effective decoding identity")
    if runtime["backend"] == "fake":
        if details != {"test_only": True, "decoding": _decoding(spec)}:
            raise I.StageRefused("judge_runtime", "invalid fake identity")
    else:
        if details.get("test_only") is not False or not isinstance(details.get("generation_config"), dict):
            raise I.StageRefused("judge_runtime", "invalid production identity")
        if details.get("generation_config_sha256") != I._digest(details["generation_config"]):
            raise I.StageRefused("judge_runtime", "generation defaults changed")
        if details.get("effective_eos_token_id") != details["generation_config"].get("eos_token_id"):
            raise I.StageRefused("judge_runtime", "EOS differs from loaded defaults")
        if not I._HEX64.fullmatch(str(details.get("actual_chat_template_sha256", ""))):
            raise I.StageRefused("judge_runtime", "missing actual chat template digest")
        for name in ("effective_pad_token_id", "tokenizer_eos_token_id"):
            if name not in details:
                raise I.StageRefused("judge_runtime", "missing effective token identity")


def judge_runtime_identity(model, spec: dict) -> dict:
    """Bind an already loaded judge for an independent probe; no model loading.

    Use with the same ``infer.TransformersBackend`` as ``run_judge``. This is
    evidence of runtime settings only, not a quality or production approval.
    """
    runtime = _runtime(model, spec)
    _check_runtime(runtime, spec)
    return runtime


def _read_done(target, record, tasks, generated, spec, config):
    result = {}
    names = _files(target / "judged", absent_ok=True)
    if names and record is None:
        raise I.StageRefused("judge_genesis_missing", "orphan judged shards are not adopted")
    total = math.ceil(len(tasks) / config["shard_size"])
    for name in names:
        index = int(name[:-5])
        shard = I._json((target / "judged" / name).read_bytes())
        if not isinstance(shard, dict):
            raise I.StageRefused("shard_schema", name)
        payload = {k: v for k, v in shard.items() if k != "payload_sha256"}
        if shard.get("payload_sha256") != I._digest(payload):
            raise I.StageRefused("shard_digest", name)
        if index >= total or shard.get("index") != index or shard.get("stage") != "judge":
            raise I.StageRefused("shard_position", name)
        chunk = tasks[index * config["shard_size"]:(index + 1) * config["shard_size"]]
        ids = [t["id"] for t in chunk]
        basis = record["identity_basis"]
        if (shard.get("schema_version") != _SCHEMA or
                shard.get("judge_identity_sha256") != record["identity_sha256"] or
                shard.get("identity_sha256") != basis["run_identity_sha256"] or
                shard.get("config_sha256") != basis["run_config_sha256"]):
            raise I.StageRefused("judge_shard_identity", name)
        if shard.get("task_ids") != ids or shard.get("input_sha256") != I._input_digest(chunk, generated[index]):
            raise I.StageRefused("shard_inputs", name)
        if shard.get("rendered_input_sha256") != basis["rendered_shard_sha256"][index]:
            raise I.StageRefused("judge_render_changed", name)
        if shard.get("system_prompt_sha256") != basis["system_prompt_sha256"]:
            raise I.StageRefused("system_prompt_changed", name)
        if shard.get("model_config_sha256") != I._digest(spec):
            raise I.StageRefused("model_config_changed", name)
        if shard.get("model") != basis["runtime_model"]:
            raise I.StageRefused("runtime_changed", name)
        I._check_rows(shard.get("rows"), ids, spec)
        result[index] = shard
    if sorted(result) != list(range(len(result))):
        raise I.StageRefused("shard_gap", "completed shards must be a contiguous prefix")
    return result


def _approval_binding(approval, run, spec, policy, implementation):
    basis = approval["identity_basis"]
    if (basis["judge_config"] != spec or basis["input_basis"]["policy_identity"] != policy or
            basis["common_implementation_sha256"] != implementation):
        raise I.StageRefused("probe_approval_identity", "approved policy, implementation or judge config differs")
    return {"receipt_sha256": approval["receipt_sha256"],
            "identity_sha256": approval["identity_sha256"],
            "cases_file_sha256": basis["input_basis"]["cases_file_sha256"],
            "probe_identity_sha256": basis["probe_identity_sha256"],
            "probe_call_ids": basis["probe_call_ids"],
            "writer_run_identity_sha256": run["identity_sha256"],
            "writer_role": "separate-judge-not-a-probe", "gpu_model_verified": True}


def _inspect(root, target, config, judge_config=None, *, owned_locks=False,
             probe_approval=None, probe_approval_sha256=None):
    require_v6_parser()
    root, target = _paths(root, target)
    if not owned_locks:
        _unlocked(root, target)
    before, output_before = _source_pins(root), _output_pins(target)
    run, tasks = I._read_inputs(root, config)
    generation_spec = I._model_config(config, "generate")
    generated = I._read_shards(root, "generate", run, tasks, config)
    total = math.ceil(len(tasks) / config["shard_size"])
    if len(generated) != total:
        raise I.StageRefused("generation_incomplete", "separate judge requires every generated shard")
    record = None
    if output_before["JUDGE.json"] is not None:
        record = I._json((target / "JUDGE.json").read_bytes())
        if (not isinstance(record, dict) or record.get("schema_version") != _SCHEMA or
                not isinstance(record.get("identity_basis"), dict) or
                record.get("identity_sha256") != I._digest(record["identity_basis"])):
            raise I.StageRefused("judge_genesis", "invalid JUDGE identity")
    stored_spec = None if record is None else record["identity_basis"].get("judge_model_config")
    selected = judge_config if judge_config is not None else (stored_spec if record else config.get("judge"))
    spec = _clone(I._model_config({"judge": selected}, "judge"))
    if record and spec != stored_spec:
        raise I.StageRefused("judge_config_changed", "new settings require a new judge directory")
    if (I._family(spec["family"]) == I._family(generation_spec["family"]) or
            spec["model_id"] == generation_spec["model_id"]):
        raise I.StageRefused("self_judging", "judge must have a different model family")
    policy, system, policy_identity = _policy()
    from .judge_probe import read_probe_approval
    approval, approval_paths = None, []
    stored_approval = record["identity_basis"].get("probe_approval") if record else None
    if (probe_approval is None) != (probe_approval_sha256 is None):
        raise I.StageRefused("probe_approval_pin", "supply both approval directory and its independent SHA256")
    if probe_approval is not None:
        path = _safe_path(Path(probe_approval))
        approval = read_probe_approval(path, probe_approval_sha256)
        approval_paths.append((path, approval["files_sha256"]))
    if output_before["probe_approval"] is not None:
        sha = stored_approval.get("receipt_sha256") if isinstance(stored_approval, dict) else probe_approval_sha256
        local = read_probe_approval(target / "probe_approval", sha)
        if approval is not None and local != approval:
            raise I.StageRefused("probe_approval_changed", "selected approval differs from the writer's sealed evidence")
        approval = local
        approval_paths.append((target / "probe_approval", local["files_sha256"]))
    elif stored_approval is not None:
        raise I.StageRefused("probe_approval_missing", "writer's portable approval is missing")
    binding = (_approval_binding(approval, run, spec, policy_identity, before["implementation"])
               if approval is not None else None)
    prompts, rendered = {}, []
    for index, shard in generated.items():
        chunk = tasks[index * config["shard_size"]:(index + 1) * config["shard_size"]]
        values = [policy.render_judge(t, r) for t, r in zip(chunk, shard["rows"])]
        if not all(isinstance(value, str) and value for value in values):
            raise I.StageRefused("judge_render", "renderer must produce nonempty strings")
        prompts[index] = values
        rendered.append(I._digest([{"system": system, "user": value} for value in values]))
    basis = {
        "run_identity_sha256": run["identity_sha256"], "run_config_sha256": run["config_sha256"],
        "run_file_sha256": before["RUN.json"], "tasks_sha256": run["tasks_sha256"],
        "generate_shard_sha256": sorted(s["payload_sha256"] for s in generated.values()),
        "generated_files": [{"index": i, "payload_sha256": s["payload_sha256"],
                             "file_sha256": before["generated"][f"{i:06d}.json"]}
                            for i, s in generated.items()],
        "judge_model": {"model_id": spec["model_id"], "family": I._family(spec["family"])},
        "judge_model_config": spec, "system_prompt_sha256": _sha_text(system),
        "rendered_shard_sha256": rendered, "policy_identity": policy_identity,
        "implementation_sha256": before["implementation"], "decoding": _decoding(spec),
        "probe_approval": binding,
    }
    if record:
        runtime = record["identity_basis"].get("runtime_model")
        _check_runtime(runtime, spec)
        _check_approved_runtime(approval, runtime, backend_injected=False)
        if I._family(runtime["model_type"]) == I._family(generated[0]["model"]["model_type"]):
            raise I.StageRefused("self_judging", "actual model families are equal")
        if record["identity_basis"] != {**basis, "runtime_model": runtime}:
            raise I.StageRefused("judge_identity_changed", "source, render, policy or implementation differs")
    done = _read_done(target, record, tasks, generated, spec, config)
    context = dict(root=root, target=target, config=config, run=run, tasks=tasks,
                   generated=generated, spec=spec, system=system, prompts=prompts,
                   basis=basis, record=record, done=done,
                   approval=approval, approval_paths=approval_paths,
                   source_pins=before, output_pins=output_before)
    _unchanged(context)
    if not owned_locks:
        _unlocked(root, target)
    return context


def _unchanged(context, model=None):
    require_v6_parser()
    for path, token in context.get("lock_tokens", []):
        _safe_path(path)
        if not path.is_file() or path.is_symlink() or path.read_bytes() != token:
            raise I.StageRefused("writer_lock_lost", "owned writer lock was removed or replaced")
    if (_source_pins(context["root"]) != context["source_pins"] or
            _output_pins(context["target"]) != context["output_pins"]):
        raise I.StageRefused("judge_snapshot_changed", "input or completed output bytes changed")
    _, system, policy = _policy()
    if system != context["system"] or policy != context["basis"]["policy_identity"]:
        raise I.StageRefused("judge_policy_changed", "new policy requires a new identity")
    from .judge_probe import approval_files, _code_pins
    for path, pins in context.get("approval_paths", []):
        if approval_files(path) != pins:
            raise I.StageRefused("probe_approval_changed", "approval evidence changed during writer/read")
    if (context.get("approval") is not None and _code_pins() !=
            context["approval"]["identity_basis"]["input_basis"]["implementation_sha256"]):
        raise I.StageRefused("probe_approval_identity", "approved probe implementation changed during writer/read")
    if (model is not None and context.get("record") is not None and
            _runtime(model, context["spec"]) != context["record"]["identity_basis"]["runtime_model"]):
        raise I.StageRefused("runtime_changed", "loaded model/tokenizer/defaults changed")


def _check_approved_runtime(approval, runtime, *, backend_injected):
    fake = runtime["backend"] == "fake" and runtime["judge_runtime"].get("test_only") is True
    if fake and approval is None:
        return  # Explicit synthetic runtime only; never a production approval.
    if backend_injected or fake:
        raise I.StageRefused("probe_approval_backend", "injected or fake backend cannot use a production approval")
    if approval is None:
        raise I.StageRefused("probe_approval_required", "real judge requires approved two-probe evidence")
    if runtime != approval["identity_basis"]["runtime_model"]:
        raise I.StageRefused("probe_approval_runtime", "loaded writer runtime differs from approved probes")


def _write_genesis(target, basis):
    record = {"schema_version": _SCHEMA, "identity_basis": basis,
              "identity_sha256": I._digest(basis), "created_at": datetime.now(UTC).isoformat()}
    final, temp = target / "JUDGE.json", target / "JUDGE.json.tmp"
    if final.exists():
        raise I.StageRefused("output_exists", str(final))
    try:
        with temp.open("xb") as stream:
            stream.write(I._canonical(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise I.StageRefused("partial_or_unknown_output", f"preserve and inspect {temp}") from exc
    if final.exists():
        raise I.StageRefused("output_exists", str(final))
    os.replace(temp, final)
    return record


def read_judge_source(run_dir, config: dict, judge_dir, *, judge_config=None,
                      expected_judge_identity_sha256=None, expected_probe_approval_sha256=None) -> dict:
    """Return the verified 05h source object, without model loading or writes.

    ``judge_dir`` is a locator, not authority. Production reads require at least
    one independently recorded expected pin; neither is inferred from JUDGE.
    Consumers must forward ``caller_expected_pins`` unchanged on every reread.
    Only an unapproved, exact fake runtime permits legacy unpinned test reads.
    """
    caller_pins = {"expected_judge_identity_sha256": expected_judge_identity_sha256,
                   "expected_probe_approval_sha256": expected_probe_approval_sha256}
    for name, value in caller_pins.items():
        if value is not None and (not isinstance(value, str) or not I._HEX64.fullmatch(value)):
            raise I.StageRefused("judge_source_pin", f"invalid external SHA256: {name}")
    context = _inspect(run_dir, judge_dir, _clone(config), judge_config)
    record = context["record"]
    if record is None:
        raise I.StageRefused("judge_genesis_missing", "JUDGE.json is required")
    runtime = record["identity_basis"]["runtime_model"]
    fake = (runtime["backend"] == "fake" and runtime["judge_runtime"].get("test_only") is True
            and context["approval"] is None)
    explicit = any(value is not None for value in caller_pins.values())
    if not fake and not explicit:
        raise I.StageRefused("judge_source_pin_required", "production source needs an independent expected identity or approval SHA")
    approval_sha = (context["basis"]["probe_approval"] or {}).get("receipt_sha256")
    if (expected_judge_identity_sha256 is not None and
            expected_judge_identity_sha256 != record["identity_sha256"]):
        raise I.StageRefused("judge_source_pin", "selected judge differs from the caller's expected identity")
    if expected_probe_approval_sha256 is not None and expected_probe_approval_sha256 != approval_sha:
        raise I.StageRefused("judge_source_pin", "selected approval differs from the caller's expected approval")
    result = {"identity_sha256": record["identity_sha256"],
              "identity_basis": _clone(record["identity_basis"]),
              "judge_dir": str(context["target"]),
              "probe_approval_sha256": approval_sha,
              "gpu_model_verified": context["approval"] is not None,
              "caller_expected_pins": caller_pins,
              "pin_provenance": "explicit-caller" if explicit else "unapproved-fake",
              "rows": [{**row, "_stage": "judge", "_backend": shard["model"]["backend"],
                        "_shard_sha256": shard["payload_sha256"],
                        "_judge_identity_sha256": record["identity_sha256"]}
                       for shard in context["done"].values() for row in shard["rows"]]}
    _unchanged(context)
    _unlocked(context["root"], context["target"])
    return result


def run_judge(run_dir, config: dict, judge_dir, *, judge_config=None,
              max_shards=None, backend_factory=None, probe_approval=None,
              probe_approval_sha256=None) -> dict:
    """Append only to one separate identity; legacy run_stage remains unchanged."""
    if max_shards is not None and (type(max_shards) is not int or max_shards < 0):
        raise I.StageRefused("invalid_max_shards", "must be a nonnegative integer")
    require_v6_parser()
    config = _clone(config)
    context = _inspect(run_dir, judge_dir, config, judge_config,
                       probe_approval=probe_approval, probe_approval_sha256=probe_approval_sha256)
    started, newly_done = time.monotonic(), 0

    def result(status, exit_code):
        count = sum(len(s["rows"]) for s in context["done"].values())
        record = context["record"]
        return {"stage": "judge", "status": status, "exit_code": exit_code,
                "planned_tasks": len(context["tasks"]), "completed_tasks": count,
                "pending_tasks": len(context["tasks"]) - count,
                "completed_shards": len(context["done"]), "new_shards": newly_done,
                "judge_identity_sha256": None if record is None else record["identity_sha256"],
                "runtime_identity_available": record is not None,
                "probe_approval_sha256": (context["basis"]["probe_approval"] or {}).get("receipt_sha256"),
                "gpu_model_verified": context["approval"] is not None and record is not None,
                "ready_for_training": False,
                "judge_dir": str(context["target"]),
                "wall_seconds": round(time.monotonic() - started, 3)}

    total = len(context["generated"])
    if len(context["done"]) == total:
        return result("complete", 0)
    if max_shards == 0:
        return result("shard_limit", 4)
    if backend_factory is None and context["approval"] is None:
        raise I.StageRefused("probe_approval_required", "explicit two-probe approval is required before model load")
    if backend_factory is None and (probe_approval is None or probe_approval_sha256 is None):
        raise I.StageRefused("probe_approval_pin", "production execution/resume requires an explicit approval SHA")
    root, target = context["root"], context["target"]
    deadline = started + context["spec"]["wall_seconds"]
    with I._writer_lock(root):
        require_v6_parser()
        root_lock = (root / ".inference.lock", (root / ".inference.lock").read_bytes())
        context["lock_tokens"] = [root_lock]
        _unchanged(context)
        target.mkdir(parents=True, exist_ok=True)
        _safe_path(target)
        with I._writer_lock(target):
            target_lock = (target / ".inference.lock", (target / ".inference.lock").read_bytes())
            fresh = _inspect(root, target, config, judge_config, owned_locks=True,
                             probe_approval=probe_approval, probe_approval_sha256=probe_approval_sha256)
            if (fresh["source_pins"] != context["source_pins"] or
                    fresh["output_pins"] != context["output_pins"] or fresh["basis"] != context["basis"]):
                raise I.StageRefused("judge_snapshot_changed", "changed while acquiring writer locks")
            context = fresh
            context["lock_tokens"] = [root_lock, target_lock]
            _unchanged(context)
            if context["approval"] is not None and context["output_pins"]["probe_approval"] is None:
                from .judge_probe import copy_probe_approval
                copied = copy_probe_approval(probe_approval, target / "probe_approval", probe_approval_sha256)
                context["output_pins"]["probe_approval"] = context["approval"]["files_sha256"]
                context["approval_paths"].append((target / "probe_approval", copied["files_sha256"]))
                _unchanged(context)
            spec, model = context["spec"], None
            try:
                for index in range(len(context["done"]), total):
                    if max_shards is not None and newly_done >= max_shards:
                        return result("shard_limit", 4)
                    if time.monotonic() >= deadline:
                        return result("time_limit", 3)
                    _unchanged(context, model)
                    require_v6_parser()
                    if model is None:
                        load_started = time.monotonic()
                        model = (backend_factory or I.TransformersBackend)(dict(spec), "judge")
                        model_load_seconds = time.monotonic() - load_started
                        runtime = judge_runtime_identity(model, spec)
                        _check_approved_runtime(context["approval"], runtime,
                                                backend_injected=backend_factory is not None)
                        if I._family(runtime["model_type"]) == I._family(context["generated"][0]["model"]["model_type"]):
                            raise I.StageRefused("self_judging", "actual architecture families are equal")
                        _unchanged(context)
                        if context["record"] is None:
                            context["record"] = _write_genesis(target, {**context["basis"], "runtime_model": runtime})
                            context["output_pins"]["JUDGE.json"] = hashlib.sha256(
                                I._canonical(context["record"]) + b"\n").hexdigest()
                        elif runtime != context["record"]["identity_basis"]["runtime_model"]:
                            raise I.StageRefused("runtime_changed", "resume requires identical loaded runtime")
                        _unchanged(context, model)
                    size = config["shard_size"]
                    chunk = context["tasks"][index * size:(index + 1) * size]
                    ids = [task["id"] for task in chunk]
                    prompts, rows = context["prompts"][index], []
                    memory = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
                    shard_started = time.monotonic()
                    for offset in range(0, len(chunk), spec["batch_size"]):
                        if time.monotonic() >= deadline:
                            return result("time_limit", 3)
                        batch = prompts[offset:offset + spec["batch_size"]]
                        output = model.generate_batch(batch, context["system"], deadline)
                        if not isinstance(output, list) or len(output) != len(batch):
                            raise I.StageRefused("backend_row_count", "batch returned wrong row count")
                        for tid, row in zip(ids[offset:offset + spec["batch_size"]], output):
                            if not isinstance(row, dict):
                                raise I.StageRefused("row_content", tid)
                            rows.append({**row, "id": tid})
                        for key in memory:
                            memory[key] = max(memory[key], getattr(model, "last_metrics", {}).get(key, 0))
                    I._check_rows(rows, ids, spec)
                    _unchanged(context, model)
                    basis, record = context["basis"], context["record"]
                    payload = {"schema_version": _SCHEMA, "stage": "judge", "index": index,
                               "identity_sha256": context["run"]["identity_sha256"],
                               "config_sha256": context["run"]["config_sha256"],
                               "judge_identity_sha256": record["identity_sha256"],
                               "task_ids": ids, "input_sha256": I._input_digest(chunk, context["generated"][index]),
                               "rendered_input_sha256": basis["rendered_shard_sha256"][index],
                               "system_prompt_sha256": basis["system_prompt_sha256"],
                               "model_config_sha256": I._digest(spec),
                               "model": _clone(record["identity_basis"]["runtime_model"]), "rows": rows,
                               "metrics": {"wall_seconds": round(time.monotonic() - shard_started, 6),
                                           "model_load_seconds": round(model_load_seconds if newly_done == 0 else 0, 6),
                                           **memory, "input_tokens": sum(r["input_tokens"] for r in rows),
                                           "output_tokens": sum(r["output_tokens"] for r in rows)},
                               "created_at": datetime.now(UTC).isoformat()}
                    folder = target / "judged"
                    folder.mkdir(exist_ok=True)
                    _safe_path(folder)
                    path = folder / f"{index:06d}.json"
                    I._write_shard(path, payload)
                    context["done"][index] = {**payload, "payload_sha256": I._digest(payload)}
                    # Add only the expected new file; never adopt changes to old files.
                    context["output_pins"]["judged"][path.name] = hashlib.sha256(
                        I._canonical(context["done"][index]) + b"\n").hexdigest()
                    _unchanged(context, model)
                    newly_done += 1
                return result("complete", 0)
            except I.StageTimeLimit:
                _unchanged(context, model)
                return result("time_limit", 3)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    _unchanged(context, model)
                    return {**result("out_of_memory", 5), "retry_policy": "explicit resume; unchanged identity only"}
                raise
            finally:
                if model is not None:
                    model.close()
