"""2026-09-09 감사 회귀: CSV, adapter 저장, 재개 회계, opt-in loader 정책.

모델 다운로드나 GPU 학습 없이 실제 CSV writer/체크포인터/CPU 로더를 검사한다.
"""

import csv
import math
import random
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from detection.budget_audit import AccountingCell, AccountingMatrix
from detection.fed_trainer import FedDetectionTrainer, LoaderReseed
from detection.resume import ResumeCheckpointer, ResumeIdentity, apply_resume, latest_resume
from detection.round_runner import _LRTrace, derive_seed, validate_detection_resume
from tests.test_detection_resume import _Counter, _FakeTrainer, _Net, _identity


def test_final_epoch_csv_keeps_columns_and_marks_validation_unmeasured(tmp_path):
    trainer = object.__new__(FedDetectionTrainer)
    trainer.csv = tmp_path / "results.csv"
    trainer.train_time_start = time.time()
    trainer.metrics = dict.fromkeys(
        ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
         "metrics/mAP50-95(B)", "val/box_loss", "val/cls_loss", "val/dfl_loss"], 0.0
    )
    losses = {"train/box_loss": 2.2, "train/cls_loss": 1.7, "train/dfl_loss": 1.4}
    lrs = {"lr/pg0": 0.0001, "lr/pg1": 0.0001, "lr/pg2": 0.0001}
    trainer.epoch = 98
    trainer.save_metrics({**losses, **trainer.metrics, **lrs})
    trainer.epoch = 99
    trainer.metrics, fitness = trainer.validate()
    assert fitness is None
    trainer.save_metrics({**losses, **trainer.metrics, **lrs})

    with trainer.csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert [len(row) for row in rows] == [15, 15, 15]
    for row in rows[1:]:
        record = dict(zip(rows[0], row, strict=True))
        assert float(record["train/box_loss"]) == 2.2
        assert float(record["lr/pg0"]) == 0.0001
        assert all(math.isnan(float(record[key])) for key in trainer.metrics)


def test_adapter_save_hook_and_restore_use_the_same_payload(tmp_path):
    source = _FakeTrainer(_Net(seed=1), epoch=4, start_epoch=4)
    reads = []

    def adapter_state(tr):
        reads.append(tr)
        return {"lin.weight": tr.model.lin.weight}

    ck = ResumeCheckpointer(tmp_path, identity=_identity(), state_dict_fn=adapter_state)
    ck.save(source)
    state = latest_resume(tmp_path)
    assert reads == [source]
    assert state.payload["canonical_keys"] == ["lin.weight"]
    fresh = _FakeTrainer(_Net(seed=99), epoch=0, start_epoch=0)
    untouched_bias = fresh.model.lin.bias.detach().clone()

    def load_adapter(model, sd):
        with torch.no_grad():
            model.lin.weight.copy_(sd["lin.weight"])

    apply_resume(fresh, state, state_dict_fn=adapter_state, load_state_dict_fn=load_adapter)
    assert torch.equal(fresh.model.lin.weight, source.model.lin.weight)
    assert torch.equal(fresh.model.lin.bias, untouched_bias)


def _partial_checkpoint(tmp_path):
    trainer = _FakeTrainer(_Net(), epoch=4, start_epoch=4)
    trainer.n_optimizer_updates = 11
    trainer._resume_lr_trace = _LRTrace([(4, 0.01)])
    ck = ResumeCheckpointer(tmp_path, identity=_identity(), step_counter=_Counter(37))
    ck.save(trainer)
    return latest_resume(tmp_path)


def test_resume_preserves_update_total_and_full_lr_trace_across_two_processes(tmp_path):
    state = _partial_checkpoint(tmp_path)
    validate_detection_resume(state)
    trainer = _FakeTrainer(_Net(seed=3), epoch=5, start_epoch=5)
    apply_resume(trainer, state)
    trainer.n_optimizer_updates = state.payload["optimizer_updates"] + 5
    trainer._resume_lr_trace = _LRTrace(state.payload["lr_trace"])
    trainer.lr = {"lr/pg0": 0.009}
    trainer._resume_lr_trace(trainer)
    ResumeCheckpointer(
        tmp_path, identity=state.identity, step_counter=_Counter(10),
        resumed_steps=state.optimizer_steps, resumed_epochs=state.epochs_ran_in_round,
    ).save(trainer)
    final = latest_resume(tmp_path)
    assert final.optimizer_steps == 47
    assert final.epochs_ran_in_round == 2
    assert final.payload["optimizer_updates"] == 16
    assert final.payload["lr_trace"] == [(4, 0.01), (5, 0.009)]


def test_completed_resume_refuses_additional_training_and_preserves_file(tmp_path):
    state = _partial_checkpoint(tmp_path)
    state.epochs_ran_in_round = 2
    state.epoch_done = 5
    with pytest.raises(ValueError, match="이미 완료"):
        validate_detection_resume(state)
    assert state.path.exists()


@pytest.mark.parametrize("missing", ["optimizer_updates", "lr_trace"])
def test_legacy_resume_missing_accounting_is_not_silently_zero(tmp_path, missing):
    state = _partial_checkpoint(tmp_path)
    del state.payload[missing]
    with pytest.raises(ValueError, match="구판 재개"):
        validate_detection_resume(state)


def test_resume_rejects_inconsistent_global_epoch(tmp_path):
    state = _partial_checkpoint(tmp_path)
    state.epoch_done = 3
    with pytest.raises(ValueError, match="전역 epoch"):
        validate_detection_resume(state)


def test_resume_rejects_incomplete_lr_history(tmp_path):
    state = _partial_checkpoint(tmp_path)
    state.payload["lr_trace"] = []
    with pytest.raises(ValueError, match="LR 이력"):
        validate_detection_resume(state)


@pytest.mark.parametrize("change", [
    {"loader_reseed_per_epoch": True}, {"loader_seed": 77}, {"profile": "pilot"},
])
def test_resume_identity_rejects_policy_or_profile_change(tmp_path, change):
    state = _partial_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="신원"):
        latest_resume(tmp_path, identity=replace(state.identity, **change))


def _accounted_round(r, **overrides):
    values = dict(round_idx=r, client_idx=0, epochs_ran=2, optimizer_steps=10,
                  num_examples=5, seed=0, stopper_class="NoEarlyStopping",
                  stopper_true_count=0, stopper_calls=2)
    values.update(overrides)
    return AccountingCell(**values)


def test_later_round_resume_stopper_audit_uses_local_consumed_epochs():
    matrix = AccountingMatrix(3, [0], 2, 6)
    matrix.record(_accounted_round(0))
    matrix.record(_accounted_round(1))
    # Round 2 begins at global epoch 4; resume at 5 means one epoch remains.
    matrix.record(_accounted_round(2, resumed_from_epoch=5, stopper_calls=0))
    report = matrix.audit()
    assert not report.ok
    assert any("stopper 호출 0회 <" in failure for failure in report.failures)
    matrix.cells[(2, 0)].stopper_calls = 1
    assert matrix.audit().ok


def test_accounting_rejects_mixed_loader_policy():
    matrix = AccountingMatrix(2, [0], 2, 4)
    matrix.record(_accounted_round(0, loader_reseed_per_epoch=False, profile="main"))
    matrix.record(_accounted_round(1, loader_reseed_per_epoch=True, profile="main"))
    assert any("loader 정책" in failure for failure in matrix.audit().failures)


class _RandomProbeDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return index, random.random(), np.random.random(), torch.rand(()), torch.initial_seed()


def test_opt_in_loader_reseeds_worker_and_sampler_by_global_epoch():
    from ultralytics.data.build import build_dataloader

    loader = build_dataloader(_RandomProbeDataset(), batch=4, workers=1, shuffle=True, device="cpu")
    trainer = SimpleNamespace(train_loader=loader, epoch=4)
    policy = LoaderReseed(derive_seed(77, 0, 1))

    def sample(reseed, epoch):
        trainer.epoch = epoch
        reseed(trainer)
        return [torch.cat(columns) for columns in zip(*list(loader))]

    try:
        a = sample(policy, 4)
        b = sample(policy, 4)
        c = sample(policy, 5)
        d = sample(LoaderReseed(derive_seed(77, 0, 2)), 4)
    finally:
        loader.close()
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[0], c[0])  # sampler permutation
    assert not torch.equal(a[0], d[0])
    assert all(not torch.equal(a[i], c[i]) for i in range(1, 5))  # worker RNGs/base seed
    assert all(not torch.equal(a[i], d[i]) for i in range(1, 5))


def test_fl_metrics_preserve_process_epochs_and_loader_policy():
    pytest.importorskip("flwr")
    from fl.server_app import _cell_from_metrics

    cell = _cell_from_metrics(2, {
        "epochs-ran": 2, "epochs-this-process": 1, "resumed-from-epoch": 5,
        "optimizer-updates": 16, "loader-reseed-per-epoch": 1,
        "loader-seed": 178, "profile": "main",
    })
    assert cell.epochs_this_process == 1
    assert cell.optimizer_updates == 16
    assert cell.loader_reseed_per_epoch is True
    assert cell.loader_seed == 178
    assert cell.profile == "main"


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_loader_policy_requires_real_boolean(value):
    from detection.round_runner import validate_loader_policy

    with pytest.raises(ValueError, match="bool"):
        validate_loader_policy(value)


def test_loader_policy_reaches_fl_client_from_server_config(monkeypatch, tmp_path):
    from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict
    from fl import client_app, client_det
    from fl.round_wiring import CANONICAL_KEYS_KEY, SERVER_ROUND_KEY
    from fl.server_app import _cell_train_config
    from fl.strategy import ARRAYS_KEY, CONFIG_KEY

    config = _cell_train_config("sep_fed", {
        "views-root": str(tmp_path), "model": "test", "num-examples": "1",
        "loader-reseed-per-epoch": True,
    }, tmp_path)
    config.update({"cell": "sep_fed", CANONICAL_KEYS_KEY: ["w"], SERVER_ROUND_KEY: 1,
                   "total-epochs": 2, "local-epochs": 2, "base-seed": 7})
    captured = []

    def fake_round(**kwargs):
        captured.append(kwargs["cfg"])
        return kwargs["weights_in"], {"num-examples": 1.0}, {}

    monkeypatch.setattr(client_det, "run_client_round", fake_round)
    monkeypatch.setattr(client_app, "_client_idx", lambda ctx: 0)
    message = Message(content=RecordDict({
        ARRAYS_KEY: ArrayRecord([np.zeros(1, dtype=np.float32)]),
        CONFIG_KEY: ConfigRecord(config),
    }), dst_node_id=1, message_type="train")
    client_app.train(message, SimpleNamespace())
    assert captured[0]["loader_reseed_per_epoch"] is True


def test_flwr_registers_loader_policy_with_legacy_default():
    import tomllib
    from pathlib import Path

    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    config = tomllib.loads(project.read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    assert config["loader-reseed-per-epoch"] is False


@pytest.mark.parametrize("value", [False, True])
def test_pilot_config_preserves_explicit_loader_policy(tmp_path, value):
    from fl.pilot_sim import _cell_train_config

    config = _cell_train_config("sep_fed", {
        "views_root": str(tmp_path), "model": "test", "num_examples": [1],
        "loader_reseed_per_epoch": value,
    }, tmp_path)
    assert config["loader-reseed-per-epoch"] is value


@pytest.mark.parametrize("kind", ["local", "central"])
def test_standalone_cells_forward_explicit_loader_policy(monkeypatch, tmp_path, kind):
    from detection import train_cell

    captured = []

    class StopAtTrainer(Exception):
        pass

    def fake_train(**kwargs):
        captured.append(kwargs)
        raise StopAtTrainer

    monkeypatch.setattr(train_cell, "train_round", fake_train)
    monkeypatch.setattr(train_cell, "AtomicLog", lambda *args, **kwargs: None)
    common = dict(model="test", total_epochs=2, base_seed=7, out_dir=tmp_path,
                  split_hash="fixture", run_stamp="new-protocol", loader_reseed_per_epoch=True)
    with pytest.raises(StopAtTrainer):
        if kind == "local":
            train_cell.run_local_cell(client_data_yamls={0: "x.yaml"}, client_num_examples={0: 1}, **common)
        else:
            train_cell.run_central_cell(data_yaml="x.yaml", num_examples=1, **common)
    assert captured[0]["loader_reseed_per_epoch"] is True
