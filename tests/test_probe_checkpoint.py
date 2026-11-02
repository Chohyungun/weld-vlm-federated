"""합성 로컬 파일로 보존 확인 성공·실패·시간 초과를 검증한다. Drive 연결·GPU 없음."""

import copy
import json
import threading
import time

import pytest

from vlm.probe_checkpoint import main, sha, verify_train_saved, wait_train_saved, write_train_done


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "a3"
    run.mkdir()
    for name, content in (("adapter.npz", b"synthetic adapter"), ("PLAN.json", b"{}"), ("rows.txt", b"synthetic\n")):
        (run / name).write_bytes(content)
    result = {"run_name": run.name, "stub": True, "train": {"adapter_sha256": sha(run / "adapter.npz")},
              "regime": {"recipe": "synthetic"}, "pairs_input": {"snapshot_digest": "b" * 64}}
    write_train_done(run, result, contract_sha256="c" * 64)
    names = [f"{run.name}/{n}" for n in ("adapter.npz", "TRAIN_DONE.json", "PLAN.json", "rows.txt")]
    names += ["RUN_CONTRACT.json", "ENV_RECEIPT.json"]
    receipt = {"kind": "train_saved", "name": run.name, "ok": True, "contract_sha256": "c" * 64,
               "adapter_sha256": sha(run / "adapter.npz"), "train_done_sha256": sha(run / "TRAIN_DONE.json"),
               "tar": {"bytes": 100, "sha256": "d" * 64, "drive_sha256": "d" * 64},
               "members": [{"path": n, "bytes": 1} for n in names]}
    return run, result, receipt


def put(run, receipt):
    (run / "TRAIN_SAVED_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_metadata_survives_without_generation_result_and_cannot_be_overwritten(saved_run):
    run, result, _ = saved_run
    rec = json.loads((run / "TRAIN_DONE.json").read_bytes())
    assert not (run / "result.json").exists()
    assert rec["train"] == result["train"] and rec["pairs_input"] == result["pairs_input"]
    with pytest.raises(SystemExit, match="덮어쓰지"):
        write_train_done(run, result, contract_sha256="c" * 64)


def test_matching_complete_receipt_is_accepted(saved_run):
    run, _, receipt = saved_run
    put(run, receipt)
    a = wait_train_saved(run, 0.1, contract_sha256="c" * 64)
    assert a["ok"] is True and a == verify_train_saved(run, "c" * 64)


@pytest.mark.parametrize("case", ["failed", "other_run", "other_contract", "other_adapter", "other_done", "transfer_mismatch",
                                  "no_tar", "missing_member", "duplicate_member", "extra_nonmember", "nonstring_member",
                                  "changed_adapter", "changed_plan", "changed_rows"])
@pytest.mark.parametrize("mode", ["wait", "immediate"])
def test_receipt_and_live_file_mismatches_block_generation(saved_run, case, mode):
    run, _, receipt = saved_run
    rec = copy.deepcopy(receipt)
    if case == "failed":
        rec["ok"] = False
    elif case == "other_run":
        rec["name"] = "other"
    elif case == "other_contract":
        rec["contract_sha256"] = "a" * 64
    elif case == "other_adapter":
        rec["adapter_sha256"] = "a" * 64
    elif case == "other_done":
        rec["train_done_sha256"] = "a" * 64
    elif case == "transfer_mismatch":
        rec["tar"]["drive_sha256"] = "a" * 64
    elif case == "no_tar":
        rec.pop("tar")
    elif case == "missing_member":
        rec["members"].pop()
    elif case == "duplicate_member":
        rec["members"][-1] = rec["members"][0]
    elif case == "extra_nonmember":
        rec["members"].append(None)
    elif case == "nonstring_member":
        rec["members"][-1]["path"] = []
    else:
        name = {"changed_adapter": "adapter.npz", "changed_plan": "PLAN.json", "changed_rows": "rows.txt"}[case]
        (run / name).write_bytes(b"changed")
    put(run, rec)
    with pytest.raises(SystemExit):
        if mode == "wait":
            wait_train_saved(run, 0.1, contract_sha256="c" * 64)
        else:
            verify_train_saved(run, "c" * 64)


def test_timeout_keeps_training_outputs(saved_run):
    run, _, _ = saved_run
    with pytest.raises(SystemExit, match="시간 초과"):
        wait_train_saved(run, 0.001, contract_sha256="c" * 64)
    assert (run / "adapter.npz").is_file() and (run / "TRAIN_DONE.json").is_file()


def test_receipt_can_arrive_after_wait_starts(saved_run):
    run, _, receipt = saved_run

    def delayed():
        time.sleep(0.02)
        put(run, receipt)

    writer = threading.Thread(target=delayed)
    writer.start()
    try:
        assert wait_train_saved(run, 1.0, contract_sha256="c" * 64)["ok"]
    finally:
        writer.join()


def test_immediate_does_not_wait_when_receipt_is_missing(saved_run, monkeypatch):
    run, _, _ = saved_run

    def no_sleep(*args):
        raise AssertionError("즉시 검사에서 대기")

    monkeypatch.setattr(time, "sleep", no_sleep)
    with pytest.raises(SystemExit, match="읽을 수 없다"):
        verify_train_saved(run, "c" * 64)


@pytest.mark.parametrize("mode", ["wait", "immediate"])
def test_current_done_identity_is_checked_on_both_routes(saved_run, mode):
    run, _, receipt = saved_run
    done = json.loads((run / "TRAIN_DONE.json").read_bytes())
    done["run_name"] = "other"
    (run / "TRAIN_DONE.json").write_text(json.dumps(done))
    receipt["train_done_sha256"] = sha(run / "TRAIN_DONE.json")
    put(run, receipt)
    with pytest.raises(SystemExit):
        if mode == "wait":
            wait_train_saved(run, .1, contract_sha256="c" * 64)
        else:
            verify_train_saved(run, "c" * 64)


def test_module_cli_uses_current_contract_bytes(saved_run, tmp_path):
    run, _, receipt = saved_run
    contract = tmp_path / "contract.json"
    contract.write_bytes(b"{}")
    done_path = run / "TRAIN_DONE.json"
    done = json.loads(done_path.read_bytes())
    done["contract_sha256"] = sha(contract)
    done_path.write_text(json.dumps(done))
    receipt.update(contract_sha256=sha(contract), train_done_sha256=sha(done_path))
    put(run, receipt)
    assert main(["--verify", str(run), str(contract)]) == 0
    contract.write_bytes(b"{\"changed\":true}")
    with pytest.raises(SystemExit):
        main(["--verify", str(run), str(contract)])


def test_unreadable_receipt_does_not_pass(saved_run):
    run, _, _ = saved_run
    (run / "TRAIN_SAVED_RECEIPT.json").write_text("{")
    with pytest.raises(SystemExit, match="읽을 수 없다"):
        wait_train_saved(run, 0.1, contract_sha256="c" * 64)


def test_harness_blocks_generation_until_backup_and_keeps_recoverable_metadata(tmp_path, monkeypatch):
    """04 전체의 대역 학습 후 보존 실패. 생성 함수에 도달하면 실패한다."""
    import transformers

    from tests.probe_fakes import write_probe_snapshot
    from tests.test_probe_inputs import ROOT, load
    from tests.uni_fakes import make_rows

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    h = load("04_암기시험_C.py")
    rows = [{**r, "width_px": 64, "height_px": 32} for r in make_rows(tmp_path / "img", "C1", 1)]
    pairs, digest = write_probe_snapshot(tmp_path / "pairs", rows)
    ids = tmp_path / "ids.txt"
    ids.write_text(rows[0]["image_id"] + "\n")
    contract = tmp_path / "RUN_CONTRACT.json"
    contract.write_text("{}")

    def offline(*args, **kwargs):
        raise OSError("합성 시험은 토크나이저를 받지 않는다")

    def no_generation(*args, **kwargs):
        raise AssertionError("보존 확인 없이 생성했다")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", offline)
    monkeypatch.setattr(h, "stub_generator", no_generation)
    args = ["--stage", "1", "--run-name", "a2", "--out-root", str(tmp_path / "out"), "--stub",
            "--rows-file", str(ids), "--pairs", str(pairs), "--pairs-digest", digest, "--steps", "1",
            "--wait-train-saved", "1", "--run-contract", str(contract)]
    with pytest.raises(SystemExit, match="시간 초과"):
        h.main(args)
    run = tmp_path / "out/a2"
    done = json.loads((run / "TRAIN_DONE.json").read_bytes())
    assert done["pairs_input"]["snapshot_digest"] == digest and done["contract_sha256"] == sha(contract)
    assert h.adapter_regime(run / "adapter.npz", stub=True)["meta_file"] == "TRAIN_DONE.json"
    assert not (run / "result.json").exists()


def test_harness_plan_window_is_the_trainer_window():
    """04 의 계획 창은 학습기의 상수(마이크로 배치 × 누적)에서 온다 — 박아 둔 32 가 A3 재시험의 계획을 4 배로 셌다(10-31)."""
    from tests.test_probe_inputs import load
    from vlm import pilot_vlm

    h = load("04_암기시험_C.py")
    assert h._window() == pilot_vlm.MICRO_BATCH * pilot_vlm.GRAD_ACCUM
    assert not hasattr(h, "WINDOW")
    spe = -(-80 // h._window())                                          # A3: 80 행 · 60 에폭 · 노출 4,800
    assert (60 * spe) // spe * 80 == 4800 and 60 * spe == {8: 600, 32: 180}[pilot_vlm.GRAD_ACCUM]
