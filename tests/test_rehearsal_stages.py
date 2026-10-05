"""리허설 단계 본체 — 목록 단계의 쓰는 순서와 거부(2판 §5-4 · §6-1 의 35), 에코 관문에서 학습 전에 멈추기(2판 §1-2 의 E′).

목록 단계는 프로세스 안에서 부른다(쓰기 실패를 주입한다). 에코 관문은 단계 실행기로 E′ 까지 돌린다.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import uni_rehearsal_world as RW
from vlm.fault import FAULT_VAR, NONCE_VAR
from vlm.rehearsal_plan import load_plan
from vlm.rehearsal_run import StageRefused, stage_lists
from vlm.uni_config import load_uni_config

KST = timezone(timedelta(hours=9))
CLEAN = {k: v for k, v in os.environ.items() if k.upper() not in (FAULT_VAR, NONCE_VAR)}


@pytest.fixture
def w(tmp_path):
    w = RW.build(tmp_path)
    w.root = w.parent / "reh-20261001T120000"
    w.root.mkdir(parents=True)
    return w


def _lists(w, **kw):
    return stage_lists(root=w.root, plan=load_plan(w.plan, purpose="rehearsal"), config=load_uni_config(w.config),
                       snapshot=w.snap, checkout=w.repo, exposure_ledger=w.exposure, **kw)


def test_목록_단계는_사본_원장_줄_목록_순으로_쓴다(w):
    rec = _lists(w)
    from vlm.exposure import read_rows

    rows = read_rows(w.exposure)
    assert [(r["event"], r["list_kind"]) for r in rows] == [("planned", "gen"), ("planned", "echo")]
    copies = sorted(p.name for p in (w.exposure.parent / "lists").iterdir())
    assert copies == [f"{w.root.name}_echo.groups.json", f"{w.root.name}_echo.txt",
                      f"{w.root.name}_gen.groups.json", f"{w.root.name}_gen.txt"]
    assert (w.exposure.parent / "lists" / f"{w.root.name}_gen.txt").read_bytes() == \
        (w.root / "lists" / "gen.txt").read_bytes()
    groups = json.loads((w.exposure.parent / "lists" / f"{w.root.name}_gen.groups.json").read_text(encoding="utf-8"))
    assert set(groups) == set((w.root / "lists" / "gen.txt").read_text(encoding="utf-8").split())
    assert rec["excluded"] == {"n_components": 1, "n_groups": 1, "n_images": 2} and rec["before_3c"] is True
    assert sorted(p.name for p in (w.root / "lists").iterdir()) == [
        "echo.txt", "gen.txt", "lists_record.json", "train_uni_central.jsonl", "train_uni_local_C1.jsonl",
        "train_uni_local_C2.jsonl", "train_uni_local_C3.jsonl"]


def test_원장_줄을_쓰지_못하면_루트에_목록을_쓰지_않는다(w, monkeypatch):
    import vlm.exposure as X

    def boom(*a, **k):
        raise OSError("원장에 쓰지 못했다")

    monkeypatch.setattr(X, "append_row", boom)
    with pytest.raises(OSError):
        _lists(w)
    assert not (w.root / "lists").exists()


@pytest.mark.parametrize(("how", "code"), [("edges_sha", "edges_hash"), ("lists_exist", "lists_exist"),
                                           ("snapshot", "snapshot_not_anchor")])
def test_목록_단계의_거부(w, how, code):
    if how == "edges_sha":
        w.edges.write_text("a_id,b_id\nv10,e0\n", encoding="utf-8")
    elif how == "lists_exist":
        (w.root / "lists").mkdir()
    else:
        cfg = w.config.read_text(encoding="utf-8").replace(w.digest, "f" * 64)
        w.config.write_text(cfg, encoding="utf-8")
    with pytest.raises(StageRefused) as exc:
        _lists(w)
    assert exc.value.code == code
    assert not w.exposure.exists()


def test_에코_카나리아가_통과가_아니면_학습을_시작하지_않는다(tmp_path):
    from scripts import rehearsal_uni as RU

    w = RW.build(tmp_path)
    # 페어 val 행 한 장의 박스를 주석과 다르게 둔다 — 에코는 학습 타깃을 그대로 내므로 정답과 어긋난다
    rows = [json.loads(x) for x in w.pairs.read_text(encoding="utf-8").splitlines()]
    v = next(r for r in rows if r["split"] == "val" and r["image_id"] == "v20")
    v["skeleton"]["defects"][0]["bbox_px"] = [1, 2, 30, 40]
    w.pairs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    import hashlib

    h = hashlib.sha256(w.pairs.read_bytes()).hexdigest()
    d = hashlib.sha256(h.encode()).hexdigest()
    (w.pairs.parent / "SNAPSHOT.sha256").write_text(f"{h}  pairs.jsonl\n# snapshot_digest {d}\n", encoding="utf-8")
    cfg = w.config.read_text(encoding="utf-8")
    import re

    w.config.write_text(re.sub(r"digest: [0-9a-f]{64}\}", f"digest: {d}}}", cfg), encoding="utf-8")
    RW.git(w.repo, "commit", "-q", "-am", "페어 지문")
    code = RU.main(["run", "--plan", str(w.plan), "--snapshot", str(w.snap), "--device", "cpu:0"],
                   checkout=w.repo, exposure_ledger=w.exposure, scoring_ledger=w.scoring, non_main_parent=w.parent,
                   launcher=RW.child_cmd(tmp_path), env=dict(CLEAN), now=datetime(2026, 10, 1, 12, 0, tzinfo=KST))
    root = w.parent / "reh-20261001T120000"
    rows = [json.loads(x) for x in (root / "rehearsal_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert code == RU.EXIT_HALT
    assert [r["name"] for r in rows if "seq" in r] == ["lists", "register", "echo_export", "echo_score"]
    gate = next(r for r in rows if r.get("kind") == "echo_gate")
    assert gate["statuses"] and gate["statuses"] != ["pass"]
    assert not (root / "train").exists()


# ================================================================ 학습 행 목록의 결속(외부 검토 1)
@pytest.fixture
def registered(w):
    from vlm.rehearsal_run import stage_register

    plan = load_plan(w.plan, purpose="rehearsal")
    cfg = load_uni_config(w.config)
    _lists(w)
    stage_register(root=w.root, plan=plan, config=cfg, checkout=w.repo, measure_env=RW.measure,
                   model_loader=RW.fake_model_loader, generator_loader=RW.generator_loader,
                   now=datetime(2026, 10, 1, 12, 0, tzinfo=KST), standin_allowed=True)
    return w, plan, cfg


def _train(w, plan, cfg, loader, tag="uni_local_C1"):
    from vlm.rehearsal_run import stage_train

    return stage_train(root=w.root, plan=plan, config=cfg, tag=tag, snapshot=w.snap, model_loader=loader,
                       standin_allowed=True, non_main_parent=w.parent, config_path=w.config)


def _rewrite(w, tag, rows, *, rehash: bool):
    import hashlib

    name = f"train_{tag}.jsonl"
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    (w.root / "lists" / name).write_bytes(raw)
    if rehash:                                   # 기록까지 고쳐 맞춘 경우 — 행의 원본 대조가 잡아야 한다
        rp = w.root / "lists" / "lists_record.json"
        rec = json.loads(rp.read_text(encoding="utf-8"))
        rec["files"][name] = hashlib.sha256(raw).hexdigest()
        rp.write_text(json.dumps(rec), encoding="utf-8")


@pytest.mark.parametrize(("how", "code"), [
    ("bytes", "train_list_hash"),               # 목록만 바꿨다
    ("image_path", "train_list_row"),           # 행의 이미지 경로를 다른 이미지(평가 쪽)로 바꾸고 기록도 맞췄다
    ("eval_id", "train_list_split"),            # 평가 id 의 행을 더하고 기록도 맞췄다
    ("other_client", "train_list_client"),      # 다른 참여자의 행을 더했다
])
def test_바꿔_끼운_학습_행_목록은_모델을_올리기_전에_거부한다(registered, how, code):
    from tests.uni_fakes import CountingLoader, fake_loader

    w, plan, cfg = registered
    tag = "uni_local_C1"
    rows = [json.loads(x) for x in (w.root / "lists" / f"train_{tag}.jsonl").read_text(encoding="utf-8").splitlines()]
    if how == "bytes":
        _rewrite(w, tag, rows[:-1] or rows + rows, rehash=False)
    elif how == "image_path":
        rows[0] = {**rows[0], "image_path": str(w.tmp / "img" / "v00.png")}
        _rewrite(w, tag, rows, rehash=True)
    elif how == "eval_id":
        _rewrite(w, tag, rows + [{**rows[0], "image_id": "e0"}], rehash=True)
    else:
        other = next(json.loads(x) for x in w.pairs.read_text(encoding="utf-8").splitlines()
                     if json.loads(x)["client"] == "C2" and json.loads(x)["split"] == "train")
        _rewrite(w, tag, rows + [other], rehash=True)
    loader = CountingLoader(fake_loader)
    with pytest.raises(StageRefused) as exc:
        _train(w, plan, cfg, loader, tag)
    assert exc.value.code == code and loader.n == 0
    assert not (w.root / "train" / f"{tag}_s1" / "adapter_last.npz").exists()


def test_목록_단계가_만든_학습_행은_결속을_지나_학습한다(registered):
    from tests.uni_fakes import CountingLoader, fake_loader

    w, plan, cfg = registered
    loader = CountingLoader(fake_loader)
    res = _train(w, plan, cfg, loader)
    assert res.status == "trained" and loader.n >= 1


# ================================================================ 목록 · 등록의 중단 변수 가드(외부 검토 7)
@pytest.mark.parametrize("cmd", ["lists", "register"])
@pytest.mark.parametrize("env_kind", ["half", "nonce_only", "planned_other_stage"])
def test_목록과_등록은_중단_변수가_어긋나면_쓰기_전에_거부한다(w, cmd, env_kind, capsys):
    from scripts import rehearsal_uni as RU
    from vlm.fault import write_planned

    shutil = __import__("shutil")
    plan_in_root = w.root / "plan.yaml"
    shutil.copyfile(w.plan, plan_in_root)
    nonce = "3" * 32
    if env_kind == "half":
        env = {**CLEAN, FAULT_VAR: "train:after_ckpt:uni_central:ep=0"}
    elif env_kind == "nonce_only":
        env = {**CLEAN, NONCE_VAR.lower(): nonce}
    else:
        fault = "train:after_ckpt:uni_central:ep=0"
        write_planned(w.root, nonce, fault, stage_no=1)
        env = {**CLEAN, FAULT_VAR: fault, NONCE_VAR: nonce}
    argv = [cmd, "--run-root", str(w.root), "--plan", str(plan_in_root)] + (["--snapshot", str(w.snap)] if cmd == "lists" else [])
    code = RU.main(argv, measure_env=RW.measure, model_loader=RW.fake_model_loader, generator_loader=RW.generator_loader,
                   checkout=w.repo, exposure_ledger=w.exposure, non_main_parent=w.parent, config_path=w.config,
                   env=env, standin_allowed=True)
    err = capsys.readouterr().err
    assert code == RU.EXIT_REFUSED and err.startswith(("[fault_env_half]", "[fault_not_for_this_call]")), err
    assert not (w.root / "lists").exists() and not (w.root / "registration").exists() and not w.exposure.exists()



def test_에코의_기대_박스가_0_이면_상태가_통과여도_학습을_시작하지_않는다(tmp_path):
    """결함 없는 val 장만이면 평가 쪽 에코는 기대 · TP · FP · FN 이 모두 0 인 채 '통과' 를 낸다 — 비어 있는 항등 검사다(외부 검토 6)."""
    from scripts import rehearsal_uni as RU

    w = RW.build(tmp_path, val_defects=False)
    code = RU.main(["run", "--plan", str(w.plan), "--snapshot", str(w.snap), "--device", "cpu:0"],
                   checkout=w.repo, exposure_ledger=w.exposure, scoring_ledger=w.scoring, non_main_parent=w.parent,
                   launcher=RW.child_cmd(tmp_path), env=dict(CLEAN), now=datetime(2026, 10, 1, 12, 0, tzinfo=KST))
    root = w.parent / "reh-20261001T120000"
    rows = [json.loads(x) for x in (root / "rehearsal_log.jsonl").read_text(encoding="utf-8").splitlines()]
    gate = next(r for r in rows if r.get("kind") == "echo_gate")
    assert code == RU.EXIT_HALT and gate["statuses"] == ["pass"] and gate["identity"] == [False]
    assert not (root / "train").exists()


def test_등록은_이음새가_없으면_두_모드가_같이_지나는_실제_적재_함수의_열쇠를_적는다(w):
    """등록의 구현 식별자 한 칸에 모델 · 에코가 같이 지나는 적재 함수가 실린다(외부 검토 4) — 모델 생성기 이음새를 주지 않았다."""
    from evaluation.prereg_unified import load_registration
    from scripts import rehearsal_uni as RU

    shutil = __import__("shutil")
    plan_in_root = w.root / "plan.yaml"
    shutil.copyfile(w.plan, plan_in_root)
    common = {"measure_env": RW.measure, "model_loader": RW.fake_model_loader, "checkout": w.repo,
              "exposure_ledger": w.exposure, "non_main_parent": w.parent, "config_path": w.config, "env": dict(CLEAN),
              "standin_allowed": True, "now": datetime(2026, 10, 1, 12, 0, tzinfo=KST)}
    assert RU.main(["lists", "--run-root", str(w.root), "--plan", str(plan_in_root), "--snapshot", str(w.snap)],
                   **common) == RU.EXIT_OK
    assert RU.main(["register", "--run-root", str(w.root), "--plan", str(plan_in_root)], **common) == RU.EXIT_OK
    reg_p = next((w.root / "registration").glob("rehearsal-*-1.json"))
    keys = load_registration(reg_p.read_bytes()).generation.impl_ids
    assert any(k.startswith("export_generator=vlm.export_generator:load_generator@") for k in keys), keys
    for seam in ("export_generator_echo", "export_generator_model", "model_loader"):         # 안쪽 구현 셋도 등록된다
        assert any(k.startswith(f"{seam}=") for k in keys), (seam, keys)



def test_목록_단계_함수도_스스로_중단_변수를_본다(w):
    """명령줄 가드와 따로 — 단계 함수를 바로 불러도 쓰기 전에 거부한다(외부 검토 7 의 남은 구별)."""
    with pytest.raises(StageRefused) as exc:
        _lists(w, env={**CLEAN, FAULT_VAR: "train:after_ckpt:uni_central:ep=0"})
    assert exc.value.code == "fault_env_half"
    assert not (w.root / "lists").exists() and not w.exposure.exists()
