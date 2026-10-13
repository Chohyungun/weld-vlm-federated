"""리허설 오케스트레이터 — 단계 실행기(2판 §1-3 의 장치)와 합성 세계 위의 단계 표 한 바퀴(2판 §1-2 · §6-1 의 13 · 27 · 28 · 30 · 35 · 45).

단계 실행기 시험은 작은 자식(`python -c`)으로 본다 — 환경 지우기(소문자 이름 포함) · 계획 기록이 자식보다 먼저 · 의도한 중단의 다섯 조건 ·
오래된 표지 · 시간 제한의 트리 끊기 · 실행 기록. 한 바퀴 시험은 `tests/uni_rehearsal_world.py` 의 세계에서 실제 진입점을 자식으로 돌린다
(작은 모델 대역 · 가짜 생성기). 실제 `outputs/` 를 건드리지 않는다 — 모든 경로를 파이썬 인자로 임시 폴더에 준다.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import rehearsal_uni as RU
from vlm.fault import EXIT_CODE, FAULT_VAR, NONCE_VAR
from vlm.rehearsal_orch import ANY_NONZERO, LOG_FILE, OrchestratorRefused, Stage, run_stage

KST = timezone(timedelta(hours=9))
CLEAN = {k: v for k, v in os.environ.items() if k.upper() not in (FAULT_VAR, NONCE_VAR)}


def _py(code: str):
    """자식 = `python -c <code>`. 인자는 쓰지 않는다."""
    def launcher(entry, argv):
        return [sys.executable, "-X", "utf8", "-c", textwrap.dedent(code)]
    return launcher


def _rows(root: Path) -> list[dict]:
    return [json.loads(x) for x in (root / LOG_FILE).read_text(encoding="utf-8").splitlines()]


FIRE = """
import json, os, sys, hashlib
from pathlib import Path
root = Path(os.environ["ROOT"])
f, n = os.environ.get("WELD_REHEARSAL_FAULT"), os.environ.get("WELD_REHEARSAL_NONCE")
planned = root / "faults" / "planned" / f"{n}.json"
assert planned.exists(), "계획 기록이 자식보다 늦다"
d = root / "faults"
seq = len(list(d.glob("*.activated.json"))) + 1
(d / f"{seq:03d}_x.activated.json").write_text(json.dumps({"nonce": n, "fault": f}), encoding="utf-8")
(d / f"{seq:03d}_x.fired.json").write_text("{}", encoding="utf-8")
os._exit(%d)
"""


def test_계획된_단계는_표식과_함께_변수를_받고_의도한_중단이면_통과다(tmp_path):
    env = {**CLEAN, "ROOT": str(tmp_path)}
    st = Stage("1", "fire", "train", (), 0, "train", fault="train:after_ckpt:uni_central:ep=0", tag="uni_central")
    res = run_stage(st, root=tmp_path, seq=1, launcher=_py(FIRE % EXIT_CODE), env_base=env, timeout_s=60)
    assert res.ok and res.exit_code == EXIT_CODE and res.row["intended_fault"], res.stderr
    row = _rows(tmp_path)[-1]
    assert row["nonce"] and len(row["nonce"]) == 32 and row["activated"][0]["nonce"] == row["nonce"]
    assert row["planned_sha256"] and row["stderr_sha256"] and row["expect"] == "fault_75"


def test_중단을_건_단계가_0_으로_끝나면_실패다(tmp_path):
    st = Stage("1", "nofire", "train", (), 0, "train", fault="train:after_ckpt:uni_central:ep=0")
    res = run_stage(st, root=tmp_path, seq=1, launcher=_py("raise SystemExit(0)"), env_base=CLEAN, timeout_s=60)
    assert not res.ok and not res.row["intended_fault"]


def test_표식이_다른_발화는_의도한_중단으로_세지_않는다(tmp_path):
    code = FIRE.replace('"nonce": n', '"nonce": "0" * 32') % EXIT_CODE
    st = Stage("1", "wrong", "train", (), 0, "train", fault="train:after_ckpt:uni_central:ep=0")
    res = run_stage(st, root=tmp_path, seq=1, launcher=_py(code), env_base={**CLEAN, "ROOT": str(tmp_path)}, timeout_s=60)
    assert not res.ok and res.exit_code == EXIT_CODE


def test_중단을_걸지_않은_단계에_새_기록이_생기면_실패고_앞_기록은_세지_않는다(tmp_path):
    d = tmp_path / "faults"
    d.mkdir()
    (d / "001_old.activated.json").write_text('{"nonce": "x"}', encoding="utf-8")      # 앞 호출의 표지
    ok = run_stage(Stage("2", "quiet", "train", (), 0, "train"), root=tmp_path, seq=1,
                   launcher=_py("raise SystemExit(0)"), env_base=CLEAN, timeout_s=60)
    assert ok.ok and ok.row["activated"] == []
    code = ("from pathlib import Path; import os; "
            f"Path(r'{d}', '002_new.activated.json').write_text('{{}}'); raise SystemExit(0)")
    bad = run_stage(Stage("2", "noisy", "train", (), 0, "train"), root=tmp_path, seq=2, launcher=_py(code),
                    env_base=CLEAN, timeout_s=60)
    assert not bad.ok


@pytest.mark.parametrize("name", [FAULT_VAR, FAULT_VAR.lower(), NONCE_VAR.lower()])
def test_계획되지_않은_자식에는_부모에_심은_변수가_없다(tmp_path, name):
    code = "import os, sys; sys.exit(3 if any(k.upper().startswith('WELD_REHEARSAL') for k in os.environ) else 0)"
    res = run_stage(Stage("0", "env", "lists", (), 0, "lists"), root=tmp_path, seq=1, launcher=_py(code),
                    env_base={**CLEAN, name: "x"}, timeout_s=60)
    assert res.ok and res.exit_code == 0


def test_기대_종료_코드의_세_꼴(tmp_path):
    for k, (expect, code, ok) in enumerate(((ANY_NONZERO, 2, True), (ANY_NONZERO, EXIT_CODE, False),
                                            (ANY_NONZERO, 0, False), ((0, 2), 2, True), ((0, 2), 3, False),
                                            (2, 2, True)), 1):
        res = run_stage(Stage("x", f"e{code}", "score", (), expect, "score"), root=tmp_path, seq=k,
                        launcher=_py(f"raise SystemExit({code})"), env_base=CLEAN, timeout_s=60)
        assert res.ok is ok, (expect, code)


def test_시간_제한을_넘기면_프로세스_트리째_끊고_실패다(tmp_path):
    pidf = tmp_path / "grandchild.pid"
    code = f"""
    import subprocess, sys, time
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open(r"{pidf}", "w").write(str(p.pid))
    time.sleep(60)
    """
    res = run_stage(Stage("x", "slow", "score", (), 0, "score"), root=tmp_path, seq=1, launcher=_py(code),
                    env_base=CLEAN, timeout_s=5)
    assert not res.ok and res.row["killed"]
    import subprocess

    pid = int(pidf.read_text())
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True,
                         check=False).stdout                       # 콘솔 인코딩이 cp949 라 바이트로 본다
    assert f'"{pid}"'.encode() not in out, "손자 프로세스가 남았다"


def test_실행_기록의_끝이_LF_가_아니면_덧붙이지_않는다(tmp_path):
    (tmp_path / LOG_FILE).write_bytes(b'{"a": 1}')
    with pytest.raises(OrchestratorRefused) as exc:
        run_stage(Stage("0", "x", "lists", (), 0, "lists"), root=tmp_path, seq=1, launcher=_py("pass"),
                  env_base=CLEAN, timeout_s=60)
    assert exc.value.code == "log_broken"


# ================================================================ 시작 전 거부
def test_오케스트레이터는_자기_환경에_중단_변수가_있으면_루트를_만들지_않는다(tmp_path):
    for name in (FAULT_VAR, NONCE_VAR.lower()):
        with pytest.raises(OrchestratorRefused) as exc:
            RU.run_rehearsal(plan_path=ROOT / "configs/rehearsal_uni.yaml", snapshot=tmp_path, device="cpu",
                             parent=tmp_path / "p", checkout=tmp_path, env={**CLEAN, name: "x"})
        assert exc.value.code == "fault_env"
    assert not (tmp_path / "p").exists()


STAGES = ("lists", "register", "train", "export", "score")
EDGES = {"path": "x/edges.csv", "sha256": "e" * 64}


def _synth_plan(tmp_path, *, timeouts, edges):
    """저장소의 리허설 계획에서 시한 · 간선 칸만 바꾼 합성 계획 — 저장소 계획의 **지금 값**에 기대지 않는다."""
    import yaml

    doc = yaml.safe_load((ROOT / "configs/rehearsal_uni.yaml").read_text(encoding="utf-8"))
    doc["timeouts_s"], doc["exclude"]["near_dup_edges"] = timeouts, edges
    p = tmp_path / "plan_synth.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


@pytest.mark.parametrize(("timeouts", "edges", "code", "word"), [
    (None, EDGES, "plan_value_missing", "timeouts_s"),
    (dict.fromkeys(STAGES), None, "plan_value_missing", "near_dup_edges"),
    (dict.fromkeys(STAGES, 60), None, "plan_value_missing", "near_dup_edges"),
    (dict.fromkeys(STAGES), EDGES, "timeouts", "lists"),
], ids=["timeouts_null", "both_values_null", "edges_null", "timeout_values_null"])
def test_빈_계획_값은_루트를_만들기_전에_멈춘다(tmp_path, capsys, timeouts, edges, code, word):
    """칸을 명시적으로 비운 합성 계획 — 시한 칸 자체가 비거나 간선이 비면 `plan_value_missing`, 시한의 단계 값이 비면 `timeouts` 다."""
    plan = _synth_plan(tmp_path, timeouts=timeouts, edges=edges)
    rc = RU.main(["run", "--plan", str(plan), "--snapshot", str(tmp_path), "--device", "cpu"],
                 non_main_parent=tmp_path / "p", checkout=tmp_path, env=dict(CLEAN))
    err = capsys.readouterr().err
    assert rc == RU.EXIT_REFUSED and err.startswith(f"[{code}]") and word in err, err
    assert not (tmp_path / "p").exists()


def test_저장소의_리허설_계획을_읽는다():
    """저장소 계획은 읽는 쪽의 꼴 검사를 지난다 — 빈 칸이 채워졌는지는 보지 않는다(그 값이 필요한 단계가 따로 본다)."""
    from vlm.rehearsal_plan import load_plan

    p = load_plan(ROOT / "configs/rehearsal_uni.yaml", purpose="rehearsal")
    assert p.purpose == "rehearsal" and "uni_fed" in p.get("cells")


def test_명령줄에는_구현을_고르는_인자가_없다():
    opts = set()
    for a in RU._parser()._actions:
        for sub in getattr(a, "choices", {}).values() if hasattr(a, "choices") and isinstance(a.choices, dict) else ():
            opts |= {s for x in sub._actions for s in x.option_strings}
    assert opts and not {o for o in opts if any(k in o for k in ("measure", "generator", "loader", "seam", "impl", "fake"))}


# ================================================================ 한 바퀴(합성 세계)
@pytest.fixture(scope="module")
def lap(tmp_path_factory):
    from tests import uni_rehearsal_world as RW

    tmp = tmp_path_factory.mktemp("lap")
    w = RW.build(tmp, cells="uni_central, uni_local_C1, uni_local_C2, uni_local_C3, uni_fed")
    code = RU.main(["run", "--plan", str(w.plan), "--snapshot", str(w.snap), "--device", "cpu:0"],
                   checkout=w.repo, exposure_ledger=w.exposure, scoring_ledger=w.scoring, non_main_parent=w.parent,
                   launcher=RW.child_cmd(tmp), env=dict(CLEAN), now=datetime(2026, 10, 1, 12, 0, tzinfo=KST),
                   flwr_procs=list, flwr_stop=lambda found: None)        # 부모 쪽 연합 정리 — 이 시험은 실제 프로세스를 보지 않는다
    root = w.parent / "reh-20261001T120000"
    return w, root, code


def test_한_바퀴가_단계_표를_기대대로_끝낸다(lap):
    _w, root, code = lap
    assert code == RU.EXIT_OK
    rows = _rows(root)
    stages = [r for r in rows if "seq" in r]
    assert all(r["ok"] for r in stages), [r["name"] for r in stages if not r["ok"]]
    names = [r["name"] for r in stages]
    assert names[:4] == ["lists", "register", "echo_export", "echo_score"]
    assert [r["stage_no"] for r in stages if r["name"].startswith("export_uni_central")] == ["3", "4", "5", "6"]
    faults = [r for r in stages if r["fault"]]
    assert [r["fault"] for r in faults] == ["train:after_ckpt:uni_central:ep=0", "export:after_lines:uni_central:n=1",
                                            "export:torn_line:uni_central:n=2", "train:before_ckpt:uni_local_C3:ep=0",
                                            "export:after_all_lines:uni_local_C3:-"]
    assert all(r["intended_fault"] and r["exit_code"] == EXIT_CODE for r in faults)
    assert all(not r["activated"] for r in stages if not r["fault"])
    kinds = [r.get("kind") for r in rows if "kind" in r]
    assert kinds[0] == "start" and kinds[-1] == "end" and "echo_gate" in kinds and "negatives_made" in kinds
    assert [r for r in rows if r.get("kind") == "not_run"] == []                # 연합 칸도 돈다(줄 B)
    # 단계 이름은 판정기와 같은 정본에서 나온다 — 연합 칸의 두 단계만 빠졌다(아직 돌지 않는다)
    from vlm.rehearsal_orch import expected_stages
    from vlm.rehearsal_plan import load_plan

    want = expected_stages(load_plan(root / "plan.yaml", purpose="rehearsal"))
    assert [n for n in want if n not in names] == []
    assert [n for n in names if n not in want] == []
    fed = [r for r in stages if r["name"] == "train_uni_fed"]
    assert len(fed) == 1 and fed[0]["exit_code"] == 0 and not fed[0]["fault"]         # 연합 칸은 중단을 넣지 않는다
    cleanup = [r for r in rows if r.get("kind") == "fed_cleanup"]
    assert len(cleanup) == 1 and cleanup[0]["ok"] is True and cleanup[0]["stage_failed"] is False   # 부모가 정리했다
    assert (root / "train" / "uni_fed_s1" / "adapter_last.npz").is_file()
    assert (root / "export" / "uni_fed_s1.export_meta.json").is_file()
    # 연합 클라이언트는 목록 단계의 행(로컬 칸과 같은 행)으로 학습했다 — 참여자 행 전체가 아니다(목록은 그 일부다)
    fed_meta = json.loads((root / "train" / "uni_fed_s1" / "adapter_last.meta.json").read_text(encoding="utf-8"))
    local = {c: json.loads((root / "train" / f"uni_local_{c}_s1" / "adapter_last.meta.json").read_text(encoding="utf-8"))
             ["train_rows_digest"] for c in ("C1", "C2", "C3")}
    assert fed_meta["train_rows_digest"] == local
    assert fed_meta["identity"]["purpose"] == "rehearsal" and fed_meta["adapter_step"] == 2


def test_구판_넷은_적용_전에_거부되고_다른_스탬프는_모델_전에_거부된다(lap):
    _w, root, _ = lap
    rows = _rows(root)
    made = next(r for r in rows if r.get("kind") == "negatives_made")["cases"]
    for case, info in made.items():
        assert (root / info["path"]).read_bytes() and \
            __import__("hashlib").sha256((root / info["path"]).read_bytes()).hexdigest() == info["sha256"]
        r = next(x for x in rows if x.get("name") == f"negative_{case}")
        out = (root / r["stdout_path"]).read_text(encoding="utf-8", errors="replace")
        err = (root / r["stderr_path"]).read_text(encoding="utf-8", errors="replace")
        assert r["exit_code"] not in (0, EXIT_CODE) and "까지의 상태를 적용한다" not in out
        assert not list((root / "negative" / case / "train").glob("*/adapter_last.npz"))
        if case == "policy_blank":
            assert "신원이 현재 실행과 다르다" in err
    other = next(x for x in rows if x.get("name") == "overwrite_other_stamp")
    assert (root / other["stderr_path"]).read_text(encoding="utf-8").startswith("[identity_mismatch] ")
    for h in (r for r in rows if r.get("kind") == "hashes"):
        assert h["before"] == h["after"], h["around"]


def test_평가_쪽_채점이_다섯_묶음을_거부_없이_받는다(lap):
    _w, root, _ = lap
    s = json.loads((root / "score" / "score_unified_rehearsal.json").read_text(encoding="utf-8"))
    assert s["bundles"]["rejected"] == {}
    assert sorted(s["bundles"]["verified"]) == ["uni_central_s1", "uni_central_s1.echo", "uni_fed_s1", "uni_local_C1_s1",
                                                 "uni_local_C2_s1", "uni_local_C3_s1"]                # 연합 묶음도 거부 없이
    assert all(e["status"] == "pass" for e in s["canaries"]["echo"] + s["canaries"]["literal"])
    assert s["status"] == "stand_in"                      # 대역으로 돈 리허설은 통과를 내지 않는다
    assert {k: len(v) for k, v in s["bundles"]["fault_events"].items()} == {"uni_central_s1": 2, "uni_local_C3_s1": 1}


def test_노출_원장과_루트_밖_파일(lap):
    """노출 원장은 정본과 같은 꼴(체크아웃의 `outputs/exposure/`)이다. 실행이 만든 목록 사본은 새 파일이고, 실행 전에 있던 파일은 그대로다(2판 §3 ④)."""
    from vlm.exposure import read_rows

    w, root, _ = lap
    assert w.exposure == w.repo / "outputs" / "exposure" / "uni_exposure_ledger.jsonl"
    rows = read_rows(w.exposure)
    assert [(r["event"], r["list_kind"]) for r in rows[:2]] == [("planned", "gen"), ("planned", "echo")]
    gen = [r for r in rows if r["event"] == "generated"]
    assert sorted((r["tag"], r["mode"]) for r in gen) == [("uni_central", "echo"), ("uni_central", "model"),
                                                           ("uni_fed", "model"), ("uni_local_C1", "model"),
                                                           ("uni_local_C2", "model"), ("uni_local_C3", "model")]
    assert all(r["run_id"] == root.name for r in rows)
    end = _rows(root)[-1]
    assert end["kind"] == "end" and end["outputs_preexisting_preserved"] and all(end["ledger_prefix_kept"].values())
    assert end["outputs_preexisting_changed"] == [] and end["outputs_new_n"] >= 4
    assert {f"exposure/lists/{root.name}_gen.txt", f"exposure/lists/{root.name}_echo.txt"} <= set(end["outputs_new"])
    assert (w.repo / "outputs" / "keep" / "old.txt").read_text(encoding="utf-8") == "앞 실행의 산출물"
    lists = json.loads((root / "lists" / "lists_record.json").read_text(encoding="utf-8"))
    assert lists["excluded"]["n_groups"] == 1 and lists["eval_intersection"] == 0
    assert not {"v00", "v01"} & set((root / "lists" / "gen.txt").read_text(encoding="utf-8").split())


def test_실행_전_파일의_보존은_새_파일을_허용하고_바뀐_파일과_사라진_파일을_잡는다():
    from vlm.rehearsal_orch import preexisting_changes

    before = {"a": [1, 10], "b": [2, 20]}
    assert preexisting_changes(before, {**before, "new": [3, 30]}) == []
    assert preexisting_changes(before, {"a": [1, 11], "b": [2, 20]}) == ["a"]
    assert preexisting_changes(before, {"a": [1, 10]}) == ["b"]


def test_두_모드의_곁_파일이_등록의_구현_열쇠와_같다(lap):
    """모델 · 에코가 같은 적재 함수를 지나므로 등록의 구현 식별자 한 칸과 두 곁 파일이 모두 맞는다(외부 검토 4)."""
    from evaluation.actuals import check_actuals
    from vlm.rehearsal_run import local_registration

    _w, root, _ = lap
    reg, _, _, _ = local_registration(root)
    for stem, mode in (("uni_central_s1", "model"), ("uni_central_s1.echo", "echo")):
        side = json.loads((root / "export" / f"{stem}.export_meta.json").read_text(encoding="utf-8"))
        assert [m.item for m in check_actuals(reg, side, mode=mode) if m.item == "impl_ids"] == [], stem


# ================================================================ 판정기(같은 한 바퀴)
def _judge(root, w, **kw):
    from vlm.rehearsal_judge import judge

    return judge(root, checkout=w.repo, exposure_ledger=w.exposure, **kw)


FED_MISSING = {"meta:uni_fed", "export:uni_fed", "stage:train_uni_fed", "stage:export_uni_fed_1"}
FED_CHECKS = {"①": ["stage_table_ok"], "②": ["env_same_all_cells", "literal_pass_all_models", "axis_ratio_called"],
              "③": ["all_bundles_verified"], "④": [], "⑤": ["sidecars_have_batch_and_device"]}


def test_판정기는_목록_기록의_미검증_층을_옮겨_싣는다(lap, tmp_path):
    """결정 21 의 3 — 목록 기록의 `unverified_strata` 가 비지 않으면 판정 산출에 그대로 실린다(판정기가 빈 목록을 박아 두지 않는다)."""
    import shutil

    w, root, _ = lap
    dst = tmp_path / root.name
    shutil.copytree(root, dst, ignore=shutil.ignore_patterns("judge.json"))
    rec_p = dst / "lists" / "lists_record.json"
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    rec["unverified_strata"] = [{"stratum": "AL|crack", "reason": "제외(R0) 뒤 후보 묶음 0 — 다른 곳에서 채우지 않는다"}]
    rec_p.write_text(json.dumps(rec, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    _judge(dst, w)
    got = json.loads((dst / "judge.json").read_text(encoding="utf-8"))["lists"]["unverified_strata"]
    assert got == rec["unverified_strata"]


def test_다섯_칸이_다_돈_한_바퀴는_빠진_것이_없고_대역_실행은_판정하지_않는다(lap):
    """한 바퀴가 다섯 칸을 다 돌았다 — 판정기가 빠진 칸 · 단계를 찾지 못하고, 관측은 모두 선다. 이음새가 대역이라 판정은 `stand_in` 이다."""
    import hashlib

    w, root, _ = lap
    code = RU.main(["judge", "--run-root", str(root)], checkout=w.repo, exposure_ledger=w.exposure)
    assert code == RU.EXIT_OK
    out = json.loads((root / "judge.json").read_text(encoding="utf-8"))
    assert out["status"] == "stand_in" and out["incomplete"] == []
    assert out["lists"]["unverified_strata"] == []                      # 이 세계는 제외 뒤에도 빈 층이 없다(결정 21 의 3 — 칸은 있다)
    assert not out["impl"]["approved"] and out["impl"]["scorer_stand_in"] is True
    assert all(c["status"] == "stand_in" for c in out["conditions"].values())
    failing = {k: sorted(n for n, v in c["checks"].items() if v is not True) for k, c in out["conditions"].items()}
    assert failing == {k: [] for k in out["conditions"]}, failing
    assert out["inputs"]["rehearsal_log.jsonl"] == hashlib.sha256((root / LOG_FILE).read_bytes()).hexdigest()
    assert out["stages"]["planned_faults"] == out["stages"]["intended_faults"] == 5
    assert out["stages"]["not_run"] == [] and out["lists"]["before_3c"] is True
    assert out["report_only"]["torn_tail_prefix"][0]["prefix_same"] is True
    assert out["conditions"]["②"]["axis_ratio"]["called"] is True
    assert RU.main(["judge", "--run-root", str(root)], checkout=w.repo) == RU.EXIT_REFUSED      # 덮지 않는다


def _drop_fed(dst):
    """연합 칸을 걷는다 — 실제로 다섯 칸을 돈 한 바퀴의 사본에서 연합 칸의 학습 · export 산출과 단계 줄,
    평가 쪽 산출의 연합 항목을 지운다. 판정기가 필수 칸의 빠짐을 잡는지를 본다."""
    import shutil

    shutil.rmtree(dst / "train" / "uni_fed_s1")
    for q in (dst / "export").glob("uni_fed_s1.*"):
        q.unlink()
    sp = dst / "score" / "score_unified_rehearsal.json"
    s = json.loads(sp.read_text(encoding="utf-8"))
    s["bundles"]["verified"] = [b for b in s["bundles"]["verified"] if b != "uni_fed_s1"]
    for k in ("literal", "axis_ratio"):
        s["canaries"][k] = [e for e in s["canaries"][k] if e.get("bundle") != "uni_fed_s1"]
    sp.write_text(json.dumps(s), encoding="utf-8")
    _write_rows(dst, [r for r in _rows(dst) if r.get("tag") != "uni_fed"])


def test_판정기는_연합_칸이_빠지면_불통과다(lap, tmp_path):
    import shutil

    w, root, _ = lap
    dst = tmp_path / root.name
    shutil.copytree(root, dst, ignore=shutil.ignore_patterns("judge.json"))
    _drop_fed(dst)
    out = _judge(dst, w)
    assert out["status"] == "fail" and set(out["incomplete"]) == FED_MISSING
    failing = {k: sorted(n for n, v in c["checks"].items() if v is not True) for k, c in out["conditions"].items()}
    assert failing == {k: sorted(v) for k, v in FED_CHECKS.items()}
    assert out["conditions"]["②"]["axis_ratio"]["called"] is False


def _floats(obj):
    if isinstance(obj, bool):
        return []
    if isinstance(obj, float):
        return [obj]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _floats(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _floats(v)]
    return []


def test_판정_파일에는_지표_값이_없다(lap):
    _w, root, _ = lap
    out = json.loads((root / "judge.json").read_text(encoding="utf-8"))
    assert _floats(out) == [], "판정 파일에 실수 값이 있다 — 지표 값을 싣지 않는다"


def _write_rows(dst, rows):
    (dst / LOG_FILE).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _approved_scorer(dst):
    sp = dst / "score" / "score_unified_rehearsal.json"
    s = json.loads(sp.read_text(encoding="utf-8"))
    s["status"], s["bundles"]["stand_in"] = "pass", []
    sp.write_text(json.dumps(s), encoding="utf-8")


TAGS5 = ("uni_central", "uni_local_C1", "uni_local_C2", "uni_local_C3", "uni_fed")


def _axis_entry(stem, status="판정 불가", **kw):
    """평가 쪽 `axis_ratio_canary` 의 꼴 그대로 — 묶음 이름 · 상태 · 사유 · 표본 수 넷."""
    return {"bundle": stem, "status": status, "reason": "등록 칸 없음", "n_defect_images": 3, "n_pairs": 2,
            "n_pair_groups": 2, "n_pairs_unusable": 0} | kw


@pytest.fixture
def copied(lap, tmp_path, monkeypatch):
    """한 바퀴의 루트를 옮겨 놓고 흔든다. 승인된 구현으로 돈 것처럼 구현 검사와 평가 쪽 대역 판정을 바꿔 끼운다 — 판정 경로를 보려는 것이다.
    다섯 칸(연합 포함)은 한 바퀴가 실제로 돌았다."""
    import shutil

    import evaluation.actuals as A

    w, root, _ = lap
    dst = tmp_path / root.name
    shutil.copytree(root, dst, ignore=shutil.ignore_patterns("judge.json"))
    monkeypatch.setattr(A, "impl_problems", lambda v: [])
    _approved_scorer(dst)
    return w, dst


def test_승인_구현으로_돈_다섯_칸은_다섯_조건이_통과다(copied):
    w, dst = copied
    out = _judge(dst, w)
    assert out["status"] == "pass", {k: c["checks"] for k, c in out["conditions"].items() if c["status"] != "pass"}
    assert out["incomplete"] == []


def test_평가_쪽이_대역으로_적으면_판정기도_판정하지_않는다(copied):
    w, dst = copied
    sp = dst / "score" / "score_unified_rehearsal.json"
    s = json.loads(sp.read_text(encoding="utf-8"))
    s["bundles"]["stand_in"] = ["uni_central_s1.echo"]
    sp.write_text(json.dumps(s), encoding="utf-8")
    out = _judge(dst, w)
    assert out["status"] == "stand_in" and out["impl"]["scorer_stand_in"] is True


def test_연합_칸이_빠지면_승인_구현이어도_불통과다(lap, tmp_path, monkeypatch):
    import shutil

    import evaluation.actuals as A

    w, root, _ = lap
    dst = tmp_path / root.name
    shutil.copytree(root, dst, ignore=shutil.ignore_patterns("judge.json"))
    monkeypatch.setattr(A, "impl_problems", lambda v: [])
    _approved_scorer(dst)
    _drop_fed(dst)
    out = _judge(dst, w)
    assert out["status"] == "fail" and set(out["incomplete"]) == FED_MISSING


def test_축_배율은_호출의_누락과_판정_불가를_가른다(copied):
    """다섯 묶음의 결과가 계약의 꼴이고 상태가 '판정 불가' 면 경로가 돈 것이다 — 상태와 표본 수만 싣고 통과를 막지 않는다.
    블록이 없거나 상태만 든 객체 하나면 호출로 보지 않아 ② 가 떨어진다(외부 검토 6 · §39 · §40)."""
    w, dst = copied
    sp = dst / "score" / "score_unified_rehearsal.json"
    original = sp.read_bytes()
    # 어긋난 블록부터 — 고치기 전 판정기에도 있는 관측(`checks.axis_ratio_called`)만으로 본다
    for bad in ({"status": "판정 불가"}, [{"status": "판정 불가"}], None):
        s = json.loads(original)
        if bad is None:
            s["canaries"].pop("axis_ratio", None)
        else:
            s["canaries"]["axis_ratio"] = bad
        sp.write_text(json.dumps(s), encoding="utf-8")
        (dst / "judge.json").unlink(missing_ok=True)
        out = _judge(dst, w)
        assert out["conditions"]["②"]["checks"]["axis_ratio_called"] is False, bad
        assert out["conditions"]["②"]["status"] == "fail" and out["status"] == "fail"
    # 평가 쪽이 다섯 묶음에 실제로 낸 블록이면 통과한다
    sp.write_bytes(original)
    (dst / "judge.json").unlink()
    out = _judge(dst, w)
    ax = out["conditions"]["②"]["axis_ratio"]
    assert out["status"] == "pass" and ax["called"] is True and ax["problems"] == []
    assert {e["bundle"] for e in ax["entries"]} == {f"{t}_s1" for t in TAGS5}

STEMS5 = {f"{t}_s1" for t in TAGS5}


def _five(**kw):
    return [_axis_entry(s, **kw) for s in sorted(STEMS5)]


@pytest.mark.parametrize("case", ["d_shape_indeterminate", "zero_counts", "pass_and_fail_mixed"])
def test_축_배율_계약을_지킨_결과는_호출로_본다(case):
    from vlm.rehearsal_judge import axis_ratio_problems

    ax = {"d_shape_indeterminate": _five(),
          "zero_counts": _five(n_defect_images=0, n_pairs=0, n_pair_groups=0, n_pairs_unusable=0),
          "pass_and_fail_mixed": [_axis_entry(s, status=st, reason="")
                                  for s, st in zip(sorted(STEMS5), ["통과", "불통과", "판정 불가", "통과", "불통과"])]}[case]
    assert axis_ratio_problems(ax, STEMS5) == []


def _bad(case):
    ax = _five()
    if case == "status_only_object":
        return {"status": "판정 불가"}
    if case == "status_only_entries":
        return [{"bundle": s, "status": "판정 불가"} for s in sorted(STEMS5)]
    if case == "single_bundle":
        return ax[:1]
    if case == "empty_list":
        return []
    if case == "missing_count":
        del ax[2]["n_pairs"]
    elif case == "negative_count":
        ax[2]["n_pair_groups"] = -1
    elif case == "bool_count":
        ax[2]["n_defect_images"] = True
    elif case == "float_count":
        ax[2]["n_pairs_unusable"] = 0.0
    elif case == "string_count":
        ax[2]["n_pairs"] = "2"
    elif case == "unknown_status":
        ax[2]["status"] = "indeterminate"
    elif case == "other_seed":
        ax = [_axis_entry(s.replace("_s1", "_s2")) for s in sorted(STEMS5)]
    elif case == "duplicate_bundle":
        ax.append(dict(ax[0]))                # 집합은 같다 — 중복만이 어긋난다
    elif case == "extra_bundle":
        ax.append(_axis_entry("uni_central_s1.echo"))
    elif case == "missing_bundle_name":
        del ax[1]["bundle"]
    return ax


@pytest.mark.parametrize("case", ["status_only_object", "status_only_entries", "single_bundle", "empty_list",
                                  "missing_count", "negative_count", "bool_count", "float_count", "string_count",
                                  "unknown_status", "other_seed", "duplicate_bundle", "extra_bundle",
                                  "missing_bundle_name"])
def test_축_배율_계약에_어긋난_결과는_호출로_보지_않는다(case):
    from vlm.rehearsal_judge import axis_ratio_problems

    assert axis_ratio_problems(_bad(case), STEMS5) != []


def test_축_배율은_묶음별_목록만_받는다():
    """집계 객체 하나는 묶음 이름이 있어도 받지 않는다 — 목록으로 감싸 받으면 어느 묶음의 결과인지의 대조가 꼴에서 빠진다."""
    from vlm.rehearsal_judge import axis_ratio_problems

    one = _axis_entry("uni_central_s1")
    assert axis_ratio_problems([one], {"uni_central_s1"}) == []
    assert axis_ratio_problems(one, {"uni_central_s1"}) != []


def test_축_배율의_상태_어휘가_평가_쪽_상수와_같다():
    import evaluation.frame_diag as F
    from vlm.rehearsal_judge import AXIS_STATUSES

    assert AXIS_STATUSES == (F.PASS, F.FAIL, F.INDETERMINATE)


def test_평가_쪽_실제_축_배율_산출을_판정기가_받는다(monkeypatch):
    """평가 쪽 진입점의 `_axis_ratio`(묶음마다 `axis_ratio_canary` 를 부르는 실제 코드)가 낸 목록을 판정기의 계약에 그대로 넣는다.
    다섯 모델 묶음 · 등록 칸 없음(판정 불가)과 등록 칸 있음(통과 · 불통과) 둘. 평가 쪽 모듈은 읽기만 한다."""
    from types import SimpleNamespace as NS

    import scripts.probe.score_unified as SU
    from vlm.rehearsal_judge import AXIS_COUNTS, axis_ratio_problems

    ids = ["a", "b", "c", "d"]
    gold = {"a": [("2011", [100.0, 100.0, 200.0, 200.0])], "b": [("100", [300.0, 50.0, 400.0, 150.0])],
            "c": [("301", [10.0, 10.0, 60.0, 60.0])], "d": []}
    monkeypatch.setattr(SU, "_gold", lambda snapshot, want: {i: gold[i] for i in want})

    def line(bx):
        return json.dumps({"bbox_px_parsed": {"defects": [{"iso_code": "2011", "bbox_px": bx}]}})

    lines = [line([102.0, 98.0, 201.0, 203.0]), line([300.0, 52.0, 401.0, 149.0]), line([12.0, 9.0, 61.0, 58.0]),
             json.dumps({"bbox_px_parsed": {"defects": []}})]
    verified = [NS(mode=SU.MODE_MODEL, files=NS(stem=s), image_ids=ids, lines=lines) for s in sorted(STEMS5)]
    verified.append(NS(mode=SU.MODE_ECHO, files=NS(stem="uni_central_s1.echo"), image_ids=ids, lines=lines))
    meta = NS(group_of={i: f"g{i}" for i in ids})
    for thr, mp in ((None, None), (0.05, 1), (0.0, 1)):
        ctx = NS(args=NS(snapshot=Path("unused")), reg=NS(canary=NS(axis_ratio_threshold=thr, axis_ratio_min_pairs=mp)))
        out = SU._axis_ratio(ctx, verified, NS(ids=ids), meta)
        assert axis_ratio_problems(out, STEMS5) == [], (thr, out)
        assert all(type(e[c]) is int for e in out for c in AXIS_COUNTS)
    # 실제 산출에서 한 묶음을 빼거나 다른 시드의 기대와 맞대면 호출로 보지 않는다
    assert axis_ratio_problems(out[1:], STEMS5) != []
    assert axis_ratio_problems(out, {s.replace("_s1", "_s2") for s in STEMS5}) != []



def test_평가_쪽_상태만_대역이어도_판정기는_판정하지_않는다(copied):
    w, dst = copied
    sp = dst / "score" / "score_unified_rehearsal.json"
    s = json.loads(sp.read_text(encoding="utf-8"))
    s["status"] = "stand_in"                     # 묶음 목록은 비었다 — 상태 한 칸만으로도 대역 실행이다
    sp.write_text(json.dumps(s), encoding="utf-8")
    out = _judge(dst, w)
    assert out["status"] == "stand_in" and out["impl"]["scorer_stand_in"] is True


def test_에코_항등_검사는_기대_박스_수가_양수여야_한다():
    from vlm.rehearsal_judge import echo_identity_ok

    ok = {"status": "pass", "n_expected": 4, "tp": 4, "fp": 0, "fn": 0, "max_coord_diff": 0.0}
    assert echo_identity_ok(ok)
    for over in ({"n_expected": 0, "tp": 0}, {"tp": 3}, {"fp": 1}, {"fn": 1}, {"max_coord_diff": 0.5},
                 {"status": "fail"}):
        assert not echo_identity_ok({**ok, **over}), over
    assert not echo_identity_ok({"status": "pass"})


@pytest.mark.parametrize(("tamper", "cond", "check"), [
    ("neg_ckpt", "①", "negatives_checkpoint_unchanged"),
    ("neg_ledger", "①", "negatives_no_data_rows"),
    ("segments", "①", "segments_same_identity"),
    ("segments_none", "①", "segments_same_identity"),
    ("stage_missing", "①", "stage_table_ok"),
    ("hashes", "④", "root_hashes_same_6p"),
    ("grid", "②", "env_same_all_cells"),
    ("echo_empty", "②", "echo_identity"),
    ("line_key", "⑤", "lines_have_latency_length_stop"),
    ("rejected", "③", "no_rejected_bundle"),
])
def test_흔든_자리는_그_조건의_관측이_떨어지고_판정이_실패다(copied, tamper, cond, check):
    w, dst = copied
    if tamper == "neg_ckpt":
        made = next(r for r in _rows(dst) if r.get("kind") == "negatives_made")["cases"]
        p = dst / made["policy_blank"]["path"]
        p.write_bytes(p.read_bytes() + b"\0")
    elif tamper == "neg_ledger":
        led = dst / "negative" / "format_mismatch" / "train" / "uni_central_s1" / "train_ledger.csv"
        led.parent.mkdir(parents=True, exist_ok=True)
        led.write_text("run_id,seed\nx,1\n", encoding="utf-8")
    elif tamper in ("segments", "segments_none"):
        p = dst / "train" / "uni_central_s1" / "adapter_last.meta.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        if tamper == "segments":
            m["resume_segments"][1]["identity_sha256"] = "0" * 64
        else:                                    # 둘 다 없다 — 같다고 보지 않는다(외부 검토 5)
            for seg in m["resume_segments"]:
                seg["identity_sha256"] = None
        p.write_text(json.dumps(m), encoding="utf-8")
    elif tamper == "stage_missing":
        _write_rows(dst, [r for r in _rows(dst) if r.get("name") != "rerun_uni_central"])
    elif tamper == "hashes":
        rows = _rows(dst)
        h = next(r for r in rows if r.get("kind") == "hashes" and r["around"] == "6'")
        h["after"]["export/attempts.jsonl"] = "0" * 64
        _write_rows(dst, rows)
    elif tamper == "grid":
        p = dst / "export" / "uni_local_C2_s1.export_meta.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        m["image_grid_thw_observed"] = {"[1, 44, 80]": 4}
        p.write_text(json.dumps(m), encoding="utf-8")
    elif tamper == "echo_empty":                 # 기대 박스가 0 인데 상태만 통과(외부 검토 6)
        p = dst / "score" / "score_unified_rehearsal.json"
        s = json.loads(p.read_text(encoding="utf-8"))
        s["canaries"]["echo"][0].update(n_expected=0, tp=0, fp=0, fn=0, status="pass")
        p.write_text(json.dumps(s), encoding="utf-8")
    elif tamper == "line_key":
        p = dst / "export" / "uni_local_C1_s1.generations.jsonl"
        lines = p.read_text(encoding="utf-8").splitlines()
        r0 = json.loads(lines[0])
        r0.pop("latency_ms")
        p.write_text("\n".join([json.dumps(r0, ensure_ascii=False)] + lines[1:]) + "\n", encoding="utf-8")
    else:
        p = dst / "score" / "score_unified_rehearsal.json"
        s = json.loads(p.read_text(encoding="utf-8"))
        s["bundles"]["rejected"] = {"uni_local_C3_s1": ["adapter_duplicate"]}
        p.write_text(json.dumps(s), encoding="utf-8")
    out = _judge(dst, w)
    assert out["conditions"][cond]["checks"][check] is False
    assert out["conditions"][cond]["status"] == "fail" and out["status"] == "fail"


def test_생성_한도_경보는_계약의_멈춤_값으로_센다(copied):
    w, dst = copied
    p = dst / "export" / "uni_local_C2_s1.generations.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["gen_stop"] = "length"                  # 계약의 한도 종료 값(evaluation.schema_v14.GenStop)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out = _judge(dst, w)
    assert out["conditions"]["⑤"]["alarm_limit_hit_over_5pct"] is True
    assert out["conditions"]["⑤"]["status"] == "pass"                # 경보는 불통과가 아니다


def test_평가_쪽이_쓴_축_배율_파일을_판정기가_읽어_판정_파일에_싣는다(lap, tmp_path):
    """끝에서 끝 — 한 바퀴에서 평가 쪽 진입점이 **실제로 쓴** `score_unified_rehearsal.json` 을 손대지 않고 판정기에 넣는다.
    판정기가 쓴 `judge.json` 을 다시 읽어, 실은 항목이 평가 쪽 파일의 항목과 같고 계약의 꼴인지 본다. 한 바퀴가 다섯 칸(연합 포함)을 돌았으므로
    어긋남이 없어야 한다 — 어긋남(묶음 빠짐 · 상태 · 표본 수 · 다른 시드)이 있으면 연결이 끊긴 것이다."""
    import shutil

    from vlm.rehearsal_judge import AXIS_COUNTS, AXIS_STATUSES

    w, root, _ = lap
    s = json.loads((root / "score" / "score_unified_rehearsal.json").read_text(encoding="utf-8"))
    d_rows = s["canaries"]["axis_ratio"]
    ran = {f"{t}_s1" for t in TAGS5}
    assert {e["bundle"] for e in d_rows} == ran and len(d_rows) == len(ran)
    assert all(e["status"] in AXIS_STATUSES and all(type(e[c]) is int and e[c] >= 0 for c in AXIS_COUNTS)
               for e in d_rows)
    dst = tmp_path / root.name
    shutil.copytree(root, dst, ignore=shutil.ignore_patterns("judge.json"))
    _judge(dst, w)
    got = json.loads((dst / "judge.json").read_text(encoding="utf-8"))["conditions"]["②"]["axis_ratio"]
    want = {e["bundle"]: {k: e.get(k) for k in ("bundle", "status", "reason", *AXIS_COUNTS)} for e in d_rows}
    assert {e["bundle"]: e for e in got["entries"]} == want
    assert got["called"] is True and got["problems"] == []


# ================================================================ 연합 칸의 학습 단계 — 거부 · 정리 · 건너뜀
def _fed_copy(lap, name: str, *, drop: bool = True):
    """한 바퀴의 루트를 같은 부모 아래 다른 루트로 옮긴다. `drop` 이면 연합 칸의 학습 산출을 걷어 처음 돌 때의 자리로 만든다."""
    import shutil

    w, root, _ = lap
    dst = w.parent / name
    shutil.copytree(root, dst, ignore=shutil.ignore_patterns("judge.json"))
    if drop:
        shutil.rmtree(dst / "train" / "uni_fed_s1")
    return w, dst


def _fed_train(w, dst, *, runner, procs, stop, extra=()):
    from tests import uni_rehearsal_world as RW

    return RU.main(["train", "--run-root", str(dst), "--plan", str(dst / "plan.yaml"), "--snapshot", str(w.snap),
                    "--tag", "uni_fed", *extra],
                   measure_env=RW.measure, model_loader=RW.fake_model_loader, checkout=w.repo,
                   exposure_ledger=w.exposure, non_main_parent=w.parent, config_path=w.config, standin_allowed=True,
                   fed_runner=runner, flwr_procs=procs, flwr_stop=stop)


def _spy(calls):
    def run(items, log_path):
        calls.append(items)
    return run


def test_연합_단계는_남은_flwr_프로세스를_못_내리면_돌지_않는다(lap, capsys):
    w, dst = _fed_copy(lap, "reh-20261001T130001")
    calls, stopped = [], []
    left = [{"pid": 4242, "name": "flower-superlink.exe", "cmdline": "flower-superlink", "kind": "target"}]
    code = _fed_train(w, dst, runner=_spy(calls), procs=lambda: left, stop=stopped.append)
    assert code == RU.EXIT_REFUSED and "[flwr_residual_before]" in capsys.readouterr().err
    assert calls == [] and stopped == [left] and not (dst / "train" / "uni_fed_s1").exists()


def test_연합_단계는_앞뒤로_정리하고_닫히지_않은_출력을_거부한다(lap, capsys):
    w, dst = _fed_copy(lap, "reh-20261001T130002")
    calls, seen = [], {"n": 0}
    left = [{"pid": 4243, "name": "flwr-simulation.exe", "cmdline": "flwr-simulation", "kind": "target"}]

    def procs():
        seen["n"] += 1
        return left if seen["n"] == 1 else []                  # 처음 본 것은 내린 뒤 사라진다

    code = _fed_train(w, dst, runner=_spy(calls), procs=procs, stop=lambda found: None)
    assert code == RU.EXIT_REFUSED and "[fed_not_closed]" in capsys.readouterr().err
    assert seen["n"] == 3                                      # 앞(찾기 · 다시 찾기) · 뒤(찾기)
    (items,) = calls
    assert items["purpose"] == "rehearsal" and items["cell"] == "uni_fed"
    assert (items["num-server-rounds"], items["local-epochs"], items["total-epochs"]) == (2, 1, 2)   # 계획의 학습량
    assert items["uni-train-root"] == (dst / "train").resolve().as_posix()
    assert items["uni-train-lists"] == (dst / "lists").resolve().as_posix()
    assert items["uni-init-adapter"] == (dst / "init" / "initial.npz").resolve().as_posix()
    assert items["rehearsal-root"] == dst.resolve().as_posix() and items["plan"] == (dst / "plan.yaml").resolve().as_posix()


def test_연합_실행이_죽어도_뒤의_정리는_한다(lap):
    w, dst = _fed_copy(lap, "reh-20261001T130003")
    seen = {"n": 0}

    def procs():
        seen["n"] += 1
        return []

    def boom(items, log_path):
        raise RuntimeError("flwr run 실패")

    with pytest.raises(RuntimeError, match="flwr run 실패"):
        _fed_train(w, dst, runner=boom, procs=procs, stop=lambda found: None)
    assert seen["n"] == 2                                      # 앞 하나 · 뒤 하나


def _denied(found):
    import psutil

    raise psutil.AccessDenied(found[0]["pid"])


def test_연합_실행이_죽고_뒤의_정리가_권한_거부여도_원래_실패가_올라온다(lap):
    """뒤의 정리(`stop()`)가 `AccessDenied` 를 내도 실행의 실패를 덮지 않는다(외부 검토 회신 `rehearsal_blockers` 의 1)."""
    w, dst = _fed_copy(lap, "reh-20261001T130006")
    seen = {"n": 0}
    left = [{"pid": 4245, "name": "flower-superlink.exe", "cmdline": "flower-superlink", "kind": "target"}]

    def procs():
        seen["n"] += 1
        return [] if seen["n"] == 1 else left                  # 앞은 비었고 실행 뒤에 남았다

    def boom(items, log_path):
        raise RuntimeError("flwr run 실패")

    with pytest.raises(RuntimeError, match="flwr run 실패"):
        _fed_train(w, dst, runner=boom, procs=procs, stop=_denied)


def test_연합_실행이_섰어도_뒤의_정리가_권한_거부면_거부한다(lap, capsys):
    w, dst = _fed_copy(lap, "reh-20261001T130007")
    calls, seen = [], {"n": 0}
    left = [{"pid": 4246, "name": "flower-superlink.exe", "cmdline": "flower-superlink", "kind": "target"}]

    def procs():
        seen["n"] += 1
        return [] if seen["n"] == 1 else left

    code = _fed_train(w, dst, runner=_spy(calls), procs=procs, stop=_denied)
    assert code == RU.EXIT_REFUSED and "[flwr_stop_failed_after]" in capsys.readouterr().err and len(calls) == 1


def test_닫힌_연합_출력은_건너뛰고_스탬프는_받지_않는다(lap, capsys):
    w, dst = _fed_copy(lap, "reh-20261001T130004", drop=False)
    calls = []
    assert _fed_train(w, dst, runner=_spy(calls), procs=list, stop=lambda f: None) == RU.EXIT_OK
    assert calls == [] and "status=skipped" in capsys.readouterr().out
    code = _fed_train(w, dst, runner=_spy(calls), procs=list, stop=lambda f: None, extra=("--stamp", "x"))
    assert code == RU.EXIT_REFUSED and "[fed_args]" in capsys.readouterr().err and calls == []


def _swap_link(dst: Path, src: Path, case: str) -> Path | None:
    """`dst/lists` 의 파일을 같은 바이트의 다른 루트 파일로 가는 링크로 바꾼다. 정션이면 `lists` 폴더 자체를 바꾸고 그 자리를 돌려준다.
    링크는 이 시험의 임시 폴더 안에서만 만든다."""
    import os
    import shutil

    if case == "normal":
        return None
    if case == "junction":
        import _winapi

        shutil.rmtree(dst / "lists")                              # 이 시험이 복사한 보통 폴더다
        _winapi.CreateJunction(str((src / "lists").resolve()), str(dst / "lists"))
        return dst / "lists"
    kind, how = case.split("_")
    name = "train_uni_local_C1.jsonl" if kind == "list" else "lists_record.json"
    (dst / "lists" / name).unlink()
    if how == "symlink":
        try:
            os.symlink(src / "lists" / name, dst / "lists" / name)
        except OSError as exc:                                    # 파일 기호 링크는 권한이 있어야 만든다(개발자 모드 · 관리자)
            pytest.skip(f"이 환경은 파일 기호 링크를 만들 수 없다: {exc}")
    else:
        os.link(src / "lists" / name, dst / "lists" / name)
    return None


LINK_CASES = ["normal", "list_hardlink", "record_hardlink", "list_symlink", "record_symlink", "junction"]


@pytest.mark.parametrize("case", LINK_CASES)
def test_연합_앞단은_다른_루트의_목록_파일로_가는_링크를_읽지_않는다(lap, capsys, case):
    """앞단의 목록 검사(`check_train_list`)도 두 파일의 최종 경로를 본다 — 해시 · 원본 행이 다 맞는 다른 루트의 정상 파일이다
    (외부 검토 회신 `rehearsal_blockers3` 의 3)."""
    import os

    w, root, _ = lap
    w, dst = _fed_copy(lap, f"reh-20261001T1310{LINK_CASES.index(case):02d}")
    junction = _swap_link(dst, root, case)
    calls = []
    try:
        code = _fed_train(w, dst, runner=_spy(calls), procs=list, stop=lambda found: None)
    finally:
        if junction is not None:
            os.rmdir(junction)                                    # 링크만 뗀다 — 가리키던 폴더는 남는다
    err = capsys.readouterr().err
    if case == "normal":                                          # 정상 내부 파일 — 앞단을 지나 실행에 닿는다
        assert len(calls) == 1 and "[train_list_path]" not in err
    else:
        assert code == RU.EXIT_REFUSED and "[train_list_path]" in err and calls == []
    assert (root / "lists" / "train_uni_local_C1.jsonl").is_file()


@pytest.mark.parametrize("case", ["list_hardlink", "record_hardlink", "junction"])
def test_로컬_학습도_다른_루트의_목록_파일로_가는_링크를_읽지_않는다(lap, capsys, case):
    import os

    from tests import uni_rehearsal_world as RW

    w, root, _ = lap
    w, dst = _fed_copy(lap, f"reh-20261001T1320{['list_hardlink', 'record_hardlink', 'junction'].index(case):02d}",
                       drop=False)
    junction = _swap_link(dst, root, case)
    loaded = []
    try:
        code = RU.main(["train", "--run-root", str(dst), "--plan", str(dst / "plan.yaml"), "--snapshot", str(w.snap),
                        "--tag", "uni_local_C1"],
                       measure_env=RW.measure, model_loader=lambda *a, **k: loaded.append(1) or RW.fake_model_loader(*a, **k),
                       checkout=w.repo, exposure_ledger=w.exposure, non_main_parent=w.parent, config_path=w.config,
                       standin_allowed=True)
    finally:
        if junction is not None:
            os.rmdir(junction)
    assert code == RU.EXIT_REFUSED and "[train_list_path]" in capsys.readouterr().err and loaded == []


def test_연합_단계는_신원이_불명확한_flwr_프로세스를_내리지_않고_시작하지_않는다(lap, capsys):
    """이 실행의 venv 의 것인지 가릴 수 없는 flwr 프로세스 — 죽이지 않고 시작을 거부한다(외부 검토 §50 요청 2)."""
    w, dst = _fed_copy(lap, "reh-20261001T130005")
    calls, stopped = [], []
    left = [{"pid": 4244, "name": "flower-superlink.exe", "cmdline": "flower-superlink", "kind": "unclear"}]
    code = _fed_train(w, dst, runner=_spy(calls), procs=lambda: left, stop=stopped.append)
    assert code == RU.EXIT_REFUSED and "[flwr_identity_unclear_before]" in capsys.readouterr().err
    assert calls == [] and stopped == []
