"""진입점의 진단 경로 — 목적 `frame_diag`(진입점 미니스펙 3판 1-10 · 8-1 · 8-3 · 계약 §31-1 · §32-2 · §32-8 · §32-9).

합성 세계 — `git init` 한 저장소에 진단 등록 · 규칙 파일을 **커밋**하고 영수증(`kind="frame_diag"`)이 그 커밋을 가리킨다.
스냅샷은 정답 박스가 장마다 다른 val 장 40 개, 모델 묶음(중앙 칸)과 에코 묶음, 학습 원장과 어댑터 meta.
여기의 문턱은 **시험용**이다 — 표본이 작은 세계에서 가지를 세우려고 정했다.

지키는 것.

1. 규칙은 **영수증의 커밋에서** 읽고 등록의 digest 와 맞댄다. 작업 트리의 규칙 파일을 읽지 않는다.
2. 원본 프레임은 통과(종료 0), 리사이즈 프레임은 불통과(종료 2) — 판정 줄이 산출물보다 먼저 원장에 있고, 산출물은 그 줄의 결정적 함수다.
3. **판정 한 번** — 같은 내용의 묶음은 다시 판정하지 않는다(다른 루트로 옮겨도 · 지연만 달라도). 산출물이 있으면 종료 3.
4. 판정 줄은 있는데 산출물이 없으면 **그 줄에서** 다시 만든다 — `call` 줄을 쓰지 않고 `frame_diag_regenerated` 줄을 쓴다. 바이트가 같다.
5. 판정하지 않는 경우 — 대역 실행 · 카나리아 불통과 · 묶음 거부. 판정 줄을 쓰지 않는다.
6. 산출에는 상태 · 후보 · 사유 · 수 · 규칙의 경로와 digest · 프레임 입력 해시뿐이다 — 통계 값이 없다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import threading
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

import scripts.probe.score_unified as SU
from data.manifest_io import ANNOTATION_COLUMNS, write_snapshot
from evaluation import frame_diag as F
from evaluation import scoring_ledger as SL
from evaluation.coord_check import coord_fixture_digest
from evaluation.eval_list import canonical_bytes, validate_eval_list
from evaluation.frame_diag_rules import build_rules_file
from evaluation.prereg_unified import CoordCfgRecord, UnifiedRegistration, dump_registration
from evaluation.scoring_ledger import read_ledger
from tests.test_bundle_v14 import IMPL_IDS, IMPL_KEYS, hx, impl, make, train_config_text
from tests.test_score_unified import NOW, _man, git
from tests.test_scoring_layer import gen_row

REPO = Path(__file__).resolve().parents[1]
IDS = [f"aihub{i:06d}" for i in range(100, 140)]
RULES_REL = "configs/registration/frame_diag_rules-20261001-1.json"
REG_REL = "configs/registration/frame_diag-20261001-1.json"
RULES = F.FrameRules(n_img=10, s_min=0.7, w_s=0.1, w_r=0.0075, w_mix=0.0019, delta=0.0055, trim=0.0,
                     n_boot=40, boot_seed=1, min_pairs=10, min_groups=10, min_pair_groups=10)
"""시험용 — 40 장 · 20 묶음 세계에서 가지를 세우는 문턱. 등록 문턱이 아니다."""
REQS = [{"noise_step": 0.1, "stat": "r", "axis": "y", "width": "w_r", "n0": 40, "n_required": 30, "g0": 20,
         "groups_required": 15, "q0": None, "pairs_required": None},
        {"noise_step": 0.1, "stat": "m", "axis": "y", "width": "axis_ratio_width", "n0": 40, "n_required": 30,
         "g0": 20, "groups_required": 25, "q0": 40, "pairs_required": 30}]
"""시험용 요구 표본 수 — 장 · 묶음 · 쌍은 닿고 쌍의 묶음(25 > 20)은 못 닿는다."""


def rules_file(rules: F.FrameRules = RULES) -> bytes:
    return build_rules_file(rules, sample_requirements=REQS, expected_sha256=hx("expected"), cases_sha256=hx("cases"),
                            procedure_sha256=hx("procedure"), noise_ladder=[0.1], summary={}, sources={})


def gold_box(i: int) -> tuple[float, float, float, float]:
    cx, cy = 300.0 + 20 * i, 200.0 + 7 * i
    return (cx - 20, cy - 20, cx + 20, cy + 20)


def _ann(image_id: str, box) -> dict:
    return {c: "" for c in ANNOTATION_COLUMNS} | {
        "ann_id": f"{image_id}#0", "image_id": image_id, "src_label_raw": "기공", "defect_type": "porosity",
        "iso_code": "2011", "polygon_json": "[]", "bbox_x1_px": box[0], "bbox_y1_px": box[1], "bbox_x2_px": box[2],
        "bbox_y2_px": box[3], "area_px": 1.5, "major_axis_px": 1.0, "minor_axis_px": 1.0, "equiv_diameter_px": 1.0,
        "major_axis_mm": None, "equiv_diameter_mm": None, "geom_valid": True, "geom_flags": "",
    }


def _central_ledger() -> bytes:
    rows = [["run_id", "seed", "cell", "split_hash", "client_id", "round", "n_train_samples", "metric_name",
             "metric_value", "bytes_up", "bytes_down", "wall_time"]]
    for e, steps in ((1, 50.0), (2, 100.0)):
        rows.append(["run_c", "20260828", "uni_central", "d", "central", str(e), "10", "optimizer_steps", str(steps),
                     "0", "0", "1"])
    rows.append(["run_c", "20260828", "uni_central", "d", "central", "100", "10", "final_adapter_step", "100.0",
                 "0", "0", "1"])
    return ("\r\n".join(",".join(r) for r in rows) + "\r\n").encode("utf-8")


def build(tmp: Path, *, predict=lambda b: b, root_name="fdiag-a", meta_extra=None, echo_edit=None) -> dict:
    repo = tmp / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    man = []
    for i, iid in enumerate(IDS):
        m = _man(iid, "val")
        m["group_id"] = f"g{i // 2}"
        man.append(m)
    snap = tmp / "snap"
    digest = write_snapshot(snap, pd.DataFrame(man), pd.DataFrame([_ann(iid, gold_box(i)) for i, iid in enumerate(IDS)]),
                            {"snapshot_id": "synthetic_v1", "capabilities": {"verdict_mode": "clause_only"}})
    (repo / "configs/registration").mkdir(parents=True)
    (repo / "configs/base.yaml").write_text(
        f"fixed_before_main_runs:\n  snapshot_id: synthetic_v1\n  snapshot_digest: {digest}\n"
        f"  uni_processor_kwargs: {{max_pixels: {1280 * 704}}}\n", encoding="utf-8")
    fixtures = REPO / "vlm/coords_fixtures/golden_fixtures.json"
    (repo / "vlm/coords_fixtures").mkdir(parents=True)
    (repo / "vlm/coords_fixtures/golden_fixtures.json").write_bytes(fixtures.read_bytes())
    rules_raw = rules_file()
    (repo / RULES_REL).write_bytes(rules_raw)

    root = repo / "outputs/rehearsal_u" / root_name
    (root / "lists").mkdir(parents=True)
    lst = canonical_bytes(IDS)
    el = validate_eval_list(lst)
    for name in ("generation.list", "scoring.list", "echo.list"):
        (root / "lists" / name).write_bytes(lst)
    plan = b"kind: frame_diag\n"
    (root / "plan.yaml").write_bytes(plan)

    r = UnifiedRegistration()
    g = r.generation
    g.purpose, g.list_split, g.snapshot_digest = "frame_diag", "val", digest
    g.eval_list_file_sha256 = g.echo_list_file_sha256 = el.file_sha256
    g.eval_list_set_sha256 = g.echo_list_set_sha256 = el.set_sha256
    g.prompt_sha256, g.chat_template_kwargs, g.gen_prefix_sha256 = hx("prompt"), {"add_generation_prompt": True}, hx("prefix")
    g.max_new_tokens, g.decoding, g.batch_size, g.padding_side = 256, {"do_sample": False}, 1, "left"
    g.processor_config_sha256, g.processor_min_pixels, g.processor_max_pixels = hx("proc"), 256, 1280 * 704
    g.patch_size, g.merge_size, g.coord_space, g.coord_cfg_hash = 16, 2, "ABS_ORIG", hx("coord")
    g.base_model_id, g.base_model_revision, g.init_adapter_digest = "local/qwen3.5-4b", "frozen", hx("init")
    g.train_config_sha256 = hashlib.sha256(train_config_text().encode("utf-8")).hexdigest()
    g.budget_n, g.budget_r, g.budget_e, g.transformers_version = 30, 3, 10, "4.57.0"
    g.impl_ids = IMPL_KEYS
    g.train_rows_digest, g.plan_sha256 = hx("rows"), hashlib.sha256(plan).hexdigest()
    g.frame_diag_rules_path, g.frame_diag_rules_sha256 = RULES_REL, hashlib.sha256(rules_raw).hexdigest()
    r.canary.coord_cfg = CoordCfgRecord(accepted=(hx("coord"),), frozen_value=hx("coord"), frozen_at="2026-10-01",
                                        current_value=hx("coord"), change_scope="first_registration",
                                        equivalence_evidence=("동결 export 없음 — 첫 등록",))
    r.canary.coord_fixture_sha256 = coord_fixture_digest(fixtures.read_bytes())
    r.metric.class_codes, r.metric.coupling_rule = ("100", "2011", "301", "401"), "decoupled_v2"
    r.seeds = {"1": 20260828}
    raw = dump_registration(r)
    (repo / REG_REL).write_bytes(raw)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "진단 등록")
    head = git(repo, "rev-parse", "HEAD")
    receipt = {"generation_sha256": r.generation_sha256(), "scoring_sha256": r.scoring_sha256(),
               "registered_at": "2026-09-21T09:00:00+09:00", "main_commit": head, "kind": "frame_diag",
               "registration_path": REG_REL, "registration_file_sha256": hashlib.sha256(raw).hexdigest()}
    (root / "registration").mkdir()
    (root / "registration/receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    ledger = root / "train/uni_central_s1/train_ledger.csv"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(_central_ledger())
    (ledger.parent / "adapter_last.meta.json").write_text(
        json.dumps({"adapter_sha256": hx("adapter"), "adapter_step": 100, "train_rows_digest": hx("rows")}),
        encoding="utf-8")
    led_sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
    common = {"generation_sha256": r.generation_sha256(), "snapshot_digest": digest, "list_split": "val",
              "batch_size": 1, "plan_sha256": g.plan_sha256, **(meta_extra or {})}
    bundles = root / "bundles"
    rows = [gen_row(iid, boxes=(("2011", predict(gold_box(i))),)) for i, iid in enumerate(IDS)]
    make(bundles, tag="uni_central", ids=IDS, rows=rows, purpose="frame_diag",
         meta_over=common | {"train_run_id": "run_c", "train_ledger_sha256": led_sha,
                             "train_ledger_path": ledger.relative_to(repo).as_posix()},
         start_over={"generation_sha256": r.generation_sha256(), "train_run_id": "run_c",
                     "train_ledger_sha256": led_sha})
    model_attempts = (bundles / "attempts.jsonl").read_bytes()
    echo_rows = [gen_row(iid, boxes=(("2011", gold_box(i)),)) for i, iid in enumerate(IDS)]
    if echo_edit:
        echo_rows = echo_edit(echo_rows)
    make(bundles, tag="uni_central", ids=IDS, rows=echo_rows, purpose="frame_diag", echo=True, meta_over=common,
         start_over={"generation_sha256": r.generation_sha256()})
    (bundles / "attempts.jsonl").write_bytes(model_attempts + (bundles / "attempts.jsonl").read_bytes())
    return {"repo": repo, "root": root, "snap": snap, "reg": r}


def argv(w: dict, *, bundles=None, out=None) -> list[str]:
    root = w["root"]
    return ["--registration", "probe", "--purpose", "frame_diag",
            "--receipt", str(root / "registration/receipt.json"), "--snapshot", str(w["snap"]),
            "--bundles", str(bundles or root / "bundles"), "--generation-list", str(root / "lists/generation.list"),
            "--scoring-list", str(root / "lists/scoring.list"), "--echo-list", str(root / "lists/echo.list"),
            "--train-root", str(root / "train"), "--out", str(out or root), "--plan", str(root / "plan.yaml"),
            "--device", "cpu:0"]


def run(w: dict, **kw) -> int:
    return SU.main(argv(w, **kw), _seam=SU.Seam(checkout=w["repo"], repo=w["repo"]), _now=NOW)


def ledger_rows(w):
    return read_ledger(w["repo"] / "outputs/main_u/scoring_ledger.jsonl")


def resize(b):
    k = 704 / 720
    return (b[0], b[1] * k, b[2], b[3] * k)


# ---------------------------------------------------------------- 판정

def test_원본_프레임은_통과하고_판정_줄이_산출물보다_먼저다(tmp_path) -> None:
    w = build(tmp_path)
    assert run(w) == SU.EXIT_OK
    rows = ledger_rows(w)
    assert [r["kind"] for _, r in rows] == ["call", "frame_diag_verdict"]
    sha, verdict = rows[-1]
    assert verdict["status"] == F.PASS and verdict["call_sha256"] == rows[0][0]
    assert verdict["rules_path"] == RULES_REL and verdict["rules_sha256"] == w["reg"].generation.frame_diag_rules_sha256
    out = w["root"] / SU.OUT_DIAG
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["verdict_line_sha256"] == sha and body["status"] == F.PASS
    assert out.read_text(encoding="utf-8") == SU._diag_output_text(sha, verdict), "산출물은 판정 줄의 결정적 함수다"


def test_리사이즈_프레임은_불통과_y_리사이즈(tmp_path) -> None:
    w = build(tmp_path, predict=resize)
    assert run(w) == SU.EXIT_REJECTED
    _, verdict = ledger_rows(w)[-1]
    assert (verdict["status"], verdict["candidate"], verdict["reason"]) == (F.FAIL, F.CAND_RESIZE, "y · 리사이즈")


def test_산출에는_통계_값이_없다(tmp_path) -> None:
    w = build(tmp_path)
    run(w)
    body = json.loads((w["root"] / SU.OUT_DIAG).read_text(encoding="utf-8"))
    keys = set(body) - {"bundles", "frame_inputs", "provisional"}
    assert keys == {"run_kind", "call_sha256", "generation_sha256", "status", "candidate", "reason", "n_images",
                    "n_pairs", "n_no_pred", "n_count_mismatch", "n_groups", "n_image_groups", "n_pair_groups",
                    "samples", "design_samples", "n_limit_no_pair", "rules_path", "rules_sha256", "output",
                    "verdict_line_sha256"}
    assert not any(isinstance(body[k], float) for k in keys)
    assert body["run_kind"] == "frame_diag", "3판 8-8 의 11 — 검토(10-02 병합 전) 6"


def test_표본_요건은_충족_미달로만_싣는다(tmp_path) -> None:
    """검토(10-02 병합 전) 4 — 규칙의 하한 넷과 규칙 파일의 요구 표본 수가 산출에 충족 · 미달로 이어진다. 수는 싣지 않는다."""
    w = build(tmp_path)
    run(w)
    body = json.loads((w["root"] / SU.OUT_DIAG).read_text(encoding="utf-8"))
    assert (body["n_images"], body["n_image_groups"], body["n_pairs"], body["n_pair_groups"]) == (40, 20, 40, 20)
    assert body["samples"] == {"images": F.MET, "image_groups": F.MET, "pairs": F.MET, "pair_groups": F.MET}
    assert body["design_samples"] == {"0.1": {"images": F.MET, "image_groups": F.MET, "pairs": F.MET,
                                              "pair_groups": F.UNMET}}
    assert set(body["frame_inputs"]) == set(SU.FRAME_INPUT_FIELDS) | {"train_config_bound"}


def test_프레임_입력_해시는_예산과_페어를_빼고_묶는다() -> None:
    side = {k: f"v_{k}" for k in SU.FRAME_INPUT_FIELDS}
    cfg = json.loads(train_config_text())
    a = SU._frame_inputs(side | {"train_config": json.dumps(cfg)})
    b = SU._frame_inputs(side | {"train_config": json.dumps(cfg | {"num_rounds": 99, "pairs_digest": "x"})})
    c = SU._frame_inputs(side | {"train_config": json.dumps(cfg | {"lr0": 1.0})})
    assert a == b and a["train_config_bound"] != c["train_config_bound"]


# ---------------------------------------------------------------- 판정 한 번 · 재생성

def test_같은_내용의_묶음은_다른_루트에서도_다시_판정하지_않는다(tmp_path) -> None:
    w = build(tmp_path)
    assert run(w) == SU.EXIT_OK
    other = w["repo"] / "outputs/rehearsal_u/fdiag-b"
    import shutil
    shutil.copytree(w["root"] / "bundles", other / "bundles")
    assert run(w, bundles=other / "bundles", out=other) == SU.EXIT_ENTRY
    assert [r["kind"] for _, r in ledger_rows(w)] == ["call", "frame_diag_verdict", "call"]


def test_같은_루트에서_산출물이_있으면_종료_3(tmp_path) -> None:
    w = build(tmp_path)
    run(w)
    assert run(w) == SU.EXIT_ENTRY
    assert [r["kind"] for _, r in ledger_rows(w)] == ["call", "frame_diag_verdict"], "판정 전 거부라 call 줄도 없다"


def test_판정_줄은_있는데_산출물이_없으면_그_줄에서_다시_만든다(tmp_path) -> None:
    w = build(tmp_path)
    run(w)
    out = w["root"] / SU.OUT_DIAG
    first = out.read_bytes()
    out.unlink()
    assert run(w) == SU.EXIT_OK
    assert out.read_bytes() == first, "같은 바이트"
    kinds = [r["kind"] for _, r in ledger_rows(w)]
    assert kinds == ["call", "frame_diag_verdict", "frame_diag_regenerated"], "call 줄을 쓰지 않는다 — 시도로 세지 않는다"
    _, regen = ledger_rows(w)[-1]
    assert "purpose" not in regen and regen["output_sha256"] == hashlib.sha256(first).hexdigest()
    assert json.loads(out.read_text(encoding="utf-8"))["run_kind"] == "frame_diag", "재생성도 같은 표지"


def test_다시_만드는_호출은_묶음과_정답을_열지_않는다(tmp_path, monkeypatch) -> None:
    w = build(tmp_path)
    run(w)
    (w["root"] / SU.OUT_DIAG).unlink()
    for name in ("_read_lists", "_verify_all", "_gold", "judge"):
        monkeypatch.setattr(SU, name, lambda *a, **k: pytest.fail("다시 만드는 호출이 계산 쪽을 불렀다"))
    assert run(w) == SU.EXIT_OK


def test_다시_만드는_호출은_영수증_검사만_한다(tmp_path, monkeypatch) -> None:
    """검토(10-02 병합 전) 2 — 스냅샷 · 계획 · 묶음이 없어지고 지금 검사기가 규칙을 못 읽어도 판정 줄에서 되살린다(§32-8)."""
    w = build(tmp_path)
    run(w)
    out = w["root"] / SU.OUT_DIAG
    first = out.read_bytes()
    out.unlink()
    shutil.rmtree(w["snap"])
    shutil.rmtree(w["root"] / "bundles")
    (w["root"] / "plan.yaml").unlink()
    for name in ("load_anchor", "check_processor", "check_snapshot", "_diag_rules", "load_rules_file",
                 "_read_lists", "_verify_all", "_gold", "judge"):
        monkeypatch.setattr(SU, name, lambda *a, _n=name, **k: pytest.fail(f"다시 만드는 호출이 {_n} 을 불렀다"))
    assert run(w) == SU.EXIT_OK
    assert out.read_bytes() == first


def test_다시_만드는_호출도_영수증의_생성_지문을_맞댄다(tmp_path, capsys) -> None:
    """재확인 5 — 판정 줄이 있고 산출물이 없을 때, 영수증의 생성 지문만 다른 값으로 바꾼 영수증으로는 되살리지 않는다(3판 1-1-가 3)."""
    w = build(tmp_path)
    run(w)
    out = w["root"] / SU.OUT_DIAG
    out.unlink()
    rp = w["root"] / "registration/receipt.json"
    rc = json.loads(rp.read_text(encoding="utf-8"))
    rp.write_text(json.dumps(rc | {"generation_sha256": "a" * 64}), encoding="utf-8")
    capsys.readouterr()
    assert run(w) == SU.EXIT_ENTRY
    assert "generation 지문" in capsys.readouterr().err
    assert not out.exists() and [r["kind"] for _, r in ledger_rows(w)] == ["call", "frame_diag_verdict"]


def test_산출물_쓰기_직전에_죽으면_줄이_남고_다음_호출이_계산_없이_되살린다(tmp_path, monkeypatch) -> None:
    """검토(10-02 병합 전) 5 — 3판 8-9 의 33. 판정 줄은 산출물보다 먼저다."""
    w = build(tmp_path)
    real = SU.write_new_text

    def dies(path, text):
        if Path(path).name == Path(SU.OUT_DIAG).name:
            raise RuntimeError("산출물 쓰기 직전에 죽었다")
        return real(path, text)
    monkeypatch.setattr(SU, "write_new_text", dies)
    with pytest.raises(RuntimeError, match="죽었다"):
        run(w)
    rows = ledger_rows(w)
    assert [r["kind"] for _, r in rows] == ["call", "frame_diag_verdict"]
    assert not (w["root"] / SU.OUT_DIAG).exists()
    monkeypatch.setattr(SU, "write_new_text", real)
    for name in ("_read_lists", "_verify_all", "_gold", "judge"):
        monkeypatch.setattr(SU, name, lambda *a, **k: pytest.fail("되살리는 호출이 계산 쪽을 불렀다"))
    sha, verdict = rows[-1]
    assert run(w) == SU._diag_exit(verdict["status"])
    assert (w["root"] / SU.OUT_DIAG).read_text(encoding="utf-8") == SU._diag_output_text(sha, verdict)
    assert [r["kind"] for _, r in ledger_rows(w)] == ["call", "frame_diag_verdict", "frame_diag_regenerated"]


def test_지연만_바꿔_다시_내보내도_다시_판정하지_않는다(tmp_path, capsys) -> None:
    """검토(10-02 병합 전) 5 — 3판 8-9 의 32. 원시 해시는 다르고 내용 해시는 같다."""
    w = build(tmp_path)
    assert run(w) == SU.EXIT_OK
    other = w["repo"] / "outputs/rehearsal_u/fdiag-c"
    shutil.copytree(w["root"] / "bundles", other / "bundles")
    [gp] = [q for q in (other / "bundles").glob("*.generations.jsonl") if ".echo." not in q.name]
    before = gp.read_bytes()
    rows = [json.loads(line) for line in before.decode("utf-8").splitlines()]
    gp.write_text("".join(json.dumps(r | {"latency_ms": r["latency_ms"] + 7.0}, ensure_ascii=False) + "\n"
                          for r in rows), encoding="utf-8", newline="\n")
    assert gp.read_bytes() != before
    capsys.readouterr()
    assert run(w, bundles=other / "bundles", out=other) == SU.EXIT_ENTRY
    assert "[DIAG_ALREADY_JUDGED]" in capsys.readouterr().err
    assert sum(r["kind"] == "frame_diag_verdict" for _, r in ledger_rows(w)) == 1


def _tamper_grid(bundles: Path, edit, observed=None) -> None:
    """모델 묶음의 줄을 `edit(i, row)` 로 바꾸고 곁 파일의 원시 해시를 맞춘다. `observed` 를 주면 곁 파일의 관측 격자도 그 값으로."""
    from evaluation.bundle_v14 import BundleFiles
    [gp] = [q for q in bundles.glob("*.generations.jsonl") if ".echo." not in q.name]
    rows = [json.loads(s) for s in gp.read_text(encoding="utf-8").splitlines()]
    gp.write_text("".join(json.dumps(edit(i, r), ensure_ascii=False) + "\n" for i, r in enumerate(rows)),
                  encoding="utf-8", newline="\n")
    side = BundleFiles(gp).sidecar
    meta = json.loads(side.read_text(encoding="utf-8"))
    meta["generations_sha256"] = hashlib.sha256(gp.read_bytes()).hexdigest()
    if observed is not None:
        meta["image_grid_thw_observed"] = observed(meta["image_grid_thw_observed"])
    side.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _drop_grid(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "image_grid_thw"}


@pytest.mark.parametrize(("edit", "observed", "code"), [
    (lambda i, r: r | {"image_grid_thw": [2, 44, 80]}, None, "line_schema"),
    (lambda i, r: r | {"image_grid_thw": [1, 45, 80]}, None, "model_input_grid_mismatch"),
    (lambda i, r: _drop_grid(r), None, "line_schema"),
    # §45 의 반례 — 첫 줄의 격자만 지우고 곁 파일 집계를 한 줄 줄인다
    (lambda i, r: _drop_grid(r) if i == 0 else r, lambda obs: {k: v - 1 for k, v in obs.items()}, "line_schema"),
])
def test_판정한_묶음에서_격자만_바꾸면_거부되고_판정이_늘지_않는다(tmp_path, edit, observed, code: str) -> None:
    """재확인 contentkey 의 1 — 격자는 내용 해시에 들므로 격자만 바꾼 묶음은 판정 한 번의 조회를 피한다. 묶음 검증이
    줄의 격자를 곁 파일의 관측 격자 · 패치 크기와 맞대 거부해야 판정 줄이 늘지 않는다."""
    w = build(tmp_path)
    assert run(w) == SU.EXIT_OK
    other = w["repo"] / "outputs/rehearsal_u/fdiag-g"
    shutil.copytree(w["root"] / "bundles", other / "bundles")
    _tamper_grid(other / "bundles", edit, observed)
    assert run(w, bundles=other / "bundles", out=other) == SU.EXIT_REJECTED
    assert sum(r["kind"] == "frame_diag_verdict" for _, r in ledger_rows(w)) == 1, "판정 줄이 늘지 않는다"
    body = json.loads((other / SU.OUT_DIAG_NOT_JUDGED).read_text(encoding="utf-8"))
    assert body["reason"] == "묶음이 거부됐다" and code in body["detail"]["uni_central_s1"]


def test_규칙을_바꾼_새_등록이어도_같은_내용은_다시_판정하지_않는다(tmp_path, capsys) -> None:
    """검토(10-02 병합 전) 5 — 3판 8-9 의 32. 새 규칙 · 새 등록 · 새 영수증이어도 같은 묶음 내용이면 멈춘다."""
    w = build(tmp_path)
    assert run(w) == SU.EXIT_OK
    repo = w["repo"]
    rel2, reg_rel2 = "configs/registration/frame_diag_rules-20261001-2.json", "configs/registration/frame_diag-20261001-2.json"
    raw_rules = rules_file(F.FrameRules(**{**asdict(RULES), "w_r": 0.5}))
    (repo / rel2).write_bytes(raw_rules)
    r2 = copy.deepcopy(w["reg"])
    r2.generation.frame_diag_rules_path, r2.generation.frame_diag_rules_sha256 = rel2, hashlib.sha256(raw_rules).hexdigest()
    assert r2.generation_sha256() != w["reg"].generation_sha256()
    raw_reg = dump_registration(r2)
    (repo / reg_rel2).write_bytes(raw_reg)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "규칙을 바꾼 새 등록")
    root2 = repo / "outputs/rehearsal_u/fdiag-d"
    shutil.copytree(w["root"], root2, ignore=shutil.ignore_patterns("score"))
    rp = root2 / "registration/receipt.json"
    rc = json.loads(rp.read_text(encoding="utf-8"))
    rp.write_text(json.dumps(rc | {"main_commit": git(repo, "rev-parse", "HEAD"), "registration_path": reg_rel2,
                                   "generation_sha256": r2.generation_sha256(), "scoring_sha256": r2.scoring_sha256(),
                                   "registration_file_sha256": hashlib.sha256(raw_reg).hexdigest()}), encoding="utf-8")
    capsys.readouterr()
    assert run({**w, "root": root2}) == SU.EXIT_ENTRY
    assert "[DIAG_ALREADY_JUDGED]" in capsys.readouterr().err
    assert sum(r["kind"] == "frame_diag_verdict" for _, r in ledger_rows(w)) == 1


def test_동시_호출_둘은_한_번만_판정한다(tmp_path, monkeypatch) -> None:
    """검토(10-02 병합 전) 3 — 같은 묶음을 다른 루트로 동시에 부르면 하나만 판정하고 다른 하나는 조회에서 멈춘다.

    A 가 판정 계산 안에서 멈춰 있는 동안 B 를 띄운다. B 가 원장 잠금에서 기다리기 시작하거나(고친 뒤) 판정 계산에 닿으면(고치기 전)
    A 를 풀어 준다. 판정 줄이 둘이면 판정 한 번이 깨진 것이다.
    """
    w = build(tmp_path)
    other = w["repo"] / "outputs/rehearsal_u/fdiag-b"
    shutil.copytree(w["root"] / "bundles", other / "bundles")
    real = SU.judge
    a_in, b_in, contended, go = threading.Event(), threading.Event(), threading.Event(), threading.Event()

    def slow(*a, **k):
        (a_in if threading.current_thread().name == "A" else b_in).set()
        assert go.wait(120), "풀리지 않았다"
        return real(*a, **k)
    monkeypatch.setattr(SU, "judge", slow)
    monkeypatch.setattr(SL, "WAIT_HOOK", lambda _lock: contended.set(), raising=False)
    codes: dict[str, int] = {}

    def call(name: str, **kw) -> None:
        try:
            codes[name] = run(w, **kw)
        except BaseException as exc:  # noqa: BLE001 — 스레드의 예외를 시험이 본다
            codes[name] = repr(exc)
    ta = threading.Thread(target=call, args=("A",), name="A")
    ta.start()
    assert a_in.wait(120), "A 가 판정 계산에 닿지 않았다"
    tb = threading.Thread(target=call, args=("B",), kwargs={"bundles": other / "bundles", "out": other}, name="B")
    tb.start()
    for _ in range(1200):
        if contended.is_set() or b_in.is_set() or not tb.is_alive():
            break
        threading.Event().wait(0.1)
    go.set()
    ta.join(120)
    tb.join(120)
    assert codes == {"A": SU.EXIT_OK, "B": SU.EXIT_ENTRY}, codes
    assert sum(r["kind"] == "frame_diag_verdict" for _, r in ledger_rows(w)) == 1
    assert contended.is_set() and not b_in.is_set(), "B 는 잠금에서 기다렸고 판정 계산에 닿지 않았다"


# ---------------------------------------------------------------- 규칙

def test_규칙은_영수증의_커밋에서_읽는다(tmp_path) -> None:
    """작업 트리의 규칙 파일을 바꿔도 판정은 커밋의 규칙으로 선다."""
    w = build(tmp_path)
    (w["repo"] / RULES_REL).write_bytes(b"{}")
    assert run(w) == SU.EXIT_OK


def test_규칙_digest_가_다르면_진입_전_종료_3(tmp_path) -> None:
    """커밋의 규칙 파일이 **유효한 다른 규칙**이면 — 꼴 검사는 지나고 digest 대조만 걸린다."""
    w = build(tmp_path)
    other = rules_file(F.FrameRules(**{**asdict(RULES), "w_r": 0.5}))
    (w["repo"] / RULES_REL).write_bytes(other)
    git(w["repo"], "commit", "-q", "-am", "규칙을 바꿨다")
    head = git(w["repo"], "rev-parse", "HEAD")
    rp = w["root"] / "registration/receipt.json"
    rc = json.loads(rp.read_text(encoding="utf-8"))
    rp.write_text(json.dumps(rc | {"main_commit": head}), encoding="utf-8")
    assert run(w) == SU.EXIT_ENTRY
    assert not (w["repo"] / "outputs/main_u/scoring_ledger.jsonl").exists(), "진입 전 거부 — 원장 줄도 없다"


# ---------------------------------------------------------------- 판정하지 않는 경우

def test_승인_구현이_아니면_묶음이_거부되고_판정하지_않는다(tmp_path) -> None:
    """진단은 대역을 받지 않는다(목적이 리허설이 아니다) — 묶음 검증이 그 칸을 거부하고 판정 줄을 쓰지 않는다(3판 1-6 의 진-4)."""
    w = build(tmp_path, meta_extra={"impl_ids": {**IMPL_IDS, "export_generator": impl(
        "export_generator", "load_generator", approved=False)}})
    assert run(w) == SU.EXIT_REJECTED
    assert "frame_diag_verdict" not in [r["kind"] for _, r in ledger_rows(w)]
    body = json.loads((w["root"] / SU.OUT_DIAG_NOT_JUDGED).read_text(encoding="utf-8"))
    assert body["reason"] == "묶음이 거부됐다" and not (w["root"] / SU.OUT_DIAG).exists()


def test_에코가_통과가_아니면_판정하지_않는다(tmp_path) -> None:
    def drop_one(rows):
        rows[0] = gen_row(IDS[0], boxes=())
        return rows
    w = build(tmp_path, echo_edit=drop_one)
    assert run(w) == SU.EXIT_REJECTED
    assert "frame_diag_verdict" not in [r["kind"] for _, r in ledger_rows(w)]
    body = json.loads((w["root"] / SU.OUT_DIAG_NOT_JUDGED).read_text(encoding="utf-8"))
    assert "카나리아" in body["reason"]
    assert not (w["root"] / SU.OUT_DIAG).exists()


def test_시험용_문턱은_검사기가_받는_꼴이다() -> None:
    assert not RULES.problems((1.0, 704 / 720)) and asdict(RULES)["min_pairs"] == 10


# ---------------------------------------------------------------- 에코 전용 — 진단 시도로 세지 않는다

def run_echo(w: dict, **kw) -> int:
    return SU.main(argv(w, **kw) + ["--echo-only"], _seam=SU.Seam(checkout=w["repo"], repo=w["repo"]), _now=NOW)


def _calls(w) -> int:
    from evaluation.scoring_ledger import count_calls
    led = w["repo"] / "outputs/main_u/scoring_ledger.jsonl"
    return count_calls(led, purpose="frame_diag", generation_sha256=w["reg"].generation_sha256()) if led.is_file() else 0


def test_에코_전용_호출은_진단의_시도_수를_늘리지_않는다(tmp_path) -> None:
    w = build(tmp_path)
    assert run_echo(w) == SU.EXIT_OK
    assert _calls(w) == 0 and not (w["repo"] / "outputs/main_u/scoring_ledger.jsonl").exists()
    assert not (w["root"] / SU.OUT_DIAG).exists() and not (w["root"] / SU.OUT_DIAG_NOT_JUDGED).exists()
    assert run(w) == SU.EXIT_OK                     # 판정 — 시도 하나
    assert _calls(w) == 1
    led = (w["repo"] / "outputs/main_u/scoring_ledger.jsonl").read_bytes()
    assert run_echo(w) == SU.EXIT_OK                # 같은 입력의 재개 — 원장은 그대로
    assert _calls(w) == 1 and (w["repo"] / "outputs/main_u/scoring_ledger.jsonl").read_bytes() == led


def test_에코_전용은_진단에서_에코가_어긋난_사유를_싣는다(tmp_path) -> None:
    def drop_one(rows):
        rows[0] = gen_row(IDS[0], boxes=())
        return rows
    w = build(tmp_path, echo_edit=drop_one)
    assert run_echo(w) == SU.EXIT_REJECTED
    body = json.loads((w["root"] / SU.OUT_ECHO_GATE).read_text(encoding="utf-8"))
    assert body["status"] == "fail" and "echo_mismatch" in body["reasons"]
    assert body["purpose"] == "frame_diag" and _calls(w) == 0


def test_에코_전용은_판정_줄에서_산출물을_되살리지_않는다(tmp_path) -> None:
    """판정 줄이 있고 산출물이 없어도 에코 전용 호출은 재생성 분기로 가지 않는다 — 진단 산출물을 만들지 않는다."""
    w = build(tmp_path)
    assert run(w) == SU.EXIT_OK
    (w["root"] / SU.OUT_DIAG).unlink()
    assert run_echo(w) == SU.EXIT_OK
    assert not (w["root"] / SU.OUT_DIAG).exists()
    assert _calls(w) == 1


def test_에코_전용도_계획_파일의_바이트가_등록과_다르면_종료_3(tmp_path, capsys) -> None:
    """에코 전용 호출의 1 단계 — 계획 파일의 바이트를 등록의 plan_sha256 과 맞댄다. 검사를 지우면 이 반례가 지난다."""
    w = build(tmp_path)
    (w["root"] / "plan.yaml").write_bytes(b"kind: frame_diag\nx: 1\n")
    capsys.readouterr()
    assert run_echo(w) == SU.EXIT_ENTRY
    assert "[PLAN_HASH]" in capsys.readouterr().err
    assert _calls(w) == 0 and not (w["root"] / SU.OUT_ECHO_GATE).exists()


def test_진단도_score_가_밖으로_이어지면_시도_전에_종료_3(tmp_path, capsys) -> None:
    """진단 산출의 경로는 원장의 판정 줄에도 적힌다 — 밖을 가리키면 판정 줄 · call 줄 전에 멈춘다(공유 회신 §57)."""
    from tests.test_score_unified import _dir_link, _unlink_dir
    w = build(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = w["root"] / "score"
    _dir_link(link, outside)
    try:
        capsys.readouterr()
        assert run(w) == SU.EXIT_ENTRY
        assert "[OUT_PATH_LINK]" in capsys.readouterr().err
        assert list(outside.iterdir()) == [] and _calls(w) == 0
        assert not (w["repo"] / "outputs/main_u/scoring_ledger.jsonl").exists()
    finally:
        _unlink_dir(link)
