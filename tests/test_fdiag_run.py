"""착수 전 프레임 진단의 오케스트레이터 경로(2판 §3-1 의 흐름 1~3) — `fdiag prepare` → 본줄기 커밋 → `fdiag receipt` → `fdiag run`.

리허설의 합성 세계(`tests/uni_rehearsal_world.py`)를 쓴다. 단계는 실제 진입점을 자식 프로세스로 돈다(학습 · 생성은 대역).
영수증 커밋은 **임시 저장소의 본줄기**다 — 실제 본줄기의 조상이 아니므로 로컬 합성 진단 영수증이고 대역을 받는다. 평가 쪽 진단 경로는
대역 묶음을 판정하지 않는다(`frame_diag_not_judged`) — 쓰는 쪽은 그 산출물의 해시와 사유를 옮겨 싣는다. 모델 · GPU 를 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.rehearsal_uni as RU
from tests import uni_rehearsal_world as RW

KST = timezone(timedelta(hours=9))
RULES_REL = "configs/registration/frame_diag_rules-20261001-1.json"
FPLAN = """kind: frame_diag
version: 1
sampling_seed: 7
seed_index: 1
cells: [uni_central]
gen: {{split: val, defect_only: true, target_boxes: 3, max_n: 8, min_groups_per_stratum: 1, max_target_tokens: 512}}
echo: {{same_as_gen: true}}
train: {{split: train, rows: {{central: 5}}, avoid_multiple_of_accum: false, central_rows: direct}}
amount: {{n: 1, r: 1, e: 1}}
exclude: {{near_dup_edges: {{path: '{edges}', sha256: {edges_sha}}}, drop_components_touching_eval: true}}
gen_limit: {{max_new_tokens: 512, provisional: true}}
timeouts_s: {{lists: 240, register: 600, train: 900, export: 600, score: 600}}
attempts_allowed: 1
"""


@pytest.fixture(scope="module", autouse=True)
def _diag_unheld():
    """진단 목록은 보류다(결정 21 의 4 · `DIAG_LISTS_ON_HOLD`) — 이 파일은 보류가 풀린 뒤의 진단 경로를 본다(자식은 시험 쪽 심이 푼다).
    보류 자체는 아래 `보류` 시험이 본다."""
    import vlm.rehearsal_lists as RL

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RL, "DIAG_LISTS_ON_HOLD", False)
        yield


def _rules() -> bytes:
    """진단 규칙 파일 — 평가 쪽 공개 함수로 정규형을 만든다. 시험용 문턱이다."""
    from evaluation import frame_diag as F
    from evaluation.frame_diag_rules import build_rules_file

    rules = F.FrameRules(n_img=10, s_min=0.7, w_s=0.1, w_r=0.0075, w_mix=0.0019, delta=0.0055, trim=0.0,
                         n_boot=40, boot_seed=1, min_pairs=10, min_groups=10, min_pair_groups=10)
    reqs = [{"noise_step": 0.1, "stat": "r", "axis": "y", "width": "w_r", "n0": 40, "n_required": 30, "g0": 20,
             "groups_required": 15, "q0": None, "pairs_required": None}]
    return build_rules_file(rules, sample_requirements=reqs, expected_sha256=RW.hx("expected"),
                            cases_sha256=RW.hx("cases"), procedure_sha256=RW.hx("procedure"), noise_ladder=[0.1],
                            summary={}, sources={})


def _main(w, argv, *, launcher=None, parent=None, **kw):
    return RU.main(argv, measure_env=RW.measure, model_loader=RW.fake_model_loader, generator_loader=RW.generator_loader,
                   checkout=w.repo, exposure_ledger=w.exposure, non_main_parent=parent or w.parent, config_path=w.config,
                   standin_allowed=True, launcher=launcher or RW.child_cmd(w.tmp), **kw)


def _prepare(tmp_path_factory, name: str):
    """흐름 1 · 2 — 진단 루트를 준비하고 등록 둘을 임시 저장소의 본줄기에 커밋한 뒤 영수증을 쓴다. 세계마다 등록이 하나다."""
    tmp = tmp_path_factory.mktemp(name)
    w = RW.build(tmp)
    (w.repo / RULES_REL).parent.mkdir(parents=True, exist_ok=True)
    (w.repo / RULES_REL).write_bytes(_rules())
    RW.git(w.repo, "add", RULES_REL)
    RW.git(w.repo, "commit", "-q", "-m", "진단 규칙")
    plan = tmp / "fplan.yaml"
    plan.write_text(FPLAN.format(edges=w.edges.as_posix(), edges_sha=hashlib.sha256(w.edges.read_bytes()).hexdigest()),
                    encoding="utf-8")
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=KST)
    assert _main(w, ["fdiag", "prepare", "--plan", str(plan), "--snapshot", str(w.snap), "--rules", RULES_REL],
                 now=now) == RU.EXIT_OK
    root = w.parent / "fdiag-20261001T120000"
    reg_dir = root / "registration"
    stem = _stem(root)                                           # 등록 단계의 날짜(자식의 시각)로 이름이 선다
    names = sorted(p.name for p in reg_dir.iterdir())
    assert names == [f"{stem}.json", f"{stem}.train_config.json"]   # 영수증은 아직 없다
    rel = f"configs/registration/{stem}.json"
    for n in names:
        shutil.copyfile(reg_dir / n, w.repo / "configs" / "registration" / n)
    RW.git(w.repo, "add", "configs/registration")
    RW.git(w.repo, "commit", "-q", "-m", "진단 등록")
    head = RW.git(w.repo, "rev-parse", "HEAD")
    assert _main(w, ["fdiag", "receipt", "--run-root", str(root), "--registration-path", rel, "--commit", head],
                 now=now + timedelta(minutes=5)) == RU.EXIT_OK
    rp = reg_dir / f"{stem}.receipt.json"
    return w, root, rp, rel, head


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    return _prepare(tmp_path_factory, "fdiag")


def _gate(ok: bool):
    """학습 전 에코 관문의 대역 — 실제 관문은 평가 쪽 호출이 없어 언제나 서지 않는다."""
    calls = []

    def gate(**kw):
        calls.append(kw["root"].name)
        return {"ok": ok, "detail": "시험의 관문"}
    gate.calls = calls
    return gate


def _stages(root: Path) -> list[str]:
    return [json.loads(x)["name"] for x in (root / "rehearsal_log.jsonl").read_text(encoding="utf-8").splitlines()
            if "name" in json.loads(x)]


def _failing(w, entry: str):
    """`entry` 단계만 종료 1 로 끝나는 자식 — 그 단계에서 죽은 실행."""
    base = RW.child_cmd(w.tmp)

    def launcher(e, argv):
        return [sys.executable, "-c", "raise SystemExit(1)"] if e == entry else base(e, argv)
    return launcher


def _verdict_then_crash(w):
    """평가 쪽 호출만 판정 줄을 쓴 직후 죽는 꼴로 바꾼 자식."""
    base = RW.child_cmd(w.tmp)
    shim = ROOT / "tests" / "uni_rehearsal_child.py"

    def launcher(e, argv):
        if e == "score":
            return [sys.executable, "-X", "utf8", str(shim), str(w.tmp), "score_verdict_then_crash", *argv]
        return base(e, argv)
    return launcher


def _stem(root: Path) -> str:
    (p,) = [p for p in (root / "registration").glob("frame_diag-*.json") if p.name.count(".") == 1]
    return p.name[: -len(".json")]


def _copy_root(w, root, name):
    """준비가 끝난 자리로 복사한다 — 실행이 낸 것(학습 · 생성 · 평가 쪽 산출 · 기록)은 옮기지 않는다."""
    dst = root.parent / name
    shutil.copytree(root, dst, ignore=shutil.ignore_patterns("logs", "rehearsal_log.jsonl", "train", "resume", "export",
                                                              "score", "frame_diag.json"))
    return dst


def _run(w, root, rp, *, device="cpu:0", gate=None, real_gate=False, **kw):
    return _main(w, ["fdiag", "run", "--run-root", str(root), "--receipt", str(rp), "--snapshot", str(w.snap),
                     "--device", device], echo_gate=None if real_gate else (gate or _gate(True)), **kw)


# ================================================================ 흐름 1 · 2
def test_준비는_목록과_진단_등록을_쓰고_멈춘다(prepared):
    from evaluation.prereg_unified import load_registration
    from vlm.pilot_vlm import train_rows_digest

    w, root, rp, rel, head = prepared
    lists = root / "lists"
    assert sorted(p.name for p in lists.iterdir()) == ["echo.txt", "gen.txt", "lists_record.json",
                                                       "train_uni_central.jsonl"]     # 참여자 목록이 없다
    rec = json.loads((lists / "lists_record.json").read_text(encoding="utf-8"))
    assert rec["central_direct"]["target"] == 5 and rec["n_train"] == {"uni_central": rec["central_direct"]["n_rows"]}
    stem = _stem(root)
    reg = load_registration((root / "registration" / f"{stem}.json").read_bytes())
    g = reg.generation
    rows = [json.loads(x) for x in (lists / "train_uni_central.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert g.purpose == "frame_diag" and g.train_rows_digest == train_rows_digest(rows)
    assert g.frame_diag_rules_path == RULES_REL
    assert g.frame_diag_rules_sha256 == hashlib.sha256(RW.git_blob(w.repo, "HEAD", RULES_REL)).hexdigest()
    assert g.max_new_tokens == 512 and (g.budget_n, g.budget_r, g.budget_e) == (1, 1, 1)
    tc = (root / "registration" / f"{stem}.train_config.json").read_bytes()
    assert hashlib.sha256(tc).hexdigest() == g.train_config_sha256 and not tc.endswith(b"\n")   # 끝 LF 없음(등록 폴더 README)
    receipt = json.loads(rp.read_text(encoding="utf-8"))
    assert receipt["kind"] == "frame_diag" and receipt["main_commit"] == head and receipt["registration_path"] == rel
    assert receipt["generation_sha256"] == reg.generation_sha256()
    log = [json.loads(x) for x in (root / "rehearsal_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r.get("kind") for r in log if r.get("kind", "").startswith("fdiag")] == ["fdiag_prepare", "fdiag_prepared"]


def test_규칙_파일이_본줄기에_없으면_루트를_만들기_전에_멈춘다(prepared, capsys):
    w, root, _rp, _rel, _head = prepared
    before = sorted(p.name for p in w.parent.iterdir())
    code = _main(w, ["fdiag", "prepare", "--plan", str(root / "plan.yaml"), "--snapshot", str(w.snap),
                     "--rules", "configs/registration/frame_diag_rules-20261001-9.json"],
                 now=datetime(2026, 10, 1, 13, 0, 0, tzinfo=KST))
    assert code == RU.EXIT_REFUSED and "[rules_not_in_main]" in capsys.readouterr().err
    assert sorted(p.name for p in w.parent.iterdir()) == before


def test_커밋한_등록이_루트의_준비_산출과_다르면_영수증을_쓰지_않는다(prepared, capsys):
    w, root, _rp, rel, head = prepared
    other = _copy_root(w, root, "fdiag-20261001T120100")
    stem = _stem(other)
    (other / "registration" / f"{stem}.receipt.json").unlink()
    p = other / "registration" / f"{stem}.train_config.json"
    p.write_bytes(p.read_bytes() + b" ")
    code = _main(w, ["fdiag", "receipt", "--run-root", str(other), "--registration-path", rel, "--commit", head])
    assert code == RU.EXIT_REFUSED and "[train_config_vs_root]" in capsys.readouterr().err
    assert not (other / "registration" / f"{stem}.receipt.json").exists()


# ================================================================ 흐름 3 — 거부(모델을 올리기 전)
@pytest.mark.parametrize(("edit", "code"), [
    (lambda r: (r / "lists" / "gen.txt").write_bytes((r / "lists" / "gen.txt").read_bytes() + b"x\n"), "lists_vs_registration"),
    (lambda r: (r / "plan.yaml").write_bytes((r / "plan.yaml").read_bytes() + b"\n"), "plan_vs_registration"),
    (lambda r: (r / "registration" / f"{_stem(r)}.json").write_bytes(b"{}"), "registration_vs_root"),
])
def test_준비_뒤에_루트가_바뀌었으면_돌지_않는다(prepared, capsys, edit, code):
    w, root, rp, _rel, _head = prepared
    other = _copy_root(w, root, f"fdiag-20261001T12{code[:2] == 'li' and '02' or code[:2] == 'pl' and '03' or '04'}00")
    edit(other)
    got = _run(w, other, other / "registration" / rp.name)
    assert got == RU.EXIT_REFUSED and f"[{code}]" in capsys.readouterr().err
    assert not (other / "logs").exists()                          # 단계를 하나도 띄우지 않았다


def test_장비_꼴이_틀리면_시작하지_않는다(prepared, capsys):
    """평가 쪽은 장비를 `{이름}:{번호}` 로 본다 — 틀린 꼴로 학습 · 생성까지 가면 평가 쪽이 묶음을 거부하고 한 번뿐인 시도를 잃는다."""
    w, root, rp, _rel, _head = prepared
    other = _copy_root(w, root, "fdiag-20261001T120600")
    assert _run(w, other, other / "registration" / rp.name, device="cuda") == RU.EXIT_REFUSED
    assert "[device_form]" in capsys.readouterr().err and not (other / "logs").exists()


def test_다른_루트의_영수증은_받지_않는다(prepared, capsys):
    w, root, rp, _rel, _head = prepared
    other = _copy_root(w, root, "fdiag-20261001T120500")
    assert _run(w, other, rp) == RU.EXIT_REFUSED and "[receipt_outside_root]" in capsys.readouterr().err


# ================================================================ 흐름 3 — 끝까지
@pytest.fixture(scope="module")
def ran(prepared):
    w, root, rp, _rel, _head = prepared
    code = _run(w, root, rp)
    return w, root, rp, code


def test_진단은_에코_학습_생성_평가_쪽_호출을_돌고_산출을_옮겨_싣는다(ran):
    _w, root, _rp, code = ran
    assert code == RU.EXIT_OK
    out = json.loads((root / "frame_diag.json").read_text(encoding="utf-8"))
    d = root / out["d_output"]["path"]
    assert out["d_output"]["sha256"] == hashlib.sha256(d.read_bytes()).hexdigest()
    body = json.loads(d.read_text(encoding="utf-8"))
    # 로컬 합성 영수증의 대역 묶음 — 평가 쪽이 판정하지 않는다. 쓰는 쪽은 그 사유를 옮긴다(판정원은 평가 쪽 하나)
    assert out["status"] == RU.NOT_JUDGED and out["reason"] == body["reason"] and out["committed_receipt"] is False
    assert [a["verdict_status"] for a in out["attempts"]] == [None] and out["attempts_allowed"] == 1
    assert set(out["inputs"]["generations"]) == {"uni_central_s1.generations.jsonl", "uni_central_s1.echo.generations.jsonl"}
    # 칸은 정해진 것뿐이다 — 기울기 · 비 · 표준오차의 값을 싣는 칸이 없다
    assert set(out) == {"kind", "run_root", "generation_sha256", "d_output", "status", "reason", "committed_receipt",
                        "attempts", "attempts_allowed", "inputs"}
    assert set(RU.D_DIAG_KEEP) <= {"status", "candidate", "reason", "samples"} | {k for k in RU.D_DIAG_KEEP if k.startswith("n_")}
    names = [json.loads(x)["name"] for x in (root / "rehearsal_log.jsonl").read_text(encoding="utf-8").splitlines()
             if "name" in json.loads(x)]
    assert names == ["lists", "register", "fdiag_echo_export", "fdiag_train", "fdiag_export", "fdiag_score"]


def test_끝난_진단은_다시_돌지_않는다(ran, capsys):
    w, root, rp, _code = ran
    assert _run(w, root, rp) == RU.EXIT_REFUSED and "[fdiag_done]" in capsys.readouterr().err


def test_같은_등록으로_새_루트를_열면_거부한다(ran, capsys):
    """루트 밖에서 본다 — 이 등록을 처음 돌린 루트가 결속돼 있다(2판 §3-1 — 같은 진단 루트에서 재개만)."""
    _w, root, rp, _code = ran
    other = _copy_root(_w, root, "fdiag-20261001T120900")
    got = _run(_w, other, other / "registration" / rp.name)
    assert got == RU.EXIT_REFUSED and "[fdiag_other_root]" in capsys.readouterr().err
    assert not (other / "logs").exists()


def test_산출물을_잃고_판정_줄도_없으면_시도로_세어_거부한다(ran, capsys):
    """평가 쪽 호출(판정하지 않음)이 원장에 남았고 산출물을 잃었다 — 다시 만들 판정 줄이 없으니 새 시도로 돌지 않는다."""
    _w, root, rp, _code = ran
    for q in ("frame_diag.json", "score/frame_diag_not_judged.json"):
        (root / q).unlink()
    n = len(_stages(root))
    got = _run(_w, root, rp)
    assert got == RU.EXIT_REFUSED and "[fdiag_attempts]" in capsys.readouterr().err and len(_stages(root)) == n


# ================================================================ 단계의 목적 갈래
def test_진단_계획의_빈_칸은_루트를_만들기_전에_멈춘다(prepared, capsys):
    """저장소의 진단 계획은 아직 간선 · 시한이 비었다(`null`) — 준비가 그 자리에서 멈춘다."""
    w, *_ = prepared
    before = sorted(p.name for p in w.parent.iterdir())
    code = _main(w, ["fdiag", "prepare", "--plan", str(ROOT / "configs" / "frame_diag_uni.yaml"), "--snapshot", str(w.snap),
                     "--rules", RULES_REL], now=datetime(2026, 10, 1, 14, 0, 0, tzinfo=KST))
    assert code == RU.EXIT_REFUSED and "[plan_value_missing]" in capsys.readouterr().err
    assert sorted(p.name for p in w.parent.iterdir()) == before


@pytest.mark.parametrize(("tag", "extra", "code"), [
    ("uni_central", (), "receipt_missing"),
    ("uni_local_C1", ("--receipt", "R"), "tag"),
    ("uni_fed", ("--receipt", "R"), "fed_args"),
])
def test_진단_루트의_학습은_영수증을_든_중앙_칸뿐이다(prepared, capsys, tag, extra, code):
    w, root, rp, _rel, _head = prepared
    other = _copy_root(w, root, f"fdiag-20261001T1207{['uni_central', 'uni_local_C1', 'uni_fed'].index(tag):02d}")
    args = tuple(str(other / "registration" / rp.name) if x == "R" else x for x in extra)
    got = _main(w, ["train", "--run-root", str(other), "--plan", str(other / "plan.yaml"), "--snapshot", str(w.snap),
                    "--tag", tag, *args])
    assert got == RU.EXIT_REFUSED and f"[{code}]" in capsys.readouterr().err
    assert not (other / "train").exists()


def test_리허설_등록은_규칙_파일을_받지_않는다(prepared, capsys):
    w, *_ = prepared
    reh = w.parent / "reh-20261001T121000"
    reh.mkdir()
    shutil.copyfile(w.plan, reh / "plan.yaml")
    got = _main(w, ["register", "--run-root", str(reh), "--plan", str(reh / "plan.yaml"), "--rules", RULES_REL])
    assert got == RU.EXIT_REFUSED and "[rules_not_for_purpose]" in capsys.readouterr().err
    assert not (reh / "registration").exists()


# ================================================================ 남은 산출물의 출처 · 학습 전 에코 관문 · 첫 루트(세계 하나)
@pytest.fixture(scope="module")
def prepared3(tmp_path_factory):
    return _prepare(tmp_path_factory, "fdiag3")


@pytest.mark.parametrize("case", ["other_registration", "not_in_ledger", "link_outside"])
def test_루트에_남은_평가_쪽_산출물은_출처를_본_뒤에만_옮긴다(prepared3, capsys, case):
    """이름만 보고 받지 않는다 — 다른 등록의 결과 · 원장에 없는 결과 · 루트 밖으로 가는 링크(외부 검토 회신 `fdiag` 의 1)."""
    import os

    w, root, rp, _rel, _head = prepared3
    score = root / "score"
    score.mkdir(exist_ok=True)
    gen = json.loads(rp.read_text(encoding="utf-8"))["generation_sha256"]
    if case == "other_registration":
        target = score / "frame_diag_not_judged.json"
        target.write_text(json.dumps({"kind": "frame_diag_not_judged", "generation_sha256": "0" * 64,
                                      "call_sha256": "1" * 64, "reason": "다른 등록"}), encoding="utf-8")
        code = "fdiag_output_provenance"
    elif case == "not_in_ledger":
        target = score / "frame_diag_score.json"
        target.write_text(json.dumps({"generation_sha256": gen, "verdict_line_sha256": "2" * 64, "status": "통과",
                                      "call_sha256": "3" * 64}), encoding="utf-8")
        code = "fdiag_output_provenance"
    else:
        outside = w.tmp / "outside_score.json"
        outside.write_text(json.dumps({"generation_sha256": gen}), encoding="utf-8")
        target = score / "frame_diag_score.json"
        os.link(outside, target)
        code = "fdiag_output_path"
    try:
        got = _run(w, root, rp)
        assert got == RU.EXIT_REFUSED and f"[{code}]" in capsys.readouterr().err
        assert not (root / "frame_diag.json").exists() and len(_stages(root)) == 2      # 준비의 두 단계 뒤로 띄운 단계가 없다
    finally:
        target.unlink()
    assert not (w.parent / RU.BINDINGS_DIR).exists()            # 거부된 실행은 루트를 결속하지 않는다


def test_에코_관문이_통과가_아니면_학습을_시작하지_않는다(prepared3):
    """2판 §3-1 — "통과가 아니면 학습을 시작하지 않는다". 에코 생성 뒤 관문에서 멈추고 학습 단계는 0 번이다(외부 검토 회신 `fdiag` 의 2)."""
    w, root, rp, _rel, _head = prepared3
    gate = _gate(False)
    assert _run(w, root, rp, gate=gate) == RU.EXIT_HALT
    assert gate.calls == [root.name]
    assert _stages(root) == ["lists", "register", "fdiag_echo_export"]          # 학습 · 생성 · 평가 쪽 호출 없음
    assert not (root / "train").exists()


def test_평가_쪽_호출_전에_죽으면_같은_루트에서만_재개한다(prepared3, capsys):
    """에코 관문에서 멈춘 실행(앞 시험)으로 이 등록은 이 루트에 결속됐다 — 다른 루트는 거부, 같은 루트는 이어서 돈다(외부 검토 회신 `fdiag` 의 3)."""
    w, root, rp, _rel, _head = prepared3
    other = _copy_root(w, root, "fdiag-20261001T121500")
    assert _run(w, other, other / "registration" / rp.name) == RU.EXIT_REFUSED
    assert "[fdiag_other_root]" in capsys.readouterr().err and not (other / "logs").exists()
    # 같은 루트 — 학습에서 죽는다(평가 쪽 호출 전)
    assert _run(w, root, rp, launcher=_failing(w, "train")) == RU.EXIT_HALT
    assert _stages(root)[-1] == "fdiag_train" and not list(root.glob("score/*.json"))
    # 같은 루트 — 이어서 끝까지. 닫힌 에코 생성은 건너뛴다
    before = len(_stages(root))
    assert _run(w, root, rp) == RU.EXIT_OK
    assert _stages(root)[before:] == ["fdiag_train", "fdiag_export", "fdiag_score"]
    assert json.loads((root / "frame_diag.json").read_text(encoding="utf-8"))["status"] == RU.NOT_JUDGED
    assert _run(w, other, other / "registration" / rp.name) == RU.EXIT_REFUSED     # 끝난 뒤에도 다른 루트는 거부
    assert "[fdiag_other_root]" in capsys.readouterr().err


# ================================================================ 판정 줄만 남은 경우의 다시 만들기(세계 하나)
@pytest.fixture(scope="module")
def prepared4(tmp_path_factory):
    return _prepare(tmp_path_factory, "fdiag4")


def test_판정_줄만_남고_산출물이_없으면_평가_쪽이_그_줄에서_다시_만든다(prepared4):
    """평가 쪽은 판정 줄을 산출물보다 먼저 쓴다 — 그 사이에 죽으면 시도 수는 닿았고 산출물은 없다. 학습 · 생성 · 판정 계산을 다시 하지 않고
    평가 쪽 호출 하나로 그 줄에서 산출물을 되살린다(평가 쪽은 `call` 줄을 쓰지 않는다 — 외부 검토 회신 `fdiag` 의 4).
    판정 줄의 값은 시험이 지어낸 것이다(대역 묶음은 평가 쪽이 판정하지 않는다)."""
    from evaluation.scoring_ledger import read_ledger

    w, root, rp, _rel, _head = prepared4
    assert _run(w, root, rp, launcher=_verdict_then_crash(w)) == RU.EXIT_HALT
    assert _stages(root)[-1] == "fdiag_score" and not (root / "score" / "frame_diag_score.json").exists()
    rows_before = read_ledger(w.scoring)
    before = len(_stages(root))
    assert _run(w, root, rp) == RU.EXIT_OK
    assert _stages(root)[before:] == ["fdiag_score_regenerate"]                   # 학습 · 생성을 다시 띄우지 않는다
    rows_after = read_ledger(w.scoring)
    kinds = [r["kind"] for _s, r in rows_after[len(rows_before):]]
    assert kinds == ["frame_diag_regenerated"]                                     # 새 call 줄 없음 — 새 시도가 아니다
    out = json.loads((root / "frame_diag.json").read_text(encoding="utf-8"))
    assert out["status"] == "통과" and out["d_output"]["kind"] == "frame_diag_verdict"
    assert [a["verdict_status"] for a in out["attempts"]] == ["통과"]


def test_산출물의_등록_지문이_다르면_원장의_호출이_맞아도_받지_않는다(tmp_path):
    """출처 검사의 첫 칸 — 본문의 `generation_sha256` 만 다른 꼴(원장의 호출 · 묶음은 이 등록 · 이 루트의 것)."""
    from datetime import datetime as dt

    from evaluation.scoring_ledger import append_line
    from vlm.rehearsal_run import StageRefused

    root = tmp_path / "fdiag-20261001T130000"
    (root / "export").mkdir(parents=True)
    (root / "score").mkdir()
    gen_file = root / "export" / "uni_central_s1.generations.jsonl"
    gen_file.write_bytes(b'{"image_id": "v1"}\n')
    ledger = tmp_path / "ledger.jsonl"
    mine, other = "a" * 64, "b" * 64
    call = append_line(ledger, "call", {"purpose": "frame_diag", "generation_sha256": mine,
                                        "bundles": [{"generations_sha256": hashlib.sha256(gen_file.read_bytes()).hexdigest()}]},
                       at=dt(2026, 10, 1, 13, 0, tzinfo=KST))
    out = root / "score" / "frame_diag_not_judged.json"
    body = {"kind": "frame_diag_not_judged", "call_sha256": call, "reason": "x"}
    out.write_text(json.dumps({**body, "generation_sha256": mine}), encoding="utf-8")
    assert RU._d_output(root, out, gen_sha=mine, ledger=ledger, checkout=tmp_path)[0]["reason"] == "x"   # 맞는 꼴은 받는다
    out.write_text(json.dumps({**body, "generation_sha256": other}), encoding="utf-8")
    with pytest.raises(StageRefused) as exc:
        RU._d_output(root, out, gen_sha=mine, ledger=ledger, checkout=tmp_path)
    assert exc.value.code == "fdiag_output_provenance" and "다른 등록" in str(exc.value)


# ================================================================ 결속의 신원 — 루트 별칭(세계 하나)
@pytest.fixture(scope="module")
def prepared5(tmp_path_factory):
    return _prepare(tmp_path_factory, "fdiag5")


def _alias(w, name: str, target: Path) -> Path:
    """`target` 을 가리키는 정션 — 끝 이름이 `name` 이다. 이 시험의 임시 폴더 안에만 만든다."""
    import _winapi

    d = w.tmp / f"alias_{name}"
    d.mkdir()
    link = d / name
    _winapi.CreateJunction(str(target.resolve()), str(link))
    return link


def test_끝_이름이_같은_별칭으로_다른_준비_사본을_넘기면_결속이_거부한다(prepared5, capsys):
    """처음 돌린 루트 A 와 **끝 이름이 같고** 실제로는 준비 사본 B 를 가리키는 정션 — 이름으로 견주면 지난다. 푼 경로로 견주어 거부한다
    (외부 검토 회신 `fdiag2` 의 3)."""
    import os

    w, root, rp, _rel, _head = prepared5
    assert _run(w, root, rp, gate=_gate(False)) == RU.EXIT_HALT                 # A 가 결속된다(평가 쪽 호출 전에 멈춘 실행)
    other = _copy_root(w, root, "fdiag-20261001T121700")                       # 준비 사본 B — 비본실험 부모 바로 아래
    link = _alias(w, root.name, other)
    try:
        got = _run(w, link, link / "registration" / rp.name)
        assert got == RU.EXIT_REFUSED and "[fdiag_other_root]" in capsys.readouterr().err
    finally:
        os.rmdir(link)                                                         # 링크만 뗀다 — 가리키던 폴더는 남는다
    assert not (other / "logs").exists() and (other / "plan.yaml").is_file()


def test_다른_이름의_별칭이라도_처음_루트를_가리키면_이어서_돈다(prepared5):
    """결속의 신원은 푼 경로다 — 끝 이름이 다른 별칭이 처음 루트 A 를 가리키면 같은 루트로 이어서 돈다(앞 시험의 멈춘 실행 뒤)."""
    import os

    w, root, rp, _rel, _head = prepared5
    link = _alias(w, "fdiag-20261001T235959", root)
    before = len(_stages(root))
    try:
        assert _run(w, link, link / "registration" / rp.name) == RU.EXIT_OK
    finally:
        os.rmdir(link)
    assert _stages(root)[before:] == ["fdiag_train", "fdiag_export", "fdiag_score"]
    body = json.loads((w.parent / RU.BINDINGS_DIR / f"{json.loads(rp.read_text(encoding='utf-8'))['generation_sha256']}.json")
                      .read_text(encoding="utf-8"))
    assert body["run_root"] == root.resolve().as_posix()                        # 푼 경로로 적혔다
    assert (root / "frame_diag.json").is_file()


# ================================================================ 결속 파일의 자리(외부 검토 §56 — 세계 하나)
@pytest.fixture(scope="module")
def prepared6(tmp_path_factory):
    return _prepare(tmp_path_factory, "fdiag6")


def _symlink_or_skip(src: Path, dst: Path, *, directory: bool) -> None:
    import os

    try:
        os.symlink(src, dst, target_is_directory=directory)
    except OSError as exc:                                       # 기호 링크는 권한이 있어야 만든다(개발자 모드 · 관리자)
        pytest.skip(f"이 환경은 기호 링크를 만들 수 없다: {exc}")


@pytest.mark.parametrize("case", ["dir_junction", "dir_symlink", "file_hardlink", "file_symlink"])
def test_결속_폴더나_파일이_밖을_가리키면_읽지도_쓰지도_않는다(prepared6, capsys, case):
    """`fdiag_bindings` 를 비본실험 부모 밖으로 잇는 정션 · 기호 링크, 결속 파일의 밖으로 가는 하드 링크 · 기호 링크 — 단계 전에 거부하고
    밖에는 아무것도 생기지 않는다. 밖의 결속 파일은 이 루트를 첫 루트로 적어 두었다(이름만 보면 지나는 꼴)."""
    import _winapi
    import os

    w, root, rp, _rel, _head = prepared6
    gen = json.loads(rp.read_text(encoding="utf-8"))["generation_sha256"]
    outside = w.tmp / f"outside_{case}"
    outside.mkdir()
    forged = outside / f"{gen}.json"
    bdir = w.parent / RU.BINDINGS_DIR
    made: list[Path] = []
    try:
        if case.startswith("dir"):
            if case == "dir_junction":
                _winapi.CreateJunction(str(outside.resolve()), str(bdir))
            else:
                _symlink_or_skip(outside.resolve(), bdir, directory=True)
            made.append(bdir)
        else:
            forged.write_text(json.dumps({"run_root": root.resolve().as_posix()}), encoding="utf-8")
            bdir.mkdir()
            if case == "file_hardlink":
                os.link(forged, bdir / f"{gen}.json")
            else:
                _symlink_or_skip(forged, bdir / f"{gen}.json", directory=False)
            made += [bdir / f"{gen}.json", bdir]
        before = sorted(q.name for q in outside.iterdir())
        got = _run(w, root, rp, gate=_gate(False))
        assert got == RU.EXIT_REFUSED and "[fdiag_binding_path]" in capsys.readouterr().err
        assert len(_stages(root)) == 2                                          # 준비의 두 단계 뒤로 띄운 단계가 없다
        assert sorted(q.name for q in outside.iterdir()) == before              # 밖에 새 결속 파일이 생기지 않았다
    finally:
        for q in made:
            if q.is_dir() and not q.is_symlink() and os.path.normcase(str(q.resolve())) == os.path.normcase(str(q)):
                os.rmdir(q)                                                     # 이 시험이 만든 빈 보통 폴더
            elif q.is_dir():
                os.rmdir(q)                                                     # 정션 · 폴더 기호 링크는 링크만 뗀다
            elif q.exists() or q.is_symlink():
                q.unlink()


def test_비본실험_부모_위의_공유_정션은_막지_않는다(prepared6):
    """`outputs` 같은 위쪽의 공유 정션은 결속을 막지 않는다 — 부모를 그 정션으로 주어도 결속은 푼 부모 아래에 선다."""
    import _winapi
    import os

    w, root, rp, _rel, _head = prepared6
    alias = w.tmp / "parent_alias"
    _winapi.CreateJunction(str(w.parent.resolve()), str(alias))
    try:
        assert _run(w, root, rp, gate=_gate(False), parent=alias) == RU.EXIT_HALT          # 에코 관문에서 멈춘다(학습 전)
    finally:
        os.rmdir(alias)
    gen = json.loads(rp.read_text(encoding="utf-8"))["generation_sha256"]
    body = json.loads((w.parent / RU.BINDINGS_DIR / f"{gen}.json").read_text(encoding="utf-8"))
    assert body["run_root"] == root.resolve().as_posix()


# ================================================================ 학습 전 에코 관문 — 평가 쪽 에코 전용 호출(지시 20261005l 의 2)
def _echo_fake(w, entry: str):
    """평가 쪽 호출 가운데 **에코 전용**만 대역(`entry`)으로 바꾼 자식 — 진단 호출은 실제 진입점이다."""
    base = RW.child_cmd(w.tmp)
    shim = ROOT / "tests" / "uni_rehearsal_child.py"

    def launcher(e, argv):
        if e == "score":
            return [sys.executable, "-X", "utf8", str(shim), str(w.tmp), entry, *argv]
        return base(e, argv)
    return launcher


def _calls(w, gen: str) -> int:
    from evaluation.scoring_ledger import count_calls

    return count_calls(w.scoring, purpose="frame_diag", generation_sha256=gen) if w.scoring.is_file() else 0


@pytest.fixture(scope="module")
def prepared7(tmp_path_factory):
    return _prepare(tmp_path_factory, "fdiag7")


def test_실제_에코_관문은_평가_쪽_에코_전용_호출을_부르고_통과가_아니면_학습하지_않는다(prepared7):
    """실제 관문 — 평가 쪽 `--echo-only` 를 자식 단계로 부르고 산출을 읽는다. 대역 묶음은 평가 쪽이 통과를 내지 않는다 — 학습 0 번,
    채점 원장은 그대로(시도가 늘지 않는다). 같은 루트에서 다시 불러도 같다(평가 쪽은 같은 바이트의 산출을 받는다)."""
    w, root, rp, _rel, _head = prepared7
    gen = json.loads(rp.read_text(encoding="utf-8"))["generation_sha256"]
    ledger = w.scoring.read_bytes() if w.scoring.is_file() else None
    for _ in range(2):
        assert _run(w, root, rp, real_gate=True) == RU.EXIT_HALT
        body = json.loads((root / "score" / "echo_gate.json").read_text(encoding="utf-8"))
        assert body["kind"] == "echo_gate" and body["status"] == "fail" and body["reasons"]
        assert not (root / "train").exists() and "fdiag_train" not in _stages(root)
        assert (w.scoring.read_bytes() if w.scoring.is_file() else None) == ledger and _calls(w, gen) == 0
    assert _stages(root) == ["lists", "register", "fdiag_echo_export", "fdiag_echo_gate", "fdiag_echo_gate"]


@pytest.fixture(scope="module")
def prepared8(tmp_path_factory):
    return _prepare(tmp_path_factory, "fdiag8")


def test_에코가_어긋나면_학습하지_않고_시도도_늘지_않는다(prepared8):
    w, root, rp, _rel, _head = prepared8
    gen = json.loads(rp.read_text(encoding="utf-8"))["generation_sha256"]
    assert _run(w, root, rp, real_gate=True, launcher=_echo_fake(w, "score_echo_mismatch")) == RU.EXIT_HALT
    assert _stages(root)[-1] == "fdiag_echo_gate" and not (root / "train").exists()
    rows = [json.loads(x) for x in (root / "rehearsal_log.jsonl").read_text(encoding="utf-8").splitlines()]
    (gate,) = [r for r in rows if r.get("kind") == "fdiag_echo_gate"]
    assert gate == {"kind": "fdiag_echo_gate", "ok": False, "detail": ["echo_mismatch"]}
    assert _calls(w, gen) == 0


def test_에코_관문_산출이_다른_등록의_것이면_받지_않는다(prepared8, capsys):
    w, root, rp, _rel, _head = prepared8
    assert _run(w, root, rp, real_gate=True, launcher=_echo_fake(w, "score_echo_other_registration")) == RU.EXIT_REFUSED
    assert "[echo_gate_provenance]" in capsys.readouterr().err and not (root / "train").exists()


def test_에코_관문_산출이_이_루트의_에코_목록을_본_것이_아니면_받지_않는다(prepared8, capsys):
    """등록 지문은 맞고 산출이 본 에코 목록의 해시만 다른 꼴 — 받지 않는다(산출의 `inputs` 대조)."""
    w, root, rp, _rel, _head = prepared8
    assert _run(w, root, rp, real_gate=True, launcher=_echo_fake(w, "score_echo_other_list")) == RU.EXIT_REFUSED
    assert "[echo_gate_provenance]" in capsys.readouterr().err and not (root / "train").exists()


def test_에코_관문이_통과면_학습으로_이어지고_시도는_진단_호출_하나다(prepared8):
    w, root, rp, _rel, _head = prepared8
    gen = json.loads(rp.read_text(encoding="utf-8"))["generation_sha256"]
    before = len(_stages(root))
    assert _run(w, root, rp, real_gate=True, launcher=_echo_fake(w, "score_echo_pass")) == RU.EXIT_OK
    assert _stages(root)[before:] == ["fdiag_echo_gate", "fdiag_train", "fdiag_export", "fdiag_score"]
    assert _calls(w, gen) == 1                                                  # 에코 관문은 시도로 세지 않는다
    assert json.loads((root / "frame_diag.json").read_text(encoding="utf-8"))["status"] == RU.NOT_JUDGED


@pytest.fixture(scope="module")
def prepared9(tmp_path_factory):
    return _prepare(tmp_path_factory, "fdiag9")


def test_에코_관문_산출이_밖으로_가는_링크면_읽지_않는다(prepared9, capsys):
    """쓰는 쪽은 산출을 루트 아래의 보통 파일로만 읽는다 — 밖의 통과 산출로 가는 하드 링크를 두면 거부하고 학습하지 않는다."""
    import os

    w, root, rp, _rel, _head = prepared9
    outside = w.tmp / "outside_echo_gate.json"
    outside.write_text(json.dumps({"kind": "echo_gate", "status": "pass"}), encoding="utf-8")
    (root / "score").mkdir()
    os.link(outside, root / "score" / "echo_gate.json")
    try:
        assert _run(w, root, rp, real_gate=True, launcher=_echo_fake(w, "score_echo_noop")) == RU.EXIT_REFUSED
        assert "[echo_gate_path]" in capsys.readouterr().err and not (root / "train").exists()
    finally:
        (root / "score" / "echo_gate.json").unlink()


# ================================================================ 에코 입력을 고르는 규칙 · 그 읽기 · 밖으로 가는 score(지시 20261005m)
def test_묶음을_고르는_규칙이_평가_쪽과_같다():
    """쓰는 쪽의 `GEN_NAME` 은 평가 쪽 `_NAME` 과 같은 꼴이다 — 한쪽만 바뀌면 이 시험이 떨어진다(외부 검토 회신 `echowire` 의 1)."""
    import scripts.probe.score_unified as SU

    assert RU.GEN_NAME.pattern == SU._NAME.pattern


@pytest.fixture(scope="module")
def prepared10(tmp_path_factory):
    return _prepare(tmp_path_factory, "fdiag10")


def test_밖으로_가는_score_는_사유_코드로_거부한다(prepared10, capsys):
    """`<루트>/score` 가 체크아웃 밖을 가리키는 정션 — 경로를 풀다 예외로 끝나지 않고 평가 쪽을 부르기 전에 거부한다(`echowire` 의 2)."""
    import _winapi
    import os

    w, root, rp, _rel, _head = prepared10
    outside = w.tmp / "outside_score"
    outside.mkdir()
    _winapi.CreateJunction(str(outside.resolve()), str(root / "score"))
    try:
        got = _run(w, root, rp, real_gate=True, launcher=_echo_fake(w, "score_echo_mismatch"))
        assert got == RU.EXIT_REFUSED and "[fdiag_output_path]" in capsys.readouterr().err
        assert len(_stages(root)) == 2 and not list(outside.iterdir())
    finally:
        os.rmdir(root / "score")


def _echo_gate_row(root):
    rows = [json.loads(x) for x in (root / "rehearsal_log.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r.get("kind") == "fdiag_echo_gate"][-1]


@pytest.mark.parametrize("extra", ["plain", "hardlink_outside"])
def test_평가_쪽이_고르지_않는_생성_파일은_쓰는_쪽도_읽지_않는다(prepared10, extra):
    """`export/backup.echo.generations.jsonl` 은 평가 쪽 이름 규칙 밖이다 — 평가 쪽이 보지 않으니 쓰는 쪽도 맞대지 않는다(받은 산출이 그대로 선다).
    그 파일이 밖으로 가는 하드 링크여도 읽지 않는다. 관문은 대역이 낸 불일치로 멈춘다(출처 검사는 지났다)."""
    import os

    w, root, rp, _rel, _head = prepared10
    if not (root / "export").exists():
        assert _run(w, root, rp, real_gate=True, launcher=_echo_fake(w, "score_echo_mismatch")) == RU.EXIT_HALT
    backup = root / "export" / "backup.echo.generations.jsonl"
    if extra == "plain":
        backup.write_bytes(b'{"image_id": "x"}\n')
    else:
        outside = w.tmp / "outside_backup.jsonl"
        outside.write_bytes(b'{"image_id": "y"}\n')
        os.link(outside, backup)
    try:
        assert _run(w, root, rp, real_gate=True, launcher=_echo_fake(w, "score_echo_mismatch")) == RU.EXIT_HALT
        assert _echo_gate_row(root) == {"kind": "fdiag_echo_gate", "ok": False, "detail": ["echo_mismatch"]}
    finally:
        backup.unlink()


def test_고른_에코_생성_파일이_밖으로_가는_링크면_읽지_않는다(prepared10, capsys):
    """평가 쪽 규칙으로 고른 파일도 `read_under` 와 같은 보호로 읽는다 — 같은 바이트의 밖 파일로 가는 하드 링크면 거부하고 학습하지 않는다."""
    import os
    import shutil

    w, root, rp, _rel, _head = prepared10
    if not (root / "export").exists():
        assert _run(w, root, rp, real_gate=True, launcher=_echo_fake(w, "score_echo_mismatch")) == RU.EXIT_HALT
    echo = root / "export" / "uni_central_s1.echo.generations.jsonl"
    keep = w.tmp / "echo_keep.jsonl"
    shutil.copyfile(echo, keep)
    echo.unlink()
    os.link(keep, echo)
    try:
        assert _run(w, root, rp, real_gate=True, launcher=_echo_fake(w, "score_echo_pass")) == RU.EXIT_REFUSED
        assert "[echo_gate_path]" in capsys.readouterr().err and not (root / "train").exists()
    finally:
        echo.unlink()
        shutil.copyfile(keep, echo)


# ================================================================ 진단 산출 파일 자체가 밖으로 가는 링크(지시 20261005n 의 덧붙임 · §59)
def test_진단_산출의_자리는_파일을_풀지_않고_루트가_체크아웃_밖이면_사유_코드로_거부한다(tmp_path):
    """자리는 푼 루트 + 이름이다 — 파일이 밖으로 가는 링크여도 여기서 `ValueError` 로 끝나지 않는다. 루트가 체크아웃 밖이면 `fdiag_output_path`."""
    from vlm.rehearsal_run import StageRefused

    co = tmp_path / "co"
    root = co / "outputs" / "rehearsal_u" / "fdiag-20261001T130000"
    root.mkdir(parents=True)
    assert RU._diag_out_rel(root, co) == "outputs/rehearsal_u/fdiag-20261001T130000/score/frame_diag_score.json"
    with pytest.raises(StageRefused) as exc:
        RU._diag_out_rel(root, tmp_path / "elsewhere")
    assert exc.value.code == "fdiag_output_path"


def test_진단_산출_파일이_밖으로_가는_기호_링크면_사유_코드로_거부한다(prepared10, capsys):
    """`score` 는 보통 폴더이고 `score/frame_diag_score.json` 만 체크아웃 밖을 가리키는 기호 링크 — 예외가 아니라 `fdiag_output_path` 로 멈춘다
    (단계 없이). 이 환경이 파일 기호 링크를 만들 수 없으면 건너뛴다 — 같은 갈래는 위의 단위 시험이 본다."""
    import os

    w, root, rp, _rel, _head = prepared10
    outside = w.tmp / "outside_diag_score.json"
    outside.write_text("{}", encoding="utf-8")
    (root / "score").mkdir(exist_ok=True)
    link = root / "score" / "frame_diag_score.json"
    try:
        os.symlink(outside, link)
    except OSError as exc:
        pytest.skip(f"이 환경은 파일 기호 링크를 만들 수 없다: {exc}")
    try:
        before = len(_stages(root))
        assert _run(w, root, rp, real_gate=True, launcher=_echo_fake(w, "score_echo_mismatch")) == RU.EXIT_REFUSED
        assert "[fdiag_output_path]" in capsys.readouterr().err and len(_stages(root)) == before
    finally:
        link.unlink()


# ================================================================ 진단 보류(결정 21 의 4 · 덧붙임의 2)
def test_보류_중에는_진단_준비가_루트를_만들기_전에_거부한다(prepared, capsys, monkeypatch):
    import vlm.rehearsal_lists as RL

    monkeypatch.setattr(RL, "DIAG_LISTS_ON_HOLD", True)
    w, root, _rp, _rel, _head = prepared
    before = sorted(p.name for p in w.parent.iterdir())
    code = _main(w, ["fdiag", "prepare", "--plan", str(root / "plan.yaml"), "--snapshot", str(w.snap), "--rules", RULES_REL],
                 now=datetime(2026, 10, 1, 15, 0, 0, tzinfo=KST))
    assert code == RU.EXIT_REFUSED and "[lists_diag_on_hold]" in capsys.readouterr().err
    assert sorted(p.name for p in w.parent.iterdir()) == before


def test_보류_중에는_목록_단계가_매니페스트를_읽기_전에_거부한다(prepared, capsys, monkeypatch):
    """목록 단계(`lists`)를 진단 루트에 직접 불러도 같은 사유 하나로 멈춘다 — 목록 폴더를 만들지 않는다."""
    import vlm.rehearsal_lists as RL

    monkeypatch.setattr(RL, "DIAG_LISTS_ON_HOLD", True)
    w, root, _rp, _rel, _head = prepared
    other = w.parent / "fdiag-20261001T150100"
    other.mkdir()
    shutil.copyfile(root / "plan.yaml", other / "plan.yaml")
    code = _main(w, ["lists", "--run-root", str(other), "--plan", str(other / "plan.yaml"), "--snapshot", str(w.snap)])
    assert code == RU.EXIT_REFUSED and "[lists_diag_on_hold]" in capsys.readouterr().err
    assert not (other / "lists").exists()
