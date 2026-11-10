"""A3 대체 승인(D7, 미니스펙 `_workspace/2026-10-31-main2/MINISPEC_a3_substitute.md` §7) — 학습 쪽 시작 문과 채점 쪽 프레임 표지가
**같은 검증 함수**로 같은 판정을 내는지, 코드 · 증거가 다르면 둘 다 거부하는지 본다.

합성 git 저장소에 두 커밋을 둔다 — "A3 커밋"(동일성 파일 · 설정, 배치 1)과 "지금"(같은 동일성, 배치 8, 설계 결정 기록 절 추가).
증거 넷은 합성 JSON · 문서다(실제 A3 산출이 아니다).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import early_coord_check as E
from evaluation import ecc_substitute as S
from evaluation.ecc_frame_check import FRAME_CHECKED_BY_A3_SUBSTITUTE, frame_check_by_a3_substitute
from vlm import ecc as VE

SECTION = "## 2026-10-31 — 시험용 A3 대체 승인 결정"
PROMPT = "원본 이미지의 절대 픽셀로 결함 상자를 적어라.\n"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                           "-c", "core.autocrlf=false", *args], capture_output=True, check=True).stdout.decode("utf-8").strip()


def base_yaml(batch: int = 1, coord: str = "ABS_ORIG") -> str:
    return ("fixed_before_main_runs:\n  uni_model: {id: Qwen/Qwen3.5-4B, revision: " + "a" * 40 + "}\n"
            "  uni_processor_kwargs: {max_pixels: 16777216, min_pixels: 65536}\n"
            "  uni_chat_template_kwargs: {enable_thinking: false}\n"
            f"  uni_prompt_path: {S.PROMPT_REL}\n  uni_coord_space: {coord}\n  uni_max_new_tokens: 1280\n"
            f"  uni_pairs: {{path: data/processed/pairs_main_v2/pairs.jsonl, digest: {'b' * 64}}}\n  uni_batch_size: {batch}\n")


def write_files(repo: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))


def commit(repo: Path, msg: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def evidence(d: Path, *, a3_commit: str, check_over=None, result_over=None, probe_over=None, n_rows: int = 16) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    chk = {"pass": True, "execution_commit": a3_commit, "result_member": "return_a3/a3-test-0001/result.json",
           "a3_values": {"precision": 1.0, "recall": 1.0, "threshold": 0.9, "scope": "train_box.class_aware_iou50", "pass": True}}
    chk.update(check_over or {})
    res = {"run_name": "a3-test-0001", "regime": {"recipe": "json_line_then_basis",
                                                 "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
                                                 "max_new_tokens": 1280}}
    res.update(result_over or {})
    probe = {"n_rows": n_rows, "passes": {"1": {"status": "ok", "records": {f"r{i}": {} for i in range(n_rows)}},
                                          "8": {"status": "ok", "compare_vs_batch1": {"tokens_equal_rate": 1.0}}}}
    if probe_over:
        probe_over(probe)
    paths = {"a3_check": d / "A3_CHECK_258.json", "a3_result": d / "result.json", "a3_gate_form": d / "A3_GATE_FORM.md",
             "batch_equivalence": d / "batch_equivalence.json"}
    paths["a3_check"].write_text(json.dumps(chk), encoding="utf-8")
    paths["a3_result"].write_text(json.dumps(res), encoding="utf-8")
    paths["a3_gate_form"].write_text("# A3 양식 — G1 판단 가(시험용)\n", encoding="utf-8")
    paths["batch_equivalence"].write_text(json.dumps(probe), encoding="utf-8")
    return paths


def make_world(tmp: Path, *, approval_over=None, **ev_kw) -> SimpleNamespace:
    repo = tmp / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    ident = {p: f"# {p} 동일성 대상(시험용)\nx = 1\n" for p in S.IDENTITY_FILES if p != S.PROMPT_REL}
    write_files(repo, {**ident, S.PROMPT_REL: PROMPT, S.CONFIG_REL: base_yaml(1),
                       S.TRAIN_REG_REL: "recipe: json_line_then_basis\n", S.DECISION_LOG_REL: "# 의사결정로그\n"})
    a3 = commit(repo, "A3 실행 커밋")
    write_files(repo, {S.CONFIG_REL: base_yaml(8), S.DECISION_LOG_REL: f"# 의사결정로그\n\n{SECTION}\n- 사용자 결정\n"})
    head = commit(repo, "지금")
    d = tmp / "approval"
    paths = evidence(d, a3_commit=a3, **ev_kw)
    doc = S.build_approval(repo=repo, rev="HEAD", a3_commit=a3, evidence=paths, decision="A3 결과로 조기 좌표 점검을 대신한다",
                           decision_log_section=SECTION, decided_at="2026-10-31T09:00:00+09:00")
    if approval_over:
        approval_over(doc)
    raw = S.dump_approval(doc)
    (d / "ecc_a3_substitute.json").write_bytes(raw)
    return SimpleNamespace(repo=repo, a3=a3, head=head, dir=d, path=d / "ecc_a3_substitute.json", raw=raw, doc=doc, paths=paths)


def train_side(w, seed_dir: Path, *, pin="p" * 64, permit=None):
    return VE.check_a3_substitute(w.path, seed_dir=seed_dir, permit_path=permit, pin_sha256=pin, seed=1,
                                  start_adapter_digest="d" * 64, repo=w.repo)


def score_side(w, rev=None):
    return frame_check_by_a3_substitute(w.path.read_bytes(), evidence_dir=w.dir, registered_sha256=hashlib.sha256(
        w.path.read_bytes()).hexdigest(), repo=w.repo, rev=rev or w.head)


def codes(w, seed_dir: Path, rev=None) -> tuple[str, str]:
    with pytest.raises(VE.EccRefused) as a:
        train_side(w, seed_dir)
    with pytest.raises(E.EarlyCheckRejected) as b:
        score_side(w, rev)
    return a.value.code, b.value.code


# ---------------------------------------------------------------- S1 · S3
def test_substitute_accepts_matching_world(tmp_path):
    w = make_world(tmp_path)
    t = train_side(w, tmp_path / "seed")
    s = score_side(w)
    assert t["approval_sha256"] == s["approval_sha256"] == hashlib.sha256(w.raw).hexdigest()
    assert s["status"] == FRAME_CHECKED_BY_A3_SUBSTITUTE and s["a3_commit"] == w.a3 == t["summary"]["a3_commit"]
    assert t["summary"]["evidence"] == s["evidence"] and [e["kind"] for e in s["evidence"]] == sorted(S.EVIDENCE_KINDS)
    assert s["batch_size"] == 8                                             # S3 — 배치는 동일성 밖(A3 커밋은 1)
    assert s["scope"]["does_not_substitute"] and s["scope"]["substitutes"]   # C2 — 범위가 산출에 실린다
    assert (tmp_path / "seed" / S.KEY_FILE).is_file()


# ---------------------------------------------------------------- S2 — 동일성
@pytest.mark.parametrize("rel", ["vlm/coords.py", "vlm/gen_parse.py", "vlm/pilot_vlm.py", "vlm/narrative.py", S.PROMPT_REL])
def test_substitute_identity_drift_refused_both_sides(tmp_path, rel):
    w = make_world(tmp_path)
    p = w.repo / rel
    p.write_bytes(p.read_bytes() + b"# drift\n")
    head = commit(w.repo, "바뀜")
    want = "substitute_regime" if rel == S.PROMPT_REL else "substitute_identity"   # 프롬프트는 등록값(프롬프트 sha)부터 다르다
    assert codes(w, tmp_path / "seed", head) == (want, "FRAME_SUBSTITUTE_" + want.removeprefix("substitute_").upper())
    assert not (tmp_path / "seed" / S.KEY_FILE).exists()                   # 키 기록 전에 멈춘다


def test_substitute_identity_config_key_refused(tmp_path):
    w = make_world(tmp_path)
    write_files(w.repo, {S.CONFIG_REL: base_yaml(8, coord="ABS_RESIZED")})
    head = commit(w.repo, "좌표 규약 바뀜")
    assert codes(w, tmp_path / "seed", head) == ("substitute_identity", "FRAME_SUBSTITUTE_IDENTITY")


def test_substitute_a3_commit_missing_refused(tmp_path):
    w = make_world(tmp_path, approval_over=lambda d: d.update(a3_commit="c" * 40))
    assert codes(w, tmp_path / "seed") == ("substitute_a3_commit", "FRAME_SUBSTITUTE_A3_COMMIT")


# ---------------------------------------------------------------- S4 — 증거
def test_substitute_evidence_sha_missing_content(tmp_path):
    w = make_world(tmp_path / "a")
    w.paths["a3_gate_form"].write_bytes(w.paths["a3_gate_form"].read_bytes() + b" ")
    assert codes(w, tmp_path / "a" / "seed") == ("substitute_evidence_sha", "FRAME_SUBSTITUTE_EVIDENCE_SHA")
    w = make_world(tmp_path / "b")
    w.paths["batch_equivalence"].unlink()
    assert codes(w, tmp_path / "b" / "seed") == ("substitute_evidence_missing", "FRAME_SUBSTITUTE_EVIDENCE_MISSING")


@pytest.mark.parametrize("kw", [
    {"check_over": {"pass": False}},
    {"check_over": {"execution_commit": "e" * 40}},
    {"check_over": {"a3_values": {"precision": 0.8, "recall": 1.0, "threshold": 0.9, "scope": "train_box.class_aware_iou50"}}},
    {"check_over": {"a3_values": {"precision": 1.0, "recall": 1.0, "threshold": 0.5, "scope": "train_box.class_aware_iou50"}}},
    # 비유한 · 범위 밖 — json.dumps 는 NaN · Infinity 를 그대로 쓰고 json.loads 는 그것을 읽는다
    {"check_over": {"a3_values": {"precision": float("nan"), "recall": 1.0, "threshold": 0.9, "scope": "train_box.class_aware_iou50"}}},
    {"check_over": {"a3_values": {"precision": 1.0, "recall": float("nan"), "threshold": 0.9, "scope": "train_box.class_aware_iou50"}}},
    {"check_over": {"a3_values": {"precision": 1.0, "recall": 1.0, "threshold": float("nan"), "scope": "train_box.class_aware_iou50"}}},
    {"check_over": {"a3_values": {"precision": float("inf"), "recall": 1.0, "threshold": 0.9, "scope": "train_box.class_aware_iou50"}}},
    {"check_over": {"a3_values": {"precision": 1.5, "recall": 1.0, "threshold": 0.9, "scope": "train_box.class_aware_iou50"}}},
    {"check_over": {"a3_values": {"precision": 1.0, "recall": 1.0, "threshold": 1.5, "scope": "train_box.class_aware_iou50"}}},
    {"result_over": {"run_name": "a3-other"}},
    {"result_over": {"regime": {"recipe": "json_only", "prompt_sha256": "0" * 64, "max_new_tokens": 1280}}},
    {"probe_over": lambda p: p["passes"]["8"]["compare_vs_batch1"].update(tokens_equal_rate=0.99)},
    {"probe_over": lambda p: p["passes"]["8"].update(status="oom")},
    {"probe_over": lambda p: p["passes"].pop("8")},
    {"n_rows": 8},
])
def test_substitute_evidence_content_refused(tmp_path, kw):
    w = make_world(tmp_path, **kw)                       # 승인서는 바뀐 증거의 sha 를 싣는다 — sha 는 맞고 내용이 어긋난다
    assert codes(w, tmp_path / "seed") == ("substitute_evidence_content", "FRAME_SUBSTITUTE_EVIDENCE_CONTENT")


# ---------------------------------------------------------------- S5 — 꼴 · 결정
@pytest.mark.parametrize(("over", "code"), [
    (lambda d: d.update(extra=1), "substitute_form"),
    (lambda d: d.pop("decision"), "substitute_form"),
    (lambda d: d.update(decided_at="2026-10-31T09:00:00"), "substitute_form"),
    (lambda d: d.update(kind="ecc_waiver"), "substitute_form"),
    (lambda d: d["evidence"].pop(), "substitute_form"),
    (lambda d: d.update(decision_log_section="## 없는 절"), "substitute_decision"),
    (lambda d: d.update(recipe="json_only"), "substitute_recipe"),
    (lambda d: d.update(prompt_sha256="0" * 64), "substitute_regime"),
    (lambda d: d.update(max_new_tokens=512), "substitute_regime"),
    (lambda d: d["identity"]["files"].update({"vlm/coords.py": "0" * 40}), "substitute_identity"),
])
def test_substitute_form_and_decision(tmp_path, over, code):
    w = make_world(tmp_path, approval_over=over)
    assert codes(w, tmp_path / "seed") == (code, "FRAME_SUBSTITUTE_" + code.removeprefix("substitute_").upper())


def test_substitute_recipe_of_the_checkout_must_be_regime2(tmp_path):
    w = make_world(tmp_path)
    write_files(w.repo, {S.TRAIN_REG_REL: "recipe: json_only\n"})
    head = commit(w.repo, "1 차 레시피")
    assert codes(w, tmp_path / "seed", head) == ("substitute_recipe", "FRAME_SUBSTITUTE_RECIPE")


# ---------------------------------------------------------------- S6 — 키 기록 · 충돌
def test_substitute_key_record_binds_pin(tmp_path):
    w = make_world(tmp_path)
    sd = tmp_path / "seed"
    first = train_side(w, sd)
    assert train_side(w, sd) == first                                     # 재개 — 같은 값이면 받는다
    with pytest.raises(VE.EccRefused) as exc:
        train_side(w, sd, pin="q" * 64)
    assert exc.value.code == "substitute_changed"
    doc = json.loads(w.raw)
    doc["decision"] = "같은 결정의 다른 문구"
    w.path.write_bytes(S.dump_approval(doc))                               # 승인서 바이트가 바뀌었다
    with pytest.raises(VE.EccRefused) as exc:
        train_side(w, sd)
    assert exc.value.code == "substitute_changed"


@pytest.mark.parametrize("which", ["permit_file", "permit_key", "waiver_key"])
def test_substitute_conflict(tmp_path, which):
    w = make_world(tmp_path)
    sd = tmp_path / "seed"
    sd.mkdir()
    permit = sd / "ecc_permit.json"
    {"permit_file": permit, "permit_key": sd / VE.PERMIT_KEY_FILE, "waiver_key": sd / VE.WAIVER_KEY_FILE}[which].write_text("{}")
    with pytest.raises(VE.EccRefused) as exc:
        train_side(w, sd, permit=permit)
    assert exc.value.code == "substitute_conflict" and not (sd / S.KEY_FILE).exists()


def test_scoring_hash_must_match_registration(tmp_path):
    w = make_world(tmp_path)
    with pytest.raises(E.EarlyCheckRejected) as exc:
        frame_check_by_a3_substitute(w.raw, evidence_dir=w.dir, registered_sha256="0" * 64, repo=w.repo, rev=w.head)
    assert exc.value.code == "FRAME_SUBSTITUTE_HASH"


# ---------------------------------------------------------------- 시작 문 — 2 차 레시피의 면제 자리
def test_is_a3_substitute_file(tmp_path):
    w = make_world(tmp_path)
    assert VE.is_a3_substitute_file(w.path)
    old = tmp_path / "old_waiver.json"
    old.write_text(json.dumps({k: "x" for k in VE.WAIVER_FIELDS}), encoding="utf-8")
    assert not VE.is_a3_substitute_file(old) and not VE.is_a3_substitute_file(tmp_path / "없음.json")


def test_real_identity_files_exist_and_a3_commit_matches_head_where_present():
    """실제 저장소 — 동일성 목록의 파일이 있고, A3 실행 커밋이 이 체크아웃에 있으면 그 커밋과 HEAD 의 동일성이 같다.

    10-31 누적 32 → 8(7172522d)로 `vlm/pilot_vlm.py` 가 바뀌어 앞 A3(0656cdb6)의 승인서는 무효가 됐다 — A3 재시험(r3)은 이 변경 뒤의 코드에서
    돈다. 이 시험은 그 코드와 HEAD 의 동일성을 고정한다(동일성 파일을 바꾸면 A3 를 다시 돌려야 한다는 경보).
    """
    for p in S.IDENTITY_FILES:
        assert (ROOT / p).is_file(), p
    a3 = "7172522d09bb82fd4180a2a94c3976edab0f5d1f"
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{a3}^{{commit}}"], capture_output=True,
                      check=False).returncode != 0:
        pytest.skip("A3 실행 커밋이 이 체크아웃에 없다")
    tree = {p: subprocess.run(["git", "-C", str(ROOT), "hash-object", "--path", p, str(ROOT / p)], capture_output=True, text=True,
                              check=True).stdout.strip() for p in S.IDENTITY_FILES}   # 작업 트리(게이트의 병합 바이트) — 줄 끝 필터를 거친 blob
    assert S.identity_at(ROOT, a3)["files"] == tree
    assert S.identity_at(ROOT, a3)["config_sha256"] == S.identity_at(ROOT, "HEAD")["config_sha256"]
