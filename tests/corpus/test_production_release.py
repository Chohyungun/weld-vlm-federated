"""학습 릴리스(release_run · verify_release) 계약 — 가짜 입력만 쓴다(실데이터 · GPU 없음).

정상 경로는 prepare_run → infer.run_stage(가짜 백엔드) → 참조용 AUDIT(보고 05 공유 계약 안) → release_run → verify_release →
학습기의 행 검사까지 지난다. 거부 시험은 한 번에 한 항목만 깨뜨린다.
"""
from __future__ import annotations

import json
import shutil

import pytest

# 은행 진입점이 없으면 건너뛰지 않고 수집에서 실패한다 — 진입점 부재는 계약 위반이다(B 05f I-1)
from corpus.parse.passage_bank import select_bank_for_generation  # noqa: F401
from corpus.production import common, infer, quality, train
from corpus.production.common import canonical_bytes, write_json_new

# 판정 v6 파서가 없으면 건너뛰지 않고 수집에서 실패한다 — 지금 판정 정책의 출력 계약이 v6 다(B 05l)
from corpus.production.quality import parse_judge_v6  # noqa: F401
from corpus.production.release import (
    JUDGE_SCHEMAS,
    JUDGE_SOURCES,
    STRUCTURED_PARSERS,
    ReleaseRefused,
    _structured_parser,
    release_run,
    verify_release,
)
from tests.corpus.production_fixtures import (
    FakeBackend,
    World,
    build_audit,
    make_judge_source,
    review_for,
    rights_for,
    sha,
)


def _stage(world, run_dir, backend="transformers"):
    FakeBackend.backend = backend
    try:
        for stage in ("generate", "judge"):
            assert infer.run_stage(run_dir, stage, world.config, backend_factory=FakeBackend)["exit_code"] == 0
    finally:
        FakeBackend.backend = "transformers"


class Case:
    """한 실행 · 한 AUDIT · 검수 · 이용 확인 파일. 시험마다 새로 만든다."""

    def __init__(self, tmp_path, *, backend="transformers", mutate_rows=None, prepare_over=None, separate=False, **quality):
        self.world = World(tmp_path, **quality)
        self.world.prepare(**(prepare_over or {}))
        self.run_dir = tmp_path / "run"
        _stage(self.world, self.run_dir, backend)
        self.audit_dir = tmp_path / "audit"
        self.judge_source = make_judge_source(self.run_dir, self.world.config, tmp_path / "judge-new") if separate else None
        # 기본은 separate 만 — 실행 안의 구판정 경로(run_config)는 허용 출처를 명시한다
        self.release_kw: dict = ({"judge_source": self.judge_source,
                                  "expected_judge_identity_sha256": self.judge_source["identity_sha256"]} if separate
                                 else {"allowed_judge_sources": JUDGE_SOURCES})
        self.audit = build_audit(self.run_dir, self.world.config, self.audit_dir, mutate_rows=mutate_rows,
                                 judge_source=self.judge_source)
        self.review = review_for(self.audit, self.audit_dir)
        self.rights = rights_for(self.audit)
        self.out = tmp_path / "release"

    def write(self):
        for name, value in (("AUDIT.json", self.audit), ("REVIEW.json", self.review), ("RIGHTS.json", self.rights)):
            path = (self.audit_dir if name == "AUDIT.json" else self.world.tmp) / name
            path.unlink(missing_ok=True)
            write_json_new(path, value)

    def release(self):
        self.write()
        return release_run(self.audit_dir, self.world.tmp / "REVIEW.json", self.world.tmp / "RIGHTS.json", self.out,
                           **self.release_kw)

    def refused(self, code):
        with pytest.raises(ReleaseRefused) as err:
            self.release()
        assert err.value.code == code, err.value
        assert not self.out.exists() and not list(self.out.parent.glob(".release.*.tmp"))


@pytest.fixture
def case(tmp_path):
    return Case(tmp_path)


def test_정상_경로(case):
    result = case.release()
    receipt = json.loads((case.out / "RELEASE.json").read_bytes())
    assert receipt["ready_for_training"] is True and receipt["schema_version"] == 1
    assert receipt["dataset_sha256"] == sha((case.out / "dataset.jsonl").read_bytes()) == result["dataset_sha256"]
    assert receipt["run_identity_sha256"] == json.loads((case.run_dir / "RUN.json").read_bytes())["identity_sha256"]
    assert receipt["counts"]["image_report"] == {"train": 4, "val": 2} and receipt["counts"]["source_qa"]["train"] == 12
    assert set(receipt["evidence"]) == {"audit", "review", "rights"}
    assert verify_release(case.out)["verified"] is True
    assert common.validate_release(case.out)["dataset_sha256"] == receipt["dataset_sha256"]
    rows = [json.loads(x) for x in (case.out / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        train._validate_row(row)                              # 학습기가 받는 꼴
    picked = train.select_rows(rows, "central", image_root=case.world.repo)
    assert len(picked) == 4 + 12                              # val 은 학습에서 빠진다


def test_시험용_backend_행은_거부(tmp_path):
    # 생성 · 판정 샤드가 fake 인데 데이터셋 행이 transformers 라고 적어도 샤드를 다시 읽어 거부한다
    def lie(rows):
        for r in rows:
            for side in ("generate", "judge"):
                r["provenance"][side]["backend"] = "transformers"
        return rows

    Case(tmp_path, backend="fake", mutate_rows=lie).refused("stage_backend")


def test_행의_backend_가_transformers_가_아니면_거부(tmp_path):
    def fake(rows):
        rows[0]["provenance"]["judge"]["backend"] = "fake"
        return rows

    Case(tmp_path, mutate_rows=fake).refused("row_backend")


@pytest.mark.parametrize("key,value,code", [
    ("dataset_sha256", "0" * 64, "dataset_sha256"),
    ("errors", 1, "audit_errors"),
    ("rejected", 1, "audit_rejected"),
    ("incomplete", 1, "audit_incomplete"),
    ("complete", False, "audit_incomplete"),
    ("planned", {"count": 1, "ids_sha256": "0" * 64}, "planned_sequence"),
    ("run_dir", None, "audit_run_dir"),
    ("user_templates", {"image_report": []}, "row_user_template"),
    ("source_documents", [], "source_documents"),
    ("source_documents", None, "audit_schema"),
])
def test_집계_항목이_어긋나면_거부(case, key, value, code):
    case.audit[key] = value
    case.refused(code)


@pytest.mark.parametrize("cut", [lambda rows: rows[:-1], lambda rows: rows[1:]])
def test_데이터셋에서_한_행이_빠지면_거부(tmp_path, cut):
    # 집계 지문 · 계획 · counts · 층을 잘린 데이터셋에 맞춰 다른 검사를 다 지나게 해도 작업 명부와의 대조에서 거부한다
    Case(tmp_path, mutate_rows=cut, minimum_source_qa_train=8).refused("planned_sequence")


def test_고유_구절_하한_미달은_거부(tmp_path):
    Case(tmp_path, minimum_distinct_train_passages=5).refused("distinct_passages")


def test_원문_QA_하한_미달은_거부(tmp_path):
    Case(tmp_path, minimum_source_qa_train=13).refused("source_qa_minimum")


def test_같은_계열_판정은_거부(tmp_path):
    def same(rows):
        rows[0]["provenance"]["judge"]["family"] = "qwen"
        return rows

    Case(tmp_path, mutate_rows=same).refused("row_family")


def test_판정이_통과가_아니면_거부(tmp_path):
    def fail(rows):
        rows[0]["provenance"]["judge"]["verdict"]["grounded"] = False
        return rows

    Case(tmp_path, mutate_rows=fail).refused("row_judge")


def test_판정이_샤드와_다르면_거부(tmp_path):
    def other(rows):
        rows[0]["provenance"]["judge"]["verdict"]["reason"] = "다른 근거"
        return rows

    Case(tmp_path, mutate_rows=other).refused("judge_mismatch")


def test_생성_prompt_를_user_message_로_복사하면_거부(tmp_path):
    prompt = {}

    def leak(rows):
        tasks = [json.loads(x) for x in (tmp_path / "run" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
        prompt["p"] = tasks[0]["prompt"]
        rows[0]["messages"][0]["content"][1]["text"] = tasks[0]["prompt"]
        return rows

    c = Case(tmp_path, mutate_rows=leak)
    c.audit["user_templates"]["image_report"].append(prompt["p"])
    c.refused("user_template_leak")                           # 틀에 숫자 · 괄호가 있다


def test_작업_명부와_다른_필드는_거부(tmp_path):
    def regroup(rows):
        rows[0]["lineage_group_id"] = "grp_other"
        return rows

    Case(tmp_path, mutate_rows=regroup).refused("task_field")


def test_image_sha256_이_없으면_거부(tmp_path):
    def drop(rows):
        del rows[0]["image_sha256"]
        return rows

    Case(tmp_path, mutate_rows=drop).refused("row_image_sha256")


@pytest.mark.parametrize("mutate,code", [
    (lambda r: r.pop("reviewer"), "review_reviewer"),
    (lambda r: r.update(reviewed_at="2026-10-08T10:00:00"), "timestamp"),
    (lambda r: r.pop("reviewed_at"), "timestamp"),
    (lambda r: r.update(target_dataset_sha256="0" * 64), "review_target"),
    (lambda r: r["samples"].pop(), "review_samples"),
    (lambda r: r["samples"][0].update(id="source_qa:none"), "review_samples"),
    (lambda r: r["samples"][0].update(decision="reject"), "review_rejected"),
    (lambda r: r.update(approved=False), "review_approved"),
])
def test_사람_승인_누락은_거부(case, mutate, code):
    mutate(case.review)
    case.refused(code)


def test_검수_표본이_층마다_최소에_못_미치면_거부(case):
    key = next(iter(case.audit["review"]["strata"]))
    case.audit["review"]["strata"][key] = []
    case.review = review_for(case.audit, case.audit_dir)
    case.refused("review_plan")


@pytest.mark.parametrize("mutate,code", [
    (lambda r: r.pop("confirmed_by"), "rights_by"),
    (lambda r: r.pop("confirmed_at"), "timestamp"),
    (lambda r: r["assets"].pop(0), "rights_missing"),         # 페어 지문의 이용 확인이 없다
    (lambda r: r["assets"][1].update(use=["research"]), "rights_missing"),   # 매니페스트(영상) 용도에 학습이 없다
    (lambda r: r["assets"].pop(), "rights_missing"),          # 원문 문서 하나의 확인이 없다
])
def test_자료_이용_확인_누락은_거부(case, mutate, code):
    mutate(case.rights)
    case.refused(code)


def test_있는_출력에는_쓰지_않는다(case):
    case.out.mkdir()
    with pytest.raises(ReleaseRefused) as err:
        case.release()
    assert err.value.code == "output_exists"


def test_발행_뒤_바뀐_데이터셋은_검증에서_거부(case):
    case.release()
    lines = (case.out / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["messages"][1]["content"][0]["text"] += " 덧붙임"
    lines[0] = canonical_bytes(row).decode("utf-8")
    (case.out / "dataset.jsonl").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    with pytest.raises(ReleaseRefused) as err:
        verify_release(case.out)
    assert err.value.code == "dataset_sha256"


def test_발행_뒤_바뀐_증빙은_검증에서_거부(case):
    case.release()
    with (case.out / "REVIEW.json").open("ab") as fh:
        fh.write(b" ")
    with pytest.raises(ReleaseRefused) as err:
        verify_release(case.out)
    assert err.value.code == "evidence_changed"


def test_영수증만_있는_디렉터리는_통과하지_않는다(case):
    case.release()
    bare = case.world.tmp / "bare"
    bare.mkdir()
    shutil.copyfile(case.out / "dataset.jsonl", bare / "dataset.jsonl")
    receipt = json.loads((case.out / "RELEASE.json").read_bytes())
    receipt["evidence"] = {}
    (bare / "RELEASE.json").write_bytes(canonical_bytes(receipt))
    with pytest.raises(ReleaseRefused) as err:
        verify_release(bare)
    assert err.value.code == "evidence_changed"
    with pytest.raises(ReleaseRefused):
        common.validate_release(bare)


def test_영수증의_승인_기록을_고치면_거부(case):
    case.release()
    receipt = json.loads((case.out / "RELEASE.json").read_bytes())
    receipt["reviewer"] = "다른 사람"
    (case.out / "RELEASE.json").write_bytes(canonical_bytes(receipt))
    with pytest.raises(ReleaseRefused) as err:
        verify_release(case.out)
    assert err.value.code == "release_evidence"


def test_user_message_가_작업의_학습_질문이_아니면_거부(tmp_path):
    other = "이 영상을 한국어로 자세히 설명하라."

    def swap(rows):
        rows[0]["messages"][0]["content"][1]["text"] = other
        return rows

    c = Case(tmp_path, mutate_rows=swap)
    c.audit["user_templates"]["image_report"].append(other)     # 틀 목록에 있어도 작업 행이 정본이다
    c.refused("user_text")


@pytest.mark.parametrize("mutate,code", [
    (lambda s: s.update(row_sha256="0" * 64), "review_row"),
    (lambda s: s.update(reviewer=" "), "review_reviewer"),
    (lambda s: s.update(reviewed_at="2026-10-08T10:00:00"), "timestamp"),
])
def test_검수_표본의_행_지문과_표본별_기록(case, mutate, code):
    mutate(case.review["samples"][0])
    case.refused(code)


def test_표본별_검수자_시각을_채워도_통과한다(case):
    for s in case.review["samples"]:
        s.update(reviewer="검수자 병", reviewed_at="2026-10-08T12:00:00+09:00")
    assert case.release()["exit_code"] == 0


def _swap_text(kind, which, new):
    def mutate(rows):
        r = next(x for x in rows if x["kind"] == kind)
        msg = r["messages"][0 if which == "user" else 1]
        msg["content"][-1]["text"] = new(msg["content"][-1]["text"])
        return rows
    return mutate


@pytest.mark.parametrize("kind,which,new,code", [
    ("source_qa", "user", lambda s: s + " 다른 질문", "message_mismatch"),
    ("source_qa", "assistant", lambda s: "판정받지 않은 답이다", "message_mismatch"),
    ("image_report", "assistant", lambda s: s + " 원인은 수소 혼입이다.", "message_mismatch"),
])
def test_학습_메시지는_판정받은_생성문의_정규_렌더와_정확히_같아야_한다(tmp_path, kind, which, new, code):
    Case(tmp_path, mutate_rows=_swap_text(kind, which, new)).refused(code)


def test_원문_QA_의_근거_id_를_바꾸면_거부(tmp_path):
    def swap(rows):
        r = next(x for x in rows if x["kind"] == "source_qa")
        r["source_ids"] = ["DOC9-P99"]                          # 작업 명부에 없는 근거 id
        return rows
    Case(tmp_path, mutate_rows=swap).refused("task_field")


@pytest.mark.parametrize("key,value", [("status", "blocked"), ("blockers", ["errors"])])
def test_집계가_사람_검수_대기가_아니면_거부(case, key, value):
    case.audit[key] = value
    case.refused("audit_blocked")


@pytest.mark.parametrize("ids", [[1], [None], [""], [["DOC1-P01"]]])
def test_렌더러는_근거_id_원소_꼴이_틀리면_거부한다(ids):
    from corpus.production.release import render_messages

    task = {"id": "source_qa:DOC1-P01:q1", "kind": "source_qa", "source_ids": ["DOC1-P01"]}
    text = json.dumps({"question": "질문", "answer": "답", "source_ids": ids}, ensure_ascii=False)
    with pytest.raises(ReleaseRefused) as err:
        render_messages(task, text)
    assert err.value.code == "generation_schema"


def test_영상_설명만이면_원문_QA_하한_0_에서_릴리스된다(tmp_path):
    """사용자 결정(10-08): 이번 실행은 영상 설명만 — 원문 QA 하한 0 · 0 ≥ 0 이라 하한 검사가 막지 않는다."""
    c = Case(tmp_path, prepare_over={"source_rights_path": None}, minimum_source_qa_train=0, target_source_qa_train=0,
             minimum_distinct_train_passages=0)
    assert c.audit["source_documents"] == []
    basis = c.audit["run_identity_basis"]
    c.rights["assets"] = [a for a in c.rights["assets"] if a["sha256"] != basis["source_bank_sha256"]]   # 쓰지 않은 은행은 확인 대상이 아니다
    result = c.release()
    assert result["counts"]["source_qa"] == {"train": 0, "val": 0} and result["counts"]["image_report"] == {"train": 4, "val": 2}
    assert verify_release(c.out)["verified"] is True


# --------------------------------------------------------------------------- 판정 신원(B 05h)


def test_따로_둔_판정은_폴더_다시_읽기까지_지나고_시험용_판정이라_릴리스되지_않는다(tmp_path):
    """판정 신원 쓰기는 주입한 백엔드로 fake 만 받는다 — 신원 · 핀 · 폴더 다시 읽기 · 행 대조를 모두 지난 뒤 판정 행의 백엔드에서 멈춘다."""
    Case(tmp_path, separate=True).refused("stage_backend")


def test_실행_설정_판정은_신원_null_로_영수증에(case):
    case.release()
    assert json.loads((case.out / "RELEASE.json").read_bytes())["judge_identity"] == {"source": "run_config", "identity_sha256": None,
                                                                                    "caller_expected_pins": None}


@pytest.mark.parametrize("mutate", [
    lambda a: a["judge_identity"]["identity_basis"].update(judge_prompt_sha256="e" * 64),     # 근거 지문이 아니다
    lambda a: a["judge_identity"].update(identity_basis={**a["judge_identity"]["identity_basis"], "run_identity_sha256": "0" * 64}),
    lambda a: a["judge_identity"].update(source="other"),
])
def test_집계의_판정_신원이_어긋나면_거부(tmp_path, mutate):
    c = Case(tmp_path, separate=True)
    mutate(c.audit)
    c.refused("audit_judge_identity")


def test_행의_판정_신원이_집계와_다르면_거부(tmp_path):
    def other(rows):
        rows[0]["provenance"]["judge"]["identity_sha256"] = "f" * 64
        return rows
    Case(tmp_path, separate=True, mutate_rows=other).refused("row_judge_identity")


@pytest.mark.parametrize("mutate", [
    lambda js: js["rows"].__setitem__(0, {**js["rows"][0], "_judge_identity_sha256": "f" * 64}),     # 다른 신원의 행
    lambda js: js["rows"].pop(),                                                                       # 모자란 행
    lambda js: js["rows"].__setitem__(0, {**js["rows"][0], "text": js["rows"][0]["text"].replace("true", "false", 1)}),
    lambda js: js["rows"].__setitem__(0, {**js["rows"][0], "_shard_sha256": "0" * 64}),
    lambda js: js["rows"].__setitem__(0, {**js["rows"][0], "_backend": "transformers"}),           # fake 행을 운영 백엔드라 적음
])
def test_호출자의_판정_객체를_고쳐도_판정_폴더를_다시_읽어_거부(tmp_path, mutate):
    c = Case(tmp_path, separate=True)
    mutate(c.judge_source)
    c.refused("judge_source_unverified")


@pytest.mark.parametrize("judge_dir", [None, "elsewhere-judge"])
def test_판정_폴더가_없거나_다른_자리면_거부(tmp_path, judge_dir):
    c = Case(tmp_path, separate=True)
    if judge_dir is None:
        del c.judge_source["judge_dir"]
    else:
        c.judge_source["judge_dir"] = str(tmp_path / judge_dir)
    c.refused("judge_source_unverified")


def test_따로_둔_판정이_주어지지_않으면_거부(tmp_path):
    c = Case(tmp_path, separate=True)
    c.release_kw = {"expected_judge_identity_sha256": c.judge_source["identity_sha256"]}
    c.refused("judge_source_missing")


def test_따로_둔_판정에_외부_핀이_없으면_거부(tmp_path):
    c = Case(tmp_path, separate=True)
    c.release_kw = {"judge_source": c.judge_source}
    c.refused("judge_source_pin_required")


def test_외부_핀이_판정_신원과_다르면_거부(tmp_path):
    c = Case(tmp_path, separate=True)
    c.release_kw = {"judge_source": c.judge_source, "expected_judge_identity_sha256": "0" * 64}
    c.refused("judge_source_unverified")


def test_실행_설정_판정인데_따로_둔_판정을_섞으면_거부(case, tmp_path):
    case.release_kw = {"judge_source": make_judge_source(case.run_dir, case.world.config, tmp_path / "judge-new"),
                       "allowed_judge_sources": JUDGE_SOURCES}
    case.refused("judge_identity_mixed")


def test_기본으로는_따로_둔_판정만_받는다(case):
    case.release_kw = {}                                        # 기본값 — 실행 안의 구판정은 명시 인자로만(결정 10-12)
    case.refused("judge_source_not_allowed")


def test_영수증의_판정_신원을_고치면_검증에서_거부(tmp_path):
    c = Case(tmp_path)                  # 통과 릴리스는 실행 안의 구판정으로만 만들 수 있다(시험용 따로 둔 판정은 릴리스되지 않는다)
    c.release()
    receipt = json.loads((c.out / "RELEASE.json").read_bytes())
    receipt["judge_identity"]["identity_sha256"] = "0" * 64
    (c.out / "RELEASE.json").write_bytes(canonical_bytes(receipt))
    with pytest.raises(ReleaseRefused) as err:
        verify_release(c.out)
    assert err.value.code == "release_identity"


def test_릴리스_핀이_집계가_읽은_핀과_다르면_거부(tmp_path):
    c = Case(tmp_path, separate=True)
    c.audit["judge_identity"]["caller_expected_pins"] = {"expected_judge_identity_sha256": None,
                                                         "expected_probe_approval_sha256": "a" * 64}
    c.refused("judge_source_unverified")


def test_판정_객체에_칸을_덧붙이면_거부(tmp_path):
    c = Case(tmp_path, separate=True)
    c.judge_source["gpu_model_verified"] = True            # 다시 읽은 값(fake 라 거짓)과 다르다
    c.refused("judge_source_unverified")



# --------------------------------------------------------------------------- 구조화 판정 대조 갈래(v6 · v5 보존 — B 05k I-1 · 05l R-9 ~ R-11)


@pytest.fixture
def operational_judge(monkeypatch):
    """판정 신원 읽기를 감싸 돌려받은 판정 행의 `_backend` 를 transformers 로 적는다 — 운영 판정이 있었다고 치는 시험 안의 흉내.
    시험 세계의 판정 신원 쓰기는 fake 만 받아 통과 릴리스가 없으므로(17절), 구조화 대조 갈래에 닿게 하려고만 쓴다."""
    real = infer.read_judge_source

    def wrapped(*args, **kwargs):
        out = real(*args, **kwargs)
        return {**out, "rows": [{**r, "_backend": "transformers"} for r in out["rows"]]}

    monkeypatch.setattr(infer, "read_judge_source", wrapped)


def test_구조화_꼴과_파서의_짝은_B_의_판_표와_같고_v5_를_보존한다():
    assert set(STRUCTURED_PARSERS) == set(quality.JUDGE_PARSER_VERSIONS)
    assert JUDGE_SCHEMAS == ("legacy_four_keys", *STRUCTURED_PARSERS)
    probe = {"text": "{}", "finish_reason": "eos"}
    for jschema, name in STRUCTURED_PARSERS.items():
        parse, version = _structured_parser(jschema)
        assert parse is getattr(quality, name) and version == quality.JUDGE_PARSER_VERSIONS[jschema]
        out = parse({"kind": "source_qa", "locked_target": None}, probe)       # 판독 실패여도 꼴 · 판은 싣는다
        assert (out["schema"], out["parser"]) == (jschema, version)


@pytest.mark.parametrize("break_it", [
    lambda m: m.delattr(quality, "parse_judge_v6"),
    lambda m: m.setattr(quality, "JUDGE_PARSER_VERSIONS", {"judge_structured_v5": "parse_judge_v5_1"}),
])
def test_구조화_파서나_그_판이_없으면_거부(monkeypatch, break_it):
    break_it(monkeypatch)
    with pytest.raises(ReleaseRefused) as err:
        _structured_parser("judge_structured_v6")
    assert err.value.code == "judge_parser"


def test_구조화_정직한_따로_둔_판정은_릴리스_검증까지_간다(tmp_path, operational_judge):
    c = Case(tmp_path, separate=True)
    c.release()
    receipt = json.loads((c.out / "RELEASE.json").read_bytes())
    assert receipt["judge_identity"]["source"] == "separate"
    assert receipt["judge_identity"]["caller_expected_pins"] == c.audit["judge_identity"]["caller_expected_pins"]
    assert verify_release(c.out)["verified"] is True
    assert (c.audit["judge_identity"]["parser_schema"], c.audit["judge_parser_version"]) == (
        quality.JUDGE_V6_SCHEMA_VERSION, quality.JUDGE_V6_PARSER_VERSION)
    rows = [json.loads(x) for x in (c.out / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    judged = [r["provenance"]["judge"] for r in rows]
    assert {(j["schema"], j["parser"]) for j in judged} == {(quality.JUDGE_V6_SCHEMA_VERSION, quality.JUDGE_V6_PARSER_VERSION)}
    assert all(j["format_normalized"] == [] for j in judged)
    # 결함 행은 v6 일곱 키로 답했고, 그 답이 그대로 행 출처에 실렸다
    defect = [j["answers"] for r, j in zip(rows, judged) if r["kind"] == "image_report" and "names_match" in j["answers"]]
    assert defect and all(tuple(sorted(a)) == tuple(sorted(quality.JUDGE_V6_KEYS["defect"])) for a in defect)


@pytest.mark.parametrize("mutate", [
    lambda j: j["answers"].update(reason="다른 이유"),            # 원 답
    lambda j: j.update(flags=["n5_absent"]),                      # 표지
    lambda j: j.update(schema="judge_structured_v5"),             # 꼴
    lambda j: j.update(parser="parse_judge_v5_1"),                # 파서 판(R-9)
    lambda j: j.update(format_normalized=["code_fence"]),        # 형식 정규화(R-9)
    lambda j: j.pop("format_normalized"),                         # 칸이 빠짐
])
def test_구조화_행의_판정_칸을_고치면_거부(tmp_path, operational_judge, mutate):
    def change(rows):
        mutate(next(r for r in rows if r["kind"] == "image_report")["provenance"]["judge"])
        return rows
    Case(tmp_path, separate=True, mutate_rows=change).refused("judge_mismatch")


@pytest.mark.parametrize("mutate", [
    lambda a: a["judge_identity"].update(parser_schema="legacy_four_keys"),       # separate 인데 옛 꼴
    lambda a: a["judge_identity"].update(parser_schema="judge_structured_v5"),    # 정책은 v6 — 다른 꼴로 읽기
    lambda a: a["judge_identity"].update(caller_expected_pins=None),             # 핀 없음
    lambda a: a.update(judge_parser_version="parse_judge_v5_1"),                 # 다른 파서 판(06f A-1)
    lambda a: a.update(judge_parser_version=None),
    lambda a: a.pop("judge_parser_version"),
])
def test_구조화_집계의_꼴_판_핀이_어긋나면_거부(tmp_path, operational_judge, mutate):
    c = Case(tmp_path, separate=True)
    mutate(c.audit)
    c.refused("audit_judge_identity")


@pytest.mark.parametrize("key,value", [("gpu_model_verified", True), ("pin_provenance", "unapproved-fake"),
                                       ("probe_approval_sha256", "a" * 64)])
def test_구조화_집계의_탐침_칸이_다시_읽은_값과_다르면_거부(tmp_path, operational_judge, key, value):
    c = Case(tmp_path, separate=True)
    c.audit["judge_identity"][key] = value
    c.refused("judge_source_unverified")


@pytest.mark.parametrize("pin", ["expected_probe_approval_sha256", "expected_judge_identity_sha256"])
def test_구조화_영수증의_핀을_고치면_검증에서_거부(tmp_path, operational_judge, pin):
    c = Case(tmp_path, separate=True)
    c.release()
    receipt = json.loads((c.out / "RELEASE.json").read_bytes())
    receipt["judge_identity"]["caller_expected_pins"][pin] = "b" * 64
    (c.out / "RELEASE.json").write_bytes(canonical_bytes(receipt))
    with pytest.raises(ReleaseRefused) as err:
        verify_release(c.out)
    assert err.value.code == "release_identity"


@pytest.mark.parametrize("jschema", ["judge_structured_v5", "judge_structured_v6"])
def test_실행_안의_구판정이_구조화_꼴이라_적히면_거부(case, jschema):
    case.audit["judge_identity"]["parser_schema"] = jschema
    case.refused("audit_judge_identity")


def test_실행_안의_구판정_집계에_파서_판이_있으면_거부(case):
    case.audit["judge_parser_version"] = quality.JUDGE_V6_PARSER_VERSION
    case.refused("audit_judge_identity")
