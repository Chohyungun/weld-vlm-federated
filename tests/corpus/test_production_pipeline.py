"""작업 명부 → 생성 · 판정 → 품질 집계 → 릴리스를 한 번에 잇는 시험 — 가짜 입력만 쓴다(실데이터 · GPU 없음).

소유: 작업 명부 · 릴리스(`corpus.production.prepare` · `release`) 쪽과 품질 집계 · 구절 은행(`corpus.production.quality` ·
`corpus.parse.passage_bank`) 쪽의 공동 파일이다. 한쪽 계약을 바꾸면 이 파일을 함께 고친다 — 한쪽만 고치지 않는다.

흐름: `prepare_run`(구절은 `passage_bank.select_bank_for_generation` 한 곳에서만 거른다) → `infer.run_stage`(가짜 백엔드) →
`quality.audit_run` → 사람 검수 양식 채우기 → `release_run` → `verify_release`. 거부 시험은 한 번에 한 항목만 깨뜨리고,
어느 단계가 어떤 코드로 멈추는지까지 본다.

구조화 판정(v6) 경로: 생성 → 따로 둔 판정(`infer.run_judge` — 주입한 백엔드는 시험용 `fake` 만 받는다) → 판정 신원 읽기
(`infer.read_judge_source`) → `quality.audit_run(judge_source=…)`(행 종류별 구조화 답을 엄격히 읽는다) → 릴리스. 시험용 판정은
집계에서 막히고 릴리스도 멈춘다 — 실제 판정 모델은 승인된 두 탐침을 거쳐야 하므로 이 시험에서는 통과 릴리스까지 가지 않는다.

구조화 판정(v7) 경로: 생성(v7 생성 계약의 글) → 판정 신원 쓰기 v7(`infer.run_judge(policy_version="v7")`, 출력 160 토큰, 정지기를
켠 시험용 판정 모델 — 결함 행 · 원문 QA 만 부르고 정상 행은 부르지 않는다) → 판정 신원 읽기(꼴을 스스로 고른다) →
`quality.audit_run(judge_source=…)`(정상 행은 결정적 영수증으로, 결함 행은 결정적 통과와 물음 넷으로). 판정 정책 · 파서 · 쓰기 ·
읽기 · 집계는 실제 것이고 가짜는 모델 백엔드뿐이다.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import ClassVar

import pytest
import yaml

prepare = pytest.importorskip("corpus.production.prepare", reason="작업 명부 모듈이 이 트리에 아직 없다")
release = pytest.importorskip("corpus.production.release", reason="릴리스 모듈이 이 트리에 아직 없다")
fixtures = pytest.importorskip("tests.corpus.production_fixtures", reason="작업 명부 시험 세계가 이 트리에 아직 없다")

from corpus.parse.passage_bank import select_bank_for_generation, text_sha256
from corpus.production import generation_policy, infer, quality
from corpus.production.common import canonical_bytes, write_json_new
from corpus.production.quality import audit_run

#: 구절 은행 — (passage_id, doc_id, 계보, 본문, 덧붙일 칸). 은행 행의 rights_status 는 모두 pending 이다(실제 은행과 같다).
#: 넘어갈 것 넷(같은 계보 안의 필수 문맥 하나 · 역사 자료 하나 포함), 빠질 것 셋(검수 기록 없음 · 그것을 문맥으로 삼음 · 보류).
BANK = [
    ("DOCA-PA", "DOC-A", "LG-A", "Porosity is a cavity formed by trapped gas.", {}),
    ("DOCA-PB", "DOC-A", "LG-A", "Lack of fusion is incomplete bonding between weld and base metal.",
     {"required_context_passage_ids": ["DOCA-PA"], "generation_constraints": ["Do not generalise beyond the cited joint type"]}),
    ("DOCB-PA", "DOC-B", "LG-B", "Slag inclusion is trapped non-metallic solid in the weld.", {}),
    ("HIST-PA", "HIST-A", "LG-H", "Blowhole.—A hole formed by trapped gases.", {"flags": ["historical_only"]}),
    ("DOCC-PA", "DOC-C", "LG-C", "Undercut is a groove melted into the base metal.", {}),
    ("DOCC-PB", "DOC-C", "LG-C", "Overlap is weld metal lying on the surface without fusion.",
     {"required_context_passage_ids": ["DOCC-PA"]}),
    ("DOCD-PA", "DOC-D", "LG-D", "A paid standard sets a limit here.", {"source_policy_holds": ["paid_standard_numeric"]}),
]
SELECTED = ["DOCA-PA", "DOCA-PB", "DOCB-PA", "HIST-PA"]
NO_REVIEW = {"DOCC-PA"}
#: 가짜 생성문이 구절마다 다른 문장을 쓰게 하는 한국어 낱말(숫자를 쓰면 추적되지 않는 수가 된다)
WORD = {"DOCA-PA": "기공", "DOCA-PB": "융합 불량", "DOCB-PA": "슬래그 혼입", "HIST-PA": "블로홀"}
#: 관점 지시 안의 낱말 → 짧은 관점 이름(지시문을 그대로 옮기면 지시 누출로 기각된다)
VIEW = {"정의": "정의", "조건": "조건", "적용 범위": "범위"}


def _doc_sha(doc_id: str) -> str:
    return fixtures.sha(f"source {doc_id}".encode())


def _bank_rows() -> list[dict]:
    rows = []
    for pid, doc, lineage, text, extra in BANK:
        row = {"schema_version": 2, "passage_id": pid, "doc_id": doc, "edition": "test edition",
               "source_file_sha256": _doc_sha(doc), "locator": {"pdf_page_start": 3, "pdf_page_end": 3},
               "text_kind": "verbatim_extract", "text": text, "text_sha256": text_sha256(text),
               "facets": [], "defect_classes": [], "flags": [], "lineage_group_id": lineage,
               "scope": f"{doc} 의 서술 범위", "rights_status": "pending"}
        row.update(extra)
        rows.append(row)
    return rows


class Bank:
    """은행 네 파일(포인터 · 본문 · 빌드 보고 · 등록부)과 사용자 권리 결정 · 사람 검수 기록."""

    def __init__(self, world):
        rows = _bank_rows()
        world.write_bank(rows)                       # 포인터 · 본문 · 빌드 보고(포인터 옆) · config 의 포인터 지문
        tmp = world.tmp
        self.pointer, self.full = world.bank, world.bank_text
        self.registry = tmp / "sources.yaml"
        self.registry.write_text(yaml.safe_dump(
            {"documents": [{"doc_id": d, "sha256": _doc_sha(d)} for d in sorted({r["doc_id"] for r in rows})]}),
            encoding="utf-8")
        self.rights = tmp / "bank_rights.jsonl"
        self.rights.write_bytes(b"".join(canonical_bytes(
            {"doc_id": d, "source_file_sha256": _doc_sha(d), "rights_status": "approved", "approved_uses": ["qa_generation"],
             "decided_by": "결정자", "decided_at": "2026-10-08T09:00:00+09:00"}) + b"\n" for d in sorted({r["doc_id"] for r in rows})))
        self.review = tmp / "bank_review.jsonl"
        self.review.write_bytes(b"".join(canonical_bytes(
            {"passage_id": r["passage_id"], "text_sha256": r["text_sha256"], "decision": "approved", "reviewer": "검수자",
             "reviewed_at": "2026-10-08T09:30:00+09:00", "flags_checked": list(r["flags"])}) + b"\n"
            for r in rows if r["passage_id"] not in NO_REVIEW))

    def kwargs(self) -> dict:
        return {"source_text_path": self.full, "source_registry_path": self.registry,
                "source_rights_path": self.rights, "source_review_path": self.review}

    def select(self):
        return select_bank_for_generation(self.pointer, self.full, registry=self.registry,
                                          rights_path=self.rights, review_path=self.review, use="qa_generation")


def _judge_answer(prompt: str, ok: bool) -> dict:
    """판정 입력에 행 종류(`row_class`)가 있으면 구조화 판정의 물음에 답하고, 없으면 옛 네 키로 답한다."""
    reason = "근거와 일치" if ok else "근거에 없는 내용"
    row_class = json.loads(prompt).get("row_class")
    if row_class == "defect_free":
        return {"claims_defect": not ok, "asserts_verdict": False, "states_no_marked_defect": ok,
                "has_limitation": True, "language_ok": True, "reason": reason}
    if row_class == "defect":
        return {"names_match": ok, "counts_match": ok, "positions_match": ok, "has_limitation": True,
                "asserts_verdict_dimension_or_cause": False, "language_ok": True, "reason": reason}
    return {"grounded": ok, "scope_ok": True, "language_ok": True, "reason": reason}


class PipelineBackend(fixtures.FakeBackend):
    """설명은 주석마다, 질문 · 답은 구절 · 관점마다 다르게 쓴다. `reject_word` 가 든 판정 입력은 근거 불일치로 판정한다."""

    reject_word: str | None = None

    def generate_batch(self, prompts, system, deadline):
        out = []
        for p in prompts:
            if self.stage == "judge":
                ok = not (self.reject_word and self.reject_word in p)
                text = json.dumps(_judge_answer(p, ok), ensure_ascii=False)
            elif "구절 id:" in p:
                pid = re.search(r"구절 id: (\S+)", p).group(1)
                ask = re.search(r"질문의 관점: ([^\n]+)", p).group(1)
                view = next(v for k, v in VIEW.items() if k in ask)
                text = json.dumps({"question": f"{WORD[pid]}에 대해 {view} 관점에서 구절은 무엇을 말하는가?",
                                   "answer": f"구절은 {WORD[pid]}을 {view} 관점에서 설명한다.", "source_ids": [pid]},
                                  ensure_ascii=False)
            else:
                # 결정적 정책을 지나는 꼴 — 결함 행은 근거의 제한 문장 그대로, 정상 행은 결함 없음 + 합격 · 안전을 뜻하지 않음
                ev = json.loads(p.split("주석: ", 1)[1])
                where = ", ".join(f"{d['name']}({d['where']})" for d in ev["defect_positions"])
                text = (f"영상에 표시된 결함은 {where} 이다. {ev['limitation']}" if where
                        else "표시된 결함이 없다. 결함이 표시되지 않았다는 것이 합격이나 안전을 뜻하지는 않는다.")
            out.append({"text": text, "finish_reason": "eos", "input_tokens": 20, "output_tokens": 15})
        return out


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    PipelineBackend.backend, PipelineBackend.reject_word = "transformers", None


def _world(tmp_path):
    world = fixtures.World(tmp_path, minimum_source_qa_train=9, minimum_distinct_train_passages=3)
    return world, Bank(world)


def _prepare(world, bank, name="run"):
    return prepare.prepare_run(world.pairs_dir / "pairs.jsonl", world.bank, world.config, world.tmp / name,
                               repo_root=world.repo, **bank.kwargs())


def _stages(world, run_dir, *, generate_backend="transformers"):
    for stage in ("generate", "judge"):
        PipelineBackend.backend = generate_backend if stage == "generate" else "transformers"
        assert infer.run_stage(run_dir, stage, world.config, backend_factory=PipelineBackend)["exit_code"] == 0
    PipelineBackend.backend = "transformers"


def _fill_review(audit_dir) -> dict:
    """사람 검수자가 양식을 채운 것의 흉내 — 표본 전부 승인."""
    review = json.loads((audit_dir / "REVIEW_TEMPLATE.json").read_bytes())
    review.update(reviewer="검수자 갑", reviewed_at="2026-10-08T10:00:00+09:00", approved=True)
    for s in review["samples"]:
        s.update(decision="approve", comment="")
    return review


def _release(world, audit_dir, out):
    audit = json.loads((audit_dir / "AUDIT.json").read_bytes())
    write_json_new(world.tmp / "REVIEW.json", _fill_review(audit_dir))
    write_json_new(world.tmp / "RIGHTS.json", fixtures.rights_for(audit))
    # 이 시험의 판정은 실행 안의 구판정(run_config)이다 — 릴리스 기본은 따로 둔 판정만이라 허용 출처를 명시한다(총괄 결정 10-12)
    return release.release_run(audit_dir, world.tmp / "REVIEW.json", world.tmp / "RIGHTS.json", out,
                               allowed_judge_sources=("run_config",))


def _run(tmp_path, world, bank, **stage_kw):
    _prepare(world, bank)
    run_dir = world.tmp / "run"
    _stages(world, run_dir, **stage_kw)
    result = audit_run(run_dir, world.config, world.tmp / "audit")
    return run_dir, result, json.loads((world.tmp / "audit" / "AUDIT.json").read_bytes())


def _tasks(run_dir) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- 정상 경로


def test_정상_경로는_은행_경계를_지난_구절만으로_릴리스까지_간다(tmp_path):
    world, bank = _world(tmp_path)
    run_dir, result, audit = _run(tmp_path, world, bank)
    run = json.loads((run_dir / "RUN.json").read_bytes())
    selected, summary = bank.select()

    # 거름은 한 곳이다 — RUN.json 의 수 · 사유가 은행 진입점의 요약 그대로이고, 그 입력 지문이 실행 신원에 들어간다
    assert run["counts"]["source_bank"] == summary
    assert run["identity_basis"]["source_bank_inputs_sha256"] == summary["inputs_sha256"]
    assert [r["passage_id"] for r in selected] == SELECTED
    assert summary["reasons"] == {"context:required_not_selectable": 1, "hold:source_policy:paid_standard_numeric": 1,
                                  "review:none": 1}
    src = [t for t in _tasks(run_dir) if t["kind"] == "source_qa"]
    assert sorted({t["source_ids"][0] for t in src}) == SELECTED and len(src) == 3 * len(SELECTED)
    dependent = next(t for t in src if t["source_ids"] == ["DOCA-PB"])
    assert set(dependent["evidence"]["context_passages"]) == {"DOCA-PA"}

    assert result["status"] == "ready_for_human_review" and result["exit_code"] == 0, audit["blockers"]
    n_image = sum(t["kind"] == "image_report" for t in _tasks(run_dir))
    assert audit["policy"]["rows_checked"] == n_image > 0 and audit["policy"]["reasons"] == {}  # 결정적 정책이 영상 설명에 걸렸다
    out = tmp_path / "release"
    assert _release(world, tmp_path / "audit", out)["exit_code"] == 0
    verified = release.verify_release(out)
    assert verified["verified"] is True and verified["counts"]["source_qa"]["train"] + verified["counts"]["source_qa"]["val"] == 12


def _separate_judge(world, run_dir, name="judge-v6"):
    """따로 둔 판정을 판정 신원 쓰기로 돌리고 판정 신원 읽기로 객체를 받는다. 주입한 백엔드는 시험용(`fake`)만 받는다."""
    target = world.tmp / name
    PipelineBackend.backend = "fake"
    try:
        assert infer.run_judge(run_dir, world.config, target, backend_factory=PipelineBackend)["exit_code"] == 0
    finally:
        PipelineBackend.backend = "transformers"
    return infer.read_judge_source(run_dir, world.config, target)


def test_구조화_판정_경로는_따로_둔_판정으로_집계까지_가고_시험용_판정은_릴리스되지_않는다(tmp_path):
    world, bank = _world(tmp_path)
    _prepare(world, bank)
    run_dir = world.tmp / "run"
    assert infer.run_stage(run_dir, "generate", world.config, backend_factory=PipelineBackend)["exit_code"] == 0
    src = _separate_judge(world, run_dir)
    result = audit_run(run_dir, world.config, world.tmp / "audit", judge_source=src)
    audit = json.loads((world.tmp / "audit" / "AUDIT.json").read_bytes())
    tasks = _tasks(run_dir)

    # 판정 신원 · 출력 계약 — 정책 지문에 묶인 구조화 v6 로 읽었고, 핀은 판정 신원 읽기를 부른 값 그대로다
    ji = audit["judge_identity"]
    assert ji["source"] == "separate" and ji["identity_sha256"] == src["identity_sha256"]
    assert ji["parser_schema"] == "judge_structured_v6" and audit["policy"]["bound_to_judge_identity"] is True
    assert audit["judge_parser_version"] == "parse_judge_v6"
    assert ji["caller_expected_pins"] == src["caller_expected_pins"] and ji["pin_provenance"] == "unapproved-fake"
    # 막는 것은 시험용 판정 백엔드 하나 — 모든 행이 구조화 판정으로 읽혀 통과했다
    assert result["status"] == "blocked" and audit["blockers"] == ["non_transformers_backend"], audit["blockers"]
    assert audit["counts"]["accepted"] == len(tasks) and audit["errors"] == 0 and audit["rejected"] == 0
    n_image = sum(t["kind"] == "image_report" for t in tasks)
    assert audit["policy"]["rows_checked"] == n_image > 0 and audit["policy"]["reasons"] == {}
    assert audit["policy"]["model_agreement"]["denominator"] == n_image == audit["policy"]["model_agreement"]["both_pass"]
    rows = [json.loads(x) for x in (world.tmp / "audit" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    seen = set()
    for row, task in zip(rows, tasks):
        judge = row["provenance"]["judge"]
        assert row["id"] == task["id"] and judge["schema"] == "judge_structured_v6" and judge["backend"] == "fake"
        assert judge["parser"] == "parse_judge_v6" and judge["format_normalized"] == []
        assert {judge["verdict"][k] for k in ("grounded", "scope_ok", "language_ok")} == {True}
        cls = next(c for c, keys in quality.JUDGE_V6_KEYS.items() if set(judge["answers"]) == set(keys))
        assert (cls == "source_qa") == (task["kind"] == "source_qa")
        seen.add(cls)
    assert seen == {"defect_free", "defect", "source_qa"}    # 정상 · 결함 · 원문 QA 세 꼴을 모두 지났다

    # 시험용 판정의 행은 릴리스하지 않는다 — 집계가 막은 그대로 행 검사에서 멈춘다
    _refused(world, tmp_path, "row_backend")


# --------------------------------------------------------------------------- 구조화 판정 v7 — 실제 쓰기 · 읽기 · 정책 · 집계


#: v7 결함 행의 물음 넷("참 = 문제 없음") — 키는 집계의 v7 표에서 읽는다
V7_QUESTIONS = tuple(k for k in quality.JUDGE_V7_KEYS["defect"] if k != "reason")


def _v7_contract_text(ev: dict) -> str:
    """v7 생성 계약의 글 — 정상 행은 정해진 두 문장(생성 정책의 첫 꼴), 결함 행은 (이름, 구역)마다 'NAME k개가 ZONE에 표시되어
    있다.' 뒤에 정해진 제한 문장. 이름 · 구역 · 수는 생성 지시의 주석에서만 읽는다."""
    if not ev["defect_positions"]:
        return f"{generation_policy.NORMAL_ABSENCE_FORMS[0]} {generation_policy.NORMAL_LIMITATION_FORMS[0]}"
    cells = Counter((d["name"], d["where"]) for d in ev["defect_positions"])
    return " ".join(f"{n} {k}개가 {w}에 표시되어 있다." for (n, w), k in cells.items()) + " " + generation_policy.LIMITATION_FORMS[0]


class V7PipelineBackend(PipelineBackend):
    """영상 설명은 v7 생성 계약의 글로 쓰고, 판정은 v7 물음에 모두 통과로 답한다. 판정은 정지기를 켜야 답하고 종료는
    `json_closed` 다. 판정 입력의 행 종류를 `judged` 에 남기고, 정상 행 입력이 오면 실패한다(정상 행은 판정 모델을 부르지 않는다)."""

    judged: ClassVar[list] = []
    stopping = False

    def enable_json_object_stopping(self):
        assert self.stage == "judge"
        self.stopping = True

    def generate_batch(self, prompts, system, deadline):
        if self.stage != "judge":
            rows = super().generate_batch(prompts, system, deadline)
            for p, r in zip(prompts, rows):
                if "구절 id:" not in p:
                    r["text"] = _v7_contract_text(json.loads(p.split("주석: ", 1)[1]))
            return rows
        assert self.stopping, "v7 판정은 정지기를 켠 모델에만 간다"
        out = []
        for p in prompts:
            request = json.loads(p)
            kind = request.get("row_class", request.get("task_kind"))
            assert kind != "defect_free", "정상 행이 판정 모델에 갔다"
            V7PipelineBackend.judged.append(kind)
            answer = ({**dict.fromkeys(V7_QUESTIONS, True), "reason": "주석과 같다"} if kind == "defect"
                      else {"grounded": True, "scope_ok": True, "language_ok": True, "reason": "근거와 일치"})
            out.append({"text": json.dumps(answer, ensure_ascii=False), "finish_reason": "json_closed",
                        "input_tokens": 20, "output_tokens": 15})
        return out


def _separate_judge_v7(world, run_dir, name="judge-v7"):
    """v7 판정을 판정 신원 쓰기로 돌리고(출력 160 토큰 · 시험용 판정 모델) 판정 신원 읽기로 객체를 받는다 — 읽기에 꼴을 주지
    않는다(판정 폴더의 꼴로 스스로 고른다)."""
    target = world.tmp / name
    V7PipelineBackend.judged = []
    PipelineBackend.backend = "fake"
    try:
        done = infer.run_judge(run_dir, world.config, target, judge_config={**world.config["judge"], "max_new_tokens": 160},
                               policy_version="v7", backend_factory=V7PipelineBackend)
    finally:
        PipelineBackend.backend = "transformers"
    assert done["exit_code"] == 0 and done["policy_version"] == "v7" and done["completed_tasks"] == done["planned_tasks"]
    return infer.read_judge_source(run_dir, world.config, target)


def test_v7_판정_경로는_실제_쓰기_읽기_정책으로_집계까지_가고_정상_행은_판정_모델을_부르지_않는다(tmp_path):
    world, bank = _world(tmp_path)
    _prepare(world, bank)
    run_dir = world.tmp / "run"
    assert infer.run_stage(run_dir, "generate", world.config, backend_factory=V7PipelineBackend)["exit_code"] == 0
    src = _separate_judge_v7(world, run_dir)
    result = audit_run(run_dir, world.config, world.tmp / "audit", judge_source=src)
    audit = json.loads((world.tmp / "audit" / "AUDIT.json").read_bytes())
    tasks = _tasks(run_dir)
    normal = [t["id"] for t in tasks if quality._judge_row_class(t) == "defect_free"]
    model = [t["id"] for t in tasks if t["id"] not in normal]
    kinds = Counter(quality._judge_row_class(t) for t in tasks)
    assert kinds["defect_free"] == len(normal) == 2 and kinds["defect"] > 0 and kinds["source_qa"] > 0

    # 판정 신원 읽기가 v7 을 골랐다 — 영수증은 모든 행, 판정 행은 모델 대상만, 정상 행은 판정 모델에 가지 않았다
    assert src["policy_version"] == "v7" and src["output_schema"] == quality.JUDGE_V7_SCHEMA_VERSION
    assert src["all_task_ids"] == [t["id"] for t in tasks] == [r["id"] for r in src["deterministic_receipts"]]
    assert src["model_task_ids"] == model == [r["id"] for r in src["rows"]]
    assert Counter(V7PipelineBackend.judged) == {"defect": kinds["defect"], "source_qa": kinds["source_qa"]}
    assert {r["finish_reason"] for r in src["rows"]} == {"json_closed"}

    # 집계 — v7 파서 · 판정 신원에 묶인 실제 v7 정책 · 결정적 영수증
    ji = audit["judge_identity"]
    assert ji["source"] == "separate" and ji["identity_sha256"] == src["identity_sha256"]
    assert ji["parser_schema"] == "judge_structured_v7" and audit["judge_parser_version"] == "parse_judge_v7"
    assert audit["audit_version"] == "quality_v10" and audit["policy"]["bound_to_judge_identity"] is True
    assert audit["policy"]["identity_sha256"] == fixtures.sha(canonical_bytes(src["identity_basis"]["policy_identity"]))
    # 막는 것은 시험용 판정 백엔드 하나 — 모든 행이 수락됐다(정상 행은 judge_missing 이 아니다)
    assert result["status"] == "blocked" and audit["blockers"] == ["non_transformers_backend"], audit["blockers"]
    assert audit["counts"]["accepted"] == len(tasks) and audit["complete"] is True and audit["incomplete"] == 0
    det = audit["deterministic_receipts"]
    assert det["rows"] == det["planned"] == len(src["deterministic_receipts"]) == len(tasks)
    assert det["policy_identity_sha256"] == audit["policy"]["identity_sha256"]
    assert det["model_task_ids"] == {"count": len(model), "ids_sha256": quality.ids_sha256(model)}
    assert det["normal"]["receipts"] == det["normal"]["deterministic_pass"] == len(normal)
    assert det["normal"]["status"] == {"accepted": len(normal)} and det["normal"]["reasons"] == {}
    assert det["model"]["receipts"] == det["model"]["deterministic_pass"] == len(model)
    assert det["model"]["status"] == {"accepted": len(model)}

    rows = {r["id"]: r for r in map(json.loads, (world.tmp / "audit" / "dataset.jsonl").read_text(encoding="utf-8").splitlines())}
    assert list(rows) == [t["id"] for t in tasks]
    for t in tasks:
        judge = rows[t["id"]]["provenance"]["judge"]
        if t["id"] in normal:
            assert judge == {"applicable": False, "identity_sha256": src["identity_sha256"]}
        elif t["kind"] == "image_report":
            assert judge["verdict"] == {**dict.fromkeys(V7_QUESTIONS, True), "reason": "주석과 같다"}
            assert list(judge["verdict"]) == [*V7_QUESTIONS, "reason"]
            assert judge["schema"] == "judge_structured_v7" and judge["parser"] == "parse_judge_v7"
        else:
            assert judge["verdict"] == {"grounded": True, "scope_ok": True, "language_ok": True, "reason": "근거와 일치"}


# --------------------------------------------------------------------------- 같은 바이트 영상


def _same_bytes(world, a: str, b: str) -> None:
    """영상 b 의 바이트를 a 와 같게 바꾸고 매니페스트 · 페어를 다시 봉인한다(시험 세계에서만)."""
    rows = {r["image_id"]: r for r in world.manifest_rows}
    (world.repo / rows[b]["rel_path"]).write_bytes((world.repo / rows[a]["rel_path"]).read_bytes())
    rows[b]["sha256"] = rows[a]["sha256"]
    world.write_manifest()
    world.seal_pairs()


def test_같은_계보의_같은_바이트는_허용되고_엇갈린_주석은_전수_검수로_간다(tmp_path):
    world, bank = _world(tmp_path)
    _same_bytes(world, "img:0001", "img:0002")        # 같은 묶음 grp_a — 결함 둘 대 정상, 주석이 엇갈린다
    run_dir, result, audit = _run(tmp_path, world, bank)
    run = json.loads((run_dir / "RUN.json").read_bytes())
    assert run["counts"]["duplicate_image_same_lineage"] == 1
    assert audit["duplicate_image_sha256"] == [] and audit["identical_images_same_lineage"]["conflicting_target_rows"] == 2
    full = audit["review"]["strata"]["image_report|identical_bytes_conflicting_target"]
    assert sorted(full) == ["image_report:img:0001", "image_report:img:0002"]
    assert result["status"] == "ready_for_human_review", audit["blockers"]
    assert _release(world, tmp_path / "audit", tmp_path / "release")["exit_code"] == 0


def test_다른_계보의_같은_바이트는_명부에서_멈춘다(tmp_path):
    world, bank = _world(tmp_path)
    _same_bytes(world, "img:0001", "img:0003")        # grp_a 와 grp_b
    with pytest.raises(prepare.PrepareRefused) as err:
        _prepare(world, bank)
    assert err.value.code == "image_duplicate_cross_lineage"
    assert not (tmp_path / "run").exists()


# --------------------------------------------------------------------------- 릴리스 거부


def _refused(world, tmp_path, code):
    out = tmp_path / "release"
    with pytest.raises(release.ReleaseRefused) as err:
        _release(world, tmp_path / "audit", out)
    assert err.value.code == code, err.value
    assert not out.exists() and not list(tmp_path.glob(".release.*.tmp"))


def test_시험용_백엔드의_생성문은_집계에서_막히고_릴리스되지_않는다(tmp_path):
    world, bank = _world(tmp_path)
    _, result, audit = _run(tmp_path, world, bank, generate_backend="fake")
    assert result["status"] == "blocked" and "non_transformers_backend" in audit["blockers"]
    _refused(world, tmp_path, "row_backend")


def test_한_행이라도_기각되면_부분_릴리스하지_않는다(tmp_path):
    world, bank = _world(tmp_path)
    PipelineBackend.reject_word = "슬래그 혼입"        # 그 구절의 QA 세 행만 판정에서 떨어진다
    _, result, audit = _run(tmp_path, world, bank)
    assert result["status"] == "blocked" and audit["rejected"] == 3 and "rejected_rows" in audit["blockers"]
    _refused(world, tmp_path, "audit_rejected")


# --------------------------------------------------------------------------- 렌더 규칙의 두 구현(품질 집계 · 릴리스)


class TrickyBackend(PipelineBackend):
    """생성문을 까다로운 꼴로 — QA 는 json 울타리 하나와 앞뒤 공백, 설명은 앞뒤 공백과 가운데 줄바꿈."""

    def generate_batch(self, prompts, system, deadline):
        rows = super().generate_batch(prompts, system, deadline)
        if self.stage == "generate":
            for p, r in zip(prompts, rows):
                if "구절 id:" in p:
                    r["text"] = "\n ```json\n" + r["text"] + "\n```  \n"
                else:
                    r["text"] = "  " + r["text"].replace(". ", ".\n", 1) + " \n"
        return rows


def test_품질_집계와_릴리스의_렌더_규칙이_같은_바이트를_낸다(tmp_path):
    """렌더 규칙은 두 곳(quality 의 데이터셋 행 · release.render_messages)에 있다 — 같은 생성문에서 같은 바이트를 내야 한다."""
    world, bank = _world(tmp_path)
    _prepare(world, bank)
    run_dir = world.tmp / "run"
    for stage in ("generate", "judge"):
        assert infer.run_stage(run_dir, stage, world.config, backend_factory=TrickyBackend)["exit_code"] == 0
    result = audit_run(run_dir, world.config, world.tmp / "audit")
    assert result["status"] == "ready_for_human_review", result
    tasks = {t["id"]: t for t in _tasks(run_dir)}
    gen = {r["id"]: r["text"] for r in infer.read_stage_rows(run_dir, "generate", world.config)}
    rows = [json.loads(x) for x in (world.tmp / "audit" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["kind"] for r in rows} == {"image_report", "source_qa"}
    assert any("\n" in r["messages"][1]["content"][0]["text"] for r in rows if r["kind"] == "image_report")
    for r in rows:
        assert canonical_bytes(r["messages"]) == canonical_bytes(release.render_messages(tasks[r["id"]], gen[r["id"]]))
