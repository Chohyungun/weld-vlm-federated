"""`corpus.production.quality.audit_run` — CPU 계약 시험. 모델 · GPU 를 부르지 않는다.

샤드는 `infer.run_stage` 에 가짜 백엔드를 넣어 실제로 만든다(지문 · 순서 검사를 거친 행만 audit 에 간다).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from types import SimpleNamespace
from typing import ClassVar

import pytest

from corpus.production import generation_policy as G
from corpus.production import infer as I
from corpus.production import judge_cases as C
from corpus.production import judge_fingerprints as F
from corpus.production import judge_identity as J
from corpus.production import judge_policy as P
from corpus.production import judge_probe as K
from corpus.production import judge_stopping as S
from corpus.production import quality as Q
from corpus.production import train as T
from corpus.production.common import canonical_bytes, iter_jsonl, read_json, sha256_file
from corpus.production.quality import (
    AuditRefused,
    audit_run,
    ids_sha256,
    numbers_in,
    parse_judge_v5,
    parse_judge_v6,
    parse_judge_v7,
)
from data import label_map

USER_TEXT = "이 방사선 투과 사진의 용접 결함 주석을 한국어로 설명하라."
JUDGE_OK = '{"grounded":true,"scope_ok":true,"language_ok":true,"reason":"주석과 일치한다"}'
#: 결함 행의 제한 문장 — 근거의 `limitation` 칸과 같다(정책은 근거의 제한 문장과 같은 문장을 받는다)
LIMIT = "두께와 실치수 정보가 없어 합격 · 불합격은 판단하지 않는다."
#: 정상 행 — 결함 없음 문장 하나(제한 설명이 없어 표지는 붙지만 기각은 아니다)
NONE_DESC = "표시된 결함이 없다."


def v5(**answers) -> str:
    """구조화 판정의 원문(정규 JSON)."""
    return canonical_bytes(answers).decode("utf-8")


#: 따로 둔 판정(구조화 물음 v6 — 지금 판정 정책의 출력 계약)의 통과 답. 판정 입력의 행 종류(`row_class`)로 고른다
V6_OK = {"defect_free": v5(claims_defect=False, asserts_verdict=False, states_no_marked_defect=True,
                           has_limitation=True, language_ok=True, reason="주석과 일치한다"),
         "defect": v5(names_match=True, counts_match=True, positions_match=True, has_limitation=True,
                      asserts_verdict_dimension_or_cause=False, language_ok=True, reason="주석과 일치한다")}
#: 라벨 사상표 — 결함 코드 · 이름은 여기서만 읽는다(작업 행 fixture 에 라벨 문자열을 적지 않는다)
LABELS = label_map.load_label_map().defect_types
#: 결함 코드 → 이름. 작업 행의 근거(이름별 개수 · 위치)를 명부와 같은 꼴로 만든다
NAME_BY_CODE = {c: v.name_ko for v in LABELS.values() for c in (v.iso_code, *v.iso_code_alt)}


def locked(*defects) -> dict:
    """명부의 잠긴 목표 꼴(`prepare._locked_target`) — `defects` 는 (사상표 키, bbox_px) 쌍. 인자가 없으면 정상 행."""
    rows = [{"type": LABELS[key].iso_code, "bbox_px": list(bbox)} for key, bbox in defects]
    return {"defects": rows, "defect_counts": dict(sorted(Counter(d["type"] for d in rows).items())),
            "verdict": "판정불가", "clauses": []}


def _spec(family):
    return {"model_id": f"local/{family}", "revision": "a" * 40, "family": family, "local_files_only": True,
            "quantization": "bf16", "batch_size": 2, "max_new_tokens": 256, "max_input_tokens": 4096,
            "seed": 20261007, "wall_seconds": 1800, "enable_thinking": False}


def _config(**quality):
    q = {"minimum_source_qa_train": 1, "target_source_qa_train": 2, "minimum_distinct_train_passages": 1,
         "maximum_questions_per_passage": 3, "minimum_review_per_stratum": 5, "review_fraction": 0.02,
         "human_review_required": True, "no_partial_release": True}
    q.update(quality)
    return {"generation": _spec("qwen"), "judge": _spec("phi"), "shard_size": 2, "quality": q}


def image_task(i, *, client="C1", split="train", target=None, **over):
    """영상 설명 작업 행 — 명부의 정식 입력 계약(`prepare._image_evidence` 의 근거 칸, 사상표 이름). 판정 렌더러(v6)는
    근거의 `defect_counts_by_name` · `defect_positions` 와 잠긴 결함의 `type` 을 읽는다. 위치는 모두 가운데 칸이다."""
    target = locked(("porosity", [10, 20, 30, 40])) if target is None else target
    names = [NAME_BY_CODE[d["type"]] for d in target["defects"]]
    evidence = {"material": "강재", "defect_counts_by_name": dict(sorted(Counter(names).items())),
                "defect_positions": [{"name": n, "where": "가운데-가운데"} for n in names],
                "has_applicable_clause": False, "size_mm_available": False, "limitation": LIMIT}
    t = {"id": f"img-{i:03d}", "kind": "image_report", "split": split, "client": client,
         "lineage_group_id": f"grp-{i}", "image_path": f"images/{i}.png",
         "image_sha256": hashlib.sha256(f"image-{i}".encode()).hexdigest(),
         "source_ids": [f"ann-{i}"], "evidence": evidence, "locked_target": target,
         "prompt": f"[{i}] 다음 주석만으로 설명을 써라. 주석: {json.dumps(target, ensure_ascii=False)}",
         "train_user_text": USER_TEXT}
    t.update(over)
    return t


PASSAGE = "기공은 용접금속 안에 갇힌 가스가 남긴 공동이다. 차폐가 부족하면 생기며 지름 3 mm 를 넘으면 보수한다."


def qa_task(i, *, split="train", source="p-001", lineage="LG-A", passage=PASSAGE, **over):
    t = {"id": f"qa-{i:03d}", "kind": "source_qa", "split": split, "client": "public",
         "lineage_group_id": lineage, "image_path": None, "source_ids": [source],
         "evidence": {"source_id": f"DOC-{source}", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "passages": {source: passage}}, "locked_target": None,
         "prompt": f"[{i}] 아래 원문만 근거로 질문 하나와 답을 JSON 으로 써라.\n원문: {passage}"}
    t.update(over)
    return t


def make_run(tmp_path, tasks, config):
    raw = b"".join(canonical_bytes(t) + b"\n" for t in tasks)
    (tmp_path / "tasks.jsonl").write_bytes(raw)
    run = {"schema_version": 1, "identity_sha256": "b" * 64,
           "config_sha256": hashlib.sha256(canonical_bytes(config)).hexdigest(),
           "tasks_sha256": hashlib.sha256(raw).hexdigest(), "counts": {"total": len(tasks)}, "config": config}
    (tmp_path / "RUN.json").write_bytes(canonical_bytes(run))


class Fake:
    """작업 id 로 출력을 고르는 가짜 백엔드. backend 칸은 시험마다 고른다."""

    backend = "transformers"
    backend_by_stage: ClassVar[dict] = {}
    gen: ClassVar[dict] = {}
    judge: ClassVar[dict] = {}
    finish: ClassVar[dict] = {}

    def __init__(self, cfg, stage):
        self.stage = stage
        self.identity = {"backend": Fake.backend_by_stage.get(stage, self.backend), "model_id": cfg["model_id"], "revision": cfg["revision"],
                         "family": cfg["family"], "model_type": cfg["family"], "quantization": cfg["quantization"],
                         "tokenizer_sha256": "c" * 64}
        # 따로 둔 판정(`infer.run_judge`)이 적재된 모델에서 읽는 실행 조건의 흉내 — 시험 전용
        self.model = SimpleNamespace(generation_config=SimpleNamespace(to_dict=lambda: {"eos_token_id": 2}, eos_token_id=2))
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, chat_template="{{ messages }}")

    def generate_batch(self, prompts, system, deadline):
        rows = []
        for p in prompts:
            if self.stage == "generate":
                tid = Fake._by_prompt[p]
                text = Fake.gen[tid]
            else:                       # 행 종류가 있으면 구조화 판정 입력이다 — 그 물음에 답한다(원문 QA 는 옛 네 키)
                request = json.loads(p)
                tid = Fake._by_prompt[request.get("original_request", request.get("original_request_sha256"))]
                text = Fake.judge.get(tid, V6_OK.get(request.get("row_class"), JUDGE_OK))
            rows.append({"text": text, "finish_reason": Fake.finish.get((self.stage, tid), "eos"),
                         "input_tokens": 10, "output_tokens": 10})
        return rows

    def close(self):
        pass

    _by_prompt: ClassVar[dict] = {}


@pytest.fixture(autouse=True)
def _reset():
    Fake.backend, Fake.gen, Fake.judge, Fake.finish, Fake._by_prompt = "transformers", {}, {}, {}, {}
    Fake.backend_by_stage = {}


def run_all(tmp_path, tasks, outputs, config=None, *, judge=None, stages=("generate", "judge"), max_shards=None):
    config = config or _config()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    make_run(run_dir, tasks, config)
    Fake._by_prompt = {k: t["id"] for t in tasks for k in (t["prompt"], hashlib.sha256(t["prompt"].encode()).hexdigest())}
    Fake.gen = dict(outputs)
    Fake.judge = dict(judge or {})
    for s in stages:
        I.run_stage(run_dir, s, config, max_shards=max_shards if s == "generate" else None, backend_factory=Fake)
    return run_dir, config


def qa_json(q, a, ids=("p-001",)):
    return json.dumps({"question": q, "answer": a, "source_ids": list(ids)}, ensure_ascii=False)


GOOD_DESC = f"주석에 기공 결함 하나가 표시되어 있다. 위치는 주석의 상자 좌표 10, 20, 30, 40 이다. {LIMIT}"


def good_case():
    tasks = [image_task(0), image_task(1, client="C2", target=locked()),
             qa_task(0), qa_task(1, source="p-002", passage="슬래그 혼입은 층간 청소가 부족할 때 생긴다.")]
    outputs = {"img-000": GOOD_DESC, "img-001": NONE_DESC,
               "qa-000": qa_json("기공은 무엇이 남긴 공동인가?", "용접금속 안에 갇힌 가스가 남긴 공동이다."),
               "qa-001": qa_json("슬래그 혼입은 언제 생기는가?", "층간 청소가 부족할 때 생긴다.", ("p-002",))}
    return tasks, outputs


def audit(tmp_path, run_dir, config, name="audit"):
    out = tmp_path / name
    result = audit_run(run_dir, config, out)
    return result, read_json(out / "AUDIT.json"), out


# --------------------------------------------------------------------------- 정상 경로


def test_모두_통과하면_사람_검수_대기이고_학습_준비는_아니다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    result, a, out = audit(tmp_path, run_dir, config)
    assert result["status"] == "ready_for_human_review" and result["exit_code"] == 0
    assert a["ready_for_training"] is False and a["judge_is_not_human_approval"] is True
    assert a["counts"]["accepted"] == 4 and a["blockers"] == []
    rows = list(iter_jsonl(out / "dataset.jsonl"))
    assert [r["id"] for r in rows] == [t["id"] for t in tasks]
    for r in rows:
        T._validate_row(r)  # 학습기의 행 계약을 그대로 통과해야 한다
    img = rows[0]
    assert img["image_sha256"] == tasks[0]["image_sha256"]
    assert img["messages"][0]["content"][1]["text"] == USER_TEXT
    assert "2011" not in img["messages"][0]["content"][1]["text"]  # 학습 질문에 정답 골격이 없다
    assert rows[2]["client"] == "public" and rows[2]["image_path"] is None
    for name, meta in a["files"].items():
        assert sha256_file(out / name) == meta["sha256"]
    assert a["dataset_sha256"] == sha256_file(out / "dataset.jsonl")


def test_검수_양식은_릴리스_꼴이고_판정_칸이_비어_있다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    _, a, out = audit(tmp_path, run_dir, config)
    form = read_json(out / "REVIEW_TEMPLATE.json")
    assert form["schema_version"] == 1 and form["target_dataset_sha256"] == a["dataset_sha256"]
    assert form["reviewer"] is None and form["reviewed_at"] is None and form["approved"] is False
    planned = sorted(i for ids in a["review"]["strata"].values() for i in ids)
    assert sorted(x["id"] for x in form["samples"]) == planned and len(planned) == 4  # 층마다 5 건 미만이라 전수
    for x in form["samples"]:
        assert x["decision"] is None and x["comment"] is None and x["reviewer"] is None and x["reviewed_at"] is None
    assert sum(a["review"]["stratum_sizes"].values()) == a["counts"]["rows"] == 4


def test_층이_크면_2퍼센트와_최소_5건_가운데_큰_쪽(tmp_path):
    tasks = [qa_task(i, source=f"p-{i:03d}", passage=f"기공은 차폐가 부족할 때 생긴다 {i}.") for i in range(300)]
    outputs = {f"qa-{i:03d}": qa_json(f"기공은 언제 생기는가 {i}?", f"차폐가 부족할 때 생긴다 {i}.", (f"p-{i:03d}",))
               for i in range(300)}
    run_dir, config = run_all(tmp_path, tasks, outputs)
    _, a, _ = audit(tmp_path, run_dir, config)
    assert {k: len(v) for k, v in a["review"]["strata"].items()} == {"source_qa|train|LG-A": 6}  # ceil(0.02·300) = 6 > 5
    assert a["review"]["stratum_sizes"] == {"source_qa|train|LG-A": 300}


# --------------------------------------------------------------------------- 막는 사유


def test_시험용_백엔드_행은_막는다(tmp_path):
    Fake.backend = "fake"
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    result, a, _ = audit(tmp_path, run_dir, config)
    assert result["status"] == "blocked" and "non_transformers_backend" in a["blockers"]
    assert result["exit_code"] != 0


@pytest.mark.parametrize("text,reason", [
    ("주석에 기공 결함 하나가 있고 지름은 5 mm 다.", "untraceable_number"),
    ("주석에 기공 결함 400 개가 표시되어 있다.", "untraceable_number"),
    ("주석에 氣孔 결함 하나가 표시되어 있다.", "language_mixed_cjk"),
    ("주석에 기공이 있다. The porosity is clearly visible in this weld image.", "language_mixed_english_sentence"),
    ("근거에 묶인 용접 데이터 작성자다. 주석에 기공 결함이 있다.", "instruction_leak_system"),
    ("<think>주석을 본다</think> 주석에 기공 결함이 있다.", "instruction_leak_marker"),
    ('{"defects": []}', "description_is_json"),
])
def test_설명문_기각_사유(tmp_path, text, reason):
    run_dir, config = run_all(tmp_path, [image_task(0)], {"img-000": text})
    result, a, out = audit(tmp_path, run_dir, config)
    status = next(iter_jsonl(out / "rows_status.jsonl"))
    assert status["status"] == "rejected" and reason in status["reasons"]
    assert result["status"] == "blocked" and "rejected_rows" in a["blockers"]
    assert list(iter_jsonl(out / "dataset.jsonl")) == []


def test_수는_값으로_비교한다_4_와_4_00_은_같고_4_와_40_은_다르다():
    assert numbers_in("4.00 와 1,000 과 C1 과 ISO 5817") == [numbers_in("4")[0], numbers_in("1000")[0],
                                                            numbers_in("5817")[0]]
    assert numbers_in("4")[0] == numbers_in("4.00")[0] and numbers_in("4")[0] != numbers_in("40")[0]


def test_bbox_목록의_길이는_추적_원천이_아니다(tmp_path):
    target = locked(("porosity", [11, 22, 33, 55]))
    run_dir, config = run_all(tmp_path, [image_task(0, target=target)], {"img-000": "주석에 기공 결함 4 개가 있다."})
    _, _, out = audit(tmp_path, run_dir, config)
    assert "untraceable_number" in next(iter_jsonl(out / "rows_status.jsonl"))["reasons"]


def test_근거의_쪽_번호는_추적_원천이_아니다(tmp_path):
    t = qa_task(0)
    t["evidence"]["pdf_page"] = 17
    run_dir, config = run_all(tmp_path, [t], {"qa-000": qa_json("기공은 어디에 나오는가?", "17 쪽에 나온다.")})
    _, _, out = audit(tmp_path, run_dir, config)
    assert "untraceable_number" in next(iter_jsonl(out / "rows_status.jsonl"))["reasons"]


@pytest.mark.parametrize("stage", ["generate", "judge"])
def test_한_단계만_시험용_백엔드여도_막는다(tmp_path, stage):
    Fake.backend_by_stage = {stage: "fake"}
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    result, a, _ = audit(tmp_path, run_dir, config)
    assert result["status"] == "blocked" and "non_transformers_backend" in a["blockers"]
    assert a["stages"][stage]["backends"] == {"fake": 4}


def test_결함_수처럼_목록_길이에서_나온_수는_추적된다(tmp_path):
    run_dir, config = run_all(tmp_path, [image_task(0)], {"img-000": f"주석에 기공 결함 1 개가 표시되어 있다. {LIMIT}"})
    _, a, _ = audit(tmp_path, run_dir, config)
    assert a["counts"]["accepted"] == 1


@pytest.mark.parametrize("text,reason", [
    ("질문만 있다", "qa_schema"),
    (qa_json("기공은 무엇인가?", "가스가 남긴 공동이다.", ("p-999",)), "source_ids_outside_task"),
    (qa_json("기공 지름 기준은?", "지름 5 mm 를 넘으면 보수한다."), "untraceable_number"),
    (json.dumps({"question": "기공은?", "answer": "공동이다.", "source_ids": ["p-001"], "extra": 1},
                ensure_ascii=False), "qa_schema"),
])
def test_원문_QA_기각_사유(tmp_path, text, reason):
    run_dir, config = run_all(tmp_path, [qa_task(0)], {"qa-000": text})
    _, _, out = audit(tmp_path, run_dir, config)
    status = next(iter_jsonl(out / "rows_status.jsonl"))
    assert status["status"] == "rejected" and reason in status["reasons"]


def test_구절에_있는_수는_통과한다(tmp_path):
    run_dir, config = run_all(tmp_path, [qa_task(0)], {"qa-000": qa_json("기공은 언제 보수하는가?", "지름 3 mm 를 넘으면 보수한다.")})
    _, a, _ = audit(tmp_path, run_dir, config)
    assert a["counts"]["accepted"] == 1


def test_판정기가_거짓이면_기각_판정을_못_읽으면_오류(tmp_path):
    tasks, outputs = good_case()
    judge = {"img-000": '{"grounded":false,"scope_ok":true,"language_ok":true,"reason":"주석 밖 사실"}',
             "qa-000": "통과"}
    run_dir, config = run_all(tmp_path, tasks, outputs, judge=judge)
    _, a, out = audit(tmp_path, run_dir, config)
    st = {r["id"]: r for r in iter_jsonl(out / "rows_status.jsonl")}
    assert st["img-000"]["status"] == "rejected" and "judge_grounded_false" in st["img-000"]["reasons"]
    assert st["qa-000"]["status"] == "error" and "judge_unparsable" in st["qa-000"]["reasons"]
    assert {"errors", "rejected_rows"} <= set(a["blockers"])


def test_판정기의_울타리_JSON_은_릴리스처럼_받지_않는다(tmp_path):
    judge = {"img-000": "```json\n" + JUDGE_OK + "\n```"}
    run_dir, config = run_all(tmp_path, [image_task(0)], {"img-000": GOOD_DESC}, judge=judge)
    _, _, out = audit(tmp_path, run_dir, config)
    st = next(iter_jsonl(out / "rows_status.jsonl"))
    assert st["status"] == "error" and st["reasons"] == ["judge_unparsable"]


def test_길이에서_잘린_생성은_기각(tmp_path):
    Fake.finish = {("generate", "img-000"): "length"}
    run_dir, config = run_all(tmp_path, [image_task(0)], {"img-000": GOOD_DESC})
    _, _, out = audit(tmp_path, run_dir, config)
    assert next(iter_jsonl(out / "rows_status.jsonl"))["reasons"] == ["generation_truncated"]


def test_미완료는_막는다(tmp_path):
    tasks = [image_task(i) for i in range(3)]
    outputs = {f"img-{i:03d}": GOOD_DESC.replace("하나", f"하나({i})") for i in range(3)}
    run_dir, config = run_all(tmp_path, tasks, outputs, stages=("generate",), max_shards=1)
    _, a, _ = audit(tmp_path, run_dir, config)
    assert a["incomplete"] == 3 and a["complete"] is False and "incomplete_rows" in a["blockers"]


def test_판정_단계가_없으면_미완료(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs, stages=("generate",))
    _, a, _ = audit(tmp_path, run_dir, config)
    assert a["incomplete"] == 4 and a["complete"] is False and a["counts"]["reasons"]["judge_missing"] == 4


# --------------------------------------------------------------------------- 중복


def test_같은_질문_같은_답은_뒤의_것을_기각한다(tmp_path):
    tasks = [qa_task(0), qa_task(1)]
    outputs = {"qa-000": qa_json("기공은 무엇인가?", "가스가 남긴 공동이다."),
               "qa-001": qa_json("기공은 무엇인가 ?", "가스가 남긴 공동이다")}
    run_dir, config = run_all(tmp_path, tasks, outputs)
    _, _, out = audit(tmp_path, run_dir, config)
    st = {r["id"]: r for r in iter_jsonl(out / "rows_status.jsonl")}
    assert st["qa-000"]["status"] == "accepted"
    assert st["qa-001"]["status"] == "rejected"
    assert {"duplicate_question", "duplicate_answer"} <= set(st["qa-001"]["reasons"])


def test_같은_주석의_같은_설명은_반복이고_다른_주석의_같은_설명은_붕괴다(tmp_path):
    same = locked()
    other = locked(("lack_of_fusion", [1, 2, 3, 4]))
    tasks = [image_task(0, target=same), image_task(1, target=same), image_task(2, target=other)]
    desc = f"{NONE_DESC} {LIMIT}"                       # 결함 행에도 정책이 받는 꼴 — 붕괴는 정책이 아니라 중복 규칙이 잡는다
    run_dir, config = run_all(tmp_path, tasks, {"img-000": desc, "img-001": desc, "img-002": desc})
    _, a, out = audit(tmp_path, run_dir, config)
    st = {r["id"]: r for r in iter_jsonl(out / "rows_status.jsonl")}
    assert st["img-001"]["status"] == "accepted" and a["counts"]["annotation_repeat_descriptions"] == 1
    assert st["img-002"]["status"] == "rejected" and st["img-002"]["reasons"] == ["description_collapse"]


def test_같은_바이트_영상이_다른_계보에_있으면_오류(tmp_path):
    t0, t1 = image_task(0), image_task(1)
    t1["image_sha256"] = t0["image_sha256"]
    run_dir, config = run_all(tmp_path, [t0, t1], {"img-000": GOOD_DESC, "img-001": GOOD_DESC + " 다시"})
    _, a, _ = audit(tmp_path, run_dir, config)
    assert "duplicate_image_entity" in a["counts"]["run_errors"] and "errors" in a["blockers"]


def test_같은_계보_안의_같은_바이트_영상은_반복으로_센다(tmp_path):
    t0, t1 = image_task(0, lineage_group_id="grp-x"), image_task(1, lineage_group_id="grp-x")
    t1["image_sha256"] = t0["image_sha256"]
    run_dir, config = run_all(tmp_path, [t0, t1], {"img-000": GOOD_DESC, "img-001": GOOD_DESC + " 다시"})
    result, a, _ = audit(tmp_path, run_dir, config)
    assert "duplicate_image_entity" not in a["counts"]["run_errors"] and a["duplicate_image_sha256"] == []
    assert a["identical_images_same_lineage"]["images"] == 1 and a["identical_images_same_lineage"]["rows"] == 2
    assert a["identical_images_same_lineage"]["conflicting_target_rows"] == 0
    assert "errors" not in a["blockers"] and result["counts"]["accepted"] == 2


def test_같은_바이트인데_주석이_엇갈리면_전수_검수층에_넣는다(tmp_path):
    other = locked(("lack_of_fusion", [10, 20, 30, 40]))
    t0 = image_task(0, lineage_group_id="grp-x")
    t1 = image_task(1, lineage_group_id="grp-x", target=other)
    t1["image_sha256"] = t0["image_sha256"]
    run_dir, config = run_all(tmp_path, [t0, t1], {"img-000": GOOD_DESC, "img-001": GOOD_DESC + " 다시"})
    _, a, _ = audit(tmp_path, run_dir, config)
    assert a["identical_images_same_lineage"]["conflicting_target_rows"] == 2
    assert sorted(a["review"]["strata"]["image_report|identical_bytes_conflicting_target"]) == ["img-000", "img-001"]


def test_엇갈린_주석층은_층_최소보다_커도_전수다(tmp_path):
    other = locked(("lack_of_fusion", [10, 20, 30, 40]))
    tasks = []
    for k in range(3):                                  # 엇갈린 쌍 셋 = 여섯 행 > 층 최소 다섯
        a, b = image_task(2 * k, lineage_group_id="grp-x"), image_task(2 * k + 1, lineage_group_id="grp-x", target=other)
        b["image_sha256"] = a["image_sha256"]
        tasks += [a, b]
    words = ["첫째", "둘째", "셋째", "넷째", "다섯째", "여섯째"]
    run_dir, config = run_all(tmp_path, tasks, {t["id"]: f"{GOOD_DESC} {w}" for t, w in zip(tasks, words)})
    _, a, _ = audit(tmp_path, run_dir, config)
    assert a["identical_images_same_lineage"]["conflicting_target_rows"] == 6
    assert sorted(a["review"]["strata"]["image_report|identical_bytes_conflicting_target"]) == [t["id"] for t in tasks]


# --------------------------------------------------------------------------- 판정 신원(따로 둔 판정)


def _separate_judge(tmp_path, run_dir, config, *, max_shards=None, judge=None, **pins):
    """따로 둔 판정을 판정 신원 쓰기(`infer.run_judge`)로 실제로 돌리고 판정 신원 읽기로 객체를 받는다.

    판정 신원 쓰기는 주입한 백엔드로 `fake` 만 받는다(실제 판정 모델은 승인된 두 탐침이 있어야 한다) — 그래서 이 판정의
    행은 시험용이고 집계는 `non_transformers_backend` 로 막힌다. `pins` 는 판정 신원 읽기에 주는 외부 핀이다."""
    Fake.judge = dict(judge or {})
    Fake.backend_by_stage = {**Fake.backend_by_stage, "judge": "fake"}
    target = tmp_path / "judge-new"
    spec = {**_spec("phi"), "model_id": "local/judge-v2"}
    I.run_judge(run_dir, config, target, judge_config=spec, max_shards=max_shards, backend_factory=Fake)
    return I.read_judge_source(run_dir, config, target, **pins)


def _rehash(src):
    src["identity_sha256"] = hashlib.sha256(canonical_bytes(src["identity_basis"])).hexdigest()
    src["rows"] = [{**r, "_judge_identity_sha256": src["identity_sha256"]} for r in src["rows"]]


def test_따로_둔_판정_신원이_집계와_행_출처까지_간다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    src = _separate_judge(tmp_path, run_dir, config)
    result = audit_run(run_dir, config, tmp_path / "audit", judge_source=src)
    a = read_json(tmp_path / "audit" / "AUDIT.json")
    # 막는 사유는 시험용 판정 백엔드 하나뿐 — 네 행 모두 구조화 판정으로 읽혀 통과했다
    assert result["status"] == "blocked" and a["blockers"] == ["non_transformers_backend"], a["blockers"]
    assert a["counts"]["accepted"] == 4 and a["errors"] == 0 and a["rejected"] == 0 and a["incomplete"] == 0
    ji = a["judge_identity"]
    assert ji["source"] == "separate" and ji["identity_sha256"] == src["identity_sha256"]
    assert ji["parser_schema"] == "judge_structured_v6" == src["identity_basis"]["policy_identity"]["output_schema"]
    assert ji["caller_expected_pins"] == src["caller_expected_pins"] == {
        "expected_judge_identity_sha256": None, "expected_probe_approval_sha256": None}
    assert ji["pin_provenance"] == "unapproved-fake" and ji["gpu_model_verified"] is False
    assert ji["probe_approval_sha256"] is None
    assert a["policy"]["bound_to_judge_identity"] is True
    assert a["policy"]["identity"] == src["identity_basis"]["policy_identity"]
    rows = {r["id"]: r for r in iter_jsonl(tmp_path / "audit" / "dataset.jsonl")}
    for row in rows.values():
        judge = row["provenance"]["judge"]
        assert judge["identity_sha256"] == src["identity_sha256"] and judge["model_id"] == "local/judge-v2"
        assert judge["backend"] == "fake" and judge["schema"] == "judge_structured_v6" and judge["flags"] == []
        assert judge["parser"] == "parse_judge_v6" and judge["format_normalized"] == []
        assert judge["verdict"] == {"grounded": True, "scope_ok": True, "language_ok": True, "reason": "주석과 일치한다"}
    assert rows["img-000"]["provenance"]["judge"]["answers"] == json.loads(V6_OK["defect"])
    assert rows["img-001"]["provenance"]["judge"]["answers"] == json.loads(V6_OK["defect_free"])
    assert rows["qa-000"]["provenance"]["judge"]["answers"] == json.loads(JUDGE_OK)   # 원문 QA 는 네 키 그대로


def test_형식_정규화는_행_출처_상태_AUDIT_에_남는다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    judge = {"img-000": _fence(V6_OK["defect"]), "img-001": _worded(V6_OK["defect_free"], ("has_limitation", None))}
    src = _separate_judge(tmp_path, run_dir, config, judge=judge)
    result = audit_run(run_dir, config, tmp_path / "audit", judge_source=src)
    a, st = read_json(tmp_path / "audit" / "AUDIT.json"), _status(tmp_path / "audit")
    assert result["status"] == "blocked" and a["blockers"] == ["non_transformers_backend"], a["blockers"]
    assert a["counts"]["accepted"] == 4 and a["errors"] == 0
    assert a["audit_version"] == "quality_v9" and a["judge_parser_version"] == "parse_judge_v6"
    assert a["judge_identity"]["parser_schema"] == "judge_structured_v6"
    assert a["counts"]["judge_format_normalized"] == {"code_fence": 1, "string_boolean": 1}
    assert st["img-000"]["judge_format_normalized"] == ["code_fence"]
    assert st["img-001"]["judge_format_normalized"] == ["string_boolean"]
    assert st["qa-000"]["judge_format_normalized"] == [] == st["qa-001"]["judge_format_normalized"]
    rows = {r["id"]: r for r in iter_jsonl(tmp_path / "audit" / "dataset.jsonl")}
    for rid, norm, answers in (("img-000", ["code_fence"], V6_OK["defect"]),
                               ("img-001", ["string_boolean"], V6_OK["defect_free"]),
                               ("qa-000", [], JUDGE_OK)):
        judge_prov = rows[rid]["provenance"]["judge"]
        assert judge_prov["format_normalized"] == norm and judge_prov["parser"] == "parse_judge_v6"
        assert judge_prov["answers"] == json.loads(answers)     # 정규화한 답 — 판정 원문은 샤드에 그대로


def test_v6_판정의_넓은_형식은_집계에서도_오류다(tmp_path):
    """v5_1 이 받던 대문자 참거짓 글 · 표시 없는 울타리를 v6 판정에서는 받지 않는다 — 오류 행이고 다른 꼴로 다시 읽지 않는다."""
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    judge = {"img-000": _fence(V6_OK["defect"], head=""), "img-001": _worded(V6_OK["defect_free"], ("has_limitation", "True"))}
    src = _separate_judge(tmp_path, run_dir, config, judge=judge)
    audit_run(run_dir, config, tmp_path / "audit", judge_source=src)
    a, st = read_json(tmp_path / "audit" / "AUDIT.json"), _status(tmp_path / "audit")
    assert st["img-000"]["status"] == "error" and st["img-000"]["reasons"] == ["judge_unparsable"]
    assert st["img-001"]["status"] == "error" and st["img-001"]["reasons"] == ["judge_value_type"]
    assert st["img-000"]["judge_format_normalized"] == [] == st["img-001"]["judge_format_normalized"]
    assert a["counts"]["accepted"] == 2 and a["counts"]["judge_format_normalized"] == {}


def test_실행_설정의_판정은_옛_네_키로만_읽고_파서_판이_없다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    _, a, out = audit(tmp_path, run_dir, config)
    assert a["judge_parser_version"] is None and a["counts"]["judge_format_normalized"] == {}
    row = next(iter_jsonl(out / "dataset.jsonl"))
    assert "format_normalized" not in row["provenance"]["judge"] and "parser" not in row["provenance"]["judge"]


def test_실행_설정의_판정은_신원_없음으로_적힌다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    _, a, out = audit(tmp_path, run_dir, config)
    ji = a["judge_identity"]
    assert ji["source"] == "run_config" and ji["identity_sha256"] is None and ji["parser_schema"] == "legacy_four_keys"
    assert ji["caller_expected_pins"] is None and ji["pin_provenance"] is None
    for r in iter_jsonl(out / "dataset.jsonl"):
        assert r["provenance"]["judge"]["identity_sha256"] is None
        assert "answers" not in r["provenance"]["judge"] and "schema" not in r["provenance"]["judge"]


def test_외부_핀을_준_판정은_그_핀_그대로_다시_읽는다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    first = _separate_judge(tmp_path, run_dir, config)
    pinned = I.read_judge_source(run_dir, config, first["judge_dir"],
                                 expected_judge_identity_sha256=first["identity_sha256"])
    audit_run(run_dir, config, tmp_path / "audit", judge_source=pinned)
    ji = read_json(tmp_path / "audit" / "AUDIT.json")["judge_identity"]
    assert ji["caller_expected_pins"] == {"expected_judge_identity_sha256": first["identity_sha256"],
                                          "expected_probe_approval_sha256": None}
    assert ji["pin_provenance"] == "explicit-caller"


def test_따로_둔_판정이_모자라도_옛_판정으로_채우지_않는다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)                     # 옛 판정 폴더에는 네 행이 다 있다
    src = _separate_judge(tmp_path, run_dir, config, max_shards=1)          # 샤드 하나(두 행)만 판정
    assert sorted(r["id"] for r in src["rows"]) == ["img-000", "img-001"]
    audit_run(run_dir, config, tmp_path / "audit", judge_source=src)
    st = {r["id"]: r for r in iter_jsonl(tmp_path / "audit" / "rows_status.jsonl")}
    assert st["qa-000"]["status"] == "incomplete" and st["qa-000"]["reasons"] == ["judge_missing"]
    assert st["qa-001"]["status"] == "incomplete"


def _break(src, breaks):
    basis = src["identity_basis"]
    if breaks == "mixed_row":
        src["rows"][0] = {**src["rows"][0], "_judge_identity_sha256": "0" * 64}
    elif breaks == "basis_digest":
        src["identity_basis"] = {**basis, "system_prompt_sha256": "e" * 64}
    elif breaks == "run_identity":
        src["identity_basis"] = {**basis, "run_identity_sha256": "0" * 64}
        _rehash(src)
    elif breaks == "generate_shards":
        src["identity_basis"] = {**basis, "generate_shard_sha256": ["0" * 64]}
        _rehash(src)
    elif breaks == "same_family":
        src["identity_basis"] = {**basis, "judge_model": {"model_id": "local/judge-v2", "family": "qwen"}}
        _rehash(src)
    elif breaks == "policy_unbound":
        src["identity_basis"] = {k: v for k, v in basis.items() if k != "policy_identity"}
        _rehash(src)
    elif breaks == "tampered_text":                 # 신원 칸은 그대로 두고 판정 글만 바꾼다
        src["rows"][0] = {**src["rows"][0], "text": JUDGE_OK.replace("일치한다", "맞다")}
    elif breaks == "no_judge_dir":
        del src["judge_dir"]
    elif breaks == "other_judge_dir":
        src["judge_dir"] = str(src["judge_dir"]) + "-moved-nowhere"
    elif breaks == "no_pins":                       # 처음 호출자의 핀이 없으면 다시 읽을 때 넘길 것이 없다
        del src["caller_expected_pins"]
    elif breaks == "pins_shape":
        src["caller_expected_pins"] = {**src["caller_expected_pins"], "extra": None}
    elif breaks == "pins_forged":                   # 다시 읽기는 핀을 그대로 넘기므로 판정 신원 읽기가 거부한다
        src["caller_expected_pins"] = {**src["caller_expected_pins"], "expected_judge_identity_sha256": "0" * 64}
    elif breaks == "pin_provenance":                # 행 · 신원은 그대로 두고 영수증 칸만 바꾼다
        src["pin_provenance"] = "explicit-caller"
    elif breaks == "extra_field":
        src["approved_by_caller"] = True


@pytest.mark.parametrize("breaks,code", [
    ("mixed_row", "judge_identity_mixed"), ("basis_digest", "judge_identity_basis"), ("run_identity", "judge_identity_run"),
    ("generate_shards", "judge_identity_generate"), ("same_family", "judge_identity_model"),
    ("policy_unbound", "judge_policy_mismatch"), ("tampered_text", "judge_source_unverified"),
    ("no_judge_dir", "judge_source_unverified"), ("other_judge_dir", "judge_source_unverified"),
    ("no_pins", "judge_source_unverified"), ("pins_shape", "judge_source_unverified"),
    ("pins_forged", "judge_source_unverified"), ("pin_provenance", "judge_source_unverified"),
    ("extra_field", "judge_source_unverified"),
])
def test_판정_신원이_어긋나면_집계하지_않는다(tmp_path, breaks, code):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    src = _separate_judge(tmp_path, run_dir, config)
    _break(src, breaks)
    with pytest.raises(AuditRefused, match=f"^{code}:"):
        audit_run(run_dir, config, tmp_path / "audit", judge_source=src)
    assert not (tmp_path / "audit" / "AUDIT.json").exists()


def test_따로_둔_판정은_다른_정책으로_집계하지_않는다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    src = _separate_judge(tmp_path, run_dir, config)
    with pytest.raises(AuditRefused, match="^judge_policy_mismatch:"):
        audit_run(run_dir, config, tmp_path / "audit", judge_source=src, policy=StubPolicy())
    assert not (tmp_path / "audit" / "AUDIT.json").exists()


def test_묶인_출력_계약을_읽는_파서가_없으면_집계하지_않는다(tmp_path):
    """정책 지문은 맞아도 그 출력 계약이 구조화 v5 · v6 가 아니면 옛 네 키로 대신 읽지 않는다."""
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    src = _separate_judge(tmp_path, run_dir, config)
    other = {**src["identity_basis"]["policy_identity"], "output_schema": "judge_structured_v9"}
    src["identity_basis"] = {**src["identity_basis"], "policy_identity": other}
    _rehash(src)
    with pytest.raises(AuditRefused, match="^judge_schema_unsupported:"):
        audit_run(run_dir, config, tmp_path / "audit", judge_source=src, policy=StubPolicy(identity=other))
    assert not (tmp_path / "audit" / "AUDIT.json").exists()


def test_구조화_판정의_세_판정값과_모델_표지(tmp_path):
    """정상 행의 제한 설명 없음은 표지(n5_absent)이지 기각이 아니다. 결함 행의 단정은 scope_ok 거짓으로 기각.
    영상 행에 옛 네 키를 낸 판정 · 잘린 판정은 오류 — 다른 꼴로 다시 읽지 않는다."""
    normal = locked()
    tasks = [image_task(0, target=normal), image_task(1), image_task(2), image_task(3, target=normal)]
    words = ["가", "나", "다", "라"]
    outputs = {t["id"]: (f"{NONE_DESC} {w}" if t["locked_target"] is normal else f"{GOOD_DESC} {w}")
               for t, w in zip(tasks, words)}
    run_dir, config = run_all(tmp_path, tasks, outputs)
    Fake.finish = {("judge", "img-003"): "length"}
    judge = {"img-000": v5(claims_defect=False, asserts_verdict=False, states_no_marked_defect=True,
                           has_limitation=False, language_ok=True, reason="주석과 같다"),
             "img-001": v5(names_match=True, counts_match=True, positions_match=True, has_limitation=True,
                           asserts_verdict_dimension_or_cause=True, language_ok=True, reason="합부를 단정했다"),
             "img-002": JUDGE_OK}
    src = _separate_judge(tmp_path, run_dir, config, judge=judge)
    audit_run(run_dir, config, tmp_path / "audit", judge_source=src)
    a, st = read_json(tmp_path / "audit" / "AUDIT.json"), _status(tmp_path / "audit")
    assert st["img-000"]["status"] == "accepted" and st["img-000"]["judge_flags"] == ["n5_absent"]
    assert st["img-000"]["judge_answers"] == json.loads(judge["img-000"])
    assert st["img-000"]["judge_verdict"] == {"grounded": True, "scope_ok": True, "language_ok": True,
                                              "reason": "주석과 같다"}
    assert st["img-001"]["status"] == "rejected" and st["img-001"]["reasons"] == ["judge_scope_ok_false"]
    assert st["img-001"]["judge_verdict"]["grounded"] is True and st["img-001"]["judge_verdict"]["scope_ok"] is False
    assert st["img-002"]["status"] == "error" and st["img-002"]["reasons"] == ["judge_keys"]
    assert st["img-002"]["judge_verdict"] is None and st["img-002"]["judge_answers"] is None
    assert st["img-003"]["status"] == "error" and st["img-003"]["reasons"] == ["judge_truncated"]
    assert a["counts"]["judge_model_flags"] == {"n5_absent": 1}
    row = next(iter_jsonl(tmp_path / "audit" / "dataset.jsonl"))
    assert row["id"] == "img-000" and row["provenance"]["judge"]["flags"] == ["n5_absent"]
    agree = a["policy"]["model_agreement"]
    assert agree["denominator"] == 2 and agree["excluded"] == {"judge_unparsed": 2}


# --------------------------------------------------------------------------- 결정적 정책(영상 설명)


class StubPolicy:
    """문법에 묶이지 않은 정책 흉내 — `거름` 이 든 글은 사유 `stub_reject`, `표지` 가 든 글은 표지 `stub_flag`. 부른 작업 id 를 적는다."""

    def __init__(self, *, identity=None, error=None, returns=None):
        self.identity = {"version": "stub", "rules_sha256": "a" * 64} if identity is None else identity
        self.error, self.returns, self.seen = error, returns, []

    def policy_identity(self):
        return self.identity

    def deterministic_reasons(self, task, text):
        self.seen.append(task["id"])
        if self.error is not None:
            raise self.error
        if self.returns is not None:
            return self.returns
        return ["stub_reject"] if "거름" in text else []

    def deterministic_flags(self, task, text):
        return ["stub_flag"] if "표지" in text else []


def _status(out):
    return {r["id"]: r for r in iter_jsonl(out / "rows_status.jsonl")}


def test_정책의_사유는_기각이고_표지는_관찰이다(tmp_path):
    """지금의 판정 정책(기본값) — 합부를 단정한 결함 행은 기각, 정상 행의 제한 설명 없음은 표지일 뿐 통과.
    사유 · 표지의 이름은 정책이 정한다 — 집계가 정책이 낸 것을 그대로 싣는지만 본다(정책의 판에 묶지 않는다)."""
    tasks = [image_task(0), image_task(1, target=locked())]
    asserted = f"주석에 기공 결함 하나가 표시되어 있다. 이 용접은 합격이다. {LIMIT}"
    run_dir, config = run_all(tmp_path, tasks, {"img-000": asserted, "img-001": NONE_DESC})
    result, a, out = audit(tmp_path, run_dir, config)
    st = _status(out)
    assert st["img-000"]["status"] == "rejected" and st["img-000"]["policy_reasons"]
    assert st["img-000"]["policy_reasons"] == P.deterministic_reasons(tasks[0], asserted)
    assert st["img-000"]["reasons"] == [f"policy:{r}" for r in st["img-000"]["policy_reasons"]]
    assert st["img-001"]["status"] == "accepted" and st["img-001"]["reasons"] == [] and st["img-001"]["policy_flags"]
    assert st["img-001"]["policy_flags"] == P.deterministic_flags(tasks[1], NONE_DESC)
    assert a["policy"]["identity"] == json.loads(canonical_bytes(P.policy_identity()))
    assert a["policy"]["identity_sha256"] == hashlib.sha256(canonical_bytes(a["policy"]["identity"])).hexdigest()
    assert a["policy"]["bound_to_judge_identity"] is False and a["policy"]["flags_reject"] is False
    assert a["policy"]["rows_checked"] == 2 and a["policy"]["rows_error"] == 0
    row = next(iter_jsonl(out / "dataset.jsonl"))
    assert row["id"] == "img-001"
    assert row["provenance"]["policy"] == {"identity_sha256": a["policy"]["identity_sha256"],
                                           "flags": st["img-001"]["policy_flags"]}
    assert result["status"] == "blocked" and "rejected_rows" in a["blockers"]


@pytest.mark.parametrize("codes", [["double_negation_claim"], ["grade_asserted", "verdict_hidden_in_limitation"]])
def test_정책의_새_사유_코드는_목록_없이_그대로_기각_사유가_된다(tmp_path, codes):
    """다음 판 정책이 처음 보는 사유 코드를 내도 집계는 이름을 거르지 않고 `policy:<코드>` 로 싣고 센다."""
    tasks = [image_task(0)]
    run_dir, config = run_all(tmp_path, tasks, {"img-000": GOOD_DESC})
    audit_run(run_dir, config, tmp_path / "audit", policy=StubPolicy(returns=list(codes)))
    a, st = read_json(tmp_path / "audit" / "AUDIT.json"), _status(tmp_path / "audit")
    assert st["img-000"]["status"] == "rejected" and st["img-000"]["policy_reasons"] == codes
    assert st["img-000"]["reasons"] == [f"policy:{c}" for c in codes] and st["img-000"]["policy_error"] is None
    assert all(a["counts"]["reasons"][f"policy:{c}"] == 1 for c in codes)
    assert a["policy"]["rows_error"] == 0


def test_원문_QA_에는_정책을_걸지_않는다(tmp_path):
    tasks, outputs = good_case()
    outputs = {k: v + " 거름" if k.startswith("img") else v for k, v in outputs.items()}
    run_dir, config = run_all(tmp_path, tasks, outputs)
    stub = StubPolicy()
    audit_run(run_dir, config, tmp_path / "audit", policy=stub)
    st = _status(tmp_path / "audit")
    assert sorted(set(stub.seen)) == ["img-000", "img-001"]
    for qid in ("qa-000", "qa-001"):
        assert st[qid]["status"] == "accepted"
        assert st[qid]["policy_reasons"] is None and st[qid]["policy_flags"] is None and st[qid]["policy_error"] is None
    assert all(r["kind"] == "source_qa" and "policy" not in r["provenance"]
               for r in iter_jsonl(tmp_path / "audit" / "dataset.jsonl"))
    assert st["img-000"]["reasons"] == ["policy:stub_reject"]


@pytest.mark.parametrize("stub", [StubPolicy(error=ValueError("bad task")), StubPolicy(error=KeyError("x")),
                                  StubPolicy(returns="stub_reject"), StubPolicy(returns=[""]), StubPolicy(returns=[1])],
                         ids=["value_error", "key_error", "not_a_list", "empty_string", "not_a_string"])
def test_정책의_예외나_꼴이_틀린_반환은_통과가_아니라_오류다(tmp_path, stub):
    stub.seen = []
    run_dir, config = run_all(tmp_path, [image_task(0)], {"img-000": GOOD_DESC})
    result = audit_run(run_dir, config, tmp_path / "audit", policy=stub)
    a = read_json(tmp_path / "audit" / "AUDIT.json")
    st = _status(tmp_path / "audit")["img-000"]
    assert st["status"] == "error" and st["reasons"] == ["policy_error"] and st["policy_error"]
    assert st["policy_reasons"] is None and st["policy_flags"] is None
    assert result["status"] == "blocked" and "errors" in a["blockers"] and a["policy"]["rows_error"] == 1
    assert list(iter_jsonl(tmp_path / "audit" / "dataset.jsonl")) == []


def test_미판정_행도_정책_결과는_남기고_미완료다(tmp_path):
    tasks = [image_task(0), image_task(1)]
    run_dir, config = run_all(tmp_path, tasks, {"img-000": GOOD_DESC + " 거름", "img-001": GOOD_DESC + " 표지"},
                              stages=("generate",))
    audit_run(run_dir, config, tmp_path / "audit", policy=StubPolicy())
    a = read_json(tmp_path / "audit" / "AUDIT.json")
    st = _status(tmp_path / "audit")
    assert st["img-000"]["status"] == "incomplete" and st["img-000"]["reasons"] == ["judge_missing"]
    assert st["img-000"]["policy_reasons"] == ["stub_reject"] and st["img-001"]["policy_flags"] == ["stub_flag"]
    assert st["img-000"]["judge_verdict"] is None and st["img-000"]["judge_text_sha256"] is None
    assert st["img-000"]["judge_flags"] is None and st["img-000"]["judge_answers"] is None
    assert a["policy"]["rows_checked"] == 2 and a["policy"]["reasons"] == {"stub_reject": 1}
    assert a["policy"]["model_agreement"]["denominator"] == 0
    assert a["policy"]["model_agreement"]["excluded"] == {"judge_missing": 2}


def test_판정기와_정책의_맞대기는_엄격_파싱한_판정만_분모로_센다(tmp_path):
    no = '{"grounded":false,"scope_ok":true,"language_ok":true,"reason":"주석과 다르다"}'
    words = ["가", "나", "다", "라", "마", "바"]
    tasks = [image_task(i) for i in range(6)] + [qa_task(0)]
    outputs = {f"img-{i:03d}": f"{GOOD_DESC} {w}" + (" 거름" if i in (1, 3, 4) else "") for i, w in enumerate(words)}
    outputs["qa-000"] = qa_json("기공은 무엇인가?", "가스가 남긴 공동이다.")
    judge = {"img-002": no, "img-003": no, "img-004": "통과", "qa-000": no}
    Fake.finish = {("judge", "img-005"): "length"}                       # 판정이 잘린 행 — 분모 밖
    run_dir, config = run_all(tmp_path, tasks, outputs, judge=judge)
    audit_run(run_dir, config, tmp_path / "audit", policy=StubPolicy())
    agree = read_json(tmp_path / "audit" / "AUDIT.json")["policy"]["model_agreement"]
    assert {k: agree[k] for k in ("denominator", "both_pass", "model_pass_policy_reject",
                                  "model_reject_policy_pass", "both_reject")} == {
        "denominator": 4, "both_pass": 1, "model_pass_policy_reject": 1, "model_reject_policy_pass": 1, "both_reject": 1}
    assert agree["excluded"] == {"judge_unparsed": 2} and agree["policy_reasons_when_model_passed"] == {"stub_reject": 1}
    assert agree["by_class"] == {"defect": {k: agree[k] for k in ("denominator", "both_pass", "model_pass_policy_reject",
                                                                 "model_reject_policy_pass", "both_reject")}}


def test_원문_판정과_엄격_파싱한_판정은_정책_결과와_따로_남는다(tmp_path):
    run_dir, config = run_all(tmp_path, [image_task(0)], {"img-000": GOOD_DESC + " 거름"})
    audit_run(run_dir, config, tmp_path / "audit", policy=StubPolicy())
    st = _status(tmp_path / "audit")["img-000"]
    assert st["status"] == "rejected" and st["reasons"] == ["policy:stub_reject"]
    assert st["judge_verdict"] == json.loads(JUDGE_OK)                   # 정책이 기각해도 판정기의 판정은 그대로
    assert st["judge_text_sha256"] == hashlib.sha256(JUDGE_OK.encode("utf-8")).hexdigest()
    assert st["judge_flags"] == [] and st["judge_answers"] is None      # 옛 네 키 판정 — 구조화 답이 없다


@pytest.mark.parametrize("stub,code", [
    (SimpleNamespace(policy_identity=lambda: {"v": 1}, deterministic_reasons=lambda t, x: []), "policy_interface"),
    (StubPolicy(identity={}), "policy_identity"),
    (StubPolicy(identity=["not", "a", "mapping"]), "policy_identity"),
])
def test_정책의_꼴이_틀리면_집계하지_않는다(tmp_path, stub, code):
    run_dir, config = run_all(tmp_path, [image_task(0)], {"img-000": GOOD_DESC})
    with pytest.raises(AuditRefused, match=f"^{code}:"):
        audit_run(run_dir, config, tmp_path / "audit", policy=stub)
    assert not (tmp_path / "audit").exists()


# --------------------------------------------------------------------------- 명부 계약 · 분할 · 부족량


@pytest.mark.parametrize("over,reason", [
    ({"image_sha256": "XYZ"}, "task_image_sha256_invalid"),
    ({"train_user_text": None}, "task_train_user_text_missing"),
])
def test_명부_계약_위반은_오류(tmp_path, over, reason):
    run_dir, config = run_all(tmp_path, [image_task(0, **over)], {"img-000": GOOD_DESC})
    _, _, out = audit(tmp_path, run_dir, config)
    st = next(iter_jsonl(out / "rows_status.jsonl"))
    assert st["status"] == "error" and reason in st["reasons"]


def test_학습_질문이_이미지마다_다르면_오류(tmp_path):
    tasks = [image_task(0), image_task(1, train_user_text=USER_TEXT + " 기공 하나")]
    run_dir, config = run_all(tmp_path, tasks, {"img-000": GOOD_DESC, "img-001": GOOD_DESC + " 둘째"})
    _, a, _ = audit(tmp_path, run_dir, config)
    assert "train_user_text_varies" in a["counts"]["run_errors"]


def test_계보가_train_과_val_에_걸치면_오류(tmp_path):
    """infer 는 같은 kind 안의 교차만 막는다. kind 가 달라도 같은 계보면 한 분할이어야 한다."""
    tasks = [image_task(0, split="val", lineage_group_id="LG-A"), qa_task(0)]
    outputs = {"img-000": GOOD_DESC, "qa-000": qa_json("기공은 무엇인가?", "가스가 남긴 공동이다.")}
    run_dir, config = run_all(tmp_path, tasks, outputs)
    _, a, _ = audit(tmp_path, run_dir, config)
    assert a["lineage_crossing_split"] == ["LG-A"] and "lineage_crosses_split" in a["counts"]["run_errors"]
    assert "errors" in a["blockers"]


def test_원문_QA_부족량을_숫자_그대로_적는다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs,
                              _config(minimum_source_qa_train=10, target_source_qa_train=20,
                                      minimum_distinct_train_passages=3334))
    result, a, _ = audit(tmp_path, run_dir, config)
    sq = a["source_qa"]
    assert sq["accepted"]["train"] == 2 and sq["distinct_train_passages"] == 2
    assert sq["short_of_minimum"] == 8 and sq["short_of_target"] == 18 and sq["distinct_passages_short"] == 3332
    assert "source_qa_below_minimum" in a["blockers"] and result["status"] == "blocked"


def test_구절당_질문_상한을_넘는_계획은_오류(tmp_path):
    tasks = [qa_task(i) for i in range(4)]
    outputs = {f"qa-{i:03d}": qa_json(f"기공 질문 {i}?", f"가스가 남긴 공동이다 {i}.") for i in range(4)}
    run_dir, config = run_all(tmp_path, tasks, outputs)
    _, a, _ = audit(tmp_path, run_dir, config)
    assert "questions_per_passage_planned_over_limit" in a["counts"]["run_errors"]
    assert a["source_qa"]["passages_planned_over_limit"] == ["p-001"]


def test_커버리지는_참여자_결함별_계획_대_채택(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    _, a, _ = audit(tmp_path, run_dir, config)
    cov = a["coverage"]["image_report"]
    assert cov["C1"]["2011"] == {"planned": 1, "accepted": 1}
    assert cov["C2"]["none"] == {"planned": 1, "accepted": 1}
    assert a["coverage"]["source_qa"]["LG-A"] == {"planned": 2, "accepted": 2}


# --------------------------------------------------------------------------- 공유 계약(릴리스가 읽는 칸)


def test_AUDIT_는_릴리스가_읽는_칸을_가진다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    _, a, out = audit(tmp_path, run_dir, config)
    run = read_json(run_dir / "RUN.json")
    assert a["complete"] is True and a["errors"] == 0 and a["rejected"] == 0 and a["incomplete"] == 0
    assert (out / a["run_dir"]).resolve() == run_dir.resolve()
    assert a["run_identity_sha256"] == run["identity_sha256"] and a["tasks_sha256"] == run["tasks_sha256"]
    assert a["config_sha256"] == run["config_sha256"] and a["config"] == run["config"]
    assert a["planned"] == {"count": 4, "ids_sha256": ids_sha256([t["id"] for t in tasks])}
    assert a["counts"]["rows"] == 4 and a["counts"]["image_report"] == {"train": 2, "val": 0}
    assert a["counts"]["source_qa"] == {"train": 2, "val": 0}
    assert a["user_templates"] == {"image_report": [USER_TEXT]}
    assert {d["source_id"] for d in a["source_documents"]} == {"DOC-p-001", "DOC-p-002"}


def test_데이터셋_행은_출처와_생성문을_그대로_가진다(tmp_path):
    desc = "주석에 기공 결함 하나가 표시되어 있다.\n두께 정보가 없어 합부는 판단하지 않는다."
    run_dir, config = run_all(tmp_path, [image_task(0)], {"img-000": desc})
    _, _, out = audit(tmp_path, run_dir, config)
    row = next(iter_jsonl(out / "dataset.jsonl"))
    assert row["messages"][1]["content"][0]["text"] == desc  # 줄바꿈까지 그대로 — 릴리스가 샤드와 대조한다
    gen = I.read_stage_rows(run_dir, "generate", config)[0]
    p = row["provenance"]
    assert p["generate"]["backend"] == "transformers" and p["generate"]["family"] == "qwen"
    assert p["generate"]["shard_sha256"] == gen["_shard_sha256"]
    assert p["generate"]["text_sha256"] == hashlib.sha256(gen["text"].encode("utf-8")).hexdigest()
    assert p["judge"]["family"] == "phi" and p["judge"]["verdict"] == json.loads(JUDGE_OK)


def test_원문_QA_행의_source_ids_는_작업_행_그대로다(tmp_path):
    run_dir, config = run_all(tmp_path, [qa_task(0)], {"qa-000": qa_json("기공은 무엇인가?", "가스가 남긴 공동이다.")})
    _, _, out = audit(tmp_path, run_dir, config)
    row = next(iter_jsonl(out / "dataset.jsonl"))
    assert row["source_ids"] == ["p-001"] and "image_sha256" not in row and row["client"] == "public"


@pytest.mark.parametrize("text", ["이 영상의 결함 3곳을 설명하라.", "이 영상을 {설명}하라.", "이 영상의 기공을 설명하라."])
def test_학습_질문_틀에_숫자_괄호_결함_이름이_있으면_오류(tmp_path, text):
    run_dir, config = run_all(tmp_path, [image_task(0, train_user_text=text)], {"img-000": GOOD_DESC})
    _, _, out = audit(tmp_path, run_dir, config)
    st = next(iter_jsonl(out / "rows_status.jsonl"))
    assert st["status"] == "error" and "task_train_user_text_leaks_target" in st["reasons"]


def test_원문_QA_작업에_문서_지문이_없으면_오류(tmp_path):
    t = qa_task(0)
    t["evidence"] = {"passages": t["evidence"]["passages"]}
    run_dir, config = run_all(tmp_path, [t], {"qa-000": qa_json("기공은 무엇인가?", "가스가 남긴 공동이다.")})
    _, _, out = audit(tmp_path, run_dir, config)
    assert next(iter_jsonl(out / "rows_status.jsonl"))["reasons"] == ["task_source_document_missing"]


def test_지문의_숫자_조각은_추적_원천이_아니다(tmp_path):
    t = qa_task(0)
    t["evidence"]["source_sha256"] = "7" + "a" * 63
    run_dir, config = run_all(tmp_path, [t], {"qa-000": qa_json("기공은 언제 생기는가?", "7 가지 조건에서 생긴다.")})
    _, _, out = audit(tmp_path, run_dir, config)
    assert "untraceable_number" in next(iter_jsonl(out / "rows_status.jsonl"))["reasons"]


# --------------------------------------------------------------------------- 출력 · 설정


def test_출력_디렉터리가_비어_있지_않으면_거부(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    out = tmp_path / "audit"
    out.mkdir()
    (out / "old.txt").write_text("x", encoding="utf-8")
    with pytest.raises(AuditRefused):
        audit_run(run_dir, config, out)


def test_설정에_품질_칸이_없으면_거부(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    bad = {k: v for k, v in config.items() if k != "quality"}
    with pytest.raises(AuditRefused):
        audit_run(run_dir, bad, tmp_path / "audit")


def test_명부를_바꾸면_샤드_검사에서_멈춘다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    (run_dir / "tasks.jsonl").write_bytes((run_dir / "tasks.jsonl").read_bytes() + b"\n")
    with pytest.raises(I.StageRefused):
        audit_run(run_dir, config, tmp_path / "audit")


def test_감사_결과는_결정적이다(tmp_path):
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    _, a1, o1 = audit(tmp_path, run_dir, config, "a1")
    _, a2, o2 = audit(tmp_path, run_dir, config, "a2")
    assert (o1 / "AUDIT.json").read_bytes() == (o2 / "AUDIT.json").read_bytes()
    assert a1["dataset_sha256"] == a2["dataset_sha256"]


# --------------------------------------------------------------------------- 구조화 판정의 판독(parse_judge_v5)


NORMAL_TASK = image_task(0, target=locked())
DEFECT_TASK = image_task(1)
QA_TASK = qa_task(0)
N_ANSWERS = {"claims_defect": False, "asserts_verdict": False, "states_no_marked_defect": True,
             "has_limitation": True, "language_ok": True, "reason": "주석과 같다"}
D_ANSWERS = {"fields_match": True, "has_limitation": True, "asserts_verdict_dimension_or_cause": False,
             "language_ok": True, "reason": "주석과 같다"}


def _row(text, finish="eos"):
    return {"text": text, "finish_reason": finish, "input_tokens": 10, "output_tokens": 10}


@pytest.mark.parametrize("task,change,expected,flags", [
    # 정상 행 — grounded = 표시 결함 없음을 말함 ∧ 결함을 주장하지 않음, scope_ok = 합부 · 안전을 단정하지 않음
    (NORMAL_TASK, {}, (True, True, True), []),
    (NORMAL_TASK, {"has_limitation": False}, (True, True, True), ["n5_absent"]),
    (NORMAL_TASK, {"claims_defect": True}, (False, True, True), []),
    (NORMAL_TASK, {"states_no_marked_defect": False}, (False, True, True), []),
    (NORMAL_TASK, {"asserts_verdict": True, "has_limitation": False}, (True, False, True), ["n5_absent"]),
    (NORMAL_TASK, {"language_ok": False}, (True, True, False), []),
    # 결함 행 — grounded = 필드 일치, scope_ok = 제한 설명 있음 ∧ 합부 · 치수 · 원인을 단정하지 않음
    (DEFECT_TASK, {}, (True, True, True), []),
    (DEFECT_TASK, {"fields_match": False}, (False, True, True), []),
    (DEFECT_TASK, {"has_limitation": False}, (True, False, True), []),
    (DEFECT_TASK, {"asserts_verdict_dimension_or_cause": True}, (True, False, True), []),
    (DEFECT_TASK, {"has_limitation": False, "asserts_verdict_dimension_or_cause": True, "language_ok": False},
     (True, False, False), []),
])
def test_구조화_답에서_세_판정값을_만든다(task, change, expected, flags):
    answers = {**(N_ANSWERS if task is NORMAL_TASK else D_ANSWERS), **change}
    out = parse_judge_v5(task, _row(v5(**answers)))
    assert out["errors"] == [] and out["answers"] == answers and out["flags"] == flags
    verdict = out["verdict"]
    assert (verdict["grounded"], verdict["scope_ok"], verdict["language_ok"]) == expected
    assert verdict["reason"] == answers["reason"]
    assert all(type(verdict[k]) is bool for k in ("grounded", "scope_ok", "language_ok"))
    assert out["rejected"] == [f"judge_{k}_false" for k, ok in zip(("grounded", "scope_ok", "language_ok"), expected)
                               if not ok]


def test_원문_QA_는_네_키_그대로다():
    out = parse_judge_v5(QA_TASK, _row(JUDGE_OK))
    assert out["row_class"] == "source_qa" and out["errors"] == [] and out["flags"] == []
    assert out["verdict"] == json.loads(JUDGE_OK) == out["answers"]
    no = parse_judge_v5(QA_TASK, _row(JUDGE_OK.replace('"scope_ok":true', '"scope_ok":false')))
    assert no["rejected"] == ["judge_scope_ok_false"]


def _dup(answers):
    """같은 키를 두 번 쓴 JSON — 중복을 하나로 접어 읽으면 키 집합이 맞아 보인다."""
    return v5(**answers)[:-1] + ',"reason":"다른 답"}'


@pytest.mark.parametrize("task,text,finish,code", [
    (NORMAL_TASK, v5(**N_ANSWERS), "length", "judge_truncated"),
    (NORMAL_TASK, v5(**N_ANSWERS)[:-7], "eos", "judge_unparsable"),                        # 잘린 JSON
    (NORMAL_TASK, _dup(N_ANSWERS), "eos", "judge_unparsable"),                             # 중복 키
    (NORMAL_TASK, v5(**N_ANSWERS).replace('"language_ok":true', '"language_ok":NaN'), "eos", "judge_unparsable"),
    # 벗기지 않는 울타리 — 중첩 · 앞뒤 글 · 둘째 울타리 · 닫는 울타리가 제 줄에 없음 · 다른 언어 표시 · 여는 줄의 글 · 물결
    (NORMAL_TASK, "```json\n```json\n" + v5(**N_ANSWERS) + "\n```\n```", "eos", "judge_unparsable"),
    (NORMAL_TASK, "판정:\n```json\n" + v5(**N_ANSWERS) + "\n```", "eos", "judge_unparsable"),
    (NORMAL_TASK, "```json\n" + v5(**N_ANSWERS) + "\n```\n끝", "eos", "judge_unparsable"),
    (NORMAL_TASK, "```json\n" + v5(**N_ANSWERS) + "\n```\n```json\n" + v5(**N_ANSWERS) + "\n```", "eos",
     "judge_unparsable"),
    (NORMAL_TASK, "```json\n" + v5(**N_ANSWERS) + "```", "eos", "judge_unparsable"),
    (NORMAL_TASK, "```python\n" + v5(**N_ANSWERS) + "\n```", "eos", "judge_unparsable"),
    (NORMAL_TASK, "```json " + v5(**N_ANSWERS) + "\n```", "eos", "judge_unparsable"),
    (NORMAL_TASK, "~~~json\n" + v5(**N_ANSWERS) + "\n~~~", "eos", "judge_unparsable"),
    # 울타리를 벗겨도 안의 잘림 · 중복 키 · 여분 키는 그대로 오류
    (NORMAL_TASK, "```json\n" + v5(**N_ANSWERS)[:-7] + "\n```", "eos", "judge_unparsable"),
    (NORMAL_TASK, "```json\n" + _dup(N_ANSWERS) + "\n```", "eos", "judge_unparsable"),
    (NORMAL_TASK, "```json\n" + v5(**N_ANSWERS, extra=True) + "\n```", "eos", "judge_keys"),
    (NORMAL_TASK, v5(**N_ANSWERS) + " 끝", "eos", "judge_unparsable"),                      # 덧붙인 글
    (NORMAL_TASK, "판정: " + v5(**N_ANSWERS), "eos", "judge_unparsable"),
    (NORMAL_TASK, "[]", "eos", "judge_not_object"),
    (NORMAL_TASK, '"참"', "eos", "judge_not_object"),
    (NORMAL_TASK, v5(**N_ANSWERS, extra=True), "eos", "judge_keys"),
    (NORMAL_TASK, v5(**{k: v for k, v in N_ANSWERS.items() if k != "has_limitation"}), "eos", "judge_keys"),
    (NORMAL_TASK, v5(**D_ANSWERS), "eos", "judge_keys"),                                   # 결함 행의 키
    (DEFECT_TASK, v5(**N_ANSWERS), "eos", "judge_keys"),                                   # 정상 행의 키
    (NORMAL_TASK, JUDGE_OK, "eos", "judge_keys"),                                          # 옛 네 키로 대신 읽지 않는다
    (DEFECT_TASK, JUDGE_OK, "eos", "judge_keys"),
    (QA_TASK, v5(**N_ANSWERS), "eos", "judge_keys"),
    (NORMAL_TASK, v5(**{**N_ANSWERS, "claims_defect": 0}), "eos", "judge_value_type"),     # 0 · 1 은 참 · 거짓이 아니다
    (DEFECT_TASK, v5(**{**D_ANSWERS, "fields_match": 1}), "eos", "judge_value_type"),
    # 참거짓 칸의 글은 "true" · "false" 만 읽는다 — 문장(첫 GPU 탐침에서 본 꼴) · 앞뒤 공백 · 다른 말 · 전각은 오류
    (DEFECT_TASK, v5(**{**D_ANSWERS, "has_limitation": "합격·안전·적합·양호를 단정할 수 없음"}), "eos",
     "judge_value_type"),
    (NORMAL_TASK, v5(**{**N_ANSWERS, "asserts_verdict": "용접의 합격·안전·적합·양호를 단정하는가?"}), "eos",
     "judge_value_type"),
    (NORMAL_TASK, v5(**{**N_ANSWERS, "asserts_verdict": "합격"}), "eos", "judge_value_type"),
    (DEFECT_TASK, v5(**{**D_ANSWERS, "has_limitation": " true"}), "eos", "judge_value_type"),
    (DEFECT_TASK, v5(**{**D_ANSWERS, "has_limitation": "yes"}), "eos", "judge_value_type"),
    (DEFECT_TASK, v5(**{**D_ANSWERS, "has_limitation": "참"}), "eos", "judge_value_type"),
    (DEFECT_TASK, v5(**{**D_ANSWERS, "has_limitation": "ｔｒｕｅ"}), "eos", "judge_value_type"),
    (DEFECT_TASK, v5(**{**D_ANSWERS, "has_limitation": "1"}), "eos", "judge_value_type"),
    (DEFECT_TASK, v5(**{**D_ANSWERS, "language_ok": None}), "eos", "judge_value_type"),
    (QA_TASK, JUDGE_OK.replace('"grounded":true', '"grounded":"yes"'), "eos", "judge_value_type"),
    (NORMAL_TASK, v5(**{**N_ANSWERS, "reason": ""}), "eos", "judge_reason_missing"),
    (DEFECT_TASK, v5(**{**D_ANSWERS, "reason": "  "}), "eos", "judge_reason_missing"),
    (DEFECT_TASK, v5(**{**D_ANSWERS, "reason": 3}), "eos", "judge_reason_missing"),
])
def test_구조화_판정을_엄격하게_읽는다(task, text, finish, code):
    out = parse_judge_v5(task, _row(text, finish))
    assert out["errors"] == [code]
    assert out["verdict"] is None and out["answers"] is None and out["rejected"] == [] and out["flags"] == []


def _fence(text, head="json", nl="\n"):
    return "```" + head + nl + text + nl + "```"


def _worded(text, *pairs):
    """참거짓 칸의 값을 글로 — (키, 글) 쌍마다."""
    for k, word in pairs:
        for b in ("true", "false"):
            text = text.replace(f'"{k}":{b}', f'"{k}":"{word if word is not None else b}"')
    return text


@pytest.mark.parametrize("task,answers,text,normalized", [
    # ① 출력 전체를 감싼 울타리 하나 — json 표시는 있거나 없거나, 대소문자 무관, 앞뒤 공백 · CRLF 는 괜찮다
    (NORMAL_TASK, N_ANSWERS, _fence(v5(**N_ANSWERS)), ["code_fence"]),
    (DEFECT_TASK, D_ANSWERS, _fence(v5(**D_ANSWERS), head=""), ["code_fence"]),
    (DEFECT_TASK, D_ANSWERS, "\n  " + _fence(v5(**D_ANSWERS), head="JSON") + "  \n", ["code_fence"]),
    (DEFECT_TASK, D_ANSWERS, _fence(v5(**D_ANSWERS), nl="\r\n"), ["code_fence"]),
    (QA_TASK, json.loads(JUDGE_OK), _fence(JUDGE_OK), ["code_fence"]),
    # ② 물음 답의 "true" · "false" 글(대소문자 무관)
    (DEFECT_TASK, D_ANSWERS, _worded(v5(**D_ANSWERS), ("has_limitation", None)), ["string_boolean"]),
    (NORMAL_TASK, {**N_ANSWERS, "asserts_verdict": False},
     _worded(v5(**N_ANSWERS), ("asserts_verdict", "FALSE"), ("claims_defect", "False")), ["string_boolean"]),
    (QA_TASK, json.loads(JUDGE_OK), _worded(JUDGE_OK, ("grounded", "True")), ["string_boolean"]),
    # 둘 다 — 표지는 이 순서
    (DEFECT_TASK, {**D_ANSWERS, "fields_match": True},
     _fence(_worded(v5(**D_ANSWERS), ("fields_match", "TRUE"))), ["code_fence", "string_boolean"]),
    # reason 의 글은 참거짓으로 읽지 않는다
    (NORMAL_TASK, {**N_ANSWERS, "reason": "true"}, v5(**{**N_ANSWERS, "reason": "true"}), []),
])
def test_뜻이_같은_형식만_정규화하고_표지를_남긴다(task, answers, text, normalized):
    out = parse_judge_v5(task, _row(text))
    assert out["errors"] == [] and out["format_normalized"] == normalized
    assert out["answers"] == answers and all(type(out["answers"][k]) is bool for k in answers if k != "reason")
    plain = parse_judge_v5(task, _row(canonical_bytes(answers).decode("utf-8")))
    assert plain["format_normalized"] == []
    # 정규화한 답은 처음부터 바른 꼴로 낸 답과 같은 판정값 · 기각 · 표지를 낸다
    assert (out["verdict"], out["rejected"], out["flags"]) == (plain["verdict"], plain["rejected"], plain["flags"])
    assert out["parser"] == "parse_judge_v5_1" and out["schema"] == "judge_structured_v5"


def test_정규화한_뒤의_오류에도_적용한_정규화는_남는다():
    """울타리를 벗긴 뒤 문장 값이 남으면 judge_value_type — 무엇을 벗겼는지는 진단으로 남는다(판정은 오류 그대로)."""
    text = _fence(v5(**{**N_ANSWERS, "asserts_verdict": "합격·안전·적합·양호를 단정할 수 없음"}))
    out = parse_judge_v5(NORMAL_TASK, _row(text))
    assert out["errors"] == ["judge_value_type"] and out["format_normalized"] == ["code_fence"]
    assert out["verdict"] is None and out["answers"] is None
    both = parse_judge_v5(DEFECT_TASK, _row(_fence(_worded(v5(**{**D_ANSWERS, "language_ok": 1}),
                                                          ("fields_match", "true")))))
    assert both["errors"] == ["judge_value_type"] and both["format_normalized"] == ["code_fence", "string_boolean"]
    assert Q.JUDGE_V5_FORMAT_NORMALIZATIONS == ("code_fence", "string_boolean")


@pytest.mark.parametrize("task,row,code", [
    ({"kind": "image_report", "locked_target": None}, _row(v5(**N_ANSWERS)), "judge_task_unsupported"),
    ({"kind": "image_report", "locked_target": {"defects": "없음"}}, _row(v5(**N_ANSWERS)), "judge_task_unsupported"),
    ({"kind": "unknown", "locked_target": {"defects": []}}, _row(v5(**N_ANSWERS)), "judge_task_unsupported"),
    ("image_report", _row(v5(**N_ANSWERS)), "judge_task_unsupported"),
    (NORMAL_TASK, None, "judge_row_invalid"),
    (NORMAL_TASK, {"text": None, "finish_reason": "eos"}, "judge_row_invalid"),
    (NORMAL_TASK, [v5(**N_ANSWERS)], "judge_row_invalid"),
])
def test_행_종류를_작업_행으로_못_고르거나_행이_없으면_오류다(task, row, code):
    out = parse_judge_v5(task, row)
    assert out["errors"] == [code] and out["verdict"] is None


def test_판독_결과의_꼴은_탐침이_받는_계약이고_입력을_바꾸지_않는다():
    task, row = json.loads(json.dumps(NORMAL_TASK)), _row(v5(**N_ANSWERS))
    before = canonical_bytes([task, row])
    out = parse_judge_v5(task, row)
    assert canonical_bytes([task, row]) == before
    assert set(out) == {"schema", "parser", "row_class", "errors", "rejected", "verdict", "flags", "answers",
                        "format_normalized"}
    assert out["schema"] == Q.JUDGE_V5_SCHEMA_VERSION == "judge_structured_v5"
    assert out["parser"] == Q.JUDGE_V5_PARSER_VERSION == "parse_judge_v5_1" and out["format_normalized"] == []
    assert set(out["verdict"]) == {"grounded", "scope_ok", "language_ok", "reason"}


@pytest.mark.parametrize("task", [NORMAL_TASK, DEFECT_TASK, QA_TASK], ids=["defect_free", "defect", "source_qa"])
def test_행_종류와_키는_판정_지시의_렌더러와_같다(task):
    """파서는 판정 정책을 불러오지 않는다(신원 폐쇄 밖) — 같은 규칙인지는 여기서 대조한다. 지금 렌더러는 v6 다 —
    v5 의 키 · 판은 아래 v5 회귀 시험이 따로 본다."""
    cls = Q._judge_row_class(task)
    assert cls == (P.row_class(task) or "source_qa")
    rendered = json.loads(P.render_judge(task, {"text": "글", "finish_reason": "eos"}))
    if cls == "source_qa":
        assert "expected_output_keys" not in rendered and Q.JUDGE_V6_KEYS[cls] == Q.JUDGE_KEYS
    else:
        assert rendered["row_class"] == cls and rendered["output_schema"] == Q.JUDGE_V6_SCHEMA_VERSION
        assert sorted(rendered["expected_output_keys"]) == list(Q.JUDGE_V6_KEYS[cls]) == sorted(P.OUTPUT_KEYS[cls])
        assert len(Q.JUDGE_V6_KEYS[cls]) == {"defect_free": 6, "defect": 7}[cls]
        # 정식 입력 계약 — 렌더러가 근거의 이름별 개수를 그대로 투영한다(사상표 이름)
        assert rendered["locked_target"]["defect_counts_by_name"] == task["evidence"]["defect_counts_by_name"]
        assert set(task["evidence"]["defect_counts_by_name"]) <= {v.name_ko for v in LABELS.values()}


def test_v5_키와_판은_지난_판_그대로_남는다():
    """v5 회귀 — v5 원문 · 영수증을 읽는 파서의 키 · 판. 지금 렌더러(v6)와 대조하지 않는다."""
    assert Q.JUDGE_V5_SCHEMA_VERSION == "judge_structured_v5" and Q.JUDGE_V5_PARSER_VERSION == "parse_judge_v5_1"
    assert Q.JUDGE_V5_KEYS == {
        "defect_free": ("asserts_verdict", "claims_defect", "has_limitation", "language_ok", "reason",
                        "states_no_marked_defect"),
        "defect": ("asserts_verdict_dimension_or_cause", "fields_match", "has_limitation", "language_ok", "reason"),
        "source_qa": ("grounded", "language_ok", "reason", "scope_ok")}
    assert Q.JUDGE_V5_KEYS["defect_free"] == Q.JUDGE_V6_KEYS["defect_free"]
    assert Q.JUDGE_V5_KEYS["defect"] != Q.JUDGE_V6_KEYS["defect"]


def test_구조화_파서만_판정_신원의_선택_폐쇄에_묶인다():
    """판정 신원은 파서와 그 의존의 폐쇄만 묶는다 — 집계 코드가 폐쇄에 들어오면 무관한 집계 수정도 판정 신원을 깬다."""
    report = F.require_v5_parser()
    assert report["capabilities"] == {"legacy_four_keys": True, "judge_structured_v5": True}
    mine = {s.split(":", 1)[1] for s in report["symbols"] if s.startswith(Q.__name__ + ":")}
    assert mine == {"JUDGE_KEYS", "JUDGE_V5_KEYS", "JUDGE_V5_SCHEMA_VERSION", "JUDGE_V5_PARSER_VERSION",
                    "JUDGE_V5_FORMAT_NORMALIZATIONS", "JUDGE_V5_BOOL_STRINGS", "_judge", "_judge_row_class",
                    "_unfence_judge_text", "parse_json", "parse_judge_v5"}


def test_v6_파서의_폐쇄는_v5_파서를_담지_않고_정책이_그것을_묶는다():
    report = F.require_v6_parser()
    assert report["selected_schema"] == "judge_structured_v6" and report["capabilities"] == {"judge_structured_v6": True}
    mine = {s.split(":", 1)[1] for s in report["symbols"] if s.startswith(Q.__name__ + ":")}
    assert mine == {"JUDGE_V6_KEYS", "JUDGE_V6_SCHEMA_VERSION", "JUDGE_V6_PARSER_VERSION", "JUDGE_V6_FORMAT_NORMALIZATIONS",
                    "JUDGE_V6_BOOL_STRINGS", "JUDGE_V6_FENCE_OPEN", "JUDGE_V6_FENCE_CLOSE", "_judge_row_class",
                    "_unfence_judge_text_v6", "parse_json", "parse_judge_v6"}
    assert P.policy_identity()["parser_implementation"] == report and P.model_parser_ready() is True
    assert J.require_v6_parser() is parse_judge_v6
    assert Q.JUDGE_PARSER_VERSIONS == {"judge_structured_v5": "parse_judge_v5_1", "judge_structured_v6": "parse_judge_v6",
                                      "judge_structured_v7": "parse_judge_v7"}


# --------------------------------------------------------------------------- 판정 탐침의 진입(가짜 판정기 — GPU 아님)


class ProbeFake:
    """탐침용 가짜 판정기 — 판정 입력의 행 종류에 맞는 통과 답을 낸다. `extra` 면 키를 하나 더 붙인다."""

    extra = False

    def __init__(self, cfg, stage):
        self.identity = {"backend": "fake", "model_id": cfg["model_id"], "revision": cfg["revision"],
                         "family": cfg["family"], "model_type": cfg["family"], "quantization": cfg["quantization"],
                         "tokenizer_sha256": "c" * 64}

    def generate_batch(self, prompts, system, deadline):
        rows = []
        for p in prompts:
            answers = json.loads(V6_OK[json.loads(p)["row_class"]])
            if ProbeFake.extra:
                answers["extra"] = True
            rows.append(_row(v5(**answers)))
        return rows

    def close(self):
        pass


@pytest.fixture
def probe_world(tmp_path, monkeypatch):
    """합성 원본 넷(50 · 50 · 25 · 25)과 그 지문 — 실제 사례 파일은 읽지 않는다. 탐침의 저장소 게이트 검사는 격리한다."""
    repo = tmp_path / "probe-repository"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(K, "_repository_root", lambda: repo)
    monkeypatch.setattr(ProbeFake, "extra", False)
    names = sorted(v.name_ko for v in label_map.load_label_map(label_map.DEFAULT_LABEL_MAP).defect_types.values())[:2]
    sets = {"N+": "judge_nplus_g1", "D+": "judge_dplus_g1", "ADV1": "judge_adv_negative_g1",
            "ADV2": "judge_adv_negative_g2"}
    paths, pins = {}, {}
    for group, count in K.V5_GROUP_COUNTS.items():
        normal = group in {"N+", "ADV1"}
        fields = {"defect_counts_by_name": {} if normal else {n: 1 for n in names},
                  "defect_positions": {} if normal else {n: ["가운데"] for n in names}}
        original = {"set": sets[group], "row_kind": "defect_free" if normal else "defect",
                    "cases": [{"id": f"unit-{group}-{i}", "text": f"합성 후보 {group} {i}",
                               "expected": "accept" if group in {"N+", "D+"} else "reject", "fields": fields}
                              for i in range(count)]}
        raw = json.dumps(original, ensure_ascii=False).encode("utf-8")
        path = tmp_path / f"{group.replace('+', 'plus')}-synthetic.json"
        path.write_bytes(raw)
        paths[group], pins[group] = path, hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(K, "V5_SOURCE_SHA256", pins)
    monkeypatch.setattr(K, "V6_SOURCE_SHA256", dict(pins))
    cases = tmp_path / "v6-cases.json"
    cases.write_bytes(canonical_bytes(C.build_v6_cases(paths, expected_source_sha256=pins)))
    config = {"generation": {**_spec("qwen"), "batch_size": 50}, "judge": {**_spec("phi"), "batch_size": 50}}
    return cases, config, tmp_path


def test_판정_탐침이_이_파서로_진입한다(probe_world):
    """`parser_missing` 없이 들어가 행마다 이 파서의 판독(세 판정값 · 모델 표지)을 싣는다. 판별력 측정이 아니다."""
    cases, config, tmp = probe_world
    out = K.run_probe(cases, config, tmp / "probe", backend_factory=ProbeFake)
    assert out["status"] == "complete" and out["exit_code"] == 0
    assert out["summary"]["protocol"] == K.V6_PROTOCOL and out["summary"]["completed_cases"] == 150
    assert out["summary"]["parser_error_rows"] == 0 and out["summary"]["truncated_rows"] == 0
    tasks = {c["id"]: c["task"] for c in json.loads(cases.read_bytes())["cases"]}
    rows = list(iter_jsonl(tmp / "probe" / "outputs.jsonl"))
    assert len(rows) == 150
    for row in rows:
        mine = parse_judge_v6(tasks[row["id"]], row["raw"])
        assert row["model_errors"] == [] and row["model_accept"] is True and row["format_normalized"] == []
        assert row["derived_verdict"] == mine["verdict"] and row["model_flags"] == mine["flags"]
    basis = read_json(tmp / "probe" / "PROBE.json")["identity_basis"]["input_basis"]
    assert basis["policy_identity"]["parser_implementation"] == F.require_v6_parser()
    assert basis["parser_schema_version"] == "judge_structured_v6"


def test_판정_탐침에서도_키가_하나_더_있으면_판독_오류다(probe_world):
    cases, config, tmp = probe_world
    ProbeFake.extra = True
    out = K.run_probe(cases, config, tmp / "probe", backend_factory=ProbeFake)
    assert out["summary"]["parser_error_rows"] == 150 and out["summary"]["thresholds_passed"] is False
    assert all(r["model_errors"] == ["judge_keys"] and r["derived_verdict"] is None
               for r in iter_jsonl(tmp / "probe" / "outputs.jsonl"))


# --------------------------------------------------------------------------- 구조화 판정 v6 의 판독(parse_judge_v6)


D6_ANSWERS = {"names_match": True, "counts_match": True, "positions_match": True, "has_limitation": True,
              "asserts_verdict_dimension_or_cause": False, "language_ok": True, "reason": "주석과 같다"}
NORMALIZED_FORMS = ([], ["code_fence"], ["string_boolean"], ["code_fence", "string_boolean"])


@pytest.mark.parametrize("task,answers,expected,flags", [
    # 결함 행 — grounded = 이름 ∧ 개수 ∧ 위치, scope_ok = 제한 설명 있음 ∧ 합부 · 치수 · 원인을 단정하지 않음
    (DEFECT_TASK, {}, (True, True, True), []),
    (DEFECT_TASK, {"names_match": False}, (False, True, True), []),
    (DEFECT_TASK, {"counts_match": False}, (False, True, True), []),
    (DEFECT_TASK, {"positions_match": False}, (False, True, True), []),
    (DEFECT_TASK, {"names_match": False, "counts_match": False, "positions_match": False}, (False, True, True), []),
    (DEFECT_TASK, {"has_limitation": False}, (True, False, True), []),
    (DEFECT_TASK, {"asserts_verdict_dimension_or_cause": True}, (True, False, True), []),
    (DEFECT_TASK, {"has_limitation": False, "asserts_verdict_dimension_or_cause": True}, (True, False, True), []),
    (DEFECT_TASK, {"positions_match": False, "has_limitation": False, "language_ok": False}, (False, False, False), []),
    # 정상 행 — v5 와 같다
    (NORMAL_TASK, {}, (True, True, True), []),
    (NORMAL_TASK, {"has_limitation": False}, (True, True, True), ["n5_absent"]),
    (NORMAL_TASK, {"claims_defect": True}, (False, True, True), []),
    (NORMAL_TASK, {"states_no_marked_defect": False}, (False, True, True), []),
    (NORMAL_TASK, {"asserts_verdict": True}, (True, False, True), []),
    # 원문 QA — 네 키 그대로
    (QA_TASK, {"scope_ok": False}, (True, False, True), []),
])
def test_v6_결함_행은_일곱_키이고_grounded_는_이름_개수_위치의_곱이다(task, answers, expected, flags):
    base = {DEFECT_TASK["id"]: D6_ANSWERS, NORMAL_TASK["id"]: N_ANSWERS}.get(task["id"], json.loads(JUDGE_OK))
    answers = {**base, **answers}
    out = parse_judge_v6(task, _row(v5(**answers)))
    assert out["errors"] == [] and out["answers"] == answers and out["flags"] == flags
    grounded, scope_ok, language_ok = expected
    assert out["verdict"] == {"grounded": grounded, "scope_ok": scope_ok, "language_ok": language_ok,
                              "reason": answers["reason"]}
    assert out["rejected"] == [f"judge_{k}_false" for k, v in zip(("grounded", "scope_ok", "language_ok"), expected)
                               if v is not True]
    assert len(Q.JUDGE_V6_KEYS["defect"]) == 7 and "fields_match" not in Q.JUDGE_V6_KEYS["defect"]


@pytest.mark.parametrize("task,answers,text,normalized", [
    # ① 출력 전체가 소문자 json 울타리 하나 — 앞뒤의 JSON 공백은 괜찮다
    (DEFECT_TASK, D6_ANSWERS, _fence(v5(**D6_ANSWERS)), ["code_fence"]),
    (NORMAL_TASK, N_ANSWERS, "\n " + _fence(v5(**N_ANSWERS)) + "\n\t", ["code_fence"]),
    (QA_TASK, json.loads(JUDGE_OK), _fence(JUDGE_OK), ["code_fence"]),
    (DEFECT_TASK, D6_ANSWERS, _fence(json.dumps(D6_ANSWERS, ensure_ascii=False, indent=1)), ["code_fence"]),
    # ② 정확히 "true" · "false" 글 — 하나 · 여럿
    (DEFECT_TASK, D6_ANSWERS, _worded(v5(**D6_ANSWERS), ("positions_match", None)), ["string_boolean"]),
    (NORMAL_TASK, N_ANSWERS, _worded(v5(**N_ANSWERS), *((k, None) for k in N_ANSWERS if k != "reason")),
     ["string_boolean"]),
    # 둘 다 — 표지는 이 순서
    (DEFECT_TASK, D6_ANSWERS, _fence(_worded(v5(**D6_ANSWERS), ("names_match", None), ("asserts_verdict_dimension_or_cause", None))),
     ["code_fence", "string_boolean"]),
    # reason 의 글은 참거짓으로 읽지 않는다
    (DEFECT_TASK, {**D6_ANSWERS, "reason": "true"}, v5(**{**D6_ANSWERS, "reason": "true"}), []),
])
def test_v6_는_좁은_정규화_둘만_하고_표지를_순서대로_남긴다(task, answers, text, normalized):
    out = parse_judge_v6(task, _row(text))
    assert out["errors"] == [] and out["format_normalized"] == normalized and normalized in NORMALIZED_FORMS
    assert out["answers"] == answers and all(type(out["answers"][k]) is bool for k in answers if k != "reason")
    plain = parse_judge_v6(task, _row(canonical_bytes(answers).decode("utf-8")))
    assert plain["format_normalized"] == []
    # 정규화한 답은 처음부터 바른 꼴로 낸 답과 같은 판정값 · 기각 · 표지를 낸다
    assert (out["verdict"], out["rejected"], out["flags"]) == (plain["verdict"], plain["rejected"], plain["flags"])
    assert out["parser"] == "parse_judge_v6" and out["schema"] == "judge_structured_v6"


D6_TEXT = v5(**D6_ANSWERS)


@pytest.mark.parametrize("text,finish,code", [
    # v5_1 이 받던 넓은 울타리 — v6 는 벗기지 않는다
    (_fence(D6_TEXT, head=""), "eos", "judge_unparsable"),
    (_fence(D6_TEXT, head="JSON"), "eos", "judge_unparsable"),
    (_fence(D6_TEXT, head="Json"), "eos", "judge_unparsable"),
    (_fence(D6_TEXT, head="json "), "eos", "judge_unparsable"),
    (_fence(D6_TEXT, nl="\r\n"), "eos", "judge_unparsable"),
    # 울타리 하나가 출력 전체가 아니다
    ("설명\n" + _fence(D6_TEXT), "eos", "judge_unparsable"),
    (_fence(D6_TEXT) + "\n끝", "eos", "judge_unparsable"),
    (_fence(D6_TEXT) + "\n" + _fence("{}"), "eos", "judge_unparsable"),
    (_fence(_fence(D6_TEXT)), "eos", "judge_unparsable"),
    ("```json\n" + D6_TEXT + "```", "eos", "judge_unparsable"),
    ("```json" + D6_TEXT + "\n```", "eos", "judge_unparsable"),
    ("~~~json\n" + D6_TEXT + "\n~~~", "eos", "judge_unparsable"),
    ("```python\n" + D6_TEXT + "\n```", "eos", "judge_unparsable"),
    ("```json\n```", "eos", "judge_unparsable"),
    (D6_TEXT + " {}", "eos", "judge_unparsable"),
    # 울타리를 벗긴 뒤의 오류는 그대로
    (_fence(D6_TEXT[:-5]), "eos", "judge_unparsable"),
    (_fence('{"names_match":false,' + D6_TEXT[1:]), "eos", "judge_unparsable"),
    (_fence(v5(**{**D6_ANSWERS, "extra": False})), "eos", "judge_keys"),
    (_fence(D6_TEXT), "length", "judge_truncated"),
    # 키 — v5 의 fields_match 꼴 · 빠진 키 · 여분 키
    (v5(**D_ANSWERS), "eos", "judge_keys"),
    (v5(**{k: v for k, v in D6_ANSWERS.items() if k != "positions_match"}), "eos", "judge_keys"),
    (v5(**{**D6_ANSWERS, "fields_match": True}), "eos", "judge_keys"),
    # 참거짓 칸의 값 — 정확한 "true" · "false" 밖은 뜻을 짐작하지 않는다
    *[(v5(**{**D6_ANSWERS, "counts_match": value}), "eos", "judge_value_type")
      for value in ("True", "TRUE", "False", " true", "true ", "yes", "참", "ｔｒｕｅ", "1", 1, 0, None,
                    "이름마다 글이 말한 개수의 합이 잠긴 개수와 같은가?")],
    (v5(**{**D6_ANSWERS, "reason": " "}), "eos", "judge_reason_missing"),
    (D6_TEXT.replace("true", "NaN", 1), "eos", "judge_unparsable"),
])
def test_v6_는_넓은_형식과_다른_꼴을_오류로_둔다(text, finish, code):
    out = parse_judge_v6(DEFECT_TASK, _row(text, finish))
    assert out["errors"] == [code]
    assert out["verdict"] is None and out["answers"] is None and out["rejected"] == [] and out["flags"] == []
    assert out["format_normalized"] in NORMALIZED_FORMS


def test_v6_정규화한_뒤의_오류에도_적용한_정규화는_남는다():
    text = _fence(v5(**{**N_ANSWERS, "asserts_verdict": "합격·안전·적합·양호를 단정할 수 없음"}))
    out = parse_judge_v6(NORMAL_TASK, _row(text))
    assert out["errors"] == ["judge_value_type"] and out["format_normalized"] == ["code_fence"]
    both = parse_judge_v6(DEFECT_TASK, _row(_fence(_worded(v5(**{**D6_ANSWERS, "language_ok": "True"}),
                                                          ("names_match", None)))))
    assert both["errors"] == ["judge_value_type"] and both["format_normalized"] == ["code_fence", "string_boolean"]
    assert Q.JUDGE_V6_FORMAT_NORMALIZATIONS == ("code_fence", "string_boolean")


def test_v6_와_v5_는_서로의_출력을_대신_읽지_않는다():
    assert parse_judge_v5(DEFECT_TASK, _row(D6_TEXT))["errors"] == ["judge_keys"]
    assert parse_judge_v6(DEFECT_TASK, _row(v5(**D_ANSWERS)))["errors"] == ["judge_keys"]
    # 정상 행 · 원문 QA 는 키가 같다 — 그래도 표지(판 · 파서)는 각자의 것
    for task, text in ((NORMAL_TASK, v5(**N_ANSWERS)), (QA_TASK, JUDGE_OK)):
        old, new = parse_judge_v5(task, _row(text)), parse_judge_v6(task, _row(text))
        assert old["verdict"] == new["verdict"] and old["errors"] == new["errors"] == []
        assert (old["schema"], old["parser"]) == ("judge_structured_v5", "parse_judge_v5_1")
        assert (new["schema"], new["parser"]) == ("judge_structured_v6", "parse_judge_v6")
    # 집계의 읽기도 판정 출처의 꼴 하나로만
    assert Q._read_judge(DEFECT_TASK, _row(D6_TEXT), "judge_structured_v6")[0] == []
    assert Q._read_judge(DEFECT_TASK, _row(D6_TEXT), "judge_structured_v5")[0] == ["judge_keys"]
    assert Q._read_judge(DEFECT_TASK, _row(D6_TEXT), "legacy_four_keys")[0] == ["judge_unparsable"]


def test_v6_판독_결과의_꼴은_탐침이_받는_계약이고_입력을_바꾸지_않는다():
    task, row = json.loads(json.dumps(DEFECT_TASK)), _row(_fence(_worded(D6_TEXT, ("names_match", None))))
    before = canonical_bytes([task, row])
    out = parse_judge_v6(task, row)
    assert canonical_bytes([task, row]) == before
    assert set(out) == {"schema", "parser", "row_class", "errors", "rejected", "verdict", "flags", "answers",
                        "format_normalized"}
    assert out["schema"] == Q.JUDGE_V6_SCHEMA_VERSION == P.OUTPUT_SCHEMA_VERSION == "judge_structured_v6"
    assert out["parser"] == Q.JUDGE_V6_PARSER_VERSION == "parse_judge_v6"
    assert type(out["format_normalized"]) is list and out["format_normalized"] == ["code_fence", "string_boolean"]
    assert set(out["verdict"]) == {"grounded", "scope_ok", "language_ok", "reason"}
    for bad in (None, "image_report", {"kind": "image_report", "locked_target": None}):
        assert parse_judge_v6(bad, row)["errors"] == ["judge_task_unsupported"]
    for bad in (None, [D6_TEXT], {"text": None, "finish_reason": "eos"}):
        assert parse_judge_v6(task, bad)["errors"] == ["judge_row_invalid"]


@pytest.mark.parametrize("kind", ["defect_free", "defect"])
def test_v6_정책의_형식_예시는_그대로_읽히고_키가_같다(kind):
    """판정 지시의 한 줄 예시(정책)가 이 파서로 정규화 없이 읽힌다 — 파서는 정책을 불러오지 않으므로 여기서 대조한다."""
    task = NORMAL_TASK if kind == "defect_free" else DEFECT_TASK
    example = P.FORMAT_EXAMPLES[kind]
    out = parse_judge_v6(task, _row(canonical_bytes(example).decode("utf-8")))
    assert out["errors"] == [] and out["answers"] == example and out["format_normalized"] == []
    assert sorted(example) == list(Q.JUDGE_V6_KEYS[kind]) == sorted(P.OUTPUT_KEYS[kind])


# --------------------------------------------------------------------------- 구조화 판정 v7(결함 행 다섯 키 · 정상 행은 판정 모델 없음)


#: v7 결함 행의 통과 답 — 물음 넷은 모두 "참 = 문제 없음"
D7_ANSWERS = {"has_limitation": True, "language_ok": True, "no_unannotated_attributes": True,
              "no_unsupported_relation": True, "reason": "주석과 같다"}
D7_TEXT = v5(**D7_ANSWERS)
D7_QUESTIONS = ("has_limitation", "language_ok", "no_unannotated_attributes", "no_unsupported_relation")


@pytest.mark.parametrize("task,answers,rejected", [
    (DEFECT_TASK, {}, []),
    *[(DEFECT_TASK, {k: False}, [f"judge_{k}_false"]) for k in D7_QUESTIONS],
    (DEFECT_TASK, dict.fromkeys(D7_QUESTIONS, False), [f"judge_{k}_false" for k in D7_QUESTIONS]),
    (DEFECT_TASK, {"no_unsupported_relation": False, "has_limitation": False},
     ["judge_has_limitation_false", "judge_no_unsupported_relation_false"]),
    # 원문 QA — 옛 네 키 그대로
    (QA_TASK, {}, []),
    (QA_TASK, {"scope_ok": False, "grounded": False}, ["judge_grounded_false", "judge_scope_ok_false"]),
])
def test_v7_결함_행은_다섯_키이고_물음_넷은_참이_문제_없음이다(task, answers, rejected):
    defect = task["id"] == DEFECT_TASK["id"]
    answers = {**(D7_ANSWERS if defect else json.loads(JUDGE_OK)), **answers}
    out = parse_judge_v7(task, _row(v5(**answers)))
    assert out["errors"] == [] and out["answers"] == answers and out["flags"] == [] and out["format_normalized"] == []
    assert out["rejected"] == rejected
    if defect:
        # 판정값은 물음 넷 그대로다 — grounded · scope_ok 를 만들지 않는다(grounded 는 결정적 정책이 가른다)
        assert out["verdict"] == answers and list(out["verdict"]) == [*D7_QUESTIONS, "reason"]
    else:
        assert out["verdict"] == {k: answers[k] for k in ("grounded", "scope_ok", "language_ok", "reason")}
    assert Q.JUDGE_V7_KEYS == {"defect": (*D7_QUESTIONS, "reason"),
                               "source_qa": ("grounded", "language_ok", "reason", "scope_ok")}


@pytest.mark.parametrize("row", [
    _row(D7_TEXT), _row(D7_TEXT, "json_closed"), _row(v5(**N_ANSWERS)), _row(_fence(D7_TEXT)), _row(D7_TEXT, "length"),
    _row(""), {"text": None}, None,
])
def test_v7_정상_행은_판정_모델의_행을_받지_않는다(row):
    out = parse_judge_v7(NORMAL_TASK, row)
    assert out["errors"] == ["judge_not_applicable"] and out["row_class"] == "defect_free"
    assert out["verdict"] is None and out["answers"] is None and out["rejected"] == [] and out["flags"] == []
    assert out["format_normalized"] == []
    assert "defect_free" not in Q.JUDGE_V7_KEYS


@pytest.mark.parametrize("reason,flags", [
    ("가" * 40, []),
    ("가" * 41, ["reason_long"]),
    ("a" * 40, []),
    ("a" * 41, ["reason_long"]),
    ("가", []),
    # 받은 글 그대로의 길이 — 앞뒤 공백도 센다
    (" " + "가" * 38 + " ", []),
    (" " + "가" * 39 + " ", ["reason_long"]),
])
def test_v7_reason_이_40_자를_넘으면_표지만_남기고_기각하지_않는다(reason, flags):
    for task, base in ((DEFECT_TASK, D7_ANSWERS), (QA_TASK, json.loads(JUDGE_OK))):
        out = parse_judge_v7(task, _row(v5(**{**base, "reason": reason})))
        assert out["errors"] == [] and out["rejected"] == [] and out["flags"] == flags
        assert out["verdict"]["reason"] == reason
    # 기각과 표지는 따로 선다
    out = parse_judge_v7(DEFECT_TASK, _row(v5(**{**D7_ANSWERS, "language_ok": False, "reason": reason})))
    assert out["rejected"] == ["judge_language_ok_false"] and out["flags"] == flags
    assert Q.JUDGE_V7_REASON_MAX_CHARS == 40


@pytest.mark.parametrize("finish,errors", [
    ("eos", []), ("json_closed", []),
    ("length", ["judge_truncated"]), (None, ["judge_truncated"]), ("stop", ["judge_truncated"]),
    ("JSON_CLOSED", ["judge_truncated"]), ("json_closed ", ["judge_truncated"]),
])
def test_v7_종료는_eos_와_json_closed_만_끝난_것으로_받는다(finish, errors):
    assert parse_judge_v7(DEFECT_TASK, _row(D7_TEXT, finish))["errors"] == errors
    assert Q.JUDGE_V7_FINISH_REASONS == ("eos", "json_closed")
    # v6 는 그대로 eos 만
    assert parse_judge_v6(DEFECT_TASK, _row(D6_TEXT, "json_closed"))["errors"] == ["judge_truncated"]


@pytest.mark.parametrize("text,normalized", [
    (_fence(D7_TEXT), ["code_fence"]),
    ("\n " + _fence(D7_TEXT) + "\n\t", ["code_fence"]),
    (_fence(json.dumps(D7_ANSWERS, ensure_ascii=False, indent=1)), ["code_fence"]),
    (_worded(D7_TEXT, ("no_unsupported_relation", None)), ["string_boolean"]),
    (_worded(D7_TEXT, *((k, None) for k in D7_QUESTIONS)), ["string_boolean"]),
    (_fence(_worded(D7_TEXT, ("has_limitation", None), ("language_ok", None))), ["code_fence", "string_boolean"]),
])
def test_v7_는_v6_의_좁은_정규화_둘을_그대로_한다(text, normalized):
    for finish in ("eos", "json_closed"):
        out = parse_judge_v7(DEFECT_TASK, _row(text, finish))
        assert out["errors"] == [] and out["format_normalized"] == normalized and normalized in NORMALIZED_FORMS
        assert out["answers"] == out["verdict"] == D7_ANSWERS and out["rejected"] == [] and out["flags"] == []


def test_v7_울타리_규칙은_v6_와_같은_글을_같게_벗긴다():
    variants = [_fence(D7_TEXT), " \n" + _fence(D7_TEXT) + "\r\n", _fence(D7_TEXT, head=""), _fence(D7_TEXT, head="JSON"),
                _fence(D7_TEXT, head="json "), _fence(D7_TEXT, nl="\r\n"), "설명\n" + _fence(D7_TEXT),
                _fence(D7_TEXT) + "\n끝", _fence(_fence(D7_TEXT)), "```json\n" + D7_TEXT + "```",
                "```json" + D7_TEXT + "\n```", "~~~json\n" + D7_TEXT + "\n~~~", "```json\n```", "```json\n" + D7_TEXT,
                D7_TEXT]
    for text in variants:
        assert Q._unfence_judge_text_v7(text) == Q._unfence_judge_text_v6(text), text
    assert (Q.JUDGE_V7_FENCE_OPEN, Q.JUDGE_V7_FENCE_CLOSE, Q.JUDGE_V7_BOOL_STRINGS, Q.JUDGE_V7_FORMAT_NORMALIZATIONS) == (
        Q.JUDGE_V6_FENCE_OPEN, Q.JUDGE_V6_FENCE_CLOSE, Q.JUDGE_V6_BOOL_STRINGS, Q.JUDGE_V6_FORMAT_NORMALIZATIONS)


@pytest.mark.parametrize("text,finish,code", [
    # 잘림
    (D7_TEXT[:-5], "eos", "judge_unparsable"),
    (D7_TEXT[:-1], "json_closed", "judge_unparsable"),
    (D7_TEXT, "length", "judge_truncated"),
    # 중복 키 · NaN · 덧붙인 글 · 객체 아님
    ('{"has_limitation":false,' + D7_TEXT[1:], "eos", "judge_unparsable"),
    (D7_TEXT.replace("true", "NaN", 1), "eos", "judge_unparsable"),
    (D7_TEXT + " {}", "eos", "judge_unparsable"),
    (D7_TEXT + "\n끝", "json_closed", "judge_unparsable"),
    ("[" + D7_TEXT + "]", "eos", "judge_not_object"),
    # 여분 · 빠진 키 · 다른 판 · 다른 행 종류의 키
    (v5(**{**D7_ANSWERS, "extra": True}), "eos", "judge_keys"),
    *[(v5(**{k: v for k, v in D7_ANSWERS.items() if k != gone}), "eos", "judge_keys") for gone in D7_ANSWERS],
    (v5(**{**D7_ANSWERS, "states_no_marked_defect": True}), "eos", "judge_keys"),
    (D6_TEXT, "eos", "judge_keys"),
    (v5(**N_ANSWERS), "eos", "judge_keys"),
    (JUDGE_OK, "eos", "judge_keys"),
    # 참거짓 칸의 값 — 정확한 "true" · "false" 밖은 뜻을 짐작하지 않는다
    *[(v5(**{**D7_ANSWERS, "no_unannotated_attributes": value}), "eos", "judge_value_type")
      for value in ("True", "FALSE", " true", "yes", "참", "1", 1, 0, None, [], {})],
    # reason
    (v5(**{**D7_ANSWERS, "reason": " "}), "eos", "judge_reason_missing"),
    (v5(**{**D7_ANSWERS, "reason": ""}), "eos", "judge_reason_missing"),
    (v5(**{**D7_ANSWERS, "reason": 40}), "eos", "judge_reason_missing"),
    (v5(**{**D7_ANSWERS, "reason": None}), "eos", "judge_reason_missing"),
])
def test_v7_는_잘림_중복_여분_빠진_키_NaN_을_오류로_둔다(text, finish, code):
    out = parse_judge_v7(DEFECT_TASK, _row(text, finish))
    assert out["errors"] == [code]
    assert out["verdict"] is None and out["answers"] is None and out["rejected"] == [] and out["flags"] == []


def test_v7_닫는_울타리가_없는_글은_종료와_상관없이_못_읽음이다():
    """좁은 정규화는 출력 전체를 감싼 울타리만 벗긴다 — 여는 울타리만 있거나 닫는 울타리만 있는 글은 종료(`eos` ·
    `json_closed`)와 상관없이 못 읽음이다. 정규화를 넓히지 않는다.

    정지기는 앞에 공백만 둔 `{` 로 시작하는 글에서만 첫 객체의 닫힘으로 멈춘다 — 울타리로 시작한 출력은 `json_closed` 가 되지
    않고, 닫는 울타리를 쓴 뒤의 `eos`(벗겨 읽는다)나 출력 상한(`judge_truncated`)으로 끝난다."""
    assert S.first_json_object_close(D7_TEXT).closed is True
    assert S.first_json_object_close("```json\n" + D7_TEXT).eligible is False
    ok = parse_judge_v7(DEFECT_TASK, _row(D7_TEXT + "\n", "json_closed"))
    assert ok["errors"] == [] and ok["format_normalized"] == []
    fenced = parse_judge_v7(DEFECT_TASK, _row(_fence(D7_TEXT), "eos"))
    assert fenced["errors"] == [] and fenced["format_normalized"] == ["code_fence"]
    for finish in ("eos", "json_closed"):
        for text in ("```json\n" + D7_TEXT, "```json\n" + D7_TEXT + "\n", D7_TEXT + "`", D7_TEXT + "\n```"):
            out = parse_judge_v7(DEFECT_TASK, _row(text, finish))
            assert out["errors"] == ["judge_unparsable"] and out["format_normalized"] == []


def test_v7_와_v5_v6_는_서로의_출력을_대신_읽지_않는다():
    assert parse_judge_v6(DEFECT_TASK, _row(D7_TEXT))["errors"] == ["judge_keys"]
    assert parse_judge_v5(DEFECT_TASK, _row(D7_TEXT))["errors"] == ["judge_keys"]
    assert parse_judge_v7(DEFECT_TASK, _row(D6_TEXT))["errors"] == ["judge_keys"]
    # 정상 행 — v5 · v6 는 그대로 읽는다(v7 만 받지 않는다)
    normal = _row(v5(**N_ANSWERS))
    assert parse_judge_v5(NORMAL_TASK, normal)["errors"] == parse_judge_v6(NORMAL_TASK, normal)["errors"] == []
    # 원문 QA 는 키가 같다 — 판정값 · 기각은 같고 표지(판 · 파서)는 각자의 것
    old, new = parse_judge_v6(QA_TASK, _row(JUDGE_OK)), parse_judge_v7(QA_TASK, _row(JUDGE_OK))
    assert new["errors"] == [] and (old["verdict"], old["rejected"]) == (new["verdict"], new["rejected"])
    assert (new["schema"], new["parser"]) == ("judge_structured_v7", "parse_judge_v7")
    # 집계의 읽기 — 꼴 → 파서를 하나씩 짝짓는다. v7 은 v7 파서로만, 짝이 없는 꼴은 다른 판의 파서로 읽지 않고 거부한다
    got, want = Q._read_judge(DEFECT_TASK, _row(D7_TEXT), "judge_structured_v7"), parse_judge_v7(DEFECT_TASK, _row(D7_TEXT))
    assert got == (want["errors"], want["rejected"], want["verdict"], want["flags"], want["answers"], want["format_normalized"])
    assert Q._read_judge(DEFECT_TASK, _row(D6_TEXT), "judge_structured_v7")[0] == ["judge_keys"]
    assert Q._read_judge(NORMAL_TASK, _row(v5(**N_ANSWERS)), "judge_structured_v7")[0] == ["judge_not_applicable"]
    for schema in ("judge_structured_v9", None):
        with pytest.raises(AuditRefused, match="^judge_schema_unsupported:"):
            Q._read_judge(DEFECT_TASK, _row(D7_TEXT), schema)


def test_v7_판독_결과의_꼴과_판_이름():
    task = json.loads(json.dumps(DEFECT_TASK))
    row = _row(_fence(_worded(D7_TEXT, ("language_ok", None))), "json_closed")
    before = canonical_bytes([task, row])
    out = parse_judge_v7(task, row)
    assert canonical_bytes([task, row]) == before
    assert set(out) == {"schema", "parser", "row_class", "errors", "rejected", "verdict", "flags", "answers",
                        "format_normalized"}
    assert out["schema"] == Q.JUDGE_V7_SCHEMA_VERSION == "judge_structured_v7"
    assert out["parser"] == Q.JUDGE_V7_PARSER_VERSION == "parse_judge_v7"
    assert out["row_class"] == "defect" and out["format_normalized"] == ["code_fence", "string_boolean"]
    for bad in (None, "image_report", {"kind": "image_report", "locked_target": None}):
        assert parse_judge_v7(bad, row)["errors"] == ["judge_task_unsupported"]
    for bad in (None, [D7_TEXT], {"text": None, "finish_reason": "eos"}, {"text": b"{}", "finish_reason": "eos"}):
        assert parse_judge_v7(task, bad)["errors"] == ["judge_row_invalid"]
    # 집계가 읽는 꼴의 표 — v7 은 v7 파서의 판으로 든다
    assert Q.JUDGE_PARSER_VERSIONS == {"judge_structured_v5": "parse_judge_v5_1", "judge_structured_v6": "parse_judge_v6",
                                      "judge_structured_v7": "parse_judge_v7"}


def test_v7_파서의_폐쇄는_v5_v6_의_것에_닿지_않는다():
    """판정 정책 쪽의 v7 지문이 생기기 전 — 지문기의 폐쇄 걷기로 v7 의 선택 폐쇄를 보고, v5 · v6 의 파서 · 울타리 · 상수가
    들지 않음과 상수가 글자 그대로의 값임(걷기가 거부하지 않음)을 본다."""
    from corpus.production import common

    report = F._Closure((Q, common)).result([(common, "parse_json"), (common, "canonical_bytes"),
                                             (Q, "JUDGE_V7_SCHEMA_VERSION"), (Q, "parse_judge_v7")])
    mine = {s.split(":", 1)[1] for s in report["symbols"] if s.startswith(Q.__name__ + ":")}
    assert mine == {"JUDGE_V7_SCHEMA_VERSION", "JUDGE_V7_KEYS", "JUDGE_V7_PARSER_VERSION", "JUDGE_V7_FORMAT_NORMALIZATIONS",
                    "JUDGE_V7_FENCE_OPEN", "JUDGE_V7_FENCE_CLOSE", "JUDGE_V7_BOOL_STRINGS", "JUDGE_V7_FINISH_REASONS",
                    "JUDGE_V7_REASON_MAX_CHARS", "_judge_row_class", "_unfence_judge_text_v7", "parse_json",
                    "parse_judge_v7"}


def test_v7_출력_계약을_묶고도_출처가_v7_이라_적지_않으면_집계하지_않는다(tmp_path):
    """v7 은 정상 행에 판정 행이 없고 결정적 영수증으로 가르는 계약이다 — 영수증 칸이 없는(v7 이라 적지 않은) 출처에
    v7 출력 계약만 묶어 오면 정상 행을 판정 없음으로 두거나 결함 행을 다른 판의 파서로 읽게 되므로 거부한다."""
    tasks, outputs = good_case()
    run_dir, config = run_all(tmp_path, tasks, outputs)
    src = _separate_judge(tmp_path, run_dir, config)
    other = {**src["identity_basis"]["policy_identity"], "output_schema": "judge_structured_v7"}
    src["identity_basis"] = {**src["identity_basis"], "policy_identity": other}
    _rehash(src)
    with pytest.raises(AuditRefused, match="^judge_schema_mixed:"):
        audit_run(run_dir, config, tmp_path / "audit", judge_source=src, policy=StubPolicy(identity=other))
    assert not (tmp_path / "audit" / "AUDIT.json").exists()


# --------------------------------------------------------------------------- v7 판정 출처의 소비(정상 행은 결정적 영수증만)
#
# 실제 판정 신원 쓰기 · 읽기(`infer.run_judge(policy_version="v7")` · `infer.read_judge_source`)와 판정 신원에 묶인 실제 v7
# 정책으로 집계까지 잇는다 — 가짜는 판정 모델 백엔드(`V7Judge`)뿐이다. 다시 읽기를 건너뛰는 `_v7_reread` 는 위조 시험에만 쓴다
# (읽기가 먼저 막는 변조를 지나 집계 자신의 대조까지 닿게).


#: v7 생성 계약의 통과 글 — 정상 행은 정해진 두 문장, 결함 행은 이름 · 수 · 구역 문장 뒤에 정해진 제한 문장
N7_DESC = f"{G.NORMAL_ABSENCE_FORMS[0]} {G.NORMAL_LIMITATION_FORMS[0]}"
D7_DESC = f"{LABELS['porosity'].name_ko} 1개가 영상 가운데에 표시되어 있다. {G.LIMITATION_FORMS[0]}"
#: v7 정책의 결정적 검사 묶음(영수증의 `deterministic_codes` 칸)
V7_GROUPS = ("attribute", "contract", "grounded", "negative", "normal")
#: v7 판정 모델 설정 — 기본 판정과 같고 출력 토큰만 160(정지기 계약)
V7_JUDGE = {**_spec("phi"), "model_id": "local/judge-v7", "max_new_tokens": 160}


class V7Judge(Fake):
    """v7 판정 신원 쓰기에 넣는 가짜 판정 모델. 정지기(`enable_json_object_stopping`)를 켜야 답하고 종료는 기본 `json_closed`.
    결함 행 입력(행 종류 · 잠긴 칸 · 후보)에는 원 요청이 없어 후보 글로 행을 찾는다. 정상 행 입력이 오면 실패한다."""

    calls: ClassVar[list] = []

    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)
        self.stopping = False

    def enable_json_object_stopping(self):
        assert self.stage == "judge"
        self.stopping = True

    def generate_batch(self, prompts, system, deadline):
        assert self.stage == "judge" and self.stopping, "v7 판정은 정지기를 켠 모델에만 간다"
        by_text = {text: tid for tid, text in Fake.gen.items()}
        rows = []
        for p in prompts:
            request = json.loads(p)
            assert request.get("row_class") != "defect_free", "정상 행이 판정 모델에 갔다"
            qa = "original_request" in request
            tid = Fake._by_prompt[request["original_request"]] if qa else by_text[request["candidate"]]
            V7Judge.calls.append(tid)
            rows.append({"text": Fake.judge.get(tid, JUDGE_OK if qa else D7_TEXT),
                         "finish_reason": Fake.finish.get(("judge", tid), "json_closed"), "input_tokens": 10, "output_tokens": 10})
        return rows


class V7Policy:
    """정책 자리에 거는 시험용 가짜 — 지문만 낸다. 판정 신원에 묶인 실제 v7 정책과 지문이 다르거나 꼴이 다른 정책이 판단 전에
    거부되는지만 본다(판단 `evaluate_v7` 에 닿으면 시험 실패)."""

    def __init__(self, identity):
        self.identity = identity

    def policy_identity_v7(self):
        return copy.deepcopy(self.identity)

    def evaluate_v7(self, task, text, *, generation_contract=True):
        raise AssertionError("지문이 다른 정책의 판단에 닿았다")


def _sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def v7_case():
    """결함 · 정상 · 정상 · 원문 QA — 샤드 둘(크기 2): [결함, 정상] [정상, 원문 QA]. 판정을 시작하는 첫 샤드에 모델 대상이 있다
    (정상 행만인 첫 샤드는 승인된 실행 조건 없이 판정 모델을 시작하지 않는다)."""
    tasks = [image_task(0), image_task(1, client="C2", target=locked()), image_task(2, client="C3", target=locked()),
             qa_task(0)]
    outputs = {"img-000": D7_DESC, "img-001": N7_DESC, "img-002": N7_DESC,
               "qa-000": qa_json("기공은 무엇이 남긴 공동인가?", "용접금속 안에 갇힌 가스가 남긴 공동이다.")}
    return tasks, outputs


def _v7_run(tmp_path, outputs=None, *, judge=None, finish=None, max_shards=None):
    """생성(가짜) → 판정 신원 쓰기 v7(가짜 판정 모델) → 판정 신원 읽기(꼴을 주지 않는다 — 판정 폴더로 스스로 고른다)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    tasks, base = v7_case()
    run_dir, config = run_all(tmp_path, tasks, {**base, **(outputs or {})}, stages=("generate",))
    Fake.judge, Fake.finish = dict(judge or {}), {("judge", k): v for k, v in (finish or {}).items()}
    Fake.backend_by_stage, V7Judge.calls = {"judge": "fake"}, []
    done = I.run_judge(run_dir, config, tmp_path / "judge-v7", judge_config=V7_JUDGE, policy_version="v7",
                       backend_factory=V7Judge, max_shards=max_shards)
    assert done["policy_version"] == "v7" and done["exit_code"] == (0 if max_shards is None else 4), done
    return run_dir, config, I.read_judge_source(run_dir, config, tmp_path / "judge-v7")


def _v7_audit(tmp_path, outputs=None, **judge):
    run_dir, config, src = _v7_run(tmp_path, outputs, **judge)
    out = tmp_path / "audit"
    result = audit_run(run_dir, config, out, judge_source=src)
    return result, read_json(out / "AUDIT.json"), _status(out), {r["id"]: r for r in iter_jsonl(out / "dataset.jsonl")}, src


def test_v7_정상_행은_판정_모델_없이_결정적_영수증으로_수락된다(tmp_path):
    result, a, st, rows, src = _v7_audit(tmp_path)
    # 판정 신원 읽기가 v7 을 골랐다 — 영수증은 네 행 모두, 판정 모델은 결함 행 · 원문 QA 만 불렀다
    assert src["policy_version"] == "v7" and src["all_task_ids"] == ["img-000", "img-001", "img-002", "qa-000"]
    assert src["model_task_ids"] == [r["id"] for r in src["rows"]] == V7Judge.calls == ["img-000", "qa-000"]
    # 막는 사유는 시험용 판정 백엔드 하나뿐 — 정상 행 둘은 판정 행 없이 수락됐다(judge_missing 이 아니다)
    assert result["status"] == "blocked" and a["blockers"] == ["non_transformers_backend"], a["blockers"]
    assert a["counts"]["accepted"] == 4 and a["incomplete"] == 0 and a["complete"] is True
    assert a["audit_version"] == "quality_v9" and a["judge_parser_version"] == "parse_judge_v7"
    assert a["judge_identity"]["parser_schema"] == "judge_structured_v7" and a["stages"]["judge"]["rows"] == 2
    policy_sha = _digest(src["identity_basis"]["policy_identity"])
    assert a["policy"]["identity_sha256"] == policy_sha == _digest(P.policy_identity_v7())
    for rid in ("img-001", "img-002"):
        assert rows[rid]["provenance"]["judge"] == {"applicable": False, "identity_sha256": src["identity_sha256"]}
        assert rows[rid]["provenance"]["policy"] == {"identity_sha256": policy_sha, "flags": []}
        s = st[rid]
        assert s["judge_applicable"] is False and s["judge_text_sha256"] is None and s["judge_verdict"] is None
        assert s["deterministic_pass"] is True and s["deterministic_reasons"] == [] and s["policy_flags"] == []
    judge = rows["img-000"]["provenance"]["judge"]
    assert judge["verdict"] == D7_ANSWERS and judge["schema"] == "judge_structured_v7" and judge["parser"] == "parse_judge_v7"
    assert judge["backend"] == "fake" and judge["identity_sha256"] == src["identity_sha256"] and "applicable" not in judge
    assert rows["qa-000"]["provenance"]["judge"]["verdict"] == json.loads(JUDGE_OK)
    assert st["img-000"]["judge_applicable"] is True is st["qa-000"]["judge_applicable"]
    det = a["deterministic_receipts"]
    assert det["rows"] == det["planned"] == len(src["deterministic_receipts"]) == 4
    assert det["policy_identity_sha256"] == policy_sha
    assert det["model_task_ids"] == {"count": 2, "ids_sha256": ids_sha256(["img-000", "qa-000"])}
    assert det["normal"] == {"planned": 2, "receipts": 2, "deterministic_pass": 2, "deterministic_reject": 0,
                             "status": {"accepted": 2}, "reasons": {}, "flags": {}, "codes": {g: {} for g in V7_GROUPS}}
    assert det["model"]["planned"] == det["model"]["receipts"] == 2 and det["model"]["status"] == {"accepted": 2}
    agree = a["policy"]["model_agreement"]
    assert agree["denominator"] == 1 and agree["both_pass"] == 1 and agree["excluded"] == {"judge_not_applicable": 2}


def test_v7_정상_행의_결정적_기각은_사유_코드_분포로_남는다(tmp_path):
    _, a, st, rows, _ = _v7_audit(tmp_path, {"img-001": "표시된 결함이 없다. 합격이다.",
                                                  "img-002": "표시된 결함이 크고 둥근 모양이다."})
    assert V7Judge.calls == ["img-000", "qa-000"]                 # 기각될 정상 행도 판정 모델에 가지 않는다
    assert st["img-001"]["status"] == "rejected" and st["img-001"]["reasons"] == [
        "policy:normal_limitation_absent", "policy:normal_not_contract_form", "policy:normal_verdict_implied",
        "policy:verdict_asserted"]
    assert st["img-002"]["reasons"] == ["policy:normal_limitation_absent", "policy:normal_not_contract_form"]
    assert st["img-002"]["deterministic_pass"] is False and st["img-002"]["policy_flags"] == ["n5_lexical_absent"]
    assert "img-001" not in rows and "img-002" not in rows and "rejected_rows" in a["blockers"]
    normal = a["deterministic_receipts"]["normal"]
    assert normal["deterministic_reject"] == 2 and normal["status"] == {"rejected": 2}
    assert normal["reasons"] == {"normal_limitation_absent": 2, "normal_not_contract_form": 2, "normal_verdict_implied": 1,
                                 "verdict_asserted": 1}
    assert normal["codes"] == {"attribute": {}, "contract": {"normal_not_contract_form": 2}, "grounded": {},
                               "negative": {"verdict_asserted": 1},
                               "normal": {"normal_limitation_absent": 2, "normal_verdict_implied": 1}}
    assert normal["flags"] == {"n5_lexical_absent": 2}
    assert a["counts"]["reasons"]["policy:verdict_asserted"] == 1


def test_v7_결함_행은_결정적_통과와_물음_넷이_모두_참이어야_수락된다(tmp_path):
    """판정 모델이 통과로 답해도 결정적 기각이면 기각, 결정적 통과여도 물음 하나가 거짓이면 기각."""
    name = LABELS["porosity"].name_ko
    verdict = f"{name} 1개가 영상 가운데에 표시되어 있다. 합격이다. {G.LIMITATION_FORMS[0]}"
    _, a, st, _, _ = _v7_audit(tmp_path / "x", {"img-000": verdict})
    assert st["img-000"]["reasons"] == ["policy:verdict_asserted"] and V7Judge.calls == ["img-000", "qa-000"]
    assert a["policy"]["model_agreement"]["model_pass_policy_reject"] == 1
    false_one = v5(**{**D7_ANSWERS, "no_unsupported_relation": False})
    _, a, st, _, _ = _v7_audit(tmp_path / "y", judge={"img-000": false_one})
    assert st["img-000"]["reasons"] == ["judge_no_unsupported_relation_false"]
    assert a["policy"]["model_agreement"]["model_reject_policy_pass"] == 1


def test_v7_reason_long_은_표지이고_json_closed_와_eos_종료를_받는다(tmp_path):
    long = v5(**{**D7_ANSWERS, "reason": "가" * 41})
    _, a, st, rows, src = _v7_audit(tmp_path, judge={"img-000": long}, finish={"qa-000": "eos"})
    assert {r["id"]: r["finish_reason"] for r in src["rows"]} == {"img-000": "json_closed", "qa-000": "eos"}
    assert st["img-000"]["status"] == "accepted" == st["qa-000"]["status"] and st["img-000"]["judge_flags"] == ["reason_long"]
    assert rows["img-000"]["provenance"]["judge"]["flags"] == ["reason_long"]
    assert a["counts"]["judge_model_flags"] == {"reason_long": 1}


def test_v7_판정_전_샤드의_모델_대상_행만_judge_missing_이다(tmp_path):
    """판정 신원 쓰기를 샤드 하나에서 멈춘다 — 둘째 샤드의 원문 QA(모델 대상)는 judge_missing, 정상 행은 영수증 없음."""
    _, a, st, _, src = _v7_audit(tmp_path, max_shards=1)
    assert src["all_task_ids"] == ["img-000", "img-001"] and src["model_task_ids"] == V7Judge.calls == ["img-000"]
    assert st["qa-000"]["status"] == "incomplete" and st["qa-000"]["reasons"] == ["judge_missing"]
    assert st["img-002"]["status"] == "incomplete" and st["img-002"]["reasons"] == ["deterministic_receipt_missing"]
    assert st["img-002"]["deterministic_pass"] is None and st["img-002"]["judge_applicable"] is False
    assert st["img-001"]["status"] == "accepted" == st["img-000"]["status"]
    assert a["complete"] is False and "incomplete_rows" in a["blockers"] and a["counts"]["reasons"] == {
        "deterministic_receipt_missing": 1, "judge_missing": 1}
    det = a["deterministic_receipts"]
    assert det["rows"] == 2 and det["normal"]["receipts"] == 1 and det["normal"]["status"] == {"accepted": 1, "incomplete": 1}
    assert a["policy"]["model_agreement"]["excluded"] == {"judge_not_applicable": 2}


def _receipt(src, rid):
    return next(r for r in src["deterministic_receipts"] if r["id"] == rid)


def _v7_break(src, breaks):
    normal = _receipt(src, "img-001")
    if breaks == "candidate":
        normal["candidate_sha256"] = _sha_text("다른 생성문")
    elif breaks == "finish":
        normal["candidate_finish_reason"] = "length"
    elif breaks == "policy_sha":
        normal["policy_identity_sha256"] = "0" * 64
    elif breaks == "forged_pass":                       # 정책이 기각하는 정상 행(img-002)을 통과로 고쳐 쓴 영수증
        banned = _receipt(src, "img-002")
        assert banned["deterministic_pass"] is False
        banned.update(deterministic_pass=True, deterministic_reasons=[],
                      deterministic_codes={g: [] for g in V7_GROUPS})
    elif breaks == "diagnostic_view":                   # 생성 계약을 끈 진단용 판단은 운영 영수증이 아니다
        normal["generation_contract_applied"] = False
    elif breaks == "extra_key":
        normal["approved"] = True
    elif breaks == "identity":
        normal["_judge_identity_sha256"] = "0" * 64
    elif breaks == "shard":
        normal["_shard_sha256"] = "not-a-hash"
    elif breaks == "missing":
        src["deterministic_receipts"].remove(normal)
    elif breaks == "order":
        src["deterministic_receipts"].reverse()
    elif breaks == "not_prefix":
        src["all_task_ids"] = src["all_task_ids"][1:]
    elif breaks == "model_ids_drop":
        src["model_task_ids"] = [x for x in src["model_task_ids"] if x != "img-000"]
    elif breaks == "model_ids_normal":
        src["model_task_ids"] = ["img-000", "img-001", "qa-000"]
    elif breaks == "normal_judged":                     # 정상 행에 판정 모델의 행이 있다
        src["rows"].append({**src["rows"][0], "id": "img-001"})
    elif breaks == "no_receipts":
        del src["deterministic_receipts"]
    elif breaks == "policy_version":
        src["policy_version"] = "v6"
    elif breaks == "top_schema":
        del src["output_schema"]


@pytest.fixture
def _v7_reread(monkeypatch):
    """위조 시험 전용 — 다시 읽기(판정 신원 읽기)가 변조를 먼저 막지 않게 주어진 객체를 그대로 돌려준다."""
    monkeypatch.setattr(Q, "_reread_judge_source", lambda run_dir, config, src: copy.deepcopy(dict(src)))


@pytest.mark.parametrize("breaks,code", [
    ("candidate", "judge_receipt_mismatch"), ("finish", "judge_receipt_mismatch"), ("policy_sha", "judge_receipt_mismatch"),
    ("forged_pass", "judge_receipt_mismatch"), ("diagnostic_view", "judge_receipt_mismatch"),
    ("extra_key", "judge_receipt_mismatch"), ("identity", "judge_receipt_identity"), ("shard", "judge_receipt_identity"),
    ("missing", "judge_receipts"), ("order", "judge_receipts"), ("not_prefix", "judge_receipts"),
    ("model_ids_drop", "judge_model_task_ids"), ("model_ids_normal", "judge_model_task_ids"),
    ("normal_judged", "judge_row_not_applicable"), ("no_receipts", "judge_source_v7"),
    ("policy_version", "judge_source_v7"), ("top_schema", "judge_source_v7"),
])
def test_v7_영수증이_생성문_정책_판정_신원과_어긋나면_집계하지_않는다(tmp_path, _v7_reread, breaks, code):
    """실제로 쓰고 읽은 출처를 변조한다. 둘째 정상 행(img-002)은 정책이 기각하는 생성문이다 — 그 영수증을 통과로 고쳐 쓰면
    정상 행이 모델 없이 수락될 길이다. 변조하지 않은 같은 출처는 집계를 지난다(첫 시험)."""
    run_dir, config, src = _v7_run(tmp_path, {"img-002": "표시된 결함이 없다. 합격이다."})
    _v7_break(src, breaks)
    with pytest.raises(AuditRefused, match=f"^{code}:"):
        audit_run(run_dir, config, tmp_path / "audit", judge_source=src)
    assert not (tmp_path / "audit" / "AUDIT.json").exists()


def test_v7_판정_출처와_v7_정책은_함께만_건다(tmp_path):
    run_dir, config, src = _v7_run(tmp_path)
    # v7 출처에 v6 의 정책 꼴(영수증을 다시 계산할 수 없다)
    with pytest.raises(AuditRefused, match="^policy_interface:"):
        audit_run(run_dir, config, tmp_path / "a1", judge_source=src, policy=StubPolicy())
    # 판정 신원에 묶인 것과 다른 v7 정책
    other = V7Policy({"version": "image_report_judge_v7_other", "output_schema": "judge_structured_v7"})
    with pytest.raises(AuditRefused, match="^judge_policy_mismatch:"):
        audit_run(run_dir, config, tmp_path / "a2", judge_source=src, policy=other)
    # 다른 판(출력 계약)의 지문을 내는 v7 정책
    with pytest.raises(AuditRefused, match="^policy_identity:"):
        audit_run(run_dir, config, tmp_path / "a3", judge_source=src,
                  policy=V7Policy({"version": "x", "output_schema": "judge_structured_v6"}))
    assert not any((tmp_path / n / "AUDIT.json").exists() for n in ("a1", "a2", "a3"))


# --------------------------------------------------------------------------- 구역별 소계의 수치 추적(실제 v7 정책 → 집계)


def _zone_task(upper, lower):
    """기공 upper + lower 개를 위-왼쪽 · 아래-오른쪽에 둔 결함 행. bbox 좌표에 1 ~ 4 가 없다(소계가 좌표에서 오지 않게)."""
    boxes = [("porosity", [10 + 40 * i, 20, 30 + 40 * i, 40]) for i in range(upper)]
    boxes += [("porosity", [610 + 40 * i, 620, 630 + 40 * i, 640]) for i in range(lower)]
    task = image_task(9, target=locked(*boxes))
    name = LABELS["porosity"].name_ko
    task["evidence"]["defect_positions"] = [{"name": name, "where": "위-왼쪽"}] * upper + [{"name": name, "where": "아래-오른쪽"}] * lower
    return task


def _zone_text(total, upper, lower, tail=""):
    name = LABELS["porosity"].name_ko
    return f"{name} {total}개가 표시되어 있다. 위-왼쪽에 {upper}개, 아래-오른쪽에 {lower}개가 있다.{tail} {G.LIMITATION_FORMS[0]}"


@pytest.mark.parametrize("upper", [1, 2, 3, 4])
def test_v7_참인_구역별_소계는_정책과_집계를_함께_지나고_틀린_수는_어느_한쪽이_막는다(upper):
    """총 5개를 두 구역에 나눈 소계 — 소계는 잠긴 칸 투영(판정 정책의 `grouped_locked_fields`)에서만 온다."""
    lower = 5 - upper
    task = _zone_task(upper, lower)
    others = {n for s in [*Q._strings(task["locked_target"])] for n in numbers_in(s)}
    assert Q._zone_subtotals(task) == {upper, lower} and not {upper, lower} & others   # 소계는 다른 원천에 없다
    good = _zone_text(5, upper, lower, " 주석만으로 실제 형상은 알 수 없다.")
    assert P.evaluate_v7(task, good)["deterministic_reasons"] == [] and Q._content_problems(task, {"text": good}) == []
    # 바꿔 적은 소계 — 두 수 모두 잠긴 소계라 집계는 지나고, 칸별 수를 맞대는 정책이 막는다
    swapped = _zone_text(5, lower, upper)
    assert P.evaluate_v7(task, swapped)["deterministic_reasons"] == ["zone_count_mismatch"]
    assert Q._content_problems(task, {"text": swapped}) == []
    # 잠긴 칸에 없는 소계 · 총합 · 실치수는 집계가 추적 불가로 막는다
    wrong = 2 if upper in (1, 4) else 1
    assert Q._content_problems(task, {"text": _zone_text(5, wrong, 5 - wrong)}) == ["untraceable_number"]
    assert Q._content_problems(task, {"text": _zone_text(6, upper, lower)}) == ["untraceable_number"]
    assert Q._content_problems(task, {"text": _zone_text(5, upper, lower, " 지름은 7 mm 이다.")}) == ["untraceable_number"]


def test_구역별_소계는_결함_행의_대조된_잠긴_칸에서만_온다():
    task = _zone_task(1, 4)
    assert Q._zone_subtotals(task) == {1, 4}
    # 근거의 위치가 잠긴 결함과 어긋나면 소계를 하나도 내지 않는다(원천을 넓히지 않는 쪽으로 닫는다)
    broken = copy.deepcopy(task)
    broken["evidence"]["defect_positions"] = broken["evidence"]["defect_positions"][:-1]
    assert Q._zone_subtotals(broken) == set()
    assert Q._content_problems(broken, {"text": _zone_text(5, 1, 4)}) == ["untraceable_number"]
    # 정상 행 · 원문 QA 는 대상이 아니다
    assert Q._zone_subtotals(image_task(1, target=locked())) == set() == Q._zone_subtotals(qa_task(0))
