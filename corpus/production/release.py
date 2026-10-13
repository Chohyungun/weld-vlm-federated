"""학습 릴리스 — 품질 집계 · 사람 승인 · 자료 이용 확인을 모두 다시 검사한 뒤에만 `RELEASE.json` 을 발행한다.

파일 계약은 `docs/dev_log/2026-10-07-본실험코퍼스/01_Claude_실행분담.md` 의 "파일 계약" 이 정본이다. 그 문서가 정하지 않은
`AUDIT.json` · 검수용 `dataset.jsonl` · 검수 파일 · 이용 확인 파일의 필드는 보고 `05_prepare_release_A.md` 의 "공유 계약(안)" 에 적었다
(B 의 `quality.audit_run` 과 파일로 맞춘다).

두 겹으로 검사한다.

- `release_run` — AUDIT 가 가리키는 실행 디렉터리의 생성 · 판정 샤드를 `infer.read_stage_rows` 로 **직접 다시 읽어**, 데이터셋 행마다
  작업 명부의 필드 · 샤드 지문 · 생성문 해시 · `_backend` · 판정 JSON 을 대조한다. AUDIT 의 집계를 믿지 않는다.
- `verify_release` — 릴리스 디렉터리만으로(실행 디렉터리 없이) 데이터셋 · 증빙 사본의 바이트와 행 내용을 다시 검사한다.
  학습기는 `common.validate_release` 를 거쳐 이것을 부른다.

누락 허용 · 시험용 우회 · 부분 릴리스는 없다. 한 항목이라도 어긋나면 `ReleaseRefused`. 출력은 새 디렉터리에만 —
임시 디렉터리에서 만들고 스스로 `verify_release` 를 통과한 뒤에 이름을 바꿔 공개한다.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .common import canonical_bytes, iter_jsonl, parse_json, read_json, sha256_file, write_json_new

SCHEMA_VERSION = 1
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
#: 판정 셋이 모두 참이어야 한다 — 그래도 사람 승인은 따로 본다
JUDGE_KEYS = ("grounded", "scope_ok", "language_ok")
#: 이용 확인의 용도에 이 말이 있어야 학습에 쓴다
RIGHTS_USE = "sft_training"
#: 영상 설명의 user message 틀에 들면 안 되는 꼴 — 숫자 · 괄호(코드 · bbox · 조항 id · JSON 골격)
_LEAK = re.compile(r"[0-9{}\[\]]")
#: 릴리스 디렉터리의 증빙 사본
EVIDENCE = {"audit": "AUDIT.json", "review": "REVIEW.json", "rights": "RIGHTS.json"}
#: 생성 QA 의 JSON 객체 하나를 담은 울타리 하나 — B 의 quality 와 같은 허용(render_v1)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
#: 판정 출처(B 05h) — run_config: RUN.json 의 판정 칸으로 돈 첫 판정, separate: 판정 신원을 따로 둔 새 판정. 한 릴리스 안에서 섞지 않는다
JUDGE_SOURCES = ("run_config", "separate")
#: 판정 출력의 꼴(B 의 quality `parser_schema`) — 옛 네 키 · 구조화 v5 · 구조화 v6. 칸이 없으면 옛 네 키(이전 집계).
#: v6 는 v5 를 대체하지 않는다 — 판 표지로 갈리고, 한 꼴을 다른 파서로 다시 읽지 않는다(B 05l)
JUDGE_SCHEMAS = ("legacy_four_keys", "judge_structured_v5", "judge_structured_v6", "judge_structured_v7")
#: 구조화 꼴 → B 의 파서(`quality` 의 이름). 파서 판은 `quality.JUDGE_PARSER_VERSIONS` 의 같은 짝에서 읽는다(R-11 · R-13) —
#: v7 은 B 의 집계가 v7 을 읽게 되어(`quality_v9`) 그 표에 들 때 열린다. 그 전에는 `judge_parser` 로 닫혀 있다
STRUCTURED_PARSERS = {"judge_structured_v5": "parse_judge_v5", "judge_structured_v6": "parse_judge_v6",
                      "judge_structured_v7": "parse_judge_v7"}
#: 판정 모델을 부르지 않는 행이 있을 수 있는 꼴 — v7 정상 행(B 05m · 05n R-15). 그 행의 판정 칸은 '적용 안 됨' 꼴
#: `{"applicable": false, "identity_sha256": <판정 신원>}` 두 칸뿐이다(05n 의 제안 꼴 — B 의 집계 v7 소비와 함께 확정)
MODEL_OPTIONAL_SCHEMAS = ("judge_structured_v7",)
#: 구조화 판정의 행 출처에서 판정 샤드를 다시 읽은 값과 맞대는 칸(B 05k · 05j R-9 · 05l)
STRUCTURED_FIELDS = ("schema", "parser", "answers", "flags", "format_normalized")
#: 판정 신원 읽기에 넘기는 외부 기대 핀(Codex 최종 API) — 운영 원천은 하나 이상 필요하다
JUDGE_PINS = ("expected_judge_identity_sha256", "expected_probe_approval_sha256")
#: 데이터셋 행이 작업 행과 같아야 하는 필드
TASK_FIELDS = ("kind", "split", "client", "lineage_group_id", "image_path", "source_ids")


class ReleaseRefused(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.exit_code = 2
        super().__init__(f"{code}: {message}")


def _need(cond: bool, code: str, message: str) -> None:
    if not cond:
        raise ReleaseRefused(code, message)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def ids_sha256(ids: list[str]) -> str:
    """id 수열의 지문 — 각 id 를 LF 로 이어 붙인 UTF-8 바이트(끝 LF 없음)의 sha256."""
    return _sha_bytes("\n".join(ids).encode("utf-8"))


def _family(name: Any) -> str:
    return str(name or "").strip().lower()


def _hex(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.fullmatch(value))


# ------------------------------------------------------------------------------------------
# 행 검사 — 학습기(train._validate_row)가 받는 꼴 + 출처
# ------------------------------------------------------------------------------------------
def _blocks(message: dict, rid: str, *, allow_image: bool) -> tuple[list[str], list[str]]:
    content = message.get("content")
    _need(isinstance(content, list) and bool(content), "row_messages", f"{rid}: content 블록이 없다")
    texts, images = [], []
    for block in content:
        _need(isinstance(block, dict), "row_messages", f"{rid}: 블록은 객체")
        if block.get("type") == "image" and allow_image:
            _need(set(block) == {"type", "image"}, "row_messages", f"{rid}: image 블록은 로컬 참조 하나")
            images.append(block["image"])
        elif block.get("type") == "text":
            _need(set(block) == {"type", "text"} and isinstance(block["text"], str) and bool(block["text"].strip()),
                  "row_messages", f"{rid}: text 블록이 비었다")
            texts.append(block["text"])
        else:
            raise ReleaseRefused("row_messages", f"{rid}: 허용되지 않는 블록")
    _need(bool(texts), "row_messages", f"{rid}: text 가 없다")
    return texts, images


def _image_path_ok(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return False
    pure = PurePosixPath(value)
    return not pure.is_absolute() and ".." not in pure.parts and "." not in pure.parts


def _verdict_keys(jschema: str, kind: str) -> tuple[str, ...]:
    """행 판정값의 물음 키(B 05n R-14b) — v7 영상 행(결함 행)은 B 의 v7 결함 물음 넷(참 = 문제 없음), 그 밖은 옛 셋."""
    if jschema == "judge_structured_v7" and kind == "image_report":
        from . import quality

        return tuple(k for k in quality.JUDGE_V7_KEYS["defect"] if k != "reason")
    return JUDGE_KEYS


def check_row(row: Any, audit: dict) -> dict:
    """한 행의 꼴 · 출처를 검사하고 user · assistant 문자열을 돌려준다."""
    _need(isinstance(row, dict), "row_schema", "행은 객체")
    rid = row.get("id")
    _need(isinstance(rid, str) and bool(rid), "row_schema", "id")
    _need(isinstance(row.get("lineage_group_id"), str) and bool(row["lineage_group_id"]), "row_schema", f"{rid}: lineage_group_id")
    _need(row.get("split") in ("train", "val"), "row_split", f"{rid}: split 은 train · val 만")
    _need(row.get("kind") in ("image_report", "source_qa"), "row_kind", f"{rid}: kind")
    _need(isinstance(row.get("source_ids"), list) and bool(row["source_ids"])
          and all(isinstance(s, str) and s for s in row["source_ids"]), "row_sources", f"{rid}: source_ids")
    msgs = row.get("messages")
    _need(isinstance(msgs, list) and len(msgs) == 2
          and [m.get("role") if isinstance(m, dict) else None for m in msgs] == ["user", "assistant"],
          "row_messages", f"{rid}: user · assistant 한 번씩")
    user_texts, user_images = _blocks(msgs[0], rid, allow_image=True)
    assistant_texts, _ = _blocks(msgs[1], rid, allow_image=False)
    if row["kind"] == "image_report":
        _need(row.get("client") in ("C1", "C2", "C3"), "row_client", f"{rid}: 영상 행은 원래 참여자")
        _need(_image_path_ok(row.get("image_path")), "row_image", f"{rid}: image_path")
        _need(user_images == [row["image_path"]], "row_image", f"{rid}: 메시지의 이미지와 image_path 가 다르다")
        _need(_hex(row.get("image_sha256")), "row_image_sha256", f"{rid}: image_sha256(64 자리 소문자) 필수")
        templates = (audit.get("user_templates") or {}).get("image_report") or []
        _need(len(user_texts) == 1 and user_texts[0] in templates, "row_user_template",
              f"{rid}: 영상 설명의 user message 는 AUDIT 에 등록한 고정 틀이어야 한다(생성 prompt · 골격 금지)")
    else:
        _need(row.get("client") == "public" and row.get("image_path") is None and not user_images
              and "image_sha256" not in row, "row_source_qa", f"{rid}: 원문 QA 는 이미지 없는 public")
    prov = row.get("provenance")
    _need(isinstance(prov, dict), "row_provenance", f"{rid}: provenance")
    gen, judge = prov.get("generate"), prov.get("judge")
    _need(isinstance(gen, dict) and isinstance(judge, dict), "row_provenance", f"{rid}: generate · judge 출처")
    jschema = (audit.get("judge_identity") or {}).get("parser_schema", "legacy_four_keys")
    applicable = "applicable" not in judge
    if not applicable:
        # 판정 모델을 부르지 않은 행(v7 정상 행) — 판정 샤드 · 원문 · 판정값 칸이 없어야 한다. 작업 행과의 대조는 check_against_run
        _need(jschema in MODEL_OPTIONAL_SCHEMAS and row["kind"] == "image_report" and judge.get("applicable") is False
              and set(judge) == {"applicable", "identity_sha256"}, "row_judge",
              f"{rid}: 판정 '적용 안 됨' 꼴은 {MODEL_OPTIONAL_SCHEMAS} 의 영상 행에서 두 칸(applicable · identity_sha256)뿐이다")
    _need(gen.get("backend") == "transformers" and (not applicable or judge.get("backend") == "transformers"), "row_backend",
          f"{rid}: _backend 가 transformers 가 아니다(시험용 출력은 릴리스 금지)")
    _need(bool(_family(gen.get("family"))) and (not applicable or _family(gen.get("family")) != _family(judge.get("family"))),
          "row_family", f"{rid}: 생성 · 판정 계열이 같거나 비었다")
    for side in ((gen, judge) if applicable else (gen,)):
        _need(_hex(side.get("shard_sha256")) and _hex(side.get("text_sha256")), "row_provenance",
              f"{rid}: 샤드 · 출력 지문")
    if applicable:
        keys = _verdict_keys(jschema, row["kind"])
        verdict = judge.get("verdict")
        _need(isinstance(verdict, dict) and set(verdict) == {*keys, "reason"}
              and all(verdict[k] is True for k in keys)
              and isinstance(verdict["reason"], str) and bool(verdict["reason"].strip()),
              "row_judge", f"{rid}: 다른 계열 판정이 통과가 아니거나 근거가 없다")
    return {"user": "\n".join(user_texts), "assistant": "\n".join(assistant_texts)}


def render_messages(task: dict, generated_text: str) -> list[dict]:
    """정규 렌더러 render_v1 — 판정받은 생성문에서 학습 메시지를 정확히 만든다. B 의 `quality._content` · `_dataset_row`(4733fe9)와 같은 규칙:
    영상 설명은 user = [이미지, 작업의 train_user_text], assistant = [생성문의 앞뒤 공백만 뗀 것].
    원문 QA 는 생성 JSON(울타리 하나 허용)의 question · answer 앞뒤 공백만 떼어 user · assistant 하나씩. 다른 글을 덧붙이면 다르다."""
    if task["kind"] == "image_report":
        return [{"role": "user", "content": [{"type": "image", "image": task["image_path"]},
                                             {"type": "text", "text": task.get("train_user_text")}]},
                {"role": "assistant", "content": [{"type": "text", "text": generated_text.strip()}]}]
    s = generated_text.strip()
    m = _FENCE.fullmatch(s)
    if m:
        s = m.group(1)
    try:
        qa = parse_json(s)
    except ValueError as exc:
        raise ReleaseRefused("generation_unparsable", f"{task['id']}: 생성 QA 를 JSON 으로 읽을 수 없다") from exc
    _need(isinstance(qa, dict) and set(qa) == {"question", "answer", "source_ids"}
          and all(isinstance(qa[k], str) and qa[k].strip() for k in ("question", "answer"))
          and isinstance(qa["source_ids"], list) and bool(qa["source_ids"])
          and all(isinstance(x, str) and x for x in qa["source_ids"])
          and set(qa["source_ids"]) <= set(task["source_ids"]), "generation_schema", f"{task['id']}: 생성 QA 의 꼴 · 근거 id")
    return [{"role": "user", "content": [{"type": "text", "text": qa["question"].strip()}]},
            {"role": "assistant", "content": [{"type": "text", "text": qa["answer"].strip()}]}]


# ------------------------------------------------------------------------------------------
# 집계 · 증빙
# ------------------------------------------------------------------------------------------
def check_judge_identity(audit: dict) -> dict:
    """R-1 — AUDIT `judge_identity` 의 꼴과 신원 근거. 판정 모델(`model`)을 돌려준다(행 대조의 기준)."""
    ji = audit.get("judge_identity")
    _need(isinstance(ji, dict) and ji.get("source") in JUDGE_SOURCES, "audit_judge_identity", "AUDIT 에 판정 출처가 없다")
    _need(ji.get("parser_schema", "legacy_four_keys") in JUDGE_SCHEMAS, "audit_judge_identity", "판정 출력의 꼴(parser_schema)")
    if ji.get("source") == "run_config":
        _need(ji.get("parser_schema", "legacy_four_keys") == "legacy_four_keys", "audit_judge_identity",
              "실행 안의 구판정은 옛 네 키 꼴이어야 한다")
    model = ji.get("model")
    _need(isinstance(model, dict) and isinstance(model.get("model_id"), str) and bool(model["model_id"])
          and bool(_family(model.get("family"))), "audit_judge_identity", "판정 모델(model_id · family)")
    if ji["source"] == "run_config":
        judge_cfg = (audit.get("config") or {}).get("judge") or {}
        _need(ji.get("identity_sha256") is None and ji.get("identity_basis") is None
              and model["model_id"] == judge_cfg.get("model_id") and _family(model["family"]) == _family(judge_cfg.get("family")),
              "audit_judge_identity", "run_config 판정은 신원이 null 이고 모델이 config 의 판정 칸이어야 한다")
    else:
        basis = ji.get("identity_basis")
        _need(_hex(ji.get("identity_sha256")) and isinstance(basis, dict)
              and _sha_bytes(canonical_bytes(basis)) == ji["identity_sha256"], "audit_judge_identity",
              "separate 판정 신원이 근거의 정규 JSON 지문이 아니다")
        _need(basis.get("run_identity_sha256") == audit.get("run_identity_sha256"), "audit_judge_identity",
              "판정 신원 근거의 실행 신원이 AUDIT 와 다르다")
        policy = basis.get("policy_identity")
        _need(isinstance(policy, dict) and policy.get("output_schema") == ji.get("parser_schema"), "audit_judge_identity",
              "separate 판정의 꼴이 판정 신원에 묶인 정책의 출력 꼴과 다르다")
        pins = ji.get("caller_expected_pins")
        _need(isinstance(pins, dict) and set(pins) == set(JUDGE_PINS), "audit_judge_identity",
              "separate 판정의 AUDIT 에 판정 폴더를 읽은 핀(두 칸)이 없다")
        jm = basis.get("judge_model") or {}
        _need(jm.get("model_id") == model["model_id"] and _family(jm.get("family")) == _family(model["family"]),
              "audit_judge_identity", "판정 신원 근거의 모델이 judge_identity.model 과 다르다")
    return model


def counts_of(rows: list[dict]) -> dict:
    c = Counter((r["kind"], r["split"]) for r in rows)
    return {"rows": len(rows),
            "image_report": {"train": c[("image_report", "train")], "val": c[("image_report", "val")]},
            "source_qa": {"train": c[("source_qa", "train")], "val": c[("source_qa", "val")]},
            "by_client": dict(sorted(Counter(r["client"] for r in rows).items()))}


def _label_names() -> list[str]:
    from data.label_map import load_label_map

    return sorted({dt.name_ko for dt in load_label_map().defect_types.values()})


def check_audit(audit: dict, dataset_sha: str, rows: list[dict]) -> None:
    _need(audit.get("schema_version") == 1, "audit_schema", "AUDIT.json schema_version 1")
    _need(audit.get("complete") is True, "audit_incomplete", "품질 집계가 완결이 아니다(부분 통과로 릴리스하지 않는다)")
    for key in ("errors", "rejected", "incomplete"):
        _need(audit.get(key) == 0, f"audit_{key}", f"{key} 가 0 이 아니다: {audit.get(key)!r}")
    # 수에 비치지 않는 막는 사유까지 — 수 검사(구체적인 사유) 뒤에 본다(B 05c 권고)
    _need(audit.get("status") == "ready_for_human_review" and audit.get("blockers") == [], "audit_blocked",
          f"품질 집계가 사람 검수 대기가 아니다: {audit.get('status')!r} · {audit.get('blockers')!r}")
    _need(audit.get("dataset_sha256") == dataset_sha, "dataset_sha256", "dataset.jsonl 이 집계한 바이트가 아니다")
    for key in ("run_identity_sha256", "tasks_sha256", "config_sha256"):
        _need(_hex(audit.get(key)), "audit_identity", key)
    config = audit.get("config")
    _need(isinstance(config, dict) and _sha_bytes(canonical_bytes(config)) == audit["config_sha256"],
          "audit_config", "AUDIT 의 config 가 config_sha256 과 다르다")
    ids = [r["id"] for r in rows]
    _need(len(set(ids)) == len(ids), "dataset_duplicate", "같은 id 가 둘 이상")
    planned = audit.get("planned") or {}
    _need(planned.get("count") == len(rows) and planned.get("ids_sha256") == ids_sha256(ids), "planned_sequence",
          "데이터셋의 id 수열이 작업 명부의 계획과 같지 않다(누락 · 추가 · 순서)")
    lineages: dict[str, str] = {}
    for r in rows:
        prior = lineages.setdefault(r["lineage_group_id"], r["split"])
        _need(prior == r["split"], "lineage_split_overlap", r["lineage_group_id"])
    gen_cfg = config.get("generation") or {}
    judge_model = check_judge_identity(audit)
    _need(bool(_family(gen_cfg.get("family"))) and _family(gen_cfg.get("family")) != _family(judge_model.get("family"))
          and gen_cfg.get("model_id") != judge_model.get("model_id"), "audit_family", "생성 · 판정 모델이 다른 계열이 아니다")
    jid = audit["judge_identity"]["identity_sha256"]
    for r in rows:
        g, j = r["provenance"]["generate"], r["provenance"]["judge"]
        _need(j.get("identity_sha256", "absent") == jid, "row_judge_identity",
              f"{r['id']}: 행의 판정 신원이 AUDIT 의 판정 신원과 다르다")
        _need(_family(g.get("family")) == _family(gen_cfg["family"])
              and (j.get("applicable") is False
                   or (_family(j.get("family")) == _family(judge_model["family"]) and j.get("model_id") == judge_model["model_id"])),
              "row_family",
              f"{r['id']}: 행의 계열 · 모델이 집계의 생성(config) · 판정(judge_identity.model) 모델과 다르다")
    templates = (audit.get("user_templates") or {}).get("image_report") or []
    names = _label_names()
    for t in templates:
        _need(isinstance(t, str) and bool(t.strip()) and not _LEAK.search(t) and not any(n in t for n in names),
              "user_template_leak", f"영상 설명의 user message 틀에 정답 골격 꼴(숫자 · 괄호 · 결함 이름)이 있다: {t!r}")
    _need(isinstance(audit.get("source_documents"), list), "audit_schema", "source_documents 는 목록")
    q = config.get("quality") or {}
    src_train = [r for r in rows if r["kind"] == "source_qa" and r["split"] == "train"]
    per_passage = Counter(s for r in src_train for s in r["source_ids"])
    _need(len(src_train) >= int(q.get("minimum_source_qa_train", 0)), "source_qa_minimum",
          f"원문 QA train {len(src_train)} < 하한 {q.get('minimum_source_qa_train')}")
    _need(len(per_passage) >= int(q.get("minimum_distinct_train_passages", 0)), "distinct_passages",
          f"고유 구절 {len(per_passage)} < 하한 {q.get('minimum_distinct_train_passages')}")
    _need(not per_passage or max(per_passage.values()) <= int(q.get("maximum_questions_per_passage", 0)),
          "questions_per_passage", "구절당 질문 상한을 넘었다")
    mine = counts_of(rows)
    got = audit.get("counts") or {}
    _need(all(got.get(k) == mine[k] for k in mine), "audit_counts",
          "AUDIT 의 counts 가 데이터셋에서 다시 낸 값과 다르다")


def _parse_time(value: Any) -> None:
    _need(isinstance(value, str) and bool(value), "timestamp", "시각이 없다")
    try:
        t = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReleaseRefused("timestamp", f"ISO 8601 이 아니다: {value!r}") from exc
    _need(t.tzinfo is not None, "timestamp", f"시간대가 없는 시각: {value!r}")


def check_review(review: dict, audit: dict, dataset_sha: str, rows: list[dict]) -> None:
    _need(review.get("schema_version") == 1, "review_schema", "검수 파일 schema_version 1")
    _need(isinstance(review.get("reviewer"), str) and bool(review["reviewer"].strip()), "review_reviewer", "검수자가 없다")
    _parse_time(review.get("reviewed_at"))
    _need(review.get("target_dataset_sha256") == dataset_sha, "review_target", "검수 대상 SHA 가 데이터셋과 다르다")
    plan = audit.get("review") or {}
    strata, sizes = plan.get("strata"), plan.get("stratum_sizes")
    _need(isinstance(strata, dict) and bool(strata) and isinstance(sizes, dict) and set(sizes) == set(strata)
          and all(type(v) is int for v in sizes.values()), "review_plan", "AUDIT 에 검수 표본의 층 · 층 크기가 없다")
    _need(sum(sizes.values()) == len(rows), "review_plan", "층 크기의 합이 데이터셋 행 수와 다르다")
    ids_all = {r["id"] for r in rows}
    planned_ids = [i for v in strata.values() for i in v]
    _need(len(set(planned_ids)) == len(planned_ids) and set(planned_ids) <= ids_all, "review_plan", "검수 표본 id")
    q = (audit.get("config") or {}).get("quality") or {}
    per_min, frac = int(q.get("minimum_review_per_stratum", 0)), float(q.get("review_fraction", 0))
    for name, members in strata.items():
        _need(sizes[name] >= len(members) and len(members) >= min(per_min, sizes[name]), "review_plan",
              f"{name}: 층마다 최소 {per_min}(층이 그보다 작으면 전수)")
    _need(len(planned_ids) >= math.ceil(frac * len(rows)), "review_plan", f"검수 표본이 전체의 {frac} 에 못 미친다")
    samples = review.get("samples")
    _need(isinstance(samples, list) and bool(samples), "review_samples", "검수 표본 판정이 없다")
    got = [s.get("id") for s in samples if isinstance(s, dict)]
    _need(len(got) == len(samples) and sorted(got) == sorted(planned_ids), "review_samples",
          "검수한 표본이 집계가 정한 표본과 같지 않다")
    row_sha = {r["id"]: _sha_bytes(canonical_bytes(r)) for r in rows}
    for s in samples:
        _need(s.get("decision") == "approve", "review_rejected", f"{s.get('id')}: 승인이 아니다 — 하나라도 기각이면 릴리스하지 않는다")
        _need(s.get("row_sha256") == row_sha[s["id"]], "review_row", f"{s['id']}: 검수한 행의 지문이 데이터셋 행과 다르다")
        if s.get("reviewer") is not None:
            _need(isinstance(s["reviewer"], str) and bool(s["reviewer"].strip()), "review_reviewer", f"{s['id']}: 표본 검수자")
        if s.get("reviewed_at") is not None:
            _parse_time(s["reviewed_at"])
    _need(review.get("approved") is True, "review_approved", "검수 전체 승인이 아니다")


def required_assets(audit: dict) -> list[Any]:
    """이용 확인이 덮어야 하는 자산 지문 — 페어 · 영상 매니페스트, 그리고 원문 QA 가 있으면 구절 은행 · 원문 문서.
    구절 은행은 원문 QA 에만 쓰인다 — 원문 문서가 하나도 없는 실행(영상 설명만)에서는 은행의 이용 확인을 요구하지 않는다."""
    basis = audit.get("run_identity_basis") or {}
    need = [basis.get("pairs_sha256"), basis.get("manifest_snapshot_digest")]
    docs = audit.get("source_documents")
    if not isinstance(docs, list) or docs:
        need.append(basis.get("source_bank_sha256"))
    need += [d.get("source_sha256") if isinstance(d, dict) else None for d in (docs if isinstance(docs, list) else [None])]
    return need


def check_rights(rights: dict, audit: dict) -> None:
    _need(rights.get("schema_version") == 1, "rights_schema", "자료 이용 확인 schema_version 1")
    _need(isinstance(rights.get("confirmed_by"), str) and bool(rights["confirmed_by"].strip()), "rights_by", "확인자가 없다")
    _parse_time(rights.get("confirmed_at"))
    assets = rights.get("assets")
    _need(isinstance(assets, list) and bool(assets), "rights_assets", "자산 목록이 없다")
    covered: dict[str, set] = {}
    for a in assets:
        _need(isinstance(a, dict) and _hex(a.get("sha256")), "rights_assets", "자산 sha256")
        use = a.get("use")
        _need(isinstance(use, list) and all(isinstance(u, str) for u in use), "rights_assets", "용도 목록")
        covered[a["sha256"]] = set(use)
    basis = audit.get("run_identity_basis")
    _need(isinstance(basis, dict) and _sha_bytes(canonical_bytes(basis)) == audit.get("run_identity_sha256"),
          "audit_identity_basis", "AUDIT 의 신원 근거가 run_identity_sha256 과 맞지 않다")
    for sha in required_assets(audit):
        _need(_hex(sha) and RIGHTS_USE in covered.get(sha, set()), "rights_missing", f"자산 {sha!r} 의 학습 이용 확인이 없다")


def _check_all(audit: dict, dataset_path: Path, review: dict, rights: dict) -> tuple[str, list[dict], list[dict]]:
    dataset_sha = sha256_file(dataset_path)
    rows = list(iter_jsonl(dataset_path))
    _need(bool(rows), "dataset_empty", "데이터셋이 비었다")
    texts = [check_row(r, audit) for r in rows]
    check_audit(audit, dataset_sha, rows)
    check_review(review, audit, dataset_sha, rows)
    check_rights(rights, audit)
    return dataset_sha, rows, texts


# ------------------------------------------------------------------------------------------
# 실행 디렉터리 대조 — release_run 만
# ------------------------------------------------------------------------------------------
def _reread_judge_source(run_dir: Path, config: dict, judge_source: dict, pins: dict) -> dict:
    """판정 객체의 `judge_dir` 를 Codex 의 판정 신원 읽기(`infer.read_judge_source`)로 다시 읽는다 — 샤드 바이트 · 신원 ·
    생성 샤드 목록 · 실제 판정 모델은 그 읽기가 검사한다. 경로는 신원이 아니라 찾는 자리다(B 05i I-1, 05h 6-2 R-3b)."""
    from . import infer

    judge_dir = judge_source.get("judge_dir")
    _need(isinstance(judge_dir, str) and bool(judge_dir), "judge_source_unverified",
          "judge_dir 가 없다 — 호출자가 준 판정 행을 그대로 믿지 않는다")
    try:
        verified = infer.read_judge_source(run_dir, parse_json(canonical_bytes(config)), judge_dir, **pins)
    except (ValueError, OSError) as exc:          # StageRefused 는 ValueError 다
        raise ReleaseRefused("judge_source_unverified", f"판정 폴더를 다시 읽지 못했다 — {exc}") from exc
    _need(isinstance(verified, dict) and isinstance(verified.get("rows"), list), "judge_source_unverified",
          "판정 신원 읽기가 행 목록을 내지 않았다")
    return verified


def _structured_parser(schema: str):
    """구조화 꼴의 B 파서와 그 판(B 05l R-10 · R-11). 판은 `quality.JUDGE_PARSER_VERSIONS` 에서 읽는다 — 표에 그 꼴이 없거나
    파서가 없으면 거부한다(다른 꼴의 파서로 대신 읽지 않는다)."""
    from . import quality

    parse = getattr(quality, STRUCTURED_PARSERS[schema], None)
    version = (getattr(quality, "JUDGE_PARSER_VERSIONS", None) or {}).get(schema)
    _need(callable(parse) and isinstance(version, str) and bool(version), "judge_parser",
          f"{schema} 의 파서({STRUCTURED_PARSERS[schema]}) 또는 그 판이 없다")
    return parse, version


def _no_model_ids(parse, tasks: list[dict]) -> set[str]:
    """판정 모델을 부르지 않는 작업 행의 id(B 05n R-15 · R-16) — B 의 파서가 판정 행 없이 그 작업을 '적용 안 됨'
    (`judge_not_applicable`)으로 고르는 행. 한 정의(파서)를 따르고, 옛 네 키 · v5 · v6 에는 없다."""
    if parse is None:
        return set()
    return {t["id"] for t in tasks if parse(t, None)["errors"] == ["judge_not_applicable"]}


def _v7_source_receipts(audit: dict, ji: dict, verified: dict, tasks: list[dict], no_model: set[str],
                        given: list[dict]) -> tuple[dict[str, dict], str]:
    """v7 판정 출처의 모델 대상 · 결정적 영수증 · 정책 지문(B 05p R-20 ~ R-22) — 다시 읽은 출처에서 본다(집계를 믿지 않는다).
    돌려주는 것: (작업 id → 영수증, 정책 지문). 행마다의 영수증 대조(R-19)는 부르는 쪽이 한다."""
    task_ids = [t["id"] for t in tasks]
    model_ids = [tid for tid in task_ids if tid not in no_model]
    _need(verified.get("all_task_ids") == task_ids, "stage_incomplete", "v7 판정을 마친 작업 id 가 작업 명부 전체와 같은 수열이 아니다")
    _need(verified.get("model_task_ids") == model_ids and [r.get("id") for r in given] == model_ids, "judge_model_task_ids",
          "v7 모델 대상 id 가 작업 − 판정을 부르지 않는 행(명부 순서) 이나 판정 행의 수열과 다르다")
    receipts = verified.get("deterministic_receipts")
    _need(isinstance(receipts, list) and all(isinstance(r, dict) for r in receipts)
          and [r.get("id") for r in receipts] == task_ids, "stage_incomplete", "결정적 영수증이 작업 명부 전체와 하나씩 같지 않다")
    _need(all(r.get("_judge_identity_sha256") == ji["identity_sha256"] for r in receipts), "judge_receipt_identity",
          "다른 판정 신원의 결정적 영수증이 섞였다")
    policy_sha = (audit.get("policy") or {}).get("identity_sha256")
    _need(_hex(policy_sha) and policy_sha == _sha_bytes(canonical_bytes((ji.get("identity_basis") or {}).get("policy_identity"))),
          "audit_policy_identity", "AUDIT 의 정책 지문이 판정 신원 근거에 묶인 정책의 지문이 아니다")
    dr = audit.get("deterministic_receipts")
    _need(isinstance(dr, dict) and dr.get("policy_identity_sha256") == policy_sha and dr.get("rows") == len(task_ids)
          and dr.get("planned") == len(task_ids)
          and dr.get("model_task_ids") == {"count": len(model_ids), "ids_sha256": ids_sha256(model_ids)},
          "audit_deterministic_receipts", "AUDIT 의 결정적 영수증 요약이 다시 읽은 출처와 다르다")
    return {r["id"]: r for r in receipts}, policy_sha


def check_against_run(audit: dict, audit_dir: Path, rows: list[dict], texts: list[dict], judge_source: dict | None = None,
                      pins: dict | None = None) -> None:
    """AUDIT 가 가리키는 실행 디렉터리를 Codex 의 검증 읽기로 다시 읽어 행마다 대조한다.

    판정 행은 AUDIT 의 판정 출처에서만 읽는다(B 05h R-3). separate 이면 `judge_source`(B 의 `audit_run` 과 같은 꼴 —
    `identity_sha256` · `identity_basis` · `rows`)의 행만 쓰고 실행 디렉터리의 옛 판정은 읽지 않는다."""
    from . import infer

    _need(isinstance(audit.get("run_dir"), str) and bool(audit["run_dir"]), "audit_run_dir", "AUDIT 에 run_dir 가 없다")
    run_dir = Path(audit["run_dir"])
    if not run_dir.is_absolute():
        run_dir = audit_dir / run_dir
    config = audit["config"]
    run = read_json(run_dir / "RUN.json")
    for key, akey in (("identity_sha256", "run_identity_sha256"), ("tasks_sha256", "tasks_sha256"),
                      ("config_sha256", "config_sha256")):
        _need(run.get(key) == audit.get(akey), "run_identity", f"RUN.json 의 {key} 가 AUDIT 와 다르다")
    _need(run.get("identity_basis") == audit.get("run_identity_basis"), "run_identity", "신원 근거가 RUN.json 과 다르다")
    gen_rows = infer.read_stage_rows(run_dir, "generate", config)
    tasks = list(iter_jsonl(run_dir / "tasks.jsonl"))                # 지문은 위 읽기가 RUN.json 과 대조했다
    ji = audit["judge_identity"]
    jschema = ji.get("parser_schema", "legacy_four_keys")
    parse, parser_version = _structured_parser(jschema) if jschema in STRUCTURED_PARSERS else (None, None)
    no_model = _no_model_ids(parse, tasks)
    receipts: dict[str, dict] | None = None       # v7 의 결정적 영수증(작업 id → 영수증), 그 밖은 None
    policy_sha = None
    if ji["source"] == "run_config":
        _need(judge_source is None, "judge_identity_mixed", "AUDIT 는 run_config 판정인데 따로 둔 판정이 주어졌다")
        judge_rows = infer.read_stage_rows(run_dir, "judge", config)    # 자기 판정 · 샤드 · 입력 지문은 infer 가 검사한다
    else:
        _need(isinstance(judge_source, dict), "judge_source_missing", "separate 판정의 행(judge_source)이 주어지지 않았다")
        pins = dict(pins or {})
        _need(any(pins.get(k) is not None for k in JUDGE_PINS), "judge_source_pin_required",
              "separate 판정은 외부에 따로 기록한 기대 판정 신원 또는 탐침 승인 SHA256 을 명시해야 한다")
        _need(ji.get("caller_expected_pins") == pins, "judge_source_unverified",
              "릴리스의 판정 핀이 품질 집계가 판정 폴더를 읽은 핀과 다르다")
        verified = _reread_judge_source(run_dir, config, judge_source, pins)  # 호출자 객체는 참고만 — 폴더를 다시 읽는다
        _need(verified.get("identity_sha256") == ji["identity_sha256"]
              and verified.get("identity_basis") == ji["identity_basis"], "judge_identity_mixed",
              "판정 폴더의 신원이 AUDIT 의 판정 신원과 다르다")
        # judge_dir 밖의 모든 칸(신원 · 근거 · 행 · 핀 · 핀 출처 · 탐침 승인 · GPU 검증)이 다시 읽은 것과 바이트로 같아야 한다(B 05j)
        mine = {k: v for k, v in judge_source.items() if k != "judge_dir"}
        theirs = {k: v for k, v in verified.items() if k != "judge_dir"}
        _need(canonical_bytes(mine) == canonical_bytes(theirs), "judge_source_unverified",
              "주어진 판정 객체가 판정 폴더를 다시 읽은 것과 다르다(칸 · 값)")
        for key in ("probe_approval_sha256", "gpu_model_verified", "pin_provenance"):
            _need(ji.get(key) == verified.get(key), "judge_source_unverified",
                  f"AUDIT 의 judge_identity.{key} 가 판정 폴더를 다시 읽은 값과 다르다")
        given = verified.get("rows")
        _need(isinstance(given, list), "judge_source_missing", "판정 행 목록")
        _need(all(isinstance(r, dict) and r.get("_judge_identity_sha256") == ji["identity_sha256"] for r in given),
              "judge_identity_mixed", "다른 판정 신원의 행이 섞였다")
        by_id: dict[str, dict] = {}
        for r in given:
            _need(r.get("id") not in by_id, "judge_identity_mixed", f"판정 행 id 가 겹친다: {r.get('id')}")
            by_id[r.get("id")] = r
        _need(not set(by_id) & no_model, "judge_row_not_applicable",
              f"판정 모델을 부르지 않는 작업 행의 판정 행이 있다 — {len(set(by_id) & no_model)}개")
        _need(set(by_id) == {t["id"] for t in tasks} - no_model, "stage_incomplete",
              f"판정 행 {len(by_id)} · 판정을 부르는 작업 {len(tasks) - len(no_model)} — 작업 명부와 같은 집합이 아니다")
        judge_rows = [by_id.get(t["id"]) for t in tasks]              # 판정을 부르지 않는 행은 None
        if jschema in MODEL_OPTIONAL_SCHEMAS:
            receipts, policy_sha = _v7_source_receipts(audit, ji, verified, tasks, no_model, given)
    if receipts is None:
        _need(audit.get("deterministic_receipts") is None, "audit_deterministic_receipts",
              "결정적 영수증 요약은 v7 판정 출처의 집계에만 있다")
    _need(len(gen_rows) == len(tasks) and len(judge_rows) == len(tasks), "stage_incomplete",
          f"생성 {len(gen_rows)} · 판정 {len(judge_rows)} · 작업 {len(tasks)} — 전량이 아니다")
    for stage_rows in (gen_rows, judge_rows):
        bad = [r["id"] for r in stage_rows if r is not None and r.get("_backend") != "transformers"]
        _need(not bad, "stage_backend", f"_backend 가 transformers 가 아닌 행 {len(bad)}개(첫 {bad[:1]})")
    task_ids = [t["id"] for t in tasks]
    _need([r["id"] for r in rows] == task_ids, "planned_sequence", "데이터셋 id 수열이 tasks.jsonl 과 같지 않다")
    docs = sorted({(t["evidence"].get("source_id"), t["evidence"].get("source_sha256"))
                   for t in tasks if t["kind"] == "source_qa"})
    got_docs = sorted((d.get("source_id"), d.get("source_sha256")) for d in audit.get("source_documents") or [])
    _need(got_docs == docs, "source_documents", "AUDIT 의 원문 문서 목록이 작업 명부와 다르다")
    gen_cfg, judge_cfg = config["generation"], ji["model"]
    # 집계를 만든 파서의 판 = 지금 다시 읽는 파서의 판(B 05j R-9 · 06f A-1). 옛 네 키면 None — 다른 판으로 만든 집계는 여기서 멈춘다
    _need(audit.get("judge_parser_version") == parser_version, "audit_judge_identity",
          f"AUDIT 의 파서 판 {audit.get('judge_parser_version')!r} 가 지금 다시 읽는 파서의 판 {parser_version!r} 과 다르다")
    for row, txt, task, g, j in zip(rows, texts, tasks, gen_rows, judge_rows):
        rid = row["id"]
        _need(g["id"] == rid and (j is None or j["id"] == rid), "stage_order", rid)
        for key in TASK_FIELDS:
            _need(row.get(key) == task.get(key), "task_field", f"{rid}: {key} 가 작업 명부와 다르다")
        if task["kind"] == "image_report":
            _need(row.get("image_sha256") == task.get("image_sha256"), "task_field", f"{rid}: image_sha256 이 작업 명부와 다르다")
            _need(task["prompt"] not in txt["user"], "user_prompt_leak", f"{rid}: 생성 prompt 를 user message 로 복사했다")
            _need(txt["user"] == task.get("train_user_text"), "user_text", f"{rid}: user message 가 작업 행의 train_user_text 가 아니다")
        _need(row["messages"] == render_messages(task, g["text"]), "message_mismatch",
              f"{rid}: 학습 메시지가 판정받은 생성문에서 정규 렌더러로 만든 것과 다르다")
        prov = row["provenance"]
        if receipts is not None:
            # 수락의 결정적 근거(B 05p R-19 · R-22) — 그 행의 영수증이 통과이고, 같은 생성문 · 같은 정책 지문이어야 한다
            rc = receipts[rid]
            _need(rc.get("deterministic_pass") is True, "judge_receipt_rejected", f"{rid}: 결정적 영수증이 기각인 행이 데이터셋에 있다")
            _need(rc.get("candidate_sha256") == _sha_bytes(g["text"].encode("utf-8")) and rc.get("policy_identity_sha256") == policy_sha,
                  "judge_receipt_mismatch", f"{rid}: 결정적 영수증의 생성문 · 정책 지문이 이 행과 다르다")
            if row["kind"] == "image_report":
                _need((prov.get("policy") or {}).get("identity_sha256") == policy_sha, "row_policy_identity",
                      f"{rid}: 행의 정책 지문이 AUDIT 의 정책 지문과 다르다")
        if j is None:     # 판정 모델을 부르지 않은 행 — 판정 칸은 '적용 안 됨' 꼴 두 칸뿐(B 05n R-15)
            _need(prov["judge"] == {"applicable": False, "identity_sha256": ji["identity_sha256"]}, "judge_mismatch",
                  f"{rid}: 판정을 부르지 않은 행의 판정 칸이 '적용 안 됨' 꼴이 아니다")
        else:
            _need("applicable" not in prov["judge"], "judge_mismatch", f"{rid}: 판정을 부른 행인데 판정 칸이 '적용 안 됨' 꼴이다")
        for side, stage_row, spec in (("generate", g, gen_cfg), ("judge", j, judge_cfg)):
            if stage_row is None:
                continue
            p = prov[side]
            _need(p.get("backend") == stage_row["_backend"] and p.get("shard_sha256") == stage_row["_shard_sha256"]
                  and p.get("text_sha256") == _sha_bytes(stage_row["text"].encode("utf-8"))
                  and _family(p.get("family")) == _family(spec["family"]) and p.get("model_id") == spec["model_id"],
                  "provenance_mismatch", f"{rid}: {side} 출처가 샤드와 다르다")
        if j is None:
            continue
        if parse is not None:
            # 구조화 판정 — 샤드의 원 답을 그 꼴의 B 파서(한 정의)로 다시 읽어 판정값 · 답 · 표지 · 꼴 · 파서 판 · 형식 정규화를 맞댄다
            parsed = parse(task, j)
            _need(not parsed["errors"] and isinstance(parsed["verdict"], dict), "judge_json",
                  f"{rid}: 구조화 판정을 읽을 수 없다 — {parsed['errors']}")
            _need(parsed.get("schema") == jschema and parsed.get("parser") == parser_version, "judge_parser",
                  f"{rid}: 파서가 돌려준 꼴 · 판이 {jschema} · {parser_version} 이 아니다")
            _need(all(prov["judge"].get(k) == parsed.get(k) for k in STRUCTURED_FIELDS), "judge_mismatch",
                  f"{rid}: 행의 판정 답 · 표지 · 꼴 · 파서 판 · 형식 정규화가 판정 샤드와 다르다")
            verdict = parsed["verdict"]
        else:
            try:
                verdict = parse_json(j["text"])
            except ValueError as exc:
                raise ReleaseRefused("judge_json", f"{rid}: 판정 JSON 을 읽을 수 없다") from exc
        _need(verdict == prov["judge"]["verdict"], "judge_mismatch", f"{rid}: 행의 판정이 판정 샤드와 다르다")


# ------------------------------------------------------------------------------------------
# 발행 · 검증
# ------------------------------------------------------------------------------------------
def release_run(audit_dir, review_path, rights_path, output_dir, *, judge_source: dict | None = None,
                allowed_judge_sources: tuple[str, ...] = ("separate",), expected_judge_identity_sha256: str | None = None,
                expected_probe_approval_sha256: str | None = None) -> dict:
    """`judge_source` — separate 판정의 객체(`infer.read_judge_source` 의 꼴, `judge_dir` 필수 — 릴리스가 폴더를 다시 읽는다).
    `allowed_judge_sources` — 받을 판정 출처. 기본은 separate 만(총괄 결정 10-12, R-5): 실행 안의 구판정(run_config)은 명시 인자로만.
    `expected_judge_identity_sha256` · `expected_probe_approval_sha256` — 릴리스 호출자가 따로 기록한 기대 핀. separate 이면 하나 이상이
    필요하고, release 가 그대로 판정 신원 읽기에 넘긴다(Codex 최종 API — 핀은 JUDGE.json 에서 추정하지 않는다)."""
    pins = {"expected_judge_identity_sha256": expected_judge_identity_sha256,
            "expected_probe_approval_sha256": expected_probe_approval_sha256}
    audit_dir, output_dir = Path(audit_dir), Path(output_dir)
    _need(not output_dir.exists() and not output_dir.is_symlink(), "output_exists", f"{output_dir} 가 이미 있다")
    audit_path, dataset_path = audit_dir / "AUDIT.json", audit_dir / "dataset.jsonl"
    audit, review, rights = read_json(audit_path), read_json(review_path), read_json(rights_path)
    dataset_sha, rows, texts = _check_all(audit, dataset_path, review, rights)
    _need(audit["judge_identity"]["source"] in allowed_judge_sources, "judge_source_not_allowed",
          f"판정 출처 {audit['judge_identity']['source']!r} 는 이 릴리스에서 받지 않는다(허용 {list(allowed_judge_sources)})")
    check_against_run(audit, audit_dir, rows, texts, judge_source, pins)

    tmp = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    tmp.mkdir(parents=True)
    try:
        shutil.copyfile(dataset_path, tmp / "dataset.jsonl")
        for src, key in ((audit_path, "audit"), (Path(review_path), "review"), (Path(rights_path), "rights")):
            shutil.copyfile(src, tmp / EVIDENCE[key])
        _need(sha256_file(tmp / "dataset.jsonl") == dataset_sha, "copy_changed", "사본 바이트가 다르다")
        receipt = {
            "schema_version": SCHEMA_VERSION, "ready_for_training": True,
            "dataset_sha256": dataset_sha, "run_identity_sha256": audit["run_identity_sha256"],
            "tasks_sha256": audit["tasks_sha256"], "config_sha256": audit["config_sha256"],
            "counts": counts_of(rows),
            "evidence": {k: sha256_file(tmp / name) for k, name in EVIDENCE.items()},
            "reviewer": review["reviewer"], "reviewed_at": review["reviewed_at"],
            "rights_confirmed_by": rights["confirmed_by"], "rights_confirmed_at": rights["confirmed_at"],
            "judge_identity": {"source": audit["judge_identity"]["source"],
                               "identity_sha256": audit["judge_identity"]["identity_sha256"],
                               "caller_expected_pins": pins if audit["judge_identity"]["source"] == "separate" else None},
        }
        write_json_new(tmp / "RELEASE.json", receipt)
        verify_release(tmp)
        os.rename(tmp, output_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {"release_dir": str(output_dir), "dataset_sha256": dataset_sha, "counts": receipt["counts"], "exit_code": 0}


def verify_release(path) -> dict:
    """릴리스 디렉터리만으로 모든 검사를 다시 한다. 통과하면 요약을 돌려주고, 아니면 `ReleaseRefused`."""
    root = Path(path)
    _need(root.is_dir() and not root.is_symlink(), "release_dir", str(root))
    receipt = read_json(root / "RELEASE.json")
    _need(receipt.get("schema_version") == 1 and receipt.get("ready_for_training") is True, "release_receipt",
          "RELEASE.json 이 학습 허가가 아니다")
    evidence = receipt.get("evidence")
    _need(isinstance(evidence, dict) and set(evidence) == set(EVIDENCE), "evidence_changed", "증빙 지문 목록")
    for key, name in EVIDENCE.items():
        _need(evidence[key] == sha256_file(root / name), "evidence_changed", f"{name} 가 발행 때의 바이트가 아니다")
    audit, review, rights = (read_json(root / EVIDENCE[k]) for k in ("audit", "review", "rights"))
    dataset_sha, rows, _ = _check_all(audit, root / "dataset.jsonl", review, rights)
    _need(receipt.get("dataset_sha256") == dataset_sha, "dataset_sha256", "RELEASE 의 dataset_sha256 이 파일과 다르다")
    for key in ("run_identity_sha256", "tasks_sha256", "config_sha256"):
        _need(receipt.get(key) == audit.get(key), "release_identity", key)
    _need(receipt.get("counts") == counts_of(rows), "release_counts", "RELEASE 의 counts 가 데이터셋과 다르다")
    rj = receipt.get("judge_identity")
    _need(isinstance(rj, dict) and rj.get("source") == audit["judge_identity"]["source"]
          and rj.get("identity_sha256") == audit["judge_identity"]["identity_sha256"],
          "release_identity", "RELEASE 의 판정 신원이 AUDIT 와 다르다")
    pins = rj.get("caller_expected_pins")
    if rj["source"] == "separate":
        _need(isinstance(pins, dict) and set(pins) == set(JUDGE_PINS) and any(pins[k] is not None for k in JUDGE_PINS)
              and pins["expected_judge_identity_sha256"] in (None, rj["identity_sha256"])
              and pins["expected_probe_approval_sha256"] in (None, audit["judge_identity"].get("probe_approval_sha256"))
              and pins == audit["judge_identity"].get("caller_expected_pins"), "release_identity",
              "RELEASE 의 판정 핀이 없거나 판정 신원 · 탐침 승인 · AUDIT 의 핀과 다르다")
    else:
        _need(pins is None, "release_identity", "run_config 판정의 영수증에 핀이 있다")
    _need(receipt.get("reviewer") == review["reviewer"] and receipt.get("reviewed_at") == review["reviewed_at"]
          and receipt.get("rights_confirmed_by") == rights["confirmed_by"]
          and receipt.get("rights_confirmed_at") == rights["confirmed_at"],
          "release_evidence", "RELEASE 의 승인 기록이 증빙과 다르다")
    return {"release_dir": str(root), "dataset_sha256": dataset_sha, "run_identity_sha256": receipt["run_identity_sha256"],
            "counts": receipt["counts"], "verified": True}
