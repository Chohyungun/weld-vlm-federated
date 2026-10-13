"""prepare · release 시험의 가짜 세계 — 저장소 뿌리 · 봉인 매니페스트 · 봉인 페어 · 영상 바이트 · 구절 은행 · config.

실데이터는 읽지 않는다. 영상은 영상이 아닌 짧은 바이트다(해시만 쓴다). 매니페스트 스냅샷은 `data.manifest_io.verify_snapshot`,
페어 스냅샷은 `make_pairs_pilot.snapshot_digest` 와 같은 산식으로 봉인해 실제 검증기를 그대로 통과시킨다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from corpus.production.common import canonical_bytes

MANIFEST_REL = "data/interim/manifest_fake"
PAIRS_REL = "data/processed/pairs_fake"

#: (image_id, split, client, group_id, 결함 수) — eval 한 장 · 격리(페어에 없음) 한 장 포함
IMAGES = [
    ("img:0001", "train", "C1", "grp_a", 2),
    ("img:0002", "train", "C1", "grp_a", 0),
    ("img:0003", "train", "C2", "grp_b", 1),
    ("img:0004", "val", "C2", "grp_c", 1),
    ("img:0005", "train", "C3", "grp_d", 0),
    ("img:0006", "val", "C3", "grp_e", 3),
    ("img:0099", "eval", "", "grp_z", 1),
]
QUARANTINED = "img:0007"     # 매니페스트 train 이지만 페어에서 격리된 영상

#: 구절 은행(v3 포인터 꼴, 행 rights_status 는 실제 은행처럼 pending). 넘어가는 것 넷(계보 LG-A 에 문서 둘, 역사 계보 하나),
#: 빠질 것 셋 — 검수 기록 없음(DOC3-P01) · 보류 · 그것을 필수 문맥으로 삼음. 거름은 은행 진입점이 한다.
PASSAGES = [
    {"passage_id": "DOC1-P01", "doc_id": "DOC-1", "lineage_group_id": "LG-A", "text": "Porosity is a cavity formed by trapped gas."},
    {"passage_id": "DOC1-P02", "doc_id": "DOC-1", "lineage_group_id": "LG-A", "text": "Lack of fusion is incomplete bonding.",
     "required_context_passage_ids": ["DOC1-P01"], "generation_constraints": ["Do not turn experimental settings into general rules"]},
    {"passage_id": "DOC2-P01", "doc_id": "DOC-2", "lineage_group_id": "LG-A", "text": "Slag inclusion is trapped non-metallic solid."},
    {"passage_id": "HIST-P01", "doc_id": "HIST-1", "lineage_group_id": "LG-H", "text": "Blowhole.—A hole formed by trapped gases.",
     "flags": ["historical_only"]},
    {"passage_id": "DOC3-P01", "doc_id": "DOC-3", "lineage_group_id": "LG-B", "text": "Undercut is a groove melted into the base metal.",
},
    {"passage_id": "DOC3-P02", "doc_id": "DOC-3", "lineage_group_id": "LG-B", "text": "A paid standard sets a limit here.",
     "source_policy_holds": ["paid_standard_numeric"]},
    {"passage_id": "DOC3-P03", "doc_id": "DOC-3", "lineage_group_id": "LG-B", "text": "Overlap is weld metal over the surface.",
     "required_context_passage_ids": ["DOC3-P01"]},
]
USABLE = ("DOC1-P01", "DOC1-P02", "DOC2-P01", "HIST-P01")
#: 사람 검수 기록이 없는 구절(그래서 진입점이 뺀다)
NO_REVIEW = frozenset({"DOC3-P01"})


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def spec(family: str) -> dict:
    return {"model_id": f"local/{family}", "revision": "a" * 40, "family": family, "local_files_only": True,
            "quantization": "bf16", "batch_size": 2, "max_new_tokens": 64, "max_input_tokens": 8192,
            "seed": 20261007, "wall_seconds": 1800, "enable_thinking": False}


class World:
    def __init__(self, tmp_path: Path, **quality):
        self.tmp = tmp_path
        self.repo = tmp_path / "repo"
        (self.repo / ".git").mkdir(parents=True)
        self.manifest_dir = self.repo / MANIFEST_REL
        self.pairs_dir = self.repo / PAIRS_REL
        self.manifest_rows = []
        for iid, split, client, group, _ in IMAGES + [(QUARANTINED, "train", "C1", "grp_q", 0)]:
            rel = f"data/interim/tiles/RT/ST/{iid.split(':')[1]}.jpg"
            path = self.repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"not-an-image {iid}".encode())
            self.manifest_rows.append({"image_id": iid, "rel_path": rel, "sha256": sha(path.read_bytes()),
                                       "group_id": group, "split": split, "client": client})
        self.write_manifest()
        self.pair_rows = [self.pair(iid, split, client, n) for iid, split, client, _, n in IMAGES if split != "eval"]
        self.bank, self.bank_text = tmp_path / "bank_v3.jsonl", tmp_path / "bank_v3_full.jsonl"
        self.write_bank([self.passage(p) for p in PASSAGES])
        self.write_records([self.passage(p) for p in PASSAGES])
        q = {"minimum_source_qa_train": 9, "target_source_qa_train": 9, "minimum_distinct_train_passages": 3,
             "maximum_questions_per_passage": 3, "minimum_review_per_stratum": 1, "review_fraction": 0.02,
             "human_review_required": True, "no_partial_release": True}
        q.update(quality)
        self.config = {"schema_version": 1, "recipe": "test", "shard_size": 2,
                       "inputs": {"pairs_sha256": None, "pairs_snapshot_digest": None,
                                  "expected_image_train": 4, "expected_image_val": 2,
                                  "source_bank_sha256": self.bank_sha},
                       "quality": q, "generation": spec("qwen"), "judge": spec("phi")}
        self.seal_pairs()

    # -- 봉인 ---------------------------------------------------------------------------
    def write_manifest(self) -> None:
        d = self.manifest_dir
        d.mkdir(parents=True, exist_ok=True)
        with (d / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.manifest_rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(self.manifest_rows)
        (d / "annotations.csv").write_bytes(b"image_id\n")
        (d / "data_capabilities.yaml").write_bytes(b"fake: true\n")
        lines = [f"{sha((d / n).read_bytes())}  {n}" for n in ("manifest.csv", "annotations.csv", "data_capabilities.yaml")]
        self.manifest_digest = sha(("\n".join(lines) + "\n").encode())
        (d / "SNAPSHOT.sha256").write_text("\n".join(lines) + f"\n# snapshot_digest {self.manifest_digest}\n",
                                           encoding="utf-8", newline="\n")

    def seal_pairs(self, *, update_config: bool = True) -> None:
        d = self.pairs_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / "pairs.jsonl").write_bytes(b"".join(json.dumps(r, ensure_ascii=False).encode() + b"\n" for r in self.pair_rows))
        meta = {"input_snapshot": {"path": MANIFEST_REL, "snapshot_digest": self.manifest_digest}}
        (d / "PAIRS_META.json").write_bytes(json.dumps(meta).encode())
        entries = [(sha((d / n).read_bytes()), n) for n in ("pairs.jsonl", "PAIRS_META.json")]
        digest = sha("".join(h for h, _ in entries).encode())
        (d / "SNAPSHOT.sha256").write_text("".join(f"{h}  {n}\n" for h, n in entries) + f"# snapshot_digest {digest}\n",
                                           encoding="utf-8", newline="\n")
        if update_config:
            self.config["inputs"]["pairs_sha256"] = entries[0][0]
            self.config["inputs"]["pairs_snapshot_digest"] = digest

    # -- 행 -------------------------------------------------------------------------------
    @staticmethod
    def pair(iid: str, split: str, client: str, n: int) -> dict:
        defects = [{"type": "2011" if k % 2 == 0 else "401", "bbox_px": [40 + 300 * k, 344, 120 + 300 * k, 408],
                    "size_px": {"major_axis": 82.05, "equiv_diameter": 64.31}, "size_mm": None} for k in range(n)]
        clauses = ["KRA27-T15"] if n else []
        return {"image_id": iid, "image_path": f"data/interim/tiles/RT/ST/{iid.split(':')[1]}.jpg", "client": client,
                "split": split, "material": "ST", "width_px": 1280, "height_px": 720, "coord_space": "ABS_ORIG",
                "skeleton": {"defects": defects, "verdict": None, "verdict_mode": "clause_only", "clauses": clauses,
                             "candidate_rules": [{"rule_id": "KRA27-T15-01", "thickness_max": "10.01"}] if n else [],
                             "uncited_codes": []},
                "target_text": "두께 10 mm 이하: 4 mm 이하" if n else "결함이 관찰되지 않는다."}

    @staticmethod
    def passage(p: dict) -> dict:
        """본문 은행 행(코퍼스 측의 행 꼴 2판). 포인터 행은 이것에서 본문만 null 로."""
        from corpus.parse.passage_bank import text_sha256

        row = {"schema_version": 2, "doc_id": p["doc_id"], "edition": "test edition", "source_file_sha256": sha(p["doc_id"].encode()),
               "locator": {"pdf_page_start": 3, "pdf_page_end": 3}, "text_kind": "verbatim_extract",
               "text_sha256": text_sha256(p["text"]), "facets": [], "defect_classes": [], "flags": [],
               "scope": f"{p['doc_id']} 의 서술 범위", "rights_status": "pending", "human_review": None}
        row.update({k: v for k, v in p.items()})
        return row

    def write_bank(self, rows: list[dict], *, report: bool = True) -> None:
        self.bank_text.write_bytes(b"".join(canonical_bytes(r) + b"\n" for r in rows))
        self.bank.write_bytes(b"".join(canonical_bytes({**r, "text": None}) + b"\n" for r in rows))
        self.bank_sha = sha(self.bank.read_bytes())
        if report:
            (self.tmp / "bank_v3.build_report.json").write_bytes(canonical_bytes(
                {"outputs": {"pointer_bank_sha256": self.bank_sha, "full_bank_sha256": sha(self.bank_text.read_bytes())}}))
        if hasattr(self, "config"):
            self.config["inputs"]["source_bank_sha256"] = self.bank_sha

    def write_records(self, rows: list[dict], *, no_review=NO_REVIEW, rights: bool = True) -> None:
        """등록부 · 사용자 권리 결정(문서마다 qa_generation 승인) · 사람 검수 기록(no_review 밖의 구절, 위험 표지 확인)."""
        import yaml

        docs = sorted({(r["doc_id"], r["source_file_sha256"]) for r in rows})
        self.registry = self.tmp / "world_sources.yaml"
        self.registry.write_text(yaml.safe_dump({"documents": [{"doc_id": d, "sha256": h} for d, h in docs]}), encoding="utf-8")
        self.rights = self.tmp / "world_rights.jsonl"
        self.rights.write_bytes(b"".join(canonical_bytes(
            {"doc_id": d, "source_file_sha256": h, "rights_status": "approved", "approved_uses": ["qa_generation"],
             "decided_by": "결정자", "decided_at": "2026-10-08T09:00:00+09:00"}) + b"\n" for d, h in docs) if rights else b"")
        self.review = self.tmp / "world_review.jsonl"
        self.review.write_bytes(b"".join(canonical_bytes(
            {"passage_id": r["passage_id"], "text_sha256": r["text_sha256"], "decision": "approved", "reviewer": "검수자",
             "reviewed_at": "2026-10-08T09:30:00+09:00", "flags_checked": list(r.get("flags") or [])}) + b"\n"
            for r in rows if r["passage_id"] not in no_review))

    def bank_kwargs(self) -> dict:
        return {"source_text_path": self.bank_text, "source_registry_path": self.registry,
                "source_rights_path": self.rights, "source_review_path": self.review}

    def prepare(self, run_name: str = "run", **over):
        from corpus.production.prepare import prepare_run

        return prepare_run(self.pairs_dir / "pairs.jsonl", self.bank, self.config, self.tmp / run_name, repo_root=self.repo,
                           **{**self.bank_kwargs(), **over})


# -- infer 의 가짜 백엔드 -------------------------------------------------------------------
def _judge_answer(prompt: str) -> dict:
    """판정 입력이 구조화 판정이면(`expected_output_keys`) 그 키로 모두 통과하는 답을, 아니면 옛 네 키로 답한다.

    통과 답 — 주장 · 단정을 묻는 물음은 거짓, 나머지는 참, `reason` 은 한 문장. 키는 판정 정책이 정하므로 v6 결함 일곱 키 ·
    정상 여섯 키 · 원문 QA(옛 네 키)에 같이 맞는다."""
    try:
        keys = json.loads(prompt).get("expected_output_keys")
    except (ValueError, AttributeError):
        keys = None
    if not keys:
        return {"grounded": True, "scope_ok": True, "language_ok": True, "reason": "근거와 일치"}
    negative = {"claims_defect", "asserts_verdict", "asserts_verdict_dimension_or_cause"}
    return {k: ("근거와 일치" if k == "reason" else k not in negative) for k in keys}


class FakeBackend:
    """infer.run_stage 의 backend_factory. `backend` 로 샤드에 적히는 이름을 바꾼다(transformers 를 흉내 내는 시험용)."""

    backend: ClassVar[str] = "transformers"

    def __init__(self, cfg, stage):
        self.stage = stage
        # 따로 둔 판정(`infer.run_judge`)이 적재된 모델에서 읽는 실행 조건의 흉내 — 시험 전용(코퍼스 측의 품질 시험과 같은 꼴)
        self.model = SimpleNamespace(generation_config=SimpleNamespace(to_dict=lambda: {"eos_token_id": 2}, eos_token_id=2))
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, chat_template="{{ messages }}")
        self.identity = {"backend": self.backend, "model_id": cfg["model_id"], "revision": cfg["revision"],
                         "family": cfg["family"], "model_type": cfg["family"], "quantization": cfg["quantization"],
                         "tokenizer_sha256": "c" * 64}

    def generate_batch(self, prompts, system, deadline):
        out = []
        for p in prompts:
            if self.stage == "judge":
                text = json.dumps(_judge_answer(p), ensure_ascii=False)
            elif "구절 id:" in p:
                pid = re.search(r"구절 id: (\S+)", p).group(1)
                text = json.dumps({"question": f"{pid} 는 무엇을 설명하는가?", "answer": "구절이 정의하는 결함을 설명한다.",
                                   "source_ids": [pid]}, ensure_ascii=False)
            else:
                text = "영상 가운데 부근에 결함이 표시되어 있다. 두께와 실치수가 없어 합부는 판정하지 않는다."
            out.append({"text": text, "finish_reason": "eos", "input_tokens": 20, "output_tokens": 15})
        return out

    def close(self):
        pass


def make_judge_source(run_dir: Path, config: dict, judge_dir: Path, *, model_id: str = "local/judge-v2") -> dict:
    """따로 둔 판정을 판정 신원 쓰기(`infer.run_judge`, 가짜 백엔드)로 실제 폴더에 돌리고, 판정 신원 읽기로 객체를 받는다(시험 전용)."""
    from corpus.production import infer

    # 판정 신원 쓰기는 주입한 백엔드로 시험용(fake)만 받는다 — 그래서 이 판정의 행은 릴리스되지 않는다(릴리스는 transformers 만)
    FakeBackend.backend = "fake"
    try:
        done = infer.run_judge(run_dir, config, judge_dir, judge_config={**config["judge"], "model_id": model_id},
                               backend_factory=FakeBackend)
    finally:
        FakeBackend.backend = "transformers"
    # 릴리스는 separate 에 외부 핀을 요구한다 — 실행이 돌려준 신원을 외부 기록으로 삼아 같은 핀으로 읽는다(시험 전용)
    return infer.read_judge_source(run_dir, config, judge_dir, expected_judge_identity_sha256=done["judge_identity_sha256"])


def build_audit(run_dir: Path, config: dict, audit_dir: Path, *, mutate_rows=None, judge_source=None) -> dict:
    """B 의 `audit_run` 이 낼 꼴(보고 05 공유 계약 안)을 흉내 낸다 — 시험 전용 참조 구현."""
    from corpus.production import infer, quality
    from corpus.production.common import read_json, write_json_new
    from corpus.production.release import counts_of, ids_sha256

    run = read_json(run_dir / "RUN.json")
    tasks = [json.loads(x) for x in (run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    gen = infer.read_stage_rows(run_dir, "generate", config)
    if judge_source is None:
        judge = infer.read_stage_rows(run_dir, "judge", config)
        judge_identity = {"source": "run_config", "identity_sha256": None, "identity_basis": None,
                          "model": {"model_id": config["judge"]["model_id"], "family": config["judge"]["family"]}}
    else:
        by_id = {r["id"]: r for r in judge_source["rows"]}
        judge = [by_id[t["id"]] for t in tasks]
        judge_identity = {"source": "separate", "identity_sha256": judge_source["identity_sha256"],
                          "identity_basis": judge_source["identity_basis"], "model": dict(judge_source["identity_basis"]["judge_model"]),
                          "parser_schema": judge_source["identity_basis"]["policy_identity"]["output_schema"],
                          "caller_expected_pins": judge_source.get("caller_expected_pins"),
                          **{k: judge_source.get(k) for k in ("probe_approval_sha256", "gpu_model_verified", "pin_provenance")}}
    rows = []
    for t, g, j in zip(tasks, gen, judge):
        if t["kind"] == "image_report":
            user = [{"type": "image", "image": t["image_path"]}, {"type": "text", "text": t["train_user_text"]}]
            assistant = [{"type": "text", "text": g["text"].strip()}]
        else:
            qa = json.loads(g["text"])
            user, assistant = [{"type": "text", "text": qa["question"].strip()}], [{"type": "text", "text": qa["answer"].strip()}]
        row = {k: t[k] for k in ("id", "kind", "split", "client", "lineage_group_id", "image_path", "source_ids")}
        if t["kind"] == "image_report":
            row["image_sha256"] = t["image_sha256"]
        row["messages"] = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        row["provenance"] = {
            side: {"backend": s["_backend"], "family": m["family"], "model_id": m["model_id"],
                   "shard_sha256": s["_shard_sha256"], "text_sha256": sha(s["text"].encode())}
            for side, s, m in (("generate", g, config["generation"]), ("judge", j, judge_identity["model"]))}
        if judge_source is None:
            row["provenance"]["judge"]["verdict"] = json.loads(j["text"])
        else:
            # 코퍼스 측의 집계와 같은 짝 — 출력 계약이 v6 면 v6 파서, v5 면 v5 파서(한 꼴을 다른 파서로 읽지 않는다)
            parse = quality.parse_judge_v6 if judge_identity["parser_schema"] == quality.JUDGE_V6_SCHEMA_VERSION else quality.parse_judge_v5
            parsed = parse(t, j)
            row["provenance"]["judge"].update(verdict=parsed["verdict"], schema=parsed["schema"], parser=parsed["parser"],
                                              answers=parsed["answers"], flags=list(parsed["flags"]),
                                              format_normalized=list(parsed["format_normalized"]))
            for side in ("generate", "judge"):        # 시험용(fake) 판정 — 행 검사를 지나 폴더 대조까지 가게 transformers 로 적는다
                row["provenance"][side]["backend"] = "transformers"
        row["provenance"]["judge"]["identity_sha256"] = judge_identity["identity_sha256"]
        rows.append(row)
    if mutate_rows:
        rows = mutate_rows(rows)
    audit_dir.mkdir(parents=True)
    (audit_dir / "dataset.jsonl").write_bytes(b"".join(canonical_bytes(r) + b"\n" for r in rows))
    strata: dict[str, list] = {}
    sizes: dict[str, int] = {}
    for r in rows:
        key = f"{r['kind']}|{r['client']}"
        sizes[key] = sizes.get(key, 0) + 1
        if len(strata.setdefault(key, [])) < 1:
            strata[key].append(r["id"])
    docs = sorted({(t["evidence"]["source_id"], t["evidence"]["source_sha256"]) for t in tasks if t["kind"] == "source_qa"})
    audit = {
        "schema_version": 1, "complete": True, "errors": 0, "rejected": 0, "incomplete": 0,
        "status": "ready_for_human_review", "blockers": [],
        "run_dir": str(run_dir), "run_identity_sha256": run["identity_sha256"], "run_identity_basis": run["identity_basis"],
        "tasks_sha256": run["tasks_sha256"], "config_sha256": run["config_sha256"], "config": run["config"],
        "dataset_sha256": sha((audit_dir / "dataset.jsonl").read_bytes()),
        "planned": {"count": len(rows), "ids_sha256": ids_sha256([r["id"] for r in rows])},
        "counts": counts_of(rows),
        "user_templates": {"image_report": sorted({t["train_user_text"] for t in tasks if t["kind"] == "image_report"})},
        "source_documents": [{"source_id": s, "source_sha256": h} for s, h in docs],
        "judge_identity": judge_identity,
        # 코퍼스 측의 집계(quality_v7 ~)가 싣는 파서 판 — 구조화 꼴이면 그 꼴의 판, 옛 네 키면 None
        "judge_parser_version": quality.JUDGE_PARSER_VERSIONS.get(judge_identity.get("parser_schema", "legacy_four_keys")),
        "review": {"strata": strata, "stratum_sizes": sizes},
    }
    write_json_new(audit_dir / "AUDIT.json", audit)
    return audit


def review_for(audit: dict, audit_dir: Path) -> dict:
    """사람이 채운 검수 기록의 흉내 — 표본마다 데이터셋 행의 지문(정규형 sha256)을 싣는다(B 2판의 REVIEW_TEMPLATE 꼴)."""
    rows = {r["id"]: r for r in map(json.loads, (audit_dir / "dataset.jsonl").read_text(encoding="utf-8").splitlines())}
    ids = [i for v in audit["review"]["strata"].values() for i in v]
    return {"schema_version": 1, "reviewer": "검수자 갑", "reviewed_at": "2026-10-08T10:00:00+09:00",
            "target_dataset_sha256": audit["dataset_sha256"], "approved": True,
            "samples": [{"id": i, "decision": "approve", "comment": "", "row_sha256": sha(canonical_bytes(rows[i])),
                         "reviewer": None, "reviewed_at": None} for i in ids if i in rows]}


def rights_for(audit: dict) -> dict:
    basis = audit["run_identity_basis"]
    shas = [basis["pairs_sha256"], basis["manifest_snapshot_digest"], basis["source_bank_sha256"]]
    shas += [d["source_sha256"] for d in audit["source_documents"]]
    return {"schema_version": 1, "confirmed_by": "확인자 을", "confirmed_at": "2026-10-08T11:00:00+09:00",
            "assets": [{"sha256": s, "use": ["sft_training"], "basis": "시험"} for s in shas]}
