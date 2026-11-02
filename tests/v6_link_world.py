"""판 6 연결(2 판 명세 §2 ~ §4)의 합성 세계 — 명시한 파일만 임시 폴더에 복사하고 실제 git 저장소로 커밋한다.

ROOT/data 를 걷지 않는다. 모델 · GPU · 이미지 열기 0. 등록 · 곁 파일 · 출처 운반 파일은 평가 쪽 작성기 · 도출 함수가 지은 바이트 그대로이고,
시험은 그 바이트를 바꿔 거부 사유를 본다. 커밋 소비 시험은 임시 저장소의 실제 blob 을 읽는다 — 단순 dict 대조로 바꾸지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

from evaluation import eval_registration as ER
from evaluation import narrative_checker as NC
from evaluation import narrative_checker_v4 as NC4
from evaluation import narrative_registration as NR
from evaluation import sample_weights as SW
from evaluation.narrative_conformance import VECTOR_REL
from evaluation.prereg_unified import dump_registration

REPO = Path(__file__).resolve().parents[1]
STEM = "main-20261029-1"
REG_REL = f"{ER.REG_DIR}/{STEM}.json"
PROMPT_REL = "vlm/prompts/unified_v3_1_basis.txt"
PINNED: tuple[str, ...] = tuple(dict.fromkeys((
    NR.EXTRACTOR_REL, VECTOR_REL, NR.CHECKER_REL, "evaluation/narrative_checker_v4.py", NC.LISTS_REL, NC4.LISTS_REL,
    NC.LIMITS_REL, NC4.LIMITS_REL, *NR.LIMITS_LOADER_FILES, *NR.GENERATOR_FILES_V1, "configs/label_map.yaml",
    NR.TRAIN_EXTRACTOR_REL, NR.INPUTS_CONTRACT_REL, ER.CONFIG_REL, ER.BASE_REL, ER.TRAIN_REG_REL, PROMPT_REL,
    *[r.path for r in NR.V6_EVIDENCE_V4],
)))
"""세계에 복사하는 파일 — 신원 대조 · 프로필 · 도출 · 출처 운반에 드는 것 전부(명시 목록)."""


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                           "-c", "core.autocrlf=false", *args], capture_output=True, check=True).stdout.decode("utf-8").strip()


def copy_pinned(root: Path, rels=PINNED) -> None:
    for rel in rels:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((REPO / rel).read_bytes())


def inputs(root: Path) -> dict:
    """도출 함수의 네 입력 — 세계의 1 차 설정 · base · 학습 등록 · 프롬프트 바이트."""
    source = ER.load_eval_registration((root / ER.CONFIG_REL).read_bytes())
    base = ER.load_base((root / ER.BASE_REL).read_bytes())
    train = yaml.safe_load((root / ER.TRAIN_REG_REL).read_bytes())
    prompt_raw = (root / base["fixed_before_main_runs"]["uni_prompt_path"]).read_bytes()
    return {"source_doc": source, "base_doc": base, "train_doc": train, "prompt_raw": prompt_raw}


def regime2_doc(root: Path) -> dict:
    return ER.build_regime2_eval(**inputs(root))


def eval_bytes(doc: dict) -> bytes:
    from scripts.prepare_eval_regime2 import candidate_bytes

    return candidate_bytes(doc)


def bind_to_doc(reg, doc: dict, root: Path):
    """등록의 모집단 · 대표 지표 · 시드 · 생성 한도(· 프롬프트 sha)를 평가 설정 `doc` 과 결속한다 — `strict_binding_problems` 가 비도록."""
    sealed = doc["population"]["sample"]["sealed"][str(doc["population"]["sample"]["main_size"])]
    rep = doc["representative"]
    population = replace(reg.population, weighting=SW.WEIGHTING_SAMPLE, cluster_unit="group",
                         eval_list_file_sha256=sealed["file_sha256"], eval_list_set_sha256=sealed["set_sha256"],
                         n_images=sealed["images"], n_clusters=sealed["groups"])
    metric = replace(reg.metric, metric_id=rep["metric_id"], iou_threshold=rep["iou_threshold"], coupling_rule=rep["coupling_rule"],
                     sensitivity_variant_id=rep["sensitivity_variant"], macro_support_rule=rep["macro_support_rule"],
                     class_codes=tuple(rep["class_codes"]))
    gen = {"max_new_tokens": doc["generation"]["max_new_tokens"], "eval_list_file_sha256": sealed["file_sha256"],
           "eval_list_set_sha256": sealed["set_sha256"]}
    if doc["generation"].get("prompt_file"):
        gen["prompt_sha256"] = NR.prompt_sha256(root / doc["generation"]["prompt_file"])
    return replace(reg, population=population, metric=metric, generation=replace(reg.generation, **gen),
                   ci=replace(reg.ci, cluster_unit="group"), seeds={str(s): 20260827 + s for s in doc["seeds"]},
                   planned_seed_count=len(doc["seeds"]))


def regime2_registration(root: Path, doc: dict | None = None):
    """2 차 체제의 완결 판 6 등록 — 서술 · 조항 축은 프로필 작성기, 모집단 · 대표 지표 · 시드 · 한도는 2 차 설정과 결속."""
    from tests.test_prereg_v6 import v6

    doc = doc or regime2_doc(root)
    reg = replace(v6(), narrative=NR.narrative_block(root, profile=NR.PROFILE_REGIME2),
                  clause_axis=NR.clause_axis_block(root, profile=NR.PROFILE_REGIME2))
    return bind_to_doc(reg, doc, root)


def legacy_registration(root: Path, doc: dict, *, version6: bool = True):
    """구판 등록 — 판 6 이면 v3.1 블록(확인 자료 셋은 합성 파일), 판 5 면 블록 없음. 1 차 설정 `doc` 과 결속."""
    from tests.test_prereg_unified import filled
    from tests.test_prereg_v6 import v6

    if not version6:
        return bind_to_doc(filled(), doc, root)
    synthetic = root / "synthetic_v6_inputs.json"
    synthetic.write_bytes(b'{"synthetic": true}\n')
    reg = replace(v6(), narrative=NR.narrative_block(root, confirmation_path=synthetic, checker_vectors_path=synthetic,
                                                     inputs_contract_path=synthetic),
                  clause_axis=NR.clause_axis_block(root))
    return bind_to_doc(reg, doc, root)


def commit_files(root: Path, files: dict[str, bytes | None], message: str = "등록") -> str:
    """파일을 쓰고(값 None 이면 지우고) 한 커밋 — HEAD sha 를 돌려준다."""
    for rel, raw in files.items():
        if raw is None:
            (root / rel).unlink(missing_ok=True)
            continue
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(raw)
    if not (root / ".git").exists():
        git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", message)
    return git(root, "rev-parse", "HEAD")


def build(root: Path, *, with_sidecar: bool = True, with_sources: bool = True, eval_raw: bytes | None = None,
          sources_raw: bytes | None = None, registration=None, stem_rel: str = REG_REL) -> SimpleNamespace:
    """코드 · 설정을 복사해 첫 커밋, 등록 · 곁 파일 · 출처 운반 파일을 둘째 커밋으로 — `head` 가 등록 anchor 다."""
    copy_pinned(root)
    commit_files(root, {}, "코드 · 설정")
    doc = regime2_doc(root)
    reg = registration if registration is not None else regime2_registration(root, doc)
    reg_raw = dump_registration(reg)
    eval_raw = eval_bytes(doc) if eval_raw is None else eval_raw
    files = {stem_rel: reg_raw}
    if with_sidecar:
        files[ER.eval_sidecar_rel(stem_rel)] = eval_raw
    if with_sources:
        files[ER.v6_sources_rel(stem_rel)] = (NR.build_v6_sources(registration_raw=reg_raw, eval_raw=eval_raw, repo=root)
                                              if sources_raw is None else sources_raw)
    head = commit_files(root, files)
    return SimpleNamespace(root=root, doc=doc, reg=reg, reg_raw=reg_raw, eval_raw=eval_raw, files=files, head=head,
                           stem_rel=stem_rel)


def sources_doc(raw: bytes) -> dict:
    return json.loads(raw.decode("utf-8"))
