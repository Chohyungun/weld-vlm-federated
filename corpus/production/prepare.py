"""작업 명부 — 봉인 페어와 구절 은행을 검사해 생성 · 판정의 작업 명부(`tasks.jsonl`)와 `RUN.json` 을 만든다(CPU).

파일 계약은 `docs/dev_log/2026-10-07-본실험코퍼스/01_Claude_실행분담.md` 의 "파일 계약" 이 정본이다.

- 봉인 페어(`pairs.jsonl` · `SNAPSHOT.sha256`)의 지문을 config 의 값과 대조하고, 페어가 가리키는 봉인 매니페스트의 스냅샷을 검증한다.
- 페어를 **스트리밍**하며 행마다 매니페스트와 대조한다 — split · client 가 다르면(재배정) 거부, eval 영상 · eval 묶음이 섞이면 거부.
  `lineage_group_id` 는 매니페스트의 실제 `group_id`, `image_sha256` 은 실제 파일에서 낸다(매니페스트의 값과 다르면 거부).
- 기대 수(train 44,845 · val 5,000)와 다르면 거부한다. 원문 QA 가 하한에 못 미치면 **채우지 않고** 수를 그대로 적는다.
- 구절 거름은 은행의 진입점 하나다 — `corpus.parse.passage_bank.select_bank_for_generation`(권리 결정 · 사람 검수 · 위험 표지 · 유료 문장 ·
  보류 · 필수 문맥 사슬 · 본문 대조). 여기서는 config 에 등록한 포인터 지문만 먼저 맞추고, 그 요약을 RUN.json 에 그대로 싣는다.
  필수 문맥은 사슬 전체(폐쇄)를 근거와 생성 지시에 넣는다.
- 원문 QA 의 train/val 은 은행 계보(`lineage_group_id`) 단위로 가른다(`_source_split`).
- eval 영상 파일과 GPU 는 열지 않는다. 매니페스트의 eval 행은 id · group_id 를 거르는 데에만 읽는다.
- 새 디렉터리에만 쓴다(있으면 거부). 쓰기 전에 본체 게이트가 닫혔는지 본다.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from .common import (
    assert_gate_closed,
    canonical_bytes,
    iter_jsonl,
    read_json,
    sha256_file,
    write_json_new,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
#: 원문 QA 분할(지시 1008b) — 은행 계보 단위. 계보를 sha256 순위로 훑어 val 목표(적격 구절의 10 %, 반올림)를 넘지 않게 담되,
#: 한 문서의 구절이 목표의 50 % 를 넘게 되는 계보는 건너뛴다. 끝에 실제 val 에서 한 문서가 50 % 를 넘으면 그 문서를 가진
#: 마지막 계보부터 뺀다. 계보는 한 split 에만 속한다(분담 문서의 조건).
SOURCE_VAL_FRACTION = 0.1
SOURCE_VAL_DOC_CAP = 0.5
#: 같은 바이트 영상(06b I-1)의 정책 — 정책은 B 가 낸다. A 는 받을 자리만 둔다: 같은 계보 안은 허용하고 수를 세며, 다른 계보면 거부.
DUPLICATE_IMAGE_POLICY = "same_lineage_allowed"
#: 원문 QA 의 질문 관점(구절당 최대 3 — config 의 maximum_questions_per_passage 와 함께 쓴다)
SOURCE_PERSPECTIVES = (
    "구절이 정의하거나 설명하는 대상이 무엇인지 묻는 질문",
    "구절이 밝힌 조건 · 원인 · 과정 가운데 하나를 묻는 질문(구절에 적힌 범위 안에서만)",
    "구절의 적용 범위나 한계 · 주의점을 묻는 질문(구절에 적힌 범위 안에서만)",
)
#: 영상 설명의 학습 user message — 모든 image_report 작업에 같은 문장(B 05b 회신 8번). 숫자 · 괄호 · 결함 이름을 넣지 않는다
#: (정답 골격 누출 금지). 생성 prompt 는 학습에 쓰지 않는다.
TRAIN_USER_TEXT = "이 방사선투과 영상에 표시된 용접 결함을 한국어로 설명하라."


class PrepareRefused(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.exit_code = 2
        super().__init__(f"{code}: {message}")


# ------------------------------------------------------------------------------------------
# 입력 검사
# ------------------------------------------------------------------------------------------
def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _code_sha256() -> str:
    return _sha_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))


def _pairs_dir(pairs_path: Path) -> Path:
    p = Path(pairs_path)
    return p.parent if p.is_file() else p


def verify_pairs_snapshot(pairs_dir: Path, inputs: dict) -> dict:
    """봉인 페어 — 구성원 해시 · 결합 다이제스트(파일 해시 문자열을 이어 붙여 해싱, `make_pairs_pilot.snapshot_digest`)를
    다시 내어 계약 파일과 config 에 대조한다."""
    snap = pairs_dir / "SNAPSHOT.sha256"
    if not snap.is_file():
        raise PrepareRefused("pairs_unsealed", f"{snap} 이 없다")
    entries, recorded = [], None
    for line in snap.read_text(encoding="utf-8").splitlines():
        if line.startswith("# snapshot_digest"):
            recorded = line.split()[-1]
            continue
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        entries.append((digest, name.strip()))
    for digest, name in entries:
        if sha256_file(pairs_dir / name) != digest:
            raise PrepareRefused("pairs_member_changed", name)
    combined = _sha_bytes("".join(d for d, _ in entries).encode())
    if recorded != combined:
        raise PrepareRefused("pairs_snapshot_digest", "계약 파일의 다이제스트와 구성원이 맞지 않는다")
    if combined != inputs.get("pairs_snapshot_digest"):
        raise PrepareRefused("pairs_snapshot_digest", "config 의 pairs_snapshot_digest 와 다르다")
    members = {n: d for d, n in entries}
    if members.get("pairs.jsonl") != inputs.get("pairs_sha256"):
        raise PrepareRefused("pairs_sha256", "config 의 pairs_sha256 과 다르다")
    meta_path = pairs_dir / "PAIRS_META.json"
    if "PAIRS_META.json" not in members:
        raise PrepareRefused("pairs_meta", "PAIRS_META.json 이 봉인 구성원이 아니다")
    meta = read_json(meta_path)
    return {"pairs_sha256": members["pairs.jsonl"], "pairs_snapshot_digest": combined, "meta": meta}


def verify_manifest_snapshot(manifest_dir: Path) -> str:
    """봉인 매니페스트 — `data.manifest_io.verify_snapshot`(매니페스트의 산식)을 그대로 부른다."""
    from data.manifest_io import SnapshotVerificationError, verify_snapshot

    try:
        return verify_snapshot(manifest_dir)
    except (SnapshotVerificationError, OSError) as exc:
        raise PrepareRefused("manifest_snapshot", str(exc)) from exc


def read_manifest(manifest_dir: Path) -> tuple[dict[str, dict], set[str], set[str]]:
    """image_id → {split, client, group_id, rel_path, sha256}. eval 행은 id · group 만 걸러 낸다(영상은 열지 않는다)."""
    rows: dict[str, dict] = {}
    eval_ids: set[str] = set()
    eval_groups: set[str] = set()
    with (manifest_dir / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "eval":
                eval_ids.add(r["image_id"])
                eval_groups.add(r["group_id"])
                continue
            rows[r["image_id"]] = {"split": r["split"], "client": r["client"], "group_id": r["group_id"],
                                   "rel_path": r["rel_path"], "sha256": r["sha256"]}
    return rows, eval_ids, eval_groups


def _flag_names(row: dict) -> set[str]:
    return {str(f).split(":", 1)[0] for f in row.get("flags") or []}


def _context_closure(pid: str, by_id: dict[str, dict]) -> list[str]:
    """필수 문맥의 사슬 전체 — 방문 집합으로 순환을 닫는다(서로를 문맥으로 삼는 정상 묶음이 있다). 자기 자신은 뺀다. id 순."""
    seen: set[str] = set()
    stack = list(by_id[pid].get("required_context_passage_ids") or [])
    while stack:
        c = stack.pop()
        if c in seen or c == pid:
            continue
        if c not in by_id:
            raise PrepareRefused("source_context_missing", f"{pid}: 필수 문맥 {c} 가 넘어온 구절에 없다 — 은행 진입점의 계약 위반")
        seen.add(c)
        stack.extend(by_id[c].get("required_context_passage_ids") or [])
    return sorted(seen)


def select_source_passages(pointer_path: Path, text_path, expected_sha256: str | None, *, registry_path=None,
                           rights_path=None, review_path=None) -> tuple[list[dict], dict]:
    """구절 거름은 은행의 진입점 하나다(B 05c). config 에 등록한 포인터 지문만 먼저 맞추고, 사슬 문맥의 본문을 붙인다.
    두 번째 값은 진입점의 요약(넘김 · 뺌 · 사유별 수 · 입력 여섯의 지문) — RUN.json 에 그대로 싣는다."""
    from corpus.parse.passage_bank import DEFAULT_REGISTRY, BankRefused, select_bank_for_generation

    if not expected_sha256:
        raise PrepareRefused("source_bank_unregistered", "config 의 source_bank_sha256 이 비었다 — 포인터 은행의 지문을 먼저 등록한다")
    if sha256_file(pointer_path) != expected_sha256:
        raise PrepareRefused("source_bank_sha256", "포인터 은행의 지문이 config 와 다르다")
    try:
        selected, summary = select_bank_for_generation(
            pointer_path, None if text_path is None else Path(text_path), registry=registry_path or DEFAULT_REGISTRY,
            rights_path=rights_path, review_path=review_path, use="qa_generation")
    except BankRefused as exc:
        raise PrepareRefused(f"source_bank:{exc.code}", str(exc)) from exc
    by_id = {r["passage_id"]: r for r in selected}
    passages = []
    for r in selected:
        closure = _context_closure(r["passage_id"], by_id)          # 진입점이 사슬 전체가 넘어갈 때만 넘기므로 모두 by_id 에 있다
        passages.append({**r, "context_texts": {c: by_id[c]["text"] for c in closure}})
    return passages, summary


# ------------------------------------------------------------------------------------------
# 작업 행
# ------------------------------------------------------------------------------------------
def _label_names() -> dict[str, str]:
    from data.label_map import load_label_map

    lm = load_label_map()
    out = {}
    for dt in lm.defect_types.values():
        for code in (dt.iso_code, *dt.iso_code_alt):
            out[code] = dt.name_ko
    return out


def _locked_target(skeleton: dict) -> dict:
    defects = [{"type": d["type"], "bbox_px": list(d["bbox_px"])} for d in skeleton.get("defects") or []]
    return {"defects": defects, "defect_counts": dict(sorted(Counter(d["type"] for d in defects).items())),
            "verdict": skeleton.get("verdict"), "clauses": list(skeleton.get("clauses") or [])}


_MATERIAL_KO = {"ST": "강재", "AL": "알루미늄"}


def _where(bbox: list, width: int, height: int) -> str:
    """bbox 중심의 거친 위치 — 숫자 없이 설명하게 한다(가로 · 세로 삼등분)."""
    cx, cy = (bbox[0] + bbox[2]) / 2 / width, (bbox[1] + bbox[3]) / 2 / height
    col = "왼쪽" if cx < 1 / 3 else ("오른쪽" if cx >= 2 / 3 else "가운데")
    row = "위" if cy < 1 / 3 else ("아래" if cy >= 2 / 3 else "가운데")
    return f"{row}-{col}"


def _image_evidence(rec: dict, names: dict[str, str]) -> dict:
    """생성 · 판정 모델이 보는 근거. 허용치 표의 수치(`target_text` · `candidate_rules`)는 넣지 않는다 —
    수치 · 코드 · bbox · 조항은 모델이 쓰지 않고 `locked_target` 에서 결속한다."""
    sk = rec["skeleton"]
    defects = sk.get("defects") or []
    unknown = sorted({d["type"] for d in defects} - set(names))
    if unknown:
        raise PrepareRefused("label_code", f"{rec['image_id']}: 사상표에 없는 코드 {unknown}")
    w, h = int(rec["width_px"]), int(rec["height_px"])
    return {
        "material": _MATERIAL_KO.get(rec.get("material"), rec.get("material")),
        "defect_counts_by_name": dict(sorted(Counter(names[d["type"]] for d in defects).items())),
        "defect_positions": [{"name": names[d["type"]], "where": _where(d["bbox_px"], w, h)} for d in defects],
        "has_applicable_clause": bool(sk.get("clauses")),
        "size_mm_available": False,
        "limitation": "모재 두께와 화소당 실치수 정보가 없어 치수 기준의 합부 판정은 내리지 않는다.",
    }


def _image_prompt(evidence: dict) -> str:
    return (
        "아래 주석은 방사선투과 영상 한 장에 표시된 용접 결함이다. 이 주석만을 근거로 영상을 설명하는 한국어 문단 하나를 써라.\n"
        "- 결함 이름 · 개수 · 대략의 위치(위 · 가운데 · 아래, 왼쪽 · 가운데 · 오른쪽)만 쓴다. 쓸 수 있는 숫자는 결함 개수뿐이다.\n"
        "- 결함 코드 · 좌표 · 화소 크기 · 조항 번호 · 판정은 쓰지 않는다(따로 붙는다).\n"
        "- 결함의 실제 발생 원인, 밀리미터 치수, 합격 · 불합격을 단정하지 않는다. 주석의 한계 문장의 뜻을 한 번 밝힌다.\n"
        "- 결함이 없으면 표시된 결함이 없다고만 쓴다. 출력은 설명 문단만 — JSON · 목록 · 머리말 없이.\n"
        f"주석: {canonical_bytes(evidence).decode('utf-8')}"
    )


def _scope_text(scope) -> str:
    return scope if isinstance(scope, str) else canonical_bytes(scope).decode("utf-8")


def _historical(p: dict) -> bool:
    return "historical_only" in _flag_names(p) or str(p.get("use_boundary", "")).startswith("historical")


def _source_evidence(p: dict) -> dict:
    """생성 · 판정이 보는 근거. 쪽 번호 같은 서지 수는 넣지 않는다(수치 추적 원천을 넓히지 않게 — `source_ref` 로)."""
    return {"passage": p["text"], "passage_id": p["passage_id"], "source_id": p["doc_id"],
            "source_sha256": p["source_file_sha256"], "lineage_group_id": p["lineage_group_id"],
            "scope": p["scope"], "generation_constraints": list(p.get("generation_constraints") or []),
            "historical_only": _historical(p), "context_passages": dict(p["context_texts"])}


def _source_prompt(p: dict, perspective: str) -> str:
    lines = ["아래 원문 구절만을 근거로 한국어 질문 하나와 답 하나를 만들어라.",
             f"- 질문의 관점: {perspective}.",
             "- 답은 구절(과 아래 필수 문맥)에 적힌 내용만으로 쓴다. 구절 밖의 수치 · 기준 · 원인을 덧붙이지 않는다. 목차 · 개정 이력 · 서지에 대한 질문은 만들지 않는다.",
             f"- 이 구절의 적용 범위: {_scope_text(p['scope'])}. 이 범위를 넘어 일반화하지 않는다."]
    for c in p.get("generation_constraints") or []:
        lines.append(f"- 지켜야 할 제약: {c}")
    if _historical(p):
        lines.append("- 이 구절은 역사적 서술이다. 현행 기준 · 현대의 원인 · 예방 · 보수로 일반화하지 말고, 답에 역사적 서술임을 밝힌다.")
    lines.append('- 출력은 JSON 하나: {"question": "...", "answer": "...", "source_ids": ["<구절 id>"]}. source_ids 에는 아래 구절 id 하나만 쓴다.')
    lines.append(f"구절 id: {p['passage_id']}")
    lines.append(f"구절: {p['text']}")
    for cid, text in p["context_texts"].items():
        lines.append(f"필수 문맥({cid}, 근거로만 쓰고 출처로 적지 않는다): {text}")
    return "\n".join(lines)


def _source_split(passages: list[dict]) -> tuple[dict[str, str], dict]:
    """계보 단위 분할(지시 1008b). 반환: 계보 → split, 규칙과 수."""
    by_lin: dict[str, Counter] = defaultdict(Counter)
    for p in passages:
        by_lin[p["lineage_group_id"]][p["doc_id"]] += 1
    n = len(passages)
    target = int(n * SOURCE_VAL_FRACTION + 0.5)
    order = sorted(by_lin, key=lambda g: (_sha_bytes(f"source_split|{g}".encode()), g))
    val: list[str] = []
    docs: Counter = Counter()
    size = 0
    for g in order:
        merged = docs + by_lin[g]
        if size + sum(by_lin[g].values()) <= target and max(merged.values()) <= SOURCE_VAL_DOC_CAP * target:
            val.append(g)
            docs, size = merged, size + sum(by_lin[g].values())
    while val and max(docs.values()) > SOURCE_VAL_DOC_CAP * size:      # 실제 val 기준의 문서 상한
        top = max(docs, key=lambda d: (docs[d], d))
        g = next(x for x in reversed(val) if by_lin[x][top])
        val.remove(g)
        docs, size = docs - by_lin[g], size - sum(by_lin[g].values())
    split = {g: ("val" if g in val else "train") for g in by_lin}
    info = {"rule": "lineage_sha256_rank_greedy_doc_cap", "val_fraction": SOURCE_VAL_FRACTION, "doc_cap": SOURCE_VAL_DOC_CAP,
            "usable_passages": n, "target_val_passages": target, "val_passages": size, "val_lineages": sorted(val),
            "lineages": len(by_lin), "max_doc_share_of_val": round(max(docs.values()) / size, 4) if size else None}
    return split, info


# ------------------------------------------------------------------------------------------
# 본체
# ------------------------------------------------------------------------------------------
def prepare_run(pairs_path, source_bank_path, config: dict, run_dir, *, source_text_path=None, source_registry_path=None,
                source_rights_path=None, source_review_path=None, repo_root=None) -> dict:
    """새 디렉터리 `run_dir` 에 `tasks.jsonl` · `RUN.json` 을 쓴다. 반환은 RUN.json 의 요약.

    `source_bank_path` — 포인터 은행(추적). `source_text_path` — 본문 은행(미추적, 없으면 넘기는 구절 0 — 사유 `no_full_bank`).
    `source_registry_path` · `source_rights_path` · `source_review_path` — 등록부 · 사용자 권리 결정 · 사람 검수 기록(은행 진입점에 그대로).

    `repo_root` — 매니페스트 · 영상의 상대 경로와 게이트를 푸는 저장소 뿌리(기본: 이 파일의 저장소). 게이트 검사는
    `common.assert_gate_closed` 가 그 뿌리의 `.git` 이 디렉터리이기를 요구하므로 워크트리에서는 거부된다(본 체크아웃 · 엘리스 클론에서 돈다).
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    assert_gate_closed(root)
    run_dir = Path(run_dir)
    if run_dir.exists() or run_dir.is_symlink():
        raise PrepareRefused("run_dir_exists", f"{run_dir} 가 이미 있다 — 새 디렉터리에만 쓴다")
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise PrepareRefused("config_schema", "config schema_version 1 이 필요하다")
    inputs, quality = config.get("inputs") or {}, config.get("quality") or {}
    exp_train, exp_val = inputs.get("expected_image_train"), inputs.get("expected_image_val")
    max_q = int(quality.get("maximum_questions_per_passage", 0))
    if not (isinstance(exp_train, int) and isinstance(exp_val, int)) or not 1 <= max_q <= len(SOURCE_PERSPECTIVES):
        raise PrepareRefused("config_inputs", "기대 수 · 구절당 질문 수가 필요하다")

    pairs_dir = _pairs_dir(Path(pairs_path))
    pairs = verify_pairs_snapshot(pairs_dir, inputs)
    snap_in = pairs["meta"].get("input_snapshot") or {}
    manifest_dir = root / str(snap_in.get("path", ""))
    manifest_digest = verify_manifest_snapshot(manifest_dir)
    if manifest_digest != snap_in.get("snapshot_digest"):
        raise PrepareRefused("manifest_snapshot", "페어가 기록한 매니페스트 판과 다르다")
    manifest, eval_ids, eval_groups = read_manifest(manifest_dir)
    passages, bank = select_source_passages(Path(source_bank_path), source_text_path, inputs.get("source_bank_sha256"),
                                            registry_path=source_registry_path, rights_path=source_rights_path,
                                            review_path=source_review_path)
    lineage_of_split, split_info = _source_split(passages)
    names = _label_names()

    tmp_dir = run_dir.with_name(f".{run_dir.name}.{uuid.uuid4().hex}.tmp")
    tmp_dir.mkdir(parents=True)
    try:
        counts: Counter = Counter()
        lineage_split: dict[str, str] = {}
        seen_ids: set[str] = set()
        image_lineage: dict[str, str] = {}
        seen_sha: set[str] = set()
        same_lineage_dup = 0
        with (tmp_dir / "tasks.jsonl").open("xb") as out:
            # 1) 영상 설명 — 페어를 스트리밍. 명부 순서는 봉인 파일의 순서 그대로다(다시 정렬하지 않는다)
            for rec in iter_jsonl(pairs_dir / "pairs.jsonl"):
                iid = rec.get("image_id")
                if not isinstance(iid, str) or not iid or iid in seen_ids:
                    raise PrepareRefused("pairs_duplicate", f"{iid!r}")
                seen_ids.add(iid)
                if rec.get("split") not in ("train", "val") or iid in eval_ids:
                    raise PrepareRefused("eval_leak", f"{iid}: 평가 영상은 작업 명부에 들 수 없다")
                m = manifest.get(iid)
                if m is None:
                    raise PrepareRefused("manifest_missing", iid)
                if m["split"] != rec["split"] or m["client"] != rec.get("client"):
                    raise PrepareRefused("split_reassigned", f"{iid}: 매니페스트 {m['split']}/{m['client']} · 페어 {rec['split']}/{rec.get('client')}")
                if m["group_id"] in eval_groups:
                    raise PrepareRefused("eval_group_leak", f"{iid}: {m['group_id']}")
                if rec.get("image_path") != m["rel_path"]:
                    raise PrepareRefused("image_path", iid)
                prior = lineage_split.setdefault(m["group_id"], rec["split"])
                if prior != rec["split"]:
                    raise PrepareRefused("lineage_split_overlap", m["group_id"])
                image_file = root / m["rel_path"]
                digest = sha256_file(image_file)
                if digest != m["sha256"] or not _HEX64.fullmatch(digest):
                    raise PrepareRefused("image_changed", f"{iid}: 파일 해시가 매니페스트와 다르다")
                first = image_lineage.setdefault(digest, m["group_id"])
                if digest in seen_sha:
                    if first != m["group_id"]:
                        raise PrepareRefused("image_duplicate_cross_lineage", f"{iid}: 같은 바이트 영상이 다른 계보({first})에 있다")
                    same_lineage_dup += 1
                seen_sha.add(digest)
                locked = _locked_target(rec["skeleton"])
                evidence = _image_evidence(rec, names)
                task = {
                    "id": f"image_report:{iid}", "kind": "image_report", "split": rec["split"], "client": rec["client"],
                    "lineage_group_id": m["group_id"], "image_path": m["rel_path"], "image_sha256": digest,
                    "source_ids": [f"{pairs_dir.name}:{iid}", *[f"limits:{c}" for c in locked["clauses"]]],
                    "evidence": evidence, "locked_target": locked, "prompt": _image_prompt(evidence),
                    "train_user_text": TRAIN_USER_TEXT,
                }
                out.write(canonical_bytes(task) + b"\n")
                counts[("image_report", rec["split"], rec["client"])] += 1
            # 2) 원문 QA — 넘길 수 있는 구절마다 관점 max_q 개(복제로 수를 채우지 않는다). 계보는 은행의 lineage_group_id
            for p in passages:
                split = lineage_of_split[p["lineage_group_id"]]
                evidence = _source_evidence(p)
                for j in range(max_q):
                    task = {
                        "id": f"source_qa:{p['passage_id']}:q{j + 1}", "kind": "source_qa", "split": split,
                        "client": "public", "lineage_group_id": p["lineage_group_id"], "image_path": None,
                        "source_ids": [p["passage_id"]], "evidence": evidence, "locked_target": None,
                        "prompt": _source_prompt(p, SOURCE_PERSPECTIVES[j]),
                        "source_ref": {"edition": p["edition"], "locator": p["locator"], "text_sha256": p["text_sha256"],
                                       "bank_pointer_sha256": bank["inputs_sha256"]["pointer_bank"]},
                    }
                    out.write(canonical_bytes(task) + b"\n")
                    counts[("source_qa", split, "public")] += 1
            out.flush()
            os.fsync(out.fileno())
        n_train = sum(v for (k, s, _), v in counts.items() if k == "image_report" and s == "train")
        n_val = sum(v for (k, s, _), v in counts.items() if k == "image_report" and s == "val")
        if (n_train, n_val) != (exp_train, exp_val):
            raise PrepareRefused("image_counts", f"train {n_train} · val {n_val} ≠ 기대 {exp_train} · {exp_val}")

        tasks_sha = sha256_file(tmp_dir / "tasks.jsonl")
        config_sha = _sha_bytes(canonical_bytes(config))
        src_train = sum(v for (k, s, _), v in counts.items() if k == "source_qa" and s == "train")
        train_passages = sum(1 for p in passages if lineage_of_split[p["lineage_group_id"]] == "train")
        basis = {
            "schema_version": SCHEMA_VERSION, "config_sha256": config_sha, "tasks_sha256": tasks_sha,
            "pairs_sha256": pairs["pairs_sha256"], "pairs_snapshot_digest": pairs["pairs_snapshot_digest"],
            "manifest_snapshot_digest": manifest_digest, "source_bank_sha256": bank["inputs_sha256"]["pointer_bank"],
            "source_text_sha256": bank["inputs_sha256"]["full_bank"], "source_bank_inputs_sha256": bank["inputs_sha256"],
            "prepare_code_sha256": _code_sha256(),
            "source_val_fraction": SOURCE_VAL_FRACTION, "source_val_doc_cap": SOURCE_VAL_DOC_CAP,
            "duplicate_image_policy": DUPLICATE_IMAGE_POLICY,
        }
        run = {
            "schema_version": SCHEMA_VERSION,
            "identity_sha256": _sha_bytes(canonical_bytes(basis)),
            "config_sha256": config_sha, "tasks_sha256": tasks_sha,
            "counts": {
                "tasks": sum(counts.values()),
                "by_kind_split_client": {f"{k}|{s}|{c}": v for (k, s, c), v in sorted(counts.items())},
                "image_report": {"train": n_train, "val": n_val},
                "source_qa": {"train": src_train, "val": sum(v for (k, s, _), v in counts.items() if k == "source_qa" and s == "val")},
                "source_passages": {"total": len(passages), "train": train_passages},
                "source_bank": bank,
                "source_split": split_info,
                "duplicate_image_same_lineage": same_lineage_dup,
                "source_qa_shortfall": {
                    "minimum_source_qa_train": quality.get("minimum_source_qa_train"),
                    "planned_source_qa_train": src_train,
                    "short_by": max(0, int(quality.get("minimum_source_qa_train") or 0) - src_train),
                    "minimum_distinct_train_passages": quality.get("minimum_distinct_train_passages"),
                    "planned_distinct_train_passages": train_passages,
                },
            },
            "config": config,
            "identity_basis": basis,
        }
        write_json_new(tmp_dir / "RUN.json", run)
        os.rename(tmp_dir, run_dir)          # 새 디렉터리를 한 번에 — 있으면 실패한다
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return {"run_dir": str(run_dir), "identity_sha256": run["identity_sha256"], "counts": run["counts"]}
