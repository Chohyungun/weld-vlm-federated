"""A3 대체 승인 — 2 차 체제의 조기 좌표 점검(ECC)을 A3 재시도의 증거로 대신한다(D7, 미니스펙 `_workspace/2026-10-31-main2/
MINISPEC_a3_substitute.md` 1 판).

사용자 결정(2026-10-31): 증거 기반 대체 승인에 찬성, 검사 거부를 없애는 수정에는 반대. 조건 셋 —
(C1) A3 와 본학습의 좌표 변환 · 전처리 · 출력 파서가 같은지 **코드와 해시로** 본다. (C2) 대신하는 검사와 대신하지 못하는 검사를
적는다(`SUBSTITUTE_SCOPE`). (C3) 학습 · 채점 양쪽에서 같은 규칙으로 검증하고 증거가 다르면 거부한다.

| 무엇 | 어디 |
|---|---|
| 승인서 꼴 · 동일성 목록 · 증거 목록 · 범위 | 이 모듈의 상수 |
| 검증(한 함수) | `verify_substitute(approval_raw, evidence_dir=, repo=, rev=)` — 학습은 실행 체크아웃의 HEAD, 채점은 영수증 커밋 |
| 학습 쪽 시작 문 | `vlm.train_cell.main_start_gate` — 2 차 레시피 + 면제 자리의 승인서 |
| 채점 쪽 프레임 표지 | `evaluation.ecc_frame_check.frame_check_by_a3_substitute` |
| 승인서 작성 | `scripts/make_a3_substitute.py` — 같은 함수로 0 건을 확인한 뒤에만 쓴다 |

동일성은 세 값이 모두 같을 때만 통과다: 승인서의 값 · 검증 자리(`rev`)의 코드 · `a3_commit` 의 코드. 증거는 승인서와 같은 폴더의
파일이고 바이트 sha 와 내용을 다시 본다. 정확도 원자료 재판정은 하지 않는다 — 증거의 정밀도 · 재현율은 A3 하네스의 저장값이다(범위에 적는다).
1 차 체제 면제(`vlm.ecc.check_waiver` · `evaluation.ecc_frame_check.frame_check_by_model_free`)와는 별개이며 그쪽 바이트를 바꾸지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

__all__ = ["CONFIG_REL", "DECISION_LOG_REL", "EVIDENCE_KINDS", "FIELDS", "IDENTITY_CONFIG_KEYS", "IDENTITY_FILES", "KEY_FILE",
           "KIND", "PROMPT_REL", "SUBSTITUTE_RECIPES", "SUBSTITUTE_SCOPE", "TRAIN_REG_REL", "SubstituteRefused",
           "build_approval", "config_identity_sha256", "dump_approval", "git_blob_sha1", "identity_at", "verify_substitute"]

KIND = "ecc_a3_substitute_v1"
SUBSTITUTE_RECIPES = ("json_line_then_basis",)
"""대체 승인을 받는 학습 레시피 — 2 차뿐이다. 1 차(`json_only`)는 종전 면제(1017a) 그대로다."""
FIELDS = ("kind", "recipe", "prompt_sha256", "max_new_tokens", "a3_commit", "identity", "evidence", "decision",
          "decision_log_section", "decided_at")
"""승인서 칸 — 정확히 이 열이다."""
IDENTITY_FILES = ("vlm/coords.py", "vlm/gen_parse.py", "vlm/pilot_vlm.py", "vlm/prompts/unified_v3_1_basis.txt",
                  "vlm/narrative.py")
"""동일성(C1)의 파일 — 좌표 변환 · 출력 파서 · 전처리(프로세서 적재 · 템플릿 인자) · 타깃 조립 · 학습 고리 · 프롬프트 · 서술 추출."""
IDENTITY_CONFIG_KEYS = ("uni_model", "uni_processor_kwargs", "uni_chat_template_kwargs", "uni_prompt_path", "uni_coord_space",
                        "uni_max_new_tokens", "uni_pairs")
"""동일성의 설정 키(`configs/base.yaml` `fixed_before_main_runs`). `uni_batch_size` 는 넣지 않는다 — 생성기의 묶음은 범위 밖(6)."""
EVIDENCE_KINDS = ("a3_check", "a3_result", "a3_gate_form", "batch_equivalence")
"""증거 kind — 닫힌 넷이 정확히 한 번씩."""
CONFIG_REL = "configs/base.yaml"
TRAIN_REG_REL = "configs/train_registration.yaml"
PROMPT_REL = "vlm/prompts/unified_v3_1_basis.txt"
DECISION_LOG_REL = "docs/의사결정로그.md"
KEY_FILE = "ECC_A3_SUBSTITUTE_KEY.json"
"""학습 쪽 시드 폴더의 키 기록 — 승인서 sha · 착수 고정 sha · 시드 · 출발 어댑터. 처음 받은 때 한 번 쓰고 그 뒤 같아야 한다."""
A3_SCOPE_KEY = "train_box.class_aware_iou50"
MIN_THRESHOLD = 0.9
MIN_PROBE_ROWS = 16

SUBSTITUTE_SCOPE: dict[str, tuple[str, ...]] = {
    "substitutes": (
        ("학습한 어댑터의 생성문이 같은 파서 · 역변환을 거쳐 정답 틀(ABS_ORIG 원본 픽셀)의 상자로 돌아온다 — A3 학습 행 64 장 결함 상자에서 "
         "종류까지 맞춘 IoU 0.5 정밀도 · 재현율(문턱 0.9 이상)"),
        ("1 차 조기 좌표 점검이 막으려던 사고(좌표 틀이 어긋난 채 본학습을 끝까지 돌림)의 경로 — 타깃 좌표 → 학습 → 생성 → 역변환 — 가 "
         "같은 코드(동일성 목록)로 한 번 끝까지 맞았다"),
    ),
    "does_not_substitute": (
        "본학습(f = 1/8 · 1 에폭)의 성능 · 학습량 충분성",
        "비노출 영상의 일반화 — A3 의 비노출 행은 0 이다",
        "참여자별 이득 · 연합 효과",
        "1 차 조기 좌표 점검의 비노출 417 장 · step 100/200 기하 판정",
        "정확도 원자료 재판정 — 증거의 정밀도 · 재현율은 A3 하네스의 저장값이다",
        "A3 이후 바뀐 생성기(묶음 생성)의 A100 동치 — 로컬 탐침만 증거에 있고 A100 확인은 본학습 착수 고정 전 배치 확인 · 생성 전 점검이 한다",
    ),
}
"""대신하는 것 · 대신하지 못하는 것(C2) — 채점 산출에 그대로 싣는다."""

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class SubstituteRefused(ValueError):
    """대체 승인을 받지 않는다. `code` 는 사유 코드(`substitute_*`), `problems` 는 모든 어긋남."""

    def __init__(self, code: str, problems: list[str]):
        super().__init__(f"[{code}] " + " · ".join(problems))
        self.code, self.problems = code, list(problems)


def git_blob_sha1(data: bytes) -> str:
    """git 의 blob 객체 id — `blob <길이>\\0` 머리 + 바이트의 sha1(커밋에서 읽은 원시 바이트에 쓴다)."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + bytes(data)).hexdigest()


def config_identity_sha256(base_raw: bytes) -> str:
    """`configs/base.yaml` 의 동일성 키 부분 → 정규 JSON(키 정렬 · UTF-8 · 공백 없음)의 sha256. 키가 없으면 `null` 로 든다."""
    import yaml

    doc = yaml.safe_load(bytes(base_raw).decode("utf-8")) or {}
    fixed = doc.get("fixed_before_main_runs") if isinstance(doc, dict) else None
    fixed = fixed if isinstance(fixed, dict) else {}
    part = {k: fixed.get(k) for k in IDENTITY_CONFIG_KEYS}
    return hashlib.sha256(json.dumps(part, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _blob(repo: Path, rev: str, path: str) -> bytes:
    from evaluation.entry_gate import EntryRejected, blob_at

    try:
        return blob_at(Path(repo), rev, path)
    except EntryRejected as exc:
        raise FileNotFoundError(f"{rev[:12]}:{path} — {exc}") from None


def identity_at(repo: Path, rev: str) -> dict[str, Any]:
    """`rev` 의 동일성 값 — 파일마다 git blob sha1, 설정 부분의 sha256. 파일이 없으면 `FileNotFoundError`."""
    return {"files": {p: git_blob_sha1(_blob(repo, rev, p)) for p in IDENTITY_FILES},
            "config_sha256": config_identity_sha256(_blob(repo, rev, CONFIG_REL))}


def _prompt_sha(raw: bytes) -> str:
    """`vlm.pilot_vlm.load_prompt` 와 같은 해시 — 줄끝을 LF 로 읽은 문자열의 UTF-8 sha256."""
    text = bytes(raw).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _expect_at(repo: Path, rev: str) -> dict[str, Any]:
    """검증 자리(`rev`)의 등록값 — 레시피(학습 등록) · 프롬프트 sha · 생성 한도 · 생성 배치(설정)."""
    import yaml

    base = yaml.safe_load(_blob(repo, rev, CONFIG_REL).decode("utf-8")) or {}
    fixed = (base.get("fixed_before_main_runs") or {}) if isinstance(base, dict) else {}
    train = yaml.safe_load(_blob(repo, rev, TRAIN_REG_REL).decode("utf-8")) or {}
    return {"recipe": train.get("recipe") if isinstance(train, dict) else None,
            "prompt_sha256": _prompt_sha(_blob(repo, rev, str(fixed.get("uni_prompt_path") or PROMPT_REL))),
            "max_new_tokens": fixed.get("uni_max_new_tokens"), "batch_size": fixed.get("uni_batch_size")}


def _json(raw: bytes) -> Any:
    try:
        return json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _form_problems(doc: Any) -> list[str]:
    if not isinstance(doc, dict) or set(doc) != set(FIELDS):
        got = sorted(doc) if isinstance(doc, dict) else []
        return [f"승인서의 칸이 {list(FIELDS)} 가 아니다 — 받은 칸 {got}"]
    bad = []
    if doc["kind"] != KIND:
        bad.append(f"kind 가 {KIND!r} 가 아니다")
    if not (isinstance(doc["prompt_sha256"], str) and _HEX64.match(doc["prompt_sha256"])):
        bad.append("prompt_sha256")
    if type(doc["max_new_tokens"]) is not int or doc["max_new_tokens"] < 1:
        bad.append("max_new_tokens")
    if not (isinstance(doc["a3_commit"], str) and _HEX40.match(doc["a3_commit"])):
        bad.append("a3_commit")
    ident = doc["identity"]
    if not (isinstance(ident, dict) and set(ident) == {"files", "config_sha256"} and isinstance(ident["files"], dict)
            and set(ident["files"]) == set(IDENTITY_FILES) and isinstance(ident["config_sha256"], str)
            and _HEX64.match(ident["config_sha256"])):
        bad.append("identity")
    ev = doc["evidence"]
    if not (isinstance(ev, list) and all(isinstance(e, dict) and set(e) == {"kind", "name", "sha256"} for e in ev)
            and sorted(e["kind"] for e in ev) == sorted(EVIDENCE_KINDS)
            and all(isinstance(e["name"], str) and e["name"] and PurePosixPath(e["name"]).name == e["name"]
                    and "\\" not in e["name"] and e["name"] not in (".", "..") for e in ev)
            and all(isinstance(e["sha256"], str) and _HEX64.match(e["sha256"]) for e in ev)
            and len({e["name"] for e in ev}) == len(ev)):
        bad.append(f"evidence — kind {list(EVIDENCE_KINDS)} 가 한 번씩, 이름은 같은 폴더의 파일 이름, sha 는 16 진 64 자")
    for k in ("decision", "decision_log_section"):
        if not (isinstance(doc[k], str) and doc[k].strip()):
            bad.append(k)
    try:
        ok_time = datetime.fromisoformat(str(doc["decided_at"])).tzinfo is not None
    except ValueError:
        ok_time = False
    if not ok_time:
        bad.append("decided_at — 시간대 있는 ISO 8601")
    return [f"승인서 칸 값이 틀렸다: {', '.join(bad)}"] if bad else []


def _evidence_problems(doc: dict, evidence_dir: Path, expect: dict) -> tuple[str | None, list[str], dict]:
    """증거 넷 — 바이트 sha · 내용. 돌려주는 값: (사유 코드, 문제, kind → 읽은 객체)."""
    got: dict[str, Any] = {}
    missing, sha_bad = [], []
    for e in doc["evidence"]:
        p = Path(evidence_dir) / e["name"]
        if not p.is_file():
            missing.append(e["name"])
            continue
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != e["sha256"]:
            sha_bad.append(e["name"])
            continue
        got[e["kind"]] = raw
    if missing:
        return "substitute_evidence_missing", [f"증거 파일이 승인서 폴더에 없다: {missing}"], {}
    if sha_bad:
        return "substitute_evidence_sha", [f"증거 파일의 바이트 sha256 이 승인서와 다르다: {sha_bad}"], {}
    bad = []
    chk, res, probe = _json(got["a3_check"]), _json(got["a3_result"]), _json(got["batch_equivalence"])
    vals = (chk or {}).get("a3_values") if isinstance(chk, dict) else None
    if not isinstance(chk, dict) or chk.get("pass") is not True or chk.get("execution_commit") != doc["a3_commit"] \
            or not isinstance(vals, dict) or vals.get("scope") != A3_SCOPE_KEY:
        bad.append("a3_check — pass 참 · 실행 커밋 = a3_commit · 범위 train_box.class_aware_iou50 이어야 한다")
    else:
        th, pr, rc = vals.get("threshold"), vals.get("precision"), vals.get("recall")
        # 유한한 수 · 범위 — json 은 NaN · Infinity 를 읽고, NaN 과의 비교는 늘 거짓이라 하한 비교만으로는 지나간다
        nums = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (th, pr, rc))
        if not nums or not (MIN_THRESHOLD <= th <= 1) or not (th <= pr <= 1) or not (th <= rc <= 1):
            bad.append(f"a3_check — 정밀도 · 재현율 · 문턱이 유한한 수이고 문턱 ∈ [{MIN_THRESHOLD}, 1] · 지표 ∈ [문턱, 1] 이어야 한다")
    run_from_check = None
    if isinstance(chk, dict) and isinstance(chk.get("result_member"), str):
        parts = PurePosixPath(chk["result_member"]).parts
        run_from_check = parts[-2] if len(parts) >= 2 else None
    rg = (res or {}).get("regime") if isinstance(res, dict) else None
    if not isinstance(rg, dict) or rg.get("recipe") != doc["recipe"] or rg.get("prompt_sha256") != doc["prompt_sha256"] \
            or rg.get("max_new_tokens") != doc["max_new_tokens"] or res.get("run_name") != run_from_check \
            or run_from_check is None:
        bad.append("a3_result — 레시피 · 프롬프트 sha · 한도가 승인서와 같고 실행 이름이 a3_check 의 결과 파일과 같아야 한다")
    n = expect.get("batch_size")
    passes = (probe or {}).get("passes") if isinstance(probe, dict) else None
    one = (passes or {}).get("1") if isinstance(passes, dict) else None
    if not isinstance(one, dict) or one.get("status") != "ok" or type(probe.get("n_rows")) is not int \
            or probe["n_rows"] < MIN_PROBE_ROWS or len(one.get("records") or {}) != probe["n_rows"]:
        bad.append(f"batch_equivalence — 배치 1 이 ok 이고 행 수 ≥ {MIN_PROBE_ROWS} 이어야 한다")
    elif type(n) is not int or n < 1:
        bad.append("batch_equivalence — 검증 자리의 설정 uni_batch_size 가 1 이상의 정수가 아니다")
    elif n > 1:
        pn = passes.get(str(n))
        cmp = (pn or {}).get("compare_vs_batch1") if isinstance(pn, dict) else None
        if not isinstance(pn, dict) or pn.get("status") != "ok" or not isinstance(cmp, dict) \
                or cmp.get("tokens_equal_rate") != 1.0:
            bad.append(f"batch_equivalence — 등록 배치 {n} 의 탐침이 ok 이고 토큰 열 일치율이 1.0 이어야 한다")
    if bad:
        return "substitute_evidence_content", bad, {}
    return None, [], {"a3_check": chk, "a3_result": res, "batch_equivalence": probe}


def verify_substitute(approval_raw: bytes, *, evidence_dir: Path, repo: Path, rev: str) -> dict[str, Any]:
    """대체 승인서를 검증 자리(`repo` 의 `rev`)와 A3 커밋 · 증거 폴더로 본다 — 학습 · 채점이 같은 함수를 부른다(C3).

    순서: 꼴 → 레시피 → 등록값(프롬프트 · 한도) → A3 커밋 → 동일성(승인서 = rev = A3 커밋) → 증거(sha · 내용) → 설계 결정 기록 절.
    통과면 요약을, 아니면 `SubstituteRefused(사유 코드, 문제)`.
    """
    doc = _json(approval_raw)
    problems = _form_problems(doc)
    if problems:
        raise SubstituteRefused("substitute_form", problems)
    try:
        expect = _expect_at(repo, rev)
    except FileNotFoundError as exc:
        raise SubstituteRefused("substitute_regime", [f"검증 자리의 설정 · 등록 · 프롬프트를 읽지 못했다: {exc}"]) from None
    if doc["recipe"] not in SUBSTITUTE_RECIPES or doc["recipe"] != expect["recipe"]:
        raise SubstituteRefused("substitute_recipe", [(f"승인서 레시피 {doc['recipe']!r} 가 대체 대상({list(SUBSTITUTE_RECIPES)})이 아니거나 "
                                                       f"검증 자리의 학습 등록 레시피 {expect['recipe']!r} 와 다르다")])
    regime = [k for k in ("prompt_sha256", "max_new_tokens") if doc[k] != expect[k]]
    if regime:
        raise SubstituteRefused("substitute_regime", [f"승인서의 {regime} 가 검증 자리의 등록값과 다르다"])
    try:
        a3 = identity_at(repo, doc["a3_commit"])
    except FileNotFoundError as exc:
        raise SubstituteRefused("substitute_a3_commit", [(f"A3 커밋의 동일성 파일을 저장소에서 읽지 못했다(코드 꾸러미에 이력이 있어야 한다): "
                                                          f"{exc}")]) from None
    try:
        here = identity_at(repo, rev)
    except FileNotFoundError as exc:
        raise SubstituteRefused("substitute_identity", [f"검증 자리의 동일성 파일을 읽지 못했다: {exc}"]) from None
    drift = [p for p in IDENTITY_FILES if not (doc["identity"]["files"][p] == here["files"][p] == a3["files"][p])]
    if not (doc["identity"]["config_sha256"] == here["config_sha256"] == a3["config_sha256"]):
        drift.append(f"{CONFIG_REL}:{list(IDENTITY_CONFIG_KEYS)}")
    if drift:
        raise SubstituteRefused("substitute_identity", [(f"A3 와 검증 자리의 코드 · 설정이 같지 않다(승인서 · 검증 자리 · A3 커밋 셋이 "
                                                         f"같아야 한다): {drift}")])
    code, problems, read = _evidence_problems(doc, Path(evidence_dir), expect)
    if code is not None:
        raise SubstituteRefused(code, problems)
    try:
        log = _blob(repo, rev, DECISION_LOG_REL).decode("utf-8")
    except FileNotFoundError:
        log = ""
    if doc["decision_log_section"].strip() not in log:
        raise SubstituteRefused("substitute_decision", ["승인서의 의사결정로그 절 제목이 검증 자리의 의사결정로그에 없다"])
    return {"approval_sha256": hashlib.sha256(bytes(approval_raw)).hexdigest(), "a3_commit": doc["a3_commit"],
            "recipe": doc["recipe"], "decision_log_section": doc["decision_log_section"].strip(),
            "decided_at": doc["decided_at"], "identity": doc["identity"],
            "evidence": sorted([{"kind": e["kind"], "sha256": e["sha256"]} for e in doc["evidence"]], key=lambda e: e["kind"]),
            "a3_values": read["a3_check"]["a3_values"], "batch_size": expect["batch_size"],
            "scope": {k: list(v) for k, v in SUBSTITUTE_SCOPE.items()}}


def build_approval(*, repo: Path, rev: str, a3_commit: str, evidence: dict[str, Path], decision: str,
                   decision_log_section: str, decided_at: str) -> dict[str, Any]:
    """승인서 객체를 짓는다 — 증거 파일 이름 · sha, 동일성은 `rev` 의 값. 쓰는 쪽(`scripts/make_a3_substitute.py`)이 같은 폴더에 증거를
    복사하고 `verify_substitute` 로 0 건을 확인한 뒤에만 쓴다."""
    exp = _expect_at(repo, rev)
    return {"kind": KIND, "recipe": exp["recipe"], "prompt_sha256": exp["prompt_sha256"], "max_new_tokens": exp["max_new_tokens"],
            "a3_commit": a3_commit, "identity": identity_at(repo, rev),
            "evidence": [{"kind": k, "name": Path(evidence[k]).name, "sha256": hashlib.sha256(Path(evidence[k]).read_bytes()).hexdigest()}
                         for k in EVIDENCE_KINDS],
            "decision": decision, "decision_log_section": decision_log_section, "decided_at": decided_at}


def dump_approval(doc: dict[str, Any]) -> bytes:
    """승인서의 정규 바이트 — 키 정렬 · UTF-8 · 들여쓰기 1 · 끝 LF."""
    return json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=1).encode("utf-8") + b"\n"
