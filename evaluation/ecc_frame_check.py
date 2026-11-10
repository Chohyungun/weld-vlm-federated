"""본실험 본채점의 프레임 표지 — 등록 `canary.frame_diag_sha256` 이 가리키는 조기 좌표 점검 통과 파일을 본다.

설계 결정 기록 10-07: 조기 좌표 점검(ECC)이 착수 전 프레임 진단을 대신한다. 그래서 목적 `main` 에서 이 칸은 **ECC 허가 산출**
(`scripts/main_uni.py ecc-accept` 가 시드 폴더에 둔 `ecc_permit.json` — 평가 쪽 통과 파일 `early_coord_check.json` 의 바이트)의
sha256 이다(`evaluation.prereg_unified.MAIN_FRAME_DIAG_MEANING`). 리허설 · 진단 목적의 뜻은 그대로다.

학습 쪽은 시작 · 재개마다 `early_coord_check.verify_permit` 으로 전부(출발 어댑터 · 착수 고정 · 점검 실행 신원 · 네 검사 산출 파일)를
대조했다. 본채점은 그 대조값을 갖지 않으므로 넷만 본다 — (1) 바이트의 sha256 이 등록과 같다 (2) 조기 점검의 산출이다(종류 · 목적 ·
분할) (3) 규칙 파일 · 대상 목록 지문이 영수증 커밋의 규칙과 같다 (4) 그 안의 step 판정 · 네 검사에서 **다시 낸** 결정이 통과이고 네 칸을
연다. 파일이 스스로 적은 결정을 믿지 않는 것은 `verify_permit` 과 같다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation import early_coord_check as E

FRAME_CHECKED_BY_ECC = "FRAME_CHECKED_BY_ECC"
"""본채점 산출의 프레임 표지 — 진단 대신 조기 좌표 점검의 통과 파일로 프레임을 확인했다."""


def frame_check_by_permit(raw: bytes, *, registered_sha256: str, rules: E.EarlyRules) -> dict:
    """통과 파일 바이트 → 산출에 싣는 표지. 어긋나면 `EarlyCheckRejected`(사유 코드 `FRAME_PERMIT_*`)."""
    sha = hashlib.sha256(raw).hexdigest()
    if not isinstance(registered_sha256, str) or sha != registered_sha256:
        raise E.EarlyCheckRejected("FRAME_PERMIT_HASH", "허가 파일의 sha256 이 등록 canary.frame_diag_sha256 과 다르다")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise E.EarlyCheckRejected("FRAME_PERMIT_FORMAT", "허가 파일을 읽지 못한다") from None
    if not isinstance(obj, dict) or obj.get("kind") != E.KIND or obj.get("purpose") != E.KIND or obj.get("split") != E.SPLIT:
        raise E.EarlyCheckRejected("FRAME_PERMIT_KIND", "조기 좌표 점검의 산출이 아니거나 목적 · 분할이 다르다")
    ev = obj.get("evidence") or {}
    if ev.get("rules_sha256") != rules.raw_sha256 or ev.get("target_list_sha256") != rules.target_sha256:
        raise E.EarlyCheckRejected("FRAME_PERMIT_RULES", "허가 파일의 규칙 파일 · 대상 목록 지문이 영수증 커밋의 규칙과 다르다")
    steps, model_free = obj.get("steps"), obj.get("model_free")
    if not isinstance(steps, list) or not isinstance(model_free, list) or not all(isinstance(s, dict) for s in steps):
        raise E.EarlyCheckRejected("FRAME_PERMIT_FORMAT", "step 판정 · 네 검사의 꼴이 아니다")
    decision = E.decide([s.get("status") for s in steps], len(rules.steps), rules.sequential_rule)
    opens = E.opens_four_cells(model_free, decision)
    if decision != obj.get("decision") or opens != obj.get("opens_four_cells"):
        raise E.EarlyCheckRejected("FRAME_PERMIT_RECOMPUTE", "허가 파일이 적은 결정 · 네 칸 판정이 그 안에서 다시 낸 값과 다르다")
    if decision["outcome"] != E.PASS or not opens:
        raise E.EarlyCheckRejected("FRAME_PERMIT_NOT_PASSED", "통과가 아니거나 네 칸을 여는 기준을 채우지 않았다")
    return {"status": FRAME_CHECKED_BY_ECC, "ecc_permit_sha256": sha, "rules_sha256": rules.raw_sha256,
            "decision_step_index": decision["step_index"], "attempt": ev.get("attempt"), "seed": ev.get("seed"),
            "pin_sha256": ev.get("pin_sha256")}


# ---------------------------------------------------------------- 허가 면제 — 모델 없는 네 검사 통과 산출 + 면제 파일의 묶음

FRAME_CHECKED_BY_A3_SUBSTITUTE = "FRAME_CHECKED_BY_A3_SUBSTITUTE"
"""본채점 산출의 프레임 표지 — 2 차 체제에서 조기 좌표 점검을 A3 대체 승인서로 대신했다(D7, 사용자 결정 2026-10-31). 대신하는 것 ·
대신하지 못하는 것은 산출의 `scope` 에 그대로 싣는다(`evaluation.ecc_substitute.SUBSTITUTE_SCOPE`)."""


def frame_check_by_a3_substitute(approval_raw: bytes, *, evidence_dir: Path, registered_sha256: str, repo: Path,
                                 rev: str) -> dict:
    """A3 대체 승인서 바이트 → 산출에 싣는 표지. 학습 쪽 시작 문과 **같은 함수**(`ecc_substitute.verify_substitute`)를 영수증 커밋
    (`rev`)으로 부른다 — 동일성(승인서 = 영수증 커밋 = A3 커밋) · 증거 sha · 내용 · 설계 결정 기록 절. 등록 `canary.frame_diag_sha256` 은
    승인서 바이트의 sha256 이다. 어긋나면 `EarlyCheckRejected`(사유 코드 `FRAME_SUBSTITUTE_*`)."""
    from evaluation import ecc_substitute as S

    sha = hashlib.sha256(bytes(approval_raw)).hexdigest()
    if not isinstance(registered_sha256, str) or sha != registered_sha256:
        raise E.EarlyCheckRejected("FRAME_SUBSTITUTE_HASH", "A3 대체 승인서의 sha256 이 등록 canary.frame_diag_sha256 과 다르다")
    try:
        summary = S.verify_substitute(approval_raw, evidence_dir=Path(evidence_dir), repo=Path(repo), rev=rev)
    except S.SubstituteRefused as exc:
        raise E.EarlyCheckRejected("FRAME_SUBSTITUTE_" + exc.code.removeprefix("substitute_").upper(),
                                   " · ".join(exc.problems)) from None
    return {"status": FRAME_CHECKED_BY_A3_SUBSTITUTE, "frame_diag_sha256": sha, "approval_sha256": sha,
            "a3_commit": summary["a3_commit"], "decision_log_section": summary["decision_log_section"],
            "decided_at": summary["decided_at"], "evidence": summary["evidence"], "identity": summary["identity"],
            "a3_values": summary["a3_values"], "batch_size": summary["batch_size"], "scope": summary["scope"]}


FRAME_CHECKED_MODEL_FREE_ONLY = "FRAME_CHECKED_MODEL_FREE_ONLY"
"""본채점 산출의 프레임 표지 — 모델 점검 허가가 사용자 결정으로 면제됐고(의사결정로그 2026-10-17), 모델 없는 네 검사의 통과만
확인했다. **미세조정 뒤 모델이 내는 생성 좌표의 틀은 이 표지가 확인하지 않는다.**"""
BUNDLE_KIND = "frame_model_free_bundle_v1"
WAIVER_FIELDS = ("decision", "decision_log_section", "model_free_admission_sha256", "step100_geometry_sha256", "decided_at")
"""면제 파일의 칸 — 학습 쪽 `vlm.ecc.WAIVER_FIELDS` 와 같다(시험이 맞댄다). 정확히 이 다섯이다."""
MODEL_FREE_SCOPE = ("모델 없는 네 검사(원본 크기 · 프로세서 입력 · 좌표 왕복 · 정답 에코)만 확인했다 — 미세조정 뒤 모델이 내는 생성 좌표의 틀은 "
                    "확인하지 않았다(모델 점검 허가 면제, 의사결정로그 2026-10-17)")


def model_free_bundle_sha256(admission_raw: bytes, waiver_raw: bytes) -> str:
    """두 파일 바이트의 sha256 을 정규 JSON 하나로 묶은 값 — 목적 main 의 `canary.frame_diag_sha256` 이 허가 대신 가리키는 값."""
    body = {"kind": BUNDLE_KIND, "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
            "waiver_sha256": hashlib.sha256(waiver_raw).hexdigest()}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _hex64(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def frame_check_by_model_free(admission_raw: bytes, waiver_raw: bytes, *, registered_sha256: str, rules: E.EarlyRules,
                              decision_log: str) -> dict:
    """모델 없는 통과 산출 · 면제 파일 → 산출에 싣는 표지. 어긋나면 `EarlyCheckRejected`(사유 코드 `FRAME_MODEL_FREE_*`).

    (1) 두 파일의 묶음 sha 가 등록과 같다 (2) 통과 산출: 네 검사가 모두 있고 통과 · 불량 0 · `passed` 참, 규칙 · 대상 지문이 영수증
    커밋의 규칙과 같다 (3) 면제 파일: 칸 다섯이 정확히 있고 결정 문구 · 절 제목이 비지 않았으며, 그 통과 산출의 sha 를 가리키고, 절 제목이
    영수증 커밋의 설계 결정 기록에 있다.
    """
    from datetime import datetime

    bundle = model_free_bundle_sha256(admission_raw, waiver_raw)
    if not isinstance(registered_sha256, str) or bundle != registered_sha256:
        raise E.EarlyCheckRejected("FRAME_MODEL_FREE_HASH", "통과 산출 · 면제 파일의 묶음 sha 가 등록 canary.frame_diag_sha256 과 다르다")
    try:
        adm, wv = json.loads(admission_raw.decode("utf-8")), json.loads(waiver_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise E.EarlyCheckRejected("FRAME_MODEL_FREE_FORMAT", "통과 산출 · 면제 파일을 읽지 못한다") from None
    checks = adm.get("checks") if isinstance(adm, dict) else None
    if not isinstance(checks, list) or not all(isinstance(c, dict) for c in checks) or not E.model_free_ok(checks) \
            or any(c.get("n_bad") != 0 for c in checks) or adm.get("passed") is not True:
        raise E.EarlyCheckRejected("FRAME_MODEL_FREE_NOT_PASSED", "모델 없는 네 검사가 모두 통과 · 불량 0 이 아니다")
    if adm.get("rules_sha256") != rules.raw_sha256 or adm.get("target_list_sha256") != rules.target_sha256:
        raise E.EarlyCheckRejected("FRAME_MODEL_FREE_RULES", "통과 산출의 규칙 파일 · 대상 목록 지문이 영수증 커밋의 규칙과 다르다")
    if not isinstance(wv, dict) or set(wv) != set(WAIVER_FIELDS):
        raise E.EarlyCheckRejected("FRAME_WAIVER_FORM", f"면제 파일의 칸이 {list(WAIVER_FIELDS)} 가 아니다")
    bad = [k for k in ("decision", "decision_log_section") if not (isinstance(wv[k], str) and wv[k].strip())]
    bad += [] if wv["step100_geometry_sha256"] is None or _hex64(wv["step100_geometry_sha256"]) else ["step100_geometry_sha256"]
    try:
        datetime.fromisoformat(str(wv["decided_at"]))
    except ValueError:
        bad.append("decided_at")
    if bad:
        raise E.EarlyCheckRejected("FRAME_WAIVER_FORM", f"면제 파일의 칸 값이 틀렸다: {bad}")
    if wv["model_free_admission_sha256"] != hashlib.sha256(admission_raw).hexdigest():
        raise E.EarlyCheckRejected("FRAME_WAIVER_BINDING", "면제 파일이 가리키는 통과 산출의 sha 가 받은 통과 산출과 다르다")
    if wv["decision_log_section"].strip() not in decision_log:
        raise E.EarlyCheckRejected("FRAME_WAIVER_DECISION", "면제 파일의 의사결정로그 절 제목이 영수증 커밋의 의사결정로그에 없다")
    return {"status": FRAME_CHECKED_MODEL_FREE_ONLY, "frame_diag_sha256": bundle,
            "admission_sha256": hashlib.sha256(admission_raw).hexdigest(), "waiver_sha256": hashlib.sha256(waiver_raw).hexdigest(),
            "rules_sha256": rules.raw_sha256, "decision_log_section": wv["decision_log_section"].strip(),
            "step100_geometry_sha256": wv["step100_geometry_sha256"], "scope": MODEL_FREE_SCOPE}
