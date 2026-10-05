"""프레임 진단의 규칙 파일 — 꼴 · 만들기 · 읽기. 07번 계약 §31-1 의 1 · §32-9.

규칙 파일은 **정규 JSON 한 파일**이다(`configs/registration/frame_diag_rules-{YYYYMMDD}-{n}.json`). 진단 등록의 생성 쪽 칸
`frame_diag_rules_path` · `frame_diag_rules_sha256` 이 그 경로와 바이트 해시를 가리키고, 진입점은 **영수증의 커밋에서** 그 바이트를 읽는다.

| 칸 | 무엇 |
|---|---|
| `version` | 이 꼴의 판(`"2"` — 요구 표본 수 칸을 더했다) |
| `rules` | 문턱과 계산 설정 — `FrameRules` 의 칸 그대로(`n_img` · `s_min` · `w_s` · `w_r` · `w_mix` · `delta` · `trim` · `n_boot` · `boot_seed` · `axis_ratio_width` · `min_pairs` · `se_divisor` · `min_groups` · `min_pair_groups` · `min_valid_boot`). 검사기가 판정에 쓰는 표본 하한은 여기의 넷이다 |
| `sample_requirements` | 문턱 절차가 낸 **검사별 요구 표본 수** — 잡음 계단 · 통계 · 축마다 요구 장 · 묶음 · (축 배율이면) 쌍 수(07번 §31-2 · 3판 8-7). 진입점은 계단마다 관측 수와 맞대 **충족 · 미달**만 싣는다 |
| `checker_sha256` | 검사기(`evaluation/frame_diag.py`) 바이트(LF)의 sha256 — **지금 검사기와 같아야** 쓴다. 검사기가 바뀌면 문턱을 세운 판과 다르다 |
| `expected_sha256` · `cases_sha256` · `procedure_sha256` | 기대 판정 파일 · 합성 사례 모듈 · 문턱 절차 모듈의 sha256 — 문턱이 어디서 왔는지 |
| `noise_ladder` · `summary` · `sources` | 잡음 사다리 · 합성 정답의 요약값 · 그 출처(가정을 가정이라 적는다) |

규칙을 바꾸려면 합성 사례 시험부터 다시 하고 **사용자 승인**을 받는다(§31-1 의 3). 한 번 커밋한 규칙 파일은 고치지 않는다 — 고치면 새 규칙이다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path

from evaluation import frame_diag as F

RULES_VERSION = "2"
KEYS = frozenset({"version", "rules", "sample_requirements", "checker_sha256", "expected_sha256", "cases_sha256",
                  "procedure_sha256", "noise_ladder", "summary", "sources"})
REQ_KEYS = frozenset({"noise_step", "stat", "axis", "width", "n0", "n_required", "g0", "groups_required", "q0",
                      "pairs_required"})
UNDECIDABLE = "정할 수 없음"
CHECKER_PATH = Path(F.__file__)


class RulesFileError(ValueError):
    """규칙 파일을 쓰지 않는다 — 까닭을 든다."""


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def canonical(body: dict) -> bytes:
    return (json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def build_rules_file(rules: F.FrameRules, *, sample_requirements: list[dict], expected_sha256: str, cases_sha256: str,
                     procedure_sha256: str, noise_ladder, summary: dict, sources: dict) -> bytes:
    body = {"version": RULES_VERSION, "rules": asdict(rules), "sample_requirements": list(sample_requirements),
            "checker_sha256": lf_sha256(CHECKER_PATH),
            "expected_sha256": expected_sha256, "cases_sha256": cases_sha256, "procedure_sha256": procedure_sha256,
            "noise_ladder": list(noise_ladder), "summary": summary, "sources": sources}
    return canonical(body)


def load_rules_file(raw: bytes) -> tuple[F.FrameRules, dict]:
    """바이트 → `(FrameRules, 본문)`. 정규형 · 칸 · 판 · 검사기 지문을 본다. 어긋나면 `RulesFileError`."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise RulesFileError("규칙 파일이 JSON 이 아니다") from None
    if not isinstance(body, dict) or set(body) != KEYS:
        raise RulesFileError("규칙 파일의 칸이 꼴과 다르다")
    if canonical(body) != raw:
        raise RulesFileError("규칙 파일이 정규형이 아니다 — 손으로 고친 파일은 받지 않는다")
    if body["version"] != RULES_VERSION:
        raise RulesFileError(f"규칙 파일의 판이 {body['version']!r} 다")
    want = {f.name for f in fields(F.FrameRules)}
    if not isinstance(body["rules"], dict) or set(body["rules"]) != want:
        raise RulesFileError("규칙의 칸이 검사기의 칸과 다르다")
    _check_requirements(body["sample_requirements"])
    if body["checker_sha256"] != lf_sha256(CHECKER_PATH):
        raise RulesFileError("규칙을 세운 검사기와 지금 검사기가 다르다 — 문턱 절차를 다시 한다(§31-1 의 3)")
    try:
        rules = F.FrameRules(**body["rules"])
    except TypeError as exc:
        raise RulesFileError(f"규칙을 만들지 못했다 — {exc}") from None
    return rules, body


def _count(v) -> bool:
    return v is None or (type(v) is int and v >= 0)


def _check_requirements(reqs) -> None:
    if not isinstance(reqs, list) or not reqs:
        raise RulesFileError("요구 표본 수(sample_requirements)가 비었다 — 문턱 절차의 산출을 싣는다")
    for i, r in enumerate(reqs):
        if not isinstance(r, dict) or set(r) != REQ_KEYS:
            raise RulesFileError(f"요구 표본 수의 {i}번째 행의 칸이 꼴과 다르다")
        if not all(_count(r[k]) for k in ("n0", "n_required", "g0", "groups_required", "q0", "pairs_required")):
            raise RulesFileError(f"요구 표본 수의 {i}번째 행에 0 이상의 정수(또는 null)가 아닌 수가 있다")
        if (r["stat"] == "m") != (r["q0"] is not None):
            raise RulesFileError(f"요구 표본 수의 {i}번째 행 — 쌍 수는 축 배율(m) 행에만 있다")


def requirement_status(body: dict, verdict: dict) -> dict:
    """잡음 계단마다 관측 수가 그 계단의 **가장 큰 요구 수**에 닿는가 — `충족` · `미달` · `정할 수 없음`(요구 수가 없다).

    관측 수는 판정의 `n_images` · `n_image_groups` · `n_pairs` · `n_pair_groups` 다. 요구 수 자체와 표준오차는 싣지 않는다.
    """
    seen = {"images": verdict["n_images"], "image_groups": verdict["n_image_groups"], "pairs": verdict["n_pairs"],
            "pair_groups": verdict["n_pair_groups"]}
    out: dict[str, dict] = {}
    for step in sorted({r["noise_step"] for r in body["sample_requirements"]}):
        rows = [r for r in body["sample_requirements"] if r["noise_step"] == step]
        img = [r for r in rows if r["stat"] != "m"]
        pair = [r for r in rows if r["stat"] == "m"]
        need = {"images": [r["n_required"] for r in img], "image_groups": [r["groups_required"] for r in img],
                "pairs": [r["pairs_required"] for r in pair], "pair_groups": [r["groups_required"] for r in pair]}
        out[str(step)] = {k: (UNDECIDABLE if not v or any(x is None for x in v)
                              else F.MET if seen[k] >= max(v) else F.UNMET) for k, v in need.items()}
    return out
