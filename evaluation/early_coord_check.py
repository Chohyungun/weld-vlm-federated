"""조기 좌표 점검 — 미니스펙 10번(3판, §14 까지)의 평가 쪽 구현.

본 중앙 칸을 공통 출발점에서 몇 step 만 학습한 **버리는 점검 실행**의 생성을, 학습 분할의 "R0 뒤 남은 학습 결함 장"
목록에 대해 프레임 진단의 검사기로 판정한다. 통계와 결합 순서는 `evaluation/frame_diag.py` 를 그대로 부르고, 이 모듈은
그 위에 다섯 가지를 얹는다.

| 무엇 | 미니스펙 |
|---|---|
| 규칙 파일 `configs/early_coord_check.yaml` 읽기 — 진단 규칙 파일과 꼴 · 이름 · digest 가 다르다 | 3절 · 13-2 |
| step 마다 판정 — 대상 전부와 **묶음 단위 비노출 부분이 각각** 통과해야 통과 | 4절 · 14-1 |
| 등록할 step 목록 그대로의 순차 결정(한 번 통과 · 두 step 연속) | 13-1 의 I-2 · 14-2 |
| 모델 없이 가르는 네 검사(원본 크기 · 프로세서 실제 입력 크기 · 좌표 변환 왕복 · 정답 에코) | 7절 |
| 산출물 계약 · 착수 허가의 대조 · 학습 뷰 밖 목록의 거부 | 8절 · 13-1 의 m-2 · 14-3 |

**이 산출물은 채점 대상이 아니다.** 목적 값은 `early_coord_check` 이고 분할은 `train` 이다 — 본채점 진입점과 묶음 검증기는
이 목적의 곁 파일을 받지 않는다(시험이 고정한다). 사유 문자열에는 건수만 싣고 id 를 싣지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

import yaml

from evaluation import frame_diag as F
from evaluation.eval_list import EvalList, set_digest

KIND = "early_coord_check"
"""목적 값 · 산출물 종류 · 원장 줄의 종류. 진단(`frame_diag`)과 다르다."""
SPLIT = "train"
RULES_VERSION = "3"
ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "configs" / "early_coord_check.yaml"
CHECKER_PATH = Path(F.__file__)

PASS, FAIL, INDETERMINATE = F.PASS, F.FAIL, F.INDETERMINATE
CONTINUE = "이어 간다"
SINGLE, TWO_CONSECUTIVE = "single", "two_consecutive"
SEQUENTIAL_RULES = (SINGLE, TWO_CONSECUTIVE)
REASON_FORMAT = "형식"
REASON_NO_NEXT_STEP = "확인할 다음 step 이 없다"
MODEL_FREE_CHECKS = ("original_size", "processor_input", "coord_roundtrip", "gold_echo")

LIMIT_KEYS = frozenset({"max_hours", "max_cost_krw", "price_krw_per_hour", "price_source", "safety_factor"})
RULES_KEYS = frozenset({"kind", "version", "rules", "steps", "n", "gen_limit", "sequential_rule", "max_attempts",
                        "format_gate", "unexposed", "target_list", "model_input_wh", "model_free_checks",
                        "open_cells_when", "actions", "checker_sha256", "discrimination", "sources", "limits"})


class EarlyCheckRejected(ValueError):
    """점검을 하지 않거나 산출을 받지 않는다. 사유에는 건수만 싣는다."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def canonical(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


# ---------------------------------------------------------------- 규칙 파일

@dataclass(frozen=True)
class EarlyRules:
    frame: F.FrameRules
    steps: tuple[int, ...]
    gen_limit: int
    sequential_rule: str | None
    """`single` · `two_consecutive` — 합성 순차 판별력으로 고른다. `None` 이면 미확정이고 점검을 시작하지 않는다."""
    max_attempts: int | None
    """총괄이 첫 점검 실행 전에 적는다. `None` 이면 점검을 시작하지 않는다."""
    min_parse_rate: float
    target_sha256: str
    target_n: int
    model_input_wh: tuple[int, int]
    raw_sha256: str
    discrimination: dict | None = None
    """합성 순차 판별력 영수증의 `{path, sha256}`. `None` 이면 순차 규칙을 고르기 전의 초안이다."""
    limits: dict | None = None
    """점검 실행의 상한과 단가(13-2 의 정본) — `max_hours` · `max_cost_krw` · `price_krw_per_hour` · `price_source` ·
    `safety_factor`. 상한이 비면 점검을 시작하지 않는다. 학습 쪽은 여기서만 읽고 실행 인자는 좁히기만 한다(`effective_limits`)."""


def discrimination_inputs(rules: EarlyRules) -> dict:
    """합성 판별력이 실제로 쓴 판정 입력 — 영수증에 싣고, 규칙 파일과 같은지 시험이 본다."""
    return {"frame_rules": asdict(rules.frame), "steps": list(rules.steps), "target_n": rules.target_n,
            "model_input_wh": list(rules.model_input_wh), "min_parse_rate": rules.min_parse_rate}


def load_rules(raw: bytes) -> EarlyRules:
    """규칙 파일의 바이트 → `EarlyRules`. 꼴 · 종류 · 판 · 검사기 지문을 본다. 진단 규칙 파일(JSON · 다른 칸)은 받지 않는다."""
    try:
        body = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EarlyCheckRejected("RULES_FORMAT", f"규칙 파일을 읽지 못한다: {type(exc).__name__}") from None
    if not isinstance(body, dict) or body.get("kind") != KIND:
        raise EarlyCheckRejected("RULES_KIND", f"규칙 파일의 종류가 {KIND!r} 가 아니다 — 진단 규칙 파일은 받지 않는다")
    if set(body) != RULES_KEYS:
        missing, extra = sorted(RULES_KEYS - set(body)), sorted(set(body) - RULES_KEYS)
        raise EarlyCheckRejected("RULES_KEYS", f"칸이 다르다 — 빠짐 {missing} · 모름 {extra}")
    if str(body["version"]) != RULES_VERSION:
        raise EarlyCheckRejected("RULES_VERSION", f"판 {body['version']!r} 은 이 검사기의 판 {RULES_VERSION!r} 이 아니다")
    if body["checker_sha256"] != lf_sha256(CHECKER_PATH):
        raise EarlyCheckRejected("CHECKER_CHANGED", "규칙 파일이 가리키는 진단 검사기가 지금 검사기와 다르다 — 문턱을 다시 세운다")
    names = {f.name for f in fields(F.FrameRules)}
    if set(body["rules"]) != names:
        raise EarlyCheckRejected("RULES_KEYS", "문턱 칸이 진단 검사기의 칸과 다르다")
    frame = F.FrameRules(**body["rules"])
    steps = body["steps"]
    if not (isinstance(steps, list) and steps and all(type(s) is int and s > 0 for s in steps)
            and all(a < b for a, b in zip(steps, steps[1:]))):
        raise EarlyCheckRejected("RULES_STEPS", "step 목록은 비지 않은 양의 정수 오름차순이다")
    if body["n"] != "all":
        raise EarlyCheckRejected("RULES_N", "표본 n 은 대상 전부(`all`)다 — 쌍 하한 때문에 줄이지 않는다")
    if type(body["gen_limit"]) is not int or body["gen_limit"] < 1:
        raise EarlyCheckRejected("RULES_GEN_LIMIT", "생성 길이 L 은 양의 정수다")
    seq = body["sequential_rule"]
    if seq is not None and seq not in SEQUENTIAL_RULES:
        raise EarlyCheckRejected("RULES_SEQUENTIAL", f"순차 규칙은 {SEQUENTIAL_RULES} 가운데 하나이거나 null 이다")
    att = body["max_attempts"]
    if att is not None and (type(att) is not int or att < 1):
        raise EarlyCheckRejected("RULES_ATTEMPTS", "허용 시도 수는 1 이상의 정수이거나 null 이다")
    gate = body["format_gate"]
    rate = gate.get("min_parse_rate") if isinstance(gate, dict) else None
    if not (isinstance(rate, (int, float)) and 0 < rate <= 1):
        raise EarlyCheckRejected("RULES_FORMAT_GATE", "형식 관문의 파싱률 문턱은 (0, 1] 이다")
    un = body["unexposed"]
    if not (isinstance(un, dict) and un.get("unit") == "group" and un.get("must_pass") is True):
        raise EarlyCheckRejected("RULES_UNEXPOSED", "비노출은 묶음 단위이고 비노출 부분도 통과해야 한다(14-1)")
    tl = body["target_list"]
    if not (isinstance(tl, dict) and isinstance(tl.get("sha256"), str) and len(tl["sha256"]) == 64
            and type(tl.get("n")) is int and tl["n"] > 0):
        raise EarlyCheckRejected("RULES_TARGET", "대상 목록은 sha256 과 장 수로 적는다")
    if list(body["model_free_checks"]) != list(MODEL_FREE_CHECKS):
        raise EarlyCheckRejected("RULES_CHECKS", f"모델 없는 검사는 {MODEL_FREE_CHECKS} 넷이다")
    wh = body["model_input_wh"]
    if not (isinstance(wh, list) and len(wh) == 2 and all(type(v) is int and v > 0 for v in wh)):
        raise EarlyCheckRejected("RULES_INPUT_WH", "model_input_wh 는 양의 정수 둘이다")
    disc = body["discrimination"]
    if disc is not None and not (isinstance(disc, dict) and isinstance(disc.get("path"), str)
                                 and isinstance(disc.get("sha256"), str) and len(disc["sha256"]) == 64):
        raise EarlyCheckRejected("RULES_DISCRIMINATION", "판별력 영수증은 {path, sha256} 이거나 null 이다")
    if seq is not None and disc is None:
        raise EarlyCheckRejected("RULES_DISCRIMINATION", "순차 규칙을 적었으면 그 근거인 판별력 영수증도 적는다")
    lim = body["limits"]

    def _pos(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0

    if not (isinstance(lim, dict) and set(lim) == LIMIT_KEYS
            and all(lim[k] is None or _pos(lim[k]) for k in ("max_hours", "max_cost_krw"))
            and _pos(lim["price_krw_per_hour"]) and isinstance(lim["price_source"], str) and lim["price_source"].strip()
            and _pos(lim["safety_factor"]) and lim["safety_factor"] >= 1):
        raise EarlyCheckRejected("RULES_LIMITS", f"상한 칸은 {sorted(LIMIT_KEYS)} 이다 — 상한은 양수이거나 null, 단가는 양수와 "
                                                 "출처 문자열, 계수는 1 이상")
    return EarlyRules(frame, tuple(steps), body["gen_limit"], seq, att, float(rate), tl["sha256"].lower(), tl["n"],
                      (wh[0], wh[1]), hashlib.sha256(raw).hexdigest(), disc, dict(lim))


def effective_limits(rules: EarlyRules, *, max_hours: float | None = None, max_cost_krw: float | None = None) -> dict:
    """학습 쪽 비용 대조가 쓸 상한 — 규칙 파일의 `limits` 가 정본이고 실행 인자는 **좁히기만** 한다(넓히면 거부).

    금액은 시간 × 단가 × 계수로 센다 — `cost_krw(hours)` 와 같은 식. 돌려주는 사전의 `cost_krw_of_hour` 는 그 한 시간 값이다.
    """
    require_ready(rules)
    lim = rules.limits
    out_h, out_w = float(lim["max_hours"]), float(lim["max_cost_krw"])
    for name, given, base in (("max_hours", max_hours, out_h), ("max_cost_krw", max_cost_krw, out_w)):
        if given is not None and float(given) > base:
            raise EarlyCheckRejected("LIMITS_WIDENED", f"실행 인자 {name} 가 규칙 파일의 상한보다 넓다 — 좁히기만 한다")
    if max_hours is not None:
        out_h = float(max_hours)
    if max_cost_krw is not None:
        out_w = float(max_cost_krw)
    rate = float(lim["price_krw_per_hour"]) * float(lim["safety_factor"])
    return {"max_hours": out_h, "max_cost_krw": out_w, "cost_krw_of_hour": rate,
            "price_krw_per_hour": float(lim["price_krw_per_hour"]), "safety_factor": float(lim["safety_factor"]),
            "price_source": lim["price_source"]}


def require_ready(rules: EarlyRules) -> None:
    """점검 실행을 시작해도 되는가 — 순차 규칙과 허용 시도 수가 정해져 있어야 한다."""
    if rules.sequential_rule is None:
        raise EarlyCheckRejected("SEQUENTIAL_UNDECIDED", "순차 규칙이 미확정이다 — 두 규칙 모두 합성 거짓 통과가 0 이 아니면 시작하지 않는다(14-2)")
    if rules.max_attempts is None:
        raise EarlyCheckRejected("ATTEMPTS_UNSET", "허용 시도 수가 비었다 — 첫 점검 실행 전에 적는다")
    if not rules.limits or rules.limits.get("max_hours") is None or rules.limits.get("max_cost_krw") is None:
        raise EarlyCheckRejected("LIMITS_UNSET", "시간 · 금액 상한이 비었다 — 학습 쪽 비용표에서 골라 첫 점검 실행 전에 적는다")


# ---------------------------------------------------------------- 대상 목록

def target_ids(pair_rows: Iterable[Mapping], removed: Collection[str]) -> list[str]:
    """봉인 페어의 학습 행 가운데 골격 결함이 하나 이상인 장, 그 가운데 R0 뒤 남은 장 — 정렬해 돌려준다(13-1 의 m-4).

    `pair_rows` 는 한 줄씩 읽은 페어 행이다 — 페어 파일 전체를 메모리에 올리지 않는다.
    """
    out = set()
    for r in pair_rows:
        if r.get("split") == SPLIT and (r.get("skeleton") or {}).get("defects"):
            out.add(str(r["image_id"]))
    return sorted(out - set(removed))


def check_target_list(el: EvalList, split_of: Mapping[str, str], rules: EarlyRules | None = None) -> None:
    """목록의 id 가 모두 학습 뷰에 있고 학습 분할인가. 없는 id 를 건너뛰지 않는다 — 평가 id 는 학습 뷰에 없다(13-1 의 m-2)."""
    missing = sum(1 for i in el.ids if i not in split_of)
    other = sum(1 for i in el.ids if i in split_of and split_of[i] != SPLIT)
    if missing or other:
        raise EarlyCheckRejected("TARGET_NOT_TRAIN", f"점검 목록에 학습 뷰 밖 id 가 있다 — 뷰에 없음 {missing}개 · 학습 아님 {other}개")
    if rules is not None and (el.file_sha256 != rules.target_sha256 or len(el) != rules.target_n):
        raise EarlyCheckRejected("TARGET_MISMATCH", f"점검 목록이 규칙 파일의 대상과 다르다 — 목록 {len(el)}개 · 규칙 {rules.target_n}개")


# ---------------------------------------------------------------- 모델 없이 가르는 네 검사

def _result(name: str, ids: Collection[str], n: int, n_bad: int, **extra) -> dict:
    """검사 결과 한 건. 덮은 장의 수와 집합 지문을 싣는다 — 산출이 대상 목록과 맞댄다(I-2). id 는 싣지 않는다."""
    ids = list(ids)
    return {"name": name, "status": PASS if n and not n_bad else FAIL, "n": n, "n_bad": n_bad,
            "n_ids": len(set(ids)), "ids_set_sha256": set_digest(ids), **extra}


def check_original_size(observed: Mapping[str, tuple[int, int]], manifest: Mapping[str, tuple[int, int]]) -> dict:
    """장마다 이미지 머리의 (가로, 세로)가 매니페스트의 크기와 같은가. 빠진 장도 어긋남으로 센다."""
    bad = sum(1 for i, wh in manifest.items() if tuple(observed.get(i, ())) != tuple(wh))
    return _result("original_size", manifest.keys(), len(manifest), bad)


def processor_input_wh(grid_thw: Sequence[int], patch: int) -> tuple[int, int]:
    """실제 입력 치수 = (w × patch, h × patch). `image_grid_thw` 는 (t, h, w) 다(7절 — merge 를 곱하지 않는다)."""
    _, h, w = grid_thw
    return w * patch, h * patch


def image_tokens(grid_thw: Sequence[int], merge: int) -> int:
    """이미지 토큰 수 = t · h · w ÷ merge². 나누어떨어지지 않으면 격자가 프로세서의 것이 아니다."""
    t, h, w = grid_thw
    n, d = t * h * w, merge * merge
    if n % d:
        raise EarlyCheckRejected("GRID_NOT_DIVISIBLE", "격자 칸 수가 merge² 로 나누어떨어지지 않는다")
    return n // d


def check_processor_input(grids: Mapping[str, Sequence[int]], *, patch: int, merge: int,
                          model_input_wh: tuple[int, int]) -> dict:
    """장마다 프로세서의 실제 입력 크기가 등록의 상정 값과 같은가. 토큰 수는 함께 싣는다(판정에는 크기만 쓴다)."""
    bad, tokens = 0, set()
    for g in grids.values():
        if processor_input_wh(g, patch) != tuple(model_input_wh):
            bad += 1
        tokens.add(image_tokens(g, merge))
    return _result("processor_input", grids.keys(), len(grids), bad, image_tokens=sorted(tokens))


def check_coord_roundtrip(golds: Mapping[str, tuple[tuple[int, int], Sequence[Sequence[float]]]], *,
                          input_wh: tuple[int, int], coord_space: str) -> dict:
    """원본 → 학습용 변환 → 정수화 → 채점용 역변환의 오차가 규약의 상한 안인가. 학습 쪽 좌표 모듈의 두 함수를 그대로 쓴다."""
    from vlm.coords import CoordCfg, ImageGeom, roundtrip_budget_px, roundtrip_error_px

    cfg = CoordCfg(coord_space=coord_space)
    bad = n = 0
    for (w, h), boxes in golds.values():
        geom = ImageGeom(w, h, input_wh[0], input_wh[1])
        budget = roundtrip_budget_px(geom, cfg)
        for b in boxes:
            n += 1
            bad += int(roundtrip_error_px(b, geom, cfg) > budget)
    return _result("coord_roundtrip", golds.keys(), n, bad)


def check_gold_echo(images: Sequence[F.FrameImage], *, budget_px: float) -> dict:
    """정답을 생성 자리에 넣어 채점 경로를 지난 예측이 정답과 같은가 — 장마다 박스 수가 같고 상호 최대 IoU 짝이 모두 서며
    짝의 좌표 차가 `budget_px` 안이다. 허용치는 등록의 에코 기준(`canary.echo_max_coord_diff`)을 호출자가 준다(m-5)."""
    bad = 0
    for im in images:
        pairs = F.mutual_pairs(im.pred, im.gold) if im.pred else []
        ok = len(im.pred) == len(im.gold) and len(pairs) == len(im.gold) and all(
            max(abs(a - b) for a, b in zip(im.pred[i], im.gold[j])) <= budget_px for i, j in pairs)
        bad += int(not ok)
    return _result("gold_echo", [im.image_id for im in images], len(images), bad)


def model_free_ok(results: Sequence[Mapping]) -> bool:
    """네 검사가 모두 있고 모두 통과인가."""
    by = {r.get("name"): r.get("status") for r in results}
    return set(by) == set(MODEL_FREE_CHECKS) and all(by[k] == PASS for k in MODEL_FREE_CHECKS)


# ---------------------------------------------------------------- step 의 판정

def _pair_rate(images: Sequence[F.FrameImage]) -> float | None:
    defect = [im for im in images if im.gold]
    if not defect:
        return None
    return sum(1 for im in defect if im.pred and F.mutual_pairs(im.pred, im.gold)) / len(defect)


def check_coverage(ids: Sequence[str], target: EvalList, where: str) -> None:
    """`ids` 가 대상 목록의 **전부**이고 대상 밖 id 가 없으며 겹침이 없는가. 사유에는 건수만 싣는다(I-2)."""
    got = set(ids)
    missing, extra, dup = len(target.id_set - got), len(got - target.id_set), len(ids) - len(got)
    if missing or extra or dup:
        raise EarlyCheckRejected("TARGET_COVERAGE", f"{where} 이 대상 전부가 아니다 — 빠짐 {missing}개 · 대상 밖 {extra}개 · 겹침 {dup}개")


def step_verdict(images: Sequence[F.FrameImage], exposed_groups: Collection[str], rules: EarlyRules, *,
                 parse_rate: float, target: EvalList) -> dict:
    """한 step 의 판정. 형식 관문 → 대상 전부 · 묶음 단위 비노출 부분의 진단 판정 → 결합.

    결합(14-1): 어느 쪽이든 불통과면 불통과, 둘 다 통과면 통과, 그 밖은 판정 불가 — 비노출 부분이 표본 하한에 못 미치는
    경우도 판정 불가다. 노출 · 비노출의 짝 성립률은 판정에 쓰지 않고 싣는다.
    """
    check_coverage([im.image_id for im in images], target, "step 판정의 장")
    exposed = set(exposed_groups)
    unexposed = [im for im in images if im.group_id not in exposed]
    shown = [im for im in images if im.group_id in exposed]
    base = {"parse_rate": parse_rate, "n_images": len(images), "n_unexposed_images": len(unexposed),
            "ids_set_sha256": target.set_sha256,
            "pair_rate_exposed": _pair_rate(shown), "pair_rate_unexposed": _pair_rate(unexposed)}
    if not (isinstance(parse_rate, (int, float)) and math.isfinite(parse_rate)) or parse_rate < rules.min_parse_rate:
        return {**base, "status": INDETERMINATE, "reason": REASON_FORMAT, "all": None, "unexposed": None}
    v_all = F.judge(images, rules.model_input_wh, rules.frame)
    v_un = F.judge(unexposed, rules.model_input_wh, rules.frame)
    if FAIL in (v_all.status, v_un.status):
        status = FAIL
    elif v_all.status == PASS and v_un.status == PASS:
        status = PASS
    else:
        status = INDETERMINATE
    why = [f"{side}: {v.reason}" for side, v in (("전부", v_all), ("비노출", v_un)) if v.status != PASS and v.reason]
    return {**base, "status": status, "reason": " · ".join(why), "all": v_all.as_dict(), "unexposed": v_un.as_dict()}


# ---------------------------------------------------------------- 순차 결정

ACTIONS = {
    PASS: "점검 실행을 끝낸다. 모델 없는 네 검사가 모두 통과였으면 네 칸을 연다. 본 중앙은 공통 출발점에서 새로 시작한다",
    FAIL: "모든 학습을 멈춘다. 총괄 · 사용자가 좌표 규약 · 프롬프트 · 프로세서를 검토하고 고친 뒤 네 검사부터, 공통 출발점에서 "
          "새 시도. 실패 실행의 산출은 옆으로 옮겨 보존한다",
    INDETERMINATE: "멈추고 보고한다. 네 칸을 열지 않는다",
    CONTINUE: "점검 실행을 다음 step 까지 잇는다",
}


def decide(statuses: Sequence[str], n_steps: int, rule: str) -> dict:
    """step 마다의 판정(앞에서부터) → 결정. 등록한 step 수 `n_steps` 와 순차 규칙으로 정한다.

    - 불통과는 어느 step 이든, 보류 중이어도 그 자리에서 멈춘다.
    - 한 번 통과(`single`): 처음 통과한 step 에서 통과.
    - 두 step 연속(`two_consecutive`): 첫 통과는 보류이고 다음 step 의 통과로 확정한다. 다음 step 이 판정 불가면 보류가
      풀린다. 마지막 step 의 통과는 확인할 다음 step 이 없어 판정 불가다.
    - 끝까지 통과가 서지 않으면 마지막 step 에서 판정 불가. 아직 step 이 남았으면 `이어 간다`.
    """
    if rule not in SEQUENTIAL_RULES:
        raise EarlyCheckRejected("SEQUENTIAL_UNDECIDED", "순차 규칙이 정해지지 않았다")
    if len(statuses) > n_steps:
        raise EarlyCheckRejected("STEPS_EXCEEDED", "등록한 step 수보다 판정이 많다")
    pending = False
    for i, s in enumerate(statuses):
        if s == FAIL:
            return {"outcome": FAIL, "step_index": i, "reason": "", "action": ACTIONS[FAIL]}
        if s == PASS:
            if rule == SINGLE or pending:
                return {"outcome": PASS, "step_index": i, "reason": "", "action": ACTIONS[PASS]}
            pending = True
        else:
            pending = False
    i = len(statuses) - 1
    if len(statuses) < n_steps:
        return {"outcome": CONTINUE, "step_index": i, "reason": "", "action": ACTIONS[CONTINUE]}
    reason = REASON_NO_NEXT_STEP if pending else ""
    return {"outcome": INDETERMINATE, "step_index": i, "reason": reason, "action": ACTIONS[INDETERMINATE]}


def opens_four_cells(model_free: Sequence[Mapping], decision: Mapping) -> bool:
    """네 칸을 여는 기준(의사결정로그 10-07): 모델 없는 네 검사가 모두 통과이고 이 점검이 통과다."""
    return model_free_ok(model_free) and decision.get("outcome") == PASS


# ---------------------------------------------------------------- 산출물과 착수 허가

# 규칙 파일 · 대상 목록의 지문은 산출 함수가 규칙과 목록에서 채운다 — 호출자가 적지 않는다
EVIDENCE_KEYS = ("lineage", "model_free_files", "step_outputs", "attempt",
                 "start_adapter_digest", "step_adapter_digests", "processor_cfg_sha256", "model_input_wh",
                 "pin_sha256", "check_run_id", "seed")


def _check_cover_records(model_free: Sequence[Mapping], steps: Sequence[Mapping], set_sha: str, n: int) -> None:
    for r in model_free:
        if r.get("ids_set_sha256") != set_sha or r.get("n_ids") != n:
            raise EarlyCheckRejected("TARGET_COVERAGE", f"모델 없는 검사 {r.get('name')!r} 이 대상 전부를 덮지 않는다")
    for i, s in enumerate(steps):
        if s.get("ids_set_sha256") != set_sha or s.get("n_images") != n:
            raise EarlyCheckRejected("TARGET_COVERAGE", f"{i + 1} 번째 step 판정이 대상 전부를 덮지 않는다")


def build_output(*, rules: EarlyRules, target: EvalList, model_free: Sequence[Mapping], steps: Sequence[Mapping],
                 evidence: Mapping) -> dict:
    """`early_coord_check.json` 의 본문(14-3 의 증거 묶음). 이 파일의 sha256 이 착수 허가의 단일 키다.

    만들기 전에 본다 — 규칙이 시작할 수 있는 상태인가(순차 규칙 · 시도 수 · 상한), 대상 목록이 규칙의 대상인가, 네 검사와
    step 판정이 대상 **전부**를 덮었는가, 시도 번호가 허용 시도 수 안인가.
    """
    require_ready(rules)
    if target.file_sha256 != rules.target_sha256 or len(target) != rules.target_n:
        raise EarlyCheckRejected("TARGET_MISMATCH", "산출의 대상 목록이 규칙 파일의 대상과 다르다")
    missing = [k for k in EVIDENCE_KEYS if k not in evidence]
    if missing:
        raise EarlyCheckRejected("EVIDENCE_MISSING", f"증거 묶음의 칸이 빠졌다 — {missing}")
    if set(evidence.get("model_free_files", {})) != set(MODEL_FREE_CHECKS):
        raise EarlyCheckRejected("EVIDENCE_MISSING", "모델 없는 네 검사의 산출 파일 지문이 넷이 아니다")
    seed = evidence["seed"]
    if type(seed) is not int or seed < 1:
        raise EarlyCheckRejected("EVIDENCE_SEED", "증거의 seed 는 시드 번호(1 이상의 정수, seed_index)다")
    att = evidence["attempt"]
    if type(att) is not int or not 1 <= att <= rules.max_attempts:
        raise EarlyCheckRejected("ATTEMPTS_EXCEEDED", f"시도 번호가 허용 시도 수 {rules.max_attempts} 안의 양의 정수가 아니다")
    if sorted(r.get("name") for r in model_free) != sorted(MODEL_FREE_CHECKS):
        raise EarlyCheckRejected("EVIDENCE_MISSING", "모델 없는 네 검사의 결과가 넷이 아니다")
    _check_cover_records(model_free, steps, target.set_sha256, len(target))
    decision = decide([s["status"] for s in steps], len(rules.steps), rules.sequential_rule)
    return {"kind": KIND, "purpose": KIND, "split": SPLIT, "scoreable": False,
            "decision": decision, "opens_four_cells": opens_four_cells(model_free, decision),
            "steps": [{"step": rules.steps[i], **s} for i, s in enumerate(steps)],
            "model_free": list(model_free), "n_targets": len(target),
            "evidence": {**dict(evidence), "rules_sha256": rules.raw_sha256, "target_list_sha256": target.file_sha256,
                         "target_set_sha256": target.set_sha256, "limits": dict(rules.limits)}}


PERMIT_EXPECTED_KEYS = ("start_adapter_digest", "pin_sha256", "check_run_id", "seed")
"""착수 허가의 대조에서 지금 실행이 **반드시** 내야 하는 값(3판). 다른 칸은 받지 않는다(모르는 칸이면 거부).

- 규칙 파일 · 대상 목록의 지문은 `rules` 객체에서 온다 — 대조값에 `rules_sha256` 을 따로 넘기지 않는다.
- 프로세서 설정은 착수 고정(`pin_sha256`)의 `common` 칸이 이미 묶는다(통합형 설정 블록 · `processor_kwargs`) — 대조값에서 뺐다.
- `seed` 는 시드 **번호**(`seed_index`, 1 · 2 · 3 …)다 — 시드 값이 아니다. 산출도 같은 뜻으로 싣는다.
- `check_run_id` 와 착수 허가 키(`expected_sha256`)는 통과 파일 **밖**의 기록(점검 통과 때 남긴 시도 기록)에서 온다.
"""


def verify_permit(raw: bytes, *, expected_sha256: str, rules: EarlyRules, expected: Mapping, file_sha256_of) -> dict:
    """네 학습 조건과 본 중앙의 **시작 · 재개마다** 부르는 대조(14-3). 어긋나면 `EarlyCheckRejected` — 시작하지 않는다.

    - `expected_sha256`: 통과 파일의 sha256 — 착수 허가의 단일 키. 바이트가 다르면 읽기 전에 거부한다.
    - `rules`: 지금 쓰는 규칙 파일. 순차 규칙 · 시도 수 · 상한이 서 있어야 하고, 통과 파일의 규칙 · 대상 지문이 이 값과 같아야 한다.
    - `expected`: 지금 실행의 출발점 · 착수 고정 · 점검 실행 신원 · 시드 번호(`PERMIT_EXPECTED_KEYS` 전부, 그 밖의 칸은 거부).
    - `file_sha256_of(path)`: 네 검사 산출 파일의 **지금** 지문. 없거나 다르면 거부한다.

    파일이 스스로 적은 결정과 네 칸 판정은 믿지 않는다 — step 판정 · 네 검사에서 다시 내어 같을 때만 받는다.
    """
    if not isinstance(expected_sha256, str) or hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
        raise EarlyCheckRejected("PERMIT_SHA", "통과 파일의 sha256 이 착수 허가의 키와 다르다")
    if not callable(file_sha256_of):
        raise EarlyCheckRejected("PERMIT_ARGS", "네 검사 산출 파일의 지문을 내는 함수가 없다 — 파일 대조를 건너뛰지 않는다")
    require_ready(rules)
    absent = [k for k in PERMIT_EXPECTED_KEYS if k not in expected]
    if absent:
        raise EarlyCheckRejected("PERMIT_EXPECTED_INCOMPLETE", f"지금 실행의 대조값이 빠졌다 — {absent}")
    unknown = sorted(set(expected) - set(PERMIT_EXPECTED_KEYS))
    if unknown:
        raise EarlyCheckRejected("PERMIT_EXPECTED_UNKNOWN", f"대조하지 않는 칸을 넘겼다 — {unknown}(규칙은 rules 로, 프로세서는 착수 고정으로)")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EarlyCheckRejected("PERMIT_FORMAT", "통과 파일을 읽지 못한다") from None
    if not isinstance(obj, dict) or obj.get("kind") != KIND or obj.get("purpose") != KIND or obj.get("split") != SPLIT:
        raise EarlyCheckRejected("PERMIT_KIND", "조기 좌표 점검의 산출이 아니거나 목적 · 분할이 다르다")
    ev = obj.get("evidence") or {}
    if ev.get("rules_sha256") != rules.raw_sha256 or ev.get("target_list_sha256") != rules.target_sha256:
        raise EarlyCheckRejected("PERMIT_MISMATCH", "통과 파일의 규칙 파일 · 대상 목록 지문이 지금 규칙과 다르다")
    differ = sorted(k for k in PERMIT_EXPECTED_KEYS if ev.get(k) != expected[k])
    if differ:
        raise EarlyCheckRejected("PERMIT_MISMATCH", f"통과 파일의 증거가 지금 실행과 다르다 — {differ}")
    att = ev.get("attempt")
    if type(att) is not int or not 1 <= att <= rules.max_attempts:
        raise EarlyCheckRejected("ATTEMPTS_EXCEEDED", "통과 파일의 시도 번호가 허용 시도 수 밖이다")
    steps, model_free = obj.get("steps") or [], obj.get("model_free") or []
    if not isinstance(steps, list) or not isinstance(model_free, list) or not all(isinstance(s, dict) for s in steps):
        raise EarlyCheckRejected("PERMIT_FORMAT", "step 판정 · 네 검사의 꼴이 아니다")
    _check_cover_records(model_free, steps, ev.get("target_set_sha256"), rules.target_n)
    decision = decide([s.get("status") for s in steps], len(rules.steps), rules.sequential_rule)
    if decision != obj.get("decision") or opens_four_cells(model_free, decision) != obj.get("opens_four_cells"):
        raise EarlyCheckRejected("PERMIT_RECOMPUTE", "통과 파일이 적은 결정 · 네 칸 판정이 그 안의 step 판정 · 네 검사에서 다시 낸 값과 다르다")
    if decision["outcome"] != PASS or not opens_four_cells(model_free, decision):
        raise EarlyCheckRejected("PERMIT_NOT_PASSED", "통과가 아니거나 네 칸을 여는 기준을 채우지 않았다")
    files = ev.get("model_free_files") or {}
    if set(files) != set(MODEL_FREE_CHECKS):
        raise EarlyCheckRejected("PERMIT_MISMATCH", "모델 없는 네 검사의 산출 파일 지문이 넷이 아니다")
    bad = sorted(k for k, f in files.items() if file_sha256_of(f.get("path")) != f.get("sha256"))
    if bad:
        raise EarlyCheckRejected("PERMIT_MISMATCH", f"네 검사의 산출 파일이 없거나 바뀌었다 — {bad}")
    return obj


def unexposed_counts(consumed_by_step: Sequence[Collection[str]], group_of: Mapping[str, str],
                     target_boxes: Mapping[str, int], rules: EarlyRules) -> list[dict]:
    """비노출 부분의 미리 세기(14-1) — 학습 쪽의 표본 순서에서 step 마다 그때까지 소비한 학습 행의 image_id 를 받아, 대상 가운데
    묶음이 한 번도 나오지 않은 장 · 묶음 · 정답 박스를 센다. 모델 없이 첫 학습 전에 돈다. 하한에 못 미치는 step 은 통과할 수 없다."""
    out = []
    for i, consumed in enumerate(consumed_by_step):
        exposed = {group_of[x] for x in consumed if x in group_of}
        ids = [x for x in target_boxes if group_of.get(x) not in exposed]
        groups = {group_of[x] for x in ids}
        boxed = {group_of[x] for x in ids if target_boxes[x] > 0}
        boxes = sum(target_boxes[x] for x in ids)
        f = rules.frame
        out.append({"step_index": i, "n_images": len(ids), "n_groups": len(groups), "n_boxes": boxes,
                    "n_groups_with_box": len(boxed),
                    "floors_reachable": len(ids) >= f.n_img and len(groups) >= f.min_groups
                    and boxes >= f.min_pairs and len(boxed) >= f.min_pair_groups})
    return out


def refuse_for_scoring(meta: Mapping) -> None:
    """채점 쪽 방어 — 점검의 산출 · 곁 파일은 채점 대상이 아니다."""
    if meta.get("kind") == KIND or meta.get("purpose") == KIND:
        raise EarlyCheckRejected("NOT_SCOREABLE", "조기 좌표 점검의 산출은 채점하지 않는다")
