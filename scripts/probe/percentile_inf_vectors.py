"""백분위 방식이 ±무한을 만났을 때의 거동을 **합성 벡터로** 잰다.

회복률 구간의 백분위·무한 처리 규약(보류-4)을 정하기 전에 알아야 하는 것은 라이브러리의
거동이다 — 유한과 무한 사이를 보간하면 무엇이 나오는지, 무한 꼬리가 몇 건일 때 끝점이
무한이 되는지. **이것은 실험 결과가 아니다.** 규약을 고른 뒤에야 볼 수 있는 값이 아니라
고르기 전에 알 수 있는 값이다.

## 입력은 이 파일 안의 합성 벡터뿐이다

파일을 하나도 읽지 않는다. 읽는 인자가 없고, 측정 구간에서 파일 열기가 한 번이라도
일어나면 값을 내지 않고 멈춘다(`_Watch`). 쓰는 파일은 `--out` 하나이고 배타 생성이다.

## 벡터의 꼴

길이 `n` 의 오름차순 벡터를 `−무한` `n₋` 건 · 유한 · `+무한` `n₊` 건으로 만든다.
**유한값은 자기 자리 번호다**(0부터). 그래서 끝점으로 나온 유한값이 곧 집힌 순위다 —
`49.0` 이면 49번째 추첨이고 `49.975` 면 49번째와 50번째 사이다.
넣을 때는 자리를 섞는다(정렬된 입력에만 서는 거동을 재지 않으려고).

## 맞대는 것

검토 27번의 3-4절과 4-3절의 표는 라이브러리 소스에서 손으로 도출한 것이다. 그 칸을
`CLAIMS_*` 에 **도출된 그대로** 적고 측정과 칸마다 맞댄다. 손 도출과 이 실행이 같은 값을
낼 때까지 어느 쪽도 인용하지 않는다.

사용: python scripts/probe/percentile_inf_vectors.py --out <새 JSON 경로>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
from pathlib import Path

import numpy as np

ALPHA = 0.05
#: 동결 모듈이 백분위를 부르는 식 그대로다(`[100 * alpha / 2, 100 * (1 - alpha / 2)]`).
Q_PERCENT: tuple[float, float] = (100 * ALPHA / 2, 100 * (1 - ALPHA / 2))

METHODS: tuple[str, ...] = (
    "linear", "lower", "higher", "nearest", "inverted_cdf", "closest_observation",
    "hazen", "weibull", "median_unbiased",
)
#: 기본 길이, 제외가 있어 길이가 달라진 경우, 가중이 정확히 0 이 되는 경계(`(n−1)·0.025` 가 정수).
RUN_SIZES: tuple[tuple[int, str], ...] = (
    (2000, "기본 — 재표집 전부가 집계에 들어간다"),
    (1937, "길이가 2000 이 아닌 경우"),
    (2001, "선형 보간의 가중이 0 이 되는 경계 — (n−1)·0.025 = 50"),
)
#: 길이를 훑는 띠. 둘째는 재표집 2,000 에서 제외가 10 % 이하인 길이다 — 제외가 그보다 많으면
#: 미정의 비율 규칙이 구간을 쓰지 않으므로, 구간이 실제로 쓰이는 길이는 이 띠 안이다.
SWEEP_BANDS: tuple[tuple[str, int, int], ...] = (("2..2500", 2, 2500), ("1800..2000", 1800, 2000))
#: 하한은 내림, 상한은 올림 — 두 방식에서 한 끝씩 가져온 구간의 이름.
OUTER = "lower_lo_higher_hi"
#: 끝점이 추첨값 그대로인 방식. 보간 식을 지나지 않는다.
ORDER_STAT_METHODS: tuple[str, ...] = ("lower", "higher", "nearest", "inverted_cdf", "closest_observation")

# --- 측정 구간의 파일 열기 감시 -----------------------------------------------------------


class _Watch:
    """측정 구간에서 이 스레드가 파일을 열면 센다. **0 이 아니면 값을 내지 않는다.**

    감사 훅은 떼어낼 수 없으므로 한 번만 달고 구간을 깃발로 연다.
    """

    installed = False
    active = False
    thread = 0
    opened: list[str] = []

    @classmethod
    def _hook(cls, event: str, args: tuple) -> None:
        if cls.active and event == "open" and threading.get_ident() == cls.thread:
            cls.opened.append(str(args[0]))

    @classmethod
    def start(cls) -> None:
        if not cls.installed:
            sys.addaudithook(cls._hook)
            cls.installed = True
        cls.opened = []
        cls.thread = threading.get_ident()
        cls.active = True

    @classmethod
    def stop(cls) -> int:
        cls.active = False
        if cls.opened:
            for p in cls.opened:
                print(f"측정 구간에서 열린 파일: {p}", file=sys.stderr)
            raise SystemExit("측정 구간에서 파일을 열었다 — 값을 내지 않는다")
        return len(cls.opened)


# --- 벡터와 끝점 ---------------------------------------------------------------------------


def _cell(x: float) -> float | str:
    """끝점 하나. 유한이면 수, 아니면 상태 이름이다.

    **표준 JSON 은 `NaN`·`Infinity` 를 담지 못한다.** 맨값으로 내면 읽는 쪽이 갈리므로
    `"nan"` · `"pos_inf"` · `"neg_inf"` 로 적는다.
    """
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "pos_inf" if x > 0 else "neg_inf"
    return x


def _state(cell: float | str) -> str:
    return cell if isinstance(cell, str) else "finite"


def _stride(n: int) -> int:
    for p in (7, 11, 13, 17, 19, 23, 29, 31):
        if n % p:
            return p
    raise ValueError(f"자리를 섞을 보폭을 못 찾았다: n={n}")


def make_vector(n: int, n_neg: int, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """(정렬본, 섞은 것). 유한값은 자기 자리 번호다."""
    if n_neg + n_pos > n:
        raise ValueError("무한의 수가 길이를 넘는다")
    x = np.arange(n, dtype=float)
    x[:n_neg] = -np.inf
    if n_pos:
        x[n - n_pos:] = np.inf
    order = (np.arange(n) * _stride(n) + 3) % n
    return x, x[order]


def endpoints(vec: np.ndarray, method: str) -> tuple[float | str, float | str]:
    with np.errstate(invalid="ignore"):
        lo, hi = np.percentile(vec, list(Q_PERCENT), method=method)
    return _cell(lo), _cell(hi)


def formula_k(n: int) -> int:
    """검토 27번 6-1절의 순위 식 — `((n − 1) × 25) // 1000`. 정수 나눗셈이다."""
    return ((n - 1) * 25) // 1000


def _counts(n: int) -> list[int]:
    k = formula_k(n)
    return [k - 1, k, k + 1, k + 2, 4 * (k + 1)]


def vector_specs(n: int) -> list[tuple[str, int, int]]:
    """(이름, `−무한` 건수, `+무한` 건수)."""
    cs = _counts(n)
    specs = [("finite_only", 0, 0)]
    specs += [(f"pos_inf_{c}", 0, c) for c in cs]
    specs += [(f"neg_inf_{c}", c, 0) for c in cs]
    specs += [(f"both_{c}", c, c) for c in cs]
    specs += [("all_pos_inf", 0, n), ("all_neg_inf", n, 0), ("half_neg_half_pos", n // 2, n - n // 2)]
    return specs


def measure_run(n: int, note: str) -> tuple[dict, int]:
    vectors, order_dependent = [], 0
    for name, n_neg, n_pos in vector_specs(n):
        ordered, shuffled = make_vector(n, n_neg, n_pos)
        by_method = {}
        for m in METHODS:
            lo, hi = endpoints(shuffled, m)
            if (lo, hi) != endpoints(ordered, m):
                order_dependent += 1
            by_method[m] = [lo, hi]
        vectors.append({
            "name": name, "n_neg_inf": n_neg, "n_finite": n - n_neg - n_pos, "n_pos_inf": n_pos,
            "lo_hi_by_method": by_method,
            OUTER: [by_method["lower"][0], by_method["higher"][1]],
        })
    k = formula_k(n)
    finite = vectors[0]
    inside = {m: _inside(*finite["lo_hi_by_method"][m]) for m in METHODS}
    inside[OUTER] = _inside(*finite[OUTER])
    return ({"n": n, "note": note, "formula_k": k, "formula_lo_rank": k,
             "formula_hi_rank": (n - 1) - k, "tail_counts": _counts(n),
             # 유한만 있는 벡터에서 구간이 품는 추첨 수와, 1 − α 가 되려면 필요한 최소 수
             "draws_inside_finite_only": inside, "draws_needed_for_1_minus_alpha": -(-19 * n // 20),
             "vectors": vectors},
            order_dependent)


def _inside(lo: float, hi: float) -> int:
    """닫힌 구간 `[lo, hi]` 안에 든 추첨 수. 유한값이 자리 번호라서 셀 수 있다."""
    return math.floor(hi) - math.ceil(lo) + 1


# --- 보간 식의 칸 (27번 3-4절 첫 표) --------------------------------------------------------

#: 두 원소 벡터 `[a, b]` 에 `q = 100·t` 를 주면 선형 보간의 아래 이웃이 a, 위 이웃이 b, 가중이 t 다.
LERP_PAIRS: tuple[tuple[str, float, float], ...] = (
    ("finite|finite", 0.0, 1.0),
    ("finite|pos_inf", 0.0, math.inf),
    ("neg_inf|finite", -math.inf, 0.0),
    ("pos_inf|pos_inf", math.inf, math.inf),
    ("neg_inf|neg_inf", -math.inf, -math.inf),
)
LERP_T: tuple[float, ...] = (0.0, 0.025, 0.25, 0.5, 0.75, 0.975)


def measure_lerp() -> list[dict]:
    out = []
    for pair, a, b in LERP_PAIRS:
        for t in LERP_T:
            with np.errstate(invalid="ignore"):
                r = np.percentile(np.array([a, b]), 100 * t, method="linear")
            out.append({"pair": pair, "t": t, "result": _cell(r)})   # 유한이면 수, 아니면 상태 이름
    return out


# --- 길이를 훑는다 — 순위 식 · 품음 · 품는 몫 · 무한이 되는 건수 --------------------------


def _sweep_one(n: int) -> dict[str, set[str]]:
    """길이 하나에서 어긋난 것의 이름. 검사 이름 → 어긋난 방식(또는 `"-"`)."""
    vec = np.arange(n, dtype=float)
    got: dict[str, tuple[float, float]] = {}
    for m in METHODS:
        with np.errstate(invalid="ignore"):
            lo, hi = (float(v) for v in np.percentile(vec, list(Q_PERCENT), method=m))
        got[m] = (lo, hi)
    k = formula_k(n)
    got[OUTER] = (got["lower"][0], got["higher"][1])
    outer = got[OUTER]
    need_tail = -(-n // 40)                          # ⌈n·α/2⌉, α = 0.05
    bad: dict[str, set[str]] = {key: set() for key in _SWEEP_CHECKS}
    if outer != (float(k), float((n - 1) - k)):
        bad["formula_ne_lower_lo_higher_hi"].add("-")
    for m, (lo, hi) in got.items():
        if lo < outer[0] or hi > outer[1]:
            bad["not_contained_in_lower_lo_higher_hi"].add(m)
        if 20 * _inside(lo, hi) < 19 * n:
            bad["holds_less_than_1_minus_alpha"].add(m)
        if m in ORDER_STAT_METHODS or m == OUTER:
            # 하한이 p 번째 자리면 −무한 p+1 건부터, 상한이 P 번째 자리면 +무한 n−P 건부터 무한이다.
            if int(lo) + 1 != need_tail:
                bad["lo_inf_threshold_ne_ceil_n_alpha_half"].add(m)
            if n - int(hi) != need_tail:
                bad["hi_inf_threshold_ne_ceil_n_alpha_half"].add(m)
    return bad


_SWEEP_CHECKS: dict[str, str] = {
    "formula_ne_lower_lo_higher_hi":
        "순위 식(k, n−1−k)이 lower 의 하한·higher 의 상한과 다른 길이",
    "not_contained_in_lower_lo_higher_hi":
        "그 방식의 구간이 '하한 내림 + 상한 올림' 구간 밖으로 나가는 길이",
    "holds_less_than_1_minus_alpha":
        "구간이 품는 추첨이 (1−α)·n 에 못 미치는 길이 (끝점 포함, 20·안 < 19·n)",
    "lo_inf_threshold_ne_ceil_n_alpha_half":
        "하한이 −무한이 되기 시작하는 꼬리 건수가 ⌈n·α/2⌉ 와 다른 길이 (끝점이 추첨값인 방식만)",
    "hi_inf_threshold_ne_ceil_n_alpha_half":
        "상한이 +무한이 되기 시작하는 꼬리 건수가 ⌈n·α/2⌉ 와 다른 길이 (끝점이 추첨값인 방식만)",
}


def measure_sweep() -> dict:
    """유한만 있는 벡터에서 길이마다 다섯 가지를 본다(`_SWEEP_CHECKS`). 띠마다 어긋난 길이의 수를 센다."""
    top = max(hi for _, _, hi in SWEEP_BANDS)
    per_n = {n: _sweep_one(n) for n in range(2, top + 1)}
    bands = {}
    for name, lo_n, hi_n in SWEEP_BANDS:
        out: dict[str, dict] = {"n_lengths": hi_n - lo_n + 1}
        for check in _SWEEP_CHECKS:
            who = ("-",) if check.startswith("formula") else (*METHODS, OUTER)
            counts = {}
            for m in who:
                hit = [n for n in range(lo_n, hi_n + 1) if m in per_n[n][check]]
                counts[m] = {"n_lengths": len(hit), "first": hit[:3]}
            if check.endswith("ceil_n_alpha_half"):
                counts = {m: v for m, v in counts.items() if m in ORDER_STAT_METHODS or m == OUTER}
            out[check] = counts["-"] if who == ("-",) else counts
        bands[name] = out
    return {"checks": _SWEEP_CHECKS, "bands": bands}


# --- 손 도출과의 대조 -----------------------------------------------------------------------

#: 27번 3-4절 첫 표. (쌍, 가중 구간) → 도출된 상태. 가중 구간의 이름은 표의 열이다.
CLAIMS_LERP: tuple[tuple[str, str, str, str], ...] = (
    # (쌍, 열, 도출, 자리)
    ("finite|finite", "t=0", "finite", "27번 133줄"),
    ("finite|finite", "0<t<0.5", "finite", "27번 133줄"),
    ("finite|finite", "t>=0.5", "finite", "27번 133줄"),
    ("finite|pos_inf", "t=0", "nan", "27번 134줄"),
    ("finite|pos_inf", "0<t<0.5", "pos_inf", "27번 134줄"),
    ("finite|pos_inf", "t>=0.5", "nan", "27번 134줄"),
    ("neg_inf|finite", "t=0", "nan", "27번 135줄"),
    ("neg_inf|finite", "0<t<0.5", "nan", "27번 135줄"),
    ("neg_inf|finite", "t>=0.5", "neg_inf", "27번 135줄"),
    ("pos_inf|pos_inf", "t=0", "nan", "27번 136줄"),
    ("pos_inf|pos_inf", "0<t<0.5", "nan", "27번 136줄"),
    ("pos_inf|pos_inf", "t>=0.5", "nan", "27번 136줄"),
    ("neg_inf|neg_inf", "t=0", "nan", "27번 137줄"),
    ("neg_inf|neg_inf", "0<t<0.5", "nan", "27번 137줄"),
    ("neg_inf|neg_inf", "t>=0.5", "nan", "27번 137줄"),
)
_T_COLUMN = {"t=0": (0.0,), "0<t<0.5": (0.025, 0.25), "t>=0.5": (0.5, 0.75, 0.975)}

#: 27번 3-4절 둘째 표와 그 아래 문장. `n = 2000`, 선형 보간. (꼬리, 건수 범위, 도출, 자리)
#: 건수 범위는 (이상, 이하) 이고 `None` 은 열린 끝이다.
CLAIMS_LINEAR_2000: tuple[tuple[str, tuple[int | None, int | None], str, str], ...] = (
    ("pos", (None, 49), "finite", "27번 146줄"),
    ("pos", (50, 50), "pos_inf", "27번 147줄"),
    ("pos", (51, None), "nan", "27번 148줄"),
    ("neg", (None, 49), "finite", "27번 150줄"),
    ("neg", (50, 50), "neg_inf", "27번 150줄"),
    ("neg", (51, None), "nan", "27번 150줄"),
)

#: 27번 4-3절 표. `n = 2000`. 자리는 0부터 센 오름차순 자리다.
#: 건수 칸은 "그 끝점이 무한이 되기 시작하는 건수" 이고, 선형 보간 줄만 다르다 —
#: "정확히 50 에서만 무한, 51 부터 NaN"(`None` 으로 적는다).
CLAIMS_RANK_2000: tuple[tuple[str, float, float, int | None, int | None, str], ...] = (
    # (방식, 하한 자리, 상한 자리, −무한 하한이 되는 건수, +무한 상한이 되는 건수, 자리)
    ("lower", 49.0, 1949.0, 50, 51, "27번 217줄"),
    ("higher", 50.0, 1950.0, 51, 50, "27번 218줄"),
    ("nearest", 50.0, 1949.0, 51, 51, "27번 219줄"),
    ("inverted_cdf", 49.0, 1949.0, 50, 51, "27번 220줄"),
    ("linear", 49.975, 1949.025, None, None, "27번 221줄"),
    ("lower_lo_higher_hi", 49.0, 1950.0, 50, 50, "27번 222줄"),
)


def _in_range(count: int, bounds: tuple[int | None, int | None]) -> bool:
    lo, hi = bounds
    return (lo is None or count >= lo) and (hi is None or count <= hi)


def _range_text(bounds: tuple[int | None, int | None]) -> str:
    lo, hi = bounds
    return f"{hi}건 이하" if lo is None else f"{lo}건 이상" if hi is None else f"정확히 {lo}건"


def _rank_expected(threshold: int | None, count: int, inf_state: str) -> str:
    if threshold is None:                            # 선형 보간 줄
        return "finite" if count < 50 else inf_state if count == 50 else "nan"
    return inf_state if count >= threshold else "finite"


def _pick(vector: dict, method: str) -> list:
    return vector[OUTER] if method == OUTER else vector["lo_hi_by_method"][method]


def compare(lerp: list[dict], run2000: dict) -> list[dict]:
    """손 도출 표의 칸마다 측정과 맞댄다. 칸 하나에 걸리는 측정이 여럿이면 **전부** 같아야 일치다."""
    rows: list[dict] = []

    def row(table, cell, where, claimed, measured: dict, expected: dict) -> None:
        ok = bool(measured) and measured == expected
        rows.append({"table": table, "cell": cell, "where": where, "claimed": claimed,
                     "measured": measured, "verdict": "일치" if ok else "불일치"})

    # 1. 보간 식의 칸
    for pair, col, want, where in CLAIMS_LERP:
        got = {f"t={c['t']}": _state(c["result"]) for c in lerp
               if c["pair"] == pair and c["t"] in _T_COLUMN[col]}
        row("3-4 첫 표", f"{pair} · {col}", where, want, got, {k: want for k in got})

    vec = {v["name"]: v for v in run2000["vectors"]}

    def tail_states(method: str, side: str) -> dict[int, str]:
        """한쪽 꼬리만 무한인 벡터에서 그쪽 끝점의 상태. 건수 → 상태."""
        prefix, end = ("pos_inf_", 1) if side == "pos" else ("neg_inf_", 0)
        out = {0: _state(_pick(vec["finite_only"], method)[end])}
        for c in run2000["tail_counts"]:
            out[c] = _state(_pick(vec[f"{prefix}{c}"], method)[end])
        return out

    # 2. 선형 보간 — 꼬리 건수별
    for side, bounds, want, where in CLAIMS_LINEAR_2000:
        inf_name = "+무한" if side == "pos" else "−무한"
        got = {f"{c}건": st for c, st in tail_states("linear", side).items() if _in_range(c, bounds)}
        row("3-4 둘째 표", f"linear · {inf_name} {_range_text(bounds)}", where, want, got,
            {k: want for k in got})

    # 3. 방식별 자리와 무한이 되는 건수
    finite = vec["finite_only"]
    for method, lo_rank, hi_rank, neg_from, pos_from, where in CLAIMS_RANK_2000:
        picked = _pick(finite, method)
        for end, name, want in ((0, "하한", lo_rank), (1, "상한", hi_rank)):
            got = picked[end]
            ok = not isinstance(got, str) and abs(got - want) < 1e-9
            rows.append({"table": "4-3 표", "cell": f"{method} · {name} 자리", "where": where,
                         "claimed": want, "measured": got, "verdict": "일치" if ok else "불일치"})
        for side, threshold in (("neg", neg_from), ("pos", pos_from)):
            inf_state = "pos_inf" if side == "pos" else "neg_inf"
            inf_name = "+무한 상한" if side == "pos" else "−무한 하한"
            states = tail_states(method, side)
            claimed = ("정확히 50건에서만 무한, 51건부터 nan" if threshold is None
                       else f"{threshold}건 이상")
            row("4-3 표", f"{method} · {inf_name}이 되는 건수", where, claimed,
                {f"{c}건": st for c, st in states.items()},
                {f"{c}건": _rank_expected(threshold, c, inf_state) for c in states})
    return rows


# --- 조립 -----------------------------------------------------------------------------------


def _json_carrier() -> dict:
    """표준 JSON 이 비유한 수를 담는가. 구간 끝점의 표현을 정할 때의 근거다."""
    out = {}
    for name, v in (("nan", math.nan), ("pos_inf", math.inf), ("neg_inf", -math.inf)):
        try:
            json.dumps(v, allow_nan=False)
            strict = "담긴다"
        except ValueError:
            strict = "ValueError"
        out[name] = {"allow_nan=False": strict, "기본값(표준 밖)": json.dumps(v)}
    return out


_FLAT_ARRAY = re.compile(r"\[\s+([^\[\]{}]*?)\s+\]")


def dumps(result: dict) -> str:
    """표준 JSON 으로 적는다(`allow_nan=False`). 스칼라뿐인 배열은 한 줄로 접는다."""
    text = json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False)
    return _FLAT_ARRAY.sub(lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", text) + "\n"


def build() -> dict:
    """측정 전부. 인자가 없다 — 읽을 경로를 받는 자리가 없다."""
    for m in METHODS:                                # 첫 호출에 딸린 적재를 감시 구간 밖으로 뺀다
        np.percentile(np.array([0.0, 1.0]), 50.0, method=m)
    _Watch.start()
    lerp = measure_lerp()
    runs, order_dependent = [], 0
    for n, note in RUN_SIZES:
        run, dep = measure_run(n, note)
        runs.append(run)
        order_dependent += dep
    sweep = measure_sweep()
    opens = _Watch.stop()

    rows = compare(lerp, runs[0])
    n_match = sum(r["verdict"] == "일치" for r in rows)
    return {
        "schema": "percentile_inf_vectors.v1",
        "what": "합성 벡터로 잰 백분위 방식의 ±무한 거동. 실험 결과가 아니다",
        "numpy_version": np.__version__,
        "alpha": ALPHA,
        "q_percent": list(Q_PERCENT),
        "methods": list(METHODS),
        "finite_value_is_rank": True,
        "io_guard": {"file_opens_during_measurement": opens},
        "order_dependent_results": order_dependent,
        "json_carrier": _json_carrier(),
        "lerp_cells": lerp,
        "runs": runs,
        "sweep_finite_only": sweep,
        "comparison": {"n_cells": len(rows), "n_match": n_match, "n_mismatch": len(rows) - n_match,
                       "rows": rows},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path, help="새로 쓸 JSON 경로. 있으면 거부한다")
    args = ap.parse_args(argv)
    result = build()
    text = dumps(result)
    with args.out.open("x", encoding="utf-8", newline="\n") as fh:     # 배타 생성 — 덮지 않는다
        fh.write(text)
    c = result["comparison"]
    print(f"칸 {c['n_cells']} · 일치 {c['n_match']} · 불일치 {c['n_mismatch']} → {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
