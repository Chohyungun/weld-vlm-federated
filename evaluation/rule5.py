"""규칙 ⑤ 의 **등록 판정** — 07번 미니스펙 §19-1 · §22-2 ~ §22-4 · §23-1 ~ §23-2 · §24-2 ~ §24-6 · §25-1, 실험설계 8-5.

## 정의

조건 모델(중앙 · 연합) **각각**에 대해 두 조건을 본다.

- **⑤-①** 등록한 **모든 시드**에서 M1 점추정이 등록 대조선의 M1 보다 **엄격히** 크다 — 같으면 실패다.
- **⑤-②** `시드 평균 M1 − 대조선 M1` 의 대응 차 95 % 구간의 하한이 0 보다 크다 — 같은 추첨 · 같은 가중.

두 조건이 모두 참인 모델이 통과하고, **등록 집합의 모든 모델**이 통과하면 `rule5_passed` 다. 로컬 평균은 조건에 넣지 않는다.
판정은 대표 채택이 아니다 — `final_adoption` 은 늘 `None` 이다(채택은 회복률 규칙 ①~④ 와 함께 정한다).

## 이 모듈이 막는 것

| 계약 | 막는 길 |
|---|---|
| §22-2 | 시드 미지정 · 빈 사상 · 중복 값 · 번호와 값의 혼동은 `ValueError`. 번호 집합이 등록과 다르면 ⑤-① 은 `None` 이고 이름을 남긴다 |
| §22-3 · §23-2 | `n_resamples` · `min_defined_draws` 가 없거나 꼴이 틀리면 `ValueError`(참 · 거짓은 수가 아니다). 유효 + 미정의 ≠ `n_resamples` 는 회계 오류로 `ValueError`. 유효 0 또는 하한 미달이면 ⑤-② 는 `None` |
| §22-4 · §24-5 | 근거 사전의 키와 근거가 스스로 말하는 `tag` 가 다르면 `ValueError`. 직렬화할 때 `per_model` 키 = 검사한 집합, 등록 ⊆ 검사, 진단 = 검사 − 등록을 다시 본다 |
| §23-1 | 대응 차의 미정의는 **이 통계량의 것만** 센다 — 회복률의 분모 실패를 받을 자리가 없다 |
| §23-2 | 공식 판정은 등록 집합만이다. 등록 밖 모델(진단)의 실패는 `rule5_passed` 를 바꾸지 않는다 |
| §25-1 | 판정이 근거(등록 시드 전부의 M1 · 대조선 M1 · 구간 끝점 · 유효 · 미정의 추첨 수)를 함께 싣는다. `rejudge` 가 직렬화한 결과를 근거에서 다시 판정해 맞댄다 |

**등록 집합.** 조건 모델은 미니스펙이 정한 두 모델(중앙 · 연합)이다 — 등록 블록에 이 집합의 칸이 없다. 상위 집합을 공식으로 쓰려면
결과를 보기 전에 계약을 고쳐야 한다(§23-2). 대조선 M1 은 같은 목록 · 같은 가중으로 채점한 값이고 전량의 등록 상수가 아니다(25번 §4-4).

## 이 검사가 증명하지 않는 것(§25-1)

근거의 수치 자체를 악의적으로 다시 꾸민 경우의 진실성은 증명하지 않는다. 막는 것은 **근거 없이 불리언만 참인 결과**와
**근거와 어긋나는 결과**다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from evaluation import percentile_rule as PR

VERDICT_AUTHORITY = "registered"
CONDITION_MODELS: tuple[str, ...] = ("uni_central", "uni_fed")
"""등록 집합 — 07번 §19-1 의 '중앙 · 연합 각각'. 태그는 `schema_v14.UNI_MAIN_TAGS` 의 이름이다."""
CONDITIONS = ("rule5_1", "rule5_2")
DIRECTIONS = ("above", "below", "indeterminate")
"""대응 차 구간의 방향 — 하한 > 0 · 상한 < 0 · 0 을 품는다(구분 불가). 구간이 없으면 방향도 없다(`None`)."""
FINAL_ADOPTION_REASON = "대표 채택은 회복률 규칙 ①~④ 와 함께 정한다 — 규칙 ⑤ 하나로 채택하지 않는다(07번 §19-1)"
NOT_PROVEN = "근거 수치 자체의 진실성은 증명하지 않는다 — 막는 것은 근거 없는 참과 근거와 어긋난 결과다(07번 §25-1)"


@dataclass(frozen=True)
class RegisteredRule5:
    """판정이 등록에서 읽는 것 — 한 번 정규화한 뒤의 값만 담는다(§23-2 의 정규화 시점). 직접 만들어도 같은 규칙으로 본다 —
    빈 시드 사상이면 ⑤-① 의 `all([])` 이 참이 되어 검사 없이 통과한다."""

    seeds: Mapping[int, int]
    n_resamples: int
    min_defined_draws: int
    statistic: str
    condition_models: tuple[str, ...] = CONDITION_MODELS

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, Mapping) or not self.seeds:
            raise ValueError("등록 시드가 빈 사상이다 — ⑤-① 이 검사할 것이 없다")
        if any(not _plain_int(k) or k < 1 or not _plain_int(v) or v < 0 for k, v in self.seeds.items()):
            raise ValueError("등록 시드는 1 이상의 정수 번호 → 0 이상의 정수 값이다")
        if len(set(self.seeds.values())) != len(self.seeds):
            raise ValueError("시드 값이 겹친다(§22-2)")
        if not _plain_int(self.n_resamples) or self.n_resamples < 1:
            raise ValueError(f"n_resamples 가 양의 정수가 아니다: {self.n_resamples!r}")
        if not _plain_int(self.min_defined_draws) or not 1 <= self.min_defined_draws <= self.n_resamples:
            raise ValueError(f"min_defined_draws 는 1 이상 n_resamples 이하의 정수다: {self.min_defined_draws!r}")
        if not isinstance(self.statistic, str) or not self.statistic.strip():
            raise ValueError("통계량 이름이 비었다")
        if not self.condition_models or any(not isinstance(m, str) or not m for m in self.condition_models) \
                or len(set(self.condition_models)) != len(self.condition_models):
            raise ValueError("조건 모델 집합이 비었거나 겹친다")


def _plain_int(v) -> bool:
    return type(v) is int


def registered_inputs(reg) -> RegisteredRule5:
    """등록 객체(`UnifiedRegistration`)에서 판정의 입력을 읽는다. 없거나 꼴이 틀리면 `ValueError` — 미정 등록을 채운 것처럼
    통과시키지 않는다. 시드 번호는 **여기서 한 번만** 정수로 정규화한다."""
    raw = reg.seeds
    if raw is None:
        raise ValueError("등록에 시드가 없다(seeds 미지정) — 규칙 ⑤ 를 판정하지 않는다")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("등록의 시드가 빈 사상이다 — ⑤-① 이 검사할 것이 없다")
    seeds: dict[int, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not k.isdigit() or k != str(int(k)) or int(k) < 1:
            raise ValueError(f"시드 번호 {k!r} 가 1 이상의 정수를 적은 문자열이 아니다")
        if not _plain_int(v) or v < 0:
            raise ValueError(f"시드 값 {v!r} 가 0 이상의 정수가 아니다(참 · 거짓은 받지 않는다)")
        seeds[int(k)] = v
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("시드 값이 겹친다 — 번호가 달라도 같은 학습이다(§22-2)")
    planned = getattr(reg, "planned_seed_count", None)
    if planned is not None and (not _plain_int(planned) or planned != len(seeds)):
        raise ValueError(f"planned_seed_count {planned!r} 가 등록 시드 수 {len(seeds)} 와 다르다(§23-2)")
    ci = reg.ci
    n = ci.n_resamples
    if not _plain_int(n) or n < 1:
        raise ValueError(f"ci.n_resamples 가 양의 정수가 아니다: {n!r}")
    m = ci.min_defined_draws
    if not _plain_int(m) or not 1 <= m <= n:
        raise ValueError(f"ci.min_defined_draws 는 1 이상 n_resamples 이하의 정수다(등록이 정한다, §22-3): {m!r}")
    stat = ci.rule5_statistic
    if not isinstance(stat, str) or not stat.strip():
        raise ValueError("ci.rule5_statistic 이 비었다 — 등록이 통계량의 이름을 적는다")
    return RegisteredRule5(seeds=dict(sorted(seeds.items())), n_resamples=n, min_defined_draws=m, statistic=stat)


@dataclass(frozen=True)
class DiffCI:
    """대응 차 `시드 평균 M1 − 대조선 M1` 의 구간과 그 추첨 회계 — **이 통계량의** 유효 · 미정의만 센다(§23-1)."""

    lo: PR.Bound
    hi: PR.Bound
    n_defined: int
    n_undefined: int
    undefined_reasons: Mapping[str, int] = field(default_factory=dict)
    convention_id: str = PR.CONVENTION_ID

    @classmethod
    def from_draws(cls, draws, undefined_reasons: Mapping[str, int], *, alpha: float) -> DiffCI:
        """추첨값(미정의는 `NaN`)에서 — `percentile_rule.defined_interval` 이 `NaN` 을 빼고 세며 ±무한을 거부한다."""
        iv, n_def, n_undef = PR.defined_interval(draws, alpha)
        if sum(undefined_reasons.values()) != n_undef:
            raise ValueError(f"미정의 사유의 합 {sum(undefined_reasons.values())} 이 미정의 추첨 수 {n_undef} 와 다르다")
        return cls(iv.lo, iv.hi, n_def, n_undef, dict(undefined_reasons), iv.convention_id)

    def as_dict(self) -> dict:
        return {"ci_lo": self.lo.value, "ci_lo_state": self.lo.state, "ci_hi": self.hi.value, "ci_hi_state": self.hi.state,
                "n_defined": self.n_defined, "n_undefined": self.n_undefined,
                "undefined_reasons": dict(self.undefined_reasons), "convention_id": self.convention_id}

    @classmethod
    def from_dict(cls, d: Mapping) -> DiffCI:
        return cls(PR.Bound(d["ci_lo"], d["ci_lo_state"]), PR.Bound(d["ci_hi"], d["ci_hi_state"]),
                   d["n_defined"], d["n_undefined"], dict(d["undefined_reasons"]), d["convention_id"])


@dataclass(frozen=True)
class ModelEvidence:
    """모델 하나의 근거 — 시드 번호 → M1 점추정(미정의면 `None`)과 대응 차 구간."""

    tag: str
    per_seed_m1: Mapping[int, float | None]
    diff: DiffCI


@dataclass(frozen=True)
class ModelVerdict:
    tag: str
    rule5_1: bool | None
    rule5_2: bool | None
    direction: str | None
    failed: tuple[str, ...]
    reasons: tuple[str, ...]
    evidence: ModelEvidence

    @property
    def passed(self) -> bool:
        return self.rule5_1 is True and self.rule5_2 is True

    def as_dict(self, baseline_m1: float | None) -> dict:
        return {"tag": self.tag, "rule5_1": self.rule5_1, "rule5_2": self.rule5_2, "passed": self.passed,
                "direction": self.direction, "failed_conditions": list(self.failed), "reasons": list(self.reasons),
                "evidence": {"per_seed_m1": {str(k): v for k, v in sorted(self.evidence.per_seed_m1.items())},
                             "baseline_m1": baseline_m1, "diff": self.evidence.diff.as_dict()}}


@dataclass(frozen=True)
class Rule5Verdict:
    registered: RegisteredRule5
    baseline_m1: float | None
    per_model: Mapping[str, ModelVerdict]

    @property
    def checked(self) -> tuple[str, ...]:
        return tuple(sorted(self.per_model))

    @property
    def diagnostic_models(self) -> tuple[str, ...]:
        """검사 − 등록으로 **유도한다**(§24-5) — 따로 채우지 않는다."""
        return tuple(m for m in self.checked if m not in self.registered.condition_models)

    @property
    def rule5_passed(self) -> bool:
        """등록 집합만으로 낸다(§23-2)."""
        return all(self.per_model[m].passed for m in self.registered.condition_models)

    def as_dict(self) -> dict:
        reg = self.registered
        checked = self.checked
        if set(self.per_model) != set(checked) or not set(reg.condition_models) <= set(checked) \
                or set(self.diagnostic_models) != set(checked) - set(reg.condition_models):
            raise ValueError("판정이 스스로와 어긋난다 — per_model 키 · 검사 집합 · 등록 집합 · 진단 집합(§24-5)")
        return {"verdict_authority": VERDICT_AUTHORITY, "rule5_passed": self.rule5_passed,
                "final_adoption": None, "final_adoption_reason": FINAL_ADOPTION_REASON,
                "condition_models_registered": list(reg.condition_models),
                "condition_models_checked": list(checked), "diagnostic_models": list(self.diagnostic_models),
                "seeds": {str(k): v for k, v in reg.seeds.items()}, "n_resamples": reg.n_resamples,
                "min_defined_draws": reg.min_defined_draws, "statistic": reg.statistic,
                "baseline_m1": self.baseline_m1,
                "per_model": {m: self.per_model[m].as_dict(self.baseline_m1) for m in checked},
                "not_proven": NOT_PROVEN}


def _unit(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0.0 <= float(x) <= 1.0


def _judge_model(reg: RegisteredRule5, ev: ModelEvidence, baseline_m1: float | None) -> ModelVerdict:
    reasons: list[str] = []
    want = set(reg.seeds)
    got = set(ev.per_seed_m1)
    # ⑤-① — 등록 시드 전부에서 점추정이 대조선보다 엄격히 크다
    if got != want:
        missing = sorted(want - got)
        extra = sorted(got - want)
        reasons.append(f"rule5_1: 시드 번호 집합이 등록과 다르다 — 빠짐 {missing} · 등록에 없음 {extra}")
        c1 = None
    elif not _unit(baseline_m1) or not all(_unit(v) for v in ev.per_seed_m1.values()):
        reasons.append("rule5_1: M1 또는 대조선 M1 이 미정의이거나 [0, 1] 밖이다")
        c1 = None
    else:
        c1 = all(float(ev.per_seed_m1[s]) > float(baseline_m1) for s in sorted(want))
    # ⑤-② — 대응 차 구간의 하한 > 0, 유효 추첨이 등록 하한 이상
    d = ev.diff
    if not (_plain_int(d.n_defined) and _plain_int(d.n_undefined)) or d.n_defined < 0 or d.n_undefined < 0:
        raise ValueError(f"{ev.tag}: 추첨 수가 0 이상의 정수가 아니다")
    if d.n_defined + d.n_undefined != reg.n_resamples:
        raise ValueError(f"{ev.tag}: 유효 {d.n_defined} + 미정의 {d.n_undefined} ≠ 등록 재표집 수 {reg.n_resamples} — "
                         "구간이 등록한 추첨 수로 산출되지 않았다(§22-3)")
    direction = None
    if d.n_defined == 0:
        reasons.append("rule5_2: 유효 추첨이 0 이다 — 방향을 말할 수 없다(퇴화 방지, §22-3)")
        c2 = None
    elif d.n_defined < reg.min_defined_draws:
        reasons.append(f"rule5_2: 유효 추첨 {d.n_defined} 가 등록 하한 {reg.min_defined_draws} 보다 적다")
        c2 = None
    elif d.lo.state != "finite" or d.hi.state != "finite":
        reasons.append(f"rule5_2: 구간 끝점이 유한하지 않다({d.lo.state} · {d.hi.state})")
        c2 = None
    else:
        if d.lo.value > d.hi.value:
            raise ValueError(f"{ev.tag}: 구간 하한이 상한보다 크다")
        direction = "above" if d.lo.value > 0 else "below" if d.hi.value < 0 else "indeterminate"
        c2 = d.lo.value > 0
    failed = tuple(name for name, v in zip(CONDITIONS, (c1, c2), strict=True) if v is not True)
    return ModelVerdict(ev.tag, c1, c2, direction, failed, tuple(reasons), ev)


def judge_rule5_registered(reg: RegisteredRule5, *, baseline_m1: float | None,
                           evidence: Mapping[str, ModelEvidence]) -> Rule5Verdict:
    """등록 판정. `evidence` 는 검사할 자리 → 근거다 — 등록 집합 전부가 있어야 하고, 더 있는 것은 진단으로 검사한다."""
    if not isinstance(reg, RegisteredRule5):
        raise TypeError("등록 판정은 registered_inputs 가 정규화한 등록만 받는다")
    for key, ev in evidence.items():
        if ev.tag != key:
            raise ValueError(f"근거의 자리 {key!r} 와 근거의 신원 {ev.tag!r} 가 다르다(§22-4)")
    missing = [m for m in reg.condition_models if m not in evidence]
    if missing:
        raise ValueError(f"등록 조건 모델의 근거가 없다: {missing}")
    values = set(reg.seeds.values())
    numbers = set(reg.seeds)
    seed_sets = set()
    for ev in evidence.values():
        keys = set(ev.per_seed_m1)
        if any(not _plain_int(k) for k in keys):
            raise ValueError(f"{ev.tag}: 시드 번호는 정규화한 정수다 — 받은 {sorted(map(repr, keys))}")
        confused = sorted(k for k in keys if k not in numbers and k in values)
        if confused:
            raise ValueError(f"{ev.tag}: 시드 번호 자리에 시드 값이 들어왔다 {confused}(§22-2)")
        seed_sets.add(frozenset(keys))
    if len(seed_sets) > 1:
        raise ValueError("모델끼리 시드 집합이 다르다 — 대응 차가 짝이 아니다(§22-2)")
    per_model = {key: _judge_model(reg, ev, baseline_m1) for key, ev in sorted(evidence.items())}
    return Rule5Verdict(reg, baseline_m1, per_model)


def rejudge(stored: Mapping, reg: RegisteredRule5, *, scoring_sha256: str | None,
            stored_scoring_sha256: str | None) -> None:
    """직렬화한 판정을 **근거에서 다시 판정**해 맞댄다(§24-2 의 1~5 · §25-1). 어긋나면 `ValueError`.

    채점 지문 둘은 **반드시 준다** — 나중의 소비 쪽은 지금 등록의 `scoring_sha256` 과 산출이 실은 값을, 같은 실행 안(쓰기 직전)은
    둘 다 `None` 을 명시한다(대조할 다른 실행이 없다). 등록 칸(시드 · 재표집 수 · 유효 하한 · 통계량 · 조건 모델) · 판정 권한 · 최종 채택 보류 ·
    규약 id 를 맞대고, 등록 조건 모델의 근거는 반드시 있어야 한다 — 저장된 불리언으로 근거를 대신하지 않는다. 진단 모델의 근거가 빠지면
    그 모델만 대조에서 빼고 공식 판정은 막지 않는다(§25-1 금지 5).
    """
    if stored.get("verdict_authority") != VERDICT_AUTHORITY:
        raise ValueError("등록 판정이 아니다(verdict_authority)")
    if scoring_sha256 != stored_scoring_sha256:
        raise ValueError("판정의 채점 지문이 지금 등록과 다르다")
    if stored.get("final_adoption") is not None:
        raise ValueError("규칙 ⑤ 판정이 최종 채택을 적었다 — 채택은 이 판정이 정하지 않는다")
    want = {"condition_models_registered": list(reg.condition_models), "seeds": {str(k): v for k, v in reg.seeds.items()},
            "n_resamples": reg.n_resamples, "min_defined_draws": reg.min_defined_draws, "statistic": reg.statistic}
    for key, value in want.items():
        if stored.get(key) != value:
            raise ValueError(f"판정의 {key} 가 등록과 다르다")
    per = stored.get("per_model")
    if not isinstance(per, Mapping) or not set(reg.condition_models) <= set(per):
        raise ValueError("per_model 이 등록 조건 모델을 덮지 않는다")
    if sorted(per) != list(stored.get("condition_models_checked") or ()):
        raise ValueError("per_model 의 키가 검사 집합과 다르다")
    evidence, skipped = {}, []
    for key, m in per.items():
        try:
            ev = m["evidence"]
            diff = DiffCI.from_dict(ev["diff"])
            if diff.convention_id != PR.CONVENTION_ID:
                raise ValueError("규약 id 가 다르다")
            evidence[key] = ModelEvidence(m["tag"], {int(k): v for k, v in ev["per_seed_m1"].items()}, diff)
        except (KeyError, TypeError, ValueError) as exc:
            if key in reg.condition_models:
                raise ValueError(f"등록 조건 모델 {key} 의 근거가 빠졌거나 꼴이 틀렸다: {type(exc).__name__}") from None
            skipped.append(key)
    try:
        baseline = stored["baseline_m1"]
    except KeyError:
        raise ValueError("판정의 대조선 M1 이 빠졌다") from None
    again = judge_rule5_registered(reg, baseline_m1=baseline, evidence=evidence).as_dict()
    if again["rule5_passed"] != stored.get("rule5_passed"):
        raise ValueError("근거에서 다시 판정한 rule5_passed 가 저장된 값과 다르다(§25-1)")
    for key in evidence:
        if again["per_model"][key] != per[key]:
            raise ValueError(f"근거에서 다시 판정한 per_model[{key}] 가 저장된 값과 다르다(§25-1)")
