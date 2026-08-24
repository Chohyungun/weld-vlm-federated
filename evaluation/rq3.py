"""RQ3 참여 이득 지표 8종. `51_RQ3_참여이득_지표설계.md` §1.

RQ3 는 "데이터가 적거나 분포가 다른 회사도 참여할 가치가 있는가"에 답한다.
**전체 평균 하나로는 답할 수 없다.** 평균은 소규모 참여자가 손해를 보고 있어도 가려 준다.

## 평가 방식

클라이언트별 시험셋을 만들지 않는다. 다섯 칸이 같은 기준으로 채점되는 것이 공정성 주장의
근거이고(불변조건 1-3·3-7), 클라이언트마다 다른 시험셋을 두면 그 근거가 무너진다.
대신 **글로벌 평가셋의 채점 결과를 클라이언트 귀속 축으로 분해**한다. 재질과 분할 규칙이
귀속을 정하므로 분해가 가능하다.

이 방식으로 8종 중 7종이 나온다. 개인화 계층 비교는 나오지 않으며, 하지 않는다.

## 단위 표기

**`%p`(절대)와 `%`(상대)를 병기한다.** 하나만 쓰면 해석이 갈린다. 0.60 에서 0.63 으로
오른 것은 +3.0%p 이자 +5.0% 이고, 두 숫자가 주는 인상이 다르다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from statistics import pstdev

PCT_POINT = "%p"
PCT_RELATIVE = "%"


@dataclass(frozen=True)
class Delta:
    """절대·상대를 함께 들고 다니는 값. 한쪽만 쓰는 실수를 타입으로 막는다."""

    absolute: float
    """지표 원단위 차이. 보고 시 ×100 하여 `%p`."""
    baseline: float

    @property
    def points(self) -> float:
        return self.absolute * 100

    @property
    def relative(self) -> float | None:
        """기준선 대비 비율(%). 기준선이 0이면 정의되지 않는다."""
        return None if self.baseline == 0 else self.absolute / self.baseline * 100

    def as_dict(self) -> dict:
        return {
            "delta_abs": self.absolute,
            "delta_pp": self.points,
            "delta_pct": self.relative,
            "baseline": self.baseline,
        }

    def __str__(self) -> str:
        rel = "정의 불가" if self.relative is None else f"{self.relative:+.1f}%"
        return f"{self.points:+.2f}%p ({rel})"


@dataclass(frozen=True)
class ClientGain:
    """클라이언트 하나의 참여 이득. `Δᵢ = M_연합(i) − M_단독(i)`."""

    client_id: str
    federated: float
    solo: float
    n_train_samples: int = 0

    @property
    def delta(self) -> Delta:
        return Delta(self.federated - self.solo, self.solo)

    @property
    def positive(self) -> bool:
        return self.federated > self.solo

    def as_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "federated": self.federated,
            "solo": self.solo,
            "n_train_samples": self.n_train_samples,
            **self.delta.as_dict(),
        }


@dataclass(frozen=True)
class DisparityChange:
    """성능 격차 감소. 연합이 회사 간 격차를 줄이는가."""

    solo_sd: float
    federated_sd: float

    @property
    def reduction(self) -> Delta:
        """sd 감소를 절대·상대로. 부호는 '줄었으면 양수'가 되게 뒤집는다."""
        return Delta(self.solo_sd - self.federated_sd, self.solo_sd)

    def as_dict(self) -> dict:
        return {
            "solo_sd": self.solo_sd,
            "federated_sd": self.federated_sd,
            "sd_reduction": self.reduction.as_dict(),
        }


@dataclass(frozen=True)
class RoundPoint:
    """라운드 하나의 궤적 점."""

    round: int
    client_id: str
    metric: float
    delta: Delta
    cumulative_bytes: int = 0


@dataclass(frozen=True)
class Rq3Report:
    """지표 8종 산출 결과. 단일 진입점의 반환값."""

    metric_name: str
    gains: tuple[ClientGain, ...]
    disparity: DisparityChange
    trajectory: tuple[RoundPoint, ...]
    small_client: str
    caveats: tuple[str, ...] = field(default=())

    # --- ① 클라이언트별 참여 이득 -------------------------------------------
    @property
    def per_client(self) -> dict[str, Delta]:
        return {g.client_id: g.delta for g in self.gains}

    # --- ② 평균 이득 --------------------------------------------------------
    @property
    def mean_gain(self) -> Delta:
        if not self.gains:
            return Delta(0.0, 0.0)
        abs_mean = sum(g.delta.absolute for g in self.gains) / len(self.gains)
        base_mean = sum(g.solo for g in self.gains) / len(self.gains)
        return Delta(abs_mean, base_mean)

    # --- ③ 최소 이득 --------------------------------------------------------
    @property
    def min_gain(self) -> tuple[str, Delta] | None:
        """가장 손해 보는 참여자와 그 크기. **평균이 가려 주는 값이라 별도로 뗀다.**"""
        if not self.gains:
            return None
        worst = min(self.gains, key=lambda g: g.delta.absolute)
        return worst.client_id, worst.delta

    # --- ④ 소규모 클라이언트 이득 -------------------------------------------
    @property
    def small_client_gain(self) -> Delta | None:
        """RQ3 의 직접 답."""
        for g in self.gains:
            if g.client_id == self.small_client:
                return g.delta
        return None

    # --- ⑤ 이득 양수 비율 ----------------------------------------------------
    @property
    def positive_ratio(self) -> float:
        if not self.gains:
            return 0.0
        return sum(1 for g in self.gains if g.positive) / len(self.gains)

    @property
    def losers(self) -> tuple[str, ...]:
        """손해 본 참여자. **있으면 그대로 보고한다.** 숨기면 RQ3 이 무의미해진다."""
        return tuple(g.client_id for g in self.gains if not g.positive)

    # --- ⑦ 라운드별 궤적 ----------------------------------------------------
    def first_positive_round(self, client_id: str) -> int | None:
        """몇 라운드부터 이득이 나는가."""
        for p in sorted(self.trajectory, key=lambda x: x.round):
            if p.client_id == client_id and p.delta.absolute > 0:
                return p.round
        return None

    # --- ⑧ 통신량 대비 이득 -------------------------------------------------
    def gain_per_mb(self) -> dict[str, float | None]:
        """참여 비용 대비 효용. 누적 통신 바이트가 0이면 정의되지 않는다."""
        last: dict[str, int] = {}
        for p in self.trajectory:
            last[p.client_id] = max(last.get(p.client_id, 0), p.cumulative_bytes)
        out: dict[str, float | None] = {}
        for g in self.gains:
            b = last.get(g.client_id, 0)
            out[g.client_id] = None if b == 0 else g.delta.points / (b / 1_048_576)
        return out

    def as_dict(self) -> dict:
        mn = self.min_gain
        return {
            "metric_name": self.metric_name,
            "per_client_gain": {k: v.as_dict() for k, v in self.per_client.items()},
            "mean_gain": self.mean_gain.as_dict(),
            "min_gain": (
                None if mn is None else {"client_id": mn[0], **mn[1].as_dict()}
            ),
            "small_client": self.small_client,
            "small_client_gain": (
                None if self.small_client_gain is None
                else self.small_client_gain.as_dict()
            ),
            "positive_ratio": self.positive_ratio,
            "losers": list(self.losers),
            "disparity": self.disparity.as_dict(),
            "first_positive_round": {
                g.client_id: self.first_positive_round(g.client_id) for g in self.gains
            },
            "gain_per_mb": self.gain_per_mb(),
            "caveats": list(self.caveats),
        }


# --- 클라이언트 귀속 분해 -----------------------------------------------------

def attribute_by_client(
    per_image: Mapping[str, float],
    image_to_client: Mapping[str, str],
) -> dict[str, list[float]]:
    """글로벌 평가셋 채점 결과를 클라이언트 귀속 축으로 분해한다(§3).

    **클라이언트별 시험셋을 만드는 것이 아니다.** 하나의 평가셋을 하나의 채점기로 채점한
    뒤, 그 결과를 귀속 축으로 나눠 보는 것이다. 채점 기준은 다섯 칸에서 동일하게 유지된다.
    """
    out: dict[str, list[float]] = {}
    for image_id, value in per_image.items():
        client = image_to_client.get(image_id)
        if client is not None:
            out.setdefault(client, []).append(value)
    return out


# --- 단일 진입점 ---------------------------------------------------------------

def build_rq3_report(
    federated: Mapping[str, float],
    solo: Mapping[str, float],
    *,
    metric_name: str = "macro_f1",
    small_client: str = "C3",
    n_train_samples: Mapping[str, int] | None = None,
    atomic_rows: Iterable[Mapping[str, object]] | None = None,
    federated_cell: str = "sep_fed",
    solo_cell: str = "sep_local",
) -> Rq3Report:
    """지표 8종을 한 번에 낸다. 후처리 리포트의 단일 진입점.

    Args:
        federated: 클라이언트 → 연합 학습본의 귀속 분해 지표.
        solo: 클라이언트 → 단독 학습본(`분리·로컬` 칸)의 같은 지표.
            **기준선은 전체 평균이 아니라 그 클라이언트의 단독 성능이다**(§2).
        atomic_rows: C가 남긴 원자 로그 행들. 라운드별 궤적과 통신량이 여기서 나온다.

    학습 중에 지표를 만들지 않는다. 이 함수는 학습이 끝난 뒤 저장된 체크포인트를 단일
    채점기로 일괄 채점한 결과를 받아 후처리할 뿐이다(§4).
    """
    n = n_train_samples or {}
    clients = sorted(set(federated) & set(solo))
    gains = tuple(
        ClientGain(c, federated[c], solo[c], int(n.get(c, 0))) for c in clients
    )
    disparity = DisparityChange(
        solo_sd=pstdev([solo[c] for c in clients]) if len(clients) > 1 else 0.0,
        federated_sd=pstdev([federated[c] for c in clients]) if len(clients) > 1 else 0.0,
    )
    trajectory = (
        _trajectory_from_atomic(atomic_rows, solo, metric_name, federated_cell)
        if atomic_rows is not None
        else ()
    )
    caveats = [
        ("개인화 계층 없이 순수 글로벌 모델만 평가했다. "
         "'모든 참여자가 이득'이라는 서술은 이 조건과 함께 쓴다."),
        "클라이언트별 시험셋을 만들지 않고 글로벌 평가셋을 귀속 축으로 분해했다.",
    ]
    missing = sorted((set(federated) | set(solo)) - set(clients))
    if missing:
        caveats.append(f"한쪽 칸에만 있는 클라이언트를 제외했다: {missing}")
    return Rq3Report(
        metric_name=metric_name,
        gains=gains,
        disparity=disparity,
        trajectory=trajectory,
        small_client=small_client,
        caveats=tuple(caveats),
    )


def _trajectory_from_atomic(
    rows: Iterable[Mapping[str, object]],
    solo: Mapping[str, float],
    metric_name: str,
    federated_cell: str,
) -> tuple[RoundPoint, ...]:
    """원자 로그에서 라운드별 궤적을 만든다.

    통신량은 라운드마다 누적한다. 원자 로그는 라운드별 증분을 남기므로(`bytes_up`·
    `bytes_down`) 누적은 후처리 몫이다.
    """
    picked: list[tuple[int, str, float, int, int]] = []
    for r in rows:
        if str(r.get("cell", federated_cell)) != federated_cell:
            continue
        if str(r.get("metric_name")) != metric_name:
            continue
        try:
            picked.append((
                int(r["round"]), str(r["client_id"]), float(r["metric_value"]),
                int(r.get("bytes_up", 0) or 0), int(r.get("bytes_down", 0) or 0),
            ))
        except (KeyError, TypeError, ValueError):
            continue

    cumulative: dict[str, int] = {}
    out: list[RoundPoint] = []
    for rd, client, value, up, down in sorted(picked):
        cumulative[client] = cumulative.get(client, 0) + up + down
        base = solo.get(client, 0.0)
        out.append(
            RoundPoint(rd, client, value, Delta(value - base, base), cumulative[client])
        )
    return tuple(out)


# --- 보고 ---------------------------------------------------------------------

def format_report(report: Rq3Report) -> str:
    """사람이 읽는 표. **전체 평균 하나만 싣지 않는다**(§6-1)."""
    lines = [
        f"RQ3 참여 이득 (기준 지표: {report.metric_name})",
        "",
        "| 클라이언트 | 단독 | 연합 | 이득(%p) | 이득(%) | 학습 표본 |",
        "|---|---|---|---|---|---|",
    ]
    for g in report.gains:
        d = g.delta
        rel = "정의 불가" if d.relative is None else f"{d.relative:+.1f}%"
        mark = "" if g.positive else "  ← 손해"
        lines.append(
            f"| {g.client_id} | {g.solo:.4f} | {g.federated:.4f} | "
            f"{d.points:+.2f}%p | {rel} | {g.n_train_samples:,}{mark} |"
        )
    mn = report.min_gain
    lines += [
        "",
        f"- 평균 이득: {report.mean_gain}",
        f"- 최소 이득: {mn[0]} {mn[1]}" if mn else "- 최소 이득: 산출 불가",
        (f"- 소규모 클라이언트({report.small_client}) 이득: "
         f"{report.small_client_gain or '산출 불가'}"),
        (f"- 이득 양수 비율: {report.positive_ratio:.0%}"
         + (f" (손해: {', '.join(report.losers)})" if report.losers else "")),
        (f"- 성능 격차: sd {report.disparity.solo_sd:.4f} → "
         f"{report.disparity.federated_sd:.4f} ({report.disparity.reduction})"),
    ]
    if report.trajectory:
        firsts = {
            g.client_id: report.first_positive_round(g.client_id) for g in report.gains
        }
        lines.append(
            "- 이득 전환 라운드: "
            + ", ".join(f"{k} {v if v is not None else '없음'}" for k, v in firsts.items())
        )
        per_mb = report.gain_per_mb()
        lines.append(
            "- 통신량 대비 이득: "
            + ", ".join(
                f"{k} {'정의 불가' if v is None else f'{v:+.3f}%p/MB'}"
                for k, v in per_mb.items()
            )
        )
    if report.caveats:
        lines += ["", "조건:"] + [f"- {c}" for c in report.caveats]
    return "\n".join(lines)


def rows_to_client_metric(
    rows: Iterable[Mapping[str, object]],
    *,
    cell: str,
    metric_name: str,
    round_idx: int | None = None,
) -> dict[str, float]:
    """원자 로그에서 (칸, 지표)의 클라이언트별 최종값을 뽑는다.

    `round_idx` 를 주지 않으면 각 클라이언트의 **마지막 라운드** 값을 쓴다. last 채점
    원칙과 같은 결이다.
    """
    best: dict[str, tuple[int, float]] = {}
    for r in rows:
        if str(r.get("cell")) != cell or str(r.get("metric_name")) != metric_name:
            continue
        try:
            rd = int(r["round"])
            val = float(r["metric_value"])
            client = str(r["client_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if round_idx is not None and rd != round_idx:
            continue
        if client not in best or rd >= best[client][0]:
            best[client] = (rd, val)
    return {c: v for c, (_, v) in best.items()}
