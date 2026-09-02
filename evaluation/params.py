"""채점 파라미터 단일 수납처 — 임계·모집단·시드가 여기서만 나온다 (77번 과제 2·6).

**코드는 값을 읽기만 한다.** 확정 하이퍼파라미터는 `configs/` 에 둔다(개발규약). 검출
`conf` 는 다섯 칸 공통 고정 항목이므로 정착지는 `configs/base.yaml`(A 소관)이고, 총괄
배분 전까지만 여기 폴백 상수로 산다. **configs 에 키가 생기면 그쪽이 이긴다** —
`resolve_*` 가 configs 를 먼저 보고 어디서 읽었는지를 `source` 로 산출물에 남긴다.

`configs/rag.yaml`(D 소관)에는 두지 않는다. 검출 설정이지 RAG 설정이 아니다.

게이트 상수도 같은 규칙이다. A 가 content-free 천장을 재산출해 `preregistered:` 에
올리면 `--gate` 플래그 없이도 그 값으로 채점된다. 도착 전에는 `prereg.py` 의 사전등록
자명하한이 폴백이고, 산출물의 `source` 가 "A 미도착"임을 스스로 밝힌다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO / "configs" / "base.yaml"

# --------------------------------------------------------------------------------------
# 폴백 상수 — configs 로 옮기기 전의 임시 거처. 값의 근거는 77번 §과제 2.
# --------------------------------------------------------------------------------------

CONF_FALLBACK = 0.25
"""검출 신뢰도 임계. Q4 문서 기본값이고 65·66번이 이 값으로 채점했다.

**이 상수는 통합형에 대응물이 없다**(생성 모델은 신뢰도를 내지 않는다). 그래서 단일
값으로 칸을 비교하면 분리형만 저신뢰 박스를 버린 뒤 들어간다 — 감사 D-1. 비교는
`CONF_SWEEP` 곡선으로 하고, 이 값은 65·66번 재현용 운용점으로만 쓴다.
"""

CONF_FLOOR = 0.01
"""재추론 하한. 이 아래는 저장하지 않는다.

NMS 는 **더 높은 점수의 박스만이** 다른 박스를 억제하므로, 하한을 낮춰도 점수 ≥ t 인
생존 박스 집합은 변하지 않는다. 따라서 하한 1회 추론 + 사후 필터가 임계별 재추론과
같은 결과를 준다. 이 동치는 가정이 아니라 `--verify-parity` 로 실측한다(65번 산출물의
0.25 집합과 비트 단위 대조).
"""

CONF_SWEEP: tuple[float, ...] = (
    0.01, 0.02, 0.03, 0.04,
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
)
"""임계 스윕 격자. 감사 지시의 0.05~0.50 을 0.05 간격으로 채우고 **하한까지 내린다.**

0.05 아래를 더 넣는 이유는 게이트 판정 때문이다. "어떤 칸도 게이트를 못 넘는다"는
주장은 **임계를 가장 유리하게 잡아도 못 넘는다**는 뜻이어야 성립한다. 분리형 Macro-F1
은 임계가 낮을수록 커지므로(재현율이 정밀도보다 빠르게 오른다) 하한이 상한 후보다.
"""

IMGSZ = 416
DEVICE = "cpu"
MAX_DET = 300
"""Ultralytics 기본값. 명시해 둔다 — 하한 추론에서 상한에 걸리면 스윕이 왜곡된다."""

PILOT_SEED = 20260828
"""C 의 파일럿 base_seed (meta.json 실측). 65·66번과 같은 값."""

PILOT_SNAPSHOT = "data/processed/aihub71761_rt_v1_pilot3000"
FROZEN_SNAPSHOT = "data/interim/manifest_v1"

CLASS_NAMES: tuple[str, ...] = ("crack", "porosity", "lack_of_fusion", "slag_inclusion")
"""검출 4클래스. 원본 라벨 문자열이 아니라 C 의 학습 클래스 순서다(nc=4 주입 순서)."""

# configs 에서 먼저 찾아볼 키. 앞에서부터 처음 발견된 것을 쓴다.
CONF_KEYS: tuple[str, ...] = (
    "evaluation.detection.conf",
    "detection.conf",
    "predict.conf",
)
GATE_KEYS: tuple[str, ...] = (
    "preregistered.content_free_best_macro_f1",
    "preregistered.content_free_ceiling_macro_f1",
    "preregistered.gate_macro_f1",
    "preregistered.metadata_only_best_macro_f1",
)
GATE_TOLERANCE_KEYS: tuple[str, ...] = (
    "preregistered.gate_tolerance",
)


@dataclass(frozen=True)
class Resolved:
    """값 하나와 그 출처. **출처를 산출물에 싣는 것이 이 타입의 존재 이유다.**"""

    value: float
    source: str

    def as_dict(self) -> dict:
        return {"value": self.value, "source": self.source}


def _dig(cfg: object, dotted: str) -> object | None:
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def load_base_config(path: str | Path = BASE_CONFIG) -> dict:
    """`configs/base.yaml` 을 읽는다. **쓰지 않는다** — A 소관 파일이다."""
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _resolve(
    keys: Sequence[str], fallback: float, fallback_note: str, cfg: dict
) -> Resolved:
    for k in keys:
        v = _dig(cfg, k)
        if isinstance(v, (int, float)):
            return Resolved(float(v), f"configs/base.yaml:{k}")
    return Resolved(fallback, fallback_note)


def resolve_conf(cfg: dict | None = None) -> Resolved:
    """검출 conf 임계. configs 우선, 없으면 폴백 + 그 사실을 출처에 적는다."""
    return _resolve(
        CONF_KEYS, CONF_FALLBACK,
        "fallback:evaluation.params.CONF_FALLBACK (configs 미등록 — 총괄 배분 대기)",
        cfg if cfg is not None else load_base_config(),
    )


def resolve_gate(cfg: dict | None = None) -> Resolved:
    """게이트 선. A 의 content-free 천장 재산출이 도착하면 자동으로 그 값을 쓴다."""
    from evaluation.prereg import PREREG

    return _resolve(
        GATE_KEYS, PREREG.all_positive_macro_f1,
        "fallback:evaluation.prereg.all_positive_macro_f1 "
        "(A 의 content-free 천장 재산출 미도착 — 74번 A-1)",
        cfg if cfg is not None else load_base_config(),
    )


def resolve_gate_tolerance(cfg: dict | None = None) -> Resolved:
    from evaluation.prereg import TOLERANCE

    return _resolve(
        GATE_TOLERANCE_KEYS, TOLERANCE,
        "fallback:evaluation.prereg.TOLERANCE",
        cfg if cfg is not None else load_base_config(),
    )


@dataclass(frozen=True)
class ScoringParams:
    """한 번의 채점이 쓰는 파라미터 전량. 스크립트는 이것만 받는다."""

    snapshot: Path
    pilot: Path
    out: Path
    seed: int = PILOT_SEED
    imgsz: int = IMGSZ
    device: str = DEVICE
    max_det: int = MAX_DET
    conf: Resolved = field(default_factory=resolve_conf)
    conf_floor: float = CONF_FLOOR
    conf_sweep: tuple[float, ...] = CONF_SWEEP
    gate: Resolved = field(default_factory=resolve_gate)
    gate_tolerance: Resolved = field(default_factory=resolve_gate_tolerance)
    class_names: tuple[str, ...] = CLASS_NAMES

    @property
    def gate_pass_line(self) -> float:
        """통과선 = 게이트 + 여유. 게이트 자체를 넘는 것으로는 부족하다(prereg 계약)."""
        return round(self.gate.value + self.gate_tolerance.value, 4)

    def as_dict(self) -> dict:
        return {
            "snapshot": str(self.snapshot),
            "pilot": str(self.pilot),
            "seed": self.seed,
            "imgsz": self.imgsz,
            "device": self.device,
            "max_det": self.max_det,
            "conf": self.conf.as_dict(),
            "conf_floor": self.conf_floor,
            "conf_sweep": list(self.conf_sweep),
            "gate": self.gate.as_dict(),
            "gate_tolerance": self.gate_tolerance.as_dict(),
            "gate_pass_line": self.gate_pass_line,
            "class_names": list(self.class_names),
        }


def add_common_args(ap) -> None:
    """다섯 칸 채점 스크립트가 공유하는 인자. **정의를 한 벌만 둔다.**"""
    ap.add_argument("--snapshot", default=PILOT_SNAPSHOT)
    ap.add_argument("--pilot", default="outputs/pilot_c")
    ap.add_argument("--out", default="outputs/pilot_d")
    ap.add_argument("--seed", type=int, default=PILOT_SEED)
    ap.add_argument("--conf", type=float, default=None,
                    help="검출 임계 수동 지정. 지정하면 configs·폴백보다 우선한다")
    ap.add_argument("--gate", type=float, default=None,
                    help="게이트 선 수동 지정. A 의 재산출이 오면 이 플래그로 즉시 재채점한다")


def params_from_args(args) -> ScoringParams:
    cfg = load_base_config()
    conf = resolve_conf(cfg)
    gate = resolve_gate(cfg)
    if getattr(args, "conf", None) is not None:
        conf = Resolved(float(args.conf), "cli:--conf")
    if getattr(args, "gate", None) is not None:
        gate = Resolved(float(args.gate), "cli:--gate")
    return ScoringParams(
        snapshot=Path(args.snapshot), pilot=Path(args.pilot), out=Path(args.out),
        seed=int(args.seed), conf=conf, gate=gate,
        gate_tolerance=resolve_gate_tolerance(cfg),
    )
