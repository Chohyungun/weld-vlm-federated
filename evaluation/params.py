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
from functools import lru_cache
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

COORD_SPACE = "ABS_ORIG"
"""다섯 칸 공통 좌표 규약. **총괄 판정 1 (2026-09-02) · main 47c4dbc.**

카나리아-1 실측 6/6 이 절대 원본 픽셀이었고 개발규약 3-8 이 "네이티브를 따른다"를 이미
정해 두었다. NORM_1000 은 라벨 왕복에서 계통 오차를 심는다(실페어 4,560 박스 median
0.98119, IoU<0.95 379건 / ABS_ORIG 는 4,560건 전부 1.000000).

**D 는 좌표를 변환하지 않는다.** 계약 #4 의 `bbox_px` 는 이미 원본 픽셀이고, `coord_space`
는 그 값이 **어느 규약의 모델에서 나왔는지**를 기록한다. 그래서 D 가 할 일은 변환이 아니라
**대조**다 — 칸마다 선언된 규약이 이 상수와 같은지 보고, 다르면 표에 싣기 전에 드러낸다.
규약이 갈린 채로 한 표에 실리는 것이 함정 #4 가 IoU 0.938 → 0.055 를 만든 경로다.
"""

COORD_SPACE_KEYS: tuple[str, ...] = (
    "evaluation.coord_space",
    "coords.coord_space",
    "vlm.coord_space",
)

DEVICE = "cpu"
MAX_DET = 300
"""Ultralytics 기본값. 명시해 둔다 — 하한 추론에서 상한에 걸리면 스윕이 왜곡된다."""

# --------------------------------------------------------------------------------------
# 검출 프로파일 — 채점 해상도·모델 규모는 **학습 프로파일에서 받는다** (82번 §10-3-1)
# --------------------------------------------------------------------------------------

DETECTION_MODEL_BY_PROFILE: dict[str, str] = {
    "main": "yolo11s.pt",
    "pilot": "yolo11n.pt",
}
"""프로파일 → 검출 모델. 모델 규모는 C 의 `PROFILES` 에 **없다** — 거기는 학습
하이퍼파라미터뿐이고 모델 이름은 run cfg(`fl/server_app.py` 의 `cfg["model"]`, 기본
yolo11s.pt)와 스크립트 상수(`scripts/pilot_c.py` yolo11n.pt · `scripts/gate_reduced_pilot.py`
yolo11s.pt)에서 온다. 그래서 규모는 여기 두되, **npz 형상 대조가 최종 방어선**이다 —
프로파일이 틀리면 `load_yolo_from_npz` 가 어느 규모의 가중치인지 말하며 실패한다.
"""

DEFAULT_PROFILE = "pilot"
"""기본 프로파일. 지금 `outputs/pilot_c` 에 있는 다섯 칸이 전부 파일럿 프로파일
(YOLO11n · 416)이라 기존 재현 명령이 그대로 돌아야 한다. 본실험 채점은 `--profile main`
을 **명시**한다 — 빠뜨리면 형상 불일치로 시작조차 하지 않으므로 조용히 틀릴 수는 없다.
"""

PREDICT_CHUNK = 256
"""`predict_cell` 이 한 번에 Ultralytics 에 넘기는 이미지 수.

전량을 한 번에 주면 전처리가 한 텐서로 뭉친다 — 평가셋 12,461장에서 **36.7GB 단일
할당**을 시도하다 죽었다(82번 §10-3-1 결함 2, C 실측). 256장이면 약 0.8GB 다. 파일럿
653장에서는 드러나지 않던 규모 결함이다.
"""


@lru_cache(maxsize=1)
def detection_profiles() -> dict[str, dict]:
    """프로파일 → `{imgsz, model, model_cfg}`.

    **imgsz 는 C 의 `detection.round_runner.PROFILES` 를 실행 시점에 읽는다.** `configs/gpu/*.yaml`
    의 `detection` 블록은 `wired: false` 고 정본은 그 파일이 스스로 가리키는 `PROFILES` 다.
    값을 여기 베껴 두면 C 가 바꿨을 때 채점만 옛 해상도로 남는다 — 640 으로 학습한 모델을
    416 으로 추론하면 그 차이가 판정 오차가 된다(82번 결함 3).

    `detection.round_runner` 는 torch 를 끌므로 지연 import 한다.
    """
    from detection.round_runner import PROFILES

    out: dict[str, dict] = {}
    for name, model in DETECTION_MODEL_BY_PROFILE.items():
        if name not in PROFILES:
            raise KeyError(f"C 의 PROFILES 에 {name!r} 가 없다 — 프로파일 표가 갈렸다")
        out[name] = {
            "imgsz": int(PROFILES[name]["imgsz"]),
            "model": model,
            "model_cfg": model.removesuffix(".pt") + ".yaml",
        }
    return out

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
    ("fixed_before_main_runs.content_free__sel_val__fit_trainval__score_eval12461"
     ".max_macro_f1"),
    "preregistered.content_free_best_macro_f1",
    "preregistered.trivial_all_positive_macro_f1",
)
"""**실재하는 키를 앞에 둔다.** 이전 판의 후보 4개는 `configs/base.yaml` 의 어떤 키와도
겹치지 않아 "configs 에 키가 생기면 그쪽이 이긴다"는 계약이 게이트에 대해 **한 번도
발화하지 않았다**(80번 D15). A 가 76번에서 등록한 실제 경로가 첫 줄이다.

키 이름 규약은 `<계열>__sel_<족선택>__fit_<적합>__score_<채점>` 이다. `fit_eval`·`sel_eval`
이 붙은 값은 평가셋 유도라 게이트 재료가 아니므로 **후보에 넣지 않는다.**
"""

GATE_TOLERANCE_KEYS: tuple[str, ...] = (
    "fixed_before_main_runs.gate_tolerance",
    "preregistered.gate_tolerance",
)

GATE_STATUS_KEYS: tuple[str, ...] = (
    "fixed_before_main_runs.gate_status",
)
"""`판정_대기` | `적용` | `대체지표`. A 가 "값은 등록하되 자동 차단에는 쓰지 않는다"로
박아 둔 스위치다(76번 §1-4). 채점기는 이 값을 **읽어서 산출물에 싣는다** — 게이트가
왜 발동하지 않았는지가 표에 남아야 한다."""


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


def resolve_gate_status(cfg: dict | None = None) -> str:
    src = cfg if cfg is not None else load_base_config()
    for k in GATE_STATUS_KEYS:
        v = _dig(src, k)
        if isinstance(v, str) and v:
            return v
    return "미등록"


def resolve_gate(cfg: dict | None = None) -> Resolved:
    """게이트 선. A 의 content-free 천장 재산출이 도착하면 자동으로 그 값을 쓴다."""
    from evaluation.prereg import PREREG

    return _resolve(
        GATE_KEYS, PREREG.all_positive_macro_f1,
        "fallback:evaluation.prereg.all_positive_macro_f1 "
        "(A 의 content-free 천장 재산출 미도착 — 74번 A-1)",
        cfg if cfg is not None else load_base_config(),
    )


def resolve_coord_space(cfg: dict | None = None) -> tuple[str, str]:
    """(규약, 출처). configs 에 키가 있으면 그쪽이 이긴다."""
    src = cfg if cfg is not None else load_base_config()
    for k in COORD_SPACE_KEYS:
        v = _dig(src, k)
        if isinstance(v, str) and v:
            return v, f"configs/base.yaml:{k}"
    return COORD_SPACE, "fallback:evaluation.params.COORD_SPACE (총괄 판정 1)"


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
    profile: str = DEFAULT_PROFILE
    """학습 프로파일 이름. `imgsz`·`model_cfg` 를 비워 두면 여기서 채운다."""
    imgsz: int | None = None
    """None 이면 프로파일 값. 명시하면 그 값을 쓰되 `imgsz_source` 가 그 사실을 남긴다."""
    model_cfg: str | None = None
    """`DetectionModel(cfg=...)` 에 줄 yaml. None 이면 프로파일 값."""
    predict_chunk: int = PREDICT_CHUNK
    device: str = DEVICE
    max_det: int = MAX_DET
    conf: Resolved = field(default_factory=resolve_conf)
    conf_floor: float = CONF_FLOOR
    conf_sweep: tuple[float, ...] = CONF_SWEEP
    gate: Resolved = field(default_factory=resolve_gate)
    gate_tolerance: Resolved = field(default_factory=resolve_gate_tolerance)
    class_names: tuple[str, ...] = CLASS_NAMES
    coord_space: str = COORD_SPACE
    coord_space_source: str = "fallback:evaluation.params.COORD_SPACE (총괄 판정 1)"

    def __post_init__(self) -> None:
        profiles = detection_profiles()
        if self.profile not in profiles:
            raise ValueError(
                f"알 수 없는 채점 프로파일: {self.profile!r}. 허용: {sorted(profiles)}"
            )
        prof = profiles[self.profile]
        if self.imgsz is None:
            object.__setattr__(self, "imgsz", prof["imgsz"])
        if self.model_cfg is None:
            object.__setattr__(self, "model_cfg", prof["model_cfg"])
        if int(self.predict_chunk) < 1:
            raise ValueError(f"predict_chunk 는 1 이상이어야 한다 (받은 값 {self.predict_chunk})")

    @property
    def imgsz_source(self) -> str:
        """해상도가 프로파일에서 왔는지 호출부가 박았는지. 산출물에 남긴다."""
        prof = detection_profiles()[self.profile]
        if self.imgsz == prof["imgsz"]:
            return f"profile:{self.profile} (detection.round_runner.PROFILES)"
        return f"explicit:{self.imgsz} (프로파일 {self.profile} 은 {prof['imgsz']})"

    @property
    def gate_pass_line(self) -> float:
        """통과선 = 게이트 + 여유. 게이트 자체를 넘는 것으로는 부족하다(prereg 계약)."""
        return round(self.gate.value + self.gate_tolerance.value, 4)

    def as_dict(self) -> dict:
        return {
            "snapshot": str(self.snapshot),
            "pilot": str(self.pilot),
            "seed": self.seed,
            "profile": self.profile,
            "model_cfg": self.model_cfg,
            "imgsz": self.imgsz,
            "imgsz_source": self.imgsz_source,
            "predict_chunk": self.predict_chunk,
            "device": self.device,
            "max_det": self.max_det,
            "conf": self.conf.as_dict(),
            "conf_floor": self.conf_floor,
            "conf_sweep": list(self.conf_sweep),
            "gate": self.gate.as_dict(),
            "gate_tolerance": self.gate_tolerance.as_dict(),
            "gate_pass_line": self.gate_pass_line,
            "class_names": list(self.class_names),
            "coord_space": {"value": self.coord_space, "source": self.coord_space_source},
        }


def add_common_args(ap) -> None:
    """다섯 칸 채점 스크립트가 공유하는 인자. **정의를 한 벌만 둔다.**"""
    ap.add_argument("--snapshot", default=PILOT_SNAPSHOT)
    ap.add_argument("--pilot", default="outputs/pilot_c")
    ap.add_argument("--out", default="outputs/pilot_d")
    ap.add_argument("--seed", type=int, default=PILOT_SEED)
    ap.add_argument("--profile", choices=sorted(DETECTION_MODEL_BY_PROFILE),
                    default=DEFAULT_PROFILE,
                    help="학습 프로파일. 해상도·모델 규모가 여기서 온다. 본실험은 main 을 명시")
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
    space, space_src = resolve_coord_space(cfg)
    return ScoringParams(
        snapshot=Path(args.snapshot), pilot=Path(args.pilot), out=Path(args.out),
        seed=int(args.seed), profile=getattr(args, "profile", DEFAULT_PROFILE),
        conf=conf, gate=gate,
        gate_tolerance=resolve_gate_tolerance(cfg),
        coord_space=space, coord_space_source=space_src,
    )
