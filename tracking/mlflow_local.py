"""MLflow 로컬 추적 — sqlite backend + 파일 artifact store. 스펙 §8.

**외부 클라우드 실험 로깅은 금지다**(개발규약 불변조건 2-3). 이 연구의 서사 자체가
"데이터를 밖으로 내보내지 않는다"이므로, 기본이 클라우드 로깅인 도구를 쓰면 논문의 전제가
스스로 무너진다. 금지를 문서에 적는 것과 실행 시점에 막는 것은 다르므로 여기서 막는다.

    from tracking.mlflow_local import guard_no_cloud_logging, start_run
    guard_no_cloud_logging()          # WANDB_* 있으면 즉시 실패
    with start_run(cell="sep_central", seed=0, tags=...) as run: ...
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DIR = REPO_ROOT / "tracking"
"""**절대경로다.** 상대경로면 DB 위치가 CWD 에 따라 갈려 run 이 어디 기록됐는지
사후에 증명할 수 없다(80번 D12 부수). 저장소 루트 기준으로 못 박는다."""

DB_PATH = TRACKING_DIR / "mlruns.db"
ARTIFACT_DIR = TRACKING_DIR / "artifacts"
EXPERIMENT = "weld-fl"

CLOUD_ENV_PREFIXES = ("WANDB_", "COMET_", "NEPTUNE_", "CLEARML_", "MLFLOW_")
"""기본이 외부 클라우드 로깅인 도구들. 존재만 해도 실패시킨다.

**`MLFLOW_` 가 빠져 있었다.** 이 프로젝트가 채택한 추적기가 MLflow 이고 반출 경로는
`MLFLOW_TRACKING_URI=https://…` 다. 환경변수 하나로 전 run 이 외부로 나가는데 가드가
통과시켰다 — 불변조건 2-3 직결이고, 가드가 정작 채택한 도구만 안 막고 있었다(80번 D12).

로컬 sqlite 를 쓰려고 `MLFLOW_TRACKING_URI` 를 **로컬 경로로** 설정하는 것도 막는다.
`tracking_uri()` 를 코드에서 부르면 되고, 환경변수 경유는 값이 바뀌어도 흔적이 안 남는다.
"""

CLOUD_URI_SCHEMES = ("http://", "https://", "databricks", "wasbs://", "gs://", "s3://")
"""원격 추적 스토어 URI 접두. `detect_cloud_logging` 이 값까지 본다."""

REQUIRED_TAGS = (
    "git_commit",
    "manifest_sha256",
    "label_map_sha256",
    "limits_sha256",
    "grade_map_sha256",
    "rag_snapshot_sha256",
    "prompt_sha256",
    "base_ckpt_sha256",
    "coords_sha256",
    "coord_cfg_hash",
    "verdict_mode",
)
"""누락을 즉시 실패로 처리한다. 나중에 채우기로 하면 안 채워진다(§8-2)."""

CELLS = ("uni_central", "uni_fed", "sep_local", "sep_central", "sep_fed")


class CloudLoggingBlocked(RuntimeError):
    """외부 클라우드 로깅 흔적이 감지됐다."""


class MissingRunMetadata(ValueError):
    """필수 로깅 항목이 비었다."""


def detect_cloud_logging(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """차단 대상 환경변수를 찾아 이름을 돌려준다. 판정만 하고 예외를 던지지 않는다.

    이름 접두뿐 아니라 **값의 스킴도 본다** — `MLFLOW_TRACKING_URI` 가 아닌 이름으로
    원격 스토어를 가리키는 경우를 접두 목록만으로는 못 잡는다.
    """
    src = os.environ if env is None else env
    hits = {k for k in src if k.startswith(CLOUD_ENV_PREFIXES)}
    hits |= {
        k for k, v in src.items()
        if "TRACKING_URI" in k.upper() and str(v).lower().startswith(CLOUD_URI_SCHEMES)
    }
    return tuple(sorted(hits))


def guard_no_cloud_logging(env: Mapping[str, str] | None = None) -> None:
    """`WANDB_*` 등이 존재하면 **즉시 실패**한다. 경고로 넘기지 않는다.

    경고는 로그에 묻히고, 묻힌 채로 한 번 돌면 그 run 의 기록이 어디로 갔는지 사후에
    증명할 수 없다. 비반출 서사에서는 그 자체가 결함이다.
    """
    found = detect_cloud_logging(env)
    if found:
        raise CloudLoggingBlocked(
            "외부 클라우드 실험 로깅이 금지돼 있는데 관련 환경변수가 설정돼 있다: "
            f"{', '.join(found)} — 해제한 뒤 다시 실행한다 (개발규약 불변조건 2-3)."
        )


def tracking_uri(db_path: Path | str = DB_PATH) -> str:
    """sqlite backend URI. 서버형 추적 스토어를 쓰지 않는다."""
    return f"sqlite:///{Path(db_path).as_posix()}"


def run_name(cell: str, seed: int, yymmdd: str) -> str:
    """`{cell}_{seed}_{yymmdd}` (예: `sep_fed_s0_260910`). 스펙 §8-2."""
    if cell not in CELLS:
        raise ValueError(f"알 수 없는 칸: {cell!r} — {CELLS} 중 하나여야 한다")
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        raise ValueError(f"yymmdd 형식이 아니다: {yymmdd!r}")
    return f"{cell}_s{seed}_{yymmdd}"


def git_commit(cwd: Path | str = ".") -> str:
    """현 커밋 해시. 실패해도 예외를 던지지 않고 표식을 남긴다 — 추적 실패가 학습을
    막을 이유는 없지만, 무엇이 없었는지는 남아야 한다."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd),
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "UNKNOWN"


def check_required_tags(tags: Mapping[str, str]) -> tuple[str, ...]:
    """누락된 필수 태그 목록. 빈 문자열도 누락으로 본다."""
    return tuple(t for t in REQUIRED_TAGS if not str(tags.get(t, "")).strip())


def require_tags(tags: Mapping[str, str]) -> None:
    missing = check_required_tags(tags)
    if missing:
        raise MissingRunMetadata(
            f"필수 로깅 항목 누락: {', '.join(missing)} — "
            "run 종료 시점에 전부 채워져 있어야 한다 (§8-2)."
        )


@dataclass(frozen=True)
class CellFingerprint:
    """다섯 칸이 같은 출발점에서 시작했다는 증명에 쓰이는 값들."""

    cell: str
    base_ckpt_sha256: str
    coords_sha256: str
    coord_cfg_hash: str
    rag_snapshot_sha256: str


def check_cells_identical(prints: Iterable[CellFingerprint]) -> dict[str, set[str]]:
    """5칸에서 갈리면 안 되는 값이 실제로 같은지 검사한다.

    사람이 확인하면 언젠가 놓친다. 갈린 필드 → 관측된 값 집합을 돌려주고, 빈 dict 면 정상.
    """
    items = list(prints)
    out: dict[str, set[str]] = {}
    for field_name in ("base_ckpt_sha256", "coords_sha256", "coord_cfg_hash",
                       "rag_snapshot_sha256"):
        values = {getattr(p, field_name) for p in items}
        if len(values) > 1:
            out[field_name] = values
    return out


def reject_best_checkpoint(path: str | Path) -> None:
    """`best` 가 파일명에 들어오면 **경고가 아니라 실패**다.

    best 선택은 val 기준의 암묵적 조기 종료이고, 조기 종료 금지는 학습량 등가(R×E=N)라는
    공정성 주장의 근거다(불변조건 3-1·3-2). 채점 대상은 언제나 last 다.
    """
    name = Path(path).name.lower()
    if "best" in name:
        raise ValueError(
            f"best 체크포인트는 채점하지 않는다: {path} — last 를 지정한다 (불변조건 3-2)."
        )
