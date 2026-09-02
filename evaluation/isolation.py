"""평가 자산(D6) 격리를 **코드가 지키게 한다.** 77번 과제 5.

불변조건 1-4 는 "평가셋·정답 조항 목록은 어떤 학습 단계에도 투입하지 않는다"이고,
지금까지 그 보장은 문서와 사람의 주의뿐이었다. 감사(74번 P7)가 `gold_clauses.csv` 의
공개 노출을 짚었을 때 총괄이 내린 판정이 이것이다 — **노출 자체는 레드라인 위반이
아니지만 실질 통제를 걸어라.**

여기 두 겹을 둔다.

1. **정적** — 학습 경로 모듈이 D6 자산 이름을 문자열로도 언급하지 않는지 훑는다.
   값싸고, 새 파일이 들어와도 자동으로 걸린다.
2. **동적** — `forbid_d6_assets()` 안에서 D6 파일을 열면 예외가 난다. 감사 훅이라
   경로를 우회하는 구현(pandas·csv·pathlib 무엇이든)도 잡힌다.

**두 겹 다 자기참조가 되지 않게 짰다.** B 의 부등식 방향 게이트가 "원리적으로 통과할
수밖에 없는 검사"였다는 감사 M9 가 이 파일의 설계 제약이다. 그래서 시험에 양성 대조가
함께 있다 — 훅이 실제로 예외를 내는지를 먼저 확인하고, 그 다음에 학습 경로를 통과시킨다.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

D6_ASSET_NAMES: tuple[str, ...] = (
    "gold_clauses.csv",
)
"""파일명으로 식별하는 D6 자산. 경로가 아니라 이름으로 본다 — 상대·절대·정션 경로가
섞여 있어 경로 비교는 새기 쉽다.

한국어 평가셋(599행)과 낯선 데이터는 아직 저장소에 실물이 없다. 들어오면 여기 추가한다.
평가**셋 이미지**는 파일명이 학습분과 구분되지 않으므로 이름으로 막을 수 없다 —
그쪽은 `split == "eval"` 행이 학습 뷰에 들어가지 않는다는 행동 시험으로 막는다.
"""

TRAIN_ROOTS: tuple[str, ...] = (
    "detection",
    "vlm",
    "fl",
)
"""학습 경로. 이 아래 어떤 모듈도 D6 자산을 읽어서는 안 된다.

`corpus/` 는 D6 자산의 **생산자**라 통째로 넣을 수 없다(`materialize_derived.py` 가
`gold_clauses.csv` 를 만든다). 합성 학습 데이터를 만드는 쪽만 아래에서 따로 짚는다.
"""

CORPUS_TRAIN_PATHS: tuple[str, ...] = (
    "corpus/generate",
    "corpus/validate",
)
"""corpus 중 학습 데이터(D3·D4)를 만드는 경로. 생산자(`corpus/rules`)와 구분한다."""

TRAIN_SCRIPTS: tuple[str, ...] = (
    "scripts/pilot_c.py",
    "scripts/run_ablation.py",
)
"""학습을 기동하는 스크립트. 모듈이 깨끗해도 여기서 읽으면 같은 위반이다."""


class EvalAssetLeak(RuntimeError):
    """학습 경로가 D6 자산을 열었다. **잡아서 무시하지 마라** — 논문 철회 사유다."""


@dataclass(frozen=True)
class StaticHit:
    path: str
    line: int
    asset: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} — {self.asset} ({self.text.strip()[:70]})"


def _iter_sources(roots: Sequence[str], repo: Path) -> Iterator[Path]:
    for r in roots:
        p = repo / r
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*.py")):
                if "__pycache__" not in f.parts:
                    yield f


def scan_training_paths(
    repo: Path | str = REPO,
    roots: Sequence[str] | None = None,
    assets: Sequence[str] = D6_ASSET_NAMES,
) -> list[StaticHit]:
    """학습 경로 소스에서 D6 자산 이름을 찾는다. 주석에 적힌 것도 잡는다.

    주석까지 잡는 것은 과하지 않다 — "여기서 gold 를 읽으면 안 된다"는 주석과 실제
    호출을 정적으로 가르려면 결국 실행 의미를 알아야 하는데, 그건 동적 훅의 일이다.
    이 층은 **의심스러운 언급을 전부 올려** 사람이 판정하게 하는 쪽이 맞다.
    """
    repo = Path(repo)
    hits: list[StaticHit] = []
    for f in _iter_sources(list(roots or (*TRAIN_ROOTS, *CORPUS_TRAIN_PATHS, *TRAIN_SCRIPTS)),
                           repo):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            for a in assets:
                if a in line:
                    hits.append(StaticHit(
                        f.relative_to(repo).as_posix(), i, a, line))
    return hits


# --------------------------------------------------------------------------------------
# 동적 — 감사 훅. 한 번 설치되면 프로세스에서 제거할 수 없으므로 비활성 시 비용이 없게 짠다.
# --------------------------------------------------------------------------------------

_installed = False
_active: list[tuple[str, ...]] = []
"""활성 중인 금지 자산 목록 스택. 비어 있으면 훅은 아무것도 하지 않는다."""


def _audit(event: str, args: tuple) -> None:
    if not _active or event not in ("open", "os.open"):
        return
    try:
        target = str(args[0])
    except Exception:                       # noqa: BLE001 — 훅에서 죽지 않는다
        return
    name = target.replace("\\", "/").rsplit("/", 1)[-1]
    for asset in _active[-1]:
        if name == asset:
            raise EvalAssetLeak(
                f"학습 경로가 평가 자산을 열었다: {target} (불변조건 1-4)"
            )


def _install() -> None:
    global _installed
    if not _installed:
        sys.addaudithook(_audit)
        _installed = True


@contextmanager
def forbid_d6_assets(assets: Sequence[str] = D6_ASSET_NAMES):
    """이 블록 안에서 D6 자산을 열면 `EvalAssetLeak` 이 난다.

    학습 진입점을 이것으로 감싸면 규칙이 문서가 아니라 실행 시점의 계약이 된다.
    시험에서는 **양성 대조**(일부러 열어 예외를 확인)와 함께 써야 한다 — 그러지 않으면
    "아무도 안 열었다"와 "훅이 죽어 있다"를 구분할 수 없다.
    """
    _install()
    _active.append(tuple(assets))
    try:
        yield
    finally:
        _active.pop()
