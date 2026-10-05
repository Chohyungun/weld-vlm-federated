"""우회 금지 그물 — 진입점 미니스펙 3판 1-2 라 · 07번 §32-10.

채점 산출물이 나오는 길이 **검증을 지나는 길 하나뿐**이어야 한다. 그물의 이름(채점 · 규칙 ⑤ · 옛 구간 함수)을
가져오거나 부르는 모듈은 같은 파일에서 `verify_bundle` 도 불러야 한다. 예외는 **닫힌 목록**뿐이다.

- 범위는 **저장소가 추적하는 모든 `.py`** 다(`tests/` 와 `evaluation/` 자신은 뺀다). 2판은 `scripts/probe/` 만 보아
  그 밖에 둔 우회 모듈을 놓쳤다(판정 09 의 I-7). 파일 목록은 `git ls-files` 로 얻는다 — 정션을 따라 걷지 않는다.
- **구문 트리로 본다.** 글자 검색이 아니다 — 독스트링의 언급은 호출이 아니다(`boundary_convention_audit.py`).
- 닫힌 목록에 새 이름을 더하려면 이 파일을 고쳐야 한다 — 조용히 늘지 않는다. 목록의 이름이 더는 그물에 닿지
  않아도 깨진다 — 목록이 낡지 않는다.
"""

from __future__ import annotations

import ast
import subprocess
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _public_functions(rel: str) -> frozenset[str]:
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    return frozenset(n.name for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_"))


NET: dict[str, frozenset[str]] = {
    "evaluation.score": frozenset({"score_records"}),
    "evaluation.metrics.box_f1": _public_functions("evaluation/metrics/box_f1.py"),
    "evaluation.rule5": frozenset({"judge_rule5", "judge_rule5_registered"}),
    "evaluation.recovery_ci": frozenset({"percentile_ci", "recovery_from", "paired_bootstrap", "weighted_map"}),
    "evaluation.stats": frozenset({"cluster_bootstrap"}),
}
"""모듈 → 그물의 이름. `box_f1` 은 공개 함수 전부다(파일에서 센다). `rule5` 는 아직 없는 모듈이다(§29-5)."""
INTERVAL_NAMES = frozenset({"percentile_ci", "paired_bootstrap", "weighted_map", "cluster_bootstrap"})
"""구간을 내는 이름."""

CLOSED: dict[str, tuple[str, str]] = {
    "scripts/probe/score_cells.py": (
        "interval_unmarked",
        "분리형 사전실험과 파일럿 두 칸을 채점한다. 판별력 구간을 싣지만 옛 등록 블록 · 규약 문장이 없다. "
        "고정 해시 파일이라 고치지 않는다"),
    "scripts/probe/recovery_bootstrap.py": ("old_registration", "회복률 CI 생성기 — RECOVERY_CI_REGISTRATION 을 싣는다"),
    "scripts/probe/map50_independent.py": ("no_interval", "점추정 독립 검산 — recovery_from 만 가져온다"),
    "scripts/probe/neardup_rescore.py": ("old_registration", "옛 등록 블록과 interval_convention 을 싣는다"),
    "scripts/probe/diagnostics3.py": ("no_interval", "score_records 만 가져온다"),
    "scripts/probe/p9_uni_diagnosis.py": ("no_interval", "score_records 만 가져온다"),
    "scripts/probe/sweep_detection_conf.py": ("no_interval", "score_records 만 가져온다"),
    "scripts/gate_reduced_pilot.py": (
        "interval_unmarked",
        "축소 파일럿 게이트 — cluster_bootstrap 의 구간을 싣지만 옛 등록 블록 · 규약 문장이 없다. 다른 소관의 파일이다"),
    "scripts/recompute_baselines.py": ("no_interval", "score_records 만 가져온다"),
    "scripts/verify_ablation_arms.py": ("no_interval", "score_records 만 가져온다"),
}
"""닫힌 목록 열 — 사전실험 스크립트다. 통합형 묶음을 읽지 않는다(`UNI_MAIN_TAGS` 를 가져오지 않는다 — 아래 시험).

분류 셋. `old_registration` 은 구간을 싣되 옛 등록 블록이나 규약 문장을 산출에 싣는다 — 시험이 그 참조를 본다.
`no_interval` 은 구간을 내는 이름을 가져오지 않는다 — 시험이 본다. `interval_unmarked` 는 구간을 싣는데 표지가 없다 —
3판이 가정한 "둘 가운데 하나" 에 들지 않는 둘이고, 고칠 수 없는 파일이라 이름과 까닭만 적는다(보고서에 올렸다)."""


# ---------------------------------------------------------------- 검사기

def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def net_touches(tree: ast.AST) -> set[str]:
    """이 모듈이 가져오거나 부르는 그물의 이름."""
    hit: set[str] = set()
    alias_of: dict[str, str] = {}          # 지역 이름 → 그물 모듈
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module in NET:
                for a in node.names:
                    if a.name == "*":
                        hit |= NET[node.module]
                    elif a.name in NET[node.module]:
                        hit.add(a.name)
            else:
                for a in node.names:       # from evaluation import recovery_ci · from evaluation.metrics import box_f1
                    if f"{node.module}.{a.name}" in NET:
                        alias_of[a.asname or a.name] = f"{node.module}.{a.name}"
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in NET:
                    alias_of[a.asname or a.name] = a.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            owner = _dotted(node.value)
            if owner is None:
                continue
            module = alias_of.get(owner, owner)
            if module in NET and node.attr in NET[module]:
                hit.add(node.attr)
    return hit


def calls_verify(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == "verify_bundle") or \
                    (isinstance(f, ast.Attribute) and f.attr == "verify_bundle"):
                return True
    return False


def names_referenced(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.alias):
            out.add(node.asname or node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
    return out


def in_scope(rel: str) -> bool:
    return rel.endswith(".py") and not rel.startswith(("tests/", "evaluation/"))


def scan(root: Path, files: list[str]) -> dict[str, set[str]]:
    """그물에 닿고 `verify_bundle` 을 부르지 않는 모듈 → 닿은 이름."""
    out: dict[str, set[str]] = {}
    for rel in files:
        if not in_scope(rel):
            continue
        with warnings.catch_warnings():    # 남의 파일의 글자 이스케이프 경고는 이 검사와 무관하다
            warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
            tree = ast.parse((root / rel).read_text(encoding="utf-8-sig"), filename=rel)
        touched = net_touches(tree)
        if touched and not calls_verify(tree):
            out[rel] = touched
    return out


def tracked_py() -> list[str]:
    raw = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--", "*.py"],
                         capture_output=True, check=True).stdout
    return sorted(p for p in raw.decode("utf-8").split("\0") if p)


# ---------------------------------------------------------------- 저장소

@pytest.fixture(scope="module")
def found() -> dict[str, set[str]]:
    return scan(REPO, tracked_py())


def test_그물에_닿고_검증을_부르지_않는_모듈은_닫힌_목록뿐이다(found) -> None:
    extra = sorted(set(found) - set(CLOSED))
    stale = sorted(set(CLOSED) - set(found))
    assert not extra, f"닫힌 목록 밖에서 그물에 닿는 모듈 — verify_bundle 을 부르거나 목록을 고친다: {extra}"
    assert not stale, f"닫힌 목록의 이름이 더는 그물에 닿지 않는다 — 목록에서 뺀다: {stale}"


def test_닫힌_목록은_열이다() -> None:
    assert len(CLOSED) == 10
    assert sum(1 for p in CLOSED if not p.startswith("scripts/probe/")) == 3


@pytest.mark.parametrize("rel", sorted(p for p, (k, _) in CLOSED.items() if k == "old_registration"))
def test_구간을_싣는_스크립트는_옛_등록이나_규약_문장을_싣는다(rel: str) -> None:
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    assert names_referenced(tree) & {"RECOVERY_CI_REGISTRATION", "interval_convention"}


@pytest.mark.parametrize("rel", sorted(p for p, (k, _) in CLOSED.items() if k == "no_interval"))
def test_구간을_싣지_않는_스크립트는_구간_함수를_가져오지_않는다(rel: str, found) -> None:
    assert not found[rel] & INTERVAL_NAMES


@pytest.mark.parametrize("rel", sorted(CLOSED))
def test_닫힌_목록은_통합형_본실험_태그를_가져오지_않는다(rel: str) -> None:
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    assert "UNI_MAIN_TAGS" not in names_referenced(tree)


def test_글로만_적은_이름은_그물에_걸리지_않는다() -> None:
    """글자 검색이 아니어야 하는 까닭 — 이 파일은 이름을 글로만 적었다."""
    rel = "scripts/probe/boundary_convention_audit.py"
    text = (REPO / rel).read_text(encoding="utf-8")
    assert "box_f1" in text
    assert not net_touches(ast.parse(text))


# ---------------------------------------------------------------- 검사기의 이빨

SAMPLES = {
    "from evaluation.score import score_records\nscore_records([])\n": {"score_records"},
    "from evaluation import recovery_ci as rc\nrc.percentile_ci([], 0.05)\n": {"percentile_ci"},
    "import evaluation.stats\nevaluation.stats.cluster_bootstrap([], {}, seed=1)\n": {"cluster_bootstrap"},
    "import evaluation.recovery_ci as r\nr.recovery_from(1, 2, [3])\n": {"recovery_from"},
    "from evaluation.rule5 import judge_rule5_registered\n": {"judge_rule5_registered"},
    "from evaluation.recovery_ci import *\n": NET["evaluation.recovery_ci"],
    '"""score_records 와 percentile_ci 를 글로만 적는다."""\nx = "cluster_bootstrap"\n': set(),
    "from evaluation.recovery_ci import read_artifact\n": set(),
}


@pytest.mark.parametrize(("src", "want"), list(SAMPLES.items()))
def test_검사기가_가져오기와_부르기를_구문_트리로_잡는다(src: str, want: set[str]) -> None:
    assert net_touches(ast.parse(src)) == want


def test_box_f1_의_공개_함수를_가져오면_걸린다() -> None:
    name = sorted(NET["evaluation.metrics.box_f1"])[0]
    assert net_touches(ast.parse(f"from evaluation.metrics.box_f1 import {name}\n")) == {name}
    assert net_touches(ast.parse(f"from evaluation.metrics import box_f1 as b\nb.{name}()\n")) == {name}


def test_검증을_부르면_그물에_닿아도_지난다() -> None:
    src = "from evaluation.bundle_v14 import verify_bundle\nfrom evaluation.score import score_records\nverify_bundle(1)\n"
    tree = ast.parse(src)
    assert net_touches(tree) and calls_verify(tree)


def test_scripts_probe_밖에_둔_우회_모듈도_잡는다(tmp_path: Path) -> None:
    """2판의 그물은 `scripts/probe/` 만 보았다 — 그 밖에 두면 걸리지 않았다."""
    files = {"scripts/bypass.py": "from evaluation.score import score_records\n",
             "corpus/bypass.py": "from evaluation import stats\nstats.cluster_bootstrap([], {}, seed=1)\n",
             "tests/test_x.py": "from evaluation.score import score_records\n",
             "evaluation/inner.py": "from evaluation.score import score_records\n",
             "scripts/probe/ok.py": "from evaluation.score import score_records\nfrom evaluation.bundle_v14 import verify_bundle\nverify_bundle(1)\n"}
    for rel, src in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(src, encoding="utf-8")
    assert set(scan(tmp_path, sorted(files))) == {"scripts/bypass.py", "corpus/bypass.py"}
