"""채점 경로 verify 배선 — 체크리스트 17 (80번 D1).

`verify_snapshot` 은 있었다. **채점 리더가 부르지 않았을 뿐이다.** 변조 사본으로
실증됐다 — 승인 로더는 거부하는데 채점 리더는 eval 653장·gold 388장을 경고 없이
반환했다. 저장소 전체에서 호출처가 `manifest_io` 내부와 시험뿐이었다.

계약이 "잠금은 OS 읽기 전용이 아니라 이 검증이다"(Q7 확정)라고 못 박아 두고 정작
채점 경로가 그 검증을 지나지 않은 것이므로, **배선 + 우회 금지**를 함께 건다.

세 겹이다.
1. 변조 픽스처 회귀 — 한 바이트 바꾸면 리더가 거부한다.
2. AST 시험 — 채점 리더 모듈이 `verify` 를 지나지 않고 CSV 를 여는 경로가 없다.
3. 양성 대조 — 정상 스냅샷은 그대로 읽힌다(통제가 채점을 막으면 안 된다).
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

from data.manifest_io import SnapshotVerificationError
from evaluation import eval_set

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "data" / "processed" / "aihub71761_rt_v1_pilot3000"


@pytest.fixture(scope="module")
def has_pilot() -> bool:
    return (PILOT / "SNAPSHOT.sha256").exists()


# --------------------------------------------------------------------------------------
# 1. 변조 픽스처 회귀
# --------------------------------------------------------------------------------------

def test_tampered_manifest_is_rejected(tmp_path: Path, has_pilot: bool) -> None:
    """**핵심 회귀.** 이전 판은 이 입력을 경고 없이 읽어 653장을 돌려줬다."""
    if not has_pilot:
        pytest.skip("파일럿 스냅샷 없음")
    dst = tmp_path / "snap"
    shutil.copytree(PILOT, dst)
    text = (dst / "manifest.csv").read_text(encoding="utf-8")
    # 한 글자만 바꾼다 — 행 수도 열 수도 그대로라 소박한 리더는 아무것도 눈치채지 못한다.
    (dst / "manifest.csv").write_text(text.replace("eval", "evaL", 1), encoding="utf-8")

    with pytest.raises(SnapshotVerificationError):
        eval_set.read_manifest(dst)


def test_tampered_annotations_are_rejected(tmp_path: Path, has_pilot: bool) -> None:
    if not has_pilot:
        pytest.skip("파일럿 스냅샷 없음")
    dst = tmp_path / "snap"
    shutil.copytree(PILOT, dst)
    p = dst / "annotations.csv"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SnapshotVerificationError):
        eval_set.read_gold(dst, {"x"})


def test_missing_snapshot_file_is_rejected(tmp_path: Path, has_pilot: bool) -> None:
    """잠기지 않은 스냅샷은 쓰지 않는다."""
    if not has_pilot:
        pytest.skip("파일럿 스냅샷 없음")
    dst = tmp_path / "snap"
    shutil.copytree(PILOT, dst)
    (dst / "SNAPSHOT.sha256").unlink()
    with pytest.raises(SnapshotVerificationError):
        eval_set.read_manifest(dst)


# --------------------------------------------------------------------------------------
# 2. AST — 우회 경로가 생기는 것을 막는다
# --------------------------------------------------------------------------------------

READER_FUNCS = ("read_manifest", "read_gold")


def _module_ast(mod) -> ast.Module:
    return ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))


def test_readers_call_ensure_verified() -> None:
    """두 리더가 **자기 몸통 안에서** `ensure_verified` 를 부른다."""
    tree = _module_ast(eval_set)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in READER_FUNCS:
            names = {
                n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            found[node.name] = "ensure_verified" in names
    assert set(found) == set(READER_FUNCS), f"리더 함수를 못 찾았다: {found}"
    assert all(found.values()), f"검증을 안 부르는 리더가 있다: {found}"


def test_no_unverified_csv_open_in_scoring_readers() -> None:
    """`ensure_verified` 를 부르지 않는 함수가 스냅샷 CSV 를 여는 일이 없다.

    새 리더가 추가돼도 이 시험이 잡는다 — D1 의 재발 경로가 정확히 그것이다.
    """
    tree = _module_ast(eval_set)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        calls = {
            n.func.attr for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        names = {
            n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if "open" in calls and "ensure_verified" not in names:
            offenders.append(node.name)
    assert not offenders, f"검증 없이 파일을 여는 함수: {offenders}"


# --------------------------------------------------------------------------------------
# 3. 양성 대조 — 통제가 채점을 막지 않는다
# --------------------------------------------------------------------------------------

def test_intact_snapshot_still_reads(has_pilot: bool) -> None:
    if not has_pilot:
        pytest.skip("파일럿 스냅샷 없음")
    rows = eval_set.read_manifest(PILOT)
    assert len(rows) > 0
    assert str(PILOT.resolve()) in eval_set.VERIFIED
