"""동결 디렉터리 가드 — 이빨 시험. 80번 G11-1·G2.

80번 G2 가 "실패할 수 없는 검사는 없는 것보다 나쁘다"고 적었다. 가드를 붙였으면
**그 가드가 실제로 물 수 있는지**를 시험이 보여야 한다. 그래서 통과 사례만이 아니라
**막아야 할 때 실제로 막는지**를 함께 건다.

실물 동결 디렉터리에 대한 확인도 둘 넣는다 — 격리가 풀려 경쟁 매니페스트가 본 디렉터리로
돌아오면 실패한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.frozen_guard import (
    ATTIC_NAME,
    CONTRACT_NAME,
    FrozenDirectoryError,
    assert_writable,
    is_frozen,
    legacy_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V1 = REPO_ROOT / "data/interim/manifest_v1"

#: 격리 대상. `scripts/audit_frozen_dir.py` 의 목록과 같아야 한다.
QUARANTINE = (
    "manifest_e3.csv",
    "manifest_pre_e3.csv",
    "manifest_pre_histmatch.csv",
    "manifest_pre_mask.csv",
    "split_meta.json",
    "split_meta_e3.json",
)


def test_계약이_없으면_통과한다(tmp_path: Path):
    assert not is_frozen(tmp_path)
    assert_writable(tmp_path)          # 예외 없이 지나가야 한다


def test_계약이_있으면_막는다(tmp_path: Path):
    """이빨 시험 — 이게 통과하지 않으면 가드는 장식이다."""
    (tmp_path / CONTRACT_NAME).write_text("deadbeef  manifest.csv\n", encoding="utf-8")
    assert is_frozen(tmp_path)
    with pytest.raises(FrozenDirectoryError) as e:
        assert_writable(tmp_path, what="시험용 디렉터리")
    msg = str(e.value)
    # 메시지가 "무엇을 해야 하는지"를 말해야 한다. 권한 오류만 던지면 다음 사람이
    # 읽기 전용 속성을 풀고 다시 돌린다 — 그게 막으려는 사고다.
    assert "시험용 디렉터리" in msg
    assert "새 경로" in msg


def test_읽기전용_속성만으로는_막히지_않는다는_전제(tmp_path: Path):
    """가드가 파일 속성이 아니라 **계약 파일의 존재**로 판단하는지.

    속성은 누구나 풀 수 있으므로 그것에 기대면 안 된다. 계약 파일이 없으면 읽기 전용
    파일이 있어도 통과해야 하고, 계약 파일이 있으면 전부 쓰기 가능이어도 막아야 한다.
    """
    (tmp_path / "manifest.csv").write_text("x\n", encoding="utf-8")
    assert_writable(tmp_path)                       # 계약 없음 → 통과
    (tmp_path / CONTRACT_NAME).write_text("x\n", encoding="utf-8")
    with pytest.raises(FrozenDirectoryError):
        assert_writable(tmp_path)                   # 계약 있음 → 차단


def test_legacy_path_는_attic_을_먼저_본다(tmp_path: Path):
    attic = tmp_path / ATTIC_NAME
    attic.mkdir()
    (attic / "a.csv").write_text("attic\n", encoding="utf-8")
    (tmp_path / "a.csv").write_text("loose\n", encoding="utf-8")
    assert legacy_path("a.csv", root=tmp_path).read_text(encoding="utf-8") == "attic\n"


def test_legacy_path_는_본디렉터리_잔존도_찾되_알린다(tmp_path: Path, capsys):
    (tmp_path / "b.csv").write_text("loose\n", encoding="utf-8")
    p = legacy_path("b.csv", root=tmp_path)
    assert p == tmp_path / "b.csv"
    assert "격리" in capsys.readouterr().out


def test_legacy_path_는_없으면_사유를_담아_실패한다(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as e:
        legacy_path("없는파일.csv", root=tmp_path)
    assert "attic/README.md" in str(e.value)


# ---------------------------------------------------------------------------------
# 실물 동결 디렉터리
# ---------------------------------------------------------------------------------


@pytest.mark.skipif(not V1.is_dir(), reason="동결 스냅샷이 워크트리에 없다")
def test_동결_디렉터리가_실제로_잠겨_있다():
    assert is_frozen(V1), f"{CONTRACT_NAME} 이 없다 — 동결이 풀렸다"
    with pytest.raises(FrozenDirectoryError):
        assert_writable(V1)


@pytest.mark.skipif(not V1.is_dir(), reason="동결 스냅샷이 워크트리에 없다")
def test_경쟁_매니페스트가_본_디렉터리로_돌아오지_않았다():
    """격리가 풀리면 여기서 잡힌다. 80번 E16 이 지목한 사고 경로다."""
    loose = [n for n in QUARANTINE if (V1 / n).is_file()]
    assert not loose, (
        f"격리 대상이 동결 디렉터리 본체에 있다: {loose}. "
        "attic/ 으로 되돌려라 — 이름을 잘못 고르면 평가셋 소속이 26,738장 뒤집힌다"
    )


@pytest.mark.skipif(not (V1 / ATTIC_NAME).is_dir(), reason="attic 이 없다")
def test_격리본이_attic_에_전부_있고_읽기전용이다():
    for n in QUARANTINE:
        p = V1 / ATTIC_NAME / n
        assert p.is_file(), f"{n} 이 attic 에 없다"
        assert not (p.stat().st_mode & 0o200), f"{n} 이 쓰기 가능하다 — 읽기 전용으로 잠가라"


@pytest.mark.skipif(not (V1 / ATTIC_NAME / "split_meta_e3.json").is_file(),
                    reason="attic 이 없다")
def test_정본_분할메타는_동결본과_값이_같다():
    """`split_meta_e3.json` 은 폐기본이 아니라 **정본 분할의 유도 원본**이다.

    `data_capabilities.yaml` 이 이 파일에서 나왔으므로 값이 갈리면 유도가 깨진 것이다.
    반대로 `split_meta.json` 은 폐기값(0.6319)이라 달라야 정상이다.
    """
    import yaml

    caps = yaml.safe_load((V1 / "data_capabilities.yaml").read_text(encoding="utf-8"))
    frozen = caps["split_meta"]["dirichlet"]
    e3 = json.loads(legacy_path("split_meta_e3.json").read_text(encoding="utf-8"))["dirichlet"]
    assert e3["seed_used"] == frozen["seed_used"]
    assert e3["attempts"] == frozen["attempts"]
    assert e3["c1_share"] == frozen["c1_share"]

    old = json.loads(legacy_path("split_meta.json").read_text(encoding="utf-8"))["dirichlet"]
    assert old["c1_share"] != frozen["c1_share"], (
        "split_meta.json 이 정본과 같아졌다 — 폐기본이라는 전제가 깨졌으니 "
        "attic/README.md 의 분류를 다시 보라"
    )
