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
    EXPECTED_SEALED,
    SEALED_STATUSES,
    FrozenDirectoryError,
    assert_writable,
    is_frozen,
    legacy_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V1 = REPO_ROOT / "data/interim/manifest_v1"

PILOT3000_KEYS = (
    "data/processed/aihub71761_rt_v1_pilot3000",
    "data/processed/aihub71761_rt_v1_pilot3000_crop_only",
    "data/processed/aihub71761_rt_v1_pilot3000_scale_control",
)

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
# 봉인처 명부 (32번 과제 6) — 형식·어휘·필수 항목·B 명부와의 정합
# ---------------------------------------------------------------------------------


def test_명부_키는_저장소_상대_POSIX_경로다():
    """드라이브 문자·역슬래시·선행 슬래시가 들어오면 규약 2-6 위반이고 대조기의
    `base / rel` 결합이 깨진다."""
    for rel in EXPECTED_SEALED:
        assert ":" not in rel and "\\" not in rel, rel
        assert not rel.startswith("/") and not rel.endswith("/"), rel
        assert ".." not in rel.split("/"), rel


def test_명부_항목은_어휘_안의_상태와_필수_필드를_가진다():
    assert SEALED_STATUSES == ("expected", "lost", "restored")
    for rel, e in EXPECTED_SEALED.items():
        assert e["status"] in SEALED_STATUSES, f"{rel}: {e['status']!r}"
        assert e["owner"] in {"A", "B", "C", "D", "E", "F"}, f"{rel}: owner {e.get('owner')!r}"
        if e["status"] in ("lost", "restored"):
            assert e.get("record"), f"{rel}: {e['status']} 인데 record 가 없다"
        if e["status"] == "restored":
            assert e.get("evidence"), f"{rel}: restored 인데 근거 등급(evidence)이 없다"


def test_본실험_매니페스트_계약이_명부에_있다():
    """F 39번 Important 1 — 단일 진실인데 명부·백업 목록이 몰랐다."""
    e = EXPECTED_SEALED["data/interim/manifest_v1"]
    assert e["status"] == "expected" and e["owner"] == "A"


def test_pilot3000_계열_3개는_A_소유_restored_로_등록돼_있다():
    """09-16 복원 판정 → 00:58 원 경로 복사·해시 4/4×3 확인 뒤 `restored`(32번 §1-4 추기).
    근거 등급이 다르다 — pilot3000 은 64자 digest 전체, 두 팔은 8자 접두 + 구성 일치.
    실물이 없는데 이 상태면 대조기가 missing_contract 로 실패하는 것이 맞다."""
    for k in PILOT3000_KEYS:
        e = EXPECTED_SEALED[k]
        assert e["owner"] == "A" and e["status"] == "restored", k
        assert "32번" in e["record"], k
    assert "64자" in EXPECTED_SEALED[PILOT3000_KEYS[0]]["evidence"]
    for k in PILOT3000_KEYS[1:]:
        assert "8자 접두" in EXPECTED_SEALED[k]["evidence"], k


@pytest.mark.skipif(
    not all((REPO_ROOT / k).is_dir() for k in PILOT3000_KEYS), reason="복원 실물이 이 트리에 없다"
)
def test_restored_자리에_실물과_계약서가_있다():
    """`restored` 는 실물이 돌아온 뒤에만 붙인다는 규칙의 실물 확인."""
    for k in PILOT3000_KEYS:
        assert is_frozen(REPO_ROOT / k), k


def test_expected_항목은_실물이_있으면_계약서를_가진다():
    """명부가 "있어야 한다"고 한 자리에 디렉터리는 있는데 계약서가 없으면 봉인이 풀린 것이다.
    디렉터리 자체가 없는 경우(정션 없는 트리)는 여기서 판단하지 않는다 — 대조기의 몫."""
    for rel, e in EXPECTED_SEALED.items():
        d = REPO_ROOT / rel
        if e["status"] in ("expected", "restored") and d.is_dir():
            assert is_frozen(d), f"{rel}: 디렉터리는 있는데 {CONTRACT_NAME} 가 없다"


def test_B_명부와_어긋나지_않는다():
    """B 가 import 로 전환하기 전까지 두 명부가 공존한다. B 쪽 항목은 전부 여기 있고
    소유가 같아야 하며, 상태는 같거나 **A 소유 항목의 `lost → restored`** 만 다를 수 있다
    (복원은 A 가 자기 명부에서 올리고 B 판은 전환 때 사라진다). 전환 뒤에는 같은 객체라
    자명하게 통과한다 — 그때 이 시험은 지워도 된다(41번 m-1)."""
    from corpus.generate.frozen_out import EXPECTED_SEALED as b_roster

    for rel, e in b_roster.items():
        assert rel in EXPECTED_SEALED, f"B 명부에만 있다: {rel}"
        mine = EXPECTED_SEALED[rel]
        assert mine["owner"] == e["owner"], rel
        same = mine["status"] == e["status"]
        restored_by_a = mine["owner"] == "A" and e["status"] == "lost" and mine["status"] == "restored"
        assert same or restored_by_a, f"{rel}: B {e['status']} vs A {mine['status']}"


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
