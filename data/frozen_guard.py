"""동결 디렉터리 쓰기 가드 + 격리 파일 경로 해석. 80번 G11-1.

## 왜 있나

`data/interim/manifest_v1/` 을 만든 파이프라인 스크립트 셋(`run_dedup_v1` ·
`run_hist_match` · `run_border_mask`)은 **`manifest.csv` 를 직접 덮어쓴다.** 동결 전에
쓰인 코드라 스스로 "SNAPSHOT 은 잠그지 않았다"고 출력한다. 지금은 잠겨 있으므로
재실행 한 번이 동결본을 지운다 — 읽기 전용 속성에 걸려 멈추더라도 `PermissionError` 는
무슨 일이 벌어진 건지 말해 주지 않고, 누군가 속성을 풀면 조용히 성공한다.

그래서 **파일 속성이 아니라 계약의 존재로** 막는다. `SNAPSHOT.sha256` 이 있는 디렉터리는
완결된 스냅샷이고, 재파생은 항상 새 경로에 한다(개발규약 1-1·1-6).

## 봉인처 명부

`EXPECTED_SEALED` — 있어야 할 봉인 디렉터리의 단일 명부(2026-09-16, 32번 과제 6). 대조기와
백업 목록이 여기서 읽는다. 상태 어휘와 갱신 규칙은 명부 위 주석.

## 쓰는 법

    from data.frozen_guard import assert_writable, legacy_path

    assert_writable(V1)                       # 동결됐으면 FrozenDirectoryError
    p = legacy_path("manifest_pre_mask.csv")  # attic/ 을 먼저 본다
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "ATTIC_NAME",
    "CONTRACT_NAME",
    "EXPECTED_SEALED",
    "SEALED_STATUSES",
    "FrozenDirectoryError",
    "assert_writable",
    "is_frozen",
    "legacy_path",
]

#: 스냅샷 계약 파일. 이게 있으면 그 디렉터리는 완결된 동결본이다.
CONTRACT_NAME = "SNAPSHOT.sha256"

#: 격리 하위 디렉터리 이름.
ATTIC_NAME = "attic"

_V1 = Path(__file__).resolve().parents[1] / "data/interim/manifest_v1"

# ======================================================================================
# 봉인처 명부 — 저장소 전체의 **단일 명부** (2026-09-16, 32번 과제 6)
#
# B 의 `corpus/generate/frozen_out.EXPECTED_SEALED` 에서 옮겨 왔다. 계약서가 있는 디렉터리만
# 찾으면 디렉터리째 사라진 봉인본은 목록에서 같이 사라진다 — 09-11 에 data/processed 가 비었을 때
# 대조기가 "깨짐 0" 을 냈다(30번 §8-4 나). 그래서 "무엇이 있어야 하는지"를 따로 들고 대조한다.
# 두 곳에 살면 한쪽만 고쳐진다 — 09-13 에 백업 목록이 소실된 v2 를 계속 들고 있었다. 대조기
# (`frozen_out.check_registry`)·백업 목록(`verify_backup.SEALED_DIRS`)은 여기서 import 한다.
#
# 키: 저장소 루트 기준 **POSIX 상대경로**. 드라이브 문자·역슬래시·선행 슬래시 금지(규약 2-6).
# 항목: `status`(아래 어휘) · `owner`(트랙 문자) · `record`(lost·restored 는 필수 — 언제 왜 그런지)
#       · `evidence`(restored 만 — 근거 등급).
#
# 상태 어휘 (`SEALED_STATUSES`):
#   expected — 있어야 한다. 대조기: 없으면 missing_contract(실패) 또는 absent_tree(알림 — 저장 뿌리
#              자체가 이 트리에 없을 때, 정션을 안 붙인 새 clone 등).
#   lost     — 소실이 기록됐다. 없으면 lost_recorded(알림, 실패 아님). 다시 나타나면
#              lost_but_present(실패 — 같은 이름 재생성은 규약 1-6 위반).
#   restored — 소실 뒤 **동일 바이트**로 복원됐다(재생성이 아니다). 검사는 expected 와 같고
#              `evidence` 에 근거 등급을 적는다("64자 digest 전체 일치" / "8자 접두 + 구성 일치" 등).
#              원 경로에 실물이 돌아온 **뒤에만** 붙인다 — 그 전에 붙이면 대조기가 missing_contract
#              로 실패한다. 대조기의 restored 분기는 B 소관(import 전환 때 expected 와 같게).
#
# 상태 갱신은 총괄 판정을 따른다. pilot3000 계열 3개는 09-16 복원 판정(의사결정로그 17ca38b) →
# 09-16 00:58 원 경로 복사·파일별 해시 4/4×3·계약서 3장 동일 확인(32번 §1-4 추기) → `restored`.
# 이 세 자리는 git 밖 단일 사본으로 돌아온 것이라 백업 목록(B `verify_backup`)에 들어야 한다.
# ======================================================================================
SEALED_STATUSES: tuple[str, ...] = ("expected", "lost", "restored")

EXPECTED_SEALED: dict[str, dict[str, str]] = {
    # --- 본실험 매니페스트 계약 (A). 본실험 데이터의 단일 진실. git 밖(.gitignore data/interim/).
    #     F 39번 Important 1: 명부와 백업 목록이 이것을 몰랐다.
    "data/interim/manifest_v1": {
        "status": "expected", "owner": "A",
        "record": "본실험 매니페스트 계약 4/4(manifest·annotations·data_capabilities·tiles) · "
                  "digest 1f80e98b… · 동결 08-31(58번) · 위생 정리 09-02(80번 G11-1)"},
    # --- 코퍼스 봉인 (B)
    "corpus/generate/cycle_pilot": {"status": "expected", "owner": "B"},
    "corpus/generate/cycle_pilot_v2": {"status": "expected", "owner": "B"},
    "data/processed/pairs_pilot_v1": {
        "status": "expected", "owner": "B",
        "record": "09-11 소실 → 09-13 해시 일치 복원 (30번 부록 A)"},
    "data/processed/pairs_pilot_v2": {
        "status": "lost", "owner": "B",
        "record": "09-11 소실 확정 · 사본 없음 · 입력 부재로 재생성 불가 (30번 부록 A-3)"},
    # --- 파일럿 표본과 어블레이션 두 팔 (A). 새 경로 재생성 digest 가 기록값과 일치해 복원(32번 §1).
    #     `evidence` 는 근거 등급 — 기록이 64자 전체인지 8자 접두뿐인지가 다르다. 재생성이 아니라
    #     동일 바이트 복원이므로 60·61·76번과 채점 산출물이 인용하는 digest 가 그대로 유효하다.
    "data/processed/aihub71761_rt_v1_pilot3000": {
        "status": "restored", "owner": "A",
        "evidence": "60번 64자 digest 전체 일치 · 같은 생성기(ed8977f)·입력(manifest_v1 4/4) 2회 실행 동일",
        "record": "09-11 소실 · 09-16 복원 판정(17ca38b) · 09-16 00:58 원 경로 복사, "
                  "파일별 해시 4/4·계약서 동일·load_snapshot 통과 (32번 §1-4 추기)"},
    "data/processed/aihub71761_rt_v1_pilot3000_crop_only": {
        "status": "restored", "owner": "A",
        "evidence": "76번 8자 접두 일치 + 구성 수치 전량 일치(2,319장·학습 풀 1,666·공유 eval 653·출처 분포)",
        "record": "09-11 소실 · 09-16 복원 판정(17ca38b) · 09-16 00:58 원 경로 복사, "
                  "파일별 해시 4/4·계약서 동일·load_snapshot 통과 (32번 §1-4 추기)"},
    "data/processed/aihub71761_rt_v1_pilot3000_scale_control": {
        "status": "restored", "owner": "A",
        "evidence": "76번 8자 접두 일치 + 구성 수치 전량 일치(2,319장·학습 풀 1,666·공유 eval 653·출처 분포)",
        "record": "09-11 소실 · 09-16 복원 판정(17ca38b) · 09-16 00:58 원 경로 복사, "
                  "파일별 해시 4/4·계약서 동일·load_snapshot 통과 (32번 §1-4 추기)"},
}


class FrozenDirectoryError(RuntimeError):
    """동결된 스냅샷 디렉터리에 쓰려고 했다."""


def is_frozen(directory: Path) -> bool:
    return (Path(directory) / CONTRACT_NAME).is_file()


def assert_writable(directory: Path, *, what: str = "이 디렉터리") -> None:
    """동결 디렉터리면 멈춘다. 파이프라인 스크립트의 진입에서 부른다.

    메시지에 **무엇을 해야 하는지**를 적는다. "권한 없음"만 던지면 다음 사람이
    읽기 전용 속성을 풀고 다시 돌린다 — 그게 정확히 막으려는 사고다.
    """
    d = Path(directory)
    if not is_frozen(d):
        return
    raise FrozenDirectoryError(
        f"{what}({d})는 동결된 스냅샷이다 — {CONTRACT_NAME} 가 있다.\n"
        "  이 스크립트는 manifest.csv 를 덮어쓰므로 재실행하면 동결본이 사라진다.\n"
        "  재파생이 필요하면 --outdir 로 **새 경로**를 주고, 새 스냅샷으로 잠근 뒤\n"
        "  어느 쪽이 정본인지 docs/의사결정로그.md 에 남겨라.\n"
        "  근거: 개발규약 1-1·1-6, 80번 G11-1. 설명은 data/interim/manifest_v1/README.md."
    )


def legacy_path(name: str, root: Path | None = None) -> Path:
    """격리된 옛 매니페스트·분할 메타의 경로. `attic/` 을 먼저 본다.

    본 디렉터리에서 발견되면 격리가 풀린 것이므로 알린다 — 조용히 쓰면 위생이 되돌아간
    것을 아무도 모른다. 파일이 아예 없으면 사유를 담아 `FileNotFoundError`.
    """
    base = Path(root) if root is not None else _V1
    attic = base / ATTIC_NAME / name
    if attic.is_file():
        return attic
    loose = base / name
    if loose.is_file():
        print(f"  ! {name} 이 격리 디렉터리가 아니라 동결 디렉터리 본체에 있다. "
              f"attic/ 으로 되돌려라 (80번 G11-1)")
        return loose
    raise FileNotFoundError(
        f"{name} 을 {attic} 에서도 {loose} 에서도 찾지 못했다. "
        "격리 목록과 사유는 data/interim/manifest_v1/attic/README.md."
    )
