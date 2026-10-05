"""본채점 진입점의 **진입 전 관문** — 묶음·정답·이미지를 열기 전에 끝나는 검사. 진입점 미니스펙 3판 1-1-가 ·
1-1-나 · 3-5 · 07번 §32-1 · §32-3 · §32-6.

여기서 걸리면 종료 3 이다(`EntryRejected`). 이 모듈이 읽는 것은 영수증 · 등록 파일과 설정 파일의 **커밋의 바이트** ·
스냅샷 파일의 해시(`verify_snapshot`) · `data_capabilities.yaml` · 매니페스트의 네 열(열 판독기)뿐이다.
묶음 · 주석 · 이미지는 열지 않는다.

**작업 트리의 등록과 설정을 읽지 않는다.** 엄격 · 진단은 영수증 커밋의 바이트를, 리허설은 로컬 등록 파일과
`refs/heads/main` 의 설정을 읽는다. "영수증이 가리키는 등록" 과 "지금 쓰는 등록" 이 같은 것이어야 한다.

**기준 저장소는 인자가 아니다.** 본체 체크아웃은 이 모듈의 자리에서 `git rev-parse --git-common-dir` 로 찾는다.
시험이 합성 저장소를 줄 수는 있다(`repo=`) — 그러나 **커밋된 영수증**(종류가 `main` · `frame_diag` 이고 커밋이
본체 `refs/heads/main` 의 조상)으로는 그 이음새를 쓸 수 없다(`guard_seam`). 그 조상 검사는 이음새를 거치지 않는다.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import yaml

from data.manifest_io import (CAPABILITIES_FILENAME, SnapshotVerificationError, read_capabilities,
                              verify_snapshot)
from data.manifest_view import SPLITS, read_manifest_columns
from evaluation.prereg_unified import (SPLIT_OF_PURPOSE, Receipt, ReceiptMissing, RegistrationFileError,
                                       UnifiedRegistration, load_registration, require_file_hash)

MAIN_REF = "refs/heads/main"
"""'본줄기의 조상' 의 기준. 본체 체크아웃의 ref 다 — 워크트리는 같은 ref 를 공유한다. 원격 추적 ref 는 쓰지 않는다."""
BASE_CONFIG = "configs/base.yaml"
LEDGER_REL = "outputs/main_u/scoring_ledger.jsonl"
"""정본 채점 원장의 자리 — 본체 체크아웃 아래. 명령줄 인자로 받지 않는다(§32-1)."""
STRICT_OUT_REL = "outputs/main_u"
"""엄격 경로의 출력 루트는 이것뿐이다(3판 가-4′)."""
COMMITTED_KINDS = ("main", "frame_diag")
"""영수증을 본줄기에 커밋하는 종류. 리허설 영수증은 로컬이다."""
MODES = ("strict", "probe")
PROBE_PURPOSES = ("rehearsal", "frame_diag")
"""`--registration probe` 가 받는 목적. `main` 은 엄격에서만 받는다(판정 09 의 I-8 — 가)."""
META_COLUMNS = ("image_id", "split", "sha256", "group_id")
"""진입점이 매니페스트에서 읽는 열 — **모든 행에서 이 넷뿐이다**(§32-4 의 1). 층을 쓸 자리가 없다."""


class EntryRejected(RuntimeError):
    """진입 전 거부 — 종료 3. `code` 가 어느 검사인지 말한다. 값(해시·경로의 내용)은 싣지 않는다."""

    def __init__(self, code: str, reason: str):
        self.code = code
        self.reason = reason
        super().__init__(f"[{code}] {reason}")


# ── 저장소 ──────────────────────────────────────────────────────────────────

def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, check=False)


def main_checkout(start: Path | None = None) -> Path:
    """본체 체크아웃의 작업 트리 — `start`(주지 않으면 이 모듈의 자리)에서 공통 git 디렉터리를 찾아 그 부모."""
    here = Path(start) if start is not None else Path(__file__).resolve().parent
    res = _git(here, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if res.returncode != 0:
        raise EntryRejected("REPO_NOT_FOUND", "git 저장소를 찾지 못했다")
    common = Path(res.stdout.decode("utf-8").strip())
    if common.name != ".git":
        raise EntryRejected("REPO_NOT_FOUND", "공통 git 디렉터리가 작업 트리 안의 .git 이 아니다")
    return common.parent


def is_ancestor(repo: Path, commit: str, ref: str = MAIN_REF) -> bool:
    """`commit` 이 `ref` 의 조상인가. 저장소에 없는 커밋은 조상이 아니다."""
    res = _git(repo, "merge-base", "--is-ancestor", commit, ref)
    if res.returncode == 0:
        return True
    if res.returncode == 1:
        return False
    exists = _git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0
    if not exists:
        return False
    raise EntryRejected("GIT_FAILED", f"조상 검사를 하지 못했다 — {ref} 를 읽지 못했을 수 있다")


def blob_at(repo: Path, rev: str, path: str) -> bytes:
    """`rev` 의 `path` 파일 바이트. 작업 트리가 아니라 커밋에서 읽는다 — 줄끝 변환이 끼지 않는 원시 blob 이다."""
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or "\\" in path:
        raise EntryRejected("PATH_INVALID", "저장소 상대 POSIX 경로가 아니다")
    res = _git(repo, "cat-file", "blob", f"{rev}:{path}")
    if res.returncode != 0:
        raise EntryRejected("GIT_SHOW_FAILED", f"{path} 를 그 커밋에서 읽지 못했다")
    return res.stdout


def guard_seam(receipt: Receipt) -> None:
    """시험의 이음새(원장 · 출력 부모 · 저장소)를 **커밋된 영수증**과 함께 쓰는 길을 막는다(§32-1 의 1).

    조상 검사는 이음새가 준 저장소가 아니라 **이 모듈이 찾은 본체 저장소**로 한다. 커밋된 영수증으로 새 빈 원장을
    주면 판정 한 번 규칙도 쓰는 쪽의 시도 수도 걸리지 않는다.
    """
    if receipt.kind in COMMITTED_KINDS and is_ancestor(main_checkout(), receipt.main_commit):
        raise EntryRejected("SEAM_ON_COMMITTED_RECEIPT",
                            "커밋된 영수증으로는 원장 · 출력 · 저장소의 이음새를 쓸 수 없다 — 정본 경로만 쓴다")


def canonical_ledger(checkout: Path) -> Path:
    return Path(checkout) / LEDGER_REL


# ── 모드 · 목적 · 영수증 ───────────────────────────────────────────────────

def check_mode(mode: str, purpose: str) -> None:
    """모드와 목적의 조합. `main` 은 엄격에서만, 엄격은 `main` 만 받는다(§32-6)."""
    if mode not in MODES:
        raise EntryRejected("MODE_INVALID", f"모드는 {list(MODES)} 가운데 하나다")
    if mode == "strict" and purpose != "main":
        raise EntryRejected("MODE_PURPOSE", "엄격 경로는 목적 main 만 받는다")
    if mode == "probe" and purpose not in PROBE_PURPOSES:
        raise EntryRejected("PROBE_MAIN" if purpose == "main" else "MODE_PURPOSE",
                            f"탐색 채점은 목적 {list(PROBE_PURPOSES)} 만 받는다 — 본실험 묶음의 값을 보는 문을 열지 않는다")


def check_receipt_kind(mode: str, purpose: str, receipt: Receipt) -> None:
    """영수증의 종류가 경로의 목적과 같은가. 탐색 채점에 본실험 영수증이 오면 여기서 끝난다."""
    if mode == "probe" and receipt.kind == "main":
        raise EntryRejected("PROBE_MAIN", "탐색 채점에 본실험 영수증이 왔다")
    if receipt.kind != purpose:
        raise EntryRejected("RECEIPT_KIND", f"영수증의 종류가 경로의 목적 {purpose!r} 가 아니다")


def load_registration_for(receipt: Receipt, *, repo: Path | None = None) -> UnifiedRegistration:
    """영수증이 가리키는 **등록 파일의 바이트**로 등록 객체를 만든다(3판 1-1-가 의 2 · 3).

    엄격 · 진단: 커밋이 `refs/heads/main` 의 조상이어야 하고, `git cat-file blob <main_commit>:<registration_path>`.
    리허설: 로컬 파일(본체 체크아웃 아래의 `registration_path`). 어느 쪽이든 바이트의 sha256 이 영수증의
    `registration_file_sha256` 과 같을 때만 만든다. 등록의 목적은 여기서 보지 않는다 — `require_receipt` 가 본다.
    """
    root = Path(repo) if repo is not None else main_checkout()
    if receipt.kind in COMMITTED_KINDS:
        if not is_ancestor(root, receipt.main_commit):
            raise EntryRejected("RECEIPT_NOT_ANCESTOR", "영수증의 커밋이 본줄기(refs/heads/main)의 조상이 아니다")
        raw = blob_at(root, receipt.main_commit, receipt.registration_path)
    else:
        path = root / PurePosixPath(receipt.registration_path)
        if not path.is_file():
            raise EntryRejected("REGISTRATION_FILE_MISSING", "리허설 등록 파일이 없다")
        raw = path.read_bytes()
    try:
        require_file_hash(raw, receipt)
    except ReceiptMissing as exc:
        raise EntryRejected("REGISTRATION_FILE_HASH", str(exc)) from None
    try:
        return load_registration(raw)
    except RegistrationFileError as exc:
        raise EntryRejected("REGISTRATION_FILE_FORM", str(exc)) from None


# ── 동결본의 닻 ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Anchor:
    """승인된 동결본 — `configs/base.yaml` 의 `fixed_before_main_runs`."""

    snapshot_id: str
    snapshot_digest: str


def read_anchor(raw: bytes) -> Anchor:
    """설정 파일의 바이트에서 닻을 읽는다. 없거나 꼴이 틀리면 거부한다."""
    try:
        cfg = yaml.safe_load(raw.decode("utf-8"))
        block = cfg["fixed_before_main_runs"]
        sid, dig = block["snapshot_id"], block["snapshot_digest"]
    except (UnicodeDecodeError, yaml.YAMLError, KeyError, TypeError):
        raise EntryRejected("ANCHOR_MISSING", "configs/base.yaml 에 fixed_before_main_runs 의 닻이 없다") from None
    if not (isinstance(sid, str) and sid and isinstance(dig, str) and len(dig) == 64
            and all(c in "0123456789abcdef" for c in dig)):
        raise EntryRejected("ANCHOR_MISSING", "닻의 snapshot_id · snapshot_digest 꼴이 틀렸다")
    return Anchor(sid, dig)


def anchor_rev(receipt: Receipt) -> str:
    """닻을 읽을 자리 — 엄격 · 진단은 영수증 커밋, 리허설은 `refs/heads/main`(§32-3 의 3)."""
    return receipt.main_commit if receipt.kind in COMMITTED_KINDS else MAIN_REF


def load_anchor(receipt: Receipt, *, repo: Path | None = None) -> Anchor:
    root = Path(repo) if repo is not None else main_checkout()
    return read_anchor(blob_at(root, anchor_rev(receipt), BASE_CONFIG))


PROCESSOR_KWARG_FIELDS: dict[str, str] = {
    "min_pixels": "processor_min_pixels", "max_pixels": "processor_max_pixels",
    "patch_size": "patch_size", "merge_size": "merge_size",
}
"""`uni_processor_kwargs` 의 키 → 등록의 생성 쪽 칸. 여기 없는 키는 등록에 맞댈 칸이 없어 거부한다."""


def read_processor_kwargs(raw: bytes) -> dict:
    """설정 파일의 바이트에서 학습 쪽 프로세서 인자(`fixed_before_main_runs.uni_processor_kwargs`)를 읽는다.

    키는 데이터 쪽이 더한다(지시 20261001h 의 3). 없으면 등록의 프로세서 값을 맞댈 근거가 없어 거부한다.
    """
    try:
        kwargs = yaml.safe_load(raw.decode("utf-8"))["fixed_before_main_runs"]["uni_processor_kwargs"]
    except (UnicodeDecodeError, yaml.YAMLError, KeyError, TypeError):
        raise EntryRejected("PROCESSOR_KWARGS_MISSING",
                            "configs/base.yaml 에 fixed_before_main_runs.uni_processor_kwargs 가 없다") from None
    if not isinstance(kwargs, dict):
        raise EntryRejected("PROCESSOR_KWARGS_MISSING", "uni_processor_kwargs 가 사전이 아니다")
    return kwargs


def load_processor_kwargs(receipt: Receipt, *, repo: Path | None = None) -> dict:
    """닻과 같은 자리(엄격 · 진단은 영수증 커밋, 리허설은 `refs/heads/main`)의 설정에서 읽는다."""
    root = Path(repo) if repo is not None else main_checkout()
    return read_processor_kwargs(blob_at(root, anchor_rev(receipt), BASE_CONFIG))


def check_processor(registration: UnifiedRegistration, kwargs: dict) -> None:
    """등록의 프로세서 값이 설정의 프로세서 인자와 같은가 — **설정에 적힌 키만** 맞댄다. 자료형까지 본다.

    설정에 없는 키(프로세서 기본값)는 여기서 볼 수 없다 — 프로세서 전체는 `processor_config_sha256` 이 곁 파일과 맞댄다.
    **빈 사전은 받지 않는다** — 맞댈 것이 없으면 이 검사가 공회전한다(검수 16번 M-3). 설정 값이 `null` 이면 등록도
    비어 있어도 다르다고 본다 — 비어 있는 것끼리 같다고 지나가지 않게.
    """
    if not kwargs:
        raise EntryRejected("PROCESSOR_KWARGS_EMPTY", "uni_processor_kwargs 가 비었다 — 맞댈 프로세서 인자가 없다")
    unknown = sorted(set(kwargs) - set(PROCESSOR_KWARG_FIELDS))
    if unknown:
        raise EntryRejected("PROCESSOR_KWARG_UNKNOWN", f"등록에 맞댈 칸이 없는 프로세서 인자다: {unknown}")
    bad = [k for k, v in kwargs.items()
           if (got := getattr(registration.generation, PROCESSOR_KWARG_FIELDS[k])) is None
           or type(got) is not type(v) or got != v]
    if bad:
        raise EntryRejected("PROCESSOR_MISMATCH",
                            f"등록의 프로세서 값이 configs/base.yaml 의 uni_processor_kwargs 와 다르다: {sorted(bad)}")


def check_anchor(registration: UnifiedRegistration, anchor: Anchor) -> None:
    """등록의 `snapshot_digest` 가 닻과 같은가 — 등록 객체를 만든 **바로 다음**에 본다(§32-3 의 2)."""
    if registration.generation.snapshot_digest != anchor.snapshot_digest:
        raise EntryRejected("SNAPSHOT_NOT_ANCHOR",
                            "등록의 snapshot_digest 가 configs/base.yaml 의 닻과 다르다 — 승인된 동결본이 아니다")


def check_snapshot(root: Path, anchor: Anchor) -> str | None:
    """스냅샷 폴더를 검증하고 그 digest 가 닻과 같은지 본다. 흡수 상태(`absorption.status`)를 돌려준다.

    폴더는 인자로 받는다 — 무엇을 줬든 `verify_snapshot` 의 digest 가 닻과 같아야 쓴다. 이 검사는 파일의
    **바이트를 해시한다**(파싱이 아니다). 흡수 블록이 없으면 `None` 이다.
    """
    try:
        digest = verify_snapshot(root)
    except SnapshotVerificationError as exc:
        raise EntryRejected("SNAPSHOT_UNVERIFIED", str(exc)) from None
    if digest != anchor.snapshot_digest:
        raise EntryRejected("SNAPSHOT_NOT_ANCHOR", "스냅샷의 digest 가 닻과 다르다")
    caps = read_capabilities(Path(root) / CAPABILITIES_FILENAME) or {}
    absorption = caps.get("absorption") if isinstance(caps, dict) else None
    return absorption.get("status") if isinstance(absorption, dict) else None


# ── 기준 집합의 허용 ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ManifestMeta:
    """매니페스트의 네 열 — `image_id` → 분할 · 이미지 sha256 · 묶음. 정답의 요약 열은 없다."""

    split_of: Mapping[str, str]
    sha256_of: Mapping[str, str]
    group_of: Mapping[str, str]


def read_manifest_meta(root: Path) -> ManifestMeta:
    """열 판독기로 **모든 분할의 행에서 네 열만** 읽는다. 판독기 안에서 검증이 먼저 돈다."""
    df = read_manifest_columns(root, META_COLUMNS, split_filter=SPLITS)
    ids = [str(x) for x in df["image_id"]]

    def col(name: str) -> Mapping[str, str]:
        return MappingProxyType(dict(zip(ids, (str(x) for x in df[name]))))

    return ManifestMeta(split_of=col("split"), sha256_of=col("sha256"), group_of=col("group_id"))


def check_list_split(ids: Iterable[str], meta: ManifestMeta, purpose: str, *, where: str,
                     echo: bool = False) -> None:
    """목록의 모든 id 가 **검증한 동결 매니페스트에서 그 목적의 분할**인가(3판 1-1-나 · 판정 05 의 2절 가).

    에코 목록은 목적과 상관없이 `val` 이다(§31-3). 매니페스트에 없는 id 도 거부한다.
    """
    want = "val" if echo else SPLIT_OF_PURPOSE[purpose]
    ids = list(ids)
    unknown = sum(1 for i in ids if i not in meta.split_of)
    wrong = sum(1 for i in ids if i in meta.split_of and meta.split_of[i] != want)
    if unknown or wrong:
        raise EntryRejected("LIST_SPLIT", f"{where}: 기준 분할 {want!r} 밖의 id — 매니페스트에 없음 {unknown}개 · "
                                          f"다른 분할 {wrong}개")
