"""시험용 이음새의 가드 — 목적에 따라 대역을 받을지 정하고, 실제로 부른 구현을 적는다.

모델 없이 학습 경로를 시험하려면 모델 적재기를 바꿔 끼울 자리가 있어야 한다. 그 자리가 본실험에서
대역을 받으면 대역으로 만든 어댑터가 본실험 산출물이 된다. 그래서 세 가지를 지킨다.

1. **파이썬 인자로만 받는다.** CLI 에는 구현을 고르는 인자가 없다.
2. **승인 판정은 이름이 아니라 객체로 한다.** 승인 목록은 import 시점에 잡아 둔 함수 참조이고,
   실제로 부를 객체가 `is` 로 그 참조와 같아야 한다. 인자 없이 기본값으로 풀린 경우도 같은 검사를
   지난다 — 모듈 속성을 바꿔 끼운 대역도 여기서 걸린다.
3. **부르기 전에 거부한다.** `purpose="main"` 은 대역을 한 번도 부르지 않고 멈춘다.

`frame_diag` 는 호출부가 대역을 허락할 때(로컬 합성 영수증)만 대역을 받는다. 그 판단은 영수증을
읽는 쪽의 몫이라 여기서는 `standin_allowed` 로만 받는다. `rehearsal` 은 대역을 받고 식별자를 적는다.

이 모듈은 torch 를 가져오지 않는다.
"""

from __future__ import annotations

import hashlib
import inspect
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable

from fl.uni_run_config import PURPOSES

__all__ = ["SeamRejected", "check_seam", "impl_id", "REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parents[1]


class SeamRejected(ValueError):
    """목적이 받지 않는 대역이 왔다. **대역을 부르기 전에** 올린다.

    `ValueError` 하위다 — 연합 서버·클라이언트에서 `SystemExit` 로 올리면 SuperLink 의 실패 처리를
    건너뛰고 인프로세스 시뮬레이션이 성공으로 끝난다(`fl/atomic_log.LedgerIdentityMismatch` 와 같은 까닭).
    """

    code = "seam_standin_rejected"


def _git_blob_sha1(data: bytes) -> str:
    """git 이 이 바이트에 매길 blob id. 작업 트리의 바이트로 잰다."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _head_blob_sha1(rel: str) -> str | None:
    """`HEAD:<rel>` 의 blob id. 저장소 밖이거나 git 이 없으면 None."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", f"HEAD:{rel}"],
                             capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and len(sha) == 40 else None


def impl_id(fn: Callable, *, seam: str, approved: bool) -> dict[str, Any]:
    """부른 객체의 구현 식별자. 어댑터 meta · 초기 어댑터 proof · 실행 기록에 싣는다.

    `blob_sha1` 은 **작업 트리 바이트**로 잰 git blob id 이고 `head_blob_sha1` 은 `HEAD` 의 것이다.
    둘이 다르면 커밋되지 않은 코드로 돈 것이다. 판정기가 둘을 보고 가른다.
    """
    # 호출 가능한 인스턴스는 그 타입에서 찾는다 — 함수·메서드·클래스가 아니면 소스 파일을 물을 수 없다.
    target = fn if (inspect.isfunction(fn) or inspect.ismethod(fn) or inspect.isclass(fn)) else type(fn)
    try:
        src = inspect.getsourcefile(target)
    except TypeError:
        src = None
    rel: str | None = None
    blob = head = None
    if src is not None:
        p = Path(src).resolve()
        try:
            rel = p.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = None
        blob = _git_blob_sha1(p.read_bytes())
        head = _head_blob_sha1(rel) if rel is not None else None
    return {
        "seam": seam,
        "approved": bool(approved),
        "module": getattr(target, "__module__", None),
        "qualname": getattr(target, "__qualname__", None),
        "source_path": rel if rel is not None else (str(src) if src else None),
        "in_repo": rel is not None,
        "blob_sha1": blob,
        "head_blob_sha1": head,
    }


def check_seam(fn: Callable, approved_fns: Iterable[Callable], *, seam: str, purpose: str,
               standin_allowed: bool = False) -> dict[str, Any]:
    """`fn` 을 부르기 전에 목적과 맞댄다. 받으면 구현 식별자를, 받지 않으면 `SeamRejected`."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose 가 {list(PURPOSES)} 가운데 하나가 아니다: {purpose!r}")
    approved = any(fn is a for a in approved_fns)
    if not approved:
        if purpose == "main":
            raise SeamRejected(
                f"본실험(purpose=main)은 승인된 실제 구현만 받는다 — {seam} 에 대역이 왔다: "
                f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}")
        if purpose == "frame_diag" and not standin_allowed:
            raise SeamRejected(
                f"프레임 진단은 로컬 합성 영수증일 때만 대역을 받는다 — {seam} 에 대역이 왔다")
    return impl_id(fn, seam=seam, approved=approved)
