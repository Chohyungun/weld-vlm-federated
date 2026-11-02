"""1차 체제의 회귀 입력. 현재 운영 설정과 분리하고 보존 바이트 지문을 검증한다."""
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/diag_20261027/base_b16e2306.yaml"
BASE_SHA256 = "052e73116d0ad6d51a54e3350af856b899cc20c7a79fb6727d631dd7d4c2f511"
REG_COMMIT = "a1655a0d14b73243320c197e075e1a648b276d16"
REG_SHA256 = "455b937655281dd115c9e186dfa608dc24fd2117f58ebac366bbce2e2451b5ee"


def historical_base() -> Path:
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA256
    return BASE


@lru_cache(maxsize=1)
def historical_registration() -> bytes:
    raw = subprocess.run(["git", "-c", "safe.directory=" + ROOT.as_posix(), "-C", str(ROOT),
                          "cat-file", "blob", REG_COMMIT + ":configs/train_registration.yaml"],
                         capture_output=True, check=True).stdout
    assert hashlib.sha256(raw).hexdigest() == REG_SHA256
    return raw


def legacy_default(monkeypatch, tmp: Path) -> Path:
    """구판 합성 세계의 기본 등록만 보존 사본으로 지정한다. 판독기와 거부 규칙은 대역으로 바꾸지 않는다."""
    from vlm import registration as R

    p = tmp / "first_train_registration.yaml"
    p.write_bytes(historical_registration())
    monkeypatch.setattr(R, "REG_PATH", p)
    return p
