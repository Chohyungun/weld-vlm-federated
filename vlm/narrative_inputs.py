"""학습 전 계약 대조 — 학습 쪽 호출 자리(C 08 2 판 §5-3 · D 07 §17-4 · 평가 쪽 인터페이스 07e §2-5 · §4).

> 학습 전 계약 대조는 두 자리에서 D 의 `narrative_inputs_problems` 를 부른다. 하나는 학습 등록의 검증(`vlm.registration.registration_problems`
> — 착수 고정을 쓸 때와 시작 · 재개마다 `main_start_gate` 가 착수 고정을 다시 낼 때)이고, 다른 하나는 모델을 올리기 전의 공통 사전
> 점검(`vlm.uni_config.main_run_problems` — 로컬 · 중앙의 사전 가드와 연합 서버)이다. 하나라도 문제가 있으면 착수 고정을 쓰지 않고
> 모델을 올리지 않는다.

- **언제 부르나** — 학습 등록에 `narrative_inputs: {version, sha256}` 칸이 있을 때만. 칸이 없는 등록(1 차 등록 바이트)은 지금 그대로다.
  칸이 있는데 평가 쪽 함수가 없거나(평가 쪽 2 단계 전) 예외로 멈추면 **닫힌 쪽**이다 — 맞대지 못한 것을 맞았다고 하지 않는다.
- **넘기는 것** — 학습 등록이 묶은 계약 파일(`configs/narrative_inputs.v<판>.json`)의 바이트와 등록의 sha256, 그리고 이 모듈이 **실제
  바이트를 본 결과**(`VerifiedPairs`): 학습이 읽을 페어 폴더 · 그 스냅샷 계약의 digest · 스냅샷 대조 결과 · 봉인 명부의 상태 ·
  PAIRS_META 바이트 · 계약에 적힌 PAIRS_META 해시 · 설정의 페어 digest · 행별 토큰 표 출처의 페어 digest. 표 · 로더 · 생성 경로의
  바이트는 평가 쪽 함수가 `repo` 에서 다시 잰다.
- **판 6 ↔ 계약 ↔ 학습 PIN**(`v6_problems`) — 내보내기 사전 점검이 본실험 모델 묶음에서 부를 자리의 함수다. 판 6 에 계약 칸
  (`narrative.inputs_contract_sha256`)이 있을 때만 맞댄다. **배선은 판 6 발급 구현(2 차 등록 묶음)과 함께 한다** — 지금 작성기가 짓는
  판 6 은 생성 경로 칸이 한 파일 sha 라 평가 쪽 대조에서 `hash_rule` 로 멈추고(07e §8 의 2), 지금 켜면 판 6 내보내기가 모두 멈춘다.

계약 파일과 학습 등록의 새 칸은 데이터 쪽 소관이다 — 이 모듈은 읽기만 하고 쓰지 않는다(2 차 등록 때 승인 아래 둔다).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["CONTRACT_REL", "ENTRY_KEY", "contract_rel", "entry_problems", "problems", "registration_entry",
           "v6_problems", "verified_pairs"]

#: 학습 등록의 칸 이름과 그 칸의 꼴(`early_coord_check_rules: {version, sha256}` 와 같은 꼴 — 09 §6 Q12).
ENTRY_KEY = "narrative_inputs"
ENTRY_KEYS = ("sha256", "version")
#: 계약 파일의 자리 — 판마다 한 파일. `configs/**/*.json` 은 연합 서버의 앱 묶음에도 든다.
CONTRACT_REL = "configs/narrative_inputs.v{version}.json"
#: 시드 폴더의 착수 고정 — 실행기(`scripts/main_uni.py` 의 `PIN_FILE`)와 같은 이름. 다섯 칸의 어댑터 폴더가 시드 폴더 바로 아래다.
PIN_NAME = "PIN.json"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_PREFIX = f"{ENTRY_KEY}: "


def contract_rel(version: int) -> str:
    return CONTRACT_REL.format(version=version)


def entry_problems(entry: Any) -> list[str]:
    """학습 등록의 칸 꼴 — 정확히 `{version, sha256}`, 판은 양의 정수(참 · 거짓 아님), 해시는 소문자 16진 64 자."""
    if not isinstance(entry, Mapping) or tuple(sorted(entry)) != ENTRY_KEYS:
        return [f"학습 등록의 {ENTRY_KEY} 칸은 {{version, sha256}} 이다 — 받은 것 {entry!r}"]
    out = []
    if type(entry["version"]) is not int or entry["version"] < 1:
        out.append(f"{ENTRY_KEY}.version 은 양의 정수다: {entry['version']!r}")
    if not isinstance(entry["sha256"], str) or not _HEX64.fullmatch(entry["sha256"]):
        out.append(f"{ENTRY_KEY}.sha256 은 소문자 16진 64 자다")
    return out


def registration_entry(registration_path: Path | None = None) -> tuple[Mapping | None, list[str]]:
    """학습 등록 파일의 `narrative_inputs` 칸(없으면 None) — 파일을 읽지 못하면 (None, [사유]). 등록의 나머지 대조는
    `vlm.registration` 이 따로 한다(착수 고정 · 실행기)."""
    from vlm.registration import RegistrationRejected, load_registration

    try:
        reg, _ = load_registration(registration_path)
    except RegistrationRejected as exc:
        return None, list(exc.problems)
    return reg.get(ENTRY_KEY), []


def _snapshot(folder: Path) -> tuple[str | None, dict[str, str]]:
    """페어 폴더의 `SNAPSHOT.sha256` — (적힌 digest, {이름: 적힌 해시}). 없으면 (None, {})."""
    f = folder / "SNAPSHOT.sha256"
    if not f.is_file():
        return None, {}
    digest, members = None, {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("# snapshot_digest"):
            digest = line.split()[-1]
        elif line.strip() and not line.startswith("#"):
            parts = line.split(None, 1)
            if len(parts) == 2:
                members[parts[1].strip()] = parts[0]
    return digest, members


def _rel_posix(p: Path, repo: Path) -> str:
    try:
        return Path(p).relative_to(repo).as_posix() if Path(p).is_absolute() else Path(p).as_posix()
    except ValueError:
        return Path(p).as_posix()


def token_table_pairs_digest(recipe: str, repo: Path) -> str | None:
    """행별 토큰 표 출처의 페어 digest(`vlm.budget.token_table_problems` 가 보는 값). 출처 파일이 없으면 None."""
    from vlm.budget import PROVENANCE_SUFFIX, TOKEN_TABLES

    if recipe not in TOKEN_TABLES:
        return None
    table = Path(repo) / TOKEN_TABLES[recipe][0]
    prov = table.with_name(table.name + PROVENANCE_SUFFIX)
    try:
        rec = json.loads(prov.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    got = rec.get("pairs_digest") if isinstance(rec, Mapping) else None
    return got if isinstance(got, str) else None


def verified_pairs(pairs_path: str | Path, *, repo: Path, config_pairs_digest: str | None, recipe: str):
    """평가 쪽 `VerifiedPairs` — 이 모듈이 실제 바이트를 본 사실을 채운다. 사실을 얻지 못한 칸은 None 이다(평가 쪽이
    `pairs_unverified` 로 거부한다 — 검증을 수행한 뒤의 빈 문제 목록과 다르다)."""
    from data.frozen_guard import EXPECTED_SEALED
    from evaluation.narrative_registration import VerifiedPairs
    from vlm.uni_config import pairs_snapshot_problems

    pp = Path(pairs_path)
    full = pp if pp.is_absolute() else Path(repo) / pp
    folder = full.parent
    rel = _rel_posix(folder, Path(repo))
    digest, members = _snapshot(folder)
    meta = folder / "PAIRS_META.json"
    sealed = EXPECTED_SEALED.get(rel)
    return VerifiedPairs(
        path=rel, snapshot_digest=digest,
        snapshot_problems=tuple(pairs_snapshot_problems(full, config_pairs_digest)),
        sealed_status=(sealed or {}).get("status") if isinstance(sealed, Mapping) else None,
        pairs_meta_raw=meta.read_bytes() if meta.is_file() else None,
        snapshot_pairs_meta_sha256=members.get("PAIRS_META.json"),
        config_pairs_digest=config_pairs_digest,
        token_table_pairs_digest=token_table_pairs_digest(recipe, Path(repo)))


def problems(entry: Any, *, repo: Path, pairs_path: str | Path | None, config_pairs_digest: str | None,
             recipe: str) -> list[str]:
    """학습 등록의 `narrative_inputs` 칸을 평가 쪽 `narrative_inputs_problems` 로 맞댄다. 빈 목록이면 맞다.

    줄은 모두 `narrative_inputs: ` 로 시작한다(평가 쪽 줄은 `<사유 코드>: <설명>` 꼴 그대로 뒤에 붙는다). `repo` 는 학습이 쓸 저장소
    뿌리다 — 계약 파일 · 페어 · 토큰 표를 여기서 읽고, 평가 쪽 함수도 표 · 로더 · 생성 경로를 여기서 다시 잰다."""
    form = entry_problems(entry)
    if form:
        return [_PREFIX + x for x in form]
    try:
        from evaluation.narrative_registration import narrative_inputs_problems
    except ImportError:
        return [_PREFIX + "평가 쪽 대조 함수(evaluation.narrative_registration.narrative_inputs_problems)가 없다 — "
                "학습 전 계약을 맞댈 수 없다(닫힌 쪽)"]
    rel = contract_rel(entry["version"])
    p = Path(repo) / rel
    if not p.is_file():
        return [_PREFIX + f"학습 등록이 묶은 계약 파일이 없다: {rel}"]
    if pairs_path is None:
        return [_PREFIX + "학습이 읽을 페어 경로를 받지 못했다"]
    try:
        facts = verified_pairs(pairs_path, repo=Path(repo), config_pairs_digest=config_pairs_digest, recipe=recipe)
        got = narrative_inputs_problems(p.read_bytes(), registered_sha256=entry["sha256"], repo=Path(repo), pairs=facts)
    except Exception as exc:  # noqa: BLE001 — 맞대지 못한 것은 거부다(닫힌 쪽)
        return [_PREFIX + f"대조가 예외로 멈췄다 — {type(exc).__name__}: {str(exc)[:200]}"]
    return [_PREFIX + str(x) for x in got]


def v6_problems(registration: Any, *, read_at_receipt: Callable[[str], bytes], adapter_dir: Path,
                adapter_meta: Mapping[str, Any], repo: Path | None = None) -> list[str]:
    """판 6 ↔ 계약 ↔ 학습 PIN(평가 쪽 `v6_contract_problems`) — 내보내기 사전 점검의 본실험 모델 묶음이 부른다(판 6 연결 2 판 §4-3).

    판 6 에 계약 칸(`narrative.inputs_contract_sha256`)이 없으면 부르지 않는다(빈 목록). 계약 바이트는 영수증 커밋의 학습 등록이
    묶은 판의 파일에서(`read_at_receipt` — 그 커밋의 blob), 착수 고정은 이 어댑터의 시드 폴더에서, 어댑터 쪽 고정 해시는 어댑터 meta 의
    `start_gate.pin_sha256` 에서 읽는다(학습 당시 값 — 지금 등록 파일을 다시 읽은 값으로 대신하지 않는다). `repo` 를 주면 계약의
    실물 파일 대조(평가 쪽 `contract_code_problems` — 표 raw · 로더 묶음 · 생성 묶음의 LF sha)를 PIN 대조 앞에 잇는다."""
    sha = getattr(getattr(registration, "narrative", None), "inputs_contract_sha256", None)
    if not sha:
        return []
    try:
        from evaluation.narrative_registration import v6_contract_problems
    except ImportError:
        return [_PREFIX + "평가 쪽 대조 함수(evaluation.narrative_registration.v6_contract_problems)가 없다 — "
                "판 6 의 계약을 맞댈 수 없다(닫힌 쪽)"]
    from vlm.registration import registration_from_bytes

    try:
        entry = registration_from_bytes(read_at_receipt("configs/train_registration.yaml")).get(ENTRY_KEY)
    except Exception as exc:  # noqa: BLE001 — 커밋에서 읽지 못한 것도 거부다(닫힌 쪽)
        return [_PREFIX + f"영수증 커밋의 학습 등록을 읽지 못했다 — {type(exc).__name__}: {str(exc)[:200]}"]
    if entry is None:
        return [_PREFIX + "영수증 커밋의 학습 등록에 이 칸이 없다 — 판 6 의 계약을 맞댈 수 없다"]
    form = entry_problems(entry)
    if form:
        return [_PREFIX + x for x in form]
    try:
        contract_raw = read_at_receipt(contract_rel(entry["version"]))
    except Exception as exc:  # noqa: BLE001 — 커밋에서 읽지 못한 것도 거부다(닫힌 쪽)
        return [_PREFIX + f"영수증 커밋에 계약 파일이 없다 — {type(exc).__name__}: {str(exc)[:200]}"]
    if repo is not None:
        try:
            from evaluation.narrative_registration import contract_code_problems
        except ImportError:
            return [_PREFIX + "평가 쪽 대조 함수(evaluation.narrative_registration.contract_code_problems)가 없다 — "
                    "계약의 실물 파일을 맞댈 수 없다(닫힌 쪽)"]
        try:
            code = contract_code_problems(contract_raw, repo=Path(repo))
        except Exception as exc:  # noqa: BLE001 — 맞대지 못한 것은 거부다(닫힌 쪽)
            return [_PREFIX + f"계약 실물 대조가 예외로 멈췄다 — {type(exc).__name__}: {str(exc)[:200]}"]
        if code:
            return [_PREFIX + str(x) for x in code]
    pin = Path(adapter_dir).parent / PIN_NAME
    pins: Sequence[str] = [s for s in [(adapter_meta.get("start_gate") or {}).get("pin_sha256")] if isinstance(s, str)]
    try:
        got = v6_contract_problems(registration, contract_raw, pin_raw=pin.read_bytes() if pin.is_file() else b"",
                                   adapter_pin_sha256=pins)
    except Exception as exc:  # noqa: BLE001 — 맞대지 못한 것은 거부다(닫힌 쪽)
        return [_PREFIX + f"대조가 예외로 멈췄다 — {type(exc).__name__}: {str(exc)[:200]}"]
    return [_PREFIX + str(x) for x in got]
