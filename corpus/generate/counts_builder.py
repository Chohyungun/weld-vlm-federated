"""counts.json 산출 — 채택량의 **회계** (스펙 §7-5).

n_k 는 manifest 에서 파생할 수 없다 — 3단계 검증 폐기·quarantine·정상 다운샘플을 거치면
실제 학습 표본 수가 manifest 수치와 달라진다. **채택 레코드에서 기계 산출**한다
(손 계산 금지). 스키마 변경은 승인 사항. S4 스냅샷 대상.

소비 규약: `clients[k]` 는 **train+val 을 합친** 회계이고 학습기는 train 만 읽는다. 그러니 이 값을
그대로 집계 가중으로 쓰면 val 이 가중에 섞인다. 통합형의 집계 가중은 감독 토큰 수이고 이 파일에서
나오지 않는다. 분할 축은 PAIRS_META.json 의 counts_by_split 에 있다.
소비처는 로드 시 counts.json 이 가리키는 스냅샷 해시를 대조하고 불일치면 실행을 중단한다 (§8).

`n_generated`(생성량) 는 §7-5 정합 3항 "discarded 합계 = 생성량 − 채택량" 을 **파일만
가지고 재검증**하기 위해 counts.json 본문에 기록한다. 이 값이 없으면 build 시점에만
성립하고, 확정 후 discarded 를 손으로 고쳐도 validate 가 통과한다 (검토 finding #4).
값의 정의는 validation_report(§6-4) 의 `n_generated`(stage0 투입량) 와 같다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "CountsError",
    "DISCARD_STAGES",
    "COUNTS_TOP_KEYS",
    "build_counts",
    "validate_counts",
    "cross_check_counts",
    "write_counts",
]

# 스펙 §7-5 스키마 그대로 (키 추가·삭제는 승인 사항)
DISCARD_STAGES: tuple[str, ...] = (
    "stage0_numeric_lock", "stage1_rule", "stage2_judge", "stage3_expert", "quarantine",
)
COUNTS_TOP_KEYS: tuple[str, ...] = (
    "asset", "version", "clients", "n_generated", "discarded", "aux",
    "limits_snapshot", "manifest_snapshot", "generated_by",
)
CLIENT_KEYS: tuple[str, ...] = ("n_total", "n_defect", "n_normal")


class CountsError(Exception):
    """counts.json 정합 위반 — fail-closed."""


def _sha_prefixed(h: str) -> str:
    return h if h.startswith("sha256:") else f"sha256:{h}"


def _record_fields(rec: Any) -> tuple[str, int]:
    """채택 레코드에서 (image_id, 결함 수). PairGold 또는 dict 를 받는다."""
    if isinstance(rec, Mapping):
        return str(rec["image_id"]), len(rec["defects"])
    return str(rec.image_id), len(rec.defects)


def build_counts(
    adopted: Sequence[Any],
    client_of: Mapping[str, str],
    *,
    n_generated: int,
    discarded: Mapping[str, int],
    limits_sha256: str,
    manifest_sha256: str,
    git_commit: str,
    date: str,
    asset: str = "d4_pairs",
    version: str = "v1",
    aux: Optional[Mapping[str, int]] = None,
    clients: Sequence[str] = ("C1", "C2", "C3"),
) -> dict:
    """최종 채택 레코드에서 counts.json 내용을 기계 산출한다 (§7-5).

    adopted: 3단계 검증 통과·최종 채택 페어 레코드 (PairGold | {"image_id", "defects"}).
    client_of: image_id → 클라이언트 (manifest 계약 #2 상속 — B 는 분할을 재계산하지 않는다).
    n_generated: 생성량 (오버샘플 포함 투입 건수). 본문에 그대로 기록해 validate 가
    파일만으로 정합 3항을 재검산하게 한다.
    정합 검증 (스냅샷 확정 조건): clients 합계 = 채택 행수, n_defect + n_normal = n_total,
    discarded 합계 = 생성량 − 채택량. 위반 시 CountsError.
    """
    per = {c: {"n_total": 0, "n_defect": 0, "n_normal": 0} for c in clients}
    for rec in adopted:
        image_id, n_def = _record_fields(rec)
        client = client_of.get(image_id)
        if client is None:
            raise CountsError(f"client_of 에 없는 image_id: {image_id} — fail-closed")
        if client not in per:
            raise CountsError(f"미정의 클라이언트 {client!r} (image_id={image_id})")
        per[client]["n_total"] += 1
        per[client]["n_defect" if n_def > 0 else "n_normal"] += 1

    unknown = set(discarded) - set(DISCARD_STAGES)
    if unknown:
        raise CountsError(f"미정의 discarded 단계: {sorted(unknown)} (스키마 §7-5 밖)")
    disc = {s: int(discarded.get(s, 0)) for s in DISCARD_STAGES}

    aux_out = {"b_qa_total": 0, "c_reasoning_total": 0}
    for k, v in (aux or {}).items():
        if k not in aux_out:
            raise CountsError(f"미정의 aux 키: {k!r}")
        aux_out[k] = int(v)

    counts = {
        "asset": asset,
        "version": version,
        "clients": per,
        "n_generated": int(n_generated),
        "discarded": disc,
        "aux": aux_out,
        "limits_snapshot": _sha_prefixed(limits_sha256),
        "manifest_snapshot": _sha_prefixed(manifest_sha256),
        "generated_by": {"git_commit": git_commit, "date": date},
    }
    validate_counts(counts, adopted)
    return counts


def _nonneg_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CountsError(f"{what} 정수 아님: {value!r}")
    if value < 0:
        raise CountsError(f"{what} 음수: {value}")
    return value


def validate_counts(counts: Mapping[str, Any], adopted: Sequence[Any]) -> None:
    """정합 검증 (T35 · §7-5 정합 3항 전부).

    ① 스키마 키 전수 ② clients 합계 = 채택 행수 ③ n_defect + n_normal = n_total
    ④ **discarded 합계 = n_generated − 채택량** — 파일에 기록된 생성량으로 재계산하므로
    확정 후 discarded 를 고쳐도 여기서 걸린다 (finding #4)
    ⑤ 채택 레코드의 image_id 유일성 — 같은 이미지가 두 번 실리면 "합계 = 행수" 는 그대로
    성립하면서 n_k 만 부풀고, FedAvg 가중치가 조용히 틀어진다.

    단계 간 재배분(합계 유지)과 생성량·폐기량을 함께 맞춘 위조는 counts.json 만으로는
    닫히지 않는다 — `cross_check_counts` 로 validation_report 와 교차 대조해야 한다.
    """
    if tuple(counts.keys()) != COUNTS_TOP_KEYS:
        raise CountsError(
            f"최상위 키 불일치: {tuple(counts.keys())} ≠ {COUNTS_TOP_KEYS} (스키마 동결)"
        )
    image_ids = [_record_fields(rec)[0] for rec in adopted]
    if len(set(image_ids)) != len(image_ids):
        dups = sorted({i for i in image_ids if image_ids.count(i) > 1})
        raise CountsError(
            f"채택 레코드 image_id 중복: {dups} — 중복분이 n_k 에 합산된다 (fail-closed)"
        )
    total = 0
    for client, cc in counts["clients"].items():
        if tuple(cc.keys()) != CLIENT_KEYS:
            raise CountsError(f"클라이언트 {client} 키 불일치: {tuple(cc.keys())}")
        for key in CLIENT_KEYS:
            _nonneg_int(cc[key], f"{client}.{key}")
        if cc["n_defect"] + cc["n_normal"] != cc["n_total"]:
            raise CountsError(
                f"{client}: n_defect({cc['n_defect']}) + n_normal({cc['n_normal']}) "
                f"≠ n_total({cc['n_total']})"
            )
        total += cc["n_total"]
    if total != len(adopted):
        raise CountsError(f"clients 합계 {total} ≠ 채택 레코드 행수 {len(adopted)}")

    if tuple(counts["discarded"].keys()) != DISCARD_STAGES:
        raise CountsError(f"discarded 키 불일치: {tuple(counts['discarded'].keys())}")
    disc_sum = sum(
        _nonneg_int(v, f"discarded.{s}") for s, v in counts["discarded"].items()
    )
    n_generated = _nonneg_int(counts["n_generated"], "n_generated")
    if disc_sum != n_generated - len(adopted):
        raise CountsError(
            f"discarded 합계 {disc_sum} ≠ 생성량 {n_generated} − 채택량 {len(adopted)} "
            "(§7-5 정합 3항 — 사후 변조 또는 생성량 오기)"
        )


def cross_check_counts(
    counts: Mapping[str, Any], validation_report: Mapping[str, Any]
) -> None:
    """counts.json ↔ validation_report(§6-4) 교차 대조 — 스냅샷 확정 조건.

    §7-5 정합 3항은 합계 검산이라 ① 단계 간 재배분(합계 유지) ② 생성량·폐기량을 함께 맞춘
    자기정합 위조를 잡지 못한다. 단계별 폐기 사유 분포는 논문에 싣는 수치이므로 합계만으로는
    방어되지 않는다. 검증 보고서의 단계별 n_in·n_pass 와 맞춰 본다:

    - `n_generated` = 보고서의 stage0 투입량
    - 단계별 discarded = 보고서 각 단계의 n_fail
    - `quarantine` = 보고서의 quarantine

    보고서에 없는 단계는 대조하지 않는다 (아직 돌지 않은 단계) — 다만 그 단계의 discarded 가
    0 이 아니면 근거 없는 수치이므로 거부한다.
    """
    stages = validation_report.get("stages") or {}
    rep_generated = validation_report.get("n_generated")
    if rep_generated is not None and int(rep_generated) != int(counts["n_generated"]):
        raise CountsError(
            f"n_generated 불일치: counts={counts['n_generated']} ↔ "
            f"validation_report={rep_generated}"
        )
    for stage in DISCARD_STAGES:
        if stage == "quarantine":
            rep_q = validation_report.get("quarantine")
            if rep_q is not None and int(rep_q) != int(counts["discarded"][stage]):
                raise CountsError(
                    f"quarantine 불일치: counts={counts['discarded'][stage]} ↔ "
                    f"validation_report={rep_q}"
                )
            continue
        block = stages.get(stage)
        if block is None:
            if counts["discarded"][stage]:
                raise CountsError(
                    f"{stage}: counts 는 {counts['discarded'][stage]}건 폐기인데 "
                    "validation_report 에 그 단계가 없다 (근거 없는 수치)"
                )
            continue
        n_fail = int(block["n_in"]) - int(block["n_pass"])
        if n_fail != int(counts["discarded"][stage]):
            raise CountsError(
                f"{stage} 폐기 건수 불일치: counts={counts['discarded'][stage]} ↔ "
                f"validation_report n_in−n_pass={n_fail} (단계 간 재배분 의심)"
            )


def write_counts(path: str | Path, counts: Mapping[str, Any]) -> str:
    """counts.json 원자적 기록 + sha256 반환 (S4 스냅샷 해시 재료). 표기 고정.

    바이트로 쓰고 **기록된 파일을 다시 읽어** 해시를 계산한다. `write_text` 는 플랫폼
    개행 변환을 거치므로 Windows 에서 파일에는 CRLF 가 들어가고, 그러면 반환 해시가 실제
    파일 바이트와 달라진다. 그 해시를 SNAPSHOT.sha256·논문에 실으면 §8 의 소비처 해시
    대조가 항상 불일치해 소비처의 n_k 로드가 통째로 중단된다 (검토 critical).
    """
    payload = json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(payload.encode("utf-8"))
    tmp.replace(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()
