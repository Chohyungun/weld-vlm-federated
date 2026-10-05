"""통합형 재개 신원 — 검출의 `ResumeIdentity` 를 넓히지 않고 필드를 더한 자기 신원(미니스펙 12번 §4-1).

검출 체크포인트와의 뒤호환이 걸려 있어 `detection.resume.ResumeIdentity` 에 필드를 더하지 않는다 — 더하면
검출이 저장하는 신원 사전이 바뀐다. 저장 · 대조 경로는 그대로 `ResumeCheckpointer` · `latest_resume` 를 쓰고,
되살릴 클래스만 이것으로 넘긴다(`latest_resume(..., identity=UniResumeIdentity(...))`).

| 필드 | 왜 |
|---|---|
| `cell` · `client_tag` | 두 칸이 round 0 · client 0 으로 같다(m-7) |
| `num_rounds` | 학습률 예산이 전역 R 에 걸려 있다 |
| `pairs_digest` | 경로 문자열이 아니라 내용이다 — 페어 순서가 학습 계약이다 |
| `prompt_sha256` | 생성 · 감독 구간이 갈린다 |
| `coord_cfg_hash` · `target_contract_sha256` | 타깃 규약이 바뀌면 다른 실험이다 |
| `input_adapter_digest` | 이 학습이 출발한 어댑터(로컬 · 중앙은 공통 초기 어댑터, 연합은 그 라운드에 받은 글로벌) |
| `micro_batch` · `grad_accum` · `lr0` · `lrf` | 창 경계와 스케줄 |
| `processor_config_sha256` | 해상도(`min/max_pixels`) 같은 프로세서 설정이 바뀌면 입력이 다르다 — 중간 체크포인트를 이어 받지 않는다 |

모두 기본값이 있다 — 필드가 없던 파일은 기본값으로 읽혀 채운 값과 어긋나 거부된다(`mismatch` 가 전 필드를 본다).
"""

from __future__ import annotations

from dataclasses import dataclass

from detection.resume import ResumeIdentity

__all__ = ["UniResumeIdentity"]


@dataclass(frozen=True)
class UniResumeIdentity(ResumeIdentity):
    cell: str = ""
    client_tag: str = ""
    num_rounds: int = 0
    pairs_digest: str = ""
    prompt_sha256: str = ""
    coord_cfg_hash: str = ""
    target_contract_sha256: str = ""
    input_adapter_digest: str = ""
    micro_batch: int = 0
    grad_accum: int = 0
    lr0: float = 0.0
    lrf: float = 0.0
    processor_config_sha256: str = ""
