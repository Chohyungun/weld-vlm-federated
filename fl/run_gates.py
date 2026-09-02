"""학습 진입점이 실제로 부르는 게이트 — 80번 체크리스트 18항 (C 몫).

## 왜 이 파일이 필요한가

80번의 한 줄 답이 이것이었다: 반복해서 나온 형태는 "결함이 있는데 시험이 초록"이 아니라
**"검사를 만들어 두고 부르지 않았다"**였다. `deterministic_torch()` 호출처 0건,
`check_cells_identical` 자기 시험 외 호출처 0건 — 함수는 저장소에 있는데 학습이 그것을
지나가지 않았다.

그래서 여기서 파는 것은 새 검사가 아니라 **호출**이다. 그리고 호출했다는 사실을
`gates_evaluated` 로 산출물에 남긴다 — 통과 여부가 아니라 **판정 자체를 했는지**를
산출물이 증명하게 하는 것이 G1-6 의 요점이다.

## 소관

`tracking/mlflow_local.py` 는 트랙 D 소유라 **수정하지 않고 호출만** 한다. 게이트 함수의
정의는 D 가, 학습 경로에서의 호출은 C 가 책임진다.

## cudnn 결정론은 선언이 아니라 실측을 남긴다

`configs/gpu/16gb.yaml` 이 "base.yaml 의 seed 절에서 관리한다(cudnn.deterministic)"고
적는데 그 절에 cudnn 키가 없다 — **존재하지 않는 곳을 가리키는 포인터**였다(80번 D13).
여기서는 설정을 읽지 않고 `torch.backends.cudnn.deterministic` 의 실효값을 그대로 적는다.
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = ["GATE_NAMES", "apply_run_gates", "fingerprint_for_cell"]

#: 이 모듈이 부르는 게이트. 산출물의 `gates_evaluated` 와 대조된다.
GATE_NAMES = ("deterministic_torch", "check_cells_identical")


def fingerprint_for_cell(cell: str, *, coord_cfg: Any = None, base_ckpt: str = "",
                         rag_snapshot: str = "") -> Any:
    """이 칸의 `CellFingerprint`. 다섯 칸이 같은 출발점이었음을 대조하는 단위다."""
    from tracking.mlflow_local import CellFingerprint
    from vlm.coords import coord_cfg_hash, coords_source_sha256

    if coord_cfg is None:
        from vlm.pilot_vlm import COORD_CFG as coord_cfg  # noqa: N806

    return CellFingerprint(
        cell=str(cell),
        base_ckpt_sha256=str(base_ckpt),
        coords_sha256=coords_source_sha256(),
        coord_cfg_hash=coord_cfg_hash(coord_cfg),
        rag_snapshot_sha256=str(rag_snapshot),
    )


def apply_run_gates(
    *,
    cell: str,
    fingerprints: Iterable[Any] | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    """학습 시작 시 게이트를 **실제로 부르고** 무엇을 불렀는지 돌려준다.

    Args:
        fingerprints: 이 run 안에서 같아야 하는 지문들. 연합 칸이면 세 클라이언트가
            여기 들어온다 — 한 라운드 안에서 클라이언트끼리 전처리·좌표 설정이 갈리면
            그 run 의 집계는 서로 다른 전제 위의 평균이다.
        deterministic: cudnn 결정론을 걸지 성능을 살릴지. 다섯 칸 공통 고정 항목이라
            기본값을 끄지 마라.

    Returns:
        `gates_evaluated`·`gate_results`·`cudnn_deterministic`(실측)을 담은 dict.
        호출부가 그대로 산출물 JSON 에 싣는다.
    """
    evaluated: list[str] = []
    results: dict[str, Any] = {}

    from fl.seeding import deterministic_torch

    if deterministic:
        deterministic_torch()
    evaluated.append("deterministic_torch")

    # 선언이 아니라 실효값을 읽는다.
    try:
        import torch

        results["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        results["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except ImportError:  # pragma: no cover - torch 없는 환경
        results["cudnn_deterministic"] = None
        results["cudnn_benchmark"] = None

    prints = list(fingerprints or [])
    if len(prints) >= 2:
        from tracking.mlflow_local import check_cells_identical

        diverged = check_cells_identical(prints)
        evaluated.append("check_cells_identical")
        results["cells_identical"] = not diverged
        if diverged:
            detail = {k: sorted(v) for k, v in diverged.items()}
            results["cells_diverged"] = detail
            raise RuntimeError(
                f"{cell}: 같아야 할 값이 칸/클라이언트마다 다르다 — 집계가 서로 다른 "
                f"전제 위의 평균이 된다: {detail}"
            )
    else:
        # **통과로 적지 않는다 — 지문이 2개 미만이면.** 구판은 `if prints:` 라서 지문
        # 1개짜리 호출(통합형 학습이 정확히 그렇다)이 대조 없이 `cells_identical: true`
        # 를 기록했다(85번 ⑦). 대조는 두 개부터 성립한다. 하나는 자기 자신과 같을
        # 뿐이고, 그것을 통과로 적는 것이 이 게이트가 잡으려던 무이빨 형태다.
        results["cells_identical"] = None
        results["cells_identical_note"] = (
            f"지문 {len(prints)}개 — 대조 미적용(2개부터 성립). "
            "다섯 칸 대조는 채점 단계(D)의 몫이다."
        )

    return {
        "cell": str(cell),
        "gates_evaluated": evaluated,
        "gates_registry": list(GATE_NAMES),
        "gate_results": results,
    }
