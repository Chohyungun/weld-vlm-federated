"""등록 클라이언트 — 칸 디스패처 하나가 세 경로(④·⑦·스모크)를 전부 받는다.

## 왜 하나인가 (85번 ⑤·⑥)

`flwr run` 은 앱당 clientapp 을 **하나** 등록한다. 이전 구조는 셋으로 갈라져 있었다.

- `fl.client_det:app` — pyproject 에 등록된 유일한 앱인데 train 핸들러가 **죽어 있었다.**
  구판 키 이름(round)을 읽는데 서버는 SERVER_ROUND_KEY 값을 보내고, `run_cfg["data-yaml"]` 등
  서버가 보내지 않는 키를 읽고, 서버가 실제로 보내는 키는 전부 무시했다. 리터럴 금지
  시험의 검사 목록에 하필 이 파일만 빠져 있어 아무도 몰랐다.
- `fl.client_vlm:app` — 살아 있는 핸들러였지만 **어디에도 등록돼 있지 않았다.**
- `fl.pilot_sim:client_app` — 실제로 도는 유일한 핸들러였는데 이름이 "pilot" 이라
  본실험 진입점으로 보이지 않았다.

"등록된 것은 죽어 있고, 살아 있는 것은 등록돼 있지 않다" — F1 과 같은 부류의 사각이다.
그래서 핸들러를 **이 파일 하나로** 모은다. pyproject 가 이것을 등록하고,
`pilot_sim`(run_simulation 경로)도 같은 객체를 쓴다. 두 진입점이 같은 클라이언트를
타므로 한쪽에서만 죽는 배선이 원리적으로 없다.

학습 본체는 여기 없다 — 검출은 `fl.client_det.run_client_round`, 통합형은
`fl.client_vlm.run_client_round` 순수 함수가 한다. 이 파일은 자료형 변환과 칸 분기만 한다.

## ⑦ 을 `flwr run` 으로 띄우는 법 (85번 ⑥)

별도 앱이 아니라 **run-config 의 `cell`** 로 고른다:

    flwr run . local-sim --run-config 'cell="uni_fed" ...'

pyproject 의 구판 주석("uni-fed 페더레이션, 클라이언트 앱이 다르다")은 틀렸었다 —
그런 페더레이션은 정의된 적이 없고 클라이언트 앱은 하나다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from flwr.clientapp import ClientApp
    from flwr.common import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "fl extra 가 설치되지 않았다. `uv sync --extra fl` 로 flwr 를 설치해야 한다."
    ) from exc

import numpy as np

from fl.round_wiring import ALL_CELLS, CANONICAL_KEYS_KEY, SERVER_ROUND_KEY, SMOKE_CELL
from fl.strategy import ARRAYS_KEY, CONFIG_KEY, METRICS_KEY, WEIGHT_KEY

__all__ = ["app", "smoke_client_round"]

app = ClientApp()


def _client_idx(context: Context) -> int:
    node_cfg = dict(context.node_config or {})
    if "partition-id" not in node_cfg:
        raise RuntimeError(
            "시뮬레이션 백엔드가 partition-id 를 넣지 않았다. 클라이언트를 식별할 수 없으므로 "
            "추측하지 않고 멈춘다 — 세 클라이언트가 같은 데이터를 돌면 지표는 초록인 채 "
            "연합이 무의미해진다."
        )
    return int(node_cfg["partition-id"])


def smoke_client_round(round_idx: int, client_idx: int, cfg: Any) -> tuple[list, dict, dict]:
    """더미 2텐서. 학습하지 않고 **배선만** 통과시킨다.

    클라이언트마다 값이 조금씩 다르다 — 전부 같으면 집계가 항등이 되어 평균 산술이
    돌았는지 알 수 없다. 같은 이유로 **가중(토큰)과 페어 수도 다른 값**이다. 85번 ① 의
    중복 키 사고에서 스모크가 두 키에 같은 값을 실어 충돌을 원리적으로 못 봤다 —
    값이 다르면 어느 쪽이 전송됐는지 회계에서 구분된다.
    """
    scale = 1.0 + 0.01 * client_idx
    arrays = [
        (np.arange(12, dtype=np.float32).reshape(4, 3) * scale),
        (np.arange(3, dtype=np.float32) * scale),
    ]
    fail_at = str(cfg.get("smoke-fail-at", ""))
    if fail_at == f"{round_idx},{client_idx}":
        raise RuntimeError(
            f"스모크 의도적 실패: r{round_idx} c{client_idx} — 회계가 남는지 본다"
        )

    from detection import serialize

    tokens = 1000 + client_idx          # 가중 (판정 2 의 단위)
    pairs = 100 + client_idx            # 페어 수 — 일부러 가중과 다른 값
    metrics = {
        WEIGHT_KEY: float(tokens),
        "num-examples": float(pairs),
        "supervised-tokens": float(tokens),
        "epochs-ran": float(cfg["local-epochs"]),
        "optimizer-steps": 5.0, "optimizer-updates": 5.0, "resumed-from-epoch": -1.0,
        "param-l2": serialize.params_l2_norm(arrays),
        "payload-bytes": float(serialize.payload_nbytes(arrays)),
        "seed": float(cfg["base-seed"]), "client-idx": float(client_idx),
        "lr": 1e-4, "momentum": float("nan"), "budget-fired-at": -1.0,
        "peak-vram-gb": 0.0, "stopper-true-count": -1.0,
    }
    strings = {
        "keys-digest": serialize.keys_digest(list(cfg[CANONICAL_KEYS_KEY])),
        "optimizer": "AdamW", "arg-optimizer": "AdamW", "stopper-class": "",
        "weight-unit": "supervised_tokens",
    }
    return arrays, metrics, strings


@app.train()
def train(msg: Message, context: Context) -> Message:
    """서버가 보낸 파라미터로 칸에 맞는 로컬 학습을 돌리고 결과를 돌려준다."""
    cfg = msg.content[CONFIG_KEY] if CONFIG_KEY in msg.content else {}
    if CANONICAL_KEYS_KEY not in cfg:
        raise RuntimeError(
            "서버가 정본 키 리스트를 보내지 않았다. ArrayRecord 는 리스트 경로에서 키를 "
            "인덱스 문자열로 바꾸므로 이름은 별도로 전달돼야 한다."
        )
    client_idx = _client_idx(context)
    round_idx = int(cfg[SERVER_ROUND_KEY]) - 1        # flwr 는 1부터 센다
    canonical_keys = list(cfg[CANONICAL_KEYS_KEY])
    arrays_in = msg.content[ARRAYS_KEY].to_numpy_ndarrays()
    cell = str(cfg["cell"])

    if cell == "sep_fed":
        from fl.client_det import run_client_round

        run_cfg = {
            "data_yaml": str(Path(str(cfg["views-root"])) / f"client{client_idx}" / "data.yaml"),
            "model": str(cfg["model"]),
            "total_epochs": int(cfg["total-epochs"]),
            "local_epochs": int(cfg["local-epochs"]),
            "base_seed": int(cfg["base-seed"]),
            "num_examples": int(cfg[f"num-examples-{client_idx}"]),
            "project": str(cfg["project"]),
            # 라운드 안에서 죽어도 그 라운드를 0부터 다시 돌지 않는다. 신원(라운드·클라이언트·
            # 시드)이 다르면 거부되므로 옆 라운드 상태를 물려받는 경로는 없다.
            "resume_root": str(cfg["resume-root"]) if cfg.get("resume-root") else None,
            "run_id": str(cfg.get("run-stamp", "")),
        }
        arrays_out, metrics, strings = run_client_round(
            weights_in=arrays_in, canonical_keys=canonical_keys,
            round_idx=round_idx, client_idx=client_idx, cfg=run_cfg,
            profile=str(cfg.get("profile", "main")),
        )
    elif cell == "uni_fed":
        from fl.client_vlm import run_client_round

        run_cfg = {
            "client_tag": str(cfg[f"client-tag-{client_idx}"]),
            "local_epochs": int(cfg["local-epochs"]),
            "num_rounds": int(cfg["num-rounds"]),
            "base_seed": int(cfg["base-seed"]),
            "resume_root": str(cfg["resume-root"]) if cfg.get("resume-root") else None,
            "run_id": str(cfg.get("run-stamp", "")),
        }
        arrays_out, metrics, strings = run_client_round(
            adapter_in=arrays_in, canonical_keys=canonical_keys,
            round_idx=round_idx, client_idx=client_idx, cfg=run_cfg,
        )
    elif cell == SMOKE_CELL:
        arrays_out, metrics, strings = smoke_client_round(round_idx, client_idx, cfg)
    else:
        raise ValueError(f"연합 칸이 아니다: {cell!r}. 허용: {ALL_CELLS}")

    return Message(
        content=RecordDict({
            ARRAYS_KEY: ArrayRecord(arrays_out),
            METRICS_KEY: MetricRecord(metrics),
            CONFIG_KEY: ConfigRecord(strings),
        }),
        reply_to=msg,
    )
