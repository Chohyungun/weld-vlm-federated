"""본실험 규모 결함 3건 — 82번 §10-3-1 (C 가 §4-6 게이트에서 실측, D 몫).

| # | 결함 | 여기서 고정하는 것 |
|---|---|---|
| 1 | `load_yolo_from_npz` 가 `yolo11n.yaml` 하드코딩 | `model_cfg` 인자. 규모가 틀리면 어느 프로파일인지 말하며 실패 |
| 2 | `predict_cell` 청킹 부재 (12,461장 → 36.7GB 단일 할당) | 내부 청킹. 호출 크기 ≤ 청크, 순서 보존 |
| 3 | `IMGSZ = 416` 고정 | 프로파일에서 받는다. C 의 `PROFILES` 와 같은 값 |

셋 다 파일럿 규모(653장·YOLO11n·416)에서는 보이지 않던 결함이라, 시험도 파일럿 값이
아니라 **본실험 값(YOLO11s·640·12,461장)** 으로 건다.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from evaluation.params import (
    CLASS_NAMES,
    DEFAULT_PROFILE,
    DETECTION_MODEL_BY_PROFILE,
    PREDICT_CHUNK,
    ScoringParams,
    add_common_args,
    detection_profiles,
    params_from_args,
)

REPO = Path(__file__).resolve().parents[1]
MAIN_CKPT = REPO / "outputs" / "gate_c" / "reduced_pilot" / "weights" / "last.npz"


def _params(**kw) -> ScoringParams:
    return ScoringParams(snapshot=Path("s"), pilot=Path("p"), out=Path("o"), **kw)


# --------------------------------------------------------------------------------------
# 결함 3 — 해상도는 프로파일에서
# --------------------------------------------------------------------------------------

def test_profile_imgsz_is_read_from_track_c_profiles() -> None:
    """D 의 프로파일 표가 C 의 `PROFILES` 와 같은 해상도를 낸다 — 베낀 값이 아니라 읽은 값."""
    from detection.round_runner import PROFILES

    profiles = detection_profiles()
    assert set(profiles) == set(DETECTION_MODEL_BY_PROFILE)
    for name, prof in profiles.items():
        assert prof["imgsz"] == PROFILES[name]["imgsz"], f"{name}: 해상도가 C 와 갈린다"


def test_main_profile_is_yolo11s_at_640() -> None:
    p = _params(profile="main")
    assert p.imgsz == 640
    assert p.model_cfg == "yolo11s.yaml"
    assert p.imgsz_source.startswith("profile:main")


def test_pilot_profile_is_yolo11n_at_416() -> None:
    p = _params(profile="pilot")
    assert p.imgsz == 416
    assert p.model_cfg == "yolo11n.yaml"


def test_default_profile_is_pilot_for_existing_reproductions() -> None:
    """`outputs/pilot_c` 의 다섯 칸이 전부 파일럿 프로파일이라 기존 명령이 그대로 돌아야 한다."""
    assert DEFAULT_PROFILE == "pilot"
    assert _params().model_cfg == "yolo11n.yaml"


def test_explicit_imgsz_is_kept_and_declared() -> None:
    """C 의 게이트 스크립트가 쓰는 형태(`ScoringParams(..., imgsz=640)`)가 계속 돌고,
    산출물이 그 값이 프로파일이 아니라 호출부에서 왔음을 밝힌다."""
    p = _params(imgsz=640)
    assert p.imgsz == 640
    assert p.imgsz_source.startswith("explicit:640")
    assert p.as_dict()["imgsz_source"] == p.imgsz_source


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="프로파일"):
        _params(profile="huge")


def test_as_dict_carries_profile_and_chunk() -> None:
    d = _params(profile="main").as_dict()
    assert d["profile"] == "main"
    assert d["model_cfg"] == "yolo11s.yaml"
    assert d["imgsz"] == 640
    assert d["predict_chunk"] == PREDICT_CHUNK


def test_cli_profile_flag_reaches_params() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(["--profile", "main"])
    p = params_from_args(args)
    assert p.profile == "main" and p.imgsz == 640 and p.model_cfg == "yolo11s.yaml"
    assert params_from_args(ap.parse_args([])).profile == DEFAULT_PROFILE
    with pytest.raises(SystemExit):
        ap.parse_args(["--profile", "huge"])


# --------------------------------------------------------------------------------------
# 결함 2 — 청킹
# --------------------------------------------------------------------------------------

class _Result:
    names: ClassVar[dict[int, str]] = {0: "crack"}
    boxes: ClassVar[list] = []


class _FakeYolo:
    """`predict` 가 받은 `source` 의 크기를 기록한다. 추론은 하지 않는다."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def predict(self, source, **kw):
        self.calls.append(list(source))
        for _ in source:
            yield _Result()


def _rows(n: int) -> list[dict]:
    return [{"image_id": f"aihub71761:{i:08d}", "rel_path": f"img/{i}.png"} for i in range(n)]


def test_predict_cell_chunks_and_preserves_order() -> None:
    from evaluation.detect_infer import predict_cell

    yolo = _FakeYolo()
    rows = _rows(10)
    recs = predict_cell(yolo, rows, Path("."), "sep_central", None,
                        _params(predict_chunk=3), conf=0.25)
    assert [len(c) for c in yolo.calls] == [3, 3, 3, 1]
    assert [r.image_id for r in recs] == [r["image_id"] for r in rows]
    assert {r.coord_space for r in recs} == {"ABS_ORIG"}


def test_predict_cell_never_hands_ultralytics_the_whole_eval_set() -> None:
    """**본실험 크기 그대로.** 12,461장을 주면 기본 청크로 49번 나눠 부르고, 어느 호출도
    256장을 넘지 않는다 — 36.7GB 단일 할당이 나던 경로다."""
    from evaluation.detect_infer import predict_cell

    n_eval = 12_461
    yolo = _FakeYolo()
    recs = predict_cell(yolo, _rows(n_eval), Path("."), "sep_central", None,
                        _params(profile="main"), conf=0.25)
    assert len(recs) == n_eval
    assert len(yolo.calls) == math.ceil(n_eval / PREDICT_CHUNK) == 49
    assert max(len(c) for c in yolo.calls) <= PREDICT_CHUNK
    assert sum(len(c) for c in yolo.calls) == n_eval


def test_predict_cell_empty_rows_makes_no_call() -> None:
    from evaluation.detect_infer import predict_cell

    yolo = _FakeYolo()
    assert predict_cell(yolo, [], Path("."), "sep_central", None, _params(), conf=0.25) == []
    assert yolo.calls == []


def test_chunk_must_be_positive() -> None:
    with pytest.raises(ValueError, match="predict_chunk"):
        _params(predict_chunk=0)


# --------------------------------------------------------------------------------------
# 결함 1 — 모델 규모는 인자
# --------------------------------------------------------------------------------------

def _npz_from(model_cfg: str, tmp_path: Path) -> Path:
    """`model_cfg` 규모의 초기 상태를 C 의 serialize 규약으로 npz 에 쓴다."""
    from ultralytics.nn.tasks import DetectionModel

    from detection import serialize

    dm = DetectionModel(cfg=model_cfg, nc=len(CLASS_NAMES), verbose=False)
    sd = dm.state_dict()
    keys = serialize.canonical_keys(sd)
    arrays = serialize.state_dict_to_ndarrays(sd, keys)
    p = tmp_path / f"{Path(model_cfg).stem}.npz"
    np.savez(p, *arrays)
    return p


def test_load_respects_model_cfg_and_names_the_mismatch(tmp_path: Path) -> None:
    from evaluation.detect_infer import ProfileMismatchError, load_yolo_from_npz

    npz_s = _npz_from("yolo11s.yaml", tmp_path)
    yolo = load_yolo_from_npz(npz_s, CLASS_NAMES, 640, model_cfg="yolo11s.yaml")
    assert yolo is not None

    # 이전 판의 하드코딩 그대로 — 본실험 가중치를 파일럿 규모에 넣으면 죽어야 하고,
    # 죽을 때 무엇이 틀렸는지를 말해야 한다.
    with pytest.raises(ProfileMismatchError) as ei:
        load_yolo_from_npz(npz_s, CLASS_NAMES, 416, model_cfg="yolo11n.yaml")
    msg = str(ei.value)
    assert "yolo11n.yaml" in msg and "yolo11s" in msg and "--profile" in msg


def test_loader_leaves_no_temp_file_beside_the_checkpoint(tmp_path: Path) -> None:
    """임시 `.pt` 를 npz 옆에 두지 않는다 — 그곳은 C 의 산출 폴더다."""
    from evaluation.detect_infer import load_yolo_from_npz

    npz = _npz_from("yolo11n.yaml", tmp_path)
    before = set(tmp_path.iterdir())
    load_yolo_from_npz(npz, CLASS_NAMES, 416, model_cfg="yolo11n.yaml")
    assert set(tmp_path.iterdir()) == before


@pytest.mark.skipif(not MAIN_CKPT.exists(), reason="C 의 §4-6 게이트 가중치(YOLO11s)가 없다")
def test_real_main_profile_checkpoint_loads_with_main_profile() -> None:
    """C 가 실제로 죽었던 입력 — 본실험 프로파일로 학습한 `last.npz`."""
    from evaluation.detect_infer import ProfileMismatchError, load_yolo_from_npz

    main = _params(profile="main")
    assert load_yolo_from_npz(MAIN_CKPT, main.class_names, main.imgsz,
                              model_cfg=main.model_cfg) is not None
    pilot = _params(profile="pilot")
    with pytest.raises(ProfileMismatchError, match="yolo11s"):
        load_yolo_from_npz(MAIN_CKPT, pilot.class_names, pilot.imgsz,
                           model_cfg=pilot.model_cfg)
