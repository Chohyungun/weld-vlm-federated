"""detection/ 교환 규약·회계 매트릭스 테스트 (트랙 C · 지정 함정 구간 #1).

**이 파일은 선택적 의존성 없이 돈다.** torch(기본 의존성)와 표준 라이브러리만 쓴다.
ultralytics 가 필요한 트레이너 검사는 `tests/test_detection_trainer.py` 에 분리했고
그쪽은 detection extra 가 없으면 skip 된다.

GPU·데이터셋 없이 돌 수 있는 계약과 회계 로직을 검사한다. 실제 학습 등가성 검증
(래퍼 R=1×E=100 대 stock 100 epoch, LR 궤적 일치)은 파일럿 스모크에서 따로 확인한다.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from detection import serialize
from detection.budget_audit import AccountingCell, AccountingMatrix
from detection.round_runner import FIXED_OVERRIDES, derive_seed


def _fake_state_dict() -> dict[str, torch.Tensor]:
    """BatchNorm 을 포함한 소형 state_dict. 정수 버퍼가 섞인 상황을 재현한다."""
    return {
        "conv.weight": torch.randn(4, 3, 3, 3),
        "bn.weight": torch.randn(4),
        "bn.bias": torch.randn(4),
        "bn.running_mean": torch.randn(4),
        "bn.running_var": torch.rand(4) + 0.5,
        "bn.num_batches_tracked": torch.tensor(17, dtype=torch.int64),
    }


# --------------------------------------------------------------------------
# 교환 규약
# --------------------------------------------------------------------------

def test_직렬화_왕복이_값을_보존한다():
    sd = _fake_state_dict()
    keys = serialize.canonical_keys(sd)
    arrays = serialize.state_dict_to_ndarrays(sd, keys)
    back = serialize.ndarrays_to_state_dict(arrays, keys, sd)

    assert list(back.keys()) == keys
    for k in keys:
        assert torch.allclose(back[k].float(), sd[k].float()), k
        assert back[k].dtype == sd[k].dtype, f"{k}: dtype 이 복원되지 않았다"


def test_부동소수는_fp32_정수버퍼는_원dtype으로_나간다():
    sd = _fake_state_dict()
    sd["conv.weight"] = sd["conv.weight"].to(torch.float16)  # 반정밀도로 학습된 경우
    keys = serialize.canonical_keys(sd)
    arrays = serialize.state_dict_to_ndarrays(sd, keys)

    by_key = dict(zip(keys, arrays))
    assert by_key["conv.weight"].dtype == np.float32, "부동소수는 fp32 로 승격해 보낸다"
    assert by_key["bn.num_batches_tracked"].dtype == np.int64, "정수 버퍼는 원 dtype 을 지킨다"


def test_키_순서가_바뀌면_해시가_달라진다():
    sd = _fake_state_dict()
    keys = serialize.canonical_keys(sd)
    swapped = [keys[1], keys[0], *keys[2:]]
    assert serialize.keys_digest(keys) != serialize.keys_digest(swapped)


@pytest.mark.parametrize("mode", ["missing_key", "extra_key", "shape", "dtype_family"])
def test_무결성_검사가_어긋남을_잡는다(mode: str):
    sd = _fake_state_dict()
    keys = serialize.canonical_keys(sd)
    arrays = serialize.state_dict_to_ndarrays(sd, keys)

    if mode == "missing_key":
        with pytest.raises(serialize.SerializeError):
            serialize.state_dict_to_ndarrays(sd, [*keys, "없는.키"])
    elif mode == "extra_key":
        with pytest.raises(serialize.SerializeError):
            serialize.state_dict_to_ndarrays(sd, keys[:-1])
    elif mode == "shape":
        bad = list(arrays)
        bad[0] = np.zeros((1, 1, 1, 1), dtype=np.float32)
        with pytest.raises(serialize.SerializeError):
            serialize.assert_compatible(bad, keys, sd)
    else:
        bad = list(arrays)
        idx = keys.index("bn.num_batches_tracked")
        bad[idx] = bad[idx].astype(np.float32)  # 정수 버퍼를 실수로 보내면 잡혀야 한다
        with pytest.raises(serialize.SerializeError):
            serialize.assert_compatible(bad, keys, sd)


def test_norm은_정수버퍼를_제외한다():
    sd = _fake_state_dict()
    keys = serialize.canonical_keys(sd)
    arrays = serialize.state_dict_to_ndarrays(sd, keys)
    manual = float(
        np.sqrt(sum(float(np.sum(a.astype(np.float64) ** 2)) for a in arrays
                    if np.issubdtype(a.dtype, np.floating)))
    )
    assert serialize.params_l2_norm(arrays) == pytest.approx(manual, rel=1e-12)


# --------------------------------------------------------------------------
# 조기 종료 차단
# --------------------------------------------------------------------------

def test_공통고정_항목은_칸별로_덮어쓸_수_없다():
    from detection.round_runner import train_round

    with pytest.raises(ValueError, match="공통 고정"):
        train_round(
            data_yaml="x.yaml", model="yolo11s.pt", total_epochs=100, local_epochs=2,
            extra_overrides={"optimizer": "AdamW"},
        )
    with pytest.raises(ValueError, match="공통 고정"):
        train_round(
            data_yaml="x.yaml", model="yolo11s.pt", total_epochs=100, local_epochs=2,
            extra_overrides={"patience": 5},
        )


def test_고정값이_조기종료와_최적화를_막는다():
    assert FIXED_OVERRIDES["patience"] >= 10000, "patience 는 epoch 수보다 훨씬 커야 한다"
    assert FIXED_OVERRIDES["optimizer"] == "SGD", "'auto' 는 lr0·momentum 을 버린다"
    assert FIXED_OVERRIDES["mlflow"] is False, "로깅 경로는 서버 시점 하나로 통일한다"
    assert FIXED_OVERRIDES["save"] is False and FIXED_OVERRIDES["val"] is False


def test_파생시드는_재현되고_라운드마다_다르다():
    assert derive_seed(7, 3, 1) == derive_seed(7, 3, 1)
    assert derive_seed(7, 3, 1) != derive_seed(7, 4, 1)
    assert derive_seed(7, 3, 1) != derive_seed(7, 3, 2)
    # 라운드와 클라이언트 축이 서로 상쇄되지 않아야 한다
    assert derive_seed(0, 1, 0) != derive_seed(0, 0, 1)


# --------------------------------------------------------------------------
# 회계 매트릭스 — 머지 차단 조건
# --------------------------------------------------------------------------

def _cell(r: int, c: int, *, epochs: int = 2, opt: str = "SGD", arg_opt: str = "SGD") -> AccountingCell:
    return AccountingCell(
        round_idx=r, client_idx=c, epochs_ran=epochs, optimizer_steps=100,
        num_examples=1000, seed=r * 10 + c, optimizer=opt, lr=0.01, momentum=0.937,
        arg_optimizer=arg_opt, arg_lr0=0.01, arg_momentum=0.937,
    )


def _full_matrix(rounds: int = 3, clients=(0, 1, 2), epochs: int = 2) -> AccountingMatrix:
    m = AccountingMatrix(num_rounds=rounds, client_ids=clients, local_epochs=epochs,
                         total_epochs=rounds * epochs)
    for r in range(rounds):
        for c in clients:
            m.record(_cell(r, c, epochs=epochs))
    return m


def test_정상_매트릭스는_통과한다():
    report = _full_matrix().audit()
    assert report.ok, report.failures
    assert report.total_epochs_by_client == {0: 6, 1: 6, 2: 6}
    assert report.total_optimizer_steps == 900


def test_빈_셀은_실패다():
    """실패한 클라이언트는 로그에 아예 없을 수 있다 — 부재 자체를 실패로 정의한다."""
    m = AccountingMatrix(num_rounds=3, client_ids=(0, 1, 2), local_epochs=2, total_epochs=6)
    for r in range(3):
        for c in (0, 1, 2):
            if (r, c) == (1, 2):  # 라운드 1 에서 C3 이탈
                continue
            m.record(_cell(r, c))
    report = m.audit()
    assert not report.ok
    assert any("빈 셀" in f for f in report.failures)


def test_epoch_수가_모자라면_실패다():
    m = AccountingMatrix(num_rounds=2, client_ids=(0,), local_epochs=2, total_epochs=4)
    m.record(_cell(0, 0, epochs=2))
    m.record(_cell(1, 0, epochs=1))
    report = m.audit()
    assert not report.ok
    assert any("epochs_ran=1" in f for f in report.failures)
    assert any("총 epoch 3" in f for f in report.failures)


def test_optimizer_auto는_실패다():
    """설정에 auto 가 남아 있으면 명시한 lr0·momentum 이 버려진다."""
    m = AccountingMatrix(num_rounds=1, client_ids=(0,), local_epochs=2, total_epochs=2)
    m.record(_cell(0, 0, opt="AdamW", arg_opt="auto"))
    report = m.audit()
    assert not report.ok
    assert any("auto" in f for f in report.failures)


def test_실사용_optimizer가_셀마다_다르면_실패다():
    m = AccountingMatrix(num_rounds=2, client_ids=(0,), local_epochs=2, total_epochs=4)
    m.record(_cell(0, 0, opt="SGD"))
    m.record(_cell(1, 0, opt="AdamW", arg_opt="AdamW"))
    report = m.audit()
    assert not report.ok
    assert any("셀마다 다르다" in f for f in report.failures)


def test_설정과_실사용이_어긋나면_실패다():
    m = AccountingMatrix(num_rounds=1, client_ids=(0,), local_epochs=2, total_epochs=2)
    m.record(_cell(0, 0, opt="AdamW", arg_opt="SGD"))
    report = m.audit()
    assert not report.ok
    assert any("실사용" in f for f in report.failures)


def test_같은_셀을_두_번_기록하지_않는다():
    m = AccountingMatrix(num_rounds=1, client_ids=(0,), local_epochs=2, total_epochs=2)
    m.record(_cell(0, 0))
    with pytest.raises(ValueError, match="이미 기록"):
        m.record(_cell(0, 0))


def test_csv에_실사용_최적화_컬럼이_들어간다(tmp_path):
    p = _full_matrix(rounds=1, clients=(0,)).to_csv(tmp_path / "accounting.csv")
    header = p.read_text(encoding="utf-8").splitlines()[0]
    for col in ("optimizer", "lr", "momentum", "arg_optimizer", "arg_lr0", "arg_momentum"):
        assert col in header, f"회계 매트릭스에 {col} 컬럼이 있어야 한다"
    assert "participated" in header and "epochs_ran" in header


def test_json_요약에_감사결과가_담긴다(tmp_path):
    import json

    p = _full_matrix(rounds=2, clients=(0, 1)).to_json(tmp_path / "accounting.json")
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["audit"]["ok"] is True
    assert payload["local_epochs"] == 2 and payload["total_epochs"] == 4


# --------------------------------------------------------------------------
# 파일럿 프로파일 — 값은 다르되 칸 간 동일은 유지한다
# --------------------------------------------------------------------------

def test_파일럿_프로파일이_규칙을_풀지_않는다():
    """축소해도 조기 종료·optimizer·로깅 규칙은 그대로다."""
    from detection.round_runner import FIXED_PILOT

    assert FIXED_PILOT["patience"] >= 10000
    assert FIXED_PILOT["optimizer"] == "SGD"
    assert FIXED_PILOT["mlflow"] is False
    assert FIXED_PILOT["save"] is False and FIXED_PILOT["val"] is False
    assert FIXED_PILOT["deterministic"] is True


def test_파일럿은_크기만_줄인다():
    from detection.round_runner import FIXED_OVERRIDES, FIXED_PILOT

    assert FIXED_PILOT["imgsz"] == 416 < FIXED_OVERRIDES["imgsz"]
    assert FIXED_PILOT["batch"] < FIXED_OVERRIDES["batch"]
    # 줄인 키 외에는 본실험과 같아야 한다
    diff = {k for k in FIXED_OVERRIDES if FIXED_PILOT[k] != FIXED_OVERRIDES[k]}
    assert diff == {"imgsz", "batch", "close_mosaic"}, f"예상 밖 차이: {diff}"


def test_파일럿에서도_칸별_덮어쓰기는_막힌다():
    from detection.round_runner import train_round

    with pytest.raises(ValueError, match="공통 고정"):
        train_round(data_yaml="x.yaml", model="yolo11n.pt", total_epochs=6,
                    local_epochs=2, profile="pilot", extra_overrides={"imgsz": 640})


def test_알수없는_프로파일은_거부된다():
    from detection.round_runner import train_round

    with pytest.raises(ValueError, match="알 수 없는 프로파일"):
        train_round(data_yaml="x.yaml", model="yolo11n.pt", total_epochs=6,
                    local_epochs=2, profile="quick")
