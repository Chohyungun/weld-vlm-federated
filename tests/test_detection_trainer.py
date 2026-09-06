"""detection/fed_trainer.py 테스트 — **detection extra(ultralytics) 필요**.

기본 테스트 스위트는 선택적 의존성 없이도 완주해야 하므로, ultralytics 가 없으면 이 파일
전체를 skip 한다. 의존성이 없다는 이유로 489건이 collection error 로 멈추면 안 된다.

ultralytics 를 쓰지 않는 검사(직렬화 규약, 회계 매트릭스, 공통 고정 위반)는
`tests/test_detection_fed.py` 에 있고 그쪽은 기본 환경에서 그대로 돈다.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ultralytics", reason="detection extra 미설치 — uv sync --extra detection")

from detection.fed_trainer import NoEarlyStopping, RoundBudget  # noqa: E402


def test_스텁_stopper는_중단시키지_않고_이력을_남긴다():
    s = NoEarlyStopping()
    # 검증 게이트가 참조하는 속성이 없으면 학습 루프가 AttributeError 로 죽는다
    assert s.possible_stop is False
    for e in range(1, 6):
        assert s(e, 0.1) is False
    assert len(s.calls) == 5


def test_스텁_stopper는_fitness가_없어도_동작한다():
    """validate() 를 no-op 으로 만들면 fitness 가 None 으로 들어온다."""
    s = NoEarlyStopping()
    assert s(1, None) is False
    assert s.calls == [(1, None)]


def test_예산_콜백은_E_도달에서만_멈춘다():
    class FakeTrainer:
        def __init__(self, start: int) -> None:
            self.start_epoch = start
            self.epoch = start
            self.stop = False

    budget = RoundBudget(local_epochs=2)
    t = FakeTrainer(start=10)  # 라운드 5, E=2 인 상황

    t.epoch = 10
    budget(t)
    assert t.stop is False and budget.epochs_ran == 1

    t.epoch = 11
    budget(t)
    assert t.stop is True and budget.epochs_ran == 2
    assert budget.fired_at_epoch == 11


def test_트레이너가_접촉점_넷을_실제로_오버라이드한다():
    """stock 동작과 다르다는 것을 클래스 수준에서 확인한다.

    - final_eval: stock 은 best.pt 를 로드해 검증한다
    - save_model: stock 은 EMA 가중치를 저장한다
    - validate:   클라이언트는 검증하지 않는다
    """
    from ultralytics.engine.trainer import BaseTrainer

    from detection.fed_trainer import FedDetectionTrainer

    for name in ("_setup_train", "validate", "final_eval", "save_model"):
        assert name in FedDetectionTrainer.__dict__, f"{name} 오버라이드가 없다"
        assert getattr(FedDetectionTrainer, name) is not getattr(BaseTrainer, name, None)


def test_검증_로더_접촉점을_오버라이드한다():
    """접촉점 6 — 검증 로더만 workers=0 (14번 §B-3, 15번 G3). stock 과 다른 함수여야 한다."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    from detection.fed_trainer import FedDetectionTrainer

    assert "get_dataloader" in FedDetectionTrainer.__dict__, "get_dataloader 오버라이드가 없다"
    assert FedDetectionTrainer.get_dataloader is not DetectionTrainer.get_dataloader


def test_no_op_들이_학습루프가_기대하는_형태를_돌려준다():
    """no-op 이 루프의 계약을 깨면 학습이 죽는다."""
    from detection.fed_trainer import FedDetectionTrainer

    dummy = object.__new__(FedDetectionTrainer)  # __init__ 없이 메서드만 검사
    metrics, fitness = FedDetectionTrainer.validate(dummy)
    assert metrics == {} and fitness is None
    # save_model 은 falsy 를 돌려줘야 on_model_save 콜백이 돌지 않는다
    assert FedDetectionTrainer.save_model(dummy) is False
    assert FedDetectionTrainer.final_eval(dummy) is None
