"""B — 미사용 검증 DataLoader 의 워커를 0 으로 만드는 접촉점 6 의 계약 (14번 §B-7 T2~T5, 15번 G3).

**detection extra(ultralytics·torch) 필요, GPU 불필요.** 기본 스위트는 선택 의존성 없이도
완주해야 하므로 없으면 파일 전체를 skip 한다.

무엇을 고정하는가
- T2: `FedDetectionTrainer.get_dataloader(mode="val")` 가 워커 0·단일 프로세스 반복자·`drop_last=False`
      ·`shuffle=False` 로더를 돌려주고 `val_loader_workers == 0` 을 남긴다. 학습 경로는 stock 에
      그대로 위임한다. 상류가 워커를 다시 띄우면 `RuntimeError` 로 드러난다(조용한 회귀 금지).
- T3: 검증 로더 생성이 전역 RNG(python·numpy·torch)와 **학습 로더의 생성기 상태**를 건드리지
      않는다 — B 전후 학습 배치열·워커 시드가 같아야 한다는 등가 논증의 파이썬 가시 부분.
- T4: `check_file_speeds` 가 전역 `random` 을 소비한다는 현상 고정 — 접촉점 6 이 `build_dataset`
      을 stock 그대로 두는 이유(생략하면 주 프로세스 RNG 궤적이 갈라진다).
- T5: 상류 카나리 — 버전·`test_loader` 참조 수·`workers * 2` 규칙·`InfiniteDataLoader` 즉시 반복자.
      하나라도 바뀌면 접촉점 6 의 전제를 다시 봐야 한다.
"""

from __future__ import annotations

import inspect
import random
from types import SimpleNamespace

import pytest

pytest.importorskip("ultralytics", reason="detection extra 미설치 — uv sync --extra detection")
pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402


class _Toy(torch.utils.data.Dataset):
    """인덱스를 그대로 돌려주는 최소 데이터셋. 순서와 워커 수만 관찰한다."""

    def __init__(self, n: int = 640) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


def _dummy_trainer(workers: int = 8):
    """`__init__` 없이 접촉점 6 만 검사할 트레이너 껍데기. CPU 라 `trainer.py:164` 가 workers=0 을
    강제하는 실트레이너 대신 `args.workers=8` 을 그대로 든 스파이를 쓴다."""
    from detection.fed_trainer import FedDetectionTrainer

    t = object.__new__(FedDetectionTrainer)
    t.args = SimpleNamespace(workers=workers, compile=False)
    t.device = torch.device("cpu")
    t.val_loader_workers = None
    t.build_dataset_calls: list = []

    def build_dataset(img_path, mode="train", batch=None):
        t.build_dataset_calls.append((img_path, mode, batch))
        return _Toy(640)

    t.build_dataset = build_dataset
    return t


def _rng_snapshot():
    return (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state().clone())


def _rng_equal(a, b) -> bool:
    return a[0] == b[0] and a[1] == b[1] and bool(torch.equal(a[2], b[2]))


# ---- T2 -----------------------------------------------------------------------------------
def test_T2_검증_로더는_워커0_단일프로세스_비셔플로_만들어진다():
    from torch.utils.data.dataloader import _SingleProcessDataLoaderIter

    from detection.fed_trainer import FedDetectionTrainer

    t = _dummy_trainer()
    loader = FedDetectionTrainer.get_dataloader(t, "views/val", batch_size=64, rank=-1, mode="val")

    assert loader.num_workers == 0
    assert isinstance(loader.iterator, _SingleProcessDataLoaderIter)  # 프로세스·핀 스레드·프리페치 없음
    assert loader.batch_size == 64 and loader.drop_last is False
    assert t.val_loader_workers == 0
    assert t.build_dataset_calls == [("views/val", "val", 64)]  # 데이터셋 생성은 stock 그대로 1회
    first = next(iter(loader))
    assert [int(v) for v in first] == list(range(64))  # shuffle=False
    loader.close()  # 워커가 없으니 무해해야 한다 (trainer.py:646 close 루프)


def test_T2_학습_로더는_stock_경로에_그대로_위임한다(monkeypatch):
    from ultralytics.models.yolo.detect import DetectionTrainer

    from detection.fed_trainer import FedDetectionTrainer

    calls: list = []

    def fake(self, dataset_path, batch_size=16, rank=0, mode="train"):
        calls.append((dataset_path, batch_size, rank, mode))
        return "SENTINEL"

    monkeypatch.setattr(DetectionTrainer, "get_dataloader", fake)
    t = _dummy_trainer()
    out = FedDetectionTrainer.get_dataloader(t, "views/train", batch_size=32, rank=-1, mode="train")
    assert out == "SENTINEL"
    assert calls == [("views/train", 32, -1, "train")]
    assert t.val_loader_workers is None  # 학습 경로는 증빙을 건드리지 않는다


def test_T2_상류가_검증_워커를_되살리면_즉시_실패한다(monkeypatch):
    import ultralytics.data as ud

    from detection.fed_trainer import FedDetectionTrainer

    monkeypatch.setattr(ud, "build_dataloader", lambda *a, **k: SimpleNamespace(num_workers=2))
    t = _dummy_trainer()
    with pytest.raises(RuntimeError, match="B 전제 위반"):
        FedDetectionTrainer.get_dataloader(t, "views/val", batch_size=64, rank=-1, mode="val")
    assert t.val_loader_workers is None


# ---- T3 -----------------------------------------------------------------------------------
def test_T3_검증_로더_생성이_전역_RNG_와_학습_로더_생성기를_건드리지_않는다():
    from ultralytics.data.build import build_dataloader

    from detection.fed_trainer import FedDetectionTrainer

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    train = build_dataloader(_Toy(256), batch=8, workers=0, shuffle=True, device="cpu")
    gen_before = train.generator.get_state().clone()
    snap = _rng_snapshot()

    t = _dummy_trainer()
    loader = FedDetectionTrainer.get_dataloader(t, "views/val", batch_size=64, rank=-1, mode="val")
    _ = next(iter(loader))  # 한 배치를 실제로 만들어도

    assert _rng_equal(snap, _rng_snapshot()), "검증 로더 생성이 전역 RNG 를 소비했다"
    assert torch.equal(gen_before, train.generator.get_state()), "학습 로더 생성기 상태가 바뀌었다"


def test_T3_stock_경로도_전역_RNG_를_소비하지_않는다_현상_고정():
    """B 가 꺼진 경로(워커 spawn + 프리페치)도 전역 RNG 를 안 쓴다 — B 전후 등가의 다른 쪽 절반.
    `_base_seed` 는 로더 자신의 generator 에서 뽑히고(torch), Windows spawn 의 이름·파이프는
    stdlib 사설 Random 만 쓴다(14번 §B-4). 워커 2개 실제 spawn 이라 수 초 걸린다."""
    from ultralytics.data.build import build_dataloader

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    snap = _rng_snapshot()
    loader = build_dataloader(_Toy(256), batch=8, workers=2, shuffle=False, device="cpu")
    try:
        assert loader.num_workers == 2
        _ = next(iter(loader))  # 워커 spawn + 프리페치 완료
        assert _rng_equal(snap, _rng_snapshot())
    finally:
        loader.close()


# ---- T4 -----------------------------------------------------------------------------------
def test_T4_check_file_speeds_는_전역_random_을_소비한다_현상_고정(tmp_path):
    from ultralytics.data.utils import check_file_speeds

    files = []
    for i in range(8):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(b"x" * 1024)
        files.append(str(p))
    random.seed(0)
    before = random.getstate()
    check_file_speeds(files, threshold_ms=1e9, threshold_mb=0.0, max_files=5, prefix="")
    assert random.getstate() != before, (
        "check_file_speeds 가 더 이상 전역 random 을 쓰지 않는다 — 접촉점 6 이 build_dataset 을 "
        "stock 그대로 두는 이유가 바뀌었다(등가 논증 재검토)"
    )


# ---- T5 -----------------------------------------------------------------------------------
def test_T5_상류_카나리():
    import ultralytics
    from ultralytics.data.build import InfiniteDataLoader
    from ultralytics.engine import trainer as tr
    from ultralytics.models.yolo.detect import DetectionTrainer

    assert ultralytics.__version__ == "8.4.120", "Ultralytics 판올림 — 접촉점 6 의 전제(아래 3건)를 다시 본다"
    src = inspect.getsource(tr)
    assert src.count("self.test_loader") == 2, "test_loader 참조가 늘었다 — 검증 로더를 소비하는 새 경로?"
    assert src.count("self.validator(") == 2, "validator 호출처가 늘었다 — no-op 으로 막히지 않는 분기?"
    assert "self.args.workers * 2" in inspect.getsource(DetectionTrainer.get_dataloader)
    assert "self.iterator = super().__iter__()" in inspect.getsource(InfiniteDataLoader.__init__)
