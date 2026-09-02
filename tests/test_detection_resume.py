"""재개 전용 체크포인트 시험.

`detection/resume.py` 는 ultralytics 를 import 하지 않는다 — 트레이너에 붙지만 저장·복원
자체는 순수 torch 다. 그래서 detection extra 없이도 이 시험이 돈다. 트레이너 결합
(`RoundBudget` 의 재개분 합산)만 ultralytics 를 요구하므로 그 부분만 따로 가드한다.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from detection import serialize
from detection.resume import (
    FORMAT,
    ResumeCheckpointer,
    ResumeIdentity,
    apply_resume,
    clear_resume,
    latest_resume,
    loader_generator_state,
    restore_loader_generator,
)


def _identity(**over) -> ResumeIdentity:
    base = dict(run_id="run-a", round_idx=2, client_idx=1, seed=20115,
                total_epochs=100, local_epochs=2,
                model="yolo11s.pt", data="/x/data.yaml")
    base.update(over)
    return ResumeIdentity(**base)


class _Net(torch.nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = torch.nn.Linear(4, 3)
        with torch.no_grad():
            self.lin.weight.copy_(torch.rand(3, 4, generator=g))
            self.lin.bias.copy_(torch.rand(3, generator=g))
        # BatchNorm 을 넣어 정수 버퍼(num_batches_tracked, 0차원)까지 왕복시킨다
        self.bn = torch.nn.BatchNorm1d(3)


class _Counter:
    def __init__(self, n: int = 0) -> None:
        self.n = n


class _FakeTrainer:
    """`save`/`apply_resume` 가 만지는 속성만 가진 최소 트레이너."""

    def __init__(self, net: _Net, epoch: int, start_epoch: int) -> None:
        self.model = net
        self.optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.937)
        self.epoch = epoch
        self.start_epoch = start_epoch
        self.scaler = None
        self.train_loader = None
        self.device = torch.device("cpu")

    def step_once(self) -> None:
        """momentum 버퍼가 실제로 차게 한다 — 비어 있으면 복원 시험이 무의미하다."""
        self.model.train()
        out = self.model.bn(self.model.lin(torch.ones(2, 4)))
        out.sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


# -- 저장·복원 -------------------------------------------------------------


def test_roundtrip_restores_weights_optimizer_and_counters(tmp_path):
    net = _Net(seed=1)
    tr = _FakeTrainer(net, epoch=5, start_epoch=4)
    tr.step_once()
    ck = ResumeCheckpointer(tmp_path, identity=_identity(), step_counter=_Counter(37),
                            resumed_epochs=0, resumed_steps=0)
    path = ck.save(tr)
    assert path.exists()

    state = latest_resume(tmp_path, identity=_identity())
    assert state is not None
    assert state.epoch_done == 5
    assert state.next_epoch == 6
    assert state.epochs_ran_in_round == 2      # epoch 5 - start 4 + 1
    assert state.optimizer_steps == 37
    assert state.payload["format"] == FORMAT

    # 다른 인스턴스에 복원 — 가중치와 momentum 버퍼가 모두 되살아나야 한다
    fresh = _Net(seed=99)
    tr2 = _FakeTrainer(fresh, epoch=0, start_epoch=0)
    apply_resume(tr2, state)
    assert torch.equal(fresh.lin.weight, net.lin.weight)
    assert torch.equal(fresh.bn.num_batches_tracked, net.bn.num_batches_tracked)
    buf_src = tr.optimizer.state[net.lin.weight]["momentum_buffer"]
    buf_dst = tr2.optimizer.state[fresh.lin.weight]["momentum_buffer"]
    assert torch.allclose(buf_src, buf_dst)
    assert tr2.start_epoch == 6


def test_zero_dim_integer_buffer_survives(tmp_path):
    """`num_batches_tracked` 는 0차원 int64 다. 모양이 (1,) 로 바뀌면 복원이 깨진다."""
    net = _Net(seed=2)
    net.bn.num_batches_tracked.fill_(11)
    tr = _FakeTrainer(net, epoch=0, start_epoch=0)
    ck = ResumeCheckpointer(tmp_path, identity=_identity())
    ck.save(tr)
    state = latest_resume(tmp_path)
    idx = state.payload["canonical_keys"].index("bn.num_batches_tracked")
    arr = state.payload["weights"][idx]
    assert arr.shape == ()
    assert arr.dtype == np.int64
    assert int(arr) == 11


def test_rng_is_restored(tmp_path):
    import random

    net = _Net(seed=3)
    tr = _FakeTrainer(net, epoch=0, start_epoch=0)
    random.seed(1234)
    ck = ResumeCheckpointer(tmp_path, identity=_identity())
    ck.save(tr)
    expected = [random.random() for _ in range(3)]

    random.seed(999)                       # 상태를 헝클어뜨린다
    state = latest_resume(tmp_path)
    apply_resume(_FakeTrainer(_Net(seed=4), 0, 0), state)
    assert [random.random() for _ in range(3)] == expected


# -- 신원 대조 -------------------------------------------------------------


@pytest.mark.parametrize("field,value", [
    ("run_id", "run-b"), ("round_idx", 3), ("client_idx", 0), ("seed", 1),
    ("total_epochs", 50), ("local_epochs", 1), ("model", "yolo11n.pt"),
    ("data", "/y/data.yaml"),
])
def test_identity_mismatch_refuses(tmp_path, field, value):
    tr = _FakeTrainer(_Net(), epoch=0, start_epoch=0)
    ResumeCheckpointer(tmp_path, identity=_identity()).save(tr)
    with pytest.raises(ValueError, match="신원"):
        latest_resume(tmp_path, identity=_identity(**{field: value}))


def test_identity_match_accepts(tmp_path):
    tr = _FakeTrainer(_Net(), epoch=0, start_epoch=0)
    ResumeCheckpointer(tmp_path, identity=_identity()).save(tr)
    assert latest_resume(tmp_path, identity=_identity()) is not None


# -- 파일 관리 -------------------------------------------------------------


def test_torn_write_stays_in_tmp_and_latest_is_readable(tmp_path):
    """`.tmp` 잔해는 재개 후보로 잡히지 않는다."""
    tr = _FakeTrainer(_Net(), epoch=7, start_epoch=6)
    ResumeCheckpointer(tmp_path, identity=_identity()).save(tr)
    (tmp_path / "resume_ep0009.tmp").write_bytes(b"\x00\x01broken")
    state = latest_resume(tmp_path)
    assert state.epoch_done == 7


def test_corrupt_latest_falls_back_to_previous(tmp_path):
    ck = ResumeCheckpointer(tmp_path, identity=_identity(), keep=3)
    for ep in (4, 5):
        ck.save(_FakeTrainer(_Net(seed=ep), epoch=ep, start_epoch=4))
    (tmp_path / "resume_ep0005.pt").write_bytes(b"garbage")
    state = latest_resume(tmp_path)
    assert state.epoch_done == 4


def test_all_unreadable_returns_none(tmp_path):
    (tmp_path / "resume_ep0001.pt").write_bytes(b"garbage")
    assert latest_resume(tmp_path) is None


def test_missing_dir_returns_none(tmp_path):
    assert latest_resume(tmp_path / "nope") is None


def test_keep_prunes_oldest(tmp_path):
    ck = ResumeCheckpointer(tmp_path, identity=_identity(), keep=2)
    for ep in range(5):
        ck.save(_FakeTrainer(_Net(), epoch=ep, start_epoch=0))
    left = sorted(p.name for p in tmp_path.glob("resume_ep*.pt"))
    assert left == ["resume_ep0003.pt", "resume_ep0004.pt"]


def test_clear_resume_removes_everything(tmp_path):
    ck = ResumeCheckpointer(tmp_path, identity=_identity(), keep=3)
    for ep in range(3):
        ck.save(_FakeTrainer(_Net(), epoch=ep, start_epoch=0))
    (tmp_path / "resume_ep0009.tmp").write_bytes(b"x")
    assert clear_resume(tmp_path) == 4
    assert latest_resume(tmp_path) is None


def test_save_failure_does_not_kill_training(tmp_path):
    """체크포인트가 못 써져도 학습은 계속돼야 한다 — 재개는 보험이지 전제가 아니다."""
    ck = ResumeCheckpointer(tmp_path, identity=_identity())
    ck.canonical_keys = ["없는키"]          # state_dict 에 없는 키 → 저장이 던진다
    ck(_FakeTrainer(_Net(), epoch=0, start_epoch=0))   # 콜백 경로는 삼킨다
    assert ck.last_error is not None
    assert ck.n_saves == 0


# -- 누적 회계 -------------------------------------------------------------


def test_resumed_epochs_and_steps_accumulate(tmp_path):
    """재개한 프로세스의 체크포인트는 이전 프로세스 몫을 이어서 센다."""
    ck = ResumeCheckpointer(tmp_path, identity=_identity(), step_counter=_Counter(10),
                            resumed_epochs=1, resumed_steps=100)
    ck.save(_FakeTrainer(_Net(), epoch=5, start_epoch=5))   # 이번 프로세스에서 1 epoch
    state = latest_resume(tmp_path)
    assert state.epochs_ran_in_round == 2      # 이전 1 + 이번 1
    assert state.optimizer_steps == 110        # 이전 100 + 이번 10


# -- 로더 셔플 상태 --------------------------------------------------------


def test_loader_generator_restore_does_not_replay_used_orders():
    """재개한 런이 **이미 쓴 순열을 반복하지 않는다** — 이것이 복원이 보장하는 전부다.

    무중단 런과 같은 순열까지는 보장하지 않는다(`restore_loader_generator` 문서 참조).
    여기서는 실제 `InfiniteDataLoader` 로 그 경계를 못 박는다.
    """
    pytest.importorskip("ultralytics")
    from ultralytics.data.build import build_dataloader

    class _Toy(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 64

        def __getitem__(self, i: int) -> int:
            return i

    def order(loader) -> list[int]:
        return [int(v) for b in loader for v in b]

    loader = build_dataloader(_Toy(), batch=8, workers=0, shuffle=True, device="cpu")

    class T:
        train_loader = loader

    ep0 = order(loader)
    saved = loader_generator_state(T())
    ep1 = order(loader)

    assert restore_loader_generator(T(), saved) is True
    after = order(T.train_loader)
    assert after != ep0, "복원 후에도 epoch 0 순열을 반복하면 복원이 의미가 없다"
    # 무중단 런과 같아지지는 않는다 — base seed 를 한 번 더 뽑기 때문이다.
    # 이 단언이 깨지면 상류가 바뀐 것이고, 그때 재개의 재현성 주장을 올릴 수 있다.
    assert after != ep1


def test_loader_generator_absent_is_reported():
    class T:
        train_loader = None

    assert loader_generator_state(T()) is None
    assert restore_loader_generator(T(), None) is False


# -- 트레이너 결합 (ultralytics 필요) ---------------------------------------


def test_round_budget_counts_resumed_epochs():
    pytest.importorskip("ultralytics")
    from detection.fed_trainer import RoundBudget

    class T:
        epoch = 5
        start_epoch = 5
        stop = False

    # E=2 라운드에서 1 epoch 을 이미 돌고 죽었다 → 이번 프로세스는 1 epoch 만 더 돌아야 한다
    b = RoundBudget(2, resumed_epochs=1)
    t = T()
    b(t)
    assert b.epochs_ran == 2
    assert t.stop is True
    assert b.fired_at_epoch == 5


def test_loader_reseed_is_a_pure_function_of_seed_and_epoch():
    """재시드가 켜지면 로더 상태가 **이력이 아니라 (시드, epoch)** 의 함수가 된다.

    이것이 재개를 정확하게 만드는 근거다. 되돌리기가 아니라 다시 계산하기라서 프리페치나
    base seed draw 같은 내부 사정에 걸리지 않는다.
    """
    pytest.importorskip("ultralytics")
    from detection.fed_trainer import LoaderReseed

    a, b = LoaderReseed(20115), LoaderReseed(20115)
    assert [a.seed_for(e) for e in range(5)] == [b.seed_for(e) for e in range(5)]
    # 시드가 다르면 순열도 달라야 한다 — 숨은 기본값 #9 가 고쳐지는 지점이다
    assert LoaderReseed(0).seed_for(3) != LoaderReseed(1).seed_for(3)
    # epoch 이 1 늘 때 시드가 멀리 움직인다. 인접 입력이 인접 시드가 되면 생성기의
    # 확산에 기대게 되고, 그 기대가 어긋나도 지표에는 흔적이 남지 않는다.
    for e in range(6):
        assert abs(a.seed_for(e) - a.seed_for(e + 1)) > 2**40
    assert all(0 <= a.seed_for(e) < 2**63 - 1 for e in (0, 1, 99, 10_000))
    # 시드가 인접해도 마찬가지다
    assert abs(LoaderReseed(100).seed_for(0) - LoaderReseed(101).seed_for(0)) > 2**40


def test_loader_reseed_changes_order_and_is_reproducible():
    pytest.importorskip("ultralytics")
    from detection.fed_trainer import LoaderReseed
    from ultralytics.data.build import build_dataloader

    class _Toy(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 64

        def __getitem__(self, i: int) -> int:
            return i

    def order_after_reseed(seed: int, epoch: int) -> list[int]:
        loader = build_dataloader(_Toy(), batch=8, workers=0, shuffle=True, device="cpu")

        class T:
            train_loader = loader

        t = T()
        t.epoch = epoch
        LoaderReseed(seed)(t)
        return [int(v) for b in loader for v in b]

    # 같은 (시드, epoch) 은 같은 순서 — 프로세스가 새로 떠도 재현된다
    assert order_after_reseed(7, 3) == order_after_reseed(7, 3)
    # 시드가 다르면 순서가 다르다
    assert order_after_reseed(7, 3) != order_after_reseed(8, 3)
    # epoch 이 다르면 순서가 다르다
    assert order_after_reseed(7, 3) != order_after_reseed(7, 4)


def test_round_budget_without_resume_unchanged():
    pytest.importorskip("ultralytics")
    from detection.fed_trainer import RoundBudget

    class T:
        epoch = 0
        start_epoch = 0
        stop = False

    b = RoundBudget(2)
    t = T()
    b(t)
    assert b.epochs_ran == 1
    assert t.stop is False
