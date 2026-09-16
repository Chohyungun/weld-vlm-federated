"""34번 §3-1 8-3 — AMP 갱신 카운터 의미 분리: 시도 수 vs 실제 적용 수.

종전 `fed_trainer.optimizer_step` 은 상류 호출마다 +1 — `torch.amp.GradScaler.step` 이 inf/NaN
기울기에서 `optimizer.step` 을 건너뛰어도 셌다. 지금은 `optimizer.step` 뒤에 불리는 **공개 훅**
(`optimizer.register_step_post_hook`)이 실제 호출을 따로 센다.

게이트 조건: 계측이 학습 산술을 바꾸지 않는다 — 소형 CPU 학습에서 계측 전후 최종 가중치가
**바이트 동일**(같은 시드·같은 입력). 아래 `_state_bytes` 는 가중치·옵티마이저 상태뿐 아니라
`param_groups`(lr·momentum)·GradScaler 상태·전역 RNG 까지 넣는다 — LR 궤적이나 scaler 성장만
바꾸는 변이도 잡히게 하기 위해서다.

`optimizer.step` 을 감싸지 않는 이유(정적 검토 Important): torch `LRScheduler.__init__` 이 같은
자리를 `step_fn.__func__` 로 감싸므로 우리가 먼저 감싸면 그 뒤 스케줄러 생성이 AttributeError 로
죽고, 래퍼 클로저가 optimizer↔trainer 순환을 만든다. 공개 훅은 둘 다 없다.
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from detection.fed_trainer import AppliedStepCounter, FedDetectionTrainer, attach_applied_counter


class _Net(torch.nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.l1 = torch.nn.Linear(8, 16)
        self.l2 = torch.nn.Linear(16, 1)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def _data(seed: int = 1, n: int = 24):
    g = torch.Generator().manual_seed(seed)
    xs = torch.randn(n, 8, generator=g)
    ys = torch.randn(n, 1, generator=g)
    return list(zip(xs.split(4), ys.split(4)))


def _state_bytes(model, optimizer, scaler=None) -> bytes:
    """두 실행이 산술적으로 같으면 이 값이 같다 — 가중치·모멘텀·lr·scaler·RNG 전부."""
    out = bytearray()
    for k, v in model.state_dict().items():
        out += k.encode() + v.detach().cpu().numpy().tobytes()
    for st in optimizer.state_dict()["state"].values():
        for k, v in st.items():
            if torch.is_tensor(v):
                out += k.encode() + v.detach().cpu().numpy().tobytes()
    for group in optimizer.param_groups:
        out += repr(sorted((k, v) for k, v in group.items() if k != "params")).encode()
    if scaler is not None:
        out += repr(sorted(scaler.state_dict().items())).encode()
    out += torch.get_rng_state().numpy().tobytes()
    return bytes(out)


# ==========================================================================
# 실제 트레이너 경로 — 계측 전후 바이트 동일 (게이트 조건)
# ==========================================================================

def _bare_trainer(scaler, *, counted: bool) -> FedDetectionTrainer:
    """`optimizer_step` 이 만지는 속성만 채운 트레이너(`__init__` 을 타지 않는다)."""
    tr = object.__new__(FedDetectionTrainer)
    tr.model = _Net(0)
    tr.optimizer = torch.optim.SGD(tr.model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)
    tr.scaler = scaler
    tr.device = torch.device("cpu")
    tr.ema = None
    tr.n_optimizer_updates = 0
    tr.applied_base = 0
    tr._applied = AppliedStepCounter()
    if not counted:
        # 계측을 끈 대조군 — 훅을 걸지 않도록 이미 걸린 것처럼 표시한다.
        tr.optimizer._fed_applied_counter = tr._applied
    return tr


def _run_trainer(scaler, *, counted: bool, inject: set[int] = frozenset()):
    torch.manual_seed(0)
    tr = _bare_trainer(scaler, counted=counted)
    for i, (x, y) in enumerate(_data()):
        tr.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(tr.model(x), y)
        tr.scaler.scale(loss).backward()
        if i in inject:
            tr.model.l1.weight.grad[0, 0] = float("inf")
        tr.optimizer_step()
    return tr


@pytest.mark.parametrize("enabled", [False, True])
def test_계측_전후_최종_상태가_바이트_동일하다(enabled):
    """게이트 조건. 실제 `FedDetectionTrainer.optimizer_step`(= unscale_ → clip → scaler.step →
    update → zero_grad) 을 그대로 타고, 훅을 건 실행과 걸지 않은 실행을 바이트로 비교한다."""
    plain = _run_trainer(torch.amp.GradScaler("cpu", enabled=enabled), counted=False)
    hooked = _run_trainer(torch.amp.GradScaler("cpu", enabled=enabled), counted=True)
    assert plain.n_optimizer_updates == hooked.n_optimizer_updates == 6
    assert hooked.n_optimizer_updates_applied == 6
    assert plain.n_optimizer_updates_applied == 0          # 훅을 걸지 않았으므로 세지 않는다
    assert _state_bytes(plain.model, plain.optimizer, plain.scaler) == \
        _state_bytes(hooked.model, hooked.optimizer, hooked.scaler)


def test_바이트_비교가_이빨이_있다():
    """다른 시드면 달라야 한다 — 비교가 공허하지 않음을 고정."""
    a = _run_trainer(torch.amp.GradScaler("cpu", enabled=True), counted=True)
    torch.manual_seed(5)
    b = _bare_trainer(torch.amp.GradScaler("cpu", enabled=True), counted=True)
    b.model = _Net(7)
    b.optimizer = torch.optim.SGD(b.model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)
    for x, y in _data():
        b.optimizer.zero_grad()
        b.scaler.scale(torch.nn.functional.mse_loss(b.model(x), y)).backward()
        b.optimizer_step()
    assert _state_bytes(a.model, a.optimizer, a.scaler) != _state_bytes(b.model, b.optimizer, b.scaler)


# ==========================================================================
# (a) 스킵된 시도는 적용 수에 들어가지 않는다
# ==========================================================================

def test_진짜_GradScaler가_inf_에서_건너뛴_시도는_적용_수에_없다():
    """(a) 실물 경로 — inf 기울기 2건을 주면 attempts=N, applied=N−2."""
    tr = _run_trainer(torch.amp.GradScaler("cpu", enabled=True), counted=True, inject={1, 4})
    assert tr.n_optimizer_updates == 6
    assert tr.n_optimizer_updates_applied == 4


class _SkippingScaler:
    """`unscale_`/`step`/`update` 만 가진 가짜. 지정한 호출에서 `optimizer.step` 을 건너뛴다."""

    def __init__(self, skip_at: set[int]):
        self.skip_at = skip_at
        self.calls = 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        i, self.calls = self.calls, self.calls + 1
        if i in self.skip_at:
            return None
        return optimizer.step()

    def update(self):
        pass


def test_가짜_scaler_가_두_배치를_건너뛰면_attempts_N_applied_N_minus_2():
    """(a) 스킵 판정을 우리가 통제하는 경우."""
    tr = _run_trainer(_SkippingScaler(skip_at={0, 3}), counted=True)
    assert tr.n_optimizer_updates == 6
    assert tr.n_optimizer_updates_applied == 4


def test_AMP_off_이면_둘이_같다():
    """(b) 비활성 GradScaler(=AMP off)는 항상 `optimizer.step` 을 부른다."""
    tr = _run_trainer(torch.amp.GradScaler("cpu", enabled=False), counted=True)
    assert tr.n_optimizer_updates == tr.n_optimizer_updates_applied == 6


# ==========================================================================
# 훅의 성질 — 멱등 · 옵티마이저 재생성 · 스케줄러 순서 · 순환 참조
# ==========================================================================

def test_훅은_한_번만_걸리고_두_번_걸어도_중복으로_세지_않는다():
    counter = AppliedStepCounter()
    opt = torch.optim.SGD(_Net().parameters(), lr=0.1)
    assert attach_applied_counter(opt, counter) is True
    assert attach_applied_counter(opt, counter) is False
    opt.step()
    assert counter.n == 1


def test_옵티마이저가_새로_만들어져도_계수가_이어진다():
    """상류 OOM 자동 축소는 `_build_train_pipeline` 에서 옵티마이저를 새로 만든다."""
    tr = _bare_trainer(torch.amp.GradScaler("cpu", enabled=False), counted=True)
    for x, y in _data()[:2]:
        tr.optimizer.zero_grad()
        tr.scaler.scale(torch.nn.functional.mse_loss(tr.model(x), y)).backward()
        tr.optimizer_step()
    assert tr.n_optimizer_updates_applied == 2
    tr.optimizer = torch.optim.SGD(tr.model.parameters(), lr=0.05)      # 재생성
    tr.model(torch.ones(2, 8)).sum().backward()
    tr.optimizer_step()
    assert tr.n_optimizer_updates == 3 and tr.n_optimizer_updates_applied == 3


def test_훅은_스케줄러_생성_순서와_무관하다():
    """`optimizer.step` 을 감싸면 이 순서에서 스케줄러 생성이 AttributeError 로 죽는다(실측).
    공개 훅은 `optimizer.step` 을 건드리지 않으므로 순서 제약이 없다."""
    counter = AppliedStepCounter()
    model = _Net()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    attach_applied_counter(opt, counter)                  # 훅 먼저
    torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)
    torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)  # 재구성도 죽지 않는다
    model(torch.ones(1, 8)).sum().backward()
    opt.step()
    assert counter.n == 1
    opt.load_state_dict(opt.state_dict())                  # 상태 복원 뒤에도 남는다
    opt.step()
    assert counter.n == 2


def test_계수_상자는_트레이너를_참조하지_않는다():
    """optimizer → 훅 → 트레이너 → optimizer 순환을 만들면 라운드마다 트레이너를 새로 만드는
    ④ 에서 가중치 한 벌이 gc 패스까지 더 상주한다."""
    tr = _bare_trainer(torch.amp.GradScaler("cpu", enabled=False), counted=True)
    tr.model(torch.ones(2, 8)).sum().backward()
    tr.optimizer_step()
    ref = weakref.ref(tr)
    gc.disable()
    try:
        del tr
        alive = ref() is not None
    finally:
        gc.enable()
        gc.collect()
    assert alive is False, "훅이 트레이너를 강참조한다 — 순환이 생겼다"


def test_계측은_optimizer_step_에서만_걸린다():
    """`__init__` 으로 올리면 옵티마이저가 없을 때 걸릴 수 있고, 재생성 대응도 잃는다."""
    import ast

    import detection.fed_trainer as ft

    tree = ast.parse(Path(ft.__file__).read_text(encoding="utf-8"))
    callers = [fn.name for fn in ast.walk(tree)
               if isinstance(fn, ast.FunctionDef)
               for inner in ast.walk(fn)
               if isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "attach_applied_counter"]
    assert callers == ["optimizer_step"], callers


# ==========================================================================
# RoundResult · meta · 재개 payload
# ==========================================================================

def _round_result(**kw):
    from detection.round_runner import RoundResult

    base = dict(ndarrays=[], num_examples=1, round_idx=0, client_idx=0, epochs_ran=1, seed=0,
                optimizer_steps=1, param_l2_norm=0.0, payload_bytes=0)
    base.update(kw)
    return RoundResult(**base)


def test_필드는_시도_수_하나에_적용_수는_None_기본이다():
    r = _round_result(optimizer_updates=7)
    assert r.optimizer_updates == 7
    assert r.optimizer_step_attempts == 7          # 이름만 다른 별칭(저장 필드는 하나)
    assert r.optimizer_updates_applied is None     # 미계측과 0 을 구분한다


def test_meta_에는_두_값이_남고_원자_로그_지표는_늘지_않는다(monkeypatch, tmp_path):
    """설계: `RoundResult`·②③ meta 에 두 값, 원자 로그에는 추가하지 않는다(라운드당 행 수 불변)."""
    import csv
    import json

    from detection import train_cell

    monkeypatch.setattr(train_cell, "train_round",
                        lambda **kw: _round_result(optimizer_updates=3, optimizer_updates_applied=2,
                                                   seed=kw["base_seed"]))
    train_cell.run_central_cell(data_yaml="a.yaml", num_examples=1, model="m", total_epochs=2,
                                base_seed=7, out_dir=tmp_path, split_hash="h", run_stamp="s")
    names = {r["metric_name"] for r in
             csv.DictReader((tmp_path / "atomic_log.csv").open(encoding="utf-8"))}
    assert names == {"epochs_ran", "optimizer_steps", "optimizer_updates", "param_l2", "lr",
                     "peak_vram_gb"}
    meta = json.loads((tmp_path / "sep_central.meta.json").read_text(encoding="utf-8"))
    assert meta["optimizer_updates"] == 3 and meta["optimizer_updates_applied"] == 2
    assert "optimizer_step_attempts" not in meta      # 프로퍼티는 직렬화되지 않는다(필드 하나)


def _checkpoint(rdir: Path, data: Path, *, epoch_done: int, updates: int, applied=None,
                consumed: int | None = None):
    """실제 체크포인트를 쓴다. `applied=None` 이면 구판(키 부재)을 흉내 낸다."""
    from detection.resume import ResumeCheckpointer, ResumeIdentity
    from detection.round_runner import _LRTrace, derive_seed
    from tests.test_detection_resume import _Counter, _FakeTrainer, _Net as _CkNet

    ident = ResumeIdentity(run_id="t", round_idx=0, client_idx=0, seed=derive_seed(0, 0, 0),
                           total_epochs=4, local_epochs=4, model="m.pt", data=str(data.resolve()),
                           loader_reseed_per_epoch=False, loader_seed=None, profile="pilot")
    tr = _FakeTrainer(_CkNet(), epoch=epoch_done, start_epoch=0)
    tr.n_optimizer_updates = updates
    if applied is not None:
        tr.n_optimizer_updates_applied = applied
    tr._resume_lr_trace = _LRTrace([(e, 0.01) for e in range(epoch_done + 1)])
    ck = ResumeCheckpointer(rdir, identity=ident, step_counter=_Counter(updates * 2),
                            resumed_epochs=(consumed or (epoch_done + 1)) - 1)
    ck.save(tr)


def test_체크포인트가_적용_수를_싣고_되읽는다(tmp_path):
    from detection.resume import latest_resume

    _checkpoint(tmp_path, tmp_path / "d.yaml", epoch_done=1, updates=11, applied=9)
    st = latest_resume(tmp_path)
    assert st.payload["optimizer_updates"] == 11
    assert st.payload["optimizer_updates_applied"] == 9


def test_미상은_다음_체크포인트까지_전파된다(monkeypatch, tmp_path):
    """정적 검토 Important. 구판 체크포인트로 재개하면 적용 수를 알 수 없다. 그 미상을
    다음 체크포인트에 0 으로 굳히면, 재개 한 번 뒤부터 결손된 값이 확정값으로 보고되고
    그 차이가 'AMP 스킵 증거' 로 읽힌다."""
    from detection.resume import latest_resume
    from detection.round_runner import train_round

    import detection.fed_trainer as ft
    from tests._round_fakes import RoundTrainer

    rdir = tmp_path / "_resume"
    data = tmp_path / "d.yaml"
    _checkpoint(rdir, data, epoch_done=0, updates=3, applied=None, consumed=1)   # 구판
    assert "optimizer_updates_applied" not in latest_resume(rdir).payload

    captured = {}

    class Tr(RoundTrainer):
        def __init__(self, overrides=None, **kw):
            super().__init__(overrides, **kw)
            del self.n_optimizer_updates_applied        # 프로퍼티 대신 round_runner 값으로 본다

        def train(self):
            captured["applied_known"] = self.applied_known
            captured["applied_base"] = self.applied_base
            # 이번 프로세스에서 2 회 적용된 것으로 둔다
            self.n_optimizer_updates_applied = self.applied_base + 2

    monkeypatch.setattr(ft, "FedDetectionTrainer", Tr)
    res = train_round(data_yaml=str(data), model="m.pt", total_epochs=4, local_epochs=4,
                      project=tmp_path / "runs", profile="pilot", run_id="t", resume_dir=rdir)
    assert captured == {"applied_known": False, "applied_base": 0}
    assert res.optimizer_updates == 3 and res.optimizer_updates_applied is None

    # 그 프로세스가 남기는 체크포인트에도 키가 없어야 한다(미상 전파).
    from detection.resume import ResumeCheckpointer
    from detection.round_runner import _LRTrace
    from tests.test_detection_resume import _Counter, _FakeTrainer, _Net as _CkNet

    ident = latest_resume(rdir).identity
    ck_tr = _FakeTrainer(_CkNet(), epoch=1, start_epoch=1)
    ck_tr.applied_known = False                 # round_runner 가 심은 미상 표식
    ck_tr.n_optimizer_updates = 5
    ck_tr.n_optimizer_updates_applied = 2       # 값은 있으나 이전 프로세스 몫이 빠졌다
    ck_tr._resume_lr_trace = _LRTrace([(0, 0.01), (1, 0.01)])
    ResumeCheckpointer(rdir, identity=ident, step_counter=_Counter(10),
                       resumed_epochs=1).save(ck_tr)
    assert "optimizer_updates_applied" not in latest_resume(rdir).payload
    assert latest_resume(rdir).payload["optimizer_updates"] == 5     # 시도 수는 이어진다


def test_신판_재개는_적용_수를_이어_센다(monkeypatch, tmp_path):
    from detection.round_runner import train_round

    import detection.fed_trainer as ft
    from tests._round_fakes import RoundTrainer

    rdir = tmp_path / "_resume"
    data = tmp_path / "d.yaml"
    _checkpoint(rdir, data, epoch_done=0, updates=3, applied=3, consumed=1)

    class Tr(RoundTrainer):
        def train(self):
            self.n_optimizer_updates_applied = self.applied_base + 2    # 이번 프로세스 2 회

    monkeypatch.setattr(ft, "FedDetectionTrainer", Tr)
    res = train_round(data_yaml=str(data), model="m.pt", total_epochs=4, local_epochs=4,
                      project=tmp_path / "runs", profile="pilot", run_id="t", resume_dir=rdir)
    assert res.optimizer_updates_applied == 5                            # 3 + 2


def test_적용_수가_시도_수를_넘는_체크포인트는_거부한다(tmp_path):
    """손상·혼입 체크포인트가 조용히 통과해 산출물로 나가지 않게 한다."""
    from detection.resume import latest_resume
    from detection.round_runner import validate_detection_resume

    _checkpoint(tmp_path, tmp_path / "d.yaml", epoch_done=0, updates=3, applied=9, consumed=1)
    with pytest.raises(ValueError, match="적용 수가 시도 수"):
        validate_detection_resume(latest_resume(tmp_path))


def test_구판_체크포인트는_여전히_통과한다(tmp_path):
    """뒤호환 — 적용 수 키가 없는 체크포인트를 거부하지 않는다."""
    from detection.resume import latest_resume
    from detection.round_runner import validate_detection_resume

    _checkpoint(tmp_path, tmp_path / "d.yaml", epoch_done=0, updates=3, applied=None, consumed=1)
    validate_detection_resume(latest_resume(tmp_path))


def test_문면이_시도_수로_정정됐다():
    """게이트 시험 (c). 같은 저장소가 같은 값을 '실제 갱신 횟수' 라고 말하면 새 의미가 무의미하다."""
    root = Path(__file__).resolve().parent.parent
    updated = {
        "detection/budget_audit.py": "시도",
        "fl/client_det.py": "시도",
        "detection/fed_trainer.py": "시도",
        "docs/dev_log/2026-09-03-본실험/10_검출3칸_C.md": "시도",
        "docs/구현설계.md": "시도",
    }
    for rel, word in updated.items():
        src = (root / rel).read_text(encoding="utf-8")
        assert word in src, rel
    # 세 시드 값은 수정하지 않는다 — 각주로만 의미를 밝힌다.
    ten = (root / "docs/dev_log/2026-09-03-본실험/10_검출3칸_C.md").read_text(encoding="utf-8")
    assert "71,202" in ten and "적용** 수는 이 실행에 계측이 없어" in ten
