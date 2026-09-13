"""F05 독립 재현: (1) 후반 라운드 재개 셀의 stopper 감사식 (2) 예산 완료 체크포인트 재개 시 추가 1 epoch (RoundBudget 의미)."""
import sys
from types import SimpleNamespace
from detection.budget_audit import AccountingCell, AccountingMatrix
from detection.fed_trainer import RoundBudget
tree = sys.argv[1]
def cell(r, **kw):
    v = dict(round_idx=r, client_idx=0, epochs_ran=2, optimizer_steps=10, num_examples=5, seed=0,
             stopper_class="NoEarlyStopping", stopper_true_count=0, stopper_calls=2); v.update(kw)
    return AccountingCell(**v)
m = AccountingMatrix(3, [0], 2, 6)
m.record(cell(0)); m.record(cell(1))
# 라운드 2(전역 epoch 4·5) 를 전역 epoch 5 에서 재개 → 이 프로세스는 1 epoch 만 돌았는데 stopper 호출 0 이면 잡혀야 한다
m.record(cell(2, resumed_from_epoch=5, stopper_calls=0))
rep = m.audit()
print(f"[{tree}] (1) 후반 라운드 재개·stopper 0회 → audit ok={rep.ok} | stopper 관련 실패={[f for f in rep.failures if 'stopper' in f or '좌표' in f]}")
# (2) 예산 완료 체크포인트(E=2 중 2 완료)로 RoundBudget 을 만들면, 첫 epoch 종료 콜백에서야 stop 이 켜진다 = 1 epoch 추가 학습
b = RoundBudget(2, resumed_epochs=2)
tr = SimpleNamespace(epoch=4, start_epoch=4, stop=False)
before = getattr(b, "epochs_ran", None)
b(tr)
print(f"[{tree}] (2) RoundBudget(E=2, resumed=2): 콜백 전 epochs_ran={before} → 1 epoch 돈 뒤 epochs_ran={b.epochs_ran}, stop={tr.stop} (stop 이 콜백 뒤에야 켜짐 = 추가 1 epoch)")
try:
    from detection.round_runner import validate_detection_resume
    st = SimpleNamespace(identity=SimpleNamespace(local_epochs=2, round_idx=2), epochs_ran_in_round=2, next_epoch=6,
                         optimizer_steps=10, payload={"optimizer_updates": 5, "lr_trace": [(4, .01), (5, .01)]})
    try:
        validate_detection_resume(st); print(f"[{tree}] (3) validate_detection_resume: 완료 체크포인트 통과(거부 없음)")
    except ValueError as e: print(f"[{tree}] (3) validate_detection_resume: 거부 — {e}")
except ImportError:
    print(f"[{tree}] (3) validate_detection_resume 없음(패치 전)")
