"""F03 독립 재현: 실제 Ultralytics save_metrics 로 마지막 epoch 행을 쓰면 열 수가 어떻게 되는가 (패치 전/후)."""
import sys, csv, math, time, tempfile, pathlib
from detection.fed_trainer import FedDetectionTrainer
tree = sys.argv[1]
t = object.__new__(FedDetectionTrainer)
t.csv = pathlib.Path(tempfile.mkdtemp()) / "results.csv"; t.train_time_start = time.time()
metric_keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "val/box_loss", "val/cls_loss", "val/dfl_loss"]
t.metrics = dict.fromkeys(metric_keys, 0.0)        # upstream _setup_train 이 0 으로 초기화하는 그 dict
losses = {"train/box_loss": 2.2, "train/cls_loss": 1.7, "train/dfl_loss": 1.4}; lrs = {"lr/pg0": 1e-4, "lr/pg1": 1e-4, "lr/pg2": 1e-4}
for ep in (97, 98):
    t.epoch = ep; t.save_metrics({**losses, **t.metrics, **lrs})
t.epoch = 99
t.metrics, fit = t.validate()                     # upstream 은 final_epoch 에 validate() 반환으로 self.metrics 를 교체한다
t.save_metrics({**losses, **t.metrics, **lrs})
rows = list(csv.reader(t.csv.open(newline="", encoding="utf-8")))
hdr = rows[0]; last = dict(zip(hdr, rows[-1]))
print(f"[{tree}] validate() → {t.metrics if len(str(t.metrics))<60 else str(t.metrics)[:60]+'…'} | 행별 열 수 {[len(r) for r in rows]}")
print(f"[{tree}] 헤더로 읽은 마지막 행: precision={last.get('metrics/precision(B)')!r} mAP50-95={last.get('metrics/mAP50-95(B)')!r} lr/pg0={last.get('lr/pg0')!r}")
print(f"[{tree}] 마지막 행 val/metrics 열이 NaN 인가: {all(math.isnan(float(last[k])) for k in metric_keys) if all(last.get(k) not in (None, '') for k in metric_keys) else '열 결손'}")
