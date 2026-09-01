"""한 사이클 파일럿 — 학습 전 구간 실행기 (52번 계획 ②③④).

사용:
    uv run python scripts/pilot_c.py views    # 학습 뷰 4벌 생성
    uv run python scripts/pilot_c.py init     # 공통 초기 가중치 (동일 출발 증명)
    uv run python scripts/pilot_c.py cell2    # ② 분리·로컬 (C1/C2/C3)
    uv run python scripts/pilot_c.py cell3    # ③ 분리·중앙
    uv run python scripts/pilot_c.py cell4    # ④ 분리·연합 (Flower R=3)

파일럿 상수는 52번 계획이 정본이다: YOLO11n · 416 · batch 2(+accum 은 검출에서 불필요
— 실측상 메모리가 병목이 아니다) · R=3 E=2 (N=6 등가) · 시드 1세트.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SNAPSHOT_DIR = "data/processed/aihub71761_rt_v1_pilot3000"
OUT_ROOT = Path("outputs/pilot_c").resolve()
MODEL = "yolo11n.pt"
R, E = 3, 2
N = R * E
BASE_SEED = 20260828          # 파일럿 표본 추출 시드를 그대로 쓴다 (기록 목적의 선택)
PROFILE = "pilot"
RUN_STAMP = "260831"


def _snapshot():
    from data.manifest_io import load_snapshot

    sn = load_snapshot(SNAPSHOT_DIR)
    # 스냅샷 digest 는 SNAPSHOT.sha256 마지막 줄의 `# snapshot_digest <hex>` 주석이다.
    txt = (Path(SNAPSHOT_DIR) / "SNAPSHOT.sha256").read_text(encoding="utf-8")
    digest = next(l.split()[-1] for l in txt.splitlines() if "snapshot_digest" in l)
    return sn, digest


def cmd_views() -> None:
    from data.manifest_io import split_view
    from detection.dataset_view import build_yolo_view

    sn, digest = _snapshot()
    print(f"스냅샷 {sn.snapshot_id} digest {digest[:8]}…")
    for tag, client in (("client0", "C1"), ("client1", "C2"), ("client2", "C3"), ("central", None)):
        t0 = time.perf_counter()
        r = build_yolo_view(sn, out_dir=OUT_ROOT / "views" / tag, train_client=client)
        n_train = len(split_view(sn.manifest, "train", client=client))
        print(
            f"  {tag:8s} train {r.n_images['train']:5d}장(manifest {n_train}) "
            f"박스 {r.n_boxes['train']:5d} / val {r.n_images['val']}장 "
            f"/ 배경 {r.n_background} / geom 제외 {r.n_geom_invalid} → {r.data_yaml}"
        )


def cmd_init() -> None:
    from detection.init_weights import build_initial_weights

    arrays, keys, _ = build_initial_weights(
        pretrained=MODEL, nc=4, seed=BASE_SEED, cache_path=OUT_ROOT / "initial.npz"
    )
    from detection import serialize

    print(f"초기 가중치 {len(arrays)} 텐서, keys_digest {serialize.keys_digest(keys)[:12]}…")
    print(f"payload {serialize.payload_nbytes(arrays)/1e6:.1f} MB → {OUT_ROOT/'initial.npz'}")


def _initial():
    from detection.init_weights import build_initial_weights

    return build_initial_weights(
        pretrained=MODEL, nc=4, seed=BASE_SEED, cache_path=OUT_ROOT / "initial.npz"
    )


def _num_examples(sn) -> dict[int, int]:
    from data.manifest_io import split_view

    return {i: len(split_view(sn.manifest, "train", client=c))
            for i, c in enumerate(("C1", "C2", "C3"))}


def cmd_cell2() -> None:
    from detection.train_cell import run_local_cell

    sn, digest = _snapshot()
    arrays, keys, _ = _initial()
    t0 = time.perf_counter()
    results = run_local_cell(
        client_data_yamls={i: OUT_ROOT / "views" / f"client{i}" / "data.yaml" for i in range(3)},
        client_num_examples=_num_examples(sn),
        model=MODEL, total_epochs=N, base_seed=BASE_SEED,
        out_dir=OUT_ROOT / "sep_local", split_hash=digest, run_stamp=RUN_STAMP,
        profile=PROFILE, initial_weights=arrays, canonical_keys=keys,
        resume_root=OUT_ROOT / "_resume" / "sep_local",
    )
    for i, r in results.items():
        print(f"  C{i+1}: {r.epochs_ran}ep steps {r.optimizer_steps} "
              f"peak {r.peak_vram_gb:.2f}GB opt {r.effective_optimizer['optimizer']}")
    print(f"② 완주 {time.perf_counter()-t0:.0f}s")


def cmd_cell3() -> None:
    from data.manifest_io import split_view
    from detection.train_cell import run_central_cell

    sn, digest = _snapshot()
    arrays, keys, _ = _initial()
    t0 = time.perf_counter()
    r = run_central_cell(
        data_yaml=OUT_ROOT / "views" / "central" / "data.yaml",
        num_examples=len(split_view(sn.manifest, "train")),
        model=MODEL, total_epochs=N, base_seed=BASE_SEED,
        out_dir=OUT_ROOT / "sep_central", split_hash=digest, run_stamp=RUN_STAMP,
        profile=PROFILE, initial_weights=arrays, canonical_keys=keys,
        resume_root=OUT_ROOT / "_resume" / "sep_central",
    )
    print(f"  central: {r.epochs_ran}ep steps {r.optimizer_steps} peak {r.peak_vram_gb:.2f}GB")
    print(f"③ 완주 {time.perf_counter()-t0:.0f}s")


def cmd_cell4() -> None:
    from fl.pilot_sim import run_pilot_fed

    sn, digest = _snapshot()
    arrays, keys, ref = _initial()
    n_ex = _num_examples(sn)
    t0 = time.perf_counter()
    run_pilot_fed(
        {
            "out_dir": OUT_ROOT / "sep_fed",
            "views_root": OUT_ROOT / "views",
            "model": MODEL,
            "num_rounds": R,
            "local_epochs": E,
            "total_epochs": N,
            "num_clients": 3,
            "base_seed": BASE_SEED,
            "run_stamp": RUN_STAMP,
            "split_hash": digest,
            "profile": PROFILE,
            "canonical_keys": keys,
            "initial_arrays": arrays,
            "reference_sd": ref,
            "num_examples": [n_ex[0], n_ex[1], n_ex[2]],
        }
    )
    audit = json.loads((OUT_ROOT / "sep_fed" / "audit.json").read_text(encoding="utf-8"))
    print(f"④ 완주 {time.perf_counter()-t0:.0f}s / 회계 ok={audit['ok']} "
          f"총 스텝 {audit['total_optimizer_steps']}")





def cmd_cell6() -> None:
    """⑥ 통합·중앙 — 전체 train 페어, N=R*E epochs."""
    from fl.atomic_log import AtomicLog, new_run_id
    from vlm.pilot_vlm import load_pairs, train_rounds
    from detection.train_cell import save_cell_weights  # noqa: F401 (스키마 참고)
    import numpy as np

    _, digest = _snapshot()
    rows = load_pairs("train")
    out = OUT_ROOT / "uni_central"; out.mkdir(parents=True, exist_ok=True)
    log = AtomicLog(out / "atomic_log.csv", run_id=new_run_id("uni_central", BASE_SEED, RUN_STAMP),
                    seed=BASE_SEED, cell="uni_central", split_hash=digest)

    def cb(ep, mean_ce, steps, wall):
        log.log_round(round_idx=ep, client_id="central", n_train_samples=len(rows),
                      metrics={"mean_ce": mean_ce, "optimizer_steps": float(steps)}, wall_time=wall)
        print(f"  ep{ep}: ce {mean_ce:.4f} steps {steps} ({wall:.0f}s)", flush=True)

    # ⑥ 은 단일 런이라 도중에 죽으면 처음부터다(파일럿에서 10.2시간). 재개 경로를 켠다 —
    # 산출물 디렉터리 밖에 둬서 채점·내보내기가 훑는 트리에 재개 가중치를 남기지 않는다.
    arrays, keys, m, _ = train_rounds(rows=rows, epochs=N, round_idx=0, client_idx=0,
                                      base_seed=BASE_SEED, log_cb=cb,
                                      resume_dir=str(OUT_ROOT / "_resume" / "uni_central"),
                                      run_id=RUN_STAMP)
    np.savez(out / "adapter_last.npz", **{k: a for k, a in zip(keys, arrays)})
    (out / "adapter_last.meta.json").write_text(
        __import__("json").dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"⑥ 완주 {m['wall_s']:.0f}s / steps {m['optimizer_steps']} / peak {m['peak_vram_gb']:.2f}GB "
          f"/ 감독토큰 {m['supervised_tokens']:,} / 어댑터 {m['payload_bytes']/1e6:.1f}MB")


def cmd_cell7() -> None:
    """⑦ 통합·연합 — R 라운드 x E epochs, 어댑터 행렬별 가중 평균.

    전송 계층(Flower)은 ④에서 검증됐다. 여기서는 함정 #3(집계)이 대상이라
    인프로세스 순차 루프로 돈다 — 라운드마다 GPU 에 클라이언트 하나만 올라가는
    실행 형태는 시뮬레이션과 동일하다(동시성 1).
    """
    import json as _json
    import numpy as np
    import torch
    from detection.budget_audit import AccountingCell, AccountingMatrix
    from fl.aggregate import weighted_fedavg
    from fl.atomic_log import AtomicLog, RoundTimer, new_run_id
    from vlm.pilot_vlm import load_pairs, train_rounds

    _, digest = _snapshot()
    counts = _json.loads(Path("data/processed/pairs_pilot_v1/counts.json").read_text(encoding="utf-8"))
    clients = ["C1", "C2", "C3"]
    shards = {i: load_pairs("train", client=c) for i, c in enumerate(clients)}
    # n_k: counts.json 이 단일 소스(선언값). 실측(train 페어 수)과 함께 회계에 남긴다.
    n_declared = {i: counts["clients"][c]["n_total"] for i, c in enumerate(clients)}
    n_train = {i: len(shards[i]) for i in shards}
    print(f"n_k 선언(counts.json n_total) {n_declared} / 실측(train) {n_train}")

    out = OUT_ROOT / "uni_fed"; out.mkdir(parents=True, exist_ok=True)
    log = AtomicLog(out / "atomic_log.csv", run_id=new_run_id("uni_fed", BASE_SEED, RUN_STAMP),
                    seed=BASE_SEED, cell="uni_fed", split_hash=digest)
    acc = AccountingMatrix(num_rounds=R, client_ids=list(shards), local_epochs=E, total_epochs=N)
    timer = RoundTimer()

    global_arrays, keys, ref_sd = None, None, None
    for r in range(R):
        client_payloads, weights = [], []
        for i in sorted(shards):
            arrays, keys, m, ref_sd = train_rounds(
                rows=shards[i], epochs=E, round_idx=r, client_idx=i, base_seed=BASE_SEED,
                adapter_in=global_arrays, adapter_keys=keys,
                resume_dir=str(OUT_ROOT / "_resume" / "uni_fed" / f"r{r:03d}_c{i}"),
                run_id=RUN_STAMP,
            )
            client_payloads.append(arrays); weights.append(n_train[i])
            log.log_round(round_idx=r, client_id=i, n_train_samples=n_train[i],
                          metrics={"optimizer_steps": float(m["optimizer_steps"]),
                                   "supervised_tokens": float(m["supervised_tokens"]),
                                   "param_l2": m["param_l2"], "peak_vram_gb": m["peak_vram_gb"]},
                          bytes_up=m["payload_bytes"], bytes_down=m["payload_bytes"],
                          wall_time=m["wall_s"])
            acc.record(AccountingCell(
                round_idx=r, client_idx=i, epochs_ran=E, optimizer_steps=m["optimizer_steps"],
                num_examples=n_train[i], seed=m["seed"], param_l2_norm=m["param_l2"],
                payload_bytes=m["payload_bytes"], optimizer="AdamW", lr=1e-4,
                momentum=float("nan"), arg_optimizer="AdamW", arg_lr0=1e-4,
                arg_momentum=float("nan")))
            print(f"  r{r} c{i}: steps {m['optimizer_steps']} tok {m['supervised_tokens']:,} "
                  f"peak {m['peak_vram_gb']:.2f}GB ({m['wall_s']:.0f}s)", flush=True)
        agg = weighted_fedavg(client_payloads, weights, keys,
                              {k: torch.as_tensor(v) for k, v in ref_sd.items()})
        global_arrays = agg.ndarrays
        np.savez(out / f"adapter_r{r+1:03d}.npz", **{k: a for k, a in zip(keys, global_arrays)})
        log.log_round(round_idx=r, client_id="server", n_train_samples=agg.total_examples,
                      metrics={"global_l2": agg.global_norm}, wall_time=timer.lap())
        print(f"  r{r} 집계: global_l2 {agg.global_norm:.3f}", flush=True)

    # 병합은 라운드 중 금지 — 최종 어댑터만 저장. 평가용 병합은 채점 단계 몫.
    np.savez(out / "adapter_last.npz", **{k: a for k, a in zip(keys, global_arrays)})
    rep = acc.audit()
    acc.to_csv(out / "accounting.csv"); acc.to_json(out / "accounting.json")
    gaps = log.audit_rounds(R, list(shards) + ["server"])
    print(f"⑦ 완주 / 회계 ok={rep.ok and not gaps} 총 스텝 {rep.total_optimizer_steps} "
          f"/ 결측 {gaps or '없음'}")





def cmd_cell7resume() -> None:
    """⑦ r2 재개 — adapter_r002.npz 주입, 마지막 라운드만.

    회계 매트릭스는 인메모리라 중단으로 r0·r1 셀이 사라졌다. 원자 로그가 정확히 이런
    복구를 위해 존재한다 — r0·r1 셀을 로그에서 재구성해 전 라운드 감사를 완성한다.
    """
    import json as _json
    import numpy as np
    import pandas as pd
    import torch
    from detection.budget_audit import AccountingCell, AccountingMatrix
    from detection.round_runner import derive_seed
    from fl.aggregate import weighted_fedavg
    from fl.atomic_log import AtomicLog, RoundTimer, new_run_id
    from vlm.pilot_vlm import load_pairs, train_rounds

    _, digest = _snapshot()
    clients = ["C1", "C2", "C3"]
    shards = {i: load_pairs("train", client=c) for i, c in enumerate(clients)}
    n_train = {i: len(shards[i]) for i in shards}
    out = OUT_ROOT / "uni_fed"

    loaded = np.load(out / "adapter_r002.npz")
    keys = list(loaded.files)                      # np.savez 키드 저장 — 순서 보존
    global_arrays = [loaded[k] for k in keys]
    print(f"재개: adapter_r002 주입 ({len(keys)} 텐서), r2 만 실행", flush=True)

    log = AtomicLog(out / "atomic_log.csv", run_id=new_run_id("uni_fed", BASE_SEED, RUN_STAMP),
                    seed=BASE_SEED, cell="uni_fed", split_hash=digest)
    acc = AccountingMatrix(num_rounds=R, client_ids=list(shards), local_epochs=E, total_epochs=N)

    # r0·r1 셀을 원자 로그에서 복원 — 시드는 파생 공식으로 재계산(결정론)
    df = pd.read_csv(out / "atomic_log.csv")
    hist = df[df.client_id != "server"].pivot_table(
        index=["round", "client_id"], columns="metric_name", values="metric_value", aggfunc="last")
    up = df[df.client_id != "server"].groupby(["round", "client_id"])["bytes_up"].last()
    for (r, c), row in hist.iterrows():
        c = int(c)
        acc.record(AccountingCell(
            round_idx=int(r), client_idx=c, epochs_ran=E,
            optimizer_steps=int(row["optimizer_steps"]), num_examples=n_train[c],
            seed=derive_seed(BASE_SEED, int(r), c), param_l2_norm=float(row["param_l2"]),
            payload_bytes=int(up.loc[(r, str(c))]), optimizer="AdamW", lr=1e-4,
            momentum=float("nan"), arg_optimizer="AdamW", arg_lr0=1e-4,
            arg_momentum=float("nan")))
    print(f"원자 로그에서 셀 {len(acc.cells)}개 복원", flush=True)

    timer = RoundTimer()
    r = R - 1  # r2
    client_payloads, weights = [], []
    ref_sd = None
    for i in sorted(shards):
        arrays, keys, m, ref_sd = train_rounds(
            rows=shards[i], epochs=E, round_idx=r, client_idx=i, base_seed=BASE_SEED,
            adapter_in=global_arrays, adapter_keys=keys,
            resume_dir=str(OUT_ROOT / "_resume" / "uni_fed" / f"r{r:03d}_c{i}"),
            run_id=RUN_STAMP)
        client_payloads.append(arrays); weights.append(n_train[i])
        log.log_round(round_idx=r, client_id=i, n_train_samples=n_train[i],
                      metrics={"optimizer_steps": float(m["optimizer_steps"]),
                               "supervised_tokens": float(m["supervised_tokens"]),
                               "param_l2": m["param_l2"], "peak_vram_gb": m["peak_vram_gb"]},
                      bytes_up=m["payload_bytes"], bytes_down=m["payload_bytes"],
                      wall_time=m["wall_s"])
        acc.record(AccountingCell(
            round_idx=r, client_idx=i, epochs_ran=E, optimizer_steps=m["optimizer_steps"],
            num_examples=n_train[i], seed=m["seed"], param_l2_norm=m["param_l2"],
            payload_bytes=m["payload_bytes"], optimizer="AdamW", lr=1e-4,
            momentum=float("nan"), arg_optimizer="AdamW", arg_lr0=1e-4,
            arg_momentum=float("nan")))
        print(f"  r{r} c{i}: steps {m['optimizer_steps']} tok {m['supervised_tokens']:,} "
              f"peak {m['peak_vram_gb']:.2f}GB ({m['wall_s']:.0f}s)", flush=True)
    agg = weighted_fedavg(client_payloads, weights, keys,
                          {k: torch.as_tensor(v) for k, v in ref_sd.items()})
    np.savez(out / f"adapter_r{r+1:03d}.npz", **{k: a for k, a in zip(keys, agg.ndarrays)})
    np.savez(out / "adapter_last.npz", **{k: a for k, a in zip(keys, agg.ndarrays)})
    log.log_round(round_idx=r, client_id="server", n_train_samples=agg.total_examples,
                  metrics={"global_l2": agg.global_norm}, wall_time=timer.lap())
    rep = acc.audit()
    acc.to_csv(out / "accounting.csv"); acc.to_json(out / "accounting.json")
    gaps = log.audit_rounds(R, list(shards) + ["server"])
    print(f"  r{r} 집계: global_l2 {agg.global_norm:.3f}", flush=True)
    print(f"⑦ 완주(재개) / 회계 ok={rep.ok and not gaps} 총 스텝 {rep.total_optimizer_steps} "
          f"/ 결측 {gaps or '없음'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["views", "init", "cell2", "cell3", "cell4",
                                    "cell6", "cell7", "cell7resume"])
    # 승격 어블레이션은 **같은 코드 경로**로 다른 스냅샷을 돌려야 한다. 별도 실행기를
    # 만들면 두 팔의 차이가 표본 때문인지 코드 때문인지 구분되지 않는다.
    ap.add_argument("--snapshot", default=None, help="스냅샷 디렉터리 덮어쓰기 (어블레이션용)")
    ap.add_argument("--out", default=None, help="산출 루트 덮어쓰기 (어블레이션용)")
    args = ap.parse_args()
    if args.snapshot:
        SNAPSHOT_DIR = args.snapshot
    if args.out:
        OUT_ROOT = Path(args.out).resolve()
    {"views": cmd_views, "init": cmd_init, "cell2": cmd_cell2, "cell3": cmd_cell3,
     "cell4": cmd_cell4, "cell6": cmd_cell6, "cell7": cmd_cell7,
     "cell7resume": cmd_cell7resume}[args.cmd]()
