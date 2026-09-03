"""본실험 검출 3칸 실행기 — ② 분리·로컬 / ③ 분리·중앙 / ④ 분리·연합 (R=50·E=2·N=100).

2026-09-03 총괄 착수 승인(의사결정로그). 파일럿·게이트에서 산 규칙 전부를 배선한다:

- **분리 실행 + epoch 경계 재개.** 모든 스테이지가 완료 마커로 멱등이라, 죽으면
  같은 명령을 다시 띄우는 것이 곧 재개다(`detection/resume.py` 경유).
- **④ 는 `flwr run` 정식 진입점.** run_simulation(pilot_sim)이 아니라 pyproject 에
  등록된 앱을 CLI 로 띄운다 — 85번 ④⑤⑥ 이 소생시킨 그 경로다. 착수 전 `preflight` 가
  같은 CLI 로 스모크 칸을 끝까지 돌려 진입점 생존을 확인한다.
- 조기 종료 금지(N=100 연속·last 채점) · 원자 로그 · 회계 finally · `gates_evaluated`
  기록 확인(첫 시드 첫 칸에서 멈춰 검사) · 스테이지 사이 GPU/커밋 여유 대기
  (트랙 B 와의 경합으로 죽은 §4-6 의 교훈).
- **detections.jsonl 원시 출력 계약(13_spec_D §2-3)을 C 가 낸다** — D 가 임시 추론을
  다시 하는 일이 없어야 한다(74번 M11). 추론 진입점은 D 의 `load_yolo_from_npz`
  (fp32 · model_cfg 명시) 하나다.

## 시드

시드 1 = 20260828 (파일럿·초기 가중치·게이트 전부의 기준 상수 — 동일 출발 증명의 앵커).
시드 2·3 은 `configs/base.yaml` 의 `experiment.seeds` 등록을 읽는다. **등록 전에는
거부한다** — 결과를 본 뒤 시드를 고르는 경로를 원리적으로 막기 위해서다(사전등록 규율).
R·E·N 도 configs 블록이 생기면 대조해, 여기 상수와 어긋나면 죽는다(정본 이중화 방지).

## 사용

    uv run python scripts/main_det.py preflight            # 결정론 + flwr run 스모크
    uv run python scripts/main_det.py chain --seed 1       # ② → ③ → ④ (분리 실행 권장)
    uv run python scripts/main_det.py export --seed 1      # detections.jsonl (CPU, 병행 가능)
    uv run python scripts/main_det.py status

산출: outputs/main_c/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

SNAPSHOT_DIR = "data/interim/manifest_v1"
OUT = Path("outputs/main_c").resolve()
VIEWS = OUT / "views"

MODEL = "yolo11s.pt"
PROFILE = "main"
#: 총괄 확정(의사결정로그 2026-09-03). configs 에 `experiment` 블록이 등록되면 대조한다.
R, E = 50, 2
N = R * E
CLIENTS = ("C1", "C2", "C3")
#: 시드 1. 파일럿 표본·초기 가중치·LoRA 초기화·§4-6 게이트가 전부 이 상수 위에 있다.
SEED1 = 20260828


# --------------------------------------------------------------------------
# 등록값 로드 — configs 가 정본이고, 미등록이면 할 수 있는 것만 한다
# --------------------------------------------------------------------------

def registered_seeds() -> dict[int, int | None]:
    """{시드 번호: base_seed 또는 None(미등록)}.

    시드 2·3 값을 여기서 지어내지 않는다. 등록 전에 돌리기 시작하면 "결과를 보고 시드를
    고르는" 경로가 열린다 — 사전등록 상수가 어느 모집단에서도 재현되지 않던 사고(80번 D2)
    와 같은 부류다. A 의 configs 등록(`experiment.seeds`)을 기다린다.
    """
    seeds: dict[int, int | None] = {1: SEED1, 2: None, 3: None}
    cfg_path = Path("configs/base.yaml")
    if not cfg_path.exists():
        return seeds
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    exp = cfg.get("experiment") or {}
    reg = exp.get("seeds")
    if reg:
        reg = [int(v) for v in reg]
        if reg[0] != SEED1:
            raise SystemExit(
                f"등록 시드 1({reg[0]})이 기준 상수 {SEED1} 과 다르다 — 파일럿·초기 가중치"
                "·게이트가 전부 이 상수 위에 있다. 총괄 판정 없이 진행하지 않는다."
            )
        for i, v in enumerate(reg[:3], start=1):
            seeds[i] = v
    for key, want in (("rounds", R), ("local_epochs", E), ("total_epochs", N)):
        if key in exp and int(exp[key]) != want:
            raise SystemExit(
                f"configs experiment.{key}={exp[key]} 가 확정값 {want} 과 다르다 — "
                "정본이 갈라졌다. 멈추고 보고한다."
            )
    return seeds


def _seed_value(seed_no: int) -> int:
    seeds = registered_seeds()
    v = seeds.get(seed_no)
    if v is None:
        raise SystemExit(
            f"시드 {seed_no} 는 아직 configs 에 등록되지 않았다(experiment.seeds). "
            "A 의 등록을 기다린다 — 여기서 값을 지어내면 사전등록 규율이 깨진다. "
            f"지금 돌릴 수 있는 것: 시드 1 ({SEED1})."
        )
    return v


def _snapshot():
    from data.manifest_io import load_snapshot

    sn = load_snapshot(SNAPSHOT_DIR)
    txt = (Path(SNAPSHOT_DIR) / "SNAPSHOT.sha256").read_text(encoding="utf-8")
    digest = next(l.split()[-1] for l in txt.splitlines() if "snapshot_digest" in l)
    return sn, digest


def _seed_dir(seed_no: int) -> Path:
    return OUT / f"seed{seed_no}"


def _commit_free_gb() -> float:
    """Windows **커밋 차지** 여유(GlobalMemoryStatusEx.ullAvailPageFile).

    `psutil.virtual_memory().available`(물리 RAM)이 아니다 — 그 지표를 잘못 써서 시드 1
    체인이 6시간 헛대기 후 자멸했다(이 기계의 평시 물리 여유가 ~13GB 라 16GB 문턱을
    영원히 못 넘는다). §4-6 을 실제로 죽인 것은 물리 RAM 이 아니라 커밋 고갈이었다
    (건강 시 24~30GB, 고장 시 0.5GB — 전부 커밋 기준 실측).
    """
    import ctypes

    class _MSX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = _MSX()
    st.dwLength = ctypes.sizeof(_MSX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPageFile / 1e9


def _headroom_wait(need_gpu_mib: int = 9000, need_commit_gb: float = 16.0,
                   deadline_s: int = 6 * 3600) -> None:
    """GPU·호스트 **커밋** 여유를 기다린다. §4-6 이 두 번 죽은 자리다(B 와의 경합)."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30)
            used, total = (int(x) for x in q.stdout.strip().split(","))
            gpu_free = total - used
        except Exception:                                     # noqa: BLE001
            gpu_free = 0
        commit_free = _commit_free_gb()
        if gpu_free >= need_gpu_mib and commit_free >= need_commit_gb:
            return
        print(f"[대기] GPU {gpu_free}MiB(need {need_gpu_mib}) / "
              f"커밋 {commit_free:.1f}GB(need {need_commit_gb}) — 60초 후 재확인", flush=True)
        time.sleep(60)
    raise SystemExit("자원 여유를 6시간 기다렸지만 확보되지 않았다 — 멈추고 보고한다.")


def _ledger(seed_no: int, stage: str, **kw) -> None:
    """스테이지별 실측(벽시계·경합 여부)을 진행 원장에 남긴다 — 처리 속도 축의 재료."""
    p = OUT / "progress.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "seed": seed_no, "stage": stage, **kw},
                            ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# 스테이지
# --------------------------------------------------------------------------

def stage_views() -> None:
    """학습 뷰 4벌 — 시드와 무관하므로 한 번만. 하드링크라 원본 무변경."""
    from data.manifest_io import split_view
    from detection.dataset_view import build_yolo_view

    if (VIEWS / "central" / "data.yaml").exists():
        print("뷰 존재 — 건너뜀", flush=True)
        return
    sn, digest = _snapshot()
    print(f"스냅샷 {sn.snapshot_id} digest {digest[:8]}…", flush=True)
    for tag, client in (("client0", "C1"), ("client1", "C2"),
                        ("client2", "C3"), ("central", None)):
        t0 = time.perf_counter()
        r = build_yolo_view(sn, out_dir=VIEWS / tag, train_client=client)
        n_train = len(split_view(sn.manifest, "train", client=client))
        print(f"  {tag:8s} train {r.n_images['train']:6,}장(manifest {n_train:,}) "
              f"박스 {r.n_boxes['train']:6,} / geom 제외 {r.n_geom_invalid} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)


def _initial(seed_no: int):
    """시드별 공통 초기 가중치 — 같은 시드의 세 칸이 같은 출발점이라는 증명."""
    from detection.init_weights import build_initial_weights

    return build_initial_weights(
        pretrained=MODEL, nc=4, seed=_seed_value(seed_no),
        cache_path=_seed_dir(seed_no) / "initial.npz")


def _num_examples(sn) -> dict[int, int]:
    from data.manifest_io import split_view

    return {i: len(split_view(sn.manifest, "train", client=c))
            for i, c in enumerate(CLIENTS)}


def _assert_gates_recorded(meta_path: Path) -> None:
    """첫 칸 완료 직후 게이트 평가 기록을 확인하고서만 진행한다(지시서 요구)."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ge = meta.get("gates_evaluated") or []
    gr = meta.get("gate_results") or {}
    if "deterministic_torch" not in ge or gr.get("cudnn_deterministic") is not True:
        raise SystemExit(
            f"게이트 평가가 기록되지 않았다({meta_path}): gates_evaluated={ge}, "
            f"cudnn_deterministic={gr.get('cudnn_deterministic')} — 진행 중단, 보고한다."
        )
    print(f"게이트 확인: {ge} / cudnn_deterministic={gr['cudnn_deterministic']}", flush=True)


def stage_cell2(seed_no: int) -> None:
    """② 분리·로컬 — 클라이언트 3 독립, N=100 연속."""
    from detection.train_cell import run_local_cell

    sd = _seed_dir(seed_no)
    done = [(sd / "sep_local" / f"sep_local_c{i}.npz").exists() for i in range(3)]
    if all(done):
        print("② 완료 마커 존재 — 건너뜀", flush=True)
        return
    sn, digest = _snapshot()
    arrays, keys, _ = _initial(seed_no)
    seed = _seed_value(seed_no)
    stamp = f"main_s{seed_no}"
    t0 = time.perf_counter()
    results = run_local_cell(
        client_data_yamls={i: VIEWS / f"client{i}" / "data.yaml" for i in range(3)},
        client_num_examples=_num_examples(sn),
        model=MODEL, total_epochs=N, base_seed=seed,
        out_dir=sd / "sep_local", split_hash=digest, run_stamp=stamp,
        profile=PROFILE, initial_weights=arrays, canonical_keys=keys,
        resume_root=sd / "_resume" / "sep_local",
    )
    for i, r in sorted(results.items()):
        print(f"  C{i+1}: {r.epochs_ran}ep steps {r.optimizer_steps:,} "
              f"opt {r.effective_optimizer['optimizer']} peak {r.peak_vram_gb:.2f}GB",
              flush=True)
    _ledger(seed_no, "cell2", wall_s=round(time.perf_counter() - t0, 1))
    # 첫 시드 첫 칸 — 게이트 기록 확인 후에만 다음 스테이지로
    _assert_gates_recorded(sd / "sep_local" / "sep_local_c0.meta.json")


def stage_cell3(seed_no: int) -> None:
    """③ 분리·중앙 — 학습 풀 전체 N=100 연속. 가장 긴 검출 런."""
    from data.manifest_io import split_view
    from detection.train_cell import run_central_cell

    sd = _seed_dir(seed_no)
    if (sd / "sep_central" / "sep_central.npz").exists():
        print("③ 완료 마커 존재 — 건너뜀", flush=True)
        return
    sn, digest = _snapshot()
    arrays, keys, _ = _initial(seed_no)
    t0 = time.perf_counter()
    r = run_central_cell(
        data_yaml=VIEWS / "central" / "data.yaml",
        num_examples=len(split_view(sn.manifest, "train")),
        model=MODEL, total_epochs=N, base_seed=_seed_value(seed_no),
        out_dir=sd / "sep_central", split_hash=digest, run_stamp=f"main_s{seed_no}",
        profile=PROFILE, initial_weights=arrays, canonical_keys=keys,
        resume_root=sd / "_resume" / "sep_central",
    )
    print(f"  central: {r.epochs_ran}ep steps {r.optimizer_steps:,} "
          f"peak {r.peak_vram_gb:.2f}GB", flush=True)
    _ledger(seed_no, "cell3", wall_s=round(time.perf_counter() - t0, 1))


def _flwr_run_config(seed_no: int, *, cell: str, rounds: int, local_epochs: int,
                     total_epochs: int, project: Path, digest: str,
                     n_ex: dict[int, int] | None = None) -> str:
    """`flwr run --run-config` 문자열. 문자열 값은 TOML 규약대로 따옴표를 감싼다."""
    items = {
        "cell": f'"{cell}"',
        "num-server-rounds": rounds,
        "local-epochs": local_epochs,
        "total-epochs": total_epochs,
        "num-clients": 3,
        "num-classes": 4,
        "base-seed": _seed_value(seed_no),
        "run-stamp": f'"main_s{seed_no}"',
        "split-hash": f'"{digest}"',
        "model": f'"{MODEL}"',
        "project": f'"{project.as_posix()}"',
        "views-root": f'"{VIEWS.as_posix()}"',
        "profile": f'"{PROFILE}"',
        "num-examples": '"{}"'.format(",".join(str(n_ex[i]) for i in range(3))) if n_ex else '""',
        "resume-root": f'"{(project / "_resume" / cell).as_posix()}"',
    }
    return " ".join(f"{k}={v}" for k, v in items.items())


#: SuperLink 연결 설정 — flwr 1.33 이 pyproject 의 [tool.flwr.federations] 를 사용자
#: 레벨 저장소로 이관해 버리므로(첫 flwr run 이 pyproject 를 재작성했다, 실측) 값이
#: 저장소 밖에 살게 된다. 매 실행 명시해 **여기(버전 관리)가 정본**이 되게 한다.
#: GPU 1.0 = 클라이언트 동시성 1 — VLM 은 물론 검출도 batch 32 에서 7.7GB 라
#: 두 클라이언트를 동시에 못 올린다.
FEDERATION_CONFIG = ("options.num-supernodes=3 "
                     "options.backend.client-resources.num-cpus=2 "
                     "options.backend.client-resources.num-gpus=1.0")


def _flwr_run(run_config: str, log_path: Path) -> None:
    """정식 진입점. pyproject [tool.flwr] 의 등록 앱을 CLI 로 띄운다."""
    cmd = ["flwr", "run", ".", "local-sim", "--run-config", run_config,
           "--federation-config", FEDERATION_CONFIG, "--stream"]
    print("  $", " ".join(cmd[:5]), "…", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import os

    env = dict(os.environ)
    # flwr 의 FAB 설치 경로가 pyproject 를 인코딩 지정 없이 read_text() 한다 — Windows
    # 기본(cp949)에서 한글 주석 바이트에 UnicodeDecodeError 로 죽는다(실측, exit 700).
    # 프레임워크 결함이라 우리 쪽에서 파이썬 기본 인코딩을 UTF-8 로 강제한다.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(Path.cwd()), env=env).returncode
    if rc != 0:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        # 알려진 프레임워크 결함 1건만 1회 자가 복구한다: 상주 SuperLink 가 cp949 환경에서
        # 떠 있으면 그 자식(시뮬 프로세스)이 pyproject 의 한글에서 UnicodeDecodeError 로
        # 죽는다(exit 700, 실측). 데몬을 내리면 다음 flwr run 이 UTF-8 환경에서 새로 띄운다.
        # **이 시그니처가 아니면 재시도하지 않는다** — 우회 금지.
        if "UnicodeDecodeError: 'cp949'" in text and "install_from_fab" in text:
            print("  상주 SuperLink 의 cp949 환경 결함 감지 — 데몬 재기동 후 1회 재시도",
                  flush=True)
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Where-Object { $_.CommandLine -like '*superlink*' -or "
                 "$_.CommandLine -like '*flwr*' } | "
                 "ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
                 "-ErrorAction SilentlyContinue }"], capture_output=True)
            time.sleep(5)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write("\n===== cp949 데몬 재기동 후 재시도 =====\n")
                fh.flush()
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=str(Path.cwd()), env=env).returncode
        if rc != 0:
            tail = "\n".join(log_path.read_text(encoding="utf-8",
                                                errors="replace").splitlines()[-15:])
            raise SystemExit(f"flwr run 실패 rc={rc}:\n{tail}")


def stage_cell4(seed_no: int) -> None:
    """④ 분리·연합 — R=50 · E=2, flwr run 정식 진입점."""
    sd = _seed_dir(seed_no)
    out = sd / "fl" / "sep_fed"
    if (out / "audit.json").exists() and (out / f"global_r{R:03d}.npz").exists():
        audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
        if audit.get("ok"):
            print("④ 완료 마커(audit ok) 존재 — 건너뜀", flush=True)
            return
        raise SystemExit(f"④ 회계가 실패 상태다({out/'audit.json'}) — 보고한다.")
    sn, digest = _snapshot()
    # 서버 `_load_initial` 의 캐시 경로(project/fl/sep_fed/initial.npz)에 **미리 만들어
    # 둔다.** 같은 시드라 서버가 새로 만들어도 같은 값이 나오지만(결정론), 같은 파일을
    # 읽게 하면 ②③ 과의 동일 출발 대조가 파일 해시 하나로 끝난다.
    from detection.init_weights import build_initial_weights

    build_initial_weights(pretrained=MODEL, nc=4, seed=_seed_value(seed_no),
                          cache_path=out / "initial.npz")
    t0 = time.perf_counter()
    _flwr_run(
        _flwr_run_config(seed_no, cell="sep_fed", rounds=R, local_epochs=E,
                         total_epochs=N, project=sd, digest=digest,
                         n_ex=_num_examples(sn)),
        log_path=sd / "flwr_sep_fed.log",
    )
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    if not audit.get("ok"):
        raise SystemExit(f"④ 회계 실패: {audit.get('failures')} — 보고한다.")
    print(f"④ 회계 ok / 총 스텝 {audit['total_optimizer_steps']:,}", flush=True)
    _ledger(seed_no, "cell4", wall_s=round(time.perf_counter() - t0, 1))


def cmd_preflight() -> None:
    """착수 전 확인 셋 — 결정론 · 등록값 · flwr run 진입점 생존."""
    from fl.run_gates import apply_run_gates

    seeds = registered_seeds()
    print(f"시드 등록 상태: {seeds}", flush=True)
    gates = apply_run_gates(cell="preflight")
    print(f"게이트: {gates['gates_evaluated']} / {gates['gate_results']}", flush=True)
    if gates["gate_results"].get("cudnn_deterministic") is not True:
        raise SystemExit("cudnn 결정론이 걸리지 않는다 — 착수 불가.")

    # flwr run 정식 진입점을 스모크 칸으로 끝까지 — 시험이 아니라 실 CLI 다.
    scratch = OUT / "_preflight"
    _flwr_run(
        f'cell="smoke" num-server-rounds=2 local-epochs=1 total-epochs=2 '
        f'num-clients=3 num-classes=4 base-seed=1 run-stamp="preflight" '
        f'split-hash="preflight" model="{MODEL}" project="{scratch.as_posix()}" '
        f'views-root="{VIEWS.as_posix()}" profile="{PROFILE}" '
        f'num-examples="1,1,1" resume-root=""',
        log_path=OUT / "preflight_flwr.log",
    )
    audit = json.loads((scratch / "fl" / "smoke" / "audit.json").read_text(encoding="utf-8"))
    if not audit["ok"]:
        raise SystemExit(f"preflight 스모크 회계 실패: {audit['failures']}")
    print("preflight 통과 — flwr run 진입점 생존, 회계 ok", flush=True)


def cmd_chain(seed_no: int) -> None:
    """② → ③ → ④ 순차. 각 스테이지 전에 자원 여유를 기다린다. 멱등 — 재기동이 곧 재개."""
    stage_views()
    for name, fn in (("cell2", stage_cell2), ("cell3", stage_cell3),
                     ("cell4", stage_cell4)):
        _headroom_wait()
        print(f"\n########## 시드 {seed_no} · {name} ##########", flush=True)
        fn(seed_no)
    (_seed_dir(seed_no) / "CHAIN_DONE").write_text(
        time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    print(f"시드 {seed_no} 검출 3칸 완주", flush=True)


# --------------------------------------------------------------------------
# detections.jsonl — 13_spec_D §2-3 원시 출력 계약
# --------------------------------------------------------------------------

def _export_targets(seed_no: int) -> list[tuple[str, Path]]:
    sd = _seed_dir(seed_no)
    return [
        ("sep_local_c0", sd / "sep_local" / "sep_local_c0.npz"),
        ("sep_local_c1", sd / "sep_local" / "sep_local_c1.npz"),
        ("sep_local_c2", sd / "sep_local" / "sep_local_c2.npz"),
        ("sep_central", sd / "sep_central" / "sep_central.npz"),
        ("sep_fed", sd / "fl" / "sep_fed" / f"global_r{R:03d}.npz"),
    ]


def cmd_export(seed_no: int) -> None:
    """평가셋 전량 추론 → `{tag}.detections.jsonl` (§2-3).

    - 추론 진입점은 D 의 `load_yolo_from_npz` **하나**(fp32, model_cfg 명시 — .half() 금지).
    - `conf = CONF_FLOOR(0.01)` 하한 추론으로 낸다. D 가 어떤 임계로든 걸러 쓸 수 있고
      (`filter_by_conf` 동치는 D 가 비트 대조로 실측), 채점 임계를 C 가 선점하지 않는다.
    - CPU 고정 — 74번 M11(±2박스 어긋남)의 재발을 막으려면 산출 장치가 기록·고정돼야 한다.
    - 부속 meta 에 checkpoint sha256·조건 전량을 남긴다.
    """
    import hashlib

    from data.manifest_io import load_snapshot, split_view
    from evaluation.detect_infer import load_yolo_from_npz
    from evaluation.params import ScoringParams
    from vlm.coords import CoordCfg, coord_cfg_hash

    params = ScoringParams(snapshot=Path(SNAPSHOT_DIR), pilot=_seed_dir(seed_no),
                           out=_seed_dir(seed_no), seed=_seed_value(seed_no), imgsz=640)
    sn = load_snapshot(SNAPSHOT_DIR)
    eval_m = split_view(sn.manifest, "eval")
    repo = Path.cwd().resolve()
    paths = [str(repo / p) for p in eval_m["rel_path"]]
    ids = list(eval_m["image_id"])
    cfg_hash = coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG"))
    out_dir = _seed_dir(seed_no) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"평가셋 {len(ids):,}장 · conf {params.conf_floor}(하한) · imgsz 640 · CPU",
          flush=True)

    for tag, npz in _export_targets(seed_no):
        out_path = out_dir / f"{tag}.detections.jsonl"
        if out_path.exists():
            print(f"  {tag}: 존재 — 건너뜀", flush=True)
            continue
        if not npz.exists():
            print(f"  {tag}: 가중치 없음({npz}) — 건너뜀", flush=True)
            continue
        t0 = time.perf_counter()
        yolo = load_yolo_from_npz(npz, params.class_names, 640,
                                  model_cfg="yolo11s.yaml")
        n_boxes = 0
        tmp_path = out_path.with_suffix(".part")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for i in range(0, len(paths), int(params.predict_chunk)):
                results = yolo.predict(paths[i:i + int(params.predict_chunk)],
                                       imgsz=640, conf=params.conf_floor,
                                       device="cpu", max_det=params.max_det,
                                       verbose=False)
                for image_id, res in zip(ids[i:i + int(params.predict_chunk)], results):
                    boxes = [
                        {"cls": int(c), "xyxy_px": [round(float(v), 2) for v in xy],
                         "conf": round(float(cf), 4)}
                        for c, xy, cf in zip(res.boxes.cls, res.boxes.xyxy, res.boxes.conf)
                    ]
                    n_boxes += len(boxes)
                    fh.write(json.dumps(
                        {"image_id": image_id, "boxes": boxes,
                         "coord_space": "ABS_ORIG", "coord_cfg_hash": cfg_hash},
                        ensure_ascii=False) + "\n")
        tmp_path.replace(out_path)
        wall = time.perf_counter() - t0
        (out_dir / f"{tag}.detections.meta.json").write_text(json.dumps({
            "checkpoint": str(npz),
            "checkpoint_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
            "model_cfg": "yolo11s.yaml", "imgsz": 640, "conf": params.conf_floor,
            "max_det": params.max_det, "device": "cpu", "precision": "fp32",
            "eval_images": len(ids), "n_boxes": n_boxes,
            "coord_space": "ABS_ORIG", "coord_cfg_hash": cfg_hash,
            "wall_s": round(wall, 1), "seed": _seed_value(seed_no),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {tag}: 박스 {n_boxes:,} ({wall:.0f}s) → {out_path.name}", flush=True)
        del yolo
    _ledger(seed_no, "export")


def cmd_status() -> None:
    for k in (1, 2, 3):
        sd = _seed_dir(k)
        marks = {
            "②": all((sd / "sep_local" / f"sep_local_c{i}.npz").exists() for i in range(3)),
            "③": (sd / "sep_central" / "sep_central.npz").exists(),
            "④": (sd / "fl" / "sep_fed" / "audit.json").exists(),
            "export": all((sd / "predictions" / f"{t}.detections.jsonl").exists()
                          for t, _ in _export_targets(k)),
        }
        print(f"시드 {k}: " + "  ".join(f"{n} {'O' if v else '·'}" for n, v in marks.items()))
    p = OUT / "progress.jsonl"
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines()[-6:]:
            print(" ", l)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["preflight", "views", "chain", "cell2", "cell3",
                                    "cell4", "export", "status"])
    ap.add_argument("--seed", type=int, default=1, help="시드 번호 (1~3)")
    a = ap.parse_args()
    dispatch = {
        "preflight": cmd_preflight,
        "views": stage_views,
        "status": cmd_status,
        "chain": lambda: cmd_chain(a.seed),
        "cell2": lambda: stage_cell2(a.seed),
        "cell3": lambda: stage_cell3(a.seed),
        "cell4": lambda: stage_cell4(a.seed),
        "export": lambda: cmd_export(a.seed),
    }
    dispatch[a.cmd]()
