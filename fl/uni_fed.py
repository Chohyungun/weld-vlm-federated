"""통합형 연합 칸의 앞뒤 — 시작 전 원장 검사, 끝난 뒤 어댑터 저장과 끝 행(리허설 2판 §0-1 · §2-6 · §2-7).

두 서버 경로(`fl/server_app.py` 의 `flwr run` · `fl/pilot_sim.py` 의 인프로세스)가 이 모듈 하나를 부른다.
검출 연합의 저장(`_save_round` 의 위치 배열 `np.savez`)은 바꾸지 않는다 — 통합형일 때만 여기를 탄다.

## 순서

1. **시작 전** — 원장(`atomic_log.csv`)에 데이터 행이 하나라도 있으면 시작하지 않는다. 연합은 재개가 없고,
   같은 `run_id` 의 재기동은 `AtomicLog` 가 통과시켜 앞 실행의 행 위에 덧붙이므로 그 원장은 겹친 행으로
   영구 거부 상태가 된다. 다시 돌 때는 새 스탬프와 새 출력 폴더를 쓴다. 본실험은 계획 파일을 받지 않는다.
2. **라운드마다** — 클라이언트가 문자열 필드(`CLIENT_STRINGS`)로 보낸 학습 쪽 값과 라운드의 글로벌 어댑터를 모은다.
3. **회계 감사가 통과한 뒤** — 모은 값이 라운드 · 클라이언트 사이에서 같은지 보고, 원장의 라운드 행을 규칙
   ①~③ 으로 본 뒤, 키가 있는 `adapter_last.npz` 와 meta 를 내구 저장하고 끝 행(`client_id="server"` ·
   `round=R` · 값 `R`)을 쓴다. 감사에 실패한 실행은 여기 오지 않는다(`finalize_accounting` 이 먼저 올린다).

거부는 `ValueError` 하위 예외다 — `SystemExit` 는 SuperLink 의 실패 처리를 건너뛰고 인프로세스 시뮬레이션이
성공으로 끝난다(`fl/atomic_log.LedgerIdentityMismatch` 와 같은 까닭).

이 모듈은 모듈 수준에서 torch 를 가져오지 않는다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fl.atomic_log import END_METRIC, last_data_row
from fl.uni_run_config import client_run_cfg, down_config

__all__ = ["UniFedRejected", "CLIENT_STRINGS", "client_strings", "assert_fresh_ledger",
           "check_fed_ledger", "check_fed_prior", "fed_identity", "UniFedRun", "FED_TAG"]

FED_TAG = "uni_fed"

#: 클라이언트가 라운드마다 싣는 학습 쪽 값. `MetricRecord` 는 문자열을 받지 않아 `ConfigRecord` 로 나른다.
CLIENT_STRINGS = ("uni-client-tag", "uni-model-id", "uni-model-revision", "uni-train-rows-digest",
                  "uni-grid-observed", "uni-processor-config-sha256", "uni-config-blocks", "uni-impl-ids",
                  "uni-prompt-sha256", "uni-template-mode", "uni-coord-cfg-hash",
                  "uni-target-contract-sha256")
#: 빈 값이 뜻을 갖는 필드 — 판을 적지 않은 파일럿 모델은 판이 빈 문자열이다. 나머지는 비면 거부한다.
EMPTY_OK = ("uni-model-revision",)


class UniFedRejected(ValueError):
    """통합형 연합 서버가 시작하지 않거나 원장을 닫지 않는다. `code` 가 사유다."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def client_strings(m: Mapping[str, Any], *, rows: Sequence[Mapping[str, Any]], client_tag: str) -> dict[str, str]:
    """클라이언트 쪽 — `train_rounds` 의 지표와 학습한 행에서 서버로 보낼 문자열을 만든다."""
    from vlm.pilot_vlm import train_rows_digest

    return {
        "uni-client-tag": str(client_tag),
        "uni-model-id": str(m["model_id"]),
        # 클라이언트가 **실제로 올린** 판 — 서버가 등록값과 맞댄다. 판을 적지 않았으면 빈 문자열이다.
        "uni-model-revision": "" if m.get("model_revision") is None else str(m["model_revision"]),
        "uni-train-rows-digest": train_rows_digest(list(rows)),
        "uni-grid-observed": _canonical(m.get("train_input_grid_observed")),
        "uni-processor-config-sha256": str(m["processor_config_sha256"]),
        "uni-config-blocks": _canonical(m["config_blocks"]),
        "uni-impl-ids": _canonical(m["impl_ids"]),
        "uni-prompt-sha256": str(m["prompt_sha256"]),
        "uni-template-mode": str(m["template_mode"]),
        "uni-coord-cfg-hash": str(m["coord_cfg_hash"]),
        "uni-target-contract-sha256": str(m["target_contract_sha256"]),
    }


def assert_fresh_ledger(path: str | Path) -> None:
    """원장에 데이터 행이 하나라도 있으면 시작하지 않는다(리허설 2판 §2-6 — 연합은 같은 원장에 이어 쓰지 않는다)."""
    if last_data_row(path) is not None:
        raise UniFedRejected("fed_ledger_not_fresh",
                             f"통합형 연합 원장에 이미 행이 있다: {path}. 새 스탬프와 새 출력 폴더로 다시 돈다")


def check_fed_ledger(path: str | Path, *, num_rounds: int, num_clients: int, closed: bool) -> None:
    """연합 원장의 라운드 행 규칙(리허설 2판 §2-6). 끝 행을 뺀 **데이터 행**에 건다.

    ① `client_id="server"` 행의 `round` 집합이 정확히 `{0, …, R−1}` 이다.
    ② 같은 `(round, client_id, metric_name)` 이 둘이면 겹친 행으로 읽지 않고 거부한다.
    ③ 라운드마다 `optimizer_steps` 행을 가진 정수 `client_id` 집합이 정확히 `{0, …, K−1}` 이다(`-1` 은 받지 않는다).
    ④ (`closed` 면) 마지막 데이터 행이 끝 행이고 `client_id="server"` · `round = R` · 값 `= max(①)+1 = R` 이다.
    """
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if any(r.values())]
    data = rows
    if closed:
        if not rows or rows[-1]["metric_name"] != END_METRIC:
            raise UniFedRejected("fed_end_missing", "닫힌 원장이어야 하는데 마지막 행이 끝 행이 아니다")
        end, data = rows[-1], rows[:-1]
    if any(r["metric_name"] == END_METRIC for r in data):
        raise UniFedRejected("fed_end_misplaced", "끝 행이 마지막이 아닌 자리에 있다")
    seen: set[tuple[str, str, str]] = set()
    for r in data:
        key = (r["round"], r["client_id"], r["metric_name"])
        if key in seen:
            raise UniFedRejected("fed_duplicate_row", f"같은 행이 둘이다: {key}")
        seen.add(key)
    server_rounds = {int(r["round"]) for r in data if r["client_id"] == "server"}
    if server_rounds != set(range(num_rounds)):
        raise UniFedRejected("fed_server_rounds", f"server 행의 라운드 {sorted(server_rounds)} ≠ 0..{num_rounds - 1}")
    for rd in range(num_rounds):
        ids = {r["client_id"] for r in data if r["metric_name"] == "optimizer_steps" and int(r["round"]) == rd}
        want = {str(i) for i in range(num_clients)}
        if ids != want:
            raise UniFedRejected("fed_client_ids", f"라운드 {rd} 의 참여자 {sorted(ids)} ≠ {sorted(want)}")
    if closed:
        v = float(end["metric_value"])
        if (end["client_id"] != "server" or int(end["round"]) != num_rounds
                or not v.is_integer() or int(v) != max(server_rounds) + 1):
            raise UniFedRejected("fed_end_value", f"끝 행이 server · round {num_rounds} · 값 {num_rounds} 가 아니다: {end}")


def fed_identity(*, run_id: str, seed_index: int | None, seed_value: int, purpose: str, num_rounds: int,
                 local_epochs: int, total_epochs: int, model_id: str, model_revision: str | None,
                 pairs_digest: str | None, prompt_sha256: str, chat_template_kwargs: Mapping[str, Any],
                 coord_space: str, init_digest: str, snapshot_digest: str, plan_sha256: str | None,
                 client_tags: Sequence[str], processor_kwargs: Mapping[str, Any] | None,
                 expected_supervised_tokens: Mapping[str, int] | None) -> dict[str, Any]:
    """연합 칸 meta 의 `identity` — **서버가 쓰고 실행기가 되짚는** 한 함수다(검수 14번 I-1).

    값은 모두 등록값(설정 · 실행 설정)과 서버가 낸 값에서 온다. 클라이언트가 보고한 값은 `finish` 가 이것과 맞댄 뒤에만
    기록에 오른다 — 그래서 실행기는 모델을 올리지 않고 설정만으로 같은 신원을 다시 낼 수 있다.
    """
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.pilot_vlm import (GRAD_ACCUM, LR, MICRO_BATCH, SHUFFLE_POLICY, TARGET_CONTRACT_SHA256,
                               template_mode_of)
    from vlm.schedule import LRF

    return {
        "run_id": str(run_id), "tag": FED_TAG, "cell": FED_TAG, "client": None,
        "seed_index": seed_index, "seed_value": int(seed_value), "purpose": purpose,
        "budget": {"num_rounds": int(num_rounds), "local_epochs": int(local_epochs),
                   "total_epochs": int(total_epochs)},
        "model_id": model_id, "model_revision": model_revision,
        "pairs_digest": pairs_digest, "prompt_sha256": prompt_sha256,
        "template_mode": template_mode_of(chat_template_kwargs), "shuffle_policy": SHUFFLE_POLICY,
        "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space=coord_space)),
        "target_contract_sha256": TARGET_CONTRACT_SHA256,
        "init_adapter_digest": init_digest,
        "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM, "lr0": LR, "lrf": LRF,
        "snapshot_digest": snapshot_digest, "plan_sha256": plan_sha256,
        "client_tags": [str(c) for c in client_tags],
        "processor_kwargs": None if processor_kwargs is None else dict(processor_kwargs),
        "expected_supervised_tokens": (None if expected_supervised_tokens is None
                                       else {str(k): int(v) for k, v in expected_supervised_tokens.items()}),
    }


def check_fed_prior(out_dir: str | Path, *, identity: Mapping[str, Any], run_id: str, num_rounds: int,
                    num_clients: int) -> dict[str, Any]:
    """닫힌 연합 산출물이 **지금 요청의 것인지** — 로컬 · 중앙의 `_verify_prior` 와 같은 재료로 본다(검수 14번 I-1).

    ① meta 의 `identity_sha256` 이 그 `identity` 에서 다시 난다 ② `identity` 가 지금 요청의 것과 같다(초기 어댑터 digest ·
    모델 판 · 페어 · 프롬프트 · 프로세서 설정 · 토큰 기대값 · 예산이 모두 그 안에 있다) ③ 실행 id ④ 어댑터 파일의 sha 가 meta 와 같다
    ⑤ 어댑터 스텝이 R 이다 ⑥ 원장이 규칙 ①~④ 로 닫혔다. 하나라도 어긋나면 `UniFedRejected`. 돌려주는 값은 meta 다.
    """
    from vlm.train_cell import ADAPTER_FILE, META_FILE, _sha, _sha256_file

    d = Path(out_dir)
    meta_p = d / META_FILE
    if not meta_p.exists():
        raise UniFedRejected("meta_missing", f"연합 어댑터는 있는데 meta 가 없다: {d}")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    old, old_sha = meta.get("identity"), meta.get("identity_sha256")
    if not isinstance(old, dict) or not isinstance(old_sha, str):
        raise UniFedRejected("identity_missing", f"앞 연합 산출물의 meta 에 identity · identity_sha256 이 없다: {d}")
    if _sha(old) != old_sha:
        raise UniFedRejected("identity_sha_broken", f"앞 연합 산출물 meta 의 identity_sha256 이 그 identity 와 맞지 않는다: {d}")
    want = dict(identity)
    if old != want:
        diff = sorted(k for k in set(old) | set(want) if old.get(k) != want.get(k))
        raise UniFedRejected("identity_mismatch", f"다른 실행의 연합 산출물이 있다: {d} — 다른 칸 {diff}")
    if meta.get("train_run_id") != run_id:
        raise UniFedRejected("identity_mismatch", f"연합 산출물의 실행 id {meta.get('train_run_id')} ≠ {run_id}")
    if _sha256_file(d / ADAPTER_FILE) != meta.get("adapter_sha256"):
        raise UniFedRejected("adapter_sha_mismatch", f"연합 어댑터 파일의 sha 가 meta 와 다르다: {d}")
    if meta.get("adapter_step") != int(num_rounds):
        raise UniFedRejected("fed_step", f"연합 어댑터 스텝 {meta.get('adapter_step')} ≠ R {num_rounds}")
    check_fed_ledger(d / "atomic_log.csv", num_rounds=num_rounds, num_clients=num_clients, closed=True)
    return meta


class UniFedRun:
    """통합형 연합 서버 한 실행의 앞뒤. `get(key, default)` 는 서버 설정을 대시 키로 읽는 함수다.

    `out_dir` 은 `resolve()` 하기 **전의** 경로로 준다 — meta 의 원장 경로를 로컬 · 중앙과 같은 꼴(작업 트리 기준 상대)로 적는다.
    """

    def __init__(self, *, get: Callable[[str, Any], Any], out_dir: Path, run_id: str, base_seed: int,
                 split_hash: str, num_rounds: int, local_epochs: int, total_epochs: int,
                 client_tags: Sequence[str], config_path: Path | None = None,
                 non_main_parent: Path | None = None):
        """초기 어댑터를 읽기 **전에** 만든다 — 목적과 계획의 검사(2판 §1-4 의 순서 5)와 본실험의 설정 대조가
        모델 적재보다 앞선다. 본실험은 필수 키가 비면(`client_run_cfg`) · 값이 설정 파일과 다르면 · 출력이 비본실험
        부모 아래면 시작하지 않는다 — 실행 설정의 옛 기본값(파일럿의 R/E/N · 시드 · 출력 루트)으로 내려가지 않는다.
        `config_path` · `non_main_parent` 는 시험이 임시 설정 · 임시 부모를 줄 자리다(파이썬 인자로만).

        **무엇보다 먼저** 고의 중단 변수를 본다 — 연합의 서버는 목적과 무관하게 거부한다(2판 §1-3 의 연합 ③). 상주 SuperLink 의
        자식은 띄운 쪽이 아니라 데몬의 환경을 물려받으므로 받는 쪽에서도 막는다."""
        from vlm.fault import FaultRefused, assert_no_fault_env

        try:
            assert_no_fault_env(who="통합형 연합 서버는")
        except FaultRefused as exc:
            raise UniFedRejected(exc.code, str(exc)) from None
        self.uni = client_run_cfg(down_config(get))
        self.purpose = self.uni["purpose"]
        plan = self.uni["plan"]
        if self.purpose == "main" and plan:
            raise UniFedRejected("main_with_plan", "본실험은 계획 파일을 받지 않는다")
        if self.purpose != "main" and not plan:
            raise UniFedRejected("plan_missing", f"{self.purpose} 는 계획 파일이 있어야 한다")
        raw_idx = str(get("uni-seed-index", "") or "")
        self.seed_index = int(raw_idx) if raw_idx else None
        self.out_dir = Path(out_dir)
        resume = str(get("resume-root", "") or "")
        self.resume_root = Path(resume) if resume else None
        init_file = str(get("uni-init-adapter", "") or "")
        if self.purpose == "main" and init_file:
            raise UniFedRejected("main_with_init_adapter", "본실험은 초기 어댑터 파일을 받지 않는다 — 실행기의 캐시 자리 하나다")
        from vlm.run_root import run_root_problems

        # 순서 4 — 출력 · 재개 · 계획 파일 경로가 모두 루트 아래다(2판 §1-4). 서버는 재개 폴더를 클라이언트에 내려보내고
        # 클라이언트는 그 아래에 체크포인트를 쓴다. 목록 · 등록 · 영수증은 연합 서버가 받는 경로가 아니다.
        train_lists = self.uni.get("train_lists")
        paths = [self.out_dir] + ([self.resume_root] if self.resume_root else []) + ([Path(plan)] if plan else []) \
            + ([Path(init_file)] if init_file else []) + ([Path(train_lists)] if train_lists else [])
        bad_root = run_root_problems(self.purpose, self.uni["rehearsal_root"], paths, parent=non_main_parent)
        if bad_root:
            raise UniFedRejected("run_root", " · ".join(bad_root))
        # 계획 파일은 루트 검사를 지난 뒤에 읽는다.
        self.plan_sha256 = hashlib.sha256(Path(plan).read_bytes()).hexdigest() if plan else None
        self.run_id = str(run_id)
        self.base_seed = int(base_seed)
        self.split_hash = str(split_hash)
        self.num_rounds, self.local_epochs, self.total_epochs = int(num_rounds), int(local_epochs), int(total_epochs)
        if self.total_epochs != self.num_rounds * self.local_epochs:
            raise UniFedRejected("budget", f"total {self.total_epochs} ≠ R {self.num_rounds} × E {self.local_epochs}")
        self.client_tags = [str(t) for t in client_tags]
        from vlm.pilot_vlm import MODEL_ID, load_prompt

        # 서버가 정본으로 두는 값 — 클라이언트의 보고를 이것과 맞댄다(검수 14번 I-2). 비었으면 파일럿 기본값이다(리허설 · 진단만).
        self.model_id = self.uni["model_id"] or MODEL_ID
        self.model_revision = self.uni["model_revision"]
        _, self.prompt_sha256 = load_prompt(self.uni["prompt_path"])
        if self.purpose == "main":
            from vlm.uni_config import load_uni_config, main_run_problems

            bad = main_run_problems(
                load_uni_config(config_path), model_id=self.uni["model_id"],
                model_revision=self.uni["model_revision"], pairs_path=self.uni["pairs_path"],
                pairs_digest=self.uni["pairs_digest"], chat_template_kwargs=self.uni["chat_template_kwargs"],
                num_rounds=self.num_rounds, local_epochs=self.local_epochs, total_epochs=self.total_epochs,
                seed_index=self.seed_index, seed_value=self.base_seed, snapshot_digest=self.split_hash,
                prompt_path=self.uni["prompt_path"], coord_space=self.uni["coord_space"],
                output_paths=[self.out_dir] + ([self.resume_root] if self.resume_root else []),
                processor_kwargs=self.uni["processor_kwargs"],
                expected_supervised_tokens=self.uni["expected_supervised_tokens"])
            if bad:
                raise UniFedRejected("main_config_mismatch", "본실험 설정과 다르다: " + " · ".join(bad))
        exp = self.uni["expected_supervised_tokens"]
        if exp is not None and sorted(exp) != sorted(self.client_tags):
            raise UniFedRejected("tokens_expected_keys", f"감독 토큰 기대값의 참여자 {sorted(exp)} ≠ {self.client_tags}")
        self.canonical_keys: list[str] = []
        self.init_digest: str | None = None
        self.init_tensor_digest: Any = None
        self.started_at = time.time()
        self.rounds: dict[int, dict[int, dict[str, str]]] = {}
        self.final: tuple[int, list] | None = None

    def set_initial(self, arrays: Sequence[Any], canonical_keys: Sequence[str]) -> None:
        """서버가 내려보낼 공통 초기 어댑터. 두 digest 를 여기서 낸다."""
        from vlm.init_adapter import adapter_proof, init_adapter_digest

        arrays = list(arrays)
        self.canonical_keys = list(canonical_keys)
        self.init_digest = init_adapter_digest(arrays)
        self.init_tensor_digest = adapter_proof(arrays, self.canonical_keys)["tensor_digest"]

    def expected_identity(self, init_digest: str) -> dict[str, Any]:
        """이 실행의 meta `identity` — 서버의 등록값으로 낸다. 실행기가 설정으로 같은 값을 낸다(`fed_identity`)."""
        from vlm.pilot_vlm import _template_kwargs

        return fed_identity(
            run_id=self.run_id, seed_index=self.seed_index, seed_value=self.base_seed, purpose=self.purpose,
            num_rounds=self.num_rounds, local_epochs=self.local_epochs, total_epochs=self.total_epochs,
            model_id=self.model_id, model_revision=self.model_revision, pairs_digest=self.uni["pairs_digest"],
            prompt_sha256=self.prompt_sha256, chat_template_kwargs=_template_kwargs(self.uni["chat_template_kwargs"]),
            coord_space=self.uni["coord_space"] or "ABS_ORIG", init_digest=init_digest,
            snapshot_digest=self.split_hash, plan_sha256=self.plan_sha256, client_tags=self.client_tags,
            processor_kwargs=self.uni["processor_kwargs"],
            expected_supervised_tokens=self.uni["expected_supervised_tokens"])

    # -- 라운드마다 ------------------------------------------------------------
    def observe(self, server_round: int, cells: Sequence[Mapping[str, Any]], agg: Any) -> None:
        per: dict[int, dict[str, str]] = {}
        for c in cells:
            idx = int(c.get("client-idx", -1))
            missing = [k for k in CLIENT_STRINGS if k not in c]
            if missing:
                raise UniFedRejected("client_strings_missing", f"라운드 {server_round} 클라이언트 {idx} 가 {missing} 를 보내지 않았다")
            empty = [k for k in CLIENT_STRINGS if k not in EMPTY_OK and not str(c[k])]
            if empty:
                raise UniFedRejected("client_strings_empty", f"라운드 {server_round} 클라이언트 {idx} 의 {empty} 가 비었다")
            if c.get("supervised-tokens") is None:
                raise UniFedRejected("client_tokens_missing", f"라운드 {server_round} 클라이언트 {idx} 가 감독 토큰을 보내지 않았다")
            per[idx] = {k: str(c[k]) for k in CLIENT_STRINGS}
            grid = json.loads(per[idx]["uni-grid-observed"])
            if isinstance(grid, dict) and grid and c.get("num-examples") is not None \
                    and sum(int(v) for v in grid.values()) != int(float(c["num-examples"])):
                # 로컬 칸과 같은 대조 — 관측 격자가 이 라운드의 학습 행을 모두 셌는가(`vlm/train_cell.py`).
                raise UniFedRejected("grid_rows_mismatch",
                                     f"라운드 {server_round} 클라이언트 {idx} 의 격자 합 ≠ 학습 행 {c['num-examples']}")
            # 감독 토큰은 라운드마다 참여자의 행 전체를 E 번 돈 값이다 — 기대값이 있으면 그 라운드에서 맞댄다(12번 §2-4).
            per[idx]["supervised-tokens"] = str(int(float(c["supervised-tokens"])))
        self.rounds[int(server_round)] = per
        self.final = (int(server_round), list(agg.ndarrays))

    # -- 감사 통과 뒤 ------------------------------------------------------------
    def _one(self, key: str, *, per_client: bool = False) -> Any:
        """라운드 사이에서 같아야 하는 값. `per_client` 면 참여자마다, 아니면 모든 참여자가 같아야 한다."""
        vals: dict[int, set[str]] = {}
        for per in self.rounds.values():
            for idx, s in per.items():
                vals.setdefault(idx, set()).add(s[key])
        if any(len(v) != 1 for v in vals.values()):
            raise UniFedRejected("client_value_drift", f"{key} 가 라운드 사이에서 바뀌었다")
        by_idx = {i: next(iter(v)) for i, v in vals.items()}
        if per_client:
            return by_idx
        if len(set(by_idx.values())) != 1:
            raise UniFedRejected("client_value_split", f"{key} 가 참여자마다 다르다")
        return next(iter(by_idx.values()))

    def finish(self, *, atomic: Any, report: Any) -> dict[str, Any]:
        """회계 감사가 통과한 뒤에만 부른다. 어댑터와 meta 를 저장하고 원장을 닫는다. 쓴 meta 의 파일 필드를 돌려준다."""
        from vlm.pilot_vlm import _template_kwargs
        from vlm.train_cell import _record_path, _sha, save_adapter_cell, train_config_object

        if not getattr(report, "ok", False):
            raise UniFedRejected("audit_not_ok", "회계 감사를 통과하지 않은 실행의 원장은 닫지 않는다")
        if self.init_digest is None:
            raise UniFedRejected("initial_missing", "공통 초기 어댑터를 받지 않았다(set_initial)")
        R, K = self.num_rounds, len(self.client_tags)
        if sorted(self.rounds) != list(range(1, R + 1)):
            raise UniFedRejected("rounds_observed", f"본 라운드 {sorted(self.rounds)} ≠ 1..{R}")
        if any(sorted(per) != list(range(K)) for per in self.rounds.values()):
            raise UniFedRejected("clients_observed", "라운드마다 모든 참여자를 보지 못했다")
        if self.final is None or self.final[0] != R:
            raise UniFedRejected("final_missing", "마지막 라운드의 글로벌 어댑터가 없다")
        ledger = self.out_dir / "atomic_log.csv"
        check_fed_ledger(ledger, num_rounds=R, num_clients=K, closed=False)

        tags = self._one("uni-client-tag", per_client=True)
        if [tags[i] for i in range(K)] != self.client_tags:
            raise UniFedRejected("client_tags", f"참여자 태그 {tags} ≠ 서버의 {self.client_tags}")
        # 클라이언트가 **실제로 학습한 것**을 서버의 등록값과 맞댄다(검수 14번 I-2). `_one` 은 참여자 사이 · 라운드 사이의
        # 일치만 본다 — 셋이 같은 값으로 등록값과 다르면 여기서 걸린다. 기록에는 맞댄 뒤의 등록값이 오른다.
        for key, want in (("uni-model-id", self.model_id),
                          ("uni-model-revision", "" if self.model_revision is None else str(self.model_revision)),
                          ("uni-prompt-sha256", self.prompt_sha256)):
            got = self._one(key)
            if got != want:
                raise UniFedRejected("client_registration_mismatch", f"클라이언트가 보고한 {key} {got!r} ≠ 서버의 {want!r}")
        model_id = self.model_id
        kwargs = _template_kwargs(self.uni["chat_template_kwargs"])
        coord_space = self.uni["coord_space"] or "ABS_ORIG"
        metrics_like = {
            "processor_config_sha256": self._one("uni-processor-config-sha256"),
            "config_blocks": json.loads(self._one("uni-config-blocks")),
            "prompt_sha256": self._one("uni-prompt-sha256"),
            "template_mode": self._one("uni-template-mode"),
            "coord_cfg_hash": self._one("uni-coord-cfg-hash"),
            "target_contract_sha256": self._one("uni-target-contract-sha256"),
        }
        train_config = train_config_object(
            model_id=model_id, model_revision=self.uni["model_revision"],
            pairs_digest=self.uni["pairs_digest"], chat_template_kwargs=kwargs, coord_space=coord_space,
            num_rounds=R, local_epochs=self.local_epochs, total_epochs=self.total_epochs,
            prompt_sha256=self.prompt_sha256, init_digest=self.init_digest, metrics=metrics_like)
        tc_text = _canonical(train_config)

        grids = {i: json.loads(g) for i, g in self._one("uni-grid-observed", per_client=True).items()}
        if any(not g for g in grids.values()):
            raise UniFedRejected("grid_missing", "관측 격자를 보내지 않은 참여자가 있다")
        grid: dict[str, int] = {}
        for g in grids.values():
            for k, n in g.items():
                grid[k] = grid.get(k, 0) + int(n)
        impl: dict[str, Any] = {}
        # 참여자는 같은 적재기로 학습해야 한다 — 식별자가 참여자마다 다르면 합치지 않고 거부한다(`_one` 의 `client_value_split`).
        for item in json.loads(self._one("uni-impl-ids")):
            impl[_canonical(item)] = item
        proof_p = self.out_dir / "initial.proof.json"
        if proof_p.exists():
            for item in json.loads(proof_p.read_text(encoding="utf-8")).get("impl_ids") or []:
                impl[_canonical(item)] = item
        rows_digest = self._one("uni-train-rows-digest", per_client=True)
        exp = self.uni["expected_supervised_tokens"]
        tokens = self._one("supervised-tokens", per_client=True)
        if exp is not None:
            for i, tag in enumerate(self.client_tags):
                want = int(exp[tag]) * self.local_epochs
                if int(tokens[i]) != want:
                    raise UniFedRejected("supervised_tokens", f"{tag} 의 라운드당 감독 토큰 {tokens[i]} ≠ 기대값 {want}")

        identity = self.expected_identity(self.init_digest)
        sha = _sha(identity)
        meta = {
            "adapter_step": R, "adapter_step_unit": "fed_rounds",
            "identity": identity, "identity_sha256": sha,
            # 세 참여자 값을 참여자 이름 순으로 싣는다(리허설 2판 §2-7 의 연합 칸).
            "train_rows_digest": {self.client_tags[i]: rows_digest[i] for i in sorted(rows_digest,
                                                                                      key=lambda i: self.client_tags[i])},
            "train_input_grid_observed": grid,
            "supervised_tokens_per_round": {self.client_tags[i]: int(tokens[i]) for i in range(K)},
            "supervised_tokens_expected": (None if exp is None
                                           else {t: int(exp[t]) * self.local_epochs for t in self.client_tags}),
            "processor_config_sha256": metrics_like["processor_config_sha256"],
            "coord_cfg_hash": metrics_like["coord_cfg_hash"],
            # 연합은 재개가 없다 — 서버 프로세스 하나다. 단위는 라운드 번호(0 부터)다.
            "resume_segments": [{"start_epoch": 0, "end_epoch": R - 1, "unit": "fed_rounds",
                                 "identity_sha256": sha, "started_at": self.started_at, "ended_at": time.time()}],
            "train_config": tc_text,
            "train_config_sha256": hashlib.sha256(tc_text.encode("utf-8")).hexdigest(),
            "init_adapter_digest": self.init_digest,
            "init_adapter_tensor_digest": self.init_tensor_digest,
            "train_run_id": self.run_id,
            "train_ledger_path": _record_path(ledger),
            "metrics": {"audit": report.as_dict() if hasattr(report, "as_dict") else dict(report),
                        "rounds": R, "clients": K},
            "impl_ids": [impl[k] for k in sorted(impl)],
            "fault_events": [],
        }
        if self.purpose != "main":
            meta["plan_sha256"] = self.plan_sha256
        files = save_adapter_cell(self.out_dir, self.final[1], self.canonical_keys, meta)
        atomic.close(client_id="server", step=R)
        check_fed_ledger(ledger, num_rounds=R, num_clients=K, closed=True)
        return files
