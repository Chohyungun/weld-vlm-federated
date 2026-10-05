"""1.4 묶음 검증 — 07번 미니스펙 §13-3 · §16-1 · §16-2 · §18-5.

거부 코드는 **하나하나가 실제로 서야** 뜻이 있다. "거부됐다"만 보는 시험은 조건이 사라진 것을
못 잡고, 조건이 사라진 검증기는 통과시키는 기계일 뿐이다. 그래서 여기서는 코드별로 걸고,
마지막에 **어느 코드를 덮었는지 세어** 표와 맞댄다.

단계 순서도 성질이다. 바이트가 깨진 파일의 내용을 해석하면 오류가 원인을 가린다 —
앞 단계가 걸리면 뒤 단계 사유가 섞여 나오지 않아야 한다.

맨 끝의 메타 시험은 모듈 전역(`COVERED`)을 읽으므로 **이 파일이 한 프로세스에서 돌아야 한다.**
지금 이 저장소에는 시험을 나눠 돌리는 플러그인이 없다. 나중에 들어와 나눠 돌리면 그 시험은
조용히 통과하지 않고 크게 실패한다(덮은 집합이 부분집합이 되므로) — 그때 한 워커로 묶으면 된다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import datetime
from pathlib import Path

import pytest

from evaluation.bundle_v14 import (
    ATTEMPT_REASONS,
    SIDECAR_ECHO_ABSENT,
    ATTEMPTS_COMMON_REQUIRED,
    ATTEMPTS_END_REQUIRED,
    ATTEMPTS_START_REQUIRED,
    ENV_FIELDS,
    EVENT_END,
    EVENT_START,
    MODE_ECHO,
    MODE_MODEL,
    REASON_FIRST,
    REASON_RESUME,
    REASON_TORN_TAIL,
    SIDECAR_REQUIRED,
    START_STAMP_ECHO_ABSENT,
    START_STAMP_REQUIRED,
    BundleFiles,
    LedgerView,
    Registration,
    TrainingArtifact,
    VerifiedBundle,
    check_env_consistency,
    population_digest,
    verify_bundle,
)
from evaluation.actuals import impl_key
from evaluation.eval_list import canonical_bytes, set_digest, validate_eval_list
from evaluation.prereg_unified import SPLIT_OF_PURPOSE
from evaluation.reject_v14 import STAGE_OF, BundleRejected, RejectCode, Stage

IDS = ["aihub000001", "aihub000002", "aihub000003"]
IDS9 = [f"aihub{i:06d}" for i in range(1, 10)]
"""회계 시험용 — 자리 차가 뜻을 가지려면 목록이 몇 개는 돼야 한다."""
WH = [1280, 704]
PATCH = 16
GRID_KEY = "[1, 44, 80]"      # 44*16 = 704, 80*16 = 1280
#: 시각에는 **초 아래 자릿수**가 있어야 한다(§28-4).
T_START = "2026-09-21T10:00:00.000000+09:00"
T_END = "2026-09-21T11:00:00.000000+09:00"
T_STAMP = "2026-09-21T09:30:00.000000+09:00"
RECEIPT = datetime.fromisoformat("2026-09-21T09:00:00+09:00")
DEVICE = "NVIDIA GeForce RTX 5060 Ti:0"
"""§28-3 나 — 이름과 번호를 함께. 번호만으로는 다른 장비가 같은 값이 된다."""


def at(minute: int, second: int = 0) -> str:
    """시도마다 다른 시각. 오프셋과 초 아래 자릿수를 함께 준다."""
    return f"2026-09-21T10:{minute:02d}:{second:02d}.000000+09:00"

COVERED: set[RejectCode] = set()
"""이 파일이 실제로 세운 코드 — 이름이 나온 것이 아니라 **거부로 실제로 올라온 것**이다.
맨 끝의 메타 시험이 표와 맞댄다."""


def hx(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def impl(seam: str, qualname: str, *, approved: bool = True, in_repo: bool = True,
         blob: str | None = "a" * 40, head: str | None = "a" * 40) -> dict:
    """쓰는 쪽 `vlm.seams.impl_id` 의 꼴 — 리허설 2판 반영판 §1-4."""
    return {"seam": seam, "approved": approved, "module": "vlm.export_run", "qualname": qualname,
            "source_path": "vlm/export_run.py", "in_repo": in_repo,
            "blob_sha1": blob, "head_blob_sha1": head}


IMPL_IDS = {"export_preflight": impl("export_preflight", "preflight"),
            "export_generator": impl("export_generator", "load_generator")}
"""곁 파일의 `impl_ids` — export 는 이음새 이름 → 식별자 사전으로 낸다(`vlm/export_run.py`)."""
IMPL_KEYS = tuple(sorted(impl_key(d) for d in IMPL_IDS.values()))
"""등록의 `generation.impl_ids` — 식별자마다 열쇠 하나."""


def projection(purpose: str, elist) -> dict:
    """그 묶음의 모드로 낸 등록 투영의 최소 — 목적 · 구현 열쇠 · 목록 해시. 다른 항목은 맞대지 않는다."""
    return {"purpose": purpose, "impl_ids": IMPL_KEYS,
            "eval_list_file_sha256": elist.file_sha256, "eval_list_set_sha256": elist.set_sha256}


def row(image_id: str, **over) -> dict:
    r = {"image_id": image_id, "coord_space": "ABS_ORIG", "coord_cfg_hash": hx("coord"),
         "model_input_wh": list(WH), "gen_stop": "eos", "image_sha256": hx(image_id),
         "image_grid_thw": [1, 44, 80]}      # GRID_KEY — 쓰는 쪽은 줄마다 관측 격자를 싣는다
    r.update(over)
    return r


def body_of(rows) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


CHAT_KW = {"add_generation_prompt": True}


def train_config_text(**over) -> str:
    """학습 설정 원문 — 정규화 JSON. 대응 표(리허설 미니스펙 2판 반영판 2-5)의 왼쪽 키가 곁 파일의 값과 맞는다."""
    cfg = {"prompt_sha256": hx("prompt"), "template_mode": "add_generation_prompt=True",
           "coord_cfg_hash": hx("coord"), "model_id": "local/qwen3.5-4b", "model_revision": "frozen",
           "num_rounds": 3, "local_epochs": 10, "total_epochs": 30, "init_adapter_digest": hx("init"),
           "processor_config_sha256": hx("proc"), "lora": {"r": 16}, "pairs_digest": hx("pairs")}
    cfg.update(over)
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def tokens_of(ids, n: int) -> bytes:
    return body_of({"image_id": i, "token_ids": [1] * n, "token_logprobs": [-0.5] * n} for i in ids)


ADAPTER = hx("adapter")


def start_stamp_of(tag: str, seed_index: int, echo: bool, elist) -> dict:
    """**계약을 지킨** 시작 도장 — 모델 아홉 / 에코 일곱(§26-3 · §28-3 가, 승인).

    앞 판의 정상 자료는 `{"ok": true}` 였다 — 검사 대상 필드가 처음부터 없어서 그것으로
    통과한 시험은 내용 대조 경로를 확인하지 못했다.
    """
    s = {"tag": tag, "seed_index": seed_index,
         "mode": MODE_ECHO if echo else MODE_MODEL,
         "generation_sha256": hx("gen_fp"),
         "eval_list_file_sha256": elist.file_sha256,
         "eval_list_set_sha256": elist.set_sha256,
         "started_at": T_STAMP, "device": DEVICE, "purpose": "main"}
    if not echo:
        s["scored_adapter_sha256"] = ADAPTER
        s["train_run_id"] = "run_c1"
        s["train_ledger_sha256"] = hx("ledger")
    return s


def att_start(tag: str, seed_index: int, no: int, stamp_sha: str, first_id, **over) -> dict:
    """**계약을 지킨** 시작 줄(§27-1 · §28-3 나). 시도가 시작할 때 그 시도가 적는다."""
    r = {"event": EVENT_START, "tag": tag, "seed_index": seed_index, "attempt_no": no,
         "start_stamp_sha256": stamp_sha,
         "reason": REASON_FIRST if no == 1 else REASON_RESUME,
         "started_at": at(2 * no), "first_id": first_id, "device": DEVICE}
    r.update(over)
    return r


def att_end(tag: str, seed_index: int, no: int, stamp_sha: str, last_id, n_written: int,
            **over) -> dict:
    """**계약을 지킨** 끝 줄(§27-1). 시도가 스스로 끝날 때 그 시도가 적는다.

    **죽은 시도에는 이 줄이 없다** — 다시 시작한 쪽이 대신 적지 않는다.
    """
    r = {"event": EVENT_END, "tag": tag, "seed_index": seed_index, "attempt_no": no,
         "start_stamp_sha256": stamp_sha,
         "last_id": last_id, "finished_at": at(2 * no + 1), "n_written": n_written}
    r.update(over)
    return r


def plan_rows(tag: str, seed_index: int, stamp_sha: str, ids, plan) -> list[dict]:
    """경우를 줄로 옮긴다.

    `plan` 은 시도마다 `(first_pos, reason, n_written)` 이다. `first_pos` 가 `None` 이면
    빈 구간이고, `n_written` 이 `None` 이면 **끝 줄이 없다**(죽은 시도).
    """
    rows: list[dict] = []
    for no, (first_pos, reason, n_written) in enumerate(plan, 1):
        first_id = None if first_pos is None else ids[first_pos]
        over = {} if reason is None else {"reason": reason}
        rows.append(att_start(tag, seed_index, no, stamp_sha, first_id, **over))
        if n_written is not None:
            last_id = None if n_written == 0 else ids[min(len(ids) - 1, first_pos + n_written - 1)]
            rows.append(att_end(tag, seed_index, no, stamp_sha, last_id, n_written))
    return rows


def jsonl_bytes(rows) -> bytes:
    """줄 단위 파일은 **바이트로** 쓴다 — `write_text` 는 Windows 에서 LF 를 CRLF 로 바꾼다."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


@dataclass
class Fixture:
    files: BundleFiles
    registration: Registration
    artifact: TrainingArtifact | None
    ledger: LedgerView | None
    receipt: datetime | None = RECEIPT

    def verify(self, **over):
        kw = {"registration": self.registration, "artifact": self.artifact,
              "ledger": self.ledger, "receipt_time": self.receipt}
        kw.update(over)
        return verify_bundle(self.files, **kw)

    def rejects(self, **over) -> set[RejectCode]:
        with pytest.raises(BundleRejected) as exc:
            self.verify(**over)
        codes = exc.value.codes()
        COVERED.update(codes)
        return codes

    def detail(self, **over) -> dict:
        with pytest.raises(BundleRejected) as exc:
            self.verify(**over)
        COVERED.update(exc.value.codes())
        return exc.value.rejections[0].detail


def make(tmp: Path, *, tag: str = "uni_local_C1", seed_index: int = 1, echo: bool = False,
         ids=None, list_ids=None, rows=None, raw: bytes | None = None,
         meta_over: dict | None = None, drop_meta=(), tokens: bytes | None = None,
         no_attempts: bool = False, no_start: bool = False,
         start_over: dict | None = None, drop_start=(), start_raw: bytes | None = None,
         attempts_rows=None, attempts_raw: bytes | None = None, attempts_plan=None,
         attempts_edit=None, side_raw_tail: bytes = b"", purpose: str = "main") -> Fixture:
    """계약을 지키는 묶음 하나를 만든다. 시험이 무너뜨리고 싶은 곳만 덮어쓴다."""
    from evaluation.schema_v14 import tag_parts

    ids = IDS if ids is None else ids
    tmp.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_s{seed_index}" + (".echo" if echo else "")
    gen = tmp / f"{stem}.generations.jsonl"
    gen.write_bytes(raw if raw is not None else body_of(rows or [row(i) for i in ids]))

    # 목록을 **먼저** 만든다 — 도장과 시도 행이 그 해시·id 를 싣는다.
    elist = validate_eval_list(canonical_bytes(list_ids if list_ids is not None else ids))
    lids = list(elist.ids) or list(ids) or [IDS[0]]

    start = tmp / f"{stem}.export_start.json"
    if not no_start:
        if start_raw is not None:
            start.write_bytes(start_raw)
        else:
            stamp = start_stamp_of(tag, seed_index, echo, elist)
            stamp["purpose"] = purpose
            for k in drop_start:
                stamp.pop(k, None)
            stamp.update(start_over or {})
            start.write_bytes(json.dumps(stamp, ensure_ascii=False).encode("utf-8"))
    # 시도 줄은 **도장 해시**를 싣는다(§27-5) — 도장을 쓴 뒤에 만든다.
    stamp_sha = hashlib.sha256(start.read_bytes() if start.exists() else b"").hexdigest()
    a_rows = None
    if not no_attempts:
        att = tmp / "attempts.jsonl"
        if attempts_raw is not None:
            att.write_bytes(attempts_raw)
        else:
            if attempts_rows is not None:
                a_rows = list(attempts_rows)
            elif attempts_plan is not None:
                a_rows = plan_rows(tag, seed_index, stamp_sha, lids, attempts_plan)
            else:
                a_rows = plan_rows(tag, seed_index, stamp_sha, lids,
                                   [(0, None, len(lids))])
            if attempts_edit is not None:
                # 도장 해시는 `make` 만 아는 값이다 — 줄을 만든 뒤에 손보게 한다.
                a_rows = attempts_edit(a_rows) or a_rows
            att.write_bytes(jsonl_bytes(a_rows))
    if tokens is not None:
        (tmp / f"{stem}.tokens.jsonl").write_bytes(tokens)

    # 곁 파일의 시각·장비는 **시도 줄에서 끌어온다**(§27-6 의 관계 4·5 · §28-3 나).
    # 쓰는 쪽이 그 값을 옮겨 적는다 — 시험이 그 순서를 따른다.
    t_first, t_last, dev_last = T_START, T_END, DEVICE
    if a_rows:
        st = [r for r in a_rows if r.get("event") == EVENT_START]
        en = [r for r in a_rows if r.get("event") == EVENT_END]
        if st:
            t_first = st[0].get("started_at", T_START)
            dev_last = st[-1].get("device", DEVICE)
        if en:
            t_last = en[-1].get("finished_at", T_END)

    n_lines = len(gen.read_bytes().decode("utf-8", "ignore").rstrip("\n").split("\n")) if gen.read_bytes() else 0
    cell, client = tag_parts(tag)
    meta = {
        "cell": cell, "client": client, "tag": tag,
        "seed_index": seed_index, "seed_value": 20260828,
        "mode": MODE_ECHO if echo else MODE_MODEL, "n_lines": n_lines,
        "generations_sha256": hashlib.sha256(gen.read_bytes()).hexdigest(),
        "eval_list_file_sha256": elist.file_sha256,
        "eval_list_set_sha256": elist.set_sha256,
        "start_stamp_sha256": hashlib.sha256(
            start.read_bytes() if start.exists() else b"").hexdigest(),
        "snapshot_digest": hx("snap"),
        "coord_space": "ABS_ORIG", "coord_cfg_hash": hx("coord"),
        "generation_sha256": hx("gen_fp"), "prompt_sha256": hx("prompt"),
        "chat_template_kwargs": {"add_generation_prompt": True},
        "gen_prefix_sha256": hx("prefix"), "max_new_tokens": 256,
        "decoding": {"do_sample": False}, "batch_size": 8, "padding_side": "left",
        "processor_config_sha256": hx("proc"),
        "processor_min_pixels": 256, "processor_max_pixels": 1280 * 704,
        "patch_size": PATCH, "merge_size": 2,
        "image_grid_thw_observed": {GRID_KEY: n_lines},
        "transformers_version": "4.57.0",
        "base_model_id": "local/qwen3.5-4b", "base_model_revision": "frozen",
        "train_run_id": "run_c1", "train_ledger_path": "ledger.csv",
        "train_ledger_sha256": hx("ledger"),
        "train_config": train_config_text(),
        "train_config_sha256": hashlib.sha256(train_config_text().encode("utf-8")).hexdigest(),
        "budget_n": 30, "budget_r": 3, "budget_e": 10,
        "purpose": purpose, "impl_ids": IMPL_IDS, "fault_events": [], "list_split": SPLIT_OF_PURPOSE[purpose],
        "start_checkpoint_sha256": None, "init_adapter_digest": hx("init"),
        "export_commit": "0" * 40, "started_at": t_first, "finished_at": t_last,
        "device": dev_last,
        "attempts": len([r for r in (a_rows or [])
                         if r.get("event") == EVENT_START
                         and r.get("start_stamp_sha256") == stamp_sha]) or 1,
    }
    if not echo:
        meta["scored_adapter_sha256"] = hx("adapter")
        meta["scored_adapter_step"] = 100
    else:
        # 에코 곁 파일에도 학습 원장 참조가 없어야 한다(2026-09-30 결정).
        for k in SIDECAR_ECHO_ABSENT:
            meta.pop(k, None)
    if tokens is not None:
        meta["tokens_sha256"] = hashlib.sha256(tokens).hexdigest()
    for k in drop_meta:
        meta.pop(k, None)
    meta.update(meta_over or {})
    (tmp / f"{stem}.export_meta.json").write_bytes(
        json.dumps(meta, ensure_ascii=False).encode("utf-8") + side_raw_tail)

    reg = Registration(generation_sha256=hx("gen_fp"), coupling_rule="decoupled_v2",
                       seeds={seed_index: 20260828}, values=projection(purpose, elist), eval_list=elist, purpose=purpose,
                       echo_list=elist if echo else None, coord_space="ABS_ORIG")
    art = None if echo else TrainingArtifact(ADAPTER, 100)
    led = None if echo else LedgerView("run_c1", hx("ledger"), 100, cell, client)
    return Fixture(BundleFiles(gen), reg, art, led)


# ---------------------------------------------------------------- 통과

def test_계약을_지킨_묶음은_통과한다(tmp_path: Path) -> None:
    got = make(tmp_path).verify()
    assert got.tag == "uni_local_C1"
    assert (got.cell, got.client) == ("uni_local", "C1")
    assert (got.seed_index, got.seed_value) == (1, 20260828)
    assert got.mode == MODE_MODEL and got.is_echo is False
    assert got.image_ids == tuple(IDS)
    assert len(got.lines) == 3
    assert got.notes["coupling_rule"] == "decoupled_v2"
    assert got.notes["echo_exempt"] == (), "모델 묶음에는 면제가 없다"   # 동결로 튜플이다


def test_에코_묶음은_어댑터_없이_통과한다(tmp_path: Path) -> None:
    got = make(tmp_path, echo=True).verify()
    assert got.is_echo and got.mode == MODE_ECHO
    assert got.notes["echo_exempt"], "에코에는 면제 목록이 실린다"


def test_중앙_칸은_참여자가_없다(tmp_path: Path) -> None:
    assert make(tmp_path, tag="uni_central").verify().client is None


# ---------------------------------------------------------------- A 존재·스키마

def test_파일_이름이_계약의_꼴이_아니면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    bad = f.files.generations.with_name("uni_local_C1.generations.jsonl")
    f.files.generations.rename(bad)
    with pytest.raises(BundleRejected) as exc:
        verify_bundle(BundleFiles(bad), registration=f.registration, artifact=f.artifact,
                      ledger=f.ledger, receipt_time=RECEIPT)
    COVERED.update(exc.value.codes())
    assert exc.value.codes() == {RejectCode.TAG_MISMATCH}


def test_본실험_태그가_아니면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    bad = f.files.generations.with_name("sep_central_s1.generations.jsonl")
    f.files.generations.rename(bad)
    with pytest.raises(BundleRejected) as exc:
        verify_bundle(BundleFiles(bad), registration=f.registration, artifact=f.artifact,
                      ledger=f.ledger, receipt_time=RECEIPT)
    assert exc.value.codes() == {RejectCode.TAG_MISMATCH}


def test_곁_파일이_없으면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    f.files.sidecar.unlink()
    assert f.rejects() == {RejectCode.SIDECAR_MISSING}


def test_출발_각인이_없으면_거부한다(tmp_path: Path) -> None:
    assert make(tmp_path, no_start=True).rejects() == {RejectCode.START_STAMP_MISSING}


def test_시도_기록이_없으면_거부한다(tmp_path: Path) -> None:
    assert make(tmp_path, no_attempts=True).rejects() == {RejectCode.ATTEMPTS_MISSING}


def test_영수증_없이는_채점하지_않는다(tmp_path: Path) -> None:
    assert make(tmp_path).rejects(receipt_time=None) == {RejectCode.RECEIPT_MISSING}


def test_존재_단계가_걸리면_뒤를_보지_않는다(tmp_path: Path) -> None:
    """없는 파일의 해시를 셀 수 없다."""
    f = make(tmp_path, no_attempts=True, meta_over={"generations_sha256": hx("틀린값")})
    assert f.rejects() == {RejectCode.ATTEMPTS_MISSING}


def test_바이트_단계가_걸리면_등록_단계를_보지_않는다(tmp_path: Path) -> None:
    """앞 단계의 사유만 나간다 — 등록 단계에도 심은 결함이 있지만 거기까지 가지 않는다.

    CR 을 **시도 기록**에 심는다. 생성 파일의 바이트는 줄을 나누는 자리에서 한 번 더 걸려서,
    거기 심으면 단계 경계의 멈춤(`if rej: raise`)을 지워도 줄 나누기의 멈춤이 먼저 걸려 이 시험이 떨어지지 않는다.
    """
    f = make(tmp_path, meta_over={"generation_sha256": hx("딴등록")})
    f.files.attempts.write_bytes(f.files.attempts.read_bytes().replace(b"\n", b"\r\n"))
    codes = f.rejects()
    assert codes == {RejectCode.CR_IN_FILE}
    assert {STAGE_OF[c] for c in codes} == {Stage.BYTES}


def test_내용_단계가_걸리면_등록_단계를_보지_않는다(tmp_path: Path) -> None:
    ids = list(IDS)
    f = make(tmp_path, rows=[row(i) for i in [ids[0], ids[0], *ids[2:]]],
             meta_over={"generation_sha256": hx("딴등록")})
    codes = f.rejects()
    assert codes == {RejectCode.DUPLICATE_IMAGE_ID}
    assert {STAGE_OF[c] for c in codes} == {Stage.CONTENT}


def test_곁_파일이_JSON_이_아니면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    f.files.sidecar.write_text("{ 망가진", encoding="utf-8")
    assert f.rejects() == {RejectCode.SIDECAR_SCHEMA}


def test_곁_파일이_객체가_아니면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    f.files.sidecar.write_text("[1, 2]", encoding="utf-8")
    assert f.detail()["reason"] == "객체가 아니다"


@pytest.mark.parametrize("key", ["cell", "n_lines", "started_at", "budget_r"])
def test_필수_키가_빠지면_무엇이_빠졌는지_말한다(tmp_path: Path, key: str) -> None:
    f = make(tmp_path, drop_meta=(key,))
    assert f.detail()["missing"] == [key]


def test_해시_자리에_해시가_아닌_것이_있으면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"prompt_sha256": "abc"})
    assert f.detail()["not_sha256"] == ["prompt_sha256"]


@pytest.mark.parametrize("bad", [-1, "3", 1.0, True])
def test_수_자리에_수가_아닌_것이_있으면_거부한다(tmp_path: Path, bad) -> None:
    f = make(tmp_path, meta_over={"batch_size": bad})
    assert f.detail()["not_int"] == "batch_size"


def test_모르는_모드는_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"mode": "dryrun"})
    assert RejectCode.SIDECAR_SCHEMA in f.rejects()


def test_이름의_모드와_곁_파일의_모드가_다르면_거부한다(tmp_path: Path) -> None:
    # 곁 파일이 에코라 말하면 원장 참조도 없어야 한다(2026-09-30 결정) — 그것까지 맞춰 놓고
    # **모드 불일치만** 남긴다. 아니면 스키마 단계가 먼저 걸려 이 경로를 못 본다.
    f = make(tmp_path, meta_over={"mode": MODE_ECHO}, drop_meta=SIDECAR_ECHO_ABSENT)
    assert f.rejects() == {RejectCode.MODE_MISMATCH}


def test_토큰_파일과_해시는_함께_있거나_함께_없어야_한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"tokens_sha256": hx("tok")})
    assert f.rejects() == {RejectCode.TOKENS_FILE_MISMATCH}


def test_토큰_파일이_있고_해시가_맞으면_통과한다(tmp_path: Path) -> None:
    rows = [row(i, n_new_tokens=5) for i in IDS]
    assert make(tmp_path, rows=rows, tokens=tokens_of(IDS, 5)).verify().mode == MODE_MODEL


# ---------------------------------------------------------------- B 바이트

def test_생성_파일_해시가_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"generations_sha256": hx("딴것")})
    assert f.rejects() == {RejectCode.GENERATIONS_HASH_MISMATCH}


def test_출발_각인_해시가_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"start_stamp_sha256": hx("딴것")})
    assert f.rejects() == {RejectCode.START_STAMP_HASH_MISMATCH}


def test_토큰_해시가_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, tokens=b'{"t": 1}\n', meta_over={"tokens_sha256": hx("딴것")})
    assert f.rejects() == {RejectCode.TOKENS_HASH_MISMATCH}


def test_BOM(tmp_path: Path) -> None:
    assert make(tmp_path, raw=b"\xef\xbb\xbf" + body_of([row(i) for i in IDS])).rejects() \
        == {RejectCode.BOM_IN_FILE}


def test_CR(tmp_path: Path) -> None:
    assert make(tmp_path, raw=body_of([row(i) for i in IDS]).replace(b"\n", b"\r\n")).rejects() \
        == {RejectCode.CR_IN_FILE}


def test_끝_개행이_없으면_쓰다_만_파일이다(tmp_path: Path) -> None:
    assert make(tmp_path, raw=body_of([row(i) for i in IDS])[:-1]).rejects() \
        == {RejectCode.TORN_TAIL}


def test_UTF8_이_아니면_거부한다(tmp_path: Path) -> None:
    assert make(tmp_path, raw=body_of([row(i) for i in IDS]) + b"\xff\n").rejects() \
        == {RejectCode.NOT_UTF8}


def test_빈_줄이_있으면_어느_줄인지_말한다(tmp_path: Path) -> None:
    raw = body_of([row(IDS[0])]) + b"\n" + body_of([row(IDS[1])])
    f = make(tmp_path, raw=raw)
    with pytest.raises(BundleRejected) as exc:
        f.verify()
    COVERED.update(exc.value.codes())
    assert exc.value.codes() == {RejectCode.BLANK_LINE}
    assert exc.value.rejections[0].where.endswith(":2")


def test_줄_수는_셋을_한_번에_맞댄다(tmp_path: Path) -> None:
    """실제 줄 수·곁 파일의 수·목록 길이. 둘만 맞대면 셋째가 조용히 어긋난다."""
    f = make(tmp_path, meta_over={"n_lines": 99})
    d = f.detail()
    assert (d["actual"], d["sidecar"], d["list"]) == (3, 99, 3)


def test_목록이_더_길면_줄_수에서_걸린다(tmp_path: Path) -> None:
    f = make(tmp_path, ids=IDS[:2], list_ids=IDS)
    assert f.rejects() == {RejectCode.LINE_COUNT_MISMATCH}


# ---------------------------------------------------------------- C 내용

def test_줄이_JSON_이_아니면_거부한다(tmp_path: Path) -> None:
    raw = body_of([row(IDS[0])]) + b"{ \xea\xb9\xa8\xec\xa7\x84\n" + body_of([row(IDS[2])])
    assert make(tmp_path, raw=raw).rejects() == {RejectCode.LINE_SCHEMA}


def test_줄에_image_id_가_없으면_거부한다(tmp_path: Path) -> None:
    rows = [row(IDS[0]), {"coord_space": "ABS_ORIG"}, row(IDS[2])]
    assert make(tmp_path, rows=rows).rejects() == {RejectCode.LINE_SCHEMA}


def test_같은_이미지가_두_번_나오면_거부한다(tmp_path: Path) -> None:
    rows = [row(IDS[0]), row(IDS[0]), row(IDS[2])]
    f = make(tmp_path, rows=rows, list_ids=IDS)
    assert RejectCode.DUPLICATE_IMAGE_ID in f.rejects()


def test_순서가_다르면_어디서부터_갈렸는지_말한다(tmp_path: Path) -> None:
    rows = [row(IDS[0]), row(IDS[2]), row(IDS[1])]
    f = make(tmp_path, rows=rows, list_ids=IDS)
    with pytest.raises(BundleRejected) as exc:
        f.verify()
    COVERED.update(exc.value.codes())
    d = next(r for r in exc.value.rejections if r.code is RejectCode.ID_SEQUENCE_MISMATCH).detail
    assert d["first_diff_line"] == 2
    assert d["same_set"] is True, "집합은 같고 순서만 다르다는 것을 말해 준다"


def test_목록_해시가_등록과_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"eval_list_file_sha256": hx("딴목록")})
    assert RejectCode.REGISTRATION_ITEM_MISMATCH in f.rejects()


def test_에코에서_목록_해시가_다르면_에코_사유로_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, echo=True, meta_over={"eval_list_set_sha256": hx("딴목록")})
    assert RejectCode.ECHO_LIST_MISMATCH in f.rejects()


def test_에코_목록이_등록돼_있지_않으면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, echo=True)
    f.registration = replace(f.registration, echo_list=None)
    assert f.rejects() == {RejectCode.ECHO_LIST_MISMATCH}


def test_줄의_좌표_규약이_곁_파일과_다르면_거부한다(tmp_path: Path) -> None:
    rows = [row(IDS[0]), row(IDS[1], coord_space="NORM_1000"), row(IDS[2])]
    assert make(tmp_path, rows=rows).rejects() == {RejectCode.COORD_SPACE_MISMATCH}


def test_좌표_설정_해시가_줄과_다르면_거부한다(tmp_path: Path) -> None:
    rows = [row(IDS[0], coord_cfg_hash=hx("딴설정"))] + [row(i) for i in IDS[1:]]
    assert make(tmp_path, rows=rows).rejects() == {RejectCode.COORD_SPACE_MISMATCH}


def test_곁_파일의_좌표_규약이_등록과_다르면_거부한다(tmp_path: Path) -> None:
    rows = [row(i, coord_space="NORM_1000") for i in IDS]
    f = make(tmp_path, rows=rows, meta_over={"coord_space": "NORM_1000"})
    assert RejectCode.REGISTRATION_ITEM_MISMATCH in f.rejects()


@pytest.mark.parametrize("bad", [[1280], [1280, 0], "1280x704", [1280.0, 704.0], None])
def test_입력_크기의_꼴이_아니면_거부한다(tmp_path: Path, bad) -> None:
    rows = [row(IDS[0], model_input_wh=bad)] + [row(i) for i in IDS[1:]]
    assert make(tmp_path, rows=rows).rejects() == {RejectCode.LINE_SCHEMA}


def test_줄마다_입력_크기가_다르면_거부한다(tmp_path: Path) -> None:
    """줄의 격자는 그 줄의 입력 크기와 맞게 두어 — 줄 사이 입력 크기 검사가 먼저 걸리는 것을 본다."""
    rows = [row(IDS[0]), row(IDS[1], model_input_wh=[1280, 720], image_grid_thw=[1, 45, 80]), row(IDS[2])]
    assert make(tmp_path, rows=rows).rejects() == {RejectCode.MODEL_INPUT_WH_MISMATCH}


def test_입력_크기가_관측_격자에서_나온_값이_아니면_거부한다(tmp_path: Path) -> None:
    """문자 일치의 동어반복을 막는 닻 — 곁 파일이 스스로를 베끼는 것을 잡는다. 줄이 격자를 실으면 줄의 격자를 센 것과
    곁 파일의 관측 격자가 같아야 하므로 그 집계가 먼저 잡는다(곁 파일만 보는 닻 `_check_grid` 는 그 뒤의 방어다)."""
    f = make(tmp_path, meta_over={"image_grid_thw_observed": {"[1, 45, 80]": 3}})
    d = f.detail()
    assert d["sidecar"] == {"[1, 45, 80]": 3} and d["in_lines"] == {GRID_KEY: 3}


def test_관측_격자가_두_가지면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"image_grid_thw_observed": {GRID_KEY: 2, "[1, 45, 80]": 1}})
    assert f.rejects() == {RejectCode.MODEL_INPUT_GRID_MISMATCH}


@pytest.mark.parametrize("bad", [[1, 44], [1, 44, 80, 1], [1.0, 44, 80], [True, 44, 80], [0, 44, 80], [1, 0, 80],
                                 "1x44x80", None])
def test_줄의_관측_격자의_꼴이_아니면_거부한다(tmp_path: Path, bad) -> None:
    """줄의 격자는 내용 해시에 든다 — 꼴을 보지 않으면 아무 값으로 판정 한 번의 열쇠를 바꿀 수 있다(재확인 contentkey 의 1)."""
    rows = [row(IDS[0], image_grid_thw=bad)] + [row(i) for i in IDS[1:]]
    f = make(tmp_path, rows=rows)
    assert f.rejects() == {RejectCode.LINE_SCHEMA}
    assert "정수" in f.detail()["reason"]


def test_줄의_관측_격자의_시간_축은_1_이다(tmp_path: Path) -> None:
    rows = [row(IDS[0], image_grid_thw=[2, 44, 80])] + [row(i) for i in IDS[1:]]
    f = make(tmp_path, rows=rows)
    assert f.rejects() == {RejectCode.LINE_SCHEMA} and "시간 축" in f.detail()["reason"]


def test_줄의_관측_격자에_패치를_곱하면_그_줄의_입력_크기다(tmp_path: Path) -> None:
    rows = [row(IDS[0], image_grid_thw=[1, 45, 80])] + [row(i) for i in IDS[1:]]
    f = make(tmp_path, rows=rows)
    assert f.rejects() == {RejectCode.MODEL_INPUT_GRID_MISMATCH} and "model_input_wh" in f.detail()["reason"]


@pytest.mark.parametrize("purpose", ["main", "rehearsal", "frame_diag"])
@pytest.mark.parametrize("echo", [False, True])
def test_격자_없는_묶음은_모든_목적에서_거부한다(tmp_path: Path, purpose: str, echo: bool) -> None:
    """계약 §37 — 격자는 생성 시점에만 나온다. 줄에도 곁 파일에도 격자가 없는 묶음(구판 · 격자를 내지 않는 대역)은
    목적 · 모드와 무관하게 거부한다. 사후 보충으로 받지 않는다."""
    rows = [{k: v for k, v in row(i).items() if k != "image_grid_thw"} for i in IDS]
    f = make(tmp_path, rows=rows, echo=echo, purpose=purpose, meta_over={"image_grid_thw_observed": {}})
    assert RejectCode.LINE_SCHEMA in f.rejects() and "image_grid_thw 가 없다" in f.detail()["reason"]


def test_줄에서_격자를_지우면_거부한다(tmp_path: Path) -> None:
    """격자를 모든 줄에서 지워도 내용 열쇠가 바뀐다 — 줄마다 격자가 있어야 한다(07번 §37)."""
    rows = [{k: v for k, v in row(i).items() if k != "image_grid_thw"} for i in IDS]
    f = make(tmp_path, rows=rows)
    assert f.rejects() == {RejectCode.LINE_SCHEMA} and "image_grid_thw 가 없다" in f.detail()["reason"]


def test_줄_하나의_격자를_지우고_곁_파일_집계를_맞춰_줄여도_거부한다(tmp_path: Path) -> None:
    """§45 의 반례 — 첫 줄의 격자만 지우고 곁 파일 집계를 3 → 2 로 줄인다(원시 해시는 도우미가 맞춘다).
    있는 줄끼리 센 집계와 곁 파일이 같아도, 줄마다 격자가 있어야 하고 집계의 합이 전체 줄 수여야 한다."""
    rows = [{k: v for k, v in row(IDS[0]).items() if k != "image_grid_thw"}] + [row(i) for i in IDS[1:]]
    f = make(tmp_path, rows=rows, meta_over={"image_grid_thw_observed": {GRID_KEY: 2}})
    assert f.rejects() == {RejectCode.LINE_SCHEMA}
    assert "image_grid_thw 가 없다" in f.detail()["reason"]


@pytest.mark.parametrize(("ids", "observed", "why"), [
    (IDS, {GRID_KEY: 3.0}, "실수"),
    ([IDS[0]], {GRID_KEY: True}, "bool"),
    (IDS, {GRID_KEY: 3, "[1, 45, 80]": 0}, "0"),
    (IDS, {GRID_KEY: 2}, "합계"),
])
def test_곁_파일_집계는_양의_정수이고_합계가_전체_줄_수다(tmp_path: Path, ids, observed, why: str) -> None:
    """곁 파일의 관측 격자 빈도 — bool 아닌 양의 정수, 합계 = 전체 줄 수. `3.0 == 3` · `True == 1` 이라 사전 비교만으로는 지난다."""
    f = make(tmp_path, ids=ids, rows=[row(i) for i in ids], meta_over={"image_grid_thw_observed": observed})
    assert f.rejects() == {RejectCode.MODEL_INPUT_GRID_MISMATCH}, why
    assert "합계가 전체 줄 수" in f.detail()["reason"]


def test_격자_표기를_읽을_수_없으면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"image_grid_thw_observed": {"1x44x80": 3}})
    assert RejectCode.MODEL_INPUT_GRID_MISMATCH in f.rejects()


def test_읽은_그림이_동결_매니페스트와_다르면_거부한다(tmp_path: Path) -> None:
    want = dict.fromkeys(IDS, hx("딴그림"))
    f = make(tmp_path)
    d = f.detail(image_sha256_of=want)
    assert d["n_lines"] == 3 and d["first_line"] == 1


def test_매니페스트를_주지_않으면_그림은_보지_않는다(tmp_path: Path) -> None:
    assert make(tmp_path).verify().image_ids == tuple(IDS)


def test_종료를_정할_수_없는_줄이_있으면_거부한다(tmp_path: Path) -> None:
    rows = [row(IDS[0], gen_stop="stop_undetermined")] + [row(i) for i in IDS[1:]]
    f = make(tmp_path, rows=rows)
    assert f.detail()["n_lines"] == 1


# ---------------------------------------------------------------- C 모집단

def test_채점_모집단이_생성_목록_밖이면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    d = f.detail(scoring_population=[*IDS, "aihub999999"])
    assert d["n_outside"] == 1


def test_모집단은_생성_목록의_부분집합이면_된다(tmp_path: Path) -> None:
    assert make(tmp_path).verify(scoring_population=IDS[:2]).image_ids == tuple(IDS)


def test_덩어리가_쪼개진_것은_거부하지_않는다(tmp_path: Path) -> None:
    """제외 단위가 영상이라 덩어리는 정당하게 쪼개진다 — 전체 구성원을 요구하면 정상 모집단이 거부된다(§18-5).

    생성 목록에 1·2번만 남은 덩어리를 모집단이 통째로 담았다. 원래 3번이 있었어도 통과해야 한다.
    """
    f = make(tmp_path)
    group = {IDS[0]: "g1", IDS[1]: "g1", IDS[2]: "g2"}
    assert f.verify(scoring_population=IDS, group_of=group).image_ids == tuple(IDS)


def test_남은_것_가운데_일부만_골라_담으면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    group = {IDS[0]: "g1", IDS[1]: "g1", IDS[2]: "g2"}
    d = f.detail(scoring_population=[IDS[0], IDS[2]], group_of=group)
    assert d["n_incomplete_clusters"] == 1


def test_에코에는_모집단_검사를_걸지_않는다(tmp_path: Path) -> None:
    f = make(tmp_path, echo=True)
    assert f.verify(scoring_population=["아무거나"]).is_echo


# ---------------------------------------------------------------- D 등록·원장

def test_생성_지문이_등록과_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"generation_sha256": hx("딴등록")})
    assert RejectCode.GENERATION_FINGERPRINT_MISMATCH in f.rejects()


def test_등록값은_자료형까지_맞댄다(tmp_path: Path) -> None:
    """`256` 과 `"256"` 은 다르다 — 같다고 보면 설정이 바뀐 것을 놓친다."""
    f = make(tmp_path, meta_over={"max_new_tokens": 256})
    f.registration = replace(f.registration, values={**f.registration.values, "max_new_tokens": "256"})
    d = f.detail()
    assert d["registered_type"] == "str" and d["sidecar_type"] == "int"
    assert d["equal"] is False


def test_등록값이_곁_파일에_없으면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    f.registration = replace(f.registration, values={**f.registration.values, "없는키": 1})
    assert f.detail()["reason"] == "곁 파일에 없다"


def test_등록값이_같으면_통과한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    f.registration = replace(f.registration, values={**f.registration.values, "max_new_tokens": 256,
                                                     "padding_side": "left"})
    assert f.verify().seed_value == 20260828


def test_곁_파일의_태그가_이름과_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"tag": "uni_fed"})
    assert RejectCode.TAG_MISMATCH in f.rejects()


def test_칸_참여자가_태그와_어긋나면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"client": "C2"})
    assert RejectCode.TAG_MISMATCH in f.rejects()


def test_이름의_시드_번호와_곁_파일이_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"seed_index": 2})
    assert RejectCode.SEED_MAPPING_MISMATCH in f.rejects()


def test_등록에_없는_시드_번호는_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, seed_index=9)
    f.registration = replace(f.registration, seeds={1: 20260828})
    assert RejectCode.SEED_NOT_REGISTERED in f.rejects()


def test_시드_번호에_붙은_값이_등록과_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    f.registration = replace(f.registration, seeds={1: 11111111})
    d = next(r for r in _rejections(f) if r.code is RejectCode.SEED_MAPPING_MISMATCH).detail
    assert d["registered"] == 11111111 and d["sidecar"] == 20260828


def _rejections(f: Fixture, **over):
    with pytest.raises(BundleRejected) as exc:
        f.verify(**over)
    COVERED.update(exc.value.codes())
    return exc.value.rejections


@pytest.mark.parametrize("key", ["started_at", "finished_at"])
def test_오프셋_없는_시각은_거부한다(tmp_path: Path, key: str) -> None:
    """naive 시각은 9시간 창에서 거짓 통과를 만든다."""
    f = make(tmp_path, meta_over={key: "2026-09-21T10:00:00"})
    assert RejectCode.TIME_NAIVE in f.rejects()


def test_시각이_시각이_아니면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"started_at": "어제"})
    assert RejectCode.TIME_NAIVE in f.rejects()


def test_시작이_종료보다_늦으면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"started_at": T_END, "finished_at": T_START})
    d = next(r for r in _rejections(f) if r.code is RejectCode.TIME_ORDER).detail
    assert d["pair"] == "started>finished"


def test_등록보다_먼저_생성했으면_거부한다(tmp_path: Path) -> None:
    """등록이 먼저였다는 것이 사전등록의 전부다."""
    late = datetime.fromisoformat("2026-09-22T00:00:00+09:00")
    f = make(tmp_path)
    pairs = {r.detail.get("pair") for r in _rejections(f, receipt_time=late)
             if r.code is RejectCode.TIME_ORDER}
    # 곁 파일과 시도 줄이 **둘 다** 등록 시각과 맞대진다(§27-6 의 관계 1·2).
    assert "started<receipt" in pairs and "start<receipt" in pairs


# ---------------------------------------------------------------- D 학습

def test_학습_산출물이_없으면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    assert RejectCode.ARTIFACT_META_MISSING in f.rejects(artifact=None)


def test_채점한_어댑터가_산출물과_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"scored_adapter_sha256": hx("딴어댑터")})
    assert RejectCode.LEDGER_MISMATCH in f.rejects()


def test_원장이_없으면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    assert RejectCode.LEDGER_MISMATCH in f.rejects(ledger=None)


def test_run_신원이_원장과_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"train_run_id": "딴run"})
    assert RejectCode.LEDGER_MISMATCH in f.rejects()


def test_마지막_스텝이_아니면_거부한다(tmp_path: Path) -> None:
    """best 금지·last 채점이 서는 자리다 — 중간 체크포인트를 채점하는 것을 여기서 막는다."""
    f = make(tmp_path, meta_over={"scored_adapter_step": 60})
    f.artifact = TrainingArtifact(hx("adapter"), 60)
    d = next(r for r in _rejections(f) if r.code is RejectCode.ADAPTER_NOT_FINAL).detail
    assert d["artifact_step"] == 60 and d["ledger_final_step"] == 100


def test_같은_어댑터를_두_칸에서_채점하면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path)
    d = next(r for r in _rejections(f, seen_adapters={hx("adapter"): ("uni_fed", 1)})
             if r.code is RejectCode.ADAPTER_DUPLICATE).detail
    assert d["also_in"] == "uni_fed_s1"


def test_에코에_어댑터가_실려_있으면_거부한다(tmp_path: Path) -> None:
    """에코는 모델을 부르지 않는다."""
    f = make(tmp_path, echo=True, meta_over={"scored_adapter_sha256": hx("adapter")})
    assert RejectCode.ECHO_ADAPTER_PRESENT in f.rejects()


def test_에코는_원장과_산출물을_요구하지_않는다(tmp_path: Path) -> None:
    assert make(tmp_path, echo=True).verify(artifact=None, ledger=None).is_echo


# ------------------------------------------------- 시작 기록의 내용 (§26-3 · §28-3 가)
#
# **해시가 맞는 것과 내용이 맞는 것은 다르다.** 다른 묶음의 도장을 놓고 그 해시만 곁 파일에
# 적으면 해시 검사는 통과한다. 도장은 추론 전에 쓰므로 필수 아홉이 그 시점에 다 있다.

def test_앞_판의_빈_도장은_거부된다(tmp_path: Path) -> None:
    """`{"ok": true}` — 검사 대상 필드가 처음부터 없는 자료였다."""
    f = make(tmp_path, start_raw=b'{"ok": true}')
    assert RejectCode.START_STAMP_SCHEMA in f.rejects()


def test_도장이_JSON_이_아니면_거부한다(tmp_path: Path) -> None:
    assert RejectCode.START_STAMP_SCHEMA in make(tmp_path, start_raw=b"not json").rejects()


def test_도장의_계약_키가_없으면_빠진_이름을_말한다(tmp_path: Path) -> None:
    f = make(tmp_path, drop_start=("mode", "device"))
    assert RejectCode.START_STAMP_SCHEMA in f.rejects()
    assert set(f.detail()["missing"]) == {"mode", "device"}


def test_도장의_원장_참조가_없으면_거부한다(tmp_path: Path) -> None:
    """§26-3 이 아홉으로 올린 넷 가운데 둘 — 학습이 끝난 뒤 export 라 그 시점에 있다."""
    f = make(tmp_path, drop_start=("train_run_id", "train_ledger_sha256"))
    assert RejectCode.START_STAMP_SCHEMA in f.rejects()
    assert set(f.detail()["missing"]) == {"train_run_id", "train_ledger_sha256"}


def test_도장의_시각에_초_아래_자릿수가_없으면_거부한다(tmp_path: Path) -> None:
    """§28-4 — 같은 바이트의 두 도장이 사실상 생기지 않게 한다."""
    f = make(tmp_path, start_over={"started_at": "2026-09-21T09:30:00+09:00"})
    assert RejectCode.START_STAMP_SCHEMA in f.rejects()
    assert f.detail()["item"] == "started_at"


def test_도장의_장비_꼴이_어긋나면_거부한다(tmp_path: Path) -> None:
    """§28-3 나 — 번호만으로는 다른 장비가 같은 값이 된다."""
    f = make(tmp_path, start_over={"device": "0"})
    assert RejectCode.START_STAMP_SCHEMA in f.rejects()
    assert f.detail()["item"] == "device"


def test_모델_모드_도장에_어댑터가_없으면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, drop_start=("scored_adapter_sha256",))
    assert RejectCode.START_STAMP_SCHEMA in f.rejects()


def test_에코_도장에_어댑터나_원장이_있으면_거부한다(tmp_path: Path) -> None:
    """§28-3 가 — 에코는 모델을 부르지 않고 그 학습에 기대지 않는다."""
    f = make(tmp_path, echo=True, start_over={"train_run_id": "run_c1"})
    assert RejectCode.ECHO_ADAPTER_PRESENT in f.rejects()
    assert f.detail()["present"] == ["train_run_id"]


def test_에코_도장은_일곱으로_통과한다(tmp_path: Path) -> None:
    """에코 필수는 아홉에서 원장 둘을 뺀 일곱이다."""
    assert make(tmp_path, echo=True).verify(artifact=None, ledger=None).is_echo


def test_다른_묶음의_도장을_놓고_해시를_맞춰도_거부한다(tmp_path: Path) -> None:
    """**이 변이가 요점이다.** 픽스처가 도장 파일에서 해시를 다시 계산하므로 해시는 맞는다."""
    f = make(tmp_path, start_over={"tag": "uni_fed"})
    assert RejectCode.START_STAMP_MISMATCH in f.rejects()
    assert RejectCode.START_STAMP_HASH_MISMATCH not in f.rejects(), "해시는 맞은 채로 걸려야 한다"


def test_도장의_시드_번호가_불리언이면_거부한다(tmp_path: Path) -> None:
    """`True == 1` 을 막는다(19번 Minor 5-3)."""
    f = make(tmp_path, start_over={"seed_index": True})
    assert RejectCode.START_STAMP_SCHEMA in f.rejects()
    assert f.detail()["not_int"] == "seed_index"


def test_도장의_등록_지문이_등록값과_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, start_over={"generation_sha256": hx("딴등록")})
    assert RejectCode.START_STAMP_MISMATCH in f.rejects()


def test_도장의_목록_해시가_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, start_over={"eval_list_file_sha256": hx("딴목록")})
    assert RejectCode.START_STAMP_MISMATCH in f.rejects()


@pytest.mark.parametrize("key", ["train_run_id", "train_ledger_sha256"])
def test_도장과_곁_파일의_원장_신원이_다르면_거부한다(tmp_path: Path, key: str) -> None:
    """09-30 에 계약에 없는 대조라 뺐고, **계약 07번 §30-9 가 되살렸다**(맞대기 04 의 D-하) — 쓰는 쪽은 도장의 값을
    곁 파일에 옮기고 다르면 쓰지 않는다. 읽는 쪽은 둘이 같은지 본다. 앞 판의 시험은 뺐다는 사실을 고정했었다."""
    other = "딴run" if key == "train_run_id" else hx("딴원장")
    f = make(tmp_path, start_over={key: other})
    assert RejectCode.START_STAMP_MISMATCH in f.rejects()


def test_도장의_어댑터가_곁_파일과_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, start_over={"scored_adapter_sha256": hx("딴어댑터")})
    assert RejectCode.START_STAMP_MISMATCH in f.rejects()


# --------------------------------- 도장과 시도 기록 사이 (§27-6 의 관계 1·3 · §28-3 나)
#
# **앞 판이 뺀 넷이다.** 시도 줄과 곁 파일만 보고 도장을 보지 않아서, 도장이 등록 전 시각이거나
# 첫 시도보다 늦거나 장비가 다른 묶음이 통과했다. 넷 다 시험이 없어 통과 수가 그것을 못 드러냈다.
# 등록 전 생성 자체는 관계 2 가 막는다 — 넷이 막는 것은 **도장과 시도 기록 사이의 모순**이다.

def test_관계1_도장이_영수증보다_이르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, start_over={"started_at": "2026-09-21T08:00:00.000000+09:00"})
    assert RejectCode.TIME_ORDER in f.rejects()
    assert "stamp<receipt" in {r.detail.get("pair") for r in _rejections(f)}


def test_관계3_도장이_첫_시도보다_늦으면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, start_over={"started_at": "2026-09-21T23:00:00.000000+09:00"})
    assert RejectCode.TIME_ORDER in f.rejects()
    assert "first_start<stamp" in {r.detail.get("pair") for r in _rejections(f)}


def test_관계3_도장과_첫_시도가_같은_시각이면_통과한다(tmp_path: Path) -> None:
    """비엄격이다 — 한 시각을 둘에 쓰면 같다."""
    f = make(tmp_path, start_over={"started_at": at(2)})
    assert f.verify().tag == "uni_local_C1"


def test_도장의_장비가_첫_시작_줄과_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, start_over={"device": "NVIDIA A100-SXM4-40GB:0"})
    assert RejectCode.START_STAMP_MISMATCH in f.rejects()
    assert f.detail()["reason"] == "도장이 첫 시작 줄과 다르다"


def test_도장의_시각에_오프셋이_없으면_거부한다(tmp_path: Path) -> None:
    """시도 줄과 곁 파일만 보면 오프셋 없는 도장이 통과한다."""
    f = make(tmp_path, start_over={"started_at": "2026-09-21T09:30:00.000000"})
    assert RejectCode.TIME_NAIVE in f.rejects()
    assert f.detail()["item"] == "started_at"


# --------------------------------- 결정 셋 (2026-09-30)

@pytest.mark.parametrize("key", ["train_run_id", "train_ledger_sha256", "train_ledger_path"])
def test_에코_곁_파일에_원장_참조가_있으면_거부한다(tmp_path: Path, key) -> None:
    """같은 묶음이 두 말을 하지 않게 한다 — 도장에서 그렇게 정했다(§28-3 다).

    **셋 다 원장을 가리키는 키다.** `train_ledger_path` 는 이름 그대로 원장 참조라 함께 뺀다 —
    하나라도 남으면 도장은 "학습과 무관" 이라 하고 곁 파일은 원장을 가리킨다.
    """
    f = make(tmp_path, echo=True, meta_over={key: "값"})
    assert RejectCode.ECHO_ADAPTER_PRESENT in f.rejects()
    assert f.detail()["present"] == [key]


def test_에코_곁_파일의_학습_설정과_예산은_그대로_필수다(tmp_path: Path) -> None:
    """결정의 문장이 "원장 참조" 라 여기까지 넓히지 않았다 — 미정으로 올렸다(§28-8)."""
    assert set(SIDECAR_ECHO_ABSENT) == {"train_run_id", "train_ledger_sha256",
                                        "train_ledger_path"}
    for key in ("train_config_sha256", "budget_n", "budget_r", "budget_e"):
        assert key in SIDECAR_REQUIRED and key not in SIDECAR_ECHO_ABSENT
    assert make(tmp_path, echo=True).verify(artifact=None, ledger=None).is_echo


def test_주인_없는_줄_하나가_폴더의_묶음을_거부한다(tmp_path: Path) -> None:
    """**덜 막는 쪽으로 틀리지 않는다** — 주인을 모르는 줄은 어느 묶음의 회계에서 빠진 줄일 수 있다."""
    def add_orphan(rows):
        stray = dict(rows[0])
        stray.pop("tag")
        stray["seed_index"] = 9
        return rows + [stray]

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=add_orphan)
    assert RejectCode.ATTEMPTS_SCHEMA in f.rejects()
    assert f.detail()["missing"] == ["tag"]


def test_없어야_하는_항목이_빈_값이어도_거부한다(tmp_path: Path) -> None:
    """**키가 있으면 값이 비어도 있는 것이다** — 쓰는 쪽은 키를 쓰지 않으면 된다."""
    f = make(tmp_path, echo=True, start_over={"scored_adapter_sha256": None})
    assert RejectCode.ECHO_ADAPTER_PRESENT in f.rejects()
    assert f.detail()["present"] == ["scored_adapter_sha256"]


# --------------------------------- 회계 — 첫 시도의 빈 시작 (22번 1-5)

def test_목록이_비지_않았는데_첫_시도가_빈_시작이면_거부한다(tmp_path: Path) -> None:
    """`null` 은 "쓸 것이 없었다" 는 뜻이다 — 그것은 목록 처음에서 시작한 것이 아니다."""
    f = nine(tmp_path, [(None, None, 0), (0, REASON_RESUME, 9)])
    assert RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH in f.rejects()
    assert f.detail()["reason"] == "첫 시도가 목록 처음에서 시작하지 않았다"


# ------------------------------------------- 시도 기록 — 계약의 경우 여덟 (§27-2 · §28-2)
#
# **바르게 쓴 묶음이 통과하는 것**을 경우마다 본다. 죽은 시도는 끝 줄이 없는 것으로 드러나고
# 아무도 대신 적지 않는다 — 그래서 추정한 시각·추정한 끝 id 가 기록에 들어가지 않는다.
# 목록은 아홉 장이고 배치는 1 이다.

def nine(tmp: Path, plan, **kw) -> Fixture:
    return make(tmp, ids=IDS9, attempts_plan=plan, **kw)


def test_경우1_첫_시도가_스스로_끝남(tmp_path: Path) -> None:
    got = nine(tmp_path, [(0, None, 9)]).verify()
    assert list(got.notes["attempts_spans"]) == [9]
    assert got.notes["attempts_end_rows"] == 1


def test_경우2_첫_시도가_줄을_쓴_뒤_죽음(tmp_path: Path) -> None:
    """시도 1 이 4줄 쓰고 죽었다. 시도 2 가 5번째부터 이어 쓴다."""
    got = nine(tmp_path, [(0, None, None), (4, REASON_RESUME, 5)]).verify()
    assert list(got.notes["attempts_spans"]) == [4, 5]
    assert got.notes["attempts_start_rows"] == 2 and got.notes["attempts_end_rows"] == 1


def test_경우3_첫_시도가_줄_없이_죽음(tmp_path: Path) -> None:
    """끝 id 를 적을 주체가 없어졌다 — 끝 줄이 없다."""
    got = nine(tmp_path, [(0, None, None), (0, REASON_RESUME, 9)]).verify()
    assert list(got.notes["attempts_spans"]) == [0, 9]


def test_경우4_둘째_뒤_시도가_줄_쓴_뒤_죽음(tmp_path: Path) -> None:
    """시작 시각의 닻이 필요 없다 — 그 시도가 시작할 때 스스로 적었다."""
    got = nine(tmp_path, [(0, None, 3), (3, REASON_RESUME, None),
                          (5, REASON_RESUME, 4)]).verify()
    assert list(got.notes["attempts_spans"]) == [3, 2, 4]


def test_경우5_둘째_뒤_시도가_줄_없이_죽음(tmp_path: Path) -> None:
    got = nine(tmp_path, [(0, None, 3), (3, REASON_RESUME, None),
                          (3, REASON_RESUME, 6)]).verify()
    assert list(got.notes["attempts_spans"]) == [3, 0, 6]


def test_경우6_꼬리_재생성_중_죽음(tmp_path: Path) -> None:
    """사유가 바뀌지 않으므로 정당한 재시도가 바-16 위반으로 읽히지 않는다."""
    def cut(rows):
        for r in rows:
            if r["attempt_no"] == 2 and r["event"] == EVENT_START:
                r["truncated_ids"] = [IDS9[3]]
        return rows

    got = nine(tmp_path, [(0, None, None), (3, REASON_TORN_TAIL, None),
                          (3, REASON_RESUME, 6)], attempts_edit=cut).verify()
    assert list(got.notes["attempts_spans"]) == [3, 0, 6]
    assert got.notes["attempts_reasons"][1] == REASON_TORN_TAIL


def test_경우7_다_쓰고_끝_줄_전에_죽음(tmp_path: Path) -> None:
    """규칙 3 — 다음 시도의 시작이 `null` 이면 그 자리를 줄 수로 본다."""
    got = nine(tmp_path, [(0, None, None), (None, REASON_RESUME, 0)]).verify()
    assert list(got.notes["attempts_spans"]) == [9, 0]


def test_경우8_끝_줄을_적고_멈춘_뒤_재개(tmp_path: Path) -> None:
    """§28-2 — 재개의 뜻이 "목록을 다 쓰지 못하고 멈춘 뒤" 로 넓었다."""
    got = nine(tmp_path, [(0, None, 3), (3, REASON_RESUME, 6)]).verify()
    assert list(got.notes["attempts_spans"]) == [3, 6]
    assert got.notes["attempts_end_rows"] == 2


# ------------------------------------------- 시도 기록 — 회계 (§28-1)

def test_회계는_첫_시도가_목록_처음에서_시작했는지_본다(tmp_path: Path) -> None:
    """합 등식이 실제로 보는 것 하나 — 따로 단언한다."""
    f = nine(tmp_path, [(1, None, 8)])
    assert RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH in f.rejects()
    assert f.detail()["reason"] == "첫 시도가 목록 처음에서 시작하지 않았다"


def test_회계는_시작_자리가_뒤로_가지_않는지_본다(tmp_path: Path) -> None:
    """합 등식이 실제로 보는 것 둘."""
    f = nine(tmp_path, [(0, None, None), (5, REASON_RESUME, None),
                        (2, REASON_RESUME, 7)])
    assert RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH in f.rejects()
    assert f.detail()["reason"] == "시작 자리가 뒤로 갔다"


def test_끝_줄이_적은_수가_자리_차와_다르면_거부한다(tmp_path: Path) -> None:
    f = nine(tmp_path, [(0, None, 8)])
    assert RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH in f.rejects()
    d = f.detail()
    assert d["n_written"] == 8 and d["from_positions"] == 9


@pytest.mark.parametrize("last_kind", ["null", "first"])
def test_쓴_줄이_있으면_끝_식별자가_그_구간의_끝이어야_한다(tmp_path: Path, last_kind) -> None:
    """**규칙 5.** 시작 자리와 줄 수만 보면 이 모순이 통과한다 — 아홉 줄을 썼다면서 끝을
    첫 id 나 `null` 로 적는다. 자리 차 9 와 적힌 9 는 맞고 시작보다 앞서지도 않는다.
    """
    def change(rows):
        rows[-1]["last_id"] = None if last_kind == "null" else rows[0]["first_id"]
        return rows

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=change)
    assert RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH in f.rejects()
    assert f.detail()["reason"] == ("쓴 줄이 있는데 구간의 끝이 비었다" if last_kind == "null"
                                   else "끝 식별자가 그 구간의 끝이 아니다")


def test_쓴_줄이_없으면_끝_식별자도_없어야_한다(tmp_path: Path) -> None:
    """쓰지 않은 시도에 끝 식별자가 남는다. 구간 길이는 0·9 라 회계는 선다."""
    def change(rows):
        rows[1]["last_id"] = rows[0]["first_id"]
        return rows

    f = nine(tmp_path, [(0, None, 0), (0, REASON_RESUME, 9)], attempts_edit=change)
    assert RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH in f.rejects()
    assert f.detail()["reason"] == "쓴 줄이 없는데 끝 식별자가 있다"


def test_쓴_줄이_없어도_시작_식별자는_비우지_않아도_된다(tmp_path: Path) -> None:
    """**`first_id` 까지 `null` 로 강제하지 않는다** — 쓸 것이 있어 시작했으나 한 줄도
    쓰기 전에 정상 종료한 경우다. 끝 식별자만 비면 된다.
    """
    got = nine(tmp_path, [(0, None, 0), (0, REASON_RESUME, 9)]).verify()
    assert list(got.notes["attempts_spans"]) == [0, 9]


def test_죽은_시도에는_끝_식별자_규칙을_걸지_않는다(tmp_path: Path) -> None:
    """끝 줄이 없으면 맞댈 값이 없다. 죽은 시도의 회계는 그대로다."""
    got = nine(tmp_path, [(0, None, None), (4, REASON_RESUME, 5)]).verify()
    assert list(got.notes["attempts_spans"]) == [4, 5]
    assert got.notes["attempts_end_rows"] == 1


def test_죽은_시도가_남긴_줄이_세지므로_합이_선다(tmp_path: Path) -> None:
    """**앞 판이 여기서 깨졌다** — 적힌 수를 더하면 4+5 가 아니라 5 만 세어 거부됐다."""
    got = nine(tmp_path, [(0, None, None), (4, REASON_RESUME, 5)]).verify()
    assert sum(got.notes["attempts_spans"]) == 9


# ------------------------------------------- 시도 기록 — 줄의 계약

def test_앞_판의_빈_시도_줄은_거부된다(tmp_path: Path) -> None:
    assert RejectCode.ATTEMPTS_SCHEMA in make(tmp_path, attempts_raw=b"{}\n").rejects()


def test_시도_줄에_주인이_없으면_고르기_전에_거부한다(tmp_path: Path) -> None:
    def drop(rows):
        rows[0].pop("tag")
        return rows

    f = make(tmp_path, attempts_edit=drop)
    assert RejectCode.ATTEMPTS_SCHEMA in f.rejects()
    assert f.detail()["missing"] == ["tag"]


def test_event_가_두_값이_아니면_거부한다(tmp_path: Path) -> None:
    def bend(rows):
        rows[0]["event"] = "starting"
        return rows

    f = make(tmp_path, attempts_edit=bend)
    assert RejectCode.ATTEMPTS_SCHEMA in f.rejects()
    assert f.detail()["item"] == "event"


def test_시작_줄의_필수_키가_없으면_거부한다(tmp_path: Path) -> None:
    def drop(rows):
        rows[0].pop("device")
        return rows

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=drop)
    assert RejectCode.ATTEMPTS_SCHEMA in f.rejects()
    assert f.detail()["missing"] == ["device"]


def test_마지막_시도에_끝_줄이_없으면_거부한다(tmp_path: Path) -> None:
    """곁 파일은 마지막 시도가 스스로 끝난 뒤에만 생긴다(§27-1)."""
    f = nine(tmp_path, [(0, None, 9), (8, REASON_RESUME, None)])
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()
    assert f.detail()["reason"] == "마지막 시도에 끝 줄이 없다"


def test_시작_줄_없는_끝_줄은_거부한다(tmp_path: Path) -> None:
    def add(rows):
        sha = rows[0]["start_stamp_sha256"]
        return rows + [att_end("uni_local_C1", 1, 7, sha, IDS9[8], 0)]

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=add)
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()
    assert f.detail()["reason"] == "시작 줄 없는 끝 줄"


def test_시도_번호가_1부터_이어지지_않으면_거부한다(tmp_path: Path) -> None:
    """§28-4 — 도장마다 1부터 빈틈 없이."""
    def shift(rows):
        for r in rows:
            r["attempt_no"] = 2
        return rows

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=shift)
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()


def test_첫_시도의_사유가_first_pass_가_아니면_거부한다(tmp_path: Path) -> None:
    f = nine(tmp_path, [(0, REASON_RESUME, 9)])
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()
    assert f.detail()["item"] == "reason"


def test_어휘_밖의_사유는_거부한다(tmp_path: Path) -> None:
    """닫힌 목록이다(§27-3). 앞 판의 `aborted_unfinished` 도 이제 어휘에 없다."""
    f = nine(tmp_path, [(0, "aborted_unfinished", 9)])
    assert RejectCode.ATTEMPTS_SCHEMA in f.rejects()
    assert f.detail()["value"] == "aborted_unfinished"


def test_꼬리_재생성이_아닌데_잘라_낸_id_가_있으면_거부한다(tmp_path: Path) -> None:
    def add(rows):
        rows[0]["truncated_ids"] = [IDS9[0]]
        return rows

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=add)
    assert RejectCode.ATTEMPTS_SCHEMA in f.rejects()
    assert f.detail()["item"] == "truncated_ids"


def test_꼬리_재생성에_잘라_낸_id_가_없으면_거부한다(tmp_path: Path) -> None:
    f = nine(tmp_path, [(0, None, None), (3, REASON_TORN_TAIL, 6)])
    assert RejectCode.ATTEMPTS_SCHEMA in f.rejects()
    assert f.detail()["item"] == "truncated_ids"


def test_시도_횟수가_시작_줄_수와_다르면_거부한다(tmp_path: Path) -> None:
    f = nine(tmp_path, [(0, None, 9)], meta_over={"attempts": 2})
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()
    d = f.detail()
    assert d["start_rows_for_bundle"] == 1 and d["sidecar_attempts"] == 2


def test_다른_도장에_딸린_줄은_골라지지_않는다(tmp_path: Path) -> None:
    """§27-5 — 개명한 옛 시도의 줄이 같은 태그·시드로 남는다."""
    def prepend_old(rows):
        return plan_rows("uni_local_C1", 1, hx("옛도장"), IDS9, [(0, None, 9)]) + rows

    got = nine(tmp_path, [(0, None, 9)], attempts_edit=prepend_old).verify()
    assert got.notes["attempts_start_rows"] == 1
    assert got.notes["attempts_rows_total"] == 4


def test_시도_줄의_시드가_불리언이면_이_묶음의_줄이_아니다(tmp_path: Path) -> None:
    """`True == 1` 인 줄이 시드 1 의 줄로 골라지지 않는다(19번 Minor 5-3 의 나머지 반).

    골라지지 않으므로 이 묶음의 시작 줄이 0 이 되고 횟수 대조가 거부한다.
    """
    def flip(rows):
        for r in rows:
            r["seed_index"] = True
        return rows

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=flip)
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()
    assert f.detail()["start_rows_for_bundle"] == 0


def test_시도_줄의_시각에_초_아래_자릿수가_없으면_거부한다(tmp_path: Path) -> None:
    def blunt(rows):
        rows[0]["started_at"] = "2026-09-21T10:02:00+09:00"
        return rows

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=blunt)
    assert RejectCode.ATTEMPTS_SCHEMA in f.rejects()


@pytest.mark.parametrize("bad", ["0", ":0", "name:", "NVIDIA:x", 0])
def test_시도_줄의_장비_꼴이_어긋나면_거부한다(tmp_path: Path, bad) -> None:
    """§28-3 나 의 `"{이름}:{인덱스}"` **꼴**을 본다.

    **이름의 어휘는 계약에 없다** — 그래서 `cuda:0` 처럼 종류만 적은 값은 꼴로는 통과한다.
    그 자리를 구현이 스스로 정하지 않는다(보고의 계약 빈자리).
    """
    def bend(rows):
        rows[0]["device"] = bad
        return rows

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=bend)
    assert RejectCode.ATTEMPTS_SCHEMA in f.rejects()
    assert f.detail()["item"] == "device"


def test_구간의_id_가_목록_밖이면_거부한다(tmp_path: Path) -> None:
    def outside(rows):
        rows[0]["first_id"] = "없는id"
        return rows

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=outside)
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()
    assert f.detail()["reason"] == "생성 목록에 없다"


def test_구간의_id_가_문자열이_아니면_예외가_아니라_거부다(tmp_path: Path) -> None:
    """해시할 수 없는 값에서 죽지 않는다(19번 Minor 5-4)."""
    def unhashable(rows):
        rows[0]["first_id"] = {"a": 1}
        return rows

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=unhashable)
    assert RejectCode.ATTEMPTS_SCHEMA in f.rejects()


def test_시각_관계_여덟_가운데_죽은_시도의_시작_순서(tmp_path: Path) -> None:
    """관계 8 — 끝 줄 유무와 **무관**하다(§28-4)."""
    def late_first(rows):
        rows[0]["started_at"] = at(30)      # 시도 1 이 시도 2 보다 늦다
        return rows

    f = nine(tmp_path, [(0, None, None), (4, REASON_RESUME, 5)], attempts_edit=late_first)
    assert RejectCode.TIME_ORDER in f.rejects()
    # **짝까지 단언한다** — 곁 파일 쪽 검사가 같은 코드를 내므로 코드만 보면 관계 8 을 지워도 통과한다.
    assert "start<prev_start" in {r.detail.get("pair") for r in _rejections(f)}


def test_시도_줄의_시작이_영수증보다_이르면_거부한다(tmp_path: Path) -> None:
    def early(rows):
        rows[0]["started_at"] = "2026-09-21T08:00:00.000000+09:00"
        return rows

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=early)
    assert RejectCode.TIME_ORDER in f.rejects()


def test_곁_파일_시각이_첫_시작과_다르면_거부한다(tmp_path: Path) -> None:
    """관계 4."""
    f = nine(tmp_path, [(0, None, 9)], meta_over={"started_at": at(59)})
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()
    assert f.detail()["item"] == "started_at"


def test_관계5_곁_파일_종료가_마지막_끝_줄과_다르면_거부한다(tmp_path: Path) -> None:
    f = nine(tmp_path, [(0, None, 9)], meta_over={"finished_at": at(59)})
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()
    assert f.detail()["item"] == "finished_at"


def test_관계6_끝이_같은_시도의_시작보다_이르면_거부한다(tmp_path: Path) -> None:
    def early_end(rows):
        for r in rows:
            if r["event"] == EVENT_END:
                r["finished_at"] = at(1)
        return rows

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=early_end)
    assert RejectCode.TIME_ORDER in f.rejects()
    assert "end<start" in {r.detail.get("pair") for r in _rejections(f)}


def test_관계7_다음_시작이_앞_종료보다_이르면_거부한다(tmp_path: Path) -> None:
    """끝 줄이 있을 때만 걸리는 관계다. 관계 8 은 건드리지 않게 앞 시작보다는 늦게 둔다."""
    def squeeze(rows):
        for r in rows:
            if r["event"] == EVENT_START and r["attempt_no"] == 2:
                r["started_at"] = at(2, 30)
        return rows

    f = nine(tmp_path, [(0, None, 3), (3, REASON_RESUME, 6)], attempts_edit=squeeze)
    pairs = {r.detail.get("pair") for r in _rejections(f)}
    assert "start<prev_end" in pairs and "start<prev_start" not in pairs


def test_이_묶음의_시작_줄이_없으면_거부한다(tmp_path: Path) -> None:
    """다른 도장·다른 묶음의 줄만 있고 곁 파일도 0 이라 적은 경우."""
    def only_other(rows):
        return plan_rows("uni_fed", 2, hx("딴도장"), IDS9, [(0, None, 9)])

    f = nine(tmp_path, [(0, None, 9)], attempts_edit=only_other,
             meta_over={"attempts": 0})
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()
    assert f.detail()["reason"] == "이 묶음의 시작 줄이 없다"


def test_꼬리를_잘라_내고_끝까지_쓴_묶음은_통과한다(tmp_path: Path) -> None:
    """바-16 의 가장 흔한 길 — 앞 시도가 꼬리를 찢고 죽고, 다음 시도가 잘라 내고 완주한다."""
    def cut(rows):
        for r in rows:
            if r["event"] == EVENT_START and r["attempt_no"] == 2:
                r["truncated_ids"] = [IDS9[3]]
        return rows

    got = nine(tmp_path, [(0, None, None), (3, REASON_TORN_TAIL, 6)],
               attempts_edit=cut).verify()
    assert list(got.notes["attempts_spans"]) == [3, 6]
    assert got.notes["attempts_end_rows"] == 1


def test_곁_파일_장비가_마지막_시작_줄과_다르면_거부한다(tmp_path: Path) -> None:
    f = nine(tmp_path, [(0, None, 9)], meta_over={"device": "AMD MI300:1"})
    assert RejectCode.ATTEMPTS_ROW_MISMATCH in f.rejects()
    assert f.detail()["item"] == "device"


def test_두_장비에_걸친_묶음은_거부가_아니라_보고다(tmp_path: Path) -> None:
    """§28-3 나 — 걸친 사실을 보고에 싣는다."""
    other = "NVIDIA A100-SXM4-40GB:0"

    def span(rows):
        for r in rows:
            if r["event"] == EVENT_START and r["attempt_no"] == 2:
                r["device"] = other
        return rows

    got = nine(tmp_path, [(0, None, 3), (3, REASON_RESUME, 6)],
               attempts_edit=span).verify()
    assert got.notes["devices_spanned"] is True
    assert list(got.notes["devices"]) == sorted({DEVICE, other})


def test_보고에_쓰는_시도_횟수의_출처를_적는다(tmp_path: Path) -> None:
    """§13-3 바-17 — 보고에 싣는 값은 곁 파일의 `attempts` 다."""
    got = nine(tmp_path, [(0, None, 9)]).verify()
    assert got.notes["attempts_source"] == "sidecar" and got.notes["attempts"] == 1
    # 원시 식별자를 알림으로 내보내지 않는다(19번 Minor 5-5).
    assert list(got.notes["attempts_first_positions"]) == [0]
    assert "attempts_intervals" not in got.notes


# ------------------------------------------------- 파일 종류별 바이트 규약 (§14-1 가)

def test_곁_파일에_CR_이_있으면_거부한다(tmp_path: Path) -> None:
    """`read_text` 는 CRLF 를 조용히 정규화한다 — 그러면 계약의 거부가 일어나지 않는다."""
    f = make(tmp_path, side_raw_tail=b"\r\n")
    assert RejectCode.CR_IN_FILE in f.rejects()


def test_도장에_CR_이_있으면_해시가_맞아도_거부한다(tmp_path: Path) -> None:
    """정상 도장에 CR 만 붙인다 — 스키마까지 어긴 자료로는 이 경로를 못 본다(19번 Minor 3-1 가)."""
    stamp = start_stamp_of("uni_local_C1", 1, False, validate_eval_list(canonical_bytes(IDS)))
    raw = json.dumps(stamp, ensure_ascii=False).encode("utf-8") + b"\r\n"
    f = make(tmp_path, start_raw=raw)
    assert RejectCode.CR_IN_FILE in f.rejects()
    assert RejectCode.START_STAMP_HASH_MISMATCH not in f.rejects()
    assert RejectCode.START_STAMP_SCHEMA not in f.rejects(), "바이트에서 멈춰야 한다"


def test_시도_기록에_BOM_이_있으면_거부한다(tmp_path: Path) -> None:
    body = jsonl_bytes(plan_rows("uni_local_C1", 1, hx("x"), IDS, [(0, None, 3)]))
    f = make(tmp_path, attempts_raw=b"\xef\xbb\xbf" + body)
    assert RejectCode.BOM_IN_FILE in f.rejects()


def test_시도_기록의_꼬리가_찢어지면_거부한다(tmp_path: Path) -> None:
    body = jsonl_bytes(plan_rows("uni_local_C1", 1, hx("x"), IDS, [(0, None, 3)]))
    f = make(tmp_path, attempts_raw=body[:-1])
    assert RejectCode.TORN_TAIL in f.rejects()


def test_시도_기록에_빈_줄이_있으면_거부한다(tmp_path: Path) -> None:
    body = jsonl_bytes(plan_rows("uni_local_C1", 1, hx("x"), IDS, [(0, None, 3)]))
    f = make(tmp_path, attempts_raw=b"\n" + body)
    assert RejectCode.BLANK_LINE in f.rejects()


def test_tokens_에_CR_이_있으면_해시가_맞아도_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, tokens=b'{"t": 1}\r\n')
    assert RejectCode.CR_IN_FILE in f.rejects()
    assert RejectCode.TOKENS_HASH_MISMATCH not in f.rejects(), "해시는 맞은 채로 걸려야 한다"


def test_tokens_의_꼬리가_찢어지면_거부한다(tmp_path: Path) -> None:
    """tokens 도 줄 단위 파일이다 — `jsonl` 연결을 고정한다(19번 Minor 3-1 나)."""
    f = make(tmp_path, tokens=b'{"t": 1}')
    assert RejectCode.TORN_TAIL in f.rejects()


def test_JSON_문서에는_끝_개행을_요구하지_않는다(tmp_path: Path) -> None:
    """JSONL 의 끝 개행 규칙을 문서에 그대로 옮기지 않는다 — 규약에 없는 것을 만들지 않는다."""
    assert make(tmp_path / "a").verify().tag == "uni_local_C1"          # 끝 개행 없음
    stamp = start_stamp_of("uni_local_C1", 1, False, validate_eval_list(canonical_bytes(IDS)))
    with_lf = json.dumps(stamp, ensure_ascii=False).encode("utf-8") + b"\n"
    assert make(tmp_path / "b", start_raw=with_lf, side_raw_tail=b"\n").verify().mode == MODE_MODEL


# ------------------------------------------------- 검증 뒤 변경 방지 (§26-6)

def test_검증을_지나지_않은_묶음은_조립되지_않는다(tmp_path: Path) -> None:
    """진입점이 이 객체를 직접 만들면 검증을 건너뛴 묶음이 채점에 들어간다."""
    good = make(tmp_path).verify()
    with pytest.raises(TypeError, match="verify_bundle"):
        VerifiedBundle(files=good.files, tag=good.tag, cell=good.cell, client=good.client,
                       seed_index=good.seed_index, seed_value=good.seed_value, mode=good.mode,
                       sidecar=dict(good.sidecar), lines=good.lines, image_ids=good.image_ids)


def test_검증된_묶음을_바꿔_끼울_수_없다(tmp_path: Path) -> None:
    """`replace` 는 초기화 인자를 옮긴다 — 조립 창이 닫혀 있어 걸린다."""
    good = make(tmp_path).verify()
    with pytest.raises(TypeError, match="verify_bundle"):
        replace(good, lines=("딴 줄",))


def test_검증된_묶음의_속성은_대입할_수_없다(tmp_path: Path) -> None:
    good = make(tmp_path).verify()
    with pytest.raises(FrozenInstanceError):
        good.lines = ("딴 줄",)


def test_알림도_바꿀_수_없다(tmp_path: Path) -> None:
    """알림은 보고로 나가는 값이다 — 곁 파일과 같은 규칙으로 둔다."""
    good = make(tmp_path).verify()
    with pytest.raises(TypeError, match="바꿀 수 없다"):
        good.notes["attempts_source"] = "딴것"


def test_곁_파일_항목을_바꿀_수_없다(tmp_path: Path) -> None:
    """판정에 쓰는 내용이다 — 묶음 사이 환경 일치가 검증 뒤에 이것을 다시 읽는다."""
    good = make(tmp_path).verify()
    with pytest.raises(TypeError, match="바꿀 수 없다"):
        good.sidecar["processor_config_sha256"] = hx("딴것")


def test_겹쳐_든_값도_바꿀_수_없다(tmp_path: Path) -> None:
    """겉만 복사하면 안쪽은 같은 객체를 가리킨다."""
    good = make(tmp_path).verify()
    with pytest.raises(TypeError, match="바꿀 수 없다"):
        good.sidecar["chat_template_kwargs"]["add_generation_prompt"] = False


def test_동결한_뒤에도_읽는_쪽이_읽는다(tmp_path: Path) -> None:
    """꼴을 바꾸면 소비하는 자리를 같이 맞춘다 — `json.dumps` 가 그 자리다."""
    good = make(tmp_path).verify()
    assert json.dumps(good.sidecar["chat_template_kwargs"], sort_keys=True) == \
        '{"add_generation_prompt": true}'
    assert check_env_consistency([good]) == []


# ---------------------------------------------------------------- 진입점 과제 2 (3판 1-8 · 07번 §30-4 · §30-9)

def test_토큰_파일의_줄_수가_생성과_다르면_거부한다(tmp_path: Path) -> None:
    rows = [row(i, n_new_tokens=5) for i in IDS]
    f = make(tmp_path, rows=rows, tokens=tokens_of(IDS[:2], 5))
    assert f.rejects() == {RejectCode.TOKENS_CONTENT_MISMATCH}
    assert f.detail()["reason"] == "줄 수"


def test_토큰_파일의_id_수열이_다르면_거부한다(tmp_path: Path) -> None:
    rows = [row(i, n_new_tokens=5) for i in IDS]
    f = make(tmp_path, rows=rows, tokens=tokens_of([IDS[1], IDS[0], IDS[2]], 5))
    assert f.detail()["reason"] == "id 수열"


@pytest.mark.parametrize("tok", [
    lambda i: {"image_id": i, "token_ids": [1] * 4, "token_logprobs": [-0.5] * 5},
    lambda i: {"image_id": i, "token_ids": [1] * 4, "token_logprobs": [-0.5] * 4},
    lambda i: {"image_id": i, "token_ids": [1] * 5},
])
def test_토큰_줄의_길이가_n_new_tokens_와_다르면_거부한다(tmp_path: Path, tok) -> None:
    rows = [row(i, n_new_tokens=5) for i in IDS]
    f = make(tmp_path, rows=rows, tokens=body_of(tok(i) for i in IDS))
    assert f.detail()["reason"] == "길이"


@pytest.mark.parametrize("purpose", ["main", "frame_diag"])
def test_본실험과_진단은_고의_중단_기록이_있으면_거부한다(tmp_path: Path, purpose: str) -> None:
    f = make(tmp_path, purpose=purpose, meta_over={"fault_events": [{"at": "stage_c", "kind": "kill"}]})
    assert f.rejects() == {RejectCode.FAULT_EVENTS_PRESENT}


def test_리허설은_고의_중단_기록을_알림에_옮긴다(tmp_path: Path) -> None:
    ev = [{"at": "stage_c", "kind": "kill"}]
    vb = make(tmp_path, purpose="rehearsal", meta_over={"fault_events": ev}).verify()
    assert list(vb.notes["fault_events"]) == [dict(e) for e in ev]


def _impl_registration(f, value):
    f.registration = replace(f.registration, values={**f.registration.values, "impl_ids": value})
    return f


OTHER_KEYS = ("export_generator=vlm.other:load_generator@vlm/other.py",
              "export_preflight=vlm.export_run:preflight@vlm/export_run.py")
UNAPPROVED = [  # (식별자 덮어쓰기, 사유 조각)
    ({"approved": False}, "승인 목록 밖"),
    ({"in_repo": False}, "저장소 밖"),
    ({"blob": "b" * 40}, "커밋되지 않은 코드"),
    ({"blob": None, "head": None}, "커밋되지 않은 코드"),
]


def _sidecar_impl(**over) -> dict:
    return {"export_preflight": IMPL_IDS["export_preflight"],
            "export_generator": impl("export_generator", "load_generator", **over)}


@pytest.mark.parametrize("purpose", ["main", "frame_diag"])
def test_승인_구현이_아니면_본실험과_진단은_그_칸을_거부한다(tmp_path: Path, purpose: str) -> None:
    f = _impl_registration(make(tmp_path, purpose=purpose), OTHER_KEYS)
    assert f.rejects() == {RejectCode.REGISTRATION_ITEM_MISMATCH}
    assert f.detail()["item"] == "impl_ids"


@pytest.mark.parametrize(("over", "why"), UNAPPROVED)
@pytest.mark.parametrize("purpose", ["main", "frame_diag"])
def test_열쇠가_같아도_승인된_실제_구현이_아니면_거부한다(tmp_path: Path, purpose: str, over: dict, why: str) -> None:
    """이름 문자열만 맞대면 대역이 같은 이름을 적는 것을 막지 못한다 — 승인 표시 · 저장소 · blob 을 본다(검수 16번 I-4)."""
    f = make(tmp_path, purpose=purpose, meta_over={"impl_ids": _sidecar_impl(**over)})
    d = f.detail()
    assert d["item"] == "impl_ids" and any(why in x for x in d["problems"])


@pytest.mark.parametrize(("over", "why"), UNAPPROVED)
def test_승인_구현이_아니면_리허설은_대역_실행으로_적는다(tmp_path: Path, over: dict, why: str) -> None:
    assert make(tmp_path, purpose="rehearsal", meta_over={"impl_ids": _sidecar_impl(**over)}).verify().notes["stand_in"]


def test_열쇠가_다르면_리허설은_대역_실행으로_적는다(tmp_path: Path) -> None:
    f = _impl_registration(make(tmp_path, purpose="rehearsal"), OTHER_KEYS)
    assert f.verify().notes["stand_in"] is True


def test_승인_구현이면_대역이_아니다(tmp_path: Path) -> None:
    """등록의 열쇠 튜플과 곁 파일의 식별자 사전에서 낸 열쇠는 같은 것으로 본다. 목록 꼴(어댑터 meta)도 받는다."""
    assert make(tmp_path, purpose="rehearsal").verify().notes["stand_in"] is False
    as_list = list(IMPL_IDS.values())[::-1]
    assert make(tmp_path / "l", purpose="rehearsal", meta_over={"impl_ids": as_list}).verify().notes["stand_in"] is False


def test_리허설의_면제는_impl_ids_하나뿐이다(tmp_path: Path) -> None:
    """검수 16번 M-11 — 리허설이어도 다른 항목의 어긋남은 거부한다."""
    f = make(tmp_path, purpose="rehearsal")
    f.registration = replace(f.registration, values={**f.registration.values, "max_new_tokens": 512})
    assert f.rejects() == {RejectCode.REGISTRATION_ITEM_MISMATCH}
    assert f.detail()["item"] == "max_new_tokens"


# ---------------------------------------------------------------- 등록 투영의 결속 (검수 16번 I-5)

@pytest.mark.parametrize(("edit", "why"), [
    (lambda r: replace(r, purpose="rehearsal"), "purpose"),
    (lambda r: replace(r, values={k: v for k, v in r.values.items() if k != "impl_ids"}), "impl_ids"),
    (lambda r: replace(r, values={**r.values, "eval_list_file_sha256": "f" * 64}), "목록 해시"),
    (lambda r: replace(r, purpose="probe", values={**r.values, "purpose": "probe"}), "어휘"),
])
def test_등록_투영이_스스로와_묶음의_모드와_맞지_않으면_거부한다(tmp_path: Path, edit, why: str) -> None:
    """목적은 대역 면제와 고의 중단 통과를 여는 스위치다 — 값과 다른 목적, 다른 모드의 투영으로는 검증하지 않는다."""
    f = make(tmp_path, meta_over={"impl_ids": _sidecar_impl(approved=False),
                                  "fault_events": [{"kind": "kill"}]})
    f.registration = edit(f.registration)
    d = f.detail()
    assert d["item"] == "registration_projection" and why in d["reason"]


def test_본실험_값에_리허설_목적을_붙여도_대역과_고의_중단이_지나지_않는다(tmp_path: Path) -> None:
    f = make(tmp_path, purpose="main", meta_over={"impl_ids": _sidecar_impl(approved=False),
                                                  "fault_events": [{"kind": "kill"}]})
    f.registration = replace(f.registration, purpose="rehearsal")
    assert f.rejects() == {RejectCode.REGISTRATION_ITEM_MISMATCH}


def test_에코_묶음에_모델_모드의_투영을_주면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, echo=True)
    f.registration = replace(f.registration, echo_list=validate_eval_list(canonical_bytes(IDS)),
                             values={**f.registration.values, "eval_list_set_sha256": "e" * 64})
    assert "모드" in f.detail()["reason"]


# ---------------------------------------------------------------- 학습 행 digest — 어댑터 meta 와 (검수 16번 I-4 나)

def test_학습_행_digest_는_곁_파일이_아니라_어댑터_meta_와_맞댄다(tmp_path: Path) -> None:
    f = make(tmp_path, purpose="frame_diag")
    f.registration = replace(f.registration, train_rows_digest=hx("rows"))
    f.artifact = replace(f.artifact, train_rows_digest=hx("rows"))
    assert f.verify()
    f.artifact = replace(f.artifact, train_rows_digest=hx("다른 행"))
    assert f.detail() == {"item": "train_rows_digest", "reason": "어댑터 meta 의 값이 등록과 다르다"}
    f.artifact = replace(f.artifact, train_rows_digest=None)
    assert f.rejects() == {RejectCode.LEDGER_MISMATCH}


def test_학습_행_digest_가_등록에_없으면_보지_않는다(tmp_path: Path) -> None:
    f = make(tmp_path)
    f.artifact = replace(f.artifact, train_rows_digest=hx("아무거나"))
    assert f.verify()


def test_곁_파일은_학습_행_digest_를_싣지_않아도_된다(tmp_path: Path) -> None:
    """쓰는 쪽 2판 반영판 §2-5 의 곁 파일 표에 없다. 진단 묶음이 이 키 없이 지나야 한다."""
    f = make(tmp_path, purpose="frame_diag")
    f.registration = replace(f.registration, train_rows_digest=hx("rows"))
    f.artifact = replace(f.artifact, train_rows_digest=hx("rows"))
    assert "train_rows_digest" not in json.loads(f.files.sidecar.read_text(encoding="utf-8"))
    assert f.verify()


# ---------------------------------------------------------------- 파일은 한 번 읽는다 (검수 16번 M-8)

def test_파일마다_한_번_읽는다(tmp_path: Path, monkeypatch) -> None:
    f = make(tmp_path, rows=[row(i, n_new_tokens=5) for i in IDS], tokens=tokens_of(IDS, 5))
    counts: dict[str, int] = {}
    real = Path.read_bytes

    def counting(self):
        counts[self.name] = counts.get(self.name, 0) + 1
        return real(self)
    monkeypatch.setattr(Path, "read_bytes", counting)
    f.verify()
    assert all(n == 1 for n in counts.values()), counts
    assert counts.get(f.files.tokens.name) == 1


def test_도장의_목적이_곁_파일과_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, start_over={"purpose": "rehearsal"})
    assert RejectCode.START_STAMP_MISMATCH in f.rejects()


@pytest.mark.parametrize(("over", "detail"), [
    ({"purpose": "probe"}, {"not_vocab": "purpose"}),
    ({"impl_ids": "export.main_uni"}, {"not_impl_ids": "impl_ids"}),
    ({"impl_ids": ["export.main_uni"]}, {"not_impl_ids": "impl_ids"}),
    ({"impl_ids": []}, {"not_impl_ids": "impl_ids"}),
    ({"impl_ids": {"export_generator": impl("export_preflight", "preflight")}}, {"not_impl_ids": "impl_ids"}),
    ({"impl_ids": [impl("x", "a"), impl("x", "b")]}, {"not_impl_ids": "impl_ids"}),
    ({"impl_ids": [{**impl("x", "a"), "extra": 1}]}, {"not_impl_ids": "impl_ids"}),
    ({"impl_ids": [{**impl("x", "a"), "approved": 1}]}, {"not_impl_ids": "impl_ids"}),
    ({"impl_ids": [impl("x", "a", blob="xyz")]}, {"not_impl_ids": "impl_ids"}),
    ({"fault_events": None}, {"not_list": "fault_events"}),
    ({"train_config": {"a": 1}}, {"not_str": "train_config"}),
])
def test_새_필수_키의_꼴이_틀리면_거부한다(tmp_path: Path, over: dict, detail: dict) -> None:
    f = make(tmp_path, meta_over=over)
    assert f.detail() == detail


@pytest.mark.parametrize("key", ["purpose", "train_config", "impl_ids", "fault_events",
                                 "list_split", "start_checkpoint_sha256", "init_adapter_digest"])
def test_새_필수_키가_빠지면_거부한다(tmp_path: Path, key: str) -> None:
    assert make(tmp_path, drop_meta=(key,)).detail()["missing"] == [key]


def test_학습_설정_원문이_정규형이_아니면_거부한다(tmp_path: Path) -> None:
    text = json.dumps(json.loads(train_config_text()), indent=1)
    f = make(tmp_path, meta_over={"train_config": text,
                                  "train_config_sha256": hashlib.sha256(text.encode()).hexdigest()})
    assert f.detail()["reason"] == "정규화 JSON 원문이 아니다"


def test_학습_설정_원문의_해시가_다르면_거부한다(tmp_path: Path) -> None:
    f = make(tmp_path, meta_over={"train_config": train_config_text(lora={"r": 8})})
    assert f.detail()["reason"] == "다시 해시한 값이 train_config_sha256 과 다르다"


@pytest.mark.parametrize(("key", "value"), [
    ("prompt_sha256", hx("딴프롬프트")), ("template_mode", "enable_thinking=False"),
    ("coord_cfg_hash", hx("딴좌표")), ("model_id", "local/other"), ("model_revision", "x"),
    ("num_rounds", 4), ("local_epochs", 9), ("total_epochs", "30"), ("init_adapter_digest", None),
    ("processor_config_sha256", hx("딴프로세서")),
])
def test_학습_설정_원문의_값이_대응_표대로_곁_파일과_같아야_한다(tmp_path: Path, key: str, value) -> None:
    text = train_config_text(**{key: value})
    f = make(tmp_path, meta_over={"train_config": text,
                                  "train_config_sha256": hashlib.sha256(text.encode()).hexdigest()})
    d = f.detail()
    assert d["reason"] == "대응 표의 값이 곁 파일과 다르다" and d["keys"] == [key]


def test_짝이_없는_학습_설정_키는_대조하지_않는다(tmp_path: Path) -> None:
    text = train_config_text(lora={"r": 64}, pairs_digest=hx("다른 페어"))
    assert make(tmp_path, meta_over={"train_config": text,
                                     "train_config_sha256": hashlib.sha256(text.encode()).hexdigest()}).verify()


def test_학습_설정의_대응_키가_빠지면_거부한다(tmp_path: Path) -> None:
    cfg = json.loads(train_config_text())
    del cfg["num_rounds"]
    text = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    f = make(tmp_path, meta_over={"train_config": text,
                                  "train_config_sha256": hashlib.sha256(text.encode()).hexdigest()})
    assert f.detail()["keys"] == ["num_rounds"]


def test_에코_곁_파일도_학습_설정_원문을_싣는다(tmp_path: Path) -> None:
    assert make(tmp_path, echo=True).verify().is_echo
    assert make(tmp_path / "e2", echo=True, drop_meta=("train_config",)).detail()["missing"] == ["train_config"]


def test_계약_키_목록은_비어_있지_않다() -> None:
    """이름을 지우면 검사가 통째로 꺼진다 — 목록 자체를 고정한다."""
    assert {"mode", "tag", "train_run_id", "started_at", "device"} <= set(START_STAMP_REQUIRED)
    assert len(START_STAMP_REQUIRED) == 10 and len(START_STAMP_ECHO_ABSENT) == 3
    assert "purpose" in START_STAMP_REQUIRED, "07번 §30-4"
    assert {"purpose", "train_config", "impl_ids", "fault_events"} <= set(SIDECAR_REQUIRED), "07번 §30-4"
    assert {"list_split", "start_checkpoint_sha256", "init_adapter_digest"} <= set(SIDECAR_REQUIRED), "검수 16번 M-2"
    assert {"event", "attempt_no", "start_stamp_sha256"} <= set(ATTEMPTS_COMMON_REQUIRED)
    assert {"reason", "started_at", "first_id", "device"} == set(ATTEMPTS_START_REQUIRED)
    assert {"last_id", "finished_at", "n_written"} == set(ATTEMPTS_END_REQUIRED)
    assert ATTEMPT_REASONS == (REASON_FIRST, REASON_RESUME, REASON_TORN_TAIL)


# ---------------------------------------------------------------- 묶음 사이

def test_한_묶음만으로는_환경을_맞댈_수_없다(tmp_path: Path) -> None:
    assert check_env_consistency([make(tmp_path).verify()]) == []


def test_환경이_다르면_무엇이_다른지_말한다(tmp_path: Path) -> None:
    a = make(tmp_path / "a", tag="uni_local_C1").verify()
    b = make(tmp_path / "b", tag="uni_fed", meta_over={"transformers_version": "4.58.0"}).verify()
    out = check_env_consistency([a, b])
    COVERED.update(r.code for r in out)
    assert [r.code for r in out] == [RejectCode.ENV_MISMATCH]
    assert out[0].detail["item"] == "transformers_version"


def test_환경이_같으면_비어_있다(tmp_path: Path) -> None:
    a = make(tmp_path / "a", tag="uni_local_C1").verify()
    b = make(tmp_path / "b", tag="uni_fed").verify()
    assert check_env_consistency([a, b]) == []


def test_맞대는_환경_항목이_곁_파일_필수_키_안에_있다() -> None:
    assert set(ENV_FIELDS) <= set(SIDECAR_REQUIRED)


def test_모집단_신원은_목록_집합_해시와_같은_함수다() -> None:
    assert population_digest(IDS) == set_digest(IDS)
    assert population_digest(IDS) == population_digest(reversed(IDS))


# ---------------------------------------------------------------- 메타

def test_이_파일이_덮은_코드를_표와_맞댄다() -> None:
    """무엇을 안 덮었는지 세어 둔다 — 세지 않으면 조건이 늘어도 시험이 그대로다."""
    canary = {RejectCode.LITERAL_MATCH_FAILED, RejectCode.PARSER_DISAGREEMENT}
    elsewhere = {RejectCode.LIST_NOT_CANONICAL}      # `test_eval_list.py` 가 세운다
    todo = set(RejectCode) - COVERED - canary - elsewhere
    assert todo == set(), f"아직 시험이 없는 거부 사유: {sorted(c.value for c in todo)}"
    assert all(STAGE_OF[c] is Stage.CANARY for c in canary), "카나리아는 별도 하네스에서 선다"
    assert not (COVERED & canary), "카나리아가 여기서 섰다면 이 목록을 줄여야 한다"
