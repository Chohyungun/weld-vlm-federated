"""export 쓰기 객체 — 한 묶음의 파일 다섯을 **정해진 순서로만** 쓴다(리허설 2판 §2-1 ~ §2-4, 평가 계약 §13-3 · §14-1).

파일은 `{stem}.export_start.json`(시작 도장) · `attempts.jsonl`(폴더 공용 시도 기록) · `{stem}.generations.jsonl` ·
`{stem}.tokens.jsonl`(선택) · `{stem}.export_meta.json`(곁 파일)이다. `stem = {tag}_s{n}`, 에코는 `{tag}_s{n}.echo`.
모두 UTF-8 · 줄끝 LF · BOM 없음이고 바이트로 쓴다.

## 상태 기계

`NEW → STAMPED → STARTED → WRITING → ENDED → SEALED`. 생성 파일은 `STARTED` 에서 부른 `open_generations()` 만 연다.
`open_generations()` 는 파일을 열기 전에 시도 기록의 마지막 줄과 도장의 바이트를 **다시 읽어** 방금 쓴 것과 같은지 본다.
핸들은 밖으로 내주지 않는다 — 줄은 `write()` 로만 쓰고, 그 함수가 목록 순서와 "토큰 줄 → 생성 줄" 의 순서를 지킨다.
도장 · 시작 줄 · 줄 하나 · 끝 줄 · 곁 파일 모두 쓴 뒤 `flush` → `fsync` 하고 다음으로 간다. 시도 기록의 한 줄은 `write` 한 번이다.
도장과 곁 파일은 tmp 에 쓰고 fsync 한 뒤 `os.rename` 으로 이름을 준다 — 대상이 있으면 실패하므로 배타 생성의 뜻이 선다.
시각은 역행하지 않는다 — 새 프로세스는 이 도장의 마지막 기록 시각 이상에서만 시작하고, 쓸 때마다 앞 시각 이상인지 본다.

## 시작 절차(`begin`) — 쓰기 전에 거부할 것을 먼저 거부한다

1. 시도 기록 전체를 평가 쪽 규칙(바이트 · 줄마다 JSON 객체 · 공통 키 · `event` 두 값)으로 본다. 어긋나면 덧붙이지 않고 멈춘다.
   끝 줄을 덧붙이기 전에도 같은 검사를 한다.
2. 곁 파일이 있으면 완결 묶음이다 — 시작 줄을 쓰지 않고 멈춘다.
3. 도장이 있으면 이어 갈 수 있는지 본다(읽힘 · 딸린 시작 줄 · 지문 · 생성 줄의 id 가 목록의 앞부분 · 코드와 파서의 판 · 멈춘 개명).
   이어 갈 수 없으면 기본은 거부다. `abort_mismatched=True` 일 때만 생성 → 토큰 → 도장 순으로 `.aborted-{k}` 를 붙이고 4 로 간다.
4. 도장이 없으면 쓰고 `attempt_no=1` · `first_pass` 다. 도장 없이 생성 · 토큰 파일만 있으면 거부한다.
5. 도장이 맞으면 생성 파일의 완결 줄 수 `m` 을 센다. 꼬리가 찢겼으면 **먼저** `retry_torn_tail` 시작 줄을 쓰고 되읽은 뒤,
   찢긴 바이트를 `{stem}.truncated-{attempt_no}.bin` 으로 남기고 잘라 낸다. 토큰 파일은 따로 판정한다.
6. 시작 줄을 쓰고 되읽는다. 그 뒤에만 생성 파일을 연다.

이 모듈은 등록 · 영수증 · 실측값 대조를 모른다 — 그 점검은 호출부(`vlm/export_run.py`)의 이음새다.
목록은 평가 쪽 검사(`evaluation.eval_list.validate_eval_list`)를 지난 `EvalList` 로만 받는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "MODES",
    "RESUME_KEYS",
    "STAMP_KEYS",
    "ExportRefused",
    "ExportStateError",
    "ExportWriter",
    "StartPlan",
    "check_attempts_file",
    "stem_of",
]

MODES = ("model", "echo")
REASON_FIRST, REASON_RESUME, REASON_TORN = "first_pass", "resume_after_interruption", "retry_torn_tail"
#: 호출부가 도장에 넣는 값. 나머지(`tag` · `seed_index` · `mode` · 목록 해시 둘 · `started_at` · `device`)는 이 객체가 낸다.
STAMP_KEYS = ("purpose", "export_commit", "parser_sha256", "generation_sha256",
              "train_run_id", "train_ledger_sha256", "scored_adapter_sha256")
#: 에코 도장 · 곁 파일에 **키 자체가 없어야** 하는 것(평가 계약 §28-3 가 · 다).
ECHO_ABSENT_STAMP = ("train_run_id", "train_ledger_sha256", "scored_adapter_sha256")
ECHO_ABSENT_SIDECAR = ("train_run_id", "train_ledger_sha256", "train_ledger_path", "scored_adapter_sha256",
                       "scored_adapter_step")
#: 이어 쓰려면 도장과 지금 값이 같아야 하는 것(리허설 2판 §2-4 의 3).
RESUME_KEYS = ("generation_sha256", "scored_adapter_sha256", "eval_list_file_sha256", "eval_list_set_sha256",
               "mode", "purpose", "tag", "seed_index", "train_run_id", "train_ledger_sha256",
               "export_commit", "parser_sha256")
_ATTEMPT_COMMON = ("event", "tag", "seed_index", "attempt_no", "start_stamp_sha256")
_RENAME_TRIES = 5


class ExportRefused(ValueError):
    """시작하지 않거나 쓰지 않는다. `code` 가 사유다."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


class ExportStateError(RuntimeError):
    """상태 기계를 건너뛰는 호출 — 쓰는 순서를 어기려 했다."""


@dataclass(frozen=True)
class StartPlan:
    attempt_no: int
    reason: str
    first_id: str | None
    m: int
    """이 시도가 시작할 때 생성 파일에 있던 완결 줄 수."""
    truncated_ids: tuple[str, ...] = ()


def stem_of(tag: str, seed_index: int, mode: str) -> str:
    return f"{tag}_s{int(seed_index)}" + (".echo" if mode == "echo" else "")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _write_all(path: Path, data: bytes, mode: str) -> None:
    with open(path, mode) as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _tmp_then_rename(final: Path, data: bytes) -> None:
    """tmp 에 쓰고 fsync 한 뒤 이름을 준다. 대상이 이미 있으면 쓰지 않는다. 남은 tmp(앞 프로세스의 잔해)는 덮는다."""
    if final.exists():
        raise ExportRefused("exists", f"이미 있다: {final.name}")
    tmp = final.with_name(final.name + ".tmp")
    _write_all(tmp, data, "wb")
    os.rename(tmp, final)


def _rename_retry(src: Path, dst: Path) -> None:
    """다른 프로세스가 잡고 있으면(WinError 32) 정해진 횟수만 다시 시도하고, 그래도 안 되면 올린다."""
    for n in range(_RENAME_TRIES):
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if n == _RENAME_TRIES - 1:
                raise
            time.sleep(0.2)


def check_attempts_file(path: Path) -> list[dict]:
    """시도 기록 전체를 평가 쪽 규칙으로 본다. 어긋나면 `ExportRefused` — 덧붙이지 않는다. 줄 목록을 돌려준다."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):
        raise ExportRefused("attempts_broken", f"시도 기록의 바이트가 규약 밖이다(BOM · CR · 끝 개행): {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ExportRefused("attempts_broken", f"시도 기록이 UTF-8 이 아니다: {path}") from None
    rows = []
    for i, line in enumerate(text[:-1].split("\n"), 1):
        if not line.strip():
            raise ExportRefused("attempts_broken", f"시도 기록 {i} 번째 줄이 비었다")
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            raise ExportRefused("attempts_broken", f"시도 기록 {i} 번째 줄이 JSON 이 아니다") from None
        if not isinstance(row, dict) or any(k not in row for k in _ATTEMPT_COMMON) \
                or row["event"] not in ("start", "end"):
            raise ExportRefused("attempts_broken", f"시도 기록 {i} 번째 줄의 꼴이 틀리다")
        rows.append(row)
    return rows


def _complete(path: Path) -> tuple[int, int, bytes]:
    """(완결 줄 수, 완결 바이트 길이, 찢긴 꼬리 바이트). 파일이 없으면 (0, 0, b"")."""
    if not path.exists():
        return 0, 0, b""
    raw = path.read_bytes()
    cut = raw.rfind(b"\n") + 1
    return raw[:cut].count(b"\n"), cut, raw[cut:]


def _aware(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ExportRefused("clock_unreadable", f"기록의 시각을 읽을 수 없다: {value!r}")
    try:
        t = datetime.fromisoformat(value)
    except ValueError:
        raise ExportRefused("clock_unreadable", f"기록의 시각을 읽을 수 없다: {value!r}") from None
    if t.tzinfo is None:
        raise ExportRefused("clock_unreadable", f"기록의 시각에 오프셋이 없다: {value!r}")
    return t


def _default_clock() -> datetime:
    return datetime.now().astimezone()


class ExportWriter:
    """한 묶음(태그 · 시드 · 모드)의 쓰기 객체. 파일을 여는 것은 이 객체 하나다."""

    def __init__(self, folder: str | Path, *, tag: str, seed_index: int, mode: str, eval_list,
                 stamp_fields: Mapping[str, Any], device: str, with_tokens: bool = False,
                 clock: Callable[[], datetime] | None = None, abort_mismatched: bool = False):
        from evaluation.eval_list import EvalList

        if mode not in MODES:
            raise ExportRefused("mode", f"모드는 {MODES} 가운데 하나다: {mode!r}")
        if not isinstance(eval_list, EvalList):
            raise ExportRefused("eval_list_unchecked", "목록은 validate_eval_list 를 지난 EvalList 로만 받는다")
        fields = dict(stamp_fields)
        unknown = sorted(set(fields) - set(STAMP_KEYS))
        if unknown:
            raise ExportRefused("stamp_unknown_key", f"도장에 넣지 않는 키다(이 객체가 내거나 없는 키): {unknown}")
        if mode == "echo":
            present = [k for k in ECHO_ABSENT_STAMP if k in fields]
            if present:
                raise ExportRefused("echo_adapter_present", f"에코 도장에는 이 키가 없어야 한다: {present}")
        need = [k for k in STAMP_KEYS if not (mode == "echo" and k in ECHO_ABSENT_STAMP) and not fields.get(k)]
        if need:
            raise ExportRefused("stamp_incomplete", f"도장에 필요한 값이 없다: {need}")
        self.folder = Path(folder)
        self.tag, self.seed_index, self.mode = str(tag), int(seed_index), mode
        self.ids = list(eval_list.ids)
        self.fields = {**fields, "tag": self.tag, "seed_index": self.seed_index, "mode": self.mode,
                       "eval_list_file_sha256": eval_list.file_sha256, "eval_list_set_sha256": eval_list.set_sha256}
        self.device = str(device)
        self.with_tokens = bool(with_tokens)
        self.clock = clock or _default_clock
        self.abort_mismatched = bool(abort_mismatched)
        s = stem_of(self.tag, self.seed_index, self.mode)
        self.stem = s
        self.p_gen = self.folder / f"{s}.generations.jsonl"
        self.p_tok = self.folder / f"{s}.tokens.jsonl"
        self.p_stamp = self.folder / f"{s}.export_start.json"
        self.p_side = self.folder / f"{s}.export_meta.json"
        self.p_att = self.folder / "attempts.jsonl"
        self.state = "NEW"
        self.plan: StartPlan | None = None
        self.stamp_sha: str | None = None
        self._start_line: bytes | None = None
        self._last_time: datetime | None = None
        self._gen_fh = None
        self._tok_fh = None
        self.n_written = 0
        self.last_id: str | None = None

    # ------------------------------------------------------------------ 시각
    def _tick(self) -> str:
        t = self.clock()
        if t.tzinfo is None:
            raise ExportRefused("clock_naive", "시각에 오프셋이 없다")
        if self._last_time is not None and t < self._last_time:
            raise ExportRefused("clock_back", f"시계가 뒤로 갔다: {t.isoformat()} < {self._last_time.isoformat()}")
        self._last_time = t
        return t.isoformat(timespec="microseconds")

    def _need(self, *states: str) -> None:
        if self.state not in states:
            raise ExportStateError(f"{self.state} 에서는 할 수 없다 — {states} 에서만")

    # ------------------------------------------------------------------ 도장 · 이어 쓰기 판정
    def _rows_of(self, rows: list[dict], stamp_sha: str) -> list[dict]:
        return [r for r in rows if r.get("tag") == self.tag and type(r.get("seed_index")) is int
                and r["seed_index"] == self.seed_index and r.get("start_stamp_sha256") == stamp_sha]

    def _aborted(self, p: Path, k: int) -> Path:
        return p.with_name(f"{p.name}.aborted-{k}")

    def _half_renamed_k(self) -> int | None:
        """생성 · 토큰은 `.aborted-k` 로 갔는데 도장은 아직 제 이름인 k — 앞 프로세스가 개명 도중에 죽었다."""
        ks = []
        for p in (self.p_gen, self.p_tok):
            for q in self.folder.glob(f"{p.name}.aborted-*"):
                tail = q.name.rsplit("-", 1)[-1]
                if tail.isdigit() and not self._aborted(self.p_stamp, int(tail)).exists():
                    ks.append(int(tail))
        return max(ks) if ks else None

    def _resume_problems(self, rows: list[dict]) -> list[str]:
        raw = self.p_stamp.read_bytes()
        try:
            stamp = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return ["도장을 JSON 으로 읽을 수 없다"]
        if not isinstance(stamp, dict):
            return ["도장이 객체가 아니다"]
        out: list[str] = []
        if self._half_renamed_k() is not None:
            out.append("앞 프로세스의 개명이 도중에 멈췄다(생성 · 토큰만 .aborted 로 갔다)")
        mine = self._rows_of(rows, _sha(raw))
        if not any(r["event"] == "start" for r in mine):
            out.append("도장에 딸린 시작 줄이 없다 — 도장을 쓰고 시작 줄 전에 죽었다")
        for k in RESUME_KEYS:
            if stamp.get(k) != self.fields.get(k):
                out.append(f"{k} 가 도장과 다르다")
        if not self.with_tokens and self.p_tok.exists():
            out.append("토큰 파일을 켠 시도와 끈 시도가 섞인다")
        m, _, tail = _complete(self.p_gen)
        if m > len(self.ids) or (m == len(self.ids) and tail):
            out.append("생성 파일의 줄이 목록보다 많다")
        elif m:
            got = []
            for line in self.p_gen.read_bytes().split(b"\n")[:m]:
                try:
                    got.append(json.loads(line).get("image_id"))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    got.append(None)
            if got != self.ids[:m]:
                out.append("생성 파일의 id 수열이 목록의 앞부분과 다르다")
        return out

    def _abort_rename(self) -> int:
        """생성 → 토큰 → 도장(마지막) 순으로 `.aborted-{k}`.

        k 는 세 이름 모두에서 비어 있는 가장 작은 양의 정수다. 앞 프로세스가 도중에 죽어 생성 · 토큰만 옮겨졌으면
        그 k 로 나머지를 마저 옮긴다 — 한 시도의 파일이 같은 번호를 갖게 한다.
        """
        k = self._half_renamed_k()
        if k is None or any(self._aborted(p, k).exists() for p in (self.p_gen, self.p_tok) if p.exists()):
            k = 1
            while any(self._aborted(p, k).exists() for p in (self.p_gen, self.p_tok, self.p_stamp)):
                k += 1
        for p in (self.p_gen, self.p_tok, self.p_stamp):
            if p.exists():
                _rename_retry(p, self._aborted(p, k))
        return k

    def _write_stamp(self) -> None:
        stamp = {**self.fields, "started_at": self._tick(), "device": self.device}
        _tmp_then_rename(self.p_stamp, _dumps(stamp))

    # ------------------------------------------------------------------ 시작 절차
    def begin(self, *, after_stamp: Callable[[], None] | None = None) -> StartPlan:
        """시작 절차 1~6. 시작 줄을 쓰고 되읽은 데까지 간다. 돌려주는 계획의 `first_id` 가 None 이면 쓸 줄이 없다.

        `after_stamp` 는 새 도장을 쓴 직후 · 시작 줄 전에 부른다 — 고의 중단 `export:after_stamp` 의 자리다(합성 시험).
        """
        self._need("NEW")
        rows = check_attempts_file(self.p_att)                                 # 1
        if self.p_side.exists():                                               # 2
            raise ExportRefused("bundle_complete", f"완결 묶음이다(곁 파일이 있다): {self.p_side.name}")
        if self.p_stamp.exists():                                              # 3
            problems = self._resume_problems(rows)
            if problems:
                if not self.abort_mismatched:
                    raise ExportRefused("stamp_mismatch", "이어 쓸 수 없다 — " + " · ".join(problems))
                self._abort_rename()
        if not self.p_stamp.exists():                                          # 4
            orphan = [p.name for p in (self.p_gen, self.p_tok) if p.exists()]
            if orphan:
                raise ExportRefused("orphan_generations", f"도장 없이 남은 파일이 있다: {orphan}")
            self.folder.mkdir(parents=True, exist_ok=True)
            self._write_stamp()
            self.stamp_sha = _sha(self.p_stamp.read_bytes())
            self.state = "STAMPED"
            if after_stamp is not None:
                after_stamp()
            plan = StartPlan(1, REASON_FIRST, self.ids[0] if self.ids else None, 0)
            self._write_start(plan)
            return plan
        # 5 — 도장이 맞다. 새 프로세스의 시각이 이 도장의 마지막 기록 시각 이상인지 본다.
        raw_stamp = self.p_stamp.read_bytes()
        self.stamp_sha = _sha(raw_stamp)
        mine = self._rows_of(rows, self.stamp_sha)
        seen = [json.loads(raw_stamp).get("started_at")]
        seen += [r["started_at"] if r["event"] == "start" else r.get("finished_at") for r in mine]
        latest = max(_aware(s) for s in seen)
        now = self.clock()
        if now.tzinfo is None or now < latest:
            raise ExportRefused("clock_back", f"지금 시각 {now.isoformat()} 이 이 도장의 마지막 기록 "
                                              f"{latest.isoformat()} 보다 이르다")
        self._last_time = latest
        self.state = "STAMPED"
        attempt_no = max(int(r["attempt_no"]) for r in mine if r["event"] == "start") + 1
        m, cut, tail = _complete(self.p_gen)
        tm, _, ttail = _complete(self.p_tok) if self.with_tokens else (m, 0, b"")
        if self.with_tokens and tm not in (m, m + 1):
            raise ExportRefused("tokens_mismatch", f"토큰 파일의 완결 줄 {tm} 이 생성 파일 {m} 과 맞지 않는다")
        first = self.ids[m] if m < len(self.ids) else None
        if tail:
            plan = StartPlan(attempt_no, REASON_TORN, first, m, (first,))
        else:
            plan = StartPlan(attempt_no, REASON_RESUME, first, m)
        self._write_start(plan)                                                # 시작 줄을 **먼저**
        if tail:
            # 찢긴 바이트를 남기고(fsync) 생성 파일에서 잘라 낸다. 완결 줄은 한 바이트도 바꾸지 않는다.
            _write_all(self.folder / f"{self.stem}.truncated-{attempt_no}.bin", tail, "xb")
            self._truncate(self.p_gen, cut)
        if self.with_tokens and (tm == m + 1 or ttail):
            # 생성 줄을 얻지 못한 id 의 토큰 줄 하나(와 찢긴 꼬리)를 잘라 낸다. 그 바이트도 남긴다.
            raw = self.p_tok.read_bytes()
            keep = self._prefix_len(raw, m)
            _write_all(self.folder / f"{self.stem}.tokens.truncated-{attempt_no}.bin", raw[keep:], "xb")
            self._truncate(self.p_tok, keep)
        return plan

    @staticmethod
    def _prefix_len(raw: bytes, m: int) -> int:
        pos = 0
        for _ in range(m):
            pos = raw.index(b"\n", pos) + 1
        return pos

    @staticmethod
    def _truncate(path: Path, size: int) -> None:
        with open(path, "rb+") as fh:
            fh.truncate(size)
            fh.flush()
            os.fsync(fh.fileno())

    def _write_start(self, plan: StartPlan) -> None:
        row = {"event": "start", "tag": self.tag, "seed_index": self.seed_index, "attempt_no": plan.attempt_no,
               "start_stamp_sha256": self.stamp_sha, "reason": plan.reason, "started_at": self._tick(),
               "first_id": plan.first_id, "device": self.device}
        if plan.reason == REASON_TORN:
            row["truncated_ids"] = list(plan.truncated_ids)
        line = _dumps(row) + b"\n"
        _write_all(self.p_att, line, "ab")
        self._start_line = line
        self.plan = plan
        self.state = "STARTED"
        self._verify_start()

    def _verify_start(self) -> None:
        """방금 쓴 시작 줄과 도장을 다시 읽어 바이트로 맞댄다."""
        raw = self.p_att.read_bytes()
        last = raw[raw.rstrip(b"\n").rfind(b"\n") + 1:]
        if last != self._start_line:
            raise ExportRefused("start_line_reread", "시도 기록의 마지막 줄이 방금 쓴 시작 줄과 다르다")
        if _sha(self.p_stamp.read_bytes()) != self.stamp_sha:
            raise ExportRefused("stamp_reread", "도장의 바이트가 바뀌었다")

    # ------------------------------------------------------------------ 줄
    def open_generations(self) -> None:
        """생성 파일을 연다 — `STARTED` 에서만, 되읽기를 지난 뒤에만. 시도 1 이면 배타 생성, 아니면 덧붙이기."""
        self._need("STARTED")
        self._verify_start()
        if self.plan.first_id is None:
            raise ExportStateError("쓸 줄이 없다 — 곧바로 end() 로 간다")
        mode = "xb" if self.plan.attempt_no == 1 else "ab"
        # 핸들은 줄마다의 write 에 걸쳐 열려 있어야 한다 — end() 가 닫는다.
        self._gen_fh = open(self.p_gen, mode)  # noqa: SIM115
        if self.with_tokens:
            self._tok_fh = open(self.p_tok, mode)  # noqa: SIM115
        self.state = "WRITING"

    def next_id(self) -> str | None:
        i = self.plan.m + self.n_written
        return self.ids[i] if i < len(self.ids) else None

    def write(self, row: Mapping[str, Any], tokens: Mapping[str, Any] | None = None) -> None:
        """줄 하나. 토큰 줄(켰으면)을 먼저, **생성 줄을 마지막에** 쓴다 — 생성 줄이 그 id 의 완료 표지다."""
        self._need("WRITING")
        want = self.next_id()
        if want is None or row.get("image_id") != want:
            raise ExportStateError(f"목록 순서가 아니다: {row.get('image_id')!r} ≠ {want!r}")
        if self.with_tokens and (tokens is None or tokens.get("image_id") != want):
            raise ExportStateError("토큰 줄이 없거나 id 가 다르다")
        self._tick()
        if self.with_tokens:
            self._tok_fh.write(json.dumps(dict(tokens), ensure_ascii=False).encode("utf-8") + b"\n")
            self._tok_fh.flush()
            os.fsync(self._tok_fh.fileno())
        self._gen_fh.write(json.dumps(dict(row), ensure_ascii=False).encode("utf-8") + b"\n")
        self._gen_fh.flush()
        os.fsync(self._gen_fh.fileno())
        self.n_written += 1
        self.last_id = want

    def write_half(self, row: Mapping[str, Any], tokens: Mapping[str, Any] | None = None) -> None:
        """고의 중단 `export:torn_line` 의 효과 — 토큰 줄(켰으면)은 다 쓰고, 생성 줄은 **바이트의 앞 절반만** 쓴 뒤 flush · fsync 한다.
        개행이 없다. 이 뒤에는 쓰지 않는다(호출부가 곧 죽는다)."""
        self._need("WRITING")
        want = self.next_id()
        if want is None or row.get("image_id") != want:
            raise ExportStateError(f"목록 순서가 아니다: {row.get('image_id')!r} ≠ {want!r}")
        if self.with_tokens:
            self._tok_fh.write(json.dumps(dict(tokens), ensure_ascii=False).encode("utf-8") + b"\n")
            self._tok_fh.flush()
            os.fsync(self._tok_fh.fileno())
        line = json.dumps(dict(row), ensure_ascii=False).encode("utf-8") + b"\n"
        self._gen_fh.write(line[: len(line) // 2])
        self._gen_fh.flush()
        os.fsync(self._gen_fh.fileno())
        self.state = "TORN"

    def end(self) -> None:
        """끝 줄. 이 시도가 쓴 마지막 id 와 줄 수. 0 줄이면 `last_id=null` 이다. 덧붙이기 전에 시도 기록을 다시 검사한다."""
        self._need("STARTED", "WRITING")
        for fh in (self._gen_fh, self._tok_fh):
            if fh is not None:
                fh.close()
        check_attempts_file(self.p_att)
        row = {"event": "end", "tag": self.tag, "seed_index": self.seed_index, "attempt_no": self.plan.attempt_no,
               "start_stamp_sha256": self.stamp_sha, "last_id": self.last_id, "finished_at": self._tick(),
               "n_written": self.n_written}
        _write_all(self.p_att, _dumps(row) + b"\n", "ab")
        self.state = "ENDED"

    # ------------------------------------------------------------------ 곁 파일
    def seal(self, values: Mapping[str, Any], measured: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """곁 파일. 파일 바이트를 **다시 읽어** 낸 값 · 시도 기록에서 옮긴 값 · 호출부가 잰 값(`measured`)을 넣는다.

        `values` 는 등록값 따위 옮겨 싣는 값이다. 셋 가운데 둘이 같은 키에 다른 값을 내면 쓰지 않는다. 목록을 다 쓴 묶음만 닫는다.
        """
        self._need("ENDED")
        m, _, tail = _complete(self.p_gen)
        if tail or m != len(self.ids):
            raise ExportRefused("incomplete", f"목록 {len(self.ids)} 가운데 {m} 줄만 있다 — 곁 파일을 쓰지 않는다")
        gen_raw = self.p_gen.read_bytes() if self.p_gen.exists() else b""
        derived: dict[str, Any] = {
            "tag": self.tag, "seed_index": self.seed_index, "mode": self.mode, "n_lines": m,
            "generations_sha256": _sha(gen_raw), "start_stamp_sha256": self.stamp_sha,
            "eval_list_file_sha256": self.fields["eval_list_file_sha256"],
            "eval_list_set_sha256": self.fields["eval_list_set_sha256"],
            "export_commit": self.fields["export_commit"], "parser_sha256": self.fields["parser_sha256"],
            "generation_sha256": self.fields["generation_sha256"], "purpose": self.fields["purpose"],
            "aborted_stamps": len(list(self.folder.glob(f"{self.p_stamp.name}.aborted-*"))),
        }
        if self.with_tokens:
            self._check_tokens(gen_raw)
            derived["tokens_sha256"] = _sha(self.p_tok.read_bytes())
        rows = self._rows_of(check_attempts_file(self.p_att), self.stamp_sha)
        starts = sorted((r for r in rows if r["event"] == "start"), key=lambda r: r["attempt_no"])
        ends = sorted((r for r in rows if r["event"] == "end"), key=lambda r: r["attempt_no"])
        derived.update({"attempts": len(starts), "started_at": starts[0]["started_at"],
                        "finished_at": ends[-1]["finished_at"], "device": starts[-1]["device"]})
        if self.mode == "model":
            for k in ECHO_ABSENT_STAMP:
                derived[k] = self.fields[k]
        extra = dict(measured or {})
        clash = sorted({k for k in derived if k in values and values[k] != derived[k]}
                       | {k for k in extra if k in derived and extra[k] != derived[k]}
                       | {k for k in extra if k in values and values[k] != extra[k]})
        if clash:
            raise ExportRefused("sidecar_clash", f"곁 파일의 같은 키에 다른 값이 둘 이상 왔다: {clash}")
        meta = {**dict(values), **extra, **derived}
        if self.mode == "echo":
            present = [k for k in ECHO_ABSENT_SIDECAR if k in meta]
            if present:
                raise ExportRefused("echo_adapter_present", f"에코 곁 파일에는 이 키가 없어야 한다: {present}")
        _tmp_then_rename(self.p_side, _dumps(meta))
        self.state = "SEALED"
        return meta

    def _check_tokens(self, gen_raw: bytes) -> None:
        """곁 파일 전의 자기 대조 — 줄 수 같음 · id 수열 같음 · 줄마다 `len(token_ids) == n_new_tokens`."""
        gl = [json.loads(x) for x in gen_raw.split(b"\n") if x]
        tl = [json.loads(x) for x in self.p_tok.read_bytes().split(b"\n") if x]
        if len(gl) != len(tl) or [g["image_id"] for g in gl] != [t["image_id"] for t in tl]:
            raise ExportRefused("tokens_mismatch", "토큰 파일의 줄 수 · id 수열이 생성 파일과 다르다")
        bad = [g["image_id"] for g, t in zip(gl, tl) if len(t.get("token_ids") or []) != g.get("n_new_tokens")]
        if bad:
            raise ExportRefused("tokens_mismatch", f"토큰 수가 n_new_tokens 와 다른 줄이 있다: {len(bad)}")
