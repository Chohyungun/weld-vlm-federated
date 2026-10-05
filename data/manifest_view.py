"""스냅샷의 매니페스트와 주석을 **평가 정답을 올리지 않고** 읽는 판독기.

아래 둘이 일반 경로이고, 본채점 · 본실험 export 의 엄격 경로 둘(`read_annotations_view_eval` ·
`read_image_paths_eval`)은 영수증 조건을 지난 호출에만 평가 행의 주석 · 이미지 경로를 낸다(이 모듈 아래쪽).

계약 #2 는 매니페스트·주석 읽기를 검증된 스냅샷 로더(`data.manifest_io.load_snapshot`)로만 하게 한다.
그 로더는 두 파일을 통째로 올린다 — 평가 행의 `has_defect` · `n_defects` · `defect_types` · `iso_codes` ·
`src_labels_raw` 와 평가 이미지의 주석까지 객체가 된다. 리허설·진단·본채점처럼 **평가 정답에 닿으면 안 되는**
쪽에는 그 로더를 쓸 수 없다. 이 모듈의 함수들이 계약 #2 의 예외다(정본 목록은 구현 설계 §1-3).

- `read_manifest_columns` — 매니페스트에서 고른 열만. **평가 행에서는 `image_id` · `split` · `sha256` ·
  `group_id` 넷만** 읽을 수 있다. 층(`strata_key`)은 `{재질}|{묶음 대표 클래스}` 라 정답의 요약이므로
  train·val 행에서만 읽는다.
- `read_annotations_view` — 주석 파일에서 준 id 집합의 행만. 집합에 평가 id 가 하나라도 있으면 주석 파일을
  열기 전에 거부한다.

둘 다 **`verify_snapshot` 을 먼저 부른다.** 검증이 실패하면 파일을 한 줄도 읽지 않는다.
요구가 규칙을 어기면(평가 행의 금지 열, 모르는 열) 검증 전에, 아무 파일도 열지 않고 거부한다.

**파일의 바이트는 모두 지나간다.** 검증은 파일 전체를 해시하고 CSV 는 한 줄씩 칸으로 나뉜다. 이 모듈이 하지
않는 것은 평가 행의 금지 칸과 준 집합 밖의 주석 행을 **값으로 만드는 것**(모아 두기 · 자료형 변환)이다.
자료형 규칙은 로더와 같은 상수를 쓴다 — 같은 칸이 두 경로에서 다른 값이 되지 않게.

흡수 곁파일은 열지 않는다(검증 단계의 해시 계산은 `verify_snapshot` 의 일이다). 스냅샷에 쓰지 않는다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

# 자료형 규칙은 로더의 **비공개 상수**에서 가져온다 — 두 경로가 같은 칸을 같은 값으로 만들게 하는 대가로
# 로더 모듈과 결합한다. 그 이름이 바뀌면 이 모듈은 import 에서 멈춘다(조용히 다른 규칙으로 읽지 않는다).
from data.manifest_io import (_ANN_BOOL, _ANN_FLOAT, _ANN_INT, _MANIFEST_BOOL, _MANIFEST_FLOAT,
                              _MANIFEST_INT, ANNOTATION_COLUMNS, ANNOTATIONS_FILENAME,
                              MANIFEST_COLUMNS, MANIFEST_FILENAME, NULLABLE_STR_COLUMNS,
                              ManifestError, SnapshotVerificationError, verify_snapshot)

#: 평가 행에서 읽을 수 있는 열. 식별자 · 분할 · 화소 지문 · 묶음이다. 이 밖의 열은 평가 행에 대해 요구할 수 없다.
EVAL_READABLE: frozenset[str] = frozenset({"image_id", "split", "sha256", "group_id"})
SPLITS: frozenset[str] = frozenset({"train", "val", "eval"})
#: 한 칸의 글자 상한. 파이썬 `csv` 의 기본 131,072 자는 긴 `polygon_json` 한 칸에서 멈춘다.
#: 같은 파일을 읽는 앞선 검산 스크립트와 같은 값이다. 전역 상태라 읽는 동안만 올리고 되돌린다.
FIELD_SIZE_LIMIT = 10_000_000

_INTS = frozenset(_MANIFEST_INT) | frozenset(_ANN_INT)
_FLOATS = frozenset(_MANIFEST_FLOAT) | frozenset(_ANN_FLOAT)
_BOOLS = frozenset(_MANIFEST_BOOL) | frozenset(_ANN_BOOL)


class ManifestViewError(ManifestError):
    """판독기가 요구를 거부했다. 거부 사유가 요구 자체에 있으면 아무 파일도 열지 않았다."""


@contextmanager
def _field_limit():
    old = csv.field_size_limit(FIELD_SIZE_LIMIT)
    try:
        yield
    finally:
        csv.field_size_limit(old)


def _open_csv(path: Path):
    """이 모듈이 여는 파일은 전부 여기를 지난다 — 시험이 무엇을 열었는지 본다."""
    return path.open(encoding="utf-8", newline="")


def _convert(col: str, values: list[str]) -> pd.Series:
    """로더(`manifest_io._read_csv`)와 같은 규칙으로 한 열을 값으로 만든다. **모은 칸만** 여기로 온다."""
    s = pd.Series(values, dtype="string")
    if col in _INTS:
        return pd.to_numeric(s.replace("", pd.NA), errors="raise").astype("Int64")
    if col in _FLOATS:
        return pd.to_numeric(s.replace("", pd.NA), errors="raise").astype("Float64")
    if col in _BOOLS:
        mapped = s.map({"True": True, "False": False, "true": True, "false": False})
        if mapped.isna().any():
            raise ManifestError(f"{col} 에 bool 로 읽을 수 없는 값이 있다")
        return mapped.astype("boolean")
    if col in NULLABLE_STR_COLUMNS:
        return s.replace("", pd.NA)
    return s


def _frame(columns: Sequence[str], cells: dict[str, list[str]], digest: str) -> pd.DataFrame:
    df = pd.DataFrame({c: _convert(c, cells[c]) for c in columns}, columns=list(columns))
    df.attrs["snapshot_digest"] = digest
    return df


def _header(reader, want: tuple[str, ...], path: Path) -> None:
    head = next(reader, None)
    if head is None or tuple(head) != want:
        raise ManifestViewError(f"{path}: 머리행이 계약의 열과 다르다 — 읽지 않는다")


def _check_columns(columns: Sequence[str]) -> list[str]:
    if isinstance(columns, str) or not isinstance(columns, Sequence) or not columns:
        raise ManifestViewError("열은 비어 있지 않은 이름의 목록으로 준다")
    cols = list(columns)
    unknown = [c for c in cols if c not in MANIFEST_COLUMNS]
    if unknown:
        raise ManifestViewError(f"매니페스트에 없는 열: {unknown}")
    if len(set(cols)) != len(cols):
        raise ManifestViewError("같은 열을 두 번 요구했다")
    return cols


def _check_splits(split_filter: Iterable[str]) -> frozenset[str]:
    if isinstance(split_filter, str):
        raise ManifestViewError("split_filter 는 분할 이름의 집합으로 준다 — 문자열 하나가 아니다")
    splits = frozenset(split_filter)
    if not splits or not splits <= SPLITS:
        raise ManifestViewError(f"분할은 {sorted(SPLITS)} 가운데서 하나 이상 고른다: {sorted(splits)}")
    return splits


def _scan_manifest(root: Path, cols: list[str], splits: frozenset[str]) -> dict[str, list[str]]:
    """검증을 마친 뒤에만 부른다. 고른 분할의 행에서 고른 칸만 모은다."""
    path = root / MANIFEST_FILENAME
    idx = {c: MANIFEST_COLUMNS.index(c) for c in cols}
    at_split = MANIFEST_COLUMNS.index("split")
    width = len(MANIFEST_COLUMNS)
    cells: dict[str, list[str]] = {c: [] for c in cols}
    with _field_limit(), _open_csv(path) as fh:
        reader = csv.reader(fh)
        _header(reader, MANIFEST_COLUMNS, path)
        for n, row in enumerate(reader, 2):
            if len(row) != width:
                raise ManifestViewError(f"{path}:{n} 칸 수가 계약과 다르다")
            split = row[at_split]
            if split not in SPLITS:
                raise ManifestViewError(f"{path}:{n} 모르는 분할 값")
            if split not in splits:
                continue
            for c, i in idx.items():
                cells[c].append(row[i])
    return cells


def read_manifest_columns(root: Path | str, columns: Sequence[str], *,
                          split_filter: Iterable[str]) -> pd.DataFrame:
    """매니페스트에서 `split_filter` 의 행과 `columns` 의 열만 읽는다.

    - `split_filter` 에 `"eval"` 이 있으면 `columns` 는 `EVAL_READABLE` 안이어야 한다. 아니면 **검증 전에,
      아무 파일도 열지 않고** 거부한다.
    - 그다음 `verify_snapshot` 이 돈다. 실패하면 매니페스트를 열지 않는다.
    - 머리행이 계약의 열(`MANIFEST_COLUMNS`)과 같아야 읽는다. 자료형은 로더와 같다.

    반환 프레임의 `attrs["snapshot_digest"]` 에 검증한 지문을 싣는다 — 부르는 쪽이 승인된 지문과 **반환 직후,
    연산 전에** 맞댄다. `attrs` 는 pandas 연산을 지나며 이어진다는 보장이 없다.
    """
    cols = _check_columns(columns)
    splits = _check_splits(split_filter)
    if "eval" in splits:
        banned = [c for c in cols if c not in EVAL_READABLE]
        if banned:
            raise ManifestViewError(
                f"평가 행에서 읽을 수 없는 열이다: {banned} — 평가 행은 {sorted(EVAL_READABLE)} 만 읽는다")
    root = Path(root)
    digest = verify_snapshot(root)
    return _frame(cols, _scan_manifest(root, cols, splits), digest)


def _scan_annotations(root: Path, want: set[str]) -> dict[str, list[str]]:
    """검증과 id 검사를 마친 뒤에만 부른다. `want` 의 이미지에 붙은 행만 모은다."""
    path = root / ANNOTATIONS_FILENAME
    at_image = ANNOTATION_COLUMNS.index("image_id")
    width = len(ANNOTATION_COLUMNS)
    cells: dict[str, list[str]] = {c: [] for c in ANNOTATION_COLUMNS}
    with _field_limit(), _open_csv(path) as fh:
        reader = csv.reader(fh)
        _header(reader, ANNOTATION_COLUMNS, path)
        for n, row in enumerate(reader, 2):
            if len(row) != width:
                raise ManifestViewError(f"{path}:{n} 칸 수가 계약과 다르다")
            if row[at_image] not in want:
                continue
            for c, v in zip(ANNOTATION_COLUMNS, row):
                cells[c].append(v)
    return cells


def read_annotations_view(root: Path | str, ids: Iterable[str]) -> pd.DataFrame:
    """주석 파일을 한 줄씩 흘려 읽어 `ids` 의 이미지에 붙은 행만 남긴다.

    - `verify_snapshot` 이 먼저 돈다.
    - 매니페스트의 `image_id` · `split` 으로 평가 id 를 가린다. `ids` 에 평가 id 가 하나라도 있거나 매니페스트에
      없는 id 가 있으면 **주석 파일을 열기 전에** 거부한다.
    - 준 집합 밖의 행은 값으로 만들지 않는다. 열은 계약의 주석 열 전부이고 자료형은 로더와 같다.
    """
    if isinstance(ids, str):
        raise ManifestViewError("ids 는 image_id 의 집합으로 준다 — 문자열 하나가 아니다")
    want = set(ids)
    if not all(isinstance(i, str) for i in want):
        raise ManifestViewError("ids 는 문자열 image_id 만 받는다")
    root = Path(root)
    digest = verify_snapshot(root)

    meta = _scan_manifest(root, ["image_id", "split"], SPLITS)
    split_of = dict(zip(meta["image_id"], meta["split"]))
    unknown = sorted(want - split_of.keys())
    if unknown:
        raise ManifestViewError(f"매니페스트에 없는 id {len(unknown)}개 — 읽지 않는다")
    n_eval = sum(1 for i in want if split_of[i] == "eval")
    if n_eval:
        raise ManifestViewError(f"평가 id {n_eval}개가 들어 있다 — 평가 이미지의 주석은 읽지 않는다")

    return _frame(ANNOTATION_COLUMNS, _scan_annotations(root, want), digest)


# --------------------------------------------------------------------------------------
# 엄격 경로의 평가 정답 뷰 — 본채점만 부른다
# --------------------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[1]
#: 엄격 경로 평가 정답 뷰의 호출 기록. 거부든 허용이든 호출마다 한 줄이다.
CALL_LOG: Path = _REPO / "outputs" / "eval_truth_view" / "calls.jsonl"
#: 승인된 동결본의 닻이 적힌 설정의 저장소 상대 경로. **영수증 커밋에서** `git show` 로 읽는다 — 작업 트리의 설정은 읽지 않는다.
_BASE_CONFIG_PATH = "configs/base.yaml"
#: 영수증 커밋의 조상 관계와 등록 바이트를 묻는 저장소. 워크트리는 ref 를 본체와 같이 쓴다.
_GIT_ROOT: Path = _REPO


class EvalViewRefused(ManifestViewError):
    """엄격 경로의 조건을 지나지 못했다. 주석 파일을 열지 않았다."""


def _read_small(path: Path) -> bytes:
    """영수증 같은 작은 파일을 바이트로 읽는다. 이 모듈이 여는 파일은 `_open_csv` · 여기 · 호출 기록뿐이다."""
    with path.open("rb") as fh:
        return fh.read()


def _append_call_log(entry: dict) -> None:
    CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
    with CALL_LOG.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=_GIT_ROOT, capture_output=True, check=False)


def _anchor(commit: str) -> str:
    """영수증 커밋의 설정에서 닻을 낸다. 체크아웃의 한 줄을 고쳐 다른 지문을 지나게 하지 못하게 한다."""
    shown = _git("show", f"{commit}:{_BASE_CONFIG_PATH}")
    if shown.returncode != 0:
        raise EvalViewRefused(f"영수증의 커밋에서 설정({_BASE_CONFIG_PATH})을 읽지 못했다")
    try:
        cfg = yaml.safe_load(shown.stdout.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EvalViewRefused("영수증 커밋의 설정을 읽지 못했다") from exc
    fixed = cfg.get("fixed_before_main_runs") if isinstance(cfg, dict) else None
    digest = fixed.get("snapshot_digest") if isinstance(fixed, dict) else None
    if not isinstance(digest, str) or len(digest) != 64:
        raise EvalViewRefused("영수증 커밋의 설정에 승인된 동결본의 닻(fixed_before_main_runs.snapshot_digest)이 없다")
    return digest


def _receipt_data(raw: bytes) -> dict:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EvalViewRefused(f"영수증을 읽지 못했다 — {exc}") from exc
    if not isinstance(data, dict):
        raise EvalViewRefused("영수증을 읽지 못했다 — JSON 객체가 아니다")
    return data


def _receipt(data: dict):
    """평가 쪽의 `Receipt` 로 만든다 — 꼴 검사는 그 클래스가 한다. `read_receipt` 와 같은 키를 같은 방식으로 넘긴다."""
    from evaluation.prereg_unified import RECEIPT_KEYS, Receipt, ReceiptMissing
    try:
        return Receipt(**{k: data[k] for k in RECEIPT_KEYS}, device=data.get("device", ""))
    except KeyError as exc:
        raise EvalViewRefused(f"영수증을 읽지 못했다 — {exc.args[0]} 이 없다") from None
    except (ReceiptMissing, TypeError, ValueError) as exc:
        raise EvalViewRefused(f"영수증을 읽지 못했다 — {exc}") from exc


def _check_eval_call(root: Path, ids: Iterable[str], receipt, entry: dict) -> tuple[set[str], str]:
    """조건을 순서대로 본다. 앞이 어긋나면 뒤를 보지 않는다. 지난 만큼 `entry` 를 채운다."""
    from evaluation.eval_list import set_digest
    from evaluation.prereg_unified import (ReceiptMissing, RegistrationFileError, RegistrationIncomplete,
                                           RegistrationInvalid, registration_for_receipt, require_receipt)
    if not isinstance(receipt, (str, Path)):
        raise EvalViewRefused("영수증은 파일 경로로 준다 — 읽은 객체는 받지 않는다(파일 바이트로 기록한다)")
    if isinstance(ids, str):
        raise EvalViewRefused("ids 는 image_id 의 집합으로 준다 — 문자열 하나가 아니다")
    want = set(ids)
    if not all(isinstance(i, str) for i in want):
        raise EvalViewRefused("ids 는 문자열 image_id 만 받는다")
    entry.update(ids_set_sha256=set_digest(want), n_ids=len(want))
    if not want:
        raise EvalViewRefused("ids 가 비었다 — 읽을 것이 없으면 주석 파일을 열지 않는다")

    # 1. 영수증 — 한 번 읽은 바이트로 해시와 파싱을 함께 한다. 종류가 main
    try:
        raw = _read_small(Path(receipt))
    except OSError as exc:
        raise EvalViewRefused("영수증 파일을 읽지 못했다") from exc
    entry["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    data = _receipt_data(raw)
    kind = data.get("kind")
    entry["receipt_kind"] = kind if isinstance(kind, str) else None
    # 종류는 이 함수가 먼저 본다 — 평가 쪽이 영수증 종류를 늘려도 main 밖은 여기서 멈춘다
    if kind != "main":
        raise EvalViewRefused(f"영수증의 종류가 {kind!r} 다 — 평가 정답은 main 영수증으로만 읽는다")
    rec = _receipt(data)
    entry["main_commit"] = rec.main_commit

    # 2. 커밋이 본줄기의 조상
    anc = _git("merge-base", "--is-ancestor", rec.main_commit, "refs/heads/main")
    if anc.returncode == 1:
        raise EvalViewRefused("영수증의 커밋이 refs/heads/main 의 조상이 아니다")
    if anc.returncode != 0:
        raise EvalViewRefused("영수증의 커밋과 refs/heads/main 의 관계를 확인하지 못했다")

    # 3. 등록 — 영수증 커밋의 바이트로. 생성 쪽과 채점 쪽 모두
    shown = _git("show", f"{rec.main_commit}:{rec.registration_path}")
    if shown.returncode != 0:
        raise EvalViewRefused("영수증의 커밋에서 등록 파일을 읽지 못했다")
    try:
        reg, _ = registration_for_receipt(shown.stdout, rec, side="generation")
        require_receipt(reg, rec, side="scoring")
    except (ReceiptMissing, RegistrationFileError, RegistrationIncomplete, RegistrationInvalid) as exc:
        raise EvalViewRefused(f"등록이 영수증과 맞지 않거나 완결되지 않았다 — {type(exc).__name__}") from exc

    # 4. 목적과 분할 — 명시로 본다(목적이 비면 영수증 대조가 건너뛴다)
    if reg.generation.purpose != "main":
        raise EvalViewRefused(f"등록의 목적이 {reg.generation.purpose!r} 다 — main 이어야 한다")
    if reg.generation.list_split != "eval":
        raise EvalViewRefused(f"등록의 기준 분할이 {reg.generation.list_split!r} 다 — eval 이어야 한다")

    # 5. 닻 — 영수증 커밋의 설정에서. 작업 트리의 설정은 읽지 않는다
    anchor = _anchor(rec.main_commit)
    if reg.generation.snapshot_digest != anchor:
        raise EvalViewRefused("등록의 동결본 지문이 영수증 커밋의 닻과 다르다")

    # 6. 검증 — 그 지문도 닻과 같아야 한다
    digest = verify_snapshot(root)
    entry["snapshot_digest"] = digest
    if digest != anchor:
        raise EvalViewRefused("검증한 스냅샷의 지문이 영수증 커밋의 닻과 다르다")

    # 7. 평가 id 만
    meta = _scan_manifest(root, ["image_id", "split"], SPLITS)
    split_of = dict(zip(meta["image_id"], meta["split"]))
    unknown = want - split_of.keys()
    if unknown:
        raise EvalViewRefused(f"매니페스트에 없는 id {len(unknown)}개 — 읽지 않는다")
    n_other = sum(1 for i in want if split_of[i] != "eval")
    if n_other:
        raise EvalViewRefused(f"평가가 아닌 id {n_other}개가 들어 있다 — 평가 id 만 받는다")
    return want, digest


def read_annotations_view_eval(root: Path | str, ids: Iterable[str], *, receipt: Path | str) -> pd.DataFrame:
    """엄격 경로(본채점)의 평가 정답 뷰. `ids` 의 **평가** 이미지에 붙은 주석 행만 돌려준다.

    부르는 조건은 코드로 건다 — `ids` 가 비어 있지 않고, 영수증의 종류가 `main` 이고, 그 커밋이 `refs/heads/main` 의
    조상이고, 그 커밋의 등록이 생성·채점 두 쪽 모두 영수증과 맞고 목적이 `main`·기준 분할이 `eval` 이고, 등록의
    동결본 지문과 검증한 지문이 **영수증 커밋의** `configs/base.yaml` 닻과 같고(작업 트리의 설정은 읽지 않는다),
    `ids` 가 전부 평가 id 일 때만 주석 파일을 연다. 하나라도 어기면 **읽기 전에** `EvalViewRefused`(검증 실패는
    `SnapshotVerificationError`). 영수증 파일은 한 번 읽고, 그 바이트로 해시를 기록하고 파싱한다.

    거부든 허용이든 호출마다 `CALL_LOG` 에 한 줄을 남긴다. 허용 호출은 **기록을 쓴 뒤에** 주석 파일을 연다 —
    기록을 쓰지 못하면 읽지 않는다. 반환 프레임의 `attrs["snapshot_digest"]` 는 검증한 지문이다.
    """
    frame = sys._getframe(1)
    caller = f"{frame.f_globals.get('__name__')}:{frame.f_code.co_name}"
    del frame
    return _run_strict(Path(root), ids, receipt, "annotations", caller,
                       lambda root, want, digest: _frame(ANNOTATION_COLUMNS, _scan_annotations(root, want), digest))


#: 엄격 경로의 이미지 경로 판독기가 평가 행에서 내는 열 — 키와 경로뿐이다.
EVAL_PATH_COLUMNS: tuple[str, ...] = ("image_id", "rel_path")


def read_image_paths_eval(root: Path | str, ids: Iterable[str], *, receipt: Path | str) -> pd.DataFrame:
    """엄격 경로(본실험 export)의 평가 이미지 경로. `ids` 의 **평가** 이미지에 대해 `image_id` · `rel_path` 만 낸다.

    부르는 조건은 `read_annotations_view_eval` 과 **같은 검사**(`_check_eval_call`)로 건다 — `ids` 가 비어 있지 않고,
    영수증의 종류가 `main` 이고, 그 커밋이 `refs/heads/main` 의 조상이고, 등록이 생성 · 채점 두 쪽 모두 영수증과
    맞고 목적이 `main` · 기준 분할이 `eval` 이고, 등록과 검증한 지문이 영수증 커밋의 닻과 같고, `ids` 가 전부 평가
    id 일 때만 경로 열을 연다. 평가 행에서 식별자 넷(`EVAL_READABLE`) 밖의 열은 여전히 `read_manifest_columns`
    가 막는다 — 이 함수가 여는 것은 경로 하나뿐이다.

    거부든 허용이든 호출마다 `CALL_LOG` 에 한 줄(`view = "image_paths"`)을 남기고, 허용 호출은 기록을 쓴 뒤 읽는다.
    반환 프레임의 `attrs["snapshot_digest"]` 는 검증한 지문이다.
    """
    frame = sys._getframe(1)
    caller = f"{frame.f_globals.get('__name__')}:{frame.f_code.co_name}"
    del frame

    def read(root: Path, want: set[str], digest: str) -> pd.DataFrame:
        meta = _scan_manifest(root, list(EVAL_PATH_COLUMNS), frozenset({"eval"}))
        keep = [n for n, iid in enumerate(meta["image_id"]) if iid in want]
        return _frame(EVAL_PATH_COLUMNS, {c: [meta[c][n] for n in keep] for c in EVAL_PATH_COLUMNS}, digest)
    return _run_strict(Path(root), ids, receipt, "image_paths", caller, read)


def _run_strict(root: Path, ids: Iterable[str], receipt, view: str, caller: str, read) -> pd.DataFrame:
    """엄격 경로 판독기 둘의 공통 순서 — 조건 검사 → 기록 → 읽기. 거부도 기록한다. 기록을 못 쓰면 읽지 않는다."""
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "view": view,
             "caller": caller, "pid": os.getpid(),
             "receipt_path": str(receipt) if isinstance(receipt, (str, Path)) else None,
             "receipt_sha256": None, "receipt_kind": None, "main_commit": None,
             "ids_set_sha256": None, "n_ids": None, "snapshot_digest": None, "outcome": None, "reason": ""}
    try:
        want, digest = _check_eval_call(root, ids, receipt, entry)
    except (EvalViewRefused, SnapshotVerificationError, ManifestViewError) as exc:
        entry.update(outcome="refused", reason=str(exc)[:300])
        try:
            _append_call_log(entry)
        except OSError:
            pass                        # 기록을 못 남겨도 거부는 그대로다
        raise
    entry["outcome"] = "allowed"
    try:
        _append_call_log(entry)
    except OSError as exc:
        raise EvalViewRefused("호출 기록을 쓰지 못했다 — 읽지 않는다") from exc
    return read(root, want, digest)
