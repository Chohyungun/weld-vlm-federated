"""판정 한 번 규칙의 열쇠 — 내용 해시(07번 §32-2).

지키는 것.

1. **제외 키만 다른 두 생성 파일은 같은 열쇠다** — 다시 내보내 지연이 달라져도, 좌표 코드의 주석이 바뀌어도 걸린다.
2. **모델 출력이나 좌표 결과가 한 글자라도 다르면 다른 열쇠다.**
3. **분류는 닫혀 있다** — 모르는 키 · 빠진 내용 키면 해시를 내지 않는다.
4. **기대 해시는 두 구현이 같은 값을 낸 뒤에 고정한다** — 모듈의 함수와 이 파일 안의 독립 계산을 맞댄다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.content_key import (
    CONTENT_KEYS,
    EXCLUDED_KEYS,
    OPTIONAL_CONTENT_KEYS,
    ContentKeyError,
    generations_content_sha256,
)


def line(image_id: str = "aihub71761:11", **over) -> dict:
    row = {"image_id": image_id, "image_sha256": "a" * 64,
           "text": '{"defects":[{"iso_code":"2011","bbox":[10,20,30,40]}]}',
           "bbox_px_parsed": {"defects": [{"iso_code": "2011", "bbox_px": [10.0, 20.0, 30.0, 40.0]}],
                              "verdict": None, "cited_clauses": []},
           "parse_error": None, "gen_stop": "eos", "n_new_tokens": 17, "n_bad_items_dropped": 0,
           "model_input_wh": [1280, 704],
           "latency_ms": 812.5, "raw_output_ref": "outputs/x/1.txt",
           "coord_space": "ABS_ORIG", "coord_cfg_hash": "c598d549"}
    row.update(over)
    return row


def raw(*rows: dict, sep: str = "\n") -> bytes:
    """줄의 키 순서와 공백은 쓰는 쪽마다 다를 수 있다 — 내용 해시는 그것에 흔들리지 않아야 한다."""
    return (sep.join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")


def independent(rows: list[dict]) -> str:
    """계약 §32-2 의 글에서 따로 짠 계산 — 모듈의 코드를 부르지 않는다."""
    keep = ["image_id", "image_sha256", "text", "bbox_px_parsed", "parse_error", "gen_stop",
            "n_new_tokens", "n_bad_items_dropped", "model_input_wh"]
    maybe = ["image_grid_thw"]      # 있으면 내용(10-04 개정)
    body = "".join(json.dumps({k: r[k] for k in keep + [x for x in maybe if x in r]}, sort_keys=True,
                              ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_분류가_계약의_표와_같다() -> None:
    assert set(CONTENT_KEYS) == {"image_id", "image_sha256", "text", "bbox_px_parsed", "parse_error",
                                 "gen_stop", "n_new_tokens", "n_bad_items_dropped", "model_input_wh"}
    assert set(EXCLUDED_KEYS) == {"latency_ms", "raw_output_ref", "coord_space", "coord_cfg_hash"}
    assert not set(CONTENT_KEYS) & set(EXCLUDED_KEYS)
    assert set(OPTIONAL_CONTENT_KEYS) == {"image_grid_thw"}
    assert not set(OPTIONAL_CONTENT_KEYS) & (set(CONTENT_KEYS) | set(EXCLUDED_KEYS))


@pytest.mark.parametrize(("key", "other"), [("latency_ms", 99.0), ("raw_output_ref", "elsewhere/1.txt"),
                                            ("coord_space", "ABS_RESIZED"), ("coord_cfg_hash", "0cc62b38")])
def test_제외_키만_다르면_같은_열쇠다(key: str, other) -> None:
    """다시 내보내 지연이 달라진 경우 — 원시 바이트 해시는 갈리지만 판정 한 번 규칙은 걸려야 한다."""
    a, b = raw(line(), line("aihub71761:12")), raw(line(), line("aihub71761:12", **{key: other}))
    assert hashlib.sha256(a).digest() != hashlib.sha256(b).digest()
    assert generations_content_sha256(a) == generations_content_sha256(b)


def test_키_순서와_공백이_달라도_같은_열쇠다() -> None:
    a = raw(line())
    b = (json.dumps(line(), ensure_ascii=False, sort_keys=True, indent=None, separators=(", ", ": ")) + "\n").encode()
    assert generations_content_sha256(a) == generations_content_sha256(b)


@pytest.mark.parametrize("change", [
    {"text": '{"defects":[{"iso_code":"2011","bbox":[10,20,30,41]}]}'},
    {"bbox_px_parsed": {"defects": [{"iso_code": "2011", "bbox_px": [10.0, 20.0, 30.0, 40.001]}],
                        "verdict": None, "cited_clauses": []}},
    {"image_sha256": "b" * 64},
    {"gen_stop": "length"},
    {"model_input_wh": [1280, 720]},
])
def test_모델_출력이나_좌표가_다르면_다른_열쇠다(change: dict) -> None:
    assert generations_content_sha256(raw(line())) != generations_content_sha256(raw(line(**change)))


def test_줄_순서가_다르면_다른_열쇠다() -> None:
    a, b = line("aihub71761:11"), line("aihub71761:12")
    assert generations_content_sha256(raw(a, b)) != generations_content_sha256(raw(b, a))


def test_분류되지_않은_키가_있으면_해시를_내지_않는다() -> None:
    with pytest.raises(ContentKeyError, match="분류되지 않은 키"):
        generations_content_sha256(raw(line(latency_us=812500)))


@pytest.mark.parametrize("key", ["text", "bbox_px_parsed", "model_input_wh"])
def test_내용_키가_빠지면_해시를_내지_않는다(key: str) -> None:
    row = line()
    del row[key]
    with pytest.raises(ContentKeyError, match="내용 키가 없다"):
        generations_content_sha256(raw(row))


def test_null_인_내용은_null_로_남는다() -> None:
    """파싱 실패 줄 — 짝 규칙대로 `bbox_px_parsed` 가 null 이다. 빠진 키와 다르다."""
    row = line(parse_error="no_json", bbox_px_parsed=None)
    assert generations_content_sha256(raw(row)) == independent([row])


@pytest.mark.parametrize("bad", [b"", b"\n", b'{"a":1}', b'{"a":1}\r\n', b"[1]\n", b"\xff\n"])
def test_꼴이_틀린_파일은_받지_않는다(bad: bytes) -> None:
    with pytest.raises(ContentKeyError):
        generations_content_sha256(bad)


def test_두_구현이_같은_값을_낸다() -> None:
    rows = [line("aihub71761:11"), line("aihub71761:12", parse_error="truncated", bbox_px_parsed=None,
                                         gen_stop="length", text="{\"defects\":[한글")]
    assert generations_content_sha256(raw(*rows)) == independent(rows)


def test_기지_입력의_기대_해시() -> None:
    """두 구현이 같은 값을 낸 뒤 고정한 값이다(위 시험). 정규화 규칙이 바뀌면 여기서 깨진다."""
    rows = [line("aihub71761:11"), line("aihub71761:12", parse_error="truncated", bbox_px_parsed=None,
                                         gen_stop="length", text="{\"defects\":[한글")]
    assert generations_content_sha256(raw(*rows)) == EXPECTED


EXPECTED = "6c4feffffa9ccf41f33e4b565fa59550b335333ae3353a9180c40163246cf975"


# ---------------------------------------------------------------- 관측 격자 `image_grid_thw` — 있으면 내용 (10-04)

def test_관측_격자가_있는_줄도_열쇠를_낸다() -> None:
    """쓰는 쪽은 줄마다 프로세서의 격자를 싣는다 — 분류에 없으면 모든 묶음이 거부된다(쓰는 쪽 보고 6절의 18)."""
    rows = [line(image_grid_thw=[1, 44, 80]), line("aihub71761:12", image_grid_thw=[1, 44, 80])]
    assert generations_content_sha256(raw(*rows)) == independent(rows)


def test_관측_격자는_내용이다_값이_다르면_열쇠가_다르고_없는_줄과도_다르다() -> None:
    """같은 이미지 · 같은 프로세서 설정이면 같은 값이라 판정 한 번의 열쇠에 든다. 실행마다 바뀌는 칸이 아니다."""
    a = generations_content_sha256(raw(line(image_grid_thw=[1, 44, 80])))
    assert a == generations_content_sha256(raw(line(image_grid_thw=[1, 44, 80], latency_ms=1.0)))
    assert a != generations_content_sha256(raw(line(image_grid_thw=[1, 68, 120])))
    assert a != generations_content_sha256(raw(line()))


def test_분류는_여전히_닫혀_있다() -> None:
    with pytest.raises(ContentKeyError, match="분류되지 않은 키"):
        generations_content_sha256(raw(line(image_grid_thw=[1, 44, 80], image_grid_thw_extra=1)))


def _classification_problems(row_keys: list[set[str]]) -> list[str]:
    """쓴 줄마다의 키 집합 → 분류와 어긋난 것. 줄마다 분류 밖 키가 없고 내용 키가 다 있어야 하며, "있으면 내용" 의 키는 쓴 줄 어딘가에 있어야 한다."""
    known = set(CONTENT_KEYS) | set(OPTIONAL_CONTENT_KEYS) | set(EXCLUDED_KEYS)
    out = []
    for i, ks in enumerate(row_keys, 1):
        if not ks <= known:
            out.append(f"{i}번째 줄의 분류 밖 키: {sorted(ks - known)}")
        if not set(CONTENT_KEYS) <= ks:
            out.append(f"{i}번째 줄에 없는 내용 키: {sorted(set(CONTENT_KEYS) - ks)}")
    seen = set().union(*row_keys) if row_keys else set()
    if not set(OPTIONAL_CONTENT_KEYS) <= seen:
        out.append(f"쓰는 쪽이 내지 않은 분류 키: {sorted(set(OPTIONAL_CONTENT_KEYS) - seen)}")
    return out


_TEXTS = (
    '{"defects": [{"iso_code": "2011", "bbox_2d": [10, 20, 40, 55]}], "verdict": "합격", "cited_clauses": []}',
    "산문만 있고 JSON 이 없다",
    ('```json\n{"defects": [{"iso_code": "2011", "bbox_2d": [5, 5, 5, 9]}, '
     '{"iso_code": "2011", "bbox_2d": [1.5, 2, 30, 40]}], "verdict": "불합격"}\n```'),
    '{"defects": [{"iso_code": "2011", "bbox_2d": [1, 2',
)
"""대역 생성기의 출력 넷 — 정상 · 해독 불가 · 깨진 항목이 섞인 펜스 · 잘림. `line_fields` 의 두 갈래를 다 지난다."""


def _written_row_keys(tmp, mode: str, monkeypatch) -> list[set[str]]:
    """쓰는 쪽 생성 경로의 **공개 진입점** `vlm.export_run.run_export` 를 대역 생성기로 실제로 한 번 돌려 쓴 줄의 키 집합을 읽는다.

    대역 적재기 · 생성기는 이 시험 안에 둔다 — 다른 트랙의 시험 파일을 가져오지 않는다. 진입점의 인자 계약은 판마다 다르다:
    사전 점검을 인자로 받는 판(`preflight` · `stamp_fields`)과 안에서 부르는 판(`receipt_path` · `measure_env`). 뒤의 판은
    사전 점검 함수(`vlm.export_preflight.export_preflight`)만 실행 중에 대역으로 바꾼다 — 등록 · 동결본 · 계획의 세계를 짓지 않는다.
    목록을 하나 더 두고 `stop_after_lines` 로 멈춰 **봉인 전에** 돌아온다(곁 파일 · 노출 원장을 쓰지 않는다). 생성 줄은 그 전에 다 쓰였다.
    """
    import inspect
    from types import SimpleNamespace

    from PIL import Image

    from evaluation.eval_list import canonical_bytes, validate_eval_list
    from vlm import export_run as ER

    ids = [f"aihub{n:06d}" for n in range(1, len(_TEXTS) + 2)]            # 마지막 하나는 쓰지 않는다 — 봉인 전 멈춤
    run_root = tmp / "reh-20261004T000000"
    lists = run_root / "lists"
    lists.mkdir(parents=True)
    list_file = lists / "gen.list"
    list_file.write_bytes(canonical_bytes(ids))
    paths = {}
    for i, iid in enumerate(ids):
        q = tmp / "img" / f"{iid}.png"
        q.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (1280, 720), i * 20).save(q)
        paths[iid] = q
    text_of = dict(zip(ids, _TEXTS + (_TEXTS[0],)))

    def gen(img, image_id):
        return ER.GenOut(text=text_of[image_id], gen_stop="eos", n_new_tokens=5, model_input_wh=(1280, 704),
                         grid_thw=(1, 44, 80), token_ids=[1, 2, 3, 4, 5], token_logprobs=[-0.1] * 5, latency_ms=1.0)

    def loader(*_a, **_k):
        return gen

    stamp = {"generation_sha256": "a" * 64, "export_commit": "c" * 40, "parser_sha256": ER.parser_sha256()}
    if mode == "model":
        stamp |= {"train_run_id": "uni_local_C1_s7_t1", "train_ledger_sha256": "b" * 64, "scored_adapter_sha256": "d" * 64}
    folder = run_root / ("export_echo" if mode == "echo" else "export")
    common = {"folder": folder, "tag": "uni_local_C1", "seed_index": 1, "mode": mode, "purpose": "rehearsal",
              "image_path_of": paths, "load_generator": loader, "device": "cpu:0", "run_root": run_root,
              "non_main_parent": tmp, "with_tokens": mode == "model", "stop_after_lines": len(_TEXTS),
              "standin_allowed": True, "env": {}}
    params = inspect.signature(ER.run_export).parameters
    if "preflight" in params:
        from vlm.coords import CoordCfg
        res = ER.run_export(**common, seed_value=7, eval_list_path=list_file, preflight=lambda **_k: None,
                            stamp_fields=stamp, sidecar_values={}, coord_cfg=CoordCfg(coord_space="ABS_ORIG"))
    else:
        import vlm.export_preflight as EP
        pre = SimpleNamespace(eval_list=validate_eval_list(canonical_bytes(ids)), seed_value=7,
                              gen_cfg=SimpleNamespace(coord_space="ABS_ORIG", mode=mode),
                              stamp_fields={**stamp, "purpose": "rehearsal"}, sidecar_values={}, stand_in=True)
        monkeypatch.setattr(EP, "export_preflight", lambda **_k: pre)
        receipt = run_root / "registration" / "receipt.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text("{}", encoding="utf-8")
        res = ER.run_export(**common, receipt_path=receipt, list_path=list_file, snapshot_root=tmp / "snap",
                            measure_env=lambda *_a, **_k: {})
    assert res.n_written == len(_TEXTS) and not res.sealed, "봉인 전에 멈췄어야 한다"
    [gp] = list(folder.glob("*.generations.jsonl"))
    return [set(json.loads(s)) for s in gp.read_text(encoding="utf-8").splitlines() if s.strip()]


@pytest.mark.parametrize("mode", ["model", "echo"])
def test_쓰는_쪽이_실제로_쓴_줄의_키가_분류와_맞다(tmp_path, monkeypatch, mode: str) -> None:
    """쓰는 쪽이 줄에 키를 더하면 이 시험이 먼저 떨어진다 — 본채점이 묶음을 `content_key` 로 거부하기 전에.

    레코드 실패가 섞인 줄(해독 불가 · 절단)까지 넣어 `line_fields` 의 두 갈래를 다 지나게 한다.
    """
    rows = _written_row_keys(tmp_path, mode, monkeypatch)
    assert len(rows) == len(_TEXTS)
    assert _classification_problems(rows) == []


def test_다른_트랙의_시험_파일을_가져오지_않는다() -> None:
    """공개 진입점만 부른다 — 다른 트랙의 시험 도우미가 옮겨지거나 이름이 바뀌어도 이 시험이 깨지지 않게."""
    import ast
    src = Path(__file__).read_text(encoding="utf-8")
    mods = {n.module or "" for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m == "tests" or m.startswith("tests.") for m in mods), sorted(mods)


@pytest.mark.parametrize(("extra", "why"), [
    ({"image_tokens"}, "분류 밖 키"),
    (set(), "없는 내용 키"),
])
def test_분류_대조가_떨어지는_꼴(extra: set[str], why: str) -> None:
    """대조 함수 자체의 단언 — 분류 밖 키가 하나 늘거나 내용 키가 하나 빠지면 떨어진다."""
    base = set(CONTENT_KEYS) | set(OPTIONAL_CONTENT_KEYS) | set(EXCLUDED_KEYS)
    rows = [base | extra] if extra else [base - {"text"}]
    assert any(why in m for m in _classification_problems(rows))


@pytest.mark.parametrize("where", ["writer", "line_fields"])
def test_쓰는_쪽이_어느_층에서_키를_더해도_떨어진다(tmp_path, monkeypatch, where: str) -> None:
    """쓰는 쪽 파일을 고치지 않고 — 실행 중에만 쓰는 함수 · `line_fields` 에 키를 하나 끼워 쓴 줄에 드러나는지 본다."""
    import vlm.export_writer as EW
    import vlm.gen_parse as GP
    if where == "writer":
        real = EW.ExportWriter.write

        def write(self, row, tokens=None):
            return real(self, {**row, "image_tokens": 3}, tokens)
        monkeypatch.setattr(EW.ExportWriter, "write", write)
    else:
        real_lf = GP.line_fields
        monkeypatch.setattr(GP, "line_fields", lambda *a, **k: {**real_lf(*a, **k), "image_tokens": 3})
    rows = _written_row_keys(tmp_path, "model", monkeypatch)
    assert any("분류 밖 키" in m and "image_tokens" in m for m in _classification_problems(rows))

