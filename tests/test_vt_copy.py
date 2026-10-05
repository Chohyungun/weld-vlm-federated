"""VT-1b · VT-1e — 원천 복사 · 자원 지킴 · 쓰기 가드. 실물을 읽지 않는다."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from data.vt.copy import (
    CopyMismatch,
    check_reference,
    copy_zip,
    drivefs_server_md5,
    hash_file,
    load_jsonl_tolerant,
    verify,
)
from data.vt.guard import Guard, GuardStop, WriteForbidden, assert_vt_writable, gate_lock_path


class _Clock:
    def __init__(self):
        self.slept = 0.0

    def sleep(self, s):
        self.slept += s


def _guard(c=20.0, mem=10.0, gate_path=None, sleep=None, log=None, c_fn=None, mem_fn=None):
    clk = _Clock()
    g = Guard(c_free_min_gb=10.0, mem_free_min_gb=6.0, c_wait_max_s=600, wait_step_s=30,
              gate_path=gate_path, c_free_fn=c_fn or (lambda: c), mem_free_fn=mem_fn or (lambda: mem),
              sleep_fn=sleep or clk.sleep, log=log)
    return g, clk


def _src(tmp_path: Path, n: int = 10_000) -> Path:
    p = tmp_path / "drive" / "TS_VTST_x.zip"
    p.parent.mkdir(parents=True)
    p.write_bytes(bytes((i * 7 + 3) % 251 for i in range(n)))
    return p


def test_구간마다_다시_열어_이어_쓴_사본이_한_번에_읽은_해시와_같다(tmp_path):
    src = _src(tmp_path)
    g, _ = _guard()
    calls = []
    orig = g.wait
    g.wait = lambda: (calls.append(1), orig())[1]
    r = copy_zip(src, tmp_path / "out" / src.name, chunk_bytes=1024, read_bytes=100, guard=g, log=lambda d: None)
    assert r["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert r["md5"] == hashlib.md5(src.read_bytes()).hexdigest()
    assert r["chunks"] == 10 and len(calls) == 10          # 구간마다 그 앞에서 지킴을 본다
    assert (tmp_path / "out" / src.name).read_bytes() == src.read_bytes()
    assert not (tmp_path / "out" / (src.name + ".part")).exists()


def test_끊긴_사본은_그_바이트를_다시_해시한_뒤_이어_쓴다(tmp_path):
    src = _src(tmp_path)
    dst = tmp_path / "out" / src.name
    dst.parent.mkdir()
    (dst.parent / (src.name + ".part")).write_bytes(src.read_bytes()[:3000])
    g, _ = _guard()
    logs = []
    r = copy_zip(src, dst, chunk_bytes=4096, read_bytes=512, guard=g, log=logs.append)
    assert r["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert logs[0] == {"resume": src.name, "offset": 3000}
    assert r["chunks"] == 2                                # 남은 7,000 바이트만


def test_다시_읽은_해시가_다르면_최종_이름으로_바꾸지_않고_멈춘다(tmp_path, monkeypatch):
    import data.vt.copy as cp
    src = _src(tmp_path)
    dst = tmp_path / "out" / src.name
    monkeypatch.setattr(cp, "hash_file", lambda path, rb: ("0" * 64, "0" * 32, path.stat().st_size))
    g, _ = _guard()
    with pytest.raises(CopyMismatch):
        cp.copy_zip(src, dst, chunk_bytes=4096, read_bytes=512, guard=g, log=lambda d: None)
    # 믿을 수 없는 사본은 비켜 둔다 — 다시 돌려도 그것을 이어 받지 않는다
    assert not dst.exists() and not (dst.parent / (src.name + ".part")).exists()
    assert (dst.parent / (src.name + ".part.bad0")).exists()


def test_원천보다_긴_이어_쓰기는_비켜_두고_처음부터_받는다(tmp_path):
    src = _src(tmp_path, 100)
    dst = tmp_path / "out" / src.name
    dst.parent.mkdir()
    (dst.parent / (src.name + ".part")).write_bytes(b"x" * 200)
    g, _ = _guard()
    r = copy_zip(src, dst, chunk_bytes=64, read_bytes=16, guard=g, log=lambda d: None)
    assert r["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert (dst.parent / (src.name + ".part.bad0")).read_bytes() == b"x" * 200


def test_끝난_사본은_다시_받지_않는다(tmp_path):
    src = _src(tmp_path, 100)
    dst = tmp_path / "out" / src.name
    dst.parent.mkdir()
    dst.write_bytes(b"done")
    g, _ = _guard()
    with pytest.raises(FileExistsError):
        copy_zip(src, dst, chunk_bytes=64, read_bytes=16, guard=g, log=lambda d: None)


def test_C_여유가_시한_안에_돌아오지_않으면_멈춘다():
    g, clk = _guard(c=9.0)
    with pytest.raises(GuardStop):
        g.wait()
    assert clk.slept >= 600


def test_C_여유가_돌아오면_잇는다():
    seq = iter([9.0, 9.0, 12.0])
    g, clk = _guard(c_fn=lambda: next(seq))
    assert g.wait()["c_free_gb"] == 12.0
    assert clk.slept == 60 and g.waits == 2


def test_게이트_락이_있으면_풀릴_때까지_기다리고_C_시한을_세지_않는다(tmp_path):
    """C: 가 문턱 아래인 채로 게이트가 열려 있다 — 게이트 동안의 기다림은 C: 시한(600 초)에 세지 않는다(06b I-1)."""
    lock = tmp_path / "CTO_GATE_OPEN"
    lock.write_text("x", encoding="utf-8")
    n = {"i": 0}

    def sleep(s):
        n["i"] += 1
        if n["i"] == 40:                                   # 20 분 기다린 뒤 게이트가 닫히고 C: 도 돌아온다
            lock.unlink()
    c = {"v": 9.0}
    g, _ = _guard(gate_path=lock, sleep=sleep, c_fn=lambda: c["v"] if lock.exists() else 12.0)
    g.wait()
    assert n["i"] == 40


def test_메모리가_모자라면_기다린다():
    seq = iter([4.0, 5.9, 6.5])
    g, clk = _guard(mem_fn=lambda: next(seq))
    assert g.wait()["mem_free_gb"] == 6.5
    assert clk.slept == 60


def test_워크트리의_게이트_락은_본체_git_폴더에_있다(tmp_path):
    main_git = tmp_path / "main" / ".git"
    (main_git / "worktrees" / "wt_A").mkdir(parents=True)
    wt = tmp_path / "wt_A"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {(main_git / 'worktrees' / 'wt_A').as_posix()}\n", encoding="utf-8")
    assert gate_lock_path(wt) == main_git / "CTO_GATE_OPEN"
    assert gate_lock_path(tmp_path / "main") == main_git / "CTO_GATE_OPEN"


def test_쓰기_가드는_동결_RT_자리와_허락된_뿌리_밖을_막는다(tmp_path):
    root = tmp_path
    staging = root / "data/interim/vt_build_r1"
    raw = root / "data/raw/aihub71761_vt/_zips"
    allowed = [staging, raw]
    assert_vt_writable(staging / "tiles", allowed_roots=allowed, repo_root=root)
    assert_vt_writable(raw, allowed_roots=allowed, repo_root=root)
    for bad in ["data/interim/manifest_v1/x", "data/interim/manifest_v3_rawlabels", "data/raw/aihub71761/_zips",
                "data/interim/aihub_labels", "data/interim/tiles_v1/VT", "data/processed/pairs_main_v1",
                "data/interim"]:
        with pytest.raises(WriteForbidden):
            assert_vt_writable(root / bad, allowed_roots=allowed + [root / "data"], repo_root=root)
    with pytest.raises(WriteForbidden):
        assert_vt_writable(root / "outputs/x", allowed_roots=allowed, repo_root=root)


def test_쓰기_가드는_계약서가_있는_디렉터리를_막는다(tmp_path):
    staging = tmp_path / "data/interim/vt_build_r1"
    (staging / "sealed").mkdir(parents=True)
    (staging / "sealed" / "SNAPSHOT.sha256").write_text("x", encoding="utf-8")
    with pytest.raises(WriteForbidden):
        assert_vt_writable(staging / "sealed" / "a", allowed_roots=[staging], repo_root=tmp_path)


def _db(tmp_path, rows):
    db = tmp_path / "metadata_sqlite_db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE items (local_title TEXT, trashed INTEGER, proto BLOB)")
    con.executemany("INSERT INTO items VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()
    return db


def test_드라이브_캐시의_서버_md5는_휴지통을_빼고_읽는다(tmp_path):
    md5 = hashlib.md5(b"a").hexdigest()
    # 실물 캐시처럼 제목은 TEXT 로 넣는다 — 바이트로 넣으면 바이트로 묶은 질의의 결함(늘 0 건)을 못 잡는다
    db = _db(tmp_path, [("A.zip", 0, b"\x01\x02" + md5.encode() + b"\x00"),
                        ("A.zip", 1, b"ffffffffffffffffffffffffffffffff"),
                        ("B.zip", 0, b"")])
    got = drivefs_server_md5(["A.zip", "B.zip", "C.zip"], db)
    assert got == {"A.zip": [md5], "B.zip": [], "C.zip": []}


def _rec(name, data):
    return {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "md5": hashlib.md5(data).hexdigest()}


def test_대조는_세_참조를_맞대고_서버_md5_대조군은_라벨_zip_이다():
    a, b, c = b"aaaa", b"bb", b"ccc"
    listing = [(4, "Training/01/TS_A.zip"), (2, "Training/02/TL_B.zip"), (3, "Validation/02/VL_C.zip")]
    recs = {"TS_A.zip": _rec("TS_A.zip", a), "TL_B.zip": _rec("TL_B.zip", b), "VL_C.zip": _rec("VL_C.zip", c)}
    lab = {"TL_B.zip": hashlib.sha256(b).hexdigest(), "VL_C.zip": hashlib.sha256(c).hexdigest()}
    md5 = {n: [r["md5"]] for n, r in recs.items()}
    r = verify(recs, listing, lab, md5)
    assert r["all_ok"] and r["server_md5_checked"] == 3 and r["server_md5_control"].startswith("라벨 zip")
    # 원천 zip 이 캐시에 없어도 라벨 대조군이 서면 증거로 센다 — 잡힌 것만 맞댄다
    r = verify(recs, listing, lab, {**md5, "TS_A.zip": []})
    assert r["all_ok"] and r["server_md5_checked"] == 2
    # 라벨 zip 하나가 캐시에 없으면 대조군이 서지 않는다 — md5 를 하나도 세지 않는다(통과로도 실패로도)
    r = verify(recs, listing, lab, {**md5, "VL_C.zip": []})
    assert r["all_ok"] and r["server_md5_checked"] == 0 and r["server_md5_control"].startswith("대조 불가")
    # 대조군이 섰는데 원천 하나가 다르면 실패
    assert not verify(recs, listing, lab, {**md5, "TS_A.zip": ["0" * 32]})["all_ok"]
    # 라벨 해시가 다르면 · 라벨 해시를 다 맞대지 못하면(이름이 다르면) · 크기가 다르면 · 빠지면 · 디스크와 다르면 실패
    assert not verify(recs, listing, {**lab, "TL_B.zip": "0" * 64}, md5)["all_ok"]
    assert not verify(recs, listing, {"TL_B.zip": lab["TL_B.zip"], "VL_Ｃ.zip": lab["VL_C.zip"]}, md5)["all_ok"]
    assert not verify({**recs, "TS_A.zip": {**recs["TS_A.zip"], "bytes": 5}}, listing, lab, md5)["all_ok"]
    assert not verify({k: v for k, v in recs.items() if k != "TS_A.zip"}, listing, lab, md5)["all_ok"]
    assert not verify(recs, listing, lab, md5, {"TS_A.zip": 4, "TL_B.zip": 2})["all_ok"]
    assert verify(recs, listing, lab, md5, {"TS_A.zip": 4, "TL_B.zip": 2, "VL_C.zip": 3})["all_ok"]


def test_이어_쓰던_사본이_원천의_접두와_다르면_비켜_두고_처음부터_받는다(tmp_path):
    src = _src(tmp_path)
    dst = tmp_path / "out" / src.name
    dst.parent.mkdir()
    bad = bytearray(src.read_bytes())
    bad[5000] ^= 0xFF                                  # 길이는 같고 한 바이트가 다른 사본(끊긴 꼬리 · 깨진 블록)
    (dst.parent / (src.name + ".part")).write_bytes(bytes(bad))
    g, _ = _guard()
    logs = []
    r = copy_zip(src, dst, chunk_bytes=4096, read_bytes=512, guard=g, log=logs.append)
    assert r["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert logs[0]["resume_rejected"] == src.name and (dst.parent / (src.name + ".part.bad0")).exists()


def test_참조_해시_앞자리는_8_자_이상():
    with pytest.raises(CopyMismatch):
        check_reference(Path(__file__), "")


def test_JSONL_은_마지막_줄만_잘렸으면_버리고_중간이_깨졌으면_멈춘다(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": ', encoding="utf-8")
    assert load_jsonl_tolerant(p) == [{"a": 1}, {"a": 2}]
    p.write_text('{"a": 1}\n{"a": \n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl_tolerant(p)


def test_C_를_보지_않는_지킴은_C_여유가_낮아도_기다리지_않는다():
    g = Guard(c_free_min_gb=10.0, mem_free_min_gb=6.0, c_wait_max_s=600, wait_step_s=30, gate_path=None,
              c_free_fn=lambda: 1.0, mem_free_fn=lambda: 10.0, sleep_fn=lambda s: None, check_c=False)
    assert g.wait()["c_free_gb"] == 1.0 and g.waits == 0


def test_hash_file_은_sha256_md5_크기를_낸다(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"hello" * 1000)
    assert hash_file(p, 7) == (hashlib.sha256(p.read_bytes()).hexdigest(),
                               hashlib.md5(p.read_bytes()).hexdigest(), 5000)


def test_기록은_적힌_해시와_맞을_때만_읽힌다(tmp_path):
    from data.vt.records import RecordHashMismatch, read_csv_checked, write_csv
    h = write_csv(tmp_path / "r.csv", ["a", "b"], [{"a": 1, "b": "x"}])
    assert read_csv_checked(tmp_path / "r.csv", h) == [{"a": "1", "b": "x"}]
    (tmp_path / "r.csv").write_bytes((tmp_path / "r.csv").read_bytes() + b"2,y\n")
    with pytest.raises(RecordHashMismatch):
        read_csv_checked(tmp_path / "r.csv", h)


def test_C_시한은_600_초까지는_기다리고_넘으면_멈춘다():
    """30 초씩 스무 번(600 초)은 기다리고 스물한 번째에 멈춘다 — 시한의 경계(06b m-21)."""
    seq = iter([9.0] * 20 + [12.0])
    g, clk = _guard(c_fn=lambda: next(seq))
    assert g.wait()["c_free_gb"] == 12.0 and clk.slept == 600
    g2, _ = _guard(c_fn=lambda: 9.0)
    with pytest.raises(GuardStop):
        g2.wait()
    assert g2.waits == 21


def test_참조_해시의_앞자리가_다르면_멈춘다(tmp_path):
    p = tmp_path / "r.tsv"
    p.write_text("x", encoding="utf-8")
    good = hashlib.sha256(b"x").hexdigest()[:16]
    check_reference(p, good)
    with pytest.raises(CopyMismatch):
        check_reference(p, "0" * 16)


def test_잘린_꼬리는_이어_쓰기_전에_고친다(tmp_path):
    from data.vt.copy import repair_jsonl_tail
    p = tmp_path / "r.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a"', encoding="utf-8")
    assert repair_jsonl_tail(p) and p.read_text(encoding="utf-8") == '{"a": 1}\n{"a": 2}\n'
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"a": 3}\n')
    assert load_jsonl_tolerant(p) == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert not repair_jsonl_tail(p)


def test_원천_이미지_zip_의_내용_증거는_서버_md5_뿐이고_없으면_따로_싣는다():
    a, b = b"aaaa", b"bb"
    listing = [(4, "Training/01/TS_A.zip"), (2, "Training/02/TL_B.zip")]
    recs = {"TS_A.zip": _rec("TS_A.zip", a), "TL_B.zip": _rec("TL_B.zip", b)}
    lab = {"TL_B.zip": hashlib.sha256(b).hexdigest()}
    md5 = {n: [r["md5"]] for n, r in recs.items()}
    r = verify(recs, listing, lab, md5)
    assert r["all_ok"] and r["image_content_evidence_all"] and r["image_zips_with_content_evidence"] == 1
    r = verify(recs, listing, lab, {**md5, "TS_A.zip": []})
    assert r["all_ok"] and not r["image_content_evidence_all"]


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="정션은 Windows 에만 있다")
def test_쓰기_가드는_정션을_풀어_동결_자리를_알아본다(tmp_path):
    """스테이징 안의 정션이 동결 매니페스트를 가리키면 막는다(06b m-1). 붙인 정션은 시험이 뗀다."""
    import subprocess
    root = tmp_path / "repo"
    frozen = root / "data/interim/manifest_v1"
    frozen.mkdir(parents=True)
    staging = root / "data/interim/vt_build_r1"
    staging.mkdir(parents=True)
    link = staging / "link"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(frozen)], check=True, capture_output=True)
    try:
        with pytest.raises(WriteForbidden):
            assert_vt_writable(link / "x.csv", allowed_roots=[staging], repo_root=root)
        assert_vt_writable(staging / "ok.csv", allowed_roots=[staging], repo_root=root)
    finally:
        __import__("os").rmdir(link)                   # 링크만 뗀다 — 대상은 그대로다
    assert frozen.is_dir()


def test_설정의_경로는_상대_경로이고_상위로_올라가지_않는다():
    from data.vt.config import repo_path
    with pytest.raises(ValueError):
        repo_path("data/../../x")
    with pytest.raises(ValueError):
        repo_path("C:/x")


def test_구간마다_원천을_다시_연다(tmp_path, monkeypatch):
    """한 손잡이로 끝까지 읽으면 공유 드라이브 캐시가 C: 를 채운다(3판 1-2) — 구간 수만큼 원천을 연다(06b m-21)."""
    from pathlib import Path as _P
    src = _src(tmp_path)
    opened = []
    real = _P.open

    def spy(self, *a, **k):
        if self == src:
            opened.append(1)
        return real(self, *a, **k)
    monkeypatch.setattr(_P, "open", spy)
    g, _ = _guard()
    r = copy_zip(src, tmp_path / "out" / src.name, chunk_bytes=1024, read_bytes=100, guard=g, log=lambda d: None)
    assert r["chunks"] == 10 and len(opened) == 10


def test_한글_가운데에서_잘린_꼬리도_그_줄만_버리고_고친다(tmp_path):
    """줄을 바이트로 가른 뒤 줄마다 푼다 — 여러 바이트 글자의 가운데에서 잘려도 앞의 줄은 산다(06c N-5)."""
    from data.vt.copy import repair_jsonl_tail
    p = tmp_path / "r.jsonl"
    good = '{"name": "TS_VTST_정상.zip"}\n'.encode()
    cut = '{"name": "TS_VTST_정상'.encode()[:-1]          # '상' 의 가운데에서 잘렸다
    p.write_bytes(good + cut)
    assert load_jsonl_tolerant(p) == [{"name": "TS_VTST_정상.zip"}]
    assert repair_jsonl_tail(p) and p.read_bytes() == good
    p.write_bytes(cut + b"\n" + good)                           # 가운데가 깨졌으면 멈춘다
    with pytest.raises((ValueError, UnicodeDecodeError)):
        load_jsonl_tolerant(p)
