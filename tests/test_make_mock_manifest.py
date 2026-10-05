"""mock 생성기 — 덮어쓰기 가드 · 결정성 · 동결본의 기대값.

세 가지는 서로 다른 것을 본다. 섞지 않는다(검토 (다)-2, 09-21 추기 7번).

- **가드**: 산출물이 이미 있는 경로에는 쓰지 않는다.
- **결정성**: 같은 입력이면 같은 바이트다. 동결본과 같다는 뜻이 **아니다** — `mock_riawelc_v1` 은
  08-21 사상표 교체 뒤로 현행 입력에서 동결본이 나오지 않으며, 그래야 한다는 기대값을 두지 않는다.
- **동결본의 기대값**: 스냅샷 디렉터리 **밖**(`data/mock/FROZEN_RECORD.yaml`)에 고정한 값과 대조한다.
  `verify_snapshot()` 은 같은 디렉터리의 잠금 기록과만 대조하므로, 파일과 잠금 기록이 함께 바뀌면 통과한다.

아래는 가드의 배경이다.


`data/mock/` 은 동결 스냅샷이다(개발규약 1-6). 생성기의 기본 출력이 바로 그 경로라서, 인자 없이
한 번 돌리면 봉인본이 덮였다 — `mock_riawelc_v1` 은 사상표 표기가 바뀐 뒤라 바이트까지 달라진다
(65번 §8·§9). 문서의 경고가 아니라 코드가 막는다.

**이 시험은 실물 `data/mock/` 에 쓰지 않는다.** 출력은 전부 `tmp_path` 이고, 기본 경로를 다루는
시험은 `write_snapshot` 을 "불리면 실패" 로 바꿔 놓은 채 돌린다.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

from data.manifest_io import (
    ANNOTATIONS_FILENAME,
    CAPABILITIES_FILENAME,
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    SNAPSHOT_MEMBERS,
    load_snapshot,
    verify_snapshot,
)
from scripts import make_mock_manifest as M

REPO = Path(__file__).resolve().parents[1]
REAL_MOCK = REPO / "data" / "mock"
RECORD = REAL_MOCK / "FROZEN_RECORD.yaml"           # 스냅샷 디렉터리 밖
PROFILES = ("mock_aihub_v1", "mock_riawelc_v1")     # configs/mock_profile.yaml 의 순서
SMALL = "mock_riawelc_v1"                           # 300행 — 빠르다


def _tree(root: Path) -> dict[str, str]:
    """경로 → 내용 해시. 쓰기가 한 바이트도 없었는지 보는 데 쓴다."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_새_경로에는_정상_생성된다(tmp_path):
    out = tmp_path / "fresh"
    assert M.main(["--profile", SMALL, "--out-root", str(out)]) == 0
    snap = load_snapshot(out / SMALL)               # 해시 검증까지 통과해야 한다
    assert len(snap.manifest) == 300


#: 생성기가 쓰는 네 파일 각각, 그리고 생성기가 모르는 파일. **무엇이든 있으면 남의 자료다.**
OCCUPANTS = [SNAPSHOT_FILENAME, MANIFEST_FILENAME, ANNOTATIONS_FILENAME, CAPABILITIES_FILENAME,
             "남이_둔_파일.txt"]


@pytest.mark.parametrize("existing", OCCUPANTS)
def test_산출물이_있는_경로는_거부하고_기존_바이트가_그대로다(tmp_path, capsys, existing):
    d = tmp_path / SMALL
    d.mkdir()
    (d / existing).write_bytes(b"sealed-bytes\r\nnot-to-be-touched\n")
    before = _tree(tmp_path)

    rc = M.main(["--profile", SMALL, "--out-root", str(tmp_path)])

    assert rc == 2                                  # 명세가 2 다 — 다른 비정상 코드로 바뀌는 회귀도 잡는다
    assert _tree(tmp_path) == before                # 새 파일도, 바뀐 파일도 없다
    err = capsys.readouterr().err
    assert str(d) in err and "--out-root" in err and "새" in err


@pytest.mark.parametrize("existing", OCCUPANTS)
def test_여러_프로필_중_하나만_선재해도_아무것도_쓰지_않는다(tmp_path, existing):
    """순서상 먼저인 프로필이 쓰인 뒤에 멈추면 반쯤 쓴 상태가 남는다 — 쓰기 전에 전부 본다.

    나중 프로필에 `annotations.csv` 나 `data_capabilities.yaml` **만** 있어도 같아야 한다(검토 (라)-1).
    """
    later = tmp_path / PROFILES[1]                  # 나중 프로필만 선재
    later.mkdir()
    (later / existing).write_text("sealed\n", encoding="utf-8")
    before = _tree(tmp_path)

    assert M.main(["--out-root", str(tmp_path)]) == 2

    assert _tree(tmp_path) == before
    assert not (tmp_path / PROFILES[0]).exists()    # 먼저인 프로필 디렉터리조차 생기지 않는다


def test_빈_디렉터리는_허용한다(tmp_path):
    """비어 있으면 덮을 것이 없다. 안내문의 새 출력 폴더 절차가 폴더를 먼저 만들어도 막히지 않는다."""
    (tmp_path / SMALL).mkdir()
    assert M.main(["--profile", SMALL, "--out-root", str(tmp_path)]) == 0
    assert len(load_snapshot(tmp_path / SMALL).manifest) == 300


def test_하위_디렉터리만_있어도_비어_있지_않은_것이다(tmp_path):
    d = tmp_path / SMALL
    (d / "sub").mkdir(parents=True)
    before = _tree(tmp_path)
    assert M.main(["--profile", SMALL, "--out-root", str(tmp_path)]) == 2
    assert _tree(tmp_path) == before and not (d / MANIFEST_FILENAME).exists()


def test_그_자리에_파일이_있으면_거부한다(tmp_path, capsys):
    """스냅샷 디렉터리 이름으로 **파일**이 놓여 있는 경우 — 역시 남의 것이다."""
    (tmp_path / SMALL).write_text("디렉터리가 아니다\n", encoding="utf-8")
    before = _tree(tmp_path)
    assert M.main(["--profile", SMALL, "--out-root", str(tmp_path)]) == 2
    assert _tree(tmp_path) == before
    assert str(tmp_path / SMALL) in capsys.readouterr().err


def test_기본_출력은_저장소의_data_mock_이고_그_상태에서_거부된다(monkeypatch, capsys):
    """기본 경로가 동결본이라는 사실과, 인자 없는 실행이 거부된다는 것. 실물에는 쓰지 않는다."""
    assert M.OUT_ROOT == REAL_MOCK
    assert all((REAL_MOCK / name / SNAPSHOT_FILENAME).is_file() for name in PROFILES)

    def _must_not_write(*_a, **_k):
        raise AssertionError("가드가 뚫렸다 — write_snapshot 이 불렸다")

    monkeypatch.setattr(M, "write_snapshot", _must_not_write)   # 가드가 깨져도 실물은 안전하다
    before = _tree(REAL_MOCK)

    rc = M.main([])                                  # 인자 없음 = 기본 --out-root

    assert rc == 2 == M.EXIT_OCCUPIED
    assert _tree(REAL_MOCK) == before
    err = capsys.readouterr().err
    assert all(str(REAL_MOCK / name) in err for name in PROFILES)   # 걸린 곳을 전부 알려 준다


def test_강제_덮어쓰기_옵션은_없다(tmp_path):
    for flag in ("--force", "--overwrite", "-f"):
        with pytest.raises(SystemExit) as e:
            M.main(["--profile", SMALL, "--out-root", str(tmp_path / "x"), flag])
        assert e.value.code == 2                    # argparse: 모르는 인자
    assert not (tmp_path / "x").exists()


# ------------------------------------------------------------------ 결정성

def test_같은_입력으로_두_번_만들면_바이트가_같다(tmp_path):
    """생성 시각이 상수라 입력이 같으면 바이트가 같다. **동결본과의 비교가 아니다.**

    범위는 **같은 프로세스 안의 두 번 · 기본 판정 모드**다. 별도 프로세스 사이의 결정성과 안내문의
    `absolute` 예제는 이 시험이 보지 않는다 — 범위를 넓히지 않았다(검토 (라)-5).
    """
    a, b = tmp_path / "a", tmp_path / "b"
    assert M.main(["--out-root", str(a)]) == 0
    assert M.main(["--out-root", str(b)]) == 0
    first = _tree(a)
    assert first == _tree(b)
    assert {k.split("/")[0] for k in first} == set(PROFILES)       # 두 프로필 모두 만들어졌다
    assert all(f"{name}/{SNAPSHOT_FILENAME}" in first for name in PROFILES)


# ------------------------------------------------------ 동결본의 기대값(밖에 고정)

def _digest_from_bytes(d: Path) -> tuple[dict[str, str], str]:
    """구성 파일의 바이트에서 직접 다시 계산한다 — 디렉터리 안의 잠금 기록을 믿지 않는다."""
    files = {m: hashlib.sha256((d / m).read_bytes()).hexdigest() for m in SNAPSHOT_MEMBERS}
    lines = "".join(f"{files[m]}  {m}\n" for m in SNAPSHOT_MEMBERS)
    return files, hashlib.sha256(lines.encode("utf-8")).hexdigest()


def test_동결본이_스냅샷_밖에_고정한_기대값과_같다():
    """생성기가 파일과 `SNAPSHOT.sha256` 을 함께 바꿔도 여기서 걸린다 — 기대값이 그 디렉터리 밖에 있다.

    구성 파일의 바이트, 거기서 다시 계산한 digest, **잠금 파일 자체의 바이트**, 그리고 디렉터리의
    **실제 파일 집합**을 본다. 잠금 목록 밖의 파일이 하나 끼어도, 하나가 빠져도 걸린다.
    """
    rec = yaml.safe_load(RECORD.read_text(encoding="utf-8"))
    assert set(rec["snapshots"]) == set(PROFILES)
    for name, want in rec["snapshots"].items():
        d = REAL_MOCK / name
        lock = want["lock_file"]
        assert {e.name for e in d.iterdir()} == {*want["files"], lock["name"]}, name     # 늘지도 줄지도 않았다
        assert all(e.is_file() for e in d.iterdir()), name
        files, digest = _digest_from_bytes(d)
        assert files == want["files"], name
        assert digest == want["snapshot_digest"], name
        assert hashlib.sha256((d / lock["name"]).read_bytes()).hexdigest() == lock["sha256"], name
        assert verify_snapshot(d) == want["snapshot_digest"], name      # 잠금 기록도 같은 값을 말한다


def test_data_mock_의_디렉터리는_기록된_스냅샷뿐이다():
    """잠금 파일이 **없는** 새 디렉터리도 잡는다 — `*/SNAPSHOT.sha256` 로만 찾으면 그런 디렉터리는 안 보인다."""
    rec = yaml.safe_load(RECORD.read_text(encoding="utf-8"))
    assert {e.name for e in REAL_MOCK.iterdir() if e.is_dir()} == set(rec["snapshots"])


#: `history_status` 에 쓸 수 있는 값과 그 뜻.
HISTORY_STATUS = {
    "공개된_적_없음": "교체 커밋의 부모에 있던 판. 공개 저장소에 올라간 적이 없다",
    "공개_이력_재작성으로_제거": "공개된 적은 있으나 2026-09-21 이력 재작성으로 공개본에서 사라졌다",
}


def _entries() -> dict[str, dict]:
    """기록의 입력·코드 항목 전부. **이 파일에서 이 함수는 하나뿐이어야 한다** —
    같은 이름을 두 번 정의하면 뒤엣것이 앞엣것을 덮어 앞을 쓰는 시험이 조용히 죽는다(09-25).
    """
    rec = yaml.safe_load(RECORD.read_text(encoding="utf-8"))
    rvi, cur = rec["restoration_verified_inputs"], rec["current_inputs"]
    out = {"복원_사상표": rvi["label_map"], "복원_프로필": rvi["mock_profile"],
           "환경_uv_lock": rvi["environment"]["uv_lock"], "현행_사상표": cur["label_map"]}
    out.update({f"생성기:{rel}": v for rel, v in rvi["generator"].items()})
    return out


def _recorded_blobs() -> list[tuple[str, str, str, str | None]]:
    """기록에 적힌 `(이름, sha256, git_blob, 로컬 보존 ref)` 전부.

    마지막 값이 `None` 이 아니면 그 blob 은 **공개본에 없다** — 그 ref 를 가진 저장소에서만 대조된다.
    `None` 인 항목은 현재 트리의 파일과 같은 blob 이므로 **공개본에서도 반드시 대조돼야 한다.**
    """
    return [(label, v["sha256"], v["git_blob"], None if v.get("in_public_history", True) else v["local_ref"])
            for label, v in _entries().items()]


def _git(*args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    """git 을 **못 부르는 것**은 건너뛸 사유가 아니라 고장이다 — 예외를 그대로 올린다."""
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, input=stdin, timeout=30, check=False)


def _ref_exists(ref: str) -> bool:
    """보존 ref 가 이 저장소에 있는가. 전체 경로(`refs/frozen/...`)와 브랜치 이름을 모두 받는다.

    blob 에 직접 건 ref 는 `refs/heads/` 아래에 있지 않다 — 브랜치로만 찾으면 있는 ref 도 없다고 본다.
    """
    cand = [ref] if ref.startswith("refs/") else [f"refs/heads/{ref}", ref]
    return any(_git("rev-parse", "--verify", "--quiet", c).returncode == 0 for c in cand)


def test_기록의_입력_해시는_형식을_갖췄다():
    """**형식 검사다.** 값이 실제 입력과 대응하는지는 보지 않는다 — 그것은 아래 시험의 일이다.

    사상표 `version` 은 복원 검증 입력에서도 지금도 1 이라, 버전으로는 둘을 구별하지 못한다는 사실도 여기서 고정한다.
    """
    rec = yaml.safe_load(RECORD.read_text(encoding="utf-8"))
    then, now = rec["restoration_verified_inputs"]["label_map"], rec["current_inputs"]["label_map"]
    assert then["version_field"] == now["version_field"] == 1
    assert then["sha256"] != now["sha256"] and then["git_blob"] != now["git_blob"]
    for _label, sha256, blob, _local_ref in _recorded_blobs():
        assert len(sha256) == 64 and set(sha256) <= set("0123456789abcdef")
        assert len(blob) == 40 and set(blob) <= set("0123456789abcdef")
    assert "복원이 검증된 입력" in rec["restoration_verified_inputs"]["description"]


@pytest.mark.parametrize("label, sha256, blob, local_ref", _recorded_blobs(), ids=[x[0] for x in _recorded_blobs()])
def test_기록의_입력_해시가_Git_에_있는_내용과_같다(label, sha256, blob, local_ref):
    """**내부 이력 대조다.** 기록된 blob 을 Git 에서 꺼내 내용의 sha256 과 blob id 를 다시 계산한다.

    **공개본은 이 대조를 할 수 없다.** 공개 저장소는 부모 없는 단일 커밋이라 과거 판의 blob 이 객체 DB 에 없다.
    그 사실을 실패가 아니라 **사유를 적은 건너뜀**으로 남긴다 — 검증하지 못한 범위를 지우지 않기 위해서다.
    현재 파일 쪽 대조는 `test_병기한_지금_값이_실제_현행_파일과_같다` 와
    `test_바뀌지_않은_항목은_공개본에서도_대조된다` 가 맡는다.

    건너뛰는 경우는 셋뿐이고 전부 "이 트리에서는 볼 수 없다" 는 사실이다 — 통과로 세지 않는다.

    1. 이 트리가 Git 저장소가 아니다(`git archive` 로 뜬 사본).
    2. 얕은 클론이라 과거 객체가 없다.
    3. **기록이 공개본에 없다고 밝힌 blob**(`in_public_history: false`)이고, 그것을 품은 보존 ref 도 이 저장소에 없다.

    그 밖의 누락은 **실패다**: 공개본에 있어야 할 객체가 없거나, 보존 ref 는 있는데 그 blob 이 없거나(기록의 id 가
    틀렸다), git 이 다른 이유로 실패한 경우. 기록의 id 를 없는 값으로 바꿔도 조용히 건너뛰지 않는다.
    """
    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip(f"이 트리는 Git 저장소가 아니다 — {label} 을 대조하지 못했다")

    have = _git("cat-file", "-e", blob).returncode == 0
    if not have:
        if _git("rev-parse", "--is-shallow-repository").stdout.strip() == b"true":
            pytest.skip(f"얕은 클론이라 과거 객체가 없다 — {label} 을 대조하지 못했다")
        if local_ref is not None and not _ref_exists(local_ref):
            status = _entries()[label].get("history_status", "(사유 없음)")
            pytest.skip(f"{label}: {status}. 보존 ref {local_ref} 도 이 저장소에 없다 — "
                        f"**공개본에서는 당시 판을 대조할 수 없다**")
        pytest.fail(f"{label}: 기록된 blob {blob} 이 이 저장소에 없다 — 기록의 id 가 틀렸거나 객체가 사라졌다")

    r = _git("cat-file", "blob", blob)
    assert r.returncode == 0, (label, r.stderr[:200])
    content = r.stdout
    assert hashlib.sha256(content).hexdigest() == sha256, label
    assert hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest() == blob, label   # Git 객체 id


# ------------------------------ 공개본이 검증할 수 있는 범위와 없는 범위

def test_바뀌지_않은_항목은_공개본에서도_대조된다():
    """기록된 blob 이 **현재 파일의 blob 과 같은** 항목은 단일 커밋 공개본에도 그 객체가 있다.

    그런 항목에 건너뛰기 장치를 달면 안 된다 — 달면 공개본에서 아무것도 확인하지 않고 통과한다.
    대조는 git 없이 **내용에서** 한다(`_lf_sha256`·`_blob_id`).
    """
    checked = 0
    for label, v in _entries().items():
        if not v.get("in_public_history", True):
            continue
        rel = v.get("path") or label.split(":", 1)[-1]
        assert (REPO / rel).is_file(), f"{label}: 경로 {rel} 가 없다 — 기록에 `path` 를 적어라"
        assert v["sha256"] == _lf_sha256(rel), f"{label}({rel}): 기록된 해시가 현재 파일과 다르다"
        assert v["git_blob"] == _blob_id(rel), f"{label}({rel}): 기록된 blob 이 현재 파일과 다르다"
        checked += 1
    assert checked >= 4, f"공개본에서 대조되는 항목이 {checked}개뿐이다 — 장치가 과하게 붙었다"


def test_공개본에_없는_항목은_사유와_보존_ref_를_함께_적는다():
    """**이빨 시험.** 사유 없이 건너뛰기 장치만 붙이면 "검증 못 함" 이 "검증함" 처럼 보인다.

    `in_public_history: false` 를 붙였으면 `history_status` 로 **왜** 없는지 적고 보존 ref 를 준다.
    그리고 공개 이력 재작성으로 사라진 항목은 **현재 판의 해시**(`after_public_cleanup`)가 반드시 있어야 한다 —
    공개본이 확인할 수 있는 것이 하나도 없으면 안 되기 때문이다.
    """
    marked = 0
    for label, v in _entries().items():
        if v.get("in_public_history", True):
            assert "history_status" not in v, f"{label}: 공개본에 있는 항목에 사유를 붙이지 않는다"
            continue
        marked += 1
        status = v.get("history_status")
        assert status in HISTORY_STATUS, f"{label}: history_status 가 없거나 모르는 값이다 — {status!r}"
        assert v.get("local_ref"), f"{label}: 보존 ref 가 없다"
        if status == "공개_이력_재작성으로_제거":
            assert "after_public_cleanup" in v, f"{label}: 현재 판의 해시가 없어 공개본이 확인할 것이 없다"
    assert marked == 6, f"건너뛰기 장치가 붙은 항목이 {marked}개다 — 늘거나 줄면 근거를 다시 적어라"


# ------------------------------ 해시가 갈라진 항목의 병기 형식 (09-21 공개본 표현 정리)

#: 동결 검증 당시의 해시. **이 표는 시험 안에 박아 둔다** — 기록의 값을 새 값으로 덮고 병기 블록을 지우면
#: 기록만 보는 검사로는 그 사실을 알 수 없기 때문이다. 여기 적힌 값이 기록에서 사라지면 실패한다.
AT_FREEZE = {
    "복원_프로필": "1ef4df080abbf626766b1a97e2ce865f9aa99674fea9ef17b3a82a2c845745d0",
    "생성기:scripts/make_mock_manifest.py": "ed2b4a1748c425710cd0b04f12d0a8e2b5ab4e7dc1e854908d6d247ed95f8a64",
    "생성기:data/manifest_io.py": "d36de98e74480d6ef54b47d751bb49d3a3cc02e83d3521a36441b122cd324816",
    "생성기:data/label_map.py": "63b41398e02edbb16621f4651ce36a058f8f28ffe7bc4987e68fcf7d25640eae",
    "현행_사상표": "e09492734ba47b2636826444ca8627154cd7735e82ac75da4ddd69ff4bfba775",
}
#: 각 항목이 가리키는 저장소 안의 파일.
AT_FREEZE_PATH = {
    "복원_프로필": "configs/mock_profile.yaml",
    "생성기:scripts/make_mock_manifest.py": "scripts/make_mock_manifest.py",
    "생성기:data/manifest_io.py": "data/manifest_io.py",
    "생성기:data/label_map.py": "data/label_map.py",
    "현행_사상표": "configs/label_map.yaml",
}


def _lf_bytes(rel: str) -> bytes:
    """저장소 파일의 내용. 기록의 규칙대로 줄끝을 LF 로 정규화한다(`.gitattributes` 가 이 파일들을 `text eol=lf` 로 잡는다)."""
    return (REPO / rel).read_bytes().replace(b"\r\n", b"\n")


def _lf_sha256(rel: str) -> str:
    """내용 해시. 위 바이트에서 낸다."""
    return hashlib.sha256(_lf_bytes(rel)).hexdigest()


def _blob_id(rel: str) -> str:
    """blob id 를 **내용에서 직접** 낸다 — 커밋 여부와 무관하다.

    `HEAD:<파일>` 을 보면 안 되는 이유: 머지 검수는 머지를 커밋하지 않고 열어 둔 채 시험을 돌린 뒤
    통과해야 커밋한다. 그때 HEAD 는 아직 머지 전이라 새 파일이 아니라 **옛 파일의 blob** 이 나온다.
    작업 트리에 커밋하지 않은 변경이 있는 사람도 같은 자리에서 깨진다. blob id 는 "깃이 저장할 값"
    이므로 내용에서 내는 편이 뜻에 맞고, 위의 sha256 과 같은 바이트에서 나오되 표기만 깃 형식이라
    어디서 돌려도 같은 답이 된다.

    `git hash-object --path` 는 `.gitattributes` 를 반영하므로 그것을 먼저 쓴다. git 이 없으면
    같은 값을 직접 계산한다 — 이 파일들에는 clean 필터가 없고 내용이 이미 LF 라 두 값이 같다.
    """
    data = _lf_bytes(rel)
    r = _git("hash-object", "--path", rel, "--stdin", stdin=data)
    if r.returncode == 0:
        return r.stdout.decode().strip()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.mark.parametrize("label", sorted(AT_FREEZE))
def test_해시가_갈라진_항목은_당시_값과_지금_값을_함께_적는다(label):
    """**한쪽만 있으면 실패한다.**

    당시 값을 지우고 새 값만 남기면 "이 바이트가 동결본을 복원했다" 는 기록이 거짓이 되고,
    새 값을 적지 않으면 기록이 현행 파일을 가리키지 못한다. 둘을 함께 적어야 사슬이 끊긴 자리가 남는다.
    """
    v = _entries()[label]
    assert v["sha256"] == AT_FREEZE[label], f"{label}: 동결 검증 당시의 해시가 기록에서 사라졌다 — 덮어쓰지 않는다"
    after = v.get("after_public_cleanup")
    assert after is not None, f"{label}: 정리 뒤 해시가 없다 — 당시 값만 남으면 기록이 현행 파일을 가리키지 못한다"
    assert after["sha256"] != v["sha256"] and after["git_blob"] != v["git_blob"], label
    for key in ("sha256", "git_blob"):
        assert len(after[key]) == (64 if key == "sha256" else 40)
        assert set(after[key]) <= set("0123456789abcdef"), label
    assert after.get("changed") and after.get("evidence"), f"{label}: 무엇이 바뀌었는지와 그 근거가 없다"


@pytest.mark.parametrize("label", sorted(AT_FREEZE))
def test_병기한_지금_값이_실제_현행_파일과_같다(label):
    """새로 적은 값이 트리의 파일과 맞는지 본다. 값만 적고 파일이 또 바뀌면 여기서 걸린다.

    두 값 모두 **작업 트리의 내용**에서 낸다(`_blob_id` 의 독스트링 참조). 커밋된 상태를 기준으로 삼으면
    머지를 열어 둔 채 도는 검수와 커밋 전 작업 트리에서 실패한다 — 파일이 맞는데도 실패하는 시험이다.
    """
    after = _entries()[label]["after_public_cleanup"]
    rel = AT_FREEZE_PATH[label]
    assert after["sha256"] == _lf_sha256(rel), f"{label}({rel}): 병기한 해시가 현행 파일과 다르다"
    assert after["git_blob"] == _blob_id(rel), f"{label}({rel}): 병기한 blob id 가 현행 파일의 내용과 다르다"


def test_갈라지지_않은_항목은_병기하지_않는다():
    """반대 방향이다. 바뀌지 않은 파일에 병기 블록이 붙으면 그것도 기록이 틀린 것이다.

    `복원_사상표` 는 예외다 — 08-21 교체 **이전** 판이라 애초에 현행 파일이 아니고,
    그 짝이 `현행_사상표` 다. 기록이 `in_public_history: false` 로 그 사실을 이미 말한다.
    """
    for label, v in _entries().items():
        if label in AT_FREEZE:
            continue
        if v.get("in_public_history", True) is False:
            assert "after_public_cleanup" not in v, f"{label}: 과거 판에는 병기하지 않는다"
            continue
        rel = v.get("path") or label.split(":", 1)[-1]
        if rel == "환경_uv_lock":
            rel = "uv.lock"
        assert "after_public_cleanup" not in v, f"{label}: 바뀌지 않았는데 병기 블록이 있다"
        assert v["sha256"] == _lf_sha256(rel), f"{label}({rel}): 파일이 바뀌었는데 병기가 없다"


def test_병기는_한_번의_변경을_가리키고_재확인_결과를_담는다():
    """병기가 "값이 달라졌다" 로 끝나면 근거가 없다. 그 변경이 결과를 바꾸지 않았다는 확인까지 함께 있어야 한다."""
    rec = yaml.safe_load(RECORD.read_text(encoding="utf-8"))
    assert "after_public_cleanup" in rec, "병기한 항목이 있는데 그 변경을 설명하는 블록이 없다"
    blk = rec["after_public_cleanup"]
    assert len(blk["commit"]) == 40 and set(blk["commit"]) <= set("0123456789abcdef")
    assert blk["changed_files"] == len(AT_FREEZE)
    assert blk["why_not_overwrite"] and blk["change_kind"] and blk["scope"]
    re = blk["reverify"]
    assert re["how"]
    # 재확인이 말하는 digest 가 이 기록의 동결 기대값과 같은 값이어야 한다 — 다른 숫자를 적어 두면 근거가 아니다.
    aih, ria = re["mock_aihub_v1"], re["mock_riawelc_v1"]
    assert aih["same_as_frozen"] is True
    assert aih["snapshot_digest"] == rec["snapshots"]["mock_aihub_v1"]["snapshot_digest"]
    assert aih["lock_file_sha256"] == rec["snapshots"]["mock_aihub_v1"]["lock_file"]["sha256"]
    assert ria["same_as_frozen"] is False
    assert ria["snapshot_digest"] == rec["current_inputs"]["effect"]["mock_riawelc_v1_digest_with_current_inputs"]
    assert ria["snapshot_digest"] != rec["snapshots"]["mock_riawelc_v1"]["snapshot_digest"]


def test_재확인_digest_가_현행_코드의_실제_산출과_같다(tmp_path):
    """기록된 재확인 결과를 **다시 돌려서** 확인한다. 적어 놓기만 한 값이면 여기서 갈린다."""
    blk = yaml.safe_load(RECORD.read_text(encoding="utf-8"))["after_public_cleanup"]["reverify"]
    assert M.main(["--out-root", str(tmp_path)]) == 0
    _files, digest = _digest_from_bytes(tmp_path / "mock_aihub_v1")
    assert digest == blk["mock_aihub_v1"]["snapshot_digest"]
    lock = tmp_path / "mock_aihub_v1" / SNAPSHOT_FILENAME
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == blk["mock_aihub_v1"]["lock_file_sha256"]
    _files, digest = _digest_from_bytes(tmp_path / "mock_riawelc_v1")
    assert digest == blk["mock_riawelc_v1"]["snapshot_digest"]


# ------------------------------ 병기 값을 낸 커밋이 기록에 있는가 (2026-10-01)

def _introducing_commits(blob: str, rel: str) -> set[str]:
    """그 blob 을 경로에 **들인** 커밋들. `--find-object` 는 나타나거나 사라지는 커밋을 모두 내므로 나타난 쪽만 고른다.

    HEAD 이력만 보지 않고 저장소의 ref 전부에서 찾는다 — 머지 검수는 머지를 커밋하지 않고 연 채 시험을 돌리므로
    그때 HEAD 는 아직 병합되지 않은 커밋을 품지 않는다(`_blob_id` 의 독스트링과 같은 사정).
    """
    r = _git("log", "--all", "--format=%H", f"--find-object={blob}", "--", rel)
    assert r.returncode == 0, r.stderr[:200]
    out = set()
    for c in r.stdout.decode().split():
        now = _git("rev-parse", "--verify", "--quiet", f"{c}:{rel}")
        before = _git("rev-parse", "--verify", "--quiet", f"{c}^:{rel}")
        if now.stdout.decode().strip() == blob and before.stdout.decode().strip() != blob:
            out.add(c)
    return out


def test_병기_값을_낸_커밋이_최상위_블록이_적은_커밋_안에_있다():
    """병기 값마다 그것을 **들인 커밋**을 이력에서 찾아, 최상위 블록이 적은 커밋(첫 변경과 `later_changes`) 안에 있는지 본다.

    항목의 값을 갱신하고 블록의 요약을 그대로 두면 요약이 거짓이 된다. 09-30 에 `manifest_io` 가 그랬다 —
    병기 값은 `65b2e5a` 가 냈는데 블록은 `ac31631` 한 번의 변경만 적었다. 형식 시험은 그것을 보지 못했다.

    공개본은 부모 없는 단일 커밋이라 적힌 커밋이 없다. 그때는 사유를 적고 건너뛴다(통과로 세지 않는다).
    """
    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("이 트리는 Git 저장소가 아니다 — 이력으로 대조할 수 없다")
    rec = yaml.safe_load(RECORD.read_text(encoding="utf-8"))
    blk = rec["after_public_cleanup"]
    later = blk.get("later_changes") or []
    allowed = {blk["commit"], *(e["commit"] for e in later)}
    missing = [c for c in allowed if _git("cat-file", "-e", f"{c}^{{commit}}").returncode != 0]
    if missing:
        pytest.skip(f"기록이 적은 커밋 {sorted(missing)} 이 이 저장소에 없다 — 공개본에서는 이력으로 대조할 수 없다")

    checked = 0
    for label in sorted(AT_FREEZE):
        after = _entries()[label]["after_public_cleanup"]
        rel = AT_FREEZE_PATH[label]
        intro = _introducing_commits(after["git_blob"], rel)
        assert intro, f"{label}({rel}): 병기 blob 을 들인 커밋을 이력에서 찾지 못했다"
        assert intro & allowed, (f"{label}({rel}): 병기 값을 낸 커밋 {sorted(c[:12] for c in intro)} 이 "
                                 f"최상위 블록이 적은 커밋 {sorted(c[:12] for c in allowed)} 에 없다")
        # 병기 블록 안에 따로 남긴 앞 시점의 값도 그 커밋이 낸 것이어야 한다.
        at = after.get("at_cleanup_commit")
        if at:
            assert at["commit"] == blk["commit"], label
            assert at["commit"] in _introducing_commits(at["git_blob"], rel), f"{label}: 앞 시점 값의 커밋이 다르다"
        checked += 1
    assert checked == len(AT_FREEZE) == blk["changed_files"]

    # 뒤이은 변경은 적은 파일을 실제로 바꿨고, 재확인이 가리키는 지문이 첫 변경의 재확인과 같다
    # (그 지문은 `test_재확인_digest_가_현행_코드의_실제_산출과_같다` 가 현행 코드로 다시 만든다).
    for e in later:
        touched = set(_git("show", "--name-only", "--format=", e["commit"]).stdout.decode().split())
        assert set(e["files"]) <= touched, e["commit"]
        rv, first = e["reverify"], blk["reverify"]
        assert rv["mock_aihub_v1_snapshot_digest"] == first["mock_aihub_v1"]["snapshot_digest"]
        assert rv["mock_aihub_v1_lock_file_sha256"] == first["mock_aihub_v1"]["lock_file_sha256"]
        assert rv["mock_riawelc_v1_snapshot_digest"] == first["mock_riawelc_v1"]["snapshot_digest"]
        assert "코드 추가" in e["change_kind"], "코드 추가를 숨기지 않는다"
