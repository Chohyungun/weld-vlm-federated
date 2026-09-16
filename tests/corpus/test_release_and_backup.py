"""공개 검수기·백업 대조기 이빨 시험. 09-08 총괄 판정 (24번 A-4 ①②).

둘 다 **통과가 곧 승인**이 되는 도구다. 공개 검수기가 놓치면 AI허브 파생물이나 규정
원문이 공개 저장소로 나가고, 백업 대조기가 놓치면 깨진 사본을 정본으로 믿는다. 그래서
"잡는가" 만이 아니라 **"안 잡으면 어떻게 되는가"** 를 같이 건다.

실물 봉인본은 읽기만 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.generate.frozen_out import CONTRACT_NAME
from corpus.validate import screen_public_release as S
from corpus.validate import verify_backup as B

REPO = Path(__file__).resolve().parents[2]


def _seal(d: Path, files: dict[str, str]) -> Path:
    """실물 + 계약서를 갖춘 가짜 봉인 디렉터리."""
    import hashlib

    d.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, body in files.items():
        p = d / name
        p.write_text(body, encoding="utf-8", newline="")
        entries.append((hashlib.sha256(p.read_bytes()).hexdigest(), name))
    digest = hashlib.sha256("".join(h for h, _ in entries).encode()).hexdigest()
    (d / CONTRACT_NAME).write_text(
        "\n".join(f"{h}  {n}" for h, n in entries) + f"\n# snapshot_digest {digest}\n",
        encoding="utf-8", newline="")
    return d


# ------------------------------------------------------------- 공개 검수기

def test_유료표준_식별자를_잡는다():
    hits = S.screen_text("이 판정은 ISO 5817 C 등급을 따른다", set())
    assert [h[0] for h in hits] == ["paid_std"]


def test_aihub_식별자를_잡는다():
    """AI허브 파생 식별자가 공개 저장소로 나가면 레드라인 1 위반이다."""
    hits = S.screen_text('{"image_id": "aihub71761:13121"}', set())
    assert "aihub_id" in [h[0] for h in hits]


def test_로컬_절대경로를_잡는다():
    hits = S.screen_text(r"E:\Fedvlm_for_welding\corpus 에서 읽었다", set())
    assert "local_path" in [h[0] for h in hits]


def test_원문_연속_일치를_잡는다():
    """규정 원문을 길게 그대로 실으면 재배포다 (.gitignore:54)."""
    src = "가" * 30 + "이 조항은 용접부의 표면 결함을 다루며 검사원이 확인한다" + "나" * 30
    sh = {src[i:i + S.SHINGLE] for i in range(len(src) - S.SHINGLE + 1)}
    assert S.screen_text(src[20:20 + S.SHINGLE + 10], sh)[0][0] == "verbatim_source"


def test_짧은_겹침은_잡지_않는다():
    """한계 미만까지 잡으면 흔한 표현마다 걸려 검수가 무의미해진다."""
    src = "용접부의 표면 결함을 다룬다 " * 20
    sh = {src[i:i + S.SHINGLE] for i in range(len(src) - S.SHINGLE + 1)}
    assert S.screen_text("용접부의 표면", sh) == []


def test_원천이_없으면_통과로_적지_않는다(monkeypatch, tmp_path):
    """원천 문서가 없으면 재배포 판정을 **못 한** 것이다. 조용히 clear 로 떨어지면 안 된다."""
    monkeypatch.setattr(S, "SOURCE_DOCS", (tmp_path / "없음.md",))
    monkeypatch.setattr(S, "TARGET_DIRS", ())
    r = S.screen()
    assert r["n_source_shingles"] == 0
    assert "warning" in r and "근거가 아니다" in r["warning"]


def test_검수는_적발되면_차단으로_끝난다(monkeypatch, tmp_path):
    """이빨 시험 — 적발이 종료코드에 반영되지 않으면 게이트가 아니다."""
    d = _seal(tmp_path / "cycle_x", {"a.jsonl": json.dumps(
        {"text": "ISO 5817 을 인용한다"}, ensure_ascii=False) + "\n"})
    monkeypatch.setattr(S, "TARGET_DIRS", (d,))
    monkeypatch.setattr(S, "untracked_members", lambda _d: ["a.jsonl"])
    assert S.main([]) == 1

    # 깨끗하면 0 이어야 한다. 늘 1이면 그것도 게이트가 아니다.
    (d / "a.jsonl").write_text(
        json.dumps({"text": "안전한 문장"}, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="")
    assert S.main([]) == 0


# ------------------------------------------------------------- 백업 대조기

def test_목록은_계약서_자신도_담는다(tmp_path):
    """계약서를 빼면 사본만으로는 무엇이 있어야 하는지 알 수 없다 — 사본이 자립해야 한다."""
    d = _seal(tmp_path / "pairs_x", {"pairs.jsonl": "{}\n", "counts.json": "{}\n"})
    plan = B.build_plan([d])
    assert {i["name"] for i in plan["items"]} == {"pairs.jsonl", "counts.json",
                                                 CONTRACT_NAME}
    assert plan["total_bytes"] > 0 and not plan["problems"]
    # 레드라인 문구가 목록에 붙어 있어야 사람이 목적지를 고를 때 본다.
    assert "국외" in plan["redline"]


def test_계약에_있는데_실물이_없으면_목록이_문제로_적는다(tmp_path):
    d = _seal(tmp_path / "pairs_x", {"pairs.jsonl": "{}\n"})
    (d / "pairs.jsonl").unlink()
    plan = B.build_plan([d])
    assert plan["problems"] and "실물이 없다" in plan["problems"][0]


def test_사본이_같으면_통과한다(tmp_path):
    d = _seal(tmp_path / "pairs_x", {"pairs.jsonl": "{}\n"})
    plan = B.build_plan([d])
    dest = tmp_path / "drive"
    (dest / d.name).mkdir(parents=True)
    for it in plan["items"]:
        (dest / it["rel"]).write_bytes(Path(it["src"]).read_bytes())
    r = B.verify(plan, dest)
    assert r["verdict"] == "ok" and r["n_bad"] == 0


@pytest.mark.parametrize("damage", ["mismatch", "missing"])
def test_사본이_다르면_잡는다(tmp_path, damage):
    """이빨 시험 — 손상과 누락을 둘 다 잡아야 한다. 하나만 잡으면 나머지로 샌다."""
    d = _seal(tmp_path / "pairs_x", {"pairs.jsonl": "{}\n", "counts.json": "{}\n"})
    plan = B.build_plan([d])
    dest = tmp_path / "drive"
    (dest / d.name).mkdir(parents=True)
    for it in plan["items"]:
        (dest / it["rel"]).write_bytes(Path(it["src"]).read_bytes())

    target = dest / d.name / "pairs.jsonl"
    if damage == "mismatch":
        target.write_bytes(b"{} \n")
    else:
        target.unlink()

    r = B.verify(plan, dest)
    assert r["verdict"] == "failed"
    assert [x["verdict"] for x in r["rows"] if x["rel"].endswith("pairs.jsonl")] == [damage]


def test_원본이_계약과_어긋나면_사본도_통과하지_않는다(tmp_path):
    """원본↔사본만 보면 원본이 이미 틀어져 있어도 같이 틀어진 채 통과한다."""
    d = _seal(tmp_path / "pairs_x", {"pairs.jsonl": "{}\n"})
    (d / "pairs.jsonl").write_text("변조됨\n", encoding="utf-8", newline="")
    plan = B.build_plan([d])                      # 계약서는 옛 해시를 들고 있다
    dest = tmp_path / "drive"
    (dest / d.name).mkdir(parents=True)
    for it in plan["items"]:
        (dest / it["rel"]).write_bytes(Path(it["src"]).read_bytes())
    r = B.verify(plan, dest)
    assert r["verdict"] == "failed"
    bad = [x for x in r["rows"] if x["verdict"] == "contract_mismatch"]
    assert bad and bad[0]["matches_source"] is True   # 사본은 원본과 같은데도 막혔다


def test_verify_는_dest_없이_돌지_않는다():
    assert B.main(["verify"]) == 2


# ------------------------------------------------------------- 실물 (읽기만)

def test_실물_백업_목록이_계약과_맞는다():
    """총괄이 옮길 실물. 목록이 계약과 어긋나면 사람이 잘못된 것을 복사한다.

    09-08 판정으로 cycle_pilot·cycle_pilot_v2 가 한 벌에 들어왔다(추적 전환 기각).
    09-15 부터 목록은 봉인처 명부에서 파생한다 — 소실 기록(v2)은 빠지고, 봉인 아닌
    `judge_labels/` 가 들어온다. 그래서 파일 수를 상수로 박지 않고 구성으로 센다.
    """
    plan = B.build_plan(B.DEFAULT_DIRS)
    if plan["problems"]:
        pytest.skip(f"이 트리에서 대상이 온전하지 않다: {plan['problems']}")
    assert {Path(d).name for d in B.DEFAULT_DIRS} == set(plan["dirs"])
    expected = 0
    for d in B.SEALED_DIRS:
        names, _ = B.snapshot_summary(d)
        expected += len(names) + 1                      # 구성원 + 계약서
    for d in B.UNSEALED_DIRS:
        expected += sum(1 for p in d.iterdir() if p.is_file())
    assert plan["n_files"] == expected, (plan["n_files"], expected)
    for it in plan["items"]:
        if it["contract_sha256"] is not None:
            assert it["sha256"] == it["contract_sha256"], it["rel"]


def test_백업_목록은_봉인처_명부에서_파생된다():
    """두 목록이 따로 살면 한쪽만 고쳐진다 — 09-13 에 이 목록이 소실된 v2 를 계속 들고 있었다."""
    from corpus.generate.frozen_out import EXPECTED_SEALED

    expected = {B.REPO / rel for rel, e in EXPECTED_SEALED.items()
                if e["status"] in B.BACKED_UP_STATUSES}
    lost = {B.REPO / rel for rel, e in EXPECTED_SEALED.items() if e["status"] == "lost"}
    assert set(B.SEALED_DIRS) == expected
    assert not (set(B.DEFAULT_DIRS) & lost), "소실 기록된 곳은 옮길 실물이 없다"
    assert B.REPO / "corpus/validate/judge_labels" in B.DEFAULT_DIRS


def test_복원된_봉인처도_백업_목록에_든다():
    """`restored` 는 소실 뒤 동일 바이트로 돌아온 자리다 — git 밖 단일 사본이 다시 생긴
    것이라 백업이 급하다. 상태 어휘를 늘린 쪽(A)과 목록을 파생하는 쪽(B)이 따로 살면
    목록이 조용히 줄어든다 — 09-13 에 반대 방향으로 같은 일이 있었다 (41번 I-1)."""
    reg = {"a/expected": {"status": "expected", "owner": "B"},
           "b/restored": {"status": "restored", "owner": "A", "record": "복원", "evidence": "64자 일치"},
           "c/lost": {"status": "lost", "owner": "B", "record": "소실"}}
    got = {p.as_posix().rsplit("/", 2)[-2] + "/" + p.name for p in B.sealed_dirs(reg, Path("/r"))}
    assert got == {"a/expected", "b/restored"}


def test_사람_라벨_폴더가_백업_목록에_있고_미추적분이_위험분으로_잡힌다(tmp_path, monkeypatch):
    """labels_*.jsonl 은 미추적이다. 백업 목록에 없으면 09-11 과 같은 단일 사본 구조다."""
    d = tmp_path / "judge_labels"
    d.mkdir()
    (d / "sheet_v1.jsonl").write_text("{}\n", encoding="utf-8", newline="")
    (d / "labels_v1.jsonl").write_text('{"labeler": "x"}\n', encoding="utf-8", newline="")
    monkeypatch.setattr(B, "UNSEALED_DIRS", (d,))
    monkeypatch.setattr(B, "tracked_names", lambda _d: frozenset({"sheet_v1.jsonl"}))
    plan = B.build_plan([d])
    assert not plan["problems"]
    by = {it["name"]: it for it in plan["items"]}
    assert by["labels_v1.jsonl"]["tracked"] is False and by["labels_v1.jsonl"]["sealed"] is False
    assert by["sheet_v1.jsonl"]["tracked"] is True
    assert plan["n_at_risk"] == 1
    # 봉인 아닌 항목은 계약 대조가 없다 — 그 사실이 항목에 남아야 verify 가 속지 않는다
    assert all(it["contract_sha256"] is None for it in plan["items"])


def test_봉인도_아니고_목록에도_없는_디렉터리는_여전히_문제로_적는다(tmp_path):
    """아무 디렉터리나 넘기면 전부 담아 주는 것이 아니다 — 명시된 곳만."""
    d = tmp_path / "아무거나"
    d.mkdir()
    (d / "x.txt").write_text("x", encoding="utf-8")
    plan = B.build_plan([d])
    assert plan["problems"] and "봉인본이 아니다" in plan["problems"][0]


def test_위험분과_안전분을_목록이_가른다():
    """★(git 밖)과 ㆍ(git 에도 있음)을 안 가르면 사람이 무엇이 유일본인지 모른다."""
    plan = B.build_plan(B.DEFAULT_DIRS)
    if plan["problems"]:
        pytest.skip("이 트리에서 대상이 온전하지 않다")
    assert 0 < plan["n_at_risk"] < plan["n_files"]
    assert plan["at_risk_bytes"] < plan["total_bytes"]
    # 추적분은 git 이 들고 있다는 뜻이므로 실제로 추적 중이어야 한다.
    from corpus.generate.frozen_out import tracked_names
    for it in plan["items"]:
        d = Path(it["src"]).parent
        assert it["tracked"] == (it["name"] in (tracked_names(d) or frozenset())), it["rel"]
