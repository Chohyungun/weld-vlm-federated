"""실행 실패와 미대조 범위가 겹칠 때의 보고. d56c91b 보완용.

외부 검토가 쓴 두 사례를 그대로 들였다. 들일 때 고치기 전 코드(c1e38b3)에서 둘 다 실패하고
지금 코드에서 통과하는 것을 확인했다 — 앞선 단언을 지난 뒤 이번 수정이 바꾼 값에서 멈춘다.
"""

from corpus.validate import seal_pairs_main as S


def test_실행실패에도_미대조_사실을_별도로_남긴다():
    row = S._check(
        "verify_pairs_main", 0,
        findings=["계약 구성원 부재"],
        broken=["보고와 종료 코드 불일치"],
        limited=["E 레코드"],
    )
    assert row["상태"] == S.BROKEN
    # 기존 범위제한 필드는 상태의 별칭으로 유지한다. 사실은 새 필드에 기록한다.
    assert row["범위제한"] is False
    assert row["미대조_있음"] is True
    assert "E 레코드" in row["사유"]


def test_실행실패와_범위제한을_최종_보고에도_함께_남긴다(tmp_path, monkeypatch):
    # verify()의 구성원 부재 조기 반환 모양에 잘못된 종료 코드 0을 주입한다.
    # 생산자의 정상 보고를 가장하는 픽스처가 아니라 소비자의 오류 처리를 보는 사례다.
    verify = {
        "build": "build",
        "failures": {"G_member_missing": 1},
        "failure_examples": {"G_member_missing": ["counts.json"]},
        "not_checked": ["E 레코드", "F 격리·귀속·회계", "A 게이트 재검", "B 조합", "D 정상"],
        "coverage": "partial",
        "ok": False,
        "verdict": "불일치 — 계약 구성원이 없거나 비어 대조를 이어갈 수 없다",
    }
    # 검사 1의 합성 보고를 분리해 보기 위한 대역이다. 실물 빌드의 두 검사 동시 실행은 아니다.
    independent = {
        "mismatch": {}, "blocking": {}, "not_compared": [],
        "coverage": "full", "verdict": "일치", "sealable": True,
        "meta_note": None, "counts_note": None,
        "n_expected_keep": 4, "n_expected_discard": 1, "n_got": 4,
    }
    replies = iter([(0, verify, ""), (0, independent, "")])
    checker = tmp_path / "pairs_independent.py"
    checker.write_text("# 대역\n", encoding="utf-8")
    monkeypatch.setattr(S, "INDEPENDENT", checker)
    monkeypatch.setattr(S, "_run", lambda argv: next(replies))
    report = S.check(tmp_path / "build", tmp_path / "snap", tmp_path, tmp_path / "out")
    assert report["봉인_가능"] is False
    assert report["판정"] == "불통과"
    assert report["검사"][0]["상태"] == S.BROKEN
    assert report["실행실패_검사"] == ["verify_pairs_main"]
    assert report["범위제한_있음"] is True
    assert report["범위제한_검사"] == ["verify_pairs_main"]
