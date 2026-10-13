"""평가 층화 표본(data.eval_sample) — D 25번 §2 명세. 가짜 매니페스트 행만 쓴다(실데이터 · 영상 없음).

고정하는 것: 층(정수 비교) · 배분 식 · 순서 열쇠 · 결정성(입력 행 순서와 무관) · 포함 관계 · 묶음 불가분 · 평가셋 밖 id 0 · 산출 꼴 · 거부.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict

import pytest

from data import eval_sample as E
from evaluation.eval_list import set_digest, validate_eval_list


def _row(iid, gid, material, n, split="eval"):
    return {"image_id": iid, "split": split, "group_id": gid, "material": material, "n_defects": str(n)}


def population(n_groups=300, *, with_trainval=True):
    """묶음 크기 1 ~ 9, 묶음마다 재질 하나, 정상 · 결함이 섞인 묶음이 있다(결정적)."""
    rows = []
    for g in range(n_groups):
        material = "AL" if g % 6 == 0 else "ST"
        size = 1 + (g * 5) % 9
        for i in range(size):
            n = 0 if g % 4 == 0 or (g + i) % 5 == 0 else 1 + (g * 3 + i) % 6
            rows.append(_row(f"img-{g:04d}-{i}", f"grp-{g:04d}", material, n))
    if with_trainval:
        rows += [_row(f"tr-{k}", f"grp-tr-{k}", "ST", k % 3, split="train") for k in range(40)]
        rows += [_row(f"va-{k}", "grp-0001", "ST", 0, split="val") for k in range(5)]      # 같은 묶음 이름이어도 평가셋 밖
    return rows


@pytest.fixture(scope="module")
def pop():
    return population()


@pytest.mark.parametrize("total,n,band", [(0, 1, "0"), (0, 7, "0"), (1, 1, "1"), (7, 5, "1"), (3, 2, "2-3"),
                                          (2, 1, "2-3"), (17, 5, "2-3"), (7, 2, "4+"), (9, 1, "4+")])
def test_묶음_층은_평균_결함_수의_정수_비교(total, n, band):
    assert E.group_band(total, n) == band          # 1.5 는 2-3, 3.5 는 4+ — 경계는 정수 비교로만


@pytest.mark.parametrize("g,size,pop_n,n_h", [(81, 3000, 12461, 20), (9, 2000, 12461, 2), (1, 2000, 12461, 1),
                                              (400, 4000, 12461, 128), (40, 12461, 12461, 40), (40, 20000, 12461, 40),
                                              (5, 1, 10, 2)])
def test_배분은_묶음_수_고정_최소_둘_층_상한(g, size, pop_n, n_h):
    assert E.allocate(g, size, pop_n) == n_h


def test_순서_열쇠는_시드_구분자_그룹의_sha256_소문자():
    key = E.rank_key(E.SEED, "grp_ST_0a")
    assert re.fullmatch(r"[0-9a-f]{64}", key[0]) and key[1] == "grp_ST_0a"
    assert key[0] == hashlib.sha256(f"{E.SEED}|grp_ST_0a".encode()).hexdigest()


def test_결정적이고_입력_행_순서와_무관하다(pop):
    a = E.sample_eval(pop, (200, 300))
    shuffled = list(pop)
    random.Random(5).shuffle(shuffled)
    assert E.sample_eval(shuffled, (200, 300)) == a
    assert E.sample_eval(pop, (300,), seed="다른 시드")["samples"][300]["ids"] != a["samples"][300]["ids"]


def test_크기가_커지면_앞의_표본을_품는다(pop):
    res = E.sample_eval(pop, (150, 250, 400))
    s = [set(res["samples"][k]["ids"]) for k in (150, 250, 400)]
    assert s[0] <= s[1] <= s[2]


def test_묶음은_나뉘지_않고_평가셋_밖_id_는_없다(pop):
    res = E.sample_eval(pop, (300,))["samples"][300]
    eval_by_group = defaultdict(set)
    for r in pop:
        if r["split"] == "eval":
            eval_by_group[r["group_id"]].add(r["image_id"])
    ids = set(res["ids"])
    assert ids <= set().union(*eval_by_group.values())
    touched = {g for g, members in eval_by_group.items() if members & ids}
    assert all(eval_by_group[g] <= ids for g in touched) and len(touched) == res["groups"]


def test_층별_묶음_수는_배분_식대로(pop):
    res = E.sample_eval(pop, (300,))
    n = res["population"]["eval_images"]
    for s, v in res["samples"][300]["strata"].items():
        assert v["n_h"] == E.allocate(res["strata"][s]["G_h"], 300, n)
    assert res["samples"][300]["groups"] == sum(v["n_h"] for v in res["samples"][300]["strata"].values())


@pytest.mark.parametrize("rows,code", [
    ([_row("a", "g", "ST", 0, split="train")], "empty"),
    ([_row("a", "g", "ST", 0), _row("a", "h", "ST", 0)], "duplicate_id"),
    ([_row("a", "g", "ST", 0), _row("b", "g", "AL", 0)], "mixed_material"),
    ([_row("a", "g", "ST", "x")], "n_defects"),
    ([_row("a", "g", "ST", -1)], "n_defects"),
    ([_row("a", "", "ST", 0)], "empty_field"),
    ([{k: v for k, v in _row("a", "g", "ST", 0).items() if k != "material"}], "columns"),
])
def test_거부(rows, code):
    with pytest.raises(E.SampleRefused) as err:
        E.sample_eval(rows, (1,))
    assert err.value.code == code


def test_산출은_정규형_목록_곁_파일_지문이고_덮어쓰지_않는다(tmp_path, pop):
    res = E.sample_eval(pop, (200, 300))
    meta = E.write_samples(res, tmp_path / "out", "d" * 64)
    for size in (200, 300):
        m = meta["sizes"][str(size)]
        raw = (tmp_path / "out" / m["list_file"]).read_bytes()
        checked = validate_eval_list(raw)
        ids = list(checked.ids)
        assert ids == sorted(ids) == res["samples"][size]["ids"]                         # 코드포인트 오름차순
        assert m["file_sha256"] == hashlib.sha256(raw).hexdigest() and m["set_sha256"] == set_digest(ids)
        side = (tmp_path / "out" / m["side_file"]).read_bytes()
        assert m["side_sha256"] == hashlib.sha256(side).hexdigest() and b"\r" not in side and side.endswith(b"\n")
        lines = side.decode("utf-8").splitlines()
        assert lines[0] == "image_id,stratum,G_h,n_h" and [x.split(",")[0] for x in lines[1:]] == ids
    assert json.loads((tmp_path / "out" / "SAMPLE.json").read_text(encoding="utf-8"))["seed"] == E.SEED
    with pytest.raises(FileExistsError):
        E.write_samples(res, tmp_path / "out", "d" * 64)


def _write_manifest(path, rows):
    cols = list(E.COLUMNS) + ["has_defect", "rel_path"]                                  # 쓰지 않는 열은 버린다
    lines = [",".join(cols)] + [",".join([r[c] for c in E.COLUMNS] + ["?", "x.jpg"]) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_명령행은_스냅샷을_대조하고_새_폴더에_쓴다(tmp_path, monkeypatch, pop):
    mdir = tmp_path / "manifest"
    mdir.mkdir()
    _write_manifest(mdir / "manifest.csv", pop)
    import data.manifest_io as M

    monkeypatch.setattr(M, "verify_snapshot", lambda root: "d" * 64)
    args = ["--manifest-dir", str(mdir), "--expect-digest", "d" * 64, "--out", str(tmp_path / "s"), "--sizes", "200", "300"]
    assert E.main(args) == 0
    meta = json.loads((tmp_path / "s" / "SAMPLE.json").read_text(encoding="utf-8"))
    assert meta["sizes"]["300"]["set_sha256"] == set_digest(E.sample_eval(pop, (300,))["samples"][300]["ids"])
    with pytest.raises(E.SampleRefused) as err:
        E.main([*args[:3], "e" * 64, args[4], str(tmp_path / "t"), *args[6:]])
    assert err.value.code == "snapshot"
