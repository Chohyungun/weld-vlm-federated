"""감사 대조선은 eval 정답을 적합에 쓰지 않으며 같은 출처의 교락을 드러낸다."""

import csv

import pytest

from evaluation import content_free, discrimination_baseline as baseline


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    rows = [
        dict(image_id="train_a", split="train", iso_codes="100", has_defect="True"),
        dict(image_id="val_b", split="val", iso_codes="", has_defect="False"),
        dict(image_id="eval_a", split="eval", iso_codes="100", has_defect="True"),
        dict(image_id="eval_b", split="eval", iso_codes="", has_defect="False"),
        dict(image_id="tile", split="eval", iso_codes="", has_defect="False"),
    ]
    monkeypatch.setattr(baseline, "read_manifest", lambda path: rows)
    monkeypatch.setattr(baseline, "ensure_verified", lambda path: "fixture-digest")
    bins = {"train_a": 0, "val_b": 1, "eval_a": 0, "eval_b": 1}
    monkeypatch.setattr(content_free, "bins_for", lambda ids, k, path: {i: bins[i] for i in ids})
    with (tmp_path / "tiles.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "provenance"])
        writer.writeheader()
        writer.writerows({"image_id": r["image_id"],
                          "provenance": "N-tile" if r["image_id"] == "tile" else "N-crop"}
                         for r in rows)
    return tmp_path, rows


def test_fit_excludes_eval_and_positive_metadata_counterexample(snapshot, monkeypatch):
    path, _ = snapshot
    real_fit = baseline.fit_idq

    def audited_fit(k, ids, gold, classes, **kwargs):
        assert set(ids) == set(gold) == {"train_a", "val_b"}
        assert k == 512
        return real_fit(k, ids, gold, classes, **kwargs)

    monkeypatch.setattr(baseline, "fit_idq", audited_fit)
    result = baseline.compute_metadata_baseline(path, classes=["100"])
    assert result["delta_auc"] == 1.0
    assert result["n_fit"] == 2
    assert result["n_defect"] == result["n_normal"] == 1
    assert result["image_pixels_read"] is False
    assert result["status"].startswith("post_hoc")


@pytest.mark.parametrize("ids", [{"train_a"}, {"tile"}, {"missing"}])
def test_population_mismatch_is_rejected(snapshot, ids):
    with pytest.raises(ValueError, match="eval/N-crop"):
        baseline.compute_metadata_baseline(snapshot[0], image_ids=ids)


def test_empty_population_is_undefined_not_zero(snapshot):
    result = baseline.compute_metadata_baseline(snapshot[0], image_ids=[])
    assert result["delta_auc"] is None
    assert result["n_defect"] == result["n_normal"] == 0


@pytest.mark.parametrize("complete", [False, True])
def test_scoring_compares_model_and_baseline_on_same_population(tmp_path, monkeypatch, complete):
    from types import SimpleNamespace

    from evaluation.schema import PredictionRecord
    from scripts.probe import score_cells

    rows = [dict(image_id="d", group_id="g", has_defect="True"),
            dict(image_id="n", group_id="g", has_defect="False")]
    pop = SimpleNamespace(rows=rows, classes=["2011"])
    raw = tmp_path / "raw.jsonl"
    ids = ["d", "n"] if complete else ["d"]
    records = [PredictionRecord(
        schema_version="1.3", image_id=i, cell="sep_central", seed=1,
        defects=[], verdict="판정불가", cited_clauses=[], parse_ok=True,
    ) for i in ids]
    raw.write_text("\n".join(r.model_dump_json() for r in records), encoding="utf-8")
    monkeypatch.setattr(score_cells, "raw_record_path", lambda params, tag: raw)
    captured = []

    def fake_baseline(path, *, image_ids, classes):
        captured.append((path, image_ids, classes))
        return {"delta_auc": 0.5, "status": "post_hoc_fixture"}

    monkeypatch.setattr(baseline, "compute_metadata_baseline", fake_baseline)
    params = SimpleNamespace(snapshot=tmp_path)
    provenance = {"d": "N-crop", "n": "N-crop"}
    if not complete:
        with pytest.raises(ValueError, match="다른 모집단"):
            score_cells.threshold_free_block(params, pop, provenance, ["central"])
        assert captured == []
    else:
        result = score_cells.threshold_free_block(params, pop, provenance, ["central"])
        assert captured == [(tmp_path, {"d", "n"}, ["2011"])]
        assert result["metadata_baseline"]["delta_auc"] == 0.5
        assert result["n_defect"] == result["n_normal"] == 1


def test_scoring_keeps_constant_baseline_separate_from_shortcut_immunity():
    from types import SimpleNamespace
    from scripts.probe.score_cells import discrimination_block

    result = discrimination_block(SimpleNamespace(rows=[]), {}, {})
    assert "상수 예측기" in result["baseline"]
    assert "다른 메타데이터 교락" in result["limitation"]
