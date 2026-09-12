"""Check pilot sampling and diagnostics independently of network downloads."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def pilot(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "coverage_pilot", directory / "benchmark_feature_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_sample_and_tail_cap(pilot, monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "M4Info", {"Yearly": SimpleNamespace(n_ts=4)})
    path = tmp_path / "Yearly-train.csv"
    path.write_text("V1,V2,V3,V4,V5\nY1,1,2,3,4\nY2,5,6,,\nY3,7,8,9,\nY4,10,11,12,13\n")
    series, lengths = pilot.sample_training(path, "Yearly", 4, 9, 2)
    assert lengths == {"Y1": 4, "Y2": 2, "Y3": 3, "Y4": 4}
    assert {uid: values.tolist() for uid, values in series.items()} == {
        "Y1": [3.0, 4.0],
        "Y2": [5.0, 6.0],
        "Y3": [8.0, 9.0],
        "Y4": [12.0, 13.0],
    }
    first, _ = pilot.sample_training(path, "Yearly", 2, 17, 4)
    second, _ = pilot.sample_training(path, "Yearly", 2, 17, 4)
    assert list(first) == list(second)
    with pytest.raises(ValueError, match="between"):
        pilot.sample_training(path, "Yearly", 5, 0, 3)
    path.write_text("V1,V2,V3,V4\nY1,1,,3\nY2,1,2,3\nY3,1,2,3\nY4,1,2,3\n")
    with pytest.raises(ValueError, match="Internal missing"):
        pilot.sample_training(path, "Yearly", 4, 9, 2)


def test_retention_reports_short_series_instead_of_hiding_them(pilot):
    features = pl.DataFrame(
        {
            "unique_id": ["short", "long1", "long2", "long3"],
            **{
                name: [np.nan if name == "lumpiness" else 0.0, 1.0, 2.0, 3.0]
                for name in pilot.FEATURE_NAMES
            },
        }
    )
    report = pilot.retention(
        features, {"short": 10, "long1": 40, "long2": 50, "long3": 60}
    )
    assert report["fraction"] == 0.75
    assert report["shortest_quartile_retention"] == 0.0
    assert report["dropped_ids"] == ["short"]
    assert report["undefined_per_feature"]["lumpiness"] == 1


def test_artifacts_use_valid_json_for_undefined_features(pilot, tmp_path):
    path = tmp_path / "result.json"
    pilot.save_json(
        path, {"values": np.array([np.inf, np.nan, 1.0]), "passed": np.bool_(True)}
    )
    assert json.loads(path.read_text()) == {"values": [None, None, 1.0], "passed": True}
    assert "NaN" not in path.read_text()


def test_control_series_share_the_same_baseline(pilot):
    base, variants = pilot.controlled_series(5, draws=2, length=48)
    again, _ = pilot.controlled_series(5, draws=2, length=48)
    for uid, values in base.items():
        np.testing.assert_array_equal(values, again[uid])
        np.testing.assert_array_equal(variants["level_shift"][uid][:24], values[:24])
        np.testing.assert_allclose(variants["level_shift"][uid][24:], values[24:] + 5)
        np.testing.assert_allclose(
            variants["variance_shift"][uid][24:], values[24:] * 5
        )
        assert np.count_nonzero(variants["spike"][uid] != values) == 1


def test_generation_matches_reference_lengths_and_records_sources(pilot):
    real = {"a": np.arange(24.0), "b": np.arange(31.0)}
    corpora, provenance, failures, configs = pilot.generate_matched(real, 12, "MS", 123)
    assert set(corpora) == {"balanced_pool", "synforecast_mar"}
    for name, values in corpora.items():
        assert not failures[name]
        assert {uid: len(x) for uid, x in values.items()} == {"a": 24, "b": 31}
        assert set(provenance[name]) == set(real)
    assert configs["synforecast_mar"]["seasonal_period"] == 12


def test_availability_reports_seasonal_short_tail(pilot):
    values = np.arange(93.0)
    short = pilot.compute_feature_set(values, 52, 52)
    long = pilot.compute_feature_set(np.arange(120.0), 52, 52)
    report = pilot.availability_summary(
        ["short", "long"],
        [93, 120],
        np.array(
            [[row[name] for name in pilot.FEATURE_NAMES] for row in (short, long)]
        ),
    )
    assert report["n_retained"] == 1
    assert report["shortest_quartile_retention"] == 0
    assert report["dropped_examples_first_10"][0]["undefined_features"] == [
        "seasonal_strength",
        "lumpiness",
        "max_level_shift",
        "max_var_shift",
    ]
    assert not report["passes_pilot_retention_criteria"]
    assert np.isfinite(list(pilot.compute_feature_set(values, None, 10).values())).all()


def test_availability_reads_all_rows_and_records_cap(pilot, monkeypatch, tmp_path):
    monkeypatch.setattr(
        pilot, "M4Info", {"Quarterly": SimpleNamespace(n_ts=2, seasonality=4)}
    )
    path = tmp_path / "Quarterly-train.csv"
    path.write_text(
        "id,observations\nQ1,"
        + ",".join(map(str, range(12)))
        + ",,,,\nQ2,"
        + ",".join(map(str, range(20)))
        + "\n"
    )
    args = SimpleNamespace(cache=tmp_path, download=False, max_length=16)
    report = pilot.audit_availability(args, "Quarterly")
    assert report["n_series"] == report["recipes"]["primary"]["n_retained"] == 2
    assert report["n_capped"] == 1
    assert report["effective_length_quantiles_min_p25_p50_p75_max"][-1] == 16
    assert report["source"]["split"] == "train"
    path.write_text("id,x,y,z\nQ1,1,,3\nQ2,1,2,3\n")
    with pytest.raises(ValueError, match="internally missing"):
        pilot.audit_availability(args, "Quarterly")


def test_availability_cli_uses_all_groups_and_writes_only_summary(
    pilot, monkeypatch, tmp_path
):
    groups = []

    def audit(args, group):
        assert args.n_series is None
        groups.append(group)
        return {"n_series": 3}

    monkeypatch.setattr(pilot, "audit_availability", audit)
    monkeypatch.setattr(pilot, "environment_metadata", lambda: {})
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--availability-only", "--output", str(tmp_path)]
    )
    pilot.main()
    assert groups == list(pilot.AVAILABILITY_RECIPES)
    assert [p.name for p in tmp_path.iterdir()] == ["summary.json"]
    monkeypatch.setattr(sys, "argv", ["pilot", "--availability-only", "--quick"])
    with pytest.raises(SystemExit) as exc:
        pilot.main()
    assert exc.value.code == 2


def test_matched_cohort_excludes_failed_draws_from_every_source(pilot):
    frames = {
        name: pl.DataFrame(
            {
                "unique_id": ["a", "b", "c", "d"],
                **{feature: [1.0, 2.0, 3.0, 4.0] for feature in pilot.FEATURE_NAMES},
            }
        )
        for name in ("real", "balanced_pool", "pretraining_pool", "synforecast_mar")
    }
    frames["pretraining_pool"] = frames["pretraining_pool"].with_columns(
        pl.when(pl.col("unique_id") == "c")
        .then(float("nan"))
        .otherwise(pl.col("acf1"))
        .alias("acf1")
    )
    assert pilot.matched_complete_ids(frames) == ["a", "b", "d"]
    with pytest.raises(ValueError, match="two shared"):
        pilot.matched_complete_ids({"real": frames["real"].head(1)})


def test_three_source_generation_uses_independent_seeds_and_exact_lengths(
    pilot, monkeypatch
):
    prototype = pilot.MARGenerator(
        min_length=24, max_length=24, seasonal_period=4, freq="MS", engine="polars"
    )
    monkeypatch.setattr(pilot, "interpretable_pool", lambda **_kwargs: [prototype])
    monkeypatch.setattr(pilot, "pretraining_pool", lambda **_kwargs: [prototype])
    reference = {"short": np.arange(24.0), "long": np.arange(31.0)}
    corpora, provenance, failures, configs = pilot.generate_matched(
        reference, 4, "MS", 123, include_pretraining=True
    )
    assert set(corpora) == {"balanced_pool", "pretraining_pool", "synforecast_mar"}
    for name in corpora:
        assert not failures[name]
        assert {uid: len(x) for uid, x in corpora[name].items()} == {
            "short": 24,
            "long": 31,
        }
        assert set(provenance[name]) == set(reference)
    assert not np.array_equal(
        corpora["balanced_pool"]["short"], corpora["pretraining_pool"]["short"]
    )
    assert "pretraining_pool" in configs
