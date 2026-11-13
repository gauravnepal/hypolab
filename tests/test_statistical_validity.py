"""Checks that the verification layer is statistically sound, not just that it runs."""

import numpy as np
import pandas as pd
import pytest

from hypolab.statistical_tester import StatisticalTester, benjamini_hochberg


def _pearson_hyp(a, b):
    return {
        "hypothesis": f"{a}~{b}",
        "test_type": "pearson_correlation",
        "variables": [a, b],
    }


# --- Pearson pairs values from the same row -------------------------------


def test_pearson_keeps_rows_paired_when_values_are_missing():
    """Missing values in different rows must not shift x against y."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "y": 3 * x + rng.normal(scale=0.01, size=200)})
    df.loc[rng.choice(200, 30, replace=False), "x"] = np.nan
    df.loc[rng.choice(200, 30, replace=False), "y"] = np.nan

    result = StatisticalTester(df).run_all([_pearson_hyp("x", "y")])[0]

    assert result["status"] == "completed"
    assert result["pearson_r"] > 0.99
    assert result["n"] == len(df.dropna())
    assert result["significant"] is True


def test_pearson_reports_confidence_interval_around_r():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    df = pd.DataFrame({"x": x, "y": 0.5 * x + rng.normal(size=500)})

    result = StatisticalTester(df).run_all([_pearson_hyp("x", "y")])[0]

    low, high = result["ci_95"]
    assert low < result["pearson_r"] < high
    assert low > 0


def test_pearson_skips_constant_column():
    df = pd.DataFrame({"x": [1.0] * 20, "y": np.arange(20.0)})
    result = StatisticalTester(df).run_all([_pearson_hyp("x", "y")])[0]
    assert result["status"] == "skipped"


# --- Benjamini-Hochberg ---------------------------------------------------


def test_benjamini_hochberg_matches_statsmodels():
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(2)
    p_values = np.concatenate([rng.uniform(size=30), rng.uniform(0, 0.01, size=10)])

    expected = multitest.multipletests(p_values, method="fdr_bh")[1]

    assert np.allclose(benjamini_hochberg(p_values.tolist()), expected)


def test_benjamini_hochberg_handles_empty_and_single():
    assert benjamini_hochberg([]) == []
    assert benjamini_hochberg([0.03]) == [0.03]


def test_weak_result_survives_alone_but_not_among_many_tests():
    """The same p-value means less when it is one of many tests."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(size=(300, 8)), columns=list("abcdefgh"))
    # Give a~b a weak relationship that lands near p = 0.02 on its own.
    df["b"] = 0.14 * df["a"] + rng.normal(size=300)
    alone = StatisticalTester(df).run_all([_pearson_hyp("a", "b")])[0]
    assert 0.001 < alone["p_value"] < 0.05, "fixture should give a borderline p"
    assert alone["significant"] is True

    pairs = [("a", "b")] + [(x, y) for x in "cdefgh" for y in "cdefgh" if x < y]
    among_many = StatisticalTester(df).run_all([_pearson_hyp(x, y) for x, y in pairs])[
        0
    ]
    assert among_many["significant_uncorrected"] is True
    assert among_many["q_value"] > among_many["p_value"]
    assert among_many["significant"] is False


def test_correction_spans_separate_run_all_calls():
    """Testing in several batches must not avoid the correction."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    tester = StatisticalTester(df)

    tester.run_all([_pearson_hyp("a", "b")])
    tester.run_all([_pearson_hyp("c", "d"), _pearson_hyp("e", "f")])

    assert {r["n_tests_corrected"] for r in tester.results} == {3}


def test_correction_can_be_turned_off():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
    result = StatisticalTester(df, correction="none").run_all([_pearson_hyp("a", "b")])[
        0
    ]
    assert result["q_value"] is None
    assert result["significant"] == result["significant_uncorrected"]


def test_unknown_correction_is_rejected():
    with pytest.raises(ValueError):
        StatisticalTester(pd.DataFrame({"a": [1]}), correction="bonferoni")


# --- False discoveries on pure noise --------------------------------------


def _noise_run(seed, correction):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(300, 7)), columns=list("abcdefg"))
    pairs = [(x, y) for x in df.columns for y in df.columns if x < y][:20]
    tester = StatisticalTester(df, correction=correction)
    tester.run_all([_pearson_hyp(x, y) for x, y in pairs])
    return tester.get_summary()["significant"]


def test_pure_noise_rarely_produces_a_finding():
    """With 20 tests on noise, at most ~5% of runs may report anything."""
    runs = 300
    with_any_finding = sum(_noise_run(seed, "bh") > 0 for seed in range(runs))
    # Expected rate is 5%; 10% leaves room for sampling error over 300 runs.
    assert with_any_finding / runs < 0.10


def test_pure_noise_without_correction_shows_why_it_is_needed():
    runs = 300
    with_any_finding = sum(_noise_run(seed, "none") > 0 for seed in range(runs))
    assert with_any_finding / runs > 0.40


# --- Effect sizes, shared keys, and guards --------------------------------


@pytest.fixture
def mixed_df():
    rng = np.random.default_rng(6)
    n = 400
    group = rng.choice(["A", "B", "C"], n)
    binary = rng.choice(["X", "Y"], n)
    score = (
        pd.Series(group).map({"A": 0.0, "B": 0.5, "C": 1.0}).to_numpy()
        + (binary == "Y") * 0.6
        + rng.normal(size=n)
    )
    return pd.DataFrame(
        {
            "score": score,
            "other": rng.normal(size=n),
            "group": group,
            "binary": binary,
        }
    )


def test_every_completed_test_reports_p_value_and_effect_size(mixed_df):
    hyps = [
        {"hypothesis": "h", "test_type": "anova", "variables": ["score", "group"]},
        {"hypothesis": "h", "test_type": "t_test", "variables": ["score", "binary"]},
        {
            "hypothesis": "h",
            "test_type": "chi_square",
            "variables": ["group", "binary"],
        },
        {"hypothesis": "h", "test_type": "regression", "variables": ["score", "other"]},
        _pearson_hyp("score", "other"),
    ]
    results = StatisticalTester(mixed_df).run_all(hyps)

    for result in results:
        assert result["status"] == "completed", result
        assert 0 <= result["p_value"] <= 1
        assert 0 <= result["q_value"] <= 1
        assert result["effect_size_name"]
        assert result["effect_size"] is not None
        assert isinstance(result["significant"], bool)

    by_type = {r["test_type"]: r for r in results}
    assert 0 < by_type["anova"]["eta_squared"] < 1
    assert 0 <= by_type["chi_square"]["cramers_v"] <= 1
    assert by_type["t_test"]["cohens_d"] != 0
    assert set(by_type["t_test"]["group_sizes"]) == {"X", "Y"}


def test_chi_square_warns_on_small_expected_counts():
    df = pd.DataFrame(
        {
            "a": ["x"] * 6 + ["y"] * 3 + ["z"] * 2,
            "b": ["p", "q"] * 5 + ["p"],
        }
    )
    result = StatisticalTester(df).run_all(
        [{"hypothesis": "h", "test_type": "chi_square", "variables": ["a", "b"]}]
    )[0]
    assert result["status"] == "completed"
    assert any("expected count" in note for note in result["warnings"])


def test_unknown_column_is_skipped_with_a_clear_reason(mixed_df):
    """An LLM can propose a column that is not in the data."""
    result = StatisticalTester(mixed_df).run_all([_pearson_hyp("score", "revenue")])[0]
    assert result["status"] == "skipped"
    assert "revenue" in result["reason"]
    assert result["test_type"] == "pearson_correlation"


def test_granger_detects_a_lagged_relationship():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    y = 0.8 * np.roll(x, 1) + rng.normal(scale=0.5, size=300)
    df = pd.DataFrame({"y": y[1:], "x": x[1:]})

    result = StatisticalTester(df).run_all(
        [
            {
                "hypothesis": "x leads y",
                "test_type": "granger_causality",
                "variables": ["y", "x"],
            }
        ]
    )[0]

    assert result["status"] == "completed", result
    assert result["significant"] is True
    # The reported p-value is corrected for the number of lags tried.
    assert result["p_value"] >= result["min_p_value"]
    assert result["lags_tested"] == 4


def test_granger_flags_non_stationary_series():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(8)
    df = pd.DataFrame(
        {
            "walk_a": rng.normal(size=300).cumsum(),
            "walk_b": rng.normal(size=300).cumsum(),
        }
    )
    result = StatisticalTester(df).run_all(
        [
            {
                "hypothesis": "h",
                "test_type": "granger_causality",
                "variables": ["walk_a", "walk_b"],
            }
        ]
    )[0]
    assert result["status"] == "completed", result
    assert any("non-stationary" in note for note in result["warnings"])
