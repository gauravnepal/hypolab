"""Tests for StatisticalTester."""

import numpy as np
import pandas as pd
import pytest

from hypolab.statistical_tester import StatisticalTester


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.normal(30, 10, 100),
        "income": np.random.normal(50000, 15000, 100),
        "score": np.random.normal(75, 15, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "binary": np.random.choice(["X", "Y"], 100),
    })


def test_pearson_correlation(sample_df):
    tester = StatisticalTester(sample_df)
    hyp = {"hypothesis": "H0: r=0", "test_type": "pearson_correlation", "variables": ["age", "income"]}
    results = tester.run_all([hyp])
    assert results[0]["status"] == "completed"
    assert "pearson_r" in results[0]
    assert "p_value" in results[0]


def test_anova(sample_df):
    tester = StatisticalTester(sample_df)
    hyp = {"hypothesis": "H0: means equal", "test_type": "anova", "variables": ["age", "category"]}
    results = tester.run_all([hyp])
    assert results[0]["status"] == "completed"
    assert "f_statistic" in results[0]


def test_chi_square(sample_df):
    tester = StatisticalTester(sample_df)
    hyp = {"hypothesis": "H0: independent", "test_type": "chi_square", "variables": ["category", "binary"]}
    results = tester.run_all([hyp])
    assert results[0]["status"] == "completed"
    assert "chi2" in results[0]


def test_t_test_numeric_vs_binary(sample_df):
    tester = StatisticalTester(sample_df)
    hyp = {"hypothesis": "H0: means equal", "test_type": "t_test", "variables": ["age", "binary"]}
    results = tester.run_all([hyp])
    assert results[0]["status"] == "completed"
    assert "t_statistic" in results[0]


def test_t_test_two_numeric(sample_df):
    tester = StatisticalTester(sample_df)
    hyp = {"hypothesis": "H0: means equal", "test_type": "t_test", "variables": ["age", "income"]}
    results = tester.run_all([hyp])
    assert results[0]["status"] == "completed"


def test_regression(sample_df):
    tester = StatisticalTester(sample_df)
    hyp = {"hypothesis": "H0: no relationship", "test_type": "regression", "variables": ["age", "income", "score"]}
    results = tester.run_all([hyp])
    assert results[0]["status"] == "completed"
    assert "r_squared" in results[0]
    assert "coefficients" in results[0]


def test_granger_insufficient_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    tester = StatisticalTester(df)
    hyp = {"hypothesis": "H0: no granger", "test_type": "granger_causality", "variables": ["a", "b"]}
    results = tester.run_all([hyp])
    assert results[0]["status"] == "skipped"


def test_unknown_test_type(sample_df):
    tester = StatisticalTester(sample_df)
    hyp = {"hypothesis": "H0: ???", "test_type": "magic_test", "variables": ["age"]}
    results = tester.run_all([hyp])
    assert results[0]["status"] == "skipped"


def test_summary(sample_df):
    tester = StatisticalTester(sample_df)
    hyps = [
        {"hypothesis": "H1", "test_type": "pearson_correlation", "variables": ["age", "income"]},
        {"hypothesis": "H2", "test_type": "anova", "variables": ["age", "category"]},
    ]
    tester.run_all(hyps)
    summary = tester.get_summary()
    assert summary["total_hypotheses"] == 2
    assert summary["completed"] == 2
    assert "significant" in summary