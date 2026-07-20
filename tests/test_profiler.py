"""Tests for DataProfiler."""

import numpy as np
import pandas as pd
import pytest

from hypolab.profiler import DataProfiler


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.normal(30, 10, 100),
        "income": np.random.normal(50000, 15000, 100),
        "score": np.random.normal(75, 15, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "region": np.random.choice(["East", "West"], 100),
    })


def test_profiler_initialization(sample_df):
    profiler = DataProfiler(sample_df)
    assert profiler.df.shape == sample_df.shape
    assert profiler.target_col is None


def test_infer_types(sample_df):
    profiler = DataProfiler(sample_df)
    profiler._infer_types()
    assert set(profiler.numeric_cols) == {"age", "income", "score"}
    assert set(profiler.categorical_cols) == {"category", "region"}
    assert profiler.datetime_cols == []


def test_compute_numeric_stats(sample_df):
    profiler = DataProfiler(sample_df)
    profiler.run()
    assert "numeric" in profiler.profile
    assert "age" in profiler.profile["numeric"]
    assert "skewness" in profiler.profile["numeric"]["age"]


def test_compute_correlations(sample_df):
    profiler = DataProfiler(sample_df)
    profiler.run()
    corrs = profiler.profile["correlations"]
    assert isinstance(corrs, list)
    assert all("pearson_r" in c for c in corrs)


def test_detect_issues(sample_df):
    df = sample_df.copy()
    df.loc[0:4, "age"] = np.nan
    profiler = DataProfiler(df)
    profiler.run()
    issues = profiler.profile["issues"]
    missing_issues = [i for i in issues if i["issue"] == "missing_values" and i["column"] == "age"]
    assert len(missing_issues) == 1
    assert missing_issues[0]["count"] == 5


def test_get_prompt_context(sample_df):
    profiler = DataProfiler(sample_df)
    ctx = profiler.get_prompt_context()
    assert "Dataset shape" in ctx
    assert "age" in ctx


def test_to_json(sample_df):
    profiler = DataProfiler(sample_df)
    json_str = profiler.to_json()
    assert "age" in json_str
    assert "correlations" in json_str