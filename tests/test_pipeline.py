"""Tests for HypoLabPipeline (integration tests)."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, Mock

from hypolab.pipeline import HypoLabPipeline
from hypolab.config import HypoLabConfig


@pytest.fixture
def sample_df():
    """Generate reproducible sample data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.normal(30, 10, 100),
            "income": np.random.normal(50000, 15000, 100),
            "score": np.random.normal(75, 15, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        }
    )


def test_pipeline_initialization(sample_df):
    """Verify all components are initialized."""
    config = HypoLabConfig(groq_api_key="", use_local_model=False, verbose=False)
    pipe = HypoLabPipeline(sample_df, config=config)
    assert pipe.df is sample_df
    assert pipe.profiler is not None


def test_pipeline_run(sample_df, capsys):
    """Verify full pipeline executes and returns report."""
    config = HypoLabConfig(groq_api_key="", use_local_model=False, verbose=True)
    pipe = HypoLabPipeline(sample_df, config=config)
    report = pipe.run(skip_literature=True)

    assert "profile_summary" in report
    assert "hypotheses" in report
    assert "test_results" in report
    assert "test_summary" in report

    captured = capsys.readouterr()
    assert "Profiling" in captured.out or "Generating" in captured.out


def test_pipeline_report_json(sample_df):
    """Verify JSON export contains expected keys."""
    config = HypoLabConfig(groq_api_key="", use_local_model=False, verbose=False)
    pipe = HypoLabPipeline(sample_df, config=config)
    pipe.run(skip_literature=True)
    json_str = pipe.to_json()
    assert "profile_summary" in json_str
    assert "hypotheses" in json_str


def test_pipeline_report_markdown(sample_df):
    """Verify Markdown export has headers."""
    config = HypoLabConfig(groq_api_key="", use_local_model=False, verbose=False)
    pipe = HypoLabPipeline(sample_df, config=config)
    pipe.run(skip_literature=True)
    md = pipe.to_markdown()
    assert "# HypoLab Report" in md
    assert "Dataset Profile" in md


def test_pipeline_with_target(sample_df):
    """Verify target column is passed through."""
    config = HypoLabConfig(groq_api_key="", use_local_model=False, verbose=False)
    pipe = HypoLabPipeline(sample_df, target_col="score", config=config)
    report = pipe.run(skip_literature=True)
    assert len(report["hypotheses"]) > 0


def test_pipeline_literature_search_mocked(sample_df):
    """Verify literature search integration without real HTTP calls."""
    config = HypoLabConfig(groq_api_key="", use_local_model=False, verbose=False, arxiv_max_results=1)
    pipe = HypoLabPipeline(sample_df, config=config)
    
    with patch("hypolab.literature_search.requests.get") as mock_get:
        mock_get.return_value = Mock(
            text="<feed xmlns='http://www.w3.org/2005/Atom'><entry><title>Test</title><summary>S</summary><author><name>A</name></author><id>http://x</id><published>2024-01-01</published></entry></feed>",
            raise_for_status=Mock(),
        )
        report = pipe.run(skip_literature=False)
        assert "literature" in report
        mock_get.assert_called()
