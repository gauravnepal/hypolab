"""Tests for HypothesisAgent."""

import json
import pytest

from hypolab.hypothesis_agent import HypothesisAgent
from hypolab.config import HypoLabConfig


@pytest.fixture
def sample_profile():
    return json.dumps({
        "schema_summary": "Dataset shape: 100 rows x 5 columns\nNumeric columns: age, income, score\nCategorical columns: category, region",
        "correlations": [
            {"col_a": "age", "col_b": "income", "pearson_r": 0.45},
        ],
    })


def test_agent_initialization_mock():
    """Agent should initialize even without API keys (mock mode)."""
    config = HypoLabConfig(groq_api_key="", use_local_model=False)
    agent = HypothesisAgent(config)
    assert agent.config is not None


def test_generate_returns_list(sample_profile):
    config = HypoLabConfig(groq_api_key="", use_local_model=False)
    agent = HypothesisAgent(config)
    hypotheses = agent.generate(sample_profile)
    assert isinstance(hypotheses, list)
    assert len(hypotheses) > 0


def test_hypothesis_structure(sample_profile):
    config = HypoLabConfig(groq_api_key="", use_local_model=False)
    agent = HypothesisAgent(config)
    hypotheses = agent.generate(sample_profile)
    for h in hypotheses:
        assert "hypothesis" in h
        assert "test_type" in h
        assert "variables" in h
        assert "rationale" in h


def test_smart_data_driven_finds_columns():
    """Smart analysis should extract real column names from profile."""
    profile = json.dumps({
        "numeric": {"age": {"mean": 30}, "income": {"mean": 50000}},
        "categorical": {"department": {"unique_count": 3}},
        "correlations": [{"col_a": "age", "col_b": "income", "pearson_r": 0.45}],
        "schema_summary": "Numeric columns: age, income\nCategorical columns: department"
    })
    config = HypoLabConfig(groq_api_key="", use_local_model=False)
    agent = HypothesisAgent(config)
    hyps = agent.generate(profile)
    
    assert len(hyps) > 0
    # Should use actual column names, not generic placeholders
    all_vars = [v for h in hyps for v in h.get("variables", [])]
    assert "age" in all_vars or "income" in all_vars


def test_parse_valid_json():
    config = HypoLabConfig(groq_api_key="", use_local_model=False)
    agent = HypothesisAgent(config)
    test_json = '[{"hypothesis": "H0: x=0", "test_type": "t_test", "variables": ["a","b"], "rationale": "test"}]'
    parsed = agent._parse_response(test_json)
    assert len(parsed) == 1
    assert parsed[0]["test_type"] == "t_test"


def test_parse_with_markdown_fences():
    config = HypoLabConfig(groq_api_key="", use_local_model=False)
    agent = HypothesisAgent(config)
    fenced = '```json\n[{"hypothesis": "H0", "test_type": "anova", "variables": ["x"], "rationale": "r"}]\n```'
    parsed = agent._parse_response(fenced)
    assert len(parsed) == 1


def test_parse_invalid_returns_empty():
    config = HypoLabConfig(groq_api_key="", use_local_model=False)
    agent = HypothesisAgent(config)
    parsed = agent._parse_response("not json at all")
    assert parsed == []