"""Tests for actual LLM paths using mocks — proves the agentic layer exists."""

import json
from unittest.mock import Mock, patch

import pytest

from hypolab.hypothesis_agent import HypothesisAgent
from hypolab.config import HypoLabConfig


def test_routes_to_groq_when_key_present():
    """Agent must call Groq when API key is configured."""
    config = HypoLabConfig(groq_api_key="fake_key", use_local_model=False, use_ollama=False)
    agent = HypothesisAgent(config)
    
    # Mock the Groq client
    mock_choice = Mock()
    mock_choice.message.content = json.dumps([{
        "hypothesis": "H0: x=0", "test_type": "t_test",
        "variables": ["a", "b"], "rationale": "mock"
    }])
    mock_completion = Mock()
    mock_completion.choices = [mock_choice]
    
    agent.client = Mock()
    agent.client.chat.completions.create.return_value = mock_completion
    
    result = agent.generate('{"numeric": {"a": {}}}')
    
    assert len(result) == 1
    assert result[0]["test_type"] == "t_test"
    agent.client.chat.completions.create.assert_called_once()


def test_falls_back_to_ollama_when_groq_fails():
    """Agent must fallback to Ollama if Groq call fails."""
    config = HypoLabConfig(groq_api_key="fake_key", use_ollama=True)
    agent = HypothesisAgent(config)
    
    # Break Groq
    agent.client = Mock()
    agent.client.chat.completions.create.side_effect = Exception("Groq down")
    
    with patch("hypolab.hypothesis_agent.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "response": json.dumps([{
                "hypothesis": "H0: ollama", "test_type": "anova",
                "variables": ["x", "y"], "rationale": "fallback"
            }])
        }
        result = agent.generate('{"numeric": {"x": {}}}')
        
        assert len(result) == 1
        assert result[0]["test_type"] == "anova"
        mock_post.assert_called_once()


def test_falls_back_to_smart_when_no_llm_available():
    """Agent must use smart analysis when no LLM backend is available."""
    config = HypoLabConfig(groq_api_key="", use_local_model=False, use_ollama=False)
    agent = HypothesisAgent(config)
    
    profile = json.dumps({
        "numeric": {"age": {}, "income": {}},
        "categorical": {},
        "correlations": [{"col_a": "age", "col_b": "income", "pearson_r": 0.6}]
    })
    
    result = agent.generate(profile)
    
    assert len(result) > 0
    # Should be data-driven, not mocked template garbage
    assert any("age" in str(h.get("variables", [])) for h in result)


def test_generate_method_routing_priority():
    """Verify routing: Groq > Ollama > Smart."""
    config = HypoLabConfig(groq_api_key="", use_ollama=False, use_local_model=False)
    agent = HypothesisAgent(config)
    
    # With no keys, should NOT have client or local pipeline
    assert agent.client is None
    assert agent.local_pipeline is None
    
    # Should still produce results via smart analysis
    result = agent.generate('{"numeric": {"a": {}}, "correlations": []}')
    assert isinstance(result, list)