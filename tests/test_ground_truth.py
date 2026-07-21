"""Precision/recall validation: pipeline must find real relationships and ignore noise."""

import numpy as np
import pandas as pd
import pytest

from hypolab.pipeline import HypoLabPipeline
from hypolab.config import HypoLabConfig


def generate_ground_truth(n: int = 1000, noise: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data with 3 real relationships and 3 pure noise columns.
    
    Real signals:
      1. feature_a ↔ feature_b (correlation ≈ 0.7)
      2. target differs by group (ANOVA significant)
      3. feature_c predicts target (regression, β = 2.5)
    
    Noise columns:
      feature_d, feature_e, feature_f (independent of everything)
    """
    rng = np.random.default_rng(seed)
    
    # Signal 1: correlated pair
    feature_a = rng.normal(50, 10, n)
    feature_b = 0.7 * feature_a + rng.normal(0, 5, n)  # r ≈ 0.7
    
    # Signal 2: group differences for target
    group = rng.choice(["A", "B", "C"], n)
    group_effect = pd.Series(group).map({"A": 10, "B": 20, "C": 35}).values
    
    # Signal 3: feature_c predicts target
    feature_c = rng.normal(100, 20, n)
    
    # Noise columns
    feature_d = rng.normal(0, 1, n)
    feature_e = rng.normal(100, 50, n)
    feature_f = rng.choice(["X", "Y", "Z"], n)
    
    # Target = group_effect + 2.5*feature_c + noise
    target = group_effect + 2.5 * feature_c + rng.normal(0, 15, n)
    
    df = pd.DataFrame({
        "feature_a": feature_a,
        "feature_b": feature_b,
        "feature_c": feature_c,
        "feature_d": feature_d,
        "feature_e": feature_e,
        "feature_f": feature_f,
        "group": group,
        "target": target,
    })
    return df


@pytest.fixture
def ground_truth_df():
    """Pytest fixture providing ground truth dataset."""
    return generate_ground_truth(n=1000, noise=0.3)


def test_pipeline_finds_real_relationships(ground_truth_df):
    """Pipeline should discover at least 3 significant relationships."""
    config = HypoLabConfig(groq_api_key="", use_local_model=False, verbose=False)
    pipe = HypoLabPipeline(ground_truth_df, target_col="target", config=config)
    report = pipe.run(skip_literature=True)
    
    summary = report["test_summary"]
    print(f"\nGround truth results: {summary}")
    
    # Must find at least 3 significant relationships
    assert summary["significant"] >= 3, f"Expected >=3 significant, got {summary['significant']}"
    
    # All tests should complete (no errors)
    assert summary["errors"] == 0


def test_pipeline_rejects_noise_hypotheses(ground_truth_df):
    """
    The agent may propose hypotheses involving noise columns (it doesn't know they're noise),
    but the statistical verification layer MUST reject them (p > alpha or skipped).
    This is the core safety mechanism: LLM proposes, math verifies.
    """
    config = HypoLabConfig(groq_api_key="", use_local_model=False, verbose=False)
    pipe = HypoLabPipeline(ground_truth_df, target_col="target", config=config)
    report = pipe.run(skip_literature=True)
    
    noise_vars = {"feature_d", "feature_e", "feature_f"}
    
    # Find any hypotheses that touch noise variables
    noise_hyps = []
    for hyp, res in zip(report["hypotheses"], report["test_results"]):
        vars_used = set(hyp.get("variables", []))
        if vars_used & noise_vars:
            noise_hyps.append((hyp, res))
    
    # If the agent proposed noise hypotheses, the statistical layer must reject them
    for hyp, res in noise_hyps:
        is_rejected = (
            res.get("status") == "skipped" or 
            not res.get("significant", False)
        )
        assert is_rejected, (
            f"CRITICAL: Noise hypothesis was accepted as significant!\n"
            f"Hypothesis: {hyp['hypothesis']}\n"
            f"Variables: {hyp['variables']}\n"
            f"Result: p={res.get('p_value', 'N/A')}, status={res.get('status')}"
        )
    
    # Also verify: at least 3 real relationships WERE found significant
    summary = report["test_summary"]
    assert summary["significant"] >= 3, (
        f"Expected >=3 significant real relationships, got {summary['significant']}"
    )
    
    print(f"\nSafety check passed: {len(noise_hyps)} noise hypotheses proposed, all rejected.")
    print(f"Real signals found: {summary['significant']}/{summary['completed']} significant.")