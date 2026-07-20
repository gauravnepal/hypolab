"""LLM-powered hypothesis generation with Groq + local fallback."""

import json
import os
import re
import warnings
from typing import Dict, List, Optional

from .config import HypoLabConfig


class HypothesisAgent:
    """Generates testable statistical hypotheses from data profiles."""

    SYSTEM_PROMPT = """You are HypoLab, an expert data scientist and statistician.
Given a dataset profile, generate 3-5 concrete, testable statistical hypotheses.
Each hypothesis must include:
- hypothesis: a clear statement (H0 and implied H1)
- test_type: one of [pearson_correlation, anova, chi_square, t_test, granger_causality, regression]
- variables: list of involved column names
- rationale: 1-sentence reasoning based on the profile

Respond ONLY as a JSON array. No markdown, no explanations outside JSON."""

    def __init__(self, config: Optional[HypoLabConfig] = None):
        self.config = config or HypoLabConfig()
        self.client = None
        self.local_pipeline = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Groq client or local model."""
        if self.config.has_groq() and not self.config.use_local_model:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.config.groq_api_key)
            except ImportError:
                warnings.warn("groq not installed; falling back to local model.")
                self.config.use_local_model = True

        if self.config.use_local_model or not self.config.has_groq():
            self._init_local_model()

    def _init_local_model(self) -> None:
        """Load a local Hugging Face model (Phi-3-mini or similar)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_name = self.config.local_model_name
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.config.local_model_device,
                trust_remote_code=True,
            )
            self.local_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.config.local_max_tokens,
                return_full_text=False,
            )
        except Exception as e:
            warnings.warn(f"Local model initialization failed: {e}. Hypothesis generation will be mocked.")
            self.local_pipeline = None

    def generate(self, profile_json: str) -> List[Dict]:
        """Generate hypotheses from a data profile."""
        prompt = self._build_prompt(profile_json)
        
        if self.client and not self.config.use_local_model:
            raw = self._call_groq(prompt)
        elif self.local_pipeline:
            raw = self._call_local(prompt)
        else:
            raw = self._mock_response(profile_json)
        
        return self._parse_response(raw)

    def _build_prompt(self, profile_json: str) -> str:
        return f"{self.SYSTEM_PROMPT}\n\nDataset Profile:\n{profile_json}\n\nHypotheses:"

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API."""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.groq_model,
                temperature=0.3,
                max_tokens=2048,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            warnings.warn(f"Groq API call failed: {e}. Trying fallback model.")
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.config.groq_fallback_model,
                    temperature=0.3,
                    max_tokens=2048,
                )
                return chat_completion.choices[0].message.content
            except Exception as e2:
                warnings.warn(f"Groq fallback failed: {e2}. Using local/mock.")
                if self.local_pipeline:
                    return self._call_local(prompt)
                return self._mock_response(prompt)

    def _call_local(self, prompt: str) -> str:
        """Call local Hugging Face model."""
        if self.local_pipeline is None:
            return self._mock_response(prompt)
        try:
            result = self.local_pipeline(prompt, do_sample=True, temperature=0.3)
            return result[0]["generated_text"]
        except Exception as e:
            warnings.warn(f"Local model inference failed: {e}")
            return self._mock_response(prompt)

    def _mock_response(self, profile_json: str) -> str:
        """Graceful degradation: return template hypotheses when no LLM is available."""
        # Parse the profile JSON to extract actual column names
        try:
            profile = json.loads(profile_json)
        except json.JSONDecodeError:
            profile = {}
        
        # Extract numeric columns from profile["numeric"] keys
        numeric = list(profile.get("numeric", {}).keys())
        # Extract categorical columns from profile["categorical"] keys
        cat = list(profile.get("categorical", {}).keys())
        # Fallback: try to parse from schema_summary
        if not numeric and not cat:
            summary = profile.get("schema_summary", "")
            # Look for "Numeric columns (3): age, income, score"
            num_match = re.search(r"Numeric columns?\s*\(\d+\):\s*([^\n]+)", summary)
            if num_match:
                numeric = [c.strip() for c in num_match.group(1).split(",") if c.strip()]
            cat_match = re.search(r"Categorical columns?\s*\(\d+\):\s*([^\n]+)", summary)
            if cat_match:
                cat = [c.strip() for c in cat_match.group(1).split(",") if c.strip()]
        
        hypotheses = []
        if len(numeric) >= 2:
            hypotheses.append({
                "hypothesis": f"H0: {numeric[0]} and {numeric[1]} are uncorrelated; H1: significant Pearson correlation exists",
                "test_type": "pearson_correlation",
                "variables": numeric[:2],
                "rationale": f"Both {numeric[0]} and {numeric[1]} are numeric; correlation may reveal linear dependency."
            })
        if len(numeric) >= 1 and len(cat) >= 1:
            hypotheses.append({
                "hypothesis": f"H0: Mean {numeric[0]} is equal across all groups of {cat[0]}; H1: at least one group differs",
                "test_type": "anova",
                "variables": [numeric[0], cat[0]],
                "rationale": f"ANOVA tests whether {cat[0]} categories explain variance in {numeric[0]}."
            })
        if len(numeric) >= 2:
            hypotheses.append({
                "hypothesis": f"H0: {numeric[0]} does not Granger-cause {numeric[1]}; H1: predictive causality exists",
                "test_type": "granger_causality",
                "variables": numeric[:2],
                "rationale": "If temporal ordering exists, Granger causality tests directional predictive influence."
            })
        if len(cat) >= 2:
            hypotheses.append({
                "hypothesis": f"H0: {cat[0]} and {cat[1]} are independent; H1: association exists",
                "test_type": "chi_square",
                "variables": cat[:2],
                "rationale": "Chi-square test of independence for categorical association."
            })
        
        if not hypotheses:
            all_cols = numeric + cat
            hypotheses = [{
                "hypothesis": "H0: No significant pattern exists; H1: significant pattern detected",
                "test_type": "regression",
                "variables": all_cols[:2] if len(all_cols) >= 2 else all_cols,
                "rationale": "Exploratory regression to identify predictive relationships."
            }]
        
        return json.dumps(hypotheses)
    def _parse_response(self, raw: str) -> List[Dict]:
        """Parse LLM response into structured hypotheses."""
        # Clean markdown fences
        cleaned = re.sub(r"```json?", "", raw)
        cleaned = re.sub(r"```", "", cleaned)
        cleaned = cleaned.strip()
        
        try:
            hypotheses = json.loads(cleaned)
            if isinstance(hypotheses, dict):
                hypotheses = [hypotheses]
            return hypotheses
        except json.JSONDecodeError:
            warnings.warn("Failed to parse LLM response as JSON. Returning empty list.")
            # Try extracting JSON array via regex
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return []