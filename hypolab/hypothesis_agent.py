"""LLM-powered hypothesis generation with Groq + Ollama + smart data-driven fallback."""

import json
import re
import warnings
from typing import Dict, List, Optional

import requests

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
        """Initialize with chosen backend."""
        self.config = config or HypoLabConfig()
        self.client = None
        self.local_pipeline = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Groq, Ollama, or local HF model."""
        if (
            self.config.has_groq()
            and not self.config.use_ollama
            and not self.config.use_local_model
        ):
            try:
                from groq import Groq

                self.client = Groq(api_key=self.config.groq_api_key)
            except ImportError:
                warnings.warn("groq not installed; trying Ollama or local fallback.")

        if self.config.use_ollama and not self.config.has_groq():
            # Ollama doesn't need init — just HTTP calls
            pass
        elif self.config.use_local_model:
            self._init_local_model()

    def _init_local_model(self) -> None:
        """Load a local Hugging Face model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_name = self.config.local_model_name
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
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
            warnings.warn(f"Local model init failed: {e}")

    def generate(self, profile_json: str) -> List[Dict]:
        """Generate hypotheses using best available backend."""
        if self.client and self.config.has_groq():
            return self._generate_with_llm(profile_json, self._call_groq)
        elif self.config.use_ollama:
            return self._generate_with_llm(profile_json, self._call_ollama)
        elif self.local_pipeline:
            return self._generate_with_llm(profile_json, self._call_local)
        else:
            # Smart data-driven analysis — no LLM needed
            return self._smart_data_driven(profile_json)

    def _generate_with_llm(self, profile_json: str, caller) -> List[Dict]:
        """Generic LLM generation with prompt building."""
        prompt = self._build_prompt(profile_json)
        raw = caller(prompt)
        return self._parse_response(raw)

    def _build_prompt(self, profile_json: str) -> str:
        """Build prompt for LLM."""
        return (
            f"{self.SYSTEM_PROMPT}\n\nDataset Profile:\n{profile_json}\n\nHypotheses:"
        )

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
            warnings.warn(f"Groq failed: {e}")
            return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama API."""
        try:
            response = requests.post(
                self.config.ollama_url,
                json={
                    "model": self.config.ollama_model,
                    "prompt": f"{self.SYSTEM_PROMPT}\n\n{prompt}",
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            warnings.warn(f"Ollama failed: {e}. Using smart data-driven fallback.")
            return ""

    def _call_local(self, prompt: str) -> str:
        """Call local Hugging Face model."""
        if self.local_pipeline is None:
            return ""
        try:
            result = self.local_pipeline(prompt, do_sample=True, temperature=0.3)
            return result[0]["generated_text"]
        except Exception as e:
            warnings.warn(f"Local model failed: {e}")
            return ""

    def _smart_data_driven(self, profile_json: str) -> List[Dict]:
        """
        Generate intelligent hypotheses by analyzing the actual data profile.
        No LLM needed — uses correlation strength, column types, and distributions.
        """
        try:
            profile = json.loads(profile_json)
        except json.JSONDecodeError:
            profile = {}

        hypotheses = []

        # Extract columns from profile
        numeric = list(profile.get("numeric", {}).keys())
        cat = list(profile.get("categorical", {}).keys())
        correlations = profile.get("correlations", [])

        # 1. Strongest correlation hypothesis
        if correlations:
            top = correlations[0]
            r_val = abs(top.get("pearson_r", 0))
            strength = (
                "strong" if r_val > 0.5 else "moderate" if r_val > 0.3 else "weak"
            )
            hypotheses.append(
                {
                    "hypothesis": f"H0: {top['col_a']} and {top['col_b']} are uncorrelated; H1: significant {strength} Pearson correlation exists",
                    "test_type": "pearson_correlation",
                    "variables": [top["col_a"], top["col_b"]],
                    "rationale": f"Profile shows {strength} correlation (r={top['pearson_r']}) between {top['col_a']} and {top['col_b']}. Testing if this is statistically significant.",
                }
            )

        # 2. ANOVA: numeric vs categorical
        if numeric and cat:
            # Pick the numeric with highest variance (most interesting)
            best_num = numeric[0]
            num_stats = profile.get("numeric", {}).get(best_num, {})
            std = num_stats.get("std", 0) if isinstance(num_stats, dict) else 0
            hypotheses.append(
                {
                    "hypothesis": f"H0: Mean {best_num} is equal across all groups of {cat[0]}; H1: at least one group differs significantly",
                    "test_type": "anova",
                    "variables": [best_num, cat[0]],
                    "rationale": f"{best_num} has std={std:.1f} showing substantial variation; ANOVA tests if {cat[0]} explains this variance.",
                }
            )

        # 3. Second correlation (if exists)
        if len(correlations) > 1:
            second = correlations[1]
            hypotheses.append(
                {
                    "hypothesis": f"H0: {second['col_a']} and {second['col_b']} are uncorrelated; H1: significant association exists",
                    "test_type": "pearson_correlation",
                    "variables": [second["col_a"], second["col_b"]],
                    "rationale": f"Secondary correlation detected (r={second['pearson_r']}) — worth validating independently.",
                }
            )

        # 4. Regression: target prediction
        if len(numeric) >= 2:
            target = numeric[0]
            predictors = numeric[1:3]  # Top 2 predictors
            hypotheses.append(
                {
                    "hypothesis": f"H0: {', '.join(predictors)} do not predict {target}; H1: significant predictive relationship exists",
                    "test_type": "regression",
                    "variables": [target] + predictors,
                    "rationale": f"Multiple regression tests combined predictive power of {', '.join(predictors)} on {target}.",
                }
            )

        # 5. Chi-square for categorical
        if len(cat) >= 2:
            hypotheses.append(
                {
                    "hypothesis": f"H0: {cat[0]} and {cat[1]} are independent; H1: significant association exists",
                    "test_type": "chi_square",
                    "variables": cat[:2],
                    "rationale": f"Chi-square tests whether {cat[0]} and {cat[1]} exhibit categorical dependency.",
                }
            )

        # 6. T-test if binary categorical exists
        if len(numeric) >= 1 and len(cat) >= 1:
            # Check if any categorical has only 2 unique values
            cat_stats = profile.get("categorical", {}).get(cat[0], {})
            unique = (
                cat_stats.get("unique_count", 0) if isinstance(cat_stats, dict) else 0
            )
            if unique == 2:
                hypotheses.append(
                    {
                        "hypothesis": f"H0: Mean {numeric[0]} is equal between the two groups of {cat[0]}; H1: significant difference exists",
                        "test_type": "t_test",
                        "variables": [numeric[0], cat[0]],
                        "rationale": f"{cat[0]} is binary — ideal for independent samples t-test on {numeric[0]}.",
                    }
                )

        if not hypotheses:
            hypotheses = [
                {
                    "hypothesis": "H0: No significant pattern exists; H1: significant pattern detected",
                    "test_type": "regression",
                    "variables": numeric[:2] if len(numeric) >= 2 else numeric + cat,
                    "rationale": "Exploratory analysis to identify any significant relationships in the dataset.",
                }
            ]

        return hypotheses

    def _parse_response(self, raw: str) -> List[Dict]:
        """Parse LLM response into structured hypotheses."""
        if not raw or not raw.strip():
            return []

        cleaned = re.sub(r"```json?", "", raw)
        cleaned = re.sub(r"```", "", cleaned)
        cleaned = cleaned.strip()

        try:
            hypotheses = json.loads(cleaned)
            if isinstance(hypotheses, dict):
                hypotheses = [hypotheses]
            return hypotheses
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return []
