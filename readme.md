# 🧪 HypoLab — Agentic Data Pipeline

[![CI](https://github.com/gauravnepal/hypolab/actions/workflows/ci.yml/badge.svg)](https://github.com/gauravnepal/hypolab/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

&gt; **Automated hypothesis generation, statistical validation, and literature search — powered by LLMs.**

HypoLab is a modular Python pipeline that:
1. **Profiles** your dataset (numeric, categorical, datetime, correlations, outliers)
2. **Generates** testable statistical hypotheses via LLM agents (Groq Llama 3.3 or local Phi-3-mini)
3. **Validates** them with statistical tests (Pearson, ANOVA, Chi-square, T-test, Granger causality, OLS regression)
4. **Searches** arXiv for supporting academic literature
5. **Presents** everything in a beautiful Streamlit UI

---

## 🚀 Quick Start

```bash
# Clone & install
git clone https://github.com/gauravnepal/hypolab.git
cd hypolab
pip install -e ".[dev]"

# Run tests (12 offline tests)
pytest tests/ -v --cov=hypolab

# Launch Streamlit UI
streamlit run app.py