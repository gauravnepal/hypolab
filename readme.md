# 🧪 HypoLab

**LLM proposes. Math verifies.**

An agentic research assistant that combines LLM reasoning with statistical verification, ensuring every accepted insight is supported by evidence—not just AI confidence.

*Because AI should generate ideas—not unquestionable conclusions.*


![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Tests](https://img.shields.io/badge/tests-40%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/demo-Streamlit-ff4b4b.svg)



---

## 📑 Table of Contents

- [Why HypoLab Exists](#-why-hypolab-exists)
- [Architecture](#-architecture)
- [What It Actually Does](#-what-it-actually-does)
- [Quick Start](#-quick-start)
- [Live Demo](#-live-demo)
- [How It Works](#-how-it-works)
- [Repository Structure](#-repository-structure)
- [Tech Stack & Configuration](#-tech-stack--configuration)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

# 🤔 Why HypoLab Exists

Current AI data tools give you "insights" that sound convincing but can't be trusted. An LLM might claim *"sales correlate with temperature"* — but without a p-value, that's just hallucinated pattern-matching.

**HypoLab fixes this with a verification loop:**

1. **Propose** — An LLM agent (or data-driven fallback) suggests testable hypotheses
2. **Verify** — Classical statistical tests (Pearson, ANOVA, Regression, etc.) validate or reject each claim with p-values
3. **Support** — arXiv literature search provides academic context for findings that survive

The result is **explainable, falsifiable data analysis** — not AI guesswork.

###

## 🏗 Architecture

```text
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   CSV/URL   │────▶│  DataProfiler   │────▶│  Schema Summary  │
└─────────────┘     └─────────────────┘     └──────────────────┘
                                                        │
                              ┌─────────────────────────┘
                              ▼
                       ┌─────────────────┐
                       │ HypothesisAgent │◄── Groq / Ollama / Smart Analysis
                       │  (LLM Agent)    │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ StatisticalTester│◄── scipy / statsmodels
                       │  (Verification)  │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ LiteratureSearch│◄── arXiv API
                       │  (Evidence)     │
                       └─────────────────┘
```
## ✨ What It Actually Does
Table
Stage	Implementation	Output
Profile	Auto-detects numeric/categorical/datetime, correlations, outliers, missing data	Structured dataset summary
Hypothesize	Groq Llama 3.3 → Ollama local → Smart Data-Driven Analysis (zero API cost)	3-5 testable hypotheses with rationale
Validate	Pearson correlation, ANOVA, Chi-square, T-test, Granger causality, OLS regression	p-values, effect sizes, R²
Research	arXiv Atom API query built from hypothesis metadata	Supporting papers with abstracts

---

### 🧠 The 3-Tier Intelligence Stack
HypoLab automatically selects the best available backend. No API key required.
Table
Tier	Engine	When It Runs
🟢 Groq	Llama 3.3-70B	GROQ_API_KEY is set
🟡 Ollama	Llama 3.1-8B	USE_OLLAMA=true (local, free, private)
🟠 Smart Analysis	Data-driven rule engine	No key, no local model — analyzes your actual correlation matrix and column types to generate intelligent hypotheses
Even in zero-cost mode, HypoLab analyzes real data patterns — not random templates.

---

## 📂 Repository Structure

```text
hypolab/
│
├── .github/
│   └── workflows/              # GitHub Actions CI
│
├── assets/                     # Images and demo assets
├── docs/                       # Project documentation
├── notebooks/                  # Jupyter notebooks
├── references/                 # Research references
│
├── hypolab/                    # Main package
│   ├── __init__.py
│   ├── config.py
│   ├── profiler.py
│   ├── hypothesis_agent.py
│   ├── statistical_tester.py
│   ├── literature_search.py
│   └── pipeline.py
│
├── tests/
│   ├── test_ground_truth.py
│   ├── test_hypothesis_agent.py
│   ├── test_literature_search.py
│   ├── test_pipeline.py
│   ├── test_profiler.py
│   └── test_statistical_tester.py
│
├── app.py                      # Streamlit application
├── pyproject.toml              # Project metadata
├── requirements.txt
├── LICENSE
└── README.md
```


---

## 🚀 Quick Start

Clone the repository

```bash
git clone https://github.com/gauravnepal/hypolab.git
```

Move into the project

```bash
cd hypolab
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

Install dependencies

```bash
pip install -e .
```

---

### ▶ Running the App
Run all tests.

```bash
pytest tests/ -v
```

Launch the Streamlit interface.

```bash
streamlit run app.py
```

---

## 📊 Live Demo

Gapminder dataset

```text
https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv
```

Example discoveries:

- Life expectancy positively correlates with GDP
- Significant life expectancy differences across continents
- GDP and year jointly predict life expectancy
- Supporting papers retrieved from arXiv

---

### 📈 Example Output

Generated hypothesis

> Countries with higher GDP per capita tend to have higher life expectancy.

Statistical verification

```text
Pearson Correlation

r = 0.58

p < 0.001

Status:
✅ Verified
```

Supporting literature

```
3 relevant papers retrieved from arXiv
```

---

## 🧪 How It Works

The LLM **never** decides whether a hypothesis is true.

Every hypothesis is verified using classical statistical methods.

Supported methods include:

- Pearson Correlation
- Spearman Correlation
- Linear Regression
- ANOVA
- t-tests (planned)
- Chi-square (planned)

Each result reports

- p-value
- confidence
- effect size
- regression metrics
- statistical interpretation

---

### 🧪 Ground Truth Testing

To validate the pipeline, HypoLab includes a synthetic benchmark dataset.

The benchmark contains:

- ✅ 3 genuine statistical relationships
- ❌ 3 completely random variables

The hypothesis agent proposes claims for both.

The statistical layer correctly

- verifies the true relationships
- rejects the noise

This behavior is tested automatically in

```text
tests/test_ground_truth.py
```

---

## ⚙️ Tech Stack & Configuration

| Environment Variable | Description |
|----------------------|-------------|
| `GROQ_API_KEY` | Enables Groq cloud inference |
| `USE_OLLAMA` | Enables local Llama inference |
| `SIGNIFICANCE_LEVEL` | Statistical significance threshold (default 0.05) |

Example

```bash
export GROQ_API_KEY=your_key
export SIGNIFICANCE_LEVEL=0.05
```

---
📦 Tech Stack
- Python
- Pandas
- NumPy
- SciPy
- Statsmodels
- Streamlit
- pytest
- Groq API
- Ollama
- arXiv API

---

## 🛣 Roadmap

- [ ] Multi-agent orchestration
- [ ] Bayesian hypothesis testing
- [ ] Causal inference
- [ ] SHAP explainability
- [ ] PDF research report export
- [ ] LangGraph integration
- [ ] Docker deployment
- [ ] Hugging Face Spaces deployment
- [ ] More statistical tests
- [ ] Automatic visualization generation

---

# 🤝 Contributing

Contributions are welcome.

Feel free to open an issue or submit a pull request for:

- bug fixes
- new statistical methods
- UI improvements
- documentation
- additional datasets

---

# 📄 License

This project is released under the MIT License.

---

# 👨‍💻 Author

**Gaurav Nepal**

Data Engineer • Machine Learning • AI Systems

GitHub

https://github.com/gauravnepal

---

<p align="center">

**LLM proposes. Math verifies.**

</p>