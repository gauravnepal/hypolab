# 🧪 HypoLab

&lt;p align="center"&gt;

**LLM proposes. Math verifies.**

An agentic research assistant that combines LLM reasoning with statistical verification, ensuring every accepted insight is supported by evidence—not just AI confidence.

*Because AI should generate ideas—not unquestionable conclusions.*

&lt;/p&gt;

&lt;p align="center"&gt;

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Tests](https://img.shields.io/badge/tests-40%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/demo-Streamlit-ff4b4b.svg)

&lt;/p&gt;

---

## 📑 Table of Contents

- [Why HypoLab Exists](#-why-hypolab-exists)
- [Architecture](#-architecture)
- [What It Actually Does](#-what-it-actually-does)
- [The 3-Tier Intelligence Stack](#-the-3-tier-intelligence-stack)
- [Quick Start](#-quick-start)
  - [Installation](#-installation)
  - [Running the App](#-running-the-app)
- [Try It Yourself](#-try-it-yourself)
- [The Ground Truth Benchmark](#-the-ground-truth-benchmark)
- [Repository Structure](#-repository-structure)
- [Technology Stack](#-technology-stack)
- [Configuration](#%EF%B8%8F-configuration)
- [Example Output](#-example-output)
- [Testing](#-testing)
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

---

# 🏗 Architecture

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
---

# 🧠 AI Fallback Strategy

HypoLab automatically chooses the best available intelligence layer.

| Priority | Engine | Description |
|----------|---------|-------------|
| 🟢 | Groq (Llama 3.3 70B) | Fast cloud inference |
| 🟡 | Ollama (Llama 3.1 8B) | Local, private inference |
| 🟠 | Smart Analysis | Rule-based analysis requiring no LLM |

Even without an API key, HypoLab still performs meaningful statistical analysis.

---

# 📂 Repository Structure

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


*(Update this section to match your actual directory structure.)*

---

# 🚀 Installation

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

# ▶ Running the App
Run all tests.

```bash
pytest tests/ -v
```

Launch the Streamlit interface.

```bash
streamlit run app.py
```

---

# 📊 Example Dataset

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

# 📈 Example Output

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

# 🧪 Statistical Verification

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

# 🧪 Ground Truth Testing

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

# ⚙ Configuration

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

# 📦 Tech Stack

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

# 🛣 Roadmap

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