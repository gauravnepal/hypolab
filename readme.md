<!-- CI badge disabled until first workflow run -->
<!-- [![CI](https://github.com/gauravnepal/hypolab/actions/workflows/ci.yml/badge.svg)](https://github.com/gauravnepal/hypolab/actions) -->
# 🧪 HypoLab

<p align="center">

### **LLM proposes. Math verifies.**

**An agentic AI pipeline that generates statistical hypotheses, validates them with rigorous statistical tests, and retrieves supporting scientific literature.**

*Because AI should generate ideas—not unquestionable conclusions.*

</p>

---

## ✨ Overview

HypoLab is an **Agentic Data Science Pipeline** that combines Large Language Models with classical statistics.

Instead of trusting an LLM's interpretation of a dataset, HypoLab lets the AI generate **falsifiable hypotheses**, then automatically verifies each one using statistical testing before searching **arXiv** for relevant scientific papers.

The result is an explainable research assistant that combines:

- 🤖 AI reasoning
- 📈 Statistical validation
- 📚 Scientific evidence

---

## 🚀 Features

- 🤖 LLM-generated hypotheses from any CSV dataset
- 📊 Automatic statistical test selection
- ✅ p-value verification
- 📐 Effect size calculation
- 📉 Correlation analysis
- 📈 Linear regression
- 🌍 ANOVA across categorical groups
- 📚 arXiv literature retrieval
- 🧠 Multi-agent pipeline
- 🟢 Cloud LLM support (Groq)
- 🟡 Local LLM support (Ollama)
- 🟠 Zero-API fallback mode
- 🎯 Ground-truth statistical testing
- 🖥️ Interactive Streamlit interface
- 🧪 Comprehensive pytest test suite

---

# 🏗 Architecture

```text
                 ┌──────────────┐
                 │ CSV / Dataset│
                 └──────┬───────┘
                        │
                        ▼
                Data Profiler Agent
                        │
                        ▼
             Schema & Dataset Summary
                        │
                        ▼
               LLM Hypothesis Agent
                        │
          Generates Falsifiable Claims
                        │
                        ▼
            Statistical Verification
                        │
          Pearson • ANOVA • Regression
                        │
                        ▼
        p-values • Effect Sizes • R²
                        │
                        ▼
             arXiv Literature Search
                        │
                        ▼
          Verified Research Report
```

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

# 📂 Project Structure

```text
hypolab/
│
├── agents/
│   ├── hypothesis_agent.py
│   ├── profiler_agent.py
│   └── literature_agent.py
│
├── statistics/
│   ├── correlation.py
│   ├── regression.py
│   └── anova.py
│
├── tests/
│   ├── test_ground_truth.py
│   ├── test_statistics.py
│   └── ...
│
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

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

Launch the Streamlit interface.

```bash
streamlit run app.py
```

Run all tests.

```bash
pytest tests/ -v
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