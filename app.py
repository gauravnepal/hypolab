"""Streamlit frontend for HypoLab."""

import json

import pandas as pd
import streamlit as st

from hypolab.config import HypoLabConfig
from hypolab.pipeline import HypoLabPipeline

st.set_page_config(page_title="HypoLab", page_icon="🧪", layout="wide")

st.title("🧪 HypoLab — Agentic Data Pipeline")
st.markdown(
    "Upload a CSV or paste a URL, let an AI agent generate hypotheses, validate them statistically, and find supporting literature."
)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
groq_key = st.sidebar.text_input(
    "Groq API Key (optional)",
    type="password",
    help="Leave empty to use Smart Analysis or Ollama",
)
use_ollama = st.sidebar.checkbox(
    "Use Ollama (Local LLM)",
    value=False,
    help="Requires Ollama running on localhost:11434",
)
use_local = st.sidebar.checkbox(
    "Use HuggingFace Local Model",
    value=False,
    help="Requires transformers + GPU (slow)",
)
skip_lit = st.sidebar.checkbox("Skip Literature Search", value=False)
alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)

st.sidebar.markdown("---")
st.sidebar.info("""
**Pipeline Steps:**
1. 📊 Profile Dataset
2. 🧠 AI Hypothesis Generation
3. 📈 Statistical Testing
4. 📚 arXiv Literature Search
""")

# Mode indicator banner
config_preview = HypoLabConfig(
    groq_api_key=groq_key if groq_key else None,
    use_ollama=use_ollama,
    use_local_model=use_local,
)

if config_preview.has_groq():
    st.success("🟢 **LLM Mode:** Groq Llama 3.3 — Full AI hypothesis generation")
elif use_ollama:
    st.info("🟡 **Local LLM Mode:** Ollama — Running on your Mac (free & private)")
elif use_local:
    st.info("🟡 **Local HF Mode:** HuggingFace model — GPU required")
else:
    st.warning(
        "🟠 **Smart Analysis Mode:** Data-driven hypotheses — No LLM needed, analyzes your actual data patterns"
    )

# Initialize session state for sample data
if "sample_df" not in st.session_state:
    st.session_state.sample_df = None

# ============================================================
# DATA LOADING: URL or File Upload or Sample Data
# ============================================================
st.subheader("📁 Load Data")

col1, col2 = st.columns([2, 1])

with col1:
    csv_url = st.text_input(
        "Paste a CSV URL",
        placeholder="https://raw.githubusercontent.com/.../data.csv",
        help="Direct link to a raw CSV file",
    )

with col2:
    uploaded_file = st.file_uploader("Or upload CSV", type=["csv"])

df = None
load_error = None

# Priority: URL > Uploaded File > Session State
if csv_url:
    try:
        df = pd.read_csv(csv_url)
        st.success(f"Loaded from URL: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        # macOS SSL fallback: use requests without verification
        if "CERTIFICATE_VERIFY_FAILED" in str(e) or "SSL" in str(e):
            try:
                import requests
                from io import StringIO
                import urllib3
                
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                response = requests.get(csv_url, verify=False, timeout=30)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text))
                st.success(f"Loaded from URL (SSL bypass): {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e2:
                load_error = str(e2)
                st.error(f"Could not load URL: {e2}")
        else:
            load_error = str(e)
            st.error(f"Could not load URL: {e}")
# ============================================================
# PIPELINE EXECUTION
# ============================================================
if df is not None:
    with st.expander("🔍 Preview Data"):
        st.dataframe(df.head(20), use_container_width=True)

    # Target selection
    target_col = st.selectbox(
        "🎯 Select Target Column (optional)", ["None"] + list(df.columns)
    )
    target_col = None if target_col == "None" else target_col

    if st.button("🚀 Run HypoLab Pipeline", type="primary"):
        config = HypoLabConfig(
            groq_api_key=groq_key if groq_key else None,
            use_ollama=use_ollama,
            use_local_model=use_local,
            significance_level=alpha,
            verbose=False,
        )

        with st.spinner("Running pipeline... This may take a minute."):
            pipe = HypoLabPipeline(df, target_col=target_col, config=config)
            report = pipe.run(skip_literature=skip_lit)

        # Display results
        st.markdown("---")
        st.header("📋 Results")

        # Summary metrics
        summary = report.get("test_summary", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hypotheses", summary.get("total_hypotheses", 0))
        col2.metric("✅ Significant", summary.get("significant", 0))
        col3.metric("❌ Not Significant", summary.get("insignificant", 0))
        col4.metric("⚠️ Skipped", summary.get("skipped", 0))

        # Profile
        with st.expander("📊 Dataset Profile", expanded=False):
            st.text(report.get("profile_summary", ""))

        # Hypotheses & Results
        st.subheader("🧪 Hypotheses & Statistical Tests")

        for i, (hyp, res) in enumerate(
            zip(report["hypotheses"], report["test_results"]), 1
        ):
            with st.container():
                cols = st.columns([3, 2])
                with cols[0]:
                    st.markdown(
                        f"**{i}. {hyp.get('test_type', 'Unknown').replace('_', ' ').title()}**"
                    )
                    st.markdown(f"📝 *{hyp.get('hypothesis', '')}*")
                    st.markdown(
                        f"🔧 Variables: `{', '.join(hyp.get('variables', []))}`"
                    )
                    st.caption(f"💡 {hyp.get('rationale', '')}")

                with cols[1]:
                    if res.get("status") == "completed":
                        if res.get("significant"):
                            st.success(
                                f"✅ Significant (p={res.get('p_value', 'N/A')})"
                            )
                        else:
                            st.error(
                                f"❌ Not Significant (p={res.get('p_value', 'N/A')})"
                            )

                        stats_md = []
                        for key in [
                            "pearson_r",
                            "f_statistic",
                            "chi2",
                            "t_statistic",
                            "r_squared",
                            "best_lag",
                        ]:
                            if key in res:
                                stats_md.append(
                                    f"**{key.replace('_', ' ').title()}:** {res[key]}"
                                )
                        if stats_md:
                            st.markdown(" | ".join(stats_md))
                    else:
                        st.warning(
                            f"⚠️ {res.get('status', 'Unknown')}: {res.get('reason', '')}"
                        )
            st.divider()

        # Literature
        if not skip_lit and report.get("literature"):
            st.subheader("📚 Supporting Literature")
            for hyp_id, papers in report["literature"].items():
                if papers:
                    with st.expander(f"Papers for: {hyp_id[:60]}..."):
                        for p in papers:
                            st.markdown(
                                f"**[{p.get('title', 'Untitled')}]({p.get('url', '')})**"
                            )
                            authors = p.get("authors", [])
                            author_str = (
                                ", ".join(authors[:3]) if authors else "Unknown"
                            )
                            st.caption(
                                f"Authors: {author_str} | {p.get('published', 'N/A')}"
                            )
                            st.write(p.get("summary", "")[:300] + "...")

        # Download report
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📥 Download JSON Report",
                data=pipe.to_json(),
                file_name="hypolab_report.json",
                mime="application/json",
            )
        with col_dl2:
            st.download_button(
                "📥 Download Markdown Report",
                data=pipe.to_markdown(),
                file_name="hypolab_report.md",
                mime="text/markdown",
            )

else:
    st.info("👆 Upload a CSV, paste a URL, or generate sample data to get started.")

    # Demo section
    st.markdown("---")
    st.subheader("🎮 Try with Sample Data")
    if st.button("Generate Sample Dataset"):
        import numpy as np

        np.random.seed(42)
        sample_df = pd.DataFrame(
            {
                "age": np.random.normal(35, 12, 200),
                "income": np.random.normal(55000, 18000, 200),
                "satisfaction_score": np.random.normal(7.5, 1.5, 200),
                "department": np.random.choice(
                    ["Sales", "Engineering", "HR", "Marketing"], 200
                ),
                "tenure_years": np.random.normal(5, 3, 200),
            }
        )
        # Add real correlation
        sample_df["income"] = (
            sample_df["income"]
            + sample_df["tenure_years"] * 2000
            + np.random.normal(0, 5000, 200)
        )

        # Store in session state and rerun immediately
        st.session_state.sample_df = sample_df
        st.rerun()