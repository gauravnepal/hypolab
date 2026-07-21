"""Main orchestrator — ties profiler, agent, tester, and literature search together."""

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import HypoLabConfig
from .profiler import DataProfiler
from .hypothesis_agent import HypothesisAgent
from .statistical_tester import StatisticalTester
from .literature_search import LiteratureSearch


class HypoLabPipeline:
    """End-to-end agentic data pipeline."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        config: Optional[HypoLabConfig] = None,
    ):
        """Initialize all pipeline stages."""
        self.df = df
        self.target_col = target_col
        self.config = config or HypoLabConfig()
        self.profiler = DataProfiler(df, target_col)
        self.agent = HypothesisAgent(self.config)
        self.tester: Optional[StatisticalTester] = None
        self.lit_search = LiteratureSearch(max_results=self.config.arxiv_max_results)

        self.profile: Optional[Dict] = None
        self.hypotheses: List[Dict] = []
        self.test_results: List[Dict] = []
        self.literature: Dict[str, List[Dict]] = {}

    def run(self, skip_literature: bool = False) -> Dict[str, Any]:
        """Execute full pipeline: profile → hypothesize → test → search."""
        # Step 1: Profile
        self._log("Profiling dataset...")
        self.profile = self.profiler.run()

        # Step 2: Generate hypotheses
        self._log("Generating hypotheses via LLM...")
        profile_json = self.profiler.to_json()
        self.hypotheses = self.agent.generate(profile_json)
        self._log(f"   Generated {len(self.hypotheses)} hypotheses")

        # Step 3: Statistical testing
        self._log("Running statistical tests...")
        self.tester = StatisticalTester(self.df, alpha=self.config.significance_level)
        self.test_results = self.tester.run_all(self.hypotheses)

        summary = self.tester.get_summary()
        self._log(
            f"   Completed: {summary['completed']}, Significant: {summary['significant']}"
        )

        # Step 4: Literature search (optional)
        if not skip_literature:
            self._log("Searching arXiv literature...")
            for hyp in self.hypotheses:
                hyp_id = hyp.get("hypothesis", "")[:50]
                papers = self.lit_search.search_for_hypothesis(hyp)
                self.literature[hyp_id] = papers
            self._log(f"   Searched {len(self.literature)} hypotheses")

        return self.get_report()

    def get_report(self) -> Dict[str, Any]:
        """Compile full pipeline report."""
        return {
            "profile_summary": self.profile.get("schema_summary", "")
            if self.profile
            else "",
            "hypotheses": self.hypotheses,
            "test_results": self.test_results,
            "test_summary": self.tester.get_summary() if self.tester else {},
            "literature": {
                k: [{"title": p["title"], "url": p["url"]} for p in v]
                for k, v in self.literature.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string."""
        return json.dumps(self.get_report(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Export report as Markdown."""
        report = self.get_report()
        lines = ["# HypoLab Report\n"]

        lines.append("## Dataset Profile\n")
        lines.append(f"```\n{report['profile_summary']}\n```\n")

        lines.append("## Hypotheses & Test Results\n")
        for i, (hyp, res) in enumerate(
            zip(report["hypotheses"], report["test_results"]), 1
        ):
            lines.append(
                f"### {i}. {hyp.get('test_type', 'Unknown').replace('_', ' ').title()}\n"
            )
            lines.append(f"**Hypothesis:** {hyp.get('hypothesis', '')}\n")
            lines.append(f"**Variables:** {', '.join(hyp.get('variables', []))}\n")
            lines.append(f"**Rationale:** {hyp.get('rationale', '')}\n")

            if res.get("status") == "completed":
                sig = "Significant" if res.get("significant") else "Not Significant"
                lines.append(f"**Result:** {sig} (p={res.get('p_value', 'N/A')})\n")
                for key in [
                    "pearson_r",
                    "f_statistic",
                    "chi2",
                    "t_statistic",
                    "r_squared",
                ]:
                    if key in res:
                        lines.append(
                            f"- **{key.replace('_', ' ').title()}:** {res[key]}\n"
                        )
            else:
                lines.append(
                    f"**Result:** {res.get('status', 'Unknown')} — {res.get('reason', '')}\n"
                )

            hyp_id = hyp.get("hypothesis", "")[:50]
            papers = self.literature.get(hyp_id, [])
            if papers:
                lines.append("**Literature:**\n")
                for p in papers[:2]:
                    lines.append(f"- [{p['title']}]({p['url']})\n")
            lines.append("---\n")

        summary = report.get("test_summary", {})
        lines.append("## Summary\n")
        lines.append(f"- Total hypotheses: {summary.get('total_hypotheses', 0)}\n")
        lines.append(f"- Completed tests: {summary.get('completed', 0)}\n")
        lines.append(f"- Significant findings: {summary.get('significant', 0)}\n")
        lines.append(
            f"- Skipped: {summary.get('skipped', 0)} | Errors: {summary.get('errors', 0)}\n"
        )

        return "\n".join(lines)

    def _log(self, message: str) -> None:
        """Print if verbose mode is on."""
        if self.config.verbose:
            print(message)
