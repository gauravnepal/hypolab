"""Data profiling module — generates structured dataset summaries."""

import json
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class DataProfiler:
    """Profiles a pandas DataFrame and extracts schema-aware metadata."""

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.df = df.copy()
        self.target_col = target_col
        self.profile: Dict[str, Any] = {}
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.datetime_cols: List[str] = []

    def run(self) -> Dict[str, Any]:
        """Execute full profiling pipeline."""
        self._infer_types()
        self._compute_numeric_stats()
        self._compute_categorical_stats()
        self._compute_correlations()
        self._detect_issues()
        self._summarize_schema()
        return self.profile

    def _infer_types(self) -> None:
        """Classify columns into numeric, categorical, datetime."""
        for col in self.df.columns:
            dtype = self.df[col].dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                self.datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

    def _compute_numeric_stats(self) -> None:
        """Compute descriptive statistics for numeric columns."""
        if not self.numeric_cols:
            return
        desc = self.df[self.numeric_cols].describe().T
        desc["skewness"] = self.df[self.numeric_cols].skew()
        desc["kurtosis"] = self.df[self.numeric_cols].kurtosis()
        desc["missing_pct"] = self.df[self.numeric_cols].isna().mean() * 100
        # FIX: orient="index" so column names are top-level keys
        self.profile["numeric"] = desc.round(4).to_dict(orient="index")

    def _compute_categorical_stats(self) -> None:
        """Compute statistics for categorical columns."""
        cat_stats = {}
        for col in self.categorical_cols:
            vc = self.df[col].value_counts()
            cat_stats[col] = {
                "unique_count": int(self.df[col].nunique()),
                "top_categories": vc.head(5).to_dict(),
                "missing_pct": round(self.df[col].isna().mean() * 100, 2),
            }
        self.profile["categorical"] = cat_stats

    def _compute_correlations(self) -> None:
        """Compute Pearson correlation matrix for numeric columns."""
        if len(self.numeric_cols) < 2:
            self.profile["correlations"] = []
            return
        corr = self.df[self.numeric_cols].corr(method="pearson")
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_pairs.append({
                    "col_a": corr.columns[i],
                    "col_b": corr.columns[j],
                    "pearson_r": round(corr.iloc[i, j], 4),
                })
        corr_pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
        self.profile["correlations"] = corr_pairs[:50]

    def _detect_issues(self) -> None:
        """Detect data quality issues."""
        issues = []
        for col in self.df.columns:
            missing = self.df[col].isna().sum()
            if missing > 0:
                issues.append({
                    "column": col,
                    "issue": "missing_values",
                    "count": int(missing),
                    "pct": round(missing / len(self.df) * 100, 2),
                })
            if col in self.numeric_cols:
                outliers = self._detect_outliers_iqr(col)
                if outliers > 0:
                    issues.append({
                        "column": col,
                        "issue": "outliers_iqr",
                        "count": int(outliers),
                        "pct": round(outliers / len(self.df) * 100, 2),
                    })
        self.profile["issues"] = issues

    def _detect_outliers_iqr(self, col: str) -> int:
        """Count outliers using IQR method."""
        s = self.df[col].dropna()
        if len(s) == 0:
            return 0
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return int(((s < lower) | (s > upper)).sum())

    def _summarize_schema(self) -> None:
        """Create a human-readable schema summary for the LLM."""
        lines = [
            f"Dataset shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns",
            f"Numeric columns ({len(self.numeric_cols)}): {', '.join(self.numeric_cols)}",
            f"Categorical columns ({len(self.categorical_cols)}): {', '.join(self.categorical_cols)}",
        ]
        if self.target_col:
            lines.append(f"Target variable: {self.target_col}")
        if self.profile.get("correlations"):
            top_corr = self.profile["correlations"][:3]
            lines.append("Top correlations:")
            for c in top_corr:
                lines.append(f"  - {c['col_a']} <-> {c['col_b']}: r={c['pearson_r']}")
        self.profile["schema_summary"] = "\n".join(lines)

    def get_prompt_context(self) -> str:
        """Return a concise context string for the LLM agent."""
        if not self.profile:
            self.run()
        return self.profile.get("schema_summary", "")

    def to_json(self) -> str:
        """Serialize profile to JSON."""
        # FIX: auto-run if profile is empty
        if not self.profile:
            self.run()
        return json.dumps(self.profile, indent=2, default=str)