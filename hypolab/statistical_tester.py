"""Statistical hypothesis testing — Pearson, ANOVA, Granger, Chi-square, T-test."""

from typing import Any, Dict, List

import pandas as pd
from scipy import stats


class StatisticalTester:
    """Runs statistical tests based on hypothesis specifications."""

    VALID_TESTS = {
        "pearson_correlation",
        "anova",
        "chi_square",
        "t_test",
        "granger_causality",
        "regression",
    }

    def __init__(self, df: pd.DataFrame, alpha: float = 0.05):
        self.df = df.copy()
        self.alpha = alpha
        self.results: List[Dict[str, Any]] = []

    def run_all(self, hypotheses: List[Dict]) -> List[Dict]:
        """Execute all specified hypotheses."""
        for hyp in hypotheses:
            test_type = hyp.get("test_type", "").lower().strip()
            variables = hyp.get("variables", [])
            if test_type not in self.VALID_TESTS:
                self.results.append(
                    {
                        "hypothesis": hyp.get("hypothesis", ""),
                        "test_type": test_type,
                        "status": "skipped",
                        "reason": f"Unknown test type: {test_type}",
                    }
                )
                continue

            try:
                result = self._dispatch(test_type, variables, hyp)
                self.results.append(result)
            except Exception as e:
                self.results.append(
                    {
                        "hypothesis": hyp.get("hypothesis", ""),
                        "test_type": test_type,
                        "status": "error",
                        "reason": str(e),
                    }
                )
        return self.results

    def _dispatch(self, test_type: str, variables: List[str], hyp: Dict) -> Dict:
        """Route to appropriate test method."""
        dispatch_map = {
            "pearson_correlation": self._pearson,
            "anova": self._anova,
            "chi_square": self._chi_square,
            "t_test": self._t_test,
            "granger_causality": self._granger,
            "regression": self._regression,
        }
        return dispatch_map[test_type](variables, hyp)

    def _clean_numeric(self, col: str) -> pd.Series:
        """Return cleaned numeric series, dropping NaNs."""
        s = pd.to_numeric(self.df[col], errors="coerce").dropna()
        return s

    def _pearson(self, variables: List[str], hyp: Dict) -> Dict:
        """Pearson correlation test."""
        if len(variables) < 2:
            return {"status": "skipped", "reason": "Need 2 numeric variables"}
        x = self._clean_numeric(variables[0])
        y = self._clean_numeric(variables[1])
        min_len = min(len(x), len(y))
        x, y = x.iloc[:min_len], y.iloc[:min_len]

        if min_len < 3:
            return {"status": "skipped", "reason": "Insufficient data after cleaning"}

        r, pvalue = stats.pearsonr(x, y)
        return {
            "hypothesis": hyp.get("hypothesis", ""),
            "test_type": "pearson_correlation",
            "variables": variables[:2],
            "pearson_r": round(r, 4),
            "p_value": round(pvalue, 6),
            "significant": pvalue < self.alpha,
            "alpha": self.alpha,
            "n": min_len,
            "status": "completed",
        }

    def _anova(self, variables: List[str], hyp: Dict) -> Dict:
        """One-way ANOVA: numeric ~ categorical."""
        if len(variables) < 2:
            return {
                "status": "skipped",
                "reason": "Need numeric + categorical variables",
            }

        numeric_col = variables[0]
        cat_col = variables[1]

        df_clean = self.df[[numeric_col, cat_col]].dropna()
        df_clean[numeric_col] = pd.to_numeric(df_clean[numeric_col], errors="coerce")
        df_clean = df_clean.dropna()

        groups = [group[numeric_col].values for _, group in df_clean.groupby(cat_col)]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            return {"status": "skipped", "reason": "Need at least 2 groups with data"}

        f_stat, pvalue = stats.f_oneway(*groups)
        return {
            "hypothesis": hyp.get("hypothesis", ""),
            "test_type": "anova",
            "variables": variables[:2],
            "f_statistic": round(f_stat, 4),
            "p_value": round(pvalue, 6),
            "significant": pvalue < self.alpha,
            "alpha": self.alpha,
            "n_groups": len(groups),
            "status": "completed",
        }

    def _chi_square(self, variables: List[str], hyp: Dict) -> Dict:
        """Chi-square test of independence."""
        if len(variables) < 2:
            return {"status": "skipped", "reason": "Need 2 categorical variables"}

        df_clean = self.df[variables[:2]].dropna()
        contingency = pd.crosstab(df_clean[variables[0]], df_clean[variables[1]])

        if contingency.size == 0:
            return {"status": "skipped", "reason": "Empty contingency table"}

        chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)
        return {
            "hypothesis": hyp.get("hypothesis", ""),
            "test_type": "chi_square",
            "variables": variables[:2],
            "chi2": round(chi2, 4),
            "p_value": round(pvalue, 6),
            "dof": dof,
            "significant": pvalue < self.alpha,
            "alpha": self.alpha,
            "status": "completed",
        }

    def _t_test(self, variables: List[str], hyp: Dict) -> Dict:
        """Independent two-sample t-test."""
        if len(variables) < 2:
            return {"status": "skipped", "reason": "Need 2 variables"}

        numeric_col = variables[0]
        group_col = variables[1]

        df_clean = self.df[[numeric_col, group_col]].dropna()
        df_clean[numeric_col] = pd.to_numeric(df_clean[numeric_col], errors="coerce")
        df_clean = df_clean.dropna()

        unique_groups = df_clean[group_col].unique()
        if len(unique_groups) != 2:
            x = self._clean_numeric(variables[0])
            y = self._clean_numeric(variables[1])
            if len(x) < 2 or len(y) < 2:
                return {"status": "skipped", "reason": "Need 2 groups/samples"}
            t_stat, pvalue = stats.ttest_ind(x, y, equal_var=False)
            return {
                "hypothesis": hyp.get("hypothesis", ""),
                "test_type": "t_test",
                "variables": variables[:2],
                "t_statistic": round(t_stat, 4),
                "p_value": round(pvalue, 6),
                "significant": pvalue < self.alpha,
                "alpha": self.alpha,
                "status": "completed",
            }

        group_a = df_clean[df_clean[group_col] == unique_groups[0]][numeric_col]
        group_b = df_clean[df_clean[group_col] == unique_groups[1]][numeric_col]

        t_stat, pvalue = stats.ttest_ind(group_a, group_b, equal_var=False)
        return {
            "hypothesis": hyp.get("hypothesis", ""),
            "test_type": "t_test",
            "variables": variables[:2],
            "t_statistic": round(t_stat, 4),
            "p_value": round(pvalue, 6),
            "significant": pvalue < self.alpha,
            "alpha": self.alpha,
            "status": "completed",
        }

    def _granger(self, variables: List[str], hyp: Dict) -> Dict:
        """Granger causality test (requires statsmodels)."""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            return {"status": "skipped", "reason": "statsmodels not installed"}

        if len(variables) < 2:
            return {
                "status": "skipped",
                "reason": "Need 2 numeric time-series variables",
            }

        df_clean = self.df[variables[:2]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df_clean) < 10:
            return {"status": "skipped", "reason": "Insufficient time-series data"}

        maxlag = min(4, len(df_clean) // 5)
        try:
            gc_result = grangercausalitytests(df_clean, maxlag=maxlag, verbose=False)
            pvalues = {
                lag: round(gc_result[lag][0]["ssr_ftest"][1], 6)
                for lag in range(1, maxlag + 1)
            }
            min_p = min(pvalues.values())
            best_lag = min(pvalues, key=pvalues.get)

            return {
                "hypothesis": hyp.get("hypothesis", ""),
                "test_type": "granger_causality",
                "variables": variables[:2],
                "best_lag": best_lag,
                "min_p_value": min_p,
                "p_values_by_lag": pvalues,
                "significant": min_p < self.alpha,
                "alpha": self.alpha,
                "status": "completed",
            }
        except Exception as e:
            return {"status": "error", "reason": f"Granger test failed: {e}"}

    def _regression(self, variables: List[str], hyp: Dict) -> Dict:
        """Simple OLS regression (first var = target, rest = predictors)."""
        try:
            import statsmodels.api as sm
        except ImportError:
            return {"status": "skipped", "reason": "statsmodels not installed"}

        if len(variables) < 2:
            return {"status": "skipped", "reason": "Need at least 2 variables"}

        target = variables[0]
        predictors = variables[1:]

        df_clean = (
            self.df[[target] + predictors]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        if len(df_clean) < 3:
            return {"status": "skipped", "reason": "Insufficient data"}

        y = df_clean[target]
        X = df_clean[predictors]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        return {
            "hypothesis": hyp.get("hypothesis", ""),
            "test_type": "regression",
            "variables": variables,
            "r_squared": round(model.rsquared, 4),
            "adj_r_squared": round(model.rsquared_adj, 4),
            "f_statistic": round(model.fvalue, 4),
            "f_pvalue": round(model.f_pvalue, 6),
            "significant": model.f_pvalue < self.alpha,
            "alpha": self.alpha,
            "coefficients": {k: round(v, 4) for k, v in model.params.to_dict().items()},
            "status": "completed",
        }

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate summary of all test results."""
        completed = [r for r in self.results if r.get("status") == "completed"]
        significant = [r for r in completed if r.get("significant")]
        return {
            "total_hypotheses": len(self.results),
            "completed": len(completed),
            "significant": len(significant),
            "insignificant": len(completed) - len(significant),
            "skipped": len([r for r in self.results if r.get("status") == "skipped"]),
            "errors": len([r for r in self.results if r.get("status") == "error"]),
            "significant_tests": significant,
        }
