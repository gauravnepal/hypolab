"""Statistical hypothesis testing — Pearson, ANOVA, Granger, Chi-square, T-test, OLS.

Every completed test reports its p-value under the same ``p_value`` key together
with an effect size. After ``run_all`` each result also carries a
Benjamini-Hochberg ``q_value``, and ``significant`` is decided on that q-value,
so the verdict accounts for how many hypotheses were tested.
"""

import math
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    # A q-value can never be smaller than the q-value of a smaller p-value.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q.tolist()


def _round_p(p: float) -> float:
    """Round to 6 significant digits, so 3e-45 is not reported as 0.0."""
    return float(f"{float(p):.6g}")


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
    VALID_CORRECTIONS = {"bh", "none"}

    def __init__(self, df: pd.DataFrame, alpha: float = 0.05, correction: str = "bh"):
        if correction not in self.VALID_CORRECTIONS:
            raise ValueError(
                f"Unknown correction '{correction}'; use one of {sorted(self.VALID_CORRECTIONS)}"
            )
        self.df = df.copy()
        self.alpha = alpha
        self.correction = correction
        self.results: List[Dict[str, Any]] = []

    def run_all(self, hypotheses: List[Dict]) -> List[Dict]:
        """Execute all specified hypotheses, then correct for multiple testing."""
        for hyp in hypotheses:
            test_type = str(hyp.get("test_type", "")).lower().strip()
            variables = hyp.get("variables", [])

            if test_type not in self.VALID_TESTS:
                result = {
                    "status": "skipped",
                    "reason": f"Unknown test type: {test_type}",
                }
            elif not isinstance(variables, list):
                result = {"status": "skipped", "reason": "Variables must be a list"}
            else:
                unknown = [v for v in variables if v not in self.df.columns]
                if unknown:
                    # An LLM can propose columns that do not exist in the data.
                    result = {
                        "status": "skipped",
                        "reason": f"Unknown column(s): {', '.join(map(str, unknown))}",
                    }
                else:
                    try:
                        result = self._dispatch(test_type, variables, hyp)
                    except Exception as e:
                        result = {"status": "error", "reason": str(e)}

            # Skipped and error results come back without their context.
            result.setdefault("hypothesis", hyp.get("hypothesis", ""))
            result.setdefault("test_type", test_type)
            self.results.append(result)

        self._apply_correction()
        return self.results

    def _apply_correction(self) -> None:
        """Set ``q_value`` and ``significant`` on every completed test.

        The correction covers everything this tester has run, including earlier
        ``run_all`` calls, so testing in several batches does not avoid it.
        """
        completed = [r for r in self.results if r.get("status") == "completed"]
        if not completed:
            return

        if self.correction == "bh":
            q_values = benjamini_hochberg([r["p_value"] for r in completed])
        else:
            q_values = [None] * len(completed)

        for result, q in zip(completed, q_values):
            uncorrected = bool(result["p_value"] < self.alpha)
            result["significant_uncorrected"] = uncorrected
            result["q_value"] = None if q is None else _round_p(q)
            result["significant"] = uncorrected if q is None else bool(q <= self.alpha)
            result["correction"] = self.correction
            result["n_tests_corrected"] = len(completed)

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

    def _completed(
        self,
        hyp: Dict,
        test_type: str,
        variables: List[str],
        p_value: float,
        notes: Optional[List[str]] = None,
        **fields: Any,
    ) -> Dict:
        """Build a completed result with the keys every test shares."""
        if p_value is None or not math.isfinite(float(p_value)):
            return {
                "status": "skipped",
                "reason": "Test statistic is undefined (no variation in the data?)",
            }
        return {
            "hypothesis": hyp.get("hypothesis", ""),
            "test_type": test_type,
            "variables": variables,
            **fields,
            "p_value": _round_p(p_value),
            "alpha": self.alpha,
            "warnings": notes or [],
            "status": "completed",
        }

    def _clean_numeric(self, col: str) -> pd.Series:
        """Return cleaned numeric series, dropping NaNs."""
        s = pd.to_numeric(self.df[col], errors="coerce").dropna()
        return s

    def _numeric_frame(self, cols: List[str]) -> pd.DataFrame:
        """Rows where every listed column is present and numeric.

        Dropping missing values per column and then lining the columns up by
        position pairs values from different rows; this keeps rows intact.
        """
        return self.df[cols].apply(pd.to_numeric, errors="coerce").dropna()

    @staticmethod
    def _pearson_ci(r: float, n: int, level: float = 0.95) -> Optional[List[float]]:
        """Confidence interval for a correlation via the Fisher z-transform."""
        if n <= 3 or abs(r) >= 1:
            return None
        z = math.atanh(r)
        half_width = stats.norm.ppf(0.5 + level / 2) / math.sqrt(n - 3)
        return [
            round(math.tanh(z - half_width), 4),
            round(math.tanh(z + half_width), 4),
        ]

    @staticmethod
    def _cohens_d(a: pd.Series, b: pd.Series) -> Optional[float]:
        """Standardised mean difference using the pooled standard deviation."""
        na, nb = len(a), len(b)
        pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (
            na + nb - 2
        )
        if not pooled_var > 0:
            return None
        return round(float((a.mean() - b.mean()) / math.sqrt(pooled_var)), 4)

    def _pearson(self, variables: List[str], hyp: Dict) -> Dict:
        """Pearson correlation test on rows where both variables are present."""
        if len(variables) < 2:
            return {"status": "skipped", "reason": "Need 2 numeric variables"}
        col_x, col_y = variables[:2]
        if col_x == col_y:
            return {"status": "skipped", "reason": "Need 2 different variables"}

        pair = self._numeric_frame([col_x, col_y])
        n = len(pair)
        if n < 3:
            return {"status": "skipped", "reason": "Insufficient data after cleaning"}
        if pair[col_x].nunique() < 2 or pair[col_y].nunique() < 2:
            return {
                "status": "skipped",
                "reason": "A variable is constant; correlation is undefined",
            }

        r, pvalue = stats.pearsonr(pair[col_x], pair[col_y])
        r = float(r)
        return self._completed(
            hyp,
            "pearson_correlation",
            variables[:2],
            pvalue,
            pearson_r=round(r, 4),
            ci_95=self._pearson_ci(r, n),
            effect_size=round(r, 4),
            effect_size_name="pearson_r",
            n=n,
        )

    def _anova(self, variables: List[str], hyp: Dict) -> Dict:
        """One-way ANOVA: numeric ~ categorical."""
        if len(variables) < 2:
            return {
                "status": "skipped",
                "reason": "Need numeric + categorical variables",
            }

        numeric_col = variables[0]
        cat_col = variables[1]
        if numeric_col == cat_col:
            return {"status": "skipped", "reason": "Need 2 different variables"}

        df_clean = self.df[[numeric_col, cat_col]].copy()
        df_clean[numeric_col] = pd.to_numeric(df_clean[numeric_col], errors="coerce")
        df_clean = df_clean.dropna()

        groups = [
            group[numeric_col].to_numpy(dtype=float)
            for _, group in df_clean.groupby(cat_col)
        ]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            return {"status": "skipped", "reason": "Need at least 2 groups with data"}

        f_stat, pvalue = stats.f_oneway(*groups)

        values = np.concatenate(groups)
        grand_mean = values.mean()
        ss_total = float(((values - grand_mean) ** 2).sum())
        ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups))
        eta_squared = round(ss_between / ss_total, 4) if ss_total > 0 else None

        return self._completed(
            hyp,
            "anova",
            variables[:2],
            pvalue,
            f_statistic=round(float(f_stat), 4),
            eta_squared=eta_squared,
            effect_size=eta_squared,
            effect_size_name="eta_squared",
            n_groups=len(groups),
            n=len(values),
        )

    def _chi_square(self, variables: List[str], hyp: Dict) -> Dict:
        """Chi-square test of independence."""
        if len(variables) < 2:
            return {"status": "skipped", "reason": "Need 2 categorical variables"}
        if variables[0] == variables[1]:
            return {"status": "skipped", "reason": "Need 2 different variables"}

        df_clean = self.df[variables[:2]].dropna()
        contingency = pd.crosstab(df_clean[variables[0]], df_clean[variables[1]])

        if contingency.size == 0:
            return {"status": "skipped", "reason": "Empty contingency table"}
        if min(contingency.shape) < 2:
            return {
                "status": "skipped",
                "reason": "Each variable needs at least 2 categories",
            }

        chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)
        n = int(contingency.to_numpy().sum())
        cramers_v = round(math.sqrt(chi2 / (n * (min(contingency.shape) - 1))), 4)

        notes = []
        low_share = float((expected < 5).mean())
        if low_share > 0.2:
            notes.append(
                f"{low_share:.0%} of cells have an expected count below 5; "
                "the chi-square approximation may be unreliable"
            )

        return self._completed(
            hyp,
            "chi_square",
            variables[:2],
            pvalue,
            notes=notes,
            chi2=round(float(chi2), 4),
            dof=int(dof),
            cramers_v=cramers_v,
            effect_size=cramers_v,
            effect_size_name="cramers_v",
            n=n,
        )

    def _t_test(self, variables: List[str], hyp: Dict) -> Dict:
        """Independent two-sample t-test (Welch)."""
        if len(variables) < 2:
            return {"status": "skipped", "reason": "Need 2 variables"}

        numeric_col = variables[0]
        group_col = variables[1]
        if numeric_col == group_col:
            return {"status": "skipped", "reason": "Need 2 different variables"}

        df_clean = self.df[[numeric_col, group_col]].copy()
        df_clean[numeric_col] = pd.to_numeric(df_clean[numeric_col], errors="coerce")
        df_clean = df_clean.dropna()

        notes = []
        unique_groups = df_clean[group_col].unique()
        if len(unique_groups) == 2:
            group_a = df_clean[df_clean[group_col] == unique_groups[0]][numeric_col]
            group_b = df_clean[df_clean[group_col] == unique_groups[1]][numeric_col]
            labels = [str(unique_groups[0]), str(unique_groups[1])]
        else:
            # No two-level grouping column: compare the two columns as samples.
            group_a = self._clean_numeric(variables[0])
            group_b = self._clean_numeric(variables[1])
            labels = [str(variables[0]), str(variables[1])]
            notes.append(
                "Compared two numeric columns as independent samples; if the "
                "rows are paired, a paired test is the appropriate one"
            )

        if len(group_a) < 2 or len(group_b) < 2:
            return {
                "status": "skipped",
                "reason": (
                    "Need 2 groups/samples "
                    f"('{group_col}' has {len(unique_groups)} distinct values)"
                ),
            }

        t_stat, pvalue = stats.ttest_ind(group_a, group_b, equal_var=False)
        cohens_d = self._cohens_d(group_a, group_b)

        return self._completed(
            hyp,
            "t_test",
            variables[:2],
            pvalue,
            notes=notes,
            t_statistic=round(float(t_stat), 4),
            cohens_d=cohens_d,
            effect_size=cohens_d,
            effect_size_name="cohens_d",
            group_sizes={labels[0]: len(group_a), labels[1]: len(group_b)},
        )

    @staticmethod
    def _run_granger(data: pd.DataFrame, maxlag: int) -> Dict:
        """Call statsmodels across versions (``verbose`` was removed in 0.15)."""
        from statsmodels.tsa.stattools import grangercausalitytests

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return grangercausalitytests(data, maxlag=maxlag, verbose=False)
            except TypeError:
                return grangercausalitytests(data, maxlag=maxlag)

    def _granger(self, variables: List[str], hyp: Dict) -> Dict:
        """Granger causality test (requires statsmodels).

        Rows are assumed to be in time order. Tests whether the second variable
        helps predict the first.
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            return {"status": "skipped", "reason": "statsmodels not installed"}

        if len(variables) < 2:
            return {
                "status": "skipped",
                "reason": "Need 2 numeric time-series variables",
            }
        if variables[0] == variables[1]:
            return {"status": "skipped", "reason": "Need 2 different variables"}

        df_clean = self._numeric_frame(variables[:2])
        if len(df_clean) < 10:
            return {"status": "skipped", "reason": "Insufficient time-series data"}

        notes = []
        for col in variables[:2]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adf_p = adfuller(df_clean[col])[1]
            except Exception:
                continue
            if adf_p > 0.05:
                notes.append(
                    f"{col} may be non-stationary (ADF p={adf_p:.3f}); "
                    "Granger results on non-stationary series can be spurious"
                )

        maxlag = min(4, len(df_clean) // 5)
        try:
            gc_result = self._run_granger(df_clean, maxlag)
        except Exception as e:
            return {"status": "error", "reason": f"Granger test failed: {e}"}

        pvalues = {
            lag: _round_p(gc_result[lag][0]["ssr_ftest"][1])
            for lag in range(1, maxlag + 1)
        }
        best_lag = min(pvalues, key=pvalues.get)
        min_p = pvalues[best_lag]
        # The best of several lags is reported, so correct for the lags tried.
        lag_corrected_p = min(1.0, min_p * maxlag)

        return self._completed(
            hyp,
            "granger_causality",
            variables[:2],
            lag_corrected_p,
            notes=notes,
            best_lag=best_lag,
            min_p_value=min_p,
            p_values_by_lag=pvalues,
            lags_tested=maxlag,
            n=len(df_clean),
        )

    def _regression(self, variables: List[str], hyp: Dict) -> Dict:
        """Simple OLS regression (first var = target, rest = predictors)."""
        try:
            import statsmodels.api as sm
        except ImportError:
            return {"status": "skipped", "reason": "statsmodels not installed"}

        if len(variables) < 2:
            return {"status": "skipped", "reason": "Need at least 2 variables"}

        target = variables[0]
        predictors = [v for v in dict.fromkeys(variables[1:]) if v != target]
        if not predictors:
            return {"status": "skipped", "reason": "Need at least 1 predictor"}

        df_clean = self._numeric_frame([target] + predictors)
        if len(df_clean) < max(3, len(predictors) + 2):
            return {"status": "skipped", "reason": "Insufficient data"}

        y = df_clean[target]
        X = df_clean[predictors]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        r_squared = round(float(model.rsquared), 4)

        return self._completed(
            hyp,
            "regression",
            variables,
            model.f_pvalue,
            r_squared=r_squared,
            adj_r_squared=round(float(model.rsquared_adj), 4),
            f_statistic=round(float(model.fvalue), 4),
            f_pvalue=_round_p(model.f_pvalue),
            coefficients={
                k: round(float(v), 4) for k, v in model.params.to_dict().items()
            },
            coefficient_p_values={
                k: _round_p(v) for k, v in model.pvalues.to_dict().items()
            },
            effect_size=r_squared,
            effect_size_name="r_squared",
            n=len(df_clean),
        )

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate summary of all test results."""
        completed = [r for r in self.results if r.get("status") == "completed"]
        significant = [r for r in completed if r.get("significant")]
        uncorrected = [r for r in completed if r.get("significant_uncorrected")]
        return {
            "total_hypotheses": len(self.results),
            "completed": len(completed),
            "significant": len(significant),
            "significant_uncorrected": len(uncorrected),
            "insignificant": len(completed) - len(significant),
            "skipped": len([r for r in self.results if r.get("status") == "skipped"]),
            "errors": len([r for r in self.results if r.get("status") == "error"]),
            "correction": self.correction,
            "significant_tests": significant,
        }
