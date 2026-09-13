"""Survival analysis for customer churn, in pure Python.

The public surface, in the order you would use it:

    from survival import data, nonparametric, cox, metrics

    cohort = data.weibull_ph(n=800)                       # or any other generator in data.REGIMES
    curve  = nonparametric.kaplan_meier(list(cohort.rows))
    fit    = cox.fit(cohort)                              # Efron ties, model-based standard errors
    checks = cox.test_proportionality(list(cohort.rows), fit)

Read ``nonparametric.naive_churn_rate`` and ``data.immortal_time`` first. Between them they cover the two
errors that account for most wrong churn analysis: dividing churns by customers regardless of how long each
was observed, and using information from after the moment being predicted.
"""

from __future__ import annotations

from . import cox, data, metrics, nonparametric
from .cox import CoxFit, fit, fit_cox, test_proportionality
from .data import Cohort, Interval
from .nonparametric import SurvivalCurve, kaplan_meier, log_rank, restricted_mean

__all__ = [
    "Cohort",
    "CoxFit",
    "Interval",
    "SurvivalCurve",
    "cox",
    "data",
    "fit",
    "fit_cox",
    "kaplan_meier",
    "log_rank",
    "metrics",
    "nonparametric",
    "restricted_mean",
    "test_proportionality",
]
__version__ = "1.0.0"
