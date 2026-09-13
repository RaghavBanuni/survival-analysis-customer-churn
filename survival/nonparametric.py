"""Non-parametric survival: Kaplan-Meier, Nelson-Aalen, log-rank, RMST -- and the naive churn rate.

Start with the arithmetic everyone gets wrong. "Churn rate" is usually computed as customers who churned
divided by customers observed, which treats a customer signed up three weeks ago and still subscribed as
evidence of retention on the same footing as one observed for three years. It is not a survival estimate of
anything; it is a mixture of survival and observation windows, and it is biased downward by however much of
the cohort is still young. ``naive_churn_rate`` computes it so it can be compared against Kaplan-Meier at a
fixed horizon, which is the comparison that shows the size of the error.

The estimators here are the standard ones, written out:

    S(t)   = prod over event times t_i <= t of (1 - d_i/n_i)                Kaplan-Meier
    H(t)   = sum over event times t_i <= t of d_i/n_i                      Nelson-Aalen
    Var(S) = S(t)^2 * sum d_i / (n_i (n_i - d_i))                          Greenwood

with ``n_i`` the number **at risk** at ``t_i``, which is where left truncation enters: a subject is at risk
only on ``(start, stop]``, so ``n_i = #{rows : start_j < t_i <= stop_j}``. Getting that wrong is the quiet
version of the same error as the naive churn rate.

Confidence intervals are computed on the complementary log-log scale, ``log(-log S)``, not on ``S`` itself.
A symmetric interval on ``S`` runs outside ``[0, 1]`` near the tails and is visibly wrong wherever it
matters most; the log-log interval cannot, because it is transformed back through a monotone map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data import Cohort, Interval

Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------------------------
# distribution tails, needed for the tests and the p-values
# ---------------------------------------------------------------------------------------------


def normal_sf(z: float) -> float:
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def chi2_sf(x: float, df: int = 1) -> float:
    """Upper tail of the chi-square distribution: ``Q(df/2, x/2)``.

    Implemented as the regularised upper incomplete gamma, by series below the transition point and by the
    Lentz continued fraction above it. For ``df = 1`` this reduces to ``erfc(sqrt(x/2))``, which is used as
    a check in the tests -- an incomplete gamma written from scratch deserves an independent comparison.
    """
    if x < 0.0:
        raise ValueError("chi-square statistic cannot be negative")
    if df < 1:
        raise ValueError("degrees of freedom must be at least 1")
    if x == 0.0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    if z < a + 1.0:
        # lower series, then complement
        total = 1.0 / a
        term = total
        for index in range(1, 500):
            term *= z / (a + index)
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        lower = total * math.exp(-z + a * math.log(z) - math.lgamma(a))
        return max(0.0, 1.0 - lower)

    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for index in range(1, 500):
        an = -index * (index - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h * math.exp(-z + a * math.log(z) - math.lgamma(a))


# ---------------------------------------------------------------------------------------------
# risk sets
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskPoint:
    """One distinct event time, with the risk set and the number of events there."""

    time: float
    at_risk: int
    events: int


def risk_table(rows: "list[Interval]") -> list[RiskPoint]:
    """Event times with their risk sets, honouring ``(start, stop]`` observation windows.

    A subject is at risk at ``t`` if ``start < t <= stop``. The left-open convention matters: a subject
    entering at exactly ``t`` is not yet at risk there, and one whose window ends at ``t`` still is.
    """
    times = sorted({row.stop for row in rows if row.event})
    table: list[RiskPoint] = []
    for time in times:
        at_risk = sum(1 for row in rows if row.start < time <= row.stop)
        events = sum(1 for row in rows if row.event and row.stop == time)
        if at_risk <= 0:
            raise ValueError(f"an event at t={time} with nobody at risk: the data is inconsistent")
        table.append(RiskPoint(time=time, at_risk=at_risk, events=events))
    return table


# ---------------------------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------------------------


@dataclass
class SurvivalCurve:
    """A step function with pointwise complementary log-log confidence limits."""

    times: list[float]
    survival: list[float]
    lower: list[float]
    upper: list[float]
    at_risk: list[int]
    events: list[int]
    variance_terms: list[float]  # cumulative sum of d/(n(n-d)), the Greenwood factor

    def at(self, time: float) -> float:
        """``S(time)``: the last step at or before ``time``. ``S(t) = 1`` before the first event."""
        value = 1.0
        for point, survival in zip(self.times, self.survival):
            if point <= time:
                value = survival
            else:
                break
        return value

    def interval_at(self, time: float) -> tuple[float, float]:
        low, high = 1.0, 1.0
        for point, lower, upper in zip(self.times, self.lower, self.upper):
            if point <= time:
                low, high = lower, upper
            else:
                break
        return low, high

    def median(self) -> float | None:
        """First time at which survival falls to 0.5 or below, or ``None`` if it never does.

        Reporting "median lifetime: 34 weeks" when the curve never reaches 0.5 is one of the more common
        ways a churn deck misleads; the honest answer is that the median is not yet estimable.
        """
        for time, survival in zip(self.times, self.survival):
            if survival <= 0.5:
                return time
        return None

    def median_interval(self) -> tuple[float | None, float | None]:
        """Brookmeyer-Crowley interval: the set of times whose confidence band contains 0.5."""
        low: float | None = None
        high: float | None = None
        for time, lower, upper in zip(self.times, self.lower, self.upper):
            if upper <= 0.5 and low is None:
                low = time
            if lower <= 0.5 and high is None:
                high = time
        return low, high

    def quantile(self, probability: float) -> float | None:
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must lie strictly inside (0, 1)")
        target = 1.0 - probability
        for time, survival in zip(self.times, self.survival):
            if survival <= target:
                return time
        return None

    def table(self, limit: int = 12) -> str:
        header = f"{'time':>8}{'at risk':>10}{'events':>9}{'S(t)':>10}{'95% CI':>22}"
        lines = [header, "-" * len(header)]
        stride = max(len(self.times) // limit, 1)
        for index in range(0, len(self.times), stride):
            lines.append(
                f"{self.times[index]:>8.1f}{self.at_risk[index]:>10}{self.events[index]:>9}"
                f"{self.survival[index]:>10.4f}"
                f"{f'{self.lower[index]:.4f} - {self.upper[index]:.4f}':>22}"
            )
        return "\n".join(lines)


def kaplan_meier(rows: "list[Interval]", z: float = Z_95) -> SurvivalCurve:
    """The product-limit estimator with Greenwood variance and log-log confidence limits."""
    table = risk_table(rows)
    times: list[float] = []
    survival: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    at_risk: list[int] = []
    events: list[int] = []
    terms: list[float] = []

    running = 1.0
    accumulated = 0.0
    for point in table:
        running *= 1.0 - point.events / point.at_risk
        if point.at_risk > point.events:
            accumulated += point.events / (point.at_risk * (point.at_risk - point.events))
        times.append(point.time)
        survival.append(running)
        at_risk.append(point.at_risk)
        events.append(point.events)
        terms.append(accumulated)

        if running <= 0.0 or running >= 1.0 or accumulated <= 0.0:
            # At S = 0 or S = 1 the log-log transform is undefined; the honest interval is degenerate.
            lower.append(max(running, 0.0))
            upper.append(min(running, 1.0))
            continue
        # Var(log(-log S)) = accumulated / (log S)^2, then map back through S^exp(+-z*se).
        standard_error = math.sqrt(accumulated) / abs(math.log(running))
        lower.append(running ** math.exp(z * standard_error))
        upper.append(running ** math.exp(-z * standard_error))

    return SurvivalCurve(times, survival, lower, upper, at_risk, events, terms)


def nelson_aalen(rows: "list[Interval]") -> tuple[list[float], list[float], list[float]]:
    """Cumulative hazard ``H(t) = sum d_i/n_i`` with its variance ``sum d_i/n_i^2``.

    Reported alongside Kaplan-Meier because the two answer different questions: ``S`` is "how many are
    left", ``H`` is "how fast are they leaving". A straight line in ``H`` means a constant hazard, and
    curvature is visible there long before it is visible in ``S``.
    """
    times: list[float] = []
    hazard: list[float] = []
    variance: list[float] = []
    running = 0.0
    accumulated = 0.0
    for point in risk_table(rows):
        running += point.events / point.at_risk
        accumulated += point.events / (point.at_risk**2)
        times.append(point.time)
        hazard.append(running)
        variance.append(accumulated)
    return times, hazard, variance


# ---------------------------------------------------------------------------------------------
# the naive comparison
# ---------------------------------------------------------------------------------------------


def naive_churn_rate(cohort: Cohort) -> float:
    """Churned divided by observed: the number in most dashboards, and it answers no question.

    It mixes survival with observation length. A cohort half of which joined last month will look loyal,
    and the same cohort a year later will look disloyal, with no change in behaviour.
    """
    pairs = cohort.observed()
    return sum(1 for _, event in pairs if event) / len(pairs)


def naive_versus_km(cohort: Cohort, horizon: float) -> tuple[float, float, float]:
    """``(naive churn fraction, Kaplan-Meier churn by horizon, absolute gap)``.

    Kaplan-Meier answers "what fraction churn within ``horizon``, given what we observed"; the naive rate
    answers nothing in particular. The gap grows with the censoring rate, which is why it is worst in
    exactly the fast-growing cohorts a business cares most about.
    """
    curve = kaplan_meier(list(cohort.rows))
    return naive_churn_rate(cohort), 1.0 - curve.at(horizon), abs(
        (1.0 - curve.at(horizon)) - naive_churn_rate(cohort)
    )


# ---------------------------------------------------------------------------------------------
# restricted mean survival time
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RMST:
    """Area under the survival curve up to ``tau``: expected lifetime within the observed horizon."""

    tau: float
    value: float
    standard_error: float

    @property
    def interval(self) -> tuple[float, float]:
        return (
            self.value - Z_95 * self.standard_error,
            self.value + Z_95 * self.standard_error,
        )

    def summary(self) -> str:
        low, high = self.interval
        return f"RMST to {self.tau:g}: {self.value:.2f} (95% CI {low:.2f} - {high:.2f})"


def restricted_mean(curve: SurvivalCurve, tau: float) -> RMST:
    """``integral of S`` to ``tau``, with the Klein-Moeschberger variance.

    RMST is the estimate to prefer when hazards are not proportional. It needs no proportionality
    assumption, it is always estimable (unlike the median, which requires the curve to reach 0.5), and its
    units are the ones a business already uses: expected weeks of subscription within the first year.
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive")

    area = 0.0
    previous_time = 0.0
    previous_survival = 1.0
    kept: list[tuple[float, float]] = []  # (event time, S just after) inside tau
    for time, survival in zip(curve.times, curve.survival):
        if time > tau:
            break
        area += previous_survival * (time - previous_time)
        kept.append((time, survival))
        previous_time, previous_survival = time, survival
    area += previous_survival * (tau - previous_time)

    # Var = sum over event times of [area from t_i to tau]^2 * d_i / (n_i (n_i - d_i))
    variance = 0.0
    for index, (time, _) in enumerate(kept):
        tail = 0.0
        current_time = time
        current_survival = curve.survival[index]
        for later_time, later_survival in zip(curve.times[index + 1 :], curve.survival[index + 1 :]):
            if later_time > tau:
                break
            tail += current_survival * (later_time - current_time)
            current_time, current_survival = later_time, later_survival
        tail += current_survival * (tau - current_time)

        at_risk = curve.at_risk[index]
        events = curve.events[index]
        if at_risk > events:
            variance += tail * tail * events / (at_risk * (at_risk - events))
    return RMST(tau=tau, value=area, standard_error=math.sqrt(variance))


# ---------------------------------------------------------------------------------------------
# the log-rank test
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogRank:
    statistic: float
    p_value: float
    observed: float
    expected: float
    strata: int = 1

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        detail = f" across {self.strata} strata" if self.strata > 1 else ""
        return (
            f"log-rank chi-square {self.statistic:.3f} on 1 df, p = {self.p_value:.4g}{detail} "
            f"(observed {self.observed:.1f} vs expected {self.expected:.1f} events in group 1)"
        )


def log_rank(rows_a: "list[Interval]", rows_b: "list[Interval]") -> LogRank:
    """Two-sample log-rank: compare observed with expected events under equal hazards.

    At each event time the number of group-A events is hypergeometric under the null, so summing
    ``O - E`` and its variance gives a statistic that is chi-square with one degree of freedom. Note what
    it is *not*: a comparison of survival at the end of follow-up. It weights every event time equally,
    which makes it powerful against proportional differences and nearly powerless against crossing
    hazards -- where the early and late contributions cancel. ``python -m survival ph`` shows that failure.
    """
    combined = list(rows_a) + list(rows_b)
    times = sorted({row.stop for row in combined if row.event})
    observed = 0.0
    expected = 0.0
    variance = 0.0

    for time in times:
        at_risk_a = sum(1 for row in rows_a if row.start < time <= row.stop)
        at_risk_b = sum(1 for row in rows_b if row.start < time <= row.stop)
        at_risk = at_risk_a + at_risk_b
        if at_risk < 2:
            continue
        events_a = sum(1 for row in rows_a if row.event and row.stop == time)
        events = events_a + sum(1 for row in rows_b if row.event and row.stop == time)
        if events == 0:
            continue
        observed += events_a
        expected += events * at_risk_a / at_risk
        if at_risk > 1:
            variance += (
                events
                * (at_risk_a / at_risk)
                * (at_risk_b / at_risk)
                * (at_risk - events)
                / (at_risk - 1)
            )

    if variance <= 0.0:
        return LogRank(statistic=0.0, p_value=1.0, observed=observed, expected=expected)
    statistic = (observed - expected) ** 2 / variance
    return LogRank(
        statistic=statistic,
        p_value=chi2_sf(statistic, 1),
        observed=observed,
        expected=expected,
    )


def stratified_log_rank(strata: "list[tuple[list[Interval], list[Interval]]]") -> LogRank:
    """Sum ``O - E`` and its variance across strata before forming the statistic.

    This is how to compare two groups while allowing every stratum its own baseline hazard -- the
    non-parametric analogue of a stratified Cox model. Pooling the strata first would let a difference in
    stratum composition masquerade as a difference in survival, which is Simpson's paradox with censoring.
    """
    if not strata:
        raise ValueError("no strata supplied")
    observed = 0.0
    expected = 0.0
    variance = 0.0
    for rows_a, rows_b in strata:
        single = log_rank(rows_a, rows_b)
        observed += single.observed
        expected += single.expected
        if single.statistic > 0.0:
            variance += (single.observed - single.expected) ** 2 / single.statistic
    if variance <= 0.0:
        return LogRank(0.0, 1.0, observed, expected, strata=len(strata))
    statistic = (observed - expected) ** 2 / variance
    return LogRank(
        statistic=statistic,
        p_value=chi2_sf(statistic, 1),
        observed=observed,
        expected=expected,
        strata=len(strata),
    )


def split_by(cohort: Cohort, index: int) -> "tuple[list[Interval], list[Interval]]":
    """Split a cohort's rows on a binary covariate, keeping every row of a subject together."""
    baseline = {}
    for row in cohort.rows:
        current = baseline.get(row.subject)
        if current is None or row.start < current:
            baseline[row.subject] = row.x[index]
    group_a = [row for row in cohort.rows if baseline[row.subject] > 0.5]
    group_b = [row for row in cohort.rows if baseline[row.subject] <= 0.5]
    return group_a, group_b
