"""Cox proportional hazards: partial likelihood, Efron ties, Newton-Raphson, and the PH check.

Cox's insight is that the baseline hazard need never be estimated to learn the covariate effects. At each
event time, ask only *which* member of the risk set failed, conditional on one of them failing:

    L(beta) = prod over event times of  exp(x_j'beta) / sum over the risk set of exp(x_k'beta)

The baseline hazard cancels from every factor, so a semi-parametric model buys the covariate effects at the
price of saying nothing about the shape of the hazard -- an excellent trade when the question is "which
customers churn faster", and the wrong model when the question is "how many will churn next month".

Three implementation points that decide whether the numbers are right:

**Ties.** Weekly churn data is nothing but ties. Breslow's approximation uses the full risk-set denominator
for every tied event, which double-counts the tied subjects and biases coefficients towards zero -- badly,
when ties are heavy. Efron's correction averages over the orders in which the tied events could have
happened, at negligible cost. Efron is the default here; ``python -m survival ties`` measures the gap on
data where the truth is known, and the difference is not academic.

**Left truncation and time-varying covariates.** Both are the same computation once the data is in
``(start, stop]`` form: the risk set at ``t`` is every row with ``start < t <= stop``, and a subject's
covariates are whatever their row says at that ``t``. No separate code path.

**Clustered standard errors.** A subject contributing many rows contributes many correlated score terms.
The model-based information matrix assumes independence and produces intervals that are too narrow, so the
robust sandwich estimator, with score residuals summed within subject, is available and is what to report
whenever rows outnumber subjects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data import Cohort, Interval
from .nonparametric import Z_95, chi2_sf, normal_sf

Vector = list[float]
Matrix = list[list[float]]


# ---------------------------------------------------------------------------------------------
# small dense linear algebra, enough for a handful of covariates
# ---------------------------------------------------------------------------------------------


def cholesky(matrix: Matrix) -> Matrix:
    """Lower-triangular ``L`` with ``L L' = matrix``; raises if the matrix is not positive definite.

    The failure is informative rather than annoying: the observed information matrix is singular exactly
    when the data cannot identify the coefficients -- a collinear pair of covariates, or a covariate that
    perfectly separates who churned, where the likelihood has no interior maximum and the coefficient
    diverges. Silently regularising that away would hide the problem.
    """
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            total = matrix[row][column] - sum(
                lower[row][k] * lower[column][k] for k in range(column)
            )
            if row == column:
                if total <= 1e-14:
                    raise ValueError(
                        "information matrix is not positive definite: the coefficients are not "
                        "identified (collinear covariates, or a covariate that separates the events)"
                    )
                lower[row][column] = math.sqrt(total)
            else:
                lower[row][column] = total / lower[column][column]
    return lower


def cholesky_solve(matrix: Matrix, vector: Vector) -> Vector:
    lower = cholesky(matrix)
    size = len(vector)
    forward = [0.0] * size
    for row in range(size):
        forward[row] = (
            vector[row] - sum(lower[row][k] * forward[k] for k in range(row))
        ) / lower[row][row]
    solution = [0.0] * size
    for row in reversed(range(size)):
        solution[row] = (
            forward[row] - sum(lower[k][row] * solution[k] for k in range(row + 1, size))
        ) / lower[row][row]
    return solution


def inverse(matrix: Matrix) -> Matrix:
    size = len(matrix)
    columns = []
    for index in range(size):
        unit = [1.0 if position == index else 0.0 for position in range(size)]
        columns.append(cholesky_solve(matrix, unit))
    return [[columns[column][row] for column in range(size)] for row in range(size)]


# ---------------------------------------------------------------------------------------------
# the partial likelihood
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PartialLikelihood:
    value: float
    gradient: Vector
    information: Matrix


def _stratum_rows(rows: "list[Interval]") -> "dict[int, list[Interval]]":
    grouped: dict[int, list[Interval]] = {}
    for row in rows:
        grouped.setdefault(row.stratum, []).append(row)
    return grouped


def partial_likelihood(
    rows: "list[Interval]", beta: Vector, ties: str = "efron"
) -> PartialLikelihood:
    """Log partial likelihood with its score vector and observed information matrix.

    Efron's correction, written out. With ``d`` tied events at one time, risk-set total ``S`` and tied
    total ``S_D``, the contribution is

        sum over tied events of eta_j  -  sum over r = 0..d-1 of log(S - (r/d) S_D)

    The intuition is that after the first of the tied events, the risk set has lost one of them, but which
    one is unknown, so a fraction ``r/d`` of the tied mass is removed on average. Setting every ``r`` term
    to ``r = 0`` recovers Breslow, which is why one loop with a shrinking denominator covers both.

    The information matrix is the negative Hessian:

        I = sum over r of [ Z2_r / D_r  -  (Z_r / D_r)(Z_r / D_r)' ]

    with ``Z_r`` and ``Z2_r`` the correspondingly discounted first and second covariate moments. That form
    makes it manifestly a sum of covariance matrices, hence positive semi-definite, which is why Newton
    steps on this problem are so well behaved.
    """
    if ties not in {"efron", "breslow"}:
        raise ValueError("ties must be 'efron' or 'breslow'")
    width = len(beta)
    total = 0.0
    gradient = [0.0] * width
    information = [[0.0] * width for _ in range(width)]

    for stratum_rows in _stratum_rows(rows).values():
        times = sorted({row.stop for row in stratum_rows if row.event})
        for time in times:
            at_risk = [row for row in stratum_rows if row.start < time <= row.stop]
            tied = [row for row in stratum_rows if row.event and row.stop == time]
            if not at_risk:
                raise ValueError(f"an event at t={time} with an empty risk set")

            risk_sum = 0.0
            risk_first = [0.0] * width
            risk_second = [[0.0] * width for _ in range(width)]
            for row in at_risk:
                weight = math.exp(sum(b * value for b, value in zip(beta, row.x)))
                risk_sum += weight
                for i in range(width):
                    risk_first[i] += weight * row.x[i]
                    for j in range(width):
                        risk_second[i][j] += weight * row.x[i] * row.x[j]

            tied_sum = 0.0
            tied_first = [0.0] * width
            tied_second = [[0.0] * width for _ in range(width)]
            for row in tied:
                linear = sum(b * value for b, value in zip(beta, row.x))
                total += linear
                for i in range(width):
                    gradient[i] += row.x[i]
                weight = math.exp(linear)
                tied_sum += weight
                for i in range(width):
                    tied_first[i] += weight * row.x[i]
                    for j in range(width):
                        tied_second[i][j] += weight * row.x[i] * row.x[j]

            count = len(tied)
            for r in range(count):
                fraction = (r / count) if ties == "efron" else 0.0
                denominator = risk_sum - fraction * tied_sum
                if denominator <= 0.0:
                    raise ValueError("non-positive risk denominator: check for duplicated rows")
                total -= math.log(denominator)
                mean = [
                    (risk_first[i] - fraction * tied_first[i]) / denominator for i in range(width)
                ]
                for i in range(width):
                    gradient[i] -= mean[i]
                    for j in range(width):
                        second = (
                            risk_second[i][j] - fraction * tied_second[i][j]
                        ) / denominator
                        information[i][j] += second - mean[i] * mean[j]
    return PartialLikelihood(value=total, gradient=gradient, information=information)


# ---------------------------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------------------------


@dataclass
class CoxFit:
    """A fitted Cox model, with everything needed to judge it."""

    beta: Vector
    names: tuple[str, ...]
    standard_errors: Vector
    covariance: Matrix
    log_likelihood: float
    null_log_likelihood: float
    iterations: int
    n_rows: int
    n_subjects: int
    n_events: int
    ties: str
    robust: bool
    converged: bool

    @property
    def hazard_ratios(self) -> Vector:
        return [math.exp(value) for value in self.beta]

    def wald(self, index: int) -> tuple[float, float]:
        """``(z, two-sided p)`` for one coefficient."""
        z = self.beta[index] / self.standard_errors[index]
        return z, 2.0 * normal_sf(abs(z))

    def interval(self, index: int, z: float = Z_95) -> tuple[float, float]:
        """Confidence interval for the **hazard ratio**, built on the log scale then exponentiated."""
        half = z * self.standard_errors[index]
        return math.exp(self.beta[index] - half), math.exp(self.beta[index] + half)

    def likelihood_ratio(self) -> tuple[float, float]:
        """Global test against the null model: ``2(l(beta) - l(0))``, chi-square on ``p`` df.

        Preferred over the Wald test when any coefficient is large, where the quadratic approximation the
        Wald test relies on is poor -- and it is the only one of the two that cannot be made arbitrarily
        small by a badly scaled covariate.
        """
        statistic = 2.0 * (self.log_likelihood - self.null_log_likelihood)
        return statistic, chi2_sf(max(statistic, 0.0), len(self.beta))

    def summary(self) -> str:
        header = (
            f"{'covariate':<16}{'coef':>9}{'HR':>9}{'se':>9}{'z':>8}{'p':>10}{'95% CI (HR)':>22}"
        )
        lines = [
            f"Cox proportional hazards ({self.ties} ties, "
            f"{'robust' if self.robust else 'model-based'} se)",
            f"  {self.n_subjects} subjects, {self.n_rows} rows, {self.n_events} events, "
            f"{self.iterations} Newton iterations",
            "",
            header,
            "-" * len(header),
        ]
        for index, name in enumerate(self.names):
            z, p = self.wald(index)
            low, high = self.interval(index)
            lines.append(
                f"{name:<16}{self.beta[index]:>9.4f}{self.hazard_ratios[index]:>9.3f}"
                f"{self.standard_errors[index]:>9.4f}{z:>8.2f}{p:>10.3g}"
                f"{f'{low:.3f} - {high:.3f}':>22}"
            )
        statistic, p_value = self.likelihood_ratio()
        lines.append("")
        lines.append(
            f"likelihood ratio test {statistic:.2f} on {len(self.beta)} df, p = {p_value:.4g}"
        )
        return "\n".join(lines)


def fit_cox(
    rows: "list[Interval]",
    names: tuple[str, ...],
    ties: str = "efron",
    robust: bool = False,
    tolerance: float = 1e-9,
    max_iterations: int = 50,
) -> CoxFit:
    """Newton-Raphson on the partial likelihood, with step halving as the only safeguard needed.

    The partial log-likelihood is concave in ``beta`` (the Hessian is minus a sum of covariance matrices),
    so Newton converges in a handful of iterations from any sensible start and no line search is required
    in practice. Step halving is retained because concavity does not prevent the *first* step from
    overshooting when a covariate is badly scaled, and halving costs nothing when it is not needed.
    """
    width = len(names)
    if width == 0:
        raise ValueError("a Cox model needs at least one covariate")
    for row in rows:
        if len(row.x) != width:
            raise ValueError("covariate width does not match the supplied names")

    beta = [0.0] * width
    null_value = partial_likelihood(rows, beta, ties).value
    current = partial_likelihood(rows, beta, ties)
    iterations = 0
    converged = False

    for iterations in range(1, max_iterations + 1):
        step = cholesky_solve(current.information, current.gradient)
        scale = 1.0
        while True:
            candidate = [b + scale * delta for b, delta in zip(beta, step)]
            try:
                proposed = partial_likelihood(rows, candidate, ties)
            except (OverflowError, ValueError):
                proposed = None
            if proposed is not None and proposed.value >= current.value - 1e-12:
                break
            scale /= 2.0
            if scale < 1e-10:
                raise RuntimeError("the Newton step could not increase the partial likelihood")
        improvement = proposed.value - current.value
        beta, current = candidate, proposed
        if abs(improvement) < tolerance and max(abs(g) for g in current.gradient) < 1e-6:
            converged = True
            break

    covariance = inverse(current.information)
    if robust:
        covariance = _sandwich(rows, beta, covariance)
    errors = [math.sqrt(covariance[index][index]) for index in range(width)]

    return CoxFit(
        beta=beta,
        names=names,
        standard_errors=errors,
        covariance=covariance,
        log_likelihood=current.value,
        null_log_likelihood=null_value,
        iterations=iterations,
        n_rows=len(rows),
        n_subjects=len({row.subject for row in rows}),
        n_events=sum(1 for row in rows if row.event),
        ties=ties,
        robust=robust,
        converged=converged,
    )


def fit(cohort: Cohort, **kwargs) -> CoxFit:
    """Fit every covariate in a cohort."""
    return fit_cox(list(cohort.rows), cohort.names, **kwargs)


def fit_subset(cohort: Cohort, indices: "list[int]", **kwargs) -> CoxFit:
    """Fit using only some covariates -- how the naive and correct encodings are compared."""
    rows = [
        Interval(
            row.subject,
            row.start,
            row.stop,
            row.event,
            tuple(row.x[index] for index in indices),
            row.stratum,
        )
        for row in cohort.rows
    ]
    return fit_cox(rows, tuple(cohort.names[index] for index in indices), **kwargs)


# ---------------------------------------------------------------------------------------------
# residuals: robust variance and the proportionality check
# ---------------------------------------------------------------------------------------------


def score_residuals(rows: "list[Interval]", beta: Vector) -> "dict[int, Vector]":
    """Score residuals summed within subject, using the Breslow risk-set weights.

    For row ``j`` the residual is its own contribution to the score:

        U_j = delta_j (x_j - xbar(t_j))  -  sum over event times t_i <= stop_j, with j at risk,
                                            of (x_j - xbar(t_i)) w_j d_i / S_i

    The first term is what the subject contributed by failing; the second removes what it contributed by
    being available to fail and not doing so. The residuals sum to the score, which is zero at the
    maximum -- and that identity is asserted in the tests, because it is the only cheap check that catches
    an error in this loop.
    """
    width = len(beta)
    residuals: dict[int, Vector] = {}
    for stratum_rows in _stratum_rows(rows).values():
        times = sorted({row.stop for row in stratum_rows if row.event})
        means: dict[float, Vector] = {}
        totals: dict[float, float] = {}
        counts: dict[float, int] = {}
        for time in times:
            at_risk = [row for row in stratum_rows if row.start < time <= row.stop]
            weights = [math.exp(sum(b * v for b, v in zip(beta, row.x))) for row in at_risk]
            total = sum(weights)
            totals[time] = total
            counts[time] = sum(1 for row in stratum_rows if row.event and row.stop == time)
            means[time] = [
                sum(weight * row.x[i] for weight, row in zip(weights, at_risk)) / total
                for i in range(width)
            ]

        for row in stratum_rows:
            accumulated = residuals.setdefault(row.subject, [0.0] * width)
            weight = math.exp(sum(b * v for b, v in zip(beta, row.x)))
            if row.event:
                for i in range(width):
                    accumulated[i] += row.x[i] - means[row.stop][i]
            for time in times:
                if not (row.start < time <= row.stop):
                    continue
                factor = weight * counts[time] / totals[time]
                for i in range(width):
                    accumulated[i] -= (row.x[i] - means[time][i]) * factor
    return residuals


def _sandwich(rows: "list[Interval]", beta: Vector, bread: Matrix) -> Matrix:
    """``I^-1 (sum_subject U U') I^-1``: the cluster-robust covariance.

    Use it whenever a subject contributes several rows, or whenever the proportional hazards assumption is
    doubted. It is consistent for the variance of ``beta`` under misspecification of the *dependence*
    structure, but it cannot rescue a coefficient that means nothing because the hazard ratio changes with
    time. A robust standard error on a meaningless estimate is still meaningless.
    """
    width = len(beta)
    meat = [[0.0] * width for _ in range(width)]
    for residual in score_residuals(rows, beta).values():
        for i in range(width):
            for j in range(width):
                meat[i][j] += residual[i] * residual[j]
    middle = [[sum(bread[i][k] * meat[k][j] for k in range(width)) for j in range(width)] for i in range(width)]
    return [
        [sum(middle[i][k] * bread[k][j] for k in range(width)) for j in range(width)]
        for i in range(width)
    ]


@dataclass(frozen=True)
class ProportionalityTest:
    """Per-covariate Schoenfeld correlation with time, plus a permutation p-value."""

    name: str
    correlation: float
    p_value: float
    n_events: int

    @property
    def violated(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        verdict = "PH VIOLATED" if self.violated else "no evidence against PH"
        return (
            f"{self.name:<16} corr(residual, time) = {self.correlation:+.3f}  "
            f"p = {self.p_value:.4g}  {verdict}"
        )


def schoenfeld_residuals(rows: "list[Interval]", beta: Vector) -> "list[tuple[float, Vector]]":
    """``(event time, x_j - xbar(t_j))`` for each event: the covariate surprise at each failure.

    Under proportional hazards these are noise around zero with no relation to time. If the hazard ratio is
    actually rising, then late failures are increasingly drawn from the high-covariate group and the
    residuals trend upwards. That trend *is* the time-varying coefficient: to first order the scaled
    residual estimates ``beta(t) - beta``, which is why plotting them is more informative than any p-value.
    """
    width = len(beta)
    output: list[tuple[float, Vector]] = []
    for stratum_rows in _stratum_rows(rows).values():
        for time in sorted({row.stop for row in stratum_rows if row.event}):
            at_risk = [row for row in stratum_rows if row.start < time <= row.stop]
            weights = [math.exp(sum(b * v for b, v in zip(beta, row.x))) for row in at_risk]
            total = sum(weights)
            mean = [
                sum(weight * row.x[i] for weight, row in zip(weights, at_risk)) / total
                for i in range(width)
            ]
            for row in stratum_rows:
                if row.event and row.stop == time:
                    output.append((time, [row.x[i] - mean[i] for i in range(width)]))
    return output


def test_proportionality(
    rows: "list[Interval]", fit_result: CoxFit, permutations: int = 2000, seed: int = 0
) -> "list[ProportionalityTest]":
    """Correlate Schoenfeld residuals with event-time rank; get the p-value by permutation.

    Grambsch and Therneau's chi-square test does this with an asymptotic variance built from the
    information matrix. A permutation test is used here instead, and deliberately: under proportional
    hazards the residuals are exchangeable with respect to time, so shuffling the time labels gives the
    null distribution directly, with no asymptotics to get wrong and no covariance algebra to mis-derive.
    It costs a few thousand correlation computations, which at this scale is free.

    Rank of event time rather than raw time, because a single very late churn otherwise dominates the
    correlation. This mirrors the ``identity`` versus ``rank`` transform choice in the standard tools, and
    the choice can change the verdict for borderline cases -- which is a reason to look at the residual
    trend, not only at the p-value.
    """
    import random as _random

    residuals = schoenfeld_residuals(rows, fit_result.beta)
    if len(residuals) < 5:
        raise ValueError("too few events to test proportionality")
    ranks = list(range(len(residuals)))
    rng = _random.Random(seed)

    def correlate(values: Vector, order: "list[int]") -> float:
        n = len(values)
        mean_value = sum(values) / n
        mean_order = sum(order) / n
        covariance = sum((v - mean_value) * (o - mean_order) for v, o in zip(values, order))
        spread_value = math.sqrt(sum((v - mean_value) ** 2 for v in values))
        spread_order = math.sqrt(sum((o - mean_order) ** 2 for o in order))
        if spread_value == 0.0 or spread_order == 0.0:
            return 0.0
        return covariance / (spread_value * spread_order)

    results: list[ProportionalityTest] = []
    for index, name in enumerate(fit_result.names):
        values = [residual[index] for _, residual in residuals]
        observed = correlate(values, ranks)
        extreme = 0
        for _ in range(permutations):
            shuffled = ranks[:]
            rng.shuffle(shuffled)
            if abs(correlate(values, shuffled)) >= abs(observed) - 1e-15:
                extreme += 1
        # add-one smoothing so the p-value can never be reported as exactly zero
        results.append(
            ProportionalityTest(
                name=name,
                correlation=observed,
                p_value=(extreme + 1) / (permutations + 1),
                n_events=len(residuals),
            )
        )
    return results


# ---------------------------------------------------------------------------------------------
# baseline hazard and prediction
# ---------------------------------------------------------------------------------------------


def baseline_cumulative_hazard(
    rows: "list[Interval]", beta: Vector, stratum: int = 0
) -> "list[tuple[float, float]]":
    """Breslow's estimator: ``H0(t) = sum over events of d_i / sum over the risk set of exp(x'beta)``.

    This is the piece Cox's partial likelihood threw away, recovered afterwards so the model can produce a
    survival curve rather than only a hazard ratio. It is a step function, and it is noisier than the
    coefficients -- the coefficients pool information across all event times while each step of ``H0``
    uses one.
    """
    stratum_rows = [row for row in rows if row.stratum == stratum]
    output: list[tuple[float, float]] = []
    running = 0.0
    for time in sorted({row.stop for row in stratum_rows if row.event}):
        at_risk = [row for row in stratum_rows if row.start < time <= row.stop]
        total = sum(math.exp(sum(b * v for b, v in zip(beta, row.x))) for row in at_risk)
        events = sum(1 for row in stratum_rows if row.event and row.stop == time)
        running += events / total
        output.append((time, running))
    return output


def predicted_survival(
    baseline: "list[tuple[float, float]]", beta: Vector, x: "tuple[float, ...]"
) -> "list[tuple[float, float]]":
    """``S(t|x) = exp(-H0(t) * exp(x'beta))``: the proportional hazards assumption, made visible.

    Every predicted curve is the same baseline curve raised to a power. Curves that cross, or a group whose
    curve is steeper early and flatter late, cannot be produced by this model at all -- which is exactly
    why the proportionality check comes before any prediction is trusted.
    """
    factor = math.exp(sum(b * v for b, v in zip(beta, x)))
    return [(time, math.exp(-hazard * factor)) for time, hazard in baseline]
