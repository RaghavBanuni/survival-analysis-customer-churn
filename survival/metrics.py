"""Evaluating a survival model: concordance, IPCW Brier scores, and calibration against Kaplan-Meier.

Ordinary classification metrics do not apply here, and the reason is worth being precise about. Accuracy or
AUC on "churned within 12 weeks" requires knowing, for every customer, whether they churned within 12
weeks -- and for the censored ones nobody knows. Dropping them keeps only customers observed long enough to
churn, which is selection on the outcome; counting them as retained records a guess as data. Both are common
and both bias the reported number upwards, because the discarded or mislabelled customers are exactly the
ambiguous ones.

The three measures here handle censoring explicitly:

* **Concordance (Harrell's C).** Of all pairs whose order is *knowable*, how many does the model rank
  correctly. A pair is comparable only when the one who failed first is known to have failed; two censored
  customers, or a censored customer who left the data before the other's event, contribute nothing.
* **Brier score with inverse-probability-of-censoring weights (Graf et al.).** A proper scoring rule for the
  predicted survival probability at a horizon, where each observation is reweighted by its probability of
  still being under observation. This measures calibration and discrimination together, which is what a
  retention forecast is actually judged on.
* **Calibration against Kaplan-Meier within bins.** Group customers by predicted survival, then compare the
  mean prediction against the Kaplan-Meier estimate inside the group. Comparing against a raw churn
  fraction would reintroduce the censoring bias the model was built to avoid.

Concordance and Brier disagree more often than people expect, and when they do the model is well ranked and
badly scaled -- fine for prioritising a save-team queue, useless for forecasting revenue. Both are reported.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .data import Cohort, Interval
from .nonparametric import kaplan_meier


@dataclass(frozen=True)
class Concordance:
    """Harrell's C with the pair counts that produced it."""

    value: float
    comparable: int
    concordant: float
    tied_predictions: int

    @property
    def usable_fraction(self) -> float:
        return self.comparable and self.concordant / self.comparable

    def summary(self) -> str:
        return (
            f"C-index {self.value:.4f} over {self.comparable} comparable pairs "
            f"({self.tied_predictions} tied predictions counted as half)"
        )


def concordance_index(
    times: "list[float]", events: "list[bool]", risk: "list[float]"
) -> Concordance:
    """Fraction of orderable pairs the risk score ranks correctly; ties in the score count as half.

    ``risk`` is higher-is-sooner-failure -- for a Cox model that is the linear predictor ``x'beta``, and no
    baseline hazard is needed because concordance only depends on the ranking.

    Note the metric's blind spot: a model can score 0.75 while its predicted probabilities are wildly
    miscalibrated, since any monotone transformation of the risk score leaves C untouched. It also degrades
    as censoring grows heavier, because the comparable-pair set shrinks towards the early failures.
    """
    if not (len(times) == len(events) == len(risk)):
        raise ValueError("times, events and risk must have the same length")
    comparable = 0
    concordant = 0.0
    tied = 0
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            # The pair is orderable only if the earlier of the two is a known failure.
            if times[i] < times[j] and events[i]:
                earlier, later = i, j
            elif times[j] < times[i] and events[j]:
                earlier, later = j, i
            elif times[i] == times[j] and events[i] and events[j]:
                continue  # tied failure times carry no ordering information
            else:
                continue
            comparable += 1
            if risk[earlier] > risk[later]:
                concordant += 1.0
            elif risk[earlier] == risk[later]:
                concordant += 0.5
                tied += 1
    if comparable == 0:
        raise ValueError("no comparable pairs: every subject is censored before any event")
    return Concordance(
        value=concordant / comparable,
        comparable=comparable,
        concordant=concordant,
        tied_predictions=tied,
    )


def censoring_curve(times: "list[float]", events: "list[bool]"):
    """Kaplan-Meier of the **censoring** distribution: the reverse-KM estimate of ``G(t)``.

    Flip the event indicator and run the same estimator. This is what makes the weighted Brier score
    possible: the weight for an observation is one over its probability of still being under observation,
    and that probability has to be estimated from the same data.

    The assumption underneath is independent censoring -- that the reason people leave the dataset is
    unrelated to their churn risk. It is an assumption, it is not testable from the data, and where it fails
    (a support outage prompting both churn and account deletion) every method here fails with it.
    """
    rows = [
        Interval(subject=index, start=0.0, stop=time, event=not event, x=(0.0,))
        for index, (time, event) in enumerate(zip(times, events))
    ]
    return kaplan_meier(rows)


@dataclass(frozen=True)
class BrierPoint:
    horizon: float
    score: float
    reference: float  # the same score for a marginal Kaplan-Meier prediction

    @property
    def skill(self) -> float:
        """Fraction of the reference model's error removed. Negative means worse than no model."""
        return 0.0 if self.reference <= 0 else 1.0 - self.score / self.reference


def brier_score(
    times: "list[float]",
    events: "list[bool]",
    predicted_survival: "list[float]",
    horizon: float,
) -> BrierPoint:
    """IPCW Brier score at one horizon, with a marginal Kaplan-Meier model as the reference.

        BS(t) = 1/n * sum [ S(t|x)^2 * 1{T <= t, event} / G(T)  +  (1 - S(t|x))^2 * 1{T > t} / G(t) ]

    Customers censored before ``t`` contribute nothing directly; their information is carried by the weights
    of those who remain. The reference is the population Kaplan-Meier curve, which is the honest baseline: a
    covariate model that cannot beat "everyone churns at the average rate" has earned nothing, and reporting
    a raw Brier score without it hides that.
    """
    n = len(times)
    if not (n == len(events) == len(predicted_survival)):
        raise ValueError("inputs must have the same length")
    censoring = censoring_curve(times, events)
    weight_at_horizon = censoring.at(horizon)
    if weight_at_horizon <= 0.0:
        raise ValueError(f"no observation weight left at horizon {horizon}: choose an earlier horizon")

    marginal = kaplan_meier(
        [Interval(index, 0.0, time, event, (0.0,)) for index, (time, event) in enumerate(zip(times, events))]
    ).at(horizon)

    total = 0.0
    reference = 0.0
    for time, event, survival in zip(times, events, predicted_survival):
        if time <= horizon and event:
            weight = censoring.at(time)
            if weight <= 0.0:
                continue
            total += survival**2 / weight
            reference += marginal**2 / weight
        elif time > horizon:
            total += (1.0 - survival) ** 2 / weight_at_horizon
            reference += (1.0 - marginal) ** 2 / weight_at_horizon
    return BrierPoint(horizon=horizon, score=total / n, reference=reference / n)


def integrated_brier(
    times: "list[float]",
    events: "list[bool]",
    predictor: "callable",
    horizons: "list[float]",
) -> float:
    """Trapezoidal average of the Brier score over horizons: one number for the whole curve.

    ``predictor(horizon)`` returns the predicted survival probability for every subject at that horizon.
    Averaging over horizons stops a model being tuned to look good at the one week someone chose to report.
    """
    if len(horizons) < 2:
        raise ValueError("need at least two horizons to integrate")
    points = [brier_score(times, events, predictor(horizon), horizon) for horizon in horizons]
    total = 0.0
    for (t0, p0), (t1, p1) in zip(
        [(point.horizon, point.score) for point in points],
        [(point.horizon, point.score) for point in points][1:],
    ):
        total += 0.5 * (p0 + p1) * (t1 - t0)
    return total / (horizons[-1] - horizons[0])


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    predicted: float
    observed: float  # Kaplan-Meier inside the bin, so censoring is handled
    observed_low: float
    observed_high: float

    @property
    def within_interval(self) -> bool:
        return self.observed_low <= self.predicted <= self.observed_high


def calibration(
    times: "list[float]",
    events: "list[bool]",
    predicted_survival: "list[float]",
    horizon: float,
    bins: int = 5,
) -> "list[CalibrationBin]":
    """Group by predicted survival, then compare the mean prediction with Kaplan-Meier in the group.

    The comparison must be against Kaplan-Meier, not against the observed churn fraction: within a bin the
    censored customers are still censored, and counting them as retained would make every model look
    over-pessimistic. Each bin also carries the Kaplan-Meier confidence interval, so a deviation can be read
    as a miscalibration rather than as thin data -- with a hundred customers per bin, a five-point gap is
    usually noise.
    """
    if bins < 2:
        raise ValueError("need at least two bins")
    order = sorted(range(len(times)), key=lambda index: predicted_survival[index])
    size = max(len(order) // bins, 1)
    output: list[CalibrationBin] = []
    for start in range(0, len(order), size):
        chunk = order[start : start + size]
        if len(chunk) < 5:
            break
        rows = [
            Interval(subject=position, start=0.0, stop=times[index], event=events[index], x=(0.0,))
            for position, index in enumerate(chunk)
        ]
        curve = kaplan_meier(rows)
        low, high = curve.interval_at(horizon)
        output.append(
            CalibrationBin(
                lower=min(predicted_survival[index] for index in chunk),
                upper=max(predicted_survival[index] for index in chunk),
                count=len(chunk),
                predicted=sum(predicted_survival[index] for index in chunk) / len(chunk),
                observed=curve.at(horizon),
                observed_low=low,
                observed_high=high,
            )
        )
    return output


def calibration_table(bins: "list[CalibrationBin]") -> str:
    header = f"{'predicted range':>22}{'n':>6}{'predicted':>11}{'KM observed':>13}{'KM 95% CI':>22}{'ok':>5}"
    lines = [header, "-" * len(header)]
    for entry in bins:
        lines.append(
            f"{f'{entry.lower:.3f} - {entry.upper:.3f}':>22}{entry.count:>6}{entry.predicted:>11.3f}"
            f"{entry.observed:>13.3f}"
            f"{f'{entry.observed_low:.3f} - {entry.observed_high:.3f}':>22}"
            f"{'yes' if entry.within_interval else 'NO':>5}"
        )
    return "\n".join(lines)


def split_subjects(cohort: Cohort, holdout: float = 0.3, seed: int = 0) -> tuple[Cohort, Cohort]:
    """Split by **subject**, never by row.

    Splitting rows would put week 1 of a customer in training and week 30 in test, so the test set would
    contain rows from customers the model has already seen churn -- leakage that flatters every metric here.
    Splitting by subject is the minimum; splitting by signup cohort in calendar time is stricter still, and
    is what to do when the product itself is changing.
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError("holdout must lie strictly inside (0, 1)")
    subjects = sorted({row.subject for row in cohort.rows})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    cut = int(len(subjects) * (1.0 - holdout))
    train_ids = set(subjects[:cut])
    train = tuple(row for row in cohort.rows if row.subject in train_ids)
    test = tuple(row for row in cohort.rows if row.subject not in train_ids)
    if not train or not test:
        raise ValueError("the split left one side empty; adjust the holdout fraction")
    return (
        Cohort(rows=train, names=cohort.names, truth=cohort.truth),
        Cohort(rows=test, names=cohort.names, truth=cohort.truth),
    )


def survival_at(baseline: "list[tuple[float, float]]", linear: float, horizon: float) -> float:
    """``S(horizon | x) = exp(-H0(horizon) exp(x'beta))``, reading the baseline step function."""
    hazard = 0.0
    for time, value in baseline:
        if time <= horizon:
            hazard = value
        else:
            break
    return math.exp(-hazard * math.exp(linear))
