"""Synthetic subscription cohorts, in counting-process form, with the truth attached.

Every generator emits ``(start, stop, event]`` rows rather than one row per customer. That shape costs
nothing when covariates are fixed and is the only honest way to express the three things that actually
happen to subscription data:

* **Right censoring.** The customer is still subscribed when the data was pulled. Their lifetime is not
  short; it is unknown and at least this long.
* **Staggered entry / left truncation.** Customers sign up on different dates, so at any tenure only some
  of them are under observation. A risk set that ignores entry time counts customers who were not yet
  observable.
* **Time-varying covariates.** Usage, plan and discounts change during the relationship, and the value that
  matters for the hazard at week 30 is the value at week 30.

``immortal_time`` is the generator to read first. In it, feature adoption has **exactly zero** effect on the
hazard, and a naive analysis that labels a customer "adopter" for their whole tenure -- using information
from the future -- will report a large protective effect anyway. Any survival implementation that cannot
reproduce that bias, and then remove it, has not been tested on the thing that goes wrong in practice.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Interval:
    """One ``(start, stop]`` observation window for one subject, with covariates held constant on it."""

    subject: int
    start: float
    stop: float
    event: bool
    x: tuple[float, ...]
    stratum: int = 0

    def __post_init__(self) -> None:
        if self.stop <= self.start:
            raise ValueError(f"subject {self.subject}: stop {self.stop} is not after start {self.start}")
        if self.start < 0.0:
            raise ValueError(f"subject {self.subject}: negative start time")


@dataclass(frozen=True)
class Cohort:
    """A set of intervals, plus the covariate names and whatever truth generated them."""

    rows: tuple[Interval, ...]
    names: tuple[str, ...]
    truth: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("a cohort needs at least one interval")
        width = len(self.names)
        by_subject: dict[int, list[Interval]] = {}
        for row in self.rows:
            if len(row.x) != width:
                raise ValueError(f"subject {row.subject}: {len(row.x)} covariates, expected {width}")
            by_subject.setdefault(row.subject, []).append(row)
        for subject, rows in by_subject.items():
            ordered = sorted(rows, key=lambda item: item.start)
            for earlier, later in zip(ordered, ordered[1:]):
                if later.start < earlier.stop - 1e-12:
                    raise ValueError(f"subject {subject}: overlapping intervals")
            for row in ordered[:-1]:
                if row.event:
                    raise ValueError(f"subject {subject}: an event before the final interval")

    @property
    def n_subjects(self) -> int:
        return len({row.subject for row in self.rows})

    @property
    def n_events(self) -> int:
        return sum(1 for row in self.rows if row.event)

    @property
    def width(self) -> int:
        return len(self.names)

    def event_times(self) -> list[float]:
        return sorted({row.stop for row in self.rows if row.event})

    def observed(self) -> list[tuple[float, bool]]:
        """One ``(total observed time, did it end in churn)`` pair per subject, for the simple estimators."""
        last: dict[int, Interval] = {}
        for row in self.rows:
            current = last.get(row.subject)
            if current is None or row.stop > current.stop:
                last[row.subject] = row
        return [(row.stop, row.event) for _, row in sorted(last.items())]

    def censoring_rate(self) -> float:
        pairs = self.observed()
        return sum(1 for _, event in pairs if not event) / len(pairs)

    def group(self, index: int) -> list[int]:
        """Baseline value of covariate ``index`` per subject, for two-sample comparisons."""
        first: dict[int, Interval] = {}
        for row in self.rows:
            current = first.get(row.subject)
            if current is None or row.start < current.start:
                first[row.subject] = row
        return [int(row.x[index]) for _, row in sorted(first.items())]


def fixed_covariate_cohort(
    times: list[float], events: list[bool], covariates: list[tuple[float, ...]], names: tuple[str, ...]
) -> Cohort:
    """Build a cohort from the familiar one-row-per-customer form."""
    if not (len(times) == len(events) == len(covariates)):
        raise ValueError("times, events and covariates must have the same length")
    rows = tuple(
        Interval(subject=index, start=0.0, stop=time, event=event, x=tuple(x))
        for index, (time, event, x) in enumerate(zip(times, events, covariates))
    )
    return Cohort(rows=rows, names=names)


# ---------------------------------------------------------------------------------------------
# Weibull proportional hazards: the model's own assumptions
# ---------------------------------------------------------------------------------------------


def weibull_ph(
    n: int = 800,
    beta: tuple[float, ...] = (0.8, -0.5, 0.35),
    shape: float = 1.3,
    scale: float = 40.0,
    horizon: float = 52.0,
    dropout_rate: float = 0.004,
    seed: int = 0,
) -> Cohort:
    """Churn times from a Weibull baseline with proportional covariate effects.

    The baseline hazard is ``h0(t) = (k/lambda) (t/lambda)^(k-1)`` and the subject hazard is
    ``h0(t) exp(x'beta)``, so the survival function inverts exactly:

        S(t) = exp(-(t/lambda)^k exp(x'beta))   =>   T = lambda * (-log(U) / exp(x'beta))^(1/k)

    which is worth doing properly rather than by rejection sampling: it makes the recovered coefficients a
    test of the estimator rather than of the simulator. ``shape > 1`` means the churn hazard rises with
    tenure; ``shape < 1`` means early churn dominates, which is what most subscription products look like.

    Censoring is both administrative (the data was pulled at ``horizon``) and random (customers vanish for
    reasons unrelated to churn, at ``dropout_rate`` per unit time).
    """
    if n < 2:
        raise ValueError("need at least two subjects")
    if shape <= 0.0 or scale <= 0.0:
        raise ValueError("Weibull shape and scale must be positive")
    rng = random.Random(seed)
    names = ("annual_plan", "discounted", "engagement")
    if len(beta) != len(names):
        raise ValueError(f"beta must have {len(names)} entries")

    times: list[float] = []
    events: list[bool] = []
    covariates: list[tuple[float, ...]] = []
    for _ in range(n):
        annual = 1.0 if rng.random() < 0.4 else 0.0
        discounted = 1.0 if rng.random() < 0.3 else 0.0
        engagement = rng.gauss(0.0, 1.0)
        x = (annual, discounted, engagement)
        linear = sum(coefficient * value for coefficient, value in zip(beta, x))

        uniform = rng.random()
        churn = scale * (-math.log(uniform) / math.exp(linear)) ** (1.0 / shape)
        dropout = rng.expovariate(dropout_rate) if dropout_rate > 0 else math.inf
        censor = min(dropout, horizon)

        times.append(min(churn, censor))
        events.append(churn <= censor)
        covariates.append(x)

    truth = {
        "beta": beta,
        "shape": shape,
        "scale": scale,
        "regime": "Weibull proportional hazards, exactly as assumed",
    }
    cohort = fixed_covariate_cohort(times, events, covariates, names)
    return Cohort(rows=cohort.rows, names=names, truth=truth)


def staggered_entry(
    n: int = 600, beta: tuple[float, ...] = (0.7,), horizon: float = 52.0, seed: int = 1
) -> Cohort:
    """Customers observed only from some tenure onwards: left truncation, not censoring.

    A customer who signed up two years before the data window is only observed from tenure 104 onwards, and
    they are only in the risk set from then. Treating their start as zero silently claims they survived a
    period nobody watched -- and because survivors are over-represented among long-tenure customers, it
    biases the early hazard downwards.
    """
    rng = random.Random(seed)
    rows: list[Interval] = []
    for subject in range(n):
        segment = 1.0 if rng.random() < 0.5 else 0.0
        linear = beta[0] * segment
        entry = rng.uniform(0.0, 30.0)  # tenure already accrued when observation began
        churn = 45.0 * (-math.log(rng.random()) / math.exp(linear)) ** (1.0 / 1.2)
        if churn <= entry:
            continue  # never observed: this is the truncation, and dropping them is correct
        stop = min(churn, entry + horizon)
        rows.append(
            Interval(subject, start=entry, stop=stop, event=churn <= entry + horizon, x=(segment,))
        )
    return Cohort(
        rows=tuple(rows),
        names=("enterprise",),
        truth={"beta": beta, "regime": "staggered entry with left truncation"},
    )


# ---------------------------------------------------------------------------------------------
# regimes the Cox model is wrong about
# ---------------------------------------------------------------------------------------------


def _discrete_time_path(
    rng: random.Random,
    horizon: int,
    hazard: "callable",
    covariate: "callable",
) -> tuple[list[tuple[int, float]], int, bool]:
    """Walk one subject forward period by period, returning its covariate path and outcome.

    Discrete-time simulation is used wherever the hazard depends on time or on a changing covariate,
    because then no closed-form inverse exists. Per period the event probability is ``1 - exp(-h)``, which
    is the exact discretisation of a piecewise-constant continuous hazard rather than an approximation.
    """
    path: list[tuple[int, float]] = []
    for period in range(horizon):
        value = covariate(period)
        path.append((period, value))
        if rng.random() < 1.0 - math.exp(-hazard(period, value)):
            return path, period + 1, True
    return path, horizon, False


def non_proportional(n: int = 700, horizon: int = 52, seed: int = 2) -> Cohort:
    """A covariate whose hazard ratio crosses one: the assumption Cox cannot bend.

    Annual contracts churn *more* than monthly ones early (buyer's remorse at the first renewal decision)
    and much less later. The true log hazard ratio is ``+0.6 - 0.035 * t``, crossing zero around week 17.
    A Cox model fitted to this will report the time-average of that effect -- a number that is not wrong so
    much as meaningless, and the Schoenfeld residual test is what says so.
    """
    rng = random.Random(seed)
    rows: list[Interval] = []
    for subject in range(n):
        annual = 1.0 if rng.random() < 0.5 else 0.0

        def hazard(period: int, value: float, annual=annual) -> float:
            baseline = 0.02 + 0.0004 * period
            return baseline * math.exp((0.6 - 0.035 * period) * annual)

        _, stop, event = _discrete_time_path(rng, horizon, hazard, lambda period: annual)
        rows.append(Interval(subject, 0.0, float(stop), event, (annual,)))
    return Cohort(
        rows=tuple(rows),
        names=("annual_plan",),
        truth={
            "log_hazard_ratio": "0.6 - 0.035 t, crossing zero near t = 17",
            "regime": "non-proportional hazards: the effect reverses",
        },
    )


def time_varying_usage(n: int = 600, horizon: int = 52, seed: int = 3) -> Cohort:
    """Usage that decays before churn, expressed as one row per week where it changes.

    The hazard depends on *current* usage with true coefficient -0.9. Summarising usage by its baseline
    value throws away the signal; summarising it by its average over the whole relationship uses the
    future, and looks even better while being unusable for prediction.
    """
    rng = random.Random(seed)
    rows: list[Interval] = []
    truth_beta = -0.9
    for subject in range(n):
        level = rng.uniform(0.4, 1.6)
        drift = rng.uniform(-0.03, 0.01)

        def usage(period: int, level=level, drift=drift) -> float:
            return max(level + drift * period, 0.05)

        def hazard(period: int, value: float) -> float:
            return 0.03 * math.exp(truth_beta * value)

        path, stop, event = _discrete_time_path(rng, horizon, hazard, usage)
        for period, value in path:
            is_last = period == stop - 1
            rows.append(
                Interval(
                    subject,
                    start=float(period),
                    stop=float(period + 1),
                    event=event and is_last,
                    x=(value,),
                )
            )
    return Cohort(
        rows=tuple(rows),
        names=("usage",),
        truth={"beta": (truth_beta,), "regime": "time-varying covariate, weekly resolution"},
    )


def immortal_time(n: int = 900, horizon: int = 52, seed: int = 4) -> Cohort:
    """Feature adoption with **no effect at all** on churn -- and the bias that says otherwise.

    Adoption happens at a random week drawn independently of the hazard. Two encodings of the same data:

    * the naive one, ``adopted_ever``, which marks the customer as an adopter for their entire tenure;
    * the correct one, ``adopted_now``, which is zero before the adoption week and one after.

    The naive encoding cannot be true of the early weeks, because to adopt in week 30 you must survive to
    week 30. That survival gets credited to adoption. The bias is large, entirely artefactual, and it is
    the single most common error in churn analysis -- ``python -m survival immortal`` measures it.
    """
    rng = random.Random(seed)
    rows: list[Interval] = []
    for subject in range(n):
        adoption = rng.randint(1, horizon + 10)  # may never happen inside the horizon

        def hazard(period: int, value: float) -> float:
            return 0.035  # constant, and independent of adoption: that is the point

        _, stop, event = _discrete_time_path(rng, horizon, hazard, lambda period: 0.0)
        adopted_ever = 1.0 if adoption < stop else 0.0

        if adoption >= stop:
            rows.append(Interval(subject, 0.0, float(stop), event, (adopted_ever, 0.0)))
        else:
            rows.append(Interval(subject, 0.0, float(adoption), False, (adopted_ever, 0.0)))
            rows.append(Interval(subject, float(adoption), float(stop), event, (adopted_ever, 1.0)))
    return Cohort(
        rows=tuple(rows),
        names=("adopted_ever", "adopted_now"),
        truth={
            "beta": (0.0, 0.0),
            "regime": "adoption has zero effect; the naive encoding invents one",
        },
    )


REGIMES = {
    "weibull": weibull_ph,
    "staggered": staggered_entry,
    "nonph": non_proportional,
    "timevarying": time_varying_usage,
    "immortal": immortal_time,
}
