"""Command line demonstrations. Each one is an argument, and most of them are about a failure.

    python -m survival km            Kaplan-Meier against the naive churn rate
    python -m survival cox           fit, hazard ratios, baseline hazard, predicted curves
    python -m survival ties          Efron against Breslow on heavily tied weekly data
    python -m survival ph            non-proportional hazards: what Cox reports and what it means
    python -m survival immortal      a zero effect that a naive encoding turns into a large one
    python -m survival truncation    staggered entry, with and without the entry times
    python -m survival metrics       held-out concordance, IPCW Brier, calibration
    python -m survival all           every demonstration in sequence

The figures depend on the seed and are printed rather than promised. Every claim these demos make about a
direction or a sign is also asserted in ``tests/``, which is where to look for the version that fails loudly.
"""

from __future__ import annotations

import math
import sys

from . import cox, data, metrics
from .nonparametric import (
    kaplan_meier,
    log_rank,
    naive_churn_rate,
    nelson_aalen,
    restricted_mean,
    split_by,
)


def _rule(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def demo_km() -> None:
    _rule("Kaplan-Meier, and the number it replaces")
    cohort = data.weibull_ph(n=900, seed=11)
    curve = kaplan_meier(list(cohort.rows))

    print(f"{cohort.n_subjects} customers, {cohort.n_events} churns, "
          f"{cohort.censoring_rate():.1%} still subscribed at the pull date\n")
    print(curve.table())

    horizon = 26.0
    naive = naive_churn_rate(cohort)
    km_churn = 1.0 - curve.at(horizon)
    print(
        f"\nnaive churn rate (churned / observed): {naive:.1%}"
        f"\nKaplan-Meier churn by week {horizon:g}:     {km_churn:.1%}"
        f"\ngap:                                   {abs(km_churn - naive) * 100:.1f} points"
    )
    print(
        "The naive figure is not a worse estimate of the same quantity; it is an estimate of nothing in\n"
        "particular, because it averages customers observed for one week with customers observed for a year."
    )

    median = curve.median()
    low, high = curve.median_interval()
    if median is None:
        print("\nmedian lifetime: not reached inside the observation window -- not estimable, and saying")
        print("'the median is beyond 52 weeks' is the only honest summary.")
    else:
        print(f"\nmedian lifetime {median:g} weeks (95% CI {low} - {high})")

    print("\n" + restricted_mean(curve, tau=52.0).summary())
    print("RMST needs no proportionality assumption and is always estimable, unlike the median.")

    times, hazard, _ = nelson_aalen(list(cohort.rows))
    print("\nNelson-Aalen cumulative hazard (curvature here means a non-constant hazard):")
    for index in range(0, len(times), max(len(times) // 6, 1)):
        print(f"  t = {times[index]:>5.1f}   H = {hazard[index]:.3f}")

    annual, monthly = split_by(cohort, index=0)
    print("\n" + log_rank(annual, monthly).summary())


def demo_cox() -> None:
    _rule("Cox proportional hazards")
    cohort = data.weibull_ph(n=900, seed=12)
    fit = cox.fit(cohort)
    print(fit.summary())

    truth = cohort.truth["beta"]
    print("\nrecovery against the generating coefficients:")
    for index, name in enumerate(cohort.names):
        low, high = fit.interval(index)
        covered = math.exp(truth[index])
        print(
            f"  {name:<14} true HR {covered:>6.3f}   estimated {fit.hazard_ratios[index]:>6.3f}   "
            f"CI {low:.3f} - {high:.3f}   {'covers' if low <= covered <= high else 'MISSES'}"
        )
    print(
        "\nThe baseline hazard was never estimated to get these numbers -- it cancels from every factor of\n"
        "the partial likelihood. That is the whole trick, and the reason a Cox model says nothing about\n"
        "when churn happens until the baseline is recovered separately:"
    )
    baseline = cox.baseline_cumulative_hazard(list(cohort.rows), fit.beta)
    for index in range(0, len(baseline), max(len(baseline) // 5, 1)):
        time, hazard = baseline[index]
        print(f"  t = {time:>5.1f}   H0 = {hazard:.4f}")

    print("\npredicted survival for two customers, from the same baseline curve raised to a power:")
    engaged = cox.predicted_survival(baseline, fit.beta, (0.0, 0.0, 1.5))
    fragile = cox.predicted_survival(baseline, fit.beta, (1.0, 1.0, -1.5))
    for horizon in (13.0, 26.0, 52.0):
        first = metrics.survival_at(baseline, sum(b * v for b, v in zip(fit.beta, (0.0, 0.0, 1.5))), horizon)
        second = metrics.survival_at(baseline, sum(b * v for b, v in zip(fit.beta, (1.0, 1.0, -1.5))), horizon)
        print(f"  week {horizon:>4.0f}:  engaged monthly {first:.3f}    discounted annual {second:.3f}")
    assert engaged and fragile


def demo_ties() -> None:
    _rule("Efron against Breslow, on data that is nothing but ties")
    cohort = data.weibull_ph(n=800, seed=13)
    # Weekly billing data: everything is observed on a week boundary, so ties are the rule, not the exception.
    rounded = data.Cohort(
        rows=tuple(
            data.Interval(row.subject, row.start, float(math.ceil(row.stop)), row.event, row.x)
            for row in cohort.rows
        ),
        names=cohort.names,
        truth=cohort.truth,
    )
    events = rounded.n_events
    distinct = len(rounded.event_times())
    print(f"{events} churns at {distinct} distinct times: {events / distinct:.1f} tied events per time\n")

    truth = cohort.truth["beta"]
    efron = cox.fit(rounded, ties="efron")
    breslow = cox.fit(rounded, ties="breslow")
    header = f"{'covariate':<14}{'true':>9}{'Efron':>10}{'Breslow':>10}{'Breslow bias':>14}"
    print(header)
    print("-" * len(header))
    for index, name in enumerate(rounded.names):
        print(
            f"{name:<14}{truth[index]:>9.4f}{efron.beta[index]:>10.4f}{breslow.beta[index]:>10.4f}"
            f"{breslow.beta[index] - truth[index]:>14.4f}"
        )
    print(
        "\nBreslow's approximation keeps the full risk set in the denominator for every tied event, so the\n"
        "tied subjects are counted as though still at risk for each other's failure. The result is shrinkage\n"
        "towards zero, and it grows with the number of ties -- which in weekly subscription data is large."
    )


def demo_ph() -> None:
    _rule("When the hazard ratio changes with time")
    cohort = data.non_proportional(n=700, seed=14)
    print(f"truth: log hazard ratio = {cohort.truth['log_hazard_ratio']}\n")

    fit = cox.fit(cohort)
    print(fit.summary())
    print(
        "\nThe coefficient is a time-average of an effect that reverses sign. It is not an estimate of\n"
        "anything a business could act on: annual plans are riskier early and safer late, and this single\n"
        "number says neither."
    )

    print("\nSchoenfeld residual test:")
    for result in cox.test_proportionality(list(cohort.rows), fit, permutations=1500):
        print("  " + result.summary())

    annual, monthly = split_by(cohort, 0)
    print("\n" + log_rank(annual, monthly).summary())
    print(
        "The log-rank test weights every event time equally, so an effect that is positive early and\n"
        "negative late cancels. A large p-value here is not evidence that the groups behave alike."
    )

    print("\nRestricted mean survival, which needs no proportionality assumption:")
    for label, rows in (("annual", annual), ("monthly", monthly)):
        rmst = restricted_mean(kaplan_meier(rows), tau=52.0)
        print(f"  {label:<9}{rmst.summary()}")
    print("\nAnd the two survival curves at three horizons, where the crossing is visible directly:")
    annual_curve = kaplan_meier(annual)
    monthly_curve = kaplan_meier(monthly)
    for horizon in (8.0, 20.0, 45.0):
        print(
            f"  week {horizon:>4.0f}:  annual S = {annual_curve.at(horizon):.3f}   "
            f"monthly S = {monthly_curve.at(horizon):.3f}"
        )


def demo_immortal() -> None:
    _rule("Immortal time bias: a zero effect that a naive encoding makes large")
    cohort = data.immortal_time(n=900, seed=15)
    print(
        "Adoption of the feature has exactly zero effect on the hazard in this data. Two encodings:\n"
        "  adopted_ever  1 for the whole tenure if the customer ever adopted -- uses the future\n"
        "  adopted_now   0 before the adoption week, 1 after -- uses only what was known at the time\n"
    )
    naive = cox.fit_subset(cohort, [0])
    correct = cox.fit_subset(cohort, [1])
    for label, fit in (("naive", naive), ("time-varying", correct)):
        low, high = fit.interval(0)
        z, p = fit.wald(0)
        print(
            f"  {label:<14} HR {fit.hazard_ratios[0]:.3f}  (95% CI {low:.3f} - {high:.3f})  p = {p:.3g}"
        )
    print(
        "\nThe naive hazard ratio is far below one and highly significant, from data where the truth is\n"
        "exactly one. Nothing about the sample size or the model fixes it: to be labelled an adopter in\n"
        "week 30 a customer must survive to week 30, and that survival is credited to adoption. This is the\n"
        "mechanism behind most reports that 'customers who use feature X churn less'."
    )


def demo_truncation() -> None:
    _rule("Staggered entry: left truncation is not censoring")
    cohort = data.staggered_entry(n=700, seed=16)
    correct = kaplan_meier(list(cohort.rows))
    naive_rows = [
        data.Interval(row.subject, 0.0, row.stop, row.event, row.x) for row in cohort.rows
    ]
    naive = kaplan_meier(naive_rows)
    print(
        "Customers were already partway through their tenure when observation began. The correct risk set\n"
        "at tenure t excludes anyone who had not yet been observed at t; the naive one pretends everybody\n"
        "was watched from signup.\n"
    )
    header = f"{'tenure':>8}{'correct S(t)':>15}{'naive S(t)':>13}{'difference':>13}"
    print(header)
    print("-" * len(header))
    for horizon in (10.0, 20.0, 30.0, 40.0, 50.0):
        first, second = correct.at(horizon), naive.at(horizon)
        print(f"{horizon:>8.0f}{first:>15.3f}{second:>13.3f}{second - first:>13.3f}")
    print(
        "\nThe naive curve is biased upwards early, because the customers contributing to the early risk set\n"
        "are the ones who had already survived long enough to be recruited. Survivorship, arithmetically."
    )


def demo_metrics() -> None:
    _rule("Held-out evaluation")
    cohort = data.weibull_ph(n=1100, seed=17)
    train, test = metrics.split_subjects(cohort, holdout=0.35, seed=3)
    fit = cox.fit(train)
    baseline = cox.baseline_cumulative_hazard(list(train.rows), fit.beta)

    observed = test.observed()
    times = [time for time, _ in observed]
    events = [event for _, event in observed]
    first_row = {}
    for row in test.rows:
        first_row.setdefault(row.subject, row)
    covariates = [first_row[subject].x for subject in sorted(first_row)]
    linear = [sum(b * v for b, v in zip(fit.beta, x)) for x in covariates]

    print(cox.CoxFit.summary(fit).splitlines()[1])
    print("\n" + metrics.concordance_index(times, events, linear).summary())

    for horizon in (13.0, 26.0, 39.0):
        predictions = [metrics.survival_at(baseline, value, horizon) for value in linear]
        point = metrics.brier_score(times, events, predictions, horizon)
        print(
            f"  week {horizon:>4.0f}:  Brier {point.score:.4f}   "
            f"Kaplan-Meier reference {point.reference:.4f}   skill {point.skill:+.1%}"
        )

    horizon = 26.0
    predictions = [metrics.survival_at(baseline, value, horizon) for value in linear]
    print(f"\ncalibration at week {horizon:g}, against Kaplan-Meier inside each bin:")
    print(metrics.calibration_table(metrics.calibration(times, events, predictions, horizon, bins=5)))
    print(
        "\nA good C-index with poor calibration would mean the ranking is usable for a save-team queue and\n"
        "the probabilities are not usable for a revenue forecast. They are different claims and both are\n"
        "reported here for that reason."
    )


DEMOS = {
    "km": demo_km,
    "cox": demo_cox,
    "ties": demo_ties,
    "ph": demo_ph,
    "immortal": demo_immortal,
    "truncation": demo_truncation,
    "metrics": demo_metrics,
}


def main(argv: "list[str] | None" = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    name = arguments[0]
    if name == "all":
        for demo in DEMOS.values():
            demo()
        return 0
    if name not in DEMOS:
        print(f"unknown demonstration {name!r}; choose from {', '.join(DEMOS)} or 'all'")
        return 2
    DEMOS[name]()
    return 0
