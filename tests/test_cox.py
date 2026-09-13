"""The Cox model, the proportionality check, the immortal-time bias, and held-out evaluation.

The most valuable tests here need no simulation: one full partial likelihood and the Efron/Breslow tie
contributions, both worked out on paper in the docstrings. After that come identities that must hold
exactly -- the analytic gradient against central differences, the score vanishing at the maximum, the score
residuals summing to the score, Cholesky against a hand-inverted 2x2. An identity binds harder than a
comparison against another library's output, because it cannot be satisfied by two errors that agree.

Where two estimators see identical data (Efron against Breslow, naive against time-varying, truncated
against untruncated) the difference is deterministic and the assertion is on its sign. Where an estimate is
compared against the truth the tolerance is in standard errors.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survival import cox, data, metrics  # noqa: E402
from survival.data import Cohort, Interval  # noqa: E402
from survival.nonparametric import kaplan_meier  # noqa: E402


def _predictions(cohort, fit, horizon: float) -> "tuple[list[float], list[bool], list[float]]":
    """Observed times, event flags and predicted survival at ``horizon``. Fixed covariates only."""
    baseline = cox.baseline_cumulative_hazard(list(cohort.rows), fit.beta)
    observed = cohort.observed()
    first: dict[int, Interval] = {}
    for row in cohort.rows:
        first.setdefault(row.subject, row)
    predictions = [
        metrics.survival_at(
            baseline, sum(b * v for b, v in zip(fit.beta, first[subject].x)), horizon
        )
        for subject in sorted(first)
    ]
    return [time for time, _ in observed], [event for _, event in observed], predictions


class TestPartialLikelihood:
    def test_partial_likelihood_by_hand(self):
        """Three subjects, x = (1, 0, 0), churning at t = 1, 2, 3, evaluated at beta = 0.5.

            t = 1  risk = all three   S = e^0.5 + 1 + 1 = 3.6487212707...
                   contribution = 0.5 - log(3.6487212707)
            t = 2  risk = {2, 3}      S = 2      contribution = -log 2
            t = 3  risk = {3}         S = 1      contribution = -log 1 = 0

            l(0.5) = 0.5 - log(3.6487212707) - log 2 = -1.4876268...

        Score: only the first event contributes, ``x - xbar = 1 - e^0.5/3.6487212707``. The information at
        that risk set is ``E[x^2] - E[x]^2``, and since x is binary ``E[x^2] = E[x]``.
        """
        rows = [
            Interval(0, 0.0, 1.0, True, (1.0,)),
            Interval(1, 0.0, 2.0, True, (0.0,)),
            Interval(2, 0.0, 3.0, True, (0.0,)),
        ]
        total = math.exp(0.5) + 2.0
        result = cox.partial_likelihood(rows, [0.5])
        assert result.value == pytest.approx(0.5 - math.log(total) - math.log(2.0), abs=1e-12)
        mean = math.exp(0.5) / total
        assert result.gradient[0] == pytest.approx(1.0 - mean, abs=1e-12)
        assert result.information[0][0] == pytest.approx(mean - mean * mean, abs=1e-12)

    def test_efron_and_breslow_differ_exactly_as_derived(self):
        """Two tied events at t = 1 among three subjects, at beta = 0, so S = 3 and S_D = 2.

            Breslow: -(log 3 + log 3)              both tied events see the full risk set
            Efron:   -(log 3 + log(3 - 0.5 * 2))   the second sees half the tied mass removed
        """
        rows = [
            Interval(0, 0.0, 1.0, True, (1.0,)),
            Interval(1, 0.0, 1.0, True, (0.0,)),
            Interval(2, 0.0, 2.0, False, (0.0,)),
        ]
        breslow = cox.partial_likelihood(rows, [0.0], ties="breslow")
        efron = cox.partial_likelihood(rows, [0.0], ties="efron")
        assert breslow.value == pytest.approx(-2.0 * math.log(3.0), abs=1e-12)
        assert efron.value == pytest.approx(-(math.log(3.0) + math.log(2.0)), abs=1e-12)

    def test_without_ties_the_two_corrections_agree(self):
        rows = list(data.weibull_ph(n=150, seed=23).rows)  # continuous times: ties have probability zero
        beta = [0.2, -0.1, 0.3]
        assert cox.partial_likelihood(rows, beta, "efron").value == pytest.approx(
            cox.partial_likelihood(rows, beta, "breslow").value, rel=1e-12
        )

    @pytest.mark.parametrize("ties", ["efron", "breslow"])
    def test_gradient_against_central_differences(self, ties):
        rows = [
            Interval(row.subject, row.start, float(math.ceil(row.stop)), row.event, row.x)
            for row in data.weibull_ph(n=200, seed=24).rows
        ]  # rounded to weeks, so the Efron branch is genuinely exercised
        beta = [0.3, -0.25, 0.15]
        analytic = cox.partial_likelihood(rows, beta, ties)
        step = 1e-6
        for index in range(len(beta)):
            up, down = list(beta), list(beta)
            up[index] += step
            down[index] -= step
            numeric = (
                cox.partial_likelihood(rows, up, ties).value
                - cox.partial_likelihood(rows, down, ties).value
            ) / (2.0 * step)
            assert analytic.gradient[index] == pytest.approx(numeric, rel=1e-5, abs=1e-6)

    def test_information_against_differences_of_the_gradient(self):
        rows = list(data.weibull_ph(n=200, seed=25).rows)
        beta = [0.3, -0.25, 0.15]
        analytic = cox.partial_likelihood(rows, beta)
        step = 1e-5
        for i in range(len(beta)):
            up, down = list(beta), list(beta)
            up[i] += step
            down[i] -= step
            gradient_up = cox.partial_likelihood(rows, up).gradient
            gradient_down = cox.partial_likelihood(rows, down).gradient
            for j in range(len(beta)):
                numeric = -(gradient_up[j] - gradient_down[j]) / (2.0 * step)
                assert analytic.information[i][j] == pytest.approx(numeric, rel=1e-4, abs=1e-6)

    def test_information_is_symmetric_and_positive_definite(self):
        information = cox.partial_likelihood(
            list(data.weibull_ph(n=200, seed=26).rows), [0.1, 0.1, 0.1]
        ).information
        for i in range(3):
            for j in range(3):
                assert information[i][j] == pytest.approx(information[j][i], rel=1e-12)
        cox.cholesky(information)  # raises if it is not positive definite

    def test_cholesky_solve_against_a_hand_inverted_matrix(self):
        """[[4, 1], [1, 3]] has determinant 11 and inverse (1/11)[[3, -1], [-1, 4]]."""
        matrix = [[4.0, 1.0], [1.0, 3.0]]
        solution = cox.cholesky_solve(matrix, [1.0, 2.0])
        assert solution[0] == pytest.approx(1.0 / 11.0, rel=1e-12)
        assert solution[1] == pytest.approx(7.0 / 11.0, rel=1e-12)
        inverted = cox.inverse(matrix)
        assert inverted[0][0] == pytest.approx(3.0 / 11.0, rel=1e-12)
        assert inverted[0][1] == pytest.approx(-1.0 / 11.0, rel=1e-12)
        assert inverted[1][1] == pytest.approx(4.0 / 11.0, rel=1e-12)


class TestFitting:
    def test_score_vanishes_at_the_maximum(self):
        cohort = data.weibull_ph(n=400, seed=27)
        fit = cox.fit(cohort)
        assert fit.converged
        gradient = cox.partial_likelihood(list(cohort.rows), fit.beta).gradient
        assert max(abs(value) for value in gradient) < 1e-6

    def test_score_residuals_sum_to_the_score(self):
        """An exact identity: within each risk set the weighted residuals sum to zero, so the cross terms
        cancel and what remains is the Breslow score. The robust variance depends on this entirely."""
        cohort = data.weibull_ph(n=250, seed=28)
        beta = [0.4, -0.3, 0.2]
        totals = [0.0, 0.0, 0.0]
        for residual in cox.score_residuals(list(cohort.rows), beta).values():
            for index in range(3):
                totals[index] += residual[index]
        expected = cox.partial_likelihood(list(cohort.rows), beta, ties="breslow").gradient
        for index in range(3):
            assert totals[index] == pytest.approx(expected[index], rel=1e-8, abs=1e-8)

    def test_coefficients_recover_the_truth(self):
        cohort = data.weibull_ph(n=1500, seed=29)
        fit = cox.fit(cohort)
        for index, truth in enumerate(cohort.truth["beta"]):
            assert abs(fit.beta[index] - truth) < 3.0 * fit.standard_errors[index], (
                f"{cohort.names[index]}: {fit.beta[index]:.3f} against {truth:.3f}"
            )

    def test_likelihood_ratio_test_rejects_a_real_effect(self):
        statistic, p_value = cox.fit(data.weibull_ph(n=600, seed=30)).likelihood_ratio()
        assert statistic > 20.0
        assert p_value < 1e-4

    def test_likelihood_ratio_test_keeps_its_size_under_the_null(self):
        cohort = data.weibull_ph(n=600, beta=(0.0, 0.0, 0.0), seed=31)
        _, p_value = cox.fit(cohort).likelihood_ratio()
        assert p_value > 0.01  # a true null: the threshold is deliberately loose

    def test_breslow_shrinks_towards_zero_when_ties_are_heavy(self):
        """Efron is the default for a measurable reason, not a stylistic one.

        Both estimators see identical data, so the difference carries no sampling error and the assertion
        can be on its sign. Which of the two lands closer to the truth on a single draw *does* involve
        sampling error, so that is deliberately not asserted.
        """
        cohort = data.weibull_ph(n=1200, seed=32)
        rounded = Cohort(
            rows=tuple(
                Interval(row.subject, row.start, float(math.ceil(row.stop)), row.event, row.x)
                for row in cohort.rows
            ),
            names=cohort.names,
            truth=cohort.truth,
        )
        efron = cox.fit(rounded, ties="efron")
        breslow = cox.fit(rounded, ties="breslow")
        for index in range(3):
            assert abs(breslow.beta[index]) < abs(efron.beta[index]), cohort.names[index]

    def test_robust_standard_errors_leave_the_estimate_alone(self):
        cohort = data.weibull_ph(n=500, seed=33)
        plain = cox.fit(cohort)
        robust = cox.fit(cohort, robust=True)
        assert plain.beta == pytest.approx(robust.beta, rel=1e-12)
        for index in range(3):
            ratio = robust.standard_errors[index] / plain.standard_errors[index]
            assert 0.5 < ratio < 2.0  # correctly specified, one row per subject: they should be close

    def test_hazard_ratio_interval_is_symmetric_on_the_log_scale(self):
        fit = cox.fit(data.weibull_ph(n=300, seed=34))
        low, high = fit.interval(0)
        assert math.log(low) + math.log(high) == pytest.approx(2.0 * fit.beta[0], rel=1e-12)

    def test_collinear_covariates_are_refused_rather_than_fudged(self):
        rows = [
            Interval(row.subject, row.start, row.stop, row.event, (row.x[0], row.x[0]))
            for row in data.weibull_ph(n=200, seed=35).rows
        ]  # the same covariate twice: no unique maximum exists
        with pytest.raises(ValueError, match="not identified"):
            cox.fit_cox(rows, ("duplicate_a", "duplicate_b"))

    def test_stratification_allows_separate_baselines(self):
        cohort = data.weibull_ph(n=500, seed=36)
        rows = [
            Interval(row.subject, row.start, row.stop, row.event, row.x, stratum=row.subject % 2)
            for row in cohort.rows
        ]
        stratified = cox.fit_cox(rows, cohort.names)
        assert stratified.converged
        for index, truth in enumerate(cohort.truth["beta"]):
            assert abs(stratified.beta[index] - truth) < 4.0 * stratified.standard_errors[index]

    def test_baseline_hazard_is_non_decreasing(self):
        cohort = data.weibull_ph(n=300, seed=50)
        fit = cox.fit(cohort)
        values = [
            hazard for _, hazard in cox.baseline_cumulative_hazard(list(cohort.rows), fit.beta)
        ]
        assert values == sorted(values)
        assert values[0] > 0.0

    def test_predicted_curves_cannot_cross(self):
        """A structural consequence: every predicted curve is the baseline curve raised to a power.

        The customer with the larger linear predictor has the higher hazard, so their survival curve lies
        below the other's at *every* horizon. This is not a nice property of the implementation, it is the
        proportional hazards assumption showing through -- data whose groups genuinely cross cannot be
        represented at all, which is why the Schoenfeld check comes before any prediction is trusted.
        """
        cohort = data.weibull_ph(n=300, seed=51)
        fit = cox.fit(cohort)
        baseline = cox.baseline_cumulative_hazard(list(cohort.rows), fit.beta)
        risky = (0.0, 0.0, 1.5)  # engagement enters with a positive coefficient in this generator
        safe = (1.0, 1.0, -1.5)
        assert sum(b * v for b, v in zip(fit.beta, risky)) > sum(
            b * v for b, v in zip(fit.beta, safe)
        )
        risky_curve = cox.predicted_survival(baseline, fit.beta, risky)
        safe_curve = cox.predicted_survival(baseline, fit.beta, safe)
        assert all(
            risky_survival <= safe_survival
            for (_, risky_survival), (_, safe_survival) in zip(risky_curve, safe_curve)
        )


class TestProportionalHazards:
    def test_a_reversing_effect_is_flagged(self):
        cohort = data.non_proportional(n=700, seed=37)
        fit = cox.fit(cohort)
        result = cox.test_proportionality(list(cohort.rows), fit, permutations=1000)[0]
        assert result.violated
        assert result.correlation < 0.0  # the true log hazard ratio falls with time

    def test_proportional_data_is_not_flagged(self):
        cohort = data.weibull_ph(n=700, seed=38)
        fit = cox.fit(cohort)
        for result in cox.test_proportionality(list(cohort.rows), fit, permutations=1000):
            assert abs(result.correlation) < 0.15, result.summary()

    def test_the_permutation_p_value_is_never_exactly_zero(self):
        """Add-one smoothing: 200 permutations cannot support a claim below 1/201."""
        cohort = data.non_proportional(n=400, seed=52)
        fit = cox.fit(cohort)
        result = cox.test_proportionality(list(cohort.rows), fit, permutations=200)[0]
        assert result.p_value >= 1.0 / 201.0

    def test_schoenfeld_residuals_are_centred_at_each_event_time(self):
        """The weighted mean is subtracted at each event time, so a risk set of one gives residual zero."""
        rows = [Interval(0, 0.0, 1.0, True, (1.0,)), Interval(1, 0.0, 2.0, True, (0.0,))]
        residuals = cox.schoenfeld_residuals(rows, [0.0])
        assert residuals[-1][1][0] == pytest.approx(0.0, abs=1e-12)


class TestImmortalTime:
    """The headline: a true hazard ratio of exactly one, reported as a large protective effect."""

    def test_the_naive_encoding_invents_a_large_effect(self):
        naive = cox.fit_subset(data.immortal_time(n=900, seed=39), [0])
        assert naive.beta[0] < -0.5  # strongly "protective"
        _, p_value = naive.wald(0)
        assert p_value < 0.001  # and confidently so

    def test_the_time_varying_encoding_finds_the_truth(self):
        correct = cox.fit_subset(data.immortal_time(n=900, seed=39), [1])
        assert abs(correct.beta[0]) < 3.0 * correct.standard_errors[0]

    def test_more_data_does_not_help_the_naive_encoding(self):
        """Bias, not variance: the estimate does not drift towards zero as the sample grows."""
        small = cox.fit_subset(data.immortal_time(n=500, seed=40), [0]).beta[0]
        large = cox.fit_subset(data.immortal_time(n=1400, seed=41), [0]).beta[0]
        assert small < -0.4 and large < -0.4


class TestTruncation:
    def test_ignoring_entry_times_biases_early_survival_upwards(self):
        """Same events, larger early risk sets, so the naive curve must sit above the correct one.

        Deterministic on identical data, so the assertion is on the sign of the difference.
        """
        cohort = data.staggered_entry(n=800, seed=42)
        correct = kaplan_meier(list(cohort.rows))
        naive = kaplan_meier(
            [Interval(row.subject, 0.0, row.stop, row.event, row.x) for row in cohort.rows]
        )
        assert naive.at(15.0) > correct.at(15.0)


@pytest.mark.slow
class TestTimeVaryingCovariates:
    def test_the_current_value_of_a_changing_covariate_is_recovered(self):
        cohort = data.time_varying_usage(n=300, seed=43)
        fit = cox.fit(cohort)
        assert abs(fit.beta[0] - cohort.truth["beta"][0]) < 3.0 * fit.standard_errors[0]

    def test_collapsing_to_the_baseline_value_attenuates_the_coefficient(self):
        cohort = data.time_varying_usage(n=300, seed=43)
        first: dict[int, Interval] = {}
        last: dict[int, Interval] = {}
        for row in cohort.rows:
            if row.subject not in first or row.start < first[row.subject].start:
                first[row.subject] = row
            if row.subject not in last or row.stop > last[row.subject].stop:
                last[row.subject] = row
        collapsed = [
            Interval(subject, 0.0, last[subject].stop, last[subject].event, first[subject].x)
            for subject in sorted(first)
        ]
        assert abs(cox.fit_cox(collapsed, cohort.names).beta[0]) < abs(cox.fit(cohort).beta[0])


class TestMetrics:
    def test_perfect_reversed_and_tied_rankings(self):
        times = [1.0, 2.0, 3.0, 4.0]
        events = [True] * 4
        assert metrics.concordance_index(times, events, [4.0, 3.0, 2.0, 1.0]).value == 1.0
        assert metrics.concordance_index(times, events, [1.0, 2.0, 3.0, 4.0]).value == 0.0
        assert metrics.concordance_index(times, events, [1.0, 1.0, 1.0, 1.0]).value == 0.5

    def test_unorderable_pairs_are_excluded_from_the_denominator(self):
        """A censored subject cannot be ordered against an event that happened after it left."""
        with pytest.raises(ValueError, match="no comparable pairs"):
            metrics.concordance_index([5.0, 9.0], [False, True], [1.0, 2.0])
        result = metrics.concordance_index([5.0, 9.0], [True, False], [2.0, 1.0])
        assert result.comparable == 1
        assert result.value == 1.0

    def test_a_fitted_model_ranks_held_out_customers_better_than_chance(self):
        cohort = data.weibull_ph(n=900, seed=44)
        train, test = metrics.split_subjects(cohort, holdout=0.3, seed=1)
        fit = cox.fit(train)
        times, events, _ = _predictions(test, fit, 26.0)
        first: dict[int, Interval] = {}
        for row in test.rows:
            first.setdefault(row.subject, row)
        risk = [sum(b * v for b, v in zip(fit.beta, first[s].x)) for s in sorted(first)]
        assert metrics.concordance_index(times, events, risk).value > 0.58

    def test_censoring_curve_is_the_reverse_kaplan_meier(self):
        """One censoring at t = 6 with two subjects then at risk, so G(6) = 1 - 1/2."""
        curve = metrics.censoring_curve([4.0, 5.0, 6.0, 8.0], [True, True, False, True])
        assert curve.at(5.0) == pytest.approx(1.0, abs=1e-12)
        assert curve.at(6.0) == pytest.approx(0.5, abs=1e-12)

    def test_brier_score_of_a_coin_flip_is_a_quarter(self):
        """With no censoring, predicting 0.5 for everyone scores exactly 0.25 -- a fixed point."""
        point = metrics.brier_score([1.0, 2.0, 3.0, 4.0], [True] * 4, [0.5] * 4, 2.0)
        assert point.score == pytest.approx(0.25, abs=1e-12)

    def test_a_fitted_model_beats_the_marginal_kaplan_meier_reference(self):
        cohort = data.weibull_ph(n=800, seed=45)
        fit = cox.fit(cohort)
        times, events, predictions = _predictions(cohort, fit, 26.0)
        point = metrics.brier_score(times, events, predictions, 26.0)
        assert point.score < point.reference
        assert point.skill > 0.0

    def test_calibration_bins_track_kaplan_meier_for_a_correct_model(self):
        cohort = data.weibull_ph(n=1200, seed=46)
        fit = cox.fit(cohort)
        times, events, predictions = _predictions(cohort, fit, 26.0)
        bins = metrics.calibration(times, events, predictions, 26.0, bins=4)
        assert len(bins) >= 4
        assert sum(1 for entry in bins if entry.within_interval) >= len(bins) - 2
        assert bins[0].observed < bins[-1].observed  # monotone: the ranking is real

    def test_splits_never_share_a_subject(self):
        cohort = data.time_varying_usage(n=120, seed=47)
        train, test = metrics.split_subjects(cohort, holdout=0.25, seed=2)
        assert {row.subject for row in train.rows} & {row.subject for row in test.rows} == set()
        assert train.n_subjects + test.n_subjects == cohort.n_subjects

    def test_integrated_brier_averages_over_horizons(self):
        observed = data.weibull_ph(n=500, seed=48).observed()
        value = metrics.integrated_brier(
            [time for time, _ in observed],
            [event for _, event in observed],
            lambda horizon: [0.6] * len(observed),
            [10.0, 20.0, 30.0],
        )
        assert 0.0 < value < 1.0
