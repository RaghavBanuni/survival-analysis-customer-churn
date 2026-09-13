"""The test suite. Hand-computed values first, then properties, then recovery against known truth.

Three kinds of test appear here and they carry different weight:

1. **Arithmetic pinned by hand.** The Kaplan-Meier steps, the Greenwood variance, the Efron and Breslow
   contributions and one full partial likelihood are computed on paper in the docstrings and asserted to
   1e-12. These are the tests that would catch a rewrite that quietly changes the estimator.
2. **Identities that must hold.** The score at the maximum is zero, the score residuals sum to the score,
   ``chi2_sf(x, 1)`` equals ``erfc(sqrt(x/2))`` and ``chi2_sf(x, 2)`` equals ``exp(-x/2)``. An independent
   closed form is a far better check on an incomplete gamma written from scratch than any table lookup.
3. **Recovery and direction on synthetic data.** Coefficients within a few standard errors of the truth,
   Breslow shrinking towards zero relative to Efron, the immortal-time bias appearing and then vanishing.
   These use fixed seeds, and their tolerances are stated in standard errors rather than in decimals so
   that a failure means a bug and not a different random draw.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survival import cox, data, metrics  # noqa: E402
from survival.data import Cohort, Interval  # noqa: E402
from survival.nonparametric import (  # noqa: E402
    chi2_sf,
    kaplan_meier,
    log_rank,
    naive_churn_rate,
    naive_versus_km,
    nelson_aalen,
    restricted_mean,
    risk_table,
    split_by,
    stratified_log_rank,
)


def rows_from(pairs: "list[tuple[float, bool]]") -> "list[Interval]":
    return [
        Interval(subject=index, start=0.0, stop=time, event=event, x=(0.0,))
        for index, (time, event) in enumerate(pairs)
    ]


# The worked example used throughout: six customers, three of them censored patterns.
#
#   times   4   5   5   6*  8   9*        (* = still subscribed when the data was pulled)
#
#   t = 4:  n = 6, d = 1   S = 1 - 1/6                     = 0.8333333...
#   t = 5:  n = 5, d = 2   S = (5/6)(1 - 2/5)              = 0.5
#   t = 8:  n = 2, d = 1   S = (1/2)(1 - 1/2)              = 0.25
#
#   Greenwood at t = 5:  1/(6*5) + 2/(5*3) = 1/30 + 2/15 = 1/6
#   so Var(S(5)) = 0.5^2 / 6 = 1/24 and se = 0.2041241...
WORKED = [(4.0, True), (5.0, True), (5.0, True), (6.0, False), (8.0, True), (9.0, False)]


class TestRiskSets:
    def test_risk_table_matches_the_hand_computation(self):
        table = risk_table(rows_from(WORKED))
        assert [(point.time, point.at_risk, point.events) for point in table] == [
            (4.0, 6, 1),
            (5.0, 5, 2),
            (8.0, 2, 1),
        ]

    def test_left_truncation_removes_subjects_before_entry(self):
        """A subject entering at tenure 3 is not at risk at tenure 2, and must not be counted there."""
        rows = [
            Interval(0, 0.0, 2.0, True, (0.0,)),
            Interval(1, 3.0, 6.0, True, (0.0,)),
        ]
        table = {point.time: point.at_risk for point in risk_table(rows)}
        assert table[2.0] == 1  # only subject 0 is under observation
        assert table[6.0] == 1

    def test_interval_convention_is_left_open(self):
        """``start < t <= stop``: entry at exactly t is not yet at risk, exit at exactly t still is."""
        rows = [Interval(0, 0.0, 5.0, True, (0.0,)), Interval(1, 5.0, 9.0, True, (0.0,))]
        table = {point.time: point.at_risk for point in risk_table(rows)}
        assert table[5.0] == 1
        assert table[9.0] == 1

    def test_an_event_with_nobody_at_risk_is_rejected(self):
        with pytest.raises(ValueError, match="nobody at risk"):
            risk_table([Interval(0, 5.0, 9.0, True, (0.0,)), Interval(1, 0.0, 3.0, False, (0.0,))])
            # the second row does not reach t = 9 and the first has not entered before it; constructed
            # to be impossible rather than merely unlikely


class TestKaplanMeier:
    def test_steps_are_exact(self):
        curve = kaplan_meier(rows_from(WORKED))
        assert curve.times == [4.0, 5.0, 8.0]
        assert curve.survival[0] == pytest.approx(5.0 / 6.0, abs=1e-12)
        assert curve.survival[1] == pytest.approx(0.5, abs=1e-12)
        assert curve.survival[2] == pytest.approx(0.25, abs=1e-12)

    def test_survival_is_flat_between_events_and_one_before_the_first(self):
        curve = kaplan_meier(rows_from(WORKED))
        assert curve.at(0.0) == 1.0
        assert curve.at(3.9) == 1.0
        assert curve.at(4.0) == pytest.approx(5.0 / 6.0)
        assert curve.at(7.5) == pytest.approx(0.5)
        assert curve.at(100.0) == pytest.approx(0.25)

    def test_greenwood_variance_matches_the_hand_computation(self):
        curve = kaplan_meier(rows_from(WORKED))
        assert curve.variance_terms[1] == pytest.approx(1.0 / 6.0, abs=1e-12)
        standard_error = curve.survival[1] * math.sqrt(curve.variance_terms[1])
        assert standard_error == pytest.approx(0.20412414523193154, abs=1e-12)

    def test_log_log_limits_stay_inside_the_unit_interval(self):
        """The reason for the transform: a symmetric interval on S would leave [0, 1] here."""
        curve = kaplan_meier(rows_from(WORKED))
        for lower, survival, upper in zip(curve.lower, curve.survival, curve.upper):
            assert 0.0 <= lower <= survival <= upper <= 1.0

    def test_censoring_only_removes_from_the_risk_set(self):
        """Censoring at 6 changes no step before 6 -- the defining property of the estimator."""
        without = kaplan_meier(rows_from([(4.0, True), (5.0, True), (5.0, True), (8.0, True)]))
        with_censor = kaplan_meier(rows_from(WORKED))
        assert with_censor.survival[0] == pytest.approx(without.survival[0])
        assert with_censor.survival[1] == pytest.approx(without.survival[1])
        # ... and does change the step after it, since the risk set is smaller
        assert with_censor.survival[2] != pytest.approx(without.survival[2])

    def test_median_is_the_first_time_at_or_below_a_half(self):
        assert kaplan_meier(rows_from(WORKED)).median() == 5.0

    def test_median_is_none_when_the_curve_never_reaches_a_half(self):
        curve = kaplan_meier(rows_from([(4.0, True), (5.0, False), (9.0, False), (12.0, False)]))
        assert curve.median() is None

    def test_quantile_reads_the_curve(self):
        curve = kaplan_meier(rows_from(WORKED))
        assert curve.quantile(0.5) == 5.0
        assert curve.quantile(0.1) == 4.0  # 10% churned by the first event
        assert curve.quantile(0.9) is None  # never observed

    def test_nelson_aalen_accumulates_the_hazard(self):
        times, hazard, variance = nelson_aalen(rows_from(WORKED))
        assert times == [4.0, 5.0, 8.0]
        assert hazard[1] == pytest.approx(1.0 / 6.0 + 2.0 / 5.0, abs=1e-12)
        assert variance[1] == pytest.approx(1.0 / 36.0 + 2.0 / 25.0, abs=1e-12)


class TestNaiveChurnRate:
    def test_naive_rate_is_churns_over_customers(self):
        cohort = Cohort(rows=tuple(rows_from(WORKED)), names=("dummy",))
        assert naive_churn_rate(cohort) == pytest.approx(4.0 / 6.0)

    def test_naive_rate_understates_churn_when_censoring_is_heavy(self):
        """The whole reason Kaplan-Meier exists, asserted as an inequality on real generated data."""
        cohort = data.weibull_ph(n=700, seed=5)
        assert cohort.censoring_rate() > 0.2
        naive, km_churn, gap = naive_versus_km(cohort, horizon=52.0)
        assert km_churn > naive
        assert gap > 0.02

    def test_the_naive_rate_depends_on_when_you_look(self):
        """Same behaviour, different observation window, different 'churn rate' -- the incoherence."""
        early = data.weibull_ph(n=600, horizon=13.0, seed=6)
        late = data.weibull_ph(n=600, horizon=52.0, seed=6)
        assert naive_churn_rate(early) < naive_churn_rate(late)
        # Kaplan-Meier at a common horizon is stable across the two windows, which the naive rate is not.
        common = 13.0
        first = 1.0 - kaplan_meier(list(early.rows)).at(common)
        second = 1.0 - kaplan_meier(list(late.rows)).at(common)
        assert abs(first - second) < 0.05


class TestRestrictedMean:
    def test_area_under_a_hand_computed_curve(self):
        """Three subjects, all churning: S = 1, 2/3, 1/3 on (0,1], (1,2], (2,3], so RMST(3) = 2."""
        curve = kaplan_meier(rows_from([(1.0, True), (2.0, True), (3.0, True)]))
        assert restricted_mean(curve, tau=3.0).value == pytest.approx(2.0, abs=1e-12)

    def test_rmst_is_bounded_by_tau_and_increases_with_it(self):
        curve = kaplan_meier(list(data.weibull_ph(n=400, seed=7).rows))
        short = restricted_mean(curve, tau=13.0)
        long = restricted_mean(curve, tau=39.0)
        assert 0.0 < short.value <= 13.0
        assert short.value < long.value <= 39.0
        assert long.standard_error > 0.0

    def test_rmst_is_estimable_when_the_median_is_not(self):
        curve = kaplan_meier(rows_from([(4.0, True), (9.0, False), (12.0, False)]))
        assert curve.median() is None
        assert restricted_mean(curve, tau=12.0).value > 0.0


class TestTailProbabilities:
    @pytest.mark.parametrize("x", [0.05, 0.5, 1.0, 2.7, 3.84, 8.0, 15.0, 40.0])
    def test_one_degree_of_freedom_matches_the_closed_form(self, x):
        assert chi2_sf(x, 1) == pytest.approx(math.erfc(math.sqrt(x / 2.0)), rel=1e-10)

    @pytest.mark.parametrize("x", [0.1, 1.0, 4.0, 9.0, 25.0])
    def test_two_degrees_of_freedom_matches_the_closed_form(self, x):
        assert chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2.0), rel=1e-10)

    def test_the_critical_value_is_where_it_should_be(self):
        assert chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-9)

    def test_monotone_and_bounded(self):
        values = [chi2_sf(x, 3) for x in (0.5, 1.0, 5.0, 20.0)]
        assert values == sorted(values, reverse=True)
        assert all(0.0 <= value <= 1.0 for value in values)
        assert chi2_sf(0.0, 4) == 1.0


class TestLogRank:
    def test_identical_groups_give_a_small_statistic(self):
        cohort = data.weibull_ph(n=600, beta=(0.0, 0.0, 0.0), seed=8)
        group_a, group_b = split_by(cohort, 0)  # the covariate has no effect here
        result = log_rank(group_a, group_b)
        assert result.p_value > 0.05
        assert not result.significant

    def test_separated_hazards_are_detected(self):
        cohort = data.weibull_ph(n=700, beta=(1.0, 0.0, 0.0), seed=9)
        group_a, group_b = split_by(cohort, 0)
        result = log_rank(group_a, group_b)
        assert result.statistic > 10.0
        assert result.p_value < 0.001
        assert result.observed > result.expected  # the high-hazard group churns more than expected

    def test_p_value_is_the_chi_square_tail_of_the_statistic(self):
        cohort = data.weibull_ph(n=400, beta=(0.6, 0.0, 0.0), seed=10)
        result = log_rank(*split_by(cohort, 0))
        assert result.p_value == pytest.approx(chi2_sf(result.statistic, 1), rel=1e-12)

    def test_crossing_hazards_defeat_the_log_rank_test(self):
        """The failure the test is famous for: early and late differences cancel to nothing.

        The generating log hazard ratio is +0.6 early and strongly negative late. The groups' survival
        curves genuinely differ, and RMST says so, while the log-rank statistic is small.
        """
        cohort = data.non_proportional(n=800, seed=21)
        group_a, group_b = split_by(cohort, 0)
        assert log_rank(group_a, group_b).p_value > 0.05
        early_gap = kaplan_meier(group_a).at(8.0) - kaplan_meier(group_b).at(8.0)
        late_gap = kaplan_meier(group_a).at(45.0) - kaplan_meier(group_b).at(45.0)
        assert early_gap < 0.0 < late_gap  # the curves cross: worse early, better late

    def test_stratified_test_pools_the_evidence(self):
        cohort = data.weibull_ph(n=500, beta=(0.7, 0.0, 0.0), seed=22)
        group_a, group_b = split_by(cohort, 0)
        half = len(group_a) // 2
        other = len(group_b) // 2
        strata = [
            (group_a[:half], group_b[:other]),
            (group_a[half:], group_b[other:]),
        ]
        pooled = stratified_log_rank(strata)
        assert pooled.strata == 2
        assert pooled.p_value < 0.05


class TestCoxLikelihood:
    def test_partial_likelihood_by_hand(self):
        """Three subjects, x = (1, 0, 0), churning at t = 1, 2, 3, evaluated at beta = 0.5.

            t = 1  risk = all three   S = e^0.5 + 1 + 1 = 3.6487212707
                   contribution = 0.5 - log(3.6487212707)
            t = 2  risk = {2, 3}      S = 2      contribution = 0 - log 2
            t = 3  risk = {3}         S = 1      contribution = 0 - log 1 = 0

            l(0.5) = 0.5 - log(3.6487212707) - log 2 = -1.4876268...

        Score at the same point: only the first event contributes, x - xbar = 1 - e^0.5/3.6487212707,
        and the information is E[x^2] - E[x]^2 at that risk set.
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
        """Two tied events at t = 1 among three subjects, at beta = 0, where S = 3 and S_D = 2.

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

    def test_without_ties_efron_and_breslow_agree(self):
        rows = list(data.weibull_ph(n=150, seed=23).rows)  # continuous times: ties have probability zero
        beta = [0.2, -0.1, 0.3]
        assert cox.partial_likelihood(rows, beta, "efron").value == pytest.approx(
            cox.partial_likelihood(rows, beta, "breslow").value, rel=1e-12
        )

    @pytest.mark.parametrize("ties", ["efron", "breslow"])
    def test_gradient_against_central_differences(self, ties):
        rows = list(data.weibull_ph(n=200, seed=24).rows)
        rounded = [
            Interval(row.subject, row.start, float(math.ceil(row.stop)), row.event, row.x)
            for row in rows
        ]  # rounded so the Efron branch is genuinely exercised
        beta = [0.3, -0.25, 0.15]
        analytic = cox.partial_likelihood(rounded, beta, ties)
        step = 1e-6
        for index in range(len(beta)):
            up = list(beta)
            down = list(beta)
            up[index] += step
            down[index] -= step
            numeric = (
                cox.partial_likelihood(rounded, up, ties).value
                - cox.partial_likelihood(rounded, down, ties).value
            ) / (2.0 * step)
            assert analytic.gradient[index] == pytest.approx(numeric, rel=1e-5, abs=1e-6)

    def test_information_matrix_against_differences_of_the_gradient(self):
        rows = list(data.weibull_ph(n=200, seed=25).rows)
        beta = [0.3, -0.25, 0.15]
        analytic = cox.partial_likelihood(rows, beta)
        step = 1e-5
        for i in range(len(beta)):
            up = list(beta)
            down = list(beta)
            up[i] += step
            down[i] -= step
            gradient_up = cox.partial_likelihood(rows, up).gradient
            gradient_down = cox.partial_likelihood(rows, down).gradient
            for j in range(len(beta)):
                numeric = -(gradient_up[j] - gradient_down[j]) / (2.0 * step)
                assert analytic.information[i][j] == pytest.approx(numeric, rel=1e-4, abs=1e-6)

    def test_information_is_symmetric_and_positive_definite(self):
        rows = list(data.weibull_ph(n=200, seed=26).rows)
        information = cox.partial_likelihood(rows, [0.1, 0.1, 0.1]).information
        for i in range(3):
            for j in range(3):
                assert information[i][j] == pytest.approx(information[j][i], rel=1e-12)
        cox.cholesky(information)  # raises if it is not positive definite


class TestCoxFitting:
    def test_score_vanishes_at_the_maximum(self):
        cohort = data.weibull_ph(n=400, seed=27)
        fit = cox.fit(cohort)
        assert fit.converged
        gradient = cox.partial_likelihood(list(cohort.rows), fit.beta).gradient
        assert max(abs(value) for value in gradient) < 1e-6

    def test_score_residuals_sum_to_the_score(self):
        """An identity, and the only cheap check on the residual loop that catches an error in it."""
        cohort = data.weibull_ph(n=250, seed=28)
        beta = [0.4, -0.3, 0.2]
        residuals = cox.score_residuals(list(cohort.rows), beta)
        totals = [0.0, 0.0, 0.0]
        for residual in residuals.values():
            for index in range(3):
                totals[index] += residual[index]
        expected = cox.partial_likelihood(list(cohort.rows), beta, ties="breslow").gradient
        for index in range(3):
            assert totals[index] == pytest.approx(expected[index], rel=1e-8, abs=1e-8)

    def test_coefficients_recover_the_truth(self):
        cohort = data.weibull_ph(n=1500, seed=29)
        fit = cox.fit(cohort)
        for index, truth in enumerate(cohort.truth["beta"]):
            error = abs(fit.beta[index] - truth)
            assert error < 3.0 * fit.standard_errors[index], (
                f"{cohort.names[index]}: {fit.beta[index]:.3f} vs {truth:.3f}"
            )

    def test_likelihood_ratio_test_rejects_a_real_effect(self):
        cohort = data.weibull_ph(n=600, seed=30)
        statistic, p_value = cox.fit(cohort).likelihood_ratio()
        assert statistic > 20.0
        assert p_value < 1e-4

    def test_likelihood_ratio_test_keeps_its_size_under_the_null(self):
        cohort = data.weibull_ph(n=600, beta=(0.0, 0.0, 0.0), seed=31)
        _, p_value = cox.fit(cohort).likelihood_ratio()
        assert p_value > 0.05

    def test_breslow_shrinks_towards_zero_when_ties_are_heavy(self):
        """Efron is the default for a measurable reason, not a stylistic one."""
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
        truth = cohort.truth["beta"]
        for index in (0, 1):  # the two binary covariates, where the effect is largest
            assert abs(breslow.beta[index]) < abs(efron.beta[index])
            assert abs(efron.beta[index] - truth[index]) < abs(breslow.beta[index] - truth[index])

    def test_robust_standard_errors_are_available_and_sane(self):
        cohort = data.weibull_ph(n=500, seed=33)
        plain = cox.fit(cohort)
        robust = cox.fit(cohort, robust=True)
        assert plain.beta == pytest.approx(robust.beta, rel=1e-12)  # the point estimate is unchanged
        for index in range(3):
            ratio = robust.standard_errors[index] / plain.standard_errors[index]
            assert 0.5 < ratio < 2.0  # correctly specified: the two should be close

    def test_hazard_ratio_interval_is_symmetric_on_the_log_scale(self):
        fit = cox.fit(data.weibull_ph(n=300, seed=34))
        low, high = fit.interval(0)
        assert math.log(low) + math.log(high) == pytest.approx(2.0 * fit.beta[0], rel=1e-12)

    def test_collinear_covariates_are_refused_rather_than_fudged(self):
        base = data.weibull_ph(n=200, seed=35)
        rows = [
            Interval(row.subject, row.start, row.stop, row.event, (row.x[0], row.x[0]))
            for row in base.rows
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


class TestProportionalHazards:
    def test_a_reversing_effect_is_flagged(self):
        cohort = data.non_proportional(n=700, seed=37)
        fit = cox.fit(cohort)
        result = cox.test_proportionality(list(cohort.rows), fit, permutations=1000)[0]
        assert result.violated
        # the true log hazard ratio falls with time, so the residuals trend downwards
        assert result.correlation < 0.0

    def test_proportional_data_is_not_flagged(self):
        cohort = data.weibull_ph(n=700, seed=38)
        fit = cox.fit(cohort)
        for result in cox.test_proportionality(list(cohort.rows), fit, permutations=1000):
            assert abs(result.correlation) < 0.15, result.summary()

    def test_schoenfeld_residuals_are_centred_at_each_event_time(self):
        """At each event time the weighted mean is subtracted, so a single-event time has residual zero."""
        rows = [Interval(0, 0.0, 1.0, True, (1.0,)), Interval(1, 0.0, 2.0, True, (0.0,))]
        residuals = cox.schoenfeld_residuals(rows, [0.0])
        assert residuals[-1][1][0] == pytest.approx(0.0, abs=1e-12)  # last event: risk set of one


class TestImmortalTime:
    """The headline result: a true hazard ratio of exactly one, reported as a large protective effect."""

    def test_the_naive_encoding_invents_a_large_effect(self):
        cohort = data.immortal_time(n=900, seed=39)
        naive = cox.fit_subset(cohort, [0])
        assert naive.beta[0] < -0.5  # strongly "protective"
        _, p_value = naive.wald(0)
        assert p_value < 0.001  # and confidently so

    def test_the_time_varying_encoding_finds_the_truth(self):
        cohort = data.immortal_time(n=900, seed=39)
        correct = cox.fit_subset(cohort, [1])
        assert abs(correct.beta[0]) < 3.0 * correct.standard_errors[0]

    def test_more_data_does_not_help_the_naive_encoding(self):
        """Bias, not variance: the estimate does not move towards zero as the sample grows."""
        small = cox.fit_subset(data.immortal_time(n=500, seed=40), [0]).beta[0]
        large = cox.fit_subset(data.immortal_time(n=1400, seed=41), [0]).beta[0]
        assert small < -0.4 and large < -0.4


class TestTruncation:
    def test_ignoring_entry_times_biases_early_survival_upwards(self):
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
        truth = cohort.truth["beta"][0]
        assert abs(fit.beta[0] - truth) < 3.0 * fit.standard_errors[0]

    def test_collapsing_to_the_baseline_value_loses_the_signal(self):
        """Summarising a time-varying covariate by its first value attenuates the coefficient."""
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
        flattened = cox.fit_cox(collapsed, cohort.names)
        full = cox.fit(cohort)
        assert abs(flattened.beta[0]) < abs(full.beta[0])


class TestMetrics:
    def test_perfect_and_reversed_rankings(self):
        times = [1.0, 2.0, 3.0, 4.0]
        events = [True, True, True, True]
        assert metrics.concordance_index(times, events, [4.0, 3.0, 2.0, 1.0]).value == 1.0
        assert metrics.concordance_index(times, events, [1.0, 2.0, 3.0, 4.0]).value == 0.0
        assert metrics.concordance_index(times, events, [1.0, 1.0, 1.0, 1.0]).value == 0.5

    def test_censored_pairs_are_excluded_from_the_denominator(self):
        """A censored subject can only be ordered against events that happened before it left."""
        times = [5.0, 9.0]
        events = [False, True]
        with pytest.raises(ValueError, match="no comparable pairs"):
            metrics.concordance_index(times, events, [1.0, 2.0])
        # ... whereas an event first is orderable against a later censoring
        result = metrics.concordance_index([5.0, 9.0], [True, False], [2.0, 1.0])
        assert result.comparable == 1
        assert result.value == 1.0

    def test_a_fitted_model_ranks_better_than_chance(self):
        cohort = data.weibull_ph(n=900, seed=44)
        train, test = metrics.split_subjects(cohort, holdout=0.3, seed=1)
        fit = cox.fit(train)
        observed = test.observed()
        first: dict[int, Interval] = {}
        for row in test.rows:
            first.setdefault(row.subject, row)
        risk = [
            sum(b * v for b, v in zip(fit.beta, first[subject].x)) for subject in sorted(first)
        ]
        result = metrics.concordance_index(
            [time for time, _ in observed], [event for _, event in observed], risk
        )
        assert result.value > 0.6

    def test_censoring_curve_is_the_reverse_kaplan_meier(self):
        times = [4.0, 5.0, 6.0, 8.0]
        events = [True, True, False, True]
        curve = metrics.censoring_curve(times, events)
        # one censoring at t = 6 with two subjects at risk: G(6) = 1 - 1/2
        assert curve.at(6.0) == pytest.approx(0.5, abs=1e-12)
        assert curve.at(5.0) == pytest.approx(1.0, abs=1e-12)

    def test_brier_score_of_a_perfect_prediction_beats_the_reference(self):
        cohort = data.weibull_ph(n=800, seed=45)
        fit = cox.fit(cohort)
        baseline = cox.baseline_cumulative_hazard(list(cohort.rows), fit.beta)
        observed = cohort.observed()
        times = [time for time, _ in observed]
        events = [event for _, event in observed]
        first: dict[int, Interval] = {}
        for row in cohort.rows:
            first.setdefault(row.subject, row)
        predictions = [
            metrics.survival_at(
                baseline, sum(b * v for b, v in zip(fit.beta, first[subject].x)), 26.0
            )
            for subject in sorted(first)
        ]
        point = metrics.brier_score(times, events, predictions, 26.0)
        assert point.score < point.reference  # better than the marginal Kaplan-Meier curve
        assert point.skill > 0.0

    def test_brier_score_of_a_coin_flip_is_a_quarter(self):
        """With no censoring, predicting 0.5 for everyone scores exactly 0.25 -- a fixed point to check."""
        times = [1.0, 2.0, 3.0, 4.0]
        events = [True, True, True, True]
        point = metrics.brier_score(times, events, [0.5] * 4, 2.0)
        assert point.score == pytest.approx(0.25, abs=1e-12)

    def test_calibration_bins_agree_with_kaplan_meier_for_a_correct_model(self):
        cohort = data.weibull_ph(n=1200, seed=46)
        fit = cox.fit(cohort)
        baseline = cox.baseline_cumulative_hazard(list(cohort.rows), fit.beta)
        observed = cohort.observed()
        first: dict[int, Interval] = {}
        for row in cohort.rows:
            first.setdefault(row.subject, row)
        predictions = [
            metrics.survival_at(
                baseline, sum(b * v for b, v in zip(fit.beta, first[subject].x)), 26.0
            )
            for subject in sorted(first)
        ]
        bins = metrics.calibration(
            [time for time, _ in observed],
            [event for _, event in observed],
            predictions,
            26.0,
            bins=4,
        )
        assert len(bins) >= 4
        assert sum(1 for entry in bins if entry.within_interval) >= len(bins) - 1
        # and the bins must be ordered: higher predicted survival, higher observed survival
        assert bins[0].observed < bins[-1].observed

    def test_splits_never_share_a_subject(self):
        cohort = data.time_varying_usage(n=120, seed=47)
        train, test = metrics.split_subjects(cohort, holdout=0.25, seed=2)
        shared = {row.subject for row in train.rows} & {row.subject for row in test.rows}
        assert shared == set()
        assert train.n_subjects + test.n_subjects == cohort.n_subjects

    def test_integrated_brier_averages_over_horizons(self):
        cohort = data.weibull_ph(n=500, seed=48)
        observed = cohort.observed()
        times = [time for time, _ in observed]
        events = [event for _, event in observed]
        value = metrics.integrated_brier(
            times, events, lambda horizon: [0.6] * len(times), [10.0, 20.0, 30.0]
        )
        assert 0.0 < value < 1.0


class TestDataValidation:
    def test_overlapping_intervals_are_rejected(self):
        with pytest.raises(ValueError, match="overlapping"):
            Cohort(
                rows=(
                    Interval(0, 0.0, 5.0, False, (1.0,)),
                    Interval(0, 3.0, 8.0, True, (1.0,)),
                ),
                names=("x",),
            )

    def test_an_event_before_the_final_interval_is_rejected(self):
        with pytest.raises(ValueError, match="event before the final interval"):
            Cohort(
                rows=(
                    Interval(0, 0.0, 5.0, True, (1.0,)),
                    Interval(0, 5.0, 8.0, False, (1.0,)),
                ),
                names=("x",),
            )

    def test_zero_length_intervals_are_rejected(self):
        with pytest.raises(ValueError, match="not after start"):
            Interval(0, 4.0, 4.0, True, (1.0,))

    def test_covariate_width_must_match_the_names(self):
        with pytest.raises(ValueError, match="expected"):
            Cohort(rows=(Interval(0, 0.0, 5.0, True, (1.0, 2.0)),), names=("only_one",))

    def test_every_generator_produces_a_valid_cohort(self):
        for name, generator in data.REGIMES.items():
            cohort = generator(n=80, seed=49) if name != "staggered" else generator(n=80, seed=49)
            assert cohort.n_subjects > 0, name
            assert cohort.n_events > 0, name
            assert 0.0 <= cohort.censoring_rate() <= 1.0, name
            assert all(row.stop > row.start for row in cohort.rows), name
