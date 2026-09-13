"""Non-parametric estimators: hand-computed arithmetic first, then properties.

Tolerances follow a rule. Where two estimators are compared on the *same* data the difference carries no
sampling error, so the assertion is on its sign. Where an estimate is compared against the truth the
tolerance is in standard errors. Null hypotheses are tested at p > 0.01 rather than p > 0.05, because a
correct implementation that fails the suite one run in twenty teaches people to ignore failures.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survival import data  # noqa: E402
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


# The worked example used throughout: six customers.
#
#   times   4   5   5   6*  8   9*        (* = still subscribed when the data was pulled)
#
#   t = 4:  n = 6, d = 1   S = 1 - 1/6            = 0.8333333...
#   t = 5:  n = 5, d = 2   S = (5/6)(1 - 2/5)     = 0.5
#   t = 8:  n = 2, d = 1   S = (1/2)(1 - 1/2)     = 0.25
#
#   Greenwood at t = 5:  1/(6*5) + 2/(5*3) = 1/30 + 2/15 = 1/6
#   so Var(S(5)) = 0.5^2 / 6 = 1/24 and se = 0.2041241452319315...
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
        rows = [Interval(0, 0.0, 2.0, True, (0.0,)), Interval(1, 3.0, 6.0, True, (0.0,))]
        table = {point.time: point.at_risk for point in risk_table(rows)}
        assert table[2.0] == 1
        assert table[6.0] == 1

    def test_interval_convention_is_left_open(self):
        """``start < t <= stop``: entry at exactly t is not yet at risk, exit at exactly t still is."""
        rows = [Interval(0, 0.0, 5.0, True, (0.0,)), Interval(1, 5.0, 9.0, True, (0.0,))]
        table = {point.time: point.at_risk for point in risk_table(rows)}
        assert table[5.0] == 1
        assert table[9.0] == 1

    def test_an_event_is_always_in_its_own_risk_set(self):
        """The invariant that makes the empty-risk-set guard in ``risk_table`` unreachable.

        Every interval satisfies ``start < stop``, so at its own event time the failing row satisfies
        ``start < stop <= stop`` and is in the risk set. No valid cohort can trigger the guard. It stays as
        a defence against a future change to the interval convention, and this test records why it cannot
        be exercised rather than pretending to exercise it.
        """
        rows = [Interval(0, 5.0, 9.0, True, (0.0,)), Interval(1, 0.0, 3.0, True, (0.0,))]
        table = {point.time: point.at_risk for point in risk_table(rows)}
        assert table[9.0] == 1  # only the failing subject, which is enough
        assert table[3.0] == 1  # disjoint windows: neither is ever in the other's risk set


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
        assert with_censor.survival[2] != pytest.approx(without.survival[2])

    def test_median_is_the_first_time_at_or_below_a_half(self):
        assert kaplan_meier(rows_from(WORKED)).median() == 5.0

    def test_median_is_none_when_the_curve_never_reaches_a_half(self):
        curve = kaplan_meier(rows_from([(4.0, True), (5.0, False), (9.0, False), (12.0, False)]))
        assert curve.median() is None

    def test_quantile_reads_the_curve(self):
        curve = kaplan_meier(rows_from(WORKED))
        assert curve.quantile(0.5) == 5.0
        assert curve.quantile(0.1) == 4.0
        assert curve.quantile(0.9) is None

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
        cohort = data.weibull_ph(n=700, seed=5)
        assert cohort.censoring_rate() > 0.2
        naive, km_churn, gap = naive_versus_km(cohort, horizon=52.0)
        assert km_churn > naive
        assert gap > 0.02

    def test_the_naive_rate_depends_on_when_you_look(self):
        """Same seed, so identical churn times; only the pull date differs.

        The naive rate moves with the observation window. Kaplan-Meier at a common horizon does not, which
        is the property that makes it a statement about customers rather than about reporting dates.
        """
        early = data.weibull_ph(n=600, horizon=13.0, seed=6)
        late = data.weibull_ph(n=600, horizon=52.0, seed=6)
        assert naive_churn_rate(early) < naive_churn_rate(late)
        first = 1.0 - kaplan_meier(list(early.rows)).at(13.0)
        second = 1.0 - kaplan_meier(list(late.rows)).at(13.0)
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
    """The incomplete gamma against closed forms, which is a better check than any printed table."""

    @pytest.mark.parametrize("x", [0.05, 0.5, 1.0, 2.7, 3.84, 8.0, 15.0, 40.0])
    def test_one_degree_of_freedom_matches_erfc(self, x):
        assert chi2_sf(x, 1) == pytest.approx(math.erfc(math.sqrt(x / 2.0)), rel=1e-10)

    @pytest.mark.parametrize("x", [0.1, 1.0, 4.0, 9.0, 25.0])
    def test_two_degrees_of_freedom_matches_an_exponential(self, x):
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
        assert log_rank(*split_by(cohort, 0)).p_value > 0.01

    def test_separated_hazards_are_detected(self):
        cohort = data.weibull_ph(n=700, beta=(1.0, 0.0, 0.0), seed=9)
        result = log_rank(*split_by(cohort, 0))
        assert result.statistic > 10.0
        assert result.p_value < 0.001
        assert result.observed > result.expected

    def test_p_value_is_the_chi_square_tail_of_the_statistic(self):
        result = log_rank(*split_by(data.weibull_ph(n=400, beta=(0.6, 0.0, 0.0), seed=10), 0))
        assert result.p_value == pytest.approx(chi2_sf(result.statistic, 1), rel=1e-12)

    def test_crossing_hazards_blunt_the_log_rank_test(self):
        """The failure the test is famous for: early and late differences partly cancel.

        Two comparisons with comparable event counts -- one with a constant hazard ratio, one where it
        reverses sign near week 17. The crossing case yields a far smaller statistic even though the groups
        genuinely differ at both ends, which the sign flip in the survival gap below establishes directly.

        The assertion is the *ordering* of the two statistics, not a p-value threshold: how much of the
        effect cancels depends on how events are distributed in time, so a fixed threshold would be a guess
        dressed as a check. The Schoenfeld test does detect this data -- see tests/test_cox.py.
        """
        crossing = log_rank(*split_by(data.non_proportional(n=800, seed=21), 0))
        proportional = log_rank(
            *split_by(data.weibull_ph(n=800, beta=(0.6, 0.0, 0.0), seed=21), 0)
        )
        assert crossing.statistic < proportional.statistic

        annual, monthly = split_by(data.non_proportional(n=800, seed=21), 0)
        early_gap = kaplan_meier(annual).at(8.0) - kaplan_meier(monthly).at(8.0)
        late_gap = kaplan_meier(annual).at(45.0) - kaplan_meier(monthly).at(45.0)
        assert early_gap < 0.0 < late_gap  # worse early, better late: the curves cross

    def test_stratified_test_pools_the_evidence(self):
        cohort = data.weibull_ph(n=500, beta=(0.7, 0.0, 0.0), seed=22)
        group_a, group_b = split_by(cohort, 0)
        half, other = len(group_a) // 2, len(group_b) // 2
        pooled = stratified_log_rank(
            [(group_a[:half], group_b[:other]), (group_a[half:], group_b[other:])]
        )
        assert pooled.strata == 2
        assert pooled.p_value < 0.05


class TestDataValidation:
    def test_overlapping_intervals_are_rejected(self):
        with pytest.raises(ValueError, match="overlapping"):
            Cohort(
                rows=(Interval(0, 0.0, 5.0, False, (1.0,)), Interval(0, 3.0, 8.0, True, (1.0,))),
                names=("x",),
            )

    def test_an_event_before_the_final_interval_is_rejected(self):
        with pytest.raises(ValueError, match="event before the final interval"):
            Cohort(
                rows=(Interval(0, 0.0, 5.0, True, (1.0,)), Interval(0, 5.0, 8.0, False, (1.0,))),
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
            cohort = generator(n=120, seed=49)
            assert cohort.n_subjects > 0, name
            assert cohort.n_events > 0, name
            assert 0.0 <= cohort.censoring_rate() <= 1.0, name
            assert all(row.stop > row.start for row in cohort.rows), name
