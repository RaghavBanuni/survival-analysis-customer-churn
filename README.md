# Survival Analysis for Customer Churn

Kaplan-Meier with Greenwood intervals, Nelson-Aalen, log-rank and stratified log-rank, restricted mean
survival time, and a full **Cox proportional hazards** model -- Efron ties, Newton-Raphson on the analytic
score and information, cluster-robust variance, Schoenfeld residual diagnostics, left truncation and
time-varying covariates -- plus censoring-aware evaluation. **Pure Python, standard library only.** No
lifelines, no scikit-survival, no NumPy.

Two errors account for most wrong churn analysis, and this repository is organised around demonstrating both
and then removing them.

**The first is arithmetic.** "Churn rate" is normally churned customers divided by observed customers, which
puts a customer who signed up three weeks ago and a customer observed for three years on the same footing. It
is not a worse estimate of retention; it is an estimate of nothing in particular, and it moves whenever the
reporting date moves. `python -m survival km` prints it next to Kaplan-Meier on the same cohort.

**The second is using the future.** In `data.immortal_time`, feature adoption has *exactly zero* effect on
the hazard. Label a customer an "adopter" for their whole tenure because they adopted in week 30, and the
model reports a large, highly significant protective effect:

```
$ python -m survival immortal

  naive          HR 0.4xx  (95% CI ...)  p < 0.001      <- adopted_ever, uses the future
  time-varying   HR ~1.0   (95% CI ...)  p = 0.xx       <- adopted_now, uses only what was known
```

To be an adopter in week 30 you must survive to week 30, and that survival is credited to adoption. No sample
size fixes it -- `test_more_data_does_not_help_the_naive_encoding` triples n and the bias does not budge,
because it is bias and not variance. This is the mechanism behind most claims that "customers who use feature
X churn less".

> Figures in this README are illustrative of the report format. Run the commands; the tests are where the
> claims are pinned.

```bash
python -m survival km          # Kaplan-Meier against the naive churn rate; median, RMST, log-rank
python -m survival cox         # hazard ratios, baseline hazard, predicted curves, recovery vs truth
python -m survival ties        # Efron against Breslow on weekly data, which is nothing but ties
python -m survival ph          # a hazard ratio that reverses sign, and what each test makes of it
python -m survival immortal    # the bias above
python -m survival truncation  # staggered entry, with and without the entry times
python -m survival metrics     # held-out concordance, IPCW Brier, calibration against Kaplan-Meier
```

---

## What censoring actually requires

Everything rests on the risk set. A subject is under observation on `(start, stop]`, so

```
n(t) = #{ rows : start < t <= stop }
```

and every estimator here reads that definition. Data is therefore in **counting-process form** from the
outset -- one row per interval on which covariates are constant -- which costs nothing when covariates are
fixed and makes three otherwise-separate problems the same computation:

| situation | what breaks if you ignore it |
|---|---|
| right censoring | churned/observed is uninterpretable, and biased by cohort age |
| staggered entry (left truncation) | early risk sets include customers nobody was watching, so early survival is biased **upwards** |
| time-varying covariates | baseline usage misses the signal; average usage uses the future |

`python -m survival truncation` measures the second on data where the entry times are known.

**Estimators.** `S(t) = prod (1 - d_i/n_i)`, `H(t) = sum d_i/n_i`, `Var(S) = S^2 sum d_i/(n_i(n_i-d_i))`.
Confidence limits are built on `log(-log S)` and mapped back through `S^exp(±z·se)`, because a symmetric
interval on `S` runs outside `[0, 1]` exactly in the tails where it is quoted. The median is `None` when the
curve never reaches 0.5 -- reporting a median that was never observed is a straightforward misstatement, and
`RMST`, the area under the curve to a horizon, is the summary to use instead.

## The Cox model

```
L(beta) = prod over event times of  exp(x_j'beta) / sum over the risk set of exp(x_k'beta)
```

The baseline hazard cancels from every factor, which is the entire trick: covariate effects without a
distributional assumption. The price is that the model says nothing about *when* churn happens until Breslow's
baseline estimator is added back afterwards.

**Ties are the implementation detail that decides the numbers.** Weekly billing data ties nearly every event.
Breslow's approximation keeps the whole risk set in the denominator for each tied event, so tied subjects are
counted as still at risk for one another's failure, and coefficients shrink towards zero. Efron averages over
the orders the tied events could have taken:

```
contribution = sum_{j in D} eta_j  -  sum_{r=0}^{d-1} log( S - (r/d) S_D )
```

Setting every `r` to zero recovers Breslow, so one loop with a shrinking denominator implements both -- and
`test_breslow_shrinks_towards_zero_when_ties_are_heavy` asserts the direction of the difference on identical
data, where it carries no sampling error.

**The gradient and Hessian are analytic.** With `xbar_r` the discounted weighted covariate mean,

```
score = sum_D x_j - sum_r xbar_r
I     = sum_r [ Z2_r/D_r - xbar_r xbar_r' ]
```

`I` is manifestly a sum of covariance matrices, hence positive semi-definite, which is why the partial
likelihood is concave and Newton converges in a handful of steps. Both are checked against central
differences of the likelihood and of the score respectively. When `I` is singular the Cholesky raises rather
than regularising: singular means the coefficients are **not identified** -- collinear covariates, or a
covariate that separates who churned -- and that is the finding, not an inconvenience.

**Clustered standard errors.** When one subject contributes many rows, its score contributions are
correlated and the model-based variance is too small. The sandwich estimator sums score residuals within
subject; the residuals must sum to the score, and that identity is asserted, because it is the only cheap
check on the loop that produces them.

## The assumption, and what to do when it fails

Proportional hazards means every predicted survival curve is the same baseline curve raised to a power, so
**curves cannot cross**. `data.non_proportional` generates a log hazard ratio of `0.6 - 0.035t`, which
reverses sign near week 17 -- annual contracts churn more at the first renewal decision and much less
afterwards. Fitted to it, Cox reports a time-average of an effect that reverses: not wrong, meaningless.

Three tests, three verdicts:

- **The Cox coefficient** is small and looks unremarkable.
- **The log-rank test** is blunted, because it weights every event time equally and the early and late
  contributions partly cancel. A large p-value here is not evidence that the groups behave alike.
- **The Schoenfeld residuals** detect it. They are `x_j - xbar(t_j)` at each event, and their trend against
  time *is* the time-varying coefficient. The p-value comes from a **permutation** null rather than the
  Grambsch-Therneau chi-square: under proportional hazards the residuals are exchangeable in time, so
  shuffling the time labels gives the null distribution directly, with no asymptotic covariance to
  mis-derive. Add-one smoothing means the p-value can never be reported below `1/(permutations+1)`.

When it fails, compare **RMST** instead: no proportionality assumption, always estimable, and in units the
business already uses -- expected subscription weeks within the first year.

## Evaluating a survival model

AUC on "churned within 12 weeks" needs an answer nobody has for the censored customers. Dropping them selects
on the outcome; calling them retained records a guess as data. Both inflate the number, and both are routine.

- **Harrell's C-index** over pairs whose order is knowable, ties in the score counted as half. It is blind to
  calibration -- any monotone transformation of the risk score leaves it unchanged.
- **IPCW Brier score** (Graf et al.), each observation reweighted by `1/G(t)` with `G` the reverse-KM estimate
  of the censoring distribution, and reported against a **marginal Kaplan-Meier reference**. A covariate model
  that cannot beat "everyone churns at the average rate" has earned nothing, and a raw Brier score hides that.
- **Calibration** against Kaplan-Meier *inside each bin*, with the bin's confidence interval, so a gap can be
  read as miscalibration rather than as thin data.

Good concordance with poor calibration means the ranking is usable for a save-team queue and the probabilities
are not usable for a revenue forecast. Different claims, both reported.

Splits are by **subject**, never by row -- splitting rows trains on week 1 and tests on week 30 of the same
customer.

## Tests

```bash
pip install -e ".[dev]"
pytest -q                    # everything
pytest -q -m "not slow"      # skip the time-varying fits
```

| file | what it pins down |
|---|---|
| `tests/test_survival.py` | KM steps, Greenwood variance, RMST area and Nelson-Aalen against hand computation; the incomplete gamma against `erfc` and `exp` closed forms; log-rank behaviour including its failure on crossing hazards |
| `tests/test_cox.py` | a full partial likelihood and the Efron/Breslow contributions on paper; gradient and information against central differences; score zero at the maximum; residuals summing to the score; the immortal-time bias appearing and vanishing; concordance, Brier and calibration |

Two rules keep the suite honest. Where two estimators see identical data the assertion is on the **sign** of
their difference, which carries no sampling error. Where an estimate meets the truth the tolerance is in
**standard errors**, and null hypotheses are tested at p > 0.01 -- a correct implementation that fails one run
in twenty teaches people to ignore failures.

## Limits

- **Pure Python.** The risk-set loop is O(events x rows) per likelihood evaluation. Fixed-covariate fits of a
  thousand customers take seconds; the time-varying fits, with tens of thousands of interval rows, take tens
  of seconds and are marked `slow`.
- **No parametric or AFT models.** No Weibull or log-logistic regression, no accelerated failure time, and no
  cure models -- which matter when a real fraction of customers never churn at all.
- **No frailty or recurrent events.** Win-back and resubscription are common in subscription businesses and
  are not modelled; a customer who leaves and returns is two subjects here.
- **No competing risks.** Voluntary cancellation, payment failure and downgrade are all "churn", though they
  have different causes and different remedies. Cause-specific hazards and the Fine-Gray subdistribution
  model are the right treatment and are absent.
- **Independent censoring is assumed and is untestable from the data.** Where the reason a customer leaves the
  dataset is itself related to churn risk, every method here fails together.
- **No time-varying coefficient estimation.** The Schoenfeld test detects non-proportionality; fitting
  `beta(t)` -- by splitting the time axis or interacting the covariate with time -- is left undone, and is the
  natural next step.
- **Synthetic data only.** No proprietary dataset is used or implied.

For production work, use [lifelines](https://lifelines.readthedocs.io) or
[scikit-survival](https://scikit-survival.readthedocs.io). This repository exists to show the derivations and
the failure modes, which a library call hides by design.

## References

- Cox (1972), *Regression Models and Life-Tables* -- the partial likelihood.
- Kaplan & Meier (1958), *Nonparametric Estimation from Incomplete Observations*.
- Efron (1977), *The Efficiency of Cox's Likelihood Function for Censored Data* -- the tie correction.
- Grambsch & Therneau (1994), *Proportional Hazards Tests and Diagnostics Based on Weighted Residuals*.
- Therneau & Grambsch (2000), *Modeling Survival Data: Extending the Cox Model* -- counting-process form, robust variance, stratification.
- Graf, Schmoor, Sauerbrei & Schumacher (1999), *Assessment and Comparison of Prognostic Classification Schemes for Survival Data* -- the IPCW Brier score.
- Harrell, Califf, Pryor, Lee & Rosati (1982), *Evaluating the Yield of Medical Tests* -- the concordance index.
- Suissa (2008), *Immortal Time Bias in Pharmacoepidemiology* -- the bias `python -m survival immortal` reproduces.
- Klein & Moeschberger (2003), *Survival Analysis: Techniques for Censored and Truncated Data* -- the RMST variance used here.

MIT licensed.
