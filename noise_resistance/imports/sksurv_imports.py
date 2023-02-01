import numpy
from sklearn.utils import check_consistent_length

"""Imports from https://github.com/sebp/scikit-survival/ due to package incompatability with other needed libraries."""


class StepFunction:
    """Callable step function.
    .. math::
        f(z) = a * y_i + b,
        x_i \\leq z < x_{i + 1}
    Parameters
    ----------
    x : ndarray, shape = (n_points,)
        Values on the x axis in ascending order.
    y : ndarray, shape = (n_points,)
        Corresponding values on the y axis.
    a : float, optional, default: 1.0
        Constant to multiply by.
    b : float, optional, default: 0.0
        Constant offset term.
    """

    def __init__(self, x, y, a=1.0, b=0.0):
        check_consistent_length(x, y)
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def __call__(self, x):
        """Evaluate step function.
        Parameters
        ----------
        x : float|array-like, shape=(n_values,)
            Values to evaluate step function at.
        Returns
        -------
        y : float|array-like, shape=(n_values,)
            Values of step function at `x`.
        """
        x = numpy.atleast_1d(x)
        if not numpy.isfinite(x).all():
            raise ValueError("x must be finite")
        if numpy.min(x) < self.x[0] or numpy.max(x) > self.x[-1]:
            raise ValueError(
                "x must be within [%f; %f]" % (self.x[0], self.x[-1])
            )
        i = numpy.searchsorted(self.x, x, side="left")
        not_exact = self.x[i] != x
        i[not_exact] -= 1
        value = self.a * self.y[i] + self.b
        if value.shape[0] == 1:
            return value[0]
        return value

    def __repr__(self):
        return "StepFunction(x=%r, y=%r, a=%r, b=%r)" % (
            self.x,
            self.y,
            self.a,
            self.b,
        )


def _compute_counts(event, time, order=None):
    """Count right censored and uncensored samples at each unique time point.
    Parameters
    ----------
    event : array
        Boolean event indicator.
    time : array
        Survival time or time of censoring.
    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.
    Returns
    -------
    times : array
        Unique time points.
    n_events : array
        Number of events at each time point.
    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.
    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = numpy.argsort(time, kind="mergesort")

    uniq_times = numpy.empty(n_samples, dtype=time.dtype)
    uniq_events = numpy.empty(n_samples, dtype=int)
    uniq_counts = numpy.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = numpy.resize(uniq_times, j)
    n_events = numpy.resize(uniq_events, j)
    total_count = numpy.resize(uniq_counts, j)
    n_censored = total_count - n_events

    # offset cumulative sum by one
    total_count = numpy.r_[0, total_count]
    n_at_risk = n_samples - numpy.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored


class BreslowEstimator:
    """Breslow's estimator of the cumulative hazard function.
    Attributes
    ----------
    cum_baseline_hazard_ : :class:`sksurv.functions.StepFunction`
        Cumulative baseline hazard function.
    baseline_survival_ : :class:`sksurv.functions.StepFunction`
        Baseline survival function.
    """

    def fit(self, linear_predictor, event, time):
        """Compute baseline cumulative hazard function.
        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.
        event : array-like, shape = (n_samples,)
            Contains binary event indicators.
        time : array-like, shape = (n_samples,)
            Contains event/censoring times.
        Returns
        -------
        self
        """
        risk_score = numpy.exp(linear_predictor)
        order = numpy.argsort(time, kind="mergesort")
        risk_score = risk_score[order]
        uniq_times, n_events, n_at_risk, _ = _compute_counts(
            event, time, order
        )

        divisor = numpy.empty(n_at_risk.shape, dtype=float)
        value = numpy.sum(risk_score)
        divisor[0] = value
        k = 0
        for i in range(1, len(n_at_risk)):
            d = n_at_risk[i - 1] - n_at_risk[i]
            value -= risk_score[k : (k + d)].sum()
            k += d
            divisor[i] = value

        assert k == n_at_risk[0] - n_at_risk[-1]

        y = numpy.cumsum(n_events / divisor)
        self.cum_baseline_hazard_ = StepFunction(uniq_times, y)
        self.baseline_survival_ = StepFunction(
            self.cum_baseline_hazard_.x,
            numpy.exp(-self.cum_baseline_hazard_.y),
        )
        return self

    def get_cumulative_hazard_function(self, linear_predictor):
        """Predict cumulative hazard function.
        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.
        Returns
        -------
        cum_hazard : ndarray, shape = (n_samples,)
            Predicted cumulative hazard functions.
        """
        risk_score = numpy.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = numpy.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(
                x=self.cum_baseline_hazard_.x,
                y=self.cum_baseline_hazard_.y,
                a=risk_score[i],
            )
        return funcs

    def get_survival_function(self, linear_predictor):
        """Predict survival function.
        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.
        Returns
        -------
        survival : ndarray, shape = (n_samples,)
            Predicted survival functions.
        """
        risk_score = numpy.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = numpy.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(
                x=self.baseline_survival_.x,
                y=numpy.power(self.baseline_survival_.y, risk_score[i]),
            )
        return funcs
