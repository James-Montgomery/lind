"""
hypothesis:


Note on estimating the population variance: We often use n-1 instead of n when estimating the
population variance (Bessel's correction), where n is the number of samples. This method corrects
the bias in the estimation of the population variance. It also partially corrects the bias in the
estimation of the population standard deviation. However, the correction often increases the mean
squared error in these estimations. When n is large this correction is small.

"""

import logging

from numpy import linspace, where
from scipy.stats import norm, t

# set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# define public functions (ignored by jupyter notebooks)
__all__ = ["get_zscore", "get_tscore"]


####################################################################################################


_z_scores = linspace(-3, 3, 1000)
_z_probabilities = norm(loc=0.0, scale=1.0).cdf(_z_scores)


def get_zscore(p: float = 0.975) -> float:
    """
    get_zscore

    Find the z score for the probability p. Z scores are appropriate when the population variance
    is known or the sample size is large enough that the sample variance can be assumed to be a
    good proxy for the population variance.

    Parameters
    ----------
    p: float
        the probability that the user wants converted to a z score

    Returns
    -------
    float
        the corresponding z score

    Examples
    --------
    >>> z_score = get_zscore(0.99)

    """

    if p <= 0.0 or p >= 1.0:
        raise ValueError("Input p must be a float between 0 and 1 non-inclusive.")
    return _z_scores[
        where(_z_probabilities <= p)[0][-1]
    ]


_t_scores = linspace(-3, 3, 1000)


def get_tscore(p: float = 0.975, sample_size: int = 1000000000) -> float:
    """
    get_tscore

    Find the t score for the probability p. T scores are appropriate when the population variance
    is unknown. This is usually the case for small sample sizes.

    Parameters
    ----------
    p: float
        the probability that the user wants converted to a t score
    sample_size: int
        the size of the sample used in the experiment

    Returns
    -------
    float
        the corresponding t score

    Examples
    --------
    >>> t_score = get_tscore(0.99, 1000)

    """

    if p <= 0.0 or p >= 1.0:
        raise ValueError("Input p must be a float between 0 and 1 non-inclusive.")
    sample_size -= 1 # Bessel correction
    if sample_size == 0.0:
        raise ValueError("The input sample_size must be greater than 1.")
    t_probabilities = t(loc=0.0, scale=1.0, df=sample_size).cdf(_t_scores)
    return _t_scores[
        where(t_probabilities <= p)[0][-1]
    ]
