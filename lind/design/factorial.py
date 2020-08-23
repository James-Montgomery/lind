"""
factorial: This module contains tools for designing factorial experiments.
"""

import logging
from typing import Union, List, Optional

from itertools import product, combinations
from fractions import Fraction

import numpy as np
from scipy.special import binom

import pandas as pd
from patsy import dmatrix  # pylint: disable=no-name-in-module

# set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# define public functions (ignored by jupyter notebooks)
__all__ = ['design_full_factorial', 'design_partial_factorial']


####################################################################################################


def _array_to_string(arr_like: Union[List, np.ndarray]) -> np.ndarray:
    """Utility for converting experiment design string into an array of factors"""
    return np.array_str(np.asarray(arr_like)).replace("[", "").replace("]", "")


def _k_combo(k: int, res: int) -> int:
    """The number of combinations of k factors given a specific resolution"""
    return binom(
        np.full(k - res + 1, k),
        np.arange(res - 1, k, 1)
    ).sum() + k


_k_combo_vec = np.vectorize(_k_combo, excluded=['res'],
                            doc="The number of combinations of k factors "
                                "given a specific resolution")


def _filter_by_length(words: Union[List, np.ndarray], size: int = 1, operator: str = "eq") -> List:
    """"""
    if operator == "eq":
        return [word for word in words if len(word) == size]
    if operator == "lt":
        return [word for word in words if len(word) < size]
    if operator == "gt":
        return [word for word in words if len(word) > size]
    raise Exception("Invalid operator {}".format(operator))


####################################################################################################


def design_full_factorial(factors: List[List],
                          factor_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    design_full_factorial

    This function helps create a full factorial experiment design. Given how easy it is to design a
    full factorial experiment once the factors and levels have been specified, this is more of a
    convenience function.

    Parameters
    ----------
    factors : List[List]
        a list of lists representing factors and levels
    factor_names : List[str]
        a list of names for the factors in the first argument. Must share the order of the first
        argument.

    Returns
    -------
    pd.DataFrame

    Example
    -------
    >>> # create full factorial design for a 2 level 3 factor experiment
    >>> design_df = design_full_factorial(factors=[[-1, 1], [-1,1], [-1, 1]],
    >>>     factor_names=["factor_one", "factor_two", "factor_three"])
    """
    assert factor_names is None or len(factor_names) == len(factors), \
        "The length of factor_names must match the length of factors."
    factor_names = factor_names if factor_names is not None else \
        ["x{}".format(i) for i in range(len(factors))]
    return pd.DataFrame(data=list(product(*factors)), columns=factor_names)


def design_partial_factorial(k: int, res: int) -> pd.DataFrame:
    """
    design_partial_factorial

    This function helps design 2 level partial factorial experiments. These experiments are often
    described using the syntax l**(k-p) where l represents the level of each factor, k represents
    the total number of factors considered, and p represents a scaling factor relative to the full
    factorial design.

    This function assumes that l=2. Users are not asked to set p, instead the user sets a minimum
    desired resolution for their experiment. Resolution describes the kind of aliasing incurred by
    scaling down from a full to a partial factorial design. Higher resolutions have less potential
    aliasing (confounding).

    Resolution number is determined through the defining relation of the partial factorial design.
    For the 6 factor design 2**(6-p) with factors ABCDEF, example defining relations (I) are shown
    below. The resolution cannot exceed the number of factors in the experiment. So a 6 factor
    experiment can be at most a resolution 6 (otherwise it would be a full factorial experiment).
    * Res I: I = A
    * Res II: I = AB
    * Res III: I = ABC
    * Res IV: I = ABCD
    * Res V: I = ABCDE
    * Res VI: I = ABCDEF

    Practically we tend to use resolution III-, IV- and V-designs.
    * Res I: Cannot distinguish between levels within main effects (not useful).
    * Res II: Main effects may be aliased with other main effects (not useful).
    * Res III: Main effects may be aliased with two-way interactions.
    * Res IV: Two-way interactions may be aliased with each other.
    * Res V: Two-way interactions may be aliased with three-way interactions.
    * Res VI: Three-way interactions may be aliased with each other.

    Parameters
    ----------
    k : int
        the total number of factors considered in the experiment
    res : int
        the desired minimum resolution of the experiment

    Returns
    -------
    pd.DataFrame
        A dataframe with the partial factorial design

    Example
    -------
    >>> # create partial factorial design for a 2 level 4 factor resolution III experiment
    >>> design_df = design_partial_factorial(k=4, res=3)
    """

    assert res <= k, "Resolution must be smaller than or equal to the number of factors."

    # Assume l=2 and use k specified by user to solve for p in design
    n = np.arange(res - 1, k, 1)
    k_minus_p = k - 1 if res == k else n[~(_k_combo_vec(n, res) < k)][0]
    logging.info("Partial Factorial Design: l=2, k={}, p={}".format(k, k - k_minus_p))
    logging.info("Ratio to Full Factorial Design: {}".format(Fraction(2**k_minus_p)))

    # identify the main effects and interactions for the design
    main_factors = np.arange(k_minus_p)
    interactions = [_array_to_string(main_factors).replace(" ", ":")] if res == k else \
        [
            _array_to_string(c).replace(" ", ":")
            for r in range(res - 1, k_minus_p)
            for c in combinations(main_factors, r)
        ][:k - k_minus_p]

    # combine main effects and interactions into a single design string (format inspired by patsy)
    factors = " ".join([_array_to_string(main_factors)] + interactions)
    logging.info("Design string: {}".format(factors))

    main_factors = _filter_by_length(factors.split(" "), 1, "eq")
    two_level_full_factorial = [[-1, 1] for _ in main_factors]
    full_factorial_design = design_full_factorial(two_level_full_factorial)

    interactions = [
        ["x" + i for i in j.split(":")]
        for j in _filter_by_length(factors.split(" "), 1, "gt")
    ]

    # code below replaced by patsy
    # partial_factorial_design = append_interactions(full_factorial_design, interactions, False)

    design = "+".join(full_factorial_design.columns.tolist() + [":".join(i) for i in interactions])
    partial_factorial_design = dmatrix(design, full_factorial_design, return_type='dataframe').drop(
        columns=["Intercept"], axis=1)

    partial_factorial_design.columns = \
        ["x{}".format(i) for i in range(partial_factorial_design.shape[1])]

    return partial_factorial_design
