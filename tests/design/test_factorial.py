"""

"""

import pandas as pd

import pytest
from pandas.util.testing import assert_frame_equal


from lind.design.factorial import design_full_factorial, design_partial_factorial


########################################################################################################################


@pytest.mark.parametrize("factors, factor_names, expected_output", [
    ([[-1, 1], [-1, 1], [-1, 1]], ["factor_one", "factor_two", "factor_three"],
     pd.DataFrame({
        'factor_one': {0: -1, 1: -1, 2: -1, 3: -1, 4: 1, 5: 1, 6: 1, 7: 1},
        'factor_two': {0: -1, 1: -1, 2: 1, 3: 1, 4: -1, 5: -1, 6: 1, 7: 1},
        'factor_three': {0: -1, 1: 1, 2: -1, 3: 1, 4: -1, 5: 1, 6: -1, 7: 1}
    })),
    ([[-1, 0, 1], ["high", "low"], [-1, 0, 1, 2]], ["factor_one", "factor_two", "factor_three"],
     pd.DataFrame({
        'factor_one': {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0,
        13: 0, 14: 0, 15: 0, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1},
        'factor_two': {0: 'high', 1: 'high', 2: 'high', 3: 'high', 4: 'low', 5: 'low', 6: 'low', 7: 'low', 8: 'high',
        9: 'high', 10: 'high', 11: 'high', 12: 'low', 13: 'low', 14: 'low', 15: 'low', 16: 'high', 17: 'high',
        18: 'high', 19: 'high', 20: 'low', 21: 'low', 22: 'low', 23: 'low'},
        'factor_three': {0: -1, 1: 0, 2: 1, 3: 2, 4: -1, 5: 0, 6: 1, 7: 2, 8: -1, 9: 0, 10: 1, 11: 2, 12: -1, 13: 0,
        14: 1, 15: 2, 16: -1, 17: 0, 18: 1, 19: 2, 20: -1, 21: 0, 22: 1, 23: 2}
     }))
    ])
def test_design_full_factorial_expected(factors, factor_names, expected_output):
    """
    test_design_full_factorial expected

    Check that the proper full factorial designs are returned.
    """

    assert_frame_equal(
        design_full_factorial(factors, factor_names),
        expected_output
    )


def test_design_full_factorial_value_error():
    """
    test_design_full_factorial value error

    Make sure an exception is thrown if the input lengths do not match.
    """

    with pytest.raises(Exception) as execinfo:
        design_full_factorial(
            factors=[[0, 1], [0, 1], [0, 1]],
            factor_names=["Factor_One", "Factor_Two"]
        )

    assert str(execinfo.value) == "The length of factor_names must match the length of factors."


########################################################################################################################


def design_partial_factorial():
    """
    design_partial_factorial


    """
    pass

