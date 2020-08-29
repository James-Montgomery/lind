"""
plackett_burman module tests
"""

from numpy import any

import pytest

from lind.design.plackett_burman import fetch_plackett_burman_design

####################################################################################################


@pytest.mark.parametrize("num_factors", [i for i in range(1, 48)]+[7.0])
def test_fetch_plackett_burman_design_orthogonal(num_factors):
    """
    fetch_plackett_burman_design orthogonal

    Ensure that this function returns orthogonal designs. These designs are validated by trusted
    sources, but my motto is "trust but verify".
    """
    assert not any(fetch_plackett_burman_design(num_factors).sum(axis=0).values)


@pytest.mark.parametrize("num_factors", ["bad input", 7.1])
def test_fetch_plackett_burman_design_value_error(num_factors):
    """
    fetch_plackett_burman_design value error

    Make sure an exception is thrown if the input lengths do not match.
    """

    with pytest.raises(Exception) as execinfo:
        fetch_plackett_burman_design(num_factors)

    assert str(execinfo.value) == "Input num_factors must be an integer."


if __name__ == '__main__':
    pytest.main(__file__)