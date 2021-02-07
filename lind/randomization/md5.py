"""
md5: Hashing can be used to generate reproducible pseudo-randomization. This can be useful in
contexts where the user does not want to store a fixed seed to ensure replicability of test
randomization.

Examples
--------
>>> # numpy randomization with fixed seed
>>> random_state = np.random.RandomState(42)
>>> random_state.choice(["Lauren", "Sam", "Ben"], size=1)
>>> # hd5 random sample (no seed required)
>>> md5shuffle(["Lauren", "Sam", "Ben"])[0]

"""

import logging
from typing import Union, List

from numpy import ndarray, vectorize, asarray
from hashlib import md5

# set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# define public functions (ignored by jupyter notebooks)
__all__ = ["md5shuffle", "draw_percentile"]


####################################################################################################


_hash_size = md5().digest_size
_hash_max_length = 2.0**(8.0*_hash_size)


def _str_to_md5_hexidec(s: str) -> hex:
    """utility for converting a string into an  MD5 hexidecimal hash"""
    hd5 = md5(s.encode())
    # byte = hd5.digest()
    hexadecimal = hd5.hexdigest()
    return hexadecimal


_str_to_md5_hexidec = vectorize(_str_to_md5_hexidec,
                                doc="Utility for converting an array of strings into an array of  "
                                    "MD5 hexidecimal hashs")


def md5shuffle(arr: ndarray) -> ndarray:
    """
    md5shuffle

    Will shuffle the input array pseudo-randomly in a deterministic manner using MD5 hashing.

    Parameters
    ----------
    arr: list, ndarray
        The array of values that you want shuffled

    Returns
    -------
    ndarray
        the input array in a shuffled order


    Examples
    --------
    >>> shuffle([i for i in range(1000)])

    """

    return arr[
        _str_to_md5_hexidec(asarray(arr).astype(str)).argsort()
    ]


# def _find_hash_space(s):
#     """Utility to convert first 6 hex digits to an int"""
#     return int(s[:6], 16)


# _vfind_hash_space = vectorize(_find_hash_space)


def _hash_to_int(s: str) -> int:
    """utility to transform md5 hash to an integer"""
    return int(s, _hash_size)


_vhash_to_int = vectorize(_hash_to_int(),
                          doc="utility to transform an array of md5 hashs to an array of integers")


def draw_percentile(arr: Union[List, ndarray], lb: float = 0.25, ub: float = 0.75) -> ndarray:
    """
    draw_percentile

    Draw array values that fall within a certain percentile of the hash space.

    Parameters
    ----------
    arr: list, ndarray
        An array of objects that you want to sample from
    lb: float, optional
        The lower bound of the percentile; must be between 0 and 1
    ub: float, optional
        The upper bound of the percentile; must be between 0 and 1; must be greater than lb

    Returns
    -------
    ndarray
        an array of values from arr that fall within the specified percentile of the hash space

    Examples
    --------
    >>> draw_percentile([i for i in range(1000)], lb=0.25, ub=0.75) # sample 50% of inputs

    """

    if ub <= lb:
        raise ValueError("Input ub must be greater than input lb.")

    hash_arr = _str_to_md5_hexidec(asarray(arr).astype(str))

    # hash_spaces = _vfind_hash_space(hash_arr)
    # max_hash = 0xffffff
    # lb *= max_hash
    # ub *= max_hash
    # return arr[
    #   np.where((hash_spaces >= lb) & (hash_spaces <= ub))[0]
    # ]

    hash_arr = _vhash_to_int(hash_arr).astype(float)
    percentiles = hash_arr / _hash_max_length
    return arr[
        (percentiles >= lb) & (percentiles <= ub)
    ]
