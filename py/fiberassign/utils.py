# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
fiberassign.utils
=======================

Utility functions.

"""
from __future__ import absolute_import, division, print_function

import os

from ._internal import (Logger, Timer, GlobalTimers, Circle, Segments, Shape,
                        Environment)

# Multiprocessing environment setup

default_mp_proc = None
"""Default number of multiprocessing processes.
Set globally on first import.
"""

if "SLURM_CPUS_PER_TASK" in os.environ:
    default_mp_proc = int(os.environ["SLURM_CPUS_PER_TASK"])
else:
    import multiprocessing as _mp
    default_mp_proc = max(1, _mp.cpu_count() // 2)


def option_list(opts):
    """Convert key, value pairs into a list.

    This converts a dictionary into an options list that can be passed to
    ArgumentParser.parse_args().  The value for each dictionary key will be
    converted to a string.  Values that are True will be assumed to not have
    a string argument added to the options list.

    Args:
        opts (dict):  Dictionary of options.

    Returns:
        (list): The list of options.

    """
    optlist = []
    for key, val in opts.items():
        keystr = "--{}".format(key)
        if val is not None:
            if isinstance(val, bool):
                if val:
                    optlist.append(keystr)
            else:
                optlist.append(keystr)
                if isinstance(val, float):
                    optlist.append("{:.14e}".format(val))
                elif isinstance(val, (list, tuple)):
                    optlist.extend(val)
                else:
                    optlist.append("{}".format(val))
    return optlist

# This is effectively the "Painter's Partition Problem".

def distribute_required_groups(A, max_per_group):
    ngroup = 1
    total = 0
    for i in range(A.shape[0]):
        total += A[i]
        if total > max_per_group:
            total = A[i]
            ngroup += 1
    return ngroup

def distribute_partition(A, k):
    low = np.max(A)
    high = np.sum(A)
    while low < high:
        mid = low + int((high - low) / 2)
        required = distribute_required_groups(A, mid)
        if required <= k:
            high = mid
        else:
            low = mid + 1
    return low

def distribute_discrete(sizes, groups, pow=1.0):
    """Distribute indivisible blocks of items between groups.

    Given some contiguous blocks of items which cannot be
    subdivided, distribute these blocks to the specified
    number of groups in a way which minimizes the maximum
    total items given to any group.  Optionally weight the
    blocks by a power of their size when computing the
    distribution.

    Args:
        sizes (list): The sizes of the indivisible blocks.
        groups (int): The number of groups.
        pow (float): The power to use for weighting

    Returns:
        A list of tuples.  There is one tuple per group.
        The first element of the tuple is the first item
        assigned to the group, and the second element is
        the number of items assigned to the group.
    """
    chunks = np.array(sizes, dtype=np.int64)
    weights = np.power(chunks.astype(np.float64), pow)
    max_per_proc = float(distribute_partition(weights.astype(np.int64), groups))

    target = np.sum(weights) / groups

    dist = []

    off = 0
    curweight = 0.0

    for cur in range(0, weights.shape[0]):
        if curweight + weights[cur] > max_per_proc:
            dist.append((off, cur - off))
            over = curweight - target
            curweight = weights[cur] + over
            off = cur
        else:
            curweight += weights[cur]

    dist.append((off, weights.shape[0] - off))

    if len(dist) != groups:
        raise RuntimeError(
            "Number of distributed groups different than number requested"
        )
    return dist
