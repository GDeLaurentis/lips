#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe

import mpmath
import numpy
import random
import re

mpmath.mp.dps = 300

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


LeviCivita = numpy.array([[0, 1], [-1, 0]])
MinkowskiMetric = numpy.diag([1, -1, -1, -1])
Pauli_zero = numpy.diag([1, 1])
Pauli_x = numpy.array([[0, 1], [1, 0]])
Pauli_y = numpy.array([[0, -1j], [1j, 0]])
Pauli_z = numpy.array([[1, 0], [0, -1]])
Pauli = numpy.array([Pauli_zero, Pauli_x, Pauli_y, Pauli_z])
Pauli_bar = numpy.array([Pauli_zero, -Pauli_x, -Pauli_y, -Pauli_z])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


pSijk = re.compile(r'^(?:s|S)(?:_){0,1}(\d+)$')
pMi = re.compile(r'^(?:m|M)(?:_){0,1}(\d)$')
pMVar = re.compile(r'^((?:m|M|μ)(?:_){0,1}[a-zA-Z]*[\d]*)$')
pd5 = re.compile(r'^δ5$')
ptr5 = re.compile(r'^tr5_(\d+)$|^tr5\(([\d\|\+\-]+)\)$')
ptr = re.compile(r'^tr\((?P<middle>(?:(?:\([\d+\+|-]{1,}\))|(?:[\d+\+|-]{1,}))*)\)$')   # the 'middle' pattern should be like in pNB
pDijk = re.compile(r'(?:^Δ_(\d+)$)|(?:^Δ_(\d+)\|(\d+)\|(\d+)$)')
pOijk = re.compile(r'^(?:Ω_)(\d+)$')
pPijk = re.compile(r'^(?:Π_)(\d+)$')
pAu = re.compile(r'^(?:⟨|<)(\d+)(?:\|)$')
pAd = re.compile(r'^(?:\|)(\d+)(?:⟩|>)$')
pA2 = re.compile(r'^(?:⟨|<)(\d+)(?:\|)(\d+)(?:⟩|>)$')
pS2 = re.compile(r'^(?:\[)(\d+)(?:\|)(\d+)(?:\])$')
pSd = re.compile(r'^(?:\[)(\d+)(?:\|)$')
pSu = re.compile(r'^(?:\|)(\d+)(?:\])$')
p3B = re.compile(r'^(?:⟨|\[)(\d+)(?:\|\({0,1})([\d+[\+-]*]*)(?:\){0,1}\|)(\d+)(?:⟩|\])$')
pNB = re.compile(r'^(?:⟨|\[)(?P<start>\d+)(?:\|)(?P<middle>(?:(?:\([\d+\+|-]{1,}\))|(?:[\d+\+|-]{1,}))*)(?:\|)(?P<end>\d+)(?:⟩|\])$')
pNB_open_begin = re.compile(r'^(?:\|)(?P<middle>(?:(?:\([\d+|-]{1,}\))|(?:[\d+|-]{1,}))*)(?:\|)(?P<end>\d+)(?:⟩|\])$')
pNB_open_end = re.compile(r'^(?:⟨|\[)(?P<start>\d+)(?:\|)(?P<middle>(?:(?:\([\d+|-]{1,}\))|(?:[\d+|-]{1,}))*)(?:\|)$')

# '(⟨a|b|c+d|e|a]-⟨b|f|c+d|e|b])'  -  from two-loop five-point one-mass alphabet
p5Bdiff = re.compile(r'^\(⟨(?P<a>\d+)\|(?P<b>\d+)\|\({0,1}(?P<cd>[\d+[\+]*]*)\){0,1}\|(?P<e>\d+)\|(?P=a)\]\-⟨(?P=b)\|(?P<f>\d+)\|\({0,1}(?P=cd)\){0,1}\|(?P=e)\|(?P=b)\]\)$')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def rand_frac():
    return mpmath.mpc(random.randrange(-100, 101)) / mpmath.mpc(random.randrange(1, 201))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def flatten(temp_list, recursion_level=0, treat_list_subclasses_as_list=True, treat_tuples_as_lists=False, max_recursion=None):
    from sympy.matrices.dense import MutableDenseMatrix
    from numpy import ndarray
    flat_list = []
    for entry in temp_list:
        if type(entry) is list and (max_recursion is None or recursion_level < max_recursion):
            flat_list += flatten(entry, recursion_level=recursion_level + 1, treat_list_subclasses_as_list=treat_list_subclasses_as_list,
                                 treat_tuples_as_lists=treat_tuples_as_lists, max_recursion=max_recursion)
        elif ((issubclass(type(entry), list) or type(entry) in [MutableDenseMatrix, ndarray]) and
              treat_list_subclasses_as_list is True and (max_recursion is None or recursion_level < max_recursion)):
            flat_list += flatten(entry, recursion_level=recursion_level + 1, treat_list_subclasses_as_list=treat_list_subclasses_as_list,
                                 treat_tuples_as_lists=treat_tuples_as_lists, max_recursion=max_recursion)
        elif (type(entry) is tuple and treat_tuples_as_lists is True and (max_recursion is None or recursion_level < max_recursion)):
            flat_list += flatten(entry, recursion_level=recursion_level + 1, treat_list_subclasses_as_list=treat_list_subclasses_as_list,
                                 treat_tuples_as_lists=treat_tuples_as_lists, max_recursion=max_recursion)
        else:
            flat_list += [entry]
    return flat_list


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def subs_dict(text, substitutions):
    pattern = re.compile("|".join(map(re.escape, substitutions.keys())))
    result = pattern.sub(lambda match: substitutions[match.group(0)], text)
    return result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def det2x2(array):
    return array[0, 0] * array[1, 1] - array[0, 1] * array[1, 0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def ldot(oP1, oP2):
    return numpy.trace(numpy.dot(oP1.r2_sp, oP2.r2_sp_b)) / 2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def indexing_decorator(func):
    """Rebases a list to start from index 1."""

    def decorated(self, index, *args):

        # for now do not decorate slices (might want to shift this as well)
        if isinstance(index, slice) or isinstance(index, str):
            return func(self, index, *args)

        if index < 1:
            raise IndexError('Indices start from 1')
        elif index > 0 and index < len(self) + 1:
            index -= 1
        elif index > len(self):
            raise IndexError('Indices can\'t exceed {}'.format(len(self)))

        return func(self, index, *args)

    return decorated


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class myException(Exception):
    pass
