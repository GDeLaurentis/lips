#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe

import mpmath
import numpy
import random
import re
import warnings

from syngular import flatten  # noqa

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
pDijk = re.compile(r'^Δ_(\d+(?:\|\d+)*)$')
pOijk = re.compile(r'^(?:Ω_)(\d+)$')
pPijk = re.compile(r'^(?:Π_)(\d+)$')
pAu = re.compile(r'^(?:⟨|<)(\d+)(?:\|)$')
pAd = re.compile(r'^(?:\|)(\d+)(?:⟩|>)$')
pA2 = re.compile(r'^(?:⟨|<)(\d+)(?:\|)(\d+)(?:⟩|>)$')
pS2 = re.compile(r'^(?:\[)(\d+)(?:\|)(\d+)(?:\])$')
pSd = re.compile(r'^(?:\[)(\d+)(?:\|)$')
pSu = re.compile(r'^(?:\|)(\d+)(?:\])$')
p3B = re.compile(r'^(?:⟨|\[)(\d+)(?:\|\({0,1})([\d+[\+-]*]*)(?:\){0,1}\|)(\d+)(?:⟩|\])$')
pNB = re.compile(r'^(?:<|⟨|\[)(?P<start>\d+)\|(?P<middle>(?:\(?(?:\d+[\+|-]?)+\)?\|?)+)\|(?P<end>\d+)(?:⟩|\]|>)$')
pNB_open_begin = re.compile(r'^(?:\|)(?P<middle>(?:(?:\([\d]+(?:[\+|-]\d+)*\))|(?:[\d]+(?:[\+|-]\d+)*))+)(?:\|)(?P<end>\d+)(?:⟩|\])$')
pNB_open_end = re.compile(r'^(?:⟨|\[)(?P<start>\d+)(?:\|)(?P<middle>(?:(?:\([\d]+(?:[\+|-]\d+)*\))|(?:[\d]+(?:[\+|-]\d+)*))+)(?:\|)$')
pNB_double_open = re.compile(r'^(?:\|)(?P<middle>(?:(?:\([\d]+(?:[\+|-]\d+)*\))|(?:[\d]+(?:[\+|-]\d+)*))+)(?:\|)$')

# '(⟨a|b|c+d|e|a]-⟨b|f|c+d|e|b])'  -  from two-loop five-point one-mass alphabet
p5Bdiff = re.compile(r'^\(⟨(?P<a>\d+)\|(?P<b>\d+)\|\({0,1}(?P<cd>[\d+[\+]*]*)\){0,1}\|(?P<e>\d+)\|(?P=a)\]\-⟨(?P=b)\|(?P<f>\d+)\|\({0,1}(?P=cd)\){0,1}\|(?P=e)\|(?P=b)\]\)$')

bold_digits = {'0': '𝟎', '1': '𝟏', '2': '𝟐', '3': '𝟑', '4': '𝟒', '5': '𝟓', '6': '𝟔', '7': '𝟕', '8': '𝟖', '9': '𝟗'}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def rand_frac():
    return mpmath.mpc(random.randrange(-100, 101)) / mpmath.mpc(random.randrange(1, 201))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def subs_dict(text, substitutions, escape=True):
    if escape:
        pattern = re.compile("|".join(re.escape(k) for k in substitutions))
        return pattern.sub(lambda m: substitutions[m.group(0)], text)
    else:
        # Build a list of (compiled_pattern, replacement)
        patterns = [(re.compile(k), v) for k, v in substitutions.items()]
        for pat, repl in patterns:
            text = pat.sub(repl, text)
        return text


def rsubs_dict(text, substitutions, escape=True):
    reverse_subs = {v: k for k, v in substitutions.items()}
    if escape:
        pattern = re.compile("|".join(re.escape(k) for k in reverse_subs))
        return pattern.sub(lambda m: reverse_subs[m.group(0)], text)
    else:
        patterns = [(re.compile(k), v) for k, v in reverse_subs.items()]
        for pat, repl in patterns:
            text = pat.sub(repl, text)
        return text


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def det(matrix):
    if matrix.shape == (1, 1):
        return matrix[0, 0]

    det_sum = 0
    for col in range(matrix.shape[1]):
        sub_matrix = matrix[1:, [i for i in range(matrix.shape[1]) if i != col]]
        det_sum += (-1) ** col * matrix[0, col] * det(sub_matrix)

    return det_sum


def det2x2(matrix):
    warnings.warn("det2x2 is deprecated, use det.", DeprecationWarning, stacklevel=2)
    return det(matrix)


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
