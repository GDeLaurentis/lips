# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import sympy
import mpmath
import numpy
import subprocess
import itertools

from copy import deepcopy
from lips.fields.finite_field import ModP

if sys.version_info.major < 3:
    from whichcraft import which
else:
    from shutil import which


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


if which('Singular') is not None:
    singular_found = True
    test = subprocess.Popen(["Singular", "--execute", "$"], stdout=subprocess.PIPE)
    output = test.communicate()[0]
    singular_clean_up_lines = output.decode("utf-8").split("\n")
    singular_clean_up_lines = singular_clean_up_lines + ["empty list", "// ** groebner base computations with inexact coefficients can not be trusted due to rounding errors", "Auf Wiedersehen."]
else:
    singular_found = False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def univariate_floating_point_solver(equation, root_dict):
    """Returns all possible solutions of 'equation' over arbitrary precision complex numbers."""
    equation = sympy.sympify(equation).subs(root_dict)
    free_symbols = list(equation.free_symbols)
    assert len(free_symbols) == 1
    symbol = free_symbols[0]
    solutions = list(map(mpmath.mpc, sympy.nroots(equation, n=300, maxsteps=500)))  # mpmath.polyroots is faster, but the parsing is more complicated
    return update_root_dict(symbol, solutions, root_dict)


def univariate_finite_field_solver(equation, root_dict, prime):
    """Returns all possible solutions of 'equation' over a finite field of cardinality 'prime'.
       If already satisfied returns True, if no solution exists returns False."""
    equation = sympy.sympify(equation).subs(root_dict)
    if isinstance(equation, sympy.numbers.Integer) and equation % prime == 0:
        return True
    free_symbols = list(equation.free_symbols)
    if len(free_symbols) < 1:
        return False
    if len(free_symbols) > 1:
        raise Exception("Too many free parameters.")
    symbol = free_symbols[0]
    equation = sympy.poly(equation, modulus=prime)
    if equation == 0:
        return True
    pre_factor, factors = sympy.factor_list(sympy.factor(equation, modulus=prime))
    factors = [factor[0] for factor in factors]
    if pre_factor % prime == 0:
        return True
    linear_factors = [factor for factor in factors if sympy.diff(factor, symbol) == 1]
    if linear_factors == []:
        return False
    solutions = [ModP(int(sympy.solve(factor)[0]), prime) for factor in factors]
    return update_root_dict(symbol, solutions, root_dict)


def update_root_dict(symbol, solutions, root_dict):
    """Given solutions and root_dict returns updated root_dicts."""
    root_dicts = [deepcopy(root_dict)]
    root_dicts[0].update({symbol: solutions[0]})
    for solution in solutions[1:]:
        new_root_dict = deepcopy(root_dict)
        new_root_dict.update({symbol: solution})
        root_dicts.append(new_root_dict)
    return root_dicts


def lex_groebner_solve(equations, prime=None):
    """Returns the variety corresponding to a given zero dimensional ideal in lexicographic groebner basis form.
       The variety take the form of a list of dictionaries for the possible values of the variables."""
    root_dicts = [{}]
    for equation in equations:
        temp_dicts = []
        for i, root_dict in enumerate(root_dicts):
            if prime is None:
                sols = univariate_floating_point_solver(equation, root_dict)
            else:
                sols = univariate_finite_field_solver(equation, root_dict, prime)
            if sols is True:
                temp_dicts += [root_dict]
            elif sols is False:
                continue
            else:
                temp_dicts += sols
        else:
            root_dicts = temp_dicts
    return root_dicts


def check_solutions(equations, root_dicts, prime=None):
    """Checks that all solutions in root_dicts solve the equations."""
    for root_dict in root_dicts:
        check = []
        for equation in equations:
            check += [sympy.sympify(equation)]
            for key, value in root_dict.items():
                if prime is None:
                    check[-1] = sympy.simplify(check[-1].subs(key, value))
                else:
                    check[-1] = check[-1].subs(key, value) % prime
        if prime is None:
            assert all([numpy.isclose(complex(entry), 0) for entry in check])
        else:
            assert all([entry == 0 for entry in check])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def lips_symbols(multiplicity):
    """Returns a list of sympy symbols: [a1, b1, c1, d1, ...].
       If using make_analytical_d we have a1=oPs[1].r_sp_d[0,0], b1=oPs[1].r_sp_d[1,0], c1=oPs[1].l_sp_d[0,0], d1=oPs[1].l_sp_d[0,1]."""
    la = sympy.symbols('a1:{}'.format(multiplicity + 1))
    lb = sympy.symbols('b1:{}'.format(multiplicity + 1))
    lc = sympy.symbols('c1:{}'.format(multiplicity + 1))
    ld = sympy.symbols('d1:{}'.format(multiplicity + 1))
    iters = map(iter, [la, lb, lc, ld])
    return list(next(it) for it in itertools.cycle(iters))
