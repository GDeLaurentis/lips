#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   ___          _   _    _          ___      _
#  | _ \__ _ _ _| |_(_)__| |___ ___ / __| ___| |_
#  |  _/ _` | '_|  _| / _| / -_|_-<_\__ \/ -_)  _|
#  |_| \__,_|_|  \__|_\__|_\___/__(_)___/\___|\__|

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
import mpmath

from .tools import pSijk, pDijk, pOijk, pPijk, pA2, pS2, pNB, myException

mpmath.mp.dps = 300

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_Set:

    def set(self, temp_string, temp_value, fix_mom=True, mode=1, itr=2, prec=0.1):
        """Constructs a given collinear limit phase space."""
        for i in range(itr):
            self._set_inner(temp_string, temp_value, fix_mom, mode)
            # check it worked
            abs_diff = abs(self.compute(temp_string) - temp_value)
            error = abs(100) * abs(abs_diff / temp_value)
            if error < prec:   # if error is less than 1 in 1000 or it is compatible with zero
                if i == 0:
                    return True
                else:
                    print("Succeded to set {} to {} but in {} tries.".format(temp_string, temp_value, i + 1))
                    return True
            if "nan" in str(self.compute(temp_string)):
                myException("NaN encountered in set!")
                break
        myException("Failed to set {} to {}. The target was {}, the actual value was {}, the error was {}.".format(temp_string, temp_value, temp_value, self.compute(temp_string), error))
        return False

    def _set_inner(self, temp_string, temp_value, fix_mom=True, mode=1):
        self.check_consistency(temp_string)                          # Check consistency of string

        if pA2.findall(temp_string) != []:                           # Sets ⟨A|B⟩  --- Changes: |B⟩, Don't touch: ⟨A|

            self._set_A2(temp_string, temp_value, fix_mom)

        elif pS2.findall(temp_string) != []:                         # Sets [A|B]  --- Changes: |B], Don't touch: [A|

            self._set_S2(temp_string, temp_value, fix_mom)

        elif pNB.findall(temp_string) != []:                         # Sets ⟨a|(b+c)|...|d]  --- Changes: ⟨a| (mode=1), |b⟩ (mode=2)

            raise Exception("Not implement in alpha version.")

        elif pSijk.findall(temp_string) != []:                       # Sets S_ijk  --- Changes: ⟨i| (mode=1) or |i] (mode=2), Don't touch ijk...

            raise Exception("Not implement in alpha version.")

        elif pDijk.findall(temp_string) != []:                       # Sets Δ_ijk  --- Changes: last two [j] moment, Don't touch [j]'s or [i]'s

            raise Exception("Not implement in alpha version.")

        elif pOijk.findall(temp_string) != []:

            raise Exception("Not implement in alpha version.")

        elif pPijk.findall(temp_string) != []:

            raise Exception("Not implement in alpha version.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_A2(self, temp_string, temp_value, fix_mom=True):       # ⟨A|B⟩ = (a, b).(c, d) = ac+bd = X ----> c = (X - bd)/a

        A, B = list(map(int, pA2.findall(temp_string)[0]))
        X = temp_value
        plist = list(map(int, self._complementary(map(str, [A, B]))))  # free momenta
        if len(plist) < 2:                                           # need at least 4 particles to fix mom cons (i.e. two free ones)
            myException("Set_A2 called with less than 4 particles. Cound't fix momentum conservation.")
        a, b = self[A].r_sp_u[0, 0], self[A].r_sp_u[0, 1]            # ⟨A| = (a, b)
        c, d = self[B].r_sp_d[0, 0], self[B].r_sp_d[1, 0]            # |B⟩ = (c, d)
        c = (X - b * d) / a                                          # c = (X - b * d) / a
        self[B].r_sp_d = numpy.array([c, d], dtype=type(c))          # set |B⟩
        if fix_mom is True:
            self.fix_mom_cons(plist[0], plist[1], axis=2)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_S2(self, temp_string, temp_value, fix_mom=True):       # [A|B] = (a, b).(c, d) = ac+bd = X ----> c = (X - bd)/a

        A, B = list(map(int, pS2.findall(temp_string)[0]))
        X = temp_value
        plist = list(map(int, self._complementary(map(str, [A, B]))))  # free momenta
        if len(plist) < 2:                                           # need at least 4 particles to fix mom cons (i.e. two free ones)
            myException("Set_S2 called with less than 4 particles. Cound't fix momentum conservation.")
        a, b = self[A].l_sp_d[0, 0], self[A].l_sp_d[0, 1]            # [A| = (a, b)
        c, d = self[B].l_sp_u[0, 0], self[B].l_sp_u[1, 0]            # |B] = (c, d)
        c = (X - b * d) / a                                          # c = (X - b * d) / a
        self[B].l_sp_u = numpy.array([c, d], dtype=type(c))          # set |B]
        if fix_mom is True:
            self.fix_mom_cons(plist[0], plist[1], axis=1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
