#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   ___          _   _    _          ___      _
#  | _ \__ _ _ _| |_(_)__| |___ ___ / __| ___| |_
#  |  _/ _` | '_|  _| / _| / -_|_-<_\__ \/ -_)  _|
#  |_| \__,_|_|  \__|_\__|_\___/__(_)___/\___|\__|

# Author: Giuseppe


from __future__ import unicode_literals

import numpy as np
import re
import gmpTools

from antares.core.bh_patch import accuracy
from antares.core.tools import flatten, pSijk, pDijk, pOijk, pPijk, pA2, pS2, pNB, myException


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_Set:

    def set(self, temp_string, temp_value, fix_mom=True, mode=1, itr=50, prec=0.1):
        """Constructs a given collinear limit phase space."""
        for i in range(itr):
            if self.set_inner(temp_string, temp_value, fix_mom, mode) == "run me again":
                self.set_inner(temp_string, temp_value, fix_mom, mode)
            actual, target = abs(self.compute(temp_string)), abs(temp_value)
            error = abs(100) * abs((actual - target) / target)
            compatible_with_zero = abs(target - actual) < 10 ** -(0.9 * accuracy())
            if compatible_with_zero is False:
                compatible_with_zero = str(abs(0)) == str(target)
            if error < prec or compatible_with_zero is True:  # if error is less than 1 in 1000 or it is compatible with zero
                if i == 0:
                    return True
                else:
                    print("Succeded to set {} to {} but in {} tries.".format(temp_string, temp_value, i + 1))
                    return True
            if "nan" in str(actual):
                myException("NaN encountered in set!")
                break
        myException("Failed to set {} to {}. The target was {}, the actual value was {}, the error was {}.".format(temp_string, temp_value, target, actual, error))
        return False

    def set_inner(self, temp_string, temp_value, fix_mom=True, mode=1):
        self.check_consistency(temp_string)                          # Check consistency of string

        if pA2.findall(temp_string) != []:                           # Sets ⟨A|B⟩  --- Changes: |B⟩, Don't touch: ⟨A|

            self.set_A2(temp_string, temp_value, fix_mom)

        elif pS2.findall(temp_string) != []:                         # Sets [A|B]  --- Changes: |B], Don't touch: [A|

            self.set_S2(temp_string, temp_value, fix_mom)

        elif pNB.findall(temp_string) != []:                         # Sets ⟨a|(b+c)|...|d]  --- Changes: ⟨a| (mode=1), |b⟩ (mode=2)

            self.set_NB(temp_string, temp_value, fix_mom, mode)

        elif pSijk.findall(temp_string) != []:                       # Sets S_ijk  --- Changes: ⟨i| (mode=1) or |i] (mode=2), Don't touch ijk...

            self.set_Sijk(temp_string, temp_value, fix_mom, mode)

        elif pDijk.findall(temp_string) != []:                       # Sets Δ_ijk  --- Changes: last two [j] moment, Don't touch [j]'s or [i]'s

            self.set_Dijk(temp_string, temp_value, fix_mom, mode)

        elif pOijk.findall(temp_string) != []:

            self.set_Oijk(temp_string, temp_value, fix_mom)

        elif pPijk.findall(temp_string) != []:

            self.set_Pijk(temp_string, temp_value, fix_mom)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def set_A2(self, temp_string, temp_value, fix_mom=True):       # ⟨A|B⟩ = (a, b).(c, d) = ac+bd = X ----> c = (X - bd)/a

        A, B = map(int, pA2.findall(temp_string)[0])
        X = temp_value
        plist = map(int, self._complementary(map(unicode, [A, B])))  # free momenta
        if len(plist) < 2:                                           # need at least 4 particles to fix mom cons (i.e. two free ones)
            myException("Set_A2 called with less than 4 particles. Cound't fix momentum conservation.")
        a, b = self[A].r_sp_u[0, 0], self[A].r_sp_u[0, 1]            # ⟨A| = (a, b)
        c, d = self[B].r_sp_d[0, 0], self[B].r_sp_d[1, 0]            # |B⟩ = (c, d)
        c = (X - b * d) / a                                          # c = (X - b * d) / a
        self[B].r_sp_d = np.array([c, d])                            # set |B⟩
        if fix_mom is True:
            self.fix_mom_cons(plist[0], plist[1])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def set_S2(self, temp_string, temp_value, fix_mom=True):       # [A|B] = (a, b).(c, d) = ac+bd = X ----> c = (X - bd)/a

        A, B = map(int, pS2.findall(temp_string)[0])
        X = temp_value
        plist = map(int, self._complementary(map(unicode, [A, B])))  # free momenta
        if len(plist) < 2:                                           # need at least 4 particles to fix mom cons (i.e. two free ones)
            myException("Set_S2 called with less than 4 particles. Cound't fix momentum conservation.")
        a, b = self[A].l_sp_d[0, 0], self[A].l_sp_d[0, 1]            # [A| = (a, b)
        c, d = self[B].l_sp_u[0, 0], self[B].l_sp_u[1, 0]            # |B] = (c, d)
        c = (X - b * d) / a                                          # c = (X - b * d) / a
        self[B].l_sp_u = np.array([c, d])                            # set |B]
        if fix_mom is True:
            self.fix_mom_cons(plist[0], plist[1])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def set_NB(self, temp_string, temp_value, fix_mom=True, mode=1):
        if "-" in temp_string:                                  # a minus sign would mess up inversion
            myException("Detected minus in string. Not implemented.")

        lNB, lNBs, lNBms, lNBe = self._get_lNB(temp_string)
        plist = self._complementary([lNB])

        if len(plist) < 2 and fix_mom is True:                  # if necessary look for alternative way to write it which allows to fix mom cons
            for i, iNBm in enumerate(lNBms):                    # try to flip the i^th bracket and see if len(plist) >= 2
                _lNBms = [entry for entry in lNBms]
                if i == 0:                                      # this is close to the head, what I call extremum of middle
                    alt = self._complementary(iNBm + [lNBs])
                elif i == len(lNBms) - 1:                         # this is close to the tail, what I call extremum of middle
                    alt = self._complementary(iNBm + [lNBe])
                else:                                           # this is not close to either head or tail
                    alt = self._complementary(iNBm)
                _lNBms[i] = alt
                plist = self._complementary([lNBs] + _lNBms + [lNBe])
                if len(plist) >= 2:                             # if this fixes it then reconstruct bc and temp_string
                    lNBms = _lNBms
                    start = temp_string[0] + unicode(lNBs) + "|"
                    end = "|" + unicode(lNBe) + temp_string[len(temp_string) - 1]
                    middle = "".join(string + "|" for string in ["(" + "".join(unicode(entry) + "+" for entry in item)[:-1] + ")" for item in lNBms])
                    middle = middle[:-1]
                    temp_string = start + middle + end
                    break

        unique_head_or_tail = []                                # see if head or tail are unique,
        unique_extrema_of_middle = []                           # or if there is a unique entry in extrema of middle
        temp_list = [[lNBs]] + lNBms + [[lNBe]]

        for i, ientry in enumerate([lNBs, lNBe]):               # see if head or tail are unique
            for k, kentry in enumerate(temp_list):
                if ((i == 0 and k == 0) or (i == 1 and k == 0 and len(lNBms) % 2 == 1) or
                   (i == 1 and k == len(temp_list) - 1) or (i == 0 and k == len(temp_list) - 1 and len(lNBms) % 2 == 1)):
                    continue
                if ientry in kentry:                            # compare to others, if it appear it's not what we are looking for
                    break
            else:                                               # if the previous loop completed normally then this is unique
                unique_head_or_tail = i                         # save if the unique entry is the head (0) or tail (1) or neither ([])
                break
            if unique_head_or_tail != []:
                break

        for i, ientry in enumerate(temp_list):                  # identify a unique entry in extrema of middle
            if i != 1 and i != len(temp_list) - 2:                # if it is head, tail or middle of middle skip it
                continue
            for j, jtem in enumerate(ientry):
                for k, kentry in enumerate(temp_list):
                    if k == i:                                  # this is the same list
                        continue
                    if jtem in kentry:                          # compare to others, if it appear it's not what we are looking for
                        break
                else:                                           # if the previous loop completed normally then this is unique
                    unique_extrema_of_middle = [jtem, j, i - 1]   # save the unique entry, the j^th position in which it appears in the bracket,
                    break                                       # and (i-1)^th bracket (because it gets compared with "bc" which doesn't contain head)
            if unique_extrema_of_middle != []:
                break

        a_or_s = temp_string[0]
        bc = ["".join(unicode(entry) + "+" for entry in item)[:-1] for item in lNBms]
        a, d = lNBs, lNBe

        # print temp_string, unique_head_or_tail                # debugging tool

        if (((unique_head_or_tail != [] or                      # this sets the head (or tail) leaving middle unchanged
              (lNBs == lNBe and lNBs not in flatten(lNBms) and
               lNBe not in flatten(lNBms) and len(lNBms) % 2 == 0)) and mode == 1) or
           (unique_extrema_of_middle == [])):
            if unique_head_or_tail == 1:                        # if the unique one is in the tail, flip it so that it is in the head
                temp_string = temp_string[::-1]
                temp_string = temp_string.replace("(", "X").replace(")", "(").replace("X", ")")
                temp_string = temp_string.replace("⟨", "X").replace("⟩", "⟨").replace("X", "⟩")
                temp_string = temp_string.replace("[", "X").replace("]", "[").replace("X", "]")
                self.set_inner(temp_string, temp_value, fix_mom, mode)
                return
            rest = np.array([[1, 0], [0, 1]])                   # initialise the rest to the identity matrix
            for i in range(len(bc)):
                comb_mom = re.sub(r'(\d)', r'self[\1].four_mom', bc[i])
                comb_mom = eval(comb_mom)
                if a_or_s == "⟨":
                    rest = np.dot(rest, self._four_mom_to_r2_sp_bar(comb_mom))
                    a_or_s = "["                                # needs to alternate the contraction of indices
                elif a_or_s == "[":
                    rest = np.dot(rest, self._four_mom_to_r2_sp(comb_mom))
                    a_or_s = "⟨"                                # needs to alternate the contraction of indices
            if a == d and len(lNBms) % 2 == 0:
                K11, K12, K21, K22 = rest[0, 0], rest[0, 1], rest[1, 0], rest[1, 1]
                if temp_string[0] == "⟨":
                    B = self[a].r_sp_d[1, 0]
                    A = (B * K11 - B * K22 + gmpTools.csqrt((B * K11 - B * K22) * (B * K11 - B * K22) + 4 * K21 * (B * B * K12 - temp_value))) / (2 * K21)
                    self[a].r_sp_d = np.array([A, B])
                else:
                    B = self[a].l_sp_d[0, 1]
                    A = (B * K11 - B * K22 + gmpTools.csqrt((B * K11 - B * K22) * (B * K11 - B * K22) + 4 * K12 * (B * B * K21 - temp_value))) / (2 * K12)
                    self[a].l_sp_d = np.array([A, B])
            else:
                if a_or_s == "⟨":
                    rest = np.dot(rest, self[d].r_sp_d)
                elif a_or_s == "[":
                    rest = np.dot(rest, self[d].l_sp_u)
                a_or_s = temp_string[0]                         # reset the angle_or_square bracket variable
                if a_or_s == "⟨":
                    _a, _b = self[a].r_sp_u[0, 0], self[a].r_sp_u[0, 1]  # ⟨A| = (a, b)
                    _c, _d = rest[0, 0], rest[1, 0]             # |rest⟩ = (c, d)
                    _a = (temp_value - _b * _d) / _c                     # a = (X - b*d)/c
                    self[a].r_sp_u = np.array([_a, _b])         # set ⟨A|
                elif a_or_s == "[":
                    _a, _b = self[a].l_sp_d[0, 0], self[a].l_sp_d[0, 1]  # [A| = (a, b)
                    _c, _d = rest[0, 0], rest[1, 0]             # |rest⟩ = (c, d)
                    _a = (temp_value - _b * _d) / _c                     # a = (X - b*d)/c
                    self[a].l_sp_d = np.array([_a, _b])         # set [A|

        elif (unique_extrema_of_middle != []):                  # this sets the unique element in the middle
            _bc = bc[unique_extrema_of_middle[2]]               # start by rearranging the list so that the unique item is at the beginnig
            if unique_extrema_of_middle[1] != 0:
                _bc = _bc[2 * unique_extrema_of_middle[1]] + _bc[1:2 * unique_extrema_of_middle[1]] + _bc[0] + _bc[2 * unique_extrema_of_middle[1] + 1:]
            bc[unique_extrema_of_middle[2]] = _bc
            if unique_extrema_of_middle[2] == 0:                # rewrite the string representing the invariant
                start = temp_string[0] + unicode(a) + "|"
                end = "|" + unicode(d) + temp_string[len(temp_string) - 1]
                middle = "".join(["(" + item + ")|" for item in bc])
                middle = middle[:-1]
            elif unique_extrema_of_middle[2] == len(bc) - 1:
                if temp_string[len(temp_string) - 1] == "⟩":
                    start = "⟨" + unicode(d) + "|"
                elif temp_string[len(temp_string) - 1] == "]":
                    start = "[" + unicode(d) + "|"
                if temp_string[0] == "⟨":
                    end = "|" + unicode(a) + "⟩"
                elif temp_string[0] == "[":
                    end = "|" + unicode(a) + "]"
                a, d = d, a
                bc.reverse()
                middle = "".join(["(" + item + ")|" for item in bc])
                middle = middle[:-1]
            temp_string = start + middle + end

            total = self.compute(temp_string)                   # compute the all thing: ⟨i|j+k|l]
            b = int(temp_string[4])

            part = self.compute(temp_string[:5] + temp_string[temp_string.find(")"):])
            if a_or_s == '⟨' and ")|" in temp_string:
                end_of_part = self.compute("[" + temp_string[4] + temp_string[temp_string.find(")|") + 1:])
            elif a_or_s == '⟨':
                end_of_part = self.compute("[" + temp_string[4] + temp_string[temp_string.find("|"):])
            elif a_or_s == '[' and ")|" in temp_string:
                end_of_part = self.compute("⟨" + temp_string[4] + temp_string[temp_string.find(")|") + 1:])
            elif a_or_s == '[':
                end_of_part = self.compute("⟨" + temp_string[4] + temp_string[temp_string.find("|"):])
            target = (temp_value - (total - part)) / end_of_part
            if a_or_s == '⟨':
                self.set_A2("⟨{}|{}⟩".format(a, b), target, False)
            elif a_or_s == '[':
                self.set_S2("[{}|{}]".format(a, b), target, False)

        if fix_mom is True:                                     # indentify which momenta to use to fix mom conservation.
            plist = self._complementary([lNBs] + lNBms + [lNBe])
            if len(plist) >= 2:                                 # two free particles = no problema
                self.fix_mom_cons(plist[0], plist[1])
            elif (len(plist) == 1 and                           # one free particle, but unique in head or tail
                  unique_head_or_tail != []):                   # can still use the head or tail other bracket!
                plist += [a]
                a_or_s = temp_string[0]                         # reset the angle_or_square bracket variable
                if a_or_s == "[":
                    self.fix_mom_cons(plist[0], plist[1], real_momenta=False, axis=1)
                else:
                    self.fix_mom_cons(plist[0], plist[1], real_momenta=False, axis=2)
            else:                                               # not enough particles to fix mom cons :( hopefully this is very rare/impossible
                plist += [a]
                a_or_s = temp_string[0]                         # reset the angle_or_square bracket variable
                if a_or_s == "[":
                    self.fix_mom_cons(plist[0], plist[1], real_momenta=False, axis=1)
                else:
                    self.fix_mom_cons(plist[0], plist[1], real_momenta=False, axis=2)
                if not self.momentum_conservation_check():
                    myException("Not enough particles to fix mom cons!")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def set_Sijk(self, temp_string, temp_value, fix_mom=True, mode=1):

        ijk = map(int, pSijk.findall(temp_string)[0])
        if len(ijk) == 2:
            print("Warning: You are not supposed to call set with s_ij. Call set either ⟨i|j⟩ or [i|j] instead.")
            return

        rest = self.compute(temp_string)                        # Compute the whole thing: s_ijk = ⟨i|j+k|i]+⟨j|k⟩[k|j]
        if mode == 1:                                           # Choose to set ⟨i| ...
            str1 = "⟨{}|(".format(ijk[0])
        elif mode == 2:                                         # ... or |i]
            str1 = "[{}|(".format(ijk[0])
        for i in range(1, len(ijk)):
            str1 += "{}+".format(ijk[i])
        str1 = str1[:-1]                                        # remove the rogue "+" from the string
        if mode == 1:                                           # and close the string
            str1 += ")|{}]".format(ijk[0])
        elif mode == 2:
            str1 += ")|{}⟩".format(ijk[0])
        rest = rest - self.compute(str1)                        # str1 = ⟨i|j+k|i], hence rest -> ⟨j|k|j]     (Note: this example is for
        self.set_inner(str1, temp_value - rest, fix_mom)        # set ⟨i|j+k|i] to (X - ⟨j|k|j])               len(ijk)==3, but it works for any ijk)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def set_Dijk(self, temp_string, temp_value, fix_mom=True, mode=1):                  # Sets Δ_ijk  --- Changes: last two [j] moment, Don't touch [j]'s or [i]'s

        ijk = map(int, pDijk.findall(temp_string)[0])
        NonOverlappingLists = self.ijk_to_3NonOverlappingLists(ijk)
        run_me_again = False

        if len(NonOverlappingLists[0]) == 2 and len(NonOverlappingLists[1]) == 2:    # Hard solution for |i⟩ (mode=1) or |i] (mode=2)
            a, b = self[NonOverlappingLists[0][0]].r_sp_d[0, 0], self[NonOverlappingLists[0][0]].r_sp_d[1, 0]
            c, d = self[NonOverlappingLists[0][0]].l_sp_d[0, 0], self[NonOverlappingLists[0][0]].l_sp_d[0, 1]
            e, f = self[NonOverlappingLists[0][1]].r_sp_d[0, 0], self[NonOverlappingLists[0][1]].r_sp_d[1, 0]
            g, h = self[NonOverlappingLists[0][1]].l_sp_d[0, 0], self[NonOverlappingLists[0][1]].l_sp_d[0, 1]
            i, j = self[NonOverlappingLists[1][0]].r_sp_d[0, 0], self[NonOverlappingLists[1][0]].r_sp_d[1, 0]
            k, l = self[NonOverlappingLists[1][0]].l_sp_d[0, 0], self[NonOverlappingLists[1][0]].l_sp_d[0, 1]
            m, n = self[NonOverlappingLists[1][1]].r_sp_d[0, 0], self[NonOverlappingLists[1][1]].r_sp_d[1, 0]
            o, p = self[NonOverlappingLists[1][1]].l_sp_d[0, 0], self[NonOverlappingLists[1][1]].l_sp_d[0, 1]
            X = 4 * temp_value
            if mode == 2:
                a, b, c, d = c, d, a, b
                e, f, g, h = g, h, e, f
                i, j, k, l = k, l, i, j
                m, n, o, p = o, p, m, n
            elif mode == 3:
                a, b, c, d, e, f, g, h = e, f, g, h, a, b, c, d
                i, j, k, l, m, n, o, p = m, n, o, p, i, j, k, l
            elif mode == 4:               # combine mode 2 and mode 3
                a, b, c, d = c, d, a, b
                e, f, g, h = g, h, e, f
                i, j, k, l = k, l, i, j
                m, n, o, p = o, p, m, n
                a, b, c, d, e, f, g, h = e, f, g, h, a, b, c, d
                i, j, k, l, m, n, o, p = m, n, o, p, i, j, k, l
            QB = (-2 * b * d * d * i * j * k * k - 2 * d * f * h * i * j * k * k +
                  2 * d * e * h * j * j * k * k + 4 * b * c * d * i * j * k * l + 2 * d * f * g * i * j * k * l +
                  2 * c * f * h * i * j * k * l - 2 * d * e * g * j * j * k * l - 2 * c * e * h * j * j * k * l -
                  2 * b * c * c * i * j * l * l - 2 * c * f * g * i * j * l * l + 2 * c * e * g * j * j * l * l -
                  2 * b * d * d * j * k * m * o - 2 * d * f * h * j * k * m * o + 2 * b * c * d * j * l * m * o +
                  4 * d * f * g * j * l * m * o - 2 * c * f * h * j * l * m * o - 2 * b * d * d * i * k * n * o -
                  2 * d * f * h * i * k * n * o + 4 * d * e * h * j * k * n * o + 2 * b * c * d * i * l * n * o -
                  2 * d * f * g * i * l * n * o + 4 * c * f * h * i * l * n * o - 2 * d * e * g * j * l * n * o -
                  2 * c * e * h * j * l * n * o - 2 * b * d * d * m * n * o * o - 2 * d * f * h * m * n * o * o +
                  2 * d * e * h * n * n * o * o + 2 * b * c * d * j * k * m * p - 2 * d * f * g * j * k * m * p +
                  4 * c * f * h * j * k * m * p - 2 * b * c * c * j * l * m * p - 2 * c * f * g * j * l * m * p +
                  2 * b * c * d * i * k * n * p + 4 * d * f * g * i * k * n * p - 2 * c * f * h * i * k * n * p -
                  2 * d * e * g * j * k * n * p - 2 * c * e * h * j * k * n * p - 2 * b * c * c * i * l * n * p -
                  2 * c * f * g * i * l * n * p + 4 * c * e * g * j * l * n * p + 4 * b * c * d * m * n * o * p +
                  2 * d * f * g * m * n * o * p + 2 * c * f * h * m * n * o * p - 2 * d * e * g * n * n * o * p -
                  2 * c * e * h * n * n * o * p - 2 * b * c * c * m * n * p * p - 2 * c * f * g * m * n * p * p +
                  2 * c * e * g * n * n * p * p)
            QA = (d * d * j * j * k * k - 2 * c * d * j * j * k * l + c * c * j * j * l * l + 2 * d * d * j * k * n * o -
                  2 * c * d * j * l * n * o + d * d * n * n * o * o - 2 * c * d * j * k * n * p +
                  2 * c * c * j * l * n * p - 2 * c * d * n * n * o * p + c * c * n * n * p * p)
            QC = (b * b * d * d * i * i * k * k + 2 * b * d * f * h * i * i * k * k + f * f * h * h * i * i * k * k -
                  2 * b * d * e * h * i * j * k * k - 2 * e * f * h * h * i * j * k * k + e * e * h * h * j * j * k * k -
                  2 * b * b * c * d * i * i * k * l - 2 * b * d * f * g * i * i * k * l - 2 * b * c * f * h * i * i * k * l -
                  2 * f * f * g * h * i * i * k * l + 2 * b * d * e * g * i * j * k * l + 2 * b * c * e * h * i * j * k * l +
                  4 * e * f * g * h * i * j * k * l - 2 * e * e * g * h * j * j * k * l + b * b * c * c * i * i * l * l +
                  2 * b * c * f * g * i * i * l * l + f * f * g * g * i * i * l * l - 2 * b * c * e * g * i * j * l * l -
                  2 * e * f * g * g * i * j * l * l + e * e * g * g * j * j * l * l + 2 * b * b * d * d * i * k * m * o +
                  4 * b * d * f * h * i * k * m * o + 2 * f * f * h * h * i * k * m * o - 2 * b * d * e * h * j * k * m * o -
                  2 * e * f * h * h * j * k * m * o - 2 * b * b * c * d * i * l * m * o - 2 * b * d * f * g * i * l * m * o -
                  2 * b * c * f * h * i * l * m * o - 2 * f * f * g * h * i * l * m * o - 2 * b * d * e * g * j * l * m * o +
                  4 * b * c * e * h * j * l * m * o + 2 * e * f * g * h * j * l * m * o - 2 * b * d * e * h * i * k * n * o -
                  2 * e * f * h * h * i * k * n * o + 2 * e * e * h * h * j * k * n * o + 4 * b * d * e * g * i * l * n * o -
                  2 * b * c * e * h * i * l * n * o + 2 * e * f * g * h * i * l * n * o - 2 * e * e * g * h * j * l * n * o +
                  b * b * d * d * m * m * o * o + 2 * b * d * f * h * m * m * o * o + f * f * h * h * m * m * o * o -
                  2 * b * d * e * h * m * n * o * o - 2 * e * f * h * h * m * n * o * o + e * e * h * h * n * n * o * o -
                  2 * b * b * c * d * i * k * m * p - 2 * b * d * f * g * i * k * m * p - 2 * b * c * f * h * i * k * m * p -
                  2 * f * f * g * h * i * k * m * p + 4 * b * d * e * g * j * k * m * p - 2 * b * c * e * h * j * k * m * p +
                  2 * e * f * g * h * j * k * m * p + 2 * b * b * c * c * i * l * m * p + 4 * b * c * f * g * i * l * m * p +
                  2 * f * f * g * g * i * l * m * p - 2 * b * c * e * g * j * l * m * p - 2 * e * f * g * g * j * l * m * p -
                  2 * b * d * e * g * i * k * n * p + 4 * b * c * e * h * i * k * n * p + 2 * e * f * g * h * i * k * n * p -
                  2 * e * e * g * h * j * k * n * p - 2 * b * c * e * g * i * l * n * p - 2 * e * f * g * g * i * l * n * p +
                  2 * e * e * g * g * j * l * n * p - 2 * b * b * c * d * m * m * o * p - 2 * b * d * f * g * m * m * o * p -
                  2 * b * c * f * h * m * m * o * p - 2 * f * f * g * h * m * m * o * p + 2 * b * d * e * g * m * n * o * p +
                  2 * b * c * e * h * m * n * o * p + 4 * e * f * g * h * m * n * o * p - 2 * e * e * g * h * n * n * o * p +
                  b * b * c * c * m * m * p * p + 2 * b * c * f * g * m * m * p * p + f * f * g * g * m * m * p * p -
                  2 * b * c * e * g * m * n * p * p - 2 * e * f * g * g * m * n * p * p + e * e * g * g * n * n * p * p - X)
            a = (-QB - gmpTools.csqrt(QB**2 - 4 * QA * QC)) / (2 * QA)
            if mode == 1:
                self[NonOverlappingLists[0][0]].r_sp_d = np.array([a, b])
            elif mode == 2:
                self[NonOverlappingLists[0][0]].l_sp_d = np.array([a, b])
            elif mode == 3:
                self[NonOverlappingLists[0][1]].r_sp_d = np.array([a, b])
            elif mode == 4:
                self[NonOverlappingLists[0][1]].l_sp_d = np.array([a, b])

        else:
            run_me_again = True
            ijk = map(int, pDijk.findall(temp_string)[0])       # First work in phase space with 3 massive momenta K1, K2, K3
            temp_oParticles = self.ijk_to_3Ks(ijk)              # K1 = Sum_Pi's, K2 = Sum_Pj's, K3 = Sum_Pk's
            K1s = temp_oParticles.ldot(1, 1)
            K10 = temp_oParticles[1].four_mom[0]
            K1V = np.array([temp_oParticles[1].four_mom[1], temp_oParticles[1].four_mom[2], temp_oParticles[1].four_mom[3]])
            K2V = np.array([temp_oParticles[2].four_mom[1], temp_oParticles[2].four_mom[2], temp_oParticles[2].four_mom[3]])
            a = K10**2 - K1s
            b = -2 * K10 * (np.dot(K1V, K2V))
            c = np.dot(K1V, K2V)**2 + K1s * np.dot(K2V, K2V) - temp_value
            K20 = (-b + gmpTools.csqrt(b**2 - 4 * a * c)) / (2 * a)                   # Solve quadratic equation and fix K_2^0
            temp_oParticles[2].four_mom = np.array([K20, K2V[0], K2V[1], K2V[2]])

            NonOLLists = self.ijk_to_3NonOverlappingLists(ijk, 2)   # Convert solution from K_2 to Pj's by decaying the last two
            from particles import Particles
            FakeoParticles2 = Particles(len(NonOLLists[1]) + 1)
            FakeoParticles2[1] = temp_oParticles[2]
            for i, particle in enumerate(NonOLLists[1]):
                FakeoParticles2[i + 2] = particle
            FakeoParticles2.fix_mom_cons(len(FakeoParticles2) - 1, len(FakeoParticles2), real_momenta=False, axis=mode)
            for i, particle in enumerate(FakeoParticles2):
                if i == 0:
                    continue
                else:
                    index = (ijk[1] - 1 + i) % len(self)
                    if index == 0:
                        index = len(self)
                    self[index] = particle

        if fix_mom is True:                                     # Fix_mom_cons by changing last two Pk's
            NonOverlappingLists = self.ijk_to_3NonOverlappingLists(ijk)
            ultima = NonOverlappingLists[2][len(NonOverlappingLists[2]) - 1]
            penultima = NonOverlappingLists[2][len(NonOverlappingLists[2]) - 2]
            if ultima == 0:
                ultima = len(self)
            if penultima == 0:
                penultima = len(self)
            self.fix_mom_cons(penultima, ultima)
        if run_me_again is True:
            return "run me again"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def set_Oijk(self, temp_string, temp_value, fix_mom=True):

        ijk = map(int, pOijk.findall(temp_string)[0])
        nol = self.ijk_to_3NonOverlappingLists(ijk)
        A, B = nol[2][0], nol[2][1]
        C, D = nol[0][0], nol[0][1]
        E, F = nol[1][0], nol[1][1]

        a, b = self[A].r_sp_d[0, 0], self[A].r_sp_d[1, 0]
        c, d = self[A].l_sp_d[0, 0], self[A].l_sp_d[0, 1]
        e, f = self[B].r_sp_d[0, 0], self[B].r_sp_d[1, 0]
        g, h = self[B].l_sp_d[0, 0], self[B].l_sp_d[0, 1]
        i, j = self[C].r_sp_d[0, 0], self[C].r_sp_d[1, 0]
        k, l = self[C].l_sp_d[0, 0], self[C].l_sp_d[0, 1]
        m, n = self[D].r_sp_d[0, 0], self[D].r_sp_d[1, 0]
        o, p = self[D].l_sp_d[0, 0], self[D].l_sp_d[0, 1]

        X = temp_value

        QB = (b * d**2 * f * g * i * k - b * c * d * f * h * i * k + d * f**2 * g * h * i * k - c * f**2 * h**2 * i * k +
              b * d**2 * e * g * j * k - b * c * d * e * h * j * k - d * e * f * g * h * j * k + c * e * f * h**2 * j * k +
              2 * b * d**2 * i * j * k**2 + 2 * d * f * h * i * j * k**2 - 2 * d * e * h * j**2 * k**2 - b * c * d * f * g * i * l -
              d * f**2 * g**2 * i * l + b * c**2 * f * h * i * l + c * f**2 * g * h * i * l - b * c * d * e * g * j * l +
              d * e * f * g**2 * j * l + b * c**2 * e * h * j * l - c * e * f * g * h * j * l - 4 * b * c * d * i * j * k * l -
              2 * d * f * g * i * j * k * l - 2 * c * f * h * i * j * k * l + 2 * d * e * g * j**2 * k * l +
              2 * c * e * h * j**2 * k * l + 2 * b * c**2 * i * j * l**2 + 2 * c * f * g * i * j * l**2 -
              2 * c * e * g * j**2 * l**2 - b * d**2 * f * g * m * o + b * c * d * f * h * m * o - d * f**2 * g * h * m * o +
              c * f**2 * h**2 * m * o + b * d**2 * j * k * m * o + d * f * h * j * k * m * o - b * c * d * j * l * m * o -
              2 * d * f * g * j * l * m * o + c * f * h * j * l * m * o - b * d**2 * e * g * n * o + b * c * d * e * h * n * o +
              d * e * f * g * h * n * o - c * e * f * h**2 * n * o + b * d**2 * i * k * n * o + d * f * h * i * k * n * o -
              2 * d * e * h * j * k * n * o - b * c * d * i * l * n * o + d * f * g * i * l * n * o - 2 * c * f * h * i * l * n * o +
              d * e * g * j * l * n * o + c * e * h * j * l * n * o + b * c * d * f * g * m * p + d * f**2 * g**2 * m * p -
              b * c**2 * f * h * m * p - c * f**2 * g * h * m * p - b * c * d * j * k * m * p + d * f * g * j * k * m * p -
              2 * c * f * h * j * k * m * p + b * c**2 * j * l * m * p + c * f * g * j * l * m * p + b * c * d * e * g * n * p -
              d * e * f * g**2 * n * p - b * c**2 * e * h * n * p + c * e * f * g * h * n * p - b * c * d * i * k * n * p -
              2 * d * f * g * i * k * n * p + c * f * h * i * k * n * p + d * e * g * j * k * n * p + c * e * h * j * k * n * p +
              b * c**2 * i * l * n * p + c * f * g * i * l * n * p - 2 * c * e * g * j * l * n * p)

        QA = (-d**2 * f * g * j * k + c * d * f * h * j * k - d**2 * j**2 * k**2 + c * d * f * g * j * l -
              c**2 * f * h * j * l + 2 * c * d * j**2 * k * l - c**2 * j**2 * l**2 + d**2 * f * g * n * o -
              c * d * f * h * n * o - d**2 * j * k * n * o + c * d * j * l * n * o - c * d * f * g * n * p +
              c**2 * f * h * n * p + c * d * j * k * n * p - c**2 * j * l * n * p)

        QC = (-b**2 * d**2 * e * g * i * k + b**2 * c * d * e * h * i * k - b * d * e * f * g * h * i * k +
              b * c * e * f * h**2 * i * k + b * d * e**2 * g * h * j * k - b * c * e**2 * h**2 * j * k -
              b**2 * d**2 * i**2 * k**2 - 2 * b * d * f * h * i**2 * k**2 - f**2 * h**2 * i**2 * k**2 +
              2 * b * d * e * h * i * j * k**2 + 2 * e * f * h**2 * i * j * k**2 - e**2 * h**2 * j**2 * k**2 +
              b**2 * c * d * e * g * i * l + b * d * e * f * g**2 * i * l - b**2 * c**2 * e * h * i * l -
              b * c * e * f * g * h * i * l - b * d * e**2 * g**2 * j * l + b * c * e**2 * g * h * j * l +
              2 * b**2 * c * d * i**2 * k * l + 2 * b * d * f * g * i**2 * k * l + 2 * b * c * f * h * i**2 * k * l +
              2 * f**2 * g * h * i**2 * k * l - 2 * b * d * e * g * i * j * k * l - 2 * b * c * e * h * i * j * k * l -
              4 * e * f * g * h * i * j * k * l + 2 * e**2 * g * h * j**2 * k * l - b**2 * c**2 * i**2 * l**2 -
              2 * b * c * f * g * i**2 * l**2 - f**2 * g**2 * i**2 * l**2 + 2 * b * c * e * g * i * j * l**2 +
              2 * e * f * g**2 * i * j * l**2 - e**2 * g**2 * j**2 * l**2 + b**2 * d**2 * e * g * m * o -
              b**2 * c * d * e * h * m * o + b * d * e * f * g * h * m * o - b * c * e * f * h**2 * m * o -
              b**2 * d**2 * i * k * m * o - 2 * b * d * f * h * i * k * m * o - f**2 * h**2 * i * k * m * o +
              b * d * e * h * j * k * m * o + e * f * h**2 * j * k * m * o + b**2 * c * d * i * l * m * o +
              b * d * f * g * i * l * m * o + b * c * f * h * i * l * m * o + f**2 * g * h * i * l * m * o +
              b * d * e * g * j * l * m * o - 2 * b * c * e * h * j * l * m * o - e * f * g * h * j * l * m * o -
              b * d * e**2 * g * h * n * o + b * c * e**2 * h**2 * n * o + b * d * e * h * i * k * n * o +
              e * f * h**2 * i * k * n * o - e**2 * h**2 * j * k * n * o - 2 * b * d * e * g * i * l * n * o +
              b * c * e * h * i * l * n * o - e * f * g * h * i * l * n * o + e**2 * g * h * j * l * n * o -
              b**2 * c * d * e * g * m * p - b * d * e * f * g**2 * m * p + b**2 * c**2 * e * h * m * p +
              b * c * e * f * g * h * m * p + b**2 * c * d * i * k * m * p + b * d * f * g * i * k * m * p +
              b * c * f * h * i * k * m * p + f**2 * g * h * i * k * m * p - 2 * b * d * e * g * j * k * m * p +
              b * c * e * h * j * k * m * p - e * f * g * h * j * k * m * p - b**2 * c**2 * i * l * m * p -
              2 * b * c * f * g * i * l * m * p - f**2 * g**2 * i * l * m * p + b * c * e * g * j * l * m * p +
              e * f * g**2 * j * l * m * p + b * d * e**2 * g**2 * n * p - b * c * e**2 * g * h * n * p +
              b * d * e * g * i * k * n * p - 2 * b * c * e * h * i * k * n * p - e * f * g * h * i * k * n * p +
              e**2 * g * h * j * k * n * p + b * c * e * g * i * l * n * p + e * f * g**2 * i * l * n * p -
              e**2 * g**2 * j * l * n * p - X)

        a = (-QB - gmpTools.csqrt(QB**2 - 4 * QA * QC)) / (2 * QA)

        self[A].r_sp_d = np.array([a, b])
        self.fix_mom_cons(E, F)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def set_Pijk(self, temp_string, temp_value, fix_mom=True):

        ijk = map(int, pPijk.findall(temp_string)[0])
        nol = self.ijk_to_3NonOverlappingLists(ijk)
        A, B = nol[2][0], nol[2][1]
        C, D = nol[0][0], nol[0][1]
        E, F = nol[1][0], nol[1][1]

        a, b = self[A].r_sp_d[0, 0], self[A].r_sp_d[1, 0]
        c, d = self[A].l_sp_d[0, 0], self[A].l_sp_d[0, 1]
        e, f = self[B].r_sp_d[0, 0], self[B].r_sp_d[1, 0]
        g, h = self[B].l_sp_d[0, 0], self[B].l_sp_d[0, 1]
        i, j = self[C].r_sp_d[0, 0], self[C].r_sp_d[1, 0]
        k, l = self[C].l_sp_d[0, 0], self[C].l_sp_d[0, 1]
        m, n = self[D].r_sp_d[0, 0], self[D].r_sp_d[1, 0]
        o, p = self[D].l_sp_d[0, 0], self[D].l_sp_d[0, 1]

        X = temp_value

        a = ((b * d * i * k + f * h * i * k - e * h * j * k - b * c * i * l - f * g * i * l + e * g * j * l +
              n * e * h * o - b * d * m * o - f * h * m * o - n * e * g * p + b * c * m * p + f * g * m * p - X) /
             (d * j * k - c * j * l - n * d * o + n * c * p))

        self[A].r_sp_d = np.array([a, b])
        self.fix_mom_cons(E, F)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
