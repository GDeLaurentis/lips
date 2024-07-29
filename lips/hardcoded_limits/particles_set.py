# -*- coding: utf-8 -*-

#   ___          _   _    _          ___      _
#  | _ \__ _ _ _| |_(_)__| |___ ___ / __| ___| |_
#  |  _/ _` | '_|  _| / _| / -_|_-<_\__ \/ -_)  _|
#  |_| \__,_|_|  \__|_\__|_\___/__(_)___/\___|\__|

# Author: Giuseppe

import numpy
import re

from ..tools import flatten, pSijk, pDijk, pOijk, pPijk, pA2, pS2, pNB, ptr5, p5Bdiff, myException


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_Set:

    # PRIVATE METHODS

    def _set(self, temp_string, temp_value, fix_mom=True, mode=1):
        """Constructs a singular phase space point."""
        self._set_inner(temp_string, temp_value, fix_mom, mode)
        abs_diff = abs(self.compute(temp_string) - temp_value)
        if fix_mom is True:
            mom_cons, on_shell = self.momentum_conservation_check(), self.onshell_relation_check()
            if mom_cons is False:
                raise myException("Setting: {} to {}. Momentum conservation is not satisfied: ", temp_string, temp_value, max(map(abs, flatten(self.total_mom))))
            elif on_shell is False:
                raise myException("Setting: {} to {}. On shellness is not satisfied: ", temp_string, temp_value, max(map(abs, flatten(self.masses))))
        if not abs_diff <= self.field.tollerance:
            raise myException("Failed to set {} to {}. Instead got {}. Absolute difference {}.".format(
                temp_string, temp_value, self.compute(temp_string), abs_diff))

    def _set_inner(self, temp_string, temp_value, fix_mom=True, mode=1):

        self.check_consistency(temp_string)                          # Check consistency of string - !Warning! Exceptions are disabled

        if pA2.findall(temp_string) != []:                           # Sets ⟨A|B⟩  --- Changes: |B⟩, Don't touch: ⟨A|

            self._set_A2(temp_string, temp_value, fix_mom)

        elif pS2.findall(temp_string) != []:                         # Sets [A|B]  --- Changes: |B], Don't touch: [A|

            self._set_S2(temp_string, temp_value, fix_mom)

        elif pNB.findall(temp_string) != []:                         # Sets ⟨a|(b+c)|...|d]  --- Changes: ⟨a| (mode=1), |b⟩ (mode=2)

            self._set_NB(temp_string, temp_value, fix_mom, mode)

        elif pSijk.findall(temp_string) != []:                       # Sets S_ijk  --- Changes: ⟨i| (mode=1) or |i] (mode=2), Don't touch ijk...

            self._set_Sijk(temp_string, temp_value, fix_mom, mode)

        elif pDijk.findall(temp_string) != []:                       # Sets Δ_ijk  --- Changes: last two [j] moment, Don't touch [j]'s or [i]'s

            self._set_Dijk(temp_string, temp_value, fix_mom, mode)

        elif pOijk.findall(temp_string) != []:

            self._set_Oijk(temp_string, temp_value, fix_mom)

        elif pPijk.findall(temp_string) != []:

            self._set_Pijk(temp_string, temp_value, fix_mom)

        elif ptr5.findall(temp_string) != []:

            self._set_tr5(temp_string, temp_value, fix_mom)

        elif p5Bdiff.findall(temp_string) != []:

            self._set_p5Bdiff(temp_string, temp_value, fix_mom)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_A2(self, temp_string, temp_value, fix_mom=True):        # ⟨A|B⟩ = (a, b).(c, d) = ac+bd = X ----> c = (X - bd)/a

        A, B = map(int, pA2.findall(temp_string)[0])
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

        A, B = map(int, pS2.findall(temp_string)[0])
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

    def _set_NB(self, temp_string, temp_value, fix_mom=True, mode=1):
        if "-" in temp_string:                                  # a minus sign would mess up inversion
            myException("Detected minus in string. Not implemented.")

        lNB, lNBs, lNBms, lNBe = self._get_lNB(temp_string)
        plist = self._complementary([lNB])

        if len(plist) < 2 and fix_mom is True:                  # if necessary look for alternative way to write it which allows to fix mom cons
            for i, iNBm in enumerate(lNBms):                    # try to flip the i^th bracket and see if len(plist) >= 2
                _lNBms = [entry for entry in lNBms]
                if i == 0:                                      # this is close to the head, what I call extremum of middle
                    alt = self._complementary(iNBm + [lNBs])
                elif i == len(lNBms) - 1:                       # this is close to the tail, what I call extremum of middle
                    alt = self._complementary(iNBm + [lNBe])
                else:                                           # this is not close to either head or tail
                    alt = self._complementary(iNBm)
                _lNBms[i] = alt
                plist = self._complementary([lNBs] + _lNBms + [lNBe])
                if len(plist) >= 2:                             # if this fixes it then reconstruct bc and temp_string
                    lNBms = _lNBms
                    start = temp_string[0] + str(lNBs) + "|"
                    end = "|" + str(lNBe) + temp_string[len(temp_string) - 1]
                    middle = "".join(string + "|" for string in ["(" + "".join(str(entry) + "+" for entry in item)[:-1] + ")" for item in lNBms])
                    middle = middle[:-1]
                    temp_string = start + middle + end
                    temp_value = -temp_value                    # flip using mom cons causes a minus sign
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
            if i != 1 and i != len(temp_list) - 2:              # if it is head, tail or middle of middle skip it
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
        bc = ["".join(str(entry) + "+" for entry in item)[:-1] for item in lNBms]
        a, d = lNBs, lNBe

        # print(temp_string, unique_head_or_tail)                # debugging tool

        if (((unique_head_or_tail != [] or                      # this sets the head (or tail) leaving middle unchanged
              (lNBs == lNBe and lNBs not in flatten(lNBms) and
               lNBe not in flatten(lNBms) and len(lNBms) % 2 == 0)) and mode == 1) or
           (unique_extrema_of_middle == [])):
            if unique_head_or_tail == 1:                        # if the unique one is in the tail, flip it so that it is in the head
                temp_string = temp_string[::-1]
                temp_string = temp_string.replace("(", "X").replace(")", "(").replace("X", ")")
                temp_string = temp_string.replace("⟨", "X").replace("⟩", "⟨").replace("X", "⟩")
                temp_string = temp_string.replace("[", "X").replace("]", "[").replace("X", "]")
                self._set_inner(temp_string, temp_value, fix_mom, mode)
                return
            rest = numpy.array([[1, 0], [0, 1]])
            if temp_string[0] == "⟨":
                rest = ["(" + re.sub(r'(\d)', r'self[\1].r2_sp_b', entry) + ")" if i % 2 == 0 else
                        "(" + re.sub(r'(\d)', r'self[\1].r2_sp', entry) + ")" for i, entry in enumerate(bc)]
                rest = ".dot(".join(rest) + ")" * (len(rest) - 1)
            elif a_or_s == "[":
                rest = ["(" + re.sub(r'(\d)', r'self[\1].r2_sp', entry) + ")" if i % 2 == 0 else
                        "(" + re.sub(r'(\d)', r'self[\1].r2_sp_b', entry) + ")" for i, entry in enumerate(bc)]
                rest = ".dot(".join(rest) + ")" * (len(rest) - 1)
            rest = eval(rest)
            if a == d and len(lNBms) % 2 == 0:
                K11, K12, K21, K22 = rest[0, 0], rest[0, 1], rest[1, 0], rest[1, 1]
                if temp_string[0] == "⟨":
                    B = self[a].r_sp_d[1, 0]
                    A = (B * K11 - B * K22 + self.field.sqrt((B * K11 - B * K22) * (B * K11 - B * K22) + 4 * K21 * (B * B * K12 - temp_value))) / (2 * K21)
                    self[a].r_sp_d = numpy.array([A, B])
                else:
                    B = self[a].l_sp_d[0, 1]
                    A = (B * K11 - B * K22 + self.field.sqrt((B * K11 - B * K22) * (B * K11 - B * K22) + 4 * K12 * (B * B * K21 - temp_value))) / (2 * K12)
                    self[a].l_sp_d = numpy.array([A, B])
            else:
                if temp_string[-1] == "⟩":
                    rest = numpy.dot(rest, self[d].r_sp_d)
                else:
                    rest = numpy.dot(rest, self[d].l_sp_u)
                if temp_string[0] == "⟨":
                    _a, _b = self[a].r_sp_u[0, 0], self[a].r_sp_u[0, 1]  # ⟨A| = (a, b)
                    _c, _d = rest[0, 0], rest[1, 0]             # |rest⟩ = (c, d)
                    _a = (temp_value - _b * _d) / _c            # a = (X - b*d)/c
                    self[a].r_sp_u = numpy.array([_a, _b])      # set ⟨A|
                else:
                    _a, _b = self[a].l_sp_d[0, 0], self[a].l_sp_d[0, 1]  # [A| = (a, b)
                    _c, _d = rest[0, 0], rest[1, 0]             # |rest⟩ = (c, d)
                    _a = (temp_value - _b * _d) / _c            # a = (X - b*d)/c
                    self[a].l_sp_d = numpy.array([_a, _b])      # set [A|

        elif (unique_extrema_of_middle != []):                  # this sets the unique element in the middle
            _bc = bc[unique_extrema_of_middle[2]]               # start by rearranging the list so that the unique item is at the beginnig
            if unique_extrema_of_middle[1] != 0:
                _bc = _bc[2 * unique_extrema_of_middle[1]] + _bc[1:2 * unique_extrema_of_middle[1]] + _bc[0] + _bc[2 * unique_extrema_of_middle[1] + 1:]
            bc[unique_extrema_of_middle[2]] = _bc
            if unique_extrema_of_middle[2] == 0:                # rewrite the string representing the invariant
                start = temp_string[0] + str(a) + "|"
                end = "|" + str(d) + temp_string[len(temp_string) - 1]
                middle = "".join(["(" + item + ")|" for item in bc])
                middle = middle[:-1]
            elif unique_extrema_of_middle[2] == len(bc) - 1:
                if temp_string[len(temp_string) - 1] == "⟩":
                    start = "⟨" + str(d) + "|"
                elif temp_string[len(temp_string) - 1] == "]":
                    start = "[" + str(d) + "|"
                if temp_string[0] == "⟨":
                    end = "|" + str(a) + "⟩"
                elif temp_string[0] == "[":
                    end = "|" + str(a) + "]"
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
                self._set_A2("⟨{}|{}⟩".format(a, b), target, False)
            elif a_or_s == '[':
                self._set_S2("[{}|{}]".format(a, b), target, False)

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

    def _set_Sijk(self, temp_string, temp_value, fix_mom=True, mode=1):

        ijk = list(map(int, pSijk.findall(temp_string)[0]))
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
        self._set_inner(str1, temp_value - rest, fix_mom)       # set ⟨i|j+k|i] to (X - ⟨j|k|j])               len(ijk)==3, but it works for any ijk)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_Dijk(self, temp_string, temp_value, fix_mom=True, mode=1):  # Set Dijk, change angle bracket (odd mode) or square bracket (even mode)

        match_list = pDijk.findall(temp_string)[0]
        if match_list[0] == '':
            NonOverlappingLists = [list(map(int, corner)) for corner in match_list[1:]]
        else:
            NonOverlappingLists = self.ijk_to_3NonOverlappingLists(list(map(int, match_list[0])))

        to_be_changed = NonOverlappingLists[0].pop((mode - 1) // 2)

        o0 = self[to_be_changed]
        oQs = self.cluster(NonOverlappingLists)

        X = temp_value

        if mode % 2 == 1:
            [[a, b]] = o0.r_sp_u
            [[alpha], [beta]] = numpy.dot(oQs[2].r2_sp_b, o0.l_sp_u)
            [[gamma], [delta]] = numpy.dot(oQs[1].r2_sp_b, o0.l_sp_u)
        elif mode % 2 == 0:
            [[a], [b]] = o0.l_sp_u
            [[alpha, beta]] = numpy.dot(o0.r_sp_u, oQs[2].r2_sp_b)
            [[gamma, delta]] = numpy.dot(o0.r_sp_u, oQs[1].r2_sp_b)

        QA = alpha ** 2 / 4
        QB = alpha * b * beta / 2 + alpha * oQs.ldot(1, 2) - gamma * oQs.ldot(2, 2)
        QC = b ** 2 * beta ** 2 / 4 + b * beta * oQs.ldot(1, 2) - b * delta * oQs.ldot(2, 2) + oQs.ldot(1, 2) ** 2 - oQs.ldot(1, 1) * oQs.ldot(2, 2) - X

        a = (-QB + self.field.sqrt(QB**2 - 4 * QA * QC)) / (2 * QA)

        if mode % 2 == 1:
            self[to_be_changed].r_sp_u = numpy.array([[a, b]])
        elif mode % 2 == 0:
            self[to_be_changed].l_sp_u = numpy.array([[a], [b]])

        if fix_mom is True:  # Fix_mom_cons by changing last two Pk's
            self.fix_mom_cons(NonOverlappingLists[2][-1], NonOverlappingLists[2][-2])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_Oijk(self, temp_string, temp_value, fix_mom=True):

        ijk = list(map(int, pOijk.findall(temp_string)[0]))
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

        a = (-QB - self.field.sqrt(QB**2 - 4 * QA * QC)) / (2 * QA)

        self[A].r_sp_d = numpy.array([a, b])
        self.fix_mom_cons(E, F)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_Pijk(self, temp_string, temp_value, fix_mom=True):

        ijk = list(map(int, pPijk.findall(temp_string)[0]))
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

        self[A].r_sp_d = numpy.array([a, b])
        self.fix_mom_cons(E, F)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_tr5(self, temp_string, temp_value, fix_mom=True):

        abcd = ptr5.findall(temp_string)[0][0 if "_" in temp_string else 1]
        a, b, c, d = abcd.split("|") if "|" in abcd else abcd
        for _ in range(4):
            if len(a) == 1:
                break
            a, b, c, d = b, c, d, a
        else:
            raise NotImplementedError("tr5 set implementation requires at least 1 massless particle.")

        plist = list(map(int, self._complementary(flatten(re.findall(r"\d+", x) for x in (a, b, c, d)))))  # free momenta

        if len(self) < 5:  # 4-point or less
            raise myException("Particles._set_tr5 called with less than 5 particles, but tr5 is identically zero below 5-point.")

        elif len(self) == 5:  # 5-point
            """Hardcoded solution to the lexicographic Groebner basis."""
            a1, b1 = self[1].r_sp_d.flatten().tolist()
            c1, d1 = self[1].l_sp_d.flatten().tolist()
            a2, b2 = self[2].r_sp_d.flatten().tolist()
            c2, d2 = self[2].l_sp_d.flatten().tolist()
            a3, b3 = self[3].r_sp_d.flatten().tolist()
            c3, d3 = self[3].l_sp_d.flatten().tolist()
            a4, b4 = self[4].r_sp_d.flatten().tolist()
            c4, d4 = self[4].l_sp_d.flatten().tolist()
            a5, b5 = self[5].r_sp_d.flatten().tolist()
            c5, d5 = self[5].l_sp_d.flatten().tolist()

            x = 4 * temp_value

            b1 = (a2**2*b3**2*b4*c1*c2*c3*d2*d3*d4 - a2**2*b3**2*b4*c1*c2*c4*d2*d3**2 - a2**2*b3**2*b4*c1*c3**2*d2**2*d4 + a2**2*b3**2*b4*c1*c3*c4*d2**2*d3 - a2**2*b3**2*b4*c2**2*c3*d1*d3*d4 + a2**2*b3**2*b4*c2**2*c4*d1*d3**2 + a2**2*b3**2*b4*c2*c3**2*d1*d2*d4 - a2**2*b3**2*b4*c2*c3*c4*d1*d2*d3 + a2**2*b3*b4**2*c1*c2*c3*d2*d4**2 - a2**2*b3*b4**2*c1*c2*c4*d2*d3*d4 - a2**2*b3*b4**2*c1*c3*c4*d2**2*d4 + a2**2*b3*b4**2*c1*c4**2*d2**2*d3 - a2**2*b3*b4**2*c2**2*c3*d1*d4**2 + a2**2*b3*b4**2*c2**2*c4*d1*d3*d4 + a2**2*b3*b4**2*c2*c3*c4*d1*d2*d4 - a2**2*b3*b4**2*c2*c4**2*d1*d2*d3 - a2*a3*b2*b3*b4*c1*c2**2*d3**2*d4 + 2*a2*a3*b2*b3*b4*c1*c2*c4*d2*d3**2 + a2*a3*b2*b3*b4*c1*c3**2*d2**2*d4 - 2*a2*a3*b2*b3*b4*c1*c3*c4*d2**2*d3 + 2*a2*a3*b2*b3*b4*c2**2*c3*d1*d3*d4 - a2*a3*b2*b3*b4*c2**2*c4*d1*d3**2 - 2*a2*a3*b2*b3*b4*c2*c3**2*d1*d2*d4 + a2*a3*b2*b3*b4*c3**2*c4*d1*d2**2 - a2*a3*b2*b4**2*c1*c2**2*d3*d4**2 + 2*a2*a3*b2*b4**2*c1*c2*c4*d2*d3*d4 - a2*a3*b2*b4**2*c1*c4**2*d2**2*d3 + a2*a3*b2*b4**2*c2**2*c3*d1*d4**2 - 2*a2*a3*b2*b4**2*c2*c3*c4*d1*d2*d4 + a2*a3*b2*b4**2*c3*c4**2*d1*d2**2 + a2*a3*b3*b4**2*c1*c3**2*d2*d4**2 - 2*a2*a3*b3*b4**2*c1*c3*c4*d2*d3*d4 + a2*a3*b3*b4**2*c1*c4**2*d2*d3**2 - a2*a3*b3*b4**2*c2*c3**2*d1*d4**2 + 2*a2*a3*b3*b4**2*c2*c3*c4*d1*d3*d4 - a2*a3*b3*b4**2*c2*c4**2*d1*d3**2 + a2*a4*b2*b3**2*c1*c2**2*d3**2*d4 - 2*a2*a4*b2*b3**2*c1*c2*c3*d2*d3*d4 + a2*a4*b2*b3**2*c1*c3**2*d2**2*d4 - a2*a4*b2*b3**2*c2**2*c4*d1*d3**2 + 2*a2*a4*b2*b3**2*c2*c3*c4*d1*d2*d3 - a2*a4*b2*b3**2*c3**2*c4*d1*d2**2 + a2*a4*b2*b3*b4*c1*c2**2*d3*d4**2 - 2*a2*a4*b2*b3*b4*c1*c2*c3*d2*d4**2 + 2*a2*a4*b2*b3*b4*c1*c3*c4*d2**2*d4 - a2*a4*b2*b3*b4*c1*c4**2*d2**2*d3 + a2*a4*b2*b3*b4*c2**2*c3*d1*d4**2 - 2*a2*a4*b2*b3*b4*c2**2*c4*d1*d3*d4 + 2*a2*a4*b2*b3*b4*c2*c4**2*d1*d2*d3 - a2*a4*b2*b3*b4*c3*c4**2*d1*d2**2 - a2*a4*b3**2*b4*c1*c3**2*d2*d4**2 + 2*a2*a4*b3**2*b4*c1*c3*c4*d2*d3*d4 - a2*a4*b3**2*b4*c1*c4**2*d2*d3**2 + a2*a4*b3**2*b4*c2*c3**2*d1*d4**2 - 2*a2*a4*b3**2*b4*c2*c3*c4*d1*d3*d4 + a2*a4*b3**2*b4*c2*c4**2*d1*d3**2 + a3**2*b2**2*b4*c1*c2**2*d3**2*d4 - a3**2*b2**2*b4*c1*c2*c3*d2*d3*d4 - a3**2*b2**2*b4*c1*c2*c4*d2*d3**2 + a3**2*b2**2*b4*c1*c3*c4*d2**2*d3 - a3**2*b2**2*b4*c2**2*c3*d1*d3*d4 + a3**2*b2**2*b4*c2*c3**2*d1*d2*d4 + a3**2*b2**2*b4*c2*c3*c4*d1*d2*d3 - a3**2*b2**2*b4*c3**2*c4*d1*d2**2 - a3**2*b2*b4**2*c1*c2*c3*d3*d4**2 + a3**2*b2*b4**2*c1*c2*c4*d3**2*d4 + a3**2*b2*b4**2*c1*c3*c4*d2*d3*d4 - a3**2*b2*b4**2*c1*c4**2*d2*d3**2 + a3**2*b2*b4**2*c2*c3**2*d1*d4**2 - a3**2*b2*b4**2*c2*c3*c4*d1*d3*d4 - a3**2*b2*b4**2*c3**2*c4*d1*d2*d4 + a3**2*b2*b4**2*c3*c4**2*d1*d2*d3 - a3*a4*b2**2*b3*c1*c2**2*d3**2*d4 + 2*a3*a4*b2**2*b3*c1*c2*c3*d2*d3*d4 - a3*a4*b2**2*b3*c1*c3**2*d2**2*d4 + a3*a4*b2**2*b3*c2**2*c4*d1*d3**2 - 2*a3*a4*b2**2*b3*c2*c3*c4*d1*d2*d3 + a3*a4*b2**2*b3*c3**2*c4*d1*d2**2 + a3*a4*b2**2*b4*c1*c2**2*d3*d4**2 - 2*a3*a4*b2**2*b4*c1*c2*c4*d2*d3*d4 + a3*a4*b2**2*b4*c1*c4**2*d2**2*d3 - a3*a4*b2**2*b4*c2**2*c3*d1*d4**2 + 2*a3*a4*b2**2*b4*c2*c3*c4*d1*d2*d4 - a3*a4*b2**2*b4*c3*c4**2*d1*d2**2 + 2*a3*a4*b2*b3*b4*c1*c2*c3*d3*d4**2 - 2*a3*a4*b2*b3*b4*c1*c2*c4*d3**2*d4 - a3*a4*b2*b3*b4*c1*c3**2*d2*d4**2 + a3*a4*b2*b3*b4*c1*c4**2*d2*d3**2 - a3*a4*b2*b3*b4*c2*c3**2*d1*d4**2 + a3*a4*b2*b3*b4*c2*c4**2*d1*d3**2 + 2*a3*a4*b2*b3*b4*c3**2*c4*d1*d2*d4 - 2*a3*a4*b2*b3*b4*c3*c4**2*d1*d2*d3 - a4**2*b2**2*b3*c1*c2**2*d3*d4**2 + a4**2*b2**2*b3*c1*c2*c3*d2*d4**2 + a4**2*b2**2*b3*c1*c2*c4*d2*d3*d4 - a4**2*b2**2*b3*c1*c3*c4*d2**2*d4 + a4**2*b2**2*b3*c2**2*c4*d1*d3*d4 - a4**2*b2**2*b3*c2*c3*c4*d1*d2*d4 - a4**2*b2**2*b3*c2*c4**2*d1*d2*d3 + a4**2*b2**2*b3*c3*c4**2*d1*d2**2 - a4**2*b2*b3**2*c1*c2*c3*d3*d4**2 + a4**2*b2*b3**2*c1*c2*c4*d3**2*d4 + a4**2*b2*b3**2*c1*c3**2*d2*d4**2 - a4**2*b2*b3**2*c1*c3*c4*d2*d3*d4 + a4**2*b2*b3**2*c2*c3*c4*d1*d3*d4 - a4**2*b2*b3**2*c2*c4**2*d1*d3**2 - a4**2*b2*b3**2*c3**2*c4*d1*d2*d4 + a4**2*b2*b3**2*c3*c4**2*d1*d2*d3 - b2*c1*d2*x/4 + b2*c2*d1*x/4 - b3*c1*d3*x/4 + b3*c3*d1*x/4 - b4*c1*d4*x/4 + b4*c4*d1*x/4)/(a2**2*b3*b4*c1**2*c3*d2**2*d4 - a2**2*b3*b4*c1**2*c4*d2**2*d3 - 2*a2**2*b3*b4*c1*c2*c3*d1*d2*d4 + 2*a2**2*b3*b4*c1*c2*c4*d1*d2*d3 + a2**2*b3*b4*c2**2*c3*d1**2*d4 - a2**2*b3*b4*c2**2*c4*d1**2*d3 - a2*a3*b2*b4*c1**2*c3*d2**2*d4 + a2*a3*b2*b4*c1**2*c4*d2**2*d3 + 2*a2*a3*b2*b4*c1*c2*c3*d1*d2*d4 - 2*a2*a3*b2*b4*c1*c2*c4*d1*d2*d3 - a2*a3*b2*b4*c2**2*c3*d1**2*d4 + a2*a3*b2*b4*c2**2*c4*d1**2*d3 + a2*a3*b3*b4*c1**2*c2*d3**2*d4 - a2*a3*b3*b4*c1**2*c4*d2*d3**2 - 2*a2*a3*b3*b4*c1*c2*c3*d1*d3*d4 + 2*a2*a3*b3*b4*c1*c3*c4*d1*d2*d3 + a2*a3*b3*b4*c2*c3**2*d1**2*d4 - a2*a3*b3*b4*c3**2*c4*d1**2*d2 + a2*a3*b4**2*c1**2*c2*d3*d4**2 - a2*a3*b4**2*c1**2*c3*d2*d4**2 - 2*a2*a3*b4**2*c1*c2*c4*d1*d3*d4 + 2*a2*a3*b4**2*c1*c3*c4*d1*d2*d4 + a2*a3*b4**2*c2*c4**2*d1**2*d3 - a2*a3*b4**2*c3*c4**2*d1**2*d2 - a2*a4*b2*b3*c1**2*c3*d2**2*d4 + a2*a4*b2*b3*c1**2*c4*d2**2*d3 + 2*a2*a4*b2*b3*c1*c2*c3*d1*d2*d4 - 2*a2*a4*b2*b3*c1*c2*c4*d1*d2*d3 - a2*a4*b2*b3*c2**2*c3*d1**2*d4 + a2*a4*b2*b3*c2**2*c4*d1**2*d3 - a2*a4*b3**2*c1**2*c2*d3**2*d4 + a2*a4*b3**2*c1**2*c4*d2*d3**2 + 2*a2*a4*b3**2*c1*c2*c3*d1*d3*d4 - 2*a2*a4*b3**2*c1*c3*c4*d1*d2*d3 - a2*a4*b3**2*c2*c3**2*d1**2*d4 + a2*a4*b3**2*c3**2*c4*d1**2*d2 - a2*a4*b3*b4*c1**2*c2*d3*d4**2 + a2*a4*b3*b4*c1**2*c3*d2*d4**2 + 2*a2*a4*b3*b4*c1*c2*c4*d1*d3*d4 - 2*a2*a4*b3*b4*c1*c3*c4*d1*d2*d4 - a2*a4*b3*b4*c2*c4**2*d1**2*d3 + a2*a4*b3*b4*c3*c4**2*d1**2*d2 - a3**2*b2*b4*c1**2*c2*d3**2*d4 + a3**2*b2*b4*c1**2*c4*d2*d3**2 + 2*a3**2*b2*b4*c1*c2*c3*d1*d3*d4 - 2*a3**2*b2*b4*c1*c3*c4*d1*d2*d3 - a3**2*b2*b4*c2*c3**2*d1**2*d4 + a3**2*b2*b4*c3**2*c4*d1**2*d2 + a3*a4*b2**2*c1**2*c3*d2**2*d4 - a3*a4*b2**2*c1**2*c4*d2**2*d3 - 2*a3*a4*b2**2*c1*c2*c3*d1*d2*d4 + 2*a3*a4*b2**2*c1*c2*c4*d1*d2*d3 + a3*a4*b2**2*c2**2*c3*d1**2*d4 - a3*a4*b2**2*c2**2*c4*d1**2*d3 + a3*a4*b2*b3*c1**2*c2*d3**2*d4 - a3*a4*b2*b3*c1**2*c4*d2*d3**2 - 2*a3*a4*b2*b3*c1*c2*c3*d1*d3*d4 + 2*a3*a4*b2*b3*c1*c3*c4*d1*d2*d3 + a3*a4*b2*b3*c2*c3**2*d1**2*d4 - a3*a4*b2*b3*c3**2*c4*d1**2*d2 - a3*a4*b2*b4*c1**2*c2*d3*d4**2 + a3*a4*b2*b4*c1**2*c3*d2*d4**2 + 2*a3*a4*b2*b4*c1*c2*c4*d1*d3*d4 - 2*a3*a4*b2*b4*c1*c3*c4*d1*d2*d4 - a3*a4*b2*b4*c2*c4**2*d1**2*d3 + a3*a4*b2*b4*c3*c4**2*d1**2*d2 + a4**2*b2*b3*c1**2*c2*d3*d4**2 - a4**2*b2*b3*c1**2*c3*d2*d4**2 - 2*a4**2*b2*b3*c1*c2*c4*d1*d3*d4 + 2*a4**2*b2*b3*c1*c3*c4*d1*d2*d4 + a4**2*b2*b3*c2*c4**2*d1**2*d3 - a4**2*b2*b3*c3*c4**2*d1**2*d2)  # noqa
            a1 = (a2**2*b1*b3*b4*c1*c3*d2**2*d4 - a2**2*b1*b3*b4*c1*c4*d2**2*d3 - a2**2*b1*b3*b4*c2*c3*d1*d2*d4 + a2**2*b1*b3*b4*c2*c4*d1*d2*d3 - a2**2*b3**2*b4*c2*c3*d2*d3*d4 + a2**2*b3**2*b4*c2*c4*d2*d3**2 + a2**2*b3**2*b4*c3**2*d2**2*d4 - a2**2*b3**2*b4*c3*c4*d2**2*d3 - a2**2*b3*b4**2*c2*c3*d2*d4**2 + a2**2*b3*b4**2*c2*c4*d2*d3*d4 + a2**2*b3*b4**2*c3*c4*d2**2*d4 - a2**2*b3*b4**2*c4**2*d2**2*d3 - a2*a3*b1*b2*b4*c1*c3*d2**2*d4 + a2*a3*b1*b2*b4*c1*c4*d2**2*d3 + a2*a3*b1*b2*b4*c2**2*d1*d3*d4 - 2*a2*a3*b1*b2*b4*c2*c4*d1*d2*d3 + a2*a3*b1*b2*b4*c3*c4*d1*d2**2 + a2*a3*b1*b3*b4*c1*c2*d3**2*d4 - a2*a3*b1*b3*b4*c1*c4*d2*d3**2 - a2*a3*b1*b3*b4*c2*c4*d1*d3**2 - a2*a3*b1*b3*b4*c3**2*d1*d2*d4 + 2*a2*a3*b1*b3*b4*c3*c4*d1*d2*d3 + a2*a3*b1*b4**2*c1*c2*d3*d4**2 - a2*a3*b1*b4**2*c1*c3*d2*d4**2 - a2*a3*b1*b4**2*c2*c4*d1*d3*d4 + a2*a3*b1*b4**2*c3*c4*d1*d2*d4 + a2*a3*b2*b3*b4*c2**2*d3**2*d4 - 2*a2*a3*b2*b3*b4*c2*c4*d2*d3**2 - a2*a3*b2*b3*b4*c3**2*d2**2*d4 + 2*a2*a3*b2*b3*b4*c3*c4*d2**2*d3 + a2*a3*b2*b4**2*c2**2*d3*d4**2 - 2*a2*a3*b2*b4**2*c2*c4*d2*d3*d4 + a2*a3*b2*b4**2*c4**2*d2**2*d3 - a2*a3*b3*b4**2*c3**2*d2*d4**2 + 2*a2*a3*b3*b4**2*c3*c4*d2*d3*d4 - a2*a3*b3*b4**2*c4**2*d2*d3**2 - a2*a4*b1*b2*b3*c1*c3*d2**2*d4 + a2*a4*b1*b2*b3*c1*c4*d2**2*d3 - a2*a4*b1*b2*b3*c2**2*d1*d3*d4 + 2*a2*a4*b1*b2*b3*c2*c3*d1*d2*d4 - a2*a4*b1*b2*b3*c3*c4*d1*d2**2 - a2*a4*b1*b3**2*c1*c2*d3**2*d4 + a2*a4*b1*b3**2*c1*c4*d2*d3**2 + a2*a4*b1*b3**2*c2*c3*d1*d3*d4 - a2*a4*b1*b3**2*c3*c4*d1*d2*d3 - a2*a4*b1*b3*b4*c1*c2*d3*d4**2 + a2*a4*b1*b3*b4*c1*c3*d2*d4**2 + a2*a4*b1*b3*b4*c2*c3*d1*d4**2 - 2*a2*a4*b1*b3*b4*c3*c4*d1*d2*d4 + a2*a4*b1*b3*b4*c4**2*d1*d2*d3 - a2*a4*b2*b3**2*c2**2*d3**2*d4 + 2*a2*a4*b2*b3**2*c2*c3*d2*d3*d4 - a2*a4*b2*b3**2*c3**2*d2**2*d4 - a2*a4*b2*b3*b4*c2**2*d3*d4**2 + 2*a2*a4*b2*b3*b4*c2*c3*d2*d4**2 - 2*a2*a4*b2*b3*b4*c3*c4*d2**2*d4 + a2*a4*b2*b3*b4*c4**2*d2**2*d3 + a2*a4*b3**2*b4*c3**2*d2*d4**2 - 2*a2*a4*b3**2*b4*c3*c4*d2*d3*d4 + a2*a4*b3**2*b4*c4**2*d2*d3**2 - a3**2*b1*b2*b4*c1*c2*d3**2*d4 + a3**2*b1*b2*b4*c1*c4*d2*d3**2 + a3**2*b1*b2*b4*c2*c3*d1*d3*d4 - a3**2*b1*b2*b4*c3*c4*d1*d2*d3 - a3**2*b2**2*b4*c2**2*d3**2*d4 + a3**2*b2**2*b4*c2*c3*d2*d3*d4 + a3**2*b2**2*b4*c2*c4*d2*d3**2 - a3**2*b2**2*b4*c3*c4*d2**2*d3 + a3**2*b2*b4**2*c2*c3*d3*d4**2 - a3**2*b2*b4**2*c2*c4*d3**2*d4 - a3**2*b2*b4**2*c3*c4*d2*d3*d4 + a3**2*b2*b4**2*c4**2*d2*d3**2 + a3*a4*b1*b2**2*c1*c3*d2**2*d4 - a3*a4*b1*b2**2*c1*c4*d2**2*d3 - a3*a4*b1*b2**2*c2*c3*d1*d2*d4 + a3*a4*b1*b2**2*c2*c4*d1*d2*d3 + a3*a4*b1*b2*b3*c1*c2*d3**2*d4 - a3*a4*b1*b2*b3*c1*c4*d2*d3**2 - 2*a3*a4*b1*b2*b3*c2*c3*d1*d3*d4 + a3*a4*b1*b2*b3*c2*c4*d1*d3**2 + a3*a4*b1*b2*b3*c3**2*d1*d2*d4 - a3*a4*b1*b2*b4*c1*c2*d3*d4**2 + a3*a4*b1*b2*b4*c1*c3*d2*d4**2 - a3*a4*b1*b2*b4*c2*c3*d1*d4**2 + 2*a3*a4*b1*b2*b4*c2*c4*d1*d3*d4 - a3*a4*b1*b2*b4*c4**2*d1*d2*d3 + a3*a4*b2**2*b3*c2**2*d3**2*d4 - 2*a3*a4*b2**2*b3*c2*c3*d2*d3*d4 + a3*a4*b2**2*b3*c3**2*d2**2*d4 - a3*a4*b2**2*b4*c2**2*d3*d4**2 + 2*a3*a4*b2**2*b4*c2*c4*d2*d3*d4 - a3*a4*b2**2*b4*c4**2*d2**2*d3 - 2*a3*a4*b2*b3*b4*c2*c3*d3*d4**2 + 2*a3*a4*b2*b3*b4*c2*c4*d3**2*d4 + a3*a4*b2*b3*b4*c3**2*d2*d4**2 - a3*a4*b2*b3*b4*c4**2*d2*d3**2 + a4**2*b1*b2*b3*c1*c2*d3*d4**2 - a4**2*b1*b2*b3*c1*c3*d2*d4**2 - a4**2*b1*b2*b3*c2*c4*d1*d3*d4 + a4**2*b1*b2*b3*c3*c4*d1*d2*d4 + a4**2*b2**2*b3*c2**2*d3*d4**2 - a4**2*b2**2*b3*c2*c3*d2*d4**2 - a4**2*b2**2*b3*c2*c4*d2*d3*d4 + a4**2*b2**2*b3*c3*c4*d2**2*d4 + a4**2*b2*b3**2*c2*c3*d3*d4**2 - a4**2*b2*b3**2*c2*c4*d3**2*d4 - a4**2*b2*b3**2*c3**2*d2*d4**2 + a4**2*b2*b3**2*c3*c4*d2*d3*d4 + b2*d2*x/4 + b3*d3*x/4 + b4*d4*x/4)/(d1*(a2*b3**2*b4*c2*c3*d3*d4 - a2*b3**2*b4*c2*c4*d3**2 - a2*b3**2*b4*c3**2*d2*d4 + a2*b3**2*b4*c3*c4*d2*d3 + a2*b3*b4**2*c2*c3*d4**2 - a2*b3*b4**2*c2*c4*d3*d4 - a2*b3*b4**2*c3*c4*d2*d4 + a2*b3*b4**2*c4**2*d2*d3 + a3*b2**2*b4*c2**2*d3*d4 - a3*b2**2*b4*c2*c3*d2*d4 - a3*b2**2*b4*c2*c4*d2*d3 + a3*b2**2*b4*c3*c4*d2**2 - a3*b2*b4**2*c2*c3*d4**2 + a3*b2*b4**2*c2*c4*d3*d4 + a3*b2*b4**2*c3*c4*d2*d4 - a3*b2*b4**2*c4**2*d2*d3 - a4*b2**2*b3*c2**2*d3*d4 + a4*b2**2*b3*c2*c3*d2*d4 + a4*b2**2*b3*c2*c4*d2*d3 - a4*b2**2*b3*c3*c4*d2**2 - a4*b2*b3**2*c2*c3*d3*d4 + a4*b2*b3**2*c2*c4*d3**2 + a4*b2*b3**2*c3**2*d2*d4 - a4*b2*b3**2*c3*c4*d2*d3))  # noqa
            c5 = d5*(-4*a2**2*b3*b4*c1*c2*c3*d2*d4 + 4*a2**2*b3*b4*c1*c2*c4*d2*d3 + 4*a2**2*b3*b4*c2**2*c3*d1*d4 - 4*a2**2*b3*b4*c2**2*c4*d1*d3 + 4*a2*a3*b2*b4*c1*c2*c3*d2*d4 - 4*a2*a3*b2*b4*c1*c2*c4*d2*d3 - 4*a2*a3*b2*b4*c2**2*c3*d1*d4 + 4*a2*a3*b2*b4*c2**2*c4*d1*d3 - 4*a2*a3*b3*b4*c1*c2*c3*d3*d4 + 4*a2*a3*b3*b4*c1*c3*c4*d2*d3 + 4*a2*a3*b3*b4*c2*c3**2*d1*d4 - 4*a2*a3*b3*b4*c3**2*c4*d1*d2 - 4*a2*a3*b4**2*c1*c2*c4*d3*d4 + 4*a2*a3*b4**2*c1*c3*c4*d2*d4 + 4*a2*a3*b4**2*c2*c4**2*d1*d3 - 4*a2*a3*b4**2*c3*c4**2*d1*d2 + 4*a2*a4*b2*b3*c1*c2*c3*d2*d4 - 4*a2*a4*b2*b3*c1*c2*c4*d2*d3 - 4*a2*a4*b2*b3*c2**2*c3*d1*d4 + 4*a2*a4*b2*b3*c2**2*c4*d1*d3 + 4*a2*a4*b3**2*c1*c2*c3*d3*d4 - 4*a2*a4*b3**2*c1*c3*c4*d2*d3 - 4*a2*a4*b3**2*c2*c3**2*d1*d4 + 4*a2*a4*b3**2*c3**2*c4*d1*d2 + 4*a2*a4*b3*b4*c1*c2*c4*d3*d4 - 4*a2*a4*b3*b4*c1*c3*c4*d2*d4 - 4*a2*a4*b3*b4*c2*c4**2*d1*d3 + 4*a2*a4*b3*b4*c3*c4**2*d1*d2 + 4*a3**2*b2*b4*c1*c2*c3*d3*d4 - 4*a3**2*b2*b4*c1*c3*c4*d2*d3 - 4*a3**2*b2*b4*c2*c3**2*d1*d4 + 4*a3**2*b2*b4*c3**2*c4*d1*d2 - 4*a3*a4*b2**2*c1*c2*c3*d2*d4 + 4*a3*a4*b2**2*c1*c2*c4*d2*d3 + 4*a3*a4*b2**2*c2**2*c3*d1*d4 - 4*a3*a4*b2**2*c2**2*c4*d1*d3 - 4*a3*a4*b2*b3*c1*c2*c3*d3*d4 + 4*a3*a4*b2*b3*c1*c3*c4*d2*d3 + 4*a3*a4*b2*b3*c2*c3**2*d1*d4 - 4*a3*a4*b2*b3*c3**2*c4*d1*d2 + 4*a3*a4*b2*b4*c1*c2*c4*d3*d4 - 4*a3*a4*b2*b4*c1*c3*c4*d2*d4 - 4*a3*a4*b2*b4*c2*c4**2*d1*d3 + 4*a3*a4*b2*b4*c3*c4**2*d1*d2 - 4*a4**2*b2*b3*c1*c2*c4*d3*d4 + 4*a4**2*b2*b3*c1*c3*c4*d2*d4 + 4*a4**2*b2*b3*c2*c4**2*d1*d3 - 4*a4**2*b2*b3*c3*c4**2*d1*d2 + c1*x)/(-4*a2**2*b3*b4*c1*c3*d2**2*d4 + 4*a2**2*b3*b4*c1*c4*d2**2*d3 + 4*a2**2*b3*b4*c2*c3*d1*d2*d4 - 4*a2**2*b3*b4*c2*c4*d1*d2*d3 + 4*a2*a3*b2*b4*c1*c3*d2**2*d4 - 4*a2*a3*b2*b4*c1*c4*d2**2*d3 - 4*a2*a3*b2*b4*c2*c3*d1*d2*d4 + 4*a2*a3*b2*b4*c2*c4*d1*d2*d3 - 4*a2*a3*b3*b4*c1*c2*d3**2*d4 + 4*a2*a3*b3*b4*c1*c4*d2*d3**2 + 4*a2*a3*b3*b4*c2*c3*d1*d3*d4 - 4*a2*a3*b3*b4*c3*c4*d1*d2*d3 - 4*a2*a3*b4**2*c1*c2*d3*d4**2 + 4*a2*a3*b4**2*c1*c3*d2*d4**2 + 4*a2*a3*b4**2*c2*c4*d1*d3*d4 - 4*a2*a3*b4**2*c3*c4*d1*d2*d4 + 4*a2*a4*b2*b3*c1*c3*d2**2*d4 - 4*a2*a4*b2*b3*c1*c4*d2**2*d3 - 4*a2*a4*b2*b3*c2*c3*d1*d2*d4 + 4*a2*a4*b2*b3*c2*c4*d1*d2*d3 + 4*a2*a4*b3**2*c1*c2*d3**2*d4 - 4*a2*a4*b3**2*c1*c4*d2*d3**2 - 4*a2*a4*b3**2*c2*c3*d1*d3*d4 + 4*a2*a4*b3**2*c3*c4*d1*d2*d3 + 4*a2*a4*b3*b4*c1*c2*d3*d4**2 - 4*a2*a4*b3*b4*c1*c3*d2*d4**2 - 4*a2*a4*b3*b4*c2*c4*d1*d3*d4 + 4*a2*a4*b3*b4*c3*c4*d1*d2*d4 + 4*a3**2*b2*b4*c1*c2*d3**2*d4 - 4*a3**2*b2*b4*c1*c4*d2*d3**2 - 4*a3**2*b2*b4*c2*c3*d1*d3*d4 + 4*a3**2*b2*b4*c3*c4*d1*d2*d3 - 4*a3*a4*b2**2*c1*c3*d2**2*d4 + 4*a3*a4*b2**2*c1*c4*d2**2*d3 + 4*a3*a4*b2**2*c2*c3*d1*d2*d4 - 4*a3*a4*b2**2*c2*c4*d1*d2*d3 - 4*a3*a4*b2*b3*c1*c2*d3**2*d4 + 4*a3*a4*b2*b3*c1*c4*d2*d3**2 + 4*a3*a4*b2*b3*c2*c3*d1*d3*d4 - 4*a3*a4*b2*b3*c3*c4*d1*d2*d3 + 4*a3*a4*b2*b4*c1*c2*d3*d4**2 - 4*a3*a4*b2*b4*c1*c3*d2*d4**2 - 4*a3*a4*b2*b4*c2*c4*d1*d3*d4 + 4*a3*a4*b2*b4*c3*c4*d1*d2*d4 - 4*a4**2*b2*b3*c1*c2*d3*d4**2 + 4*a4**2*b2*b3*c1*c3*d2*d4**2 + 4*a4**2*b2*b3*c2*c4*d1*d3*d4 - 4*a4**2*b2*b3*c3*c4*d1*d2*d4 + d1*x)  # noqa
            b5 = (-b1*d1 - b2*d2 - b3*d3 - b4*d4)/d5  # noqa
            a5 = b5*(a2*c1*d2 - a2*c2*d1 + a3*c1*d3 - a3*c3*d1 + a4*c1*d4 - a4*c4*d1)/(b2*c1*d2 - b2*c2*d1 + b3*c1*d3 - b3*c3*d1 + b4*c1*d4 - b4*c4*d1)  # noqa

            self[1].r_sp_d = numpy.array([[a1], [b1]], dtype=object)
            self[1].l_sp_d = numpy.array([[c1, d1]], dtype=object)
            self[2].r_sp_d = numpy.array([[a2], [b2]], dtype=object)
            self[2].l_sp_d = numpy.array([[c2, d2]], dtype=object)
            self[3].r_sp_d = numpy.array([[a3], [b3]], dtype=object)
            self[3].l_sp_d = numpy.array([[c3, d3]], dtype=object)
            self[4].r_sp_d = numpy.array([[a4], [b4]], dtype=object)
            self[4].l_sp_d = numpy.array([[c4, d4]], dtype=object)
            self[5].r_sp_d = numpy.array([[a5], [b5]], dtype=object)
            self[5].l_sp_d = numpy.array([[c5, d5]], dtype=object)

        else:  # at 6-point and above there is a simple solution, as the system becomes block diagonal with a suitable choice of variables.
            X = temp_value
            A, B = self(f"[{a}|{b}|{c}|{d}|-[{a}|{d}|{c}|{b}|").flatten().tolist()
            C, D = self[a].r_sp_d[0, 0], self[a].r_sp_d[1, 0]
            C = (X - B * D) / A
            self[a].r_sp_d = numpy.array([C, D], dtype=type(C))

            if fix_mom is True:
                self.fix_mom_cons(plist[0], plist[1], axis=1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_p5Bdiff(self, temp_string, temp_value, fix_mom=True):
        # not tested beyond 6-point massless kinematics
        a, b, cd, e, f = p5Bdiff.findall(temp_string)[0]
        a, b, c, d, e, f = map(int, [a, b, ] + cd.split("+") + [e, f])

        # (⟨a|b|c+d|e|a]-⟨b|f|c+d|e|b]) = (⟨a|b|c+d|e⟩[e|a]-⟨b|f|c+d|e⟩[e|b]) = (-⟨a|b|c+d|e⟩[a|-⟨b|f|c+d|e⟩[b|)|e]
        X = temp_value
        A, B = self(f"(⟨{a}|{b}|{c}+{d}|{e}⟩[{a}|-⟨{b}|{f}|{c}+{d}|{e}⟩[{b}|)").flatten().tolist()
        C, D = self[e].l_sp_u[0, 0], self[e].l_sp_u[1, 0]
        C = (- X - B * D) / A
        self[e].l_sp_u = numpy.array([C, D], dtype=type(C))  # set |e]

        if fix_mom:
            self.fix_mom_cons(c, d)
