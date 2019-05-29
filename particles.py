#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   ___          _   _    _
#  | _ \__ _ _ _| |_(_)__| |___ ___
#  |  _/ _` | '_|  _| / _| / -_|_-<
#  |_| \__,_|_|  \__|_\__|_\___/__/

# Author: Giuseppe


from __future__ import unicode_literals

import sys
import time
import numpy as np
import random
import re
import os

from copy import deepcopy
from antares.core.bh_patch import accuracy
from antares.core.tools import settings, MinkowskiMetric, Pauli, Pauli_bar, flatten, pA2, pS2, pNB, Hyphens, vec_to_str, myException, mapThreads
from antares.core.invariants import All_Strings
from antares.particles.particle import Particle
from antares.particles.particles_compute import Particles_Compute
from antares.particles.particles_set import Particles_Set
from antares.particles.particles_set_pair import Particles_SetPair


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def indexing_decorator(func):
    """Rebases a list to start from index 1."""

    def decorated(self, index, *args):
        if index < 1:
            raise IndexError('Indices start from 1')
        elif index > 0 and index < len(self) + 1:
            index -= 1
        elif index > len(self):
            raise IndexError('Indices can\'t exceed {}'.format(len(self)))

        return func(self, index, *args)

    return decorated


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles(Particles_Compute, Particles_Set, Particles_SetPair, list):
    """Describes the kinematics of n particles."""

    def __init__(self, number_of_particles_or_particles=None, seed=None):
        """Initialisation. Requires either multiplicity of phace space or list of Particle objects."""
        list.__init__(self)
        if isinstance(number_of_particles_or_particles, int):
            random.seed(seed) if seed is not None else random.seed()
            for i in range(number_of_particles_or_particles):
                self.append(Particle())
        elif isinstance(number_of_particles_or_particles, list):
            for oParticle in number_of_particles_or_particles:
                self.append(oParticle)
        elif number_of_particles_or_particles is not None:
            raise Exception("Invalid initialisation of Particles instance.")

    def __eq__(self, other):
        if type(self) == type(other):
            return all(self[i] == other[i] for i in range(1, len(self) + 1))
        else:
            return False

    def __hash__(self):
        return hash("".join(flatten([map(str, oParticle.four_mom) for oParticle in self])))

    def randomise_all(self):
        """Randomises all particles. Breaks momentum conservation."""
        for oParticle in self:
            oParticle.randomise()

    def angles_for_squares(self):
        """Switches all angle brackets for square brackets and viceversa."""
        for oParticle in self:
            oParticle.angles_for_squares()

    def image(self, permutation):
        """Returns the image of self under a given permutation. Remember, this is a passive transformation."""
        return deepcopy(Particles(sorted(self, key=lambda x: permutation[self.index(x)])))

    def fix_mom_cons(self, A=0, B=0, real_momenta=False, axis=1):   # using real momenta changes both |⟩ and |] of A & B
        """Fixes momentum conservation using particles A and B."""
        if A == 0 and B == 0:                                       # defaults to random particles to fix mom cons
            A, B = random.sample(range(1, len(self) + 1), 2)
            self.fix_mom_cons(A, B, real_momenta, axis)

        elif A != 0 and B != 0 and real_momenta is True:            # the following results in real momenta, but changes both |⟩ and |] of A & B
            ex = np.array([0. + 0j, 0., 0., 0.])                    # excess momentum init. to 0
            for i in range(1, len(self) + 1):
                if i != A and i != B:
                    ex = ex - self[i].four_mom
            numerator = (ex[1] * ex[1] + ex[2] * ex[2] + ex[3] * ex[3] - ex[0] * ex[0])
            if axis == 1:
                denominator = (2 * ex[1] - 2 * ex[0])
            elif axis == 2:
                denominator = (2 * ex[2] - 2 * ex[0])
            elif axis == 3:
                denominator = (2 * ex[3] - 2 * ex[0])
            p_fix = numerator / denominator
            if axis == 1:
                self[A].four_mom = np.array([p_fix, p_fix, 0, 0])
            elif axis == 2:
                self[A].four_mom = np.array([p_fix, 0, p_fix, 0])
            elif axis == 3:
                self[A].four_mom = np.array([p_fix, 0, 0, p_fix])
            ex = ex - self[A].four_mom
            self[B].four_mom = ex

        elif A != 0 and B != 0:                                     # the following results in complex momenta, it changes |A⟩&|B⟩ (axis=1) or |A]&|B] (axis=2)
            K = np.array([0. + 0j, 0., 0., 0.])                       # Note: axis here is a meaningless name. It is just used as a switch.
            for i in range(1, len(self) + 1):                         # minus sum of all other momenta
                if i != A and i != B:
                    K = K - self[i].four_mom
            K11 = K[0] + K[3]
            K12 = K[1] - 1j * K[2]
            K21 = K[1] + 1j * K[2]
            K22 = K[0] - K[3]
            a, b = self[A].r_sp_d[0, 0], self[A].r_sp_d[1, 0]
            c, d = self[A].l_sp_d[0, 0], self[A].l_sp_d[0, 1]
            e, f = self[B].r_sp_d[0, 0], self[B].r_sp_d[1, 0]
            g, h = self[B].l_sp_d[0, 0], self[B].l_sp_d[0, 1]
            if axis == 1:                                           # change |A⟩ and |B⟩
                a = (g * K12 - h * K11) / (d * g - c * h)
                b = (g * K22 - h * K21) / (d * g - c * h)
                e = (d * K11 - c * K12) / (d * g - c * h)
                f = (d * K21 - c * K22) / (d * g - c * h)
                self[A].r_sp_d = np.array([a, b])
                self[B].r_sp_d = np.array([e, f])
            else:                                                   # change |A] and |B]
                c = (e * K21 - f * K11) / (b * e - a * f)
                d = (e * K22 - f * K12) / (b * e - a * f)
                g = (b * K11 - a * K21) / (b * e - a * f)
                h = (b * K12 - a * K22) / (b * e - a * f)
                self[A].l_sp_d = np.array([c, d])
                self[B].l_sp_d = np.array([g, h])

    def momentum_conservation_check(self, silent=True):
        """Returns true if momentum is conserved."""
        mom_violation = 0                                           # momentum conservation violation
        for i in range(4):
            if abs(self.total_mom[i]) > mom_violation:
                mom_violation = abs(self.total_mom[i])
        if silent is False:
            print("The largest momentum violation is {}".format(mom_violation))
        if mom_violation > 10 ** -(0.9 * accuracy()):
            myException("Momentum conservation violation.")
            return False
        return True

    def onshell_relation_check(self, silent=True):
        """Returns true if all on-shell relations are satisfied."""
        onshell_violation = 0                                       # onshell violation
        for i in range(1, len(self) + 1):
            if abs(self.ldot(i, i)) > onshell_violation:
                onshell_violation = abs(self.ldot(i, i))
        if silent is False:
            print("The largest on shell violation is {}".format(onshell_violation))
            print("{}-------------------{}".format(Hyphens, Hyphens))
        if onshell_violation > 10 ** -(0.9 * accuracy()):
            myException("Onshell relation violation.")
            return False
        return True

    def phasespace_consistency_check(self, invariants=[], silent=True):
        """Runs momentum and onshell checks. Looks for outliers in phase space."""

        if invariants == []:
            _invars = All_Strings(len(self), "2")
        else:
            _invars = [invariant for invariant in invariants]

        if silent is False:
            print ""
            print("{} Consistency check {}".format(Hyphens, Hyphens))

        mom_cons = self.momentum_conservation_check(silent)         # momentum conservation violation
        on_shell = self.onshell_relation_check(silent)              # onshell violation

        values = []                                                 # smallest and biggest invariants
        for _invar in _invars:
            values += [abs(self.compute(_invar))]
            if "n" in unicode(values[len(values) - 1]) and silent is False:
                print "not a number!", values[len(values) - 1], "invariant", _invar
                return False, False, [], []
        while True:
            Break = True
            for i in range(len(_invars) - 1):
                if values[i] > values[i + 1]:
                    t_v = values[i + 1]
                    values[i + 1] = values[i]
                    values[i] = t_v
                    t_i = _invars[i + 1]
                    _invars[i + 1] = _invars[i]
                    _invars[i] = t_i
                    Break = False
            if Break is True:
                break
        small_outliers, small_outliers_values = [], []
        big_outliers, big_outliers_values = [], []
        for i in range(len(_invars)):
            if values[i] < 0.00001:
                small_outliers += [_invars[i]]
                small_outliers_values += [values[i]]
                if silent is False:
                    print "{} = {}".format(_invars[i], values[i])
            if values[i] > 0.0001:
                break
        if silent is False:
            print "..."
        for i in range(len(_invars)):
            if values[i] > 100000:
                myException("Outliers are big!")
                big_outliers += [_invars[i]]
                big_outliers_values += [values[i]]
                if silent is False:
                    print "{} = {}".format(_invars[i], values[i])
        if silent is False:
            print("{}-------------------{}".format(Hyphens, Hyphens))
        return mom_cons, on_shell, big_outliers, small_outliers

    # BASE ONE LIST

    @indexing_decorator
    def __getitem__(self, index):
        return list.__getitem__(self, index)

    @indexing_decorator
    def __setitem__(self, index, value):
        list.__setitem__(self, index, value)

    @indexing_decorator
    def __delitem__(self, index):
        list.__delitem__(self, index)

    @indexing_decorator
    def insert(self, index, value):
        list.insert(self, index, value)

    # MISCELLANEOUS

    def _can_fix_mom_cons(self, t_s1, t_s2):                        # Intended behaviour: returns False if it is impossible, otherwise returns tuple + axis
        if ((pA2.findall(t_s1) != [] or pS2.findall(t_s1) != []) and pNB.findall(t_s2) != []):
            if pA2.findall(t_s1) != []:
                ab = map(int, list(pA2.findall(t_s1)[0]))
            elif pS2.findall(t_s1) != []:
                ab = map(int, list(pS2.findall(t_s1)[0]))
            lNB, lNBs, lNBms, lNBe = self._get_lNB(t_s2)
            plist = self._complementary(ab + lNB)
            if len(plist) >= 2:                                     # easy momentum fix: two free particles
                return (plist[0], plist[1]), 1
            elif len(plist) == 1:                                   # complicated momentum fix: one free particle
                if ab[0] not in lNB:
                    if t_s1[0] == "⟨":
                        return (plist[0], ab[0]), 2                 # plist[0] and ab[0]
                    else:
                        return (plist[0], ab[0]), 1
                elif ab[1] not in lNB:
                    if t_s1[-1] == "⟩":
                        return (plist[0], ab[1]), 2                 # plist[0] and ab[1]
                    else:
                        return (plist[0], ab[1]), 1
                elif (lNBs not in ab or t_s1[0] == t_s2[0]) and (lNBs != lNBe or len(lNBms) % 2 == 0) and all(lNBs not in lNBm for lNBm in lNBms):
                    if t_s2[0] == "⟨":
                        return (plist[0], lNBs), 2                  # plist[0] and lNBs
                    else:
                        return (plist[0], lNBs), 1
                elif (lNBe not in ab or t_s1[-1] == t_s2[-1]) and (lNBs != lNBe or len(lNBms) % 2 == 0) and all(lNBe not in lNBm for lNBm in lNBms):
                    if t_s2[-1] == "⟩":
                        return (plist[0], lNBe), 2                  # plist[0] and lNBe
                    else:
                        return (plist[0], lNBe), 1
                else:
                    myException("Not enough particles to fix mom cons! (One free particle)")
                    return False, 0
            else:                                                   # almost impossible momentum fix: no free particles
                if ab[0] not in lNB and ab[1] not in lNB:
                    if t_s1[0] == "⟨":
                        return False, 0                             # ab[0] and ab[1] would result in big outliers
                    else:
                        return False, 0
                elif (ab[0] not in lNB and lNBs not in ab and t_s1[0] == t_s2[0] and
                      (lNBs != lNBe or len(lNBms) % 2 == 0) and all(lNBs not in lNBm for lNBm in lNBms)):
                    if t_s1[0] == "⟨":
                        return (ab[0], lNBs), 2                     # ab[0] and lNBs
                    else:
                        return (ab[0], lNBs), 1
                elif (ab[1] not in lNB and lNBs not in ab and t_s1[0] == t_s2[0] and
                      (lNBs != lNBe or len(lNBms) % 2 == 0) and all(lNBs not in lNBm for lNBm in lNBms)):
                    if t_s1[0] == "⟨":
                        return (ab[1], lNBs), 2                     # ab[1] and lNBs
                    else:
                        return (ab[1], lNBs), 1
                elif (ab[0] not in lNB and lNBe not in ab and t_s1[-1] == t_s2[-1] and
                      (lNBe != lNBs or len(lNBms) % 2 == 0) and all(lNBe not in lNBm for lNBm in lNBms)):
                    if t_s1[-1] == "⟩":
                        return (ab[0], lNBe), 2                     # ab[0] and lNBe
                    else:
                        return (ab[0], lNBe), 1
                elif (ab[1] not in lNB and lNBe not in ab and t_s1[-1] == t_s2[-1] and
                      (lNBe != lNBs or len(lNBms) % 2 == 0) and all(lNBe not in lNBm for lNBm in lNBms)):
                    if t_s1[-1] == "⟩":
                        return (ab[1], lNBe), 2                     # ab[1] and lNBe
                    else:
                        return (ab[1], lNBe), 1
                elif (((lNBs not in ab or t_s1[0] == t_s2[0]) and all(lNBs not in lNBm for lNBm in lNBms)) and (lNBs != lNBe) and
                      ((lNBe not in ab or t_s1[-1] == t_s2[-1]) and all(lNBe not in lNBm for lNBm in lNBms)) and len(lNBms) % 2 == 0):
                    if t_s2[0] == "⟨":
                        return (lNBs, lNBe), 2
                    else:
                        return (lNBs, lNBe), 1
                else:
                    myException("Not enough particles to fix mom cons! (Zero free particles)")
                    return False, 0

    @staticmethod
    def _lNB_to_string(start, lNBs, lNBms, lNBe, end):
        start = start + unicode(lNBs) + "|"
        end = "|" + unicode(lNBe) + end
        middle = "".join(string + "|" for string in ["(" + "".join(unicode(entry) + "+" for entry in item)[:-1] + ")" for item in lNBms])
        middle = middle[:-1]
        t_s_new = start + middle + end
        return t_s_new

    @staticmethod
    def _get_lNB(temp_string):                                      # usage: lNB, lNBs, lNBms, lNBe = _get_lNB(temp_string)
        lNB = list(pNB.findall(temp_string)[0])
        lNB[1] = [entry.replace("(", "").replace(")", "").split("+") for entry in lNB[1].split("|")]
        lNBs, lNBms, lNBe = int(lNB[0]), [map(int, entry) for entry in lNB[1]], int(lNB[2])
        lNB = map(int, [lNBs] + [entry for sublist in lNB[1] for entry in sublist] + [lNBe])
        return lNB, lNBs, lNBms, lNBe

    def _complementary(self, temp_list):                            # returns the list obtained by using momentum conservation
        temp_list = flatten(temp_list)
        if type(temp_list) == list:                                 # make sure it is a set (no double entries)
            temp_list = set(temp_list)
        original_type = type(list(temp_list)[0])
        if type(list(temp_list)[0]) == unicode:                         # make sure entries are integers (representing particle #)
            temp_list = set(map(int, temp_list))
        temp_list = list(temp_list)
        n = len(self)
        c_list = []
        for i in range(1, n + 1):
            c_list += [i]
        for element in temp_list:
            if element in c_list:
                c_list.remove(element)
        c_list = map(original_type, c_list)
        return c_list

    def ijk_to_3Ks(self, ijk):                                      # this method is used for Delta computation and setting
        K = [0, 0, 0]
        for i in range(3):
            K[i] = np.array([0, 0, 0, 0])
            j = ijk[i]
            while (j != ijk[(i + 1) % 3] and j != ijk[(i + 2) % 3]):
                K[i] = K[i] + self[j].four_mom
                j = j + 1
                j = j % len(self)
                if j == 0:
                    j = len(self)
        temp_oParticles = Particles(3)
        temp_oParticles[1].four_mom = K[0]
        temp_oParticles[2].four_mom = K[1]
        temp_oParticles[3].four_mom = K[2]
        return temp_oParticles

    def ijk_to_3NonOverlappingLists(self, ijk, mode=1):             # this method is used for Delta computation and setting
        NonOverlappingLists = [[ijk[0]], [ijk[1]], [ijk[2]]]
        for i in range(3):
            while True:
                last_entry = NonOverlappingLists[i][len(NonOverlappingLists[i]) - 1]
                new_entry = (last_entry + 1) % len(self)
                if new_entry == 0:
                    new_entry = len(self)
                if (new_entry not in NonOverlappingLists[(i + 1) % 3] and
                   new_entry not in NonOverlappingLists[(i + 2) % 3]):
                    NonOverlappingLists[i] += [new_entry]
                else:
                    break
        if mode != 1:
            for i in range(len(NonOverlappingLists)):
                for j in range(len(NonOverlappingLists[i])):
                    NonOverlappingLists[i][j] = self[NonOverlappingLists[i][j]]
        return NonOverlappingLists

    @staticmethod
    def check_consistency(temp_string):                             # should check the consistency of a string representing un invariant - old function
        init = temp_string[0]
        clos = temp_string[len(temp_string) - 1]

        relist = re.split('[\(\)⟨⟩|\]\[]', temp_string)
        relist = filter(None, relist)

        # consistency of s_ijk "Δ", "Ω", "Π"
        if init in ["s", "S", "Δ", "Ω", "Π", "δ"]:
            if temp_string[0] == "δ" and temp_string[1] == "5":
                return
            elif temp_string[1] == "_":
                return
            else:
                myException("Expected \'_\' after \'s\' in s_ijk or d_ijk")
                return

        # consitency of first and last character
        if init != "⟨" and init != "[":
            myException("Expected opening \'⟩\' or \']\'. Found \'{}\'".format(init))
            return

        # consistency of opening and closing brackets wrt length of string
        alte = (-1) ** len(relist)
        if alte == 1 and init == "⟨" and clos != "⟩":
            myException("Expected closing \'⟩\'. Found \'{}\'".format(clos))
        elif alte == -1 and init == "⟨" and clos != "]":
            myException("Expected closing \']\'. Found \'{}\'".format(clos))
        elif alte == 1 and init == "[" and clos != "]":
            myException("Expected closing \']\'. Found \'{}\'".format(clos))
        elif alte == -1 and init == "[" and clos != "⟩":
            myException("Expected closing \']\'. Found \'{}\'".format(clos))

    @property
    def total_mom(self):
        """Total momentum of the given phase space."""
        TotalMomentum = [0j, 0, 0, 0]
        for oParticle in self:
            TotalMomentum += oParticle.four_mom
        return TotalMomentum

    def print_four_momenta(self):
        print("")
        print("{}-- Four Momenta ---{}".format(Hyphens, Hyphens))
        i = 1
        for oParticle in self:
            print "Particle {} four mom. is: ".format(i)
            print vec_to_str(oParticle.four_mom)
            i = i + 1
        print("{}-------------------{}".format(Hyphens, Hyphens))
        print("Total four momentum  is {}".format(vec_to_str(self.total_mom)))
        print("{}-------------------{}".format(Hyphens, Hyphens))

    def print_r_sp_d(self):
        print("")
        print("{}- Right Spinors D -{}".format(Hyphens, Hyphens))
        i = 1
        for oParticle in self:
            print("Particle {} right sp. is {}".format(i, oParticle.r_sp_d))
            i = i + 1
        print("{}-------------------{}".format(Hyphens, Hyphens))

    def print_l_sp_d(self):
        print("")
        print("{}- Left Spinors D -{}".format(Hyphens, Hyphens))
        i = 1
        for oParticle in self:
            print("Particle {} left sp. is {}".format(i, oParticle.l_sp_d))
            i = i + 1
        print("{}-------------------{}".format(Hyphens, Hyphens))

    def _r_sp_d_for_mathematica(self):
        msg = ""
        i = 1
        for oParticle in self:
            a = oParticle.r_sp_d[0, 0]
            a_real = repr(a.real)[5:-1]
            a_imag = repr(a.imag)[5:-1]
            a = (a_real + "+" + a_imag + "I").replace("e", "*^")
            b = oParticle.r_sp_d[1, 0]
            b_real = repr(b.real)[5:-1]
            b_imag = repr(b.imag)[5:-1]
            b = (b_real + "+" + b_imag + "I").replace("e", "*^")
            msg += 'Subscript[\[Lambda], ' + str(i) + ',1] = ' + a + ";\n"
            msg += 'Subscript[\[Lambda], ' + str(i) + ',2] = ' + b + ";\n"
            i = i + 1
        return msg

    def _l_sp_d_for_mathematica(self):
        msg = ""
        i = 1
        for oParticle in self:
            a = oParticle.l_sp_d[0, 0]
            a_real = repr(a.real)[5:-1]
            a_imag = repr(a.imag)[5:-1]
            a = (a_real + "+" + a_imag + "I").replace("e", "*^")
            b = oParticle.l_sp_d[0, 1]
            b_real = repr(b.real)[5:-1]
            b_imag = repr(b.imag)[5:-1]
            b = (b_real + "+" + b_imag + "I").replace("e", "*^")
            msg += "Subscript[\!\(\*OverscriptBox[\(\[Lambda]\), \(_\)]\), " + str(i) + ",1] = " + a + ";\n"
            msg += "Subscript[\!\(\*OverscriptBox[\(\[Lambda]\), \(_\)]\), " + str(i) + ",2] = " + b + ";\n"
            i = i + 1
        return msg

    def four_momenta_for_mathematica(self, as_spinors=False):
        msg = ""
        if as_spinors is False:
            for i, iParticle in enumerate(self):
                P0 = (repr(iParticle.four_mom[0].real) + "+" + repr(iParticle.four_mom[0].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                P1 = (repr(iParticle.four_mom[1].real) + "+" + repr(iParticle.four_mom[1].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                P2 = (repr(iParticle.four_mom[2].real) + "+" + repr(iParticle.four_mom[2].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                P3 = (repr(iParticle.four_mom[3].real) + "+" + repr(iParticle.four_mom[3].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                msg += "DeclareSpinorMomentum[{ind}, [[{P0}, {P1}, {P2}, {P3}]]]".format(
                    ind=i + 1, P0=P0, P1=P1, P2=P2, P3=P3).replace("[[", "{").replace("]]", "}").replace("+-", "-").replace("gmpTools.", "") + "\n"
            msg = msg[:-1]
            return msg
        elif as_spinors is True:
            for i, iParticle in enumerate(self):
                La0 = (repr(iParticle.r_sp_d[0, 0].real) + "+" + repr(iParticle.r_sp_d[0, 0].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                La1 = (repr(iParticle.r_sp_d[1, 0].real) + "+" + repr(iParticle.r_sp_d[1, 0].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                Lat0 = (repr(iParticle.l_sp_d[0, 0].real) + "+" + repr(iParticle.l_sp_d[0, 0].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                Lat1 = (repr(iParticle.l_sp_d[0, 1].real) + "+" + repr(iParticle.l_sp_d[0, 1].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                msg += "DeclareSpinorMomentum[Sp[{ind}], [[[[{La0}]], [[{La1}]]]], [[[[{Lat0}, {Lat1}]]]]]".format(
                    ind=i + 1, La0=La0, La1=La1, Lat0=Lat0, Lat1=Lat1).replace("[[", "{").replace("]]", "}").replace("+-", "-").replace("gmpTools.", "") + "\n"
            msg = msg[:-1]
            return msg

    def save_phase_space_point(self, invariant=""):
        if invariant != "":
            FileName = invariant.replace("⟨", "A").replace("⟩", "A").replace("[", "S").replace("]", "S").replace("|", "V")
            FileName = FileName + "1.m"
        else:
            FileName = "NoLimit1.m"
        PWD = os.getcwd()
        path = PWD + "/phase_space_cache/"
        if not os.path.exists(path):
            os.makedirs(path)
        path += FileName
        if os.path.exists(path):
            FileName = FileName[:-2] + "2.m"
        path = PWD + "/phase_space_cache/" + FileName
        with open(path, "w") as oFile:
            oFile.write(self._r_sp_d_for_mathematica())
            oFile.write(self._l_sp_d_for_mathematica())

    @staticmethod
    def _four_mom_to_r2_sp_bar(temp_four_mom):                      # this is (P^mu)_alpha,alpha_dot = |A⟩[A|
        p_lowered_index = np.dot(MinkowskiMetric, temp_four_mom)
        p_lowered_index = np.transpose(p_lowered_index)
        r2_s_b = np.array([[0j, 0j], [0j, 0j]])
        for i in range(4):
            r2_s_b = r2_s_b + np.dot(Pauli_bar[i], p_lowered_index[i])
        return r2_s_b

    @staticmethod
    def _four_mom_to_r2_sp(temp_four_mom):                          # this is (P^mu)^alpha_dot,alpha = |A]⟨A|
        p_lowered_index = np.dot(MinkowskiMetric, temp_four_mom)
        p_lowered_index = np.transpose(p_lowered_index)
        r2_s = np.array([[0j, 0j], [0j, 0j]])
        for i in range(4):
            r2_s = r2_s + np.dot(Pauli[i], p_lowered_index[i])
        return r2_s


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def phase_space_points(multiplicity=None, nbr_points=None, small_invs=None, small_invs_exps=None):
    """Returns phase space points (Particles objects) of given multiplicity in collinear limit described by small_invs & small_invs_exps."""
    time_start = time.time()
    lParticles = mapThreads(phase_space_point, multiplicity, small_invs, small_invs_exps, range(nbr_points), UseParallelisation=settings.UseParallelisation, Cores=settings.Cores)
    time_end = time.time()
    seconds = time_end - time_start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    sys.stdout.write("\rGenrated %i phase space points in %d:%02d:%02d.                                            " % (nbr_points, h, m, s))
    sys.stdout.flush()
    return lParticles


def phase_space_point(multiplicity, small_invs, small_invs_exps, nbr_point):
    oParticles = Particles(multiplicity, seed=nbr_point)
    if small_invs is None or len(small_invs) == 0:
        oParticles.fix_mom_cons()
    elif len(small_invs) == 1 and len(small_invs_exps) == 1:
        oParticles.set(small_invs[0], 10 ** -int(0.80 * accuracy() / 5))
    elif len(small_invs) == 2 and len(small_invs_exps) == 2:
        oParticles.set_pair(small_invs[0], 10 ** -int(2 * 0.80 * accuracy() / 5), small_invs[1], 10 ** -int(0.80 * accuracy() / 5))
    else:
        raise Exception("Bad format for small_invs and small_invs_exps in phase_space_points.")
    return oParticles
