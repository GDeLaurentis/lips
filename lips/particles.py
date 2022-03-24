#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   ___          _   _    _
#  | _ \__ _ _ _| |_(_)__| |___ ___
#  |  _/ _` | '_|  _| / _| / -_|_-<
#  |_| \__,_|_|  \__|_\__|_\___/__/

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
import random
import re
import os
import copy
import itertools
import sympy

from .tools import MinkowskiMetric, flatten, pA2, pS2, pNB, myException, indexing_decorator
from .particle import Particle
from .particles_compute import Particles_Compute
from .particles_eval import Particles_Eval
from .particles_set import Particles_Set
from .particles_set_pair import Particles_SetPair


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles(Particles_Compute, Particles_Eval, Particles_Set, Particles_SetPair, list):
    """Describes the kinematics of n particles. Base one list of Particle objects."""

    def __init__(self, number_of_particles_or_particles=None, seed=None, real_momenta=False, fix_mom_cons=True):
        """Initialisation. Requires either multiplicity of phace space or list of Particle objects."""
        list.__init__(self)
        if isinstance(number_of_particles_or_particles, int):
            random.seed(seed) if seed is not None else random.seed()
            for i in range(number_of_particles_or_particles):
                self.append(Particle(real_momentum=real_momenta))
        elif isinstance(number_of_particles_or_particles, list):
            for oParticle in number_of_particles_or_particles:
                self.append(oParticle)
        elif number_of_particles_or_particles is not None:
            raise Exception("Invalid initialisation of Particles instance.")
        self.oRefVec = Particle(real_momentum=real_momenta)
        if fix_mom_cons is True and max(map(abs, flatten(self.total_mom))) > 10 ** -(0.9 * 300):
            self.fix_mom_cons(real_momenta=real_momenta)

    def __call__(self, string_expression):
        return self.compute(string_expression)

    def __eq__(self, other):
        """Checks equality of each particle in particles."""
        if type(self) == type(other):
            return all(self[i] == other[i] for i in range(1, len(self) + 1))
        else:
            return False

    def __hash__(self):
        """Hash function: hash string of concatenated momenta."""
        return hash(" ".join(flatten([list(map(str, flatten(oParticle.r2_sp))) for oParticle in self])))

    # PUBLIC METHODS

    @property
    def total_mom(self):
        """Total momentum of the given phase space as a rank two spinor."""
        return sum([oParticle.r2_sp for oParticle in self])

    @property
    def masses(self):
        """Masses of all particles in phase space."""
        return [oParticle.mass for oParticle in self]

    def randomise_all(self, real_momenta=False):
        """Randomises all particles. Breaks momentum conservation."""
        for oParticle in self:
            oParticle.randomise(real_momentum=real_momenta)
        self.fix_mom_cons(real_momenta=real_momenta)

    def randomise_twistor(self):
        for i, iParticle in enumerate(self):
            iParticle.randomise_twist()
        for i, iParticle in enumerate(self):
            iParticle.comp_twist_x(self[(i + 1) % len(self) + 1])
        for i, iParticle in enumerate(self):
            iParticle.twist_x_to_mom(self[(i + 1) % len(self) + 1])

    def angles_for_squares(self):
        """Switches all angle brackets for square brackets and viceversa."""
        for oParticle in self:
            oParticle.angles_for_squares()

    def image(self, permutation_or_rule):
        """Returns the image of self under a given permutation or rule. Remember, this is a passive transformation."""
        if type(permutation_or_rule) is str:
            return copy.deepcopy(Particles(sorted(self, key=lambda x: permutation_or_rule[self.index(x)]), fix_mom_cons=False))
        else:
            assert type(permutation_or_rule[0]) is str and type(permutation_or_rule[1]) is bool
            oResParticles = self.image(permutation_or_rule[0])
            if permutation_or_rule[1] is True:
                oResParticles.angles_for_squares()
            return oResParticles

    def cluster(self, llIntegers):
        """Returns clustered particle objects according to lists of lists of integers (e.g. corners of one loop diagram)."""
        oKs = Particles(len(llIntegers), fix_mom_cons=False)
        r2_spinors = [sum([self[i].r2_sp for i in corner_as_integers]) for corner_as_integers in llIntegers]
        for i, iK in enumerate(oKs):
            iK.r2_sp = r2_spinors[i]
        return oKs

    def make_analytical_d(self, indepVars=None):
        """ """
        if indepVars is None:
            indepVars = tuple(numpy.zeros(4 * len(self), dtype=int))
        la = sympy.symbols('a1:{}'.format(len(self) + 1))
        lb = sympy.symbols('b1:{}'.format(len(self) + 1))
        lc = sympy.symbols('c1:{}'.format(len(self) + 1))
        ld = sympy.symbols('d1:{}'.format(len(self) + 1))
        for i, oParticle in enumerate(self):
            if indepVars[i * 4 + 0] == 0:
                oParticle._r_sp_d[0, 0] = la[i]
            if indepVars[i * 4 + 1] == 0:
                oParticle._r_sp_d[1, 0] = lb[i]
            if indepVars[i * 4 + 2] == 0:
                oParticle._l_sp_d[0, 0] = lc[i]
            if indepVars[i * 4 + 3] == 0:
                oParticle._l_sp_d[0, 1] = ld[i]
            oParticle._r_sp_d_to_r_sp_u()
            oParticle._l_sp_d_to_l_sp_u()
            oParticle._r1_sp_to_r2_sp()
            oParticle._r1_sp_to_r2_sp_b()
            try:
                oParticle._r2_sp_b_to_four_momentum()
                oParticle._four_mom_to_four_mom_d()
            except (TypeError, SystemError):
                oParticle._four_mom = None
                oParticle._four_mom_d = None

    def fix_mom_cons(self, A=0, B=0, real_momenta=False, axis=1):   # using real momenta changes both |⟩ and |] of A & B
        """Fixes momentum conservation using particles A and B."""
        if A == 0 and B == 0:                                       # defaults to random particles to fix mom cons
            A, B = random.sample(range(1, len(self) + 1), 2)
            self.fix_mom_cons(A, B, real_momenta, axis)

        elif A != 0 and B != 0 and real_momenta is True:            # the following results in real momenta, but changes both |⟩ and |] of A & B
            K = sum([self[k].four_mom for k in range(1, len(self) + 1) if k not in [A, B]])
            self[A].four_mom = numpy.array([- numpy.dot(K, numpy.dot(MinkowskiMetric, K)) / (2 * (K[0] - K[axis])) if k in [0, axis] else 0 for k in range(4)])
            self[B].four_mom = - self[A].four_mom - K

        elif A != 0 and B != 0:                                     # the following results in complex momenta, it changes |A⟩&|B⟩ (axis=1) or |A]&|B] (axis=2)
            K = sum([self[k].r2_sp for k in range(1, len(self) + 1) if k not in [A, B]])
            if axis == 1:                                           # change |A⟩ and |B⟩
                self[A].r_sp_u = numpy.dot(self[B].l_sp_d, K) / self.compute("[%d|%d]" % (A, B))
                self[B].r_sp_u = - numpy.dot(self[A].l_sp_d, K) / self.compute("[%d|%d]" % (A, B))
            else:                                                   # change |A] and |B]
                self[A].l_sp_u = - numpy.dot(K, self[B].r_sp_d) / self.compute("⟨%d|%d⟩" % (A, B))
                self[B].l_sp_u = numpy.dot(K, self[A].r_sp_d) / self.compute("⟨%d|%d⟩" % (A, B))

    def momentum_conservation_check(self, silent=True):
        """Returns true if momentum is conserved."""
        mom_violation = 0
        for i in range(4):
            if abs(flatten(self.total_mom)[i]) > mom_violation:
                mom_violation = abs(flatten(self.total_mom)[i])
        if silent is False:
            print("The largest momentum violation is {}".format(float(mom_violation)))
        if mom_violation > 10 ** -(0.9 * 300):
            myException("Momentum conservation violation.")
            return False
        return True

    def onshell_relation_check(self, silent=True):
        """Returns true if all on-shell relations are satisfied."""
        onshell_violation = 0
        for i in range(1, len(self) + 1):
            if abs(self.ldot(i, i)) > onshell_violation:
                onshell_violation = abs(self.ldot(i, i))
        if silent is False:
            print("The largest on shell violation is {}".format(float(onshell_violation)))
        if onshell_violation > 10 ** -(0.9 * 300):
            myException("Onshell relation violation.")
            return False
        return True

    def phasespace_consistency_check(self, invariants=[], silent=True):
        """Runs momentum and onshell checks. Looks for outliers in phase space. Returns: mom_cons, on_shell, big_outliers, small_outliers."""

        if invariants == []:
            _invars = (["⟨{}|{}⟩".format(i, j) for (i, j) in itertools.combinations(range(1, len(self) + 1), 2)] +
                       ["[{}|{}]".format(i, j) for (i, j) in itertools.combinations(range(1, len(self) + 1), 2)])
        else:
            _invars = [invariant for invariant in invariants]

        if silent is False:
            print("Consistency check:")
            # print("{} Consistency check {}".format(Hyphens, Hyphens))

        mom_cons = self.momentum_conservation_check(silent)         # momentum conservation violation
        on_shell = self.onshell_relation_check(silent)              # onshell violation

        values = []                                                 # smallest and biggest invariants
        for _invar in _invars:
            values += [abs(self.compute(_invar))]
            if "n" in str(values[len(values) - 1]) and silent is False:
                print("not a number!", values[len(values) - 1], "invariant", _invar)
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
                    print("{} = {}".format(_invars[i], float(values[i])))
            if values[i] > 0.0001:
                break
        if silent is False:
            print("...")
        for i in range(len(_invars)):
            if values[i] > 100000:
                myException("Outliers are big!")
                big_outliers += [_invars[i]]
                big_outliers_values += [values[i]]
                if silent is False:
                    print("{} = {}".format(_invars[i], float(values[i])))
        if silent is False:
            pass
            # print("{}-------------------{}".format(Hyphens, Hyphens))
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

    def _can_fix_mom_cons(self, t_s1, t_s2):
        """Checks if momentum conservation can be restored, without spoiling the collinear limit."""
        # If possible returns how to fix mom cons (tuple & axis), otherwise returns false.
        if ((pA2.findall(t_s1) != [] or pS2.findall(t_s1) != []) and pNB.findall(t_s2) != []):
            if pA2.findall(t_s1) != []:
                ab = list(map(int, list(pA2.findall(t_s1)[0])))
            elif pS2.findall(t_s1) != []:
                ab = list(map(int, list(pS2.findall(t_s1)[0])))
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
        start = start + str(lNBs) + "|"
        end = "|" + str(lNBe) + end
        middle = "".join(string + "|" for string in ["(" + "".join(str(entry) + "+" for entry in item)[:-1] + ")" for item in lNBms])
        middle = middle[:-1]
        t_s_new = start + middle + end
        return t_s_new

    @staticmethod
    def _get_lNB(temp_string):                                      # usage: lNB, lNBs, lNBms, lNBe = _get_lNB(temp_string)
        lNB = list(pNB.findall(temp_string)[0])
        lNB[1] = [entry.replace("(", "").replace(")", "").split("+") for entry in lNB[1].split("|")]
        lNBs, lNBms, lNBe = int(lNB[0]), [list(map(int, entry)) for entry in lNB[1]], int(lNB[2])
        lNB = list(map(int, [lNBs] + [entry for sublist in lNB[1] for entry in sublist] + [lNBe]))
        return lNB, lNBs, lNBms, lNBe

    def _complementary(self, temp_list):                            # returns the list obtained by using momentum conservation
        temp_list = flatten(temp_list)
        if type(temp_list) == list:                                 # make sure it is a set (no double entries)
            temp_list = set(temp_list)
        original_type = type(list(temp_list)[0])
        if type(list(temp_list)[0]) is not int:                         # make sure entries are integers (representing particle #)
            temp_list = set(map(int, temp_list))
        temp_list = list(temp_list)
        n = len(self)
        c_list = []
        for i in range(1, n + 1):
            c_list += [i]
        for element in temp_list:
            if element in c_list:
                c_list.remove(element)
        c_list = list(map(original_type, c_list))
        return c_list

    def ijk_to_3Ks(self, ijk):                                      # this method is used for Delta computation and setting
        return self.cluster(self.ijk_to_3NonOverlappingLists(ijk))

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

        relist = re.split(r'[\(\)⟨⟩|\]\[]', temp_string)
        relist = list(filter(None, relist))

        # consistency of s_ijk "Δ", "Ω", "Π"
        if init in ["s", "S", "Δ", "Ω", "Π", "δ"] or temp_string[0:3] == "tr5":
            if temp_string[0] == "δ" and temp_string[1] == "5":
                return
            elif temp_string[1] == "_" or (temp_string[0:3] == "tr5" and temp_string[4] == "_"):
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
                    ind=i + 1, P0=P0, P1=P1, P2=P2, P3=P3).replace("[[", "{").replace("]]", "}").replace("+-", "-") + "\n"
            msg = msg[:-1]
            return msg
        elif as_spinors is True:
            for i, iParticle in enumerate(self):
                La0 = (repr(iParticle.r_sp_d[0, 0].real) + "+" + repr(iParticle.r_sp_d[0, 0].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                La1 = (repr(iParticle.r_sp_d[1, 0].real) + "+" + repr(iParticle.r_sp_d[1, 0].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                Lat0 = (repr(iParticle.l_sp_d[0, 0].real) + "+" + repr(iParticle.l_sp_d[0, 0].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                Lat1 = (repr(iParticle.l_sp_d[0, 1].real) + "+" + repr(iParticle.l_sp_d[0, 1].imag) + "I").replace("e", "*^").replace("RGMP(", "").replace(")", "")
                msg += "DeclareSpinorMomentum[Sp[{ind}], [[[[{La0}]], [[{La1}]]]], [[[[{Lat0}, {Lat1}]]]]]".format(
                    ind=i + 1, La0=La0, La1=La1, Lat0=Lat0, Lat1=Lat1).replace("[[", "{").replace("]]", "}").replace("+-", "-") + "\n"
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
