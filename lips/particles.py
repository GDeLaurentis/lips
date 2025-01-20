# -*- coding: utf-8 -*-

#   ___          _   _    _
#  | _ \__ _ _ _| |_(_)__| |___ ___
#  |  _/ _` | '_|  _| / _| / -_|_-<
#  |_| \__,_|_|  \__|_\__|_\___/__/

# Author: Giuseppe

import numpy
import random
import re
import os
import copy
import itertools
import mpmath
import sympy

from sympy import NotInvertible

from syngular import Field
from pyadic import PAdic

from .tools import MinkowskiMetric, flatten, subs_dict, pNB, myException, indexing_decorator, pAu, pAd, pSu, pSd, pMVar
from .particle import Particle
from .particles_compute import Particles_Compute
from .particles_eval import Particles_Eval
from .hardcoded_limits.particles_set import Particles_Set
from .hardcoded_limits.particles_set_pair import Particles_SetPair
from .algebraic_geometry.particles_singular_variety import Particles_SingularVariety
from .particles_variety import Particles_Variety


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles(Particles_Compute, Particles_Eval, Particles_Set, Particles_SetPair, Particles_SingularVariety, Particles_Variety, list):
    """Describes the kinematics of n particles. Base one list of Particle objects."""

    # MAGIC METHODS

    def __init__(self, number_of_particles_or_particles=None, seed=None, real_momenta=False, field=Field('mpc', 0, 300),
                 fix_mom_cons=True, internal_masses=None):
        """Initialisation. Requires either multiplicity of phace space or list of Particle objects."""
        super().__init__()
        self.field = field
        self.seed = seed  # This should be removed
        random.seed(seed) if seed is not None else random.seed()
        # External Kinematics
        if isinstance(number_of_particles_or_particles, int):
            for i in range(number_of_particles_or_particles):
                self.append(Particle(real_momentum=real_momenta, field=field))
        elif isinstance(number_of_particles_or_particles, list):
            for oParticle in number_of_particles_or_particles:
                self.append(oParticle)
        elif number_of_particles_or_particles is not None:
            raise Exception("Invalid initialisation of Particles instance.")
        self.oRefVec = Particle(real_momentum=real_momenta, field=field)  # This should not be added by default
        if fix_mom_cons is True and max(map(abs, flatten(self.total_mom))) > field.tollerance:
            self.fix_mom_cons(real_momenta=real_momenta)
        # Internal Kinematics
        if isinstance(internal_masses, dict):
            self.internal_masses = set()
            for internal_mass, value in internal_masses.items():
                self.__setattr__(internal_mass, value)
        elif isinstance(internal_masses, (list, tuple, set)) and all(isinstance(internal_mass, str) for internal_mass in internal_masses):
            self.internal_masses = internal_masses
            for internal_mass in internal_masses:
                self.__setattr__(internal_mass, self.field.random())
        elif internal_masses is None:
            self.internal_masses = set()
        else:
            raise Exception(f"Internal masses not understood, received {internal_masses} of type {type(internal_masses)}.")

    def __setattr__(self, name, value):
        if pMVar.match(name):
            self.internal_masses.add(name)
        super().__setattr__(name, value)

    def __call__(self, string_expression):
        return self.compute(string_expression)

    def __eq__(self, other):
        """Checks equality of each particle in particles."""
        if isinstance(other, Particles):
            return all(self[i] == other[i] for i in range(1, len(self) + 1))
        else:
            return False

    def __hash__(self):
        """Hash function: hash string of concatenated momenta."""
        return hash(tuple([hash(oP) for oP in self]))
        # this breaks when only little group changes
        # return hash(" ".join(flatten([list(map(str, flatten(oParticle.r2_sp))) for oParticle in self])))

    # PUBLIC METHODS

    @property
    def total_mom(self):
        """Total momentum of the given phase space as a rank two spinor."""
        return sum([oParticle.r2_sp for oParticle in self])

    @property
    def masses(self):
        """Masses of all particles in phase space."""
        return [oParticle.mass for oParticle in self]

    @property
    def internal_masses_dict(self):
        return {key: getattr(self, key) for key in self.internal_masses}

    @property
    def multiplicity(self):
        return len(self)

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
            if not permutation_or_rule.isdigit():
                raise ValueError(f"Permutation to map phase space points should be a string of integers, got: {permutation_or_rule}.")
            oResParticles = copy.deepcopy(Particles(sorted(self, key=lambda x: permutation_or_rule[self.index(x)]),
                                                    field=self.field, fix_mom_cons=False, internal_masses=self.internal_masses_dict))
            oResParticles.oRefVec = copy.deepcopy(self.oRefVec)
            return oResParticles
        else:
            assert type(permutation_or_rule[0]) is str and type(permutation_or_rule[1]) is bool
            oResParticles = self.image(permutation_or_rule[0])
            if permutation_or_rule[1] is True:
                oResParticles.angles_for_squares()
            return oResParticles

    def copy(self):
        from .symmetries import identity
        return self.image(identity(len(self)))

    def cluster(self, llIntegers):
        """Returns clustered particle objects according to lists of lists of integers (e.g. corners of one loop diagram)."""
        drule1 = dict(zip(["s" + "".join(map(str, entry)) for entry in llIntegers], [f"s{i}" for i in range(1, len(llIntegers) + 1)]))
        drule2 = dict(zip(["s_" + "".join(map(str, entry)) for entry in llIntegers], [f"s_{i}" for i in range(1, len(llIntegers) + 1)]))
        clustered_internal_masses = {key: (subs_dict(val, drule1 | drule2) if isinstance(val, str) else val)
                                     for key, val in self.internal_masses_dict.items()}
        return Particles([sum([self[i] for i in corner_as_integers]) for corner_as_integers in llIntegers],
                         field=self.field, fix_mom_cons=False, internal_masses=clustered_internal_masses)

    def make_analytical_d(self, indepVars=None, symbols=('a', 'b', 'c', 'd')):
        """ """
        if indepVars is None:
            indepVars = tuple(numpy.zeros(4 * len(self), dtype=int))
        la = sympy.symbols(f'{symbols[0]}1:{len(self) + 1}')
        lb = sympy.symbols(f'{symbols[1]}1:{len(self) + 1}')
        lc = sympy.symbols(f'{symbols[2]}1:{len(self) + 1}')
        ld = sympy.symbols(f'{symbols[3]}1:{len(self) + 1}')
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
            # is the following really needed? it slows down the variety calculation,
            # plus no invariant should be computed from 4-momenta (use rank 2 spinor instead)
            try:
                oParticle._r2_sp_b_to_four_momentum()
                oParticle._four_mom_to_four_mom_d()
            except (ValueError, TypeError, SystemError, NotInvertible):
                oParticle._four_mom = None
                oParticle._four_mom_d = None

    def analytical_subs_d(self):
        la = sympy.symbols('a1:{}'.format(self.multiplicity + 1))
        lb = sympy.symbols('b1:{}'.format(self.multiplicity + 1))
        lc = sympy.symbols('c1:{}'.format(self.multiplicity + 1))
        ld = sympy.symbols('d1:{}'.format(self.multiplicity + 1))
        subs_dict = {}
        for i, iParticle in enumerate(self):
            subs_dict.update({la[i]: iParticle.r_sp_d[0, 0], lb[i]: iParticle.r_sp_d[1, 0]})
            subs_dict.update({lc[i]: iParticle.l_sp_d[0, 0], ld[i]: iParticle.l_sp_d[0, 1]})
        return subs_dict

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
        mom_violation = max(map(abs, flatten(self.total_mom)))
        if silent is False:
            print("The largest momentum violation is {}".format(float(mom_violation) if type(mom_violation) is mpmath.mpf else mom_violation))
        if mom_violation > self.field.tollerance:
            myException("Momentum conservation violation.")
            return False
        return True

    def onshell_relation_check(self, silent=True):
        """Returns true if all on-shell relations are satisfied."""
        onshell_violation = max(map(abs, flatten(self.masses)))
        if silent is False:
            print("The largest on shell violation is {}".format(float(onshell_violation) if type(onshell_violation) is mpmath.mpf else onshell_violation))
        if onshell_violation > self.field.tollerance:
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

        mom_cons = self.momentum_conservation_check(silent)         # momentum conservation violation
        on_shell = self.onshell_relation_check(silent)              # onshell violation

        if self.field.name == 'padic':
            threshold = PAdic(0, self.field.characteristic, 0, 1)
        elif self.field.name == 'finite field':
            threshold = 10 ** -300
        else:
            threshold = 10 ** -8

        values = []                                                 # smallest and biggest invariants
        for _invar in _invars:
            values += [abs(self.compute(_invar))]
            if isinstance(values[-1], numpy.ndarray):
                values[-1] = values[-1].max()
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
            if values[i] <= threshold:
                small_outliers += [_invars[i]]
                small_outliers_values += [values[i]]
                if silent is False:
                    print("{} = {}".format(_invars[i], float(values[i]) if type(values[i]) is mpmath.mpf else values[i]))
            if values[i] > threshold:
                break
        if silent is False:
            print("...")
        for i in range(len(_invars)):
            if values[i] >= 1 / threshold:
                myException("Outliers are big!")
                big_outliers += [_invars[i]]
                big_outliers_values += [values[i]]
                if silent is False:
                    print("{} = {}".format(_invars[i], float(values[i]) if type(values[i]) is mpmath.mpf else values[i]))
        return mom_cons, on_shell, big_outliers, small_outliers

    @property
    def spinors_are_in_field_extension(self):
        return any([oP.spinors_are_in_field_extension for oP in self])

    # BASE ONE LIST METHODS

    @indexing_decorator
    def __getitem__(self, index):
        if isinstance(index, str):
            if pAu.findall(index) != [] or pAd.findall(index) != [] or pSu.findall(index) != [] or pSd.findall(index) != []:
                return self.compute(index)
            elif re.findall(r"(\d)", index) != []:
                return self[int(re.findall(r"(\d)", index)[0])]
            else:
                raise IndexError(index)
        elif isinstance(index, slice):
            oNewParticles = Particles(list.__getitem__(self, index), field=self.field, fix_mom_cons=False)
            oNewParticles.oRefVec = self.oRefVec
            return oNewParticles
        else:
            return list.__getitem__(self, index)

    @indexing_decorator
    def __setitem__(self, index, value):
        if isinstance(index, str):
            if pAu.findall(index) != []:                        # ⟨A|
                A = int(pAu.findall(index)[0])
                self[A].r_sp_u = value
            elif pAd.findall(index) != []:                      # |B⟩
                B = int(pAd.findall(index)[0])
                self[B].r_sp_d = value
            elif pSu.findall(index) != []:                      # |A]
                A = int(pSu.findall(index)[0])
                self[A].l_sp_u = value
            elif pSd.findall(index) != []:                      # [B|
                B = int(pSd.findall(index)[0])
                self[B].l_sp_d = value
        else:
            list.__setitem__(self, index, value)

    @indexing_decorator
    def __delitem__(self, index):
        list.__delitem__(self, index)

    @indexing_decorator
    def insert(self, index, value):
        list.insert(self, index, value)

    # PRIVATE METHODS

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
        if isinstance(temp_list, list):                                 # make sure it is a set (no double entries)
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
            msg += r'Subscript[\[Lambda], ' + str(i) + ',1] = ' + a + ";\n"
            msg += r'Subscript[\[Lambda], ' + str(i) + ',2] = ' + b + ";\n"
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
            msg += r"Subscript[\!\(\*OverscriptBox[\(\[Lambda]\), \(_\)]\), " + str(i) + ",1] = " + a + ";\n"
            msg += r"Subscript[\!\(\*OverscriptBox[\(\[Lambda]\), \(_\)]\), " + str(i) + ",2] = " + b + ";\n"
            i = i + 1
        return msg

    def four_momenta_for_mathematica(self, as_spinors=False):
        msg = ""
        if as_spinors is False:
            for i, iParticle in enumerate(self):
                P0 = (repr(iParticle.four_mom[0].real) + "+" + repr(iParticle.four_mom[0].imag) + "I").replace("e", "*^").replace("mpf(", "").replace(")", "")
                P1 = (repr(iParticle.four_mom[1].real) + "+" + repr(iParticle.four_mom[1].imag) + "I").replace("e", "*^").replace("mpf(", "").replace(")", "")
                P2 = (repr(iParticle.four_mom[2].real) + "+" + repr(iParticle.four_mom[2].imag) + "I").replace("e", "*^").replace("mpf(", "").replace(")", "")
                P3 = (repr(iParticle.four_mom[3].real) + "+" + repr(iParticle.four_mom[3].imag) + "I").replace("e", "*^").replace("mpf(", "").replace(")", "")
                msg += "DeclareSpinorMomentum[{ind}, [[SetPrecision[{P0}, {PR}], SetPrecision[{P1}, {PR}], SetPrecision[{P2}, {PR}], SetPrecision[{P3}, {PR}] ]]]".format(
                    ind=i + 1, P0=P0, P1=P1, P2=P2, P3=P3, PR=mpmath.mp.dps).replace("[[", "{").replace("]]", "}").replace("+-", "-").replace("'", "") + "\n"
            msg = msg[:-1]
            return msg
        elif as_spinors is True:
            for i, iParticle in enumerate(self):
                La0 = str(complex(iParticle.r_sp_d[0, 0])).replace("j", "I").replace("e", "*^")
                La1 = str(complex(iParticle.r_sp_d[1, 0])).replace("j", "I").replace("e", "*^")
                Lat0 = str(complex(iParticle.l_sp_d[0, 0])).replace("j", "I").replace("e", "*^")
                Lat1 = str(complex(iParticle.l_sp_d[0, 1])).replace("j", "I").replace("e", "*^")
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
