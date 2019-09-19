#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   ___          _   _    _          ___                     _
#  | _ \__ _ _ _| |_(_)__| |___ ___ / __|___ _ __  _ __ _  _| |_ ___
#  |  _/ _` | '_|  _| / _| / -_|_-<| (__/ _ \ '  \| '_ \ || |  _/ -_)
#  |_| \__,_|_|  \__|_\__|_\___/__(_)___\___/_|_|_| .__/\_,_|\__\___|
#                                                 |_|

# Author: Giuseppe


from __future__ import unicode_literals

import sys
import numpy
import re
import mpmath

from antares.core.tools import MinkowskiMetric, pSijk, pd5, pDijk, pOijk, pPijk, pA2, pS2, pNB, ptr5

mpmath.mp.dps = 300


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_Compute:

    def ldot(self, A, B):
        """Lorentz dot product: P_A^μ * η_μν * P_B^ν."""
        p_lowered_index = numpy.dot(MinkowskiMetric, self[B].four_mom)
        p_lowered_index = numpy.transpose(p_lowered_index)
        return numpy.dot(self[A].four_mom, p_lowered_index)

    def ep(self, i, j):
        if self.helconf[i - 1] == "+":
            # ε⁺ᵢ⋅pⱼ = ⟨q|j|i] / √2⟨qi⟩
            return (numpy.dot(numpy.dot(self.oRefVec.r_sp_u, self[j].r2_sp_b), self[i].l_sp_u) /
                    (mpmath.sqrt(2) * numpy.dot(self.oRefVec.r_sp_u, self[i].r_sp_d)))[0][0]
        elif self.helconf[i - 1] == "-":
            # ε⁻ᵢ⋅pⱼ = ⟨i|j|q] / √2[iq]
            return (numpy.dot(numpy.dot(self[i].r_sp_u, self[j].r2_sp_b), self.oRefVec.l_sp_u) /
                    (mpmath.sqrt(2) * numpy.dot(self[i].l_sp_d, self.oRefVec.l_sp_u)))[0][0]

    def pe(self, i, j):
        return self.ep(j, i)

    def ee(self, i, j):
        if self.helconf[i - 1] == self.helconf[j - 1]:
            return 0
        elif self.helconf[i - 1] == "-":
            # ε⁻ᵢ⋅ε⁺ⱼ = ⟨i|q|j] / ⟨j|q|i]
            return - (numpy.dot(numpy.dot(self[i].r_sp_u, self.oRefVec.r2_sp_b), self[j].l_sp_u) /
                      (numpy.dot(numpy.dot(self[j].r_sp_u, self.oRefVec.r2_sp_b), self[i].l_sp_u)))[0][0]
        else:
            return self.ee(j, i)

    def compute(self, temp_string):
        """Computes spinor strings.\n
        Available variables: ⟨a|b⟩, [a|b], ⟨a|b+c|d], ⟨a|b+c|d+e|f], ..., s_ijk, Δ_ijk, Ω_ijk, Π_ijk, tr5_ijkl"""
        self.check_consistency(temp_string)                         # Check consistency of string

        if ptr5.findall(temp_string) != []:                         # tr5_ijkl [i|j|k|l|i⟩ - ⟨i|j|k|l|i]
            ijkl = map(int, ptr5.findall(temp_string)[0])
            return (self.compute("[{a}|{b}|{c}|{d}|{a}⟩".format(a=ijkl[0], b=ijkl[1], c=ijkl[2], d=ijkl[3])) -
                    self.compute("⟨{a}|{b}|{c}|{d}|{a}]".format(a=ijkl[0], b=ijkl[1], c=ijkl[2], d=ijkl[3])))

        if pOijk.findall(temp_string) != []:                        # Ω_ijk
            ijk = map(int, pOijk.findall(temp_string)[0])
            nol = self.ijk_to_3NonOverlappingLists(ijk)
            Omega = (2 * self.compute("s_" + "".join(map(unicode, nol[2]))) * self.compute("s_" + "".join(map(unicode, nol[1]))) -
                     (self.compute("s_" + "".join(map(unicode, nol[2]))) + self.compute("s_" + "".join(map(unicode, nol[1]))) -
                      self.compute("s_" + "".join(map(unicode, nol[0])))) * self.compute("s_" + "".join(map(unicode, nol[2] + [nol[0][0]]))))
            return Omega

        if pPijk.findall(temp_string) != []:                        # Π_ijk, eg: Π_351 = s_123-s124
            ijk = map(int, pPijk.findall(temp_string)[0])
            nol = self.ijk_to_3NonOverlappingLists(ijk)
            Pi = (self.compute("s_" + "".join(map(unicode, nol[2] + [nol[0][0]]))) - self.compute("s_" + "".join(map(unicode, nol[2] + [nol[0][1]]))))
            return Pi

        if pDijk.findall(temp_string) != []:                        # Δ_ijk
            ijk = map(int, pDijk.findall(temp_string)[0])
            temp_oParticles = self.ijk_to_3Ks(ijk)
            Delta = temp_oParticles.ldot(1, 2)**2 - temp_oParticles.ldot(1, 1) * temp_oParticles.ldot(2, 2)
            return Delta

        if pd5.findall(temp_string) != []:
            return (2 * self.compute("s_12") * self.compute("s_23") * self.compute("s_34") * self.compute("s_45") +
                    2 * self.compute("s_12") * self.compute("s_23") * self.compute("s_34") * self.compute("s_51") +
                    2 * self.compute("s_12") * self.compute("s_23") * self.compute("s_45") * self.compute("s_51") -
                    2 * self.compute("s_12") * self.compute("s_23") * self.compute("s_23") * self.compute("s_34") +
                    2 * self.compute("s_12") * self.compute("s_34") * self.compute("s_45") * self.compute("s_51") -
                    2 * self.compute("s_12") * self.compute("s_45") * self.compute("s_51") * self.compute("s_51") -
                    2 * self.compute("s_12") * self.compute("s_12") * self.compute("s_23") * self.compute("s_51") +
                    1 * self.compute("s_12") * self.compute("s_12") * self.compute("s_23") * self.compute("s_23") +
                    1 * self.compute("s_12") * self.compute("s_12") * self.compute("s_51") * self.compute("s_51") +
                    2 * self.compute("s_23") * self.compute("s_34") * self.compute("s_45") * self.compute("s_51") -
                    2 * self.compute("s_23") * self.compute("s_34") * self.compute("s_34") * self.compute("s_45") +
                    1 * self.compute("s_23") * self.compute("s_23") * self.compute("s_34") * self.compute("s_34") -
                    2 * self.compute("s_34") * self.compute("s_45") * self.compute("s_45") * self.compute("s_51") +
                    1 * self.compute("s_34") * self.compute("s_34") * self.compute("s_45") * self.compute("s_45") +
                    1 * self.compute("s_45") * self.compute("s_45") * self.compute("s_51") * self.compute("s_51"))

        elif pSijk.findall(temp_string) != []:                      # S_ijk...
            ijk = map(int, pSijk.findall(temp_string)[0])
            s = 0
            for i in range(len(ijk)):
                for j in range(i + 1, len(ijk)):
                    s = s + 2 * self.ldot(ijk[i], ijk[j])
            return s

        elif pA2.findall(temp_string) != []:                        # ⟨A|B⟩ -- contraction is up -> down : lambda[A]^alpha.lambda[B]_alpha
            A, B = map(int, pA2.findall(temp_string)[0])
            return numpy.dot(self[A].r_sp_u, self[B].r_sp_d)[0, 0]

        elif pS2.findall(temp_string) != []:                        # [A|B] -- contraction is down -> up : lambda_bar[A]_alpha_dot.lambda_bar[B]^alpha_dot
            A, B = map(int, pS2.findall(temp_string)[0])
            return numpy.dot(self[A].l_sp_d, self[B].l_sp_u)[0, 0]

        elif pNB.findall(temp_string) != []:                        # ⟨A|(B+C+..)..|D]
            abcd = pNB.search(temp_string)
            a = int(abcd.group('start'))
            bc = abcd.group('middle')
            d = int(abcd.group('end'))
            bc = re.split('[\)|\|]', bc)
            bc = [entry.replace('|', '') for entry in bc]
            bc = [entry.replace('(', '') for entry in bc]
            bc = [entry for entry in bc if entry != '']
            a_or_s = temp_string[0]
            if a_or_s == '⟨':
                result = self[a].r_sp_u
            elif a_or_s == '[':
                result = self[a].l_sp_d
            else:
                print "Critical error: string must start with ⟨ or [."
                sys.exit('Invalid string in compute.')
            for i in range(len(bc)):
                comb_mom = re.sub(r'(\d)', r'self[\1].four_mom', bc[i])
                comb_mom = eval(comb_mom)
                if a_or_s == "⟨":
                    result = numpy.dot(result, self._four_mom_to_r2_sp_bar(comb_mom))
                    a_or_s = "["                                    # needs to alternate
                elif a_or_s == "[":
                    result = numpy.dot(result, self._four_mom_to_r2_sp(comb_mom))
                    a_or_s = "⟨"                                    # needs to alternate
            if a_or_s == "⟨":
                result = numpy.dot(result, self[d].r_sp_d)
            elif a_or_s == "[":
                result = numpy.dot(result, self[d].l_sp_u)
            return result[0][0]
        else:
            print "Critical error: string {} is not implemented.".format(temp_string)
            sys.exit('Invalid string in compute.')
