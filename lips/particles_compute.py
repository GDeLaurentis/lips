#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   ___          _   _    _          ___                     _
#  | _ \__ _ _ _| |_(_)__| |___ ___ / __|___ _ __  _ __ _  _| |_ ___
#  |  _/ _` | '_|  _| / _| / -_|_-<| (__/ _ \ '  \| '_ \ || |  _/ -_)
#  |_| \__,_|_|  \__|_\__|_\___/__(_)___\___/_|_|_| .__/\_,_|\__\___|
#                                                 |_|

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
import re
import mpmath
import sys

from .tools import pSijk, pd5, pDijk, pOijk, pPijk, pA2, pAu, pAd, pS2, pSu, pSd, pNB, ptr5

if sys.version_info[0] > 2:
    unicode = str

mpmath.mp.dps = 300


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_Compute:

    def ldot(self, A, B):
        """Lorentz dot product: 2 trace(P^{α̇α}P̅\u0305_{αα̇}) = P_A^μ * η_μν * P_B^ν."""
        return numpy.trace(numpy.dot(self[A].r2_sp, self[B].r2_sp_b)) / 2

    def ep(self, i, j):
        """Contraction of polarization tensor with four momentum. Requires .helconf property to be set."""
        if self.helconf[i - 1] in ["+", "p"]:
            # ε⁺ᵢ⋅pⱼ = ⟨q|j|i] / √2⟨qi⟩
            return (numpy.dot(numpy.dot(self.oRefVec.r_sp_u, self[j].r2_sp_b), self[i].l_sp_u) /
                    (mpmath.sqrt(2) * numpy.dot(self.oRefVec.r_sp_u, self[i].r_sp_d)))[0][0]
        elif self.helconf[i - 1] in ["-", "m"]:
            # ε⁻ᵢ⋅pⱼ = ⟨i|j|q] / √2[iq]
            return (numpy.dot(numpy.dot(self[i].r_sp_u, self[j].r2_sp_b), self.oRefVec.l_sp_u) /
                    (mpmath.sqrt(2) * numpy.dot(self[i].l_sp_d, self.oRefVec.l_sp_u)))[0][0]

    def pe(self, i, j):
        """Contraction of four momentum with polarization tensor. Requires .helconf property to be set."""
        return self.ep(j, i)

    def ee(self, i, j):
        """Contraction of two polarization tensors. Requires .helconf property to be set."""
        if self.helconf[i - 1] == self.helconf[j - 1]:
            return 0
        elif self.helconf[i - 1] in ["-", "m"]:
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
            ijkl = list(map(int, ptr5.findall(temp_string)[0]))
            return (self.compute("[{a}|{b}|{c}|{d}|{a}⟩".format(a=ijkl[0], b=ijkl[1], c=ijkl[2], d=ijkl[3])) -
                    self.compute("⟨{a}|{b}|{c}|{d}|{a}]".format(a=ijkl[0], b=ijkl[1], c=ijkl[2], d=ijkl[3])))

        if pOijk.findall(temp_string) != []:                        # Ω_ijk
            ijk = list(map(int, pOijk.findall(temp_string)[0]))
            nol = self.ijk_to_3NonOverlappingLists(ijk)
            Omega = (2 * self.compute("s_" + "".join(map(unicode, nol[2]))) * self.compute("s_" + "".join(map(unicode, nol[1]))) -
                     (self.compute("s_" + "".join(map(unicode, nol[2]))) + self.compute("s_" + "".join(map(unicode, nol[1]))) -
                      self.compute("s_" + "".join(map(unicode, nol[0])))) * self.compute("s_" + "".join(map(unicode, nol[2] + [nol[0][0]]))))
            return Omega

        if pPijk.findall(temp_string) != []:                        # Π_ijk, eg: Π_351 = s_123-s124
            ijk = list(map(int, pPijk.findall(temp_string)[0]))
            nol = self.ijk_to_3NonOverlappingLists(ijk)
            Pi = (self.compute("s_" + "".join(map(unicode, nol[2] + [nol[0][0]]))) - self.compute("s_" + "".join(map(unicode, nol[2] + [nol[0][1]]))))
            return Pi

        if pDijk.findall(temp_string) != []:                        # Δ_ijk or Δ_ij|kl|lm
            match_list = pDijk.findall(temp_string)[0]
            if match_list[0] == '':
                NonOverlappingLists = [list(map(int, corner)) for corner in match_list[1:]]
            else:
                NonOverlappingLists = self.ijk_to_3NonOverlappingLists(list(map(int, match_list[0])))
            temp_oParticles = self.cluster(NonOverlappingLists)
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
            ijk = list(map(int, pSijk.findall(temp_string)[0]))
            s = 0
            for i in range(len(ijk)):
                for j in range(i + 1, len(ijk)):
                    s = s + 2 * self.ldot(ijk[i], ijk[j])
            return s

        elif pA2.findall(temp_string) != []:                        # ⟨A|B⟩ -- contraction is up -> down : lambda[A]^alpha.lambda[B]_alpha
            A, B = map(int, pA2.findall(temp_string)[0])
            return numpy.dot(self[A].r_sp_u, self[B].r_sp_d)[0, 0]

        elif pAu.findall(temp_string) != []:                        # ⟨A|
            A = int(pAu.findall(temp_string)[0])
            return self[A].r_sp_u

        elif pAd.findall(temp_string) != []:                        # |B⟩
            B = int(pAd.findall(temp_string)[0])
            return self[B].r_sp_d

        elif pS2.findall(temp_string) != []:                        # [A|B] -- contraction is down -> up : lambda_bar[A]_alpha_dot.lambda_bar[B]^alpha_dot
            A, B = map(int, pS2.findall(temp_string)[0])
            return numpy.dot(self[A].l_sp_d, self[B].l_sp_u)[0, 0]

        elif pSu.findall(temp_string) != []:                        # |A]
            A = int(pSu.findall(temp_string)[0])
            return self[A].l_sp_u

        elif pSd.findall(temp_string) != []:                        # [B|
            B = int(pSd.findall(temp_string)[0])
            return self[B].l_sp_d

        elif pNB.findall(temp_string) != []:                        # ⟨A|(B+C+..)..|D]
            abcd = pNB.search(temp_string)
            a = int(abcd.group('start'))
            bc = abcd.group('middle').replace("(", "").replace(")", "").split("|")
            d = int(abcd.group('end'))

            if temp_string[0] == "⟨":
                middle = ["(" + re.sub(r'(\d+)', r'self[\1].r2_sp_b', entry) + ")" if i % 2 == 0 else
                          "(" + re.sub(r'(\d+)', r'self[\1].r2_sp', entry) + ")" for i, entry in enumerate(bc)]
                middle = ".dot(".join(middle) + ")" * (len(middle) - 1)
                result = self[a].r_sp_u.dot(eval(middle))
            else:
                middle = ["(" + re.sub(r'(\d+)', r'self[\1].r2_sp', entry) + ")" if i % 2 == 0 else
                          "(" + re.sub(r'(\d+)', r'self[\1].r2_sp_b', entry) + ")" for i, entry in enumerate(bc)]
                middle = ".dot(".join(middle) + ")" * (len(middle) - 1)
                result = self[a].l_sp_d.dot(eval(middle))

            if temp_string[-1] == "⟩":
                return result.dot(self[d].r_sp_d)[0][0]
            else:
                return result.dot(self[d].l_sp_u)[0][0]

        else:
            return self._eval(temp_string)
