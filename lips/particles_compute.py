#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   ___          _   _    _          ___                     _
#  | _ \__ _ _ _| |_(_)__| |___ ___ / __|___ _ __  _ __ _  _| |_ ___
#  |  _/ _` | '_|  _| / _| / -_|_-<| (__/ _ \ '  \| '_ \ || |  _/ -_)
#  |_| \__,_|_|  \__|_\__|_\___/__(_)___\___/_|_|_| .__/\_,_|\__\___|
#                                                 |_|

# Author: Giuseppe

import numpy
import re
import mpmath
import warnings
import functools
import operator

from .tools import pSijk, pMi, pMVar, pd5, pDijk, pOijk, pPijk, pA2, pAu, pAd, pS2, pSu, pSd, \
    pNB, pNB_open_begin, pNB_open_end, pNB_double_open, ptr5, ptr, det, rsubs_dict, bold_digits

mpmath.mp.dps = 300


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def as_scalar_if_scalar(func):
    """Turns numpy arrays with zero dimensions into 'real' scalars."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if not hasattr(res, 'shape'):
            return res
        if res.shape == ():
            return res[()]  # pops the scalar out of array(scalar) or does nothing if array has non-trivial dimensions.
        elif functools.reduce(operator.mul, res.shape) == 1:
            return res.flatten()[0]
        else:
            return res
    return wrapper


class Particles_Compute:

    def ldot(self, A, B):
        """Lorentz dot product: 1/2 trace(P^{α̇α}P̅\u0305_{αα̇}) = P_A^μ * η_μν * P_B^ν."""
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

    @as_scalar_if_scalar
    def compute(self, original_temp_string):
        """Computes spinor strings.\n
        Available variables: ⟨a|b⟩, [a|b], ⟨a|b+c|d], ⟨a|b+c|d+e|f], ..., s_ijk, Δ_ijk, Ω_ijk, Π_ijk, tr5_ijkl"""

        # Consistency of string check is left to ast parser.
        temp_string = rsubs_dict(original_temp_string, bold_digits)

        if ptr5.findall(temp_string) != []:                         # tr5_ijkl [i|j|k|l|i⟩ - ⟨i|j|k|l|i]
            abcd = ptr5.findall(temp_string)[0][0 if "_" in temp_string else 1]
            a, b, c, d = abcd.split("|") if "|" in abcd else abcd
            for _ in range(4):
                if len(a) == 1:
                    break
                a, b, c, d = b, c, d, a
            else:
                raise NotImplementedError("tr5 implementation requires at least 1 massless particle.")
            return self.compute(f"[{a}|{b}|{c}|{d}|{a}⟩") - self.compute(f"⟨{a}|{b}|{c}|{d}|{a}]")

        if ptr.findall(temp_string) != []:                          # e.g.: tr(i+j|k-l|...)
            abcd = ptr.search(temp_string)
            bc = abcd.group("middle").replace("(", "").replace(")", "").split("|")
            middle = ["(" + re.sub(r'(\d+)', r'self[\1].r2_sp_b', entry) + ")" if i % 2 == 0 else
                      "(" + re.sub(r'(\d+)', r'self[\1].r2_sp', entry) + ")" for i, entry in enumerate(bc)]
            middle = " @ ".join(middle)
            return eval(middle).trace()

        if pOijk.findall(temp_string) != []:                        # Ω_ijk
            ijk = list(map(int, pOijk.findall(temp_string)[0]))
            nol = self.ijk_to_3NonOverlappingLists(ijk)
            Omega = (2 * self.compute("s_" + "".join(map(str, nol[2]))) * self.compute("s_" + "".join(map(str, nol[1]))) -
                     (self.compute("s_" + "".join(map(str, nol[2]))) + self.compute("s_" + "".join(map(str, nol[1]))) -
                      self.compute("s_" + "".join(map(str, nol[0])))) * self.compute("s_" + "".join(map(str, nol[2] + [nol[0][0]]))))
            return Omega

        if pPijk.findall(temp_string) != []:                        # Π_ijk, eg: Π_351 = s_123-s124
            ijk = list(map(int, pPijk.findall(temp_string)[0]))
            nol = self.ijk_to_3NonOverlappingLists(ijk)
            Pi = (self.compute("s_" + "".join(map(str, nol[2] + [nol[0][0]]))) - self.compute("s_" + "".join(map(str, nol[2] + [nol[0][1]]))))
            return Pi

        if pDijk.findall(temp_string) != []:                        # Δ_ijk or Δ_ij|kl|lm
            match = pDijk.findall(temp_string)[0]
            if "|" in match:
                indices_in_corners = [list(map(int, corner)) for corner in match.split("|")]
                r2_sps = [sum([self[_i].r2_sp for _i in indices]) for indices in indices_in_corners]
                r2_sps_b = [sum([self[_i].r2_sp_b for _i in indices]) for indices in indices_in_corners]
                n = len(indices_in_corners)
                matrix = numpy.array([[numpy.trace(numpy.dot(r2_sps[i], r2_sps_b[j])) / 2 for j in range(n)] for i in range(n)])
                return (-1) ** n * det(matrix[:-1, :-1])
            warnings.warn("The shorthand Δ_ijk is deprecated and will be removed.", DeprecationWarning, stacklevel=2)
            NonOverlappingLists = self.ijk_to_3NonOverlappingLists(list(map(int, match)))
            r2_sp_1 = sum([self[_i].r2_sp for _i in NonOverlappingLists[0]])
            r2_sp_b_2 = sum([self[_i].r2_sp_b for _i in NonOverlappingLists[1]])
            return (numpy.trace(numpy.dot(r2_sp_1, r2_sp_b_2)) / 2) ** 2 - det(r2_sp_1) * det(r2_sp_b_2)

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

        if pMi.findall(temp_string) != []:
            """Mass of the i^{th} particle (m_i = s_i)."""
            return self.compute(temp_string.replace("m", "s").replace("M", "S"))

        if pSijk.findall(temp_string) != []:                      # S_ijk...
            r"""Mandelstam variables: s_{i \dots k} = (P_i + \dots + P_k)^2. Computed via determinant of rank 2 spinor."""
            ijk = list(map(int, pSijk.findall(temp_string)[0]))
            tot_r2_sp = sum([self[_i].r2_sp for _i in ijk])
            return det(tot_r2_sp)

        if pA2.findall(temp_string) != []:                        # ⟨A|B⟩ -- contraction is up -> down : lambda[A]^alpha.lambda[B]_alpha
            A, B = map(int, pA2.findall(temp_string)[0])
            return self[A].r_sp_u @ self[B].r_sp_d

        if pAu.findall(temp_string) != []:                        # ⟨A|
            A = int(pAu.findall(temp_string)[0])
            return self[A].r_sp_u

        if pAd.findall(temp_string) != []:                        # |B⟩
            B = int(pAd.findall(temp_string)[0])
            return self[B].r_sp_d

        if pS2.findall(temp_string) != []:                        # [A|B] -- contraction is down -> up : lambda_bar[A]_alpha_dot.lambda_bar[B]^alpha_dot
            A, B = map(int, pS2.findall(temp_string)[0])
            return self[A].l_sp_d @ self[B].l_sp_u

        if pSu.findall(temp_string) != []:                        # |A]
            A = int(pSu.findall(temp_string)[0])
            return self[A].l_sp_u

        if pSd.findall(temp_string) != []:                        # [B|
            B = int(pSd.findall(temp_string)[0])
            return self[B].l_sp_d

        if pNB.findall(temp_string) != []:                        # ⟨A|(B+C+...)|...|D]
            abcd = pNB.search(temp_string)
            a = int(abcd.group('start'))
            bc = abcd.group('middle').replace("(", "").replace(")", "").split("|")
            d = int(abcd.group('end'))

            # Check the contraction is valid
            if len(bc) % 2 == 0 and temp_string[0] == "⟨" and temp_string[-1] != "⟩":
                raise SyntaxError(f"Expected closing \'⟩\', instead found \'{temp_string[-1]}\'.")
            elif len(bc) % 2 == 1 and temp_string[0] == "⟨" and temp_string[-1] != "]":
                raise SyntaxError(f"Expected closing \']\', instead found \'{temp_string[-1]}\'.")
            elif len(bc) % 2 == 0 and temp_string[0] == "[" and temp_string[-1] != "]":
                raise SyntaxError(f"Expected closing \']\', instead found \'{temp_string[-1]}\'.")
            elif len(bc) % 2 == 1 and temp_string[0] == "[" and temp_string[-1] != "⟩":
                raise SyntaxError(f"Expected closing \']\', instead found \'{temp_string[-1]}\'.")

            if temp_string[0] == "⟨":
                middle = ["(" + re.sub(r'(\d+)', r'self[\1].r2_sp_b', entry) + ")" if i % 2 == 0 else
                          "(" + re.sub(r'(\d+)', r'self[\1].r2_sp', entry) + ")" for i, entry in enumerate(bc)]
                middle = " @ ".join(middle)
                result = self[a].r_sp_u @ eval(middle)
            else:
                middle = ["(" + re.sub(r'(\d+)', r'self[\1].r2_sp', entry) + ")" if i % 2 == 0 else
                          "(" + re.sub(r'(\d+)', r'self[\1].r2_sp_b', entry) + ")" for i, entry in enumerate(bc)]
                middle = " @ ".join(middle)
                result = self[a].l_sp_d @ eval(middle)

            if temp_string[-1] == "⟩":
                result = result @ self[d].r_sp_d
            else:
                result = result @ self[d].l_sp_u

            return result

        if pNB_open_begin.findall(temp_string) != []:                        # |(B+C+...)|...|D⟩ or |(B+C+...)|...|D]
            abcd = pNB_open_begin.search(temp_string)
            bc = abcd.group('middle').replace("(", "").replace(")", "").split("|")
            d = int(abcd.group('end'))

            if (temp_string[-1] == "⟩" and len(bc) % 2 == 0) or (temp_string[-1] == "]" and len(bc) % 2 == 1):
                middle = ["(" + re.sub(r'(\d+)', r'self[\1].r2_sp_b', entry) + ")" if i % 2 == 0 else
                          "(" + re.sub(r'(\d+)', r'self[\1].r2_sp', entry) + ")" for i, entry in enumerate(bc)]
                middle = " @ ".join(middle)
                result = eval(middle)
            else:
                middle = ["(" + re.sub(r'(\d+)', r'self[\1].r2_sp', entry) + ")" if i % 2 == 0 else
                          "(" + re.sub(r'(\d+)', r'self[\1].r2_sp_b', entry) + ")" for i, entry in enumerate(bc)]
                middle = " @ ".join(middle)
                result = eval(middle)

            if temp_string[-1] == "⟩":
                result = result @ self[d].r_sp_d
            else:
                result = result @ self[d].l_sp_u

            return result

        if pNB_open_end.findall(temp_string) != []:                        # ⟨A|(B+C+...)|...| or [A|(B+C+...)|...|
            abcd = pNB_open_end.search(temp_string)
            a = int(abcd.group('start'))
            bc = abcd.group('middle').replace("(", "").replace(")", "").split("|")

            if temp_string[0] == "⟨":
                middle = ["(" + re.sub(r'(\d+)', r'self[\1].r2_sp_b', entry) + ")" if i % 2 == 0 else
                          "(" + re.sub(r'(\d+)', r'self[\1].r2_sp', entry) + ")" for i, entry in enumerate(bc)]
                middle = " @ ".join(middle)
                result = self[a].r_sp_u @ eval(middle)
            else:
                middle = ["(" + re.sub(r'(\d+)', r'self[\1].r2_sp', entry) + ")" if i % 2 == 0 else
                          "(" + re.sub(r'(\d+)', r'self[\1].r2_sp_b', entry) + ")" for i, entry in enumerate(bc)]
                middle = " @ ".join(middle)
                result = self[a].l_sp_d @ eval(middle)

            return result

        if pNB_double_open.findall(temp_string) != []:  # |(B+C+...)|...| assumes first entry is r2_sp_b (lower alpha open)
            abcd = pNB_double_open.search(temp_string)
            bc = abcd.group('middle').replace("(", "").replace(")", "").split("|")

            middle = ["(" + re.sub(r'(\d+)', r'self[\1].r2_sp_b', entry) + ")" if i % 2 == 0 else
                      "(" + re.sub(r'(\d+)', r'self[\1].r2_sp', entry) + ")" for i, entry in enumerate(bc)]
            middle = " @ ".join(middle)
            result = eval(middle)

            return result

        if pMVar.findall(temp_string) != []:
            res = getattr(self, temp_string)
            if isinstance(res, str):
                return self.compute(res)
            return res

        # if nothing matches, use abstract syntactic tree parser
        return self._eval(original_temp_string)
