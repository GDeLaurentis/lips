#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   ___          _   _    _          ___      _   ___      _
#  | _ \__ _ _ _| |_(_)__| |___ ___ / __| ___| |_| _ \__ _(_)_ _
#  |  _/ _` | '_|  _| / _| / -_|_-<_\__ \/ -_)  _|  _/ _` | | '_|
#  |_| \__,_|_|  \__|_\__|_\___/__(_)___/\___|\__|_| \__,_|_|_|

# Author: Giuseppe

from __future__ import unicode_literals

import sys
import os
import mpmath

from .tools import pSijk, pDijk, pA2, pS2, p3B, pNB, myException

local_directory = os.path.dirname(__file__)
mpmath.mp.dps = 300

if sys.version_info[0] > 2:
    unicode = str

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_SetPair:

    def set_pair(self, t_s1, t_v1, t_s2, t_v2, itr=10, prec=0.1):
        """Constructs a double collinear limit phase space."""
        for i in range(itr):
            new_invs = self._set_pair_inner(t_s1, t_v1, t_s2, t_v2)     # set it --- note: if a list is returned then switch invariants to those
            if type(new_invs) is list:                                 # used for example for s_123&⟨1|(2+3)|1] ---> s_123&⟨2|3⟩
                t_s1, t_s2 = new_invs[0], new_invs[1]
            elif type(new_invs) in [unicode, str]:
                if new_invs == "Not implemented.":
                    break
            actual1, target1 = abs(self.compute(t_s1)), abs(t_v1)
            error1 = abs(100) * abs((actual1 - target1) / target1)
            compatible_with_zero1 = abs(target1 - actual1) < 10 ** -(0.9 * 300)
            if compatible_with_zero1 is False:
                compatible_with_zero1 = str(0) == str(target1)
            actual2, target2 = abs(self.compute(t_s2)), abs(t_v2)
            error2 = abs(100) * abs((actual2 - target2) / target2)
            compatible_with_zero2 = abs(target2 - actual2) < 10 ** -(0.9 * 300)
            if compatible_with_zero2 is False:
                compatible_with_zero2 = str(0) == str(target2)
            mom_cons, on_shell, big_outliers, small_outliers = self.phasespace_consistency_check()
            if ((error1 < abs(prec) or compatible_with_zero1 is True) and (error2 < abs(prec) or compatible_with_zero2 is True) and
               mom_cons is not False and on_shell is not False):       # if error is less than 1 in 1000 or it is compatible with zero
                if i == 0 and big_outliers == []:
                    return True
                elif big_outliers == []:
                    # print("Succeded to set {} to {} and {} to {} but in {} tries.".format(t_s1, t_v1, t_s2, t_v2, i + 1))
                    return True
                else:
                    self.randomise_all()                               # try to iterate and obtain a phase space without big outliers
            if ("nan" in str(actual1) or "nan" in str(actual2)):
                raise myException("NaN encountered in set pair!")
                self.randomise_all()                                   # try to iterate and obtain a phase space without nan's
        else:
            raise myException("Failed to set {} to {} and {} to {}. Target1 was {}, target2 was {}. Actual1 was {}, actual2 was {}. Error1 was {}, error2 was {}.".format(
                t_s1, t_v1, t_s2, t_v2, target1, target2, actual1, actual2, error1, error2))
            return False
        raise myException("Failed to set {} to {} and {} to {}. Pair not implemented.".format(t_s1, t_v1, t_s2, t_v2))
        return False

    def _set_pair_inner(self, t_s1, t_v1, t_s2, t_v2):                  # Try to take care of all possible combinations

        if pA2.findall(t_s1) != [] or pS2.findall(t_s1) != []:         # First is: ⟨A|B⟩ or [A|B]

            if pA2.findall(t_s2) != [] or pS2.findall(t_s2) != []:               # Second is: ⟨C|D⟩ or [C|D]

                return self._set_pair_A2_or_S2_and_A2_or_S2(t_s1, t_v1, t_s2, t_v2)

            elif pNB.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|...|l⟩

                raise Exception("Not implement in alpha version.")

            elif pSijk.findall(t_s2) != []:                                      # Second is: S_ijk...

                raise Exception("Not implement in alpha version.")

            elif pDijk.findall(t_s2) != []:                                      # Second is: Δ_ijk

                raise Exception("Not implement in alpha version.")

        elif p3B.findall(t_s1) != []:                                  # First: ⟨i|(j+k)|l]

            if pA2.findall(t_s2) != [] or pS2.findall(t_s2) != []:               # Second: ⟨A|B⟩ or [A|B]

                raise Exception("Not implement in alpha version.")

            elif p3B.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|l]

                raise Exception("Not implement in alpha version.")

            elif pNB.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|...|l⟩

                raise Exception("Not implement in alpha version.")

            elif pSijk.findall(t_s2) != []:                                      # Second: S_ijk...

                raise Exception("Not implement in alpha version.")

            elif pDijk.findall(t_s2) != []:                                      # Second is: Δ_ijk

                raise Exception("Not implement in alpha version.")

        elif pNB.findall(t_s1) != []:                                  # First: ⟨i|(j+k)|...|l⟩

            if pA2.findall(t_s2) != [] or pS2.findall(t_s2) != []:               # Second: ⟨A|B⟩ or [A|B]

                raise Exception("Not implement in alpha version.")

            elif p3B.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|l]

                raise Exception("Not implement in alpha version.")

            elif pNB.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|...|l⟩

                raise Exception("Not implement in alpha version.")

            elif pSijk.findall(t_s2) != []:                                      # Second: S_ijk...

                raise Exception("Not implement in alpha version.")

            elif pDijk.findall(t_s2) != []:                                      # Second is: Δ_ijk

                raise Exception("Not implement in alpha version.")

        elif pSijk.findall(t_s1) != []:                                # First is: S_ijk...

            if pA2.findall(t_s2) != [] or pS2.findall(t_s2) != []:               # Second is: ⟨A|B⟩ or [A|B]

                raise Exception("Not implement in alpha version.")

            elif p3B.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|l]

                raise Exception("Not implement in alpha version.")

            elif pNB.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|...|l⟩

                raise Exception("Not implement in alpha version.")

            elif pSijk.findall(t_s2) != []:                                      # Second is: S_ijk...

                raise Exception("Not implement in alpha version.")

            elif pDijk.findall(t_s2) != []:                                      # Second is: Δ_ijk

                raise Exception("Not implement in alpha version.")

        elif pDijk.findall(t_s1) != []:                                # First: Δ_ijk

            if pA2.findall(t_s2) != [] or pS2.findall(t_s2) != []:               # Second: ⟨A|B⟩ or [A|B]

                raise Exception("Not implement in alpha version.")

            elif p3B.findall(t_s2) != []:                                        # Second: ⟨a|(b+c)|d]

                raise Exception("Not implement in alpha version.")

            elif pNB.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|...|l⟩

                raise Exception("Not implement in alpha version.")

            elif pSijk.findall(t_s2) != []:                                      # Second: S_ijk...

                raise Exception("Not implement in alpha version.")

            elif pDijk.findall(t_s2) != []:                                      # Second: Δ_ijk

                raise Exception("Not implement in alpha version.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_pair_A2_or_S2_and_A2_or_S2(self, t_s1, t_v1, t_s2, t_v2):     # Current failed: 0/870 @ 6pt

        if pA2.findall(t_s1) != []:
            ab = list(map(int, list(pA2.findall(t_s1)[0])))
        elif pS2.findall(t_s1) != []:
            ab = list(map(int, list(pS2.findall(t_s1)[0])))

        if pA2.findall(t_s2) != []:
            cd = list(map(int, list(pA2.findall(t_s2)[0])))
        elif pS2.findall(t_s2) != []:
            cd = list(map(int, list(pS2.findall(t_s2)[0])))

        overlap = []
        for i in range(2):
            if ab[i] == cd[0] or ab[i] == cd[1]:
                overlap += [ab[i]]
            else:
                ab_only = ab[i]
            if cd[i] == ab[0] or cd[i] == ab[1]:
                pass
            else:
                cd_only = cd[i]
        if ((t_s1[0] == "⟨" and t_s2[0] == "⟨") or             # if both ⟨⟩ or [] and overlap is not empty rearrange overlap in front
           (t_s1[0] == "[" and t_s2[0] == "[")) and overlap != []:
            if t_s1[0] == "⟨":
                t_s1 = "⟨{}|{}⟩".format(overlap[0], ab_only)
                t_s2 = "⟨{}|{}⟩".format(overlap[0], cd_only)
            else:
                t_s1 = "[{}|{}]".format(overlap[0], ab_only)
                t_s2 = "[{}|{}]".format(overlap[0], cd_only)
        self.set(t_s1, t_v1, fix_mom=False)                    # set it
        self.set(t_s2, t_v2, fix_mom=False)
        plist = list(map(int, self._complementary(list(set([unicode(ab[0]), unicode(ab[1]), unicode(cd[0]), unicode(cd[1])])))))
        if len(plist) >= 2:
            self.fix_mom_cons(plist[0], plist[1])
        elif len(plist) == 1:
            if t_s1[0] == "⟨":
                self.fix_mom_cons(ab[0], plist[0], real_momenta=False, axis=2)
            else:
                self.fix_mom_cons(ab[0], plist[0], real_momenta=False, axis=1)
        else:
            self.fix_mom_cons(ab[0], ab[1])
            if not self.momentum_conservation_check():
                raise myException("Not enough particles to fix mom cons!")
        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
