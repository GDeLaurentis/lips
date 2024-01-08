# -*- coding: utf-8 -*-

#   ___          _   _    _          ___      _   ___      _
#  | _ \__ _ _ _| |_(_)__| |___ ___ / __| ___| |_| _ \__ _(_)_ _
#  |  _/ _` | '_|  _| / _| / -_|_-<_\__ \/ -_)  _|  _/ _` | | '_|
#  |_| \__,_|_|  \__|_\__|_\___/__(_)___/\___|\__|_| \__,_|_|_|

# Author: Giuseppe

import numpy
import mpmath

from ..tools import flatten, pSijk, pDijk, pA2, pS2, p3B, pNB, myException


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_SetPair:

    # PRIVATE METHODS

    def _set_pair(self, t_s1, t_v1, t_s2, t_v2):
        """Constructs a doubly singular phase space point."""
        new_invs = self._set_pair_inner(t_s1, t_v1, t_s2, t_v2)     # set it --- note: if a list is returned then switch invariants to those
        if type(new_invs) is list:                                  # used for example for s_123&⟨1|(2+3)|1] ---> s_123&⟨2|3⟩
            t_s1, t_s2 = new_invs[0], new_invs[1]
        elif type(new_invs) is str and new_invs == "Not implemented.":
            raise myException("Failed to set {} to {} and {} to {}. Pair not implemented.".format(t_s1, t_v1, t_s2, t_v2))
        abs_diff1 = min(abs(self.compute(t_s1) - t_v1), abs(self.compute(t_s1) + t_v1))
        abs_diff2 = min(abs(self.compute(t_s2) - t_v2), abs(self.compute(t_s2) + t_v2))
        mom_cons, on_shell = self.momentum_conservation_check(), self.onshell_relation_check()
        # mom_cons, on_shell, big_outliers, small_outliers = self.phasespace_consistency_check()  # this would be more thorough.. is it needed tho?
        if mom_cons is False:
            raise myException("Momentum conservation is not satisfied: ", max(map(abs, flatten(self.total_mom))))
        elif on_shell is False:
            raise myException("On shellness is not satisfied: ", max(map(abs, flatten(self.masses))))
        elif not all([abs_diff1 <= self.field.tollerance, abs_diff2 <= self.field.tollerance]):
            raise myException("Failed to set {} to {} and {} to {}. Instead got {} and {}. Absoute differences: {} and {}.".format(
                t_s1, abs(t_v1), t_s2, abs(t_v2), abs(self.compute(t_s1)), abs(self.compute(t_s2)), abs_diff1, abs_diff2))

    def _set_pair_inner(self, t_s1, t_v1, t_s2, t_v2):                  # Try to take care of all possible combinations

        if pA2.findall(t_s1) != [] or pS2.findall(t_s1) != []:         # First is: ⟨A|B⟩ or [A|B]

            if pA2.findall(t_s2) != [] or pS2.findall(t_s2) != []:               # Second is: ⟨C|D⟩ or [C|D]

                return self._set_pair_A2_or_S2_and_A2_or_S2(t_s1, t_v1, t_s2, t_v2)

            elif pNB.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|...|l⟩

                return self._set_pair_A2_or_S2_and_NB(t_s1, t_v1, t_s2, t_v2)

            elif pSijk.findall(t_s2) != []:                                      # Second is: S_ijk...

                return self._set_pair_A2_or_S2_and_Sijk(t_s1, t_v1, t_s2, t_v2)

            elif pDijk.findall(t_s2) != []:                                      # Second is: Δ_ijk

                return self._set_pair_A2_or_S2_and_Dijk(t_s1, t_v1, t_s2, t_v2)

        elif p3B.findall(t_s1) != []:                                  # First: ⟨i|(j+k)|l]

            if pA2.findall(t_s2) != [] or pS2.findall(t_s2) != []:               # Second: ⟨A|B⟩ or [A|B]

                return self._set_pair_inner(t_s2, t_v2, t_s1, t_v1)

            elif p3B.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|l]

                return self._set_pair_3B_and_3B(t_s1, t_v1, t_s2, t_v2)

            elif pNB.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|...|l⟩

                return "Not implemented."

            elif pSijk.findall(t_s2) != []:                                      # Second: S_ijk...

                return self._set_pair_3B_and_Sijk(t_s1, t_v1, t_s2, t_v2)

            elif pDijk.findall(t_s2) != []:                                      # Second is: Δ_ijk

                return self._set_pair_3B_and_Dijk(t_s1, t_v1, t_s2, t_v2)

        elif pNB.findall(t_s1) != []:                                  # First: ⟨i|(j+k)|...|l⟩

            if pA2.findall(t_s2) != [] or pS2.findall(t_s2) != []:               # Second: ⟨A|B⟩ or [A|B]

                return self._set_pair_inner(t_s2, t_v2, t_s1, t_v1)

            elif p3B.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|l]

                return "Not implemented."

            elif pNB.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|...|l⟩

                if t_s1 == "⟨7|3+4|5+6|7⟩" and t_s2 == "[7|3+4|5+6|7]":
                    self._set("[7|3+4|5+6|7]", t_v2, fix_mom=False)
                    self._set("⟨7|3+4|5+6|7⟩", t_v1, fix_mom=False)
                    self.fix_mom_cons(1, 2)
                else:
                    return "Not implemented."

            elif pSijk.findall(t_s2) != []:                                      # Second: S_ijk...

                return "Not implemented."

            elif pDijk.findall(t_s2) != []:                                      # Second is: Δ_ijk

                return self._set_pair_NB_and_Dijk(t_s1, t_v1, t_s2, t_v2)

        elif pSijk.findall(t_s1) != []:                                # First is: S_ijk...

            if pA2.findall(t_s2) != [] or pS2.findall(t_s2) != []:               # Second is: ⟨A|B⟩ or [A|B]

                return self._set_pair_inner(t_s2, t_v2, t_s1, t_v1)

            elif p3B.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|l]

                return self._set_pair_inner(t_s2, t_v2, t_s1, t_v1)

            elif pNB.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|...|l⟩

                return "Not implemented."

            elif pSijk.findall(t_s2) != []:                                      # Second is: S_ijk...

                return self._set_pair_Sijk_and_Sijk(t_s1, t_v1, t_s2, t_v2)

            elif pDijk.findall(t_s2) != []:                                      # Second is: Δ_ijk

                return self._set_pair_Sijk_and_Dijk(t_s1, t_v1, t_s2, t_v2)

        elif pDijk.findall(t_s1) != []:                                # First: Δ_ijk

            if pA2.findall(t_s2) != [] or pS2.findall(t_s2) != []:               # Second: ⟨A|B⟩ or [A|B]

                return self._set_pair_inner(t_s2, t_v2, t_s1, t_v1)

            elif p3B.findall(t_s2) != []:                                        # Second: ⟨a|(b+c)|d]

                return self._set_pair_inner(t_s2, t_v2, t_s1, t_v1)

            elif pNB.findall(t_s2) != []:                                        # Second is: ⟨i|(j+k)|...|l⟩

                return self._set_pair_NB_and_Dijk(t_s2, t_v2, t_s1, t_v1)

            elif pSijk.findall(t_s2) != []:                                      # Second: S_ijk...

                return self._set_pair_inner(t_s2, t_v2, t_s1, t_v1)

            elif pDijk.findall(t_s2) != []:                                      # Second: Δ_ijk

                return "Not implemented."

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
        # if both are ⟨⟩ or [] and overlap is not empty rearrange overlap in front
        if ((t_s1[0] == "⟨" and t_s2[0] == "⟨") or (t_s1[0] == "[" and t_s2[0] == "[")) and overlap != []:
            if t_s1[0] == "⟨":
                t_s1 = "⟨{}|{}⟩".format(overlap[0], ab_only)
                t_s2 = "⟨{}|{}⟩".format(overlap[0], cd_only)
            else:
                t_s1 = "[{}|{}]".format(overlap[0], ab_only)
                t_s2 = "[{}|{}]".format(overlap[0], cd_only)
            # get the sign right too
            flipped1 = False if overlap[0] == ab[0] else True
            flipped2 = False if overlap[0] == cd[0] else True
        else:
            flipped1 = flipped2 = False
        self._set(t_s1, t_v1 if flipped1 is False else -t_v1, fix_mom=False)
        self._set(t_s2, t_v2 if flipped2 is False else -t_v2, fix_mom=False)
        plist = list(map(int, self._complementary(list(set([str(ab[0]), str(ab[1]), str(cd[0]), str(cd[1])])))))
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

    def _set_pair_A2_or_S2_and_NB(self, t_s1, t_v1, t_s2, t_v2):    # Current failed: 48/4680 @ 6pt

        if pA2.findall(t_s1) != []:
            ab = list(map(int, list(pA2.findall(t_s1)[0])))
        elif pS2.findall(t_s1) != []:
            ab = list(map(int, list(pS2.findall(t_s1)[0])))

        if "-" in t_s2:                                         # a minus sign would mess up inversion
            raise myException("Detected minus in string. Not implemented.")
        lNB, lNBs, lNBms, lNBe = self._get_lNB(t_s2)
        if self._can_fix_mom_cons(t_s1, t_s2)[0] is False:      # if necessary look for alternative way to write it which allows to fix mom cons
            for i, iNBm in enumerate(lNBms):                    # try to flip the i^th bracket and see if it allows to fix mom cons
                _lNBms = [entry for entry in lNBms]
                if i == 0 and i == len(lNBms) - 1:                # this is close to both head and tail (only for 3Brackets)
                    alt = self._complementary(iNBm + [lNBs] + [lNBe])
                elif i == 0:                                    # this is close to the head, what I call extremum of middle
                    alt = self._complementary(iNBm + [lNBs])
                elif i == len(lNBms) - 1:                         # this is close to the tail, what I call extremum of middle
                    alt = self._complementary(iNBm + [lNBe])
                else:                                           # this is not close to either head or tail
                    alt = self._complementary(iNBm)
                _lNBms[i] = alt
                t_s2_new = self._lNB_to_string(t_s2[0], lNBs, _lNBms, lNBe, t_s2[-1])
                if self._can_fix_mom_cons(t_s1, t_s2_new)[0] is not False:
                    t_s2 = t_s2_new
        lNB, lNBs, lNBms, lNBe = self._get_lNB(t_s2)
        plist = self._complementary(ab + lNB)
        # print(lNBs, lNBms, lNBe)
        if ((t_s1[0] == "⟨" and t_s2[0] == "⟨" and (t_s2[-1] != "⟩" or (t_s1[1] == t_s2[1] and t_s1[1] != t_s2[-2]))) or
           (t_s1[0] == "[" and t_s2[0] == "[" and (t_s2[-1] != "]" or (t_s1[1] == t_s2[1] and t_s1[1] != t_s2[-2])))):
            # or (lNBs == lNBe and set(lNBms[0]) == set(ab))):               # old statement
            t_s2 = t_s2[::-1]                                   # flip it
            t_s2 = t_s2.replace("(", "X").replace(")", "(").replace("X", ")")
            t_s2 = t_s2.replace("⟨", "X").replace("⟩", "⟨").replace("X", "⟩")
            t_s2 = t_s2.replace("[", "X").replace("]", "[").replace("X", "]")
        lNB, lNBs, lNBms, lNBe = self._get_lNB(t_s2)
        plist = self._complementary(ab + lNB)
        use_mode = 1
        if len(plist) == 0 or (lNBs in ab and len(lNBms) > 1):  # e.g. '⟨2|4⟩' && '⟨3|(1+2)|(1+6)|5⟩'
            use_mode = 2
            for i, entry in enumerate(lNBms[0]):                # look for free in lNBms[0]
                if entry not in set(flatten(lNBms[1:]) + ab):
                    if i != 0:                                  # put it in first place
                        lNBms[0][0], lNBms[0][i] = lNBms[0][i], lNBms[0][0]
                        t_s2 = self._lNB_to_string(t_s2[0], lNBs, lNBms, lNBe, t_s2[-1])
                    break
            else:                                               # if you don't find it flip it (hope there is a free one in lNBms[-1]
                t_s2 = t_s2[::-1]
                t_s2 = t_s2.replace("(", "X").replace(")", "(").replace("X", ")")
                t_s2 = t_s2.replace("⟨", "X").replace("⟩", "⟨").replace("X", "⟩")
                t_s2 = t_s2.replace("[", "X").replace("]", "[").replace("X", "]")
                lNB, lNBs, lNBms, lNBe = self._get_lNB(t_s2)
                for i, entry in enumerate(lNBms[0]):                # look for free in lNBms[0]
                    if entry not in set(flatten(lNBms[1:]) + ab):
                        if i != 0:                                  # put it in first place
                            lNBms[0][0], lNBms[0][i] = lNBms[0][i], lNBms[0][0]
                            t_s2 = self._lNB_to_string(t_s2[0], lNBs, lNBms, lNBe, t_s2[-1])
                        break
        lNB, lNBs, lNBms, lNBe = self._get_lNB(t_s2)
        _tuple, use_axis = self._can_fix_mom_cons(t_s1, t_s2)
        # print(_tuple, use_axis, t_s1, t_s2, use_mode)                   # debugging tool
        self._set(t_s1, t_v1, fix_mom=False)
        self._set(t_s2, t_v2, fix_mom=False, mode=use_mode)
        if _tuple is not False:
            self.fix_mom_cons(_tuple[0], _tuple[1], real_momenta=False, axis=use_axis)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_pair_A2_or_S2_and_Sijk(self, t_s1, t_v1, t_s2, t_v2):   # Current failed: 0/300 @ 6pt

        if pA2.findall(t_s1) != []:
            ab = list(map(int, list(pA2.findall(t_s1)[0])))
        elif pS2.findall(t_s1) != []:
            ab = list(map(int, list(pS2.findall(t_s1)[0])))

        ijk = list(map(int, list(pSijk.findall(t_s2)[0])))
        if ab[0] in ijk or ab[1] in ijk:                       # completely or partially overlapping
            overlap = []                                       # move the overlap at the end
            for i in range(2):
                if ab[i] in ijk:
                    ijk.remove(ab[i])
                    overlap += [ab[i]]
            str1 = 's_'
            for i in range(len(ijk)):
                str1 += '{}'.format(ijk[i])
            for entry in overlap:
                str1 += '{}'.format(entry)
            t_s2 = str1
        elif ab[0] not in ijk and ab[1] not in ijk:             # disjoint
            if len(self) == 6 and len(ijk) == 3:                # six particles -> disjoint = overlapping
                compl = self._complementary(list(map(str, ijk)))
                t_s2 = 's_{}{}{}'.format(compl[0], compl[1], compl[2])
                self._set_pair_inner(t_s1, t_v1, t_s2, t_v2)
                return
            elif len(self) > 6:                                 # more than six particles -> enough momenta for it not to be a problem
                pass
            else:                                               # less than 6 is not supposed to happen
                raise myException("S_ijk called with 5 particles!")
        if t_s1[0] == "⟨":                                      # set it using mode 2 (this reduces the number of invariants that go to zero)
            self._set(t_s1, t_v1, fix_mom=False)
            self._set(t_s2, t_v2, fix_mom=False, mode=2)
        elif t_s1[0] == "[":                                    # set it using mode 1 (this reduces the number of invariants that go to zero)
            self._set(t_s1, t_v1, fix_mom=False)
            self._set(t_s2, t_v2, fix_mom=False, mode=1)
        plist = list(map(int, self._complementary(list(set([str(entry) for entry in ijk + ab])))))
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_pair_A2_or_S2_and_Dijk(self, t_s1, t_v1, t_s2, t_v2):   # Current failed: 0/60 @ 6pt

        if pA2.findall(t_s1) != []:
            ab = list(map(int, list(pA2.findall(t_s1)[0])))
        elif pS2.findall(t_s1) != []:
            ab = list(map(int, list(pS2.findall(t_s1)[0])))

        ab = list(map(int, ab))
        match_list = pDijk.findall(t_s2)[0]
        if match_list[0] == '':
            adjacent = False
            NonOverlappingLists = [list(map(int, corner)) for corner in match_list[1:]]
        else:
            adjacent = True
            NonOverlappingLists = self.ijk_to_3NonOverlappingLists(list(map(int, match_list[0])))

        both_in_List = -1                                       # if both ab in a single List:
        for i, List in enumerate(NonOverlappingLists):
            if (ab[0] in List and ab[1] in List):
                both_in_List = i
        if both_in_List != -1:
            NonOverlappingLists = [NonOverlappingLists[both_in_List],
                                   NonOverlappingLists[(both_in_List + 1) % 3],
                                   NonOverlappingLists[(both_in_List + 2) % 3]]
        else:
            for i, List in enumerate(NonOverlappingLists):
                if (ab[0] not in List and ab[1] not in List):
                    free_List = i
            NonOverlappingLists = [NonOverlappingLists[(free_List + 1) % 3],
                                   NonOverlappingLists[(free_List + 2) % 3],
                                   NonOverlappingLists[free_List]]
        if adjacent is True:
            t_s2 = "Δ_{}{}{}".format(NonOverlappingLists[0][0], NonOverlappingLists[1][0], NonOverlappingLists[2][0])
        else:
            t_s2 = "Δ_{}|{}|{}".format("".join(map(str, NonOverlappingLists[0])), "".join(map(str, NonOverlappingLists[1])), "".join(map(str, NonOverlappingLists[2])))
        self._set(t_s1, t_v1, fix_mom=False)
        if t_s1[0] == "⟨":
            self._set(t_s2, t_v2, fix_mom=True, mode=2)
        else:
            self._set(t_s2, t_v2, fix_mom=True, mode=1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_pair_3B_and_3B(self, t_s1, t_v1, t_s2, t_v2):     # Current failed: 180/6972 @ 6pt

        def unpack(t_s):
            l3B = list(p3B.findall(t_s)[0])
            l3B[1] = l3B[1].split("+")
            l3Bs, l3Bm, l3Be = l3B[0], l3B[1], l3B[2]
            l3B = [item for sublist in l3B for item in sublist]
            l3Bc = ""
            a_or_s = t_s[0]
            for i in range(len(l3Bm)):
                l3Bc += "{}+".format(l3Bm[i])
            l3Bc = l3Bc[:-1]
            return l3B, l3Bs, l3Bm, l3Be, a_or_s, l3Bc

        def how_to_fix_mom_cons(t_s1, t_s2):
            l3B_0, l3Bs_0, l3Bm_0, l3Be_0, a_or_s_0, l3Bc_0 = unpack(t_s1)
            l3B_1, l3Bs_1, l3Bm_1, l3Be_1, a_or_s_1, l3Bc_1 = unpack(t_s2)
            free = list(map(int, self._complementary(list(set(l3B_0 + l3B_1)))))
            if len(free) >= 2:
                return free[0], free[1], False, 1
            elif len(free) == 1:
                if t_s1[0] == "⟨" and t_s2[0] == "⟨" and l3Be_1 not in l3Bm_0 and l3Be_1 != l3Bs_0 and l3Be_1 != l3Bs_1:
                    return free[0], int(l3Be_1), False, 1
                elif t_s1[0] == "⟨" and t_s2[0] == "⟨" and l3Bs_1 not in l3Bm_0 and l3Bs_1 != l3Be_0 and l3Bs_1 != l3Be_1:
                    return free[0], int(l3Bs_1), False, 2
                elif t_s1[0] == "⟨" and t_s2[0] == "⟨" and l3Be_0 not in l3Bm_1 and l3Be_0 != l3Bs_1 and l3Be_0 != l3Bs_0:
                    return free[0], int(l3Be_0), False, 1
                elif t_s1[0] == "⟨" and t_s2[0] == "⟨" and l3Bs_0 not in l3Bm_1 and l3Bs_0 != l3Be_1 and l3Bs_0 != l3Be_0:
                    return free[0], int(l3Bs_0), False, 2
                elif t_s1[0] == "[" and t_s2[0] == "[" and l3Be_1 not in l3Bm_0 and l3Be_1 != l3Bs_0 and l3Be_1 != l3Bs_1:
                    return free[0], int(l3Be_1), False, 2
                elif t_s1[0] == "[" and t_s2[0] == "[" and l3Bs_1 not in l3Bm_0 and l3Bs_1 != l3Be_0 and l3Bs_1 != l3Be_1:
                    return free[0], int(l3Bs_1), False, 1
                elif t_s1[0] == "[" and t_s2[0] == "[" and l3Be_0 not in l3Bm_1 and l3Be_0 != l3Bs_1 and l3Be_0 != l3Bs_0:
                    return free[0], int(l3Be_0), False, 2
                elif t_s1[0] == "[" and t_s2[0] == "[" and l3Bs_0 not in l3Bm_1 and l3Bs_0 != l3Be_1 and l3Bs_0 != l3Be_0:
                    return free[0], int(l3Bs_0), False, 1
                else:
                    return "Impossible", "", "", ""
            else:
                if t_s1[0] == "⟨" and t_s2[0] == "⟨" and l3Bs_0 not in l3B_1 and l3Bs_1 not in l3B_0 and l3B_0.count(l3Bs_0) == 1 and l3B_1.count(l3Bs_1) == 1:
                    return int(l3Bs_0), int(l3Bs_1), False, 2
                elif t_s1[0] == "[" and t_s2[0] == "[" and l3Bs_0 not in l3B_1 and l3Bs_1 not in l3B_0 and l3B_0.count(l3Bs_0) == 1 and l3B_1.count(l3Bs_1) == 1:
                    return int(l3Bs_0), int(l3Bs_1), False, 1
                elif t_s1[0] == "⟨" and t_s2[0] == "[" and l3Bs_0 not in l3B_1 and l3Be_1 not in l3B_0 and l3B_0.count(l3Bs_0) == 1 and l3B_1.count(l3Be_1) == 1:
                    return int(l3Bs_0), int(l3Be_1), False, 2
                elif t_s1[0] == "[" and t_s2[0] == "⟨" and l3Be_0 not in l3B_1 and l3Bs_1 not in l3B_0 and l3B_0.count(l3Be_0) == 1 and l3B_1.count(l3Bs_1) == 1:
                    return int(l3Be_0), int(l3Bs_1), False, 2
                else:
                    return "Impossible", "", "", ""

        def can_fix_mom_cons(*args):
            free1, _, _, _ = how_to_fix_mom_cons(*args)
            if free1 != "Impossible":
                return True
            else:
                return False

        def flip_using_mom_cons(t_s):
            l3B, l3Bs, l3Bm, l3Be, a_or_s, l3Bc = unpack(t_s)
            l3Bm = self._complementary(list(set([l3Bs] + [l3Be] + l3Bm)))
            l3Bc = ""
            for i in range(len(l3Bm)):
                l3Bc += "{}+".format(l3Bm[i])
            l3Bc = l3Bc[:-1]
            if a_or_s == '⟨':
                t_s = '⟨{}|({})|{}]'.format(l3Bs, l3Bc, l3Be)
            elif a_or_s == '[':
                t_s = '[{}|({})|{}⟩'.format(l3Bs, l3Bc, l3Be)
            return t_s

        def reverse_order(t_s):
            l3B, l3Bs, l3Bm, l3Be, a_or_s, l3Bc = unpack(t_s)
            if a_or_s == '⟨':
                t_s = '[{}|({})|{}⟩'.format(l3Be, l3Bc, l3Bs)
            elif a_or_s == '[':
                t_s = '⟨{}|({})|{}]'.format(l3Be, l3Bc, l3Bs)
            return t_s

        if "-" in t_s1 or "-" in t_s2:                    # a minus sign would mess up inversion
            print("Error: Detected - in string. Not implemented.")

        l3B_0, l3Bs_0, l3Bm_0, l3Be_0, a_or_s_0, l3Bc_0 = unpack(t_s1)
        l3B_1, l3Bs_1, l3Bm_1, l3Be_1, a_or_s_1, l3Bc_1 = unpack(t_s2)

        free = self._complementary(list(set(l3B_0 + l3B_1)))
        free_first_flipped = self._complementary(         # free if first is flipped
            list(set([l3Bs_0] + [l3Bs_1] + [l3Be_0] + [l3Be_1] + l3Bm_1 + self._complementary(list(set([l3Bs_0] + [l3Be_0] + l3Bm_0))))))
        free_second_flipped = self._complementary(        # free if second is flipped
            list(set([l3Bs_0] + [l3Bs_1] + [l3Be_0] + [l3Be_1] + l3Bm_0 + self._complementary(list(set([l3Bs_1] + [l3Be_1] + l3Bm_1))))))
        free_both_flipped = self._complementary(          # free if both are flipped
            list(set([l3Bs_0] + [l3Bs_1] + [l3Be_0] + [l3Be_1] +
                     self._complementary(list(set([l3Bs_0] + [l3Be_0] + l3Bm_0))) + self._complementary(list(set([l3Bs_1] + [l3Be_1] + l3Bm_1))))))

        if len(free) >= 2:
            pass
        elif len(free_first_flipped) >= 2:
            t_s1 = flip_using_mom_cons(t_s1)
        elif len(free_second_flipped) >= 2:
            t_s2 = flip_using_mom_cons(t_s2)
        elif len(free_both_flipped) >= 2:
            t_s1 = flip_using_mom_cons(t_s1)
            t_s2 = flip_using_mom_cons(t_s2)

        l3B_0, l3Bs_0, l3Bm_0, l3Be_0, a_or_s_0, l3Bc_0 = unpack(t_s1)
        l3B_1, l3Bs_1, l3Bm_1, l3Be_1, a_or_s_1, l3Bc_1 = unpack(t_s2)

        if can_fix_mom_cons(t_s1, t_s2):
            pass
        elif can_fix_mom_cons(flip_using_mom_cons(t_s1), t_s2):
            t_s1 = flip_using_mom_cons(t_s1)
        elif can_fix_mom_cons(t_s1, flip_using_mom_cons(t_s2)):
            t_s2 = flip_using_mom_cons(t_s2)
        elif can_fix_mom_cons(flip_using_mom_cons(t_s1), flip_using_mom_cons(t_s2)):
            t_s1 = flip_using_mom_cons(t_s1)
            t_s2 = flip_using_mom_cons(t_s2)

        l3B_0, l3Bs_0, l3Bm_0, l3Be_0, a_or_s_0, l3Bc_0 = unpack(t_s1)
        l3B_1, l3Bs_1, l3Bm_1, l3Be_1, a_or_s_1, l3Bc_1 = unpack(t_s2)

        # Look for differences in the middle section of the 3 brackets
        l3Bm_0_only_list = [entry for entry in l3Bm_0 if entry not in l3Bm_1]  # look for an entry in the middle of the first but not of the second
        l3Bm_1_only_list = [entry for entry in l3Bm_1 if entry not in l3Bm_0]  # look for and entry in the middle of the second ut not of the first
        # if the two centers are not the exact same and at least one entry only in one of them is not the same as the start and end of the other
        if set(l3B_0) == set(l3B_1) and len(set(l3B_0)) == 3 and len(set(l3B_1)) == 3:   # and t_v1 == t_v2:
            X, Y = t_v1, t_v2
            if l3B_0[1] == l3B_1[0]:                                     # the function is for ⟨a|b+c|a] && ⟨b|a+c|b]
                A, B, C = int(l3B_0[0]), int(l3B_0[1]), int(l3B_0[2])
            else:
                A, B, C = int(l3B_0[0]), int(l3B_0[2]), int(l3B_0[1])
            c1, d1 = self[A].l_sp_d[0, 0], self[A].l_sp_d[0, 1]
            a6, b6 = self[B].r_sp_d[0, 0], self[B].r_sp_d[1, 0]
            c6, d6 = self[B].l_sp_d[0, 0], self[B].l_sp_d[0, 1]
            a2, b2 = self[C].r_sp_d[0, 0], self[C].r_sp_d[1, 0]
            c2, d2 = self[C].l_sp_d[0, 0], self[C].l_sp_d[0, 1]
            a = -((-((a6 * c6 * d1 - a6 * c1 * d6) * X) - (a2 * c2 * d1 + a6 * c6 * d1 - a2 * c1 * d2 - a6 * c1 * d6) *
                   (a6 * b2 * c6 * d2 - a2 * b6 * c6 * d2 - a6 * b2 * c2 * d6 + a2 * b6 * c2 * d6 - Y)) /
                  (-((a2 * c2 * d1 + a6 * c6 * d1 - a2 * c1 * d2 - a6 * c1 * d6) * (-(b6 * c6 * d1) + b6 * c1 * d6)) +
                   (a6 * c6 * d1 - a6 * c1 * d6) * (-(b2 * c2 * d1) - b6 * c6 * d1 + b2 * c1 * d2 + b6 * c1 * d6)))
            b = (a6 * b2 ** 2 * c2 * c6 * d1 * d2 - a2 * b2 * b6 * c2 * c6 * d1 * d2 + a6 * b2 * b6 * c6 ** 2 * d1 * d2 - a2 * b6 ** 2 * c6 ** 2 * d1 * d2 -
                 a6 * b2 ** 2 * c1 * c6 * d2 ** 2 + a2 * b2 * b6 * c1 * c6 * d2 ** 2 -
                 a6 * b2 ** 2 * c2 ** 2 * d1 * d6 + a2 * b2 * b6 * c2 ** 2 * d1 * d6 - a6 * b2 * b6 * c2 * c6 * d1 * d6 + a2 * b6 ** 2 * c2 * c6 * d1 * d6 +
                 a6 * b2 ** 2 * c1 * c2 * d2 * d6 - a2 * b2 * b6 * c1 * c2 * d2 * d6 -
                 a6 * b2 * b6 * c1 * c6 * d2 * d6 + a2 * b6 ** 2 * c1 * c6 * d2 * d6 + a6 * b2 * b6 * c1 * c2 * d6 ** 2 - a2 * b6 ** 2 * c1 * c2 * d6 ** 2 +
                 b6 * c6 * d1 * X - b6 * c1 * d6 * X - b2 * c2 * d1 * Y - b6 * c6 * d1 * Y + b2 * c1 * d2 * Y +
                 b6 * c1 * d6 * Y) / ((-(a6 * b2) + a2 * b6) * (c2 * d1 - c1 * d2) * (c6 * d1 - c1 * d6))
            self[A].r_sp_d = numpy.array([a, b])
            if can_fix_mom_cons(t_s1, t_s2):
                self.fix_mom_cons(*how_to_fix_mom_cons(t_s1, t_s2))
            else:
                raise myException("Not supposed to happen --- custom double set for two l3Bs.")
            return
        elif l3Bm_1_only_list != [] and (l3Bm_1_only_list != [l3Bs_0] or l3Bm_1_only_list != [l3Be_0]):
            for l3Bm_1_only in l3Bm_1_only_list:      # trigger the next if statement by flipping stuff
                if l3Bm_1_only == l3Bs_0 and l3Bm_1_only != l3Be_0:
                    l3Bs_0, l3Be_0 = l3Be_0, l3Bs_0
                    if a_or_s_0 == '⟨':
                        a_or_s_0 = '['
                    elif a_or_s_0 == '[':
                        a_or_s_0 = '⟩'
                if l3Bm_1_only != l3Bs_0:
                    l3Bm_1.remove(l3Bm_1_only)        # put it in front
                    l3Bm_1 = [l3Bm_1_only] + l3Bm_1     # rebuild the second making sure l3Bm_1_only appears in the correct position
                    l3Bc_1 = ""                       # i.e. next to the entry of the same type as l3Bs_1
                    for i in range(len(l3Bm_1)):
                        l3Bc_1 += "{}+".format(l3Bm_1[i])
                    l3Bc_1 = l3Bc_1[:-1]
                    if a_or_s_0 == '⟨':
                        if a_or_s_1 == '⟨':
                            t_s2 = '⟨{}|({})|{}]'.format(l3Bs_1, l3Bc_1, l3Be_1)
                        elif a_or_s_1 == '[':
                            t_s2 = '⟨{}|({})|{}]'.format(l3Be_1, l3Bc_1, l3Bs_1)
                    elif a_or_s_0 == '[':
                        if a_or_s_1 == '⟨':
                            t_s2 = '[{}|({})|{}⟩'.format(l3Be_1, l3Bc_1, l3Bs_1)
                        elif a_or_s_1 == '[':
                            t_s2 = '[{}|({})|{}⟩'.format(l3Bs_1, l3Bc_1, l3Be_1)
                    l3Bc_0 = ""                       # rebuild the first one
                    for i in range(len(l3Bm_0)):
                        l3Bc_0 += "{}+".format(l3Bm_0[i])
                    l3Bc_0 = l3Bc_0[:-1]
                    if a_or_s_0 == '⟨':
                        t_s1 = '⟨{}|({})|{}]'.format(l3Bs_0, l3Bc_0, l3Be_0)
                    elif a_or_s_0 == '[':
                        t_s1 = '[{}|({})|{}⟩'.format(l3Bs_0, l3Bc_0, l3Be_0)
                    use_mode = 2
                    break
        elif l3Bm_0_only_list != [] and (l3Bm_0_only_list != [l3Bs_1] or l3Bm_0_only_list != [l3Be_1]):
            return self._set_pair_inner(t_s2, t_v2, t_s1, t_v1)  # flip them to trigger the above condition
        else:                                            # if the overlap is 100% in the center must use the start and end entries
            if a_or_s_0 != a_or_s_1:                     # make sure they are both flipped the same way
                t_s1 = reverse_order(t_s1)
            if l3Bs_0 != l3Bs_1:                         # now look for non overlap
                use_mode = 1
            elif l3Be_0 != l3Be_1:                       # must flip both strings
                use_mode = 1
                t_s1 = reverse_order(t_s1)
                t_s2 = reverse_order(t_s2)

        if len(t_s1) > len(t_s2):
            t_s1, t_s2 = t_s2, t_s1
            t_v1, t_v2 = t_v2, t_v1

        self._set(t_s1, t_v1, fix_mom=False)
        self._set(t_s2, t_v2, fix_mom=False, mode=use_mode)

        if can_fix_mom_cons(t_s1, t_s2):
            self.fix_mom_cons(*how_to_fix_mom_cons(t_s1, t_s2))
        else:
            raise myException("Not enough particles to fix mom cons!")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_pair_3B_and_Sijk(self, t_s1, t_v1, t_s2, t_v2):   # Current failed: 0/840 @ 6pt

        ijk = list(map(int, list(pSijk.findall(t_s2)[0])))
        l3B = list(p3B.findall(t_s1)[0])
        l3B[1] = l3B[1].split("+")
        l3Bs, l3Bm, l3Be = int(l3B[0]), list(map(int, l3B[1])), int(l3B[2])
        l3B = list(map(int, [item for sublist in l3B for item in sublist]))
        free = list(map(int, self._complementary(list(map(str, list(set(ijk + l3B)))))))
        free_first_flipped = list(map(int, self._complementary(list(map(str, list(set(l3B + list(map(int, self._complementary(list(map(str, ijk))))))))))))
        free_second_flipped = list(map(
            int, self._complementary(list(map(str, list(set(ijk + [l3Bs] + [l3Be] + list(map(int, self._complementary(list(set(map(str, l3B)))))))))))))
        if len(free) >= 2 or (len(free_first_flipped) < 2 and len(free_second_flipped) < 2):
            l3Bm_only, l3Bs_only, l3Be_only = None, False, False
            for i in l3Bm:
                if i not in ijk:
                    l3Bm_only = i
            if l3Bs not in ijk:
                l3Bs_only = True
            elif l3Be not in ijk:
                l3Be_only = True
            if l3Bs_only is True:                         # set s_abc first then ⟨i|(j+k)|l] (mode=1)
                use_mode = 1
            elif l3Be_only is True:                       # set s_abc first then [l|(j+k)|i⟩ (mode=1)
                l3Bc = ""                                 # flip it
                for i in range(len(l3Bm)):
                    l3Bc += "{}+".format(l3Bm[i])
                l3Bc = l3Bc[:-1]
                if t_s1[0] == '⟨':
                    t_s1 = '[{}|({})|{}⟩'.format(l3Be, l3Bc, l3Bs)
                else:
                    t_s1 = '⟨{}|({})|{}]'.format(l3Be, l3Bc, l3Bs)
                use_mode = 1
            elif l3Bm_only is not None:
                l3Bm.remove(l3Bm_only)                    # place the l3Bm_only at the beginning of middle (use mode=2)
                l3Bm = [l3Bm_only] + l3Bm
                l3Bc = ""
                for i in range(len(l3Bm)):
                    l3Bc += "{}+".format(l3Bm[i])
                l3Bc = l3Bc[:-1]
                if t_s1[0] == '⟨':
                    t_s1 = '⟨{}|({})|{}]'.format(l3Bs, l3Bc, l3Be)
                else:
                    t_s1 = '[{}|({})|{}⟩'.format(l3Bs, l3Bc, l3Be)
                use_mode = 2
            else:                                         # ijk contained in l3B and viceversa
                t_s1 = t_s2                               # this is the s_ijk, s_ij+s_ik, hence s_jk case basically
                t_s2 = '⟨{}|{}⟩'.format(l3Bm[0], l3Bm[1])
                self._set_pair_inner(t_s2, t_v2, t_s1, t_v1)
                return [t_s2, t_s1]                       # cheeky fix
        elif len(free_first_flipped) >= 2:                # flip it and try again
            t_s2 = 's_{}'.format(''.join(self._complementary(list(map(str, ijk)))))
            return self._set_pair_inner(t_s2, t_v2, t_s1, t_v1)
        elif len(free_second_flipped) >= 2:               # flip it and try again
            comp = list(map(int, self._complementary(list(map(str, list(set(l3B)))))))
            l3Bc = ""
            for i in range(len(comp)):
                l3Bc += "{}+".format(comp[i])
            l3Bc = l3Bc[:-1]
            if t_s1[0] == "⟨":
                t_s1 = "⟨{}|({})|{}]".format(l3Bs, l3Bc, l3Be)
            else:
                t_s1 = "[{}|({})|{}⟩".format(l3Bs, l3Bc, l3Be)
            return self._set_pair_inner(t_s2, t_v2, t_s1, t_v1)
        l3B = list(p3B.findall(t_s1)[0])                  # recompute this stuff since it changes sometimes
        l3B[1] = l3B[1].split("+")
        l3Bs, l3Bm, l3Be = int(l3B[0]), list(map(int, l3B[1])), int(l3B[2])
        l3B = list(map(int, [item for sublist in l3B for item in sublist]))
        free = list(map(int, self._complementary(list(map(str, list(set(ijk + l3B)))))))
        self._set(t_s2, t_v2, fix_mom=False)               # set it
        self._set(t_s1, t_v1, fix_mom=False, mode=use_mode)
        if len(free) >= 2:
            self.fix_mom_cons(free[0], free[1])
        elif len(free) == 1:
            if l3Bs not in ijk and t_s1[0] == "⟨":
                self.fix_mom_cons(free[0], l3Bs, real_momenta=False, axis=2)
            elif l3Bs not in ijk and t_s1[0] == "[":
                self.fix_mom_cons(free[0], l3Bs, real_momenta=False, axis=1)
            elif l3Be not in ijk and t_s1[-1] == "⟩":
                self.fix_mom_cons(free[0], l3Be, real_momenta=False, axis=2)
            elif l3Be not in ijk and t_s1[-1] == "]":
                self.fix_mom_cons(free[0], l3Be, real_momenta=False, axis=1)
        else:
            self.fix_mom_cons(ijk[0], ijk[1])
            if not self.momentum_conservation_check():
                raise myException("Not enough particles to fix mom cons!")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_pair_3B_and_Dijk(self, t_s1, t_v1, t_s2, t_v2):   # Current failed: 38/168 @ 6pt

        def unpack(t_s):
            l3B = list(p3B.findall(t_s)[0])
            l3B[1] = l3B[1].split("+")
            l3Bs, l3Bm, l3Be = l3B[0], l3B[1], l3B[2]
            l3B = [item for sublist in l3B for item in sublist]
            l3Bc = ""
            a_or_s = t_s[0]
            for i in range(len(l3Bm)):
                l3Bc += "{}+".format(l3Bm[i])
            l3Bc = l3Bc[:-1]
            return list(map(int, l3B)), int(l3Bs), list(map(int, l3Bm)), int(l3Be), a_or_s, l3Bc

        def flip_using_mom_cons(t_s):
            l3B, l3Bs, l3Bm, l3Be, a_or_s, l3Bc = unpack(t_s)
            l3Bm = self._complementary(list(set([l3Bs] + [l3Be] + l3Bm)))
            l3Bc = ""
            for i in range(len(l3Bm)):
                l3Bc += "{}+".format(l3Bm[i])
            l3Bc = l3Bc[:-1]
            if a_or_s == '⟨':
                t_s = '⟨{}|({})|{}]'.format(l3Bs, l3Bc, l3Be)
            elif a_or_s == '[':
                t_s = '[{}|({})|{}⟩'.format(l3Bs, l3Bc, l3Be)
            return t_s

        def reverse_order(t_s):
            l3B, l3Bs, l3Bm, l3Be, a_or_s, l3Bc = unpack(t_s)
            if a_or_s == '⟨':
                t_s = '[{}|({})|{}⟩'.format(l3Be, l3Bc, l3Bs)
            elif a_or_s == '[':
                t_s = '⟨{}|({})|{}]'.format(l3Be, l3Bc, l3Bs)
            return t_s

        def reorder_invariants(t_s1, t_s2):
            l3B, l3Bs, l3Bm, l3Be, a_or_s, l3Bc = unpack(t_s1)
            match_list = pDijk.findall(t_s2)[0]
            if match_list[0] == '':
                adjacent = False
                NonOverlappingLists = [list(map(int, corner)) for corner in match_list[1:]]
            else:
                adjacent = True
                NonOverlappingLists = self.ijk_to_3NonOverlappingLists(list(map(int, match_list[0])))
            if l3Bs == l3Be:
                for NonOverlappingList in NonOverlappingLists:
                    if l3Bs in NonOverlappingList and [entry for entry in NonOverlappingList if entry != l3Bs][0] in l3Bm:
                        NonOverlappingLists.remove(NonOverlappingList)
                        NonOverlappingLists += [NonOverlappingList]
                        break
                    elif l3Bs in NonOverlappingList:
                        NonOverlappingLists.remove(NonOverlappingList)
                        NonOverlappingLists = [NonOverlappingList] + NonOverlappingLists
                        break
            else:
                for NonOverlappingList in NonOverlappingLists:
                    if l3Bs == NonOverlappingList[0]:
                        NonOverlappingLists.remove(NonOverlappingList)
                        NonOverlappingLists = [NonOverlappingList] + NonOverlappingLists
                        break
                    elif l3Be == NonOverlappingList[0]:
                        NonOverlappingLists.remove(NonOverlappingList)
                        NonOverlappingLists = [NonOverlappingList] + NonOverlappingLists
                        t_s1 = reverse_order(t_s1)
                        break
            l3B, l3Bs, l3Bm, l3Be, a_or_s, l3Bc = unpack(t_s1)
            for NonOverlappingList in NonOverlappingLists:
                for entry in l3B[:-1]:
                    if entry in NonOverlappingList:
                        break
                else:
                    NonOverlappingLists.remove(NonOverlappingList)
                    NonOverlappingLists += [NonOverlappingList]
                    if adjacent is True:
                        t_s2 = "Δ_{}{}{}".format(NonOverlappingLists[0][0], NonOverlappingLists[1][0], NonOverlappingLists[2][0])
                    else:
                        t_s2 = "Δ_{}|{}|{}".format("".join(map(str, NonOverlappingLists[0])), "".join(map(str, NonOverlappingLists[1])), "".join(map(str, NonOverlappingLists[2])))
                    return t_s1, t_s2
            return False

        def there_are_two_free_particles(*args):
            if reorder_invariants(*args) is False:
                return False
            else:
                return True

        if there_are_two_free_particles(t_s1, t_s2):
            t_s1, t_s2 = reorder_invariants(t_s1, t_s2)
        elif there_are_two_free_particles(flip_using_mom_cons(t_s1), t_s2):
            t_s1 = flip_using_mom_cons(t_s1)
            t_s1, t_s2 = reorder_invariants(t_s1, t_s2)
        else:
            return "Not implemented."

        l3B, l3Bs, l3Bm, l3Be, a_or_s, l3Bc = unpack(t_s1)
        match_list = pDijk.findall(t_s2)[0]
        if match_list[0] == '':
            NonOverlappingLists = [list(map(int, corner)) for corner in match_list[1:]]
        else:
            NonOverlappingLists = self.ijk_to_3NonOverlappingLists(list(map(int, match_list[0])))

        self._set(t_s1, t_v1, fix_mom=False)
        if t_s1[0] == "⟨":
            if NonOverlappingLists[0][0] in l3B[1:]:
                self._set(t_s2, t_v2, fix_mom=False, mode=3)
            else:
                self._set(t_s2, t_v2, fix_mom=False, mode=2)
            self.fix_mom_cons(NonOverlappingLists[2][0], NonOverlappingLists[2][1], real_momenta=False, axis=1)
        else:
            self._set(t_s2, t_v2, fix_mom=False, mode=1)
            self.fix_mom_cons(NonOverlappingLists[2][0], NonOverlappingLists[2][1], real_momenta=False, axis=2)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_pair_Sijk_and_Sijk(self, t_s1, t_v1, t_s2, t_v2):   # Current failed: 0/90 @ 6pt

        ijk = list(map(int, list(pSijk.findall(t_s1)[0])))
        lmn = list(map(int, list(pSijk.findall(t_s2)[0])))
        ovrlap, ijko, lmno = [], [], []                   # in both ijk and lmn, only in ijk, only in lmn
        for i in ijk:
            if i in lmn:
                ovrlap += [i]
            else:
                ijko += [i]
        for l in lmn:
            if l in ijk:
                pass
            else:
                lmno += [l]
        plist = list(map(int, self._complementary(list(map(str, ovrlap + ijko + lmno)))))
        if len(plist) >= 2:                               # if there are at least two free particles then
            t_s1 = 's_'                                   # just make sure the overlap is at the end
            for i in range(len(ijko)):                    # and use mode=1/mode=2 for the first/second respectively
                t_s1 += '{}'.format(ijko[i])              # to avoid sending additional stuff to zero
            for i in range(len(ovrlap)):
                t_s1 += '{}'.format(ovrlap[i])
            t_s2 = 's_'
            for i in range(len(lmno)):
                t_s2 += '{}'.format(lmno[i])
            for i in range(len(ovrlap)):
                t_s2 += '{}'.format(ovrlap[i])
        else:                                             # otherwise if there are not enough free particles flip the longest
            if len(ijk) > len(lmn):                       # make sure lmn is the longest
                lmn, ijk, t_s1 = ijk, lmn, t_s2
            lmn = list(map(int, self._complementary(map(str, lmn))))
            t_s2 = 's_'
            for i in range(len(lmn)):
                t_s2 += '{}'.format(lmn[i])
            return self._set_pair_inner(t_s1, t_v1, t_s2, t_v2)
        self._set(t_s1, t_v1, fix_mom=False, mode=1)       # set it
        self._set(t_s2, t_v2, fix_mom=False, mode=2)
        if len(plist) >= 2:
            self.fix_mom_cons(plist[0], plist[1])
        else:
            self.fix_mom_cons(ijk[0], ijk[1])
            if not self.momentum_conservation_check():
                raise myException("Not enough particles to fix mom cons!")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_pair_Sijk_and_Dijk(self, t_s1, t_v1, t_s2, t_v2):  # Current failed: 8/20 @ 6pt

        from .DoubleCollinearLimit_D_and_S_a import expression_for_a
        from .DoubleCollinearLimit_D_and_S_e import expression_for_e

        Sijk = list(map(int, pSijk.findall(t_s1)[0]))
        match_list = pDijk.findall(t_s2)[0]
        if match_list[0] == '':
            NonOverlappingLists = [list(map(int, corner)) for corner in match_list[1:]]
        else:
            NonOverlappingLists = self.ijk_to_3NonOverlappingLists(list(map(int, match_list[0])))

        FirstNonOverlappingList = -1
        SecondNonOverlappingList = -1
        for i, NonOverlappingList in enumerate(NonOverlappingLists):
            if all([entry in Sijk for entry in NonOverlappingList]):
                FirstNonOverlappingList = i
                break
        if FirstNonOverlappingList != -1:
            for i, NonOverlappingList in enumerate(NonOverlappingLists):
                if i == FirstNonOverlappingList:
                    continue
                if NonOverlappingList[0] in Sijk:
                    SecondNonOverlappingList = i
                    break

        if FirstNonOverlappingList == -1 or SecondNonOverlappingList == -1:
            Sijk = self._complementary(Sijk)
            FirstNonOverlappingList = -1
            SecondNonOverlappingList = -1
            for i, NonOverlappingList in enumerate(NonOverlappingLists):
                if all([entry in Sijk for entry in NonOverlappingList]):
                    FirstNonOverlappingList = i
                    break
            if FirstNonOverlappingList != -1:
                for i, NonOverlappingList in enumerate(NonOverlappingLists):
                    if i == FirstNonOverlappingList:
                        continue
                    if NonOverlappingList[0] in Sijk:
                        SecondNonOverlappingList = i
                        break

        if FirstNonOverlappingList == -1 or SecondNonOverlappingList == -1:
            return "Not implemented."

        A, B, = NonOverlappingLists[FirstNonOverlappingList][0], NonOverlappingLists[FirstNonOverlappingList][1]
        C, D = NonOverlappingLists[SecondNonOverlappingList][0], NonOverlappingLists[SecondNonOverlappingList][1]
        if FirstNonOverlappingList > SecondNonOverlappingList:
            NonOverlappingLists.remove(NonOverlappingLists[FirstNonOverlappingList])
            NonOverlappingLists.remove(NonOverlappingLists[SecondNonOverlappingList])
        else:
            NonOverlappingLists.remove(NonOverlappingLists[SecondNonOverlappingList])
            NonOverlappingLists.remove(NonOverlappingLists[FirstNonOverlappingList])
        E, F = NonOverlappingLists[0][0], NonOverlappingLists[0][1]

        a, b = self[A].r_sp_d[0, 0], self[A].r_sp_d[1, 0]
        c, d = self[A].l_sp_d[0, 0], self[A].l_sp_d[0, 1]  # noqa --- used in eval
        e, f = self[B].r_sp_d[0, 0], self[B].r_sp_d[1, 0]
        g, h = self[B].l_sp_d[0, 0], self[B].l_sp_d[0, 1]  # noqa --- used in eval
        i, j = self[C].r_sp_d[0, 0], self[C].r_sp_d[1, 0]  # noqa --- used in eval
        k, l = self[C].l_sp_d[0, 0], self[C].l_sp_d[0, 1]  # noqa --- used in eval
        m, n = self[D].r_sp_d[0, 0], self[D].r_sp_d[1, 0]  # noqa --- used in eval
        o, p = self[D].l_sp_d[0, 0], self[D].l_sp_d[0, 1]  # noqa --- used in eval
        if isinstance(t_v1, float) and isinstance(t_v2, float):
            Y, X = mpmath.mpf(t_v1), mpmath.mpf(t_v2)      # noqa --- used in eval
        else:
            Y, X = t_v1, t_v2
        a, e = expression_for_a(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, Y, X), expression_for_e(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, Y, X)
        self[A].r_sp_d = numpy.array([a, b])
        self[B].r_sp_d = numpy.array([e, f])
        self.fix_mom_cons(E, F)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _set_pair_NB_and_Dijk(self, t_s1, t_v1, t_s2, t_v2):  # Current failed: 8/20 @ 6pt

        if t_s1 == "⟨1|3+4|5+6|1⟩" and t_s2 == "Δ_735":
            self._set("Δ_357", t_v2, fix_mom=False)
            self._set("⟨1|3+4|5+6|1⟩", t_v1, fix_mom=False)
            self.fix_mom_cons(7, 2)
        elif t_s1 == "⟨2|1+7|3+4|2⟩" and t_s2 == "Δ_735":
            self._set("⟨2|1+7|3+4|2⟩", t_v1, fix_mom=False)
            self._set("Δ_735", t_v2, mode=6, fix_mom=False)
            self.fix_mom_cons(5, 6)
        elif t_s1 == "⟨7|3+4|5+6|7⟩" and t_s2 == "Δ_735":
            self._set("Δ_357", t_v2, fix_mom=False)
            self._set("⟨7|3+4|5+6|7⟩", t_v1, fix_mom=False)
            self.fix_mom_cons(1, 2)
        elif t_s1 == "⟨7|1+2|5+6|7⟩" and t_s2 == "Δ_12|347|56":
            self._set("Δ_56|12|347", t_v2, fix_mom=False)
            self._set("⟨7|1+2|5+6|7⟩", t_v1, fix_mom=False)
            self.fix_mom_cons(3, 4)
        elif t_s1 == "[7|1+2|5+6|7]" and t_s2 == "Δ_12|347|56":
            self._set("Δ_56|12|347", t_v2, fix_mom=False)
            self._set("[7|1+2|5+6|7]", t_v1, fix_mom=False)
            self.fix_mom_cons(3, 4)
        elif t_s1 == "⟨3|5+6|1+2|3⟩" and t_s2 == "Δ_12|347|56":
            self._set("Δ_56|12|347", t_v2, fix_mom=False)
            self._set("⟨3|5+6|1+2|3⟩", t_v1, mode=1, fix_mom=False)
            self.fix_mom_cons(7, 4)
        elif t_s1 == "⟨7|3+4|5+6|7⟩" and t_s2 == "Δ_135":
            self._set("Δ_12|34|567", t_v2, fix_mom=False)
            self._set("⟨7|3+4|1+2|7⟩", t_v1, fix_mom=False)
            self.fix_mom_cons(5, 6)
        else:
            return "Not implemented."

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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
