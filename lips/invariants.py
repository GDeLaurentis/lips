#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   ___                  _          _                                    _
#  |_ _|_ ___ ____ _ _ _(_)__ _ _ _| |_ ___   __ _ ___ _ _  ___ _ _ __ _| |_ ___ _ _
#   | || ' \ V / _` | '_| / _` | ' \  _(_-<  / _` / -_) ' \/ -_) '_/ _` |  _/ _ \ '_|
#  |___|_||_\_/\__,_|_| |_\__,_|_||_\__/__/__\__, \___|_||_\___|_| \__,_|\__\___/_|
#                                        |___|___/

# Author: Giuseppe

import os
import re
import itertools
import shelve
import dbm

pA2 = re.compile(r'^(?:⟨)([0-9])(?:\|)([0-9])(?:⟩)$')
pNB = re.compile(r'^(?:⟨|\[)(?P<start>\d)(?:\|)(?P<middle>(?:(?:\([\d[\+|-]{1,}]{,1}\))|(?:[\d[\+|-]{1,}]{,1}))*)(?:\|)(?P<end>\d)(?:⟩|\])$')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Invariants(object):

    def __init__(self, n, no_cached=False, no_hard_coded_ones=False, Restrict3Brackets=True, Restrict4Brackets=True, FurtherRestrict4Brackets=True, verbose=False):

        self.multiplicity = n

        if no_cached is False and self.pw_invariants is not None and Restrict3Brackets is True and Restrict4Brackets is True:
            try:
                # Read From Cache
                with shelve.open(self.pw_invariants, 'r') as sInvariants:
                    if verbose:
                        print("Reading from cache")
                    if all(inv_type in sInvariants for inv_type in ["invs_2", "invs_3", "invs_4", "invs_5", "invs_s", "invs_D", "invs_O", "invs_P", "invs_tr5"]):
                        self.invs_2 = sInvariants[str("invs_2")]   # This is ⟨⟩ and []
                        self.invs_3 = sInvariants[str("invs_3")]   # This is ⟨|()|]
                        self.invs_4 = sInvariants[str("invs_4")]   # This is ⟨|()|()|⟩ [|()|()|]
                        self.invs_5 = sInvariants[str("invs_5")]   # This is ⟨|()|()|()|⟩ [|()|()|()|]
                        self.invs_s = sInvariants[str("invs_s")]   # This is s_ijk..
                        self.invs_D = sInvariants[str("invs_D")]   # This is Δ_ijk
                        self.invs_O = sInvariants[str("invs_O")]   # This is Ω_ijk
                        self.invs_P = sInvariants[str("invs_P")]   # This is Π_ijk
                        self.invs_tr5 = sInvariants[str("invs_tr5")]   # This is tr5
                    else:
                        raise Exception("Missing piece - regenerate invariants.")

            except dbm.error[0]:
                if verbose:
                    print("Regenerating from scratch")
                # Generate Them & Write To Cache
                with shelve.open(self.pw_invariants, 'c') as sInvariants:
                    self.GenerateFromScratch(n, no_hard_coded_ones,
                                             Restrict3Brackets=Restrict3Brackets, Restrict4Brackets=Restrict4Brackets, FurtherRestrict4Brackets=FurtherRestrict4Brackets)
                    sInvariants[str("invs_2")] = self.invs_2
                    sInvariants[str("invs_3")] = self.invs_3
                    sInvariants[str("invs_4")] = self.invs_4
                    sInvariants[str("invs_5")] = self.invs_5
                    sInvariants[str("invs_s")] = self.invs_s
                    sInvariants[str("invs_D")] = self.invs_D
                    sInvariants[str("invs_O")] = self.invs_O
                    sInvariants[str("invs_P")] = self.invs_P
                    sInvariants[str("invs_tr5")] = self.invs_tr5
        else:
            self.GenerateFromScratch(n, no_hard_coded_ones, Restrict3Brackets=Restrict3Brackets, Restrict4Brackets=Restrict4Brackets, FurtherRestrict4Brackets=FurtherRestrict4Brackets)

    def GenerateFromScratch(self, n, no_hard_coded_ones=False, Restrict3Brackets=True, Restrict4Brackets=True, FurtherRestrict4Brackets=True):
        self.invs_2 = all_strings(n, "2")   # This is ⟨⟩ and []
        self.invs_3 = all_strings(n, "3")   # This is ⟨|()|]
        self.invs_4 = all_strings(n, "4")   # This is ⟨|()|()|⟩ [|()|()|]
        self.invs_5 = all_strings(n, "5")   # This is ⟨|()|()|()|⟩ [|()|()|()|]
        self.invs_s = all_strings(n, "s")   # This is s_ijk..
        self.invs_D = all_strings(n, "D")   # This is Δ_ijk
        self.invs_O = all_strings(n, "O")   # This is Ω_ijk
        self.invs_P = all_strings(n, "P")   # This is Π_ijk
        self.invs_tr5 = all_strings(n, "tr5")   # This is tr5

        # Remove all non minimal, non neighbouring 3 Bracktes
        if Restrict3Brackets is True:
            Purge3Brackets(n, self.invs_3)

        # Remove all non minimal, non neighbouring 4 Bracktes
        if Restrict4Brackets is True:
            Purge4Brackets(n, self.invs_4, FurtherRestrict4Brackets)

        if n == 6 and no_hard_coded_ones is False:
            from lips import Particles
            oParticles = Particles(n)
            oInvariants = Invariants(n, no_cached=True, no_hard_coded_ones=True,
                                     Restrict3Brackets=Restrict3Brackets, Restrict4Brackets=Restrict4Brackets, FurtherRestrict4Brackets=FurtherRestrict4Brackets)
            invs_to_be_added = ["⟨2|(1+3)|5]", "⟨3|(1+5)|4]", "⟨6|(1+3)|4]", "⟨6|(2+4)|3]", "⟨4|(3+5)|1]", "⟨2|(1+4)|5]", "⟨4|(2+5)|1]", "⟨6|(2+5)|3]", "⟨3|(4+6)|1]", ]
            for inv in invs_to_be_added:
                oParticles.randomise_all()
                oParticles._set(inv, 10 ** -30)
                mom_cons, on_shell, big_outliers, small_outliers = oParticles.phasespace_consistency_check(oInvariants.full)
                # print mom_cons, on_shell, big_outliers, small_outliers
                if mom_cons is True and on_shell is True and len(small_outliers) == 0:
                    self.invs_3 += [inv]
        if n == 6:
            self.invs_D += ["Δ_15|26|34", "Δ_14|23|56", "Δ_13|24|56"]
        if n == 7:
            self.invs_4 += ["⟨1|(5+6)|(2+7)|1⟩", "[1|(5+6)|(2+7)|1]", "Δ_12|347|56", ]

    @property
    def full(self):
        return self.invs_2 + self.invs_3 + self.invs_4 + self.invs_5 + self.invs_s + self.invs_D + self.invs_O + self.invs_P + self.invs_tr5

    @property
    def full_minus_4_brackets(self):
        return self.invs_2 + self.invs_3 + self.invs_5 + self.invs_s + self.invs_D + self.invs_O + self.invs_P + self.invs_tr5

    @property
    def invs_N(self):
        return self.invs_3 + self.invs_4 + self.invs_5

    @property
    def pw_cache(self):
        home = os.path.expanduser("~")
        pw_cache = home + "/.cache/lips/n={}".format(self.multiplicity)
        if not os.path.exists(pw_cache):
            os.makedirs(pw_cache)
        return pw_cache

    @property
    def pw_invariants(self):
        return self.pw_cache + "/invariants"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def all_strings(n, temp_string):
    result = []

    if temp_string[0] == "s" or temp_string[0] == "S":                          # s_ijk...
        for m in range(3, n // 2 + 1):                                               # m is the number of particles in the invariant
            string = 's_'                                                       # eg: m of s_ijk is 3, m of s_ijkl is 4
            recurse_s_ijk_gen(n, 0, m, result, string)
        return result

    if temp_string[0] == "D":                                                   # D_ijk
        if n >= 6:
            return generate_D_ijk(n)
        else:
            return []

    if temp_string[0] == "O":                                                   # O_ijk
        if n == 6:
            return generate_O_ijk(n)
        else:
            return []

    if temp_string[0] == "P":                                                   # P_ijk
        if n == 6:
            return [entry.replace("Ω", "Π") for entry in generate_O_ijk(n)]
        else:
            return []

    if len(temp_string) >= 3 and temp_string[0:3] == "tr5":                     # tr5
        if n >= 5:
            return generate_tr5_ijkls(n)
        else:
            return []

    if temp_string[0] == "5":
        if n >= 7:
            return generate_5Brackets(n)
        else:
            return []

    nbr_brks = int(temp_string[0])                                              # number of brackets ⟨a|b⟩ is 2, ⟨a|(b+c)|d] is 3, etc...
    for first in range(1, n + 1):                                               # all possible first elements
        for last in range(1, n + 1):                                            # all possible last elements
            if nbr_brks % 2 == 0 and first > last:                              # if the number of brackets is even there is some symmetry, avoid overcounting
                continue
            for counter in range(2 - (nbr_brks % 2)):                           # if the number of brackets is even need to consider both ⟨⟩ []
                if counter == 0:                                                # else only ⟨] since [⟩ is taken care of automatically
                    start = "⟨{}|".format(first)                                # hence: counter = 0 only for odd, counter = (0, 1) for even
                elif counter == 1:
                    start = "[{}|".format(first)
                if nbr_brks % 2 == 0 and counter == 0:
                    end = "|{}⟩".format(last)
                elif nbr_brks % 2 == 1 and counter == 0:
                    end = "|{}]".format(last)
                elif nbr_brks % 2 == 0 and counter == 1:
                    end = "|{}]".format(last)
                if nbr_brks == 2:                                               # if it's a 2 braket then no "middle" exists
                    end = end[1:]
                    if first < last:                                            # avoid overcounting
                        result += [start + end]
                middles = []                                                    # if a "middle" exists then construct it
                for nbr_m_brks in range(1, nbr_brks - 1):                       # loop over the number of brackets in the middle
                    _middles = []                                               # loop over all possible lengths of the brackets
                    for length in range(2, n // 2 + 1):                         # keep in mind that it could be replaced with the complementary list
                        if nbr_m_brks == 1 and nbr_brks == 3:
                            comb = _inner_bracket(n, length, first, last)
                        elif nbr_m_brks == 1 and nbr_brks > 3:
                            comb = _inner_bracket(n, length, first)
                        elif nbr_m_brks == nbr_brks - 2 and nbr_brks > 3:
                            comb = _inner_bracket(n, length, last)
                        else:
                            comb = _inner_bracket(n, length)
                        if comb is None:
                            break
                        for element1 in comb:                                   # concatenate strings
                            _middle = "("
                            for element2 in element1:
                                _middle += element2
                                _middle += "+"
                            _middle = _middle[:-1]
                            _middle += ")"
                            if nbr_m_brks == 1:
                                _middles += [_middle]
                            else:
                                for middle in middles:
                                    if middle != _middle:
                                        if (nbr_m_brks == 2 and first == last):
                                            k = 1
                                            needs_skipping = False
                                            while (k < len(middle) and k < len(_middle)):
                                                if (int(middle[k]) > int(_middle[k])):
                                                    needs_skipping = True
                                                    break
                                                elif (int(middle[k]) < int(_middle[k])):
                                                    break
                                                k += 2
                                            if needs_skipping is True:
                                                continue
                                        _middles += [middle + "|" + _middle]
                    middles = [__middle for __middle in _middles]
                for middle in middles:
                    result += [start + middle + end]

    if n == 4 and nbr_brks == 2:
        result = list(filter(lambda inv: pA2.match(inv) is not None, result))

    # At this point the entries in result should be unique (mostly). However, they might be rewritten as product of easier stuff.. which is very bad.
    if nbr_brks == 4:                                                           # for now just take care of 4 brackets
        _result = [entry for entry in result]
        for invariant in _result:
            abcd = pNB.search(invariant)
            a = abcd.group('start')
            bc = abcd.group('middle')
            d = abcd.group('end')
            bc = re.split(r'[\)|\|]', bc)
            bc = [entry.replace('|', '') for entry in bc]
            bc = [entry.replace('(', '') for entry in bc]
            bc = [entry for entry in bc if entry != '']
            alt = ["+".join(_complementary(n, list(set(bc[0].split("+") + [a])))),
                   "+".join(_complementary(n, list(set(bc[1].split("+") + [d]))))]
            bc = [set(entry.split("+")) for entry in bc]
            alt = [set(entry.split("+")) for entry in alt]
            # eg: ⟨1|(2+3)|(2+3)|6⟩ ~ ⟨1|6⟩ s_23
            if bc[0] == bc[1] or bc[0] == alt[1] or alt[0] == bc[1] or alt[0] == alt[1]:
                result.remove(invariant)
            # eg: ⟨1|(2+3+4)|(2+3)|1⟩ ~ ⟨1|4⟩[4|2+3|1⟩
            # however: ⟨2|(3+4)|(5+6)|2⟩ = ⟨2|(1+7)|(5+6)|2⟩ = ⟨2|(3+4)|(1+7)|2⟩ must be included once
            elif a == d and ((bc[0].issubset(bc[1]) and len(bc[1] - bc[0]) == 1) or
                             (bc[0].issubset(alt[1]) and len(alt[1] - bc[0]) == 1) or
                             (bc[1].issubset(bc[0]) and len(bc[0] - bc[1]) == 1) or
                             (bc[1].issubset(alt[0]) and len(alt[0] - bc[1]) == 1)):
                result.remove(invariant)
            # eg: ⟨1|(3+4)|(2+5)|6⟩ = ⟨1|(3+4)|(1+3+4)|6⟩
            elif ((len(list(bc[0] - bc[1])) == 1 and
                   list(bc[0] - bc[1])[0] == d) or
                  (len(list(bc[0] - alt[1])) == 1 and
                   list(bc[0] - alt[1])[0] == d) or
                  (len(list(alt[0] - bc[1])) == 1 and
                   list(alt[0] - bc[1])[0] == d) or
                  (len(list(alt[0] - alt[1])) == 1 and
                   list(alt[0] - alt[1])[0] == d) or
                  (len(list(bc[1] - bc[0])) == 1 and
                   list(bc[1] - bc[0])[0] == a) or
                  (len(list(bc[1] - alt[0])) == 1 and
                   list(bc[1] - alt[0])[0] == a) or
                  (len(list(alt[1] - bc[0])) == 1 and
                   list(alt[1] - bc[0])[0] == a) or
                  (len(list(alt[1] - alt[0])) == 1 and
                   list(alt[1] - alt[0])[0] == a)):
                result.remove(invariant)
        _result = [entry for entry in result]
        for i, iInv in enumerate(_result):                                      # remove some doubles, eg: '⟨2|(3+4)|(5+6)|2⟩' = '⟨2|(1+7)|(5+6)|2⟩'
            abcd = pNB.search(iInv)
            a = abcd.group('start')
            bc = abcd.group('middle')
            d = abcd.group('end')
            if a != d:
                continue
            bc = re.split(r'[\)|\|]', bc)
            bc = [entry.replace('|', '') for entry in bc]
            bc = [entry.replace('(', '') for entry in bc]
            bc = [entry for entry in bc if entry != '']
            alt = ["+".join(_complementary(n, list(set(bc[0].split("+") + [a])))), "+".join(_complementary(n, list(set(bc[1].split("+") + [d]))))]
            bc = [set(entry.split("+")) for entry in bc]
            alt = [set(entry.split("+")) for entry in alt]
            if bc[0].issubset(alt[1]):
                new_alt = alt[1] - bc[0]
                new_alt = "+".join(new_alt)
                new_bc = bc[0]
                new_bc = "+".join(new_bc)
                if iInv[0] == "⟨":
                    equal_inv = "⟨{}|({})|({})|{}⟩".format(a, new_bc, new_alt, d)
                else:
                    equal_inv = "[{}|({})|({})|{}]".format(a, new_bc, new_alt, d)
                if equal_inv in result:
                    if Brackets4IsIndividuallyNeighbouring(n, equal_inv) is True and Brackets4IsIndividuallyNeighbouring(n, iInv) is True:
                        if _result.index(equal_inv) > i:
                            result.remove(equal_inv)
                    elif Brackets4IsIndividuallyNeighbouring(n, equal_inv) is True and Brackets4IsIndividuallyNeighbouring(n, iInv) is False:
                        result.remove(iInv)
                    elif Brackets4IsIndividuallyNeighbouring(n, equal_inv) is False and Brackets4IsIndividuallyNeighbouring(n, iInv) is False:
                        result.remove(equal_inv)
            if bc[1].issubset(alt[0]):
                new_alt = alt[0] - bc[1]
                new_alt = "+".join(new_alt)
                new_bc = bc[1]
                new_bc = "+".join(new_bc)
                if iInv[0] == "⟨":
                    equal_inv = "⟨{}|({})|({})|{}⟩".format(a, new_alt, new_bc, d)
                else:
                    equal_inv = "[{}|({})|({})|{}]".format(a, new_alt, new_bc, d)
                if equal_inv in result and _result.index(equal_inv) > i:
                    result.remove(equal_inv)
    return result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def recurse_s_ijk_gen(n, i, m, result, string):
    for j in range(i + 1, n + 1):
        # avoid double counting by introducing identical invariants
        if n % 2 == 0 and len(string) == n // 2 + 1:
            if j == n:
                return
        string += "{}".format(j)
        if m > 1:
            recurse_s_ijk_gen(n, j, m - 1, result, string)
        elif m == 1:
            result += [string]
        string = string[:-1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def generate_D_ijk(n):
    lDeltas = []
    for i in range(2, n - 3):
        for j in range(i):
            start1 = -j % n + 1
            start2 = i - j + 1
            for k in range(2, n - i - 1):
                start3 = start2 + k
                lDeltas += ["Δ_{}{}{}".format(start1, start2, start3)]
    return lDeltas


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def generate_O_ijk(n):
    lDeltas = generate_D_ijk(n)          # generate deltas first
    lOmegas = []
    for delta in lDeltas:
        lOmegas += ["Ω_{}{}{}".format(delta[2:][0], delta[2:][1], delta[2:][2]),
                    "Ω_{}{}{}".format(delta[2:][1], delta[2:][2], delta[2:][0]),
                    "Ω_{}{}{}".format(delta[2:][2], delta[2:][0], delta[2:][1])]
    return lOmegas


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def generate_tr5_ijkls(n):
    tr5_ijkls = ["tr5_" + "".join(entry) for entry in itertools.combinations("".join(map(str, range(1, n))), 4)]
    return tr5_ijkls


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def generate_5Brackets(n):
    l5Brackets = []
    for i in range(1, n + 1):
        particles = list(range(1, n + 1))
        particles.remove(i)
        for kk in range(len(particles)):
            if kk != 0:
                particles = particles[1:] + [particles[0]]
            for j in range(2, n - 4):
                middle1 = [entry for ii, entry in enumerate(particles) if ii < j]
                if CheckIfNeighbouring(n, middle1) is False:
                    continue
                for k in range(2, n - 2 - j):
                    middle2 = [entry for ii, entry in enumerate(particles) if (ii >= j and ii < j + k)]
                    if CheckIfNeighbouring(n, middle2) is False:
                        continue
                    middle3 = list(set(particles) - set(middle1) - set(middle2))
                    if CheckIfNeighbouring(n, middle3) is False:
                        continue
                    s_middle1 = "+".join(map(str, middle1))
                    s_middle2 = "+".join(map(str, middle2))
                    s_middle3 = "+".join(map(str, middle3))
                    l5Brackets += ["⟨{}|({})|({})|({})|{}]".format(i, s_middle1, s_middle2, s_middle3, i)]
    return l5Brackets


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def CheckIfNeighbouring(n, middle):
    a = middle[0]
    near = [a, a - 1, a + 1]
    if a == n:
        near += [1]
    if a == 1:
        near += [n]
    for j in range(len(middle)):
        for i in range(len(middle)):
            if middle[i] in near:
                near += [middle[i] + 1, middle[i] - 1]
                if middle[i] == n:
                    near += [1]
                if middle[i] == 1:
                    near += [n]
    if all(entry in near for entry in middle):
        return True
    else:
        return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def Purge4Brackets(n, invs_4, FurtherRestrict4Brackets):
    _invs_4 = [entry for entry in invs_4]
    for inv in _invs_4:
        abcd = pNB.search(inv)
        a = int(abcd.group('start'))
        bc = abcd.group('middle')
        d = int(abcd.group('end'))
        bc = re.split(r'[\)|\|]', bc)
        bc = [entry.replace('|', '') for entry in bc]
        bc = [entry.replace('(', '') for entry in bc]
        bc = [entry for entry in bc if entry != '']
        minimal = True                                                          # check if it is minimal  <---- some may be required, like for 3Brackets?
        for entry in bc:
            if len(entry) > 3:
                minimal = False
                invs_4.remove(inv)
                break
        if minimal is False:
            continue
        bc = [entry.split("+") for entry in bc]
        bc = [map(int, entry) for entry in bc]
        if FurtherRestrict4Brackets is False:                                   # check if it is neighbouring
            near = [a, a + 1, a - 1]
            if a == n:
                near += [1]
            if a == 1:
                near += [n]
            for i in range(5):
                if d in near:
                    near += [d + 1, d - 1]
                    if d == n:
                        near += [1]
                    if d == 1:
                        near += [n]
                for j in range(2):
                    if bc[j][0] in near:
                        near += [bc[j][0] + 1, bc[j][0] - 1]
                        if bc[j][0] == n:
                            near += [1]
                        if bc[j][0] == 1:
                            near += [n]
                    if bc[j][1] in near:
                        near += [bc[j][1] + 1, bc[j][1] - 1]
                        if bc[j][1] == n:
                            near += [1]
                        if bc[j][1] == 1:
                            near += [n]
            if (d not in near):
                invs_4.remove(inv)
                continue
            for i in range(2):
                if (bc[i][0] not in near or bc[i][1] not in near):
                    invs_4.remove(inv)
                    break
        elif FurtherRestrict4Brackets is True:
            if Brackets4IsIndividuallyNeighbouring(n, inv) is False:
                invs_4.remove(inv)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def Brackets4IsIndividuallyNeighbouring(n, invariant):
    abcd = pNB.search(invariant)
    a = int(abcd.group('start'))
    bc = abcd.group('middle')
    d = int(abcd.group('end'))
    bc = re.split(r'[\)|\|]', bc)
    bc = [entry.replace('|', '') for entry in bc]
    bc = [entry.replace('(', '') for entry in bc]
    bc = [entry for entry in bc if entry != '']
    bc = [entry.split("+") for entry in bc]
    bc = [list(map(int, entry)) for entry in bc]
    near = [a, a + 1, a - 1]
    if a == n:
        near += [1]
    if a == 1:
        near += [n]
    for i in range(2):
        if bc[0][0] in near:
            near += [bc[0][0] + 1, bc[0][0] - 1]
            if bc[0][0] == n:
                near += [1]
            if bc[0][0] == 1:
                near += [n]
        if bc[0][1] in near:
            near += [bc[0][1] + 1, bc[0][1] - 1]
            if bc[0][1] == n:
                near += [1]
            if bc[0][1] == 1:
                near += [n]
    if (bc[0][0] not in near or bc[0][1] not in near):
        return False
    near = [d, d + 1, d - 1]
    if d == n:
        near += [1]
    if d == 1:
        near += [n]
    for i in range(2):
        if bc[1][0] in near:
            near += [bc[1][0] + 1, bc[1][0] - 1]
            if bc[1][0] == n:
                near += [1]
            if bc[1][0] == 1:
                near += [n]
        if bc[1][1] in near:
            near += [bc[1][1] + 1, bc[1][1] - 1]
            if bc[1][1] == n:
                near += [1]
            if bc[1][1] == 1:
                near += [n]
    if (bc[1][0] not in near or bc[1][1] not in near):
        return False
    return True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def Purge3Brackets(n, invs_3):
    _invs_3 = [entry for entry in invs_3]
    for inv in _invs_3:
        abcd = pNB.search(inv)
        a = int(abcd.group('start'))
        bc = abcd.group('middle')
        d = int(abcd.group('end'))
        bc = re.split(r'[\)|\|]', bc)
        bc = [entry.replace('|', '') for entry in bc]
        bc = [entry.replace('(', '') for entry in bc]
        bc = [entry for entry in bc if entry != '']
        # # check if it is minimal       <-------- some 'non minimal' 3brackets are required
        # minimal = True
        # for entry in bc:
        #     if len(entry) > 3:
        #         minimal = False
        #         invs_3.remove(inv)
        #         break
        # if minimal is False:
        #     continue
        # check if it is neighbouring
        bc = [entry.split("+") for entry in bc]
        bc = [list(map(int, entry)) for entry in bc]
        bc_complementary = [list(map(int, _complementary(n, list(set(map(str, bc[0] + [a] + [d]))))))]
        if (Brackets3NeedsToBeRemoved(n, a, bc, d) is True and Brackets3NeedsToBeRemoved(n, a, bc_complementary, d) is True):
            invs_3.remove(inv)
            continue


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def Brackets3NeedsToBeRemoved(n, a, bc, d):
    near = [a, a + 1, a - 1]
    if a == n:
        near += [1]
    if a == 1:
        near += [n]
    for i in range(len(bc[0]) + 1):
        if d in near:
            near += [d, d + 1, d - 1]
            if d == n:
                near += [1]
            if d == 1:
                near += [n]
        for j in range(len(bc[0])):
            if bc[0][j] in near:
                near += [bc[0][j], bc[0][j] + 1, bc[0][j] - 1]
                if bc[0][j] == n:
                    near += [1]
                if bc[0][j] == 1:
                    near += [n]
    if (d not in near or
       True in [bc[0][i] not in near for i in range(len(bc[0]))]):
        return True
    return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _complementary(n, temp_list):
    c_list = []
    for i in range(1, n + 1):
        c_list += ["{}".format(i)]
    for element in temp_list:
        c_list.remove(element)
    return c_list


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _inner_bracket(n, length, adjacent1=0, adjacent2=0):
    # string for generation of combinations (up to n=9)
    c_string = ""
    for i in range(1, n + 1):
        c_string += "{}".format(i)

    # generate all possible combinations of given length
    comb = list(itertools.combinations(c_string, length))
    comb = list(list(tup) for tup in comb)

    # delte all combinations which simplify to shorter ones
    i = 0
    while i < len(comb):
        removed = False
        if str(adjacent1) in comb[i]:
            comb.remove(comb[i])
            removed = True
        elif str(adjacent2) in comb[i]:
            comb.remove(comb[i])
            removed = True
        if removed is True:
            i = i - 1
        i = i + 1
    # print comb

    # compute complementary length
    comp_length = n - length
    if adjacent1 != 0 and adjacent2 != 0:
        if adjacent1 == adjacent2:
            comp_length = comp_length - 1
        else:
            comp_length = comp_length - 2
    elif adjacent1 != 0 or adjacent2 != 0:
        comp_length = comp_length - 1

    # if same length remove repeted ones
    if comp_length == length:
        for element in comb:
            comp = _complementary(n, element)
            if str(adjacent1) in comp:
                comp.remove(str(adjacent1))
            if str(adjacent2) in comp:
                comp.remove(str(adjacent2))
            comb.remove(comp)
    elif comp_length < length:
        return None
    # print comb
    return comb


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
