# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from copy import deepcopy

from pyadic.padic import PAdic, full_range_random_padic_filling

from .tools import flatten, myException


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_Variety:

    # PUBLIC METHODS

    def variety(self, invariants, valuations, try_singular_variety_solver=True, verbose=False):
        """Constructs the required phase space point by first trying hardcoded limits and then the singular variety."""
        assert len(invariants) == len(valuations)
        try:
            if verbose:
                print("Trying hardcoded solutions...")
            if try_singular_variety_solver:
                self_backup = deepcopy(self)
            if self.field.name == 'padic':
                _valuations = tuple(PAdic(self.field.characteristic ** valuation * full_range_random_padic_filling(self.field.characteristic, self.field.digits - valuation),
                                          self.field.characteristic, self.field.digits, from_addition=True) for valuation in valuations)
            else:
                _valuations = valuations
            if len(invariants) == 1:
                self._set(invariants[0], _valuations[0])
            elif len(invariants) == 2:
                self._set_pair(invariants[0], _valuations[0], invariants[1], _valuations[1])
            else:
                raise NotImplementedError
        except Exception:
            if try_singular_variety_solver:
                for i, iParticle in enumerate(self_backup):
                    self[i + 1] = iParticle
                if verbose:
                    print("Trying singular variety solver...")
                self._singular_variety(invariants, valuations, verbose=verbose)

        if verbose:
            print("Checking result...")

        mom_cons, on_shell = self.momentum_conservation_check(), self.onshell_relation_check()
        # the following would be more thorough... is it needed tho?
        # mom_cons, on_shell, big_outliers, small_outliers = self.phasespace_consistency_check()
        if mom_cons is False:
            raise myException("Momentum conservation is not satisfied: ", max(map(abs, flatten(self.total_mom))))
        elif on_shell is False:
            raise myException("On shellness is not satisfied: ", max(map(abs, flatten(self.masses))))

        if self.field.characteristic == 0:
            abs_diffs = [min(abs(self(invariant) - valuation), abs(self(invariant) + valuation))
                         for (invariant, valuation) in zip(invariants, _valuations)]
            if not all([abs_diff <= self.field.tollerance for abs_diff in abs_diffs]):
                raise myException("Failed to set {} to {}. Instead got {}.".format(invariants, valuations, abs_diffs))
        else:
            numerical_invariants = [self(invariant) for invariant in invariants]
            abs_diffs = [abs(entry) for entry in numerical_invariants]
            if self.field.name == 'padic':
                abs_diffs = [abs_diff.n - valuation for abs_diff, valuation in zip(abs_diffs, valuations)]
            if not all(abs_diff == 0 for abs_diff in abs_diffs):
                raise myException("Failed to set {} to valuation {}. Instead got {}.".format(invariants, valuations, numerical_invariants))
