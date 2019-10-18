#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   ___          _   _    _
#  | _ \__ _ _ _| |_(_)__| |___
#  |  _/ _` | '_|  _| / _| / -_)
#  |_| \__,_|_|  \__|_\__|_\___|

# Author: Giuseppe


from __future__ import unicode_literals

import numpy
import mpmath

from tools import MinkowskiMetric, LeviCivita, rand_frac, Pauli, Pauli_bar

mpmath.mp.dps = 300


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particle(object):
    """Describes the kinematics of a single particle."""

    def __init__(self, four_mom=None, real_momentum=False):
        """Initialisation. Calls randomise if None, else initialises the four momentum."""
        if four_mom is None:
            self.randomise(real_momentum=False)
        else:
            self.four_mom = four_mom

    def __eq__(self, other):
        """Equality checks equality of four momenta."""
        if type(self) == type(other):
            return all(self.four_mom == other.four_mom)
        else:
            return False

    # GETTERS and SETTERS

    @property
    def r_sp_d(self):
        """Right spinor with index down: λ_α (column vector)."""
        return self._r_sp_d

    @r_sp_d.setter
    def r_sp_d(self, temp_r_sp_d):
        self._r_sp_d = temp_r_sp_d     # left spinor is unchanged.
        self._r_sp_d.shape = (2, 1)    # column vector
        self._r_sp_d_to_r_sp_u()
        self._r1_sp_to_r2_sp()
        self._r1_sp_to_r2_sp_b()
        self._r2_sp_b_to_four_momentum()
        self._four_mom_to_four_mom_d()

    @property
    def l_sp_d(self):
        """Left spinor with index down: λ̅_α̇ (row vector)."""
        return self._l_sp_d

    @l_sp_d.setter
    def l_sp_d(self, temp_l_sp_d):
        self._l_sp_d = temp_l_sp_d     # right spinor is unchanged.
        self._l_sp_d.shape = (1, 2)    # row vectorx
        self._l_sp_d_to_l_sp_u()
        self._r1_sp_to_r2_sp()
        self._r1_sp_to_r2_sp_b()
        self._r2_sp_b_to_four_momentum()
        self._four_mom_to_four_mom_d()

    @property
    def r_sp_u(self):
        """Right spinor with index up: λ^α (row vector)."""
        return self._r_sp_u

    @r_sp_u.setter
    def r_sp_u(self, temp_r_sp_u):
        self._r_sp_u = temp_r_sp_u     # left spinor is unchanged.
        self._r_sp_u.shape = (1, 2)    # row vector
        self._r_sp_u_to_r_sp_d()
        self._r1_sp_to_r2_sp()
        self._r1_sp_to_r2_sp_b()
        self._r2_sp_b_to_four_momentum()
        self._four_mom_to_four_mom_d()

    @property
    def l_sp_u(self):
        """Left spinor with index up: λ̅^α̇ (column vector)."""
        return self._l_sp_u

    @l_sp_u.setter
    def l_sp_u(self, temp_l_sp_u):
        self._l_sp_u = temp_l_sp_u     # right spinor is unchanged.
        self._l_sp_u.shape = (2, 1)    # column vector
        self._l_sp_u_to_l_sp_d()
        self._r1_sp_to_r2_sp()
        self._r1_sp_to_r2_sp_b()
        self._r2_sp_b_to_four_momentum()
        self._four_mom_to_four_mom_d()

    @property
    def r2_sp(self):
        """Four Momentum Slashed with upper indices: P^{α̇α}"""
        return self._r2_sp

    @r2_sp.setter
    def r2_sp(self, temp_r2_sp):
        self._r2_sp = temp_r2_sp
        self._r2_sp_to_four_momentum()
        self._four_mom_to_four_mom_d()
        self._four_mom_d_to_r2_sp_b()
        self._four_mom_to_r_sp_d()
        self._r_sp_d_to_r_sp_u()
        self._four_mom_to_l_sp_d()
        self._l_sp_d_to_l_sp_u()

    @property
    def r2_sp_b(self):
        """Four Momentum Slashed with lower indices: P̅\u0305_{αα̇}"""
        return self._r2_sp_b

    @r2_sp_b.setter
    def r2_sp_b(self, temp_r2_sp_b):
        self._r2_sp_b = temp_r2_sp_b
        self._r2_sp_b_to_four_momentum()
        self._four_mom_to_four_mom_d()
        self._four_mom_d_to_r2_sp()
        self._four_mom_to_r_sp_d()
        self._r_sp_d_to_r_sp_u()
        self._four_mom_to_l_sp_d()
        self._l_sp_d_to_l_sp_u()

    @property
    def four_mom(self):
        """Four Momentum with upper index: P^μ"""
        return self._four_mom

    @four_mom.setter
    def four_mom(self, temp_four_mom):
        self._four_mom = temp_four_mom
        self._four_mom_to_four_mom_d()
        self._four_mom_d_to_r2_sp()
        self._four_mom_d_to_r2_sp_b()
        self._four_mom_to_r_sp_d()
        self._r_sp_d_to_r_sp_u()
        self._four_mom_to_l_sp_d()
        self._l_sp_d_to_l_sp_u()

    @property
    def four_mom_d(self):
        """Four Momentum with lower index: P_μ"""
        return self._four_mom_d

    @four_mom_d.setter
    def four_mom_d(self, temp_four_mom_d):
        self._four_mom_d = temp_four_mom_d
        self._four_mom_d_to_four_mom()
        self._four_mom_d_to_r2_sp()
        self._four_mom_d_to_r2_sp_b()
        self._four_mom_to_r_sp_d()
        self._r_sp_d_to_r_sp_u()
        self._four_mom_to_l_sp_d()
        self._l_sp_d_to_l_sp_u()

    # PUBLIC METHODS

    def randomise(self, real_momentum=False):
        """Randomises its momentum."""
        while True:
            if real_momentum is False:
                p = [rand_frac() + 1j * rand_frac(), rand_frac() + 1j * rand_frac(), rand_frac() + 1j * rand_frac()]
            else:
                p = [rand_frac(), rand_frac(), rand_frac()]
            if not (abs(p[0]) == 0 and abs(p[1]) == 0 and p[2].real <= 0 and p[2].imag == 0):     # make sure it is not in the
                break                                                                             # negative z direction or null
        p2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
        p_zero = mpmath.sqrt(p2)
        self.four_mom = numpy.array([p_zero] + p)
        self._four_mom_to_four_mom_d()
        self._four_mom_d_to_r2_sp()
        self._four_mom_d_to_r2_sp_b()
        self._four_mom_to_r_sp_d()
        self._r_sp_d_to_r_sp_u()
        self._four_mom_to_l_sp_d()
        self._l_sp_d_to_l_sp_u()

    def angles_for_squares(self):
        """Flips left and right spinors."""
        self._l_sp_u, self._r_sp_u = self._r_sp_u, self._l_sp_u
        self._l_sp_u.shape = (2, 1)    # column vector
        self._r_sp_u.shape = (1, 2)    # column vector
        self._l_sp_u_to_l_sp_d()
        self._r_sp_u_to_r_sp_d()
        self._r1_sp_to_r2_sp()
        self._r1_sp_to_r2_sp_b()
        self._r2_sp_b_to_four_momentum()
        self._four_mom_to_four_mom_d()

    # PRIVATE METHODS

    def __neg__(self):
        minus_self = Particle()
        minus_self.four_mom = -1 * self.four_mom
        return minus_self

    def _four_mom_to_r_sp_d(self):     # r_sp_d is \lambda_\alpha
        lambda_one = mpmath.sqrt(self._four_mom[0] + self._four_mom[3])
        if abs(lambda_one) == 0:
            if self._four_mom[1] == 0 and self._four_mom[2] == 0:
                lambda_two = lambda_one  # i.e. zero
            else:
                raise ValueError("Encountered zero denominator in spinor.")
        else:
            lambda_two = (self._four_mom[1] + self._four_mom[2] * 1j) / lambda_one
        self._r_sp_d = numpy.array([lambda_one, lambda_two])
        self._r_sp_d.shape = (2, 1)    # column vector

    def _four_mom_to_l_sp_d(self):     # l_sp_d is \bar{\lambda}_{\dot\alpha}
        lambdabar_one = mpmath.sqrt(self._four_mom[0] + self._four_mom[3])
        if abs(lambdabar_one) == 0:
            if self._four_mom[1] == 0 and self._four_mom[2] == 0:
                lambdabar_two = lambdabar_one  # i.e. zero
            else:
                raise ValueError("Encountered zero denominator in spinor.")
        else:
            lambdabar_two = (self._four_mom[1] - self._four_mom[2] * 1j) / lambdabar_one
        self._l_sp_d = numpy.array([lambdabar_one, lambdabar_two])
        self._l_sp_d.shape = (1, 2)    # row vector

    def _r2_sp_to_four_momentum(self):
        for i in range(4):
            self._four_mom[i] = numpy.trace(numpy.dot(Pauli_bar[i], self.r2_sp)) / 2

    def _r2_sp_b_to_four_momentum(self):
        for i in range(4):
            self._four_mom[i] = numpy.trace(numpy.dot(Pauli[i], self.r2_sp_b)) / 2

    def _four_mom_to_four_mom_d(self):
        self._four_mom_d = numpy.dot(MinkowskiMetric, self.four_mom)

    def _four_mom_d_to_four_mom(self):
        self._four_mom = numpy.dot(MinkowskiMetric, self.four_mom_d)

    def _four_mom_d_to_r2_sp(self):
        self._r2_sp = numpy.tensordot(self._four_mom_d, Pauli, axes=(0, 0))

    def _four_mom_d_to_r2_sp_b(self):
        self._r2_sp_b = numpy.tensordot(self._four_mom_d, Pauli_bar, axes=(0, 0))

    def _r1_sp_to_r2_sp(self):
        self._r2_sp = numpy.dot(self.l_sp_u, self.r_sp_u)

    def _r1_sp_to_r2_sp_b(self):
        self._r_sp_d.shape = (2, 1)
        self._l_sp_d.shape = (1, 2)
        self._r2_sp_b = numpy.dot(self.r_sp_d, self.l_sp_d)

    def _r_sp_d_to_r_sp_u(self):
        self._r_sp_u = numpy.dot(LeviCivita, self.r_sp_d)
        self._r_sp_u.shape = (1, 2)    # row vector

    def _l_sp_d_to_l_sp_u(self):
        self._l_sp_d.shape = (2, 1)    # temporary column vector
        self._l_sp_u = numpy.dot(LeviCivita, self.l_sp_d)
        self._l_sp_d.shape = (1, 2)    # back to row vector
        self._l_sp_u.shape = (2, 1)    # column vector

    def _r_sp_u_to_r_sp_d(self):
        self._r_sp_u.shape = (2, 1)    # temporary column vector
        self._r_sp_d = numpy.dot(numpy.transpose(LeviCivita), self.r_sp_u)
        self._r_sp_u.shape = (1, 2)    # back to row vector
        self._r_sp_d.shape = (2, 1)    # column vector

    def _l_sp_u_to_l_sp_d(self):
        self._l_sp_d = numpy.dot(numpy.transpose(LeviCivita), self.l_sp_u)
        self._l_sp_d.shape = (1, 2)    # row vector
