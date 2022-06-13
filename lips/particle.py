# -*- coding: utf-8 -*-

#   ___          _   _    _
#  | _ \__ _ _ _| |_(_)__| |___
#  |  _/ _` | '_|  _| / _| / -_)
#  |_| \__,_|_|  \__|_\__|_\___|

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
import sympy
import mpmath
import random
import lips

from .fields.field import Field
from .fields.gaussian_rationals import GaussianRational, rand_rat_frac
from .fields.finite_field import ModP
from .fields.padic import PAdic
from .tools import MinkowskiMetric, LeviCivita, rand_frac, Pauli, Pauli_bar, flatten


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particle(object):
    """Describes the kinematics of a single particle."""

    # MAGIC METHODS

    def __init__(self, four_mom=None, r2_sp=None, real_momentum=False, field=Field('mpc', 0, 300)):
        """Initialisation. Calls randomise if None, else initialises the four momentum."""
        self.field = field
        if four_mom is None and r2_sp is None and field.name in ("mpc", "gaussian rational", "finite field", "padic"):
            self.randomise(real_momentum=real_momentum)
        elif four_mom is not None:
            self.four_mom = four_mom
        elif r2_sp is not None:
            self.r2_sp = r2_sp
        else:
            raise Exception('Bad Particle Constructor')

    def __eq__(self, other):
        """Equality: checks equality of four momenta."""
        if type(self) == type(other):
            return numpy.all(self.r2_sp == other.r2_sp)
        else:
            return False

    def __neg__(self):
        minus_self = Particle()
        minus_self.r2_sp = -1 * self.r2_sp
        return minus_self

    def __add__(self, other):
        """Sum: summs the four momenta."""
        if other == 0:
            return Particle(r2_sp=self.r2_sp)
        assert isinstance(other, Particle)
        return Particle(r2_sp=self.r2_sp + other.r2_sp)

    def __radd__(self, other):
        """Sum: summs the four momenta."""
        if other == 0:
            return Particle(r2_sp=self.r2_sp)
        assert isinstance(other, Particle)
        return Particle(r2_sp=self.r2_sp + other.r2_sp)

    def __sub__(self, other):
        """Sub: subtract the four momenta."""
        if other == 0:
            return Particle(r2_sp=self.r2_sp)
        assert isinstance(other, Particle)
        return Particle(r2_sp=self.r2_sp - other.r2_sp)

    def __mul__(self, other):
        """Mul: multiply momentum by number."""
        return Particle(r2_sp=self.r2_sp * other)

    def __rmul__(self, other):
        """Mul: multiply momentum by number."""
        return Particle(r2_sp=self.r2_sp * other)

    def __div__(self, other):
        """Div: divide momentum by number."""
        return Particle(r2_sp=self.r2_sp / other)

    def __truediv__(self, other):
        """Div: divide momentum by number."""
        return Particle(r2_sp=self.r2_sp / other)

    def __getitem__(self, key):
        return self.four_mom[key]

    def __hash__(self):
        if abs(self.mass) <= self.field.tollerance:
            return hash(tuple([tuple(self.r_sp_d.flatten()), tuple(self.l_sp_d.flatten())]))
        else:
            return hash(tuple(self.r2_sp.flatten()))

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
        # should I check if four_mom is in the field?
        try:
            self._r2_sp_b_to_four_momentum()
            self._four_mom_to_four_mom_d()
        except (TypeError, SystemError):
            self._four_mom = None
            self._four_mom_d = None

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
        # should I check if four_mom is in the field?
        try:
            self._r2_sp_b_to_four_momentum()
            self._four_mom_to_four_mom_d()
        except (TypeError, SystemError):
            self._four_mom = None
            self._four_mom_d = None

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
        # should I check if four_mom is in the field?
        try:
            self._r2_sp_b_to_four_momentum()
            self._four_mom_to_four_mom_d()
        except (TypeError, SystemError):
            self._four_mom = None
            self._four_mom_d = None

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
        # should I check if four_mom is in the field?
        try:
            self._r2_sp_b_to_four_momentum()
            self._four_mom_to_four_mom_d()
        except (TypeError, SystemError):
            self._four_mom = None
            self._four_mom_d = None

    @property
    def r2_sp(self):
        """Four Momentum Slashed with upper indices: P^{α̇α}"""
        return self._r2_sp

    @r2_sp.setter
    def r2_sp(self, temp_r2_sp):
        self._r2_sp = temp_r2_sp
        self._r2_sp_to_r2_sp_b()
        # should I check if four_mom is in the field?
        try:
            self._r2_sp_to_four_momentum()
            self._four_mom_to_four_mom_d()
        except (TypeError, SystemError):
            self._four_mom = None
            self._four_mom_d = None
        # should I check for masslessness?
        try:
            self._r2_sp_to_r_sp_d()
            self._r_sp_d_to_r_sp_u()
            self._r2_sp_to_l_sp_d()
            self._l_sp_d_to_l_sp_u()
        except (TypeError, ValueError):
            self._r_sp_u = None
            self._r_sp_d = None
            self._l_sp_u = None
            self._l_sp_d = None

    @property
    def r2_sp_b(self):
        """Four Momentum Slashed with lower indices: P̅\u0305_{αα̇}"""
        return self._r2_sp_b

    @r2_sp_b.setter
    def r2_sp_b(self, temp_r2_sp_b):
        self._r2_sp_b = temp_r2_sp_b
        self._r2_sp_b_to_r2_sp()
        # should I check if four_mom is in the field?
        try:
            self._r2_sp_b_to_four_momentum()
            self._four_mom_to_four_mom_d()
        except (TypeError, SystemError):
            self._four_mom = None
            self._four_mom_d = None
        # should I check for masslessness?
        try:
            self._r2_sp_to_r_sp_d()
            self._r_sp_d_to_r_sp_u()
            self._r2_sp_to_l_sp_d()
            self._l_sp_d_to_l_sp_u()
        except (TypeError, ValueError):
            self._r_sp_u = None
            self._r_sp_d = None
            self._l_sp_u = None
            self._l_sp_d = None

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
        # should I check for masslessness?
        try:
            self._four_mom_to_r_sp_d()
            self._r_sp_d_to_r_sp_u()
            self._four_mom_to_l_sp_d()
            self._l_sp_d_to_l_sp_u()
        except TypeError:
            self._r_sp_u = None
            self._r_sp_d = None
            self._l_sp_u = None
            self._l_sp_d = None

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
        # should I check for masslessness?
        try:
            self._four_mom_to_r_sp_d()
            self._r_sp_d_to_r_sp_u()
            self._four_mom_to_l_sp_d()
            self._l_sp_d_to_l_sp_u()
        except TypeError:
            self._r_sp_u = None
            self._r_sp_d = None
            self._l_sp_u = None
            self._l_sp_d = None

    # PUBLIC METHODS

    def randomise(self, real_momentum=False):
        if self.field.name == "mpc":
            self.randomise_mpc(real_momentum=real_momentum)
        elif self.field.name == "gaussian rational":
            self.randomise_rational()
        elif self.field.name == "finite field":
            self.randomise_finite_field()
        elif self.field.name == "padic":
            self.randomise_padic()

    def randomise_mpc(self, real_momentum=False):
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

    def randomise_rational(self):
        self._r_sp_d = numpy.array([GaussianRational(rand_rat_frac(), rand_rat_frac()), GaussianRational(rand_rat_frac(), rand_rat_frac())])
        self._r_sp_d.shape = (2, 1)
        self._r_sp_d_to_r_sp_u()
        self._four_mom = numpy.array([None, None, None, None])
        self.l_sp_d = numpy.array([GaussianRational(rand_rat_frac(), rand_rat_frac()), GaussianRational(rand_rat_frac(), rand_rat_frac())])

    def randomise_finite_field(self):
        p = self.field.characteristic
        self._r_sp_d = numpy.array([ModP(random.randrange(0, p), p), ModP(random.randrange(0, p), p)], dtype=object)
        self._r_sp_d.shape = (2, 1)
        self._r_sp_d_to_r_sp_u()
        self._four_mom = numpy.array([None, None, None, None])
        self.l_sp_d = numpy.array([ModP(random.randrange(0, p), p), ModP(random.randrange(0, p), p)], dtype=object)

    def randomise_padic(self):
        p, k = self.field.characteristic, self.field.digits
        self._r_sp_d = numpy.array([PAdic(random.randrange(0, p ** k - 1), p, k), PAdic(random.randrange(0, p ** k - 1), p, k)], dtype=object)
        self._r_sp_d.shape = (2, 1)
        self._r_sp_d_to_r_sp_u()
        self._four_mom = numpy.array([None, None, None, None])
        self.l_sp_d = numpy.array([PAdic(random.randrange(0, p ** k - 1), p, k), PAdic(random.randrange(0, p ** k - 1), p, k)], dtype=object)

    def angles_for_squares(self):
        """Flips left and right spinors."""
        self._l_sp_u, self._r_sp_u = self._r_sp_u, self._l_sp_u
        self._l_sp_u.shape = (2, 1)
        self._r_sp_u.shape = (1, 2)
        self._l_sp_u_to_l_sp_d()
        self._r_sp_u_to_r_sp_d()
        self._r1_sp_to_r2_sp()
        self._r1_sp_to_r2_sp_b()
        # should I check if four_mom is in the field?
        try:
            self._r2_sp_b_to_four_momentum()
            self._four_mom_to_four_mom_d()
        except (TypeError, SystemError):
            self._four_mom = None
            self._four_mom_d = None

    @property
    def spinors_are_in_field_extension(self):
        from .fields.field_extension import FieldExtension
        return FieldExtension in set(map(type, flatten(self.r_sp_d) + flatten(self.l_sp_d)))

    # PRIVATE METHODS

    def _r2_sp_to_r_sp_d(self):
        self._set_r_sp_d(self.r2_sp[1, 1], - self.r2_sp[1, 0])

    def _four_mom_to_r_sp_d(self):
        self._set_r_sp_d(self._four_mom[0] + self._four_mom[3], self._four_mom[1] + self._four_mom[2] * 1j)

    def _set_r_sp_d(self, P0_plus_P3, P1_plus_iP2):  # r_sp_d is \lambda_\alpha
        if lips.spinor_convention == 'symmetric' and abs(P0_plus_P3) <= self.field.tollerance:
            raise ValueError("Encountered zero denominator in spinor.")
        elif lips.spinor_convention == 'symmetric':
            lambda_one = mpmath.sqrt(P0_plus_P3)
            lambda_two = P1_plus_iP2 / lambda_one
        elif lips.spinor_convention == 'asymmetric':
            lambda_one = P0_plus_P3
            lambda_two = P1_plus_iP2
        self._r_sp_d = numpy.array([lambda_one, lambda_two], dtype=object)
        self._r_sp_d.shape = (2, 1)    # column vector

    def _r2_sp_to_l_sp_d(self):
        self._set_l_sp_d(self.r2_sp[1, 1], - self.r2_sp[0, 1])

    def _four_mom_to_l_sp_d(self):
        self._set_l_sp_d(self._four_mom[0] + self._four_mom[3], self._four_mom[1] - self._four_mom[2] * 1j)

    def _set_l_sp_d(self, P0_plus_P3, P1_minus_iP2):  # l_sp_d is \bar{\lambda}_{\dot\alpha}
        if lips.spinor_convention == 'symmetric' and abs(P0_plus_P3) <= self.field.tollerance:
            raise ValueError("Encountered zero denominator in spinor.")
        elif lips.spinor_convention == 'symmetric':
            lambda_one = mpmath.sqrt(P0_plus_P3)
            lambda_two = P1_minus_iP2 / lambda_one
        elif lips.spinor_convention == 'asymmetric':
            lambda_one = 1
            lambda_two = P1_minus_iP2 / P0_plus_P3
        self._l_sp_d = numpy.array([lambda_one, lambda_two], dtype=object)
        self._l_sp_d.shape = (1, 2)    # column vector

    def _r2_sp_to_r2_sp_b(self):
        self._r2_sp_b = (LeviCivita.dot(self.r2_sp.dot(LeviCivita.T))).T

    def _r2_sp_b_to_r2_sp(self):
        self._r2_sp = (LeviCivita.dot(self.r2_sp_b.dot(LeviCivita.T))).T

    def _r2_sp_to_four_momentum(self):
        if not hasattr(self, "_four_mom") or self._four_mom is None:
            self._four_mom = numpy.array([None, None, None, None])
        for i in range(4):
            self._four_mom[i] = numpy.trace(numpy.dot(Pauli_bar[i], self.r2_sp)) / 2

    def _r2_sp_b_to_four_momentum(self):
        if not hasattr(self, "_four_mom") or self._four_mom is None:
            self._four_mom = numpy.array([None, None, None, None])
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

    # TWISTOR METHODS

    def randomise_twist(self):
        self._twist_z = numpy.array([rand_frac(), rand_frac(), rand_frac(), rand_frac()])
        self._r_sp_d = numpy.array([self._twist_z[0], self._twist_z[1]])
        self._mu = numpy.array([self._twist_z[2], self._twist_z[3]])

    def comp_twist_x(self, other):
        x21n = self._r_sp_d[0] * other._mu[0] - self._mu[0] * other._r_sp_d[0]
        x21d = (self._r_sp_d[0] * other._r_sp_d[1] -
                other._r_sp_d[0] * self._r_sp_d[1])
        x21 = x21n / x21d
        x11 = (self._mu[0] - self._r_sp_d[1] * x21) / self._r_sp_d[0]

        x22n = self._r_sp_d[0] * other._mu[1] - self._mu[1] * other._r_sp_d[0]
        x22d = (self._r_sp_d[0] * other._r_sp_d[1] -
                other._r_sp_d[0] * self._r_sp_d[1])
        x22 = x22n / x22d
        x12 = (self._mu[1] - self._r_sp_d[1] * x22) / self._r_sp_d[0]

        self._twist_x = numpy.array([[x11, x12], [x21, x22]])

    def twist_x_to_mom(self, other):
        r_two_spinor = self._twist_x - other._twist_x
        for i in range(4):
            self._four_mom[i] = numpy.trace(numpy.dot(Pauli[i], r_two_spinor)) / 2
        self.four_mom = self._four_mom

    # OPERATIONS

    def lsq(self):
        """Lorentz dot product with itself: 2 trace(P^{α̇α}P̅\u0305_{αα̇}) = P^μ * η_μν * P^ν."""
        # A possible test is that this should match the determinant of the rank 2 spinor
        lsq = numpy.trace(numpy.dot(self.r2_sp, self.r2_sp_b)) / 2
        if isinstance(lsq, sympy.Basic):
            lsq = sympy.expand(lsq)
        return lsq

    @property
    def mass(self):
        # Technically this is the mass squared - might want to rename
        return self.lsq()
