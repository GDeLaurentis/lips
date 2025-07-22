#   ___          _   _    _
#  | _ \__ _ _ _| |_(_)__| |___
#  |  _/ _` | '_|  _| / _| / -_)
#  |_| \__,_|_|  \__|_\__|_\___|

# Author: Giuseppe

import numpy
import sympy
import lips

from copy import copy
from sympy import NotInvertible

from syngular import Field

from pyadic.field_extension import FieldExtension

from .tools import MinkowskiMetric, LeviCivita, rand_frac, Pauli, Pauli_bar, flatten


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particle(object):
    """Describes the kinematics of a single particle."""

    # MAGIC METHODS

    def __init__(self, kinematics=None, real_momentum=False, field=Field('mpc', 0, 300)):
        """
        Initialisation of Particle object.
        Calls randomise if kinematics is None,
        else initialises the appropriate variable based on the tensor shape of kinematics:
        (4, ) : four momentum;
        (2, 2) : rank 2 spinor;
        ( (2, 1), (1, 2) ) : (right spinor index down, left spinor index down).
        """
        # Should check the field is consistent with the kinematics info, or even compute it from there.
        self.field = field
        if kinematics is None and field.name in ("mpc", "gaussian rational", "rational", "finite field", "padic"):
            self.randomise(real_momentum=real_momentum)
        elif isinstance(kinematics, numpy.ndarray) and kinematics.shape == (4, ):
            self.four_mom = kinematics
        elif isinstance(kinematics, numpy.ndarray) and kinematics.shape == (2, 2):
            self.r2_sp = kinematics
        elif isinstance(kinematics, tuple) and len(kinematics) == 2 and kinematics[0].shape == (2, 1) and kinematics[1].shape == (1, 2):
            self._r_sp_d = kinematics[0]
            self._r_sp_d_to_r_sp_u()
            self.l_sp_d = kinematics[1]
        else:
            raise Exception('Bad Particle Constructor')

    def to_field(self, field):
        self._r_sp_d = numpy.vectorize(field)(self.r_sp_d)
        self._r_sp_d_to_r_sp_u()
        self.l_sp_d = numpy.vectorize(field)(self.l_sp_d)
        self.field = field

    def __eq__(self, other):
        """Equality: checks equality of four momenta."""
        if isinstance(other, Particle):
            return numpy.all(self.r2_sp == other.r2_sp)
        else:
            return False

    def __neg__(self):
        minus_self = Particle(field=self.field)
        minus_self.r2_sp = -1 * self.r2_sp
        return minus_self

    def __add__(self, other):
        """Sum: summs the four momenta."""
        if other == 0:
            return copy(self)
        assert isinstance(other, Particle)
        return Particle(kinematics=self.r2_sp + other.r2_sp, field=self.field)

    def __radd__(self, other):
        """Sum: summs the four momenta."""
        if other == 0:
            return copy(self)
        assert isinstance(other, Particle)
        return Particle(kinematics=self.r2_sp + other.r2_sp, field=self.field)

    def __sub__(self, other):
        """Sub: subtract the four momenta."""
        if other == 0:
            return copy(self)
        assert isinstance(other, Particle)
        return Particle(kinematics=self.r2_sp - other.r2_sp, field=self.field)

    def __mul__(self, other):
        """Mul: multiply momentum by number."""
        return Particle(kinematics=self.r2_sp * other, field=self.field)

    def __rmul__(self, other):
        """Mul: multiply momentum by number."""
        return Particle(kinematics=self.r2_sp * other, field=self.field)

    def __div__(self, other):
        """Div: divide momentum by number."""
        return Particle(kinematics=self.r2_sp / other, field=self.field)

    def __truediv__(self, other):
        """Div: divide momentum by number."""
        return Particle(kinematics=self.r2_sp / other, field=self.field)

    def __getitem__(self, key):
        return self.four_mom[key]

    def __hash__(self):
        if abs(self.m2) <= self.field.tollerance:
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
        except (ValueError, TypeError, SystemError, NotInvertible):
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
        except (ValueError, TypeError, SystemError, NotInvertible):
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
        except (ValueError, TypeError, SystemError, NotInvertible):
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
        except (ValueError, TypeError, SystemError, NotInvertible):
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
        except (ValueError, TypeError, SystemError, NotInvertible):
            self._four_mom = None
            self._four_mom_d = None
        # should I check for masslessness?
        try:
            self._r2_sp_to_r_sp_d()
            self._r_sp_d_to_r_sp_u()
            self._r2_sp_to_l_sp_d()
            self._l_sp_d_to_l_sp_u()
        except (ValueError, TypeError, SystemError, NotInvertible):
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
        except (ValueError, TypeError, SystemError, NotInvertible):
            self._four_mom = None
            self._four_mom_d = None
        # should I check for masslessness?
        try:
            self._r2_sp_to_r_sp_d()
            self._r_sp_d_to_r_sp_u()
            self._r2_sp_to_l_sp_d()
            self._l_sp_d_to_l_sp_u()
        except (ValueError, TypeError, SystemError, NotInvertible):
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
        except (ValueError, TypeError, SystemError, NotInvertible):
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
        except (ValueError, TypeError, SystemError, NotInvertible):
            self._r_sp_u = None
            self._r_sp_d = None
            self._l_sp_u = None
            self._l_sp_d = None

    # PUBLIC METHODS

    def randomise(self, real_momentum=False):
        if self.field.name == "mpc":
            self.randomise_mpc(real_momentum=real_momentum)
        elif self.field.name in ["rational", "gaussian rational", "finite field", "padic"]:
            self.randomise_spinors_in_field()

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
        p_zero = self.field.sqrt(p2)
        self.four_mom = numpy.array([p_zero] + p)

    def randomise_spinors_in_field(self):
        self._r_sp_d = numpy.array([self.field.random(), self.field.random()])
        self._r_sp_d.shape = (2, 1)
        self._r_sp_d_to_r_sp_u()
        self._four_mom = numpy.array([None, None, None, None])
        self.l_sp_d = numpy.array([self.field.random(), self.field.random()])

    def angles_for_squares(self):
        """Flips left and right spinors."""
        if self.l_sp_d is not None and self.r_sp_d is not None:  # massive scalars do not have these defined
            self._l_sp_d, self._r_sp_d = self._r_sp_d.T, self._l_sp_d.T
            if hasattr(self, 'spin_index'):
                if self.spin_index[0] == 'u':
                    self._l_sp_d *= -1
                elif self.spin_index[0] == 'd':
                    self._r_sp_d *= -1
                else:
                    raise Exception("Spin index not understood.")
            self._sps_d_to_sps_u()
        self._r2_sp = self._r2_sp.T
        self._r2_sp_b = self._r2_sp_b.T
        # should I check if four_mom is in the field?
        try:
            self._r2_sp_b_to_four_momentum()
            self._four_mom_to_four_mom_d()
        except (ValueError, TypeError, SystemError, NotInvertible):
            self._four_mom = None
            self._four_mom_d = None

    @property
    def spinors_are_in_field_extension(self):
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
            lambda_one = self.field.sqrt(P0_plus_P3)
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
            lambda_one = self.field.sqrt(P0_plus_P3)
            lambda_two = P1_minus_iP2 / lambda_one
        elif lips.spinor_convention == 'asymmetric':
            lambda_one = self.field(1)
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
        self._four_mom = numpy.array([numpy.trace(numpy.dot(Pauli_bar[i], self.r2_sp)) / 2 for i in range(4)])

    def _r2_sp_b_to_four_momentum(self):
        if not hasattr(self, "_four_mom") or self._four_mom is None:
            self._four_mom = numpy.array([None, None, None, None])
        self._four_mom = numpy.array([numpy.trace(numpy.dot(Pauli[i], self.r2_sp_b)) / 2 for i in range(4)])

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

    def _r_sp_d_to_r_sp_u(self):  # λ^α = ϵ^αβ λ_β
        """⟨a| = λ^α = ϵ^αβ λ_β = ϵ |a⟩ or ⟨a^I| = λ^Iα = ϵ^αβ λ_β^I = ϵ |a^I⟩"""
        self._r_sp_u = (LeviCivita @ self.r_sp_d).T

    def _l_sp_d_to_l_sp_u(self):
        """|a] = λ^α˙ = ϵ^α˙β˙ λ_β˙ = λ_β˙ ϵ^β˙α˙ = [a| ϵ or |a^I] = λ^α˙I = ϵ^α˙β˙ λ^I_β˙ = λ^I_β˙ ϵ^β˙α˙ = [a^I|"""
        self._l_sp_u = (self.l_sp_d @ LeviCivita.T).T

    def _r_sp_u_to_r_sp_d(self):
        self._r_sp_d = (self.r_sp_u @ LeviCivita).T

    def _l_sp_u_to_l_sp_d(self):
        self._l_sp_d = (LeviCivita.T @ self.l_sp_u).T

    def _sps_u_to_sps_d(self):
        self._l_sp_u_to_l_sp_d()
        self._r_sp_u_to_r_sp_d()

    def _sps_d_to_sps_u(self):
        self._l_sp_d_to_l_sp_u()
        self._r_sp_d_to_r_sp_u()

    # TWISTOR METHODS

    def randomise_twist(self):
        self._twist_z = numpy.array([[self.field.random()], [self.field.random()], [self.field.random()], [self.field.random()]])
        self.r_sp_d, self._mu = self._twist_z[:2, :], self._twist_z[2:, :]

    def comp_twist_x(self, other):
        """x^{˙αα} = (μⱼ^{˙α} λᵢ^α - μᵢ^{˙α} λⱼ^α ) / ⟨ij⟩"""
        self._twist_x = (self._mu @ other.r_sp_u - other._mu @ self.r_sp_u) / (other.r_sp_u @ self.r_sp_d)[0, 0]

    def twist_x_to_mom(self, other):
        r_two_spinor = self._twist_x - other._twist_x
        for i in range(4):
            self._four_mom[i] = numpy.trace(numpy.dot(Pauli[i], r_two_spinor)) / 2
        self.four_mom = self._four_mom

    # OPERATIONS

    def lsq(self):
        """Lorentz dot product with itself: 2 trace(P^{α̇α}P̅\u0305_{αα̇}) = P^μ * η_μν * P^ν."""
        # A possible test is that this should match the determinant of the rank 2 spinor
        # the determinant seems to be less operations, might be better - now used in compute s_ijk function
        lsq = numpy.trace(numpy.dot(self.r2_sp, self.r2_sp_b)) / 2
        if isinstance(lsq, sympy.Basic):
            lsq = sympy.expand(lsq)
        return lsq

    @property
    def m2(self):
        return self.lsq()

    @property
    def m(self):
        return self.field.sqrt(self.lsq())
