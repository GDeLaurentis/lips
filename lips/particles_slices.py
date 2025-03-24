import random
import sympy

from syngular import Ring, Ideal, flatten

from .algebraic_geometry.covariant_ideal import LipsIdeal
from .algebraic_geometry.particles_singular_variety import update_particles


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_Slices:

    # PUBLIC METHODS

    def univariate_slice(self, extra_constraints=(), seed=None, indepSets=None, algorithm=('covariant', 'generic')[0], verbose=False):
        from .particles import Particles
        random.seed(seed)
        t = sympy.symbols('t')

        if algorithm == 'covariant':  # ⟨ij⟩ is linear in t
            if indepSets is not None:
                raise NotImplementedError("IndepSet option not implemented yet with covariant algorithm.")
            self._singular_variety(extra_constraints, (1, ) * len(extra_constraints))
            oPShift = Particles(1, fix_mom_cons=False, field=self.field, seed=random.randint(1, self.field.characteristic - 1))[1]

            xs = sympy.symbols(f'x1:{len(self) + 1}')
            ys = sympy.symbols(f'y1:{len(self) + 1}')
            for i, oP in enumerate(self):
                oP.r_sp_d = oP.r_sp_d + t * xs[i] * oPShift.r_sp_d
                oP.l_sp_d = oP.l_sp_d + t * ys[i] * oPShift.l_sp_d

            equations = [sympy.poly(entry.expand(), modulus=self.field.characteristic ** self.field.digits) for entry in self.total_mom.flatten().tolist()]
            equations += [sympy.poly(self(constraint).expand(), modulus=self.field.characteristic ** self.field.digits) for constraint in extra_constraints]
            equations = [entry for entry in flatten([sympy.poly(eq, t).all_coeffs() for eq in equations]) if entry != 0]
            if verbose:
                print(f"Slicing subject to len(equations) constraints: {equations}")

            ring = Ring(self.field.characteristic, xs + ys, 'dp')
            ideal = Ideal(ring, list(map(str, equations)))
            xSubs = ideal.point_on_variety(self.field)
            self.subs(xSubs)

        elif algorithm == 'generic':  # ring-agnostic algorithm, less efficient: ⟨ij⟩ is quadratic in t
            multiplicity = len(self)
            I = LipsIdeal(multiplicity, extra_constraints)
            I.to_qring(I)
            univariate_slice = I.ring.univariate_slice(self.field)
            update_particles(self, univariate_slice(t))

        else:
            raise Exception('Complete shift algorithm not understood')
