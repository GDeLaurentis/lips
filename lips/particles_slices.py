import random
import sympy

from itertools import combinations
from copy import deepcopy
from pycoretools import flatten, TemporarySetting

from syngular import Ring, Ideal, Polynomial

from .algebraic_geometry.covariant_ideal import LipsIdeal
from .algebraic_geometry.particles_singular_variety import update_particles


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_Slices:

    # PUBLIC METHODS

    def univariate_slice(self, extra_constraints=(), extra_exact_constraints=(), extra_approximate_constraints=(), seed=None, indepSets=None,
                         algorithm=('covariant', 'generic')[0], kind=('generic', 'minimal')[0],
                         minimal_non_zero=None, codim_upper_bound=None, verbose=False):
        from .particles import Particles
        random.seed(seed)
        t = sympy.symbols('t')

        extra_exact_constraints = extra_constraints + extra_exact_constraints

        if algorithm == 'covariant':  # ⟨ij⟩ is linear in t
            if indepSets is not None:
                raise NotImplementedError("IndepSet option not implemented yet with covariant algorithm.")
            self._singular_variety(extra_exact_constraints + extra_approximate_constraints,
                                   (self.field.digits, ) * len(extra_exact_constraints) + (1, ) * len(extra_approximate_constraints),
                                   seed=seed)
            oPShift = Particles(1, fix_mom_cons=False, field=self.field, seed=random.randint(1, self.field.characteristic - 1))[1]

            xs = sympy.symbols(f'x1:{len(self) + 1}')
            ys = sympy.symbols(f'y1:{len(self) + 1}')
            for i, oP in enumerate(self):
                oP.r_sp_d = oP.r_sp_d + t * xs[i] * oPShift.r_sp_d
                oP.l_sp_d = oP.l_sp_d + t * ys[i] * oPShift.l_sp_d

            equations = [sympy.poly(entry.expand(), modulus=self.field.characteristic ** self.field.digits) for entry in self.total_mom.flatten().tolist()]
            equations += [sympy.poly(self(constraint).expand(), modulus=self.field.characteristic ** self.field.digits) for constraint in extra_exact_constraints]
            equations = [entry for entry in flatten([sympy.poly(eq, t).all_coeffs() for eq in equations]) if entry != 0]
            equations_approximate = [sympy.poly(self(constraint).expand(), modulus=self.field.characteristic ** self.field.digits) for constraint in extra_approximate_constraints]
            equations_approximate = [entry for entry in flatten([sympy.poly(eq, t).all_coeffs() for eq in equations_approximate]) if entry != 0]

            ring = Ring(self.field.characteristic, xs + ys, 'dp')
            if verbose:
                print(f"Slicing in {len(ring.variables)} variables subject to {len(equations)} exact constraints and", end="")
                print(f"{len(equations_approximate)} approximate constraints, with a codim upper bound of {codim_upper_bound}.")
            ideal = Ideal(ring, list(map(str, equations)) + list(map(str, equations_approximate)))
            if kind == 'generic':
                if codim_upper_bound is not None:
                    ideal.codim_upper_bound = codim_upper_bound
                xSubs = ideal.point_on_variety(self.field, directions=list(map(str, equations)), seed=seed, verbose=verbose)
                counter = 0
                while 0 in xSubs.values():
                    if verbose:
                        print("One of the parameters was set to exactly zero, retrying")
                    counter += 1
                    xSubs = ideal.point_on_variety(self.field, directions=list(map(str, equations)), seed=seed + counter, verbose=verbose)
            elif kind == 'minimal':  # WIP
                # Tries to find a minimal slice - i.e. tries to keep as many variables un-shifted by t
                found = False
                counter = 0
                for r in range(2, 5):
                    for nonzero_subset in combinations(ring.variables, r):
                        min_point = {str(var): var if var in nonzero_subset else 0 for var in ring.variables}
                        if verbose:
                            counter += 1
                            print(f"\rTrying set: {counter}, {min_point}    ", end="")
                        # print(f"trying: {nonzero_subset}")
                        # print(f"trying: {min_point}")
                        I = deepcopy(ideal)
                        I.generators = [Polynomial(generator, field=self.field).subs(min_point) for generator in I.generators]
                        for generator in I.generators:
                            generator.field = int
                        with (TemporarySetting("syngular", "FORCECDOTS", True), TemporarySetting("syngular", "CDOTCHAR", "*")):
                            I.generators = [str(generator) for generator in I.generators]
                        I.squash()
                        I.ring.variables = tuple(var for var, val in min_point.items() if val != 0)
                        # print("new ideal:", I)
                        if I.is_unit_ideal:
                            continue
                        xSubs = min_point | I.point_on_variety(self.field)
                        selfTest = deepcopy(self)
                        selfTest.subs(xSubs)
                        # print(f"xsubs: {xSubs}")
                        if all([val == 0 for val in xSubs.values()]):
                            # print("continuing")
                            continue
                        elif minimal_non_zero is not None:
                            min_non_zero_check = [sympy.expand(8 * selfTest(entry)) for entry in minimal_non_zero]
                            if all([entry != 0 and (isinstance(sympy.sympify(entry), sympy.Number) or
                                                    sympy.poly(entry, modulus=selfTest.field.characteristic) != 0)
                                    for entry in min_non_zero_check]):
                                found = True
                                # print("breaking1")
                                break
                        elif not all([val == 0 for val in xSubs.values()]):
                            # print("breaking2")
                            found = True
                            break
                    if found:
                        break
                else:
                    print("Coundn't find a minimal non zero set")
            else:
                raise ValueError("Univariate slice kind not understood.")
            self.subs(xSubs)

        elif algorithm == 'generic':  # ring-agnostic algorithm, less efficient: ⟨ij⟩ is quadratic in t
            multiplicity = len(self)
            I = LipsIdeal(multiplicity, extra_constraints)
            I.to_qring(I)
            univariate_slice = I.ring.univariate_slice(self.field)
            update_particles(self, univariate_slice(t))

        else:
            raise Exception('Complete shift algorithm not understood')

    def bivariate_slice(self, extra_constraints=(), seed=None, indepSets=None, algorithm=('covariant', 'generic')[0], verbose=False):
        from .particles import Particles
        random.seed(seed)
        t1 = sympy.symbols('t1')
        t2 = sympy.symbols('t2')

        if algorithm == 'covariant':  # ⟨ij⟩ is linear in t
            if indepSets is not None:
                raise NotImplementedError("IndepSet option not implemented yet with covariant algorithm.")
            self._singular_variety(extra_constraints, (self.field.digits, ) * len(extra_constraints), seed=seed)
            oPShift1 = Particles(1, fix_mom_cons=False, field=self.field, seed=random.randint(1, self.field.characteristic - 1))[1]
            oPShift2 = Particles(1, fix_mom_cons=False, field=self.field, seed=random.randint(1, self.field.characteristic - 1))[1]

            xs1 = sympy.symbols(f'x1_1:{len(self) + 1}')
            ys1 = sympy.symbols(f'y1_1:{len(self) + 1}')
            xs2 = sympy.symbols(f'x2_1:{len(self) + 1}')
            ys2 = sympy.symbols(f'y2_1:{len(self) + 1}')
            for i, oP in enumerate(self):
                oP.r_sp_d = oP.r_sp_d + t1 * xs1[i] * oPShift1.r_sp_d + t2 * xs2[i] * oPShift2.r_sp_d
                oP.l_sp_d = oP.l_sp_d + t1 * ys1[i] * oPShift1.l_sp_d + t2 * xs2[i] * oPShift2.l_sp_d

            equations = [sympy.poly(entry.expand(), modulus=self.field.characteristic ** self.field.digits) for entry in self.total_mom.flatten().tolist()]
            equations += [sympy.poly(self(constraint).expand(), modulus=self.field.characteristic ** self.field.digits) for constraint in extra_constraints]
            equations = [entry for entry in flatten([sympy.poly(eq, (t1, t2)).as_list() for eq in equations]) if entry != 0]
            if verbose:
                print(f"Slicing subject to len(equations) constraints: {equations}")

            ring = Ring(self.field.characteristic, xs1 + xs2 + ys1 + ys2, 'dp')
            ideal = Ideal(ring, list(map(str, equations)))
            xSubs = ideal.point_on_variety(self.field, seed=seed)
            counter = 0
            while 0 in xSubs.values():
                counter += 1
                xSubs = ideal.point_on_variety(self.field, seed=seed + counter)
            self.subs(xSubs)

        elif algorithm == 'generic':
            raise NotImplementedError

        else:
            raise Exception('Complete shift algorithm not understood')
