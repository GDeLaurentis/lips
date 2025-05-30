# -*- coding: utf-8 -*-

import itertools
import mpmath
import numpy
import pytest
import sympy

from sympy.functions.special.tensor_functions import LeviCivita
from fractions import Fraction as Q

from lips import Particles
from lips.tools import pSijk, pDijk
from lips.fields.field import Field

mpc = Field('mpc', 0, 300)
modp = Field('finite field', 2 ** 31 - 1, 1)
padic = Field('padic', 2 ** 31 - 1, 6)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_compute_ldot(field):
    oParticles = Particles(5, field=field)
    assert abs(oParticles.ldot(1, 2) - numpy.trace(numpy.dot(oParticles[1].r2_sp, oParticles[2].r2_sp_b)) / 2) <= field.tollerance


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_compute_lNB(field):
    oParticles = Particles(6, field=field)
    assert abs(oParticles("⟨1|2-3|4]") - oParticles("⟨1|2⟩[2|4]-⟨1|3⟩[3|4]")) <= field.tollerance
    assert abs(oParticles("⟨1|2+3|4]") - oParticles("⟨1|2⟩[2|4]+⟨1|3⟩[3|4]")) <= field.tollerance
    assert abs(oParticles("⟨1|(2-3)|4]") - oParticles("⟨1|2-3|4]")) <= field.tollerance
    assert abs(oParticles("⟨1|(2+3)|4]") - oParticles("⟨1|2+3|4]")) <= field.tollerance
    assert abs(oParticles("2⟨1|(2)|4]") - oParticles("2⟨1|2|4]")) <= field.tollerance
    assert abs(oParticles("123⟨1|(3)|4]") - oParticles("123⟨1|3|4]")) <= field.tollerance


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_compute_lNB_open(field):
    oParticles = Particles(7, field=field)
    assert numpy.all(abs(oParticles("|4+5|6+7|1]-|6+7|4+5|1]") - oParticles("|4]⟨4|6+7|1]+|5]⟨5|6+7|1]-|6]⟨6|4+5|1]-|7]⟨7|4+5|1]")) <= field.tollerance)
    assert numpy.all(abs(oParticles("[1|4+5|6+7|-[1|6+7|4+5|") - oParticles("[1|4+5|6⟩[6|+[1|4+5|7⟩[7|-[1|6+7|4⟩[4|-[1|6+7|5⟩[5|")) <= field.tollerance)


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_compute_lNB_doubly_open(field):
    oParticles = Particles(8, field=field)
    oPsClustered = oParticles.cluster([[1,], [2,], [3, 4], [5, 6], [7, 8]])
    assert numpy.all(abs(oPsClustered("⟨2|3|4|2⟩") - (oPsClustered("⟨2|") @ oPsClustered("|3|4|") @ oPsClustered("|2⟩"))[0, 0]) <= field.tollerance)


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_eval_rational_function(field):
    oParticles = Particles(7, field=field)
    assert abs(oParticles("(+96/127[35]⟨4|2+3|1]⟨16⟩)/(⟨56⟩[56]⟨1|(2+4)|3]⟨2|(1+4)|3])") -
               Q("+96/127") * oParticles("[35]") * oParticles("⟨4|2+3|1]") * oParticles("⟨1|6⟩") /
               (oParticles("⟨56⟩") * oParticles("[56]") * oParticles("⟨1|(2+4)|3]") * oParticles("⟨2|(1+4)|3]"))) <= field.tollerance


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_eval_expr_with_two_open_indices(field):
    oPs = Particles(7, field=field)
    assert numpy.all(numpy.abs(
        oPs("|1⟩⟨2|4+5|1|-|2⟩⟨3|4+5|3]⟨1|+|2⟩⟨1|4+5|2|+|2⟩⟨1|4+5|3|+|3|4+5|2⟩⟨1|") -
        oPs("-1*(|1⟩⟨1|)*⟨2|5⟩*[1|5]-1*(|1⟩⟨1|)*⟨2|4⟩*[1|4]+1*(|2⟩⟨1|)*⟨3|4⟩*[3|4]+1*(|2⟩⟨1|)*⟨3|5⟩*[3|5]-1*(|2⟩⟨2|)*⟨1|4⟩*[2|4]\
            -1*(|2⟩⟨2|)*⟨1|5⟩*[2|5]-1*(|2⟩⟨3|)*⟨1|4⟩*[3|4]-1*(|2⟩⟨3|)*⟨1|5⟩*[3|5]-1*(|3⟩⟨1|)*⟨2|4⟩*[3|4]-1*(|3⟩⟨1|)*⟨2|5⟩*[3|5]")) <= field.tollerance)


def test_particles_eval_string_with_two_open_indices():
    oPs = Particles(8, field=Field("finite field", 2 ** 31 - 19, 1), seed=0)
    oPs._singular_variety(("⟨34⟩+[34]", "⟨34⟩-⟨56⟩", "⟨56⟩+[56]"), (1, 1, 1))
    oPsClustered = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]])
    oPsClustered("+((⟨1|@(|1|2|3|4|3|-|1|3|1|4|3|+|1|3|3|4|1+2|-|3|2|3|4|1+2|)@|2]))/(⟨1|3|2]Δ_12|3|4|5)")


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_eval_trace(field):
    oPs = Particles(7, field=field)
    assert numpy.all(numpy.abs(oPs("⟨3|4+5-6-7|3]-⟨1|4+5-6-7|1]-⟨2|4+5-6-7|2]") - oPs("tr(3-1-2|4+5-6-7)")) <= field.tollerance)
    assert numpy.all(numpy.abs(oPs("|3]⟨3|4+5-6-7|-|1]⟨1|4+5-6-7|-|2]⟨2|4+5-6-7|").trace() - oPs("tr(3-1-2|4+5-6-7)")) <= field.tollerance)
    assert numpy.all(numpy.abs(oPs("|4+5-6-7|3]⟨3|-|4+5-6-7|1]⟨1|-|4+5-6-7|2]⟨2|").trace() - oPs("tr(3-1-2|4+5-6-7)")) <= field.tollerance)
    oPs = Particles(7, field=Field("finite field", 2 ** 31 - 1, 1))


def test_particles_eval_with_internal_masses_invalid():
    oPs = Particles(7, field=padic)
    with pytest.raises(AttributeError):
        oPs("-1/8(s12s23mt2(8mt2-s_123-2s_45)(s_123-2s_45)[1|3]^2)/([1|6+7|4+5|3][1|4+5|6+7|3])")


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_eval_with_internal_masses(field):
    oPs = Particles(7, field=field, internal_masses={'mt2'})
    oPs("-1/8(s12s23mt2(8mt2-s_123-2s_45)(s_123-2s_45)[1|3]^2)/([1|6+7|4+5|3][1|4+5|6+7|3])")


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_eval_with_four_mass_box_gram(field):
    oParticles = Particles(8, field=field, internal_masses={'mt'})
    oParticles('(mt^2)/(Δ_12|34|56|78)')


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_eval_with_sqrt(field):
    oPs = Particles(8, field=field, seed=0)
    assert abs(oPs('sqrt(Δ_61|23|45)') - oPs.field.sqrt(oPs("Δ_61|23|45"))) <= oPs.field.tollerance
    assert abs(oPs('1/8s123sqrt(Δ_61|23|45)') - oPs("1/8s123") * oPs.field.sqrt(oPs('Δ_61|23|45'))) <= oPs.field.tollerance


@pytest.mark.parametrize("field", [mpc, modp, padic, ])
def test_particles_eval_mass_as_alias_with_cluster(field):
    oPs = Particles(7, field=field, internal_masses={'mt2'})
    oPs.mh2 = "s_45"
    oPs_5pt = oPs.cluster([[1], [2], [3], [4, 5], [6, 7]])
    expr = "-1/8(s12s23mt2(8mt2-s_123-2s_45)(s_123-2s_45)[1|3]^2)/([1|6+7|4+5|3][1|4+5|6+7|3])"
    expr_5pt = "-1/8(s12s23mt2(8mt2-s_123-2mh2)(s_123-2mh2)[1|3]^2)/([1|5|4|3][1|4|5|3])"
    assert abs(oPs(expr) - oPs_5pt(expr_5pt)) <= oPs.field.tollerance


def test_particles_compute_Mandelstam():
    """Test computation of Mandelsta w.r.t. summed particles object."""
    oParticles = Particles(9)
    temp_string = "s_1234"
    ijk = list(map(int, pSijk.findall(temp_string)[0]))
    assert sum([oParticles[_i] for _i in ijk]).m2 == oParticles(temp_string)


def test_particles_compute_three_mass_gram():
    """Test computation of 3-mass Gram w.r.t. summed particles object."""
    oParticles = Particles(9)
    temp_string = "Δ_135|249|678"
    match = pDijk.findall(temp_string)[0]
    if "|" in match:
        NonOverlappingLists = [list(map(int, corner)) for corner in match.split("|")]
    else:
        NonOverlappingLists = oParticles.ijk_to_3NonOverlappingLists(list(map(int, match)))
    temp_oParticles = oParticles.cluster(NonOverlappingLists)
    assert temp_oParticles.ldot(1, 2)**2 - temp_oParticles.ldot(1, 1) * temp_oParticles.ldot(2, 2) == oParticles(temp_string)


def test_particles_compute_tr5_1234():
    oParticles = Particles(5)
    tr5 = 0
    for (i, j, k, l) in itertools.product(range(4), repeat=4):
        tr5 += mpmath.mpc(sympy.simplify(4j * LeviCivita(i, j, k, l) *
                                         oParticles[1].four_mom[i] *
                                         oParticles[2].four_mom[j] *
                                         oParticles[3].four_mom[k] *
                                         oParticles[4].four_mom[l]))
    assert abs(tr5 - oParticles("tr5_1234")) < 10 ** -290


def test_particles_compute_four_mass_box_gram():
    oPs = Particles(8, field=Field("finite field", 2 ** 31 - 19, 1), seed=0)
    oPs._singular_variety(("⟨34⟩+[34]", "⟨34⟩-⟨56⟩", "⟨56⟩+[56]"), (1, 1, 1))
    oPs.mt2 = oPs("s_34")
    oPs.mt = oPs("<34>")
    oPs = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]])
    assert oPs.mt ** 2 == oPs.mt2
    assert oPs("Δ_12|3|4|5") == oPs("(1/4*(s12*(tr(3|3)²-tr(3|4)²)+tr(3|4)*tr(1+2|4)*tr(1+2|3)-1/2*tr(3|3)*(tr(1+2|4)²+tr(1+2|3)²)))")
    assert oPs("Δ_12|3|4|5²") == oPs("(1/4*(s12*(tr(3|3)²-tr(3|4)²)+tr(3|4)*tr(1+2|4)*tr(1+2|3)-1/2*tr(3|3)*(tr(1+2|4)²+tr(1+2|3)²)))²")


def test_particles_compute_with_levicivita_and_transpose():
    oPs = Particles(6, field=Field("finite field", 2 ** 31 - 1, 1), seed=None)
    assert oPs("[3^|1+2|5_⟩") == oPs("⟨5_|1+2|3^]")
    assert 'ϵ' in oPs._parse("[3^|1+2|5_⟩") and 'transpose' in oPs._parse("[3^|1+2|5_⟩")


def test_particles_compute_bold_numbers():
    oPs = Particles(8, field=Field("finite field", 2 ** 31 - 19, 1), seed=0)
    oPs = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', 1), (4, 'd', 1)))
    assert numpy.all(oPs("4|𝟒]") == 4 * oPs("|𝟒]"))
    assert numpy.all(oPs("|𝟒|𝟒]") == oPs("|𝟒|") @ oPs("|𝟒]"))
    with pytest.raises(SyntaxError):
        oPs("𝟒|𝟒⟩")


def test_particles_eval_moderately_complicated_expression():
    oPs = Particles(8, field=Field("finite field", 2 ** 31 - 19, 1), seed=0)
    oPs._singular_variety(("⟨34⟩+[34]", "⟨34⟩-⟨56⟩", "⟨56⟩+[56]"), (1, 1, 1))
    oPs.mt = oPs("<34>")
    oPs = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', 1), (4, 'd', 1)))
    # just check it can be evaluated
    oPs("+(+1/48mt²(⟨2|(3)|1+2|4|1]-⟨2|4|(1+2)|3|1])tr(1+2|3+4)(s_124-s_3)²s_34(s_34-4s_3)([3|4]-⟨3|4⟩))/(⟨1|2⟩[1|2]Δ_12|3|4|5²)")
