import numpy
import pytest
import pickle
import subprocess
import hashlib

from lips.fields import Field
from lips import Particles
from lips.tools import MinkowskiMetric, Pauli_bar


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


mpc = Field('mpc', 0, 300)
modp = Field('finite field', 2 ** 31 - 1, 1)
padic = Field('padic', 2 ** 31 - 1, 6)


@pytest.mark.parametrize(
    "multiplicity, field",
    [
        (4, mpc), (4, modp), (4, padic),
        (5, mpc), (5, modp), (5, padic),
        (6, mpc), (6, modp), (6, padic),
        (7, mpc), (7, modp), (7, padic),
    ]
)
def test_particles_instantiation(multiplicity, field):
    oPs = Particles(multiplicity, field=field)
    assert oPs.momentum_conservation_check()
    assert oPs.onshell_relation_check()


@pytest.mark.parametrize(
    "multiplicity, real_momenta, axis",
    [
        (5, False, 1),
        (5, False, 2),
        (5, True, 1),
        (5, True, 2),
        (5, True, 3),
        (6, False, 1),
        (6, False, 2),
        (6, True, 1),
        (6, True, 2),
        (6, True, 3),
    ]
)
def test_particles_fix_mom_cons(multiplicity, real_momenta, axis):

    oParticles = Particles(multiplicity, real_momenta=real_momenta, fix_mom_cons=False)
    oParticles.fix_mom_cons(real_momenta=real_momenta, axis=axis)

    assert oParticles.momentum_conservation_check()
    assert oParticles.onshell_relation_check()


def test_spinor_item_setter():
    oPs = Particles(8)
    a, b, c, d = numpy.array([[1, 2]]), numpy.array([[2], [-1]]), numpy.array([[3, 4]]), numpy.array([[-4], [3]])
    oPs["[5|"] = a
    assert numpy.all(oPs["|5]"] == b)
    oPs["|5⟩"] = d
    assert numpy.all(oPs["⟨5|"] == c)


def test_equality_after_pickle():
    # Note that pickle uses __reduce__ to dump bytecode info of the object.
    oPs = Particles(6, field=Field("finite field", 2 ** 31 - 1, 1), seed=0)
    oPs2 = oPs.image(('132456', False))
    oPs3 = oPs.image(('132456', False))
    assert oPs2 == oPs3
    assert pickle.dumps(oPs2) == pickle.dumps(oPs3)


def test_particles_cluster_with_massive_fermions():
    oPs = Particles(8, field=Field("finite field", 2 ** 31 - 19, 1), seed=0)
    oPs._singular_variety(("⟨34⟩+[34]", "⟨34⟩-⟨56⟩", "⟨56⟩+[56]"), (1, 1, 1))
    oPs.mt2 = oPs("s_34")
    oPs.mt = - oPs("⟨34⟩")
    oPsClusteredTensor = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', all), (4, 'd', all)))
    oPsClusteredTensor11 = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', 1), (4, 'd', 1)))
    oPsClusteredTensor12 = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', 1), (4, 'd', 2)))
    oPsClusteredTensor21 = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', 2), (4, 'd', 1)))
    oPsClusteredTensor22 = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], [7, 8]], massive_fermions=((3, 'u', 2), (4, 'd', 2)))
    assert (oPsClusteredTensor("⟨3|5|4]") == numpy.array([[oPsClusteredTensor11("⟨3|5|4]"), oPsClusteredTensor12("⟨3|5|4]")],
                                                          [oPsClusteredTensor21("⟨3|5|4]"), oPsClusteredTensor22("⟨3|5|4]")]])).all()


def test_Dirac_equation():
    oPs = Particles(6, field=Field("finite field", 2 ** 31 - 1, 1), seed=None)
    oPsClustered = oPs.cluster([[1, ], [2, ], [3, 4], [5, 6], ], massive_fermions=((3, 'u', 1), (4, 'd', 1)))
    oPsClustered.m = - oPs("⟨3|4⟩")
    oPsClustered.mb = oPs("[3|4]")
    assert oPsClustered("s_3") == oPsClustered.m * oPsClustered.mb
    assert (oPsClustered("|3|3⟩") == - oPsClustered("m|3]")).all()
    assert (oPsClustered("|3|3]") == - oPsClustered("mb|3⟩")).all()
    assert (oPsClustered("⟨3|3|") == oPsClustered("m[3|")).all()
    assert (oPsClustered("[3|3|") == oPsClustered("mb⟨3|")).all()


def run_hashes_in_subprocess():
    code = """
import hashlib
import pickle
from lips import Particles
from syngular import Field
oPs = Particles(6, field=Field("finite field", 2 ** 31 - 1, 1), seed=0)
cache_key_raw = oPs
cache_key_bytes = pickle.dumps(cache_key_raw)
cache_key_hash = hashlib.sha256(cache_key_bytes).hexdigest()
print(hash(oPs))
print(cache_key_hash, end="")
"""
    result = subprocess.run(
        ["python", "-c", code],
        capture_output=True,
        check=True
    )
    return result.stdout.decode().split("\n")


def test_hash_vs_sha256_process_stability():
    hash1, hash1sha256 = run_hashes_in_subprocess()
    hash2, hash2sha256 = run_hashes_in_subprocess()
    assert hash1 != hash2
    assert hash1sha256 == hash2sha256


def test_pickle_round_trip():
    original = Particles(6, field=Field("finite field", 2 ** 31 - 1, 1), seed=0)

    dumped = pickle.dumps(original)
    loaded = pickle.loads(dumped)

    assert original == loaded

    hash1 = hashlib.sha256(pickle.dumps(original)).hexdigest()
    hash2 = hashlib.sha256(pickle.dumps(loaded)).hexdigest()

    assert hash1 == hash2
    assert hash(original) == hash(loaded)


def test_hashes_does_not_change_under_identity_mapping():
    oPs = Particles(6, field=Field("finite field", 2 ** 31 - 1, 1), seed=0)
    # cache_key_bytes = pickle.dumps(oPs)
    # cache_key_hash1 = hashlib.sha256(cache_key_bytes).hexdigest()
    oPsMapped = oPs.image(('123456', False, '+'))
    # cache_key_bytes = pickle.dumps(oPsMapped)
    # cache_key_hash2 = hashlib.sha256(cache_key_bytes).hexdigest()
    assert hash(oPs) == hash(oPsMapped)
    # assert cache_key_hash1 == cache_key_hash2  # this fails and I'm unsure how to fix it atm


def test_field_change():
    Fp = Field("finite field", 2 ** 31 - 1, 1)
    oPsFp = Particles(6, field=Field("rational", 0, 0), seed=0)
    oPsFp.to_field(Fp)
    Qp = Field("padic", 2 ** 31 - 1, 4)
    oPsQp = Particles(6, field=Field("rational", 0, 0), seed=0)
    oPsQp.to_field(Qp)
    assert oPsFp("⟨1|2⟩") == oPsQp("⟨1|2⟩").as_tuple[0]


@pytest.mark.parametrize(
    "field", (Field("finite field", 2 ** 31 - 19, 1), Field("mpc", 0, 300))
)
def test_massive_vector_polarization(field):
    # pseudo random phase space point at this seed
    seed = 0

    # six-point massless phase space al la BDK 9708239
    oPs6pt = Particles(6, field=field, seed=seed, real_momenta=False, fix_mom_cons=(1, 2))

    # set the values of the masses
    # oPs6pt._singular_variety(("⟨56⟩+[56]", ), (0, ), seed=seed)  # this forces chiral and anti-chiral masses to be the same
    oPs6pt.m2 = oPs6pt("s_56")
    m2 = oPs6pt.m2
    # m = field.sqrt(m2) - if you want it fro mthe sqrt
    m = oPs6pt("⟨65⟩")
    mbar = oPs6pt("[56]")

    # sanity check on the massses
    assert abs(m2 - m * mbar) <= field.tollerance

    # re-interpret the random massless six-point phase space in terms of five-point one-mass phase space
    # the massless spinors 5 and 6 become the components of the bold spinor 𝟓
    oPs5pt_u = oPs6pt.cluster([[1, ], [2, ], [3, ], [4, ], [5, 6]], massive_fermions=((5, 'u', all), ))  # up index   5^I keeping all values for I = {1, 2}
    oPs5pt_d = oPs6pt.cluster([[1, ], [2, ], [3, ], [4, ], [5, 6]], massive_fermions=((5, 'd', all), ))  # down index 5_I keeping all values for I = {1, 2}
    oPs5pt_AuSd = oPs6pt.cluster([[1, ], [2, ], [3, ], [4, ], [5, 6]], massive_spins=((5, ('d', all), ('u', all), ), ))
    oPs5pt = oPs5pt_u

    # check the p-slash is related to the out products of the spinors, explicitly:
    assert (abs(oPs5pt[5].r2_sp_b - (numpy.outer(oPs5pt_u("|𝟓⟩")[:, 0], oPs5pt_d("[𝟓|")[0, :]) +
                                     numpy.outer(oPs5pt_u("|𝟓⟩")[:, 1], oPs5pt_d("[𝟓|")[1, :]))) <= field.tollerance).all()

    # check the p-slash is related to the out products of the spinors, explicitly:
    assert (abs(oPs5pt[5].r2_sp_b - (numpy.outer(oPs5pt_AuSd("|𝟓⟩")[:, 0], oPs5pt_AuSd("[𝟓|")[0, :]) +
                                     numpy.outer(oPs5pt_AuSd("|𝟓⟩")[:, 1], oPs5pt_AuSd("[𝟓|")[1, :]))) <= field.tollerance).all()

    εm = ε11 = - numpy.einsum("i,mij,j->m", oPs5pt("⟨𝟓|")[0, :], Pauli_bar, oPs5pt("|𝟓]")[:, 0]) / (field.sqrt(2) * mbar)  # noqa

    ε22 = - numpy.einsum("i,mij,j->m", oPs5pt("⟨𝟓|")[1, :], Pauli_bar, oPs5pt("|𝟓]")[:, 1]) / (field.sqrt(2) * m)
    εp = - ε22

    ε0 = (numpy.einsum("i,mij,j->m", oPs5pt("⟨𝟓|")[0, :], Pauli_bar, oPs5pt("|𝟓]")[:, 1]) +
          numpy.einsum("i,mij,j->m", oPs5pt("⟨𝟓|")[1, :], Pauli_bar, oPs5pt("|𝟓]")[:, 0])) / (field.sqrt(2) * m) / field.sqrt(2)
    ε0c = ε0 * m / mbar

    assert numpy.all(
        abs((oPs6pt[5].four_mom - oPs6pt[6].four_mom) - (
            numpy.einsum("i,mij,j->m", oPs5pt_AuSd("⟨𝟓|")[0, :], Pauli_bar, oPs5pt_AuSd("|𝟓]")[:, 0]) -
            numpy.einsum("i,mij,j->m", oPs5pt_AuSd("⟨𝟓|")[1, :], Pauli_bar, oPs5pt_AuSd("|𝟓]")[:, 1])) / 2) <= field.tollerance
    )

    # Normalisation: $\epsilon_\lambda \cdot \epsilon_{\lambda'} = - \delta_{\lambda\lambda'}$

    assert abs(εp @ MinkowskiMetric @ εp) <= field.tollerance
    assert abs(εp @ MinkowskiMetric @ εm + 1) <= field.tollerance
    assert abs(εp @ MinkowskiMetric @ ε0) <= field.tollerance
    assert abs(εm @ MinkowskiMetric @ εm) <= field.tollerance
    assert abs(εm @ MinkowskiMetric @ ε0) <= field.tollerance
    assert abs(ε0 @ MinkowskiMetric @ ε0c + 1) <= field.tollerance

    # On-shell transversality: $p \cdot \epsilon = 0$

    assert abs(oPs5pt[5].four_mom_d @ εp) <= field.tollerance
    assert abs(oPs5pt[5].four_mom_d @ εm) <= field.tollerance
    assert abs(oPs5pt[5].four_mom_d @ ε0) <= field.tollerance

    # Completeness $\sum_\lambda \epsilon^\mu_\lambda \cdot \epsilon^\nu_{\lambda} = - \eta^{\mu\nu} + p^\mu p^\nu / m^2$

    assert (abs((numpy.outer(εp, εm) + numpy.outer(εm, εp) + numpy.outer(ε0, ε0c)) -
                (- MinkowskiMetric + numpy.outer(oPs5pt[5].four_mom, oPs5pt[5].four_mom) / (m * mbar))) <= field.tollerance).all()
