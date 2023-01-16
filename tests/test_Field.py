from lips.fields.field import Field


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_type_casting():

    assert Field("mpc", 0, 300)(0) == 0  # what about tollerance here
    assert Field("padic", 2 ** 31 - 1, 5)(0) == 0  # and here
    assert Field("finite field", 2 ** 31 - 1, 1)(0) == 0
