import itertools
import numpy


def phase_weights_compatible_symmetries(phase_weights):
    """Returns all symmetries that leave the pahse weights unchanged, including the identity."""
    phase_weights = numpy.array(phase_weights)
    base = list(range(1, len(phase_weights) + 1))
    permutations = list(itertools.permutations(base))
    phase_weights_compatible_symmetries = list(filter(None, [
        (''.join(map(str, permutation)), False, ) if all([phase_weights[i - 1] for i in permutation] == phase_weights) else
        (''.join(map(str, permutation)), True, ) if all([phase_weights[i - 1] for i in permutation] == - phase_weights) else None for permutation in permutations]))
    return phase_weights_compatible_symmetries[:]


def inverse(permutation_or_symmetry):
    """Returns the inverse of a symmetry or permutation."""
    if isinstance(permutation_or_symmetry, str):
        inverse_permutation = [0 for entry in permutation_or_symmetry]
        for i, entry in enumerate(permutation_or_symmetry):
            inverse_permutation[int(entry) - 1] = i + 1
        inverse_permutation = "".join(map(str, inverse_permutation))
        return inverse_permutation
    elif isinstance(permutation_or_symmetry, tuple):
        return (inverse(permutation_or_symmetry[0]), permutation_or_symmetry[1])
    else:
        raise Exception(f"Input for inverse not understood: {permutation_or_symmetry}.")


def identity(n):
    return (''.join(str(i) for i in range(1, n + 1)), False)
