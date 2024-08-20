# Lorentz Invariant Phase Space

[![CI Lint](https://github.com/GDeLaurentis/lips/actions/workflows/ci_lint.yml/badge.svg)](https://github.com/GDeLaurentis/lips/actions/workflows/ci_lint.yml)
[![CI Test](https://github.com/GDeLaurentis/lips/actions/workflows/ci_test.yml/badge.svg)](https://github.com/GDeLaurentis/lips/actions/workflows/ci_test.yml)
[![Coverage](https://img.shields.io/badge/Coverage-82%25-greenyellow?labelColor=2a2f35)](https://github.com/GDeLaurentis/lips/actions)
[![Docs](https://github.com/GDeLaurentis/lips/actions/workflows/cd_docs.yml/badge.svg?label=Docs)](https://gdelaurentis.github.io/lips/)
[![PyPI](https://img.shields.io/pypi/v/lips?label=PyPI)](https://pypi.org/project/lips/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/lips.svg?label=PyPI%20downloads)](https://pypi.org/project/lips/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GDeLaurentis/lips/HEAD)
[![DOI](https://zenodo.org/badge/210320784.svg)](https://zenodo.org/doi/10.5281/zenodo.11518261)
[![Python](https://img.shields.io/pypi/pyversions/lips?label=Python)](https://pypi.org/project/lips/)


`Lips` is a Python 3 library that provides a phase-space generator and manipulator that is tailored to the needs of modern theoretical calculations in quantum field theory. At present, the package is designed to handle the kinematics of scattering processes involving an arbitrary number of massless particles. Use cases include: 
    
1) generation of phase-space points over complex numbers ($\mathbb{C}$), finite fields ($\mathbb{F}_p$), and $p$-adic numbers ($\mathbb{Q}_p$);
2) generation of spinor strings representing possible kinematic singularities (related to letter of the symbol alphabet);
3) on-the-fly evaluation of arbitrary spinor-helicity expressions in any of the above mentioned fields;
4) construction of special kinematic configurations, with efficient, hard-coded solutions available up to codimension 2;
5) algebro-geometric analysis of irreducible varieties in kinematic space.

## Installation
The package is available on the [Python Package Index](https://pypi.org/project/lips/)
```console
pip install lips
```
Alternativelty, it can be installed by cloning the repo
```console
git clone https://github.com/GDeLaurentis/lips.git path/to/repo
pip install -e path/to/repo
```

## Requirements
`pip` will automatically install the required packages, which are
```
numpy, sympy, mpmath, pyadic
```
The `algebraic_gemetry` submodule requires [Singular](https://www.singular.uni-kl.de/) through the Python interface [syngular](https://github.com/GDeLaurentis/syngular). Singular needs to be installed manually (e.g. `apt-get install singular`).

## Testing

```
pytest --cov lips/ --cov-report html tests/ --verbose
```


## Citation

If you found this library useful, please consider citing it


```bibtex
@inproceedings{DeLaurentis:2023qhd,
    author = "De Laurentis, Giuseppe",
    title = "{Lips: $p$-adic and singular phase space}",
    booktitle = "{21th International Workshop on Advanced Computing and Analysis Techniques in Physics Research}: {AI meets Reality}",
    eprint = "2305.14075",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "PSI-PR-23-14",
    month = "5",
    year = "2023"
}

@phdthesis{DeLaurentis:2020xar,
    author = "De Laurentis, Giuseppe",
    title = "{Numerical techniques for analytical high-multiplicity scattering amplitudes}",
    school = "Durham U.",
    year = "2020"
}
```
