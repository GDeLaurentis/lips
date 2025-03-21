# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

- Support for arbitrary Gram determinants, e.g. for a box gram: `Δ_12|3|4|5`
- Support for spinor strings with two open indices, e.g.: `|1+2|3|4|`. The first open index is assumed to be a lower alpha.
- Raising and lowering of spinor indices does works in the presence of additional spin indices.
- `Particles.cluster` accepts a new keyword arguement `massive_fermions`, which allows to specify the states of the fermions. E.g. `massive_fermions=((3, 'u', all), (4, 'd', all))` results in tensor output with open indices IJ, or `massive_fermions=((3, 'u', 1), (4, 'd', 1))` picks the scalar I=1, J=1 component.

### Changed

- Support for Particles string evaluation containing sqrt (in arbitrary field insteald of only with `mpmath`).
- Particles compute and eval may return a tensor if additional spin indices are present.

### Fixed

- Sphinx fails if autodoc fails, instead of quitely raising a warning.

### Deprecated


## [0.4.5] - 2025-01-31

### Added

- Support for python 3.13
- Experimental feature to bypass `indepSet` calculation for `point_on_variety`

### Changed

- Improved `angles_for_squares` for massive states


## [0.4.4] - 2025-01-21

### Added

- `_sps_u_to_sps_d` and `_sps_d_to_sps_u` as shorthands to update lower-index spinors from upper-index spinors, or viceversa.

### Changed

- Conjugation operation `angles_for_squares` has been reworked to better support massive particles.
- `phasespace_consistency_check` can now handle open-index inputs.


## [0.4.3] - 2024-12-16

### Added

- `symmetries` submodule.
- `Particles.copy`.


## [0.4.2] - 2024-08-20

### Added

- `Particles.cluster` and `Particles.image` now copy the `internal_masses` values.
- Implemented new regular expression for trace of gamma five, e.g. `tr5(1|2|3|4+5)` is now valid. This shadows the definition of normal traces, `tr`, and allows for massive legs. NB: with the current implementation at least one of the four legs has to be massless (doesn't matter which one).
- `Particles.internal_masses` can be used to alias kinematic expressions to masses, e.g. `Particles.Mh2 = 's_45'` will define the squared Higgs mass as the Mandelstam 's_45'. Clustering, e.g. [4, 5] will correctly map this to 's_4'.

### Changed

- Numbers in the name of masses associated with the phase space point in `Particles.internal_masses` must appear at the end and will be interpreted as powers (for mass dimension considerations). NB: `m1` refers to the mass of the `Particle` at `Particles[1]`, `m2` to the second one, etc. Avoid name clashes.
- `Particles.variety` by default will now try the lex-groebner solver `Particles._singular_variety` when an hardcoded limit fails.

### Fixed

- Fixed issue with `Particles.variety` not correctly recognizing when an hardcoded limit failed with p-adics.

### Deprecated


## [0.4.1] - 2024-01-08

### Added

- `Particles` accepts new keyword argumnet `internal_masses`. It can be a dictionary of strings: values or one of list, tuple or set containing strings representing the masses. The format is expected to be `m` or `M` or `μ`, possibly followed by an underscore, and letters or numbers (explicitly: `^((?:m|M|μ)(?:_){0,1}[\da-zA-Z]*)$`). If it ends in numbers, those will be considered as an exponent (for mass dimension considerations).
- `Particles.from_singular_variety` classmethod to instantiate directly from a point on a variety.

### Changed

- Improved variety point generation, following updated to syngular.
- Tests run over python version 3.9 to 3.12.

### Fixed

- Fixed issue where a spinor component was the integer 1 instead of being 1 in the field.


## [0.4.0] - 2024-01-08

### Added

- Implemented valuator and setter for $(⟨a|b|c+d|e|a]-⟨b|f|c+d|e|b])$, from 2-loop 5-point 1-mass alphabet.
- Implemented evaluator for open-index spinor strings of arbitrary length.
- Implemented evalautor for $\text{tr}(a+b|c+d|\dots)$ invariants.
- Implemented singular setter for $\text{tr}_5$ at 5-point from solution to Lex-Groebner basis.

### Changed

- Updated `Particle.__init__` to accept rank-1 and rank-2 spinors as input, as well as 4-momenta.
- Updated `Particles.eval` to accept all unicode digits.
- Tweaked `Invariants` to return $\text{tr}_5$ invariant at 5-point.
- Simplified singular setter for $\text{tr}_5$ beyond 5-point.
- Number types can now be found in [pyadic](https://github.com/GDeLaurentis/pyadic).
- Field object can now be found in [syngular](https://github.com/GDeLaurentis/syngular). Import from lips is deprecated.
- Variety construction is now out-sourced to ring-agnostic algorithm in [syngular](https://github.com/GDeLaurentis/syngular).

### Fixed

- Fixed issue where manipulations (such as via arithmetic operations) of `Particles` would lose track of the underlying field.


## [0.3.1] - 2023-01-16

### Added

- p-adic and finite field phase space points.
- Algebro-geomtric phase space point manipulations: ideals and points on varieties.


## [0.1.3] - 2019-10-18

### Added

- Basic phase space generation and manipulation.
- Numerical computation of Lorentz invariant spinor strings.


[unreleased]: https://github.com/GDeLaurentis/lips/compare/v0.4.5...HEAD
[0.4.5]: https://github.com/GDeLaurentis/lips/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/GDeLaurentis/lips/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/GDeLaurentis/lips/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/GDeLaurentis/lips/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/GDeLaurentis/lips/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/GDeLaurentis/lips/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/GDeLaurentis/lips/compare/v0.1.3...v0.3.1
[0.1.3]: https://github.com/GDeLaurentis/lips/releases/tag/v0.1.3
