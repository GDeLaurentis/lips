# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Particles` accepts new keyword argumnet `internal_masses`. It can be a dictionary of strings: values or one of list, tuple or set containing strings representing the masses. The format is expected to be `m` or `M` or `μ`, possibly followed by an underscore, and letters or numbers (explicitly: `^((?:m|M|μ)(?:_){0,1}[\da-zA-Z]*)$`). If it ends in numbers, those will be considered as an exponent (for mass dimension considerations).
- `Particles.from_singular_variety` classmethod to instantiate directly from a point on a variety.

### Changed

- Improved variety point generation, following updated to syngular.
- Tests run over python version 3.9 to 3.12.

### Fixed

- Fixed issue where a spinor component was the integer 1 instead of being 1 in the field.

### Deprecated

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


[unreleased]: https://github.com/GDeLaurentis/lips/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/GDeLaurentis/lips/releases/tag/v0.4.0
[0.3.1]: https://github.com/GDeLaurentis/lips/releases/tag/v0.3.1
[0.1.3]: https://github.com/GDeLaurentis/lips/releases/tag/v0.1.3
