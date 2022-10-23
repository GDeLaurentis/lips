# Lorentz Invariant Phase Space

[![Continuous Integration Status](https://github.com/GDeLaurentis/lips/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/GDeLaurentis/lips/actions)
[![Coverage](https://img.shields.io/badge/Coverage-80%25-greenyellow?labelColor=2a2f35)](https://github.com/GDeLaurentis/lips-dev/actions)
[![PyPI Downloads](https://img.shields.io/pypi/dm/lips.svg?label=PyPI%20downloads)](https://pypi.org/project/lips/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GDeLaurentis/lips/HEAD)


## Requirements
```
numpy, mpmath, sympy
```
Algebraic-gemetry tools require
```
Singular
```

## Installation
```
pip install -e path/to/repo
```

## Testing

```
pytest --cov lips/ --cov-report html tests/ --verbose
```
