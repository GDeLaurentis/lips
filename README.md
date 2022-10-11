# Lorentz Invariant Phase Space

[![Continuous Integration Status](https://github.com/GDeLaurentis/lips-dev/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/GDeLaurentis/lips-dev/actions)
[![Coverage](https://img.shields.io/badge/Coverage-79%25-yellow?labelColor=2a2f35)](https://github.com/GDeLaurentis/lips-dev/actions)
[![PyPI Downloads](https://img.shields.io/pypi/dm/lips.svg?label=PyPI%20downloads)](https://pypi.org/project/lips/)

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
