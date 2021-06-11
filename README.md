# Lorentz Invariant Phase Space

[![Continuous Integration Status](https://github.com/GDeLaurentis/lips-dev/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/GDeLaurentis/lips-dev/actions)
[![Coverage](https://img.shields.io/badge/Coverage-69%25-orange?labelColor=2a2f35)](https://github.com/GDeLaurentis/lips-dev/actions)

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