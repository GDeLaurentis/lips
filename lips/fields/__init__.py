from .field import Field  # noqa
from .gaussian_rationals import GaussianRational  # noqa

import warnings

warnings.warn("The lips.fields module is deprecated. Please use syngular.field instead.", DeprecationWarning, stacklevel=2)
