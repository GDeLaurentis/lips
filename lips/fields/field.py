# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mpmath

from lips.fields.padic import PAdic

mpmath.mp.dps = 300


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Field(object):

    def __init__(self, *args):
        """Defaults to ('mpc', 0, 300)"""
        if args == ():
            args = ('mpc', 0, 300)
        self.set(args)

    def set(self, *args):
        """(name, characteristic, digits)"""
        if len(args) == 1:
            self.name = args[0][0]
            self.characteristic = args[0][1]
            self.digits = args[0][2]
        else:
            self.name = args[0]
            self.characteristic = args[1]
            self.digits = args[2]

    def __str__(self):
        return "({}, {}, {})".format(self.name, self.characteristic, self.digits)

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if value not in ['mpc', 'gaussian rational', 'finite field', 'padic']:
            raise Exception("Field must be one of 'mpc', 'gaussian rational', 'finite field', 'padic'.")
        else:
            self._name = value

    @property
    def characteristic(self):
        return self._characteristic

    @characteristic.setter
    def characteristic(self, value):
        if value < 0:
            raise Exception("Characteristic must be non-negative.")
        else:
            self._characteristic = value

    @property
    def digits(self):
        if self.name == 'mpc':
            return mpmath.mp.dps
        else:
            return self._digits

    @digits.setter
    def digits(self, value):
        if value < 0:
            raise Exception("Digits must be positive.")
        elif self.name == 'mpc':
            mpmath.mp.dps = value
        else:
            self._digits = value

    @property
    def tollerance(self):
        if self.name in ['gaussian rational', 'finite field']:
            return 0
        elif self.name == 'mpc':
            return mpmath.mpf('10e-{}'.format(int(min([0.95 * mpmath.mp.dps, mpmath.mp.dps - 4]))))
        elif self.name == 'padic':
            return PAdic(0, self.characteristic, 0, self.digits)

    @property
    def singular_notation(self):
        if self.name == 'mpc':
            return '(complex,{},I)'.format(self.digits)
        elif self.name in ['finite field', 'padic']:
            return str(self.characteristic)
        else:
            return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


field = Field()
