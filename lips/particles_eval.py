#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import re
import ast
import mpmath
import operator as op

mpmath.mp.dps = 300

operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

pA2 = re.compile(r'(?:\u27e8)(\d+)(?:\|)(\d+)(?:\u27e9)')
pA2bis = re.compile(r'(?:(?:\u27e8)(\d)(\d)(?:\u27e9))')
pS2 = re.compile(r'(?:\[)(\d+)(?:\|)(\d+)(?:\])')
pS2bis = re.compile(r'(?:\[)(\d)(\d)(?:\])')
pSijk = re.compile(r'(?:s_|S_)(\d+)')
pOijk = re.compile(r'(?:Ω_)(\d+)')
pPijk = re.compile(r'(?:Π_)(\d+)')
p3B = re.compile(r'(?:\u27e8|\[)(\d+)(?:\|\({0,1})([\d+[\+|-]*]*)(?:\){0,1}\|)(\d+)(?:\u27e9|\])')


class Particles_Eval:

    @staticmethod
    def _parse(string):
        string = string.replace("^", "**")
        string = pA2bis.sub(r"⟨\1|\2⟩", string)
        string = pA2.sub(r"self.compute('⟨\1|\2⟩')", string)
        string = pS2bis.sub(r"[\1|\2]", string)
        string = pS2.sub(r"self.compute('[\1|\2]')", string)
        string = pSijk.sub(r"self.compute('s_\1')", string)
        string = pOijk.sub(r"self.compute('Ω_\1')", string)
        string = pPijk.sub(r"self.compute('Π_\1')", string)
        string = p3B.sub(r"self.compute('⟨\1|(\2)|\3]')", string)
        string = re.sub(r'(\d)s', r'\1*s', string)
        string = string.replace(')s', ')*s')
        string = string.replace(')(', ')*(')
        return string

    def _eval_expr(self, expr):
        return self._eval_node(ast.parse(expr, mode='eval').body)

    def _eval_node(self, node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](self._eval_node(node.left), self._eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](self._eval_node(node.operand))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.value.id == 'self' and node.func.attr == 'compute':
                function, method = node.func.value.id, node.func.attr
                if sys.version_info[0] == 2:
                    argument = node.args[0].s.decode('utf-8')
                else:
                    argument = node.args[0].s
                allowed_func_call = "{function}.{method}('{argument}')".format(function=function, method=method, argument=argument)
                return eval(allowed_func_call)
            else:
                raise TypeError(node)
        else:
            raise TypeError(node)

    def eval(self, string):
        return self._eval_expr(self._parse(string))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
