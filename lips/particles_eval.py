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

from fractions import Fraction

operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg, ast.UAdd: op.pos}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


pA2 = re.compile(r'(?:\u27e8)(\d+)(?:\|)(\d+)(?:\u27e9)')
pA2bis = re.compile(r'(?:(?:\u27e8)(\d)(\d)(?:\u27e9))')
pAu = re.compile(r'(?:⟨|<)(\d+)(?:\|)(?!\d[⟩|>])(?!\d[\+|-])(?!\(\d[\+|-])')
pAd = re.compile(r'(?<![⟨|<]\d)(?<![\+|-]\d\))(?<![\+|-]\d)(?:\|)(\d+)(?:⟩|>)')
pS2 = re.compile(r'(?:\[)(\d+)(?:\|)(\d+)(?:\])')
pSd = re.compile(r'(?:\[)(\d+)(?:\|)(?!\d\])(?!\d[\+|-])(?!\(\d[\+|-])')
pSu = re.compile(r'(?<!\[\d)(?<![\+|-]\d\))(?<![\+|-]\d)(?:\|)(\d+)(?:\])')
pS2bis = re.compile(r'(?:\[)(\d)(\d)(?:\])')
pSijk = re.compile(r'(?:s|S)(?:_){0,1}(\d+)')
pOijk = re.compile(r'(?:Ω_)(\d+)')
pPijk = re.compile(r'(?:Π_)(\d+)')
pDijk_adjacent = re.compile(r'(?:Δ_(\d+)(?![\d\|]))')
pDijk_non_adjacent = re.compile(r'(?:Δ_(\d+)\|(\d+)\|(\d+))')
# p3B = re.compile(r'(?:\u27e8|\[)(\d+)(?:\|\({0,1})([\d+[\+|-]*]*)(?:\){0,1}\|)(\d+)(?:\u27e9|\])')
pNB = re.compile(r'((?:⟨|\[)\d+\|(?:(?:\([\d+\+|-]{1,}\))|(?:[\d+\+|-]{1,}))*\|\d+(?:⟩|\]))')
ptr5 = re.compile(r'(?:tr5_)(\d+)')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Particles_Eval:

    @staticmethod
    def _parse(string):
        string = string.replace("−", "-")
        string = string.replace(r"\scriptscriptstyle", "").replace("<", "⟨").replace(">", "⟩")
        string = string.replace(r"\frac{", "(")
        string = string.replace(r"}{", ")/(")
        string = string.replace("}+\\", ")+")
        string = string.replace("\n", "").replace(" ", "")
        string = re.sub(r"}$", ")", string)
        string = string.replace("²", "^2", ).replace("³", "^3", ).replace("⁴", "^4")
        string = string.replace("^", "**")
        string = re.sub(r"{([\d\|]+)}", r"\1", string)
        string = pA2bis.sub(r"⟨\1|\2⟩", string)
        string = pA2.sub(r"oPs.compute('⟨\1|\2⟩')", string)
        string = pAu.sub(r"oPs.compute('⟨\1|')", string)
        string = pAd.sub(r"oPs.compute('|\1⟩')", string)
        string = pS2bis.sub(r"[\1|\2]", string)
        string = pS2.sub(r"oPs.compute('[\1|\2]')", string)
        string = pSd.sub(r"oPs.compute('[\1|')", string)
        string = pSu.sub(r"oPs.compute('|\1]')", string)
        string = pSijk.sub(r"oPs.compute('s_\1')", string)
        string = pOijk.sub(r"oPs.compute('Ω_\1')", string)
        string = pPijk.sub(r"oPs.compute('Π_\1')", string)
        string = ptr5.sub(r"oPs.compute('tr5_\1')", string)
        string = pDijk_adjacent.sub(r"oPs.compute('Δ_\1')", string)
        string = pDijk_non_adjacent.sub(r"oPs.compute('Δ_\1|\2|\3')", string)
        string = pNB.sub(r"oPs.compute('\1')", string)
        string = re.sub(r'(\d)s', r'\1*s', string)
        string = re.sub(r'(\d)o', r'\1*o', string)
        string = re.sub(r'(\d)\(', r'\1*(', string)
        string = re.sub(r'\)(\d)', r')*\1', string)
        string = string.replace(')(', ')*(').replace(")o", ")*o")
        re_rat_nbr = re.compile(r"(?<!\*\*)(\d+)\/(\d+)")
        string = re_rat_nbr.sub(r"Fraction(\1,\2)", string)
        return string

    def _eval(self, string):
        return ast_eval_expr(self._parse(string), {'oPs': self})


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def ast_eval_expr(expr, locals_={}):
    return _eval_node(ast.parse(expr, mode='eval').body, locals_=locals_)


def _eval_node(node, locals_={}):
    locals().update(locals_)
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](_eval_node(node.left, locals_), _eval_node(node.right, locals_))
    elif isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](_eval_node(node.operand, locals_))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.value.id == 'oPs' and node.func.attr == 'compute':
            function, method = node.func.value.id, node.func.attr
            argument = node.args[0].s if sys.version_info[0] > 2 else node.args[0].s.decode('utf-8')
            allowed_func_call = "{function}.{method}('{argument}')".format(function=function, method=method, argument=argument)
        elif isinstance(node.func, ast.Attribute) and node.func.value.id == 'mpmath' and node.func.attr in ['mpf', 'sqrt']:
            function, method = 'mpmath', node.func.attr
            if hasattr(node.args[0], 'id'):
                argument = ast_eval_expr(node.args[0].id, locals_)
            else:
                argument = _eval_node(node.args[0], locals_)
            allowed_func_call = "{function}.{method}('{argument}')".format(function=function, method=method, argument=argument)
        elif isinstance(node.func, ast.Name) and node.func.id == 'PAdic':
            function, arguments = 'PAdic', [arg.n for arg in node.args]
            allowed_func_call = "{function}({arguments})".format(function=function, arguments=", ".join(map(str, arguments)))
        elif isinstance(node.func, ast.Name) and node.func.id == 'Fraction':
            function, arguments = 'Fraction', [arg.n for arg in node.args]
            allowed_func_call = "{function}({arguments})".format(function=function, arguments=", ".join(map(str, arguments)))
        else:
            raise TypeError(node)
        return eval(allowed_func_call)
    elif isinstance(node, ast.Name):
        if node.id in locals() and type(locals()[node.id]) in [int, float, mpmath.mpc, mpmath.mpc, Fraction]:
            return eval(node.id)
        else:
            raise TypeError(node)
    else:
        raise TypeError(node)
