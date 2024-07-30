#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe

import sys
import re
import ast
import functools
import mpmath
import operator as op

from fractions import Fraction
from pyadic import PAdic, ModP, GaussianRational

operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg, ast.UAdd: op.pos, ast.MatMult: op.matmul}


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
pMi = re.compile(r'(?:m|M)(?:_){0,1}(\d)')
pMVar = re.compile(r'(?<![a-zA-Z])((?:m|M|μ)(?:_){0,1}[a-zA-Z]*[\d]*)')
pOijk = re.compile(r'(?:Ω_)(\d+)')
pPijk = re.compile(r'(?:Π_)(\d+)')
pDijk_adjacent = re.compile(r'(?:Δ_(\d+)(?![\d\|]))')
pDijk_non_adjacent = re.compile(r'(?:Δ_(\d+)\|(\d+)\|(\d+))')
# p3B = re.compile(r'(?:\u27e8|\[)(\d+)(?:\|\({0,1})([\d+[\+|-]*]*)(?:\){0,1}\|)(\d+)(?:\u27e9|\])')
# pNB = re.compile(r'((?:⟨|\[)\d+\|(?:(?:\([\d+\+|-]{1,}\))|(?:[\d+\+|-]{1,}))*\|\d+(?:⟩|\]))')  # this messes up on strings like: '|2⟩⟨1|4+5|3|+|3|4+5|2⟩⟨1|'
pNB = re.compile(r'((?:⟨|\[)\d+\|(?:(?:(?!\|[\+|-]\|)\([\d+\+|-]{1,}\))|(?:(?!\|[\+|-]\|)[\d+\+|-]))*\|\d+(?:⟩|\]))')
pNB_open_begin = re.compile(r'(?<!⟨\d)(?<!\[\d)(?<![\+|-]\d\))(?<![\+|-]\d)((?:\|)(?:(?:\([\d+|-]{1,}\))|(?:[\d+|-]{1,}))*(?:\|)\d+(?:⟩|\]))')
pNB_open_end = re.compile(r'((?:⟨|\[)\d+(?:\|)(?:(?:\([\d+|-]{1,}\))|(?:[\d+\+|-]{1,}))*(?:\|))(?!\d⟩)(?!\d\])(?!\d[\+|-])(?!\(\d[\+|-])')
ptr5 = re.compile(r'tr5(_\d+|\([\d\|\+\-]+\))')
ptr = re.compile(r'(tr\((?:(?:\([\d+\+|-]{1,}\))|(?:[\d+\+|-]{1,})*)\))')

unicode_powers_dict = {"^0": "⁰", "^1": "¹", "^2": "²", "^3": "³", "^4": "⁴", "^5": "⁵", "^6": "⁶", "^7": "⁷", "^8": "⁸", "^9": "⁹"}


def non_unicode_powers(string):
    for hat_pow, uni_pow in unicode_powers_dict.items():
        string = string.replace(uni_pow, hat_pow)
    return string


def unicode_powers(string):
    for hat_pow, uni_pow in unicode_powers_dict.items():
        string = string.replace(hat_pow, uni_pow)
    return string


def as_scalar_if_scalar(func):
    """Turns numpy arrays with zero dimensions into 'real' scalars."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if hasattr(res, "shape"):
            if res.shape == ():
                return res[()]  # pops the scalar out of array(scalar) or does nothing if array has non-trivial dimensions.
            elif functools.reduce(op.mul, res.shape) == 1:
                return res.flatten()[0]
        return res
    return wrapper

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
        string = non_unicode_powers(string)
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
        string = pMi.sub(r"oPs.compute('m_\1')", string)
        string = pMVar.sub(r"oPs.compute('\1')", string)
        string = pOijk.sub(r"oPs.compute('Ω_\1')", string)
        string = pPijk.sub(r"oPs.compute('Π_\1')", string)
        string = ptr5.sub(r"oPs.compute('tr5\1')", string)
        string = ptr.sub(r"oPs.compute('\1')", string)
        string = pDijk_adjacent.sub(r"oPs.compute('Δ_\1')", string)
        string = pDijk_non_adjacent.sub(r"oPs.compute('Δ_\1|\2|\3')", string)
        string = pNB.sub(r"oPs.compute('\1')", string)
        string = pNB_open_begin.sub(r"oPs.compute('\1')", string)
        string = pNB_open_end.sub(r"oPs.compute('\1')", string)
        string = re.sub(r'(\d)s', r'\1*s', string)
        string = re.sub(r'(\d)o', r'\1*o', string)
        string = re.sub(r'(?<!tr)(\d)\(', r'\1*(', string)
        string = re.sub(r'\)(\d)', r')*\1', string)
        string = string.replace(')(', ')*(').replace(")o", ")*o")
        re_rat_nbr = re.compile(r"(?<!\*\*)(\d+)\/(\d+)")
        string = re_rat_nbr.sub(r"Fraction(\1,\2)", string)
        return string

    @as_scalar_if_scalar
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
        if node.id in locals() and type(locals()[node.id]) in [int, float, GaussianRational, PAdic, ModP, mpmath.mpc, mpmath.mpc, Fraction]:
            return eval(node.id)
        else:
            raise TypeError(node)
    else:
        raise TypeError(node)
