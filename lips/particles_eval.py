#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe

import sys
import re
import ast
import functools
import numpy
import mpmath
import operator as op

from fractions import Fraction
from pyadic import PAdic, ModP, GaussianRational
from syngular.monomial import non_unicode_powers

from .tools import LeviCivita


operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg, ast.UAdd: op.pos, ast.MatMult: op.matmul}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


pA2 = re.compile(r'(?:\u27e8)(\d+)(?:\|)(\d+)(?:\u27e9)')
pA2bis = re.compile(r'(?:(?:\u27e8)(\d)(\d)(?:\u27e9))')
pAu = re.compile(r'(?<!\(\')(?:⟨|<)(\d+)(?:\|)(?!\d[⟩|>])(?!\d[\+|-])(?!\(\d[\+|-])(?!\'\))')
pAd = re.compile(r'(?<!\(\')(?<![⟨|<]\d)(?<![\+|-]\d\))(?<![\+|-]\d)(?:\|)(\d+)(?:⟩|>)(?!\'\))')
pS2 = re.compile(r'(?:\[)(\d+)(?:\|)(\d+)(?:\])')
pSd = re.compile(r'(?<!\(\')(?:\[)(\d+)(?:\|)(?!\d\])(?!\d[\+|-])(?!\(\d[\+|-])(?!\'\))')
pSu = re.compile(r'(?<!\(\')(?<!\[\d)(?<![\+|-]\d\))(?<![\+|-]\d)(?:\|)(\d+)(?:\])(?!\'\))')
pS2bis = re.compile(r'(?:\[)(\d)(\d)(?:\])')
pSijk = re.compile(r'(?:s|S)(?:_){0,1}(\d+)')
pMi = re.compile(r'(?:m|M)(?:_){0,1}(\d)')
pMVar = re.compile(r'(?<![a-zA-Z])((?:m|M|μ)(?:_){0,1}[a-zA-Z]*[\d]*)')
pOijk = re.compile(r'(?:Ω_)(\d+)')
pPijk = re.compile(r'(?:Π_)(\d+)')
pDijk_adjacent = re.compile(r'(?:Δ_(\d+)(?![\d\|]))')
pDijk_non_adjacent = re.compile(r'(?:Δ_(\d+(?:\|\d+)*))')
# p3B = re.compile(r'(?:\u27e8|\[)(\d+)(?:\|\({0,1})([\d+[\+|-]*]*)(?:\){0,1}\|)(\d+)(?:\u27e9|\])')
# pNB = re.compile(r'((?:⟨|\[)\d+\|(?:(?:\([\d+\+|-]{1,}\))|(?:[\d+\+|-]{1,}))*\|\d+(?:⟩|\]))')  # this messes up on strings like: '|2⟩⟨1|4+5|3|+|3|4+5|2⟩⟨1|'
pNB = re.compile(r'((?:<|⟨|\[)\d+\|(?:\(?(?:\d+[\+|-]?)+\)?\|?)+\|\d+(?:⟩|\]|>))')
pNB_open_begin = re.compile(
    r'(?<!\(\')'  # negative lookbehind for already matched expression
    r'(?<![\(\[⟨<]\d)(?<![\+|-]\d\))(?<![\+|-]\d)'  # negative lookbehind
    r'(?<!_\d)(?<!_\d\d)(?<!_\d\d\d)(?<!_\d\d\d\d)'  # negative lookbehind for e.g. Δ_
    r'(\|(?:(?:\([\d]+(?:[\+|-]\d+)*\))|(?:[\d]+(?:[\+|-]\d+)*))+\|'  # capture
    r'\d+[⟩\]])'  # capture end
    r'(?!\'\))'  # negative lookahead for already matched expression
)
pNB_open_end = re.compile(
    r'(?<!\(\')'  # negative lookbehind for already matched expression
    r'([⟨<\[]\d+'  # capture beginning
    r'\|(?:(?:\([\d+|-]+(?:[\+|-]\d+)*\))|(?:[\d]+(?:[\+|-]\d+)*))+\|)'  # capture
    r'(?!\d[⟩>\]])(?!\d[\+|-])(?!\(\d[\+|-])'  # negative lookahead
    r'(?!\'\))'  # negative lookahead for already matched expression
)
pNB_double_open = re.compile(
    r'(?<!\(\')'  # negative lookbehind for already matched expression
    r'(?<!\'\[\d\|\'\)\]\]\)\)\[0:1, :\]@)'  # negative lookbehind to disambiguate open index position
    r'(?<![\(\[⟨<]\d)(?<![\+|-]\d\))(?<![\+|-]\d)'  # negative lookbehind
    r'(?<!_\d)(?<!_\d\d)(?<!_\d\d\d)(?<!_\d\d\d\d)'  # negative lookbehind for e.g. Δ_
    r'(?<!\|\d)(?<!\|\d\d)(?<!\|\d\d\d)(?<!\|\d\d\d\d)'  # negative lookbehind for e.g. Δ_12|
    r'(\|(?:(?:\([\d+|-]+(?:[\+|-]\d+)*\))|(?:[\d]+(?:[\+|-]\d+)*))+\|)'  # capture
    r'(?!\d[⟩>\]])(?!\d[\+|-])(?!\(\d[\+|-])'  # negative lookahead
    r'(?!\'\))'  # negative lookahead for already matched expression
)
pNB_double_open_disambiguate_alphadot = re.compile(  # DO NOT MODIFY HERE: COPY FROM ABOVE ONE
    r'(?<!\(\')'  # negative lookbehind for already matched expression
    r'(?<=\'\[\d\|\'\)\]\]\)\)\[0:1, :\]@)'  # positive lookbehind to disambiguate open index position
    r'(?<![\(\[⟨<]\d)(?<![\+|-]\d\))(?<![\+|-]\d)'  # negative lookbehind
    r'(?<!_\d)(?<!_\d\d)(?<!_\d\d\d)(?<!_\d\d\d\d)'  # negative lookbehind for e.g. Δ_
    r'(?<!\|\d)(?<!\|\d\d)(?<!\|\d\d\d)(?<!\|\d\d\d\d)'  # negative lookbehind for e.g. Δ_12|
    r'(\|(?:(?:\([\d+|-]+(?:[\+|-]\d+)*\))|(?:[\d]+(?:[\+|-]\d+)*))+\|)'  # capture
    r'(?!\d[⟩>\]])(?!\d[\+|-])(?!\(\d[\+|-])'  # negative lookahead
    r'(?!\'\))'  # negative lookahead for already matched expression
)
ptr5 = re.compile(r'tr5(_\d+|\([\d\|\+\-]+\))')
ptr = re.compile(r'(tr\((?:(?:\([\d+\+|-]{1,}\))|(?:[\d+\+|-]{1,})*)\))')
pMassiveSAu_SpinIndexd = re.compile(r'(?:⟨|<)(\d+)(_|d)\|')
pMassiveSAd_SpinIndexd = re.compile(r'\|(\d+)(_|d)(?:⟩|>)')
pMassiveSBd_SpinIndexu = re.compile(r'\[(\d+)(\^|u)\|')
pMassiveSBu_SpinIndexu = re.compile(r'\|(\d+)(\^|u)\]')


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
        string = re.sub(r"\^(?![\]⟩>\|])", r"**", string)
        string = re.sub(r"{([\d\|]+)}", r"\1", string)
        # this needs to be at top because it relies on the rest
        string = pMassiveSAu_SpinIndexd.sub(
            lambda match: f"(ϵ@block([[⟨{int(match.group(1))}|], [⟨{int(match.group(1)) + 1}|]]))[0:1, :]@|", string)
        string = pMassiveSAd_SpinIndexd.sub(
            lambda match: f"|@(block([|{int(match.group(1))}⟩, |{int(match.group(1)) + 1}⟩])@-ϵ)[:, 0:1]", string)
        string = pMassiveSBd_SpinIndexu.sub(
            lambda match: f"(-ϵ@block([[[{int(match.group(1))}|], [[{int(match.group(1)) + 1}|]]))[0:1, :]@|", string)
        string = pMassiveSBu_SpinIndexu.sub(
            lambda match: f"|@(block([|{int(match.group(1))}], |{int(match.group(1)) + 1}]])@ϵ)[:, 0:1]", string)
        string = string.replace("@|@", "@")
        string = pA2bis.sub(r"⟨\1|\2⟩", string)
        string = pA2.sub(r"oPs('⟨\1|\2⟩')", string)
        string = pS2bis.sub(r"[\1|\2]", string)
        string = pS2.sub(r"oPs('[\1|\2]')", string)
        string = pSijk.sub(r"oPs('s_\1')", string)
        string = pMi.sub(r"oPs('m_\1')", string)
        string = re.sub(r"([a-zA-Z\d]+)oPs", r"\1*oPs", string)
        string = pMVar.sub(r"oPs('\1')", string)
        string = pOijk.sub(r"oPs('Ω_\1')", string)
        string = pPijk.sub(r"oPs('Π_\1')", string)
        string = ptr5.sub(r"oPs('tr5\1')", string)
        string = ptr.sub(r"oPs('\1')", string)
        string = pDijk_adjacent.sub(r"oPs('Δ_\1')", string)
        string = pDijk_non_adjacent.sub(r"oPs('Δ_\1')", string)
        string = pNB.sub(r"oPs('\1')", string)
        # open index start
        string = pAu.sub(r"oPs('⟨\1|')", string)
        string = pAd.sub(r"oPs('|\1⟩')", string)
        string = pSd.sub(r"oPs('[\1|')", string)
        string = pSu.sub(r"oPs('|\1]')", string)
        string = pNB_open_begin.sub(r"oPs('\1')", string)
        string = pNB_open_end.sub(r"oPs('\1')", string)
        string = pNB_double_open.sub(r"oPs('\1')", string)
        string = pNB_double_open_disambiguate_alphadot.sub(r"numpy.transpose(-ϵ @ oPs('\1') @ ϵ)", string)
        # open index end
        string = string.replace("sqrt", "oPs.field.sqrt")
        string = re.sub(r'(\d)s', r'\1*s', string)
        string = re.sub(r'(\d)o', r'\1*o', string)
        string = re.sub(r'(?<!tr)(\d)\(', r'\1*(', string)
        string = re.sub(r'\)(\d)', r')*\1', string)
        string = string.replace(')(', ')*(').replace(")o", ")*o")
        re_rat_nbr = re.compile(r"(?<!\*\*)(\d+)\/(\d+)")
        string = re_rat_nbr.sub(r"Fraction(\1,\2)", string)
        string = re.sub(r'(?<!numpy\.)block', r'numpy.block', string)
        return string

    @as_scalar_if_scalar
    def _eval(self, string):
        return ast_eval_expr(self._parse(string), {'oPs': self, 'ε': LeviCivita, 'numpy': numpy})


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def ast_eval_expr(expr, locals_={}):
    return _eval_node(ast.parse(expr, mode='eval').body, locals_=locals_)


def _eval_node(node, locals_={}):
    if isinstance(node, ast.Constant):
        return node.n

    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](_eval_node(node.left, locals_), _eval_node(node.right, locals_))

    elif isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](_eval_node(node.operand, locals_))

    elif isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(el, locals_) for el in node.elts]

    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and hasattr(node.func, 'id') and node.func.id == 'oPs':
            function = node.func.id
            argument = node.args[0].s if sys.version_info[0] > 2 else node.args[0].s.decode('utf-8')
            allowed_func_call = f"{function}('{argument}')"
        elif isinstance(node.func, ast.Attribute) and node.func.attr in ['mpf', 'sqrt']:
            function, method = 'oPs.field', node.func.attr
            if hasattr(node.args[0], 'id'):
                argument = ast_eval_expr(node.args[0].id, locals_)
            else:
                argument = _eval_node(node.args[0], locals_)
            # return locals_['oPs'].field.sqrt(argument)  # could also avoid re-parsing the argument
            allowed_func_call = f"{function}.{method}({function}('{argument}'))"
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in ['numpy', ]:
            module_name, func_name = node.func.value.id, node.func.attr
            if module_name in locals_ and hasattr(locals_[module_name], func_name):
                func = getattr(locals_[module_name], func_name)
                args = [_eval_node(arg, locals_) for arg in node.args]
                kwargs = {kw.arg: _eval_node(kw.value, locals_) for kw in node.keywords}
                return func(*args, **kwargs)
            else:
                raise TypeError("Attribute not understood:", node, ast.dump(node))
        elif isinstance(node.func, ast.Name) and hasattr(node.func, 'id') and node.func.id == 'PAdic':
            function, arguments = 'PAdic', ", ".join(map(str, [arg.n for arg in node.args]))
            allowed_func_call = f"{function}({arguments})"
        elif isinstance(node.func, ast.Name) and hasattr(node.func, 'id') and node.func.id == 'Fraction':
            function, arguments = 'Fraction', ", ".join(map(str, [arg.n for arg in node.args]))
            allowed_func_call = f"{function}({arguments})"
        else:
            raise TypeError(node, ast.dump(node))
        return eval(allowed_func_call, None, locals_)

    elif isinstance(node, ast.Name):
        if node.id in locals_.keys() and isinstance(locals_[node.id],
                                                    (int, float, GaussianRational, PAdic, ModP,
                                                     mpmath.mpc, mpmath.mpc, Fraction, numpy.ndarray)):
            return eval(node.id, None, locals_)
        else:
            raise TypeError(node, ast.dump(node), locals_[node.id] if node.id in locals_.keys() else "Not in locals.")

    elif isinstance(node, ast.Subscript):
        value = _eval_node(node.value, locals_)  # Evaluate the base object

        def eval_slice(s):
            """ Recursively evaluate slice components. """
            if isinstance(s, ast.Slice):
                return slice(
                    _eval_node(s.lower, locals_) if s.lower else None,
                    _eval_node(s.upper, locals_) if s.upper else None,
                    _eval_node(s.step, locals_) if s.step else None
                )
            elif isinstance(s, ast.Tuple):  # Handle multi-dimensional slicing
                return tuple(eval_slice(dim) for dim in s.elts)
            else:  # Regular index (e.g., `x[3]`)
                return _eval_node(s, locals_)

        index = eval_slice(node.slice)
        return value[index]  # Perform the actual indexing/slicing

    else:
        raise TypeError(node, ast.dump(node))
