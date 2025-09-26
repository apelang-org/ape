#!/usr/bin/env python3

import sys, typing, traceback
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

class TokenKind(IntEnum):
  END_OF_INPUT = 128
  ERROR = 129
  INDENT = 130
  # DEDENT = 131 # NOTE(dfra): we use INDENT[length] and handle dedents in the parser.
  NEWLINE = 132

  IDENTIFIER = 133
  INTEGER = 134
  FLOAT = 135
  STRING = 136

  LTLT = 148
  GTGT = 149
  DASHGT = 150
  STARSTAR = 151
  SLASHSLASH = 152
  COLONEQ = 153
  EQEQ = 154
  BANGEQ = 155
  LTEQ = 156
  GTEQ = 157
  PLUSEQ = 158
  DASHEQ = 159
  STAREQ = 160
  ATEQ = 161
  SLASHEQ = 162
  PERCENTEQ = 163
  AMPEQ = 164
  PIPEEQ = 165
  CARETEQ = 166

  DOTDOTDOT = 167
  LTLTEQ = 168
  GTGTEQ = 169
  STARSTAREQ = 170
  SLASHSLASHEQ = 171

  KEYWORD_False = 192
  KEYWORD_None = 193
  KEYWORD_True = 194
  KEYWORD_and = 195
  KEYWORD_as = 196
  KEYWORD_assert = 197
  KEYWORD_async = 198
  KEYWORD_await = 199
  KEYWORD_break = 200
  KEYWORD_case = 201
  KEYWORD_class = 202
  KEYWORD_continue = 203
  KEYWORD_def = 204
  KEYWORD_del = 205
  KEYWORD_elif = 206
  KEYWORD_else = 207
  KEYWORD_except = 208
  KEYWORD_finally = 209
  KEYWORD_for = 210
  KEYWORD_from = 211
  KEYWORD_global = 212
  KEYWORD_if = 213
  KEYWORD_import = 214
  KEYWORD_in = 215
  KEYWORD_is = 216
  KEYWORD_lambda = 217
  KEYWORD_match = 218
  KEYWORD_nonlocal = 219
  KEYWORD_not = 220
  KEYWORD_or = 221
  KEYWORD_pass = 222
  KEYWORD_raise = 223
  KEYWORD_return = 224
  KEYWORD_try = 225
  KEYWORD_while = 226
  KEYWORD_with = 227
  KEYWORD_yield = 228

  @staticmethod
  def as_str(kind: int) -> str: return TokenKind(kind).name if kind in TokenKind else f"'{chr(kind)}'"

kwlist = {kind.name.removeprefix("KEYWORD_"): kind for kind in TokenKind if kind.name.startswith("KEYWORD_")}
oplist: dict[str, int] = {
  "!": ord('!'),
  "~": ord('~'),
  "not": TokenKind.KEYWORD_not,
  "<": ord('<'),
  ">": ord('>'),
  "+": ord('+'),
  "-": ord('-'),
  "*": ord('*'),
  "@": ord('@'),
  "%": ord('%'),
  "/": ord('/'),
  "%": ord('%'),
  "&": ord('&'),
  "|": ord('|'),
  "^": ord('^'),
  "and": TokenKind.KEYWORD_and,
  "or": TokenKind.KEYWORD_or,
  "<<": TokenKind.LTLT,
  ">>": TokenKind.GTGT,
  "->": TokenKind.DASHGT,
  "**": TokenKind.STARSTAR,
  "//": TokenKind.SLASHSLASH,
  ":=": TokenKind.COLONEQ,
  "==": TokenKind.EQEQ,
  "!=": TokenKind.BANGEQ,
  "<=": TokenKind.LTEQ,
  ">=": TokenKind.GTEQ,
  "+=": TokenKind.PLUSEQ,
  "-=": TokenKind.DASHEQ,
  "*=": TokenKind.STAREQ,
  "@=": TokenKind.ATEQ,
  "/=": TokenKind.SLASHEQ,
  "%=": TokenKind.PERCENTEQ,
  "&=": TokenKind.AMPEQ,
  "|=": TokenKind.PIPEEQ,
  "^=": TokenKind.CARETEQ,
  "...": TokenKind.DOTDOTDOT,
  "<<=": TokenKind.LTLTEQ,
  ">>=": TokenKind.GTGTEQ,
  "**=": TokenKind.STARSTAREQ,
  "//=": TokenKind.SLASHSLASHEQ,
}
opstr = {v: k for k,v in oplist.items()}

@dataclass
class Token:
  kind: int
  location: int
  length: int

  def as_str(self, s: str) -> str: return s[self.location:self.location + self.length]

class TokenizerError(Exception):
  def __init__(self, message: str, token: Token) -> None:
    super().__init__(message, token)
    self.message = message
    self.token = token

def isbasedigit(c: str, base: int) -> bool:
  if base == 2: return '0' <= c <= '1'
  if base == 8: return '0' <= c <= '7'
  if base == 10: return '0' <= c <= '9'
  if base == 16: return '0' <= c <= '9' or 'a' <= c.lower() <= 'f'
  raise NotImplementedError(base)

def token_at(s: str, p: int) -> Token:
  start = p
  newline_was_skipped = False
  beginning_of_line = start
  while True:
    while p < len(s) and s[p].isspace():
      if s[p] == '\n': newline_was_skipped = True; beginning_of_line = p + 1
      p += 1
    if p < len(s) and s[p] == '#':
      while p < len(s) and s[p] != '\n': p += 1
      continue
    break
  if p >= len(s): return Token(TokenKind.END_OF_INPUT, p, 0)
  pb = p
  while pb > 1 and s[pb - 1].isspace() and s[pb - 1] != '\n': pb -= 1
  if newline_was_skipped and start != 0:
    return Token(TokenKind.NEWLINE, beginning_of_line - 1, 1)
  if s[beginning_of_line].isspace() and (pb - 1 == start or s[pb - 1] == '\n'):
    return Token(TokenKind.INDENT, beginning_of_line, p - beginning_of_line)
  start = p
  if s[p].isalpha() or s[p] == '_':
    while p < len(s) and (s[p].isalnum() or s[p] == '_'): p += 1
    if s[start:p] in kwlist: return Token(kwlist[s[start:p]], start, p - start)
    return Token(TokenKind.IDENTIFIER, start, p - start)
  if s[p].isdigit():
    base = 10
    is_float = False
    if p + 1 < len(s) and s[p] == '0' and s[p + 1] in "box":
      p += 1
      if s[p] == 'b': base = 2
      if s[p] == 'o': base = 8
      if s[p] == 'x': base = 16
      p += 1
      if p >= len(s) or not isbasedigit(s[p], base): raise TokenizerError(f"Integer literal contains invalid input.", Token(TokenKind.INTEGER, start, p - start))
    while p < len(s) and isbasedigit(s[p], base): p += 1
    if p < len(s) and s[p] == '.':
      p += 1
      is_float = True
      if base != 10: raise TokenizerError(f"Float literal can not have a non-ten base.", Token(TokenKind.FLOAT, start, p - start))
      while p < len(s) and s[p].isdigit(): p += 1
    if p < len(s) and s[p].lower() == 'e':
      p += 1
      if p < len(s) and s[p] in "+-": p += 1
      if p >= len(s) or not s[p].isdigit(): raise TokenizerError(f"Float literal must contain a digit after exponent marker.", Token(TokenKind.FLOAT, start, p - start))
      is_float = True
      if base != 10: raise TokenizerError(f"Float literal can not have a non-ten base.", Token(TokenKind.FLOAT, start, p - start))
      while p < len(s) and s[p].isdigit(): p += 1
    return Token(TokenKind.FLOAT if is_float else TokenKind.INTEGER, start, p - start)
  if s[p] == '"':
    p += 1
    while p < len(s) and (s[p - 1] == '\\' or s[p] != '"'): p += 1
    if p >= len(s) or s[p] != '"': raise TokenizerError(f"You did not terminate your string literal starting with `{s[start:start + min(p - start, 20)]}`.", Token(TokenKind.STRING, start, p - start))
    p += 1
    return Token(TokenKind.STRING, start, p - start)
  if p + 2 < len(s) and s[p:p + 3] in oplist: return Token(oplist[s[p:p + 3]], start, 3)
  if p + 1 < len(s) and s[p:p + 2] in oplist: return Token(oplist[s[p:p + 2]], start, 2)
  if s[p] in "+-*@/%&|~^:=.,;()[]{}<>": return Token(ord(s[p]), start, 1)
  raise TokenizerError(f"I do not know of a token that starts with '{s[p]}'.", Token(TokenKind.ERROR, start, 1))

def offset_to_line_col(s: str, p: int) -> tuple[int, int]:
  line, col, i = 1, 1, 0
  while i < min(p, len(s)):
    if s[i] == '\n': line += 1; col = 0
    i += 1; col += 1
  return line, col

def print_all_tokens(s: str, p: int = 0) -> None:
  while True:
    token = token_at(s, p)
    p = token.location + token.length
    print(TokenKind.as_str(token.kind), '"' + token.as_str(s).encode("unicode_escape").decode() + '"')
    if token.kind == TokenKind.END_OF_INPUT: break

@dataclass
class Code:
  start_location: int
  end_location: int
@dataclass
class Code_Module(Code):
  name: str
  body: list[Code]
@dataclass
class Code_Literal(Code):
  class EnumLiteral(str): pass
  value: EnumLiteral | bool | int | float | str | None
  is_thick: bool = False
@dataclass
class Code_Variable(Code):
  name: str
  raised: bool = False
  disable_implicit_call: bool = False
@dataclass
class Code_Declaration(Code):
  names: list[str]
  type_expr: Code | None
  exprs: list[Code]
@dataclass
class Code_Procedure(Code):
  name: str
  constants: list[Code_Variable]
  params: list[Code_Declaration]
  brackets: bool
  return_type: Code | None
  body: list[Code]
  is_macro: bool
@dataclass
class Code_UnaryOp(Code):
  op: int
  right: Code
@dataclass
class Code_BinaryOp(Code):
  left: Code
  op: int
  right: Code
@dataclass
class Code_Compare(Code):
  left: Code
  ops: list[int]
  conds: list[Code]
@dataclass
class Code_IfExpression(Code):
  conseq: Code
  cond: Code
  alt: Code
@dataclass
class Code_Call(Code):
  expr: Code
  args: list[Code]
@dataclass
class Code_Slice(Code):
  expr: Code
  start: Code | None
  stop: Code | None
@dataclass
class Code_SubscriptOrCall(Code):
  expr: Code
  exprs: list[Code]
@dataclass
class Code_ImportFrom(Code):
  module_name: str
  names: list[str]
  aliases: list[str | None]
  excludes: list[str]
@dataclass
class Code_Return(Code):
  exprs: list[Code]
@dataclass
class Code_Assert(Code):
  cond: Code
  message: Code | None
@dataclass
class Code_With(Code):
  names: list[str]
  exprs: list[Code]
  block_name: str | None
  body: list[Code]
@dataclass
class Code_Pass(Code): pass

class Parser:
  class Error(Exception):
    def __init__(self, message: str, token: Token) -> None:
      super().__init__(message, token)
      self.message = message
      self.token = token

  COMPARES: list[int] = [TokenKind.EQEQ, TokenKind.BANGEQ, TokenKind.LTEQ, TokenKind.GTEQ, ord('<'), ord('>'), TokenKind.KEYWORD_in]
  PRECEDENCES: dict[int, int] = {
    TokenKind.KEYWORD_or: 0,
    TokenKind.KEYWORD_and: 1,
    ord('|'): 3,
    ord('^'): 4,
    ord('&'): 5,
    TokenKind.LTLT: 6, TokenKind.GTGT: 6,
    ord('+'): 7, ord('-'): 7,
    ord('*'): 8, ord('@'): 8, ord('/'): 8, TokenKind.SLASHSLASH: 8, ord('%'): 8,
  }

  def __init__(self, s: str, p: int = 0) -> None:
    self.s = s
    self.p = p
    self.indents = [0]

  def peek(self, n: int = 1) -> Token:
    assert n > 0
    token: Token | None = None
    p = self.p
    for _ in range(n):
      token = token_at(self.s, p)
      p = token.location + token.length
    assert token is not None
    return token

  def eat(self, expect: int) -> Token:
    token = token_at(self.s, self.p)
    self.p = token.location + token.length
    if expect != token.kind: raise Parser.Error(f"I expected {TokenKind.as_str(expect)} but saw {TokenKind.as_str(token.kind)}!", token)
    return token

  def parse_leaf(self) -> Code:
    result: Code | None = None
    if self.peek().kind == TokenKind.IDENTIFIER:
      name = self.eat(TokenKind.IDENTIFIER)
      result = Code_Variable(name.location, name.length, name.as_str(self.s))
    elif self.peek().kind == TokenKind.INTEGER:
      token = self.eat(TokenKind.INTEGER)
      value = int(token.as_str(self.s), base=0)
      result = Code_Literal(token.location, token.length, value)
    elif self.peek().kind == TokenKind.FLOAT:
      token = self.eat(TokenKind.FLOAT)
      value = float(token.as_str(self.s))
      result = Code_Literal(token.location, token.length, value)
    elif self.peek().kind == TokenKind.STRING:
      token = self.eat(TokenKind.STRING)
      is_thick = token.as_str(self.s).startswith('"""')
      result = Code_Literal(token.location, token.length, token.as_str(self.s)[3 if is_thick else 1:-3 if is_thick else -1], is_thick=is_thick)
    elif self.peek().kind == ord('.'):
      token = self.eat(ord('.'))
      name = self.eat(TokenKind.IDENTIFIER)
      result = Code_Literal(token.location, self.p, Code_Literal.EnumLiteral(name.as_str(self.s)))
    elif self.peek().kind == ord('('):
      self.eat(ord('('))
      result = self.parse_expression()
      self.eat(ord(')'))
    elif self.peek().kind == TokenKind.KEYWORD_nonlocal:
      token = self.eat(TokenKind.KEYWORD_nonlocal)
      name = self.eat(TokenKind.IDENTIFIER)
      result = Code_Variable(token.location, self.p, name.as_str(self.s), disable_implicit_call=True)
    elif self.peek().kind == TokenKind.KEYWORD_raise:
      token = self.eat(TokenKind.KEYWORD_raise)
      name = self.eat(TokenKind.IDENTIFIER)
      result = Code_Variable(token.location, self.p, name.as_str(self.s), raised=True, disable_implicit_call=True)
    elif self.peek().kind in [ord('-'), ord('~'), TokenKind.KEYWORD_not]:
      token = self.eat(self.peek().kind)
      right = self.parse_leaf()
      result = Code_UnaryOp(token.location, self.p, token.kind, right)
    if result is None: raise Parser.Error(f"I do not know of an expression that starts with {TokenKind.as_str(self.peek().kind)}.", self.peek())
    while True:
      if self.peek().kind == ord('('):
        token = self.eat(ord('('))
        args: list[Code] = []
        while self.peek().kind != ord(')'):
          if self.peek(2).kind == ord('='):
            args.append(self.parse_Declaration())
          else:
            args.append(self.parse_expression())
          if self.peek().kind == ord(','): self.eat(ord(','))
          else: break
        self.eat(ord(')'))
        result = Code_Call(result.start_location, self.p, result, args)
        continue
      elif self.peek().kind == ord('['):
        self.eat(ord('['))
        expr_a: Code | None = None
        expr_b: Code | None = None
        colon: bool = False
        exprs: list[Code] = []
        if self.peek().kind != ord(']'):
          if self.peek().kind != ord(':'):
            expr_a = self.parse_expression()
          if self.peek().kind == ord(':'):
            colon = True
            self.eat(ord(':'))
            if self.peek().kind != ord(']'):
              expr_b = self.parse_expression()
          elif expr_a:
            exprs.append(expr_a); expr_a = None
            if self.peek().kind == ord(','):
              self.eat(ord(','))
              while self.peek().kind != ord(']'):
                exprs.append(self.parse_expression())
                if self.peek().kind == ord(','): self.eat(ord(','))
                else: break
          self.eat(ord(']'))
        if colon:
          assert len(exprs) == 0
          result = Code_Slice(result.start_location, self.p, result, expr_a, expr_b)
        else:
          assert expr_a is None and expr_b is None
          result = Code_SubscriptOrCall(result.start_location, self.p, result, exprs)
        continue
      break
    return result

  def parse_expression(self, min_prec: int = -1) -> Code:
    start = self.p
    left = self.parse_leaf()
    while self.peek().kind in Parser.PRECEDENCES and Parser.PRECEDENCES[self.peek().kind] >= min_prec:
      op = self.eat(self.peek().kind)
      right = self.parse_expression(min_prec)
      left = Code_BinaryOp(start, self.p, left, op.kind, right)
    if self.peek().kind in Parser.COMPARES:
      ops: list[int] = []
      conds: list[Code] = []
      while self.peek().kind in Parser.COMPARES:
        ops.append(self.eat(self.peek().kind).kind)
        conds.append(self.parse_leaf())
      left = Code_Compare(left.start_location, self.p, left, ops, conds)
    if min_prec == -1:
      while True:
        if self.peek().kind == TokenKind.KEYWORD_if:
          self.eat(TokenKind.KEYWORD_if)
          cond = self.parse_expression()
          self.eat(TokenKind.KEYWORD_else)
          alt = self.parse_expression()
          left = Code_IfExpression(start, self.p, left, cond, alt)
          continue
        break
    return left

  def parse_inline_statement_or_expression(self) -> Code:
    if self.peek().kind == TokenKind.IDENTIFIER and self.peek(2).kind in [ord(','), ord(':'), ord('=')]:
      return self.parse_Declaration()
    elif self.peek().kind == TokenKind.KEYWORD_return:
      token = self.eat(TokenKind.KEYWORD_return)
      exprs: list[Code] = []
      while self.peek().kind not in [TokenKind.NEWLINE, TokenKind.END_OF_INPUT]:
        exprs.append(self.parse_expression())
        if self.peek().kind == ord(','): self.eat(ord(','))
        else: break
      return Code_Return(token.location, self.p, exprs)
    elif self.peek().kind == TokenKind.KEYWORD_assert:
      token = self.eat(TokenKind.KEYWORD_assert)
      cond = self.parse_expression()
      message: Code | None = None
      if self.peek().kind == ord(','):
        self.eat(ord(','))
        message = self.parse_expression()
      return Code_Assert(token.location, self.p, cond, message)
    elif self.peek().kind == TokenKind.KEYWORD_pass:
      token = self.eat(TokenKind.KEYWORD_pass)
      return Code_Pass(token.location, token.length)
    elif self.peek().kind == TokenKind.KEYWORD_from:
      token = self.eat(TokenKind.KEYWORD_from)
      module_name = self.eat(TokenKind.IDENTIFIER).as_str(self.s)
      self.eat(TokenKind.KEYWORD_import)
      names: list[str] = []
      aliases: list[str | None] = []
      excludes: list[str] = []
      if self.peek().kind == ord('*'):
        self.eat(ord('*'))
        if self.peek().kind == TokenKind.KEYWORD_except:
          self.eat(TokenKind.KEYWORD_except)
          while True:
            excludes.append(self.eat(TokenKind.IDENTIFIER).as_str(self.s))
            if self.peek().kind == ord(','): self.eat(ord(','))
            else: break
      else:
        while True:
          names.append(self.eat(TokenKind.IDENTIFIER).as_str(self.s))
          if self.peek().kind == TokenKind.KEYWORD_as:
            self.eat(TokenKind.KEYWORD_as)
            aliases.append(self.eat(TokenKind.IDENTIFIER).as_str(self.s))
          else:
            aliases.append(None)
          if self.peek().kind == ord(','): self.eat(ord(','))
          else: break
      return Code_ImportFrom(token.location, self.p, module_name, names, aliases, excludes)
    return self.parse_expression()

  def parse_line(self) -> list[Code]:
    line: list[Code] = []
    while self.peek().kind not in [TokenKind.NEWLINE, TokenKind.END_OF_INPUT]:
      line.append(self.parse_inline_statement_or_expression())
      if self.peek().kind == ord(';'): self.eat(ord(';'))
      else: break
    if self.peek().kind != TokenKind.END_OF_INPUT: self.eat(TokenKind.NEWLINE)
    return line

  def try_parse_block_statement(self) -> Code | None:
    if self.peek().kind == TokenKind.KEYWORD_def:
      return self.parse_Procedure()
    elif self.peek().kind == TokenKind.KEYWORD_with:
      token = self.eat(TokenKind.KEYWORD_with)
      names: list[str] = []
      exprs: list[Code] = []
      block_name: str | None = None
      while self.peek().kind == TokenKind.IDENTIFIER and self.peek(2).kind == ord('='):
        name = self.eat(TokenKind.IDENTIFIER)
        self.eat(ord('='))
        names.append(name.as_str(self.s))
        exprs.append(self.parse_expression())
        if self.peek().kind == ord(','): self.eat(ord(','))
        else: break
      if self.peek().kind != ord(':'):
        block_name = self.eat(TokenKind.IDENTIFIER).as_str(self.s)
      body = self.parse_block()
      return Code_With(token.location, self.p, names, exprs, block_name, body)
    return None

  def try_parse_block_statement_or_line(self) -> list[Code]:
    if c := self.try_parse_block_statement(): return [c]
    else: return self.parse_line()

  def parse_block(self) -> list[Code]:
    op = self.eat(ord(':'))
    body: list[Code] = []
    if self.peek().kind == TokenKind.NEWLINE:
      self.eat(TokenKind.NEWLINE)
      indent_token = self.peek()
      if indent_token.kind != TokenKind.INDENT: raise Parser.Error(f"A block expects an indent after a newline, got {TokenKind.as_str(indent_token.kind)}.", indent_token)
      first_indent = len(indent_token.as_str(self.s))
      self.indents.append(first_indent)
      while True:
        indent_token = self.peek()
        if indent_token.kind != TokenKind.INDENT: break
        indent = len(indent_token.as_str(self.s))
        if indent != first_indent:
          if indent > first_indent: raise Parser.Error(f"I expected an indent of length {first_indent} but got length {indent}.", self.peek())
          self.indents.pop()
          if indent not in self.indents: raise Parser.Error(f"Indentation of length {indent} did not match any prior block.", self.peek())
          break
        self.eat(TokenKind.INDENT)
        body += self.try_parse_block_statement_or_line()
    else: body += self.parse_line()
    if len(body) == 0: raise Parser.Error("A block must have at least one expression or statement.", op)
    return body

  def parse_Procedure(self) -> Code_Procedure:
    is_macro = False # TODO
    token = self.eat(TokenKind.KEYWORD_def)
    name = self.eat(TokenKind.IDENTIFIER)
    params: list[Code] = []
    brackets = self.peek().kind == ord('[')
    if brackets:
      self.eat(ord('['))
      decls = self.peek(2).kind == ord(':')
      while self.peek().kind != ord(']'):
        params.append(self.parse_Declaration() if decls else self.parse_expression())
        if self.peek().kind == ord(','): self.eat(ord(','))
        else: break
      self.eat(ord(']'))
    constants: list[Code] = []
    if self.peek().kind in [ord('['), ord('(')]:
      constants = params.copy(); params = []
      brackets = self.eat(self.peek().kind).kind == ord('[')
      while self.peek().kind != (ord(']') if brackets else ord(')')):
        params.append(self.parse_Declaration())
        if self.peek().kind == ord(','): self.eat(ord(','))
        else: break
      self.eat(ord(']') if brackets else ord(')'))
    return_type: Code | None = None
    if self.peek().kind == TokenKind.DASHGT:
      self.eat(TokenKind.DASHGT)
      return_type = self.parse_expression()
    body = self.parse_block()
    for constant in constants:
      if not isinstance(constant, Code_Variable): raise Parser.Error("", token)
    for param in params:
      if not isinstance(param, Code_Declaration): raise Parser.Error("", token)
    return Code_Procedure(token.location, self.p, name.as_str(self.s), typing.cast(list[Code_Variable], constants), typing.cast(list[Code_Declaration], params), brackets, return_type, body, is_macro)

  def parse_Declaration(self) -> Code_Declaration:
    start = self.p
    first_name = self.peek()
    names: list[str] = []
    while True:
      name = self.eat(TokenKind.IDENTIFIER)
      names.append(name.as_str(self.s))
      if self.peek().kind == ord(','): self.eat(ord(','))
      else: break
    type_expr: Code | None = None
    if self.peek().kind == ord(':'):
      self.eat(ord(':'))
      type_expr = self.parse_expression()
    exprs: list[Code] = []
    if self.peek().kind == ord('='):
      self.eat(ord('='))
      while True:
        exprs.append(self.parse_expression())
        if self.peek().kind == ord(','): self.eat(ord(','))
        else: break
    if type_expr is None and len(exprs) == 0: raise Parser.Error("A declaration (e.g. x: y = z) expects a type (y) and/or value (z).", first_name)
    return Code_Declaration(start, self.p, names, type_expr, exprs)

  def parse_Module(self, name: str) -> Code_Module:
    start = self.p
    body: list[Code] = []
    while self.peek().kind != TokenKind.END_OF_INPUT:
      body += self.try_parse_block_statement_or_line()
    return Code_Module(start, self.p, name, body)

def code_as_string(code: Code, level: int = 0) -> str:
  if isinstance(code, Code_Module):
    return "\n".join(code_as_string(child, level) for child in code.body)
  elif isinstance(code, Code_Literal):
    if isinstance(code.value, Code_Literal.EnumLiteral): return f".{code.value}"
    elif isinstance(code.value, bool | int | float | None): return str(code.value)
    else: return f"{'"""' if code.is_thick else '"'}{code.value}{'"""' if code.is_thick else '"'}"
  elif isinstance(code, Code_Variable):
    return f"{"nonlocal " if not code.raised and code.disable_implicit_call else ""}{"raise " if code.raised else ""}{code.name}"
  elif isinstance(code, Code_Declaration):
    result = f"{", ".join(code.names)}"
    if code.type_expr: result += f": {code_as_string(code.type_expr, level)}"
    if len(code.exprs): result += f" = {", ".join(code_as_string(expr, level) for expr in code.exprs)}"
    return result
  elif isinstance(code, Code_Procedure):
    return f"def {code.name}{f"[{", ".join(code_as_string(constant, level) for constant in code.constants)}]" if len(code.constants) else ""}{"[" if code.brackets else "("}{", ".join(code_as_string(param, level) for param in code.params)}{"]" if code.brackets else ")"}{f" -> {code_as_string(code.return_type, level)}" if code.return_type else ""}:\n{"\n".join("  " * (level + 1) + code_as_string(child, level + 1) for child in code.body)}"
  elif isinstance(code, Code_UnaryOp):
    wrap = isinstance(code.right, Code_BinaryOp | Code_IfExpression)
    return f"{opstr[code.op]}{" " if code.op == TokenKind.KEYWORD_not else ""}{"(" if wrap else ""}{code_as_string(code.right, level)}{")" if wrap else ""}"
  elif isinstance(code, Code_BinaryOp):
    left_wrap = isinstance(code.left, Code_BinaryOp) and Parser.PRECEDENCES[code.left.op] < Parser.PRECEDENCES[code.op]
    right_wrap = isinstance(code.right, Code_BinaryOp) and Parser.PRECEDENCES[code.right.op] < Parser.PRECEDENCES[code.op]
    result = f"{"(" if left_wrap else ""}{code_as_string(code.left, level)}{")" if left_wrap else ""}"
    result += f" {opstr[code.op]} "
    result += f"{"(" if right_wrap else ""}{code_as_string(code.right, level)}{")" if right_wrap else ""}"
    return result
  elif isinstance(code, Code_Compare):
    result = f"{code_as_string(code.left, level)}"
    for op, cond in zip(code.ops, code.conds):
      result += f" {opstr[op]} {code_as_string(cond, level)}"
    return result
  elif isinstance(code, Code_IfExpression):
    cond_wrap = isinstance(code.cond, Code_IfExpression)
    conseq_wrap = isinstance(code.conseq, Code_IfExpression)
    return f"{"(" if conseq_wrap else ""}{code_as_string(code.conseq, level)}{")" if conseq_wrap else ""} if {"(" if cond_wrap else ""}{code_as_string(code.cond, level)}{")" if cond_wrap else ""} else {code_as_string(code.alt, level)}"
  elif isinstance(code, Code_Call):
    return f"{code_as_string(code.expr, level)}({", ".join(code_as_string(arg, level) for arg in code.args)})"
  elif isinstance(code, Code_SubscriptOrCall):
    return f"{code_as_string(code.expr, level)}[{", ".join(code_as_string(expr, level) for expr in code.exprs)}]"
  elif isinstance(code, Code_ImportFrom):
    return f"from {code.module_name} import {"*" if len(code.names) == 0 else ", ".join(f"{name}{f" as {alias}" if alias else ""}" for name, alias in zip(code.names, code.aliases))}{f" except {", ".join(code.excludes)}" if len(code.excludes) else ""}"
  elif isinstance(code, Code_Assert):
    return f"assert {code_as_string(code.cond, level)}{f", {code_as_string(code.message, level)}" if code.message else ""}"
  elif isinstance(code, Code_With):
    return f"with{f" {", ".join(f"{name} = {code_as_string(expr, level)}" for name, expr in zip(code.names, code.exprs))}" if len(code.exprs) else ""}{f" {"," if len(code.exprs) > 0 else ""}{code.block_name}" if code.block_name else ""}:\n{"\n".join("  " * (level + 1) + code_as_string(child, level + 1) for child in code.body)}"
  elif isinstance(code, Code_Return):
    return f"return {", ".join(code_as_string(expr, level) for expr in code.exprs)}"
  elif isinstance(code, Code_Pass):
    return "pass"
  else:
    raise NotImplementedError(code.__class__.__name__)

@dataclass(frozen=True)
class Type: pass
@dataclass(frozen=True)
class VariableType(Type): name: str
@dataclass(frozen=True)
class NamedType(Type): name: str
@dataclass(frozen=True)
class ProcedureType(Type):
  parameter_types: list[Type]
  return_type: Type
  varargs_type: Type | None = None
  is_macro: bool = False

type_type = NamedType("type")
type_code = NamedType("code")
type_none = NamedType("none")
type_enum_literal = NamedType("enum_literal")
type_noreturn = NamedType("noreturn")
type_void = NamedType("void")
type_bool = NamedType("bool")
type_int = NamedType("int")
type_float = NamedType("float")
type_str = NamedType("str")
type_list = NamedType("list")
type_dict = NamedType("dict")
type_tuple = NamedType("tuple")
type_procedure_set = NamedType("procedure_set")

@dataclass
class Value:
  class Procedure:
    def __init__(self, name: str, body: typing.Callable[..., "Value"] | list[Code], constant_names: list[str], parameter_names: list[str], defaults: dict[str, "Value"], brackets: bool) -> None:
      self.name = name
      self.body = body
      self.constant_names = constant_names
      self.parameter_names = parameter_names
      self.defaults = defaults
      self.brackets = brackets

    def __call__(self, *args: "Value", **kwargs: typing.Any) -> "Value":
      if callable(self.body): return self.body(*args, **kwargs)
      else:
        procedure_scope = Scope(kwargs["evaluator"].global_scope)
        for i, arg in enumerate(args):
          procedure_scope.entries.update({self.parameter_names[i]: arg})
        for child in self.body:
          result = kwargs["evaluator"](child, procedure_scope)
          if hasattr(result, "$return"): delattr(result, "$return"); return result
        if kwargs["ty"].return_type != type_void:
          raise Evaluator.Error(f"Procedure did not return a value!", kwargs["op_code"])
        return value_void

  class ProcedureSet:
    def __init__(self, procedures: list["Value"]) -> None:
      self.procedures = procedures

  ty: Type
  contents: Type | Code | tuple["Value", ...] | Procedure | ProcedureSet | bool | int | float | str | None

  @property
  def as_type(self) -> Type:
    if self.ty != type_type or not isinstance(self.contents, Type): raise TypeError()
    return self.contents
  @property
  def as_code(self) -> Code:
    if self.ty != type_code or not isinstance(self.contents, Code): raise TypeError()
    return self.contents
  @property
  def as_tuple(self) -> tuple["Value", ...]:
    if self.ty != type_tuple or not isinstance(self.contents, tuple): raise TypeError()
    return self.contents
  @property
  def as_procedure_set(self) -> ProcedureSet:
    if self.ty != type_procedure_set or not isinstance(self.contents, Value.ProcedureSet): raise TypeError()
    return self.contents
  @property
  def as_procedure(self) -> Procedure:
    if not isinstance(self.ty, ProcedureType) or not isinstance(self.contents, Value.Procedure): raise TypeError()
    return self.contents
  @property
  def as_bool(self) -> bool:
    if self.ty != type_bool or not isinstance(self.contents, bool): raise TypeError()
    return self.contents
  @property
  def as_int(self) -> int:
    if self.ty != type_int or not isinstance(self.contents, int): raise TypeError()
    return self.contents
  @property
  def as_float(self) -> float:
    if self.ty != type_float or not isinstance(self.contents, float): raise TypeError()
    return self.contents
  @property
  def as_str(self) -> str:
    if self.ty != type_str or not isinstance(self.contents, str): raise TypeError()
    return self.contents
  @property
  def as_enum_literal(self) -> str:
    if self.ty != type_enum_literal or not isinstance(self.contents, str): raise TypeError()
    return self.contents

value_void = Value(type_void, None)
value_none = Value(type_none, None)
value_true = Value(type_bool, True)
value_false = Value(type_bool, False)

class Scope:
  def __init__(self, parent: "Scope | None") -> None:
    self.parent = parent
    self.entries: dict[str, Value] = {}

  def find(self, key: str) -> Value | None:
    if key in self.entries: return self.entries[key]
    if self.parent is None: return None
    return self.parent.find(key)

def compiler_type(kind_value: Value, **kwargs: typing.Any) -> Value:
  kind = kind_value.as_enum_literal
  if kind == "type": return Value(type_type, type_type)
  if kind == "code": return Value(type_type, type_code)
  if kind == "none": return Value(type_type, type_none)
  if kind == "enum_literal": return Value(type_type, type_enum_literal)
  if kind == "noreturn": return Value(type_type, type_noreturn)
  if kind == "void": return Value(type_type, type_void)
  if kind == "bool": return Value(type_type, type_bool)
  if kind == "int": return Value(type_type, type_int)
  if kind == "float": return Value(type_type, type_float)
  if kind == "str": return Value(type_type, type_str)
  if kind == "list": return Value(type_type, type_list)
  if kind == "dict": return Value(type_type, type_dict)
  if kind == "tuple": return Value(type_type, type_tuple)
  if kind == "procedure": raise NotImplementedError()
  if kind == "procedure_set": return Value(type_type, type_procedure_set)
  raise Evaluator.Error(f"I could not find a type that goes by the name {kind}.", kwargs["arg_codes"][0])

def compiler_type_of(value: Value, **kwargs: typing.Any) -> Value:
  return Value(type_type, value.ty)

compiler_scope = Scope(None)
compiler_scope.entries.update({
  "type": Value(type_procedure_set, Value.ProcedureSet([
    Value(ProcedureType([type_enum_literal], type_procedure_set), Value.Procedure("type", compiler_type, [], ["kind"], {"kind": Value(type_enum_literal, "type")}, True)),
    Value(ProcedureType([VariableType("T")], type_type), Value.Procedure("type", compiler_type_of, ["T"], ["value"], {}, False))
  ])),
})

class Evaluator:
  class Error(Exception):
    def __init__(self, message: str, code: Code) -> None:
      super().__init__(message, code)
      self.message = message
      self.code = code

  def __init__(self, s: str, global_scope: Scope | None = None) -> None:
    self.s = s
    self.global_scope = global_scope if global_scope else Scope(compiler_scope)

  def type_as_string(self, ty: Type) -> str:
    if isinstance(ty, NamedType): return f"type[.{ty.name}]"
    raise NotImplementedError(ty.__class__.__name__)

  def coerce(self, value: Value, ty: Type) -> Value | None:
    if value.ty == ty or isinstance(ty, VariableType): return value
    return None

  def call(self, scope: Scope, expr: Value, op_code: Code, arg_codes: list[Code], explicit_brackets: bool = False, explicit_parentheses: bool = False) -> Value:
    try: proc_set = expr.as_procedure_set
    except TypeError: raise Evaluator.Error("Attempted to call non-procedure.", op_code)
    def sort(proc: Value):
      if explicit_brackets and proc.as_procedure.brackets: return -1
      if explicit_parentheses and not proc.as_procedure.brackets: return -1
      return 1
    choices = sorted(proc_set.procedures, key=sort)
    proc_value = choices[0]
    proc = proc_value.as_procedure
    if explicit_brackets and not proc.brackets: raise Evaluator.Error(f"No overload of {proc.name}[].", op_code)
    if explicit_parentheses and proc.brackets: raise Evaluator.Error(f"No overload of {proc.name}().", op_code)
    assert isinstance(proc_value.ty, ProcedureType)
    posargs: list[Value] = [value_void] * len(proc_value.ty.parameter_types)
    namedargs: dict[str, Value] = {}
    posargi = 0
    for arg in arg_codes:
      if isinstance(arg, Code_Declaration):
        assert len(arg.names) == 1
        assert not arg.type_expr
        assert len(arg.exprs) == 1
        name = arg.names[0]
        arg_value = self(arg.exprs[0], scope)
        if name in proc.parameter_names:
          index = proc.parameter_names.index(name)
          arg_value = self.coerce(arg_value, proc_value.ty.parameter_types[index])
          if arg_value is None:
            raise Evaluator.Error(f"type mismatch", arg_codes[index])
          posargs[index] = arg_value
          posargi = index + 1
        else:
          namedargs[name] = arg_value
      else:
        arg_value = self.coerce(self(arg, scope), proc_value.ty.parameter_types[posargi])
        if arg_value is None:
          raise Evaluator.Error(f"type mismatch", arg_codes[posargi])
        posargs[posargi] = arg_value
        posargi += 1
    while posargi < len(proc.parameter_names) and proc.parameter_names[posargi] in proc.defaults:
      arg_value = self.coerce(proc.defaults[proc.parameter_names[posargi]], proc_value.ty.parameter_types[posargi])
      if arg_value is None:
        raise Evaluator.Error(f"type mismatch", arg_codes[posargi])
      posargs[posargi] = arg_value
      posargi += 1
    if posargi < len(proc.parameter_names): raise Evaluator.Error("not enough arguments provided", op_code)
    return proc(*posargs, namedargs=namedargs, ty=proc_value.ty, evaluator=self, op_code=op_code, arg_codes=arg_codes)

  def __call__(self, code: Code, scope: Scope, disable_implicit_call: bool = False) -> Value:
    if isinstance(code, Code_Module):
      scope.entries.update({"__name__": Value(type_str, code.name)})
      for child in code.body:
        self(child, scope, disable_implicit_call)
      return value_void
    elif isinstance(code, Code_Literal):
      if isinstance(code.value, bool): return value_true if code.value else value_false
      elif isinstance(code.value, int): return Value(type_int, code.value)
      elif isinstance(code.value, float): return Value(type_float, code.value)
      elif isinstance(code.value, Code_Literal.EnumLiteral): return Value(type_enum_literal, code.value)
      elif isinstance(code.value, str): return Value(type_str, code.value)
      elif code.value is None: return value_none
      raise NotImplementedError(type(code.value))
    elif isinstance(code, Code_Variable):
      value = scope.find(code.name)
      if value is None: raise Evaluator.Error("Not in scope!", code)
      if not code.disable_implicit_call and not disable_implicit_call and value.ty == type_procedure_set:
        value = self.call(scope, value, code, [])
      return value
    elif isinstance(code, Code_Declaration):
      if len(code.names) != 1: raise NotImplementedError()
      if code.type_expr: raise NotImplementedError()
      if len(code.exprs) != 1: raise NotImplementedError()
      scope.entries[code.names[0]] = self(code.exprs[0], scope, disable_implicit_call)
      return value_void
    elif isinstance(code, Code_Procedure):
      constant_names: list[str] = []
      parameter_names: list[str] = []
      parameter_types: list[Type] = []
      defaults: dict[str, Value] = {}
      for constant in code.constants:
        constant_names.append(constant.name)
      for param in code.params:
        assert len(param.names) == 1
        assert param.type_expr is not None
        assert len(param.exprs) == 0
        parameter_names.append(param.names[0])
        try: parameter_types.append(self(param.type_expr, scope, disable_implicit_call).as_type)
        except TypeError: raise Evaluator.Error(f"Parameter type expression is not a type!", param.type_expr)
      return_type = type_void
      if code.return_type:
        try: return_type = self(code.return_type, scope, disable_implicit_call).as_type
        except TypeError: raise Evaluator.Error(f"Return type expression is not a type!", code.return_type)
      _ = return_type
      proc = Value(ProcedureType(parameter_types, return_type, is_macro=code.is_macro), Value.Procedure(code.name, code.body, constant_names, parameter_names, defaults, code.brackets))
      value = scope.find(code.name)
      if value and value.ty == type_procedure_set:
        value.as_procedure_set.procedures.append(proc)
      else:
        scope.entries[code.name] = Value(type_procedure_set, Value.ProcedureSet([proc]))
      return value_void
    elif isinstance(code, Code_UnaryOp):
      right = self(code.right, scope, disable_implicit_call)
      if right.ty == type_int:
        if code.op == ord('-'): return Value(type_int, -right.as_int)
        if code.op == ord('~'): return Value(type_int, ~right.as_int)
      if right.ty == type_bool:
        if code.op == TokenKind.KEYWORD_not: return value_true if not right.as_bool else value_false
      raise Evaluator.Error(f"Unary operator {opstr[code.op]} is not defined for {self.type_as_string(right.ty)}!", code)
    elif isinstance(code, Code_BinaryOp):
      left = self(code.left, scope, disable_implicit_call)
      right = self(code.right, scope, disable_implicit_call)
      if left.ty == type_int and right.ty == type_int:
        if code.op == ord('+'): return Value(type_int, left.as_int + right.as_int)
        if code.op == ord('-'): return Value(type_int, left.as_int - right.as_int)
        if code.op == ord('*'): return Value(type_int, left.as_int * right.as_int)
        if code.op == TokenKind.SLASHSLASH: return Value(type_int, left.as_int // right.as_int)
        if code.op == ord('%'): return Value(type_int, left.as_int % right.as_int)
      raise Evaluator.Error(f"Binary operator {opstr[code.op]} is not defined between {self.type_as_string(left.ty)} and {self.type_as_string(right.ty)}!", code)
    elif isinstance(code, Code_Compare):
      result = True
      left = self(code.left, scope)
      for op, cond in zip(code.ops, code.conds):
        cond = self(cond, scope)
        if left.ty == type_int and cond.ty == type_int:
          if op == TokenKind.EQEQ: result = result and left.as_int == cond.as_int
          elif op == TokenKind.BANGEQ: result = result and left.as_int != cond.as_int
          elif op == TokenKind.LTEQ: result = result and left.as_int <= cond.as_int
          elif op == TokenKind.GTEQ: result = result and left.as_int >= cond.as_int
          elif op == ord('<'): result = result and left.as_int < cond.as_int
          elif op == ord('>'): result = result and left.as_int > cond.as_int
          else: raise Evaluator.Error(f"Comparison operator {opstr[op]} not defined between {self.type_as_string(left.ty)} and {self.type_as_string(cond.ty)}", code)
        elif left.ty == type_type and cond.ty == type_type:
          if op == TokenKind.EQEQ: result = result and left.as_type == cond.as_type
          elif op == TokenKind.BANGEQ: result = result and left.as_type != cond.as_type
          else: raise Evaluator.Error(f"Comparison operator {opstr[op]} not defined between {self.type_as_string(left.ty)} and {self.type_as_string(cond.ty)}", code)
        elif left.ty == type_str and cond.ty == type_str:
          if op == TokenKind.EQEQ: result = result and left.as_str == cond.as_str
          elif op == TokenKind.BANGEQ: result = result and left.as_str != cond.as_str
          else: raise Evaluator.Error(f"Comparison operator {opstr[op]} not defined between {self.type_as_string(left.ty)} and {self.type_as_string(cond.ty)}", code)
        else: raise Evaluator.Error(f"Comparion operator {opstr[op]} not defined between {self.type_as_string(left.ty)} and {self.type_as_string(cond.ty)}", code)
        left = cond
      return value_true if result else value_false
    elif isinstance(code, Code_Call):
      return self.call(scope, self(code.expr, scope, True), code.expr, code.args, explicit_parentheses=True)
    elif isinstance(code, Code_SubscriptOrCall):
      expr = self(code.expr, scope, True)
      if expr.ty == type_procedure_set: return self.call(scope, expr, code.expr, code.exprs, explicit_brackets=True)
      else:
        assert len(code.exprs) == 1
        # key = self(code.exprs[0], scope)
        raise NotImplementedError()
    elif isinstance(code, Code_IfExpression):
      # TODO: TYPECHECK!!1!
      return self(code.conseq, scope, disable_implicit_call) if self(code.cond, scope, disable_implicit_call).as_bool else self(code.alt, scope, disable_implicit_call)
    elif isinstance(code, Code_ImportFrom):
      src = Path(code.module_name).with_suffix(".ape").read_text()
      module = Parser(src).parse_Module(code.module_name)
      evaluator = Evaluator(src)
      evaluator(module, evaluator.global_scope)
      if len(code.names):
        for name, alias in zip(code.names, code.aliases):
          scope.entries[alias if alias else name] = evaluator.global_scope.entries[name]
      else:
        # TODO(dfra): *Should* this ignore duplicates like __name__? Obviously this won't work for procedure sets...
        scope.entries.update({k: v for k,v in evaluator.global_scope.entries.items() if k not in code.excludes and k not in scope.entries})
      return value_void
    elif isinstance(code, Code_Pass):
      return value_void
    elif isinstance(code, Code_Return):
      value = Value(type_tuple, tuple(self(expr, scope, disable_implicit_call) for expr in code.exprs)) if len(code.exprs) > 1 else (self(code.exprs[0], scope) if len(code.exprs) > 0 else value_void)
      setattr(value, "$return", True)
      return value
    elif isinstance(code, Code_Assert):
      try: cond = self(code.cond, scope, disable_implicit_call).as_bool
      except TypeError: raise Evaluator.Error("Assertion condition did not result in a boolean!", code.cond)
      if code.message: raise NotImplementedError()
      if not cond:
        raise Evaluator.Error("Assertion failed!", code.cond)
      return value_void
    elif isinstance(code, Code_With):
      with_scope = Scope(scope)
      for name, value in zip(code.names, code.exprs):
        with_scope.entries.update({name: self(value, with_scope, disable_implicit_call)})
      for child in code.body:
        self(child, with_scope, disable_implicit_call)
      return value_void
    else:
      raise NotImplementedError(code.__class__.__name__)

def dofile(path: Path, name: str | None = None) -> None:
  name = name if name else path.stem
  src = path.read_text()
  # print_all_tokens(src)
  parser = Parser(src)
  evaluator = Evaluator(src)
  try:
    module = parser.parse_Module(name)
    # print(code_as_string(module))
    evaluator(module, evaluator.global_scope)
  except TokenizerError as e:
    traceback.print_exc()
    line, col = offset_to_line_col(src, e.token.location)
    print(f"token error @ {path}:{line}:{col}: {e.message}")
  except Parser.Error as e:
    traceback.print_exc()
    line, col = offset_to_line_col(src, e.token.location)
    print(f"parse error @ {path}:{line}:{col} near '{e.token.as_str(src)}': {e.message}")
  except Evaluator.Error as e:
    traceback.print_exc()
    line, col = offset_to_line_col(src, e.code.start_location)
    print(f"evaluation error @ {path}:{line}:{col} near '{code_as_string(e.code)}': {e.message}")

def repl() -> None:
  src = ""
  repl_scope = Scope(compiler_scope)
  while True:
    pos = len(src)
    try: src += input("> ") + '\n'
    except (KeyboardInterrupt, EOFError): print(""); break
    # print_all_tokens(src, pos)
    parser = Parser(src, pos)
    evaluator = Evaluator(src, repl_scope)
    try:
      module = parser.parse_Module("__main__")
      # print(code_as_string(module))
      evaluator(module, evaluator.global_scope)
    except TokenizerError as e:
      line, col = offset_to_line_col(src, e.token.location)
      print(f"token error @ repl:{line}:{col}: {e.message}")
    except Parser.Error as e:
      line, col = offset_to_line_col(src, e.token.location)
      print(f"parse error @ repl:{line}:{col} near '{e.token.as_str(src)}': {e.message}")
    except Evaluator.Error as e:
      line, col = offset_to_line_col(src, e.code.start_location)
      print(f"evaluation error @ repl:{line}:{col} near '{code_as_string(e.code)}': {e.message}")

if __name__ == "__main__":
  if len(sys.argv) > 1: dofile(Path(sys.argv[1]), name="__main__")
  else: repl()
