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
class Code_BinaryOp(Code):
  left: Code
  op: int
  right: Code
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

  PRECEDENCES: dict[int, int] = {
    TokenKind.KEYWORD_or: 0,
    TokenKind.KEYWORD_and: 1,
    TokenKind.EQEQ: 2, TokenKind.BANGEQ: 2, TokenKind.LTEQ: 2, TokenKind.GTEQ: 2, ord('<'): 2, ord('>'): 2, TokenKind.KEYWORD_in: 2,
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
    elif self.peek().kind == TokenKind.STRING:
      token = self.eat(TokenKind.STRING)
      is_thick = token.as_str(self.s).startswith('"""')
      result = Code_Literal(token.location, token.length, token.as_str(self.s)[3 if is_thick else 1:-3 if is_thick else -1], is_thick=is_thick)
    elif self.peek().kind == ord('.'):
      token = self.eat(ord('.'))
      name = self.eat(TokenKind.IDENTIFIER)
      result = Code_Literal(token.location, self.p, Code_Literal.EnumLiteral(name.as_str(self.s)))
    if result is None: raise Parser.Error(f"I do not know of an expression that starts with {TokenKind.as_str(self.peek().kind)}.", self.peek())
    while True:
      if self.peek().kind == ord('['):
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
    return left

  def parse_inline_statement_or_expression(self) -> Code:
    if self.peek().kind == TokenKind.IDENTIFIER and self.peek(2).kind in [ord(','), ord(':'), ord('=')]:
      return self.parse_Declaration()
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
    if isinstance(code.value, bool | int | float | None): return str(code.value)
    else: return f"{'"""' if code.is_thick else '"'}{code.value}{'"""' if code.is_thick else '"'}"
  elif isinstance(code, Code_Variable):
    return f"{code.name}"
  elif isinstance(code, Code_Declaration):
    result = f"{", ".join(code.names)}"
    if code.type_expr: result += f": {code_as_string(code.type_expr, level)}"
    if len(code.exprs): result += f" = {", ".join(code_as_string(expr, level) for expr in code.exprs)}"
    return result
  elif isinstance(code, Code_Procedure):
    return f"def {code.name}{f"[{", ".join(code_as_string(constant, level) for constant in code.constants)}]" if len(code.constants) else ""}{"[" if code.brackets else "("}{", ".join(code_as_string(param, level) for param in code.params)}{"]" if code.brackets else ")"}{f" -> {code_as_string(code.return_type, level)}" if code.return_type else ""}:\n{"\n".join("  " * (level + 1) + code_as_string(child, level + 1) for child in code.body)}"
  elif isinstance(code, Code_BinaryOp):
    left_wrap = isinstance(code.left, Code_BinaryOp) and Parser.PRECEDENCES[code.left.op] < Parser.PRECEDENCES[code.op]
    right_wrap = isinstance(code.right, Code_BinaryOp) and Parser.PRECEDENCES[code.right.op] < Parser.PRECEDENCES[code.op]
    result = f"{"(" if left_wrap else ""}{code_as_string(code.left, level)}{")" if left_wrap else ""}"
    result += f" {opstr[code.op]} "
    result += f"{"(" if right_wrap else ""}{code_as_string(code.right, level)}{")" if right_wrap else ""}"
    return result
  elif isinstance(code, Code_SubscriptOrCall):
    return f"{code_as_string(code.expr, level)}[{", ".join(code_as_string(expr, level) for expr in code.exprs)}]"
  elif isinstance(code, Code_Assert):
    return f"assert {code_as_string(code.cond, level)}{f", {code_as_string(code.message, level)}" if code.message else ""}"
  elif isinstance(code, Code_With):
    return f"with{f" {", ".join(f"{name} = {code_as_string(expr, level)}" for name, expr in zip(code.names, code.exprs))}" if len(code.exprs) else ""}{f" {"," if len(code.exprs) > 0 else ""}{code.block_name}" if code.block_name else ""}:\n{"\n".join("  " * (level + 1) + code_as_string(child, level + 1) for child in code.body)}"
  elif isinstance(code, Code_Pass):
    return "pass"
  else:
    raise NotImplementedError(code.__class__.__name__)

def dofile(path: Path, name: str | None = None) -> None:
  name = name if name else path.stem
  src = path.read_text()
  # print_all_tokens(src)
  parser = Parser(src)
  try:
    module = parser.parse_Module(name)
    print(code_as_string(module))
  except TokenizerError as e:
    traceback.print_exc()
    line, col = offset_to_line_col(src, e.token.location)
    print(f"token error @ {path}:{line}:{col}: {e.message}")
  except Parser.Error as e:
    traceback.print_exc()
    line, col = offset_to_line_col(src, e.token.location)
    print(f"parse error @ {path}:{line}:{col} near '{e.token.as_str(src)}': {e.message}")

def repl() -> None:
  src = ""
  while True:
    pos = len(src)
    try: src += input("> ") + '\n'
    except (KeyboardInterrupt, EOFError): print(""); break
    # print_all_tokens(src, pos)
    parser = Parser(src, pos)
    try:
      module = parser.parse_Module("__main__")
      print(code_as_string(module))
    except TokenizerError as e:
      line, col = offset_to_line_col(src, e.token.location)
      print(f"token error @ repl:{line}:{col}: {e.message}")
    except Parser.Error as e:
      line, col = offset_to_line_col(src, e.token.location)
      print(f"parse error @ repl:{line}:{col} near '{e.token.as_str(src)}': {e.message}")

if __name__ == "__main__":
  if len(sys.argv) > 1: dofile(Path(sys.argv[1]), name="__main__")
  else: repl()
