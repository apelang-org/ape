import random, os, sys, traceback
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

class TokenKind(IntEnum):
  END_OF_INPUT = 128
  ERROR = 129
  INDENT = 130
  NEWLINE = 131

  IDENTIFIER = 132
  INTEGER = 133
  FLOAT = 134
  STRING = 135
  THICK_STRING = 136

  LTLT = 148
  GTGT = 149
  # SLASHSLASH = 150
  # STARSTAR = 151
  DASHGT = 152
  EQEQ = 153
  BANGEQ = 154
  LTEQ = 155
  GTEQ = 156
  PLUSEQ = 157
  DASHEQ = 158
  STAREQ = 159
  # ATEQ = 160
  SLASHEQ = 161
  PERCENTEQ = 162
  AMPEQ = 163
  PIPEEQ = 164
  CARETEQ = 165
  COLONEQ = 166

  LTLTEQ = 167
  GTGTEQ = 167
  # SLASHSLASHEQ = 169
  # STARSTAREQ = 170
  DOTDOTDOT = 171

  KW_False = 192
  KW_None = 193
  KW_True = 194
  KW_and = 195
  KW_as = 196
  KW_assert = 197
  # KW_async = 198
  KW_await = 199
  KW_break = 200
  KW_case = 201
  KW_class = 202
  KW_continue = 203
  KW_def = 204
  KW_del = 205
  KW_elif = 206
  KW_else = 207
  KW_except = 207
  KW_finally = 209
  KW_for = 210
  KW_from = 211
  # KW_global = 212
  KW_if = 213
  KW_import = 214
  KW_in = 215
  # KW_is = 216
  KW_lambda = 217
  KW_match = 218
  KW_nonlocal = 219
  KW_not = 220
  KW_or = 221
  KW_pass = 222
  KW_raise = 223
  KW_return = 224
  KW_try = 225
  KW_while = 226
  KW_with = 227
  KW_yield = 228

  CHEESY_not_in = 248

  @staticmethod
  def as_str(kind: int) -> str: return TokenKind(kind).name if kind in TokenKind else f"'{chr(kind)}'"

@dataclass
class Token:
  kind: int
  location: int
  length: int

  def as_str(self, s: str) -> str: return s[self.location:self.location + self.length]

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
  if newline_was_skipped and start != 0:
    return Token(TokenKind.NEWLINE, beginning_of_line - 1, 1)
  pb = p
  while pb > 1 and s[pb - 1].isspace() and s[pb - 1] != '\n': pb -= 1
  indent = p - beginning_of_line
  if s[start].isspace() and s[pb - 1] == '\n' and indent > 0:
    return Token(TokenKind.INDENT, beginning_of_line, indent)
  start = p
  if s[p].isalpha() or s[p] == '_':
    while p < len(s) and (s[p].isalnum() or s[p] == '_'): p += 1
    test = s[start:p]
    match len(test):
      case 2:
        if test == "if": return Token(TokenKind.KW_if, start, 2)
        if test == "as": return Token(TokenKind.KW_as, start, 2)
        if test == "in": return Token(TokenKind.KW_in, start, 2)
        if test == "or": return Token(TokenKind.KW_or, start, 2)
      case 3:
        if test == "and": return Token(TokenKind.KW_and, start, 3)
        if test == "def": return Token(TokenKind.KW_def, start, 3)
        if test == "del": return Token(TokenKind.KW_del, start, 3)
        if test == "for": return Token(TokenKind.KW_for, start, 3)
        if test == "not":
          peek = p
          while peek < len(s) and s[peek].isspace() and s[peek] != '\n': peek += 1
          if peek + 1 < len(s) and s[peek:peek + 2] == "in" and (peek + 2 >= len(s) or s[peek + 2].isspace() or s[peek + 2] == '#'):
            return Token(TokenKind.CHEESY_not_in, start, (peek + 2) - start)
          return Token(TokenKind.KW_not, start, 3)
        if test == "try": return Token(TokenKind.KW_try, start, 3)
      case 4:
        if test == "None": return Token(TokenKind.KW_None, start, 4)
        if test == "True": return Token(TokenKind.KW_True, start, 4)
        if test == "case": return Token(TokenKind.KW_case, start, 4)
        if test == "elif": return Token(TokenKind.KW_elif, start, 4)
        if test == "else": return Token(TokenKind.KW_else, start, 4)
        if test == "from": return Token(TokenKind.KW_from, start, 4)
        if test == "pass": return Token(TokenKind.KW_pass, start, 4)
        if test == "with": return Token(TokenKind.KW_with, start, 4)
      case 5:
        if test == "False": return Token(TokenKind.KW_False, start, 5)
        if test == "await": return Token(TokenKind.KW_await, start, 5)
        if test == "break": return Token(TokenKind.KW_break, start, 5)
        if test == "class": return Token(TokenKind.KW_class, start, 5)
        if test == "match": return Token(TokenKind.KW_match, start, 5)
        if test == "raise": return Token(TokenKind.KW_raise, start, 5)
        if test == "while": return Token(TokenKind.KW_while, start, 5)
        if test == "yield": return Token(TokenKind.KW_yield, start, 5)
      case 6:
        if test == "assert": return Token(TokenKind.KW_assert, start, 6)
        if test == "except": return Token(TokenKind.KW_except, start, 6)
        if test == "import": return Token(TokenKind.KW_import, start, 6)
        if test == "lambda": return Token(TokenKind.KW_lambda, start, 6)
        if test == "return": return Token(TokenKind.KW_return, start, 6)
      case 7:
        if test == "finally": return Token(TokenKind.KW_finally, start, 7)
      case 8:
        if test == "continue": return Token(TokenKind.KW_continue, start, 8)
        if test == "nonlocal": return Token(TokenKind.KW_nonlocal, start, 8)
      case _: pass
    return Token(TokenKind.IDENTIFIER, start, p - start)
  elif s[p].isdigit():
    while p < len(s) and s[p].isdigit(): p += 1
    return Token(TokenKind.INTEGER, start, p - start)
  elif p + 2 < len(s) and s[p:p + 3] == '"""':
    p += 3
    while p + 2 < len(s) and (s[p - 1] == '\\' or s[p:p + 3] != '"""'): p += 1
    if p + 2 >= len(s) or s[p:p + 3] != '"""': return Token(TokenKind.ERROR, start, p - start + 2)
    p += 3
    return Token(TokenKind.THICK_STRING, start, p - start)
  elif s[p] == '"':
    p += 1
    while p < len(s) and (s[p - 1] == '\\' or s[p] != '"'): p += 1
    if p >= len(s) or s[p] != '"': return Token(TokenKind.ERROR, start, p - start)
    p += 1
    return Token(TokenKind.STRING, start, p - start)
  if p + 2 < len(s):
    test = s[p:p + 3]
    if test == "<<=": return Token(TokenKind.LTLTEQ, start, 3)
    if test == ">>=": return Token(TokenKind.GTGTEQ, start, 3)
    if test == "...": return Token(TokenKind.DOTDOTDOT, start, 3)
  if p + 1 < len(s):
    test = s[p:p + 2]
    if test == "<<": return Token(TokenKind.LTLT, start, 2)
    if test == ">>": return Token(TokenKind.GTGT, start, 2)
    if test == "->": return Token(TokenKind.DASHGT, start, 2)
    if test == "==": return Token(TokenKind.EQEQ, start, 2)
    if test == "!=": return Token(TokenKind.BANGEQ, start, 2)
    if test == "<=": return Token(TokenKind.LTEQ, start, 2)
    if test == ">=": return Token(TokenKind.GTEQ, start, 2)
    if test == "+=": return Token(TokenKind.PLUSEQ, start, 2)
    if test == "-=": return Token(TokenKind.DASHEQ, start, 2)
    if test == "*=": return Token(TokenKind.STAREQ, start, 2)
    if test == "/=": return Token(TokenKind.SLASHEQ, start, 2)
    if test == "%=": return Token(TokenKind.PERCENTEQ, start, 2)
    if test == "&=": return Token(TokenKind.AMPEQ, start, 2)
    if test == "|=": return Token(TokenKind.PIPEEQ, start, 2)
    if test == "^=": return Token(TokenKind.CARETEQ, start, 2)
    if test == ":=": return Token(TokenKind.COLONEQ, start, 2)
  if s[p] in "+-*/%&|~^$:=.,;()[]{}<>": return Token(ord(s[p]), start, 1)
  return Token(TokenKind.ERROR, start, 1)

def print_all_tokens(s: str, p: int = 0) -> None:
  while True:
    token = token_at(s, p)
    p = token.location + token.length
    print(TokenKind.as_str(token.kind), f'"{token.as_str(s).encode("unicode_escape").decode()}"')
    if token.kind == TokenKind.END_OF_INPUT: break

def offset_to_line_col(s: str, p: int) -> tuple[int, int]:
  line, col, i = 1, 1, 0
  while i < min(p, len(s)):
    if s[i] == '\n': line += 1; col = 0
    i += 1; col += 1
  return line, col

PRECEDENCES: list[int] = [-1] * 256
PRECEDENCES[TokenKind.KW_or] = 0
PRECEDENCES[TokenKind.KW_and] = 1
PRECEDENCES[TokenKind.EQEQ] = 2; PRECEDENCES[TokenKind.BANGEQ] = 2; PRECEDENCES[TokenKind.LTEQ] = 2
PRECEDENCES[TokenKind.GTEQ] = 2; PRECEDENCES[ord('<')] = 2; PRECEDENCES[ord('>')] = 2
PRECEDENCES[TokenKind.KW_in] = 2; PRECEDENCES[TokenKind.CHEESY_not_in] = 2
PRECEDENCES[ord('|')] = 3
PRECEDENCES[ord('^')] = 4
PRECEDENCES[ord('&')] = 5
PRECEDENCES[TokenKind.LTLT] = 6; PRECEDENCES[TokenKind.GTGT] = 6
PRECEDENCES[ord('+')] = 7; PRECEDENCES[ord('-')] = 7
PRECEDENCES[ord('*')] = 8; PRECEDENCES[ord('/')] = 8; PRECEDENCES[ord('%')] = 8

ASSIGN_OPS: list[int] = [False] * 256
ASSIGN_OPS[TokenKind.PLUSEQ] = True
ASSIGN_OPS[TokenKind.DASHEQ] = True
ASSIGN_OPS[TokenKind.STAREQ] = True
ASSIGN_OPS[TokenKind.SLASHEQ] = True
ASSIGN_OPS[TokenKind.PERCENTEQ] = True
ASSIGN_OPS[TokenKind.AMPEQ] = True
ASSIGN_OPS[TokenKind.PIPEEQ] = True
ASSIGN_OPS[TokenKind.CARETEQ] = True
ASSIGN_OPS[TokenKind.COLONEQ] = True
ASSIGN_OPS[TokenKind.LTLTEQ] = True
ASSIGN_OPS[TokenKind.GTGTEQ] = True

@dataclass
class Code:
  start_location: int
  end_location: int
@dataclass
class Code_Error(Code): pass
@dataclass
class Code_Literal(Code):
  is_thick: bool
  value: int | float | str
@dataclass
class Code_Variable(Code):
  is_nonlocal: bool
  name: str
@dataclass
class Code_Declaration(Code):
  is_nonlocal: bool
  is_constant: bool
  name: str
  type_expr: Code | None
  value_expr: Code | None
@dataclass
class Code_UnaryOp(Code):
  op: Token
  right: Code
@dataclass
class Code_BinaryOp(Code):
  left: Code
  op: Token
  right: Code
@dataclass
class Code_Uninitialized(Code): pass
@dataclass
class Code_TypeInstantiation(Code):
  name: str
@dataclass
class Code_Yield(Code):
  body: list[Code]
@dataclass
class Code_Call(Code):
  expression: Code
  arguments: list[Code]

class Parser:
  @dataclass
  class Error:
    message: str
    token: Token
    stacktrace: traceback.StackSummary

  def __init__(self, src: str, pos: int = 0) -> None:
    self.src = src
    self.pos = pos
    self.indents = [0]
    self.errors: list[Parser.Error] = []

def peek(p: Parser, n: int = 1) -> Token:
  assert n > 0
  token: Token | None = None
  pos = p.pos
  for _ in range(n):
    token = token_at(p.src, pos)
    if token.kind == TokenKind.ERROR: p.errors.append(Parser.Error(f"A token error occured starting with '{p.src[token.location]}'.", token, traceback.extract_stack()))
    pos = token.location + token.length
  assert token is not None
  return token

def eat(p: Parser, expect: int) -> Token:
  token = token_at(p.src, p.pos)
  p.pos = token.location + token.length
  if expect != token.kind: p.errors.append(Parser.Error(f"I expected {TokenKind.as_str(expect)} but saw {TokenKind.as_str(token.kind)}.", token, traceback.extract_stack()))
  return token

def parse_leaf(p: Parser) -> Code:
  result: Code | None = None
  is_nonlocal = peek(p).kind == TokenKind.KW_nonlocal
  is_constant = peek(p).kind == ord('$')
  n = 2 if is_nonlocal or is_constant else 1
  if peek(p, n).kind == TokenKind.IDENTIFIER:
    start = p.pos
    if is_constant:
      eat(p, ord('$'))
      name = eat(p, TokenKind.IDENTIFIER).as_str(p.src)
      return Code_TypeInstantiation(start, p.pos, name)
    else:
      if is_nonlocal: eat(p, TokenKind.KW_nonlocal)
      name = eat(p, TokenKind.IDENTIFIER).as_str(p.src)
      result = Code_Variable(start, p.pos, is_nonlocal, name)
  elif peek(p).kind == TokenKind.INTEGER:
    token = eat(p, TokenKind.INTEGER)
    value = int(token.as_str(p.src), base=0)
    result = Code_Literal(token.location, token.length, False, value)
  elif peek(p).kind == TokenKind.STRING:
    token = eat(p, TokenKind.STRING)
    value = token.as_str(p.src)[1:-1]
    result = Code_Literal(token.location, token.length, False, value)
  elif peek(p).kind == TokenKind.THICK_STRING:
    token = eat(p, TokenKind.THICK_STRING)
    value = token.as_str(p.src)[3:-3]
    result = Code_Literal(token.location, token.length, True, value)
  elif peek(p).kind in [ord('-'), ord('~')]:
    token = eat(p, peek(p).kind)
    right = parse_leaf(p)
    result = Code_UnaryOp(token.location, p.pos, token, right)
  elif peek(p).kind == TokenKind.KW_not:
    token = eat(p, TokenKind.KW_not)
    right = parse_expression(p, PRECEDENCES[ord('<')])
    result = Code_UnaryOp(token.location, p.pos, token, right)
  elif peek(p).kind == ord('('):
    eat(p, ord('('))
    result = parse_expression(p)
    eat(p, ord(')'))
  if result is None:
    p.errors.append(Parser.Error(f"I do not know of an expresion that starts with {TokenKind.as_str(peek(p).kind)}.", peek(p), traceback.extract_stack()))
    result = Code_Error(p.pos, p.pos)
  while True:
    if peek(p).kind == ord('('):
      eat(p, ord('('))
      arguments: list[Code] = []
      while peek(p).kind != ord(')'):
        arguments.append(parse_expression(p))
        if peek(p).kind == ord(';'): eat(p, ord(';'))
        else: break
      eat(p, ord(')'))
      result = Code_Call(result.start_location, p.pos, result, arguments)
      continue
    break
  return result

def parse_expression(p: Parser, min_prec: int = 0) -> Code:
  left = parse_leaf(p)
  while PRECEDENCES[peek(p).kind] >= 0 and PRECEDENCES[peek(p).kind] >= min_prec:
    op = eat(p, peek(p).kind)
    right = parse_expression(p, PRECEDENCES[op.kind])
    left = Code_BinaryOp(left.start_location, p.pos, left, op, right)
  return left

def parse_inline_statement(p: Parser) -> Code:
  start = p.pos
  is_nonlocal = peek(p).kind == TokenKind.KW_nonlocal
  n = 2 if is_nonlocal else 1
  is_constant = peek(p, n).kind == ord('$')
  if is_constant or (peek(p, n).kind == TokenKind.IDENTIFIER and peek(p, n + 1).kind in [ord(':'), ord('=')]):
    if is_nonlocal: eat(p, TokenKind.KW_nonlocal)
    if is_constant: eat(p, ord('$'))
    name = eat(p, TokenKind.IDENTIFIER)
    type_expr: Code | None = None
    if peek(p).kind == ord(':'):
      eat(p, ord(':'))
      type_expr = parse_expression(p)
    value_expr: Code | None = None
    if peek(p).kind == ord('='):
      eat(p, ord('='))
      if peek(p).kind == TokenKind.KW_del:
        token = eat(p, TokenKind.KW_del)
        value_expr = Code_Uninitialized(token.location, token.length)
      else:
        value_expr = parse_expression(p)
    if type_expr is None and value_expr is None: p.errors.append(Parser.Error(f"A declaration (i.e. x: y = z) must have a type (y) or value (z).", name, traceback.extract_stack()))
    return Code_Declaration(start, p.pos, is_nonlocal, is_constant, name.as_str(p.src), type_expr, value_expr)
  else:
    result = parse_expression(p)
    if ASSIGN_OPS[peek(p).kind]:
      op = eat(p, peek(p).kind)
      right = parse_expression(p)
      result = Code_BinaryOp(start, p.pos, result, op, right)
    return result

def parse_line(p: Parser) -> list[Code]:
  line: list[Code] = []
  while peek(p).kind not in [TokenKind.NEWLINE, TokenKind.END_OF_INPUT]:
    line.append(parse_inline_statement(p))
    if peek(p).kind == ord(';'): eat(p, ord(';'))
    else: break
  if peek(p).kind != TokenKind.END_OF_INPUT: eat(p, TokenKind.NEWLINE)
  return line

def parse_block_statement_or_line(p: Parser) -> list[Code]:
  result: list[Code]
  if peek(p).kind == TokenKind.KW_yield and peek(p, 2).kind == ord(':'):
    token = eat(p, TokenKind.KW_yield)
    body = parse_block(p)
    result = [Code_Yield(token.location, p.pos, body)]
  else:
    result = parse_line(p)
  return result

def parse_block_inner(p: Parser) -> list[Code]:
  children: list[Code] = []
  indent_token = peek(p)
  if indent_token.kind != TokenKind.INDENT:
    p.errors.append(Parser.Error(f"I expected an indent after a newline.", indent_token, traceback.extract_stack()))
  if len(p.indents) > 0 and indent_token.length < p.indents[-1]:
    p.errors.append(Parser.Error(f"I expected an indent greater than length {p.indents[-1]}. Your indent was length {indent_token.length}.", indent_token, traceback.extract_stack()))
  p.indents.append(indent_token.length)
  while True:
    indent_token = peek(p)
    if indent_token.kind != TokenKind.INDENT: break
    if indent_token.length != p.indents[-1]:
      if indent_token.length > p.indents[-1]:
        p.errors.append(Parser.Error(f"I expected an indent of length {p.indents[-1]}. Your indent was length {indent_token.length}.", indent_token, traceback.extract_stack()))
      p.indents.pop()
      if indent_token.length not in p.indents:
        p.errors.append(Parser.Error(f"I do not see a previous block with an indent of length {indent_token.length}.", indent_token, traceback.extract_stack()))
      break
    eat(p, TokenKind.INDENT)
    children += parse_block_statement_or_line(p)
  return children

def parse_block(p: Parser) -> list[Code]:
  eat(p, ord(':'))
  children: list[Code] = []
  if peek(p).kind == TokenKind.NEWLINE:
    eat(p, TokenKind.NEWLINE)
    children += parse_block_inner(p)
  else:
    children += parse_block_statement_or_line(p)
    if peek(p).kind == TokenKind.INDENT and (len(p.indents) == 0 or peek(p).length > p.indents[-1]):
      children += parse_block_inner(p)
  return children

def parse_module(p: Parser) -> list[Code]:
  children: list[Code] = []
  while peek(p).kind != TokenKind.END_OF_INPUT:
    children += parse_block_statement_or_line(p)
    if len(p.errors) > 0: break
  return children

def code_block_as_string(s: str, codes: list[Code], level: int = 0) -> str:
  return f":\n" + "\n".join("  " * (level + 1) + code_as_string(s, code, level + 1) for code in codes)

def code_as_string(s: str, code: Code, level: int = 0) -> str:
  if isinstance(code, Code_Literal):
    if isinstance(code.value, str): return f"{'"""' if code.is_thick else '"'}{code.value}{'"""' if code.is_thick else '"'}"
    else: return str(code.value)
  elif isinstance(code, Code_Variable):
    return f"{"nonlocal " if code.is_nonlocal else ""}{code.name}"
  elif isinstance(code, Code_Declaration):
    result = f"{"nonlocal " if code.is_nonlocal else ""}{"$" if code.is_constant else ""}{code.name}"
    if code.type_expr: result += f": {code_as_string(s, code.type_expr, level)}"
    if code.value_expr: result += f" = {code_as_string(s, code.value_expr, level)}"
    return result
  elif isinstance(code, Code_UnaryOp):
    wrap = isinstance(code.right, Code_BinaryOp)
    space = code.op.as_str(s) == "not"
    return f"{code.op.as_str(s)}{" " if space else ""}{"(" if wrap else ""}{code_as_string(src, code.right, level)}{")" if wrap else ""}"
  elif isinstance(code, Code_BinaryOp):
    left_wrap = isinstance(code.left, Code_BinaryOp) and PRECEDENCES[code.left.op.kind] < PRECEDENCES[code.op.kind]
    right_wrap = isinstance(code.right, Code_BinaryOp) and PRECEDENCES[code.right.op.kind] < PRECEDENCES[code.op.kind]
    result = f"{"(" if left_wrap else ""}{code_as_string(src, code.left, level)}{")" if left_wrap else ""}"
    result += f" {code.op.as_str(src)} "
    result += f"{"(" if right_wrap else ""}{code_as_string(src, code.right, level)}{")" if right_wrap else ""}"
    return result
  elif isinstance(code, Code_Uninitialized):
    return "del"
  elif isinstance(code, Code_TypeInstantiation):
    return f"${code.name}"
  elif isinstance(code, Code_Call):
    return f"{code_as_string(s, code.expression, level)}({", ".join(code_as_string(s, argument, level) for argument in code.arguments)})"
  elif isinstance(code, Code_Yield):
    return f"yield{code_block_as_string(src, code.body, level)}"
  else:
    raise NotImplementedError(code.__class__.__name__)

if __name__ == "__main__":
  DEBUG = int(os.getenv("DEBUG", "1"), base=0)
  if len(sys.argv) <= 1:
    print("You must specify a file to interpret.")
    exit(1)
  path = Path(sys.argv[1])
  src = path.read_text()
  # print_all_tokens(src)
  p = Parser(src)
  module = parse_module(p)
  if len(p.errors):
    for err in reversed(p.errors):
      if DEBUG > 0:
        for i, entry in enumerate(err.stacktrace[1:]):
          print(" " * (len(err.stacktrace) - 2 - i), f"{entry.name} @ {entry.filename}:{entry.lineno}")
      line, col = offset_to_line_col(src, err.token.location)
      near = err.token.as_str(src).encode("unicode_escape").decode()
      print(f"\x1b[31mparse error\x1b[0m @ {path}:{line}:{col} near '{near[0:min(len(near), 20)]}': {err.message}")
  else:
    e = list(enumerate(module))
    random.shuffle(e) # to help test out-of-order execution before parallelism.
    for i, code in e:
      print(f"# {i}\n" + code_as_string(src, code))
