#!/usr/bin/env python3

import sys
from dataclasses import dataclass
from pathlib import Path

class Tuple(list["Code"]): pass
class Identifier(str): pass
class Keyword(str): pass
class String(str): pass
class Integer(int): pass
class Float(float): pass
@dataclass
class Code:
	start: int
	end: int
	contents: Tuple | Identifier | Keyword | String | Integer | Float

	@property
	def as_tuple(self) -> Tuple:
		if not isinstance(self.contents, Tuple): raise TypeError()
		return self.contents
	@property
	def as_identifier(self) -> Identifier:
		if not isinstance(self.contents, Identifier): raise TypeError()
		return self.contents
	@property
	def as_keyword(self) -> Keyword:
		if not isinstance(self.contents, Keyword): raise TypeError()
		return self.contents
	@property
	def as_string(self) -> String:
		if not isinstance(self.contents, String): raise TypeError()
		return self.contents
	@property
	def as_integer(self) -> Integer:
		if not isinstance(self.contents, Integer): raise TypeError()
		return self.contents
	@property
	def as_float(self) -> Float:
		if not isinstance(self.contents, Float): raise TypeError()
		return self.contents

class ParseError(Exception):
	def __init__(self, message: str, location: int) -> None:
		super().__init__(message, location)
		self.message = message
		self.location = location

def parse_code(s: str, p: int, disable_implicit_parentheses: bool = False, adjusted_indent: int = 0) -> tuple[Code | None, int]:
	initial_p = p
	indents: list[int] = []
	implicitnesses: list[bool] = []
	codes: list[Code] = []
	while True:
		start = p
		newline_was_skipped = False
		beginning_of_line = start
		while True:
			while p < len(s) and s[p].isspace():
				if s[p] == '\n': newline_was_skipped = True; beginning_of_line = p + 1
				p += 1
			if p < len(s) and s[p] == ';':
				while p < len(s) and s[p] != '\n': p += 1
				continue
			break
		if p >= len(s): break
		first_exp_of_line = newline_was_skipped or start == initial_p
		if first_exp_of_line:
			indent = p - beginning_of_line
			if len(indents) == 0 or indent > indents[-1]: indents.append(indent)
			if not disable_implicit_parentheses and s[p] not in "(),'`":
				implicitnesses.append(True)
				codes.append(Code(start, -1, Tuple()))
		start = p
		if s[p] == '(':
			p += 1
			implicitnesses.append(False)
			codes.append(Code(start, -1, Tuple()))
		elif s[p] == ')':
			p += 1
			if len(implicitnesses) == 0 or implicitnesses[-1]: raise ParseError("You have an unexpected closing parenthesis.", start)
			implicitnesses.pop()
			if len(codes) > 1:
				popped = codes.pop()
				popped.end = p
				codes[-1].as_tuple.append(popped)
		elif s[p].isdigit():
			while p < len(s) and s[p].isdigit(): p += 1
			(codes[-1].as_tuple if len(codes) else codes).append(Code(start, p, Integer(s[start:p], base=0)))
		elif s[p] == '"':
			p += 1
			while p < len(s) and (s[p - 1] == '\\' or s[p] != '"'): p += 1
			if p >= len(s) or s[p] != '"': raise ParseError("You have an unterminated string literal.", start)
			p += 1
			(codes[-1].as_tuple if len(codes) else codes).append(Code(start, p, String(s[start + 1:p - 1])))
		elif s[p] == ',':
			p += 1
			code, next_pos = parse_code(s, p, disable_implicit_parentheses=True, adjusted_indent=0)
			if code is None: raise ParseError("$insert of nothing.", start)
			p = next_pos
			(codes[-1].as_tuple if len(codes) else codes).append(Code(start, -1, Tuple([Code(start, start + 7, Identifier("$insert")), code])))
		elif s[p] == "'":
			p += 1
			code, next_pos = parse_code(s, p, disable_implicit_parentheses=True, adjusted_indent=0)
			if code is None: raise ParseError("$quote of nothing.", start)
			p = next_pos
			(codes[-1].as_tuple if len(codes) else codes).append(Code(start, -1, Tuple([Code(start, start + 6, Identifier("$quote")), code])))
		elif s[p] == '`':
			p += 1
			code, next_pos = parse_code(s, p, disable_implicit_parentheses=True, adjusted_indent=0)
			if code is None: raise ParseError("$quasiquote of nothing.", start)
			p = next_pos
			(codes[-1].as_tuple if len(codes) else codes).append(Code(start, -1, Tuple([Code(start, start + 11, Identifier("$quasiquote")), code])))
		else:
			while p < len(s) and s[p] not in " \t\n\r(),'`;": p += 1
			(codes[-1].as_tuple if len(codes) else codes).append(Code(start, p, Identifier(s[start:p])))
		if p < len(s) and p > 1 and s[p - 1] not in " \t\n\r(),'`" and s[p] not in " \t\n\r();":
			raise ParseError("You have conjoined expressions.", p)
		peek = p
		while peek < len(s) and s[peek].isspace() and s[peek] not in "\n;": peek += 1
		last_exp_of_line = peek >= len(s) or s[peek] in "\n;"
		beginning_of_next_line = peek
		while True:
			while peek < len(s) and s[peek].isspace():
				if s[peek] == '\n': beginning_of_next_line = peek + 1
				peek += 1
			if peek < len(s) and s[peek] == ';':
				while peek < len(s) and s[peek] != '\n': peek += 1
				continue
			break
		next_indent = (peek - beginning_of_next_line if peek < len(s) else 0)
		if last_exp_of_line:
			while len(indents) > 0 and next_indent <= indents[-1]:
				indents.pop()
				if len(codes) > 1:
					popped = codes.pop()
					popped.end = p
					codes[-1].as_tuple.append(popped)
			if len(implicitnesses) > 0 and implicitnesses[-1]: implicitnesses.pop()
		if len(implicitnesses) == 0 and (len(indents) == 0 or next_indent <= indents[0]): break
	if len(implicitnesses) != 0: raise ParseError("You are missing a closing parenthesis.", initial_p)
	assert len(codes) <= 1
	result = codes.pop() if len(codes) == 1 else None
	if result: result.end = p
	return result, p

def code_as_string(code: Code) -> str:
	if isinstance(code.contents, Tuple): return f"({" ".join(map(code_as_string, code.contents))})"
	elif isinstance(code.contents, Identifier): return code.contents
	elif isinstance(code.contents, Keyword): return '#' + code.contents
	elif isinstance(code.contents, String): return '"' + code.contents + '"'
	elif isinstance(code.contents, Integer): return str(code.contents)
	elif isinstance(code.contents, Float): return str(code.contents) # type: ignore
	else: raise NotImplementedError()

def offset_to_line_col(s: str, p: int) -> tuple[int, int]:
	line, col, i = 1, 1, 0
	while i < min(p, len(s)):
		if s[i] == '\n': line += 1; col = 0
		i += 1; col += 1
	return line, col

class Evaluator:
	class Error(Exception):
		def __init__(self, message: str, code: Code) -> None:
			super().__init__(message, code)
			self.message = message
			self.code = code

	@dataclass(frozen=True)
	class Type: pass
	@dataclass(frozen=True)
	class NamedType(Type): name: str

	type_type = NamedType("type")
	type_code = NamedType("code")
	type_noreturn = NamedType("noreturn")
	type_void = NamedType("void")
	type_bool = NamedType("bool")

	@dataclass
	class Value:
		ty: "Evaluator.Type"
		contents: "Evaluator.Type | Code | None"

	class Scope:
		def __init__(self, parent: "Evaluator.Scope | None") -> None:
			self.parent = parent
			self.entries: dict[str, "Evaluator.Value"] = {}

		def find(self, key: str) -> "Evaluator.Value | None":
			if key in self.entries: return self.entries[key]
			if self.parent is None: return None
			return self.parent.find(key)

	def __init__(self, s: str, global_scope: Scope | None = None) -> None:
		self.s = s
		self.global_scope = global_scope if global_scope else Evaluator.Scope(compiler_scope)

	def __call__(self, code: Code, scope: Scope) -> Value:
		raise NotImplementedError()

compiler_scope = Evaluator.Scope(None)

def dostring(src_path: str, evaluator: Evaluator, s: str, p: int = 0) -> None:
	# evaluator.global_scope.entries["__file__"] = Evaluator.Value(type_string, src_path)
	while True:
		try: code, next_pos = parse_code(s, p)
		except ParseError as e:
			line, col = offset_to_line_col(s, e.location)
			print(f"parse error @ {src_path}:{line}:{col}: {e.message}"); break
		if code is None: break
		p = next_pos
		print(code_as_string(code))
		evaluator(code, evaluator.global_scope)

def dofile(path: Path) -> Evaluator.Scope:
	src = path.read_text()
	evaluator = Evaluator(src)
	dostring(str(path), evaluator, src)
	return evaluator.global_scope

def repl() -> None:
	src = ""
	repl_scope = Evaluator.Scope(compiler_scope)
	while True:
		pos = len(src)
		try: src += input("> ")
		except (KeyboardInterrupt, EOFError): print(""); break
		evaluator = Evaluator(src, repl_scope)
		dostring("repl", evaluator, src, pos)

if __name__ == "__main__":
	if len(sys.argv) > 1: dofile(Path(sys.argv[1]))
	else: repl()
