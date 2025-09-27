#!/usr/bin/env python3

import sys, typing
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
	@dataclass(frozen=True)
	class ProcedureType(Type):
		parameter_types: list["Evaluator.Type"]
		return_type: "Evaluator.Type"
		varargs_type: "Evaluator.Type | None" = None
		is_macro: bool = False

	type_type = NamedType("type")
	type_code = NamedType("code")
	type_anytype = NamedType("anytype")
	type_noreturn = NamedType("noreturn")
	type_void = NamedType("void")
	type_bool = NamedType("bool")
	type_int = NamedType("int")
	type_float = NamedType("float")
	type_string = NamedType("string")

	@dataclass
	class Value:
		class Procedure:
			def __init__(self, body: typing.Callable[..., "Evaluator.Value"] | Tuple, name: str, parameter_names: list[str], varargs_name: str | None = None) -> None:
				self.body = body
				self.name = name
				self.parameter_names = parameter_names
				self.varargs_name = varargs_name

			def __call__(self, *args: "Evaluator.Value", **kwargs: typing.Any) -> "Evaluator.Value":
				if callable(self.body): return self.body(*args, **kwargs)
				else: raise NotImplementedError()

		ty: "Evaluator.Type"
		contents: "Evaluator.Type | Procedure | Code | Integer | Float | String | bool | None"

		@property
		def as_type(self) -> "Evaluator.Type":
			if self.ty != Evaluator.type_type or not isinstance(self.contents, Evaluator.Type): raise TypeError()
			return self.contents
		@property
		def as_procedure(self) -> Procedure:
			if not isinstance(self.ty, Evaluator.ProcedureType) or not isinstance(self.contents, Evaluator.Value.Procedure): raise TypeError()
			return self.contents
		@property
		def as_code(self) -> Code:
			if self.ty != Evaluator.type_code or not isinstance(self.contents, Code): raise TypeError()
			return self.contents
		@property
		def as_int(self) -> int:
			if self.ty != Evaluator.type_int or not isinstance(self.contents, int): raise TypeError()
			return self.contents
		@property
		def as_bool(self) -> bool:
			if self.ty != Evaluator.type_bool or not isinstance(self.contents, bool): raise TypeError()
			return self.contents

	value_void = Value(type_void, None)

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

	def coerce(self, value: Value, ty: Type) -> Value | None:
		if value.ty == ty or ty == Evaluator.type_anytype: return value
		return None

	def type_as_string(self, ty: Type) -> str:
		if isinstance(ty, Evaluator.NamedType): return f"($type '{ty.name})"
		raise NotImplementedError()

	def value_as_string(self, value: Value) -> str:
		if value.ty == Evaluator.type_type: return self.type_as_string(value.as_type)
		if value.ty == Evaluator.type_code: return code_as_string(value.as_code)
		if value.ty == Evaluator.type_int: return str(value.as_int)
		if isinstance(value.ty, Evaluator.ProcedureType): return str(value.as_procedure)
		raise NotImplementedError(self.type_as_string(value.ty))

	def __call__(self, code: Code, scope: Scope) -> Value:
		if isinstance(code.contents, Identifier):
			value = scope.find(code.contents)
			if value is None: raise Evaluator.Error("Variable not in scope.", code)
			return value
		elif isinstance(code.contents, Integer): return Evaluator.Value(Evaluator.type_int, code.contents)
		elif isinstance(code.contents, Float): return Evaluator.Value(Evaluator.type_float, code.contents)
		elif isinstance(code.contents, String): return Evaluator.Value(Evaluator.type_string, code.contents)
		assert isinstance(code.contents, Tuple), code.contents
		if len(code.contents) == 0: raise Evaluator.Error("You attempted to call a procedure without specifying a name.", code)
		op_code, *arg_codes = code.contents
		op = self(op_code, scope)
		try: proc = op.as_procedure
		except TypeError: raise Evaluator.Error("You attempted to call something that is not a procedure.", code)
		assert isinstance(op.ty, Evaluator.ProcedureType)
		if len(arg_codes) < len(op.ty.parameter_types) or (len(arg_codes) != len(op.ty.parameter_types) and op.ty.varargs_type is None):
			raise Evaluator.Error("Arity mismatch.", code)
		pargs = [self(arg_code, scope) if not op.ty.is_macro or (op.ty.parameter_types[i] if i < len(op.ty.parameter_types) else op.ty.varargs_type) != Evaluator.type_code else Evaluator.Value(Evaluator.type_code, arg_code) for i, arg_code in enumerate(arg_codes)]
		posargs, varargs = pargs[:len(op.ty.parameter_types)], pargs[len(op.ty.parameter_types):]
		for i, posarg in enumerate(posargs):
			coerced = self.coerce(posarg, op.ty.parameter_types[i])
			if coerced is None: raise Evaluator.Error(f"Argument {self.type_as_string(posarg.ty)} could not coerce to {self.type_as_string(op.ty.parameter_types[i])}", code)
			posargs[i] = coerced
		for i, vararg in enumerate(varargs):
			coerced = self.coerce(vararg, op.ty.varargs_type or Evaluator.type_void)
			if coerced is None: raise Evaluator.Error(f"Variadic argument {self.type_as_string(vararg.ty)} could not coerce to {self.type_as_string(op.ty.varargs_type or Evaluator.type_void)}", code)
			varargs[i] = coerced
		return proc(*posargs, varargs=varargs, ty=op.ty, evaluator=self, calling_scope=scope, code=code, op_code=op_code, arg_codes=arg_codes)

def compiler_block(**kwargs: typing.Any) -> Evaluator.Value:
	block_scope = Evaluator.Scope(kwargs["calling_scope"])
	for code_value in kwargs["varargs"]:
		kwargs["evaluator"](code_value.as_code, block_scope)
	return Evaluator.value_void

def compiler_define(name_value: Evaluator.Value, value: Evaluator.Value, **kwargs: typing.Any) -> Evaluator.Value:
	try: name = name_value.as_code.as_identifier
	except TypeError: raise Evaluator.Error("$define expects argument one to be an identifier.", kwargs["code"])
	kwargs["calling_scope"].entries[name] = value
	return Evaluator.value_void

def compiler_operator(kind_value: Evaluator.Value, **kwargs: typing.Any) -> Evaluator.Value:
	try: kind = kind_value.as_code.as_identifier
	except TypeError: raise Evaluator.Error("$operator expects argument one to be an identifier.", kwargs["code"])
	varargs = typing.cast(list[Evaluator.Value], kwargs["varargs"])
	if len(varargs) == 0: raise Evaluator.Error("$operator must have at least one variadic argument to determine its return type.", kwargs["code"])
	try:
		if varargs[0].ty == Evaluator.type_int:
			if kind == Identifier("+"): return Evaluator.Value(Evaluator.type_int, sum([kwargs["evaluator"].coerce(vararg, Evaluator.type_int).as_int for vararg in varargs])) # type: ignore
			if kind == Identifier("=="):
				arg0 = kwargs["evaluator"].coerce(varargs[0], Evaluator.type_int).as_int
				return Evaluator.Value(Evaluator.type_bool, all([arg0 == kwargs["evaluator"].coerce(vararg, Evaluator.type_int).as_int for vararg in varargs[1:]])) # type: ignore
	except (TypeError, AttributeError): raise Evaluator.Error(f"Arguments could not coerce to {kwargs["evaluator"].type_as_string(varargs[0].ty)}.", kwargs["code"])
	raise Evaluator.Error(f"$operator '{kind}' not defined for {kwargs["evaluator"].type_as_string(varargs[0].ty)}.", kwargs["code"])

def compiler_quote(code_value: Evaluator.Value, **kwargs: typing.Any) -> Evaluator.Value:
	return code_value

def compiler_assert(cond_value: Evaluator.Value, **kwargs: typing.Any) -> Evaluator.Value:
	if not cond_value.as_bool: raise Evaluator.Error("Assertion failed!", kwargs["code"])
	return Evaluator.value_void

compiler_scope = Evaluator.Scope(None)
compiler_scope.entries.update({
	Identifier("$block"): Evaluator.Value(Evaluator.ProcedureType([], Evaluator.type_void, Evaluator.type_code, is_macro=True), Evaluator.Value.Procedure(compiler_block, "$block", [], varargs_name="codes")),
	Identifier("$define"): Evaluator.Value(Evaluator.ProcedureType([Evaluator.type_code, Evaluator.type_anytype], Evaluator.type_void), Evaluator.Value.Procedure(compiler_define, "$define", ["name", "value"])),
	Identifier("$operator"): Evaluator.Value(Evaluator.ProcedureType([Evaluator.type_code], Evaluator.type_anytype, Evaluator.type_anytype), Evaluator.Value.Procedure(compiler_operator, "$operator", ["kind"], varargs_name="args")),
	Identifier("$quote"): Evaluator.Value(Evaluator.ProcedureType([Evaluator.type_code], Evaluator.type_code, is_macro=True), Evaluator.Value.Procedure(compiler_quote, "$quote", ["code"])),
	Identifier("$assert"): Evaluator.Value(Evaluator.ProcedureType([Evaluator.type_bool], Evaluator.type_void), Evaluator.Value.Procedure(compiler_assert, "$assert", ["cond"])),
})

def dostring(src_path: str, evaluator: Evaluator, s: str, p: int = 0, show_result: bool = False) -> None:
	# evaluator.global_scope.entries["__file__"] = Evaluator.Value(type_string, src_path)
	while True:
		try: code, next_pos = parse_code(s, p)
		except ParseError as e:
			line, col = offset_to_line_col(s, e.location)
			print(f"parse error @ {src_path}:{line}:{col}: {e.message}"); break
		if code is None: break
		p = next_pos
		# print(code_as_string(code))
		try:
			result = evaluator(code, evaluator.global_scope)
			if show_result and result is not Evaluator.value_void: print(evaluator.value_as_string(result))
		except Evaluator.Error as e:
			line, col = offset_to_line_col(s, e.code.start)
			print(f"evaluation error @ {src_path}:{line}:{col} near '{code_as_string(e.code)}': {e.message}"); break

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
		dostring("repl", evaluator, src, pos, show_result=True)

if __name__ == "__main__":
	if len(sys.argv) > 1: dofile(Path(sys.argv[1]))
	else: repl()
