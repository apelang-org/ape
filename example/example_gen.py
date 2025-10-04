kwlist = ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield', 'match', 'case']
punct = "+-*@/%&|~^:=.,;<>(){}[]"
def colorize(s: str) -> str:
  result = ""
  p = 0
  while True:
    while True:
      while p < len(s) and s[p].isspace(): result += s[p]; p += 1
      if p < len(s) and s[p] == '#':
        start = p
        while p < len(s) and s[p] != '\n': p += 1
        result += f"""<span style="color: #90AECC;">{s[start:p]}</span>"""
        continue
      break
    if p >= len(s): break
    start = p
    if s[p].isalpha() or s[p] == '_':
      while p < len(s) and (s[p].isalnum() or s[p] == '_'): p += 1
      peek = p
      while peek < len(s) and s[peek].isspace(): peek += 1
      result += f"""<span style="color: {"#FE9494" if s[start:p] in kwlist else "#E6C2ED" if (peek >= len(s) or s[peek] != '(') else "white"};">{s[start:p]}</span>"""
    elif s[p].isdigit():
      while p < len(s) and s[p].isdigit(): p += 1
      result += f"""<span style="color: #FFB6C1">{s[start:p]}</span>"""
    elif s[p] == '"':
      p += 1
      while p < len(s) and s[p] != '\n' and (s[p - 1] == '\\' or s[p] != '"'): p += 1
      p += 1
      result += f"""<span style="color: #FFB6C1">{s[start:p]}</span>"""
    elif s[p] in punct:
      while p < len(s) and s[p] in punct: p += 1
      result += f"""<span style="color: aqua;">{s[start:p]}</span>"""
    else: result += s[p]; p += 1
  return result

with open("example.ape") as f: src = f.read().strip()

with open("example.svg", "w") as f:
  lines = src.split('\n')
  pad = 7
  w = max(len(line.rstrip()) + 1 for line in lines) * 7 + pad * 2
  h = len(lines) * 15 + pad * 2
  f.write(f"""
<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
<foreignObject width="{w}" height="{h}">
  <div xmlns="http://www.w3.org/1999/xhtml">
    <div style="height: {h-pad*2}px; background-color: {"#3B567E"}; font-family: monospace; border-radius: {pad}px; padding: {pad}px;">
      <div style="white-space: pre;">{colorize(src)}</div>
    </div>
  </div>
</foreignObject>
</svg>
""")
