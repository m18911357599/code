"""Parse operator/algorithm DSL (.coll)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class CollSyntaxError(Exception):
    def __init__(self, message: str, line: int = 0):
        self.line = line
        super().__init__(f"{line}: {message}" if line else message)


@dataclass
class Match:
    ranks_ge: Optional[int] = None
    layers_ge: Optional[int] = None
    power_of_two: bool = False


@dataclass
class AlgoDef:
    name: str
    pattern: str = "ring"
    match: Match = field(default_factory=Match)
    intra: Optional[str] = None
    intra_layer: int = 0
    inter: Optional[str] = None
    inter_layer: int = 1


@dataclass
class OperatorDef:
    name: str
    reduce: str = "SUM"
    dtype: str = "i32"
    count: int = 8
    use: str = "ring"
    layer: Optional[int] = None
    root: int = 0


@dataclass
class Program:
    topo_from: Optional[str] = None
    algos: dict[str, AlgoDef] = field(default_factory=dict)
    operators: list[OperatorDef] = field(default_factory=list)


_TOKEN_RE = re.compile(
    r"""
    (?P<ws>[ \t\r]+)
    |(?P<nl>\n)
    |(?P<comment>\#[^\n]*|//[^\n]*)
    |(?P<string>"(?:\\.|[^"\\])*")
    |(?P<bare>
          \d+\.\d+
        | [A-Za-z_][A-Za-z0-9_./:-]*
      )
    |(?P<number>\d+)
    |(?P<lbrace>\{)
    |(?P<rbrace>\})
    |(?P<op>>=|<=|==)
    """,
    re.VERBOSE,
)


@dataclass
class Tok:
    kind: str
    value: str
    line: int


def _lex(src: str) -> list[Tok]:
    toks: list[Tok] = []
    line = 1
    i = 0
    while i < len(src):
        m = _TOKEN_RE.match(src, i)
        if not m:
            raise CollSyntaxError(f"unexpected {src[i]!r}", line)
        kind = m.lastgroup or ""
        raw = m.group()
        if kind == "nl":
            line += 1
        elif kind not in ("ws", "comment"):
            val = raw[1:-1] if kind == "string" else raw
            k = "ident" if kind in ("string", "bare") else kind
            toks.append(Tok(k, val, line))
        i = m.end()
    toks.append(Tok("eof", "", line))
    return toks


class Parser:
    def __init__(self, toks: list[Tok]):
        self.toks = toks
        self.i = 0

    def peek(self) -> Tok:
        return self.toks[self.i]

    def eat(self, kind: Optional[str] = None, value: Optional[str] = None) -> Tok:
        tok = self.peek()
        if tok.kind == "eof":
            raise CollSyntaxError("unexpected eof", tok.line)
        if kind and tok.kind != kind:
            raise CollSyntaxError(f"expected {kind}, got {tok.value!r}", tok.line)
        if value is not None and tok.value != value:
            raise CollSyntaxError(f"expected {value!r}", tok.line)
        self.i += 1
        return tok

    def accept(self, kind: Optional[str] = None, value: Optional[str] = None) -> Optional[Tok]:
        tok = self.peek()
        if tok.kind == "eof":
            return None
        if kind and tok.kind != kind:
            return None
        if value is not None and tok.value != value:
            return None
        self.i += 1
        return tok

    def ident(self) -> str:
        return self.eat("ident").value

    def number(self) -> int:
        tok = self.peek()
        if tok.kind == "number":
            return int(self.eat("number").value)
        if tok.kind == "ident" and tok.value.isdigit():
            return int(self.eat("ident").value)
        raise CollSyntaxError(f"expected number, got {tok.value!r}", tok.line)

    def parse(self) -> Program:
        prog = Program()
        while self.peek().kind != "eof":
            key = self.ident()
            if key == "topo":
                self.eat("ident", "from")
                prog.topo_from = self.eat("ident").value
            elif key == "algo":
                algo = self.parse_algo()
                prog.algos[algo.name] = algo
            elif key == "operator":
                prog.operators.append(self.parse_operator())
            else:
                raise CollSyntaxError(f"unknown statement {key!r}", self.peek().line)
        if "ring" not in prog.algos:
            prog.algos["ring"] = AlgoDef("ring", "ring", Match(ranks_ge=2))
        if "recursive_hd" not in prog.algos:
            prog.algos["recursive_hd"] = AlgoDef(
                "recursive_hd", "recursive_hd", Match(power_of_two=True)
            )
        if "hierarchical" not in prog.algos:
            prog.algos["hierarchical"] = AlgoDef(
                "hierarchical",
                "hierarchical",
                Match(layers_ge=2),
                intra="ring",
                intra_layer=0,
                inter="ring",
                inter_layer=1,
            )
        return prog

    def parse_algo(self) -> AlgoDef:
        name = self.ident()
        self.eat("lbrace")
        algo = AlgoDef(name)
        while self.peek().kind != "rbrace":
            k = self.ident()
            if k == "match":
                algo.match = self.parse_match()
            elif k == "pattern":
                algo.pattern = self.ident()
            elif k == "intra":
                algo.intra = self.ident()
                self.eat("ident", "on")
                self.eat("ident", "layer")
                algo.intra_layer = self.number()
            elif k == "inter":
                algo.inter = self.ident()
                self.eat("ident", "on")
                self.eat("ident", "layer")
                algo.inter_layer = self.number()
            else:
                raise CollSyntaxError(f"unknown algo field {k!r}", self.peek().line)
        self.eat("rbrace")
        if algo.pattern == "hierarchical" or algo.intra:
            algo.pattern = "hierarchical"
        return algo

    def parse_match(self) -> Match:
        m = Match()
        while True:
            tok = self.peek()
            if tok.kind in ("rbrace", "eof") or (
                tok.kind == "ident" and tok.value in ("pattern", "intra", "inter", "match")
            ):
                break
            word = self.ident()
            if word == "and":
                continue
            if word == "power_of_two":
                m.power_of_two = True
                continue
            if word == "ranks":
                self._rel_ge()
                m.ranks_ge = self.number()
            elif word == "layers":
                self._rel_ge()
                m.layers_ge = self.number()
            else:
                raise CollSyntaxError(f"unknown match {word!r}", tok.line)
        return m

    def _rel_ge(self) -> None:
        tok = self.peek()
        if tok.kind == "op" and tok.value in (">=", ">", "=="):
            self.eat("op")
        elif tok.kind == "ident" and tok.value == ">=":
            self.eat("ident")

    def parse_operator(self) -> OperatorDef:
        name = self.ident()
        self.eat("lbrace")
        op = OperatorDef(name)
        while self.peek().kind != "rbrace":
            k = self.ident()
            if k == "reduce":
                op.reduce = self.ident().upper()
            elif k == "dtype":
                op.dtype = self.ident()
            elif k == "count":
                op.count = self.number()
            elif k == "root":
                op.root = self.number()
            elif k == "use":
                op.use = self.ident()
                if self.peek().kind == "ident" and self.peek().value == "on":
                    self.eat("ident")
                    self.eat("ident", "layer")
                    op.layer = self.number()
            else:
                raise CollSyntaxError(f"unknown operator field {k!r}", self.peek().line)
        self.eat("rbrace")
        return op


def parse_coll(src: str) -> Program:
    return Parser(_lex(src)).parse()


def parse_coll_file(path: str | Path) -> Program:
    return parse_coll(Path(path).read_text(encoding="utf-8"))
