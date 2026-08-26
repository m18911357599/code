#!/usr/bin/env python3
"""Rank table DSL: describe / compile / decompile cann/hcomm rank table JSON."""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

MAX_RANK_COUNT = 65536
MAX_DEVICE_ID = 64
BACKUP_LOCAL_ID = 64
DEFAULT_TCP_PORT = 16666
MIN_TCP_PORT = 1
MAX_TCP_PORT = 65535
MAX_LEVEL_LIST = 8
MAX_NET_LAYER = 7
MAX_NET_INST_LEN = 1024
MAX_RANK_ADDR = 24
MAX_PORT_COUNT = 16
MAX_PORT_LEN = 32
MAX_ADDR_LEN = 256
MAX_PLANE_LEN = 1024
MAX_BACKUP_ADDR = 16
MAX_LISTEN_PORT = 65536
MAX_SERVER_ID_LEN = 64

NET_TYPES = {
    "CLOS",
    "TOPO_FILE_DESC",
    "1DMESH",
    "2DMESH",
    "A3_SERVER",
    "A2_AX_SERVER",
}
ADDR_TYPES = {"EID", "IPV4", "IPV6"}
EID_HEX_LEN = 32


class RankTableError(Exception):
    pass


class ParseError(RankTableError):
    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
        loc = f"{line}:{col}: " if line else ""
        super().__init__(f"{loc}{message}")


class CheckError(RankTableError):
    pass


@dataclass
class Address:
    addr_type: str
    addr: str
    ports: list[str] = field(default_factory=list)
    plane_id: str = "0"
    backup_addrs: list[str] = field(default_factory=list)


@dataclass
class Level:
    net_layer: int
    net_instance_id: str
    net_type: str = "CLOS"
    net_attr: str = ""
    addrs: list[Address] = field(default_factory=list)


@dataclass
class ControlPlane:
    addr_type: str
    addr: str
    listen_port: int


@dataclass
class Rank:
    rank_id: int
    local_id: int
    device_id: int
    replaced_local_id: Optional[int] = None
    device_port: int = DEFAULT_TCP_PORT
    host_port: int = DEFAULT_TCP_PORT
    levels: list[Level] = field(default_factory=list)
    control_plane: Optional[ControlPlane] = None


@dataclass
class V1Device:
    device_id: int
    rank_id: int
    device_ip: Optional[str] = None
    device_port: Optional[int] = None
    host_port: Optional[int] = None
    super_device_id: Optional[str] = None
    backup_device_ip: Optional[str] = None
    backup_device_port: Optional[int] = None


@dataclass
class Server:
    server_id: str
    host_ip: Optional[str] = None
    devices: list[V1Device] = field(default_factory=list)


@dataclass
class SuperPod:
    super_pod_id: str
    server_ids: list[str] = field(default_factory=list)


@dataclass
class RankTable:
    version: str
    status: str = "completed"
    detour: bool = False
    ranks: list[Rank] = field(default_factory=list)
    servers: list[Server] = field(default_factory=list)
    super_pods: list[SuperPod] = field(default_factory=list)

    @property
    def is_v2(self) -> bool:
        return self.version.startswith("2.")

    def rank_count(self) -> int:
        if self.ranks:
            return len(self.ranks)
        return sum(len(s.devices) for s in self.servers)


# ---------------------------------------------------------------------------
# lexer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>[ \t\r]+)
    |(?P<nl>\n)
    |(?P<comment>\#[^\n]*|//[^\n]*)
    |(?P<string>"(?:\\.|[^"\\])*")
    |(?P<bare>
          \d+(?:\.\d+){3}(?:/\d+)?   # IPv4
        | \d+\.\d+                   # 1.0 / 2.0
        | \d+/\d+                    # Die/Port
        | [A-Za-z_][A-Za-z0-9_./:-]*
      )
    |(?P<number>\d+)
    |(?P<lbrace>\{)
    |(?P<rbrace>\})
    |(?P<lbrack>\[)
    |(?P<rbrack>\])
    |(?P<comma>,)
    """,
    re.VERBOSE,
)


@dataclass
class Tok:
    kind: str
    value: str
    line: int
    col: int


def _unescape(s: str) -> str:
    return s.encode("utf-8").decode("unicode_escape") if "\\" in s else s


def lex(src: str) -> list[Tok]:
    tokens: list[Tok] = []
    line = 1
    col = 1
    i = 0
    n = len(src)
    while i < n:
        m = _TOKEN_RE.match(src, i)
        if not m:
            raise ParseError(f"unexpected character {src[i]!r}", line, col)
        kind = m.lastgroup or ""
        raw = m.group()
        if kind == "nl":
            line += 1
            col = 1
        elif kind in ("ws", "comment"):
            col += len(raw)
        else:
            val = raw
            if kind == "string":
                val = _unescape(raw[1:-1])
                kind = "ident"
            elif kind == "number":
                kind = "number"
            elif kind == "bare":
                kind = "ident"
            tokens.append(Tok(kind, val, line, col))
            col += len(raw)
        i = m.end()
    tokens.append(Tok("eof", "", line, col))
    return tokens


class Parser:
    def __init__(self, tokens: list[Tok]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Tok:
        return self.toks[self.i]

    def eat(self, kind: Optional[str] = None, value: Optional[str] = None) -> Tok:
        tok = self.peek()
        if tok.kind == "eof":
            raise ParseError("unexpected end of file", tok.line, tok.col)
        if kind and tok.kind != kind:
            raise ParseError(f"expected {kind}, got {tok.kind} {tok.value!r}", tok.line, tok.col)
        if value is not None and tok.value != value:
            raise ParseError(f"expected {value!r}, got {tok.value!r}", tok.line, tok.col)
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
        raise ParseError(f"expected number, got {tok.value!r}", tok.line, tok.col)

    def atom(self) -> str:
        tok = self.peek()
        if tok.kind in ("ident", "number"):
            self.i += 1
            return tok.value
        raise ParseError(f"expected identifier, got {tok.value!r}", tok.line, tok.col)

    def str_list(self) -> list[str]:
        self.eat("lbrack")
        items: list[str] = []
        if not self.accept("rbrack"):
            items.append(self.atom())
            while self.accept("comma"):
                if self.peek().kind == "rbrack":
                    break
                items.append(self.atom())
            self.eat("rbrack")
        return items

    def parse(self) -> RankTable:
        if self.peek().kind != "ident" or self.peek().value != "ranktable":
            raise ParseError("file must start with 'ranktable'", self.peek().line, self.peek().col)
        self.eat("ident", "ranktable")
        version = self.atom()
        self.eat("lbrace")
        table = RankTable(version=version)
        while self.peek().kind != "rbrace" and self.peek().kind != "eof":
            key = self.peek()
            if key.kind != "ident":
                raise ParseError(f"unexpected {key.value!r}", key.line, key.col)
            if key.value == "status":
                self.eat("ident")
                table.status = self.atom()
            elif key.value == "detour":
                self.eat("ident")
                table.detour = self.atom().lower() == "true"
            elif key.value == "rank":
                table.ranks.append(self.parse_rank())
            elif key.value == "server":
                table.servers.append(self.parse_server())
            elif key.value == "super_pod":
                table.super_pods.append(self.parse_super_pod())
            else:
                raise ParseError(f"unknown statement {key.value!r}", key.line, key.col)
        self.eat("rbrace")
        if self.peek().kind != "eof":
            raise ParseError("trailing tokens after ranktable block", self.peek().line, self.peek().col)
        return table

    def parse_rank(self) -> Rank:
        self.eat("ident", "rank")
        rank_id = self.number()
        self.eat("lbrace")
        rank = Rank(rank_id=rank_id, local_id=0, device_id=0)
        saw_local = saw_device = False
        while self.peek().kind != "rbrace":
            key = self.ident()
            if key == "local":
                rank.local_id = self.number()
                saw_local = True
            elif key == "device":
                rank.device_id = self.number()
                saw_device = True
            elif key == "replaced_local":
                rank.replaced_local_id = self.number()
            elif key == "device_port":
                rank.device_port = self.number()
            elif key == "host_port":
                rank.host_port = self.number()
            elif key == "level":
                rank.levels.append(self.parse_level())
            elif key == "control_plane":
                addr_type = self.atom()
                addr = self.atom()
                listen_kw = self.ident()
                if listen_kw != "listen":
                    raise ParseError(f"expected 'listen', got {listen_kw!r}", self.peek().line, self.peek().col)
                rank.control_plane = ControlPlane(addr_type, addr, self.number())
            else:
                raise ParseError(f"unknown rank field {key!r}", self.peek().line, self.peek().col)
        self.eat("rbrace")
        if not saw_local or not saw_device:
            raise ParseError(f"rank {rank_id} requires local and device", self.peek().line, self.peek().col)
        return rank

    def parse_level(self) -> Level:
        # 'level' already consumed
        net_layer = self.number()
        net_inst = self.atom()
        net_type = self.atom()
        net_attr = ""
        if self.peek().kind == "ident" and self.peek().value == "attr":
            self.eat("ident")
            net_attr = self.atom()
        self.eat("lbrace")
        level = Level(net_layer, net_inst, net_type, net_attr)
        while self.peek().kind != "rbrace":
            self.eat("ident", "addr")
            level.addrs.append(self.parse_addr())
        self.eat("rbrace")
        return level

    def parse_addr(self) -> Address:
        addr_type = self.atom()
        addr = self.atom()
        ports: list[str] = []
        plane = "0"
        backups: list[str] = []
        while self.peek().kind == "ident" and self.peek().value in ("ports", "plane", "backup"):
            kw = self.ident()
            if kw == "ports":
                ports = self.str_list()
            elif kw == "plane":
                plane = self.atom()
            else:
                backups = self.str_list()
        if not ports:
            raise ParseError("addr requires ports [...]", self.peek().line, self.peek().col)
        return Address(addr_type, addr, ports, plane, backups)

    def parse_server(self) -> Server:
        self.eat("ident", "server")
        server_id = self.atom()
        host_ip = None
        if self.peek().kind == "ident" and self.peek().value == "host":
            self.eat("ident")
            host_ip = self.atom()
        self.eat("lbrace")
        server = Server(server_id, host_ip)
        while self.peek().kind != "rbrace":
            self.eat("ident", "device")
            server.devices.append(self.parse_v1_device())
        self.eat("rbrace")
        return server

    def parse_v1_device(self) -> V1Device:
        device_id = self.number()
        fields: dict[str, Any] = {}
        while self.peek().kind == "ident" and self.peek().value in {
            "ip",
            "port",
            "host_port",
            "rank",
            "sdid",
            "backup_ip",
            "backup_port",
        }:
            key = self.ident()
            fields[key] = self.number() if key in ("port", "host_port", "rank", "backup_port") else self.atom()
        if "rank" not in fields:
            raise ParseError("device requires rank", self.peek().line, self.peek().col)
        return V1Device(
            device_id=device_id,
            rank_id=int(fields["rank"]),
            device_ip=fields.get("ip"),
            device_port=fields.get("port"),
            host_port=fields.get("host_port"),
            super_device_id=fields.get("sdid"),
            backup_device_ip=fields.get("backup_ip"),
            backup_device_port=fields.get("backup_port"),
        )

    def parse_super_pod(self) -> SuperPod:
        self.eat("ident", "super_pod")
        pod_id = self.atom()
        self.eat("lbrace")
        ids: list[str] = []
        while self.peek().kind != "rbrace":
            self.eat("ident", "server")
            ids.append(self.atom())
        self.eat("rbrace")
        return SuperPod(pod_id, ids)


def parse_dsl(src: str) -> RankTable:
    return Parser(lex(src)).parse()


# ---------------------------------------------------------------------------
# JSON load / emit
# ---------------------------------------------------------------------------

def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise CheckError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value)
    raise CheckError(f"{name} must be an integer, got {value!r}")


def _as_str(value: Any, name: str) -> str:
    if value is None:
        raise CheckError(f"{name} is required")
    return str(value)


def _opt_int(obj: dict[str, Any], key: str) -> Optional[int]:
    if key not in obj or obj[key] in (None, ""):
        return None
    return _as_int(obj[key], key)


def load_json(data: dict[str, Any]) -> RankTable:
    version = _as_str(data.get("version", ""), "version")
    table = RankTable(version=version, status=_as_str(data.get("status", "completed"), "status"))
    detour = data.get("detour", "")
    if isinstance(detour, bool):
        table.detour = detour
    else:
        table.detour = str(detour).lower() == "true"

    if "rank_list" in data:
        for item in data["rank_list"]:
            table.ranks.append(_load_rank(item))
        declared = data.get("rank_count")
        if declared is not None and _as_int(declared, "rank_count") != len(table.ranks):
            raise CheckError("rank_count does not match rank_list size")
        return table

    if "server_list" in data:
        for item in data["server_list"]:
            table.servers.append(_load_server(item))
        for pod in data.get("super_pod_list", []):
            table.super_pods.append(
                SuperPod(
                    _as_str(pod.get("super_pod_id"), "super_pod_id"),
                    [_as_str(s.get("server_id"), "server_id") for s in pod.get("server_list", [])],
                )
            )
        return table

    raise CheckError("JSON must contain rank_list (v2) or server_list (v1)")


def _load_rank(item: dict[str, Any]) -> Rank:
    rank = Rank(
        rank_id=_as_int(item.get("rank_id"), "rank_id"),
        local_id=_as_int(item.get("local_id"), "local_id"),
        device_id=_as_int(item.get("device_id"), "device_id"),
        device_port=_as_int(item.get("device_port", DEFAULT_TCP_PORT), "device_port"),
        host_port=_as_int(item.get("host_port", DEFAULT_TCP_PORT), "host_port"),
    )
    if "replaced_local_id" in item:
        rank.replaced_local_id = _as_int(item["replaced_local_id"], "replaced_local_id")
    for level in item.get("level_list", []):
        addrs = []
        for addr in level.get("rank_addr_list", []):
            addrs.append(
                Address(
                    addr_type=_as_str(addr.get("addr_type"), "addr_type"),
                    addr=_as_str(addr.get("addr"), "addr"),
                    ports=[_as_str(p, "ports") for p in addr.get("ports", [])],
                    plane_id=_as_str(addr.get("plane_id", "0"), "plane_id"),
                    backup_addrs=[_as_str(b, "backup_addr") for b in addr.get("backup_addr", [])],
                )
            )
        rank.levels.append(
            Level(
                net_layer=_as_int(level.get("net_layer"), "net_layer"),
                net_instance_id=_as_str(level.get("net_instance_id"), "net_instance_id"),
                net_type=_as_str(level.get("net_type", "CLOS"), "net_type"),
                net_attr=_as_str(level.get("net_attr", ""), "net_attr") if level.get("net_attr") is not None else "",
                addrs=addrs,
            )
        )
    if "control_plane" in item:
        cp = item["control_plane"]
        rank.control_plane = ControlPlane(
            _as_str(cp.get("addr_type"), "addr_type"),
            _as_str(cp.get("addr"), "addr"),
            _as_int(cp.get("listen_port"), "listen_port"),
        )
    return rank


def _load_server(item: dict[str, Any]) -> Server:
    devices = []
    for dev in item.get("device", item.get("devices", [])):
        devices.append(
            V1Device(
                device_id=_as_int(dev.get("device_id"), "device_id"),
                rank_id=_as_int(dev.get("rank_id"), "rank_id"),
                device_ip=str(dev["device_ip"]) if "device_ip" in dev else None,
                device_port=_opt_int(dev, "device_port"),
                host_port=_opt_int(dev, "host_port"),
                super_device_id=str(dev["super_device_id"]) if "super_device_id" in dev else None,
                backup_device_ip=str(dev["backup_device_ip"]) if "backup_device_ip" in dev else None,
                backup_device_port=_opt_int(dev, "backup_device_port"),
            )
        )
    return Server(
        server_id=_as_str(item.get("server_id"), "server_id"),
        host_ip=str(item["host_ip"]) if item.get("host_ip") else None,
        devices=devices,
    )


def emit_json(table: RankTable) -> dict[str, Any]:
    if table.is_v2 or table.ranks:
        return _emit_v2(table)
    return _emit_v1(table)


def _emit_v2(table: RankTable) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": table.status,
        "version": table.version if table.version.startswith("2.") else "2.0",
        "rank_count": len(table.ranks),
    }
    if table.detour:
        out["detour"] = "true"
    ranks = []
    for rank in table.ranks:
        item: dict[str, Any] = {
            "rank_id": rank.rank_id,
            "local_id": rank.local_id,
            "device_id": rank.device_id,
        }
        if rank.replaced_local_id is not None:
            item["replaced_local_id"] = rank.replaced_local_id
        if rank.device_port != DEFAULT_TCP_PORT:
            item["device_port"] = rank.device_port
        if rank.host_port != DEFAULT_TCP_PORT:
            item["host_port"] = rank.host_port
        levels = []
        for level in rank.levels:
            addrs = []
            for addr in level.addrs:
                a: dict[str, Any] = {
                    "addr_type": addr.addr_type,
                    "addr": addr.addr,
                    "ports": list(addr.ports),
                }
                if addr.plane_id != "0":
                    a["plane_id"] = addr.plane_id
                if addr.backup_addrs:
                    a["backup_addr"] = list(addr.backup_addrs)
                addrs.append(a)
            lv: dict[str, Any] = {
                "net_layer": level.net_layer,
                "net_instance_id": level.net_instance_id,
                "net_type": level.net_type,
                "net_attr": level.net_attr,
                "rank_addr_list": addrs,
            }
            levels.append(lv)
        item["level_list"] = levels
        if rank.control_plane:
            item["control_plane"] = {
                "addr_type": rank.control_plane.addr_type,
                "addr": rank.control_plane.addr,
                "listen_port": rank.control_plane.listen_port,
            }
        ranks.append(item)
    out["rank_list"] = ranks
    return out


def _s(value: Any) -> str:
    return str(value)


def _emit_v1(table: RankTable) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": table.status,
        "version": table.version,
        "server_count": _s(len(table.servers)),
        "server_list": [],
    }
    for server in table.servers:
        s: dict[str, Any] = {"server_id": server.server_id, "device": []}
        if server.host_ip:
            s["host_ip"] = server.host_ip
        for dev in server.devices:
            d: dict[str, Any] = {
                "device_id": _s(dev.device_id),
                "rank_id": _s(dev.rank_id),
            }
            if dev.device_ip is not None:
                d["device_ip"] = dev.device_ip
            if dev.device_port is not None:
                d["device_port"] = _s(dev.device_port)
            if dev.host_port is not None:
                d["host_port"] = _s(dev.host_port)
            if dev.super_device_id is not None:
                d["super_device_id"] = _s(dev.super_device_id)
            if dev.backup_device_ip is not None:
                d["backup_device_ip"] = dev.backup_device_ip
            if dev.backup_device_port is not None:
                d["backup_device_port"] = _s(dev.backup_device_port)
            s["device"].append(d)
        out["server_list"].append(s)
    if table.super_pods:
        out["super_pod_list"] = [
            {
                "super_pod_id": pod.super_pod_id,
                "server_list": [{"server_id": sid} for sid in pod.server_ids],
            }
            for pod in table.super_pods
        ]
    return out


# ---------------------------------------------------------------------------
# DSL emit
# ---------------------------------------------------------------------------

def _q(s: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./:-]*", s) or re.fullmatch(r"\d+(?:\.\d+){3}", s) or re.fullmatch(r"\d+/\d+", s):
        return s
    return json.dumps(s, ensure_ascii=False)


def _ports(ports: list[str]) -> str:
    return "[" + ", ".join(_q(p) for p in ports) + "]"


def emit_dsl(table: RankTable) -> str:
    lines = [f"ranktable {table.version} {{", f"  status {table.status}"]
    if table.detour:
        lines.append("  detour true")
    if table.ranks:
        for rank in table.ranks:
            lines.append("")
            lines.append(f"  rank {rank.rank_id} {{")
            lines.append(f"    local {rank.local_id}")
            lines.append(f"    device {rank.device_id}")
            if rank.replaced_local_id is not None:
                lines.append(f"    replaced_local {rank.replaced_local_id}")
            if rank.device_port != DEFAULT_TCP_PORT:
                lines.append(f"    device_port {rank.device_port}")
            if rank.host_port != DEFAULT_TCP_PORT:
                lines.append(f"    host_port {rank.host_port}")
            for level in rank.levels:
                head = f"    level {level.net_layer} {_q(level.net_instance_id)} {level.net_type}"
                if level.net_attr:
                    head += f" attr {_q(level.net_attr)}"
                lines.append(f"{head} {{")
                for addr in level.addrs:
                    row = f"      addr {addr.addr_type} {_q(addr.addr)} ports {_ports(addr.ports)}"
                    if addr.plane_id != "0":
                        row += f" plane {_q(addr.plane_id)}"
                    if addr.backup_addrs:
                        row += " backup [" + ", ".join(_q(b) for b in addr.backup_addrs) + "]"
                    lines.append(row)
                lines.append("    }")
            if rank.control_plane:
                cp = rank.control_plane
                lines.append(f"    control_plane {cp.addr_type} {_q(cp.addr)} listen {cp.listen_port}")
            lines.append("  }")
    for server in table.servers:
        lines.append("")
        head = f"  server {_q(server.server_id)}"
        if server.host_ip:
            head += f" host {server.host_ip}"
        lines.append(f"{head} {{")
        for dev in server.devices:
            row = f"    device {dev.device_id}"
            if dev.device_ip:
                row += f" ip {dev.device_ip}"
            if dev.device_port is not None:
                row += f" port {dev.device_port}"
            if dev.host_port is not None:
                row += f" host_port {dev.host_port}"
            if dev.super_device_id is not None:
                row += f" sdid {dev.super_device_id}"
            if dev.backup_device_ip:
                row += f" backup_ip {dev.backup_device_ip}"
            if dev.backup_device_port is not None:
                row += f" backup_port {dev.backup_device_port}"
            row += f" rank {dev.rank_id}"
            lines.append(row)
        lines.append("  }")
    for pod in table.super_pods:
        lines.append("")
        lines.append(f"  super_pod {_q(pod.super_pod_id)} {{")
        for sid in pod.server_ids:
            lines.append(f"    server {_q(sid)}")
        lines.append("  }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# checks (aligned with RankTableInfo::Check / Deserialize)
# ---------------------------------------------------------------------------

def _check_addr(addr: Address, where: str) -> None:
    if addr.addr_type not in ADDR_TYPES:
        raise CheckError(f"{where}: invalid addr_type {addr.addr_type!r}")
    if not (1 <= len(addr.addr) <= MAX_ADDR_LEN):
        raise CheckError(f"{where}: addr length out of range")
    if addr.addr_type == "IPV4":
        ipaddress.IPv4Address(addr.addr)
    elif addr.addr_type == "IPV6":
        ipaddress.IPv6Address(addr.addr)
    elif addr.addr_type == "EID":
        if len(addr.addr) != EID_HEX_LEN or not re.fullmatch(r"[0-9A-Fa-f]+", addr.addr):
            raise CheckError(f"{where}: EID must be {EID_HEX_LEN} hex chars")
        if addr.backup_addrs:
            raise CheckError(f"{where}: backup_addr does not support EID")
    if not (1 <= len(addr.ports) <= MAX_PORT_COUNT):
        raise CheckError(f"{where}: ports count must be 1..{MAX_PORT_COUNT}")
    for p in addr.ports:
        if not (1 <= len(p) <= MAX_PORT_LEN):
            raise CheckError(f"{where}: port {p!r} length out of range")
    if len(addr.plane_id) > MAX_PLANE_LEN:
        raise CheckError(f"{where}: plane_id too long")
    if len(addr.backup_addrs) > MAX_BACKUP_ADDR:
        raise CheckError(f"{where}: backup_addr exceeds {MAX_BACKUP_ADDR}")
    for b in addr.backup_addrs:
        if addr.addr_type == "IPV4":
            ipaddress.IPv4Address(b)
        elif addr.addr_type == "IPV6":
            ipaddress.IPv6Address(b)


def check(table: RankTable) -> None:
    if table.ranks:
        _check_v2(table)
    if table.servers:
        _check_v1(table)
    if not table.ranks and not table.servers:
        raise CheckError("rank table has no ranks or servers")


def _check_v2(table: RankTable) -> None:
    if table.version != "2.0":
        raise CheckError(f'version {table.version!r} is not "2.0"')
    n = len(table.ranks)
    if n == 0 or n > MAX_RANK_COUNT:
        raise CheckError(f"rankCount {n} out of range [1, {MAX_RANK_COUNT}]")
    ids = []
    local_ids = set()
    recorded_replace: Optional[int] = None
    for rank in table.ranks:
        if rank.rank_id >= n:
            raise CheckError(f"rank_id {rank.rank_id} out of range [0, {n})")
        if rank.rank_id in ids:
            raise CheckError(f"rank_id {rank.rank_id} is repeated")
        ids.append(rank.rank_id)
        if rank.local_id > BACKUP_LOCAL_ID:
            raise CheckError(f"local_id {rank.local_id} out of range [0, {BACKUP_LOCAL_ID}]")
        if rank.device_id > MAX_DEVICE_ID:
            raise CheckError(f"device_id {rank.device_id} out of range [0, {MAX_DEVICE_ID}]")
        if not (MIN_TCP_PORT <= rank.device_port <= MAX_TCP_PORT):
            raise CheckError(f"device_port {rank.device_port} out of range")
        if not (MIN_TCP_PORT <= rank.host_port <= MAX_TCP_PORT):
            raise CheckError(f"host_port {rank.host_port} out of range")
        replaced = rank.replaced_local_id if rank.replaced_local_id is not None else rank.local_id
        if rank.local_id != BACKUP_LOCAL_ID and replaced != rank.local_id:
            raise CheckError("replaced_local_id must equal local_id unless local_id is 64")
        if rank.local_id == BACKUP_LOCAL_ID:
            if rank.replaced_local_id is None:
                raise CheckError("replaced_local_id required when local_id is 64")
            if rank.replaced_local_id > BACKUP_LOCAL_ID - 1:
                raise CheckError("replaced_local_id out of range")
            if recorded_replace is not None:
                raise CheckError("multiple replaced rank is configured")
            recorded_replace = rank.replaced_local_id
        else:
            local_ids.add(rank.local_id)
        if not rank.levels:
            raise CheckError(f"rank {rank.rank_id} level_list is empty")
        if len(rank.levels) > MAX_LEVEL_LIST:
            raise CheckError("level_list exceeds maximum 8")
        prev = -1
        for level in rank.levels:
            if level.net_layer > MAX_NET_LAYER:
                raise CheckError(f"net_layer {level.net_layer} out of range [0, 7]")
            if level.net_layer <= prev:
                raise CheckError("level is not increased in sequence")
            prev = level.net_layer
            if not (1 <= len(level.net_instance_id) <= MAX_NET_INST_LEN):
                raise CheckError("net_instance_id length out of range")
            if level.net_type not in NET_TYPES:
                raise CheckError(f"invalid net_type {level.net_type!r}")
            if len(level.addrs) > MAX_RANK_ADDR:
                raise CheckError("rank_addr_list exceeds 24")
            for i, addr in enumerate(level.addrs):
                _check_addr(addr, f"rank {rank.rank_id} layer {level.net_layer} addr[{i}]")
        if rank.control_plane:
            cp = rank.control_plane
            if cp.addr_type not in ADDR_TYPES:
                raise CheckError("invalid control_plane addr_type")
            if not (1 <= len(cp.addr) <= MAX_ADDR_LEN):
                raise CheckError("control_plane addr length out of range")
            if not (1 <= cp.listen_port <= MAX_LISTEN_PORT):
                raise CheckError("listen_port out of range")
    for rid in range(n):
        if rid not in ids:
            raise CheckError(f"rank_id is not continuous, missing {rid}")
    if recorded_replace is not None and recorded_replace in local_ids:
        raise CheckError("configuring same local_id with replaced one simultaneously")
    # same (layer, instance) must share addr-list size
    sizes: dict[tuple[int, str], int] = {}
    for rank in table.ranks:
        for level in rank.levels:
            key = (level.net_layer, level.net_instance_id)
            sz = len(level.addrs)
            if key in sizes and sizes[key] != sz:
                raise CheckError(
                    f"rank_addrs size differs for net_instance_id {level.net_instance_id!r} at layer {level.net_layer}"
                )
            sizes[key] = sz


def _check_v1(table: RankTable) -> None:
    if table.version not in ("1.0", "1.2"):
        raise CheckError(f"unsupported v1 version {table.version!r}")
    if table.status not in ("completed", "initializing"):
        raise CheckError(f"invalid status {table.status!r}")
    if not table.servers:
        raise CheckError("server_list is empty")
    seen_server = set()
    seen_rank: set[int] = set()
    prev_max = -1
    for server in table.servers:
        if not server.server_id or len(server.server_id) > MAX_SERVER_ID_LEN:
            raise CheckError("server_id length must be 1..64")
        if server.server_id in seen_server:
            raise CheckError(f"duplicate server_id {server.server_id!r}")
        seen_server.add(server.server_id)
        if not server.devices:
            raise CheckError(f"server {server.server_id} has no devices")
        ranks = [d.rank_id for d in server.devices]
        if any(r in seen_rank for r in ranks):
            raise CheckError("rank_id is not unique")
        seen_rank.update(ranks)
        if min(ranks) <= prev_max:
            raise CheckError("rank_id ranges of different servers must not interleave")
        prev_max = max(prev_max, max(ranks))
        for dev in server.devices:
            if dev.device_id > MAX_DEVICE_ID:
                raise CheckError(f"device_id {dev.device_id} out of range")
            for port in (dev.device_port, dev.host_port, dev.backup_device_port):
                if port is not None and not (MIN_TCP_PORT <= port <= MAX_TCP_PORT):
                    raise CheckError(f"port {port} out of range")
            if dev.device_ip:
                _check_ip_any(dev.device_ip, "device_ip")
            if dev.backup_device_ip:
                _check_ip_any(dev.backup_device_ip, "backup_device_ip")
        if server.host_ip:
            ipaddress.IPv4Address(server.host_ip)
    n = len(seen_rank)
    if set(range(n)) != seen_rank:
        raise CheckError("v1 rank_id must be unique and cover [0, N)")
    known = {s.server_id for s in table.servers}
    for pod in table.super_pods:
        for sid in pod.server_ids:
            if sid not in known:
                raise CheckError(f"super_pod references unknown server {sid!r}")


def _check_ip_any(text: str, name: str) -> None:
    try:
        ipaddress.ip_address(text)
    except ValueError as exc:
        raise CheckError(f"{name} {text!r} is not a valid IP") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _write(path: Optional[str], text: str) -> None:
    if path:
        Path(path).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


def _load_any(path: str) -> RankTable:
    text = _read(path)
    stripped = text.lstrip()
    if stripped.startswith("{") or path.endswith(".json"):
        return load_json(json.loads(text))
    return parse_dsl(text)


def cmd_compile(args: argparse.Namespace) -> int:
    table = parse_dsl(_read(args.input))
    check(table)
    text = json.dumps(emit_json(table), indent=2, ensure_ascii=False) + "\n"
    _write(args.output, text)
    return 0


def cmd_decompile(args: argparse.Namespace) -> int:
    table = load_json(json.loads(_read(args.input)))
    check(table)
    _write(args.output, emit_dsl(table))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    table = _load_any(args.input)
    check(table)
    print(
        f"ok version={table.version} ranks={len(table.ranks)} "
        f"servers={len(table.servers)} rank_count={table.rank_count()}"
    )
    return 0


def cmd_dump(args: argparse.Namespace) -> int:
    table = _load_any(args.input)
    check(table)
    _write(args.output, emit_dsl(table))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HCOMM rank table DSL")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_io(p: argparse.ArgumentParser) -> None:
        p.add_argument("input")
        p.add_argument("-o", "--output", default=None)

    add_io(sub.add_parser("compile", help="DSL -> JSON"))
    add_io(sub.add_parser("decompile", help="JSON -> DSL"))
    add_io(sub.add_parser("check", help="validate DSL or JSON"))
    add_io(sub.add_parser("dump", help="normalize to DSL"))

    args = parser.parse_args(argv)
    try:
        return {
            "compile": cmd_compile,
            "decompile": cmd_decompile,
            "check": cmd_check,
            "dump": cmd_dump,
        }[args.cmd](args)
    except (RankTableError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
