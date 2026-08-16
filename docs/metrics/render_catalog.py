#!/usr/bin/env python3
"""Render 分类 → 度量项 → 度量子项 directory into the metrics document."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path(__file__).with_name("catalog.yaml")
DOC = ROOT / "下一代调试调优度量体系.md"

# Map catalog item id → heading prefix used in the current body.
ITEM_HEADING_PREFIX = {
    "A1": "#### A1 ",
    "A2": "#### A2 ",
    "A3": "#### A3 ",
    "A4": "#### A4 ",
    "A5": "#### A5 ",
    "A6": "#### A6 ",
    "A7": "#### A7 ",
    "B1": "#### B1 ",
    "B2": "#### B2 ",
    "B3": "#### B3 ",
    "C1": "#### C1 ",
    "C2": "#### C2 ",
    "C3": "#### C3 ",
    "C4": "#### C4 ",
    "C5": "#### C5 ",
    "D1": "#### D1 ",
    "D2": "#### D2 ",
    "D3": "#### D3 ",
    "D4": "#### D4 ",
    "D5": "#### D5 ",
    "D6": "#### D6 ",
    "D7": "#### D7 ",
    "E1": "#### E1 ",
}

EXPANDED_STUBS = {
    "A8": '''#### A8 多版本一致性（$W_{A,8}=0.04$）（主亲和 F2）

<a id="F2-A8"></a>

**度量方案**：跨 arch 版本算子行为与语义 diff

##### η_ver 跨版本行为一致率
<a id="F2-A8-eta_ver"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越大越好） | 权重 $w$ |
|---|---|---|---|---|
| 跨版本行为一致率 $\\eta_{ver}$ | 跨版本对照 | 行为一致算子 / 总算子 | $\\tau_{100}=100\\%,\\ \\tau_{80}=80\\%,\\ \\tau_{60}=50\\%$ | $0.6$ |

##### n_drift 跨版本语义漂移数
<a id="F2-A8-n_drift"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| 跨版本语义漂移数 $n_{drift}$ | arch 版本对比 | 语义漂移条目数 | $\\tau_{100}=0,\\ \\tau_{80}=2,\\ \\tau_{60}=5$ | $0.4$ |

$$
S_{A,8} = 0.6 \\cdot s_{\\eta_{ver}} + 0.4 \\cdot s_{n_{drift}}
$$
''',
    "A9": '''#### A9 同步流水复杂度（$W_{A,9}=0.04$）（主亲和 F1）

<a id="F1-A9"></a>

**度量方案**：源代码流水阶段数 + 必须同时记住的序域统计

##### n_pipe 流水级数
<a id="F1-A9-n_pipe"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| 流水级数 $n_{pipe}$ | 源代码阶段划分 | 单算子手动流水阶段数 | $\\tau_{100}=1,\\ \\tau_{80}=3,\\ \\tau_{60}=6$ | $0.5$ |

##### n_ord 序域个数
<a id="F1-A9-n_ord"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| 序域个数 $n_{ord}$ | 同步原语扫描 | 须同时记住的序域（核内/核间/Host）个数 | $\\tau_{100}=1,\\ \\tau_{80}=2,\\ \\tau_{60}=4$ | $0.5$ |

$$
S_{A,9} = 0.5 \\cdot s_{n_{pipe}} + 0.5 \\cdot s_{n_{ord}}
$$
''',
    "A10": '''#### A10 设备初始化（$W_{A,10}=0.02$）（主亲和 F5）

<a id="F5-A10"></a>

**度量方案**：AIC/AIV 操作单元初始化代码统计

##### L_init 设备初始化代码行
<a id="F5-A10-L_init"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| 设备初始化代码行 $L_{init}$ | 源代码统计 | AIC/AIV 操作单元初始化代码行 | $\\tau_{100}=0,\\ \\tau_{80}=10,\\ \\tau_{60}=30$ | $1.0$ |

$$
S_{A,10} = s_{L_{init}}
$$
''',
    "A11": '''#### A11 ID 残留对消（$W_{A,11}=0.02$）（主亲和 F1）

<a id="F1-A11"></a>

**度量方案**：GetBlockIdx 消歧与残留扫描

##### L_cancel ID 对消代码行
<a id="F1-A11-L_cancel"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| ID 对消代码行 $L_{cancel}$ | 源代码统计 | GetBlockIdx 消歧代码行 | $\\tau_{100}=0,\\ \\tau_{80}=5,\\ \\tau_{60}=15$ | $0.6$ |

##### n_resid 对消后残留次数
<a id="F1-A11-n_resid"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| 对消后残留次数 $n_{resid}$ | 运行时 ID 扫描 | 同步 ID 残留次数 | $\\tau_{100}=0,\\ \\tau_{80}=1,\\ \\tau_{60}=3$ | $0.4$ |

$$
S_{A,11} = 0.6 \\cdot s_{L_{cancel}} + 0.4 \\cdot s_{n_{resid}}
$$
''',
    "A12": '''#### A12 高层 API 替代覆盖率（$W_{A,12}=0.05$）（主亲和 F3，弱）

<a id="F3-A12"></a>

**度量方案**：高层 API 可替换手写 tiling/搬运的算子覆盖统计

##### η_hl 高层 API 覆盖率
<a id="F3-A12-eta_hl"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越大越好） | 权重 $w$ |
|---|---|---|---|---|
| 高层 API 覆盖率 $\\eta_{hl}$ | 工具链统计 | 可被高层 API 替代的算子比例 | $\\tau_{100}=100\\%,\\ \\tau_{80}=70\\%,\\ \\tau_{60}=40\\%$ | $1.0$ |

$$
S_{A,12} = s_{\\eta_{hl}}
$$
''',
    "A13": '''#### A13 自动同步消除率（$W_{A,13}=0.05$）（主亲和 F1）

<a id="F1-A13"></a>

**度量方案**：编译时自动插入/消除同步覆盖率

##### η_sync 自动同步消除率
<a id="F1-A13-eta_sync"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越大越好） | 权重 $w$ |
|---|---|---|---|---|
| 自动同步消除率 $\\eta_{sync}$ | 编译时统计 | 自动插入同步覆盖算子比例 | $\\tau_{100}=100\\%,\\ \\tau_{80}=80\\%,\\ \\tau_{60}=50\\%$ | $1.0$ |

$$
S_{A,13} = s_{\\eta_{sync}}
$$
''',
    "A14": '''#### A14 自动 tiling 命中率（$W_{A,14}=0.04$）（主亲和 F3）

<a id="F3-A14"></a>

**度量方案**：autotuning 搜索命中统计

##### η_til 自动 tiling 命中率
<a id="F3-A14-eta_til"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越大越好） | 权重 $w$ |
|---|---|---|---|---|
| 自动 tiling 命中率 $\\eta_{til}$ | autotuning 统计 | 自动搜索 tiling 命中算子比例 | $\\tau_{100}=100\\%,\\ \\tau_{80}=70\\%,\\ \\tau_{60}=40\\%$ | $1.0$ |

$$
S_{A,14} = s_{\\eta_{til}}
$$
''',
    "A15": '''#### A15 自动格式转换覆盖率（$W_{A,15}=0.02$）（主亲和 F2）

<a id="F2-A15"></a>

**度量方案**：编译时 dn2nz/nd2nz/dtype 自动转换覆盖统计

##### η_fmt 自动格式转换覆盖率
<a id="F2-A15-eta_fmt"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越大越好） | 权重 $w$ |
|---|---|---|---|---|
| 自动格式转换覆盖率 $\\eta_{fmt}$ | 编译时统计 | 自动 dn2nz/nd2nz 覆盖算子比例 | $\\tau_{100}=100\\%,\\ \\tau_{80}=70\\%,\\ \\tau_{60}=40\\%$ | $1.0$ |

$$
S_{A,15} = s_{\\eta_{fmt}}
$$
''',
    "A16": '''#### A16 用户实际感知暴露度（$W_{A,16}=0.02$）（主亲和 F4，弱）

<a id="F4-A16"></a>

**度量方案**：用户须显式配置的微架构暴露项计数

##### n_perc 用户须感知暴露项数
<a id="F4-A16-n_perc"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| 用户须感知暴露项数 $n_{perc}$ | API/文档暴露清单 | 用户必须填写的微架构参数个数 | $\\tau_{100}=0,\\ \\tau_{80}=3,\\ \\tau_{60}=8$ | $1.0$ |

$$
S_{A,16} = s_{n_{perc}}
$$
''',
    "B4": '''#### B4 地址计算（$W_{B,4}=0.17$）（主亲和 F6）

<a id="F6-B4"></a>

**度量方案**：多级内存地址偏移代码统计（从 B1 拆出，避免与 tiling 混计）

##### L_addr 地址计算代码行
<a id="F6-B4-L_addr"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| 地址计算代码行 $L_{addr}$ | 源代码统计 | 地址偏移计算代码行 | $\\tau_{100}=0,\\ \\tau_{80}=20,\\ \\tau_{60}=50$ | $1.0$ |

$$
S_{B,4} = s_{L_{addr}}
$$
''',
    "B5": '''#### B5 地址对齐（$W_{B,5}=0.13$）（主亲和 F6）

<a id="F6-B5"></a>

**度量方案**：padding / 对齐边界分支统计（从 B1 拆出）

##### n_pad 对齐边界分支数
<a id="F6-B5-n_pad"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| 对齐边界分支数 $n_{pad}$ | 源代码统计 | padding/边界分支数 | $\\tau_{100}=0,\\ \\tau_{80}=3,\\ \\tau_{60}=8$ | $1.0$ |

$$
S_{B,5} = s_{n_{pad}}
$$
''',
    "B6": '''#### B6 随路搬运（$W_{B,6}=0.13$）（主亲和 F2）

<a id="F2-B6"></a>

**度量方案**：FIXPIPE 随路互斥表 + 格式转换调用统计（从 B1 拆出）

##### n_fix 随路互斥拆分步数
<a id="F2-B6-n_fix"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| 随路互斥拆分步数 $n_{fix}$ | 硬件 FIXPIPE 互斥表 | 互斥随路组合需拆分步数 | $\\tau_{100}=0,\\ \\tau_{80}=2,\\ \\tau_{60}=5$ | $0.6$ |

##### n_fmt 格式转换调用数
<a id="F2-B6-n_fmt"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| 格式转换调用数 $n_{fmt}$ | 源代码统计 | dn2nz/nd2nz 调用次数 | $\\tau_{100}=0,\\ \\tau_{80}=2,\\ \\tau_{60}=5$ | $0.4$ |

$$
S_{B,6} = 0.6 \\cdot s_{n_{fix}} + 0.4 \\cdot s_{n_{fmt}}
$$
''',
    "B7": '''#### B7 计算行为表达（$W_{B,7}=0.04$）（主亲和 F4）

<a id="F4-B7"></a>

**度量方案**：饱和 / 舍入 / 特殊函数显式调用扫描

##### n_beh 显式饱和/舍入/特殊函数调用数
<a id="F4-B7-n_beh"></a>

| 度量指标 | 锚点来源 | 计算方法 | 锚点阈值（越小越好） | 权重 $w$ |
|---|---|---|---|---|
| 显式饱和/舍入/特殊函数调用数 $n_{beh}$ | 源代码统计 | 须手写的数值行为插桩次数 | $\\tau_{100}=0,\\ \\tau_{80}=2,\\ \\tau_{60}=6$ | $1.0$ |

$$
S_{B,7} = s_{n_{beh}}
$$
''',
}


def load_catalog() -> dict:
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))


def item_anchor(cls_id: str, item_id: str) -> str:
    return f"{cls_id}-{item_id}"


def sub_anchor(cls_id: str, item_id: str, sub_id: str) -> str:
    return f"{cls_id}-{item_id}-{sub_id}"


def render_toc(cat: dict) -> str:
    n_items = sum(len(c["items"]) for c in cat["classes"])
    n_subs = sum(len(it["subs"]) for c in cat["classes"] for it in c["items"])
    lines = [
        "## 目录接口",
        "",
        "三级路径：**度量分类** → **度量项** → **度量子项**。",
        "稳定 ID：`#{分类}-{度量项}` / `#{分类}-{度量项}-{子项}`，例如 [`#F1-E1-eta_ov`](#F1-E1-eta_ov)。",
        "机读目录：[catalog.yaml](metrics/catalog.yaml)。",
        "",
        f"| 层级 | 数量 | 说明 |",
        f"|---|---|---|",
        f"| 度量分类 | {len(cat['classes'])} | F1–F6 负载敏感轴 |",
        f"| 度量项 | {n_items} | 可独立评分的能力项 |",
        f"| 度量子项 | {n_subs} | 单一采集指标（原度量指标） |",
        "",
    ]
    for cls in cat["classes"]:
        cid, cname = cls["id"], cls["name"]
        n_i, n_s = len(cls["items"]), sum(len(it["subs"]) for it in cls["items"])
        lines.append(f"- **[{cid} {cname}](#{cid})**（{n_i} 项 / {n_s} 子项）— {cls['axis']}")
        for it in cls["items"]:
            iid, iname = it["id"], it["name"]
            flags = []
            if it.get("redline"):
                flags.append("红线")
            if it.get("weak"):
                flags.append("弱")
            if it.get("expanded"):
                flags.append("补全")
            flag = f" _{','.join(flags)}_" if flags else ""
            lines.append(f"  - [{iid} {iname}](#{item_anchor(cid, iid)}){flag}")
            for sub in it["subs"]:
                sid, sname, sym = sub["id"], sub["name"], sub["symbol"]
                lines.append(
                    f"    - [`{sym}` {sname}](#{sub_anchor(cid, iid, sid)})"
                )
        lines.append("")
    lines.append("> 正文 §3 仍按建模维 A–E 展开；本目录按分类聚合，点击即达对应项/子项锚点。")
    lines.append("")
    return "\n".join(lines)


def class_jump_bar(cat: dict) -> str:
    blocks = []
    for cls in cat["classes"]:
        cid = cls["id"]
        item_links = " · ".join(
            f"[{it['id']}](#{item_anchor(cid, it['id'])})" for it in cls["items"]
        )
        blocks.append(f'<a id="{cid}"></a>\n\n**{cid} {cls["name"]}**：{item_links}')
    return "\n".join(blocks) + "\n"


def inject_item_anchors(body: str, cat: dict) -> str:
    """Insert item/sub anchors and ##### sub headings after each known #### item."""
    item_to_cls = {}
    item_meta = {}
    for cls in cat["classes"]:
        for it in cls["items"]:
            item_to_cls[it["id"]] = cls["id"]
            item_meta[it["id"]] = it

    lines = body.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        matched_id = None
        for iid, prefix in ITEM_HEADING_PREFIX.items():
            if line.startswith(prefix):
                matched_id = iid
                break
        if matched_id is None:
            out.append(line)
            i += 1
            continue

        cls_id = item_to_cls[matched_id]
        it = item_meta[matched_id]
        # heading
        if "（主亲和" not in line:
            line = line.rstrip("\n") + f"（主亲和 {cls_id}）\n"
        out.append(line)
        # skip if anchor already present
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if f'id="{cls_id}-{matched_id}"' not in nxt and f'id="{cls_id}-{matched_id}"' not in line:
            out.append(f'\n<a id="{cls_id}-{matched_id}"></a>\n')
            sub_links = " · ".join(
                f"[`{s['symbol']}`](#{sub_anchor(cls_id, matched_id, s['id'])})"
                for s in it["subs"]
            )
            out.append(f"\n子项：{sub_links}\n")
            for s in it["subs"]:
                out.append(
                    f"\n##### {s['symbol']} {s['name']}\n"
                    f'<a id="{sub_anchor(cls_id, matched_id, s["id"])}"></a>\n'
                )
        i += 1
    return "".join(out)


def insert_expanded_items(body: str) -> str:
    marker = "---\n\n## 4. 三档性能驱动的度量画像"
    if marker not in body:
        raise SystemExit("cannot find §4 marker")
    stubs = "\n".join(EXPANDED_STUBS[k] for k in sorted(EXPANDED_STUBS, key=lambda x: (x[0], int(x[1:]))))
    note = (
        "### 3.6 目录补全项（原「篇幅所限略」）\n\n"
        "> 下列度量项在 §7/目录接口中独立存在，正文原先未展开。按同一模板补全，"
        "并从 B1/A5/A6/A7 拆出与独立项重复的子指标，避免目录双重挂载。\n\n"
    )
    return body.replace(marker, note + stubs + "\n---\n\n## 4. 三档性能驱动的度量画像", 1)


def replace_front(doc: str, toc: str, jump: str) -> str:
    intro = (
        "# 下一代调试调优度量体系（三档性能驱动版）\n\n"
        "> 本文档基于《度量.xlsx》原始度量条目、《下一代调试调优度量方案.md》参考方案与《易用性度量.md》易用性建模，"
        "刷新为**三档性能达成度（60% / 90% / 99%）驱动的客观度量体系**。\n"
        "> 核心原则：① 度量方案客观、可复现；② 三档性能驱动权重；③ 数学公式表达；"
        "④ 度量按负载轴亲和到六类；⑤ 目录接口为 **分类 → 度量项 → 度量子项**。\n\n"
        f"{toc}\n"
        "---\n\n"
        "## 0. 体系总览\n\n"
        f"{jump}\n"
        "### 0.1 三层目录结构\n\n"
        "| 层级 | 符号 | 含义 | 入口 |\n"
        "|---|---|---|---|\n"
        "| **度量分类** | F1–F6 | 负载敏感轴（互联带宽 / 精度 / Batch / FLOPs/token / 容量 / 显存敏感） | [目录接口](#目录接口) |\n"
        "| **度量项** | A1、B1、E1… | 可独立评分的能力项（36 项） | `#{分类}-{项}` |\n"
        "| **度量子项** | $\\eta_{ov}$、$n_{til}$… | 单一采集指标（原「度量指标」） | `#{分类}-{项}-{子项}` |\n\n"
        "度量方案（AST / PMU / μ-bench / 故障注入）是子项上的提取方法，不是目录层级。\n"
        "建模维 A–E 是易用性税种，作为度量项属性保留，不参与目录分层。\n\n"
    )
    # Drop everything from start through the old 0.1 section, keep from 0.2 onward.
    key = "### 0.2 度量维度（对齐易用性度量.md 第二部分）"
    idx = doc.find(key)
    if idx < 0:
        raise SystemExit("cannot find §0.2")
    return intro + doc[idx:]


def main() -> None:
    cat = load_catalog()
    doc = DOC.read_text(encoding="utf-8")
    toc = render_toc(cat)
    jump = class_jump_bar(cat)
    doc = replace_front(doc, toc, jump)
    doc = inject_item_anchors(doc, cat)
    if "### 3.6 目录补全项" not in doc:
        doc = insert_expanded_items(doc)
    DOC.write_text(doc, encoding="utf-8")
    n_items = sum(len(c["items"]) for c in cat["classes"])
    n_subs = sum(len(it["subs"]) for c in cat["classes"] for it in c["items"])
    print(f"wrote {DOC}")
    print(f"classes={len(cat['classes'])} items={n_items} subs={n_subs}")


if __name__ == "__main__":
    main()
