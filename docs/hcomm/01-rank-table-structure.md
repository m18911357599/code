# HCOMM Rank Table 结构与 DSL

> 基于 [cann/hcomm](https://gitcode.com/cann/hcomm) 源码与 CANN 文档整理。  
> 主要定义位置：
> - `src/coll_communicator_mgr/rank_graph/rank_table_info/rank_table_info.{h,cc}`
> - `src/coll_communicator_mgr/rank_graph/rank_table_info/new_rank_info.{h,cc}`
> - `src/coll_communicator_mgr/rank_graph/rank_table_info/rank_level_info.{h,cc}`
> - `src/coll_communicator_mgr/rank_graph/rank_table_info/address_info.{h,cc}`
> - `src/coll_communicator_mgr/rank_graph/rank_table_info/control_plane.{h,cc}`
> - `src/coll_communicator_mgr/rank_graph/rank_graph_builder/rank_graph_builder.cc`
> - 样例：`examples/01_communicators/02_one_device_per_process_rank_table/`

配套工具：[`tools/ranktable_dsl/`](../../tools/ranktable_dsl/README.md)。

---

## 1. 设计定位

Rank table 是 HCCL/HCOMM 的 **集群成员描述文件**（JSON，上限 1GB）。通信域初始化接口 `HcclCommInitClusterInfo` / `HcclCommInitClusterInfoConfig` 读取该文件，得到每个 rank 的设备、地址与分层网络归属，再与本机 **物理拓扑文件**（`TopoInfo`，默认 `/usr/local/Ascend/driver/topo/`）一起建成 `RankGraph`。

仓库里并存两套 on-disk 方言，内部统一落到 `RankTableInfo`：

| 方言 | `version` | 组织轴 | 使用场景 |
|------|-----------|--------|----------|
| **V1** | `1.0` / `1.2` | Server → Device | A2/A3/训练系列；样例 `rank_table.json` |
| **V2** | `2.0` | Rank → 网络层次 | Ascend 950 / Atlas 350；`RankTableInfo::Deserialize` 只接受此版本 |

V2 是当前 hcomm 解析器的 **IR**。`RankTableInfo::Check()` 强制 `version == "2.0"`。V1 JSON 由旧入口或产品侧适配后，再进入同一套 rank graph 构建。

```
ranktable.json ──► JsonParser ──► RankTableInfo
                                      │
topo/*.json    ──► PhyTopoBuilder ──► PhyTopo
                                      │
                                      ▼
                              RankGraphBuilder
                                      │
                                      ▼
                                  RankGraph
```

---

## 2. 内部 IR：`RankTableInfo`

对应 JSON 根对象。字段与校验均来自 `Deserialize` + `Check`。

```
RankTableInfo
├── version          string     必须 "2.0"
├── rankCount        u32        [1, 65536]，且 == ranks.size()
├── detour           bool       JSON "detour": "true"|"false"|省略
└── ranks[]          NewRankInfo
    ├── rankId       u32        全局唯一，连续覆盖 [0, rankCount)
    ├── localId      u32        [0, 64]；64 = BACKUP_LOCAL_ID
    ├── replacedLocalId u32     localId==64 时必填；否则 = localId
    ├── deviceId     u32        [0, 64]
    ├── devicePort   u32        [1, 65535]，默认 16666
    ├── hostPort     u32        [1, 65535]，默认 16666
    ├── controlPlane ControlPlane   可选
    └── rankLevelInfos[]  RankLevelInfo   长度 ≤ 8，netLayer 严格递增
        ├── netLayer     u32        [0, 7]
        ├── netInstId    string     长度 [1, 1024]
        ├── netType      NetType
        ├── netAttr      string     可选，默认 ""
        └── rankAddrs[]  AddressInfo     长度 ≤ 24
            ├── addrType     EID | IPV4 | IPV6
            ├── addr         IpAddress    文本长度 [1, 256]
            ├── backupAddrs  IpAddress[]  可选，≤ 16，类型同主地址
            ├── ports        set<string>  个数 [1, 16]，每项长度 [1, 32]
            ├── planeId      string       默认 "0"，长度 ≤ 1024
            └── socketPort_  u32          运行时填 devicePort，不入 JSON
```

`status` 出现在文档与样例中（`completed` / `initializing`），`RankTableInfo::Deserialize` **不读取**该字段。DSL 保留它，仅作文件可用性标记。

同一 `net_layer` 下相同 `net_instance_id` 的 `rank_addr_list` 长度必须一致（`InsertToRank` / `CheckAndInsert`）。全表最多一个 `local_id == 64` 的替换 rank，且其 `replaced_local_id` 不得与其他 rank 的 `local_id` 冲突。

---

## 3. V2 文件内容（JSON ↔ 字段）

根对象：

| JSON 键 | 类型 | 必选 | 对应成员 | 约束 |
|---------|------|------|----------|------|
| `version` | string | 是 | `version` | `"2.0"` |
| `status` | string | 文档必选 / 解析忽略 | — | `completed` / `initializing` |
| `rank_count` | uint | 是 | `rankCount` | `[1, 65536]`，等于 `rank_list` 长度 |
| `detour` | string | 否 | `detour` | `"true"` / `"false"` / 缺省=false |
| `rank_list` | array | 是 | `ranks` | 每个元素一个 rank |

`rank_list[]`：

| JSON 键 | 类型 | 必选 | 对应成员 | 约束 |
|---------|------|------|----------|------|
| `rank_id` | uint | 是 | `rankId` | `[0, rank_count)`，不重复、连续 |
| `local_id` | uint | 是 | `localId` | `[0, 64]`；本 Server 内 NPU 编号 |
| `replaced_local_id` | uint | `local_id==64` 时必选 | `replacedLocalId` | `[0, 63]` |
| `device_id` | uint | 是 | `deviceId` | `[0, 64]` |
| `device_port` | uint | 否 | `devicePort` | 默认 16666 |
| `host_port` | uint | 否 | `hostPort` | 默认 16666 |
| `level_list` | array | 是 | `rankLevelInfos` | 长度 ≤ 8，`net_layer` 升序 |
| `control_plane` | object | 否 | `controlPlane` | 控制面监听地址 |

`level_list[]`：

| JSON 键 | 类型 | 必选 | 对应成员 | 约束 |
|---------|------|------|----------|------|
| `net_layer` | uint | 是 | `netLayer` | `[0, 7]` |
| `net_instance_id` | string | 是 | `netInstId` | 同层唯一逻辑实例，长度 ≤ 1024 |
| `net_type` | string | 否 | `netType` | 缺省 `CLOS` |
| `net_attr` | string | 否 | `netAttr` | 预留 |
| `rank_addr_list` | array | 是 | `rankAddrs` | 长度 ≤ 24，每 Die 一条 |

`net_type` 枚举（`RankLevelInfo::strToNetType`）：

| JSON | 内部 `NetType` | 含义 |
|------|----------------|------|
| `CLOS` | `CLOS` | 全互通（胖树 / 交换机） |
| `TOPO_FILE_DESC` | `TOPO_FILE_DESC` | 边关系由物理拓扑文件给出 |
| `1DMESH` | `MESH_1D` | Device 直连 |
| `2DMESH` | `MESH_2D` | 二维 mesh |
| `A3_SERVER` | `A3_SERVER` | A3 Server 内拓扑 |
| `A2_AX_SERVER` | `A2_AX_SERVER` | A2 Server 内拓扑 |

文档约定：`net_layer == 0` 时 `net_type` 应为 `TOPO_FILE_DESC`；非 0 层常用 `CLOS` 或 `TOPO_FILE_DESC`。`RankGraphBuilder` 实际只实例化这两种：`InnerNetInstance` / `ClosNetInstance`。

`rank_addr_list[]`：

| JSON 键 | 类型 | 必选 | 对应成员 | 约束 |
|---------|------|------|----------|------|
| `addr_type` | string | 是 | `addrType` | `EID` / `IPV4` / `IPV6`（大小写按源码） |
| `addr` | string | 是 | `addr` | 长度 [1, 256]，格式匹配 `addr_type` |
| `ports` | string[] | 是 | `ports` | `"DieId/PortId"`，如 `"1/0"`；1~16 个 |
| `plane_id` | string | 否 | `planeId` | 默认 `"0"` |
| `backup_addr` | string[] | 否 | `backupAddrs` | ≤ 16，类型同主地址；**不支持 EID** |

`control_plane`：

| JSON 键 | 类型 | 必选 | 对应成员 |
|---------|------|------|----------|
| `addr_type` | string | 是 | `addrType` |
| `addr` | string | 是 | `addr` |
| `listen_port` | uint | 是 | `listenPort` | `[1, 65536]` |

样例（hcomm `rank_table_v2.json` 语义）：

```json
{
  "version": "2.0",
  "rank_count": 2,
  "status": "completed",
  "rank_list": [
    {
      "rank_id": 0,
      "device_id": 0,
      "local_id": 0,
      "level_list": [
        {
          "net_layer": 0,
          "net_instance_id": "az0-rack0",
          "net_type": "TOPO_FILE_DESC",
          "net_attr": "",
          "rank_addr_list": [
            { "addr_type": "IPV4", "addr": "223.0.0.10", "ports": ["0/0"] }
          ]
        }
      ]
    }
  ]
}
```

---

## 4. V1 文件内容（Server 轴）

旧产品与样例 `rank_table.json` 使用 Server/Device 轴。数值在官方文档和样例里多为 **字符串**。

根对象：

| JSON 键 | 说明 |
|---------|------|
| `status` | `completed` 才可用 |
| `version` | 典型组网 `1.0`；超节点 `1.2` |
| `server_count` | AI Server 个数（常为字符串） |
| `server_list` | Server 数组 |
| `super_pod_list` | 仅 `1.2`：超节点分组 |

`server_list[]`：

| JSON 键 | 说明 |
|---------|------|
| `server_id` | 全局唯一，长度 ≤ 64 |
| `host_ip` | Host IPv4；重执行场景建议填 |
| `device` | Device 数组（键名是 `device`，不是 `devices`） |

`device[]`：

| JSON 键 | 说明 |
|---------|------|
| `device_id` | Server 内物理 ID |
| `device_ip` | 集成网卡 IP；多机必填 |
| `device_port` | Device 监听端口 |
| `host_port` | Host 监听端口 |
| `rank_id` | 全局 rank，建议按物理邻接排序；**不同 Server 的 rank_id 区间不可交叉** |
| `super_device_id` | 超节点内 NPU ID（`1.2`） |
| `backup_device_ip` / `backup_device_port` | 借轨备用网卡 |

`super_pod_list[]`：`super_pod_id` + 内嵌 `server_list[{server_id}]`。同一物理超节点的 Server 须连续编排，禁止交叉。

另有已不推荐的 **group/instance** 模板（`group_list` / `instance_list` / `pod_name` / `devices`），仅兼容旧容器编排。

---

## 5. 配套物理拓扑文件（`TopoInfo`）

V2 rank table 的 `ports` / `local_id` 要与本机 topo 文件对齐。解析器：`src/coll_communicator_mgr/rank_graph/topo_info/`。

| JSON 键 | 约束 |
|---------|------|
| `version` | `"2.0"` |
| `peer_count` | `[1, 65]`，等于 `peer_list` 长度 |
| `peer_list[].local_id` | `[0, 64]`，不重复 |
| `edge_count` | 等于 `edge_list` 长度（去重后回写） |
| `edge_list[]` | `net_layer`、`link_type`（`PEER2PEER`/`PEER2NET`）、`protocols`、`local_a`/`local_b`、端口列表、`topo_type`、`position` |

`RankGraphBuilder`：layer 0 用 topo 的 PEER2PEER 边 + rank table 的 `port→addr` 映射建直连；高层用 `net_instance_id` 聚合成 `NetInstance`，CLOS 层再按 `plane_id` 挂 Fabric。

---

## 6. DSL

JSON 把层次摊成键值，阅读成本高。DSL 按 IR 树写成块结构，**一份文本对应一份 rank table 文件**。

### 6.1 词法

- 注释：`#` 或 `//` 到行尾
- 标识符 / 枚举：裸词（`completed`、`IPV4`、`TOPO_FILE_DESC`）
- 整数：十进制
- 字符串：`"..."`；IP / `Die/Port` 也可裸写（`192.168.1.8`、`1/0`）
- 端口列表：`[1/0, 0/4]`

### 6.2 文法（摘要）

```
file        := 'ranktable' version '{' stmt* '}'
stmt        := status | detour | rank | server | super_pod

rank        := 'rank' int '{' rank_body* '}'
rank_body   := 'local' int
             | 'device' int
             | 'replaced_local' int
             | 'device_port' int
             | 'host_port' int
             | level
             | control_plane

level       := 'level' int net_instance_id net_type [attr]? '{' addr* '}'
addr        := 'addr' addr_type address 'ports' port_list ['plane' id] ['backup' str_list]

server      := 'server' id ['host' ip] '{' device* '}'
device      := 'device' int { 'ip' | 'port' | 'host_port' | 'rank' | 'sdid' | 'backup_ip' | 'backup_port' }*

super_pod   := 'super_pod' id '{' ('server' id)* '}'
```

`rank_count` / `server_count` 由列表长度推导，不手写。

### 6.3 与 JSON 的对应

V2 单层（对应 `rank_table_v2.json`）：

```
ranktable 2.0 {
  status completed

  rank 0 {
    local 0
    device 0
    level 0 "az0-rack0" TOPO_FILE_DESC {
      addr IPV4 223.0.0.10 ports [0/0]
    }
  }

  rank 1 {
    local 1
    device 1
    level 0 "az0-rack0" TOPO_FILE_DESC {
      addr IPV4 223.0.0.28 ports [0/1]
    }
  }
}
```

V1 单机 8 卡（对应 `rank_table.json`）：

```
ranktable 1.0 {
  status completed

  server "SERVER_ID_SV1" {
    device 0 ip 192.168.1.8  rank 0
    device 1 ip 192.168.1.9  rank 1
    device 2 ip 192.168.1.10 rank 2
    device 3 ip 192.168.1.11 rank 3
    device 4 ip 192.168.1.12 rank 4
    device 5 ip 192.168.1.13 rank 5
    device 6 ip 192.168.1.14 rank 6
    device 7 ip 192.168.1.15 rank 7
  }
}
```

V2 两层 + 借轨 + 控制面：

```
ranktable 2.0 {
  status completed

  rank 0 {
    local 0
    device 0
    device_port 16666
    host_port 16665

    level 0 "az0-rack0-pod0" TOPO_FILE_DESC {
      addr IPV4 172.16.0.10 ports [1/0]
    }
    level 1 "az0" CLOS {
      addr IPV4 172.16.0.15 ports [0/4, 0/5, 0/6, 0/7] plane plane0
      addr IPV4 172.16.0.5  ports [1/5, 1/6]           plane plane1
                     backup [172.16.0.6]
    }

    control_plane IPV4 172.16.0.100 listen 16666
  }
}
```

编译器按 `version` 选择发射方言：`2.0` → `rank_list` 数值字段；`1.x` → `server_list` 字符串字段（与官方样例一致）。

---

## 7. 校验规则（对齐 `RankTableInfo::Check`）

1. `version` 对 V2 IR 必须是 `2.0`
2. `1 ≤ rank_count ≤ 65536` 且等于 rank 条数
3. `rank_id` 互不重复，恰好填满 `[0, rank_count)`
4. `local_id == 64` 至多一条；其 `replaced_local_id` 不得与其他 `local_id` 同时出现
5. 非 backup rank 的 `replaced_local_id` 必须等于 `local_id`
6. 每个 rank 的 `level_list.net_layer` 严格递增
7. 同一 `(net_layer, net_instance_id)` 上各 rank 的 `rank_addr_list` 长度相同
8. 地址 / 端口 / 枚举范围见第 2 节
