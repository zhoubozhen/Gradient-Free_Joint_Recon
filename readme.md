# fista_tranPACT

基于 TranPACT + FISTA-TV 的梯度自由联合重建包。

## 1. 参数说明总表

下面参数同时适用于 `config.json` 和 `cluster_config.json`。
 若某个参数只在本地或 cluster 场景下更常用，会在备注中说明。

------

### 1.1 `binding` 参数

| 参数                   | 类型   | 含义                        | 当前示例        | 备注                   |
| ---------------------- | ------ | --------------------------- | --------------- | ---------------------- |
| `binding.main_gpu_idx` | int    | 主计算进程使用的 GPU 编号   | `6`             | 本地运行时常手动指定   |
| `binding.prox_gpu_idx` | int    | prox worker 使用的 GPU 编号 | `7`             | 本地运行时常手动指定   |
| `binding`              | object | GPU 绑定配置对象            | `{}` 或具体索引 | cluster 配置里通常留空 |

说明：

- 本地运行时，主 GPU 和 prox GPU推荐在a9:0-2或者3-7；在ein推荐0-3或者4-7；即不要在a9:主卡0副卡3，不要在ein：主卡1副卡5.
- cluster 运行时一般由调度系统分配 GPU，因此 `cluster_config.json` 中这一项常为空对象。

------

### 1.2 `fista` 参数

| 参数                 | 类型   | 含义                          | 当前示例 | 备注                                              |
| -------------------- | ------ | ----------------------------- | -------- | ------------------------------------------------- |
| `fista.reg`          | float  | TV 正则项权重                 | `0.0001` | 越大正则越强，结果更平滑，常改                    |
| `fista.lip`          | float  | Lipschitz 常数 / 步长相关参数 | `5.0`    | FISTA 稳定性关键参数之一，常改                    |
| `fista.iter`         | int    | FISTA 最大迭代轮数            | `20`     | 每次外层调用时的内层迭代次数                      |
| `fista.prox_mode`    | int    | prox 调用模式                 | `2`      | 1:main单卡模式；2：双卡模式main+prox各一张        |
| `fista.prox_impl`    | string | prox 实现类型                 | `"mix"`  | mix:cpu+GPU混合实现,推荐；cupy：GPU实现，可能报错 |
| `fista.prox_iter`    | int    | prox 内部迭代次数             | `50`     | TV proximal 子问题迭代数，一般不需要改            |
| `fista.grad_min`     | float  | 梯度停止阈值                  | `1e-05`  | 可用于提前停止，一般不需要改                      |
| `fista.cost_min`     | float  | 代价函数停止阈值              | `0.001`  | 可用于提前停止，一般不需要改                      |
| `fista.save_freq`    | int    | 中间结果保存频率              | `1`      | 每几轮保存一次，可改                              |
| `fista.use_check`    | bool   | 是否启用收敛/发散检查         | `true`   | 开启后使用下方阈值规则，一般建议开启              |
| `fista.check_iter`   | int    | 检查频率                      | `1`      | 每若干轮检查一次                                  |
| `fista.rel_thr`      | float  | 相对变化收敛阈值              | `0.01`   | 连续小变化时可判定收敛，常改                      |
| `fista.rel_patience` | int    | 收敛判定耐心轮数              | `2`      | 连续满足多少次才停止，可改                        |
| `fista.rel_warmup`   | int    | 收敛检测预热轮数              | `2`      | 前几轮不做收敛判定，一般不改                      |
| `fista.div_rel_thr`  | float  | 发散相对变化阈值              | `0.01`   | 超过该阈值可视为异常增长，可改                    |
| `fista.div_patience` | int    | 发散判定耐心轮数              | `2`      | 连续异常多少次才判发散，可改                      |
| `fista.div_warmup`   | int    | 发散检测预热轮数              | `2`      | 前几轮不做发散判定，一般不改                      |

说明：

- 这是当前最常改的一组参数。

------

### 1.3 `fista.runtime` 参数

| 参数                          | 类型   | 含义                 | 当前示例                                      | 备注                                                     |
| ----------------------------- | ------ | -------------------- | --------------------------------------------- | -------------------------------------------------------- |
| `fista.runtime.worker_script` | string | prox worker 脚本路径 | `"${NEW_V2_ROOT}/my_code/run_prox_worker.py"` | local / cluster 初始化后可能会被改写成不同形式，不建议改 |

说明：

- 该参数指定 prox worker 的启动脚本。
- 在模板里通常写成占位形式；初始化 workdir 后：
  - local 场景可能会被改成 `my_code/run_prox_worker.py`
  - cluster 场景可能会被改成包内绝对路径
- 这是运行时依赖项，不建议手动随意改名。

------

### 1.4 顶层主流程参数

| 参数        | 类型   | 含义                       | 当前示例      | 备注                                                 |
| ----------- | ------ | -------------------------- | ------------- | ---------------------------------------------------- |
| `maxfun`    | int    | 外层优化最大函数评估次数   | `60`          | 控制 GFJR 外层搜索/评估预算，可改                    |
| `stride`    | float  | 空间下采样步长             | `1.0`         | `1.0` 表示不下采样；增大降计算量比如5.0              |
| `start`     | float  | 初始参数或初始模型缩放因子 | `1.05`        | 用于外层初始化，常改                                 |
| `pressure`  | string | 压力数据标签/数据类型      | `"nhp_3_nsp"` | 由数据加载逻辑解释                                   |
| `recon_opt` | int    | 重建流程选项               | `0`           | 0:"homo1layer",1:"3layer",2:"aubry",常改；1可能有bug |
| `ind`       | int    | 样本索引 / case 索引       | `3`           | 常用于选择具体数据                                   |
| `skullp0`   | int    | 是否启用 skull p0 相关流程 | `0`           | 0:skull区域p0置零, 1: 不使用skull_roi                    |

说明：

- 这一组位于 `fista` 后面，属于主流程控制参数。

------

### 1.5 `paths` 参数

| 参数                  | 类型   | 含义               | 当前示例                         | 备注                                 |
| --------------------- | ------ | ------------------ | -------------------------------- | ------------------------------------ |
| `paths.nhp_directory` | string | NHP 参考数据目录   | `.../KH250319_headphantom/data/` | 读取参考/辅助数据时使用              |
| `paths.directory`     | string | 当前实验主数据目录 | `.../KH250727_JRwithAubry/`      | 主输入数据路径                       |
| `paths.saving_root`   | string | 输出保存根目录     | `.../case_study`                 | 重建结果、日志、迭代输出等保存在这里 |

说明：

- 这一组是最容易因换机器、换数据而需要修改的参数。
- 这里配置的是**数据路径**和**结果输出路径**，不是包路径。
- 新建实验时，通常第一时间检查这一组是否正确。

------

### 1.6 其他较少改动的顶层参数

| 参数          | 类型  | 含义                   | 当前示例  | 备注                                     |
| ------------- | ----- | ---------------------- | --------- | ---------------------------------------- |
| `onlycor`     | bool  | 是否只执行校正相关步骤 | `false`   | 只使用cor，默认关闭                      |
| `noise`       | int   | 噪声级别控制参数       | `5`       | 与 `noise_scale` 联合使用                |
| `noise_scale` | float | 噪声幅度缩放           | `0.00026` | 基本，实际噪声幅度 = noise * noise_scale |

说明：

- 这一组位于 `paths` 之后。相对不常改，通常只在特定实验或调试时调整。

------

### 1.7 `runtime` 参数

| 参数                           | 类型 | 含义                    | 当前示例 | 备注                                |
| ------------------------------ | ---- | ----------------------- | -------- | ----------------------------------- |
| `runtime.debug_small`          | bool | 是否启用小规模调试模式  | `true`   | 常用于快速试跑，当stride=1等于false |
| `runtime.debug_only_one_fista` | bool | 是否只跑一次 FISTA 调试 | `false`  | 快速排查用                          |
| `runtime.debug_nt`             | int  | 调试时使用的时间步数    | `4800`   | 缩小时间维规模                      |
| `runtime.trace_full`           | bool | 是否开启完整 trace/log  | `true`   | 调试时很有用                        |
| `runtime.heartbeat_sec`        | int  | 心跳日志间隔（秒）      | `60`     | 避免长时间无输出                    |
| `runtime.log_gpu`              | bool | 是否记录 GPU 状态       | `true`   | 便于检查 GPU 绑定和使用情况         |

说明：

- 这一组主要控制运行时调试、日志和可观测性。不常改。

------

### 1.8 `physics` 参数

| 参数                  | 类型        | 含义                   | 当前示例 | 备注                    |
| --------------------- | ----------- | ---------------------- | -------- | ----------------------- |
| `physics.cb`          | float       | 背景声速或相关物理参数 | `1.5`    | 单位由代码约定          |
| `physics.f0`          | float       | 中心频率               | `1.0`    | 用于波场/信号模型       |
| `physics.ppw`         | int         | points per wavelength  | `4`      | 影响离散精度与稳定性    |
| `physics.fs`          | int / float | 采样频率               | `30`     | 时间采样设置            |
| `physics.space_order` | int         | Devito 空间离散阶数    | `10`     | 越高通常越耗算力        |
| `physics.to`          | int / float | 时间阶或相关传播参数   | `2`      | 具体解释由主程序定义    |
| `physics.nbl`         | int         | 吸收边界层厚度         | `16`     | Devito/PML 相关常用参数 |

说明：

- 这一组控制物理模型与数值离散。不常改。

------

### 1.9 末尾其他参数

| 参数               | 类型   | 含义             | 当前示例 | 备注                     |
| ------------------ | ------ | ---------------- | -------- | ------------------------ |
| `devito_log_level` | string | Devito 日志级别  | `"INFO"` | 常见有 `INFO`、`WARNING` |
| `out_print`        | int    | 输出信息详细程度 | `3`      | 数值越大通常日志越详细   |

说明：

- 这两个参数现在位于 JSON 末尾。一般属于输出与日志控制，不是常规调参重点。

## 2. 包结构

```
fista_tranPACT/
├── README.md
├── my_code/
│   ├── main.py
│   ├── config.json
│   ├── cluster_config.json
│   ├── run_local.sh
│   ├── run_prox_worker.py
│   ├── init_local_workdir.sh
│   ├── init_cluster_workdir.sh
│   ├── cluster_run.sh
│   └── cluster.sub
└── src/
    ├── tranPACT/
    ├── fista_tv_3d_python/
    └── gfjr_utils.py
```

设计逻辑：

- `src/`：核心工具库，作为唯一真源。
- `my_code/`：运行入口、模板脚本、配置文件。
- 本地/集群工作目录只复制 `my_code/*`。
- `src/` 不复制到每个 workdir，而是统一直接引用包内版本。

------

## 3. 使用方式

### 3.1 本地运行

先在一个新的工作目录中初始化：

```
mkdir -p /path/to/your_workdir
cd /path/to/your_workdir
init_local
```

然后运行：

```
cd my_code
bash run_local.sh
```

本地日志默认写到：

```
../logs/YYYYMMDD_HHMM.log
```

------

### 3.2 集群运行

先初始化一个新的 cluster workdir：

```
mkdir -p /path/to/your_cluster_workdir
cd /path/to/your_cluster_workdir
init_cluster
```

提交任务：

```
cd my_code
condor_submit cluster.sub
```

集群日志通常写到：

```
../logs/
../logs/condor/
```

------

## 4. 整体逻辑

这套包分成三层：

### 4.1 工具层

位于：

```
src/
```

包含：

- `tranPACT`
- `fista_tv_3d_python`
- `gfjr_utils.py`

这部分是所有 workdir 共享的公共工具库。
 如果 TranPACT / FISTA 的核心实现已经稳定，后续一般不需要频繁改这里。

------

### 4.2 模板层

位于：

```
my_code/
```

包含：

- `main.py`
- `config.json`
- `cluster_config.json`
- 初始化脚本
- 运行脚本
- worker 脚本

这部分决定“未来新初始化出来的 workdir 默认长什么样”。

------

### 4.3 工作目录层

每次执行初始化脚本后，会在当前 workdir 下生成一套 `my_code/*` 副本。

这意味着：

- 修改 `workdir/my_code/main.py` 只影响当前实验目录
- 修改 `~/my_packages/fista_tranPACT/my_code/main.py` 会影响未来新初始化的目录
- 修改 `~/my_packages/fista_tranPACT/src/*` 会影响所有 workdir

------

## 5. 推荐维护方式

### 5.1 基础工具尽量稳定

如果 `tranPACT`、`FISTA-TV`、`gfjr_utils` 已经稳定，建议尽量少改：

```
~/my_packages/fista_tranPACT/src/
```

因为这部分一旦改动，会影响所有工作目录。

------

### 5.2 实验逻辑改 workdir 里的 `main.py`

如果只是当前实验要调整逻辑、加日志、改调参流程，优先改：

```
你的workdir/my_code/main.py
你的workdir/my_code/config.json
```

这样不会影响其他目录，也不会污染模板。

------

### 5.3 模板升级改包内 `my_code`

如果你确认某个改动以后所有新工作目录都需要继承，就改：

```
~/my_packages/fista_tranPACT/my_code/
```

然后重新初始化新的 workdir。

------

## 6. `config.json` 与 `cluster_config.json`

这两个配置目前结构基本一致。

常见区别是：

- `config.json`：本地运行时可指定固定 GPU
- `cluster_config.json`：通常留空绑定，由集群环境决定 GPU 分配

## 7. `config.json` 与 `cluster_config.json` 的实际建议

### 本地运行建议

优先改：

- `binding.main_gpu_idx`
- `binding.prox_gpu_idx`
- `paths.*`
- `runtime.debug_*`
- `fista.*`

适合快速调试、确认逻辑、单机测试。

------

### 集群运行建议

优先改：

- `paths.*`
- `runtime.*`
- `physics.*`
- `fista.*`

通常不建议在 `cluster_config.json` 里写死 GPU 编号，除非集群环境就是固定卡位。

------

## 8. 常见工作流建议

### 8.1 新实验

1. 新建 workdir
2. 用 `init_local_workdir.sh` 或 `init_cluster_workdir.sh` 初始化
3. 修改该 workdir 下的 `my_code/config.json`
4. 必要时修改该 workdir 下的 `my_code/main.py`
5. 运行

------

### 8.2 只改当前实验

只改：

```
workdir/my_code/main.py
workdir/my_code/config.json
```

不动包内模板。

------

### 8.3 更新未来默认模板

改：

```
~/my_packages/fista_tranPACT/my_code/*
```

这样以后新 init 的 workdir 都继承这些改动。

------

### 8.4 修改基础算法/工具

改：

```
~/my_packages/fista_tranPACT/src/*
```

这会影响所有 workdir，需谨慎。

------

## 9. 当前版本的设计原则

当前版本已经稳定在以下原则上：

- `src/` 是唯一工具库
- `my_code/` 是模板和入口集合
- 本地/集群 workdir 只复制 `my_code/*`
- workdir 可以单独修改 `main.py`
- 基础工具与实验逻辑分层维护

这套结构的目标是：

1. 工具统一维护，不复制、不漂移
2. 实验目录可独立改逻辑，不互相污染
3. 本地和集群都能通过同一套模板快速初始化
4. 调试和长期维护都尽量清晰

