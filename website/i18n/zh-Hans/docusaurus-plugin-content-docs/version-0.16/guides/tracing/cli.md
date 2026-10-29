---
title: Trace CLI
description: "使用 CubeLoop trace CLI 检查、过滤并展示 OpenTelemetry trace span。"
---

# 使用 `cubeloop trace` 检查 trace

`JsonlSpanExporter` 在 `./cubeloop-traces/<date>/<trace_id>.jsonl` 下每条
trace 写入一个文件。`cubeloop trace` CLI（由 `trace-cli` extra 提供）读取这些
文件，让你无需重新运行即可了解一次运行的详细情况——哪些 LLM 和工具调用被
触发、顺序如何、各自返回什么、在哪里出错以及 token 数量。

```bash
pip install 'cubeloop[trace-cli]'      # 或：uv sync --extra trace-cli
cubeloop trace --help
```

`--dir` 默认为 `./cubeloop-traces`；若 trace 文件存放在其他位置，传入
`--dir <path>` 即可。每个文件是一条 **trace**：运行本身加上所有嵌套
subagent 运行（它们继承父级的 `trace_id`，因此写入同一文件）。

## `ls` —— 列出近期 trace

```bash
cubeloop trace ls          # 最新优先；-n N 限制数量
```

| 列 | 含义 |
|---|---|
| `started` | trace 开始时间（UTC） |
| `trace_id` | 传递给 `view` / `follow` / `stats` 的 ID |
| `spans` | 整条 trace 的 span 数量（含 subagent） |
| `status` | `ok` 或 `error` |
| `duration` | trace 的实际耗时 |
| `input` | 用户的 prompt，用于识别此次运行 |

### 按运行元数据过滤（`--meta`）

如果宿主在 trace 上打了 run 级别的元数据（通过 `tracing_context(metadata=…)`
——例如 cubebox 会在根 `invoke_agent` span 上记录 `conversation_id`、
`user_id`、`org_id`、`workspace_id`），可以只过滤出匹配的 trace：

```bash
cubeloop trace ls --meta conversation_id=conv_123
cubeloop trace ls --meta user_id=usr_9 --meta org_id=org_1   # 可重复 = AND，精确匹配
```

每个 `--meta KEY=VALUE` 与 trace 根节点元数据精确匹配；重复该标志表示 AND 关系。

如需将元数据值**显示**为列（而不只是过滤），添加 `--show-meta KEY[,KEY…]`：

```bash
cubeloop trace ls --show-meta conversation_id,user_id
cubeloop trace ls --meta org_id=org_1 --show-meta conversation_id   # 过滤 + 显示
```

（或通过 `cubeloop trace view <id> -v` 查看单条 trace 的所有元数据。）

## `view` —— 将 trace 渲染为 span 树

trace-id 的**前缀**即可（表格会截断 ID）；如果前缀不唯一，会列出候选结果。

```bash
cubeloop trace view 1cd97cdb
```

```
trace
└── invoke_agent  14425.8ms  [0x1cd97cdb]
    ├── cubeloop.turn  1283.1ms  [0x5cfda93e]
    │   ├── chat deepseek-v4-flash  1208.7ms  tok 6845/68  [0x0d130229]
    │   └── execute_tool subagent  9610.2ms  subagent  [0x38bdd10a]
    │       └── invoke_agent  9601.0ms  [0x8094f99b]   ← subagent 运行，嵌套其中
    │           └── cubeloop.turn  9598.4ms  [0x57c5cfc7]
    │               ├── chat deepseek-v4-flash  1190.3ms  [0x8205ca6b]
    │               └── execute_tool web_search  6500.2ms  web_search  [0xca4e59fc]
    └── cubeloop.turn  491.9ms  ERROR  [0xce25f242]
        └── chat deepseek-v4-flash  427.2ms  ERROR  [0x0bff68ec]
            └── error: Error code: 400 - ... `tool_use` ids were found without
                `tool_result` blocks immediately after: call_01_...
```

从上到下阅读：`invoke_agent`（一次运行）→ `cubeloop.turn`（一个 agent 循环
turn）→ `chat <model>`（一次 LLM 调用，含 `tok <input>/<output>`）和
`execute_tool <name>`（一次工具调用）。**subagent** 显示为
`execute_tool subagent`，其自身的 `invoke_agent → cubeloop.turn → …` 直接
嵌套在其下方。每个节点后缀的 `[0x…]` 是该 span 的 `span_id`——在原始
JSONL 中 grep 它即可检查对应的具体 span。错误信息内联显示在失败 span 的下方。

标志：

```bash
cubeloop trace view <id> --content   # 展开 gen_ai prompt / 工具参数 / 结果
cubeloop trace view <id> -v          # 展开所有 span 属性（详细，输出量大）
```

`--content` 要求该运行以 `record_content=True` 方式录制
（参见[内容记录与脱敏](./content-recording)）。

## `follow` —— 实时观察 trace

```bash
cubeloop trace follow <id>           # 轮询 span 完成情况；适合正在进行中的运行
```

## `stats` —— 跨 trace 聚合统计

```bash
cubeloop trace stats --by model                  # 延迟 p50/p95、错误率、token 数量
cubeloop trace stats --by tool --since 2026-05-20
```

`stats` 同样接受 `--meta KEY=VALUE`（语义与 `ls` 相同），仅对匹配的 trace
进行聚合——例如按单个用户或对话统计延迟/错误率/token：

```bash
cubeloop trace stats --by model --meta user_id=usr_9
cubeloop trace stats --by tool --meta conversation_id=conv_123
```

## `convert` —— 重建 API 请求体

当你需要重放某次特定的 LLM 调用——复现故障、测试 prompt 变更，或对同一
上下文执行原始 `curl`——`convert` 读取已录制的 `chat` span 并输出完整的
请求体。

需要 `record_content=True`。

```bash
# 默认：trace 中最后一个 chat span，OpenAI JSON 格式
cubeloop trace convert <trace_id>

# 选择要重建的 LLM 调用
cubeloop trace convert <trace_id> --turn 2        # 第 2 个 chat span（从 1 计数）
cubeloop trace convert <trace_id> --span 0xbb7eb1 # 按 span_id 前缀（来自 `view`）

# 输出格式
cubeloop trace convert <trace_id> --format openai     # 默认——JSON 请求体
cubeloop trace convert <trace_id> --format anthropic  # Anthropic Messages API 请求体
cubeloop trace convert <trace_id> --format curl       # 可直接执行的 curl 命令
```

`view` 输出中的 `[0x…]` span id 可直接传给 `--span`：

```
├── chat kimi-k2.6  31704.5ms  [0xbb7eb192]   ← 粘贴为：--span 0xbb7eb1
├── chat kimi-k2.6  32420.2ms  [0x7c76f48d]   ← 或：--span 0x7c76f4
```

重建后的请求体包含完整对话历史、system prompt、所有工具定义以及请求参数
（`model`、`max_tokens`、`temperature`）。可管道传给 `python -m json.tool`、
`jq` 或直接用于重放脚本。

## CLI 之外

这些文件是纯 JSONL——每行一个 span——因此可以直接解析（`jq`、`python -c`）
以提取特定属性（`gen_ai.usage.*`、`gen_ai.tool.call.result`、
`gen_ai.input.messages`……）。错误详情存放在名为
`gen_ai.client.operation.exception` 的 span **event** 中。

:::tip 为 AI agent 准备
内置的 `cubeloop-trace` skill 可驱动此 CLI 进行调试（"这次运行为何没有回复？"、
"工具结果不对"）。它内置了快速路径（`ls` → `view <prefix>`）以及
token/缓存命中率的约定。

```bash
npx skills add cubeplexai/cubeloop@cubeloop-trace -a claude-code
```
:::
