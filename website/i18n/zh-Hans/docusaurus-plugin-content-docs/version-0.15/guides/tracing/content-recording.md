---
title: 内容记录与脱敏
description: "在 CubeLoop 的 OpenTelemetry tracing 中配置 prompt 内容记录与脱敏。"
sidebar_position: 4
---

# 记录 Prompt、响应与工具 Payload

CubeLoop 的 tracing 默认只发出结构性属性——操作名称、模型、token 数量、
结束原因、耗时。**不会有 prompt 内容、模型输出、工具参数或结果离开进程。**
这是有意为之：很多 agent 场景处理 PII、客户数据或商业机密 prompt，
这些内容不应发送到第三方可观测性后端。

当你确实需要捕获内容——用于离线评测、调试偶发的工具调用问题，或构建标注
数据集——可以通过 `record_content=True` 显式开启，并配合 `redact` 回调在
数据离开进程之前剥除敏感部分。

## 开启内容记录

```python
tracer = Tracer(
    service_name="my-bot",
    agent_name="assistant",
    record_content=True,            # ← 显式开启
    exporters=[JsonlSpanExporter(directory="./cubeloop-traces")],
)
```

开启 `record_content=True` 后，每层 span 按照 OTel GenAI semconv 携带相应
内容属性：

| Span | 新增内容属性 |
|---|---|
| `invoke_agent` | `gen_ai.system_instructions`、`gen_ai.input.messages`、`gen_ai.output.messages` |
| `cubeloop.turn` | `gen_ai.input.messages`（per-turn 切片）、`gen_ai.output.messages`（per-turn 切片） |
| `chat <model>` | `gen_ai.system_instructions`、`gen_ai.input.messages`、`gen_ai.tool.definitions`、`cubeloop.llm.raw_request`、`cubeloop.llm.raw_response` |
| `execute_tool <tool_name>` | `gen_ai.tool.call.arguments`、`gen_ai.tool.call.result` |

`chat` span 的 `gen_ai.input.messages` 包含 provider 请求实际携带的
**完整时序上下文**——包括之前的 assistant turn 和工具结果，而不只是新的
用户 prompt。对于多轮工具调用运行，trace 消费方可以精确还原模型在每次
调用时所看到的内容。

## 导出前脱敏

`redact` 是一个 `(key, value) -> value | None` 回调，在每个内容属性的设置
处被调用。返回值：

- 原始值不变 → 保留原样
- 相同形状的修改后的值 → 替换原值
- `None` → 完全丢弃该属性

```python
def redact(key: str, value):
    # 在 prompt 离开进程之前剥除其中的密钥。
    if key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        return _scrub_messages(value)
    # 在生产环境中不发送原始请求/响应体——只保留规范化后的结构。
    if key in ("cubeloop.llm.raw_request", "cubeloop.llm.raw_response"):
        return None
    return value


tracer = Tracer(
    service_name="my-bot",
    record_content=True,
    redact=redact,
    exporters=[…],
)
```

`redact` 是内容的唯一拦截点——recorder 在将属性序列化为 OTel 属性之前，
每个属性只调用一次，函数返回什么就发送什么。`redact` 内部抛出的异常会被
吞掉（此时该属性被丢弃），因此有 bug 的 redactor 会以"不发送"方式失败，
而不是泄露数据。

### 常用模式

丢弃所有内容，仅保留 per-message 长度，让 dashboard 仍可正常工作：

```python
def redact(key, value):
    if key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        return [{"role": m["role"], "parts": [{"type": "text", "content": "<redacted>",
                                               "length": sum(len(p.get("content", "")) for p in m["parts"])}]}
                for m in value]
    return value
```

基于标签的选择性记录——除非某个 thread 已开启，否则全部过滤：

```python
import contextvars
RECORD = contextvars.ContextVar("trace.record_content", default=False)

def redact(key, value):
    return value if RECORD.get() else None
```

然后对需要捕获的运行调用 `RECORD.set(True)`。

## 大小预算

OTel 属性值在 recorder 内部以 JSON 序列化。大多数后端会截断或拒绝超过
几百 KB 的属性。如果每个 chat span 都记录原始 provider 响应，多轮工具调用
的 agent 运行会很快产生大量数据。对超出预算的字段，请通过 `redact` 丢弃
或做摘要处理。

## 流级别记录

`record_content` 捕获每个 `chat` span 上最终组装好的请求和响应，但不捕获
逐块的流式 chunk。如需对流式失败做事后调试——空工具调用参数、重复事件、
输出被截断——可开启 `record_stream`，在主 trace 旁边写入逐块事件日志：

```python
tracer = Tracer(
    record_content=True,            # trace convert 所需
    record_stream=True,             # ← 逐块事件日志
    stream_dir="./cubeloop-traces",   # <run_id>.stream.jsonl 的写入目录
    exporters=[JsonlSpanExporter(directory="./cubeloop-traces")],
)
```

`record_stream` 写入 `<stream_dir>/<run_id>.stream.jsonl`（每个
`StreamEvent` 一行 JSON）。每行携带 `t`（距运行开始的秒数）和 `type`。
工具调用相关行还包含 `ci`（content index）、`id`、`name`、delta 大小和
参数预览：

```json
{"t": 5.873, "type": "toolcall_start", "ci": 1, "id": "toolu_...", "name": "show_widget"}
{"t": 5.875, "type": "toolcall_delta", "ci": 1, "chars": 11, "accumulated": 11, "preview": "{\"title\": \""}
{"t": 33.177, "type": "toolcall_end",  "ci": 1, "id": "toolu_...", "args_chars": 7465, "args_preview": "{\"title\": \"CubeLoop..."}
```

通过这些数据，可以方便地确认参数 chunk 是否按预期到达，或者同一事件是否
触发了两次（例如，某个 provider 发送了两次 `finish_reason`，就会产生两条
相同 `ci` 的 `toolcall_end` 行）。

`record_stream` 与 `record_content` 相互独立——仅在调试会话中开启。
对于长时间运行、工具调用密集的 agent，文件可能会变得很大。

## 审计已记录的内容

recorder 始终在每个 span 上设置 `service.name`、`gen_ai.agent.name` 和
`cubeloop.run_id`——无论 `record_content` 是否开启。使用这些属性在 trace
后端过滤到单次运行，并直观确认哪些数据已落地。

如需深度审计，`JsonlSpanExporter` 每行写入一个 span，因此可以在将同一
exporter 指向远程后端之前，先对本地文件进行 grep / `jq` 检查：

```bash
jq -r 'select(.attributes["gen_ai.input.messages"]) | .attributes["gen_ai.input.messages"]' \
   cubeloop-traces/2026-05-19/*.jsonl
```
