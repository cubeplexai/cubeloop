---
title: 可恢复长任务
description: "使用 CubeLoop checkpointing 和恢复机制构建崩溃容错的长时间运行任务。"
---

# Recipe：可恢复长任务

当 agent 正在执行一个长时间运行的操作（一系列工具调用、多轮推理会话）
时，如果进程崩溃，你希望能从中断处继续，而不是从头开始。
CubeLoop 的追加写入 checkpointing 加上 `agent.resume()` 让**轮次之间**
的恢复变得轻而易举；而**工具执行中途**的恢复则需要多一些准备。

**预计耗时：** 15 分钟。
**依赖：** `cubeloop[sqlite]`、`ANTHROPIC_API_KEY`。

## 模式概述

有三种崩溃点需要考虑：

1. **轮次之间** —— 模型已回复，无工具可运行，循环在迭代间等待。
   `resume()` 会重新调用模型。*使用 checkpointer 即免费获得。*
2. **工具结果之后、模型调用之前** —— 工具结果已持久化。`resume()`
   看到最后一条消息是 `ToolResultMessage`，重新调用模型。
   *使用 checkpointer 即免费获得。*
3. **工具执行中途** —— 工具已启动但未完成。尚未持久化任何内容
   （CubeLoop 只持久化消息）。需要工具内部的幂等性。*需要额外处理。*

本 recipe 重点关注第 3 种情况。

## 带外部状态的幂等工具

模式：每个工具操作都有一个确定性的幂等键。在执行工作之前，先检查
是否已经完成过。

```python title="tools.py"
import os
import json
from pathlib import Path

from cubeloop import AgentToolResult, TextContent, tool


# 简单的文件支撑 job store；生产中替换为 Redis / Postgres。
JOB_DIR = Path(os.environ.get("JOB_DIR", "/tmp/cubeloop-jobs"))
JOB_DIR.mkdir(parents=True, exist_ok=True)


# execution_mode="sequential" → 一次只转码一个。只声明了 `signal`,
# 所以只有它这一个由循环提供的参数会被注入。
@tool(execution_mode="sequential")
async def transcode_video(source_path: str, output_path: str, *, signal=None) -> AgentToolResult:
    "Transcode a video file. Idempotent — safe to retry."
    job_key = f"transcode:{source_path}->{output_path}"
    job_file = JOB_DIR / f"{job_key.replace('/', '_')}.json"

    if job_file.exists():
        # 上次运行已完成。
        state = json.loads(job_file.read_text())
        return AgentToolResult(
            content=[TextContent(text=f"Already transcoded to {state['output_path']}.")],
            details=state,
        )

    # 执行实际工作（长时间运行，代价高昂）。
    # 使用 signal 在取消时干净地中止。
    output = await run_ffmpeg(source_path, output_path, signal=signal)

    # 在工作成功后再写入 job 完成标记。
    job_file.write_text(json.dumps({"output_path": output}))

    return AgentToolResult(
        content=[TextContent(text=f"Transcoded to {output}.")],
        details={"output_path": output},
    )
```

现在，如果进程在 `run_ffmpeg` 执行期间崩溃，下次 agent 运行时会发现
`job_file.exists() == False`，重新执行工作，并且只在成功后写入标记。
如果进程在标记**已写入后**崩溃，下次运行会发现标记，立即返回缓存结果，
agent 继续运行，就好像刚刚完成了一样。

## 恢复 agent

```python title="resume.py"
import asyncio
import os
import sys

from cubeloop import Agent
from cubeloop.checkpointer import SQLiteCheckpointer
from cubeloop.providers.anthropic import AnthropicProvider

from tools import transcode_video   # 上面用 @tool 装饰的 AgentTool


async def main(thread_id: str, initial_prompt: str | None):
    async with SQLiteCheckpointer("jobs.db") as cp:
        agent = Agent(
            model=AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"]).model("claude-sonnet-4-6"),
            system_prompt="You orchestrate video transcoding jobs.",
            tools=[transcode_video],
            checkpointer=cp,
            thread_id=thread_id,
        )
        agent.subscribe(lambda e, s=None: None)

        if initial_prompt:
            # 全新运行。prompt() 在第一次调用前自动加载历史记录，
            # 然后追加新的用户消息。
            await agent.prompt(initial_prompt)
        else:
            # 恢复。agent.resume() 不会自动加载 —— 只有 prompt() 会。
            # 需要先手动恢复 agent 状态。
            data = await cp.load(thread_id)
            if data is None:
                raise RuntimeError(f"No saved state for thread {thread_id!r}")
            agent.state.messages = list(data.messages)
            # `extra` 也会被恢复；它在 Agent 上是私有的，如果你的
            # middleware 需要读取，请使用 checkpointer 的视图。

            # resume 从最后一条持久化消息继续：
            #   ToolResultMessage / UserMessage → 重新调用模型
            #   没有排队 steer/follow_up 的 AssistantMessage → 抛出异常
            last = agent.state.messages[-1]
            if type(last).__name__ == "AssistantMessage":
                from cubeloop.providers.base import ToolCall
                has_pending_tools = any(isinstance(c, ToolCall) for c in last.content)
                if not has_pending_tools:
                    # 任务已正常完成，无待执行的工具调用。
                    # 实际应用中可在此处提示用户输入下一条消息。
                    print("上次运行已完成，无需恢复。")
                    return
                # 存在未执行的工具调用 —— 交由 resume() 继续执行。
            await agent.resume()


if __name__ == "__main__":
    thread_id = sys.argv[1]
    initial = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(main(thread_id, initial))
```

工作流：

```bash
# 启动一个 job：
python resume.py job-1 "Transcode /videos/a.mov to /out/a.mp4 and /videos/b.mov to /out/b.mp4"

# 中途终止：Ctrl-C。

# 恢复 —— agent 从最后一条持久化消息继续：
python resume.py job-1
```

## 三种恢复场景的代码

```python
async def smart_resume(agent, cp, thread_id):
    # resume() 不会自动加载 —— 如果 agent 状态为空，先手动恢复。
    if not agent.state.messages:
        data = await cp.load(thread_id)
        if data is None or not data.messages:
            return False           # 没有可恢复的内容
        agent.state.messages = list(data.messages)

    last = agent.state.messages[-1]
    last_role = type(last).__name__

    if last_role == "AssistantMessage":
        # 要么自然结束，要么在一轮结束后立即崩溃。
        # 除非有排队的 steering，否则 resume() 会抛出。
        # 最简单的方案：询问用户下一步做什么。
        return False

    # 最后一条是 UserMessage 或 ToolResultMessage —— 可以安全恢复。
    await agent.resume()
    return True
```

## 持久化与中止

`agent.abort()` 触发干净的拆卸并发射 `agent_end`。最后一条**完全持久化**
的消息是通过 `message_end` 提交的那条。工具执行期间的中止不会持久化
工具结果（工具未返回），因此 `resume()` 会使用包含未完成 `ToolCall`
的最后一条 `AssistantMessage` 重新调度模型。模型通常会重新发出调用 ——
你的幂等性保护机制会处理剩下的一切。

## 关于持久化部分工具状态

CubeLoop 不提供"持久化部分工具结果"的 API。预期的模式是：将部分状态
保存在工具自己的外部存储中（文件系统、Redis、S3），以工具参数为键
进行确定性寻址。上面的 `transcode_video` 用 `JOB_DIR` 实现的就是这种模式。

## 常见陷阱

- **非幂等工具** —— 没有确定性键，重试可能导致信用卡被扣两次或发送
  重复邮件。始终用幂等键包裹外部副作用。
- **job 标记放在 `/tmp`** —— 重启时会被清除。生产 job 请使用真正的
  持久化层。
- **在无队列的 assistant 消息后调用 `resume()`** —— 会抛出异常。
  要么向用户询问下一条消息，要么重新调用 `prompt()`。
- **在全新 agent 上调用 `resume()`** —— 会抛出 `No messages to continue
  from`。`resume()` 不会从 checkpointer 自动加载；只有 `prompt()` 会。
  请先手动恢复：`agent.state.messages =
  (await cp.load(thread_id)).messages`。
- **忘记在工具内部检查 signal** —— 一个长时间运行的
  `await asyncio.sleep(...)` 或忽略 `signal.is_set()` 的 `for ... in stream`
  不会响应 `abort`。请在任何热循环内部添加检查。

## 另请参见

- [多轮对话 → `resume()`](../guides/agents/multi-turn#resume--continue-from-the-last-message)
  —— 完整语义。
- [持久化聊天](./persistent-chat) —— 更简单的可重启场景。
- [SQLite Checkpointing](../guides/checkpointing/sqlite) —— 持久化内容与时机。

## 运行示例

仓库中有一份完整可运行的代码，位于
[`examples/resumable_tasks.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/resumable_tasks.py)。

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync --extra sqlite

export ANTHROPIC_API_KEY=sk-ant-...   # 或 OPENAI_API_KEY [+ OPENAI_BASE_URL]

# 启动任务（可用 Ctrl-C 中途终止以测试恢复能力）：
uv run python examples/resumable_tasks.py job-1 start

# 从中断处恢复 —— 已完成的条目会被跳过：
uv run python examples/resumable_tasks.py job-1
```
