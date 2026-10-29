---
title: Postgres + FastAPI 服务
description: "使用 PostgresCheckpointer 将 FastAPI 后端的 CubeLoop agent 部署到生产环境。"
---

# Recipe：Postgres + FastAPI 服务

一个生产形态的 HTTP 服务，用于封装 CubeLoop agent：以 FastAPI 做路由，
Server-Sent Events 做流式传输，共享的 `PostgresCheckpointer` 做持久化，
`thread_id` 从已认证用户派生。

**预计耗时：** 30 分钟。
**依赖：** `cubeloop[postgres]`、`fastapi`、`uvicorn[standard]`、
`sse-starlette`、已运行并应用 CubeLoop schema 的 Postgres 实例。

## 先建 Schema

在服务启动之前，运行 CubeLoop schema 迁移。本 recipe 最快捷的方式：

```bash
psql "$DATABASE_URL" <<'SQL'
CREATE TABLE cubepi_threads (
    thread_id TEXT PRIMARY KEY,
    parent_thread_id TEXT REFERENCES cubepi_threads(thread_id),
    forked_at_seq BIGINT,
    extra JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE cubepi_messages (
    thread_id TEXT NOT NULL REFERENCES cubepi_threads(thread_id) ON DELETE CASCADE,
    seq BIGINT NOT NULL,
    role TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (thread_id, seq)
) PARTITION BY HASH (thread_id);

-- 64 个哈希分区
DO $$
BEGIN
  FOR i IN 0..63 LOOP
    EXECUTE format(
      'CREATE TABLE cubepi_messages_p%s PARTITION OF cubepi_messages FOR VALUES WITH (MODULUS 64, REMAINDER %s)',
      i, i
    );
  END LOOP;
END$$;

CREATE INDEX ix_cubepi_messages_metadata_gin
ON cubepi_messages USING gin (metadata jsonb_path_ops);

CREATE TABLE cubepi_schema_version (version INT PRIMARY KEY);
INSERT INTO cubepi_schema_version VALUES (1);
SQL
```

在真实部署中，通过 Alembic 生成此迁移 —— 参见
[Postgres Checkpointing → 通过 Alembic 初始化](../guides/checkpointing/postgres#bootstrapping-via-alembic)。

## 服务代码

```python title="service.py"
import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cubeloop import Agent
from cubeloop.checkpointer import PostgresCheckpointer
from cubeloop.providers.anthropic import AnthropicProvider


# --- 应用生命周期 ----------------------------------------------------------

_provider = AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
_checkpointer: PostgresCheckpointer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _checkpointer
    _checkpointer = await PostgresCheckpointer(
        os.environ["DATABASE_URL"],
        min_pool_size=2,
        max_pool_size=20,
    ).__aenter__()
    yield
    await _checkpointer.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)


# --- 认证（存根 —— 替换为你的真实认证逻辑）--------------------------------

async def current_user_id() -> str:
    # 生产中：解码 JWT、查找 session 等。
    return "demo-user"


# --- 路由 ------------------------------------------------------------------

class PromptBody(BaseModel):
    text: str


@app.post("/chat/{conversation_id}/messages")
async def post_message(
    conversation_id: str,
    body: PromptBody,
    user_id: str = Depends(current_user_id),
):
    thread_id = f"{user_id}:{conversation_id}"

    async def event_generator() -> AsyncIterator[dict]:
        agent = Agent(
            model=_provider.model("claude-sonnet-4-6"),
            system_prompt="You are a helpful assistant.",
            checkpointer=_checkpointer,
            thread_id=thread_id,
        )

        queue: asyncio.Queue = asyncio.Queue()
        agent.subscribe(lambda e, s=None: queue.put_nowait(e))

        async def run():
            try:
                await agent.prompt(body.text)
            finally:
                queue.put_nowait(None)   # sentinel

        task = asyncio.create_task(run())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                # 向客户端发送一小部分事件。
                if event.type == "message_update" and event.stream_event.type == "text_delta":
                    yield {"event": "delta", "data": event.stream_event.delta}
                elif event.type == "tool_execution_start":
                    yield {"event": "tool_start", "data": event.tool_name}
                elif event.type == "agent_end":
                    yield {"event": "done", "data": ""}
        finally:
            await task

    return EventSourceResponse(event_generator())


@app.get("/chat/{conversation_id}/history")
async def get_history(
    conversation_id: str,
    user_id: str = Depends(current_user_id),
):
    thread_id = f"{user_id}:{conversation_id}"
    data = await _checkpointer.load(thread_id)
    if data is None:
        return {"messages": []}
    return {
        "messages": [m.model_dump(mode="json") for m in data.messages],
    }
```

运行：

```bash
pip install "cubeloop[postgres]" fastapi "uvicorn[standard]" sse-starlette
export DATABASE_URL=postgresql://user:pass@localhost/cubeloop
export ANTHROPIC_API_KEY=sk-…
uvicorn service:app --reload --port 8000
```

测试：

```bash
curl -N -X POST http://localhost:8000/chat/conv1/messages \
  -H "content-type: application/json" \
  -d '{"text":"hi"}'
# event: delta
# data: Hello
# event: delta
# data: !
# event: done
```

## 设计说明

- **每个进程一个 `PostgresCheckpointer`，跨请求共享。**
  它持有连接池；每个请求单独打开连接会使连接池失去意义。
- **每个请求一个 `Agent`。** Agent 拥有各自的对话状态
  （steering 队列、监听器）。不要复用。
- **`thread_id = f"{user_id}:{conversation_id}"`** —— 通过前缀隔离用户。
  Agent 只读写自己的 thread。
- **SSE 做流式传输。** 每个文本增量作为单独事件发给客户端。
  工具启动有专属事件类型 —— 客户端无需重建事件处理逻辑即可渲染
  "思考中"指示器。
- **无需负载均衡亲和性。** 状态在 Postgres 中，任何服务实例都能
  接管任意对话。

## 同一 thread 的并发问题

如果用户双击发送，两个 `POST` 请求会同时到达。两者都创建绑定到同一
`thread_id` 的 `Agent`。Postgres 咨询锁会串行化它们的追加写入，但
**内存中**的状态会分叉 —— 第二个请求的 agent 可能看不到第一个请求
正在进行的消息（`agent.state.messages`）。

对大多数聊天 UI 来说这没问题（客户端控制发送时机）。如果需要严格
排序，可以添加应用层互斥锁（按 `thread_id` 键控的 `asyncio.Lock`）
或队列。

## 生产加固检查清单

- **认证：** 将 `current_user_id()` 替换为真实的 JWT / session 验证。
- **限速：** 在 agent 构造函数中添加
  [`RateLimitMiddleware`](../guides/middleware/examples#rate-limiting)，
  以 `user_id` 为键。
- **成本跟踪：** 订阅 `agent_end`，对每个 `AssistantMessage` 的
  `usage` 求和，写入计费表。
- **可观测性：** 使用 `on_response` 捕获 `anthropic-*` 限速响应头，
  导出到 Prometheus。
- **备份：** Postgres 原生方式 —— `pg_dump`、时间点恢复。
- **优雅关闭：** uvicorn 的 lifespan handler 会关闭连接池；如有其他
  资源，添加 `signal.signal(SIGTERM, ...)`。

## 常见陷阱

- **启动时 CubeloopSchemaUninitialized** —— 迁移未运行。请先应用 schema。
- **连接池耗尽** —— 默认 `max_pool_size=10`。如果服务的并发 agent
  数量超过此值，请调大。
- **SSE 在负载均衡器后面** —— 某些负载均衡器会缓冲 SSE。禁用缓冲
  （nginx 用 `X-Accel-Buffering: no`）。
- **长请求超时** —— 重工具 agent 可能运行数分钟。
  请设置宽松的代理超时并将 uvicorn 的 `--timeout-keep-alive` 设为 600。

## 另请参见

- [Postgres Checkpointing](../guides/checkpointing/postgres) —— 后端深度说明。
- [持久化聊天](./persistent-chat) —— 使用 SQLite 的相同流程。
- [多 Provider 故障转移](./multi-provider-failover) —— 与本服务结合以提升弹性。

## 运行示例

仓库中有一份完整可运行的代码，位于
[`examples/postgres_fastapi.py`](https://github.com/cubeplexai/cubeloop/blob/main/examples/postgres_fastapi.py)。

```bash
git clone https://github.com/cubeplexai/cubeloop && cd cubeloop
uv sync --extra postgres

export DATABASE_URL=postgresql://user:pass@localhost/cubeloop
export ANTHROPIC_API_KEY=sk-ant-...   # 或 OPENAI_API_KEY [+ OPENAI_BASE_URL]

uv run --with fastapi --with "uvicorn[standard]" --with sse-starlette \
  uvicorn examples.postgres_fastapi:app --reload --port 8000

# 用 curl 测试：
curl -N -X POST http://localhost:8000/chat/conv1/messages \
  -H "content-type: application/json" \
  -d '{"text":"hi"}'
```
