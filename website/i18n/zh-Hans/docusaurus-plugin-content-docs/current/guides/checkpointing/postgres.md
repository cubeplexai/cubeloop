---
title: Postgres 检查点
description: "使用 PostgresCheckpointer 实现生产级的 agent 状态持久化。"
---

# Postgres 检查点

`PostgresCheckpointer` 是生产级持久化后端。它使用 `asyncpg` 搭配连接池、
`msgpack` 编码 payload、以及每线程的 Postgres advisory lock，让多个进程
可以安全地写入同一个 `thread_id` 而不会互相干扰。

安装 extra：

```bash
pip install "cubeloop[postgres]"
```

会拉入 `asyncpg`、`sqlalchemy` 和 `msgpack`。

## 基本用法

```python
import asyncio
from cubeloop import Agent
from cubeloop.checkpointer import PostgresCheckpointer
from cubeloop.providers.anthropic import AnthropicProvider


async def main():
    provider = AnthropicProvider(provider_id="anthropic", api_key="…")
    async with PostgresCheckpointer("postgresql://user:pass@host/dbname") as cp:
        agent = Agent(
            model=provider.model("claude-sonnet-4-6"),
            checkpointer=cp,
            thread_id="user-42",
        )
        await agent.prompt("hello")


asyncio.run(main())
```

DSN 接受 `asyncpg.create_pool(...)` 支持的任何格式。连接池大小：

```python
async with PostgresCheckpointer(
    "postgresql://…",
    min_pool_size=2,
    max_pool_size=20,
) as cp:
    …
```

## Schema

checkpointer 需要这些表：`cubepi_threads`、`cubepi_messages`、
`cubepi_runs`、`cubepi_hitl_answers` 和 `cubepi_schema_version`。
与 SQLite 不同，CubeLoop **不会自动创建这些表**——
它只在 `__aenter__` 时验证它们是否存在且 `schema_version` 是否匹配预期。

如果表不存在，你会收到 `CubeloopSchemaUninitialized`。如果版本与本版
CubeLoop 不匹配，你会收到 `CubeloopSchemaMismatch`。

原因：生产数据库属于宿主应用的迁移系统（Alembic、Atlas……），
不属于一个可能与你现有迁移冲突的第三方库。

### 通过 Alembic 引导 {#bootstrapping-via-alembic}

CubeLoop 暴露了 SQLAlchemy `MetaData`，让你的迁移可以采用其 schema：

```python
# alembic/env.py
from cubeloop.checkpointer.postgres import cubeloop_metadata, EXPECTED_SCHEMA_VERSION

target_metadata = [my_app_metadata, cubeloop_metadata]
```

然后生成一个 revision 并执行。迁移还必须写入 `cubepi_schema_version`。
使用辅助函数：

```python
# 在迁移的 upgrade() 中：
from cubeloop.checkpointer.postgres.alembic_helpers import (
    create_message_partitions_op,
    create_runs_partitions_op,
    write_schema_version_op,
)

def upgrade():
    op.create_table(...)                            # 从 cubeloop_metadata 自动生成
    op.execute(create_message_partitions_op())      # cubepi_messages 的 64 个哈希分区
    op.execute(create_runs_partitions_op())         # cubepi_runs 的 64 个哈希分区
    op.execute(write_schema_version_op())           # 记录历史 schema v5
```

两个辅助函数都返回 SQL 字符串——你需要传入 `op.execute(...)`。
`write_schema_version_op()` 是不可变的历史 v5 writer：它会删除其他版本行并写入
5，可以重复执行。未来 schema version 必须新增并调用对应版本的 writer，不能复用
这个 helper。

## 数据模型

```
cubepi_threads
    thread_id (PK)
    parent_thread_id   -- 用于 fork
    forked_at_seq      -- fork 点处的序列号
    extra              -- JSONB
    created_at / updated_at

cubepi_messages
    thread_id, seq     -- 复合 PK；按 HASH(thread_id) 分为 64 个分区
    role               -- "user" | "assistant" | "tool"
    metadata           -- JSONB（通过 GIN 索引）
    payload            -- bytea (msgpack)
    created_at

cubepi_runs
    thread_id, run_id  -- 复合 PK
    claimed_at / completed_at
    completion_seq

cubepi_hitl_answers
    thread_id, run_id, question_id -- 复合 PK
    answer                         -- JSONB
    answered_at

cubepi_schema_version
    version (PK)
```

重要属性：

- **`(thread_id, seq)` 是消息标识。** `seq` 在每线程中单调递增，
  在 `pg_advisory_xact_lock(hashtext(thread_id))` 下分配。
  两个对同一线程的并发写入者会干净地序列化。
- **`payload` 是 msgpack 编码的 `model.model_dump(mode="json")`。**
  CubeLoop 在读取时重建 Pydantic 模型。
- **`metadata` 是 JSONB，可查询。** 完整消息的 payload 内部也包含
  `metadata`，但这一列是 SQL 查询的规范视图。
- **表按 `HASH(thread_id)` 分为 64 个分区。** 跨分区均匀分布，
  无每线程瓶颈。

## 并发

advisory lock 让同一线程上的追加操作跨进程安全：

```python
# 进程 A 和进程 B 同时追加到线程 "user-42"。
# 它们通过 pg_advisory_xact_lock 序列化，各自获得连续的 seq。
```

读取（`load`）不取锁——它们在事务内是快照一致性的。

默认连接池 `min=1, max=10` 对大多数应用足够；如果你的并发 agent 数
很高，请调大 `max_pool_size`。

## `save_extra` 语义

`save_extra` 做的是 JSONB 合并，而不是替换：

```sql
extra = cubepi_threads.extra || EXCLUDED.extra
```

所以先写 `{"foo": 1}` 再写 `{"bar": 2}` 会得到 `{"foo": 1, "bar": 2}`。
中间件可以安全地写入部分 dict 而不会丢失先前的键。

## 条件式清理 pending request

跨 worker 协调 HITL 恢复的宿主可以调用
`clear_pending_request_if_matches(thread_id, question_id=..., run_id=...)`。
只有数据库中当前 pending request 的问题 ID 和所属 run ID 都仍然匹配时，
它才会清除这两个字段。比较与清理在一条 PostgreSQL 更新中完成，因此即使
另一个 worker 的新 run 复用了相同问题 ID，旧 worker 也不会误删新请求。

## Fork

`parent_thread_id` + `forked_at_seq` 列用于支持
[会话 Fork](../agents/forking)：fork 会创建一个新的 thread，并在其
`cubepi_threads` 行中写入指向源 thread 的 `parent_thread_id`，
以及拷贝时源 thread 末尾的 `forked_at_seq`（最后一条已拷贝消息的
`seq`）。

## Schema v3 → v4 migration {#schema-v3--v4-migration}

Fork 功能将 `EXPECTED_SCHEMA_VERSION` 从 3 升到 4。升级会向
`cubepi_messages` 添加 `run_id` 列 + 索引，并创建分区父表
`cubepi_runs` 及其子分区。使用 alembic helper：

```python
# 在迁移的 upgrade() 中：
from cubeloop.checkpointer.postgres.alembic_helpers import (
    upgrade_v3_to_v4_op,
    write_schema_version_op,
)

def upgrade():
    op.execute(upgrade_v3_to_v4_op())
    op.execute(write_schema_version_op())  # 将 cubepi_schema_version 升到 4
```

`upgrade_v3_to_v4_op()` 在重复执行下是幂等的（所有 DDL 都带
`IF NOT EXISTS`）。

升级前的旧消息保留 `run_id = NULL`，仍然可读；
关于混合数据的 fork 资格规则，请参阅
[旧数据行为](../agents/forking#legacy-data-behaviour)。

## CubeLoop 更名

包名更改不会改变持久化 schema：物理名称继续使用 `cubepi_*`，`EXPECTED_SCHEMA_VERSION` 保持 5。已有 0.13.6 数据库无需迁移。详见[迁移指南](../../migration/from-cubepi)。

## 常见坑

- **`CubeloopSchemaUninitialized`** —— 数据库为空或迁移未运行。
  先执行宿主 alembic upgrade。
- **`CubeloopSchemaMismatch`** —— 你升级了 CubeLoop 但未生成新的迁移。
  生成一个、执行它，CubeLoop 就会启动。

  :::info Schema v2（HITL）

  CubeLoop ≥ HITL 版本将 `EXPECTED_SCHEMA_VERSION` 从 1 提升到 2，
  并在 `cubepi_threads` 表中新增 `pending_request JSONB NULL` 列。
  宿主 alembic upgrade 必须在提升 schema_version 行之前调用
  `add_pending_request_column_op()`（来自
  `cubeloop.checkpointer.postgres.alembic_helpers`）。
  完整的跨进程流程请参阅 [HITL 指南](../hitl/overview)。
  :::
- **负载下连接池耗尽** —— 默认 `max_pool_size=10`。
  如果应用的并发 agent 数超过此值，请调大。
- **`asyncpg.exceptions.UndefinedTableError` 在 `__aenter__` 外部** ——
  表示你在 `async with` 之外使用了 checkpointer。连接池尚未连接。
  请用 context manager 包裹。
- **混用宿主 SQLAlchemy `MetaData`** —— CubeLoop 自带独立的
  `MetaData` 实例，正是为了能与你的应用模型共存而不冲突。
  不要将它们合并到你的全局 metadata 中——分别传给 Alembic。

## 另请参阅

- [SQLite 检查点](./sqlite) —— 单进程替代方案。
- [自定义后端](./custom) —— Protocol 详情。
- [配方 → Postgres + FastAPI 服务](../../recipes/postgres-fastapi)
  —— 一个可部署的 HTTP 前端 agent。
