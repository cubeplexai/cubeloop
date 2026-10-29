---
title: 从 cubepi 迁移
description: "CubePi 在 0.14 更名为 CubeLoop；安装、导入、持久化和 trace 的迁移方法。"
---

# 从 cubepi 迁移

0.14 将项目从 CubePi / `cubepi` 更名为 CubeLoop / `cubeloop`。GitHub 仓库
原地更名为 `cubeplexai/cubeloop`，文档站为 https://cubeloop.dev。

## 安装与导入

```bash
pip install cubeloop
# extras 不变：cubeloop[sqlite]、cubeloop[postgres]、cubeloop[mysql]、…
```

```python
from cubeloop import Agent, tool
from cubeloop.providers.anthropic import AnthropicProvider
```

`cubepi` 0.14.1+ 是不带依赖的占位包。所有 `cubepi` 导入都会直接报错并提示迁移，
不会安装或代理 CubeLoop。请明确替换项目依赖和全部 import。

CLI 请将 `cubepi trace` 替换为 `cubeloop trace`。

## Checkpointer schema 保持 v5

Postgres 和 MySQL 的物理表名、分区名和索引名继续使用 `cubepi_*`。这些名字属于
持久化协议，不是产品品牌。schema version 保持 5，因此已有 0.13.6 数据库升级到
0.14.1 不需要数据库迁移。SQLite 表名从未加品牌前缀，保持不变。

历史 Alembic revision 可以继续从 `cubeloop.checkpointer.*.alembic_helpers`
导入 helper；这些 helper 保持原来的 v1–v5 SQL。不要添加表名 rename revision。

已有 schema v5 数据库不需要表名迁移。如果数据库仍是更早的 schema version，
请先通过宿主 Alembic migration 正常升级到 v5，再启动 0.14.1 checkpointer。

## Tracing

新 span 使用 `cubeloop.*` 属性和 `cubeloop.turn` span 名。`cubeloop trace` 仍能读取
0.13 中使用 `cubepi.*` 的 JSONL；你自己的 dashboard 需要更新属性名。默认 JSONL
目录为 `./cubeloop-traces`。

## 环境变量

`CUBELOOP_TEST_PG_DSN`、`CUBELOOP_TEST_MYSQL_DSN`、`CUBELOOP_PG_DSN`、
`CUBELOOP_MYSQL_DSN` 替换对应的 `CUBEPI_*` 名称。新名称未设置时，0.x 的剩余
周期仍会读取旧名称。
