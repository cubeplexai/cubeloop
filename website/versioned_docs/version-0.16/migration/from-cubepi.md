---
title: From cubepi
description: "CubePi was renamed to CubeLoop in 0.14. How to update installs, imports, checkpointer schema, and traces."
---

# Migrating from cubepi

0.14 renames the project from CubePi / `cubepi` to CubeLoop / `cubeloop`.
GitHub keeps the same repository (renamed in place to `cubeplexai/cubeloop`).
The docs site is https://cubeloop.dev.

## Install and imports

```bash
pip install cubeloop
# extras are unchanged: cubeloop[sqlite], cubeloop[postgres], cubeloop[mysql], …
```

```python
from cubeloop import Agent, tool
from cubeloop.providers.anthropic import AnthropicProvider
```

`cubepi` 0.14.1+ is a tombstone package: it has no CubeLoop dependency and all
`cubepi` imports fail with migration instructions. Replace the dependency and
all imports explicitly; there is no compatibility proxy.

CLI: replace `cubepi trace` with `cubeloop trace`.

## Checkpointer schema stays at v5

Postgres and MySQL keep their `cubepi_*` physical table and index names. They
are persistent protocol identifiers, not product branding. Schema version 5 is
unchanged, so an existing 0.13.6 database needs no migration for 0.14.1.
SQLite table names were never prefixed; they stay `messages` / `runs` / ….

Historical host Alembic revisions may import helpers from
`cubeloop.checkpointer.*.alembic_helpers`; those helpers retain their original
v1–v5 SQL. Do not add a rename revision.

An existing schema-v5 database needs no rename migration. If the database is on
an earlier schema version, run its normal host Alembic migrations through v5
before opening the 0.14.1 checkpointer.

## Tracing

New spans use `cubeloop.*` attributes and `cubeloop.turn` span names.
`cubeloop trace` still reads 0.13 JSONL that used `cubepi.*`.
Dashboards you own need their own attribute-name update.

Default JSONL directory is `./cubeloop-traces`.

## Environment variables

`CUBELOOP_TEST_PG_DSN`, `CUBELOOP_TEST_MYSQL_DSN`, `CUBELOOP_PG_DSN`,
`CUBELOOP_MYSQL_DSN` replace the `CUBEPI_*` names. The old names are still
read if the new one is unset, for the rest of 0.x.
