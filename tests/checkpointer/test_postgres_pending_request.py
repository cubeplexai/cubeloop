import asyncpg
import pytest

from cubeloop.checkpointer.postgres import PostgresCheckpointer
from cubeloop.hitl.types import ApproveRequest, HitlRequest
from tests.checkpointer.test_postgres import _setup_schema as _setup_schema_v2


def _req(qid="tc-1") -> HitlRequest:
    return HitlRequest(
        question_id=qid,
        thread_id="t-1",
        payload=ApproveRequest(tool_name="bash", tool_call_id=qid, args={"cmd": "ls"}),
        created_at=0.0,
        timeout_seconds=30.0,
    )


@pytest.mark.asyncio
async def test_postgres_save_and_load_pending_request(clean_db) -> None:
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("t-1", _req())
        loaded = await cp.load_pending_request("t-1")
    assert loaded == _req()


@pytest.mark.asyncio
async def test_postgres_clear_pending_request(clean_db) -> None:
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("t-1", _req())
        await cp.save_pending_request("t-1", None)
        loaded = await cp.load_pending_request("t-1")
    assert loaded is None


@pytest.mark.asyncio
async def test_postgres_compare_and_clear_pending_request(clean_db) -> None:
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("t-1", _req(qid="q-current"), run_id="run-1")

        assert not await cp.clear_pending_request_if_matches(
            "t-1", question_id="q-replaced", run_id="run-1"
        )
        assert (await cp.load_pending_request("t-1")) == _req(qid="q-current")

        assert await cp.clear_pending_request_if_matches(
            "t-1", question_id="q-current", run_id="run-1"
        )
        assert await cp.load_pending_request("t-1") is None
        assert await cp.load_pending_run_id("t-1") is None


@pytest.mark.asyncio
async def test_postgres_compare_and_clear_preserves_replacement(clean_db) -> None:
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("t-1", _req(qid="q-answered"), run_id="run-1")
        await cp.save_pending_request("t-1", _req(qid="q-follow-up"), run_id="run-1")

        assert not await cp.clear_pending_request_if_matches(
            "t-1", question_id="q-answered", run_id="run-1"
        )
        assert (await cp.load_pending_request("t-1")) == _req(qid="q-follow-up")
        assert await cp.load_pending_run_id("t-1") == "run-1"


@pytest.mark.asyncio
async def test_postgres_compare_and_clear_requires_matching_run(clean_db) -> None:
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("t-1", _req(qid="q-same"), run_id="run-new")

        assert not await cp.clear_pending_request_if_matches(
            "t-1", question_id="q-same", run_id="run-old"
        )
        assert (await cp.load_pending_request("t-1")) == _req(qid="q-same")
        assert await cp.load_pending_run_id("t-1") == "run-new"


@pytest.mark.asyncio
async def test_postgres_pending_request_creates_thread_row_lazily(clean_db) -> None:
    """save_pending_request must INSERT … ON CONFLICT DO NOTHING for the thread row,
    so calling it on an unknown thread doesn't FK-violate."""
    await _setup_schema_v2(clean_db)
    async with PostgresCheckpointer(clean_db) as cp:
        await cp.save_pending_request("brand-new-thread", _req(qid="tc-x"))
        loaded = await cp.load_pending_request("brand-new-thread")
    assert loaded is not None
    assert loaded.question_id == "tc-x"


@pytest.mark.asyncio
async def test_postgres_v2_schema_version_enforced(clean_db) -> None:
    """If the host's alembic only ran v1 (no pending_request column, schema_version=1),
    PostgresCheckpointer().__aenter__ raises CubepiSchemaMismatch."""
    from cubeloop.checkpointer.postgres.exceptions import CubeloopSchemaMismatch

    conn = await asyncpg.connect(clean_db)
    try:
        await conn.execute(
            "CREATE TABLE cubepi_schema_version (version INTEGER PRIMARY KEY);"
        )
        await conn.execute("INSERT INTO cubepi_schema_version (version) VALUES (1);")
    finally:
        await conn.close()
    with pytest.raises(CubeloopSchemaMismatch):
        async with PostgresCheckpointer(clean_db):
            pass
