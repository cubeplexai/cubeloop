import aiomysql
import pytest

from cubeloop.checkpointer.mysql import MySQLCheckpointer
from cubeloop.checkpointer.mysql.checkpointer import _parse_dsn
from cubeloop.hitl.types import ApproveRequest, HitlRequest
from tests.checkpointer.test_mysql import _setup_schema as _setup_schema_v2


def _req(qid="tc-1") -> HitlRequest:
    return HitlRequest(
        question_id=qid,
        thread_id="t-1",
        payload=ApproveRequest(tool_name="bash", tool_call_id=qid, args={"cmd": "ls"}),
        created_at=0.0,
        timeout_seconds=30.0,
    )


@pytest.mark.asyncio
async def test_mysql_save_and_load_pending_request(clean_mysql_db) -> None:
    await _setup_schema_v2(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.save_pending_request("t-1", _req())
        loaded = await cp.load_pending_request("t-1")
    assert loaded == _req()


@pytest.mark.asyncio
async def test_mysql_clear_pending_request(clean_mysql_db) -> None:
    await _setup_schema_v2(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.save_pending_request("t-1", _req())
        await cp.save_pending_request("t-1", None)
        loaded = await cp.load_pending_request("t-1")
    assert loaded is None


@pytest.mark.asyncio
async def test_mysql_compare_and_clear_pending_request(clean_mysql_db) -> None:
    await _setup_schema_v2(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.save_pending_request("t-1", _req(qid="q-current"), run_id="run-new")

        assert not await cp.clear_pending_request_if_matches(
            "t-1", question_id="q-replaced", run_id="run-new"
        )
        assert not await cp.clear_pending_request_if_matches(
            "t-1", question_id="q-current", run_id="run-old"
        )
        assert not await cp.clear_pending_request_if_matches(
            "t-1", question_id="q-current", run_id="RUN-NEW"
        )
        assert await cp.load_pending("t-1") == (_req(qid="q-current"), "run-new")

        assert await cp.clear_pending_request_if_matches(
            "t-1", question_id="q-current", run_id="run-new"
        )
        assert await cp.load_pending("t-1") is None


@pytest.mark.asyncio
async def test_mysql_pending_request_creates_thread_row_lazily(clean_mysql_db) -> None:
    """save_pending_request must upsert the thread row so calling it on an unknown
    thread doesn't violate the FK constraint."""
    await _setup_schema_v2(clean_mysql_db)
    async with MySQLCheckpointer(clean_mysql_db) as cp:
        await cp.save_pending_request("brand-new-thread", _req(qid="tc-x"))
        loaded = await cp.load_pending_request("brand-new-thread")
    assert loaded is not None
    assert loaded.question_id == "tc-x"


@pytest.mark.asyncio
async def test_mysql_v2_schema_version_enforced(clean_mysql_db) -> None:
    """If the host's alembic only ran v1 (schema_version=1), MySQLCheckpointer
    __aenter__ raises CubepiSchemaMismatch."""
    from cubeloop.checkpointer.mysql.exceptions import CubeloopSchemaMismatch

    conn = await aiomysql.connect(autocommit=True, **_parse_dsn(clean_mysql_db))
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "CREATE TABLE cubepi_schema_version (version INT PRIMARY KEY) ENGINE=InnoDB"
            )
            await cur.execute("INSERT INTO cubepi_schema_version (version) VALUES (1)")
    finally:
        await conn.ensure_closed()

    with pytest.raises(CubeloopSchemaMismatch):
        async with MySQLCheckpointer(clean_mysql_db):
            pass
