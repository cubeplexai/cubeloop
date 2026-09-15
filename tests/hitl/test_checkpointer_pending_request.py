import tempfile
import pytest

from cubeloop.checkpointer.memory import MemoryCheckpointer
from cubeloop.checkpointer.sqlite import SQLiteCheckpointer
from cubeloop.hitl.types import ApproveRequest, HitlRequest


def _req(thread_id="t-1", qid="q-1") -> HitlRequest:
    return HitlRequest(
        question_id=qid,
        thread_id=thread_id,
        payload=ApproveRequest(tool_name="bash", tool_call_id=qid, args={"cmd": "ls"}),
        created_at=0.0,
        timeout_seconds=30.0,
    )


@pytest.fixture
async def sqlite_cp():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        async with SQLiteCheckpointer(f.name) as cp:
            yield cp


async def test_memory_save_and_load_pending():
    cp = MemoryCheckpointer()
    assert await cp.load_pending_request("t-1") is None
    req = _req()
    await cp.save_pending_request("t-1", req)
    loaded = await cp.load_pending_request("t-1")
    assert loaded == req


async def test_memory_clear_pending():
    cp = MemoryCheckpointer()
    await cp.save_pending_request("t-1", _req())
    await cp.save_pending_request("t-1", None)
    assert await cp.load_pending_request("t-1") is None


async def test_memory_compare_and_clear_pending():
    cp = MemoryCheckpointer()
    await cp.save_pending_request("t-1", _req(qid="q-current"), run_id="run-new")

    assert not await cp.clear_pending_request_if_matches(
        "t-1", question_id="q-replaced", run_id="run-new"
    )
    assert not await cp.clear_pending_request_if_matches(
        "t-1", question_id="q-current", run_id="run-old"
    )
    assert await cp.load_pending("t-1") == (_req(qid="q-current"), "run-new")

    assert await cp.clear_pending_request_if_matches(
        "t-1", question_id="q-current", run_id="run-new"
    )
    assert await cp.load_pending("t-1") is None


async def test_sqlite_save_and_load_pending(sqlite_cp):
    assert await sqlite_cp.load_pending_request("t-1") is None
    req = _req()
    await sqlite_cp.save_pending_request("t-1", req)
    loaded = await sqlite_cp.load_pending_request("t-1")
    assert loaded == req


async def test_sqlite_clear_pending(sqlite_cp):
    await sqlite_cp.save_pending_request("t-1", _req())
    await sqlite_cp.save_pending_request("t-1", None)
    assert await sqlite_cp.load_pending_request("t-1") is None


async def test_sqlite_compare_and_clear_pending(sqlite_cp):
    await sqlite_cp.save_pending_request("t-1", _req(qid="q-current"), run_id="run-new")

    assert not await sqlite_cp.clear_pending_request_if_matches(
        "t-1", question_id="q-replaced", run_id="run-new"
    )
    assert not await sqlite_cp.clear_pending_request_if_matches(
        "t-1", question_id="q-current", run_id="run-old"
    )
    assert await sqlite_cp.load_pending("t-1") == (
        _req(qid="q-current"),
        "run-new",
    )

    assert await sqlite_cp.clear_pending_request_if_matches(
        "t-1", question_id="q-current", run_id="run-new"
    )
    assert await sqlite_cp.load_pending("t-1") is None


async def test_sqlite_create_table_idempotent(sqlite_cp):
    """Re-opening a checkpointer DB with existing pending_request table is safe."""
    await sqlite_cp.save_pending_request("t-1", _req())
    # Re-entering the context manager would call CREATE TABLE IF NOT EXISTS again
    # against an existing table — must not raise.
    await sqlite_cp._db.execute(
        "CREATE TABLE IF NOT EXISTS thread_pending_request ("
        "thread_id TEXT PRIMARY KEY, request_json TEXT NOT NULL, "
        "created_at REAL NOT NULL DEFAULT (julianday('now')))"
    )


async def test_memory_hitl_answer_ledger_crud():
    cp = MemoryCheckpointer()
    await cp.save_hitl_answer("t-1", "q-1", {"decision": "approve"}, run_id="r-1")
    await cp.save_hitl_answer("t-1", "q-2", {"decision": "deny"}, run_id="r-1")
    await cp.save_hitl_answer("t-1", "q-1", {"decision": "edit"}, run_id="r-2")

    assert await cp.load_hitl_answer("t-1", "q-1", run_id="r-1") == {
        "decision": "approve"
    }
    assert await cp.load_hitl_answer("t-1", "q-1", run_id="r-2") == {"decision": "edit"}

    await cp.clear_hitl_answers("t-1", ["q-1"], run_id="r-1")
    assert await cp.load_hitl_answer("t-1", "q-1", run_id="r-1") is None
    assert await cp.load_hitl_answer("t-1", "q-2", run_id="r-1") == {"decision": "deny"}
    assert await cp.load_hitl_answer("t-1", "q-1", run_id="r-2") == {"decision": "edit"}

    await cp.clear_hitl_answers("t-1", run_id="r-1")
    assert await cp.load_hitl_answer("t-1", "q-2", run_id="r-1") is None
    assert await cp.load_hitl_answer("t-1", "q-1", run_id="r-2") == {"decision": "edit"}


async def test_sqlite_hitl_answer_ledger_crud(sqlite_cp):
    await sqlite_cp.save_hitl_answer(
        "t-1", "q-1", {"decision": "approve"}, run_id="r-1"
    )
    await sqlite_cp.save_hitl_answer("t-1", "q-2", {"decision": "deny"}, run_id="r-1")
    await sqlite_cp.save_hitl_answer("t-1", "q-1", {"decision": "edit"}, run_id="r-2")

    assert await sqlite_cp.load_hitl_answer("t-1", "q-1", run_id="r-1") == {
        "decision": "approve"
    }
    assert await sqlite_cp.load_hitl_answer("t-1", "q-1", run_id="r-2") == {
        "decision": "edit"
    }

    await sqlite_cp.clear_hitl_answers("t-1", ["q-1"], run_id="r-1")
    assert await sqlite_cp.load_hitl_answer("t-1", "q-1", run_id="r-1") is None
    assert await sqlite_cp.load_hitl_answer("t-1", "q-2", run_id="r-1") == {
        "decision": "deny"
    }
    assert await sqlite_cp.load_hitl_answer("t-1", "q-1", run_id="r-2") == {
        "decision": "edit"
    }

    await sqlite_cp.clear_hitl_answers("t-1", run_id="r-1")
    assert await sqlite_cp.load_hitl_answer("t-1", "q-2", run_id="r-1") is None
    assert await sqlite_cp.load_hitl_answer("t-1", "q-1", run_id="r-2") == {
        "decision": "edit"
    }


async def test_sqlite_hitl_answer_ledger_clear_empty_iterable(sqlite_cp):
    await sqlite_cp.save_hitl_answer("t-1", "q-1", {"decision": "approve"})
    await sqlite_cp.clear_hitl_answers("t-1", [], run_id=None)
    assert await sqlite_cp.load_hitl_answer("t-1", "q-1") == {"decision": "approve"}
