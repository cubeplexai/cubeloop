from __future__ import annotations

import pytest

import cubeloop.middleware as middleware


def test_lazy_exports_resolve_builtin_middleware() -> None:
    assert middleware.CompactionMiddleware.__name__ == "CompactionMiddleware"
    assert middleware.CompactionState.__name__ == "CompactionState"
    assert middleware.SubagentMiddleware.__name__ == "SubagentMiddleware"
    assert middleware.SubagentSpec.__name__ == "SubagentSpec"
    assert middleware.ToolResultLimitMiddleware.__name__ == "ToolResultLimitMiddleware"


def test_unknown_lazy_export_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        getattr(middleware, "MissingMiddleware")
