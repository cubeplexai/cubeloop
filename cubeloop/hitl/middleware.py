from __future__ import annotations

import inspect
from typing import Awaitable, Callable, Iterable, Protocol, Union, cast

from pydantic import BaseModel

from cubeloop.agent.types import BeforeToolCallContext, BeforeToolCallResult
from cubeloop.hitl.binding import HitlBinding
from cubeloop.hitl.channel import CheckpointedChannel, HitlChannel
from cubeloop.hitl.exceptions import HitlCancelled, HitlTimedOut
from cubeloop.hitl.policy import Approve, ApprovalDecision, AskUser, Deny
from cubeloop.middleware.base import Middleware
from cubeloop.types import JsonObject, StructuredObject


class _VarsObject(Protocol):
    __dict__: JsonObject


def _args_to_dict(args: BaseModel | JsonObject | _VarsObject) -> JsonObject:
    if isinstance(args, BaseModel):
        return args.model_dump()
    if isinstance(args, dict):
        return dict(args)
    return dict(vars(args))


class ApprovalPolicyMiddleware(Middleware):
    def __init__(
        self,
        channel: HitlChannel,
        policy: Callable[
            [BeforeToolCallContext],
            Union[ApprovalDecision, Awaitable[ApprovalDecision]],
        ],
    ):
        self._channel = channel
        self._policy = policy
        checkpointed = isinstance(channel, CheckpointedChannel)
        self.hitl = HitlBinding(
            checkpointed=checkpointed,
            run_id=getattr(channel, "_run_id", None) if checkpointed else None,
        )

    async def before_tool_call(
        self, ctx: BeforeToolCallContext, *, signal=None
    ) -> BeforeToolCallResult | None:
        decision = self._policy(ctx)
        if inspect.isawaitable(decision):
            decision = await decision

        if isinstance(decision, Approve):
            return None

        if isinstance(decision, Deny):
            return BeforeToolCallResult(
                block=True,
                deny_reason=decision.reason,
                reason=decision.reason,
                hitl_trace={"decision": "policy_deny", "reason": decision.reason},
            )

        if isinstance(decision, AskUser):
            return await self._ask_and_translate(ctx, decision, signal=signal)

        raise TypeError(f"policy returned unexpected {type(decision).__name__}")

    async def _ask_and_translate(
        self, ctx: BeforeToolCallContext, ask: AskUser, *, signal
    ) -> BeforeToolCallResult | None:
        original_args = _args_to_dict(ctx.args)
        try:
            answer = await self._channel.approve(
                tool_name=ctx.tool_call.name,
                tool_call_id=ctx.tool_call.id,
                args=original_args,
                details=ask.details,
                timeout=ask.timeout_seconds,
                signal=signal,
            )
        except HitlTimedOut:
            return BeforeToolCallResult(
                block=True,
                deny_reason="approval_timeout",
                reason="approval_timeout",
                hitl_trace={"decision": "timed_out"},
            )
        except HitlCancelled as exc:
            return BeforeToolCallResult(
                block=True,
                deny_reason=f"cancelled: {exc.reason}",
                reason=f"cancelled: {exc.reason}",
                hitl_trace={"decision": "cancelled", "reason": exc.reason},
            )

        if answer.decision == "approve":
            return BeforeToolCallResult(hitl_trace={"decision": "human_approve"})
        if answer.decision == "deny":
            return BeforeToolCallResult(
                block=True,
                deny_reason=answer.reason,
                reason=answer.reason,
                hitl_trace={"decision": "human_deny", "reason": answer.reason},
            )
        if answer.decision == "edit":
            return BeforeToolCallResult(
                edited_args=answer.edited_args,
                hitl_trace=cast(
                    StructuredObject,
                    {
                        "decision": "edit",
                        "original_args": original_args,
                        "edited_args": answer.edited_args,
                    },
                ),
            )


class ConfirmToolCallMiddleware(ApprovalPolicyMiddleware):
    """Convenience wrapper: 'always ask the human for these tool names'."""

    def __init__(
        self,
        channel: HitlChannel,
        *,
        require_confirm: Union[
            Callable[[BeforeToolCallContext], bool], Iterable[str], None
        ] = None,
        details_fn: Callable[[BeforeToolCallContext], JsonObject] | None = None,
        timeout_seconds: float | None = None,
    ):
        matcher: Callable[[BeforeToolCallContext], bool]
        if require_confirm is None:

            def matcher(ctx: BeforeToolCallContext) -> bool:
                return True

        elif callable(require_confirm):
            matcher = require_confirm
        else:
            names = set(require_confirm)

            def matcher(ctx: BeforeToolCallContext) -> bool:
                return ctx.tool_call.name in names

        def policy(ctx: BeforeToolCallContext) -> ApprovalDecision:
            if matcher(ctx):
                return AskUser(
                    timeout_seconds=timeout_seconds,
                    details=details_fn(ctx) if details_fn else None,
                )
            return Approve()

        super().__init__(channel, policy=policy)
