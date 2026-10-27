from cubeloop.providers.base import (
    AssistantMessage,
    BaseProvider,
    BoundModel,
    Content,
    ImageContent,
    Message,
    MessageStream,
    Model,
    ModelCost,
    OnChunkCallback,
    OnPayloadCallback,
    OnRequestCallback,
    OnResponseBodyCallback,
    OnResponseCallback,
    Provider,
    ProviderResponse,
    ReasoningControl,
    ReasoningEffort,
    ReasoningMode,
    ReasoningSummary,
    StreamEvent,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
    is_empty_completion,
    is_synthetic_message,
    synthetic_user_message,
    chain_providers,
    collect_agent_providers,
)
from cubeloop.providers.capability import (
    CapabilityDescriptor,
    CapabilityWarning,
    PayloadPreview,
    ReasoningCapability,
    TemperatureSpec,
    apply_reasoning_control,
    lint_capability,
    preview_payload,
)
from cubeloop.providers.faux import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from cubeloop.providers.fallback import (
    DEFAULT_TRIGGER_ERRORS,
    FallbackBoundModel,
)
from cubeloop.providers.models import models_are_equal
from cubeloop.providers.reasoning_profiles import get_capability_profile


# Lazy imports for optional providers
def get_anthropic_provider():
    from cubeloop.providers.anthropic import AnthropicProvider

    return AnthropicProvider


def get_openai_provider():
    from cubeloop.providers.openai import OpenAIProvider

    return OpenAIProvider


def get_openai_responses_provider():
    from cubeloop.providers.openai_responses import OpenAIResponsesProvider

    return OpenAIResponsesProvider


__all__ = [
    "AssistantMessage",
    "BaseProvider",
    "BoundModel",
    "Content",
    "DEFAULT_TRIGGER_ERRORS",
    "FauxProvider",
    "FallbackBoundModel",
    "ImageContent",
    "Message",
    "MessageStream",
    "Model",
    "ModelCost",
    "OnChunkCallback",
    "OnPayloadCallback",
    "OnRequestCallback",
    "OnResponseBodyCallback",
    "OnResponseCallback",
    "Provider",
    "ProviderResponse",
    "ReasoningCapability",
    "ReasoningControl",
    "ReasoningEffort",
    "ReasoningMode",
    "ReasoningSummary",
    "StreamEvent",
    "StreamOptions",
    "TextContent",
    "ThinkingContent",
    "ToolCall",
    "ToolDefinition",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "apply_reasoning_control",
    "is_empty_completion",
    "is_synthetic_message",
    "synthetic_user_message",
    "chain_providers",
    "collect_agent_providers",
    "faux_assistant_message",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "get_capability_profile",
    "lint_capability",
    "models_are_equal",
    "preview_payload",
    "CapabilityDescriptor",
    "CapabilityWarning",
    "PayloadPreview",
    "TemperatureSpec",
]
