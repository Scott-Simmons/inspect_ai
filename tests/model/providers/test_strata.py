import os
from collections.abc import AsyncIterator

import pytest
from test_helpers.utils import skip_if_no_strata

from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    Model,
    StreamEvent,
    execute_tools,
    get_model,
)
from inspect_ai.tool import Tool, tool


@pytest.fixture
async def strata_model() -> AsyncIterator[Model]:
    async with get_model(
        f"strata/{os.environ['STRATA_TEST_MODEL']}",
        config=GenerateConfig(
            max_tokens=256,
            temperature=0,
            reasoning_effort="none",
            max_connections=1,
            max_retries=0,
            attempt_timeout=60,
            timeout=90,
        ),
        max_retries=0,
        memoize=False,
    ) as model:
        yield model


@skip_if_no_strata
async def test_strata_generate(strata_model: Model) -> None:
    response = await strata_model.generate(
        "Reply with exactly STRATA_OK and nothing else."
    )
    assert response.completion.strip() == "STRATA_OK"
    assert response.stop_reason == "stop"


@skip_if_no_strata
async def test_strata_tool_roundtrip(strata_model: Model) -> None:
    receipt = "receipt_9f01c6e24b"
    executed: list[str] = []

    @tool
    def lookup_probe_receipt() -> Tool:
        async def execute(probe_id: str) -> str:
            """Look up the opaque receipt for a connectivity probe.

            Args:
                probe_id: The identifier of the probe to look up.
            """
            if probe_id != "inspect-strata-smoke":
                raise ValueError("Unknown probe identifier")
            executed.append(probe_id)
            return receipt

        return execute

    tools = [lookup_probe_receipt()]
    messages: list[ChatMessage] = [
        ChatMessageUser(
            content="Use lookup_probe_receipt to retrieve the receipt for probe_id "
            "inspect-strata-smoke. You do not know the receipt; you must retrieve "
            "it using the tool. After receiving the tool result, reply with "
            "exactly the receipt and nothing else."
        )
    ]
    response = await strata_model.generate(input=messages, tools=tools)
    assert response.stop_reason == "tool_calls"
    assert response.message.tool_calls is not None
    assert len(response.message.tool_calls) == 1
    call = response.message.tool_calls[0]
    assert call.function == "lookup_probe_receipt"
    assert call.arguments == {"probe_id": "inspect-strata-smoke"}

    messages.append(response.message)
    result = await execute_tools(messages, tools)
    assert executed == ["inspect-strata-smoke"]
    assert len(result.messages) == 1
    tool_result = result.messages[0]
    assert tool_result.role == "tool"
    assert tool_result.error is None
    assert tool_result.tool_call_id == call.id
    assert tool_result.content == receipt

    messages.extend(result.messages)
    response = await strata_model.generate(input=messages, tools=tools)
    assert response.completion.strip() == receipt
    assert response.stop_reason == "stop"
    assert not response.message.tool_calls


@skip_if_no_strata
async def test_strata_streaming(strata_model: Model) -> None:
    text: list[str] = []

    async def collect(event: StreamEvent) -> None:
        if event.type == "text":
            text.append(event.text)

    response = await strata_model.generate(
        "Reply with exactly STRATA_STREAM_OK and nothing else.", on_stream=collect
    )
    assert response.completion.strip() == "STRATA_STREAM_OK"
    assert "".join(text) == response.completion
    assert response.stop_reason == "stop"
