"""Provider-agnostic model client wrappers."""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from typing import Any

from .types import AgentConfig, AgentResponse, Usage


class ProviderAPIError(Exception):
    """Normalized provider API error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ProviderConnectionError(Exception):
    """Normalized provider connection error."""


class BaseModelClient(ABC):
    """Common interface for provider-specific clients."""

    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    async def create_message(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        """Create a response with tool use support."""

    async def create_message_streaming(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        """Fallback streaming path for providers without incremental output handling."""
        response = await self.create_message(messages=messages, system=system, tools=tools)
        text = extract_text_from_blocks(response.content)
        if text:
            print(text, file=sys.stdout)
        return response

    @abstractmethod
    async def create_message_simple(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> AgentResponse:
        """Create a simple response without tool definitions."""


class AnthropicClient(BaseModelClient):
    """Anthropic client wrapper with normalized outputs."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        import anthropic
        from anthropic import AsyncAnthropic

        self._anthropic = anthropic
        self.client = AsyncAnthropic()

    async def create_message(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        try:
            response = await self.client.messages.create(
                model=self.config.resolved_model(),
                max_tokens=self.config.max_tokens,
                system=system,
                messages=messages,
                tools=tools if tools else self._anthropic.NOT_GIVEN,
            )
        except self._anthropic.APIStatusError as exc:
            raise ProviderAPIError(exc.message, status_code=exc.status_code) from exc
        except self._anthropic.APIConnectionError as exc:
            raise ProviderConnectionError(str(exc)) from exc

        return _normalize_anthropic_message(response)

    async def create_message_streaming(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        try:
            async with self.client.messages.stream(
                model=self.config.resolved_model(),
                max_tokens=self.config.max_tokens,
                system=system,
                messages=messages,
                tools=tools if tools else self._anthropic.NOT_GIVEN,
            ) as stream:
                async for text in stream.text_stream:
                    print(text, end="", flush=True, file=sys.stdout)
                print(file=sys.stdout)
                final_message = await stream.get_final_message()
        except self._anthropic.APIStatusError as exc:
            raise ProviderAPIError(exc.message, status_code=exc.status_code) from exc
        except self._anthropic.APIConnectionError as exc:
            raise ProviderConnectionError(str(exc)) from exc

        return _normalize_anthropic_message(final_message)

    async def create_message_simple(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> AgentResponse:
        try:
            response = await self.client.messages.create(
                model=model or self.config.resolved_summary_model(),
                max_tokens=max_tokens,
                messages=messages,
            )
        except self._anthropic.APIStatusError as exc:
            raise ProviderAPIError(exc.message, status_code=exc.status_code) from exc
        except self._anthropic.APIConnectionError as exc:
            raise ProviderConnectionError(str(exc)) from exc

        return _normalize_anthropic_message(response)


class OpenAIClient(BaseModelClient):
    """OpenAI chat completions wrapper with normalized outputs."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        from openai import APIConnectionError, APIStatusError, AsyncOpenAI

        self._api_connection_error = APIConnectionError
        self._api_status_error = APIStatusError
        self.client = AsyncOpenAI()

    async def create_message(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        try:
            response = await self.client.chat.completions.create(
                model=self.config.resolved_model(),
                messages=_openai_messages(messages, system),
                tools=_openai_tools(tools) if tools else None,
                tool_choice="auto" if tools else None,
                max_completion_tokens=self.config.max_tokens,
            )
        except self._api_status_error as exc:
            raise ProviderAPIError(str(exc), status_code=exc.status_code) from exc
        except self._api_connection_error as exc:
            raise ProviderConnectionError(str(exc)) from exc

        return _normalize_openai_message(response)

    async def create_message_simple(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> AgentResponse:
        try:
            response = await self.client.chat.completions.create(
                model=model or self.config.resolved_summary_model(),
                messages=_openai_messages(messages, []),
                max_completion_tokens=max_tokens,
            )
        except self._api_status_error as exc:
            raise ProviderAPIError(str(exc), status_code=exc.status_code) from exc
        except self._api_connection_error as exc:
            raise ProviderConnectionError(str(exc)) from exc

        return _normalize_openai_message(response)


class GoogleClient(BaseModelClient):
    """Google GenAI wrapper with manual function calling."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client().aio

    async def create_message(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": self.config.max_tokens,
        }
        if system_text := _system_text(system):
            config_kwargs["system_instruction"] = system_text
        if tools:
            config_kwargs["tools"] = _google_tools(tools, self._types)
            config_kwargs["automatic_function_calling"] = self._types.AutomaticFunctionCallingConfig(
                disable=True
            )
        config = self._types.GenerateContentConfig(**config_kwargs)
        try:
            response = await self.client.models.generate_content(
                model=self.config.resolved_model(),
                contents=_google_contents(messages, self._types),
                config=config,
            )
        except Exception as exc:  # google-genai currently exposes transport-specific errors
            if "connect" in str(exc).lower() or "dns" in str(exc).lower():
                raise ProviderConnectionError(str(exc)) from exc
            raise ProviderAPIError(str(exc)) from exc

        return _normalize_google_response(response)

    async def create_message_simple(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> AgentResponse:
        config = self._types.GenerateContentConfig(max_output_tokens=max_tokens)
        try:
            response = await self.client.models.generate_content(
                model=model or self.config.resolved_summary_model(),
                contents=_google_contents(messages, self._types),
                config=config,
            )
        except Exception as exc:
            if "connect" in str(exc).lower() or "dns" in str(exc).lower():
                raise ProviderConnectionError(str(exc)) from exc
            raise ProviderAPIError(str(exc)) from exc

        return _normalize_google_response(response)


def create_client(config: AgentConfig) -> BaseModelClient:
    """Create a provider-specific client from configuration."""
    if config.provider == "anthropic":
        return AnthropicClient(config)
    if config.provider == "openai":
        return OpenAIClient(config)
    if config.provider == "google":
        return GoogleClient(config)
    raise ValueError(f"Unsupported provider: {config.provider}")


def extract_text_from_blocks(content: Any) -> str:
    """Extract text from normalized response blocks."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(part for part in parts if part)


def _normalize_anthropic_message(message: Any) -> AgentResponse:
    usage = Usage(
        input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
    )
    content: list[dict[str, Any]] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            content.append({"type": "text", "text": block.text})
        elif getattr(block, "type", None) == "tool_use":
            content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return AgentResponse(content=content, stop_reason=message.stop_reason, usage=usage)


def _normalize_openai_message(response: Any) -> AgentResponse:
    choice = response.choices[0]
    message = choice.message
    content: list[dict[str, Any]] = []
    if message.content:
        content.append({"type": "text", "text": message.content})
    for tool_call in message.tool_calls or []:
        arguments = tool_call.function.arguments or "{}"
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_arguments = {"raw_arguments": arguments}
        content.append({
            "type": "tool_use",
            "id": tool_call.id,
            "name": tool_call.function.name,
            "input": parsed_arguments,
        })

    if message.tool_calls:
        stop_reason = "tool_use"
    elif choice.finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    usage = Usage(
        input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
    )
    return AgentResponse(content=content, stop_reason=stop_reason, usage=usage)


def _normalize_google_response(response: Any) -> AgentResponse:
    content: list[dict[str, Any]] = []
    text = getattr(response, "text", None)
    if text:
        content.append({"type": "text", "text": text})

    function_calls = getattr(response, "function_calls", None) or []
    for index, function_call in enumerate(function_calls, start=1):
        content.append({
            "type": "tool_use",
            "id": getattr(function_call, "id", None) or f"google-call-{index}",
            "name": function_call.name,
            "input": dict(function_call.args or {}),
        })

    if function_calls:
        stop_reason = "tool_use"
    else:
        finish_reason = None
        if getattr(response, "candidates", None):
            finish_reason = getattr(response.candidates[0], "finish_reason", None)
        stop_reason = "max_tokens" if str(finish_reason).upper() == "MAX_TOKENS" else "end_turn"

    usage_metadata = getattr(response, "usage_metadata", None)
    usage = Usage(
        input_tokens=getattr(usage_metadata, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage_metadata, "candidates_token_count", 0) or 0,
    )
    return AgentResponse(content=content, stop_reason=stop_reason, usage=usage)


def _system_text(system: list[dict[str, Any]]) -> str:
    return "\n\n".join(block.get("text", "") for block in system if block.get("type") == "text")


def _openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
                "strict": False,
            },
        }
        for tool in tools
    ]


def _google_tools(tools: list[dict[str, Any]], types_module: Any) -> list[Any]:
    declarations = [
        types_module.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters_json_schema=tool["input_schema"],
        )
        for tool in tools
    ]
    return [types_module.Tool(function_declarations=declarations)]


def _openai_messages(
    messages: list[dict[str, Any]],
    system: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    system_text = _system_text(system)
    if system_text:
        converted.append({"role": "system", "content": system_text})

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "user" and isinstance(content, str):
            converted.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            text = extract_text_from_blocks(content)
            tool_calls = []
            for block in content or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
            assistant_message: dict[str, Any] = {"role": "assistant"}
            if text:
                assistant_message["content"] = text
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            elif "content" not in assistant_message:
                assistant_message["content"] = ""
            converted.append(assistant_message)
            continue

        if role == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    converted.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"],
                    })
                else:
                    converted.append({"role": "user", "content": str(block)})
            continue

        converted.append({"role": role, "content": str(content)})

    return converted


def _google_contents(messages: list[dict[str, Any]], types_module: Any) -> list[Any]:
    contents: list[Any] = []
    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "user" and isinstance(content, str):
            contents.append(types_module.Content(
                role="user",
                parts=[types_module.Part.from_text(text=content)],
            ))
            continue

        if role == "assistant":
            text_parts = []
            function_parts = []
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(types_module.Part.from_text(text=block["text"]))
                elif block.get("type") == "tool_use":
                    function_parts.append(types_module.Part.from_function_call(
                        name=block["name"],
                        args=block.get("input", {}),
                    ))
            parts = [*text_parts, *function_parts]
            if parts:
                contents.append(types_module.Content(role="model", parts=parts))
            continue

        if role == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_name = _find_tool_name(messages, block["tool_use_id"])
                    if not tool_name:
                        continue
                    response_payload = (
                        {"error": block["content"]}
                        if block.get("is_error")
                        else {"result": block["content"]}
                    )
                    part = types_module.Part.from_function_response(
                        name=tool_name,
                        response=response_payload,
                    )
                    contents.append(types_module.Content(role="tool", parts=[part]))
                else:
                    contents.append(types_module.Content(
                        role="user",
                        parts=[types_module.Part.from_text(text=str(block))],
                    ))
            continue

        contents.append(types_module.Content(
            role=role,
            parts=[types_module.Part.from_text(text=str(content))],
        ))

    return contents


def _find_tool_name(messages: list[dict[str, Any]], tool_use_id: str) -> str | None:
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                return block.get("name")
    return None
