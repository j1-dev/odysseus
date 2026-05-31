# src/bedrock_adapter.py
"""AWS Bedrock adapter.

Converts the OpenAI-style messages that the rest of the app speaks into
Bedrock's unified Converse / ConverseStream API. boto3 picks up credentials
from the standard AWS chain — environment vars, ~/.aws/credentials, and the
SSO cache populated by `aws sso login`. When the SSO session is alive the
short-lived token is refreshed transparently per call.

Endpoint URL convention used by the rest of Odysseus:

    bedrock://<region>[?profile=<aws-profile-name>]

`region` lives in the host slot, an optional AWS profile in the query string.
The endpoint's `api_key` column is repurposed as the profile name (mostly so
admins can fill it in via the existing UI without a schema migration).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

logger = logging.getLogger(__name__)

# Substrings of bedrock model IDs that we know expose chat-style behaviour.
# Embeddings / image / video models are dropped from the picker since the
# rest of the LLM path can't drive them.
_CHAT_FAMILIES = (
    "anthropic.claude",
    "amazon.nova",
    "meta.llama",
    "mistral.",
    "cohere.command",
    "ai21.jamba",
    "deepseek.",
)


def parse_bedrock_url(url: str) -> Tuple[str, Optional[str]]:
    """`bedrock://us-east-1?profile=dev` → ('us-east-1', 'dev')."""
    parts = urlsplit(url or "")
    region = (parts.hostname or parts.netloc or "").strip()
    profile: Optional[str] = None
    if parts.query:
        qs = parse_qs(parts.query)
        vals = qs.get("profile") or qs.get("aws_profile")
        if vals:
            profile = vals[0].strip() or None
    return region, profile


def _get_session(region: str, profile: Optional[str]):
    """Build a boto3 Session for the given region/profile.

    Sessions are cheap; we don't bother caching since the credential
    resolver is what actually does the heavy lifting and it has its own
    in-memory cache.
    """
    try:
        import boto3  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "AWS Bedrock support requires boto3. Install it with "
            "`pip install boto3` (also listed in requirements-optional.txt)."
        ) from e
    return boto3.Session(region_name=region or None, profile_name=profile or None)


def _bedrock_runtime(region: str, profile: Optional[str]):
    return _get_session(region, profile).client("bedrock-runtime")


def _bedrock_control(region: str, profile: Optional[str]):
    return _get_session(region, profile).client("bedrock")


# ── message translation ─────────────────────────────────────────────────────

_DATA_URI_RE = re.compile(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)


def _content_blocks(content: Any) -> List[Dict[str, Any]]:
    """Convert OpenAI content (str | list of blocks) → Bedrock content blocks."""
    if content is None:
        return [{"text": ""}]
    if isinstance(content, str):
        return [{"text": content}]
    blocks: List[Dict[str, Any]] = []
    for b in content:
        if not isinstance(b, dict):
            blocks.append({"text": str(b)})
            continue
        t = b.get("type")
        if t == "text":
            blocks.append({"text": b.get("text", "")})
        elif t == "image_url":
            url = (b.get("image_url") or {}).get("url", "")
            m = _DATA_URI_RE.match(url)
            if not m:
                # Bedrock only takes raw bytes; remote URLs aren't supported.
                logger.warning("Bedrock: skipping non-base64 image_url block")
                continue
            import base64
            fmt = m.group(1).lower().split("/")[-1]
            if fmt == "jpg":
                fmt = "jpeg"
            blocks.append({
                "image": {
                    "format": fmt,
                    "source": {"bytes": base64.b64decode(m.group(2))},
                }
            })
        # Other block types (tool_result, tool_use) are handled separately.
    return blocks or [{"text": ""}]


def build_converse_payload(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build kwargs for bedrock-runtime.converse() / converse_stream().

    Maps OpenAI roles to Bedrock equivalents:
      system → top-level `system` list
      user/assistant → `messages` with content blocks
      tool → user message with toolResult block
      assistant w/ tool_calls → assistant message with toolUse blocks
    """
    system_blocks: List[Dict[str, str]] = []
    out_messages: List[Dict[str, Any]] = []

    for m in messages:
        role = m.get("role")
        if role == "system":
            sys_text = m.get("content")
            if isinstance(sys_text, list):
                # Flatten any text blocks.
                sys_text = "\n".join(b.get("text", "") for b in sys_text if isinstance(b, dict) and b.get("type") == "text")
            if sys_text:
                system_blocks.append({"text": sys_text})
            continue
        if role == "tool":
            content = m.get("content", "")
            if isinstance(content, str):
                tool_content = [{"text": content}]
            else:
                tool_content = _content_blocks(content)
            out_messages.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": m.get("tool_call_id", ""),
                        "content": tool_content,
                    }
                }],
            })
            continue
        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            blocks: List[Dict[str, Any]] = []
            if m.get("content"):
                blocks.extend(_content_blocks(m["content"]))
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except (json.JSONDecodeError, TypeError):
                    args = {}
                blocks.append({
                    "toolUse": {
                        "toolUseId": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                })
            out_messages.append({"role": "assistant", "content": blocks})
            continue
        # Plain user / assistant
        if role in ("user", "assistant"):
            out_messages.append({"role": role, "content": _content_blocks(m.get("content"))})
        else:
            # Anything else: best-effort treat as user.
            out_messages.append({"role": "user", "content": _content_blocks(m.get("content"))})

    inference: Dict[str, Any] = {}
    if max_tokens and max_tokens > 0:
        inference["maxTokens"] = int(max_tokens)
    if temperature is not None:
        # Bedrock rejects values > 1.0 for some families; clamp safely.
        inference["temperature"] = max(0.0, min(float(temperature), 1.0))

    payload: Dict[str, Any] = {
        "modelId": model,
        "messages": out_messages,
    }
    if system_blocks:
        payload["system"] = system_blocks
    if inference:
        payload["inferenceConfig"] = inference

    if tools:
        bedrock_tools = []
        for t in tools:
            if t.get("type") != "function":
                continue
            fn = t["function"]
            bedrock_tools.append({
                "toolSpec": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "inputSchema": {"json": fn.get("parameters", {"type": "object", "properties": {}})},
                }
            })
        if bedrock_tools:
            payload["toolConfig"] = {"tools": bedrock_tools}

    return payload


# ── response parsing ────────────────────────────────────────────────────────

def parse_converse_response(resp: Dict[str, Any]) -> str:
    """Pull the assistant text out of a non-streaming Converse response."""
    out = resp.get("output", {}).get("message", {})
    parts: List[str] = []
    for block in out.get("content", []) or []:
        if "text" in block:
            parts.append(block["text"] or "")
    return "".join(parts)


# ── unsupported-inference-param handling ────────────────────────────────────

# Map the inferenceConfig key → substrings that appear in the validation error
# when a model rejects that param. Newer Claude (opus-4.x) deprecates
# `temperature`; some families reject `topP` or `temperature` together.
_INFERENCE_PARAMS = ("temperature", "topP", "maxTokens")


def _strip_rejected_param(payload: Dict[str, Any], err_msg: str) -> bool:
    """If err_msg names an inferenceConfig param as deprecated/unsupported,
    remove it from the payload in place. Returns True if something was removed
    (so the caller can retry)."""
    msg = (err_msg or "").lower()
    if "deprecat" not in msg and "not supported" not in msg and "isn't supported" not in msg and "unsupported" not in msg:
        return False
    cfg = payload.get("inferenceConfig") or {}
    removed = False
    for key in _INFERENCE_PARAMS:
        if key.lower() in msg and key in cfg:
            cfg.pop(key, None)
            removed = True
    if removed:
        if cfg:
            payload["inferenceConfig"] = cfg
        else:
            payload.pop("inferenceConfig", None)
    return removed


def _is_validation_error(e: Exception) -> bool:
    return type(e).__name__ == "ValidationException" or "ValidationException" in str(e)


# ── sync entry point ────────────────────────────────────────────────────────

def converse_sync(url: str, model: str, messages, temperature, max_tokens) -> str:
    region, profile = parse_bedrock_url(url)
    if not region:
        raise RuntimeError("Bedrock endpoint URL must include a region, e.g. bedrock://us-east-1")
    client = _bedrock_runtime(region, profile)
    payload = build_converse_payload(model, messages, temperature, max_tokens)
    # Retry up to a couple times, dropping each inferenceConfig param the model
    # rejects (e.g. opus-4.x deprecates `temperature`).
    for _ in range(len(_INFERENCE_PARAMS) + 1):
        try:
            resp = client.converse(**payload)
            return parse_converse_response(resp)
        except Exception as e:
            if _is_validation_error(e) and _strip_rejected_param(payload, str(e)):
                continue
            raise


# ── async entry point (boto3 is sync, so run in threadpool) ────────────────

async def converse_async(url: str, model: str, messages, temperature, max_tokens) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(converse_sync, url, model, messages, temperature, max_tokens),
    )


# ── streaming entry point ──────────────────────────────────────────────────

def _iter_stream_events(url: str, model: str, payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Drive bedrock-runtime.converse_stream() and yield raw event dicts.

    Opening the stream (the converse_stream call) is what raises a
    ValidationException for a rejected inferenceConfig param, so we can retry
    it here before any events have been yielded."""
    region, profile = parse_bedrock_url(url)
    if not region:
        raise RuntimeError("Bedrock endpoint URL must include a region, e.g. bedrock://us-east-1")
    client = _bedrock_runtime(region, profile)
    response = None
    for _ in range(len(_INFERENCE_PARAMS) + 1):
        try:
            response = client.converse_stream(**payload)
            break
        except Exception as e:
            if _is_validation_error(e) and _strip_rejected_param(payload, str(e)):
                continue
            raise
    for event in (response or {}).get("stream", []):
        yield event


async def stream_converse(url: str, model: str, messages, temperature, max_tokens, tools=None):
    """Yield the same SSE chunks as src.llm_core.stream_llm — text deltas,
    tool_calls (OpenAI-shaped), usage, and a final [DONE]. Errors bubble up
    as `event: error` chunks.
    """
    payload = build_converse_payload(model, messages, temperature, max_tokens, tools=tools)

    # Drive the (sync) boto3 EventStream from a threadpool. We push every
    # event onto an asyncio.Queue so the consumer can iterate normally.
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def _producer():
        try:
            for ev in _iter_stream_events(url, model, payload):
                loop.call_soon_threadsafe(queue.put_nowait, ("event", ev))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", e))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", SENTINEL))

    fut = loop.run_in_executor(None, _producer)

    # Tool block accumulator: index → {id, name, arguments}
    tool_blocks: Dict[int, Dict[str, Any]] = {}
    in_tool: Dict[int, str] = {}  # block index → tool_use_id
    input_tokens = 0
    output_tokens = 0

    try:
        while True:
            kind, payload_ev = await queue.get()
            if kind == "done":
                break
            if kind == "error":
                err = str(payload_ev)
                yield f'event: error\ndata: {json.dumps({"error": err, "status": 502})}\n\n'
                return
            ev = payload_ev

            if "contentBlockStart" in ev:
                start = ev["contentBlockStart"]
                idx = start.get("contentBlockIndex", 0)
                tu = (start.get("start") or {}).get("toolUse")
                if tu:
                    tool_blocks[idx] = {
                        "id": tu.get("toolUseId", f"call_{idx}"),
                        "name": tu.get("name", ""),
                        "arguments": "",
                    }
                    in_tool[idx] = tu.get("toolUseId", "")
            elif "contentBlockDelta" in ev:
                d = ev["contentBlockDelta"]
                idx = d.get("contentBlockIndex", 0)
                delta = d.get("delta") or {}
                if "text" in delta:
                    text = delta["text"] or ""
                    if text:
                        yield f'data: {json.dumps({"delta": text})}\n\n'
                elif "toolUse" in delta:
                    partial = delta["toolUse"].get("input", "") or ""
                    if idx in tool_blocks and partial:
                        tool_blocks[idx]["arguments"] += partial
                        if tool_blocks[idx]["name"] in ("create_document", "update_document", "edit_document"):
                            yield f'data: {json.dumps({"type": "tool_call_delta", "index": idx, "name": tool_blocks[idx]["name"], "arg_delta": partial})}\n\n'
            elif "metadata" in ev:
                usage = ev["metadata"].get("usage") or {}
                input_tokens = usage.get("inputTokens", 0) or input_tokens
                output_tokens = usage.get("outputTokens", 0) or output_tokens
            elif "messageStop" in ev:
                if tool_blocks:
                    calls = [tool_blocks[i] for i in sorted(tool_blocks)]
                    yield f'data: {json.dumps({"type": "tool_calls", "calls": calls})}\n\n'
                if input_tokens or output_tokens:
                    yield f'data: {json.dumps({"type": "usage", "data": {"input_tokens": input_tokens, "output_tokens": output_tokens}})}\n\n'
                yield "data: [DONE]\n\n"
                return
        # Stream ended without an explicit messageStop (shouldn't happen).
        if tool_blocks:
            calls = [tool_blocks[i] for i in sorted(tool_blocks)]
            yield f'data: {json.dumps({"type": "tool_calls", "calls": calls})}\n\n'
        yield "data: [DONE]\n\n"
    finally:
        # Drain the producer if we exited early.
        try:
            await fut
        except Exception:
            pass


# ── model discovery ────────────────────────────────────────────────────────

def list_models(url: str) -> List[str]:
    """Return invokable Bedrock chat model IDs (for the picker). Empty on failure.

    Newer Claude / Nova / Llama models reject on-demand invocation by their
    bare foundation-model ID and must be called via a cross-region
    *inference profile* (IDs prefixed `eu.` / `us.` / `apac.` / `global.`).
    So we lead with inference profiles — those are the IDs that actually
    work — then append any bare foundation models that aren't already
    covered by a profile (older models like Claude 3.x still take on-demand
    by bare ID).
    """
    region, profile = parse_bedrock_url(url)
    if not region:
        return []

    def _is_chat(mid: str) -> bool:
        return bool(mid) and any(fam in mid for fam in _CHAT_FAMILIES) and not (
            ":" in mid and "image" in mid.lower()
        )

    out: List[str] = []
    covered_foundation: set = set()

    # 1) Inference profiles — the invokable IDs for modern models.
    try:
        control = _bedrock_control(region, profile)
        paginator_kwargs = {}
        resp = control.list_inference_profiles(**paginator_kwargs)
        for p in resp.get("inferenceProfileSummaries", []) or []:
            pid = p.get("inferenceProfileId") or ""
            # Strip the region/global prefix to test the family + record which
            # underlying foundation models this profile already covers.
            stripped = pid.split(".", 1)[1] if "." in pid and pid.split(".", 1)[0] in ("eu", "us", "apac", "global") else pid
            if not _is_chat(stripped):
                continue
            out.append(pid)
            for model in (p.get("models") or []):
                arn = model.get("modelArn") or ""
                # ARN tail is the foundation model id.
                if "/" in arn:
                    covered_foundation.add(arn.rsplit("/", 1)[-1])
    except Exception as e:
        logger.warning(f"Bedrock list_inference_profiles failed: {e}")

    # 2) Bare foundation models that no profile covers (older on-demand models).
    try:
        control = _bedrock_control(region, profile)
        resp = control.list_foundation_models(byOutputModality="TEXT")
        for m in resp.get("modelSummaries", []) or []:
            mid = m.get("modelId") or ""
            if not _is_chat(mid) or mid in covered_foundation:
                continue
            out.append(mid)
    except Exception as e:
        logger.warning(f"Bedrock list_foundation_models failed: {e}")

    # De-dup while preserving order (profiles first).
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]
