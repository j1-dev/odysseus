"""Interactive tool-permission registry for the agent loop.

When the agent wants to run a tool that mutates the world (edit_file,
write_file, bash, …) and the user hasn't opted into auto-accept, the loop
PAUSES: it emits a `permission_request` SSE event and awaits the user's
decision. The frontend shows an approve/deny prompt and POSTs the answer to
`/api/agent/permission/{session_id}`, which calls `resolve()` here to wake the
waiting loop.

State is in-memory and keyed by session_id (one pending request per session at
a time — the loop only asks for one tool before continuing). It does not
survive a server restart; a pending request left dangling is simply denied by
timeout so the agent never hangs forever.
"""
import asyncio
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class _Pending:
    __slots__ = ("request_id", "event", "decision", "scope")

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.event = asyncio.Event()
        self.decision: Optional[str] = None   # "approve" | "deny"
        self.scope: str = "once"              # "once" | "session" (approve-all)


_PENDING: Dict[str, _Pending] = {}

# Tools that mutate the world and therefore require confirmation when
# auto-accept is off. Read-only tools (read_file, grep, glob, web_search, …)
# never prompt. bash/python can do anything, so they're gated too.
GATED_TOOLS = frozenset({
    "edit_file", "write_file", "notebook_edit", "bash", "python",
})

# Sessions where the user clicked "approve for the rest of this run" — every
# subsequent gated tool in the same session is auto-approved without asking.
_SESSION_GRANTS: set = set()

# How long to wait for a user decision before giving up and denying. Bounds
# a forgotten prompt so the detached run can't sit pinned open forever.
DECISION_TIMEOUT_S = 300


def create(session_id: str, request_id: str) -> _Pending:
    """Register a pending permission request for a session, replacing any
    stale one. Returns the _Pending whose event the loop awaits."""
    p = _Pending(request_id)
    _PENDING[session_id] = p
    return p


def resolve(session_id: str, request_id: str, decision: str, scope: str = "once") -> bool:
    """Record the user's decision and wake the waiting loop.

    Returns False if there's no matching pending request (already resolved,
    timed out, or a stale request_id from a previous prompt).
    """
    p = _PENDING.get(session_id)
    if p is None or p.request_id != request_id:
        return False
    p.decision = "approve" if decision == "approve" else "deny"
    p.scope = "session" if scope == "session" else "once"
    if p.decision == "approve" and p.scope == "session":
        _SESSION_GRANTS.add(session_id)
    p.event.set()
    return True


def has_session_grant(session_id: str) -> bool:
    """True if the user approved all tools for the rest of this session."""
    return session_id in _SESSION_GRANTS


async def wait_for_decision(session_id: str, p: _Pending) -> str:
    """Await the user's decision for a pending request. Returns
    "approve" or "deny". Times out to "deny" so the agent never hangs."""
    try:
        await asyncio.wait_for(p.event.wait(), timeout=DECISION_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.info("[agent-permission] %s timed out — denying", session_id)
        return "deny"
    finally:
        # One-shot: drop the pending entry once decided (or timed out).
        if _PENDING.get(session_id) is p:
            _PENDING.pop(session_id, None)
    return p.decision or "deny"


def clear_session(session_id: str) -> None:
    """Drop any pending request + session grant when a run ends. Called from
    the loop's teardown so a new run starts clean."""
    _PENDING.pop(session_id, None)
    _SESSION_GRANTS.discard(session_id)
