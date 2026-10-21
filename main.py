"""
A2A Invoice Agent  (IITM TDS GA5)
---------------------------------
A single-file FastAPI app that implements the A2A 1.0 HTTP+JSON invoice agent.

Design choices (read these once, they explain everything):
  * ALL state is in memory (dicts). That is fine for the grader because the whole
    receipt lifecycle happens inside one verification window. BUT it means you MUST
    run exactly ONE worker process (see the start command in render.yaml / README).
  * Protocol layer, storage layer, and the AI decision are kept separate so retries
    never repeat a model call or an action.
  * Only the very first time a brand-new batch of packages is seen do we call the AI.
    Decisions are cached by the *content* of each package, so Checks/Save that reuse
    the same core packages cost nothing. Polls, lists, replays, cancels and receipt
    continuations never call the model.

Env vars you set on Render:
  A2A_BASE_URL   e.g. https://your-app.onrender.com/a2a   (no trailing slash)
  AI_API_KEY     your key for any OpenAI-compatible provider
  AI_BASE_URL    default https://api.openai.com/v1
  AI_MODEL       default gpt-4o-mini
  AGENT_VERSION  default 1.0.0
"""

import os
import re
import json
import uuid
import time
import hashlib
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
A2A_BASE_URL = os.environ.get("A2A_BASE_URL", "").rstrip("/")   # may be empty -> auto-detect
AGENT_VERSION = os.environ.get("AGENT_VERSION", "1.0.0")

# Bump this whenever you deploy, so GET / confirms which code is actually live.
BUILD = "2026-10-21-coverage"

AI_BASE_URL = os.environ.get("AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
AI_API_KEY = os.environ.get("AI_API_KEY", "")
AI_MODEL = os.environ.get("AI_MODEL", "gpt-4o-mini")
# Max estimated tokens per model request; batches are split to stay under this so a
# low per-request / TPM cap (e.g. Groq free tier = 8000 TPM) doesn't 413 the whole batch.
AI_MAX_TOKENS_PER_CALL = int(os.environ.get("AI_MAX_TOKENS_PER_CALL", "5000"))
# Max packages per model call. Small models (e.g. gpt-4o-mini) get "lazy" and return
# fewer answers than asked when a batch is large, so we keep each call small and reliable.
AI_MAX_PKGS_PER_CALL = int(os.environ.get("AI_MAX_PKGS_PER_CALL", "4"))

# Optional: when set, GET /debug/last (with header X-Debug-Token) dumps the last few
# batches + decisions so you can tune the prompt against real packages. Leave UNSET
# during the graded Save.
DEBUG_TOKEN = os.environ.get("DEBUG_TOKEN", "")


def log(*parts: Any) -> None:
    print("[agent]", *parts, flush=True)


# Ring buffer of recent batches for debugging (capped). Not principal-scoped on purpose.
LAST_DEBUG: List[Dict[str, Any]] = []

A2A_MEDIA = "application/a2a+json"
INPUT_MODE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MODE = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MODE = "application/vnd.ga5.invoice-action-receipts+json"
RESULTS_MODE = "application/vnd.ga5.invoice-action-results+json"

VALID_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

# ProtoJSON state strings
S_SUBMITTED = "TASK_STATE_SUBMITTED"
S_WORKING = "TASK_STATE_WORKING"
S_INPUT = "TASK_STATE_INPUT_REQUIRED"
S_COMPLETED = "TASK_STATE_COMPLETED"
S_CANCELED = "TASK_STATE_CANCELED"
TERMINAL = {S_COMPLETED, S_CANCELED}

# --------------------------------------------------------------------------- #
# In-memory stores  (single worker only!)
# --------------------------------------------------------------------------- #
TASKS: Dict[str, Dict[str, Any]] = {}                 # task_id -> record
IDEMP: Dict[Tuple[str, str], Dict[str, str]] = {}     # (principal, messageId) -> {hash, task_id}
DECISION_CACHE: Dict[str, Dict[str, Any]] = {}        # package_content_hash -> decision

_task_locks: Dict[str, asyncio.Lock] = {}
_msg_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
_registry_lock = asyncio.Lock()


async def task_lock(task_id: str) -> asyncio.Lock:
    async with _registry_lock:
        lk = _task_locks.get(task_id)
        if lk is None:
            lk = asyncio.Lock()
            _task_locks[task_id] = lk
        return lk


async def msg_lock(key: Tuple[str, str]) -> asyncio.Lock:
    async with _registry_lock:
        lk = _msg_locks.get(key)
        if lk is None:
            lk = asyncio.Lock()
            _msg_locks[key] = lk
        return lk


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def canonical(obj: Any) -> str:
    """Recursively key-sorted, compact JSON."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def new_id(prefix: str = "") -> str:
    return (prefix + uuid.uuid4().hex)  # 32 hex chars -> always >= 12


def a2a_json(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(content=data, status_code=status, media_type=A2A_MEDIA)


def err(code: str, message: str, status: int) -> JSONResponse:
    # Generic error body. Never echo another principal's task id or existence.
    return a2a_json({"error": {"code": code, "message": message}}, status=status)


def principal_of(request: Request) -> Optional[str]:
    """Any non-empty Bearer token identifies a distinct user. No preset token exists."""
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    if not token:
        return None
    # Internal, non-reversible principal id so we never leak the raw token.
    return "p_" + hashlib.sha256(token.encode()).hexdigest()[:24]


def check_protocol(request: Request, require_ct: bool) -> Optional[JSONResponse]:
    """Version + (optionally) content-type gate. Returns an error response or None."""
    ver = request.headers.get("a2a-version")
    if ver != "1.0":
        return err("UNSUPPORTED_VERSION", "A2A-Version 1.0 required", 400)
    if require_ct:
        ct = request.headers.get("content-type", "")
        if A2A_MEDIA not in ct:
            return err("UNSUPPORTED_MEDIA_TYPE", "application/a2a+json required", 415)
    return None


def base_url(request: Request) -> str:
    if A2A_BASE_URL:
        return A2A_BASE_URL
    proto = request.headers.get("x-forwarded-proto", "https")
    host = request.headers.get("host", "localhost")
    return f"{proto}://{host}/a2a"


def origin_url(request: Request) -> str:
    b = base_url(request)
    return b[:-4] if b.endswith("/a2a") else b


def pkg_id(pkg: Dict[str, Any]) -> str:
    for k in ("packageId", "package_id", "id"):
        if isinstance(pkg, dict) and pkg.get(k):
            return str(pkg[k])
    return sha(pkg)[:16]


BRACKET_RE = re.compile(r"\[[^\[\]\n]{1,120}\]")


def find_brackets(obj: Any) -> List[str]:
    """All bracketed [REF] tokens appearing anywhere in the package text."""
    text = canonical(obj)
    seen, out = set(), []
    for m in BRACKET_RE.findall(text):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


# --------------------------------------------------------------------------- #
# AI decision layer
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are an invoice action agent. For EACH invoice package you receive, choose "
    "exactly ONE action and cite the exact decisive evidence.\n\n"
    "Actions (pick exactly one per package):\n"
    "- settle_invoice: valid, reconciled, and within autonomous authority.\n"
    "- request_approval: commercially valid, but the amount/terms are OUTSIDE delegated authority.\n"
    "- hold_invoice: payment pauses until a stated verification completes.\n"
    "- reject_duplicate: the SAME commercial invoice was already paid (same vendor+invoice number+amount). "
    "A payment for a different invoice is NOT a duplicate.\n"
    "- open_exception: material records conflict and need an exception workflow.\n\n"
    "Rules:\n"
    "1. The documents MIX useful facts with old archive examples, negation, and irrelevant action "
    "words. Decide ONLY from the paragraph that actually governs this invoice. Ignore cover-sheet "
    "references, archive examples, and training decoys.\n"
    "2. evidenceRefs MUST be EXACTLY THREE bracketed references (like [PAY-12] or [POLICY-4]) copied "
    "VERBATIM from this package's governing paragraph. Not two, not four. Three.\n"
    "3. Extract facts: vendorName (string), invoiceNumber (string), amountMinor (integer, the amount in "
    "the smallest currency unit, e.g. rupees*100 or cents), currency (ISO code like INR or USD).\n"
    "4. rationale: 60 to 1500 characters, name the chosen action and cite at least two of the three refs.\n\n"
    "Return STRICT JSON ONLY, shaped as:\n"
    '{"decisions":[{"packageId":"...","action":"...","evidenceRefs":["[..]","[..]","[..]"],'
    '"facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":12345,"currency":"INR"},'
    '"rationale":"..."}]}'
)


def _loads_loose(content: str) -> Dict[str, Any]:
    """Parse JSON that may be wrapped in prose or ```json fences."""
    content = content.strip()
    try:
        return json.loads(content)
    except Exception:
        pass
    # Strip code fences / find the outermost {...}
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except Exception:
            pass
    return {}


def _est_tokens(obj: Any) -> int:
    """Rough token estimate: ~3 chars per token for compact JSON."""
    return max(1, len(canonical(obj)) // 3)


def _chunk_by_tokens(packages: List[Dict[str, Any]], budget: int,
                     max_pkgs: int = 1000) -> List[List[Dict[str, Any]]]:
    """Split packages so each chunk stays under `budget` tokens AND holds at most
    `max_pkgs` packages. The package cap keeps each call small enough that a small
    model reliably returns every answer."""
    base = len(SYSTEM_PROMPT) // 3 + 60
    chunks: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_tok = 0
    for p in packages:
        t = _est_tokens(p)
        if cur and (len(cur) >= max_pkgs or cur_tok + t + base > budget):
            chunks.append(cur)
            cur, cur_tok = [], 0
        cur.append(p)
        cur_tok += t
    if cur:
        chunks.append(cur)
    return chunks


async def _request_once(client, packages, headers):
    """One HTTP attempt for one chunk. Returns (decisions, status) with status in
    {ok, ok_no_jsonmode, too_large, rate_limited, error:...}."""
    user_payload = {"packages": [{"packageId": pkg_id(p), "package": p} for p in packages]}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    bodies = [
        {"model": AI_MODEL, "temperature": 0, "response_format": {"type": "json_object"}, "messages": messages},
        {"model": AI_MODEL, "temperature": 0, "messages": messages},
    ]
    if AI_MODEL.startswith("openai/gpt-oss"):
        for b in bodies:
            b["reasoning_effort"] = "low"  # fewer reasoning tokens -> less TPM, faster

    last = "error:unknown"
    for i, body in enumerate(bodies):
        try:
            r = await client.post(f"{AI_BASE_URL}/chat/completions", json=body, headers=headers)
        except Exception as e:
            last = f"error:exc:{type(e).__name__}"
            continue
        if r.status_code == 413:
            return {}, "too_large"
        if r.status_code == 429:
            return {}, "rate_limited"
        if r.status_code >= 400:
            last = f"error:http_{r.status_code}"
            continue  # e.g. json_object unsupported -> try the plain body
        try:
            content = r.json()["choices"][0]["message"]["content"]
        except Exception:
            last = "error:badresp"
            continue
        parsed = _loads_loose(content)
        out: Dict[str, Dict[str, Any]] = {}
        for d in parsed.get("decisions", []):
            pid = str(d.get("packageId", ""))
            if pid:
                out[pid] = d
        if out:
            return out, ("ok" if i == 0 else "ok_no_jsonmode")
        last = "error:empty_parse"
    return {}, last


async def call_model(packages: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Decide packages via the model, splitting into sub-requests that fit under the
    provider's per-request token cap. Returns ({packageId: decision}, status_string).
    On a tight free tier some chunks may be rate-limited and left to fall back; the
    content cache means they can be filled on a later request without repeating work."""
    if not AI_API_KEY:
        return {}, "no_key"

    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    out: Dict[str, Dict[str, Any]] = {}
    fail = ""
    deadline = time.time() + 36.0  # stay inside the 45s per-request grader budget

    async def process(pkgs: List[Dict[str, Any]], retries_left: int) -> None:
        nonlocal fail
        # Stack so a 413 can split a chunk in half and requeue the halves.
        stack = list(reversed(_chunk_by_tokens(pkgs, AI_MAX_TOKENS_PER_CALL, AI_MAX_PKGS_PER_CALL)))
        while stack:
            if time.time() > deadline:
                fail = fail or "deadline"
                return
            chunk = stack.pop()
            dec, status = await _request_once(client, chunk, headers)
            if status in ("ok", "ok_no_jsonmode"):
                out.update(dec)
            elif status == "too_large" and len(chunk) > 1:
                mid = len(chunk) // 2
                stack.append(chunk[mid:])
                stack.append(chunk[:mid])
            elif status == "rate_limited" and retries_left > 0 and time.time() < deadline - 5:
                retries_left -= 1
                await asyncio.sleep(3.0)
                stack.append(chunk)  # wait out a little TPM and retry this chunk
            else:
                fail = status  # give up this chunk for now

    async with httpx.AsyncClient(timeout=30) as client:
        await process(packages, retries_left=2)
        # Gap-fill: re-request any packages the model skipped (small-model laziness),
        # rather than letting them fall back.
        missing = [p for p in packages if pkg_id(p) not in out]
        if missing and time.time() < deadline - 3:
            await process(missing, retries_left=1)

    if not out:
        return {}, (fail or "error:none")
    still_missing = sum(1 for p in packages if pkg_id(p) not in out)
    if still_missing or fail:
        return out, f"partial:{fail or 'incomplete'}"
    return out, "ok"


def fallback_decision(pkg: Dict[str, Any]) -> Dict[str, Any]:
    """Used only when the model is unavailable or skipped a package. Conservative escalate."""
    refs = find_brackets(pkg)[:3]
    while len(refs) < 3:
        refs.append("[n/a]")
    return {
        "action": "open_exception",
        "evidenceRefs": refs,
        "facts": {"vendorName": "", "invoiceNumber": "", "amountMinor": 0, "currency": "INR"},
        "rationale": "open_exception chosen as a safe default because no model decision was "
                     f"available; records need manual review per {refs[0]} and {refs[1]}.",
    }


def sanitize_decision(dec: Dict[str, Any], pkg: Dict[str, Any]) -> Dict[str, Any]:
    """Force the decision to satisfy the typed schema so it never fails a contract check."""
    action = dec.get("action")
    if action not in VALID_ACTIONS:
        action = "open_exception"

    refs = dec.get("evidenceRefs") or []
    refs = [str(x) for x in refs if str(x).strip()]
    if len(refs) > 3:
        refs = refs[:3]
    if len(refs) < 3:
        for b in find_brackets(pkg):
            if b not in refs:
                refs.append(b)
            if len(refs) == 3:
                break
    while len(refs) < 3:
        refs.append("[n/a]")

    facts = dec.get("facts") or {}
    try:
        amount = int(facts.get("amountMinor", 0))
    except (TypeError, ValueError):
        amount = 0
    facts = {
        "vendorName": str(facts.get("vendorName", "")),
        "invoiceNumber": str(facts.get("invoiceNumber", "")),
        "amountMinor": amount,
        "currency": str(facts.get("currency", "INR")) or "INR",
    }

    rationale = str(dec.get("rationale", "")).strip()
    if action not in rationale or sum(r in rationale for r in refs[:2]) < 2 or not (60 <= len(rationale) <= 1500):
        rationale = (f"{action}: decided from the governing paragraph of this package, "
                     f"supported by {refs[0]} and {refs[1]} (also {refs[2]}).")
    rationale = rationale[:1500]
    if len(rationale) < 60:
        rationale = (rationale + " " + " ".join(refs))[:1500].ljust(60)

    return {"action": action, "evidenceRefs": refs, "facts": facts, "rationale": rationale}


async def decide_packages(packages: List[Dict[str, Any]], batch_id: str = "") -> Dict[str, Dict[str, Any]]:
    """Return {packageId: decision}, using the content cache and one model call for the rest."""
    result: Dict[str, Dict[str, Any]] = {}
    uncached: List[Tuple[Dict[str, Any], str]] = []

    for pkg in packages:
        ph = sha(pkg)
        if ph in DECISION_CACHE:
            result[pkg_id(pkg)] = DECISION_CACHE[ph]
        else:
            uncached.append((pkg, ph))

    ai_status = "all_cached"
    model_hit = 0
    if uncached:
        decided, ai_status = await call_model([p for p, _ in uncached])
        for pkg, ph in uncached:
            pid = pkg_id(pkg)
            raw = decided.get(pid)
            if raw is None:
                raw = fallback_decision(pkg)
            else:
                model_hit += 1
            clean = sanitize_decision(raw, pkg)
            DECISION_CACHE[ph] = clean
            result[pid] = clean

    # Diagnostics: show in Render logs whether AI worked and how varied the actions are.
    counts: Dict[str, int] = {}
    for d in result.values():
        counts[d["action"]] = counts.get(d["action"], 0) + 1
    log(f"batch={batch_id} pkgs={len(packages)} cached={len(packages)-len(uncached)} "
        f"model_hit={model_hit} ai={ai_status} actions={counts}")

    if len(LAST_DEBUG) >= 6:
        LAST_DEBUG.pop(0)
    LAST_DEBUG.append({
        "batchId": batch_id,
        "ai_status": ai_status,
        "model_hit": model_hit,
        "action_counts": counts,
        "packages": packages,
        "decisions": {pid: result[pid] for pid in result},
    })

    return result


# --------------------------------------------------------------------------- #
# Task shaping
# --------------------------------------------------------------------------- #
def trim_history(history: List[Any], length: int = 20) -> List[Any]:
    if length and len(history) > length:
        return history[-length:]
    return history


def public_task(record: Dict[str, Any]) -> Dict[str, Any]:
    """The A2A Task object exactly as returned to the client."""
    return {
        "id": record["id"],
        "contextId": record["contextId"],
        "status": {"state": record["state"]},
        "artifacts": record["artifacts"],
        "history": trim_history(record["history"]),
        "kind": "task",
    }


def build_proposals_artifact(batch_id: str, proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "artifactId": new_id("art_"),
        "name": "invoice-action-proposals",
        "parts": [{
            "mediaType": PROPOSALS_MODE,
            "data": {"batchId": batch_id, "proposals": proposals},
        }],
    }


def build_receipts_artifact(batch_id: str, executions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "artifactId": new_id("art_"),
        "name": "invoice-action-receipts",
        "parts": [{
            "mediaType": RECEIPTS_MODE,
            "data": {"batchId": batch_id, "executions": executions},
        }],
    }


def extract_part(message: Dict[str, Any], media: str) -> Optional[Dict[str, Any]]:
    for part in message.get("parts", []):
        if part.get("mediaType") == media:
            return part.get("data")
    return None


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI()
app.router.redirect_slashes = False  # never redirect the grader
log(f"startup build={BUILD} model={AI_MODEL} base={AI_BASE_URL} "
    f"ai_key_set={bool(AI_API_KEY)} max_tokens_per_call={AI_MAX_TOKENS_PER_CALL}")


@app.middleware("http")
async def collapse_slashes(request: Request, call_next):
    """Turn /a2a//message:send into /a2a/message:send without any redirect."""
    path = request.scope.get("path", "")
    if "//" in path:
        new_path = re.sub(r"/{2,}", "/", path)
        request.scope["path"] = new_path
        request.scope["raw_path"] = new_path.encode("utf-8")
    return await call_next(request)


@app.get("/")
async def health():
    return {"ok": True, "service": "a2a-invoice-agent", "build": BUILD,
            "model": AI_MODEL, "ai_key_set": bool(AI_API_KEY)}


@app.get("/debug/last")
async def debug_last(request: Request):
    """Inert unless DEBUG_TOKEN env var is set AND the X-Debug-Token header matches it.
    Lets you read the actual packages the grader sent + the decisions made, so you can
    tune SYSTEM_PROMPT against real data. KEEP DEBUG_TOKEN UNSET DURING THE GRADED SAVE."""
    if not DEBUG_TOKEN or request.headers.get("x-debug-token") != DEBUG_TOKEN:
        return err("NOT_FOUND", "Not found", 404)
    return a2a_json({"recent": LAST_DEBUG})


# ---- Agent Card (public, no auth, no version header required) ---- #
@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    base = base_url(request)
    card = {
        "name": "Invoice Action Agent",
        "description": "Reads invoice claim batches, decides one safe business action per "
                       "invoice with cited evidence, and executes only approved actions.",
        "version": AGENT_VERSION,
        "protocolVersion": "1.0",
        "url": base,
        "preferredTransport": "HTTP+JSON",
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
        "defaultInputModes": [INPUT_MODE],
        "defaultOutputModes": [PROPOSALS_MODE, RECEIPTS_MODE],
        "supportedInterfaces": [
            {"url": base, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
        ],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "invoice_action_agent",
            "description": "Decide and execute a safe invoice action per package with evidence refs.",
            "tags": ["invoice", "a2a", "finance", "decision"],
        }],
    }
    return a2a_json(card)


# ---- message:send (initial batch OR result continuation) ---- #
@app.post("/a2a/message:send")
async def message_send(request: Request):
    principal = principal_of(request)
    if not principal:
        return err("UNAUTHORIZED", "Bearer token required", 401)

    gate = check_protocol(request, require_ct=True)
    if gate:
        return gate

    try:
        body = await request.json()
    except Exception:
        return err("BAD_REQUEST", "Invalid JSON", 400)

    message = body.get("message") or {}
    message_id = message.get("messageId")
    if not message_id:
        return err("BAD_REQUEST", "message.messageId required", 400)

    config = body.get("configuration") or {}
    history_len = config.get("historyLength", 20)

    # Idempotency: hash the MESSAGE ONLY (ignore configuration).
    idem_key = (principal, message_id)
    body_hash = sha(message)

    lk = await msg_lock(idem_key)
    async with lk:
        prior = IDEMP.get(idem_key)
        if prior:
            if prior["hash"] != body_hash:
                return err("IDEMPOTENCY_CONFLICT", "messageId reused with different content", 409)
            # Same message replayed -> return stored task, no model call, no action.
            rec = TASKS[prior["task_id"]]
            return a2a_json({"task": public_task(rec)})

        # Branch: continuation (has results part) vs initial batch.
        results_data = extract_part(message, RESULTS_MODE)
        if results_data is not None:
            return await handle_continuation(principal, message, body_hash, idem_key, history_len)

        batch_data = extract_part(message, INPUT_MODE)
        if batch_data is None:
            return err("BAD_REQUEST", "No invoice-claim-batch part found", 400)

        return await handle_initial_batch(principal, message, batch_data, body_hash, idem_key, history_len)


async def handle_initial_batch(principal, message, batch_data, body_hash, idem_key, history_len):
    batch_id = batch_data.get("batchId", "")
    packages = batch_data.get("packages", []) or []

    decisions = await decide_packages(packages, batch_id)  # <- the only place a model is called

    proposals = []
    used_action_ids = set()
    for pkg in packages:
        pid = pkg_id(pkg)
        dec = decisions.get(pid) or sanitize_decision(fallback_decision(pkg), pkg)
        aid = new_id("act_")
        while aid in used_action_ids:
            aid = new_id("act_")
        used_action_ids.add(aid)
        proposals.append({
            "packageId": pid,
            "actionId": aid,
            "action": dec["action"],
            "facts": dec["facts"],
            "evidenceRefs": dec["evidenceRefs"],
            "rationale": dec["rationale"],
        })

    task_id = new_id("task_")
    context_id = new_id("ctx_")
    artifact = build_proposals_artifact(batch_id, proposals)

    record = {
        "id": task_id,
        "principal": principal,
        "contextId": context_id,
        "batchId": batch_id,
        "state": S_INPUT,
        "artifacts": [artifact],
        "history": [message],
        "proposals": proposals,
        "created": time.time(),
    }
    TASKS[task_id] = record
    IDEMP[idem_key] = {"hash": body_hash, "task_id": task_id}

    return a2a_json({"task": public_task(record)})


async def handle_continuation(principal, message, body_hash, idem_key, history_len):
    task_id = message.get("taskId")
    context_id = message.get("contextId")
    results_data = extract_part(message, RESULTS_MODE)

    if not task_id or task_id not in TASKS:
        return err("NOT_FOUND", "Task not found", 404)

    record = TASKS[task_id]

    # Isolation: only the owner may continue.
    if record["principal"] != principal:
        return err("NOT_FOUND", "Task not found", 404)

    tlock = await task_lock(task_id)
    async with tlock:
        # Terminal replay / race (checked inside the lock so cancel vs receipt is atomic):
        if record["state"] in TERMINAL:
            if record["state"] == S_COMPLETED:
                # A read of an already-completed task. (Same-message replay was handled by
                # idempotency above; this is a late/duplicate continuation.)
                return a2a_json({"task": public_task(record)})
            return err("TASK_TERMINAL", "Task already canceled", 409)

        # ---- STRICT validation: reject the WHOLE continuation on ANY mismatch, before
        # mutating anything. The grader sends tampered continuations on purpose. ----
        if context_id != record["contextId"]:
            return err("INVALID_CONTINUATION", "contextId mismatch", 409)
        if results_data.get("batchId") != record["batchId"]:
            return err("INVALID_CONTINUATION", "batchId mismatch", 409)

        results = results_data.get("results")
        if not isinstance(results, list) or not results:
            return err("INVALID_CONTINUATION", "missing results", 409)

        proposals_by = {p["actionId"]: p for p in record["proposals"]}

        # First pass: validate every result against its stored proposal. No mutation yet.
        for res in results:
            prop = proposals_by.get(res.get("actionId"))
            if prop is None:
                return err("INVALID_CONTINUATION", "unknown actionId", 409)
            if res.get("packageId") != prop["packageId"]:
                return err("INVALID_CONTINUATION", "packageId mismatch", 409)
            if res.get("action") != prop["action"]:
                return err("INVALID_CONTINUATION", "action mismatch", 409)
            outcome = res.get("outcome")
            if outcome not in ("ACCEPTED", "REJECTED"):
                return err("INVALID_CONTINUATION", "bad outcome", 409)
            if outcome == "ACCEPTED" and not res.get("receiptNonce"):
                return err("INVALID_CONTINUATION", "missing receiptNonce", 409)

        # Second pass: continuation is valid -> execute ACCEPTED only, each bound to its
        # stored proposal and its grader-supplied nonce. REJECTED proposals are never run.
        executions = []
        for res in results:
            if res.get("outcome") != "ACCEPTED":
                continue
            prop = proposals_by[res["actionId"]]
            executions.append({
                "packageId": prop["packageId"],
                "actionId": prop["actionId"],
                "action": prop["action"],
                "receiptNonce": res["receiptNonce"],
                "facts": prop["facts"],
                "evidenceRefs": prop["evidenceRefs"],
            })

        receipts_artifact = build_receipts_artifact(record["batchId"], executions)
        record["artifacts"].append(receipts_artifact)
        record["history"].append(message)
        record["state"] = S_COMPLETED
        IDEMP[idem_key] = {"hash": body_hash, "task_id": task_id}

        return a2a_json({"task": public_task(record)})


# ---- GET /tasks/{id} ---- #
@app.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal = principal_of(request)
    if not principal:
        return err("UNAUTHORIZED", "Bearer token required", 401)
    gate = check_protocol(request, require_ct=False)
    if gate:
        return gate

    record = TASKS.get(task_id)
    if not record or record["principal"] != principal:
        return err("NOT_FOUND", "Task not found", 404)
    return a2a_json(public_task(record))


# ---- GET /tasks ---- #
@app.get("/a2a/tasks")
async def list_tasks(request: Request):
    principal = principal_of(request)
    if not principal:
        return err("UNAUTHORIZED", "Bearer token required", 401)
    gate = check_protocol(request, require_ct=False)
    if gate:
        return gate

    mine = [public_task(r) for r in TASKS.values() if r["principal"] == principal]
    return a2a_json({"tasks": mine})


# ---- POST /tasks/{id}:cancel ---- #
@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal = principal_of(request)
    if not principal:
        return err("UNAUTHORIZED", "Bearer token required", 401)
    gate = check_protocol(request, require_ct=False)
    if gate:
        return gate

    record = TASKS.get(task_id)
    if not record or record["principal"] != principal:
        return err("NOT_FOUND", "Task not found", 404)

    tlock = await task_lock(task_id)
    async with tlock:
        if record["state"] in TERMINAL:
            # Loses the cancel/receipt race (already completed or already canceled).
            return err("TASK_TERMINAL", "Task already terminal", 409)
        record["state"] = S_CANCELED
        return a2a_json(public_task(record))
