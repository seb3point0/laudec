#!/usr/bin/env python3
"""
laudec_serve: a minimal OpenAI-compatible HTTP API backed by laudec.

Exposes `haiku`, `sonnet`, and `opus` over the standard OpenAI endpoints, so any
harness or SDK that speaks `/v1/chat/completions` can use them. Each request is
answered by driving the interactive Claude TUI via laudec.

    POST /v1/chat/completions   (stream + non-stream)
    GET  /v1/models
    GET  /health

Each request shells out to `laudec` (one isolated `claude` session, tools
disabled), capped by a concurrency limit. The OpenAI `reasoning_effort` field
maps to Claude's effort level. Sampling params (temperature, max_tokens) are
accepted and ignored.

OpenAI function calling is supported. When a request carries a `tools` array,
the server teaches the text-only session the available tools and a strict text
protocol for requesting one, then parses the reply back into structured
`tool_calls` with `finish_reason="tool_calls"` (streaming and non-streaming).
Claude never executes anything — it only emits the call as text; the client
runs the tool and sends the result back on the next turn, exactly as with a real
OpenAI endpoint. So the text-only / no-execution security model is unchanged.

Run (equivalent; `laudec serve` keeps everything under one binary):
    laudec serve                                 # 127.0.0.1:8787
    python3 laudec_serve.py
    LAUDEC_SERVE_API_KEY=secret laudec serve --port 9000

Point your harness at:  http://127.0.0.1:8787/v1

laudec authenticates to Claude through your logged-in Claude Code, so the server
needs no key of its own. Set --api-key only to gate access to the endpoint, in
which case clients send that value as the bearer token.

Security: laudec runs Claude as a pure text responder (tools off), so requests
produce text only. The endpoint still fronts your Claude subscription, so it
binds to 127.0.0.1 by default; expose it wider only behind an API key and your
own network controls.

Env / flags:
    LAUDEC_SERVE_HOST         bind host        (default 127.0.0.1)
    LAUDEC_SERVE_PORT         bind port        (default 8787)
    LAUDEC_SERVE_API_KEY      require Bearer    (default: none)
    LAUDEC_SERVE_CONCURRENCY  max parallel laudec calls (default 2)
    LAUDEC_SERVE_TIMEOUT      per-request seconds (default 300)
    LAUDEC_BIN                how to invoke laudec (default: this dir's laudec.py)
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["haiku", "sonnet", "opus"]


def _env(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def laudec_cmd():
    raw = os.environ.get("LAUDEC_BIN")
    if raw:
        return shlex.split(raw)
    return [sys.executable, os.path.join(HERE, "laudec.py")]


HOST = _env("LAUDEC_SERVE_HOST", "127.0.0.1")
PORT = int(_env("LAUDEC_SERVE_PORT", "8787"))
API_KEY = _env("LAUDEC_SERVE_API_KEY", "")
CONCURRENCY = int(_env("LAUDEC_SERVE_CONCURRENCY", "2"))
TIMEOUT = float(_env("LAUDEC_SERVE_TIMEOUT", "300"))

_slots = threading.BoundedSemaphore(max(1, CONCURRENCY))


# ── mapping helpers ───────────────────────────────────────────────────────────
def normalize_model(m):
    m = (m or "").lower()
    for k in MODELS:
        if k in m:
            return k
    return "sonnet"   # lenient default for unknown model names


EFFORTS = ("low", "medium", "high", "xhigh", "max")


def model_ids():
    """Every selectable id: each bare model, plus one entry per effort level.

    Clients that only have a model dropdown (and no reasoning_effort control)
    can therefore pick the effort by choosing e.g. `opus-high` from the list.
    """
    ids = []
    for m in MODELS:
        ids.append(m)
        ids.extend(f"{m}-{e}" for e in EFFORTS)
    return ids


def parse_model(m):
    """Split an incoming model id into (claude_model, effort_or_None).

    Accepts bare names (`opus`) and effort-suffixed names (`opus-high`). The
    suffix is the effort carried by selector-only clients; an explicit
    reasoning_effort field still overrides it in do_POST.
    """
    raw = (m or "").lower()
    effort = None
    for e in EFFORTS:
        if raw.endswith("-" + e):
            effort = e
            raw = raw[: -(len(e) + 1)]
            break
    return normalize_model(raw), effort


def normalize_effort(e):
    """Map the OpenAI reasoning_effort field to a Claude effort level, or None."""
    e = (e or "").lower()
    return e if e in EFFORTS else None


def msg_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content-parts (vision etc.), keep text
        return "\n".join(
            c.get("text", "") for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        )
    return "" if content is None else str(content)


def messages_to_prompt(messages):
    """Flatten an OpenAI messages[] into a single prompt for laudec.

    Tool-calling history is preserved as readable text so a fresh, tool-less
    Claude session still has the full context: an assistant turn that requested
    tools is rendered as `[Called tool <name>(<args>)]`, and each `tool` result
    is labelled with the tool it answers (resolved via tool_call_id)."""
    # Map tool_call_id -> tool name so tool results can name their origin.
    id2name = {}
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if tc.get("id"):
                id2name[tc["id"]] = (tc.get("function") or {}).get("name")

    sys_parts, convo = [], []
    for m in messages:
        role = m.get("role")
        text = msg_text(m.get("content")).strip()
        tool_calls = m.get("tool_calls") or []
        if role == "system":
            if text:
                sys_parts.append(text)
        elif role == "assistant":
            seg = [text] if text else []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args)
                seg.append(f"[Called tool {fn.get('name')}({args})]")
            if seg:
                convo.append("Assistant: " + "\n".join(seg))
        elif role == "tool":
            name = id2name.get(m.get("tool_call_id")) or m.get("name") or "tool"
            convo.append(f"Tool result ({name}): {text}")
        elif text:
            convo.append(f"User: {text}")
    out = ("\n\n".join(sys_parts) + "\n\n") if sys_parts else ""
    if len(convo) == 1 and convo[0].startswith("User: "):
        out += convo[0][len("User: "):]   # single turn → clean prompt, no label
    else:
        out += "\n\n".join(convo)
    return out.strip()


# ── tool calling (OpenAI functions over a text-only model) ─────────────────────
# laudec runs Claude with tools disabled, so it can only emit text. To still
# support OpenAI function calling, we teach Claude the available tools and a
# strict text contract for "calling" one, then parse that text back into the
# structured tool_calls the client expects.
TOOLCALL_TAG = "tool_call"


def tool_fn(t):
    """Pull the function spec out of one entry of an OpenAI `tools` array."""
    if not isinstance(t, dict):
        return {}
    fn = t.get("function") if t.get("type", "function") == "function" else t
    return fn if isinstance(fn, dict) else {}


def tool_names(tools):
    """The set of declared tool names — used to reject hallucinated calls."""
    return {n for n in (tool_fn(t).get("name") for t in (tools or [])) if n}


def tools_to_instructions(tools, tool_choice=None):
    """Render OpenAI tool definitions into a system instruction block.

    Teaches the text-only session to answer with a fenced ```tool_call JSON
    array when (and only when) it needs tools, so we can parse it back out."""
    lines = []
    for t in tools:
        fn = tool_fn(t)
        name = fn.get("name")
        if not name:
            continue
        desc = (fn.get("description") or "").strip().replace("\n", " ")
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        lines.append(f"- {name}: {desc}\n  parameters (JSON Schema): "
                     f"{json.dumps(params)}")
    forced = ""
    if isinstance(tool_choice, dict):
        want = (tool_choice.get("function") or {}).get("name")
        if want:
            forced = (f"\n- You MUST call the `{want}` tool on this turn; "
                      "emit only its tool_call block.")
    elif tool_choice == "required":
        forced = "\n- You MUST call at least one tool on this turn."
    return (
        "# Tool use\n"
        "You can call tools to help answer. When you need one or more tools, "
        "reply with ONLY a fenced code block tagged `" + TOOLCALL_TAG + "` "
        "holding a JSON array of calls, and nothing else:\n\n"
        "```" + TOOLCALL_TAG + "\n"
        '[{"name": "<tool_name>", "arguments": {<args matching the schema>}}]\n'
        "```\n\n"
        "Rules:\n"
        "- Only emit the block when you actually need a tool; otherwise just "
        "answer in plain text.\n"
        "- Use only the exact tool names listed below. Never invent a tool.\n"
        "- `arguments` must be a JSON object of real values, not the schema, "
        "and must satisfy the tool's parameter schema.\n"
        "- Emit strict JSON: double-quoted keys and strings, no trailing "
        "commas, no comments, no surrounding markdown beyond the one fence.\n"
        "- Request several tools at once by adding more objects to the one "
        "array (preferred), or by emitting several tool_call blocks back to "
        "back with nothing between them.\n"
        "- After the tool_call block(s), STOP. NEVER write, guess, or role-play "
        "the tool's output — your turn ends the moment you emit the block, and "
        "you will be called again with the real results before you continue.\n"
        "- When calling tools, output the block with no surrounding prose."
        + forced + "\n\n"
        "## Available tools\n" + "\n".join(lines)
    )


_FENCE_RE = re.compile(
    r"```(?:" + TOOLCALL_TAG + r"|tool_calls|json)?[^\S\n]*\n?(.*?)```",
    re.S | re.I)
# Claude's native function-call markup, used as a fallback. Tolerates an
# optional namespace prefix (e.g. `antml:`) on every tag.
_FC_RE = re.compile(r"<(?:\w+:)?function_calls>(.*?)</(?:\w+:)?function_calls>",
                    re.S | re.I)
_INVOKE_RE = re.compile(
    r'<(?:\w+:)?invoke\s+name="(.*?)"\s*>(.*?)</(?:\w+:)?invoke>', re.S | re.I)
_PARAM_RE = re.compile(
    r'<(?:\w+:)?parameter\s+name="(.*?)"\s*>(.*?)</(?:\w+:)?parameter>', re.S | re.I)


def _coerce_calls(obj, valid=None):
    """Normalize a parsed JSON value into [{name, arguments}, ...] or None.

    When `valid` (a set of declared tool names) is given, calls naming an
    unknown tool are dropped — this is what keeps an unrelated ```json block in
    a normal answer from being mistaken for a tool call."""
    if isinstance(obj, dict):
        if isinstance(obj.get("tool_calls"), list):
            obj = obj["tool_calls"]
        elif obj.get("name") or obj.get("function"):
            obj = [obj]
        else:
            return None
    if not isinstance(obj, list):
        return None
    out = []
    for c in obj:
        if not isinstance(c, dict):
            continue
        if isinstance(c.get("function"), dict):   # OpenAI-shaped entry
            fn = c["function"]
            name, args = fn.get("name"), fn.get("arguments", {})
        else:
            name = c.get("name")
            args = c.get("arguments", c.get("parameters", {}))
        if not name or (valid and name not in valid):
            continue
        out.append({"name": name, "arguments": args})
    return out or None


def _parse_fenced(text, valid=None):
    """Collect tool calls from leading fenced ```tool_call block(s).

    A text-only model tends to role-play the whole agentic loop in one reply
    (call → fabricated result → next call …), so only the calls *before* any
    real prose are genuine. We take the first tool-call block plus any further
    blocks separated from it by whitespace alone (legitimate parallel calls),
    and stop at the first block preceded by prose or that isn't a tool call.
    Prose before the first block is kept as content; everything after dropped."""
    calls, content, end = None, text, None
    for m in _FENCE_RE.finditer(text):
        try:
            parsed = _coerce_calls(json.loads(m.group(1).strip()), valid)
        except (ValueError, json.JSONDecodeError):
            parsed = None
        if not parsed:
            if calls:           # a non-call fence after calls → end of the run
                break
            continue            # still scanning for the first call block
        if calls is None:
            calls, content, end = parsed, text[:m.start()], m.end()
        elif text[end:m.start()].strip():   # prose between blocks → role-play
            break
        else:                               # back-to-back parallel call blocks
            calls += parsed
            end = m.end()
    return calls, content


def _parse_native(text, valid=None):
    """Fallback for Claude's native <function_calls> markup. Only the first
    block is real; keep prose before it and drop the rest."""
    m = _FC_RE.search(text)
    if not m:
        return None, text
    calls = []
    for name, inner in _INVOKE_RE.findall(m.group(1)):
        args = {k: v.strip() for k, v in _PARAM_RE.findall(inner)}
        name = name.strip()
        if name and not (valid and name not in valid):
            calls.append({"name": name, "arguments": args})
    if not calls:
        return None, text
    return calls, text[:m.start()]


def _parse_bare(text, valid=None):
    """Last resort: the whole reply is a bare JSON array/object of calls, with
    no fence and no prose."""
    s = text.strip()
    if not s or s[0] not in "[{":
        return None, text
    try:
        calls = _coerce_calls(json.loads(s), valid)
    except (ValueError, json.JSONDecodeError):
        return None, text
    return (calls, "") if calls else (None, text)


def _arg_string(args):
    """Coerce tool-call arguments into the JSON-object *string* OpenAI expects.

    Never emit raw, unparseable text: a client doing JSON.parse on the
    arguments must not choke, so anything we can't make sense of becomes `{}`."""
    if isinstance(args, dict):
        return json.dumps(args)
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return "{}"
        try:                       # already-stringified JSON → re-serialize clean
            return json.dumps(json.loads(s))
        except (ValueError, json.JSONDecodeError):
            return "{}"
    return "{}"                     # null / list / number → no usable arguments


def parse_tool_calls(text, valid=None):
    """Return (content, tool_calls) from a model reply.

    `valid` is the set of declared tool names (used to reject hallucinated or
    accidental calls). tool_calls is an OpenAI-format list (possibly empty);
    content is any prose left after removing the tool-call markup, or None when
    nothing meaningful remains (the OpenAI convention with finish_reason
    tool_calls)."""
    calls, rest = _parse_fenced(text, valid)
    if not calls:
        calls, rest = _parse_native(text, valid)
    if not calls:
        calls, rest = _parse_bare(text, valid)
    if not calls:
        return text, []
    oai = [{
        "id": "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": c["name"], "arguments": _arg_string(c.get("arguments"))},
    } for c in calls]
    content = (rest or "").strip()
    return (content or None), oai


def est_tokens(s):
    return max(1, len(s) // 4)


def run_laudec(prompt, model, timeout, effort=None):
    cmd = laudec_cmd() + ["-m", model]
    if effort:
        cmd += ["--effort", effort]
    with _slots:                       # backpressure: cap concurrent claude sessions
        p = subprocess.run(cmd, input=prompt, capture_output=True,
                           text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "").strip() or f"laudec exited {p.returncode}")
    return p.stdout.rstrip("\n")


# ── HTTP ──────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "laudec_serve/1.0"
    protocol_version = "HTTP/1.1"

    # -- low level --
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _err(self, code, msg, etype="invalid_request_error"):
        self._json(code, {"error": {"message": msg, "type": etype}})

    def _auth_ok(self):
        if not API_KEY:
            return True
        auth = self.headers.get("Authorization", "")
        return auth.startswith("Bearer ") and auth[7:].strip() == API_KEY

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[laudec_serve] {self.address_string()} {fmt % args}\n")

    # -- routes --
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health" or path == "":
            return self._json(200, {"status": "ok"})
        if path == "/v1/models":
            if not self._auth_ok():
                return self._err(401, "invalid api key", "authentication_error")
            data = [{"id": m, "object": "model", "created": 0, "owned_by": "laudec"}
                    for m in model_ids()]
            return self._json(200, {"object": "list", "data": data})
        return self._err(404, f"unknown path {path}")

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path != "/v1/chat/completions":
            return self._err(404, f"unknown path {path}")
        if not self._auth_ok():
            return self._err(401, "invalid api key", "authentication_error")
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._err(400, "invalid JSON body")

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._err(400, "'messages' must be a non-empty array")
        requested = body.get("model")
        model, suffix_effort = parse_model(requested)
        # explicit reasoning_effort wins; otherwise use the model-id suffix
        effort = normalize_effort(body.get("reasoning_effort")) or suffix_effort
        echo_model = requested if isinstance(requested, str) and requested else model
        prompt = messages_to_prompt(messages)
        if not prompt:
            return self._err(400, "no text content in messages")
        stream = bool(body.get("stream"))

        # Function calling: if the client offered tools (and didn't disable them
        # via tool_choice="none"), teach the text-only session how to request a
        # call, then parse the reply back into structured tool_calls below.
        tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        tool_choice = body.get("tool_choice")
        if tools and tool_choice != "none":
            prompt = tools_to_instructions(tools, tool_choice) + "\n\n" + prompt
        else:
            tools = []

        try:
            text = run_laudec(prompt, model, TIMEOUT, effort)
        except subprocess.TimeoutExpired:
            return self._err(504, "laudec timed out", "timeout")
        except Exception as e:  # noqa: BLE001
            return self._err(502, f"laudec failed: {e}", "upstream_error")

        if tools:
            content, tool_calls = parse_tool_calls(text, tool_names(tools))
        else:
            content, tool_calls = text, []

        if stream:
            self._stream(content, tool_calls, echo_model)
        else:
            self._completion(content, tool_calls, echo_model, prompt)

    # -- responses --
    def _completion(self, content, tool_calls, model, prompt):
        message = {"role": "assistant", "content": content}
        finish = "stop"
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish = "tool_calls"
        billed = (content or "") + (json.dumps(tool_calls) if tool_calls else "")
        self._json(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": est_tokens(prompt),
                "completion_tokens": est_tokens(billed),
                "total_tokens": est_tokens(prompt) + est_tokens(billed),
            },
        })

    def _stream(self, content, tool_calls, model):
        cid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        # No Content-Length on a stream, so signal completion by closing the
        # connection; otherwise an HTTP/1.1 keep-alive client hangs after [DONE].
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()

        def chunk(delta, finish=None):
            payload = {
                "id": cid, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        try:
            chunk({"role": "assistant"})
            if content:
                chunk({"content": content})   # emulated: whole answer in one chunk
            if tool_calls:
                # Each call streamed whole (name + full arguments) in one delta.
                deltas = [{
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                } for i, tc in enumerate(tool_calls)]
                chunk({"tool_calls": deltas})
                chunk({}, "tool_calls")
            else:
                chunk({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    global HOST, PORT, API_KEY, TIMEOUT
    # prog defaults to basename(sys.argv[0]); laudec.py rewrites that to
    # "laudec.py serve" when invoked as a subcommand, so help text stays accurate.
    ap = argparse.ArgumentParser(
        description="OpenAI-compatible API backed by laudec.")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--api-key", default=API_KEY, help="require this Bearer token")
    ap.add_argument("--timeout", type=float, default=TIMEOUT)
    args = ap.parse_args()
    HOST, PORT, API_KEY, TIMEOUT = args.host, args.port, args.api_key, args.timeout

    if HOST not in ("127.0.0.1", "localhost", "::1") and not API_KEY:
        sys.stderr.write(
            "WARNING: binding to a non-local host without an API key, which exposes "
            "your Claude subscription as an open endpoint. Set --api-key.\n")

    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    sys.stderr.write(
        f"[laudec_serve] listening on http://{HOST}:{PORT}  "
        f"models={','.join(MODELS)} ×{len(EFFORTS)} efforts "
        f"({len(model_ids())} ids)  concurrency={CONCURRENCY}  "
        f"auth={'on' if API_KEY else 'off'}\n"
        f"[laudec_serve] base_url: http://{HOST}:{PORT}/v1\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[laudec_serve] shutting down\n")
        srv.shutdown()


if __name__ == "__main__":
    main()
