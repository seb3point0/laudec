#!/usr/bin/env python3
"""
laudec drives the interactive Claude Code TUI inside a pseudo-terminal and
returns Claude's answer to the caller, as an alternative to `claude -p`.

The user's prompt is passed straight to `claude` (which auto-runs it on boot). A
`Stop` hook, registered via `--settings`, fires when Claude finishes its turn;
its payload contains `last_assistant_message`, which a tiny `laudec.py hook <id>`
mode writes into a FIFO that the parent reads to EOF. Tools are disabled by
default (`--tools ""`), so Claude acts as a pure text responder.

A single Python file with no dependencies.

Usage:
    laudec.py "your prompt"
    laudec.py -m sonnet "quick question"          # pick a model
    echo "data" | laudec.py "summarize this"      # stdin appended to prompt
    git diff | laudec.py "review this diff"
    laudec.py -b "..."                            # bypass: run `claude -p` instead
    laudec.py serve [--port N] [--api-key K]      # OpenAI-compatible HTTP server

Env overrides:
    LAUDEC_CMD          claude binary             (default: claude)
    LAUDEC_MODEL        model alias/name          (default: sonnet)
    LAUDEC_EFFORT       effort level              (default: medium)
    LAUDEC_TOOLS        --tools value             (default: empty, all tools off)
    LAUDEC_ARGS         extra args, space-split   (e.g. "--add-dir /x")
    LAUDEC_FAST         1=trim startup (default), 0=disable the fast flags
    LAUDEC_HOOK_CMD     override the Stop-hook command laudec registers
    LAUDEC_BOOT_TIMEOUT seconds to wait for first output (default: 30)
    LAUDEC_TIMEOUT      hard overall cap, seconds (default: 300)
    LAUDEC_RUNTIME_DIR  where FIFOs live (default: $XDG_RUNTIME_DIR or /tmp)

Exit codes: 0 ok · 2 boot timeout · 3 response timeout / no delivery · 4 hook error
"""
import argparse
import fcntl
import json
import os
import pty
import re
import secrets
import select
import shlex
import signal
import struct
import sys
import termios
import threading
import time

# ── config ──────────────────────────────────────────────────────────────────
def _env(name, default, cast=str):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    try:
        return cast(v)
    except (ValueError, TypeError):
        sys.stderr.write(f"laudec: ignoring invalid {name}={v!r}\n")
        return default


# Best-effort: make the kernel SIGTERM `claude` if laudec dies (Linux only), so a
# crash/kill of the wrapper never leaves an orphaned claude session running.
# Loaded in the parent so the forked child only makes one syscall.
_LIBC = None
if sys.platform.startswith("linux"):
    try:
        import ctypes
        _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
    except Exception:  # noqa: BLE001
        _LIBC = None
_PR_SET_PDEATHSIG = 1


CMD = _env("LAUDEC_CMD", "claude")
MODEL = _env("LAUDEC_MODEL", "sonnet")
EFFORT = _env("LAUDEC_EFFORT", "medium")
TOOLS = os.environ.get("LAUDEC_TOOLS", "")   # "" = disable all tools (pure text)
EXTRA_ARGS = shlex.split(os.environ.get("LAUDEC_ARGS", ""))
FAST = _env("LAUDEC_FAST", "1") not in ("0", "false", "no")
BOOT_TIMEOUT = _env("LAUDEC_BOOT_TIMEOUT", 30.0, float)
TIMEOUT = _env("LAUDEC_TIMEOUT", 300.0, float)
COLS, ROWS = 120, 40

# Startup-trimming flags that do NOT touch auth (so subscription billing still
# works). Note: --bare is deliberately NOT here; it forces ANTHROPIC_API_KEY
# auth and never reads OAuth/keychain, which would bill against the API and
# defeat the purpose of using the interactive session.
FAST_FLAGS = [
    "--strict-mcp-config",                    # skip loading configured MCP servers
    "--disable-slash-commands",               # skip loading skills/commands
    "--no-chrome",                            # skip Chrome integration probe
    "--exclude-dynamic-system-prompt-sections",  # better cross-call cache reuse
]

# claude subcommands. If a prompt's first token is one of these, we can't pass
# it as a positional (it'd be parsed as a command), so we type it instead.
RESERVED = {
    "agents", "auth", "auto-mode", "doctor", "install", "mcp", "plugin",
    "plugins", "project", "setup-token", "ultrareview", "update", "upgrade",
    "config", "resume", "commands", "migrate-installer",
}

PASTE_START = b"\x1b[200~"
PASTE_END = b"\x1b[201~"
SUBMIT = b"\r"

READY_RE = re.compile(r"\? for shortcuts")
# Folder-trust dialog. Claude's wording has changed across versions ("Do you
# trust the files" → "Quick safety check … trust this folder"), and after ANSI
# stripping the words can run together, so \s* tolerates missing spaces.
TRUST_RE = re.compile(
    r"trust\s*this\s*folder|trust\s*the\s*files|do\s*you\s*trust|safety\s*check",
    re.I)
WORKING_RE = re.compile(r"esc to interrupt", re.I)
ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"          # CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC
    r"|\x1b[()][AB0-2]"                   # charset
    r"|[\x00-\x08\x0b-\x1f\x7f]"          # other control chars (keep \n)
)

VERBOSE = False


def log(msg):
    if VERBOSE:
        sys.stderr.write(f"[laudec] {msg}\n")
        sys.stderr.flush()


def runtime_dir():
    return (os.environ.get("LAUDEC_RUNTIME_DIR")
            or os.environ.get("XDG_RUNTIME_DIR")
            or "/tmp")


def fifo_path(req_id):
    return os.path.join(runtime_dir(), f"laudec-{req_id}.fifo")


# ── hook mode: deliver Claude's final message into the FIFO ───────────────────
def run_hook(req_id):
    """`laudec.py hook <id>`. Claude's Stop hook calls this. Reads the hook
    payload on stdin and writes `last_assistant_message` into the FIFO."""
    path = fifo_path(req_id)
    if not os.path.exists(path):
        return 0   # request already finished or torn down; nothing to deliver
    try:
        obj = json.loads(sys.stdin.read() or "{}")
        msg = obj.get("last_assistant_message")
        data = (msg if isinstance(msg, str) else "").encode()
    except Exception:  # noqa: BLE001
        data = b""
    try:
        with open(path, "wb") as f:   # blocking open rendezvous with the reader
            f.write(data)
    except (BrokenPipeError, OSError):
        return 4
    return 0


def hook_cmd(req_id):
    return (os.environ.get("LAUDEC_HOOK_CMD")
            or f"{shlex.quote(sys.executable)} "
               f"{shlex.quote(os.path.abspath(__file__))} hook {req_id}")


def hook_settings(req_id):
    return json.dumps({"hooks": {"Stop": [
        {"hooks": [{"type": "command", "command": hook_cmd(req_id)}]}
    ]}})


# ── pty session ──────────────────────────────────────────────────────────────
def set_winsize(fd, rows, cols):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


class Session:
    def __init__(self, argv):
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.environ["TERM"] = "xterm-256color"
            if _LIBC is not None:
                try:
                    _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
                except Exception:  # noqa: BLE001
                    pass
            try:
                os.execvp(argv[0], argv)
            except FileNotFoundError:
                sys.stderr.write(f"laudec: command not found: {argv[0]}\n")
            os._exit(127)
        set_winsize(self.fd, ROWS, COLS)
        self.alive = True

    def read(self, timeout):
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return b""
        try:
            chunk = os.read(self.fd, 65536)
        except OSError:
            self.alive = False
            return b""
        if not chunk:
            self.alive = False
        return chunk

    def write(self, data):
        try:
            os.write(self.fd, data)
        except OSError:
            pass

    def paste(self, text):
        self.write(PASTE_START)
        self.write(text.encode())
        self.write(PASTE_END)
        time.sleep(0.25)
        self.write(SUBMIT)

    def close(self):
        for seq in (b"\x1b", b"\x1b", b"\x03"):
            self.write(seq)
            time.sleep(0.05)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(self.pid, sig)
                time.sleep(0.1)
            except OSError:
                break
        try:
            os.waitpid(self.pid, 0)
        except OSError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass


def strip(b):
    return ANSI_RE.sub("", b.decode("utf-8", "replace"))


def common_claude_flags():
    """Flags shared by normal and bypass mode: model, effort, the startup-trimming
    flags, and LAUDEC_ARGS. Kept in one place so the two modes can't drift."""
    flags = []
    if MODEL:
        flags += ["--model", MODEL]
    if EFFORT:
        flags += ["--effort", EFFORT]
    if FAST:
        flags += FAST_FLAGS
    return flags + EXTRA_ARGS


def build_argv(req_id, prompt, as_arg):
    # Register the Stop hook + disable tools (pure text responder, no perms needed).
    argv = [CMD, "--settings", hook_settings(req_id), "--tools", TOOLS]
    argv += common_claude_flags()
    if as_arg:
        argv += ["--", prompt]   # -- so a variadic flag can't swallow the prompt
    return argv


def wait_ready(s, deadline):
    """Only used in type-fallback mode: wait for the input box to appear."""
    buf, trusted, last, got = "", False, time.time(), False
    while time.time() < deadline:
        chunk = s.read(0.2)
        if not s.alive:
            return False
        if chunk:
            buf = (buf + strip(chunk))[-8000:]
            last, got = time.time(), True
        if TRUST_RE.search(buf) and not trusted:
            s.write(SUBMIT)
            trusted, buf = True, ""
            time.sleep(0.4)
            continue
        if READY_RE.search(buf) or (got and time.time() - last > 1.5 and not chunk):
            return True
    return False


# ── bypass mode: just run `claude -p` ────────────────────────────────────────
def run_bypass(prompt):
    """Skip all laudec logic and hand the prompt straight to `claude -p`.

    Replaces this process with `claude`, so its stdout/stderr and exit code pass
    through unchanged. model/effort/fast/LAUDEC_ARGS still apply; the laudec hook
    machinery does not.
    """
    argv = [CMD, "-p"] + common_claude_flags() + ["--", prompt]
    log("bypass: exec " + " ".join(shlex.quote(a) for a in argv))
    try:
        os.execvp(CMD, argv)
    except FileNotFoundError:
        sys.stderr.write(f"laudec: command not found: {CMD}\n")
        sys.exit(127)


# ── print mode (default) ──────────────────────────────────────────────────────
def run_print(prompt):
    req_id = secrets.token_hex(8)
    path = fifo_path(req_id)
    if os.path.exists(path):
        os.unlink(path)
    os.mkfifo(path, 0o600)

    result = {}
    done = threading.Event()

    def reader():
        try:
            with open(path, "rb") as f:   # blocks until the hook writes; reads to EOF
                result["data"] = f.read()
        except Exception as e:            # noqa: BLE001
            result["error"] = e
        finally:
            done.set()

    first_token = prompt.split(maxsplit=1)[0] if prompt.split() else ""
    as_arg = ("\n" not in prompt
              and not prompt.startswith("-")
              and first_token not in RESERVED
              and _env("LAUDEC_NO_ARGPROMPT", "0") in ("0", "false", "no"))

    s = Session(build_argv(req_id, prompt, as_arg))
    threading.Thread(target=reader, daemon=True).start()
    try:
        if not as_arg:
            if not wait_ready(s, time.time() + BOOT_TIMEOUT):
                sys.stderr.write("laudec: timed out waiting for Claude to start\n")
                return 2
            s.paste(prompt)
            log("prompt pasted")
        else:
            log("prompt passed as argument")

        start = time.time()
        deadline = start + TIMEOUT
        boot_deadline = start + BOOT_TIMEOUT
        saw_output = False
        saw_working = False
        trusted = False
        recent = ""

        while not done.is_set() and time.time() < deadline:
            chunk = s.read(0.2)
            now = time.time()
            if chunk:
                saw_output = True
                txt = strip(chunk)
                recent = (recent + txt)[-4000:]
                if WORKING_RE.search(txt):
                    saw_working = True
                if TRUST_RE.search(recent) and not trusted:
                    s.write(SUBMIT)         # accept folder-trust dialog
                    trusted, recent = True, ""
                    continue

            if done.is_set():
                break
            if not s.alive:
                if done.wait(1.0):
                    break
                sys.stderr.write("laudec: Claude exited before delivering a response\n")
                return 3
            if not saw_output and now > boot_deadline:
                sys.stderr.write("laudec: timed out waiting for Claude to start\n")
                return 2
            # Output appeared but Claude never started working (no spinner): almost
            # always a login/config problem. Fail fast instead of hanging to TIMEOUT.
            if saw_output and not saw_working and now > boot_deadline:
                sys.stderr.write("laudec: Claude started but never began working, "
                                 "likely not logged in or a config error "
                                 "(try running `claude` directly)\n")
                return 2

        if not done.is_set():
            sys.stderr.write("laudec: timed out waiting for the response\n")
            return 3
        if "error" in result:
            sys.stderr.write(f"laudec: read error: {result['error']}\n")
            return 3

        data = result.get("data", b"")
        sys.stdout.buffer.write(data)
        if not data.endswith(b"\n"):
            sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
        return 0
    finally:
        s.close()
        try:
            os.unlink(path)
        except OSError:
            pass


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global VERBOSE, MODEL, EFFORT

    if len(sys.argv) >= 3 and sys.argv[1] == "hook":
        sys.exit(run_hook(sys.argv[2]))

    # `laudec serve [...]` → run the OpenAI-compatible HTTP server. Intercepted
    # before argparse so its flags (--port etc.) aren't parsed as a prompt; the
    # server module is imported lazily so normal prompt runs don't pay for it.
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        import laudec_serve
        sys.argv = [f"{sys.argv[0]} serve"] + sys.argv[2:]
        laudec_serve.main()
        return

    # Ensure SIGTERM unwinds through run_print's finally (cleanup + kill claude).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    ap = argparse.ArgumentParser(
        prog="laudec",
        description="One-shot prompt/response against the interactive Claude TUI, "
                    "delivered via a Stop hook.",
    )
    ap.add_argument("prompt", nargs="*", help="prompt text (stdin is appended)")
    ap.add_argument("-m", "--model", default=None,
                    help="model alias or full name (e.g. sonnet, opus, haiku)")
    ap.add_argument("--effort", default=None,
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="effort level for the session")
    ap.add_argument("-b", "--bypass", action="store_true",
                    help="skip laudec entirely and run `claude -p` directly")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="diagnostics on stderr")
    args = ap.parse_args()
    VERBOSE = args.verbose
    if args.model:
        MODEL = args.model
    if args.effort:
        EFFORT = args.effort

    parts = []
    if args.prompt:
        parts.append(" ".join(args.prompt))
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            parts.append(piped)
    prompt = "\n\n".join(parts).strip()
    if not prompt:
        ap.error("no prompt given (pass as args and/or via stdin)")

    if args.bypass:
        run_bypass(prompt)   # exec's `claude -p`; does not return
    sys.exit(run_print(prompt))


if __name__ == "__main__":
    main()
