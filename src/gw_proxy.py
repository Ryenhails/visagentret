"""Localhost proxy that adds the LLM server's key, so the key never enters the Codex container.

Codex (in the container) talks to http://127.0.0.1:<port>/v1 with a random
per-run session token; this process checks the token, swaps in the real key and
streams the server's reply (SSE) back.  The agent's shell runs in Codex's
network-less sandbox, so it cannot reach the proxy even though it can see the
token; other users on the node can reach the port but do not know the token.

Started by visagentret_pipeline.py:  GwProxy().start() -> (port, token), also .port/.token;  .stop()
"""
import http.client
import http.server
import json
import os
import secrets
import threading
import time

import urllib.parse

UPSTREAM = "http://127.0.0.1:8000"  # default; the pipeline passes its LLM server
KEY_FILE = os.environ.get("CODEX_GW_KEY_FILE")  # read only when no key_env is given
TAIL = 1 << 20  # bytes of each reply kept to read its usage
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host", "authorization",
       "proxy-connection", "upgrade", "te", "trailer"}


def call_usage(tail):
    """Usage from the end of a Responses API reply (SSE stream or plain JSON)."""
    for line in reversed(tail.decode("utf-8", "replace").splitlines()):
        if line.startswith("data:") and '"response.completed"' in line:
            try:
                return json.loads(line[5:])["response"].get("usage")
            except (ValueError, KeyError):
                return None
    try:
        return json.loads(tail).get("usage")
    except ValueError:
        return None


def make_handler(key, token, proxy, upstream):
    up = urllib.parse.urlsplit(upstream)
    conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # body ends at connection close: simple and SSE-safe

        def log_message(self, *a):  # keep stderr quiet
            pass

        def _proxy(self):
            if self.headers.get("Authorization", "") != f"Bearer {token}":
                self.send_error(403)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            proxy.dump_request(body)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            headers["Authorization"] = f"Bearer {key}"
            conn = conn_cls(up.netloc, timeout=900)
            t0, status, tail = time.time(), None, bytearray()
            try:
                conn.request(self.command, self.path, body=body or None, headers=headers)
                resp = conn.getresponse()
                status = resp.status
                self.send_response(resp.status, resp.reason)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.end_headers()
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    tail += chunk
                    if len(tail) > 2 * TAIL:  # amortised trim; usage sits in the last event
                        del tail[:-TAIL]
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # upstream unreachable / timeout -> 502 if nothing sent yet
                try:
                    self.send_error(502, f"{type(e).__name__}")
                except Exception:
                    pass
            finally:
                conn.close()
                proxy.log_call({"t": round(t0, 2), "path": self.path, "status": status,
                                "latency_s": round(time.time() - t0, 2), "usage": call_usage(bytes(tail))})

        do_GET = do_POST = _proxy

    return Handler


class GwProxy:
    """log_path: per-call usage log (no prompt/response content); the episode points it
    at its calls.jsonl."""
    log_path = None
    dump_path = None  # if set, the next request body is saved here once (debugging what Codex sends)

    dump_dir = None  # if set, every request body is saved there (req_000.json, ...)

    def dump_request(self, body):
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
            with open(f"{self.dump_dir}/req_{len(os.listdir(self.dump_dir)):03d}.json", "wb") as fh:
                fh.write(body)
        if self.dump_path:
            path, self.dump_path = self.dump_path, None
            with open(path, "wb") as fh:
                fh.write(body)

    def log_call(self, rec):
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def __init__(self, upstream=UPSTREAM, key_env=None):
        """key_env: name of an environment variable holding the upstream key (local vLLM);
        default: read the key from the file named by $CODEX_GW_KEY_FILE."""
        self.upstream, self.key_env = upstream, key_env

    def start(self):
        if self.key_env:
            key = os.environ.get(self.key_env, "")
        else:
            with open(KEY_FILE) as fh:
                key = fh.read().strip()
        self.token = secrets.token_hex(24)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(key, self.token, self, self.upstream))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]
        return self.port, self.token

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
