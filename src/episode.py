"""One Codex episode: the agent answers one query with a ranked list of corpus_ids.

Used by visagentret_pipeline.py (vidore-benchmark BasePipeline).
Stdlib only, so it runs in any environment.

<run_dir>/
    events.jsonl       codex exec --json trajectory
    calls.jsonl        one line per LLM call (usage, latency) from the proxy
    tool_calls.jsonl   one line per MCP tool call
    stderr.log         only if non-empty
    result.json        parsed ranking + usage + timing (presence = done; reruns skip it)
    failed.json        instead of result.json when the launcher/LLM server failed (no completed turn)
"""
import json
import os
import re
import signal
import subprocess
import time

from rrf_pad import rrf_pad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root: bin/, prompts/, tools/
SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}, "maxItems": 10}},
    "required": ["ranking"],
    "additionalProperties": False,
}
# Episode settings and their defaults (the pipeline sets the ones it uses).
DEFAULTS = {
    "corpus_dir": None,       # mounted read-only at /corpus: documents.tsv, images.zip
    "domain": None,           # retrieval-server index name
    "vdr_url": None,          # retrieval server for the MCP tools (None = no tools)
    "tool_mode": "image",     # search + get_page (page image)
    "max_views": 35,
    "model": None,
    "prompt": None,           # prompt name (recorded only)
    "timeout": 900,
    "llm_base_path": "/api/v1",
    "rrf_pad": True,
    "schema": False,
    "dump_requests": False,
    "dump_first_request": False,
    "corpus_label": None,     # recorded only
}


def load_valid_ids(corpus_dir):
    """All corpus_ids listed in <corpus_dir>/documents.tsv (as strings)."""
    ids = set()
    with open(f"{corpus_dir}/documents.tsv") as fh:
        next(fh)
        for line in fh:
            _, _, first, last = line.rstrip("\n").split("\t")
            ids.update(str(c) for c in range(int(first), int(last) + 1))
    return ids


def parse_ranking(text, valid_ids):
    """Final message -> list of valid corpus_ids (JSON first, then the last [...] in the text)."""
    cands = []
    try:
        cands = json.loads(text)["ranking"]
    except (ValueError, KeyError, TypeError):
        lists = re.findall(r"\[([^\[\]]*)\]", text)
        if lists:
            cands = re.findall(r"\d+", lists[-1])
    out = []
    for c in cands:
        c = str(c).strip()
        if c in valid_ids and c not in out:
            out.append(c)
    return out[:10], bool(cands)


def summarize_events(path):
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    n_cmd = n_turns = 0
    n_mcp = {}
    errors = []
    with open(path) as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            t = ev.get("type", "")
            if t == "turn.completed":
                n_turns += 1
                for k, v in (ev.get("usage") or {}).items():
                    usage[k] = usage.get(k, 0) + (v or 0)
            elif t == "item.completed" and (ev.get("item") or {}).get("type") == "command_execution":
                n_cmd += 1
            elif t == "item.completed" and (ev.get("item") or {}).get("type") == "mcp_tool_call":
                tool = (ev["item"].get("tool") or "?")
                n_mcp[tool] = n_mcp.get(tool, 0) + 1
            elif t in ("error", "turn.failed"):
                errors.append(str(ev)[:500])
    return usage, n_cmd, n_turns, errors, n_mcp


def summarize_calls(path):
    """Per-episode token totals from the proxy log (one line per LLM call; present even on timeout)."""
    tot = {"llm_calls": 0, "failed_calls": 0, "input": 0, "cached_input": 0, "output": 0, "reasoning": 0}
    if not os.path.exists(path):
        return tot
    with open(path) as fh:
        for line in fh:
            c = json.loads(line)
            tot["llm_calls"] += 1
            if c.get("status") != 200:
                tot["failed_calls"] += 1
            u = c.get("usage") or {}
            tot["input"] += u.get("input_tokens", 0) or 0
            tot["cached_input"] += (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
            tot["output"] += u.get("output_tokens", 0) or 0
            tot["reasoning"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    return tot


def page_views(path):
    """Successful get_page calls in an episode's tool_calls.jsonl."""
    if not os.path.exists(path):
        return 0
    with open(path) as fh:
        return sum(1 for l in fh if (c := json.loads(l)).get("tool") == "get_page" and "error" not in c)


def run_episode(query_id, query, run_dir, cfg, proxy, template, valid_ids):
    """Run one episode into run_dir (must be a real path: the Codex container binds only resolved dirs).

    cfg: dict of DEFAULTS keys; proxy: a started GwProxy owned by the caller for the episode's duration.
    Returns (result dict, ok); ok=False means an infrastructure failure (failed.json, retry later).
    """
    cfg = {**DEFAULTS, **cfg}
    os.makedirs(run_dir, exist_ok=True)
    prompt = template.format(query=query)  # stored in result.json, not as a separate file
    if cfg["schema"]:
        with open(f"{run_dir}/.answer_schema.json", "w") as fh:
            json.dump(SCHEMA, fh)

    cmd = [f"{ROOT}/bin/codex-gw", run_dir, "--json", "-o", f"{run_dir}/last_message.txt"]
    if cfg["model"]:
        cmd += ["-m", cfg["model"]]
    if cfg["schema"]:  # off by default: with GLM-5.3-Flash it suppresses all tool calls
        cmd += ["--output-schema", f"{run_dir}/.answer_schema.json"]
    env = dict(os.environ, CODEX_GW_CORPUS=cfg["corpus_dir"], CODEX_GW_PORT=str(proxy.port),
               CODEX_GW_TOKEN=proxy.token, CODEX_GW_BASE_PATH=cfg["llm_base_path"])
    if cfg["vdr_url"]:  # retriever tool via MCP (tools/mcp_vdr.py); the server runs on this node's loopback
        env["CODEX_GW_TOOLS"] = f"{ROOT}/tools"
        mcp_args = ["/tools/mcp_vdr.py", "--url", cfg["vdr_url"], "--domain", cfg["domain"], "--corpus", "/corpus",
                    "--max-views", str(cfg["max_views"]), "--mode", cfg["tool_mode"]]
        cmd += ["-c", 'mcp_servers.vdr.command="python3"',
                "-c", "mcp_servers.vdr.args=" + json.dumps(mcp_args),
                "-c", 'mcp_servers.vdr.env={VDR_LOG="%s/tool_calls.jsonl"}' % run_dir,
                "-c", "mcp_servers.vdr.tool_timeout_sec=180",
                "-c", 'mcp_servers.vdr.default_tools_approval_mode="approve"']  # else "never" policy rejects every call
    cmd += ["-"]

    proxy.log_path = f"{run_dir}/calls.jsonl"
    if cfg["dump_first_request"]:
        proxy.dump_path = f"{run_dir}/first_request.json"
    if cfg["dump_requests"]:
        proxy.dump_dir = f"{run_dir}/requests"
    t0 = time.time()
    timed_out = False
    with open(f"{run_dir}/events.jsonl", "w") as out, open(f"{run_dir}/stderr.log", "w") as err:
        # own process group, so a timeout takes down apptainer's helpers (squashfuse) too
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True,
                                env=env, start_new_session=True)
        try:
            proc.communicate(prompt, timeout=cfg["timeout"])
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = None, True
            for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 10)):
                try:
                    os.killpg(proc.pid, sig)
                    proc.wait(timeout=wait)
                    break
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    continue
    wall = time.time() - t0
    proxy.log_path = proxy.dump_dir = None

    # keep per-episode files few: last message and prompt go into result.json, empty stderr is dropped
    last = ""
    if os.path.exists(f"{run_dir}/last_message.txt"):
        with open(f"{run_dir}/last_message.txt") as fh:
            last = fh.read()
        os.remove(f"{run_dir}/last_message.txt")
    if os.path.exists(f"{run_dir}/stderr.log") and os.path.getsize(f"{run_dir}/stderr.log") == 0:
        os.remove(f"{run_dir}/stderr.log")
    ranking_raw, parsed = parse_ranking(last, valid_ids)
    # default: fill empty slots up to 10 from the RRF of the episode's own search results (no LLM calls)
    ranking = rrf_pad(ranking_raw, f"{run_dir}/events.jsonl") if cfg["rrf_pad"] else ranking_raw
    usage, n_cmd, n_turns, errors, n_mcp = summarize_events(f"{run_dir}/events.jsonl")
    res = {"query_id": str(query_id), "query": query, "ranking": ranking, "ranking_raw": ranking_raw,
           "rrf_pad": cfg["rrf_pad"], "parsed": parsed,
           "exit_code": rc, "timed_out": timed_out, "wall_s": round(wall, 1), "usage": usage,
           "tokens": summarize_calls(f"{run_dir}/calls.jsonl"),
           "n_commands": n_cmd, "n_tool_calls": n_mcp, "n_turns": n_turns, "errors": errors,
           "corpus": cfg["corpus_label"], "model": cfg["model"] or "config default", "prompt": cfg["prompt"],
           "schema": cfg["schema"], "prompt_text": prompt, "last_message": last,
           "max_views": cfg["max_views"] if cfg["vdr_url"] else None,
           "tool_mode": cfg["tool_mode"] if cfg["vdr_url"] else None}
    # No completed turn and a non-zero exit = the LLM server/launcher failed, not the agent:
    # keep it out of result.json so a rerun retries the query.
    infra_fail = rc not in (0, None) and n_turns == 0
    with open(f"{run_dir}/{'failed' if infra_fail else 'result'}.json", "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"q{query_id}: rc={rc} {wall:.0f}s turns={n_turns} cmds={n_cmd} tools={n_mcp} "
          f"tokens={res['tokens']['input']}in/{res['tokens']['output']}out/{res['tokens']['llm_calls']}calls "
          f"ranking={ranking}{' TIMEOUT' if timed_out else ''}{' INFRA-FAIL' if infra_fail else ''}", flush=True)
    if infra_fail:
        tail = open(f"{run_dir}/stderr.log").read()[-800:] if os.path.exists(f"{run_dir}/stderr.log") else ""
        print(f"  q{query_id} stderr tail:", tail, "\n  last error:", errors[-1:], flush=True)
    return res, not infra_fail
