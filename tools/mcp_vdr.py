#!/usr/bin/env python3
"""Minimal MCP stdio server (stdlib only) exposing a page retriever to Codex, vision-only.

    python3 /tools/mcp_vdr.py --url http://127.0.0.1:PORT --domain hr --corpus /corpus

Tools
  search(query, k=10)   -> top-k pages from the retrieval server: corpus_id, score, document
  get_page(corpus_id)   -> the page IMAGE (no OCR text anywhere)

/corpus is the image-only corpus (data/<domain>/corpus_img). Codex spawns MCP servers itself,
outside the agent's shell sandbox, so this process can reach the retrieval server on loopback.
Each call is appended to $VDR_LOG when set.
"""
import argparse
import base64
import json
import os
import sys
import urllib.request
import zipfile

# Each viewed page stays in the context as a ~350 KB base64 image; the gateway rejects requests above
# ~16 MB (HTTP 413, seen at ~45 images), so views per episode are capped (--max-views, default 35).

TOOLS = [
    {"name": "search",
     "description": "Search the document collection with a visual document retriever that scores page images "
                    "against the query text. Returns the top-k pages, best first: corpus_id, score and document. You can rewrite the question into focused queries and call this several times.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "search query text"},
         "k": {"type": "integer", "description": "number of pages to return (1-20, default 10)"}},
         "required": ["query"]}},
    {"name": "get_page",  # image mode only
     "description": "Look at one page: returns the rendered page image, so you can check whether it really "
                    "answers the question.",
     "inputSchema": {"type": "object", "properties": {
         "corpus_id": {"type": "integer", "description": "page id from search results"}},
         "required": ["corpus_id"]}},
]


def load_docs(corpus):
    """corpus_id -> doc_id"""
    out = {}
    with open(f"{corpus}/documents.tsv") as fh:
        next(fh)
        for line in fh:
            doc, _, first, last = line.rstrip("\n").split("\t")
            for c in range(int(first), int(last) + 1):
                out[c] = doc
    return out



def call_tool(args, docs, name, a):
    if name == "search":
        k = max(1, min(20, int(a.get("k") or 10)))
        req = urllib.request.Request(f"{args.url}/search", method="POST",
                                     data=json.dumps({"domain": args.domain, "query": a["query"], "k": k,
                                                      "with_text": False}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            hits = json.load(r)["results"]
        lines = [f"{i}. corpus_id={h['corpus_id']} score={h['score']:.2f} doc={docs.get(h['corpus_id'], '?')}"
                 for i, h in enumerate(hits, 1)]
        content = [{"type": "text", "text": "\n".join(lines) or "no results"}]
        return content, {"query": a["query"], "k": k, "ids": [h["corpus_id"] for h in hits]}
    if name == "get_page" and args.mode == "image":
        if args.views >= args.max_views:
            raise RuntimeError(f"page view limit ({args.max_views}) reached; give your final ranking now")
        cid = int(a["corpus_id"])
        if os.path.exists(f"{args.corpus}/images.zip"):  # packed corpus: one zip per domain
            if not hasattr(args, "zip"):
                args.zip = zipfile.ZipFile(f"{args.corpus}/images.zip")
            raw = args.zip.read(f"{cid}.jpg")
        else:
            with open(f"{args.corpus}/images/{cid}.jpg", "rb") as fh:
                raw = fh.read()
        data = base64.b64encode(raw).decode()
        content = [{"type": "text", "text": f"corpus_id={cid} doc={docs.get(cid, '?')}"},
                   {"type": "image", "data": data, "mimeType": "image/jpeg"}]
        args.views += 1
        return content, {"corpus_id": cid}
    raise ValueError(f"unknown tool {name}")


def log(rec):
    p = os.environ.get("VDR_LOG")
    if p:
        try:
            with open(p, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError as e:  # logging is best effort; never fail the tool call over it
            sys.stderr.write(f"VDR_LOG write failed: {e}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--corpus", default="/corpus")
    ap.add_argument("--max-views", type=int, default=35)
    ap.add_argument("--mode", choices=["image"], default="image",
                    help="image: search returns ids, get_page returns the page image")
    args = ap.parse_args()
    args.views = 0
    docs = load_docs(args.corpus)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:  # notifications need no reply
            continue
        try:
            if method == "initialize":
                res = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2025-06-18"),
                       "capabilities": {"tools": {}}, "serverInfo": {"name": "vdr", "version": "0.2"}}
            elif method == "tools/list":
                res = {"tools": TOOLS}
            elif method == "tools/call":
                p = msg["params"]
                try:
                    content, meta = call_tool(args, docs, p["name"], p.get("arguments") or {})
                    log({"tool": p["name"], **meta})
                    res = {"content": content, "isError": False}
                except Exception as e:  # tool errors go back to the model as text
                    log({"tool": p["name"], "error": f"{type(e).__name__}: {e}"})
                    res = {"content": [{"type": "text", "text": f"error: {type(e).__name__}: {e}"}], "isError": True}
            elif method == "ping":
                res = {}
            else:
                raise KeyError(method)
            out = {"jsonrpc": "2.0", "id": mid, "result": res}
        except Exception as e:
            out = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": str(e)}}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
