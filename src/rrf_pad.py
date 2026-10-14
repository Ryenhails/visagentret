"""Fill an agent's final ranking to 10 with the RRF fusion of its own search results.

The agent often returns fewer than 10 pages; the empty slots are filled, in order, with pages
from Reciprocal Rank Fusion (k=60) over every `search` tool result in the episode. No extra LLM
calls. Also used to re-score runs that predate this default:

    python src/rrf_pad.py runs/glm53_vdrtool_cs50/computer_science   # adds ranking_raw, rewrites ranking
"""
import collections
import glob
import json
import os
import re
import sys

HIT = re.compile(r"corpus_id=(\d+)")


def search_results(events_path):
    out = []
    if not os.path.exists(events_path):
        return out
    for line in open(events_path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        it = ev.get("item") or {}
        if ev.get("type") == "item.completed" and it.get("type") == "mcp_tool_call" and it.get("tool") == "search":
            text = "\n".join(c.get("text", "") for c in ((it.get("result") or {}).get("content") or []) if c.get("type") == "text")
            out.append([m for m in HIT.findall(text)])
    return out


def rrf_pad(ranking, events_path, k=10, c=60):
    rrf = collections.defaultdict(float)
    for res in search_results(events_path):
        for i, cid in enumerate(res):
            rrf[cid] += 1 / (c + i + 1)
    out = [str(x) for x in ranking]
    for cid in sorted(rrf, key=lambda x: -rrf[x]):
        if len(out) >= k:
            break
        if cid not in out:
            out.append(cid)
    return out[:k]


def main():
    for d in sys.argv[1:]:
        n = changed = 0
        for p in glob.glob(f"{d}/q*/result.json"):
            r = json.load(open(p))
            raw = r.get("ranking_raw", r["ranking"])
            padded = rrf_pad(raw, f"{os.path.dirname(p)}/events.jsonl")
            r["ranking_raw"], r["ranking"], r["rrf_pad"] = raw, padded, True
            json.dump(r, open(p, "w"), indent=1, ensure_ascii=False)
            n += 1
            changed += len(padded) != len(raw)
        print(f"{d}: {n} results, {changed} padded")


if __name__ == "__main__":
    main()
