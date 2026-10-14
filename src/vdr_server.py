"""Page retrieval server for the agent's `search` tool.

    POST /search {"domain": "hr", "query": "...", "k": 10, "with_text": false} -> {"results": [{"corpus_id", "score"[, "text"]}, ...]}
    POST /index  {"name": "visual/hr", "pages_zip": "<path>", "texts": "<jsonl path or null>"}   (evie only)
         -> {"pages": n, "encode_ms": ..., "mean_tokens": ...}; "name" is then usable as "domain" in /search
    POST /judge  {"domain": "energy", "query": "...", "ids": [...]} -> {"results": [{"corpus_id", "p_relevant",
         "prompt_tokens"}, ...]}: JEV-27B-VL page judgments (needs --jev-url; key from $JEV_API_KEY, so it never
         enters the agent's container)
    POST /region {"domain": "energy", "corpus_id": 930, "query": "...", "frac": 0.35}
         -> {"y0", "y1", "jpeg_b64"}: the full-width band (frac of the page height) whose patches match the query best
         (EVIE patch similarity, content query tokens only), cropped from the agent's 1600px page JPEG
         (needs --region-domains: patch grids come from the original page sizes)
    GET  /health

Backends
  evie  EVIE-4.5B page MaxSim (official scoring: all query tokens incl. augmentation, all page tokens).
        Needs a GPU and the EVIE colpali fork:  the EVIE env's python, PYTHONPATH=<evie>/colpali
        Page embeddings come from the cached H5 files (--domains) or are encoded live by /index from a zip of
        page images (<corpus_id>.<ext> members), exactly as the cache was built (NanoVDR-v2
        scripts/cache_teacher_mv.py: 1024 visual tokens, batch 4, tokens with norm > 0); queries are encoded live.
        A live index replaces all earlier live indexes, so GPU memory holds one corpus at a time.
        --emb-device cpu keeps cached page embeddings in host memory and scores on the CPU (the query encoder stays
        on the GPU): for sharing a MIG slice with a large judge model.
  bm25  BM25 over the OCR markdown -- CPU stand-in used only to test the tool plumbing.

    python src/vdr_server.py --backend evie --domains hr computer_science energy --port 0 --port-file run/port
    python src/vdr_server.py --backend evie --domains --port-file run/port      # live indexes only (vidore-benchmark)
"""
import argparse
import io
import json
import os
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("VDR_EMB_CACHE", f"{ROOT}/vidore_cache")  # cached EVIE page embeddings (--domains)
DATA = os.environ.get("VDR_DATA_DIR", f"{ROOT}/data")


def page_text(domain, cid):
    """ViDoRe-shipped OCR markdown of one page (the text NVIDIA's agentic pipeline shows its agent)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pagestore import page_texts
    return page_texts(domain).get(int(cid), "").strip()
SHORT = {"computer_science": "cs", "pharmaceuticals": "pharma"}


class EvieBackend:
    def __init__(self, domains, device="cuda", emb_device=None):
        import h5py
        import torch
        from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
        from colpali_engine.models.qwen3_5.colqwen3_5.modeling_colqwen3_5 import set_active_head
        self.torch = torch
        # cuDNN SDPA finds no execution plan for the bidirectional query mask on MIG H200 slices; use the other kernels
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.proc = ColQwen3_5Processor.from_pretrained("tencent/EVIE-4.5B", max_num_visual_tokens=1024)
        model = ColQwen3_5.from_pretrained("tencent/EVIE-4.5B", torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        model.enable_bidirectional_attention()
        set_active_head(model, 2048)  # same settings as the cache (scripts/cache_teacher_mv.py)
        self.model = model.to(device)
        self.device = device
        self.docs = {}
        edev = emb_device or device
        for d in domains:
            t = time.time()
            with h5py.File(f"{CACHE}/evie45_{SHORT.get(d, d)}_corpus.h5", "r") as f:
                lens = torch.from_numpy(f["doc_lengths"][:])
                emb = torch.empty(f["doc_emb"].shape, dtype=torch.float16, device=edev)
                for s in range(0, emb.shape[0], 64):  # stream to the GPU to keep host memory small
                    emb[s:s + 64] = torch.from_numpy(f["doc_emb"][s:s + 64]).to(edev)
            mask = torch.arange(emb.shape[1])[None, :] < lens[:, None]
            self.docs[d] = (emb, mask.to(edev), list(range(emb.shape[0])))  # H5 rows are corpus_ids 0..n-1
            print(f"loaded {d}: {tuple(emb.shape)} in {time.time() - t:.0f}s", flush=True)
        self.lock = threading.Lock()
        self.live = set()   # names created by /index
        self.texts = {}     # name -> {corpus_id: OCR text} for live indexes

    def index(self, name, pages_zip, texts=None, batch_size=4):
        """Encode every page image in pages_zip; store the padded embeddings on the GPU under `name`."""
        from PIL import Image
        torch = self.torch
        t0 = time.time()
        with self.lock:
            for old in self.live:  # one live corpus at a time (the largest, industrial, is ~22 GB in fp16)
                self.docs.pop(old, None)
                self.texts.pop(old, None)
            self.live.clear()
            torch.cuda.empty_cache()
            zf = zipfile.ZipFile(pages_zip)
            members = sorted(zf.namelist(), key=lambda m: int(os.path.splitext(m)[0]))
            ids = [int(os.path.splitext(m)[0]) for m in members]
            pages = []  # unpadded (n_i, D) fp16 on the CPU until the padded GPU tensor is built
            with torch.inference_mode():
                for s in range(0, len(members), batch_size):
                    imgs = [Image.open(io.BytesIO(zf.read(m))).convert("RGB") for m in members[s:s + batch_size]]
                    for obj in (self.model, getattr(self.model, "model", None)):
                        if obj is not None and hasattr(obj, "rope_deltas"):
                            obj.rope_deltas = None
                    b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in self.proc.process_images(imgs).items()}
                    out = self.model(**b)
                    h = out if isinstance(out, torch.Tensor) else out[0]
                    for v in h:
                        pages.append(v[v.float().norm(dim=-1) > 1e-6].half().cpu())
                    if (s // batch_size) % 100 == 0:
                        print(f"index {name}: {s + len(imgs)}/{len(members)} pages, {time.time() - t0:.0f}s", flush=True)
            lens = torch.tensor([p.shape[0] for p in pages])
            emb = torch.zeros((len(pages), int(lens.max()), pages[0].shape[1]), dtype=torch.float16, device=self.device)
            for i, p in enumerate(pages):
                emb[i, :p.shape[0]] = p.to(self.device)
            mask = (torch.arange(emb.shape[1])[None, :] < lens[:, None]).to(self.device)
            self.docs[name] = (emb, mask, ids)
            self.live.add(name)
            if texts:
                with open(texts) as fh:
                    self.texts[name] = {(r := json.loads(l))["corpus_id"]: r["markdown"] or "" for l in fh}
        info = {"pages": len(ids), "encode_ms": round(1000 * (time.time() - t0)),
                "mean_tokens": round(float(lens.float().mean()), 1), "shape": list(emb.shape)}
        print(f"indexed {name}: {info}", flush=True)
        return info

    def load_grids(self, domain):
        """(rows, cols) of EVIE's visual patch grid per corpus_id, from the original page sizes (Qwen-VL resize rule)."""
        import glob
        import math
        import pyarrow.parquet as pq
        from PIL import Image

        def smart_resize(h, w, factor=32, min_px=65536, max_px=1048576):
            hb, wb = round(h / factor) * factor, round(w / factor) * factor
            if hb * wb > max_px:
                beta = math.sqrt(h * w / max_px)
                hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
            elif hb * wb < min_px:
                beta = math.sqrt(min_px / (h * w))
                hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
            return hb // factor, wb // factor

        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        snap = glob.glob(f"{hub}/datasets--vidore--vidore_v3_{domain}/snapshots/*/")[0]
        grids = {}
        for f in sorted(glob.glob(snap + "corpus/*.parquet")):
            t = pq.read_table(f, columns=["corpus_id", "image"])
            for cid, img in zip(t.column("corpus_id").to_pylist(), t.column("image").to_pylist()):
                W, H = Image.open(io.BytesIO(img["bytes"])).size
                grids[int(cid)] = smart_resize(H, W)
        emb, mask, ids = self.docs[domain]
        lens = mask.sum(1).tolist()
        bad = [c for c, L in zip(ids, lens) if grids[c][0] * grids[c][1] != int(L) - 11]
        assert not bad, f"grid/token mismatch for {len(bad)} pages, e.g. {bad[:3]}"
        self.grids = getattr(self, "grids", {})
        self.grids[domain] = grids
        print(f"patch grids for {domain}: {len(grids)} pages, all match the cached token counts", flush=True)

    def region(self, domain, cid, query, frac):
        """Best full-width band of `frac` page height for the query, as fractions (y0, y1) of the page height."""
        torch = self.torch
        q = self.encode(query)[:-10]  # drop the 10 query padding tokens (they act as registers)
        emb, mask, ids = self.docs[domain]
        row = ids.index(int(cid))
        gh, gw = self.grids[domain][int(cid)]
        with torch.inference_mode():
            D = emb[row, 4:4 + gh * gw]
            heat = (q @ D.T).float().max(0).values.reshape(gh, gw).sum(1)  # per patch: best content token; per row: sum
            k = max(1, round(frac * gh))
            band = torch.nn.functional.conv1d(heat[None, None], torch.ones(1, 1, k, device=heat.device))[0, 0]
            r0 = int(band.argmax())
        return r0 / gh, (r0 + k) / gh

    def page_text(self, domain, cid):
        if domain in self.texts:
            return self.texts[domain].get(int(cid), "").strip()
        return page_text(domain, cid)

    def encode(self, text):
        torch = self.torch
        with self.lock, torch.inference_mode():
            for obj in (self.model, getattr(self.model, "model", None)):  # RoPE cache must not leak across calls
                if obj is not None and hasattr(obj, "rope_deltas"):
                    obj.rope_deltas = None
            b = {k: v.to(self.device) for k, v in self.proc.process_queries([text]).items()}
            out = self.model(**b)
            q = out[0] if not isinstance(out, torch.Tensor) else out[0]
            q = q[b["attention_mask"][0].bool()]
            return q[q.float().norm(dim=-1) > 1e-6].half()

    def scores(self, domain, q):
        torch = self.torch
        emb, mask, _ = self.docs[domain]
        out = []
        with torch.inference_mode():
            if emb.device.type == "cpu":  # host-memory embeddings: fp32 matmul in small chunks
                q = q.float().cpu()
                for s in range(0, emb.shape[0], 32):
                    S = torch.einsum("qd,nld->nql", q, emb[s:s + 32].float())
                    S = S.masked_fill(~mask[s:s + 32, None, :], float("-inf"))
                    out.append(S.max(-1).values.sum(-1))
                return torch.cat(out)
            for s in range(0, emb.shape[0], 256):
                S = torch.einsum("qd,nld->nql", q, emb[s:s + 256]).float()
                S = S.masked_fill(~mask[s:s + 256, None, :], float("-inf"))
                out.append(S.max(-1).values.sum(-1))
        return torch.cat(out).cpu()

    def search(self, domain, query, k):
        sc = self.scores(domain, self.encode(query))
        top = sc.topk(min(k, len(sc)))
        ids = self.docs[domain][2]
        return [{"corpus_id": ids[int(i)], "score": float(v)} for v, i in zip(top.values, top.indices)]


class BM25Backend:
    def __init__(self, domains):
        sys.path.insert(0, os.path.dirname(__file__))
        from bm25 import BM25, load_pages
        self.index = {d: BM25(load_pages(d)) for d in domains}

    def search(self, domain, query, k):
        return [{"corpus_id": int(c), "score": float(s)} for c, s in self.index[domain].search(query, k).items()]

    def page_text(self, domain, cid):
        return page_text(domain, cid)


def selfcheck(backend, args):
    """EVIE only: live-encoded original queries must reproduce the cached page ranking (patch_grep v1 baseline)."""
    res = json.load(open(args.selfcheck))
    for r in res:
        if r["domain"] not in backend.docs:
            continue
        got = [h["corpus_id"] for h in backend.search(r["domain"], r["query"], 10)]
        same = len(set(got) & set(r["baseline_page_top10"]))
        print(f"selfcheck {r['domain']} q{r['query_id']}: top10 overlap with cached ranking {same}/10, "
              f"top1 {got[0]} vs {r['baseline_page_top10'][0]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["evie", "bm25"], required=True)
    ap.add_argument("--domains", nargs="*", default=["hr", "computer_science", "energy"],
                    help="cached corpora to preload (may be empty: live /index only)")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--port-file", help="write the bound port here once ready")
    ap.add_argument("--selfcheck", help="patch_grep results.json to compare against (evie)")
    ap.add_argument("--emb-device", choices=["cuda", "cpu"], default="cuda", help="where cached page embeddings live (evie)")
    ap.add_argument("--jev-url", help="JEV-27B-VL server (bin/jev-serve) for /judge; key from $JEV_API_KEY")
    ap.add_argument("--region-domains", nargs="*", default=[], help="preloaded domains that serve /region (evie)")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    backend = EvieBackend(args.domains, emb_device=args.emb_device) if args.backend == "evie" else BM25Backend(args.domains)
    for d in args.region_domains:
        backend.load_grids(d)
    if args.selfcheck and args.backend == "evie":
        selfcheck(backend, args)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"ok": True, "backend": args.backend, "indexes": sorted(backend.docs) if hasattr(backend, "docs") else args.domains})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                if self.path.rstrip("/") == "/index":
                    if not isinstance(backend, EvieBackend):
                        raise ValueError("only the evie backend has live indexing")
                    self._send(200, backend.index(req["name"], req["pages_zip"], req.get("texts")))
                    return
                if self.path.rstrip("/") == "/region":
                    import base64
                    from PIL import Image
                    from pagestore import page_image_bytes
                    t = time.time()
                    y0, y1 = backend.region(req["domain"], req["corpus_id"], req["query"], float(req.get("frac", 0.35)))
                    im = Image.open(io.BytesIO(page_image_bytes(req["domain"], req["corpus_id"])))
                    crop = im.crop((0, round(y0 * im.height), im.width, round(y1 * im.height)))
                    buf = io.BytesIO()
                    crop.save(buf, "JPEG", quality=85)
                    print(json.dumps({"region": req["corpus_id"], "y0": round(y0, 3), "y1": round(y1, 3),
                                      "ms": int(1000 * (time.time() - t))}), flush=True)
                    self._send(200, {"y0": y0, "y1": y1, "jpeg_b64": base64.b64encode(buf.getvalue()).decode()})
                    return
                if self.path.rstrip("/") == "/judge":
                    if not args.jev_url:
                        raise ValueError("start the server with --jev-url to enable /judge")
                    from jev_client import judge_pages
                    t = time.time()
                    res = judge_pages(args.jev_url, os.environ.get("JEV_API_KEY", ""), req["domain"], req["query"],
                                      [int(c) for c in req["ids"]])
                    print(json.dumps({"judge": req["domain"], "n": len(res), "ms": int(1000 * (time.time() - t))}), flush=True)
                    self._send(200, {"results": res})
                    return
                t = time.time()
                hits = backend.search(req["domain"], req["query"], int(req.get("k", 10)))
                if req.get("with_text"):
                    for h in hits:
                        h["text"] = backend.page_text(req["domain"], h["corpus_id"])
                print(json.dumps({"domain": req["domain"], "query": req["query"][:200], "ms": int(1000 * (time.time() - t))}), flush=True)
                self._send(200, {"results": hits})
            except Exception as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    port = srv.server_address[1]
    if args.port_file:
        with open(args.port_file + ".tmp", "w") as fh:
            fh.write(str(port))
        os.replace(args.port_file + ".tmp", args.port_file)
    print(f"serving {args.backend} on 127.0.0.1:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
