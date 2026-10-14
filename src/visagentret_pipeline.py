"""VisAgentRet as a vidore-benchmark pipeline (BasePipeline): an LLM agent (Codex harness) answers each query by
ReAct-style search over a late-interaction page retriever (EVIE-4.5B), then returns its top-10 pages.

    vidore-benchmark pipeline evaluate --dataset-name vidore/vidore_v3_hr --language english \
        --module-path src/visagentret_pipeline.py --class-name VisAgentRetPipeline \
        --pipeline-args '{"arm": "visual", "llm_model": "zai-org/GLM-5.3-Flash", "num_concurrent": 16}'

Tools (arm "visual"): search(query, k) -> page ids + scores; get_page(corpus_id) -> the page IMAGE (capped views)

index(): built from the corpus the framework passes in (so indexing time is real):
  - every page image, losslessly (PNG), is encoded by the EVIE server (same protocol as the EVIE cache:
    1024 visual tokens, all non-zero tokens kept), replacing any earlier corpus on the GPU;
  - pages as <=1600px JPEGs in images.zip for get_page.
search(): one Codex episode per query, num_concurrent at a time. The agent's ranking is filled to 10 by RRF over
  its own search results; an episode that ends with no page at all falls back to one retriever call on the
  original query. Episodes that fail for infrastructure reasons are retried; if the LLM server is down the
  run aborts (no metrics written) and finished episodes are reused on the next run.

Deployment endpoints come from the environment (they are not part of the method, so not pipeline args):
  VISAGENTRET_LLM_URL        OpenAI-compatible server root, e.g. http://127.0.0.1:8000 (Responses API)
  VISAGENTRET_LLM_BASE_PATH  API prefix (default /v1)
  VISAGENTRET_LLM_KEY_ENV    name of the env var holding its key (default VLLM_API_KEY)
  VISAGENTRET_RETRIEVER_URL  src/vdr_server.py --backend evie
  VISAGENTRET_RUN_DIR        episode traces: <dir>/<domain>/q<qid>/ (kept for analysis)
  VISAGENTRET_WORK_DIR       node-local scratch for the exported corpus (default /tmp/$USER_visagentret_$SLURM_JOB_ID)
"""
import concurrent.futures as cf
import io
import json
import os
import queue
import shutil
import sys
import threading
import time
import urllib.request
import zipfile

from PIL import Image
from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from episode import ROOT, page_views, run_episode  # noqa: E402
from gw_proxy import GwProxy  # noqa: E402

PROMPTS = {"visual": "task_tool_v1"}
TOOL_MODE = {"visual": "image"}
README = """# Corpus

{n} pages from {d} PDF documents (ViDoRe v3, {domain}). Every page has an integer id (`corpus_id`).

- `documents.tsv`: doc_id, number of pages, first and last corpus_id (pages of a document are consecutive).
{images}"""


def _post(url, payload, timeout):
    req = urllib.request.Request(url, method="POST", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class VisAgentRetPipeline(BasePipeline):
    def __init__(self, arm="visual", llm_model="zai-org/GLM-5.3-Flash", retriever_model="tencent/EVIE-4.5B",
                 num_concurrent=16, target_top_k=10, max_page_views=20, episode_timeout_s=900, max_retries=2,
                 rrf_fill=True, page_max_side=1600, page_jpeg_quality=85, query_limit=None):
        if arm not in PROMPTS:
            raise ValueError(f"arm must be one of {sorted(PROMPTS)}")
        if retriever_model != "tencent/EVIE-4.5B":
            raise ValueError("the retrieval server implements tencent/EVIE-4.5B only")
        self.arm, self.llm_model, self.retriever_model = arm, llm_model, retriever_model
        self.num_concurrent, self.top_k, self.max_views = int(num_concurrent), int(target_top_k), int(max_page_views)
        self.timeout, self.max_retries, self.rrf_fill = int(episode_timeout_s), int(max_retries), bool(rrf_fill)
        self.page_max_side, self.jpeg_quality = int(page_max_side), int(page_jpeg_quality)
        self.query_limit = query_limit  # debugging only: run the first N queries, the rest score 0
        env = os.environ
        self.llm_url = env["VISAGENTRET_LLM_URL"].rstrip("/")
        self.llm_base_path = env.get("VISAGENTRET_LLM_BASE_PATH", "/v1")
        self.llm_key_env = env.get("VISAGENTRET_LLM_KEY_ENV", "VLLM_API_KEY")
        self.retriever_url = env["VISAGENTRET_RETRIEVER_URL"].rstrip("/")
        self.run_dir = os.path.realpath(env["VISAGENTRET_RUN_DIR"])  # the Codex container binds resolved paths
        self.work_dir = env.get("VISAGENTRET_WORK_DIR") or f"/tmp/{env.get('USER', 'u')}_visagentret_{env.get('SLURM_JOB_ID', 'x')}"
        with open(f"{ROOT}/prompts/{PROMPTS[arm]}.md") as fh:
            self.template = fh.read()

    # ------------------------------------------------------------------ index
    def index(self, corpus_ids, corpus_images, corpus_texts, dataset_name=None):
        if not dataset_name:
            raise ValueError("dataset_name is required (passed by vidore-benchmark's evaluate_retrieval)")
        t0 = time.time()
        self.domain = dataset_name.split("/")[-1].removeprefix("vidore_v3_")
        self.index_name = f"{self.arm}/{self.domain}"
        base = f"{self.work_dir}/{self.arm}/{self.domain}"
        shutil.rmtree(base, ignore_errors=True)
        os.makedirs(f"{base}/corpus")
        self.corpus_dir = f"{base}/corpus"
        self.valid_ids = set(map(str, corpus_ids))

        # documents.tsv from the dataset's corpus metadata (doc_id per page; used for the tools' `doc` field)
        from datasets import load_dataset
        meta = load_dataset(dataset_name, data_dir="corpus", split="test").select_columns(["corpus_id", "doc_id"])
        doc_of = dict(zip(map(str, meta["corpus_id"]), meta["doc_id"]))
        runs = []  # [doc_id, first, last] over consecutive corpus_ids
        for cid in corpus_ids:
            d = doc_of[str(cid)]
            if runs and runs[-1][0] == d and int(cid) == runs[-1][2] + 1:
                runs[-1][2] = int(cid)
            else:
                runs.append([d, int(cid), int(cid)])
        if len({r[0] for r in runs}) != len(runs):
            raise ValueError("a document's pages are not consecutive corpus_ids")
        with open(f"{self.corpus_dir}/documents.tsv", "w") as fh:
            fh.write("doc_id\tn_pages\tfirst_corpus_id\tlast_corpus_id\n")
            for d, a, b in runs:
                fh.write(f"{d}\t{b - a + 1}\t{a}\t{b}\n")
        with open(f"{self.corpus_dir}/README.md", "w") as fh:
            fh.write(README.format(n=len(corpus_ids), d=len(runs), domain=self.domain, images=(
                f"- `images.zip`: the rendered pages as `<corpus_id>.jpg` (longest side {self.page_max_side}px).\n")))

        # page images: lossless PNG for the retriever; JPEG for the agent's get_page
        def prep(i):
            im = corpus_images[i].convert("RGB")
            png = io.BytesIO()
            im.save(png, "PNG", compress_level=1)
            im.thumbnail((self.page_max_side, self.page_max_side), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=self.jpeg_quality, optimize=True)
            jpg = buf.getvalue()
            return str(corpus_ids[i]), png.getvalue(), jpg

        evie_zip = f"{base}/pages_for_retriever.zip"
        with zipfile.ZipFile(evie_zip, "w", zipfile.ZIP_STORED) as ez, \
                zipfile.ZipFile(f"{self.corpus_dir}/images.zip", "w", zipfile.ZIP_STORED) as az, \
                cf.ThreadPoolExecutor(min(16, os.cpu_count() or 4)) as ex:
            for cid, png, jpg in ex.map(prep, range(len(corpus_ids))):
                ez.writestr(f"{cid}.png", png)
                az.writestr(f"{cid}.jpg", jpg)
        texts = None
        t_export = time.time() - t0
        info = _post(f"{self.retriever_url}/index", {"name": self.index_name, "pages_zip": evie_zip, "texts": texts},
                     timeout=4 * 3600)
        os.remove(evie_zip)
        self.index_info = {"pages": len(corpus_ids), "documents": len(runs), "export_ms": round(1000 * t_export),
                           "retriever_encode_ms": info["encode_ms"], "retriever_mean_tokens_per_page": info["mean_tokens"]}
        print(f"[{self.index_name}] indexed: {self.index_info}", flush=True)

    # ------------------------------------------------------------------ search
    def _llm_ok(self):
        req = urllib.request.Request(f"{self.llm_url}{self.llm_base_path}/models",
                                     headers={"Authorization": f"Bearer {os.environ.get(self.llm_key_env, '')}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status == 200
        except Exception:
            return False

    def _fallback(self, query):
        hits = _post(f"{self.retriever_url}/search", {"domain": self.index_name, "query": query, "k": self.top_k},
                     timeout=300)["results"]
        return [str(h["corpus_id"]) for h in hits]

    def search(self, query_ids, queries):
        pairs = list(zip(query_ids, queries))
        if self.query_limit:
            pairs = pairs[: int(self.query_limit)]
        cfg = {"corpus_dir": self.corpus_dir, "domain": self.index_name, "vdr_url": self.retriever_url,
               "tool_mode": TOOL_MODE[self.arm], "max_views": self.max_views, "model": self.llm_model,
               "prompt": PROMPTS[self.arm], "timeout": self.timeout, "llm_base_path": self.llm_base_path,
               "rrf_pad": self.rrf_fill, "corpus_label": f"vidore-benchmark:{self.domain}"}
        root = f"{self.run_dir}/{self.domain}"
        os.makedirs(root, exist_ok=True)
        proxies = queue.Queue()
        started = []
        for _ in range(self.num_concurrent):  # one key-injecting proxy per concurrent episode (per-episode usage log)
            p = GwProxy(upstream=self.llm_url, key_env=self.llm_key_env)
            p.start()
            proxies.put(p)
            started.append(p)
        abort = threading.Event()
        lock, done, t0 = threading.Lock(), [0], time.time()

        def one(qid, query):
            d = f"{root}/q{qid}"
            rec = {"qid": str(qid), "resumed": False, "infra_error": False, "fallback": False}
            res = None
            if os.path.exists(f"{d}/result.json"):  # finished in an earlier, interrupted run
                with open(f"{d}/result.json") as fh:
                    res = json.load(fh)
                rec["resumed"] = True
            else:
                for attempt in range(1 + self.max_retries):
                    if abort.is_set():
                        break
                    p = proxies.get()
                    try:
                        res, ok = run_episode(qid, query, d, cfg, p, self.template, self.valid_ids)
                    finally:
                        proxies.put(p)
                    if ok:
                        break
                    res = None
                    if not self._llm_ok():
                        print(f"[{self.index_name}] LLM server not answering; aborting", flush=True)
                        abort.set()
                        break
                    time.sleep(30)
                rec["infra_error"] = res is None
            ranking = list(res["ranking"]) if res else []
            if not ranking and not abort.is_set():
                ranking, rec["fallback"] = self._fallback(query), True
            rec["ranking"], rec["res"] = ranking[: self.top_k], res
            rec["views"] = page_views(f"{d}/tool_calls.jsonl") if res else 0
            with lock:
                done[0] += 1
                el = time.time() - t0
                print(f"[{self.index_name}] {done[0]}/{len(pairs)} done, {el / 60:.1f} min elapsed, "
                      f"ETA {(len(pairs) - done[0]) * el / done[0] / 60:.1f} min", flush=True)
            return rec

        try:
            with cf.ThreadPoolExecutor(self.num_concurrent) as ex:
                recs = list(ex.map(lambda a: one(*a), pairs))
        finally:
            for p in started:
                p.stop()
        if abort.is_set():
            raise RuntimeError("LLM server failed during the run; finished episodes are kept and reused on rerun")

        run = {r["qid"]: {cid: 1.0 / (i + 1) for i, cid in enumerate(r["ranking"])} for r in recs}
        ok = [r for r in recs if r["res"]]
        tok = lambda k: sum(r["res"]["tokens"][k] for r in ok)
        mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        infos = {"pipeline_infos": {
            "arm": self.arm, "llm_model": self.llm_model, "retriever_model": self.retriever_model,
            "num_concurrent": self.num_concurrent, "max_page_views": self.max_views,
            "index": self.index_info, "num_queries_run": len(pairs),
            "per_query_retrieval_time_milliseconds": {r["qid"]: round(1000 * r["res"]["wall_s"]) for r in ok},
            "llm_error_query_ids": [r["qid"] for r in recs if r["infra_error"]],
            "no_answer_query_ids": [r["qid"] for r in ok if not r["res"]["ranking_raw"]],
            "timed_out_query_ids": [r["qid"] for r in ok if r["res"]["timed_out"]],
            "fallback_query_ids": [r["qid"] for r in recs if r["fallback"]],
            "resumed_query_ids": [r["qid"] for r in recs if r["resumed"]],
            "llm_total_prompt_tokens": tok("input"), "llm_total_cached_prompt_tokens": tok("cached_input"),
            "llm_total_completion_tokens": tok("output"), "llm_total_reasoning_tokens": tok("reasoning"),
            "llm_total_tokens": tok("input") + tok("output"), "llm_total_calls": tok("llm_calls"),
            "avg_llm_turns": mean([r["res"]["tokens"]["llm_calls"] for r in ok]),
            "avg_retrieval_calls": mean([r["res"]["n_tool_calls"].get("search", 0) for r in ok]),
            "avg_page_views": mean([r["views"] for r in ok]),
            "avg_agent_ranking_length": mean([len(r["res"]["ranking_raw"]) for r in ok]),
        }}
        return run, infos
