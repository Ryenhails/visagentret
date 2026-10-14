# VisAgentRet: agentic pure-vision retrieval for ViDoRe V3

This repository holds the agentic pipeline we submitted to the [ViDoRe V3 pipeline leaderboard](https://github.com/illuin-tech/vidore-benchmark).

An LLM agent answers each query by searching a visual page retriever and looking at page images. It returns its top 10 pages.

1. [EVIE-4.5B](https://huggingface.co/tencent/EVIE-4.5B) is the retriever. It scores page images with late-interaction MaxSim. The corpus is encoded in `index()`.
2. [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is the agent. It runs in the [Codex CLI](https://github.com/openai/codex) harness with two MCP tools:
   * `search(query, k)` returns page ids, scores and document names from EVIE-4.5B.
   * `get_page(corpus_id)` returns the page image (longest side 1600 px). The agent may view at most 20 pages per query.
3. The agent rewrites the query, searches several times and checks pages. It then answers with a ranked list of at most 10 page ids. Empty slots are filled by reciprocal rank fusion (k=60) over the agent's own search results. This fill makes no model call.

The agent does not use the shell. No OCR text is used at any stage.

## Results

We ran the official CLI (`vidore-benchmark pipeline evaluate --language english`) at vidore-benchmark commit `a70f23a`. Hardware: one NVIDIA B300 (288 GB), 16 CPU cores, 256 GB RAM. GLM-5.3-Flash and EVIE-4.5B share the GPU. 16 queries run concurrently.

| Dataset | Queries | nDCG@10 | Recall@10 | Indexing (s) | Search (s/query) |
|---|---:|---:|---:|---:|---:|
| computer_science | 215 | 0.8481 | 0.8539 | 160 | 22.3 |
| energy | 308 | 0.7568 | 0.7978 | 299 | 24.6 |
| finance_en | 309 | 0.7667 | 0.7965 | 345 | 24.2 |
| finance_fr | 320 | 0.6030 | 0.6805 | 293 | 27.1 |
| hr | 318 | 0.7451 | 0.7544 | 132 | 27.9 |
| industrial | 283 | 0.6395 | 0.6386 | 590 | 27.6 |
| pharmaceuticals | 364 | 0.7514 | 0.7724 | 233 | 22.9 |
| physics | 302 | 0.4901 | 0.5191 | 221 | 18.6 |
| **Average** | 2419 | **0.7001** | 0.7267 | 2273 (total) | 24.4 |

The pipeline returns 10 pages per query, so Recall@100 equals Recall@10.

### Timing and cost

Search time is the CLI's `search_throughput_ms_per_query`. It is the wall time of `search()` divided by the number of queries. Indexing covers page rendering and EVIE encoding.

With 16 episodes in flight, one episode takes 335 s at the median and 660 s at the 90th percentile. An episode stops after 900 s. This happened for 74 of 2419 queries (3.1%). For 79 queries (3.3%) the agent gave no ranking, 74 of them because of this limit. Their ranking comes from the fusion fill alone.

Per query the agent made 17.7 LLM calls, 14.2 searches and 17.9 page views on average. It read about 557k prompt tokens and wrote about 2.4k tokens. Only 1.7% of the prompt tokens were served from vLLM's prefix cache in this run. With 16 long episodes the KV cache is almost full, so earlier turns are evicted and computed again. The search time above includes this cost.

Every episode's trace is kept: the Codex events, each LLM call with its token usage, each tool call, and the final answer. The pipeline also writes per-query times, token totals and the ids of timed-out queries into `aggregated_metrics.infos` of each result file.

## Code

| File | Role |
|---|---|
| `src/visagentret_pipeline.py` | The evaluated `BasePipeline` (`VisAgentRetPipeline`). This is the `module_path`. |
| `src/episode.py` | Runs one Codex episode and parses its answer. `src/rrf_pad.py` fills the ranking. |
| `src/gw_proxy.py` | Localhost proxy between Codex and vLLM. It adds the vLLM key, so the key never enters the Codex container. It logs the token usage of each call. |
| `bin/codex-gw` | Runs `codex exec` in the Codex image. Only the episode directory and the corpus are visible inside. |
| `codex-home/` | Codex config and the model entry for GLM-5.3-Flash. |
| `tools/mcp_vdr.py` | The MCP server with the `search` and `get_page` tools (stdlib only). |
| `prompts/task_tool_v1.md` | The task prompt. |
| `src/vdr_server.py` | EVIE retrieval server with live `/index` and `/search`. The pipeline uses only these two endpoints. |
| `bin/vllm-serve` | Starts GLM-5.3-Flash in the `vllm/vllm-openai:v0.30.0` image. |
| `containers/codex.def` | Apptainer definition of the Codex image. |
| `slurm/visagentret_bench.sh` | The job that produced the results. It starts vLLM, then EVIE. It runs a 1-query smoke test, the CLI on all 8 datasets, and the CI check. |
| `src/pack_episodes.py` | Packs the episode traces of each dataset into one zip. |
| `src/finalize_submission.py`, `src/validate_submission.py` | The CLI writes `num_queries` only per language, but the CI also needs it in `overall`. These scripts copy it there and run the repository's CI check locally. |

The scripts take local paths from environment variables. They are listed at the top of `slurm/visagentret_bench.sh`, `bin/vllm-serve` and `bin/codex-gw`. They replace the paths of our machine, and our site's module loads were removed. The Codex provider got a generic name, and the model catalog keeps only the GLM-5.3-Flash entry. `src/visagentret_pipeline.py`, `src/episode.py` and `tools/mcp_vdr.py` are the files that ran, with the branches of other experiments removed. This run did not use those branches. The rest of the logic is the same as in the run.

Our runs used a Codex image with the same base image, packages and Codex version as `containers/codex.def`. That image also held tools the pipeline does not call.

## Versions

* vidore-benchmark `a70f23a`.
* GLM-5.3-Flash as the NVFP4 checkpoint [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4), revision `da920bb`. It is served by vLLM 0.30.0 (`vllm/vllm-openai:v0.30.0`) on one GPU with `--gpu-memory-utilization 0.84 --max-model-len 262144 --max-num-seqs 64 --kv-cache-dtype fp8 --speculative-config {"method":"mtp","num_speculative_tokens":5} --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice --limit-mm-per-prompt {"image":32}`.
* Codex CLI 0.156.1 (Responses API).
* EVIE-4.5B, revision `b732dca`. It runs through the colpali fork in [Tencent/EVIE](https://github.com/Tencent/EVIE): 1024 visual tokens per page and the official MaxSim scoring.

## Reproduce

Build the Codex image from `containers/codex.def`. Set the variables listed at the top of `slurm/visagentret_bench.sh`. Then run it from the repository root on a machine with one NVIDIA B300 (288 GB):

```bash
mkdir -p logs
sbatch --job-name=var_visual slurm/visagentret_bench.sh
```

The leaderboard files are written to `runs/<exp>/visual/submission/results/`. The episode traces are in `runs/<exp>/visual/episodes/<dataset>.zip`. A run that stops early can be submitted again. Finished datasets are skipped, and finished episodes of an unfinished dataset are reused.

## License

Apache-2.0 (see `LICENSE`). The models keep their own licences.
