#!/bin/bash
#SBATCH --job-name=visagentret
#SBATCH --gpus=b300:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=logs/visagentret_%x_%j.out
#SBATCH --error=logs/visagentret_%x_%j.err
#
# VisAgentRet on ViDoRe v3 through vidore-benchmark's own CLI (English queries, all 8 datasets, the leaderboard
# protocol). The job has the GPU to itself, so its timings are its own.
# From the repository root, after mkdir -p logs:
#   sbatch --job-name=var_visual slurm/visagentret_bench.sh
# Per job: one vLLM server (GLM-5.3-Flash NVFP4), one EVIE retrieval server, then per dataset
# `vidore-benchmark pipeline evaluate` with src/visagentret_pipeline.py (index = live EVIE encoding of the
# corpus it is given; search = NUM_CONCURRENT Codex episodes at a time).
#
# Set first (sbatch passes the environment on):
#   VLLM_SIF          Apptainer image of vllm/vllm-openai:v0.30.0 (used by bin/vllm-serve)
#   CODEX_SIF         Apptainer image from containers/codex.def (used by bin/codex-gw)
#   EVIE_PYTHON       python of an env with the EVIE colpali fork's dependencies
#   EVIE_COLPALI      colpali/ directory of a github.com/Tencent/EVIE checkout
#   VIDORE_BENCH_BIN  bin/ directory of an env with vidore-benchmark (commit a70f23a) installed
#   VIDORE_BENCHMARK_DIR  checkout of vidore-benchmark, for its CI check (default external/vidore-benchmark)
#   HF_HOME           Hugging Face cache with the models (default ~/.cache/huggingface)
# Needs apptainer, python3, curl and gcc on PATH (EVIE JIT-compiles Triton kernels).
#
# Outputs:
#   runs/$EXP/$ARM/submission/results/{metrics,pipeline_descriptions}/<name>/   PR-ready, checked by the repo's CI
#   runs/$EXP/$ARM/episodes/<dataset>.zip   every episode's trace (events, LLM calls, tool calls, result.json)
#   runs/$EXP/$ARM/{run_config.json, vllm.log, evie.log, cli_<dataset>.log}
# Resumable: re-submit; finished datasets are skipped, finished episodes of an interrupted dataset are reused
# (listed in infos.resumed_query_ids). The job exits as soon as the work is done or anything fails, and the
# EXIT trap stops vLLM/EVIE, so the GPU is released at once.
#
# MIG pipeline test (inside an allocation): MODEL_REPO=Qwen/Qwen3.8-27B-FP8 SERVED=... VLLM_ARGS=... \
#   DATASETS=hr QUERY_LIMIT=3 NUM_CONCURRENT=2 EXP=_migtest_bench srun --overlap ... bash slurm/visagentret_bench.sh
set -u
ARM=visual  # the pipeline's arm (output paths and pipeline_args use it)
export CC=${CC:-$(command -v gcc)} CXX=${CXX:-$(command -v g++)}
JOB=${SLURM_JOB_ID:-local$$}
export TRITON_CACHE_DIR=/tmp/${USER}_triton_$JOB
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
: "${VLLM_SIF:?}" "${CODEX_SIF:?}" "${EVIE_PYTHON:?}" "${EVIE_COLPALI:?}" "${VIDORE_BENCH_BIN:?}"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export VIDORE_BENCHMARK_DIR=${VIDORE_BENCHMARK_DIR:-$ROOT/external/vidore-benchmark}
cd "$ROOT"

MODEL_REPO=${MODEL_REPO:-nvidia/GLM-5.3-Flash-NVFP4}
SERVED=${SERVED:-zai-org/GLM-5.3-Flash}
# GLM recipe (recipes.vllm.ai/zai-org/GLM-5.3-Flash), TP=1 on one 288 GB B300; ~16% of the GPU is left for EVIE.
# (single quotes: the JSON values must reach vLLM intact; no spaces inside them, so word splitting is safe)
if [ -z "${VLLM_ARGS:-}" ]; then
    VLLM_ARGS='--gpu-memory-utilization 0.84 --max-model-len 262144 --max-num-seqs 64 --kv-cache-dtype fp8 --speculative-config {"method":"mtp","num_speculative_tokens":5} --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice --limit-mm-per-prompt {"image":32}'
fi
# used only if the first start fails: no FP8 KV cache, no MTP
if [ -z "${VLLM_FALLBACK:-}" ]; then
    VLLM_FALLBACK='--gpu-memory-utilization 0.84 --max-model-len 262144 --max-num-seqs 64 --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice --limit-mm-per-prompt {"image":32}'
fi
DATASETS=${DATASETS:-"computer_science energy hr finance_en finance_fr industrial pharmaceuticals physics"}
NUM_CONCURRENT=${NUM_CONCURRENT:-16}
QUERY_LIMIT=${QUERY_LIMIT:-}   # testing only: first N queries per dataset, the rest score 0
SMOKE=${SMOKE:-1}
EXP=${EXP:-visagentret_glm53}
PNAME=${PNAME:-visagentret-${ARM}_evie-4.5b_glm-5.3-flash}
mkdir -p "runs/$EXP/$ARM"
RUN=$(realpath "runs/$EXP/$ARM")
SUB=$RUN/submission
LOCAL=/tmp/${USER}_visagentret_$JOB
VB=$VIDORE_BENCH_BIN
mkdir -p "$RUN/episodes" "$SUB/results/metrics/$PNAME" "$LOCAL"
echo "Start: $(date) host: $(hostname) job: $JOB arm: $ARM model: $MODEL_REPO datasets: $DATASETS concurrent: $NUM_CONCURRENT ${QUERY_LIMIT:+query_limit: $QUERY_LIMIT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

VPID=""; EPID=""
cleanup() {
    for p in $EPID $VPID; do kill "$p" 2>/dev/null; done
    sleep 5
    for p in $EPID $VPID; do kill -9 "$p" 2>/dev/null; done
    rm -rf "$LOCAL" "/tmp/${USER}_vllm_$JOB" "$TRITON_CACHE_DIR"
    echo "End Time: $(date)"
}
trap cleanup EXIT

free_port() { python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])"; }
VLLM_API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(24))")
export VLLM_API_KEY

# ---------- 1. vLLM ----------
start_vllm() {  # $1 = args string, $2 = log
    VPORT=$(free_port)
    # shellcheck disable=SC2086
    bin/vllm-serve "$VPORT" "$SERVED" "$MODEL_REPO" $1 > "$2" 2>&1 &
    VPID=$!
    for i in $(seq 1 720); do  # up to 120 min (weight loading plus compile)
        if curl -sf -m 5 -H @<(printf 'Authorization: Bearer %s\n' "$VLLM_API_KEY") "http://127.0.0.1:$VPORT/v1/models" >/dev/null; then
            echo "vLLM ready on $VPORT after $((i * 10))s"; return 0
        fi
        kill -0 $VPID 2>/dev/null || { echo "vLLM exited:"; tail -40 "$2"; return 1; }
        sleep 10
    done
    echo "vLLM not ready after 120 min"; kill $VPID; return 1
}
VLLM_USED="$VLLM_ARGS"
if ! start_vllm "$VLLM_ARGS" "$RUN/vllm.log"; then
    echo "retrying with fallback args"; VLLM_USED="$VLLM_FALLBACK"
    start_vllm "$VLLM_FALLBACK" "$RUN/vllm_fallback.log" || { echo "vLLM failed twice"; exit 1; }
fi
grep -hE "KV cache|Maximum concurrency|max_num_seqs" "$RUN"/vllm*.log | tail -4

# ---------- 2. EVIE retrieval server (empty; the pipeline's index() fills it) ----------
PYTHONPATH=$EVIE_COLPALI HF_HUB_OFFLINE=1 \
  "$EVIE_PYTHON" -u src/vdr_server.py \
  --backend evie --domains --port-file "$LOCAL/evie.port" > "$RUN/evie.log" 2>&1 &
EPID=$!
for i in $(seq 1 90); do [ -f "$LOCAL/evie.port" ] && break; kill -0 $EPID 2>/dev/null || break; sleep 10; done
[ -f "$LOCAL/evie.port" ] || { echo "EVIE server failed"; tail -20 "$RUN/evie.log"; exit 1; }
EURL="http://127.0.0.1:$(cat "$LOCAL/evie.port")"
echo "EVIE at $EURL"

python3 - "$RUN/run_config.json" <<EOF
import json, subprocess, sys, time
commit = subprocess.run(["git", "-C", "$VIDORE_BENCHMARK_DIR", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump({"time": time.strftime("%F %T"), "job": "$JOB", "host": "$(hostname)", "arm": "$ARM", "pipeline_name": "$PNAME",
           "model_repo": "$MODEL_REPO", "served": "$SERVED", "vllm_image": "vllm/vllm-openai:v0.30.0", "vllm_args": """$VLLM_USED""",
           "num_concurrent": $NUM_CONCURRENT, "query_limit": "$QUERY_LIMIT", "datasets": "$DATASETS".split(),
           "vidore_benchmark_commit": commit}, open(sys.argv[1], "w"), indent=1)
EOF

# ---------- 3. the official CLI, one dataset at a time ----------
export VISAGENTRET_LLM_URL="http://127.0.0.1:$VPORT" VISAGENTRET_LLM_BASE_PATH=/v1 VISAGENTRET_LLM_KEY_ENV=VLLM_API_KEY
export VISAGENTRET_RETRIEVER_URL="$EURL" VISAGENTRET_WORK_DIR="$LOCAL/work"
# datasets: hub snapshot from $HF_HOME (online check, no re-download), arrow cache node-local
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$LOCAL/hf_datasets
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
pargs() {  # $1 = query limit (may be empty)
    python3 -c "
import json, sys
d = {'arm': '$ARM', 'llm_model': '$SERVED', 'retriever_model': 'tencent/EVIE-4.5B', 'num_concurrent': $NUM_CONCURRENT,
     'target_top_k': 10, 'max_page_views': 20, 'episode_timeout_s': 900, 'max_retries': 2, 'rrf_fill': True,
     'page_max_side': 1600, 'page_jpeg_quality': 85}
if sys.argv[1]: d['query_limit'] = int(sys.argv[1])
print(json.dumps(d))" "$1"
}
evaluate() {  # $1 = dataset, $2 = output json, $3 = query limit, $4 = log, $5 = episode dir
    env VISAGENTRET_RUN_DIR="$5" "$VB/vidore-benchmark" pipeline evaluate --dataset-name "vidore/vidore_v3_$1" --language english \
        --module-path "$ROOT/src/visagentret_pipeline.py" --class-name VisAgentRetPipeline \
        --pipeline-args "$(pargs "$3")" --output-file "$2" > "$4" 2>&1
}

# smoke gate: one HR query through the full official path; the agent must use its tools and answer
if [ "$SMOKE" = 1 ]; then
    echo "== smoke $(date +%T)"
    evaluate hr "$RUN/smoke_vidore_v3_hr.json" 1 "$RUN/cli_smoke.log" "$RUN/smoke_episodes"
    if ! python3 -I - "$RUN/smoke_episodes/hr" "$ARM" <<'EOF'
import glob, json, os, sys
d, arm = sys.argv[1:]
r = [(p, json.load(open(p))) for p in glob.glob(f"{d}/q*/result.json")]
calls = []
if r:
    log = os.path.join(os.path.dirname(r[0][0]), "tool_calls.jsonl")
    calls = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
good = lambda t: any(c.get("tool") == t and "error" not in c for c in calls)
ok = bool(r) and bool(r[0][1]["ranking_raw"]) and good("search") and good("get_page")
print("smoke", "PASS" if ok else "FAIL", "| ok search/get_page:", good("search"), good("get_page"),
      "| shell cmds:", r[0][1]["n_commands"] if r else None, "| ranking:", r[0][1]["ranking_raw"] if r else None,
      "| tokens:", r[0][1]["tokens"] if r else None)
sys.exit(0 if ok else 1)
EOF
    then
        echo "smoke test failed; stopping before the full run"; tail -30 "$RUN/cli_smoke.log"; exit 1
    fi
fi

STATUS=0
for D in $DATASETS; do
    OUT=$SUB/results/metrics/$PNAME/vidore_v3_$D.json
    [ -f "$OUT" ] && { echo "== $D: done earlier, skipping"; continue; }
    echo "== $D $(date +%T)"
    if ! evaluate "$D" "$OUT" "$QUERY_LIMIT" "$RUN/cli_$D.log" "$RUN/episodes"; then
        echo "evaluation failed for $D:"; grep -v -i "warn" "$RUN/cli_$D.log" | tail -30; STATUS=1; break
    fi
    grep -E "ndcg_cut_10|recall_10 |search_throughput|indexed:" "$RUN/cli_$D.log" | head -6
    python3 src/pack_episodes.py "$RUN/episodes/$D"
    rm -rf "$LOCAL/work/$ARM/$D"
done

# ---------- 4. PR check (passes once all 8 datasets are in) ----------
python3 src/finalize_submission.py "$SUB" "$PNAME" || echo "PR check not passed (expected until all 8 datasets are in)"
echo "Exit Code: $STATUS"
exit $STATUS
