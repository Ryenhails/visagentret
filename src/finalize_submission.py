"""Make a run's vidore-benchmark output PR-ready and check it with the repo's own CI.

    python src/finalize_submission.py <submission_root> <pipeline_name>

1. copies submission/pipeline_descriptions/<name>/description.json (versioned in this repo) into
   <submission_root>/results/pipeline_descriptions/<name>/;
2. adds aggregated_metrics.overall.num_queries: the CI requires it, but vidore-benchmark's aggregate_results
   (a70f23a) writes num_queries only per language. With --language english, overall == by_language.english,
   which is asserted before copying the count;
3. runs validate_submission.py (the CI check, verbatim).
"""
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    sub, name = os.path.abspath(sys.argv[1]), sys.argv[2]
    d = f"{sub}/results/pipeline_descriptions/{name}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f"{ROOT}/submission/pipeline_descriptions/{name}/description.json", f"{d}/description.json")
    for p in sorted(glob.glob(f"{sub}/results/metrics/{name}/*.json")):
        j = json.load(open(p))
        agg = j["aggregated_metrics"]
        eng = agg["by_language"]["english"]
        assert j["language"] == "english" and list(agg["by_language"]) == ["english"], p
        assert all(abs(agg["overall"][k] - eng[k]) < 1e-12 for k in agg["overall"] if k != "num_queries"), p
        if agg["overall"].get("num_queries") != eng["num_queries"]:
            agg["overall"]["num_queries"] = eng["num_queries"]
            with open(p, "w") as fh:
                json.dump(j, fh, indent=2)
        print(f"{os.path.basename(p)}: n={eng['num_queries']} nDCG@10={eng['ndcg_cut_10']:.4f} "
              f"R@10={eng['recall_10']:.4f} search {agg['timing']['search_throughput_ms_per_query'] / 1000:.1f} s/query")
    sys.exit(subprocess.run([sys.executable, f"{ROOT}/src/validate_submission.py", sub]).returncode)


if __name__ == "__main__":
    main()
