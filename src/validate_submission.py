"""Run vidore-benchmark's own PR check (.github/workflows/validate-results-json.yml) on a local submission.

    python src/validate_submission.py <submission_root>

<submission_root> mirrors the repo layout: results/metrics/<name>/vidore_v3_*.json and
results/pipeline_descriptions/<name>/description.json. The check's Python is taken verbatim from the
workflow file of the cloned repo, with every JSON under <submission_root>/results as the "changed files".
"""
import glob
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(os.environ.get("VIDORE_BENCHMARK_DIR", f"{ROOT}/external/vidore-benchmark"),
                        ".github/workflows/validate-results-json.yml")


def main():
    root = os.path.abspath(sys.argv[1])
    src = open(WORKFLOW).read()
    code = src.split("python - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    code = textwrap.dedent(code)
    files = sorted(os.path.relpath(p, root) for p in glob.glob(f"{root}/results/**/*.json", recursive=True))
    with open(f"{root}/changed_json_files.txt", "w") as fh:
        fh.write("\n".join(files) + "\n")
    try:
        rc = subprocess.run([sys.executable, "-c", code], cwd=root).returncode
    finally:
        os.remove(f"{root}/changed_json_files.txt")
    sys.exit(rc)


if __name__ == "__main__":
    main()
