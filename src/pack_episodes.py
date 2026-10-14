"""Pack a finished dataset's episode folders into one zip (all files kept), then remove the loose files.

    python src/pack_episodes.py runs/visagentret_glm53/visual/episodes/hr     # -> .../episodes/hr.zip

Members keep their relative paths (q<qid>/events.jsonl, ...). Loose files are removed only after the zip
is re-read and holds every file with the same size.
"""
import os
import shutil
import sys
import zipfile


def pack(d):
    d = os.path.realpath(d.rstrip("/"))
    files = sorted(os.path.relpath(os.path.join(r, f), d) for r, _, fs in os.walk(d) for f in fs)
    out = d + ".zip"
    with zipfile.ZipFile(out + ".tmp", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            zf.write(os.path.join(d, f), arcname=f)
    with zipfile.ZipFile(out + ".tmp") as zf:
        assert zf.testzip() is None
        sizes = {i.filename: i.file_size for i in zf.infolist()}
    assert sizes == {f: os.path.getsize(os.path.join(d, f)) for f in files}, d
    os.replace(out + ".tmp", out)
    shutil.rmtree(d)
    print(f"{d}: {len(files)} files -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        pack(a)
