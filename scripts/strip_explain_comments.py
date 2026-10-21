"""Remove the teaching comments that were added to the v14 tear-amodal files.

Every such comment is a FULL LINE whose text starts with `# [EXPLAIN]`, so removing
those lines never touches code. Run (dry run first):

    python scripts/strip_explain_comments.py --dry-run
    python scripts/strip_explain_comments.py

Default targets are the four v14 python files; pass other paths to strip more.
"""
import argparse
import os

DEFAULT = ["segmentation/tear_synth.py", "segmentation/dataset_2_5d_tear.py",
           "segmentation/model_tear_amodal.py", "segmentation/finetune_meniscus_v14_tear_amodal.py"]
TAG = "# [EXPLAIN]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in a.files:
        path = f if os.path.isabs(f) else os.path.join(root, f)
        lines = open(path).read().splitlines(keepends=True)
        kept = [l for l in lines if not l.lstrip().startswith(TAG)]
        print(f"{f}: removing {len(lines) - len(kept)} comment lines ({len(lines)} -> {len(kept)})")
        if not a.dry_run:
            open(path, "w").write("".join(kept))


if __name__ == "__main__":
    main()
