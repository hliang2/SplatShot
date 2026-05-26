"""
Download the pre-built NeRSemble base 3DGS models from HuggingFace Hub.

Usage:
    python download_bases.py --output_dir ./nersemble_bases
"""

import argparse
from pathlib import Path


def download(output_dir: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading base models to {output_dir} ...")
    snapshot_download(
        repo_id="TODO/splatshot-bases",   # replace with actual HF repo once created
        repo_type="dataset",
        local_dir=str(output_dir),
    )
    print(f"Done. {len(list(output_dir.glob('**/*.ply')))} PLY files downloaded.")
    print(f"\nSet NERSEMBLE_PLY = \"{output_dir.resolve()}\" in inference.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./nersemble_bases")
    args = parser.parse_args()
    download(args.output_dir)
