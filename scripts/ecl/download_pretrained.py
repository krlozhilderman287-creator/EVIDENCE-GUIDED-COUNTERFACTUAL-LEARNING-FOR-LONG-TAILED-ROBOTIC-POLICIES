"""Download the exact public MiniVLA initialization, including required metadata."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="main", help="Use a saved commit SHA to repeat a download")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    root = Path(__file__).resolve().parents[2]
    repo = "Stanford-ILIAD/minivla-libero90-prismatic"
    sha = HfApi().model_info(repo, revision=args.revision).sha
    output = root / "pretrained/minivla-libero90-prismatic"
    snapshot_download(repo, revision=sha, local_dir=str(output), allow_patterns=[
        "config.json", "dataset_statistics.json", "checkpoints/step-122500-epoch-55-loss=0.0743.pt"
    ])
    (output / "download_manifest.json").write_text(json.dumps({"repo": repo, "revision": sha}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
