#!/usr/bin/env python3
import argparse
import hashlib
import os
import time
from pathlib import Path

import wandb


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints as W&B artifacts.")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "glm-ctc-training"))
    parser.add_argument("--run-name", default="upload-checkpoints")
    args = parser.parse_args()

    run = wandb.init(project=args.project, name=args.run_name, job_type="checkpoint-upload")
    try:
        for path in args.checkpoints:
            path = path.resolve()
            step = int(path.stem.split("_")[-1])
            size = path.stat().st_size
            print(f"START {path} size={size} bytes", flush=True)
            t0 = time.time()
            sha = sha256_file(path)

            artifact = wandb.Artifact(
                name=f"ctc-checkpoint-step-{step}",
                type="model",
                metadata={"step": step, "size_bytes": size, "sha256": sha},
            )
            artifact.add_file(str(path))
            logged = run.log_artifact(artifact)
            logged.wait()

            seconds = time.time() - t0
            mib = size / 1024 / 1024
            print(
                f"DONE {path} seconds={seconds:.2f} speed={mib / seconds:.2f} MiB/s sha256={sha}",
                flush=True,
            )
    finally:
        run.finish()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
