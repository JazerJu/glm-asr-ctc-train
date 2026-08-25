#!/usr/bin/env python3
"""定期把新 checkpoint 传成 W&B artifact（低干扰版）。

对训练的干扰点有三个：读 457MB 文件、算 sha256、上传占用 CPU/网络。
本版的取舍：
  - 轮询间隔默认 30 分钟（存档本身约 40 分钟一次，5 分钟轮询纯属浪费）
  - 进程自降优先级（nice 19 + ionice idle），只在系统空闲时抢资源
  - 每轮最多传 1 个 step_* 文件（挑最新的），里程碑文件不受此限
  - sha256 分块读时主动让出 CPU
最终结果的兜底不依赖本进程：训练结束后看门狗会用 --once 全量补传一次。
"""
import argparse, hashlib, json, os, re, time, traceback
from pathlib import Path
import wandb

STEP_RE = re.compile(r"^step_(\d+)$")

def lower_priority():
    try:
        os.nice(19)
    except Exception:
        pass
    try:  # ionice idle class；容器里可能没权限，失败就算了
        os.system(f"ionice -c3 -p {os.getpid()} >/dev/null 2>&1")
    except Exception:
        pass

def sha256_file(path: Path, yield_every=16) -> str:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
            n += 1
            if n % yield_every == 0:
                time.sleep(0.05)      # 让出 CPU/磁盘，避免和 DataLoader 抢
    return h.hexdigest()

def artifact_name(path: Path) -> str:
    stem = path.stem
    m = STEP_RE.match(stem)
    return f"ctc-checkpoint-step-{m.group(1)}" if m else f"ctc-checkpoint-{stem.replace('_','-')}"

def pick(paths, done, min_age, max_steps_per_cycle):
    """选出本轮要传的：里程碑全要，step_* 只挑最新的若干个。"""
    now = time.time()
    fresh = []
    for p in paths:
        st = p.stat()
        sig = f"{st.st_size}:{int(st.st_mtime)}"
        if done.get(p.name) == sig or now - st.st_mtime < min_age:
            continue
        fresh.append((p, sig, STEP_RE.match(p.stem)))
    milestones = [(p, s) for p, s, m in fresh if not m]
    steps = sorted([(int(m.group(1)), p, s) for p, s, m in fresh if m], reverse=True)
    return milestones + [(p, s) for _, p, s in steps[:max_steps_per_cycle]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--state", type=Path, default=Path(".uploaded_ckpts.json"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "glm-ctc-training"))
    ap.add_argument("--run-name", default="ckpt-uploader")
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--min-age", type=int, default=90)
    ap.add_argument("--max-per-cycle", type=int, default=1,
                    help="每轮最多传几个 step_* 文件（0=不限，收尾补传时用）")
    ap.add_argument("--nice", action="store_true", help="自降 CPU/IO 优先级")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    if args.nice:
        lower_priority()

    done = json.loads(args.state.read_text()) if args.state.exists() else {}
    run = wandb.init(project=args.project, name=args.run_name, job_type="checkpoint-upload")
    print(f"uploader run: {run.name}  interval={args.interval}s "
          f"max_per_cycle={args.max_per_cycle} nice={args.nice}", flush=True)
    try:
        while True:
            try:
                todo = pick(sorted(args.ckpt_dir.glob("*.pt")), done,
                            args.min_age, args.max_per_cycle or 10**6)
                for path, sig in todo:
                    t0 = time.time()
                    size = path.stat().st_size
                    print(f"START {path.name} {size/2**20:.1f} MiB", flush=True)
                    art = wandb.Artifact(name=artifact_name(path), type="model",
                                         metadata={"file": path.name, "size_bytes": size,
                                                   "sha256": sha256_file(path)})
                    art.add_file(str(path))
                    run.log_artifact(art).wait()
                    done[path.name] = sig
                    args.state.write_text(json.dumps(done, indent=1))
                    dt = time.time() - t0
                    print(f"DONE {path.name} {dt:.0f}s {size/2**20/dt:.1f} MiB/s", flush=True)
            except Exception:
                traceback.print_exc()
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        run.finish()

if __name__ == "__main__":
    main()
