"""Copy best_checkpoint.pth over checkpoint.pth so the next training run
resumes from the peak policy instead of the latest (possibly degraded) one.

Usage:
    python resume_from_best.py
    python resume_from_best.py --best-path best_checkpoint.pth --ckpt-path checkpoint.pth
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-path", default="best_checkpoint.pth")
    parser.add_argument("--ckpt-path", default="checkpoint.pth")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without copying.")
    args = parser.parse_args()

    best = Path(args.best_path)
    ckpt = Path(args.ckpt_path)

    if not best.exists():
        sys.exit(f"No best checkpoint at {best}. Nothing to do.")

    payload = torch.load(best, map_location="cpu", weights_only=False)
    ep = payload.get("episode", "?")
    metric = payload.get(
        "eval_reward_at_save",
        payload.get("avg_reward_at_save", payload.get("best_avg_reward", "?")),
    )
    print(f"Best checkpoint: {best} (episode={ep}, metric={metric})")

    if ckpt.exists():
        backup = ckpt.with_suffix(ckpt.suffix + ".before_resume")
        print(f"Current checkpoint -> backup: {ckpt} -> {backup}")
        if not args.dry_run:
            shutil.copy2(ckpt, backup)

    print(f"Copy {best} -> {ckpt}")
    if not args.dry_run:
        shutil.copy2(best, ckpt)
    print("Done. Next `python train.py` will resume from the best policy.")


if __name__ == "__main__":
    main()
