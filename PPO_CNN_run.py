"""Backward-compatible launcher shim for the renamed training module."""

from train import *  # noqa: F401,F403
from train import app, main


if __name__ == "__main__":
    app.run(main)
