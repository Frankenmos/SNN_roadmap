"""Backward-compatible launcher shim for the renamed evaluation module."""

from eval import *  # noqa: F401,F403
from eval import app, main


if __name__ == "__main__":
    app.run(main)
