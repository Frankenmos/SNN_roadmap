"""Backward-compatible launcher shim for the renamed training module."""

from train import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
