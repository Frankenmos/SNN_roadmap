"""Backward-compatible launcher shim for the renamed evaluation module."""

from eval import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
