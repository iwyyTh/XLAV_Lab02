"""Provide the future training entry point in M6."""
from pathlib import Path
import torch


def main() -> None:
    """Input: future CLI/config. Output: history/checkpoint artifacts.

    TODO: wire config, dataset, model and trainer; never tune on test set.
    """
    raise NotImplementedError("M6")


if __name__ == "__main__":
    main()
