"""Command-line interface for MLXBench."""

import sys
from .benchmark import main as benchmark_main


def main():
    """Main CLI entry point."""
    benchmark_main()


if __name__ == "__main__":
    main()