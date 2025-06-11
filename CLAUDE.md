# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MLXBench is a benchmarking tool for Apple MLX-based language models. It's a rewrite of Apple's generate example from mlx-lm focused on performance benchmarking and metrics collection.

## Project Structure

```
mlxbench/
├── src/
│   └── mlxbench/
│       ├── __init__.py       # Package exports
│       ├── benchmark.py      # Main benchmarking logic
│       ├── cli.py           # CLI entry point
│       └── hwinfo.py        # Hardware detection utility
├── pyproject.toml           # Package configuration
├── README.md               # User documentation
└── CLAUDE.md              # This file
```

## Commands

### Installation (development)
```bash
pip install -e .
```

### Running the benchmark
```bash
mlxbench --model <model-name> --n-prompt <prompt-tokens> --n-generate <generation-tokens> --repeats <num-runs>
```

Key arguments:
- `--model`: HuggingFace model path (default: mlx-community/Llama-3.2-3B-Instruct-4bit)
- `--n-prompt`: Target number of prompt tokens (default: 1000)
- `--n-generate`: Number of tokens to generate (default: 100)
- `--repeats`: Number of benchmark runs (default: 1)
- `--verbose`: Print generated text output
- `--prefill-step-size`: Tokens to process at once during prefill (default: 2048)
- `--max-kv-size`: Maximum key-value cache size
- `--kv-bits`: Bits for KV cache quantization

### Output format
The script outputs CSV data with columns:
- run, model, hw_model, n_prompt, n_generate, prefill_step_size, prompt_tokens, prompt_tps, generation_tokens, generation_tps, peak_memory_gb

The hw_model column automatically includes hardware information (chip model, memory, GPU cores) for each benchmark run.

## Architecture

The package consists of several components:

### benchmark.py
Main benchmarking module that:
1. Loads MLX models using mlx_lm.utils.load
2. Tokenizes and prepares prompts by repeating base text to reach target length
3. Generates tokens using a custom generate_step generator with KV cache support
4. Collects performance metrics (tokens per second, memory usage)
5. Outputs results in CSV format for analysis

Key functions:
- `prepare_prompt()`: Repeats base prompt to reach target token count
- `generate_step()`: Generator for token generation with prefill and KV cache
- `generate()`: Main generation function that collects performance metrics
- `wired_limit()`: Context manager for handling memory limits on Apple Silicon

The implementation uses MLX streams for efficient async evaluation and supports KV cache quantization for memory optimization.

### hwinfo.py
Internal hardware detection utility that:
1. Detects Apple Silicon chips and their specifications
2. Gathers CPU core counts (performance and efficiency cores)
3. Extracts GPU core count using ioreg
4. Reports total system memory
5. Automatically included in benchmark CSV output

### cli.py
Command-line interface entry point that wraps the benchmark module for the `mlxbench` command.