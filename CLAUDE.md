# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MLXBench is a benchmarking tool for Apple MLX-based language models. It's a rewrite of Apple's generate example from mlx-lm focused on performance benchmarking and metrics collection.

## Commands

### Running the benchmark
```bash
python mlxbench.py --model <model-name> --n-prompt <prompt-tokens> --n-generate <generation-tokens> --repeats <num-runs>
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
- run, prompt_tokens, prompt_tps, generation_tokens, generation_tps, peak_memory_gb

## Architecture

The codebase consists of two main components:

### mlxbench.py
Main benchmarking script that:
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
Hardware information utility that:
1. Detects Apple Silicon chips and their specifications
2. Gathers CPU core counts (performance and efficiency cores)
3. Extracts GPU core count using ioreg
4. Reports total system memory
5. Provides formatted output for easy display

Key functions:
- `apple_hwinfo()`: Returns comprehensive hardware information as a dictionary
- `format_hwinfo()`: Formats hardware info into a concise string (e.g., "M2 Ultra | 192.0GB | 76 GPU cores")