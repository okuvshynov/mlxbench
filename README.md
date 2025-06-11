# MLXBench

A benchmarking tool for Apple MLX-based language models. This is a complete rewrite of Apple's generate example from mlx-lm focused on performance benchmarking and metrics collection.

## Features

- Comprehensive benchmarking of MLX language models
- Hardware detection for Apple Silicon Macs
- Performance metrics including tokens per second and memory usage
- CSV output for easy analysis
- Support for KV cache quantization
- Multi-model benchmarking support

## Installation

### Using pip

```bash
pip install mlxbench
```

### Using uv

```bash
uv pip install mlxbench
```

### From source

```bash
git clone https://github.com/okuvshynov/mlxbench
cd mlxbench
pip install -e .
```

## Usage

### Basic benchmarking

```bash
mlxbench --model mlx-community/Llama-3.2-3B-Instruct-4bit --n-prompt 1000 --n-generate 100 --repeats 5
```

### Key arguments

- `--model`: HuggingFace model path (default: mlx-community/Llama-3.2-3B-Instruct-4bit)
- `--n-prompt`: Target number of prompt tokens (default: 1000)
- `--n-generate`: Number of tokens to generate (default: 100)
- `--repeats`: Number of benchmark runs (default: 1)
- `--verbose`: Print generated text output
- `--prefill-step-size`: Tokens to process at once during prefill (default: 2048)
- `--max-kv-size`: Maximum key-value cache size
- `--kv-bits`: Bits for KV cache quantization

### Output format

The tool outputs CSV data with columns:
- run, prompt_tokens, prompt_tps, generation_tokens, generation_tps, peak_memory_gb

### Hardware information

```python
from mlxbench import apple_hwinfo, format_hwinfo

info = apple_hwinfo()
print(format_hwinfo(info))
# Output: M2 Ultra | 192.0GB | 76 GPU cores
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4 series)
- Python 3.9+
- mlx-lm >= 0.12.0

## License

MIT License - see LICENSE file for details.