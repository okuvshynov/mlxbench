# Copyright © 2023-2024 Apple Inc.
# Copyright © 2025 Oleksandr Kuvshynov

# Major rewrite of Apple's generate example from https://github.com/ml-explore/mlx-lm

import argparse
import contextlib
import functools
import sys
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce

from mlx_lm.models import cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load


@dataclass
class BenchmarkParam:
    """Configuration for a benchmark parameter."""
    name: str  # Parameter name for argparse and generate()
    arg_names: List[str]  # List of argument names/shortcuts for argparse
    arg_type: type  # Type for argparse
    default: Any  # Default value
    help: str  # Help text for argparse
    arg_kwargs: Dict[str, Any] = field(default_factory=dict)  # Additional argparse kwargs
    csv_name: Optional[str] = None  # Name for CSV column (defaults to name)
    csv_format: Optional[Callable[[Any], str]] = None  # Custom formatter for CSV output
    
    def get_csv_name(self) -> str:
        """Get the name to use in CSV header."""
        return self.csv_name or self.name
    
    def format_csv_value(self, value: Any) -> str:
        """Format the value for CSV output."""
        if self.csv_format:
            return self.csv_format(value)
        return str(value)


# Registry of benchmark parameters that affect individual runs
# To add a new parameter:
# 1. Add a BenchmarkParam entry here with proper configuration
# 2. Make sure generate_step() or generate() functions accept the parameter
# 3. The parameter will automatically appear in CSV output and argparse
BENCHMARK_PARAMS = [
    BenchmarkParam(
        name="n_prompt",
        arg_names=["--n-prompt", "-np"],
        arg_type=int,
        default=1000,
        help="Target number of tokens for the prompt",
        csv_name="prompt_tokens_target",
    ),
    BenchmarkParam(
        name="n_generate",
        arg_names=["--n-generate", "-n"],
        arg_type=int,
        default=100,
        help="Number of tokens to generate",
        csv_name="generation_tokens_target",
    ),
    BenchmarkParam(
        name="max_kv_size",
        arg_names=["--max-kv-size"],
        arg_type=int,
        default=None,
        help="Set the maximum key-value cache size",
    ),
    BenchmarkParam(
        name="kv_bits",
        arg_names=["--kv-bits"],
        arg_type=int,
        default=None,
        help="Number of bits for KV cache quantization. Defaults to no quantization.",
    ),
    BenchmarkParam(
        name="kv_group_size",
        arg_names=["--kv-group-size"],
        arg_type=int,
        default=64,
        help="Group size for KV cache quantization.",
    ),
    BenchmarkParam(
        name="quantized_kv_start",
        arg_names=["--quantized-kv-start"],
        arg_type=int,
        default=0,
        help="When --kv-bits is set, start quantizing the KV cache from this step onwards.",
    ),
    BenchmarkParam(
        name="prefill_step_size",
        arg_names=["--prefill-step-size", "-pss"],
        arg_type=int,
        default=2048,
        help="Number of tokens to process at once during prompt prefill",
    ),
]


def prepare_prompt(args, tokenizer) -> List[int]:
    """
    Prepare the prompt tokens based on command line arguments.
    
    Args:
        args: Command line arguments
        tokenizer: The tokenizer to use for encoding
        
    Returns:
        List[int]: The prepared prompt tokens
    """
    # Get the base prompt
    base_prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    base_prompt = sys.stdin.read() if base_prompt == "-" else base_prompt
    
    # Tokenize the base prompt
    base_tokens = tokenizer.encode(base_prompt)
    
    # Repeat tokens to reach desired length
    target_length = args.n_prompt
    if len(base_tokens) > 0:
        repeats = (target_length + len(base_tokens) - 1) // len(base_tokens)
        prompt = base_tokens * repeats
        prompt = prompt[:target_length]
    else:
        prompt = []
    
    return prompt


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    
    # Add model and prompt arguments (not part of benchmark params)
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="The quick brown fox jumps over the lazy dog. ",
        help="Base prompt text to repeat ('-' reads from stdin)",
    )
    
    # Add benchmark parameters from registry
    for param in BENCHMARK_PARAMS:
        parser.add_argument(
            *param.arg_names,
            type=param.arg_type,
            default=param.default,
            help=param.help,
            **param.arg_kwargs
        )
    
    # Add non-benchmark parameters (verbose and repeats)
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print generated text output",
    )
    parser.add_argument(
        "--repeats",
        "-r",
        type=int,
        default=1,
        help="Number of times to repeat the benchmark",
    )
    
    return parser


# A stream on the default device just for generation
generation_stream = mx.new_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield None
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)


@dataclass
class PerfMetrics:
    """
    Performance metrics from text generation.

    Args:
        prompt_tokens (int): The number of tokens in the prompt.
        prompt_tps (float): The prompt processing tokens-per-second.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        peak_memory (float): The peak memory used so far in GB.
    """

    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if (
        kv_bits is not None
        and not isinstance(prompt_cache[0], cache.QuantizedKVCache)
        and prompt_cache[0].offset > quantized_kv_start
    ):
        for i in range(len(prompt_cache)):
            if isinstance(prompt_cache[i], cache.KVCache):
                prompt_cache[i] = prompt_cache[i].to_quantized(
                    group_size=kv_group_size, bits=kv_bits
                )


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    n_generate: int,
    max_kv_size: Optional[int],
    prefill_step_size: int,
    kv_bits: Optional[int],
    kv_group_size: int,
    quantized_kv_start: int,
) -> Generator[mx.array, None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        n_generate (int): The number of tokens to generate. Use``-1`` for an infinite
          generator.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization.
        kv_group_size (int): Group size for KV cache quantization.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None.

    Yields:
        mx.array: One token.
    """

    # Create the KV cache for generation
    prompt_cache = cache.make_prompt_cache(
        model,
        max_kv_size=max_kv_size,
    )


    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    sampler = make_sampler(0.0)

    def _step(y):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=prompt_cache)
            logits = logits[:, -1, :]
            quantize_cache_fn(prompt_cache)
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            y = sampler(logprobs)
            return y

    y = prompt
    with mx.stream(generation_stream):
        while y.shape[0] > prefill_step_size:
            model(y[:prefill_step_size][None], cache=prompt_cache)
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            y = y[prefill_step_size:]
            mx.clear_cache()

        y = _step(y)

    mx.async_eval(y)
    n = 0
    while True:
        if n != n_generate:
            next_y = _step(y)
            mx.async_eval(next_y)
        if n == n_generate:
            break
        yield y.item()
        if n % 256 == 0:
            mx.clear_cache()
        y = next_y
        n += 1


def generate(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    prompt: List[int],
    verbose: bool = False,
    **kwargs,
) -> PerfMetrics:
    """
    Generate text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        tokenizer (TokenizerWrapper): The tokenizer.
        prompt (List[int]): The input prompt as integer tokens.
        verbose (bool): Whether to print generated text during generation.
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Returns:
        PerfMetrics: An instance containing the final generation metadata.
    """
    prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer

    token_generator = generate_step(prompt, model, **kwargs)
    with wired_limit(model, [generation_stream]):
        detokenizer.reset()
        tic = time.perf_counter()
        for n, token in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()

            detokenizer.add_token(token)
            
            if verbose:
                print(detokenizer.last_segment, end="", flush=True)

        detokenizer.finalize()
        if verbose:
            print(detokenizer.last_segment, end="", flush=True)
        
        return PerfMetrics(
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
        )


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    tokenizer_config = {"trust_remote_code" : True}

    model, tokenizer = load(
        args.model,
        tokenizer_config=tokenizer_config,
    )
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt = prepare_prompt(args, tokenizer)

    # Build CSV header dynamically
    header_parts = ["run"]
    # Add benchmark parameter columns
    for param in BENCHMARK_PARAMS:
        header_parts.append(param.get_csv_name())
    # Add performance metric columns
    header_parts.extend(["prompt_tokens", "prompt_tps", "generation_tokens", "generation_tps", "peak_memory_gb"])
    print(",".join(header_parts))
    
    for run_idx in range(args.repeats):
        if args.verbose:
            print("=" * 10)
        
        # Collect benchmark parameters for generate()
        generate_kwargs = {}
        for param in BENCHMARK_PARAMS:
            # Skip n_prompt as it's used for prompt preparation, not generation
            if param.name != "n_prompt":
                generate_kwargs[param.name] = getattr(args, param.name)
        
        response = generate(
            model,
            tokenizer,
            prompt,
            verbose=args.verbose,
            **generate_kwargs
        )
        
        if args.verbose:
            print()
            print("=" * 10)
        
        # Build CSV row dynamically
        row_parts = [str(run_idx + 1)]
        # Add benchmark parameter values
        for param in BENCHMARK_PARAMS:
            value = getattr(args, param.name)
            row_parts.append(param.format_csv_value(value))
        # Add performance metrics
        row_parts.extend([
            str(response.prompt_tokens),
            f"{response.prompt_tps:.3f}",
            str(response.generation_tokens),
            f"{response.generation_tps:.3f}",
            f"{response.peak_memory:.3f}"
        ])
        print(",".join(row_parts))

if __name__ == "__main__":
    main()
