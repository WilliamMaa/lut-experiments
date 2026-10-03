"""Entry wrapper: install the v8 patch, then hand off to the vLLM OpenAI
server CLI.

Usage (remote, from LLM_LUT/v8):
    python -m vllm_plugin.serve serve \
        /home/u/downloads/models/Qwen3.6-35B-A3B \
        --enforce-eager --max-model-len 131072 \
        --max-num-seqs 64 --kv-cache-dtype auto \
        > logs/vllm_plugin.log 2>&1 &

v1 requires --enforce-eager (checked below).
"""
import sys


def main() -> None:
    if "--enforce-eager" not in sys.argv:
        raise SystemExit(
            "[vllm_plugin] v1 requires --enforce-eager (per-request Python "
            "state is not CUDA-graph capturable). Aborting.")
    from . import patch
    patch()

    # vLLM commit 58b32984: the `serve` CLI subcommand normally maps the
    # positional model_tag onto args.model (vllm/entrypoints/cli/serve.py:
    # ServeSubcommand.cmd). We bypass the subcommand, so do the mapping here —
    # otherwise EngineArgs.model stays the argparse default "Qwen/Qwen3-0.6B"
    # and startup tries to fetch it from HF (crashes on offline machines).
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-"):
        model = argv.pop(0)
        argv = ["--model", model] + argv
    sys.argv = ["vllm", *argv]

    # Entry moved to vllm.entrypoints.launchers.api_server (the old
    # vllm.entrypoints.openai.api_server is a deprecated re-export with no
    # main). Fall back to the legacy path for older installs.
    try:
        from vllm.entrypoints.launchers.api_server.entry import main as _main
    except ImportError:
        from vllm.entrypoints.openai.api_server import main as _main
    _main()


if __name__ == "__main__":
    main()
