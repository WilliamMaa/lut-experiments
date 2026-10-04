"""Entry wrapper: install the v8 patch, then hand off to the vLLM OpenAI
server (vLLM 0.19.1 and 0.30-era entry points supported).

Usage (remote, from LLM_LUT/v8):
    python -m vllm_plugin.serve /home/u/downloads/models/Qwen3.6-35B-A3B \
        --enforce-eager --no-enable-prefix-caching --max-model-len 131072 \
        --tensor-parallel-size 2 --max-num-seqs 64 --port 18001

v1 requires --enforce-eager (checked below). Prefix caching is force-
disabled: block reuse across requests would alias per-request compact
regions (0.19.1 has no per-spec prefix_cacheable flag).
"""
import sys

_PREFIX_FLAGS = ("--no-enable-prefix-caching", "--enable-prefix-caching",
                 "--disable-prefix-caching")


def main() -> None:
    if "--enforce-eager" not in sys.argv:
        raise SystemExit(
            "[vllm_plugin] v1 requires --enforce-eager (per-request Python "
            "state is not CUDA-graph capturable). Aborting.")
    argv = list(sys.argv[1:])
    if not any(f in argv for f in _PREFIX_FLAGS):
        argv.append("--no-enable-prefix-caching")

    from . import patch
    patch()

    try:
        # vLLM 0.30-era entry: vllm.entrypoints.launchers.api_server.
        from vllm.entrypoints.launchers.api_server.entry import main as _main
    except ImportError:
        _serve_019(argv)
        return

    # The `serve` CLI subcommand normally maps the positional model_tag onto
    # args.model (vllm/entrypoints/cli/serve.py ServeSubcommand.cmd). We
    # bypass the subcommand, so do the mapping here — otherwise
    # EngineArgs.model stays the argparse default and startup tries to fetch
    # it from HF (crashes on offline machines).
    if argv and not argv[0].startswith("-"):
        model = argv.pop(0)
        argv = ["--model", model] + argv
    sys.argv = ["vllm", *argv]
    _main()


def _serve_019(argv) -> None:
    """vLLM 0.19.1 path: run the OpenAI api_server module as __main__."""
    # api_server.__main__ parses the positional model_tag but never maps it
    # onto args.model (that mapping lives in vllm/entrypoints/cli/serve.py,
    # which we bypass) — without this, args.model falls back to the
    # "Qwen/Qwen3-0.6B" default and startup tries to fetch from HF.
    if argv and not argv[0].startswith("-"):
        model = argv.pop(0)
        argv = ["--model", model] + argv
    sys.argv = ["api_server", *argv]
    import runpy
    runpy.run_module("vllm.entrypoints.openai.api_server",
                     run_name="__main__")


if __name__ == "__main__":
    main()
