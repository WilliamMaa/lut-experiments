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

    from vllm.entrypoints.openai import api_server

    if hasattr(api_server, "main"):
        # vLLM entrypoints expose either main() (argparse from sys.argv)
        api_server.main()
    else:  # or run_server(parsed_args)
        args = api_server.parse_args()
        api_server.run_server(args)


if __name__ == "__main__":
    main()
