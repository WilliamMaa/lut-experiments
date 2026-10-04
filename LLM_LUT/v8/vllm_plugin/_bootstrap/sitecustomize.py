"""Spawned-process bootstrap: auto-apply the v8 patch in every process.

vLLM 0.19.1 forces VLLM_WORKER_MULTIPROC_METHOD=spawn for this model
(CUDA is initialized in the API-server proc by multimodal init, so fork
is overridden and monkeypatches never reach EngineCore/Worker procs).
With spawn, every child re-runs a fresh interpreter — but a fresh
interpreter also imports `sitecustomize` from PYTHONPATH at startup.
serve.py puts this directory on PYTHONPATH and sets V8_PLUGIN_AUTOPATCH,
so each spawned proc patches itself before vllm loads the model.
"""
import os

if os.environ.get("V8_PLUGIN_AUTOPATCH") == "1":
    try:
        from vllm_plugin import patch
        patch()
    except Exception:
        import sys
        import traceback
        traceback.print_exc()
        print("[v8_plugin] FATAL: bootstrap patch failed "
              "(serving would silently use stock attention)", flush=True)
        sys.exit(1)
