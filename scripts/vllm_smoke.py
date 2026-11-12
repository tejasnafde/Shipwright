# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub>=1.0",
#     "httpx",
# ]
# ///
"""Does vLLM start on an HF Job GPU, with and without the LoRA adapter, and answer a request? (about $0.20)

The gate's first real run (round 3) died at vLLM startup, after 5 minutes and before one tau2 run. Run this first
after any change to the gate's serving (vLLM version, flags, environment), so the 90 minute gate is not the test:

  .venv/bin/hf jobs uv run --flavor a10g-large --timeout 25m --secrets HF_TOKEN scripts/vllm_smoke.py \\
      --adapter-repo mitushigoyal5/shipwright-lora --adapter-revision round-3

It builds the same `vllm serve` command as scripts/gate_eval.py (same flags, same environment), serves the base
model and the adapter, sends one greedy request to each, and prints SMOKE PASS or the end of vLLM's log.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

BASE_MODEL = "CohereLabs/tiny-aya-global"
VLLM = "vllm==0.32.0"
PORT = 8000
# Keep in step with VLLM_ENVIRONMENT in scripts/gate_eval.py.
VLLM_ENVIRONMENT = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-repo", default="mitushigoyal5/shipwright-lora")
    parser.add_argument("--adapter-revision", default="round-3")
    parser.add_argument("--startup-seconds", type=int, default=1200)
    return parser.parse_args(argv)


def serve_command(adapter_directory: str) -> list[str]:
    return [
        "uv", "run", "--no-project", "--python", "3.12", "--with", VLLM,
        "vllm", "serve", BASE_MODEL, "--served-model-name", BASE_MODEL, "--port", str(PORT),
        "--dtype", "bfloat16", "--max-model-len", "8192", "--gpu-memory-utilization", "0.85",
        "--enable-lora", "--max-lora-rank", "16", "--max-loras", "1", "--lora-modules", f"candidate={adapter_directory}",
    ]


def wait_until_ready(process: subprocess.Popen, startup_seconds: int) -> set[str]:
    import httpx

    deadline = time.time() + startup_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with {process.returncode}")
        try:
            response = httpx.get(f"http://localhost:{PORT}/v1/models", timeout=5)
            served = {model["id"] for model in response.json().get("data", [])}
            if response.status_code == 200 and {BASE_MODEL, "candidate"} <= served:
                return served
        except Exception:  # noqa: BLE001 - not up yet
            pass
        time.sleep(10)
    raise RuntimeError(f"vLLM not ready after {startup_seconds} s")


def ask(model_name: str) -> str:
    import httpx

    response = httpx.post(
        f"http://localhost:{PORT}/v1/chat/completions", timeout=120,
        json={"model": model_name, "temperature": 0, "max_tokens": 40,
              "messages": [{"role": "user", "content": "Reply with the single word: ready"}]},
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    from huggingface_hub import snapshot_download

    adapter_directory = snapshot_download(options.adapter_repo, revision=options.adapter_revision)
    log_path = "/tmp/vllm-smoke.log"
    command = serve_command(adapter_directory)
    print("+ " + " ".join(command), flush=True)
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env={**os.environ, **VLLM_ENVIRONMENT})
    try:
        served = wait_until_ready(process, options.startup_seconds)
        print(f"vLLM ready, serving {sorted(served)}", flush=True)
        print(f"base      -> {ask(BASE_MODEL)!r}", flush=True)
        print(f"candidate -> {ask('candidate')!r}", flush=True)
        print("SMOKE PASS", flush=True)
        return 0
    except Exception as smoke_error:  # noqa: BLE001
        print(f"SMOKE FAIL: {smoke_error}", flush=True)
        with open(log_path, errors="replace") as log_file:
            print("".join(log_file.readlines()[-60:]), flush=True)
        return 1
    finally:
        process.terminate()


if __name__ == "__main__":
    sys.exit(main())
