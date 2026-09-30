"""
Lab 18, step 3: sustained on-device performance (10 minutes, not 10 seconds).

Short benchmarks measure a cold, cool device. Users run models on a warm
device for minutes. This script generates continuously for `--minutes`,
logging per-minute decode tokens/s, process RSS, battery level, and (where
the OS exposes it) temperature, so thermal throttling becomes visible.

Backends:
  llama_cpp   llama-cpp-python on laptops / Raspberry Pi / Jetson
  adb         an Android device running a llama.cpp or MLC binary; battery
              and temperature read via `adb shell dumpsys battery`

Usage:
    python sustained_bench.py --backend llama_cpp --gguf mobile/model-Q4_K_M.gguf --minutes 10
    python sustained_bench.py --backend adb --cmd "/data/local/tmp/llama-cli -m /data/local/tmp/m.gguf -n 256 -p hi" --minutes 10
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time

PROMPT = "Write a short, friendly explanation of how a refund works, in five sentences."


def battery_local():
    try:
        import psutil
        b = psutil.sensors_battery()
        temps = getattr(psutil, "sensors_temperatures", lambda: {})() or {}
        t = max((x.current for v in temps.values() for x in v), default=None)
        return (b.percent if b else None), t
    except Exception:
        return None, None


def battery_adb():
    out = subprocess.run(["adb", "shell", "dumpsys", "battery"], capture_output=True, text=True).stdout
    level = re.search(r"level: (\d+)", out)
    temp = re.search(r"temperature: (\d+)", out)            # tenths of a degree C
    return (int(level.group(1)) if level else None), (int(temp.group(1)) / 10 if temp else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llama_cpp", "adb"], default="llama_cpp")
    ap.add_argument("--gguf")
    ap.add_argument("--cmd")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/sustained.jsonl")
    a = ap.parse_args()
    import os
    os.makedirs("results", exist_ok=True)
    llm = None
    if a.backend == "llama_cpp":
        from llama_cpp import Llama
        llm = Llama(model_path=a.gguf, n_ctx=4096, n_threads=a.threads, n_gpu_layers=-1, verbose=False)
    t_end = time.time() + a.minutes * 60
    minute, tokens, t_min = 0, 0, time.time()
    with open(a.out, "w") as f:
        while time.time() < t_end:
            t0 = time.time()
            if llm:
                r = llm.create_chat_completion([{"role": "user", "content": PROMPT}], max_tokens=256, temperature=0.7)
                n = r["usage"]["completion_tokens"]
            else:
                out = subprocess.run(["adb", "shell", a.cmd], capture_output=True, text=True).stdout
                m = re.search(r"eval time.*?(\d+) runs", out)
                n = int(m.group(1)) if m else 256
            tokens += n
            if time.time() - t_min >= 60:
                level, temp = battery_local() if a.backend == "llama_cpp" else battery_adb()
                rss = None
                try:
                    import psutil
                    rss = round(psutil.Process().memory_info().rss / 2 ** 30, 2)
                except Exception:
                    pass
                rec = {"minute": minute, "tok_per_s": round(tokens / (time.time() - t_min), 1),
                       "battery_pct": level, "temp_c": temp, "rss_gb": rss}
                print(json.dumps(rec)); f.write(json.dumps(rec) + "\n"); f.flush()
                minute, tokens, t_min = minute + 1, 0, time.time()


if __name__ == "__main__":
    main()
