# 🦙 llama-arwaky

> **GGUF Model Picker + llama-server launcher** — optimized for **AMD RX 6800 XT** with ROCm

---

## 🤖 Agent Context

This file provides context for AI agents and coding assistants to understand the project structure, hardware constraints, and design decisions.

### Project Summary

`llama-arwaky` is a Python-based TUI (Text User Interface) that:
1. **Scans** all `.gguf` files recursively in the project folder
2. **Presents** an interactive terminal menu for model selection
3. **Launches** `llama-server` with hardware-optimal settings for Raka's machine

### Target Hardware

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 5600G (6 cores / 12 threads) |
| GPU | AMD Radeon RX 6800 XT — **16 GB VRAM** |
| ROCm | 7.2.4 |
| Backend | `llama-server` build b9656 |

### File Structure

```
llama-arwaky/
├── llama-tui          # Main script (Python 3, no .py extension — runs as executable)
├── config.yaml        # Server & GPU configuration
├── setup_env.sh       # Exports PATH & LD_LIBRARY_PATH
├── AGENT.md           # This file — AI agent context
└── .gitignore         # Excludes .gguf models and llama binary folder
```

> ⚠️ The `llama-b9656/` directory and all `*.gguf` files are **not committed** to git due to their large size.

### How It Works

```
llama-tui
  └─► scan *.gguf files recursively (skip mmproj* files)
  └─► display interactive TUI menu
  └─► select model → launch llama-server
        ├─ auto-detect mmproj-*.gguf (for multimodal models)
        ├─ set HSA_OVERRIDE_GFX_VERSION=10.3.0 (required for RX 6800 XT on ROCm)
        └─ stream llama-server stdout directly to terminal
```

### Configuration (`config.yaml`)

| Key | Value | Reason |
|---|---|---|
| `ngl: 999` | All layers → GPU | Full VRAM offload |
| `no_kv_offload: true` | KV cache → CPU | Saves VRAM for large models |
| `flash_attn: true` | Flash Attention enabled | Faster attention computation |
| `cache_type_k/v: q4_0` | 4-bit KV cache | Reduces VRAM usage |
| `reasoning_budget: 0` | Disabled | For non-thinking/non-CoT models |
| `ctx: 4096` | Context window | Balanced memory vs. length |

### Dependencies

```bash
pip install pyyaml
```

No other dependencies — uses Python standard library only.

### Usage

```bash
# 1. Set up PATH (one-time)
source setup_env.sh

# 2. Run the TUI
python3 llama-tui
# or, if executable bit is set:
./llama-tui
```

### ROCm Environment Variables

The script automatically injects these into the subprocess environment:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0
# Required for RX 6800 XT — ROCm does not natively list it as a supported GFX target
# without this override, GPU offloading will silently fail

LD_LIBRARY_PATH=/home/raka/App/llama-cpp/llama-b9656:/opt/rocm-7.2.4/lib:$LD_LIBRARY_PATH
```

---

## 🧠 Notes for AI Agents

- **Do NOT change** `HSA_OVERRIDE_GFX_VERSION` — it is critical for RX 6800 XT GPU offloading under ROCm
- **`llama-server` binary path** is hardcoded in `llama-tui` at line 17 — update it if the folder changes
- **`llama-b9656/`** contains pre-built [llama.cpp](https://github.com/ggerganov/llama.cpp) release binaries, not source code
- The `llama-tui` file intentionally has **no `.py` extension** so it can be invoked directly as a shell executable
- Multimodal support (vision models) is automatic — the script looks for `mmproj-*.gguf` in the same directory as the selected model
- The `find_models()` function skips files starting with `mmproj` to avoid listing projection files as selectable models
- Server port is `8080` by default; the OpenAI-compatible endpoint is `http://127.0.0.1:8080/v1/chat/completions`
