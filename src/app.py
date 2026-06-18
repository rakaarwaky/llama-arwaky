"""
src/app.py — Textual TUI for llama-tui
★(◕‿◕)ノ

Screens:
  ModelScreen  — table of available GGUF models
  ServerScreen — live server log + clickable URL when ready
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    Link,
    RichLog,
    Static,
)

from .config import ROOT_DIR, load_model_config
from .models import ModelEntry, find_mmproj, find_models

LLAMA_SERVER = os.path.join(ROOT_DIR, "llama-b9656", "llama-server")

# ─── CSS ─────────────────────────────────────────────────────────────────────

APP_CSS = """
/* ── Global ─────────────────────────────────────────── */
Screen {
    background: $surface;
}

/* ── Model Screen ───────────────────────────────────── */
#model-table {
    height: 1fr;
    margin: 1 2;
    border: tall $accent 60%;
}

#model-table > .datatable--header {
    background: $accent-darken-2;
    color: $text;
    text-style: bold;
}

#model-table > .datatable--cursor {
    background: $accent;
    color: $text;
}

#hint {
    dock: bottom;
    height: 1;
    background: $surface-darken-1;
    color: $text-muted;
    padding: 0 2;
    text-align: center;
}

/* ── Server Screen ──────────────────────────────────── */
#info-bar {
    height: auto;
    background: $surface-darken-2;
    padding: 0 2;
    border-bottom: solid $accent 40%;
}

#server-log {
    height: 1fr;
    margin: 0 1;
    border: tall $accent 30%;
    padding: 0 1;
}

#copy-bar {
    height: auto;
    padding: 0 1;
    background: $surface-darken-1;
    border-top: solid $accent 40%;
}

#copy-bar Button {
    width: auto;
}

#url-bar {
    dock: bottom;
    height: auto;
    min-height: 3;
    background: $success-darken-3;
    padding: 1 2;
    border-top: solid $success 60%;
    display: none;
}

#url-bar.ready {
    display: block;
}

#url-bar Link {
    color: $success-lighten-2;
    text-style: bold underline;
}

#url-bar Label {
    color: $text-muted;
}

#url-main {
    color: $success;
    text-style: bold;
}
"""

# ─── Helper: build llama-server args from config ─────────────────────────────

def build_server_args(model_path: str, cfg: dict) -> list[str]:
    """Build complete llama-server command line from config dict."""
    scfg = cfg.get("server", {})
    mcfg = cfg.get("model", {})
    ccfg = cfg.get("cpu", {})
    kcfg = cfg.get("kv_cache", {})
    memcfg = cfg.get("memory", {})
    qcfg = cfg.get("quantization", {})
    scfg_samp = cfg.get("sampling", {})
    acfg = cfg.get("adapters", {})
    specfg = cfg.get("speculative", {})
    mmcfg = cfg.get("multimodal", {})
    rcfg = cfg.get("reasoning", {})
    chatcfg = cfg.get("chat", {})
    fcfg = cfg.get("features", {})
    authcfg = cfg.get("auth", {})
    logcfg = cfg.get("logging", {})
    perfcfg = cfg.get("performance", {})
    slotcfg = cfg.get("slots", {})
    embcfg = cfg.get("embeddings", {})
    ttscfg = cfg.get("tts", {})
    netcfg = cfg.get("network", {})
    overcfg = cfg.get("override", {})
    sleepcfg = cfg.get("sleep", {})
    idcfg = cfg.get("identity", {})

    args = [
        LLAMA_SERVER,
        "-m", model_path,
        "--host", scfg.get("host", "127.0.0.1"),
        "--port", str(scfg.get("port", 8080)),
    ]

    # Model loading
    if mcfg.get("ngl") is not None:
        args += ["-ngl", str(mcfg["ngl"])]
    if mcfg.get("ctx") is not None:
        args += ["-c", str(mcfg["ctx"])]
    if mcfg.get("batch_size") is not None:
        args += ["-b", str(mcfg["batch_size"])]
    if mcfg.get("ubatch_size") is not None:
        args += ["-ub", str(mcfg["ubatch_size"])]
    if mcfg.get("main_gpu") is not None:
        args += ["--main-gpu", str(mcfg["main_gpu"])]
    if mcfg.get("split_mode"):
        args += ["--split-mode", mcfg["split_mode"]]
    if mcfg.get("tensor_split"):
        args += ["--tensor-split", mcfg["tensor_split"]]
    if mcfg.get("mlock"):
        args.append("--mlock")
    if not mcfg.get("mmap", True):
        args.append("--no-mmap")
    if mcfg.get("no_host"):
        args.append("--no-host")
    if mcfg.get("checkpoint"):
        args.append("--checkpoint")
    if mcfg.get("rpc_servers"):
        args += ["--rpc", mcfg["rpc_servers"]]

    # CPU affinity & threading
    if ccfg.get("threads") is not None:
        args += ["-t", str(ccfg["threads"])]
    if ccfg.get("threads_batch") is not None:
        args += ["--threads-batch", str(ccfg["threads_batch"])]
    if ccfg.get("cpu_mask"):
        args += ["--cpu-mask", ccfg["cpu_mask"]]
    if ccfg.get("cpu_range"):
        args += ["--cpu-range", ccfg["cpu_range"]]
    if ccfg.get("cpu_strict"):
        args.append("--cpu-strict")
    if ccfg.get("prio") is not None:
        args += ["--prio", str(ccfg["prio"])]
    if ccfg.get("poll") is not None:
        args += ["--poll", str(ccfg["poll"])]
    if ccfg.get("cpu_mask_batch"):
        args += ["--cpu-mask-batch", ccfg["cpu_mask_batch"]]
    if ccfg.get("cpu_range_batch"):
        args += ["--cpu-range-batch", ccfg["cpu_range_batch"]]
    if ccfg.get("cpu_strict_batch"):
        args.append("--cpu-strict-batch")
    if ccfg.get("prio_batch") is not None:
        args += ["--prio-batch", str(ccfg["prio_batch"])]
    if ccfg.get("poll_batch") is not None:
        args += ["--poll-batch", str(ccfg["poll_batch"])]
    if ccfg.get("numa"):
        args += ["--numa", ccfg["numa"]]

    # KV Cache
    if kcfg.get("no_kv_offload"):
        args.append("--no-kv-offload")
    if kcfg.get("cache_type_k"):
        args += ["--cache-type-k", kcfg["cache_type_k"]]
    if kcfg.get("cache_type_v"):
        args += ["--cache-type-v", kcfg["cache_type_v"]]
    if kcfg.get("cache_type_k_draft"):
        args += ["--cache-type-k-draft", kcfg["cache_type_k_draft"]]
    if kcfg.get("cache_type_v_draft"):
        args += ["--cache-type-v-draft", kcfg["cache_type_v_draft"]]
    if kcfg.get("defrag_thold") is not None:
        args += ["--defrag-thold", str(kcfg["defrag_thold"])]
    if not kcfg.get("cache_prompt", True):
        args.append("--no-cache-prompt")
    if kcfg.get("cache_reuse") is not None:
        args += ["--cache-reuse", str(kcfg["cache_reuse"])]
    if kcfg.get("rope_scaling") and kcfg["rope_scaling"] != "none":
        args += ["--rope-scaling", kcfg["rope_scaling"]]
    if kcfg.get("rope_scale") is not None:
        args += ["--rope-scale", str(kcfg["rope_scale"])]
    if kcfg.get("rope_freq_base") is not None:
        args += ["--rope-freq-base", str(kcfg["rope_freq_base"])]
    if kcfg.get("rope_freq_scale") is not None:
        args += ["--rope-freq-scale", str(kcfg["rope_freq_scale"])]
    if kcfg.get("yarn_orig_ctx") is not None:
        args += ["--yarn-orig-ctx", str(kcfg["yarn_orig_ctx"])]
    if kcfg.get("yarn_ext_factor") is not None:
        args += ["--yarn-ext-factor", str(kcfg["yarn_ext_factor"])]
    if kcfg.get("yarn_attn_factor") is not None:
        args += ["--yarn-attn-factor", str(kcfg["yarn_attn_factor"])]
    if kcfg.get("yarn_beta_slow") is not None:
        args += ["--yarn-beta-slow", str(kcfg["yarn_beta_slow"])]
    if kcfg.get("yarn_beta_fast") is not None:
        args += ["--yarn-beta-fast", str(kcfg["yarn_beta_fast"])]

    # Memory / Flash Attention
    fa = memcfg.get("flash_attn")
    if fa and fa != "auto":
        args += ["--flash-attn", "on" if fa is True else "off" if fa is False else str(fa)]
    if memcfg.get("direct_io"):
        args.append("--direct-io")
    if not memcfg.get("op_offload", True):
        args.append("--no-op-offload")
    if not memcfg.get("mmproj_offload", True):
        args.append("--no-mmproj-offload")
    if not memcfg.get("repack", True):
        args.append("--no-repack")
    if not memcfg.get("fit", True):
        args.append("--fit", "off")
    if memcfg.get("fit_target") is not None:
        args += ["--fit-target", str(memcfg["fit_target"])]
    if memcfg.get("fit_ctx") is not None:
        args += ["--fit-ctx", str(memcfg["fit_ctx"])]

    # Quantization overrides
    if qcfg.get("override_tensor"):
        args += ["--override-tensor", qcfg["override_tensor"]]
    if qcfg.get("cpu_moe"):
        args.append("--cpu-moe")
    if qcfg.get("n_cpu_moe") is not None:
        args += ["--n-cpu-moe", str(qcfg["n_cpu_moe"])]

    # Sampling
    if scfg_samp.get("temp") is not None:
        args += ["--temp", str(scfg_samp["temp"])]
    if scfg_samp.get("top_k") is not None:
        args += ["--top-k", str(scfg_samp["top_k"])]
    if scfg_samp.get("top_p") is not None:
        args += ["--top-p", str(scfg_samp["top_p"])]
    if scfg_samp.get("min_p") is not None:
        args += ["--min-p", str(scfg_samp["min_p"])]
    if scfg_samp.get("top_n_sigma") is not None:
        args += ["--top-n-sigma", str(scfg_samp["top_n_sigma"])]
    if scfg_samp.get("typical_p") is not None:
        args += ["--typical", str(scfg_samp["typical_p"])]
    if scfg_samp.get("xtc_probability") is not None:
        args += ["--xtc-probability", str(scfg_samp["xtc_probability"])]
    if scfg_samp.get("xtc_threshold") is not None:
        args += ["--xtc-threshold", str(scfg_samp["xtc_threshold"])]
    if scfg_samp.get("repeat_last_n") is not None:
        args += ["--repeat-last-n", str(scfg_samp["repeat_last_n"])]
    if scfg_samp.get("repeat_penalty") is not None:
        args += ["--repeat-penalty", str(scfg_samp["repeat_penalty"])]
    if scfg_samp.get("presence_penalty") is not None:
        args += ["--presence-penalty", str(scfg_samp["presence_penalty"])]
    if scfg_samp.get("frequency_penalty") is not None:
        args += ["--frequency-penalty", str(scfg_samp["frequency_penalty"])]
    if scfg_samp.get("dry_multiplier") is not None:
        args += ["--dry-multiplier", str(scfg_samp["dry_multiplier"])]
    if scfg_samp.get("dry_base") is not None:
        args += ["--dry-base", str(scfg_samp["dry_base"])]
    if scfg_samp.get("dry_allowed_length") is not None:
        args += ["--dry-allowed-length", str(scfg_samp["dry_allowed_length"])]
    if scfg_samp.get("dry_penalty_last_n") is not None:
        args += ["--dry-penalty-last-n", str(scfg_samp["dry_penalty_last_n"])]
    if scfg_samp.get("dry_sequence_breaker"):
        args += ["--dry-sequence-breaker", scfg_samp["dry_sequence_breaker"]]
    if scfg_samp.get("mirostat") is not None:
        args += ["--mirostat", str(scfg_samp["mirostat"])]
    if scfg_samp.get("mirostat_lr") is not None:
        args += ["--mirostat-lr", str(scfg_samp["mirostat_lr"])]
    if scfg_samp.get("mirostat_ent") is not None:
        args += ["--mirostat-ent", str(scfg_samp["mirostat_ent"])]
    if scfg_samp.get("adaptive_target") is not None:
        args += ["--adaptive-target", str(scfg_samp["adaptive_target"])]
    if scfg_samp.get("adaptive_decay") is not None:
        args += ["--adaptive-decay", str(scfg_samp["adaptive_decay"])]
    if scfg_samp.get("dynatemp_range") is not None:
        args += ["--dynatemp-range", str(scfg_samp["dynatemp_range"])]
    if scfg_samp.get("dynatemp_exp") is not None:
        args += ["--dynatemp-exp", str(scfg_samp["dynatemp_exp"])]
    if scfg_samp.get("samplers"):
        args += ["--samplers", scfg_samp["samplers"]]
    if scfg_samp.get("sampler_seq"):
        args += ["--sampler-seq", scfg_samp["sampler_seq"]]
    if scfg_samp.get("seed") is not None:
        args += ["-s", str(scfg_samp["seed"])]
    if scfg_samp.get("ignore_eos"):
        args.append("--ignore-eos")
    if scfg_samp.get("logit_bias"):
        args += ["--logit-bias", scfg_samp["logit_bias"]]
    if scfg_samp.get("grammar"):
        args += ["--grammar", scfg_samp["grammar"]]
    if scfg_samp.get("grammar_file"):
        args += ["--grammar-file", scfg_samp["grammar_file"]]
    if scfg_samp.get("json_schema"):
        args += ["--json-schema", scfg_samp["json_schema"]]
    if scfg_samp.get("json_schema_file"):
        args += ["--json-schema-file", scfg_samp["json_schema_file"]]
    if scfg_samp.get("backend_sampling"):
        args.append("--backend-sampling")

    # Adapters (LoRA / Control Vectors)
    if acfg.get("lora"):
        args += ["--lora", acfg["lora"]]
    if acfg.get("lora_scaled"):
        args += ["--lora-scaled", acfg["lora_scaled"]]
    if acfg.get("lora_init_without_apply"):
        args.append("--lora-init-without-apply")
    if acfg.get("control_vector"):
        args += ["--control-vector", acfg["control_vector"]]
    if acfg.get("control_vector_scaled"):
        args += ["--control-vector-scaled", acfg["control_vector_scaled"]]
    if acfg.get("control_vector_layer_range"):
        args += ["--control-vector-layer-range", acfg["control_vector_layer_range"]]

    # Speculative decoding (draft model)
    if specfg.get("draft_model"):
        args += ["--model-draft", specfg["draft_model"]]
    if specfg.get("draft_hf_repo"):
        args += ["--hf-repo-draft", specfg["draft_hf_repo"]]
    if specfg.get("draft_ngl") is not None:
        args += ["--n-gpu-layers-draft", str(specfg["draft_ngl"])]
    if specfg.get("draft_threads") is not None:
        args += ["--threads-draft", str(specfg["draft_threads"])]
    if specfg.get("draft_cache_type_k"):
        args += ["--cache-type-k-draft", specfg["draft_cache_type_k"]]
    if specfg.get("draft_cache_type_v"):
        args += ["--cache-type-v-draft", specfg["draft_cache_type_v"]]

    # Multimodal
    if mmcfg.get("mmproj") and mmcfg["mmproj"] != "none":
        mmproj_path = find_mmproj(model_path)
        if mmproj_path and mmcfg["mmproj"] == "auto":
            args += ["--mmproj", mmproj_path]
        elif mmcfg["mmproj"] != "auto":
            args += ["--mmproj", mmcfg["mmproj"]]
    if mmcfg.get("image_min_tokens"):
        args += ["--image-min-tokens", str(mmcfg["image_min_tokens"])]
    if mmcfg.get("image_max_tokens"):
        args += ["--image-max-tokens", str(mmcfg["image_max_tokens"])]
    if mmcfg.get("mtmd_batch_max_tokens"):
        args += ["--mtmd-batch-max-tokens", str(mmcfg["mtmd_batch_max_tokens"])]

    # Reasoning / Thinking
    if rcfg.get("reasoning") and rcfg["reasoning"] != "auto":
        args += ["--reasoning", rcfg["reasoning"]]
    if rcfg.get("reasoning_format") and rcfg["reasoning_format"] != "auto":
        args += ["--reasoning-format", rcfg["reasoning_format"]]
    if rcfg.get("reasoning_budget") is not None:
        args += ["--reasoning-budget", str(rcfg["reasoning_budget"])]
    if rcfg.get("reasoning_budget_message"):
        args += ["--reasoning-budget-message", rcfg["reasoning_budget_message"]]

    # Chat template
    if chatcfg.get("chat_template"):
        args += ["--chat-template", chatcfg["chat_template"]]
    if chatcfg.get("chat_template_file"):
        args += ["--chat-template-file", chatcfg["chat_template_file"]]
    if chatcfg.get("chat_template_kwargs"):
        args += ["--chat-template-kwargs", chatcfg["chat_template_kwargs"]]
    if not chatcfg.get("jinja", True):
        args.append("--no-jinja")
    if chatcfg.get("skip_chat_parsing"):
        args.append("--skip-chat-parsing")
    if not chatcfg.get("prefill_assistant", True):
        args.append("--no-prefill-assistant")

    # Server features
    if not fcfg.get("ui", True):
        args.append("--no-ui")
    if fcfg.get("ui_config"):
        args += ["--ui-config", fcfg["ui_config"]]
    if fcfg.get("ui_config_file"):
        args += ["--ui-config-file", fcfg["ui_config_file"]]
    if fcfg.get("ui_mcp_proxy"):
        args.append("--ui-mcp-proxy")
    if fcfg.get("tools"):
        args += ["--tools", fcfg["tools"]]
    if fcfg.get("embedding"):
        args.append("--embedding")
    if fcfg.get("reranking"):
        args.append("--reranking")
    if fcfg.get("metrics"):
        args.append("--metrics")
    if fcfg.get("props"):
        args.append("--props")
    if not fcfg.get("slots", True):
        args.append("--no-slots")
    if fcfg.get("slot_save_path"):
        args += ["--slot-save-path", fcfg["slot_save_path"]]
    if fcfg.get("media_path"):
        args += ["--media-path", fcfg["media_path"]]
    if fcfg.get("models_dir"):
        args += ["--models-dir", fcfg["models_dir"]]
    if fcfg.get("models_preset"):
        args += ["--models-preset", fcfg["models_preset"]]
    if fcfg.get("models_max") is not None:
        args += ["--models-max", str(fcfg["models_max"])]
    if not fcfg.get("models_autoload", True):
        args.append("--no-models-autoload")

    # Authentication & SSL
    if authcfg.get("api_key"):
        args += ["--api-key", authcfg["api_key"]]
    if authcfg.get("api_key_file"):
        args += ["--api-key-file", authcfg["api_key_file"]]
    if authcfg.get("ssl_key_file"):
        args += ["--ssl-key-file", authcfg["ssl_key_file"]]
    if authcfg.get("ssl_cert_file"):
        args += ["--ssl-cert-file", authcfg["ssl_cert_file"]]

    # Logging
    if logcfg.get("log_disable"):
        args.append("--log-disable")
    if logcfg.get("log_file"):
        args += ["--log-file", logcfg["log_file"]]
    if logcfg.get("log_colors") and logcfg["log_colors"] != "auto":
        args += ["--log-colors", logcfg["log_colors"]]
    if logcfg.get("verbosity") is not None:
        args += ["--verbosity", str(logcfg["verbosity"])]
    if not logcfg.get("log_prefix", True):
        args.append("--no-log-prefix")
    if not logcfg.get("log_timestamps", True):
        args.append("--no-log-timestamps")
    if logcfg.get("log_prompts_dir"):
        args += ["--log-prompts-dir", logcfg["log_prompts_dir"]]

    # Performance
    if perfcfg.get("perf"):
        args.append("--perf")
    if perfcfg.get("check_tensors"):
        args.append("--check-tensors")

    # Slots & context
    if slotcfg.get("slot_prompt_similarity") is not None:
        args += ["--slot-prompt-similarity", str(slotcfg["slot_prompt_similarity"])]
    if slotcfg.get("keep") is not None:
        args += ["--keep", str(slotcfg["keep"])]
    if slotcfg.get("swa_full"):
        args.append("--swa-full")
    if slotcfg.get("n_predict") is not None:
        args += ["-n", str(slotcfg["n_predict"])]

    # Embeddings
    if embcfg.get("embd_normalize") is not None:
        args += ["--embd-normalize", str(embcfg["embd_normalize"])]

    # TTS
    if ttscfg.get("model_vocoder"):
        args += ["--model-vocoder", ttscfg["model_vocoder"]]
    if ttscfg.get("tts_use_guide_tokens"):
        args.append("--tts-use-guide-tokens")

    # Network
    if netcfg.get("offline"):
        args.append("--offline")

    # Override KV
    if overcfg.get("override_kv"):
        args += ["--override-kv", overcfg["override_kv"]]

    # Identity
    if idcfg.get("alias"):
        args += ["--alias", idcfg["alias"]]
    if idcfg.get("tags"):
        args += ["--tags", idcfg["tags"]]

    # Sleep
    if sleepcfg.get("sleep_idle_seconds") is not None:
        args += ["--sleep-idle-seconds", str(sleepcfg["sleep_idle_seconds"])]

    # Server timeouts
    if scfg.get("timeout") is not None:
        args += ["--timeout", str(scfg["timeout"])]
    if scfg.get("sse_ping_interval") is not None:
        args += ["--sse-ping-interval", str(scfg["sse_ping_interval"])]
    if scfg.get("threads_http") is not None:
        args += ["--threads-http", str(scfg["threads_http"])]

    # Paths
    if scfg.get("path"):
        args += ["--path", scfg["path"]]
    if scfg.get("api_prefix"):
        args += ["--api-prefix", scfg["api_prefix"]]

    return args


# ─── Screens ─────────────────────────────────────────────────────────────────

class ModelScreen(Screen):
    """Model selection — DataTable with per-row config label."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "app.quit", "Quit", show=True),
        Binding("enter", "select_model", "Launch Server", show=True),
        Binding("r", "refresh", "Refresh", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._models: list[ModelEntry] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield DataTable(id="model-table", cursor_type="row", zebra_stripes=True)
        yield Static(
            "  ↑↓ navigate   Enter launch   r refresh   q quit",
            id="hint",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._load_models()

    def _load_models(self) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns(" # ", "Model", "Size", "Config")

        self._models = find_models()
        for i, m in enumerate(self._models, 1):
            _, cfg_file = load_model_config(m.rel)
            cfg_label = cfg_file.replace(".yaml", "")
            table.add_row(
                f"[bold]{i}[/]",
                m.rel,
                f"[cyan]{m.size_gb:.1f} GB[/]",
                f"[dim]{cfg_label}[/]",
                key=str(i - 1),   # 0-indexed key
            )
        if self._models:
            table.move_cursor(row=0)
        table.focus()

    def action_refresh(self) -> None:
        self._load_models()
        self.notify("Model list refreshed")

    def action_select_model(self) -> None:
        table = self.query_one(DataTable)
        if table.cursor_row is None:
            return
        idx = table.cursor_row
        if idx >= len(self._models):
            return
        model = self._models[idx]
        cfg, cfg_name = load_model_config(model.rel)
        self.app.push_screen(ServerScreen(model, cfg, cfg_name))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_select_model()


class ServerScreen(Screen):
    """Live server log with clickable URL when ready."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q,escape", "back", "Stop & Back", show=True),
        Binding("ctrl+c", "back", "Stop Server", show=False),
        Binding("b", "open_browser", "Open Browser", show=True),
        Binding("c", "copy_logs", "Copy Logs", show=True),
    ]

    def __init__(
        self,
        model:    ModelEntry,
        cfg:      dict,
        cfg_name: str,
    ) -> None:
        super().__init__()
        self.model    = model
        self.cfg      = cfg
        self.cfg_name = cfg_name
        self._proc:          asyncio.subprocess.Process | None = None
        self._server_url:    str = ""
        self._server_ready:  bool = False
        self._log_buffer: list[str] = []

    # ── Layout ───────────────────────────────────────────

    def compose(self) -> ComposeResult:
        scfg = self.cfg.get("server", {})
        mcfg = self.cfg.get("model", {})
        sampcfg = self.cfg.get("sampling", {})
        yield Header(show_clock=True)
        with Container(id="info-bar"):
            yield Static(
                f"[bold cyan]Model[/]  {self.model.name}\n"
                f"[bold cyan]Config[/] {self.cfg_name}   "
                f"[bold cyan]Port[/] {scfg.get('port', 8080)}   "
                f"[bold cyan]ctx[/] {mcfg.get('ctx', '?')}   "
                f"[bold cyan]ngl[/] {mcfg.get('ngl', '?')}   "
                f"[bold cyan]temp[/] {sampcfg.get('temp', '?')}"
            )
        yield RichLog(id="server-log", highlight=True, markup=False, wrap=True)
        with Horizontal(id="copy-bar"):
            yield Button("📋  Copy Logs", id="copy-btn", variant="primary")
        # URL bar — hidden until server is ready
        with Container(id="url-bar"):
            yield Static("✅  Server Ready", id="url-main")
            yield Label("")   # spacer
            yield Link("", id="url-chat",    tooltip="Open llama.cpp web UI")
            yield Link("", id="url-api",     tooltip="OpenAI-compatible endpoint")
            yield Link("", id="url-metrics", tooltip="Prometheus metrics")
        yield Footer()

    def on_mount(self) -> None:
        self._start_server()

    # ── Server worker ─────────────────────────────────────

    @work(exclusive=True, thread=False)
    async def _start_server(self) -> None:  # noqa: C901
        log  = self.query_one(RichLog)

        args = build_server_args(self.model.path, self.cfg)

        env = os.environ.copy()
        env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        lib_dir = os.path.join(ROOT_DIR, "llama-b9656")
        env["LD_LIBRARY_PATH"] = (
            f"{lib_dir}:/opt/rocm-7.2.4/lib:" + env.get("LD_LIBRARY_PATH", "")
        )

        log.write(f"▶  {' '.join(os.path.basename(a) if i == 0 else a for i, a in enumerate(args))}")
        log.write("─" * 60)

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            log.write(f"[ERROR] llama-server not found:\n  {LLAMA_SERVER}")
            return

        self._proc = proc

        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            log.write(line)
            self._log_buffer.append(line)
            log.scroll_end(animate=False)

            if not self._server_ready and "HTTP server listening" in line:
                self._server_ready = True
                host = self.cfg.get("server", {}).get("host", "127.0.0.1")
                port = self.cfg.get("server", {}).get("port", 8080)
                self._server_url = f"http://{host}:{port}"
                self.call_from_thread(self._show_url_bar, host, port)

        await proc.wait()
        log.write("─" * 60)
        log.write("⏹  Server stopped.")

    def _show_url_bar(self, host: str, port: int) -> None:
        """Called from worker thread — update URL bar and open browser."""
        base = f"http://{host}:{port}"
        url_chat    = base
        url_api     = f"{base}/v1/chat/completions"
        url_metrics = f"{base}/metrics"

        self.query_one("#url-chat",    Link).update(f"🌐  {url_chat}")
        self.query_one("#url-chat",    Link).url = url_chat
        self.query_one("#url-api",     Link).update(f"🔗  {url_api}")
        self.query_one("#url-api",     Link).url = url_api
        self.query_one("#url-metrics", Link).update(f"📡  {url_metrics}")
        self.query_one("#url-metrics", Link).url = url_metrics

        url_bar = self.query_one("#url-bar")
        url_bar.add_class("ready")

        self.notify(f"Server ready at {base}", title="✅ llama-server", severity="information")

        # Auto-open browser
        try:
            subprocess.Popen(
                ["xdg-open", url_chat],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    # ── Actions ───────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy-btn":
            self.action_copy_logs()

    def action_copy_logs(self) -> None:
        if not self._log_buffer:
            self.notify("No logs to copy", severity="warning")
            return
        text = "\n".join(self._log_buffer)
        try:
            import pyperclip
            pyperclip.copy(text)
            self.notify("Logs copied to clipboard", severity="information")
        except ImportError:
            self.notify("Install pyperclip to use copy: pip install pyperclip", severity="error")
        except Exception as e:
            self.notify(f"Copy failed: {e}", severity="error")

    def action_back(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
        self.app.pop_screen()

    def action_open_browser(self) -> None:
        if self._server_url:
            try:
                subprocess.Popen(
                    ["xdg-open", self._server_url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        else:
            self.notify("Server not ready yet", severity="warning")


# ─── App ──────────────────────────────────────────────────────────────────────

class LlamaTui(App):
    """llama-tui — GGUF Model Picker + llama-server launcher ★(◕‿◕)ノ"""

    TITLE    = "★ llama-tui ★"
    SUB_TITLE = "GGUF Model Picker — RX 6800 XT ROCm"
    CSS      = APP_CSS

    def on_mount(self) -> None:
        self.push_screen(ModelScreen())