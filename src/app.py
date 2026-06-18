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

    # ── Layout ───────────────────────────────────────────

    def compose(self) -> ComposeResult:
        scfg = self.cfg["server"]
        yield Header(show_clock=True)
        with Container(id="info-bar"):
            yield Static(
                f"[bold cyan]Model[/]  {self.model.name}\n"
                f"[bold cyan]Config[/] {self.cfg_name}   "
                f"[bold cyan]Port[/] {scfg['port']}   "
                f"[bold cyan]ctx[/] {scfg['ctx']}   "
                f"[bold cyan]ngl[/] {scfg['ngl']}   "
                f"[bold cyan]temp[/] {scfg['temp']}"
            )
        yield RichLog(id="server-log", highlight=True, markup=False, wrap=True)
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
        scfg = self.cfg["server"]

        args = [
            LLAMA_SERVER,
            "-m", self.model.path,
            "--host", scfg["host"],
            "--port", str(scfg["port"]),
            "-ngl", str(scfg["ngl"]),
            "-c", str(scfg["ctx"]),
            "-t", str(scfg["threads"]),
            "--threads-batch", str(scfg["threads_batch"]),
            "--temp", str(scfg["temp"]),
            "--top-k", str(scfg["top_k"]),
            "--top-p", str(scfg["top_p"]),
        ]

        if scfg.get("flash_attn"):
            args += ["--flash-attn", "on"]
        if scfg.get("no_kv_offload"):
            args.append("--no-kv-offload")
        if scfg.get("cache_type_k"):
            args += ["--cache-type-k", scfg["cache_type_k"]]
        if scfg.get("cache_type_v"):
            args += ["--cache-type-v", scfg["cache_type_v"]]
        if scfg.get("reasoning_budget") is not None:
            args += ["--reasoning-budget", str(scfg["reasoning_budget"])]

        mmproj = find_mmproj(self.model.path)
        mcfg   = self.cfg.get("multimodal", {})
        if mmproj and mcfg.get("mmproj") == "auto":
            args += ["--mmproj", mmproj]
            log.write(f"🖼️  mmproj: {os.path.basename(mmproj)}")

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
            log.scroll_end(animate=False)

            if not self._server_ready and "HTTP server listening" in line:
                self._server_ready = True
                host = scfg["host"]
                port = scfg["port"]
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
