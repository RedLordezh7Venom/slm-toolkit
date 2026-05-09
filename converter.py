#!/usr/bin/env python3
"""
SLM Toolkit Converter — graph-based model conversion TUI
Supports: HF → GGUF → Quantized, HF → ONNX, GGML → GGUF, LoRA → GGUF
"""

import os
import sys
import subprocess
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Label, Button, Input, Select,
    Static, ListView, ListItem, Log, ProgressBar, Markdown
)
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from textual import work, on
from rich.text import Text

# ─── Conversion Graph ────────────────────────────────────────────────────────
#
#   HF_MODEL ──→ GGUF ──→ QUANTIZED_GGUF
#       │          ↑
#       │        GGML
#       │
#       └──→ ONNX
#   LORA ──→ GGUF
#
LLAMA_DIR = Path(__file__).parent / "llama.cpp"
VENV_PYTHON = LLAMA_DIR / ".venv" / "bin" / "python3"

# Quant choices
QUANT_TYPES = [
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S",
    "Q8_0", "Q6_K", "Q3_K_M", "Q2_K",
    "IQ4_NL", "IQ3_M", "F16", "BF16",
]

# Conversion edges: (src_fmt, dst_fmt) → list of steps
CONV_GRAPH = {
    ("hf",       "gguf"):      ["hf_to_gguf"],
    ("hf",       "quantized"): ["hf_to_gguf", "gguf_to_quant"],
    ("hf",       "onnx"):      ["hf_to_onnx"],
    ("gguf",     "quantized"): ["gguf_to_quant"],
    ("ggml",     "gguf"):      ["ggml_to_gguf"],
    ("ggml",     "quantized"): ["ggml_to_gguf", "gguf_to_quant"],
    ("lora",     "gguf"):      ["lora_to_gguf"],
    ("lora",     "quantized"): ["lora_to_gguf", "gguf_to_quant"],
}

@dataclass
class ConvJob:
    src_fmt: str
    dst_fmt: str
    src_path: str
    out_path: str
    outtype: str = "f16"
    qtype: str = "Q4_K_M"
    hf_repo: str = ""
    base_model: str = ""   # for LoRA
    steps: list = field(default_factory=list)


def python_cmd():
    """Return python to use — prefer venv, fall back to system."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def build_commands(job: ConvJob) -> list[tuple[str, list]]:
    """Resolve step list → actual shell commands."""
    cmds = []
    # intermediate GGUF path when chaining
    gguf_path = job.src_path
    if job.src_fmt in ("hf", "ggml", "lora"):
        gguf_path = str(Path(job.out_path).with_suffix("")) + "_converted.gguf"

    for step in job.steps:
        if step == "hf_to_gguf":
            outfile_arg = ["--outfile", gguf_path] if job.out_path else []
            cmds.append(("Convert HF → GGUF", [
                python_cmd(),
                str(LLAMA_DIR / "convert_hf_to_gguf.py"),
                job.src_path,
                "--outtype", job.outtype,
                *outfile_arg,
            ]))

        elif step == "gguf_to_quant":
            in_f = gguf_path if job.src_fmt != "gguf" else job.src_path
            out_f = job.out_path or str(Path(in_f).with_suffix("")) + f"-{job.qtype}.gguf"
            cmds.append(("Quantize GGUF", [
                str(LLAMA_DIR / "llama-quantize"),
                in_f, out_f, job.qtype,
            ]))

        elif step == "ggml_to_gguf":
            cmds.append(("Convert GGML → GGUF", [
                python_cmd(),
                str(LLAMA_DIR / "convert_llama_ggml_to_gguf.py"),
                "--input", job.src_path,
                "--output", gguf_path,
            ]))

        elif step == "lora_to_gguf":
            args = [
                python_cmd(),
                str(LLAMA_DIR / "convert_lora_to_gguf.py"),
                job.src_path,
            ]
            if job.base_model:
                args += ["--base", job.base_model]
            args += ["--outfile", gguf_path]
            cmds.append(("Convert LoRA → GGUF", args))

        elif step == "hf_to_onnx":
            out_f = job.out_path or str(Path(job.src_path).stem) + "_onnx"
            cmds.append(("Export HF → ONNX", [
                sys.executable, "-m", "optimum.exporters.onnx",
                "--model", job.src_path,
                out_f,
            ]))

    return cmds


# ─── Screens ─────────────────────────────────────────────────────────────────

class HFSearchScreen(ModalScreen):
    """Modal: search HuggingFace Hub and pick a model."""
    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self._results = []

    def compose(self) -> ComposeResult:
        with Container(id="hf-modal"):
            yield Label("🤗  Search HuggingFace Hub", id="hf-title")
            yield Input(placeholder="e.g. mistralai/Mistral-7B-v0.1  or  SmolLM", id="hf-query")
            yield Button("Search", id="hf-search-btn", variant="primary")
            yield ListView(id="hf-results")
            yield Button("Download Selected ↓", id="hf-dl-btn", variant="success")
            yield Label("", id="hf-status")

    @on(Button.Pressed, "#hf-search-btn")
    def do_search(self):
        query = self.query_one("#hf-query", Input).value.strip()
        if not query:
            return
        self.query_one("#hf-status", Label).update("Searching…")
        threading.Thread(target=self._search, args=(query,), daemon=True).start()

    def _search(self, query):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            models = list(api.list_models(search=query, limit=30, sort="downloads"))
            self._results = [m.id for m in models]
            self.app.call_from_thread(self._populate, self._results)
        except Exception as e:
            self.app.call_from_thread(
                self.query_one("#hf-status", Label).update, f"[red]Error: {e}[/red]"
            )

    def _populate(self, results):
        lv = self.query_one("#hf-results", ListView)
        lv.clear()
        for r in results:
            lv.append(ListItem(Label(r)))
        self.query_one("#hf-status", Label).update(f"{len(results)} results")

    @on(Button.Pressed, "#hf-dl-btn")
    def do_download(self):
        lv = self.query_one("#hf-results", ListView)
        if lv.highlighted_child is None:
            self.query_one("#hf-status", Label).update("[yellow]Select a model first[/yellow]")
            return
        idx = lv.index
        if idx is None or idx >= len(self._results):
            return
        repo = self._results[idx]
        self.dismiss(repo)

    def on_dismiss(self, result):
        if result and self._callback:
            self._callback(result)


class JobScreen(Screen):
    """Run a ConvJob, streaming output live."""
    BINDINGS = [Binding("escape,q", "app.pop_screen", "Back")]

    def __init__(self, job: ConvJob):
        super().__init__()
        self._job = job

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("", id="job-status")
        yield ProgressBar(total=100, id="job-progress", show_eta=False)
        yield Log(id="job-log", highlight=True)
        yield Footer()

    def on_mount(self):
        self.run_job()

    @work(thread=True)
    def run_job(self):
        job = self._job
        cmds = build_commands(job)
        log = self.query_one("#job-log", Log)
        status = self.query_one("#job-status", Label)
        pb = self.query_one("#job-progress", ProgressBar)

        total = len(cmds)
        for i, (label, cmd) in enumerate(cmds):
            self.app.call_from_thread(status.update, f"Step {i+1}/{total}: {label}")
            self.app.call_from_thread(log.write_line, f"\n{'─'*60}")
            self.app.call_from_thread(log.write_line, f"▶  {label}")
            self.app.call_from_thread(log.write_line, "  " + " ".join(cmd))
            self.app.call_from_thread(log.write_line, f"{'─'*60}")

            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                for line in proc.stdout:
                    self.app.call_from_thread(log.write_line, line.rstrip())
                proc.wait()
                if proc.returncode != 0:
                    self.app.call_from_thread(status.update, f"[red]✗ Failed at: {label}[/red]")
                    return
            except FileNotFoundError as e:
                self.app.call_from_thread(log.write_line, f"[red]Binary not found: {e}[/red]")
                self.app.call_from_thread(status.update, "[red]✗ Binary not found — did you build llama.cpp?[/red]")
                return

            pct = int(((i + 1) / total) * 100)
            self.app.call_from_thread(pb.update, progress=pct)

        self.app.call_from_thread(status.update, "[green]✓ All steps completed![/green]")


# ─── Wizard ──────────────────────────────────────────────────────────────────

SRC_FORMATS = [
    ("hf",   "HuggingFace Model (directory / repo ID)"),
    ("gguf", "GGUF File"),
    ("ggml", "GGML File (legacy)"),
    ("lora", "LoRA Adapter"),
]

DST_FORMATS = {
    "hf":   [("gguf", "GGUF"), ("quantized", "Quantized GGUF"), ("onnx", "ONNX")],
    "gguf": [("quantized", "Quantized GGUF")],
    "ggml": [("gguf", "GGUF"), ("quantized", "Quantized GGUF")],
    "lora": [("gguf", "GGUF"), ("quantized", "Quantized GGUF")],
}


class WizardScreen(Screen):
    """Step-by-step wizard to configure and launch a conversion job."""
    BINDINGS = [Binding("escape,q", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="wizard-scroll"):
            yield Label("── Step 1 · Source Format", classes="step-title")
            yield Select(
                [(b, a) for a, b in SRC_FORMATS],
                id="src-fmt", prompt="Select source format…"
            )

            yield Label("── Step 2 · Source Path / Repo", classes="step-title")
            yield Horizontal(
                Input(placeholder="Path or HF repo ID", id="src-path"),
                Button("Browse HF 🤗", id="browse-hf", variant="default"),
                id="src-row",
            )

            yield Label("── Step 3 · Target Format", classes="step-title")
            yield Select([], id="dst-fmt", prompt="Select target format…")

            yield Label("── Step 4 · Options", classes="step-title")
            yield Label("Output Format (for HF→GGUF step):", classes="opt-label")
            yield Select(
                [("f16","f16"),("bf16","bf16"),("f32","f32"),("q8_0","q8_0"),("auto","auto")],
                id="outtype", value="f16"
            )
            yield Label("Quantization Type (if quantizing):", classes="opt-label")
            yield Select([(q, q) for q in QUANT_TYPES], id="qtype", value="Q4_K_M")
            yield Label("Base Model Path (LoRA only, optional):", classes="opt-label")
            yield Input(placeholder="Base model directory", id="base-model")
            yield Label("Output Path (optional — leave blank for auto):", classes="opt-label")
            yield Input(placeholder="e.g. /home/user/output.gguf", id="out-path")

            yield Label("── Conversion Plan", classes="step-title")
            yield Static("", id="plan-display")

            yield Button("▶  Start Conversion", id="start-btn", variant="success")
        yield Footer()

    def on_mount(self):
        self.query_one("#browse-hf").display = False

    @on(Select.Changed, "#src-fmt")
    def src_changed(self, event: Select.Changed):
        src = event.value
        if src is Select.BLANK:
            return
        # show HF browse only for hf format
        self.query_one("#browse-hf").display = (src == "hf")
        # update dst options
        dst_sel = self.query_one("#dst-fmt", Select)
        opts = DST_FORMATS.get(src, [])
        dst_sel.set_options([(label, val) for val, label in opts])
        self._update_plan()

    @on(Select.Changed, "#dst-fmt")
    def dst_changed(self, _):
        self._update_plan()

    @on(Input.Changed)
    def input_changed(self, _):
        self._update_plan()

    @on(Button.Pressed, "#browse-hf")
    def browse_hf(self):
        def on_result(repo):
            self.query_one("#src-path", Input).value = repo
            self._update_plan()
        self.app.push_screen(HFSearchScreen(on_result))

    def _update_plan(self):
        src = self.query_one("#src-fmt", Select).value
        dst = self.query_one("#dst-fmt", Select).value
        plan_widget = self.query_one("#plan-display", Static)
        if src is Select.BLANK or dst is Select.BLANK:
            plan_widget.update("")
            return
        steps = CONV_GRAPH.get((src, dst), [])
        if not steps:
            plan_widget.update("[red]✗ No conversion path available[/red]")
            return
        lines = ["[bold cyan]Steps:[/bold cyan]"]
        for i, s in enumerate(steps, 1):
            label = {
                "hf_to_gguf":   "Convert HuggingFace → GGUF",
                "gguf_to_quant":"Quantize GGUF",
                "ggml_to_gguf": "Convert GGML → GGUF",
                "lora_to_gguf": "Merge LoRA → GGUF",
                "hf_to_onnx":   "Export HuggingFace → ONNX",
            }.get(s, s)
            lines.append(f"  {i}. {label}")
        plan_widget.update("\n".join(lines))

    @on(Button.Pressed, "#start-btn")
    def start(self):
        src_fmt = self.query_one("#src-fmt", Select).value
        dst_fmt = self.query_one("#dst-fmt", Select).value
        src_path = self.query_one("#src-path", Input).value.strip()
        out_path = self.query_one("#out-path", Input).value.strip()
        outtype = self.query_one("#outtype", Select).value
        qtype = self.query_one("#qtype", Select).value
        base_model = self.query_one("#base-model", Input).value.strip()

        if src_fmt is Select.BLANK:
            self.notify("Select a source format", severity="error"); return
        if dst_fmt is Select.BLANK:
            self.notify("Select a target format", severity="error"); return
        if not src_path:
            self.notify("Enter a source path or repo ID", severity="error"); return

        steps = CONV_GRAPH.get((src_fmt, dst_fmt))
        if not steps:
            self.notify("No conversion path for this combination", severity="error"); return

        job = ConvJob(
            src_fmt=src_fmt, dst_fmt=dst_fmt,
            src_path=src_path, out_path=out_path,
            outtype=outtype or "f16",
            qtype=qtype or "Q4_K_M",
            base_model=base_model,
            steps=steps,
        )
        self.app.push_screen(JobScreen(job))


# ─── Main App ────────────────────────────────────────────────────────────────

MENU_CSS = """
Screen { background: #0d0d1a; }
#menu-box { align: center middle; height: 100%; }
#menu-title { text-align: center; color: #7dd3fc; text-style: bold; margin-bottom: 1; }
#menu-subtitle { text-align: center; color: #475569; margin-bottom: 2; }
.menu-btn { width: 50; margin: 0 0 1 0; }
"""

WIZARD_CSS = """
#wizard-scroll { padding: 1 3; }
.step-title { color: #7dd3fc; text-style: bold; margin-top: 1; }
.opt-label  { color: #94a3b8; margin-top: 1; }
#src-row { height: 3; }
#src-row Input { width: 1fr; }
#src-row Button { width: 16; margin-left: 1; }
#plan-display { background: #1e293b; padding: 1; border: round #334155; margin-top: 1; }
#start-btn { margin-top: 2; width: 30; }
"""

JOB_CSS = """
#job-status { color: #7dd3fc; text-style: bold; padding: 1; }
#job-progress { margin: 0 1 1 1; }
#job-log { height: 1fr; border: round #334155; }
"""

HF_CSS = """
HFSearchScreen { align: center middle; }
#hf-modal { background: #1e293b; border: round #7dd3fc; padding: 2; width: 70; height: 35; }
#hf-title { text-align: center; color: #7dd3fc; text-style: bold; margin-bottom: 1; }
#hf-results { height: 15; border: round #334155; margin: 1 0; }
#hf-status { color: #94a3b8; }
#hf-search-btn { margin-bottom: 1; }
#hf-dl-btn { margin-top: 1; width: 30; }
"""

FULL_CSS = MENU_CSS + WIZARD_CSS + JOB_CSS + HF_CSS


class MainMenu(Screen):
    BINDINGS = [Binding("q", "app.exit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="menu-box"):
            yield Label("⚡  SLM Toolkit — Model Converter", id="menu-title")
            yield Label("Graph-based conversion · Powered by llama.cpp", id="menu-subtitle")
            yield Button("🔄  Convert / Quantize a Model", id="btn-convert", variant="primary", classes="menu-btn")
            yield Button("🤗  Browse & Download from HuggingFace", id="btn-hf", classes="menu-btn")
            yield Button("❌  Quit", id="btn-quit", classes="menu-btn")
        yield Footer()

    @on(Button.Pressed, "#btn-convert")
    def go_wizard(self): self.app.push_screen(WizardScreen())

    @on(Button.Pressed, "#btn-hf")
    def go_hf(self):
        def on_dl(repo):
            self.notify(f"Downloading {repo}…", severity="information")
            threading.Thread(target=self._dl, args=(repo,), daemon=True).start()
        self.app.push_screen(HFSearchScreen(on_dl))

    def _dl(self, repo):
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo_id=repo)
            self.app.call_from_thread(self.notify, f"✓ Saved to {path}")
        except Exception as e:
            self.app.call_from_thread(self.notify, f"✗ {e}", severity="error")

    @on(Button.Pressed, "#btn-quit")
    def quit(self): self.app.exit()


class SLMConverterApp(App):
    CSS = FULL_CSS
    TITLE = "SLM Toolkit"
    SUB_TITLE = "Model Conversion Suite"
    BINDINGS = [Binding("ctrl+c", "app.exit", "Quit", show=False)]

    def on_mount(self):
        self.push_screen(MainMenu())


if __name__ == "__main__":
    SLMConverterApp().run()
