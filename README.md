<div align="center">

# 🜂 bwoc-chat

**Native desktop chat for BWOC agents — one agent, or a whole team, in one window.**

[![Rust](https://img.shields.io/badge/rust-2024-orange.svg?logo=rust)](https://www.rust-lang.org)
[![UI: egui](https://img.shields.io/badge/ui-egui-blue.svg)](https://github.com/emilk/egui)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Backends](https://img.shields.io/badge/backends-ollama%20%C2%B7%20openai--compatible-555.svg)](#-requirements)

A tiny [egui](https://github.com/emilk/egui) window that talks to a BWOC agent —
streaming replies, live tool activity, and inline permission prompts.

</div>

---

## ✨ What it is

`bwoc-chat` is a thin **renderer** over the protocol the framework already speaks.
It spawns `bwoc-harness --chat` for an agent and draws the
[`bwoc_core::chat_proto`](../bwoc-framwork/crates/bwoc-core/src/chat_proto.rs)
event stream — the same wire format the in-terminal `bwoc chat --tui` uses.

> The **harness** owns each session, its tools, model calls, and the
> guardrail → permission pipeline. This window only renders events and routes
> your messages + permission decisions. Name several agents and each becomes its
> own harness subprocess, multiplexed into one shared transcript.

## 🚀 Features

- 🪟 **Native window** — pure-Rust [egui](https://github.com/emilk/egui)/eframe, no webview.
- 👥 **Team chat** — name several agents for a group: a message broadcasts to all,
  or `@name` addresses one; each reply streams in tagged + coloured by agent.
- ⚡ **Streaming** — assistant tokens appear live as the model generates them.
- 🔧 **Tool activity pane** — every `🔧 tool call` and `✓/✗ result` in real time, per agent.
- ⚠️ **Inline permission** — `ask`-mode tools surface an **Allow / Deny** bar.
- 🧭 **Zero config** — model + endpoint come from each agent's `config.manifest.json`.
- 🧹 **Clean shutdown** — closing the window sends `quit` and reaps every harness.

## 📦 Install

```bash
# from this directory
cargo install --path . --force      # → bwoc-chat on your PATH
# …or just run it
cargo run -- <agent>
```

## 🖱️ Usage

```bash
bwoc-chat                                  # personal assistant (~/.bwoc/personal)
bwoc-chat <agent>                          # one workspace agent, 1:1
bwoc-chat <a> <b> <c>                      # team chat — several agents, one window
bwoc-chat --here | --path <dir>            # an agent directory directly, no workspace
#         [--workspace <dir>] [--model <m>] [--endpoint <url>]
```

Named `<agent>`s are resolved from the workspace registry (via `--workspace`,
`$BWOC_WORKSPACE`, or an ancestor `.bwoc/workspace.toml`).

| Key / action | Effect |
| --- | --- |
| **Enter** / **Send** | send your message (broadcasts to every agent in a team) |
| `@name …` | address just one agent in a team (matched on its short name) |
| **Allow** / **Deny** | answer a pending permission request (per agent) |
| `/help` `/tools` `/clear` `/forget` `/quit` | client-side commands |
| close window | end the session (sends `quit`, reaps every harness) |

## 🪟 The window

```
┌ status:  agent · model · backend · tokens ─────────────┐
│ conversation                          │ tools          │
│   you:   refactor the parser          │ 🔧 read_file   │
│   agent: sure — I'd split the lexer…  │ 🔧 edit_file   │
│          (tokens streaming in…)        │ ✓ edit_file    │
├ input  [ message…                    ] [ Send ]─────────┤
└  ⚠ permission: run_command …          [ Allow ] [ Deny ]┘
```

## 🧩 How it works

```
bwoc-chat  (egui window)
   │  stdin  → ChatInput  (user / permission / quit)
   ▼  stdout ← ChatEvent  (ready / token / tool_call / tool_result /
bwoc-harness --chat            permission_request / turn_end / bye)
   └─ run_loop + tools + guardrails + permission policy → model endpoint
```

A reader `std::thread` parses the child's stdout lines into `ChatEvent`s onto an
`mpsc` channel; the egui loop drains it each frame, repaints, and writes
`ChatInput` lines back to the child's stdin.

## 📋 Requirements

- A Rust toolchain (2024 edition / rustc ≥ 1.85).
- `bwoc-harness` on `PATH` (or installed beside the running binary — it's
  resolved as a sibling of the current executable).
- A reachable model endpoint for a **harness backend** (`ollama` /
  `openai-compatible`) — e.g. `ollama serve`. Vendor-CLI backends (claude /
  codex / kimi / agy) aren't rendered here; use `bwoc spawn` for those.

## 🏗️ Why a separate project

Keeps the heavy GUI dep tree (winit / glow) out of the lean
[`bwoc-framwork`](../bwoc-framwork) workspace. `bwoc-chat` depends **only** on
`bwoc-core` (by path) for the protocol + agent resolution — never on `bwoc-cli`
or `bwoc-harness`. The harness is a runtime subprocess, not a build dependency.

## License

MIT.
