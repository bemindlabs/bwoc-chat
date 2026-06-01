# bwoc-chat

Native desktop chat for a single BWOC agent — **one agent, one window**.

A thin [egui](https://github.com/emilk/egui) frontend over the protocol the
framework already speaks: it spawns `bwoc-harness --chat` for an agent and
renders the `bwoc_core::chat_proto` event stream (the same wire format the
in-terminal `bwoc chat --tui` uses). The harness owns the session, tools, model
calls, and the guardrail→permission pipeline; this window only renders events
and sends user messages + permission decisions.

## Run

```bash
bwoc-chat <agent> [--workspace <dir>] [--model <m>] [--endpoint <url>]
```

- `<agent>` is resolved from the workspace registry (`--workspace`, `$BWOC_WORKSPACE`,
  or an ancestor `.bwoc/workspace.toml`). Only the harness backends
  (`ollama` / `openai-compatible`) are supported — they're the ones that emit a
  chat stream. Model + endpoint default to the agent's `config.manifest.json`.
- Needs `bwoc-harness` on `PATH` (or installed beside the running binary) and a
  reachable model endpoint (e.g. `ollama serve`).

## Window

```
┌ status: agent · model · backend · tokens ──────────┐
│ conversation (you / agent / system)   │ tools      │
│                                        │ 🔧 calls   │
│                                        │ ✓✗ results │
├ input  [ message…                ] [Send]──────────┤
└  ⚠ permission: <tool> … [Allow] [Deny]  (when asked)┘
```

## Why a separate project

Keeps the heavy GUI dep tree (winit / glow) out of the lean `bwoc-framwork`
workspace. It depends only on `bwoc-core` (by path) for the protocol + agent
resolution — never on `bwoc-cli` or `bwoc-harness` (the harness is a runtime
subprocess, resolved as a sibling of the running binary).
