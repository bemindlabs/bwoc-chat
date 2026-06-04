//! `bwoc-chat [<agent>...]` — native desktop chat for BWOC agents in ONE window.
//!
//! A thin egui frontend over the protocol the framework already speaks: for each
//! agent it spawns a `bwoc-harness --chat` subprocess and renders the
//! `bwoc_core::chat_proto` event stream (the same wire format `bwoc chat --tui`
//! uses). The harness owns each session, its tools, model calls, and the
//! guardrail→permission pipeline; this window only renders events and routes
//! user messages + permission decisions.
//!
//! **One window, N agents (team chat).** Name one agent for a 1:1 chat, or
//! several for a group: the user's message broadcasts to every agent (or, with a
//! leading `@name`, to just one), and each reply streams into the shared
//! transcript tagged + coloured by agent. Each agent is an independent harness
//! subprocess — they answer the user in parallel; they do not (yet) see each
//! other's replies.
//!
//! Architecture (no async): per agent, a reader `std::thread` parses the child's
//! stdout lines into `ChatEvent`s onto an `mpsc` channel; the egui update loop
//! drains every channel each frame, repaints, and writes `ChatInput` lines back
//! to each child's stdin.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use bwoc_core::chat_proto::{ChatEvent, ChatInput};
use bwoc_core::design;
use bwoc_core::manifest::Manifest;
use bwoc_core::workspace::AgentsRegistry;
use eframe::egui;

/// Default OpenAI-compatible endpoint (Ollama) when the manifest has no
/// `baseUrl`. Mirrors the harness's own default.
const DEFAULT_ENDPOINT: &str = "http://localhost:11434/v1";

/// Per-agent accent colours, assigned by index so each agent is visually
/// distinct in the shared transcript and status bar.
const PALETTE: &[(u8, u8, u8)] = &[
    (0x9E, 0xE0, 0x93), // green
    (0xE0, 0xC0, 0x60), // amber
    (0xC0, 0x90, 0xE0), // violet
    (0x90, 0xC8, 0xE0), // sky
    (0xE0, 0x90, 0x90), // rose
    (0x80, 0xD8, 0xC0), // teal
];

/// Client-side slash commands: `(name, description)`. Surfaced as a filtered
/// list when the input starts with `/`; dispatched by [`ChatApp::run_command`].
const COMMANDS: &[(&str, &str)] = &[
    ("/help", "list commands"),
    ("/tools", "list each agent's available tools"),
    (
        "/mode",
        "permission mode: default | accept-edits | bypass | plan",
    ),
    ("/clear", "wipe the conversation + tool activity"),
    ("/forget", "clear every agent's memory of this conversation"),
    ("/quit", "close the window"),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let configs = match resolve(std::env::args().skip(1).collect()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "bwoc-chat: {e}\n\n\
                 usage: bwoc-chat [<agent>...] [--here | --path <dir>] [--workspace <dir>] \
                 [--model <m>] [--endpoint <url>]\n\
                 \x20 no agent      → personal assistant at ~/.bwoc/personal (created on first use)\n\
                 \x20 --here / .    → the current directory, no workspace\n\
                 \x20 --path <dir>  → that directory directly\n\
                 \x20 <agent>       → a named agent from the workspace registry\n\
                 \x20 <a> <b> <c>   → team chat: several agents in one window"
            );
            std::process::exit(2);
        }
    };

    // Spawn one harness session per agent. A spawn failure for any agent is
    // fatal — better to fail loudly than open a half-empty team window.
    let mut sessions = Vec::with_capacity(configs.len());
    for (i, cfg) in configs.iter().enumerate() {
        let color = palette(i);
        let session = if cfg.claude_code {
            AgentSession::spawn_claude_code(cfg, color)?
        } else {
            AgentSession::spawn(cfg, color)?
        };
        sessions.push(session);
    }

    let title = if sessions.len() == 1 {
        format!("bwoc · {}", sessions[0].id)
    } else {
        let names: Vec<&str> = sessions.iter().map(|s| short(&s.id)).collect();
        format!("bwoc team · {}", names.join(", "))
    };

    let app = ChatApp::new(sessions);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([820.0, 600.0])
            .with_min_inner_size([480.0, 360.0])
            .with_title(title.clone()),
        ..Default::default()
    };
    eframe::run_native(
        &title,
        options,
        Box::new(|cc| {
            install_fonts(&cc.egui_ctx);
            // Wrap text by default so long lines (esp. space-less scripts like
            // Thai, which egui can't break on whitespace) never run past the
            // panel edge.
            cc.egui_ctx
                .style_mut(|s| s.wrap_mode = Some(egui::TextWrapMode::Wrap));
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| format!("eframe: {e}"))?;
    Ok(())
}

/// Map a design token's RGB half to an egui colour. bwoc-chat is a pixel UI,
/// so unlike the TUIs (which use the token's ANSI half to respect terminal
/// themes) it renders the exact token shade.
fn rgb(t: design::ColorToken) -> egui::Color32 {
    let (r, g, b) = t.rgb;
    egui::Color32::from_rgb(r, g, b)
}

fn palette(i: usize) -> egui::Color32 {
    let (r, g, b) = PALETTE[i % PALETTE.len()];
    egui::Color32::from_rgb(r, g, b)
}

/// An agent's short name — the registry id without the `agent-` prefix, used for
/// `@mention` matching and compact labels.
fn short(id: &str) -> &str {
    id.strip_prefix("agent-").unwrap_or(id)
}

/// Install a Thai/Unicode-capable fallback font so non-Latin text (e.g. Thai)
/// renders instead of tofu boxes — egui's built-in fonts are Latin-centric. We
/// append a broad system font as a per-glyph fallback (egui falls through the
/// family list glyph-by-glyph). Best-effort: if no candidate is found, egui
/// keeps its default and Latin still renders.
fn install_fonts(ctx: &egui::Context) {
    const CANDIDATES: &[&str] = &[
        "/System/Library/Fonts/Supplemental/Ayuthaya.ttf", // macOS — Thai + Latin, small
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", // macOS — very broad
        "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf", // Linux
        "/usr/share/fonts/noto/NotoSansThai-Regular.ttf",
    ];
    let Some(bytes) = CANDIDATES.iter().find_map(|p| std::fs::read(p).ok()) else {
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "unicode_fallback".to_owned(),
        egui::FontData::from_owned(bytes),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("unicode_fallback".to_owned());
    }
    ctx.set_fonts(fonts);
}

// ---------------------------------------------------------------------------
// Agent resolution
// ---------------------------------------------------------------------------

/// A resolved agent ready to spawn — model/endpoint already settled.
struct AgentConfig {
    agent_id: String,
    agent_path: PathBuf,
    backend: String,
    model: String,
    endpoint: String,
    /// `--claude-code`: drive the `claude` CLI in headless stream-json mode
    /// (subscription auth, no API key) instead of `bwoc-harness`, translating
    /// Claude Code's event stream into `chat_proto`.
    claude_code: bool,
}

/// Stamp the `--claude-code` flag onto every resolved config (the flag is
/// session-wide, applied after agent resolution).
fn with_claude(mut cfgs: Vec<AgentConfig>, on: bool) -> Vec<AgentConfig> {
    for c in &mut cfgs {
        c.claude_code = on;
    }
    cfgs
}

/// Resolve CLI args into one or more agents to spawn.
///
/// - 0 names, no dir flag → the personal assistant (`~/.bwoc/personal`).
/// - `--here` / `.` / `--path <dir>` → that directory directly (single only).
/// - 1 name → that workspace agent.
/// - N names → team chat: every named workspace agent in one window.
fn resolve(args: Vec<String>) -> Result<Vec<AgentConfig>, String> {
    let mut names: Vec<String> = Vec::new();
    let mut workspace: Option<PathBuf> = None;
    let mut model_override: Option<String> = None;
    let mut endpoint_override: Option<String> = None;
    let mut path_override: Option<PathBuf> = None;
    let mut here = false;
    let mut claude_code = false;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--workspace" => workspace = it.next().map(PathBuf::from),
            "--model" => model_override = it.next(),
            "--endpoint" => endpoint_override = it.next(),
            "--path" => path_override = it.next().map(PathBuf::from),
            "--here" | "." => here = true,
            "--claude-code" => claude_code = true,
            s if s.starts_with("--") => return Err(format!("unknown flag `{s}`")),
            s => names.push(s.to_string()),
        }
    }

    // ── Single-agent directory modes (no workspace, no init) ─────────────────
    if let Some(dir) = path_override {
        if !names.is_empty() {
            return Err("--path takes no agent name (single-agent mode)".into());
        }
        return Ok(with_claude(
            vec![AgentConfig::from_dir(
                dir,
                model_override,
                endpoint_override,
            )?],
            claude_code,
        ));
    }
    if here {
        if !names.is_empty() {
            return Err("--here / . takes no agent name (single-agent mode)".into());
        }
        let cwd = std::env::current_dir().map_err(|e| format!("cwd: {e}"))?;
        return Ok(with_claude(
            vec![AgentConfig::from_dir(
                cwd,
                model_override,
                endpoint_override,
            )?],
            claude_code,
        ));
    }
    // No agent named → the global **personal assistant** at `~/.bwoc/personal`.
    if names.is_empty() {
        let dir = ensure_personal_agent()?;
        return Ok(with_claude(
            vec![AgentConfig::from_dir(
                dir,
                model_override,
                endpoint_override,
            )?],
            claude_code,
        ));
    }

    // ── Workspace mode (one or more named agents from the registry) ──────────
    let workspace = workspace
        .or_else(|| std::env::var_os("BWOC_WORKSPACE").map(PathBuf::from))
        .or_else(resolve_workspace)
        .ok_or(
            "no workspace found — pass --workspace / set BWOC_WORKSPACE / run from a \
             workspace, or run with no agent for the personal assistant (or --here for \
             the current directory)",
        )?;
    let registry =
        AgentsRegistry::load(&workspace).map_err(|e| format!("failed to read agents.toml: {e}"))?;

    let mut configs = Vec::with_capacity(names.len());
    for name in &names {
        configs.push(resolve_workspace_agent(
            name,
            &workspace,
            &registry,
            model_override.clone(),
            endpoint_override.clone(),
        )?);
    }
    Ok(with_claude(configs, claude_code))
}

/// Resolve one named agent from the workspace registry into an [`AgentConfig`].
fn resolve_workspace_agent(
    name: &str,
    workspace: &std::path::Path,
    registry: &AgentsRegistry,
    model_override: Option<String>,
    endpoint_override: Option<String>,
) -> Result<AgentConfig, String> {
    let lookup = if name.starts_with("agent-") {
        name.to_string()
    } else {
        format!("agent-{name}")
    };
    let entry = registry
        .agents
        .iter()
        .find(|a| a.id == lookup)
        .ok_or_else(|| format!("no agent named '{name}' in {}", workspace.display()))?;

    // Backends the harness can render as a chat_proto stream: the OpenAI-compat
    // HTTP path and the native Anthropic provider. Other vendor CLIs (codex /
    // kimi / agy) have no harness stream — point the user at `bwoc spawn`.
    if !matches!(
        entry.backend.as_str(),
        "ollama" | "openai-compatible" | "claude" | "anthropic"
    ) {
        return Err(format!(
            "agent '{}' uses the '{}' backend — bwoc-chat renders the harness chat stream for \
             ollama / openai-compatible / claude. Use `bwoc spawn` for other vendor CLIs.",
            entry.id, entry.backend
        ));
    }

    let agent_path = workspace.join(&entry.path);
    let manifest = Manifest::load_from_path(&agent_path.join("config.manifest.json")).ok();
    let model = model_override
        .or_else(|| manifest.as_ref().map(|m| m.primary_model.clone()))
        .unwrap_or_else(|| "gemma4:latest".to_string());
    let endpoint = endpoint_override
        .or_else(|| manifest.as_ref().and_then(|m| m.base_url.clone()))
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());

    Ok(AgentConfig {
        agent_id: entry.id.clone(),
        agent_path,
        backend: entry.backend.clone(),
        model,
        endpoint,
        claude_code: false,
    })
}

impl AgentConfig {
    /// Build a config for a single agent **directory** directly — no workspace,
    /// no registry (`--path`, `--here`, or the personal assistant). The backend
    /// is assumed harness-compatible (ollama / openai-compatible); model +
    /// endpoint come from a `config.manifest.json` in the dir if present, else
    /// the defaults.
    fn from_dir(
        dir: PathBuf,
        model_override: Option<String>,
        endpoint_override: Option<String>,
    ) -> Result<Self, String> {
        if !dir.is_dir() {
            return Err(format!("not a directory: {}", dir.display()));
        }
        let agent_id = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("agent")
            .to_string();
        let manifest = Manifest::load_from_path(&dir.join("config.manifest.json")).ok();
        let model = model_override
            .or_else(|| manifest.as_ref().map(|m| m.primary_model.clone()))
            .unwrap_or_else(|| "gemma4:latest".to_string());
        let endpoint = endpoint_override
            .or_else(|| manifest.as_ref().and_then(|m| m.base_url.clone()))
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
        Ok(AgentConfig {
            agent_id,
            agent_path: dir,
            backend: "ollama".to_string(),
            model,
            endpoint,
            claude_code: false,
        })
    }
}

/// System prompt for the auto-created personal assistant.
const PERSONAL_SYSTEM_PROMPT: &str = "\
You are the user's personal assistant. Be concise, friendly, and genuinely
helpful. You can read files and run small tasks when asked; always ask before
doing anything destructive or irreversible.";

/// Default tool policy for the personal assistant: read freely, ask before
/// writing or running (the chat window shows an Allow/Deny prompt).
const PERSONAL_POLICY: &str = "\
default_mode = \"ask\"

[tools]
read_file = \"allow\"
list_dir = \"allow\"
grep = \"allow\"
";

/// Resolve (and on first use, create) the global personal assistant at
/// `~/.bwoc/personal` — a standalone agent that needs no workspace. Seeds a
/// friendly `AGENTS.md` system prompt and a sensible `.bwoc/harness-policy.toml`.
fn ensure_personal_agent() -> Result<PathBuf, String> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or("cannot find your home directory ($HOME unset)")?;
    let dir = home.join(".bwoc").join("personal");
    if !dir.join("AGENTS.md").is_file() {
        std::fs::create_dir_all(dir.join(".bwoc"))
            .map_err(|e| format!("create {}: {e}", dir.display()))?;
        std::fs::write(dir.join("AGENTS.md"), PERSONAL_SYSTEM_PROMPT)
            .map_err(|e| format!("seed AGENTS.md: {e}"))?;
        std::fs::write(
            dir.join(".bwoc").join("harness-policy.toml"),
            PERSONAL_POLICY,
        )
        .map_err(|e| format!("seed harness-policy.toml: {e}"))?;
    }
    Ok(dir)
}

/// Ancestor walk for `.bwoc/workspace.toml` from the current directory.
fn resolve_workspace() -> Option<PathBuf> {
    let mut cur = std::env::current_dir().ok()?;
    loop {
        if cur.join(".bwoc/workspace.toml").is_file() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-agent harness session
// ---------------------------------------------------------------------------

struct Pending {
    id: String,
    tool: String,
    detail: String,
}

/// One row in the activity panel: a compact one-line `summary` shown as the
/// (optionally collapsible) header, plus the full untruncated `body` rendered
/// as markdown when expanded. `body` is empty when the summary already shows
/// everything, in which case the row renders flat with no collapse arrow.
struct ActivityItem {
    summary: String,
    body: String,
}

impl ActivityItem {
    /// Build from a `summary` line and the `full` untruncated content. A body
    /// is kept only when `full` adds something the summary can't show on one
    /// line (it's multi-line, or the summary truncated it).
    fn new(summary: String, full: String) -> Self {
        let collapses = full.contains('\n') || full.chars().count() > 80;
        let body = if collapses { full } else { String::new() };
        ActivityItem { summary, body }
    }

    /// A row that is only ever a one-liner (e.g. a permission decision).
    fn line(summary: String) -> Self {
        ActivityItem {
            summary,
            body: String::new(),
        }
    }
}

/// One agent = one `bwoc-harness --chat` subprocess plus the live UI state the
/// window renders for it. Its [`Drop`] reaps the child so no harness is orphaned.
struct AgentSession {
    id: String,
    color: egui::Color32,
    /// Status-bar text (`model · backend · ready|busy|…`), updated from events.
    status: String,
    /// Tool names from this agent's `Ready` event (for `/tools`).
    tools: Vec<String>,
    /// Per-agent tool-call / result log shown in the activity panel.
    activity: Vec<ActivityItem>,
    busy: bool,
    alive: bool,
    pending: Option<Pending>,
    /// Index into [`ChatApp::convo`] of this agent's in-progress streamed reply,
    /// so concurrent agents append to their own message rather than the last one.
    cur: Option<usize>,
    stdin: ChildStdin,
    rx: Receiver<ChatEvent>,
    child: Child,
    /// True when the subprocess is the `claude` CLI (stream-json) rather than
    /// `bwoc-harness` — changes how [`Self::write`] frames stdin input.
    claude: bool,
}

impl AgentSession {
    fn spawn(cfg: &AgentConfig, color: egui::Color32) -> Result<Self, String> {
        let harness = bwoc_core::exec::binary_or_name("bwoc-harness");
        let mut child = Command::new(&harness)
            .arg("--chat")
            // Lift the workdir sandbox: a desktop chat agent reaches real files
            // anywhere on the machine. The safety gate is the per-action ask
            // prompt (Allow/Deny), not path confinement.
            .arg("--unrestricted")
            .arg("--workdir")
            .arg(&cfg.agent_path)
            // Select the provider: claude → Anthropic Messages API, otherwise
            // the OpenAI-compatible HTTP path. The harness substitutes the
            // Anthropic endpoint for a claude agent that left the default.
            .arg("--backend")
            .arg(&cfg.backend)
            .arg("--model")
            .arg(&cfg.model)
            .arg("--endpoint")
            .arg(&cfg.endpoint)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                format!(
                    "failed to spawn bwoc-harness ({harness:?}) for {}: {e}",
                    cfg.agent_id
                )
            })?;

        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");

        // Reader thread: child stdout lines → ChatEvent → channel.
        let (tx, rx) = mpsc::channel::<ChatEvent>();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(ev) = serde_json::from_str::<ChatEvent>(line) {
                    if tx.send(ev).is_err() {
                        break;
                    }
                }
            }
        });

        Ok(AgentSession {
            status: format!("{} · {} · connecting…", cfg.model, cfg.backend),
            id: cfg.agent_id.clone(),
            color,
            tools: Vec::new(),
            activity: Vec::new(),
            busy: false,
            alive: true,
            pending: None,
            cur: None,
            stdin,
            rx,
            child,
            claude: false,
        })
    }

    /// Spawn the `claude` CLI in headless stream-json mode and translate its
    /// event stream into `chat_proto` — Claude Code in a native window, using
    /// the logged-in **subscription** (no `ANTHROPIC_API_KEY`). The reader
    /// thread maps Claude's `system/init`, `stream_event`, `assistant`,
    /// `user`, and `result` events into [`ChatEvent`]s the UI already renders.
    fn spawn_claude_code(cfg: &AgentConfig, color: egui::Color32) -> Result<Self, String> {
        let claude = resolve_claude();
        let mut child = Command::new(&claude)
            .arg("-p")
            .arg("--input-format")
            .arg("stream-json")
            .arg("--output-format")
            .arg("stream-json")
            // Live token deltas (content_block_delta) so the window streams.
            .arg("--include-partial-messages")
            // stream-json output requires --verbose.
            .arg("--verbose")
            // Headless can't prompt interactively; auto-approve edits + common
            // fs commands. (A richer Allow/Deny bridge is future work.)
            .arg("--permission-mode")
            .arg("acceptEdits")
            .arg("--model")
            .arg(claude_model_alias(&cfg.model))
            // Run in the agent dir so Claude Code loads that agent's CLAUDE.md
            // (→ AGENTS.md) persona automatically.
            .current_dir(&cfg.agent_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                format!(
                    "failed to spawn claude ({claude}) for {}: {e}",
                    cfg.agent_id
                )
            })?;

        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");

        // Reader thread: claude stream-json lines → ChatEvent → channel.
        let (tx, rx) = mpsc::channel::<ChatEvent>();
        let agent_id = cfg.agent_id.clone();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                    continue;
                };
                for ev in translate_claude_event(&v, &agent_id) {
                    if tx.send(ev).is_err() {
                        return;
                    }
                }
            }
            let _ = tx.send(ChatEvent::Bye);
        });

        Ok(AgentSession {
            status: format!("{} · claude-code · connecting…", cfg.model),
            id: cfg.agent_id.clone(),
            color,
            tools: Vec::new(),
            activity: Vec::new(),
            busy: false,
            alive: true,
            pending: None,
            cur: None,
            stdin,
            rx,
            child,
            claude: true,
        })
    }

    fn write(&mut self, input: &ChatInput) {
        if self.claude {
            // Claude CLI stream-json input: only user turns are framed; the
            // other control inputs have no claude-code equivalent in this MVP.
            if let ChatInput::User { text } = input {
                let line = serde_json::json!({
                    "type": "user",
                    "message": { "role": "user", "content": text },
                });
                let _ = writeln!(self.stdin, "{line}");
                let _ = self.stdin.flush();
            }
            return;
        }
        if let Ok(line) = input.to_line() {
            let _ = writeln!(self.stdin, "{line}");
            let _ = self.stdin.flush();
        }
    }
}

/// Resolve the `claude` CLI binary — GUI apps don't inherit a shell `PATH`, so
/// probe the common install locations before falling back to bare `claude`.
fn resolve_claude() -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.local/bin/claude"),
        "/opt/homebrew/bin/claude".to_string(),
        "/usr/local/bin/claude".to_string(),
    ];
    candidates
        .into_iter()
        .find(|p| std::path::Path::new(p).exists())
        .unwrap_or_else(|| "claude".to_string())
}

/// Map a manifest model id (e.g. `claude-sonnet-4-6`) to a `claude --model`
/// alias the CLI reliably accepts. Defaults to `sonnet`.
fn claude_model_alias(model: &str) -> &'static str {
    let m = model.to_lowercase();
    if m.contains("haiku") {
        "haiku"
    } else if m.contains("opus") {
        "opus"
    } else {
        "sonnet"
    }
}

/// Translate one Claude Code stream-json event into zero or more [`ChatEvent`]s.
fn translate_claude_event(v: &serde_json::Value, agent_id: &str) -> Vec<ChatEvent> {
    match v["type"].as_str() {
        Some("system") if v["subtype"] == "init" => {
            let model = v["model"].as_str().unwrap_or("claude").to_string();
            let tools = v["tools"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|t| t.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            vec![ChatEvent::Ready {
                agent: agent_id.to_string(),
                model,
                backend: "claude-code".to_string(),
                tools,
            }]
        }
        Some("stream_event") => {
            let ev = &v["event"];
            if ev["type"] == "content_block_delta" && ev["delta"]["type"] == "text_delta" {
                let text = ev["delta"]["text"].as_str().unwrap_or_default().to_string();
                if !text.is_empty() {
                    return vec![ChatEvent::Token { text }];
                }
            }
            Vec::new()
        }
        Some("assistant") => {
            // Surface tool_use blocks as ToolCall (text is streamed separately
            // and finalised by the `result` event).
            let mut out = Vec::new();
            if let Some(blocks) = v["message"]["content"].as_array() {
                for b in blocks {
                    if b["type"] == "tool_use" {
                        out.push(ChatEvent::ToolCall {
                            id: b["id"].as_str().unwrap_or_default().to_string(),
                            name: b["name"].as_str().unwrap_or_default().to_string(),
                            args: b["input"].to_string(),
                        });
                    }
                }
            }
            out
        }
        Some("user") => {
            // Claude's own tool results arrive as user tool_result blocks.
            let mut out = Vec::new();
            if let Some(blocks) = v["message"]["content"].as_array() {
                for b in blocks {
                    if b["type"] == "tool_result" {
                        let output = b["content"]
                            .as_str()
                            .map(String::from)
                            .unwrap_or_else(|| b["content"].to_string());
                        out.push(ChatEvent::ToolResult {
                            id: b["tool_use_id"].as_str().unwrap_or_default().to_string(),
                            name: "tool".to_string(),
                            ok: !b["is_error"].as_bool().unwrap_or(false),
                            output,
                        });
                    }
                }
            }
            out
        }
        Some("result") => {
            let u = &v["usage"];
            let prompt = u["input_tokens"].as_u64().unwrap_or(0);
            let completion = u["output_tokens"].as_u64().unwrap_or(0);
            let mut out = Vec::new();
            if v["is_error"].as_bool().unwrap_or(false) {
                out.push(ChatEvent::Error {
                    message: v["result"].as_str().unwrap_or("claude error").to_string(),
                });
            } else if let Some(text) = v["result"].as_str() {
                // Final assistant text — overwrites the streamed buffer (same content).
                out.push(ChatEvent::Message {
                    text: text.to_string(),
                });
            }
            out.push(ChatEvent::TurnEnd {
                prompt_tokens: prompt,
                completion_tokens: completion,
            });
            out
        }
        _ => Vec::new(),
    }
}

impl Drop for AgentSession {
    fn drop(&mut self) {
        // Best-effort graceful shutdown, then reap so the harness never orphans.
        self.write(&ChatInput::Quit);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Who {
    User,
    Agent,
    System,
}

/// One line in the shared transcript. `agent` indexes [`ChatApp::sessions`] for
/// `Who::Agent` (its tag + colour); ignored for user/system lines.
struct Msg {
    who: Who,
    agent: usize,
    text: String,
}

struct ChatApp {
    sessions: Vec<AgentSession>,
    convo: Vec<Msg>,
    input: String,
    /// Current permission mode (`default` / `accept_edits` / `bypass`), reflected
    /// from the harness's `ModeChanged` ack. Shown in the status bar.
    mode: String,
    /// Directory whose entries the `@` completion popup lists (the launch cwd).
    cwd: PathBuf,
    /// Render cache for the CommonMark (markdown) viewer.
    md_cache: egui_commonmark::CommonMarkCache,
}

impl ChatApp {
    fn new(sessions: Vec<AgentSession>) -> Self {
        let hint = if sessions.len() == 1 {
            "Connected. Type a message and press Enter — or /help for commands.".to_string()
        } else {
            let names: Vec<String> = sessions
                .iter()
                .map(|s| format!("@{}", short(&s.id)))
                .collect();
            format!(
                "Team connected ({} agents). A message goes to all; prefix {} to address one. /help for commands.",
                sessions.len(),
                names.join(" / ")
            )
        };
        Self {
            convo: vec![Msg {
                who: Who::System,
                agent: 0,
                text: hint,
            }],
            input: String::new(),
            mode: "default".to_string(),
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            md_cache: egui_commonmark::CommonMarkCache::default(),
            sessions,
        }
    }

    fn team(&self) -> bool {
        self.sessions.len() > 1
    }

    fn any_alive(&self) -> bool {
        self.sessions.iter().any(|s| s.alive)
    }

    fn any_busy(&self) -> bool {
        self.sessions.iter().any(|s| s.busy)
    }

    /// Apply one event from agent `idx`.
    fn apply(&mut self, idx: usize, ev: ChatEvent) {
        match ev {
            ChatEvent::Ready {
                model,
                backend,
                tools,
                ..
            } => {
                let s = &mut self.sessions[idx];
                s.status = format!("{model} · {backend} · ready");
                s.tools = tools;
            }
            ChatEvent::Restored { role, text } => {
                let who = if role == "user" {
                    Who::User
                } else {
                    Who::Agent
                };
                self.convo.push(Msg {
                    who,
                    agent: idx,
                    text,
                });
            }
            ChatEvent::Token { text } => {
                // Append onto this agent's in-progress reply, or open a new one.
                match self.sessions[idx].cur {
                    Some(ci) => self.convo[ci].text.push_str(&text),
                    None => {
                        self.convo.push(Msg {
                            who: Who::Agent,
                            agent: idx,
                            text,
                        });
                        self.sessions[idx].cur = Some(self.convo.len() - 1);
                    }
                }
            }
            ChatEvent::Message { text } => {
                // Final assistant text — overwrite the streamed buffer if any.
                match self.sessions[idx].cur {
                    Some(ci) => self.convo[ci].text = text,
                    None => {
                        self.convo.push(Msg {
                            who: Who::Agent,
                            agent: idx,
                            text,
                        });
                        self.sessions[idx].cur = Some(self.convo.len() - 1);
                    }
                }
            }
            ChatEvent::ToolCall { name, args, .. } => {
                let summary = format!("» {name} {}", truncate(&args, 80));
                self.sessions[idx]
                    .activity
                    .push(ActivityItem::new(summary, args));
            }
            ChatEvent::ToolResult {
                name, ok, output, ..
            } => {
                let mark = if ok { "[ok]" } else { "[err]" };
                let summary = format!("{mark} {name}: {}", truncate(&output, 80));
                self.sessions[idx]
                    .activity
                    .push(ActivityItem::new(summary, output));
            }
            ChatEvent::PermissionRequest { id, tool, detail } => {
                self.sessions[idx].pending = Some(Pending { id, tool, detail });
            }
            ChatEvent::ModeChanged { mode } => {
                // All sessions share the broadcast mode; reflect it once.
                if self.mode != mode {
                    self.mode = mode.clone();
                    self.convo.push(Msg {
                        who: Who::System,
                        agent: 0,
                        text: format!("permission mode → {mode}"),
                    });
                }
            }
            ChatEvent::Compacted { removed } => {
                self.convo.push(Msg {
                    who: Who::System,
                    agent: idx,
                    text: format!("context compacted — folded {removed} earlier messages"),
                });
            }
            ChatEvent::TurnEnd {
                prompt_tokens,
                completion_tokens,
            } => {
                let s = &mut self.sessions[idx];
                s.busy = false;
                s.cur = None;
                s.status = format!("ready · tokens {prompt_tokens} in / {completion_tokens} out");
            }
            ChatEvent::Error { message } => {
                let s = &mut self.sessions[idx];
                s.busy = false;
                s.cur = None;
                let id = s.id.clone();
                self.convo.push(Msg {
                    who: Who::System,
                    agent: idx,
                    text: format!("{id} error: {message}"),
                });
            }
            ChatEvent::Bye => {
                let s = &mut self.sessions[idx];
                s.alive = false;
                let id = s.id.clone();
                self.convo.push(Msg {
                    who: Who::System,
                    agent: idx,
                    text: format!("{id} session ended."),
                });
            }
        }
    }

    /// Route the typed text. A bare message broadcasts to every alive agent; a
    /// leading `@name` (matched against an agent's short name) targets just that
    /// one. The raw text (with any `@name`) is shown in the transcript so the
    /// user sees who they addressed; the `@name` is stripped before sending.
    fn send_user(&mut self) {
        let raw = self.input.trim().to_string();
        if raw.is_empty() || !self.any_alive() {
            return;
        }
        self.input.clear();

        let (target, body) = self.parse_target(&raw);
        if body.is_empty() {
            self.convo.push(Msg {
                who: Who::System,
                agent: 0,
                text: "(empty message — nothing sent)".to_string(),
            });
            return;
        }
        self.convo.push(Msg {
            who: Who::User,
            agent: 0,
            text: raw,
        });
        for i in 0..self.sessions.len() {
            if !self.sessions[i].alive {
                continue;
            }
            if let Some(t) = target {
                if t != i {
                    continue;
                }
            }
            self.sessions[i].busy = true;
            self.sessions[i].cur = None;
            self.sessions[i].write(&ChatInput::User { text: body.clone() });
        }
    }

    /// Completion suggestions for an `@`-prefixed input: team agents first, then
    /// files/directories in the launch cwd. Each entry is `(label, insert)` where
    /// `insert` is the full input string to substitute when the row is clicked.
    /// Capped so the popup stays small.
    fn at_suggestions(&self, frag: &str) -> Vec<(String, String)> {
        let lower = frag.to_lowercase();
        let mut out: Vec<(String, String)> = Vec::new();

        // Agents (only meaningful in a team) — keep the `@` for routing.
        if self.sessions.len() > 1 {
            for s in &self.sessions {
                let name = short(&s.id);
                if name.to_lowercase().starts_with(&lower) {
                    out.push((format!("@{name}  (agent)"), format!("@{name} ")));
                }
            }
        }

        // Files / directories in the cwd — insert the bare name (a path the
        // agent can read); directories get a trailing `/`.
        if let Ok(rd) = std::fs::read_dir(&self.cwd) {
            let mut entries: Vec<(String, String)> = Vec::new();
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with('.') || !name.to_lowercase().starts_with(&lower) {
                    continue;
                }
                let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                let display = if is_dir {
                    format!("{name}/")
                } else {
                    name.clone()
                };
                let insert = if is_dir {
                    format!("{name}/")
                } else {
                    format!("{name} ")
                };
                entries.push((format!("{display}  (file)"), insert));
            }
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            out.extend(entries);
        }

        out.truncate(12);
        out
    }

    /// Split a leading `@name` off the message. Returns `(Some(idx), rest)` when
    /// `name` matches an agent's short name, else `(None, whole)` (broadcast).
    fn parse_target(&self, raw: &str) -> (Option<usize>, String) {
        if let Some(rest) = raw.strip_prefix('@') {
            let mut parts = rest.splitn(2, char::is_whitespace);
            let name = parts.next().unwrap_or("");
            let body = parts.next().unwrap_or("").trim();
            if let Some(idx) = self
                .sessions
                .iter()
                .position(|s| short(&s.id).eq_ignore_ascii_case(name))
            {
                return (Some(idx), body.to_string());
            }
        }
        (None, raw.to_string())
    }

    /// Handle a client-side `/command` (intercepted before it reaches a harness).
    fn run_command(&mut self, line: &str, ctx: &egui::Context) {
        let mut parts = line.trim().splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("").to_lowercase();
        let arg = parts.next().unwrap_or("").trim().to_string();
        let sys = |s: &mut Self, text: String| {
            s.convo.push(Msg {
                who: Who::System,
                agent: 0,
                text,
            })
        };
        match name.as_str() {
            "" | "help" | "?" => {
                let extra = if self.team() {
                    " · @name to address one agent, bare message broadcasts to all"
                } else {
                    ""
                };
                sys(
                    self,
                    format!("commands: /help · /tools · /mode · /clear · /forget · /quit{extra}"),
                );
            }
            "tools" => {
                for i in 0..self.sessions.len() {
                    let s = &self.sessions[i];
                    let msg = if s.tools.is_empty() {
                        format!("{}: no tools reported.", s.id)
                    } else {
                        format!("{} ({} tools): {}", s.id, s.tools.len(), s.tools.join(", "))
                    };
                    sys(self, msg);
                }
            }
            "mode" => {
                let want = arg.trim();
                if want.is_empty() {
                    sys(
                        self,
                        format!(
                            "permission mode: {} — set with /mode default | accept-edits | bypass | plan",
                            self.mode
                        ),
                    );
                } else {
                    // Broadcast the switch to every harness; each acks ModeChanged.
                    let normalized = want.replace('-', "_");
                    for s in &mut self.sessions {
                        s.write(&ChatInput::SetMode {
                            mode: normalized.clone(),
                        });
                    }
                }
            }
            "clear" => {
                self.convo.clear();
                for s in &mut self.sessions {
                    s.activity.clear();
                    s.cur = None;
                }
                sys(self, "conversation cleared.".to_string());
            }
            "forget" => {
                // Tell every harness to drop its memory + on-disk session.
                for s in &mut self.sessions {
                    s.write(&ChatInput::Forget);
                    s.activity.clear();
                    s.cur = None;
                }
                self.convo.clear();
                sys(
                    self,
                    "memory cleared — every agent starts fresh.".to_string(),
                );
            }
            "quit" | "exit" => {
                for s in &mut self.sessions {
                    s.write(&ChatInput::Quit);
                    s.alive = false;
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            other => sys(self, format!("unknown command `/{other}` — try /help")),
        }
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain every agent's channel since last frame.
        for idx in 0..self.sessions.len() {
            let events: Vec<ChatEvent> = self.sessions[idx].rx.try_iter().collect();
            for ev in events {
                self.apply(idx, ev);
            }
        }

        let mut do_send = false;
        // Permission decisions queued this frame: (session idx, allow).
        let mut perms: Vec<(usize, bool)> = Vec::new();
        let mut run_cmd: Option<&'static str> = None;
        // A clicked `@`-completion row's replacement for the input buffer.
        let mut pick: Option<String> = None;

        // ── Status bar: one chip per agent ───────────────────────────────────
        egui::TopBottomPanel::top("status").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong("bwoc-chat");
                // Permission mode badge — amber when relaxed past the safe default.
                let mode_color = if self.mode == "default" {
                    rgb(design::color::MUTED)
                } else {
                    rgb(design::color::WARNING)
                };
                ui.label(egui::RichText::new(format!("[{}]", self.mode)).color(mode_color));
                for s in &self.sessions {
                    ui.separator();
                    ui.label(egui::RichText::new(short(&s.id)).color(s.color).strong());
                    ui.weak(&s.status);
                    if s.busy {
                        ui.spinner();
                    }
                }
            });
        });

        // The markdown render cache is borrowed out of `self` so both the
        // activity panel and the transcript below can render CommonMark while
        // still iterating `&self.sessions` / `&self.convo`.
        let mut md_cache = std::mem::take(&mut self.md_cache);

        // ── Activity panel: tool calls/results, grouped by agent ─────────────
        egui::SidePanel::right("activity")
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.heading("activity");
                ui.separator();
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        let team = self.sessions.len() > 1;
                        let mut any = false;
                        // Unique salt per row so identical summaries don't share
                        // collapsing state.
                        let mut row = 0usize;
                        for s in &self.sessions {
                            if s.activity.is_empty() {
                                continue;
                            }
                            any = true;
                            if team {
                                ui.label(egui::RichText::new(short(&s.id)).color(s.color).strong());
                            }
                            for item in &s.activity {
                                row += 1;
                                if item.body.is_empty() {
                                    // Nothing to expand — render the line flat.
                                    ui.label(&item.summary);
                                } else {
                                    // Collapsed by default to keep the panel
                                    // compact; the full body (which may carry
                                    // code fences, tables, links) renders as
                                    // CommonMark when expanded.
                                    egui::CollapsingHeader::new(&item.summary)
                                        .id_salt(row)
                                        .show(ui, |ui| {
                                            egui_commonmark::CommonMarkViewer::new().show(
                                                ui,
                                                &mut md_cache,
                                                &item.body,
                                            );
                                        });
                                }
                            }
                            if team {
                                ui.add_space(4.0);
                            }
                        }
                        if !any {
                            ui.weak("(no tool activity yet)");
                        }
                    });
            });

        // ── Bottom: pending permissions + input ──────────────────────────────
        egui::TopBottomPanel::bottom("input").show(ctx, |ui| {
            ui.add_space(4.0);
            let team = self.sessions.len() > 1;
            let mut any_pending = false;
            for (i, s) in self.sessions.iter().enumerate() {
                if let Some(p) = &s.pending {
                    any_pending = true;
                    ui.horizontal_wrapped(|ui| {
                        let who = if team {
                            format!("permission [{}]: {} ", short(&s.id), p.tool)
                        } else {
                            format!("permission: {} ", p.tool)
                        };
                        ui.label(
                            egui::RichText::new(who)
                                .color(rgb(design::color::WARNING))
                                .strong(),
                        );
                        ui.label(truncate(&p.detail, 120));
                        if ui.button("Allow").clicked() {
                            perms.push((i, true));
                        }
                        if ui.button("Deny").clicked() {
                            perms.push((i, false));
                        }
                    });
                }
            }
            if any_pending {
                ui.separator();
            }

            // Slash-command list when the input starts with `/`.
            if self.input.starts_with('/') {
                let typed = &self.input[1..];
                let matches: Vec<(&'static str, &'static str)> = COMMANDS
                    .iter()
                    .copied()
                    .filter(|(name, _)| name[1..].starts_with(typed))
                    .collect();
                if matches.is_empty() {
                    ui.weak("no matching command — /help");
                }
                for (name, desc) in matches {
                    let label = egui::RichText::new(format!("{name}  —  {desc}"));
                    if ui
                        .add(egui::Button::new(label).frame(false))
                        .on_hover_text("click to run")
                        .clicked()
                    {
                        run_cmd = Some(name);
                    }
                }
                ui.separator();
            }

            // `@` completion: agents (team) + files/directories in the cwd.
            if self.input.starts_with('@') {
                let frag = self.input[1..].to_string();
                let suggestions = self.at_suggestions(&frag);
                if suggestions.is_empty() {
                    ui.weak("no matching agent or file");
                }
                for (label, insert) in suggestions {
                    if ui
                        .add(egui::Button::new(egui::RichText::new(label)).frame(false))
                        .on_hover_text("click to insert")
                        .clicked()
                    {
                        pick = Some(insert);
                    }
                }
                ui.separator();
            }

            let alive = self.any_alive();
            ui.horizontal(|ui| {
                let hint = if !alive {
                    "(all sessions ended)"
                } else if team {
                    "message all…  (@name to target · /help)"
                } else {
                    "message…  (/help for commands)"
                };
                let resp = ui.add_enabled(
                    alive,
                    egui::TextEdit::singleline(&mut self.input)
                        .desired_width(f32::INFINITY)
                        .hint_text(hint),
                );
                let entered = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if entered {
                    do_send = true;
                    resp.request_focus();
                }
                if ui.add_enabled(alive, egui::Button::new("Send")).clicked() {
                    do_send = true;
                }
            });
            ui.add_space(4.0);
        });

        // ── Central: shared transcript ───────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for msg in &self.convo {
                        let (tag, color): (&str, egui::Color32) = match msg.who {
                            Who::User => ("you", rgb(design::color::USER)),
                            Who::Agent => {
                                let s = &self.sessions[msg.agent];
                                (short(&s.id), s.color)
                            }
                            Who::System => ("·", rgb(design::color::SYSTEM)),
                        };
                        match msg.who {
                            // Assistant replies render as markdown; the tag goes
                            // on its own line so fenced blocks get full width.
                            Who::Agent => {
                                ui.label(
                                    egui::RichText::new(format!("{tag}:")).color(color).strong(),
                                );
                                egui_commonmark::CommonMarkViewer::new().show(
                                    ui,
                                    &mut md_cache,
                                    &msg.text,
                                );
                            }
                            _ => {
                                let body = egui::TextStyle::Body.resolve(ui.style());
                                let line_h = body.size * design::space::LINE_HEIGHT_FACTOR;
                                let text_color = ui.visuals().text_color();
                                let job = build_message_job(
                                    tag,
                                    &msg.text,
                                    color,
                                    text_color,
                                    body,
                                    line_h,
                                    ui.available_width(),
                                );
                                ui.label(job);
                            }
                        }
                        ui.add_space(design::space::MESSAGE_GAP);
                    }
                });
        });
        self.md_cache = md_cache;

        // ── Dispatch queued actions ──────────────────────────────────────────
        if let Some(insert) = pick {
            // A clicked `@`-completion row replaces the input buffer.
            self.input = insert;
        }
        if let Some(name) = run_cmd {
            self.input.clear();
            self.run_command(name.trim_start_matches('/'), ctx);
        } else if do_send {
            // A leading `/` is a client-side command, not a message to an agent.
            let text = self.input.trim().to_string();
            if let Some(cmd) = text.strip_prefix('/') {
                self.input.clear();
                self.run_command(cmd, ctx);
            } else {
                self.send_user();
            }
        }
        for (i, allow) in perms {
            if let Some(p) = self.sessions[i].pending.take() {
                self.sessions[i].activity.push(ActivityItem::line(format!(
                    "{} {}",
                    if allow { "[allowed]" } else { "[denied]" },
                    p.tool
                )));
                self.sessions[i].write(&ChatInput::Permission { id: p.id, allow });
            }
        }

        // Poll the channels a few times a second even without input events.
        if self.any_busy() {
            ctx.request_repaint_after(Duration::from_millis(50));
        } else {
            ctx.request_repaint_after(Duration::from_millis(120));
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    let one_line: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= max {
        one_line
    } else {
        let mut t: String = one_line.chars().take(max).collect();
        t.push('…');
        t
    }
}

#[cfg(test)]
mod claude_code_tests {
    use super::*;
    use bwoc_core::chat_proto::ChatEvent;
    use serde_json::json;

    #[test]
    fn init_becomes_ready_with_claude_code_backend() {
        let v = json!({"type":"system","subtype":"init","model":"claude-sonnet-4-5","tools":["Read","Bash"]});
        let out = translate_claude_event(&v, "agent-x");
        assert_eq!(out.len(), 1);
        match &out[0] {
            ChatEvent::Ready {
                agent,
                model,
                backend,
                tools,
            } => {
                assert_eq!(agent, "agent-x");
                assert_eq!(model, "claude-sonnet-4-5");
                assert_eq!(backend, "claude-code");
                assert_eq!(tools, &vec!["Read".to_string(), "Bash".to_string()]);
            }
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    #[test]
    fn partial_text_delta_becomes_token() {
        let v = json!({"type":"stream_event","event":{"type":"content_block_delta",
            "delta":{"type":"text_delta","text":"Hi"}}});
        let out = translate_claude_event(&v, "a");
        assert_eq!(out, vec![ChatEvent::Token { text: "Hi".into() }]);
    }

    #[test]
    fn result_emits_message_then_turn_end_with_usage() {
        let v = json!({"type":"result","subtype":"success","is_error":false,
            "result":"done","usage":{"input_tokens":12,"output_tokens":3}});
        let out = translate_claude_event(&v, "a");
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            ChatEvent::Message {
                text: "done".into()
            }
        );
        assert_eq!(
            out[1],
            ChatEvent::TurnEnd {
                prompt_tokens: 12,
                completion_tokens: 3
            }
        );
    }

    #[test]
    fn assistant_tool_use_becomes_tool_call() {
        let v = json!({"type":"assistant","message":{"content":[
            {"type":"text","text":"working"},
            {"type":"tool_use","id":"tu1","name":"Read","input":{"path":"x"}}]}});
        let out = translate_claude_event(&v, "a");
        assert_eq!(out.len(), 1);
        match &out[0] {
            ChatEvent::ToolCall { id, name, .. } => {
                assert_eq!(id, "tu1");
                assert_eq!(name, "Read");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn error_result_becomes_error_event() {
        let v = json!({"type":"result","is_error":true,"result":"Not logged in",
            "usage":{"input_tokens":0,"output_tokens":0}});
        let out = translate_claude_event(&v, "a");
        assert!(matches!(&out[0], ChatEvent::Error { message } if message == "Not logged in"));
    }
}

/// Build the single wrapping galley job for one user/system transcript line:
/// a coloured `"tag: "` prefix + body, wrapped at `max_width` so space-less
/// scripts (e.g. Thai) break at the panel edge, with an explicit `line_height`
/// so stacked Thai vowel/tone marks aren't clipped. Extracted from the render
/// loop so the wrapping + line-height behaviour can be unit-tested headlessly.
fn build_message_job(
    tag: &str,
    text: &str,
    tag_color: egui::Color32,
    text_color: egui::Color32,
    font: egui::FontId,
    line_height: f32,
    max_width: f32,
) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = max_width;
    job.append(
        &format!("{tag}: "),
        0.0,
        egui::TextFormat {
            font_id: font.clone(),
            color: tag_color,
            line_height: Some(line_height),
            ..Default::default()
        },
    );
    job.append(
        text,
        0.0,
        egui::TextFormat {
            font_id: font,
            color: text_color,
            line_height: Some(line_height),
            ..Default::default()
        },
    );
    job
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lay a message job out through egui's real text engine, headless.
    /// `Context::fonts` needs one `run()` first to construct the font atlas.
    fn layout(text: &str, max_width: f32, line_height: f32) -> std::sync::Arc<egui::Galley> {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let job = build_message_job(
            "you",
            text,
            egui::Color32::WHITE,
            egui::Color32::WHITE,
            egui::FontId::proportional(14.0),
            line_height,
            max_width,
        );
        ctx.fonts(|f| f.layout_job(job))
    }

    #[test]
    fn spaceless_text_wraps_to_width() {
        // No whitespace → egui can only break it via the explicit wrap width.
        // This is exactly the Thai-overflow case (Thai has no inter-word spaces);
        // a long no-space ASCII run reproduces it deterministically without
        // depending on a Thai font being installed on the test host.
        let long = "x".repeat(400);
        let g = layout(&long, 180.0, 20.0);
        assert!(
            g.rows.len() > 1,
            "space-less text should wrap to multiple rows, got {}",
            g.rows.len()
        );
        assert!(
            g.size().x <= 182.0,
            "galley width {} should not exceed the 180px wrap width",
            g.size().x
        );
    }

    #[test]
    fn line_height_increases_row_spacing() {
        // Same two lines, taller line_height → taller galley.
        let tight = layout("alpha\nbeta", 1000.0, 14.0);
        let roomy = layout("alpha\nbeta", 1000.0, 28.0);
        assert!(
            roomy.size().y > tight.size().y,
            "a larger line_height should produce a taller galley ({} vs {})",
            roomy.size().y,
            tight.size().y
        );
    }

    #[test]
    fn short_text_stays_one_row() {
        let g = layout("hi", 1000.0, 20.0);
        assert_eq!(g.rows.len(), 1);
    }
}
