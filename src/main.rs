//! `bwoc-chat <agent>` — native desktop chat for ONE BWOC agent, ONE window.
//!
//! A thin egui frontend over the protocol the framework already speaks:
//! it spawns `bwoc-harness --chat` for the agent and renders the
//! `bwoc_core::chat_proto` event stream (the same wire format the ratatui
//! `bwoc chat --tui` uses). The harness owns the session, tools, model calls,
//! and the guardrail→permission pipeline; this window only renders events and
//! sends user messages + permission decisions.
//!
//! Architecture (no async): a reader `std::thread` parses the child's stdout
//! lines into `ChatEvent`s onto an `mpsc` channel; the egui update loop drains
//! the channel each frame, repaints, and writes `ChatInput` lines to the
//! child's stdin. One process = one window = one agent.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use bwoc_core::chat_proto::{ChatEvent, ChatInput};
use bwoc_core::manifest::Manifest;
use bwoc_core::workspace::AgentsRegistry;
use eframe::egui;

/// Default OpenAI-compatible endpoint (Ollama) when the manifest has no
/// `baseUrl`. Mirrors the harness's own default.
const DEFAULT_ENDPOINT: &str = "http://localhost:11434/v1";

/// Client-side slash commands: `(name, description)`. Surfaced as a filtered
/// list when the input starts with `/`; dispatched by [`ChatApp::run_command`].
const COMMANDS: &[(&str, &str)] = &[
    ("/help", "list commands"),
    ("/tools", "list the agent's available tools"),
    ("/clear", "wipe the conversation + tool activity"),
    ("/quit", "close the window"),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = match Config::from_args(std::env::args().skip(1).collect()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "bwoc-chat: {e}\n\n\
                 usage: bwoc-chat [<agent>] [--here | --path <dir>] [--workspace <dir>] \
                 [--model <m>] [--endpoint <url>]\n\
                 \x20 no agent      → personal assistant at ~/.bwoc/personal (created on first use)\n\
                 \x20 --here / .    → the current directory, no workspace\n\
                 \x20 --path <dir>  → that directory directly\n\
                 \x20 <agent>       → a named agent from the workspace registry"
            );
            std::process::exit(2);
        }
    };

    // Spawn `bwoc-harness --chat` for this agent, piped both ways.
    let harness = bwoc_core::exec::binary_or_name("bwoc-harness");
    let mut child = Command::new(&harness)
        .arg("--chat")
        .arg("--workdir")
        .arg(&cfg.agent_path)
        .arg("--model")
        .arg(&cfg.model)
        .arg("--endpoint")
        .arg(&cfg.endpoint)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("failed to spawn bwoc-harness ({harness:?}): {e}"))?;

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

    let title = format!("bwoc · {}", cfg.agent_id);
    let app = ChatApp::new(cfg, stdin, rx, child);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([760.0, 560.0])
            .with_min_inner_size([480.0, 360.0])
            .with_title(title.clone()),
        ..Default::default()
    };
    eframe::run_native(
        &title,
        options,
        Box::new(|cc| {
            install_fonts(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| format!("eframe: {e}"))?;
    Ok(())
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

struct Config {
    agent_id: String,
    agent_path: PathBuf,
    backend: String,
    model: String,
    endpoint: String,
}

impl Config {
    fn from_args(args: Vec<String>) -> Result<Self, String> {
        let mut name: Option<String> = None;
        let mut workspace: Option<PathBuf> = None;
        let mut model_override: Option<String> = None;
        let mut endpoint_override: Option<String> = None;
        let mut path_override: Option<PathBuf> = None;
        let mut here = false;
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--workspace" => workspace = it.next().map(PathBuf::from),
                "--model" => model_override = it.next(),
                "--endpoint" => endpoint_override = it.next(),
                "--path" => path_override = it.next().map(PathBuf::from),
                "--here" | "." => here = true,
                s if s.starts_with("--") => return Err(format!("unknown flag `{s}`")),
                s => {
                    if name.is_some() {
                        return Err(format!("unexpected extra argument `{s}`"));
                    }
                    name = Some(s.to_string());
                }
            }
        }

        // ── Single-agent modes (no workspace, no init) ───────────────────────
        // `--path <dir>` / `--here` / `.`: run the agent in that directory
        // directly — no workspace, no registry.
        if let Some(dir) = path_override {
            return Self::from_dir(dir, model_override, endpoint_override);
        }
        if here {
            let cwd = std::env::current_dir().map_err(|e| format!("cwd: {e}"))?;
            return Self::from_dir(cwd, model_override, endpoint_override);
        }
        // No agent named → the global **personal assistant** at `~/.bwoc/personal`
        // (created on first use). A standalone agent with no workspace to init.
        let Some(name) = name else {
            let dir = ensure_personal_agent()?;
            return Self::from_dir(dir, model_override, endpoint_override);
        };

        // ── Workspace mode (named agent from the registry) ───────────────────
        let workspace = workspace
            .or_else(|| std::env::var_os("BWOC_WORKSPACE").map(PathBuf::from))
            .or_else(resolve_workspace)
            .ok_or(
                "no workspace found — pass --workspace / set BWOC_WORKSPACE / run from a \
                 workspace, or run with no agent for the personal assistant (or --here for \
                 the current directory)",
            )?;

        let registry = AgentsRegistry::load(&workspace)
            .map_err(|e| format!("failed to read agents.toml: {e}"))?;
        let lookup = if name.starts_with("agent-") {
            name.clone()
        } else {
            format!("agent-{name}")
        };
        let entry = registry
            .agents
            .iter()
            .find(|a| a.id == lookup)
            .ok_or_else(|| format!("no agent named '{name}' in {}", workspace.display()))?;

        // Only the harness-driven backends produce a chat_proto stream.
        if !matches!(entry.backend.as_str(), "ollama" | "openai-compatible") {
            return Err(format!(
                "agent '{}' uses the '{}' backend — bwoc-chat only renders the harness chat \
                 stream for ollama / openai-compatible. Use `bwoc spawn` for vendor CLIs.",
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

        Ok(Config {
            agent_id: entry.id.clone(),
            agent_path,
            backend: entry.backend.clone(),
            model,
            endpoint,
        })
    }

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
        Ok(Config {
            agent_id,
            agent_path: dir,
            backend: "ollama".to_string(),
            model,
            endpoint,
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
// UI
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Who {
    User,
    Agent,
    System,
}

struct Pending {
    id: String,
    tool: String,
    detail: String,
}

struct ChatApp {
    agent_id: String,
    status: String,
    convo: Vec<(Who, String)>,
    activity: Vec<String>,
    input: String,
    pending: Option<Pending>,
    busy: bool,
    alive: bool,
    /// Tool names the agent has, from the `Ready` event (for `/tools`).
    tools: Vec<String>,
    stdin: ChildStdin,
    rx: Receiver<ChatEvent>,
    child: Child,
    /// Render cache for the CommonMark (markdown) viewer — code blocks, lists,
    /// emphasis in assistant replies.
    md_cache: egui_commonmark::CommonMarkCache,
}

impl ChatApp {
    fn new(cfg: Config, stdin: ChildStdin, rx: Receiver<ChatEvent>, child: Child) -> Self {
        Self {
            status: format!(
                "{} · {} · {} · connecting…",
                cfg.agent_id, cfg.model, cfg.backend
            ),
            agent_id: cfg.agent_id,
            convo: vec![(
                Who::System,
                "Connected. Type a message and press Enter — or /help for commands.".to_string(),
            )],
            activity: Vec::new(),
            input: String::new(),
            pending: None,
            busy: false,
            alive: true,
            tools: Vec::new(),
            stdin,
            rx,
            child,
            md_cache: egui_commonmark::CommonMarkCache::default(),
        }
    }

    fn apply(&mut self, ev: ChatEvent) {
        match ev {
            ChatEvent::Ready {
                agent,
                model,
                backend,
                tools,
            } => {
                self.status = format!("{agent} · {model} · {backend} · ready");
                self.tools = tools;
            }
            ChatEvent::Token { text } => {
                // Append streamed tokens onto the in-progress agent message.
                match self.convo.last_mut() {
                    Some((Who::Agent, s)) if self.busy => s.push_str(&text),
                    _ => self.convo.push((Who::Agent, text)),
                }
            }
            ChatEvent::Message { text } => {
                // Final assistant text for the turn (overrides any streamed buffer).
                match self.convo.last_mut() {
                    Some((Who::Agent, s)) if self.busy => *s = text,
                    _ => self.convo.push((Who::Agent, text)),
                }
            }
            ChatEvent::ToolCall { name, args, .. } => {
                self.activity
                    .push(format!("» {name} {}", truncate(&args, 80)));
            }
            ChatEvent::ToolResult {
                name, ok, output, ..
            } => {
                let mark = if ok { "[ok]" } else { "[err]" };
                self.activity
                    .push(format!("{mark} {name}: {}", truncate(&output, 80)));
            }
            ChatEvent::PermissionRequest { id, tool, detail } => {
                self.pending = Some(Pending { id, tool, detail });
            }
            ChatEvent::TurnEnd {
                prompt_tokens,
                completion_tokens,
            } => {
                self.busy = false;
                self.status = format!(
                    "{} · ready · tokens {} in / {} out",
                    self.agent_id, prompt_tokens, completion_tokens
                );
            }
            ChatEvent::Error { message } => {
                self.busy = false;
                self.convo.push((Who::System, format!("error: {message}")));
            }
            ChatEvent::Bye => {
                self.alive = false;
                self.convo.push((Who::System, "session ended.".to_string()));
            }
        }
    }

    fn send_user(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || !self.alive {
            return;
        }
        self.convo.push((Who::User, text.clone()));
        self.input.clear();
        self.busy = true;
        self.write_input(&ChatInput::User { text });
    }

    /// Handle a client-side `/command` (intercepted before it reaches the
    /// harness). `ctx` is needed so `/quit` can close the window.
    fn run_command(&mut self, line: &str, ctx: &egui::Context) {
        let mut parts = line.trim().splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("").to_lowercase();
        let _arg = parts.next().unwrap_or("").trim();
        let sys = |s: &mut Self, text: String| s.convo.push((Who::System, text));
        match name.as_str() {
            "" | "help" | "?" => sys(
                self,
                "commands: /help · /clear (wipe view) · /quit (close)".to_string(),
            ),
            "tools" => {
                let msg = if self.tools.is_empty() {
                    "no tools reported (older harness, or none registered).".to_string()
                } else {
                    format!("{} tools: {}", self.tools.len(), self.tools.join(", "))
                };
                sys(self, msg);
            }
            "clear" => {
                self.convo.clear();
                self.activity.clear();
                sys(self, "conversation cleared.".to_string());
            }
            "quit" | "exit" => {
                self.write_input(&ChatInput::Quit);
                self.alive = false;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            other => sys(self, format!("unknown command `/{other}` — try /help")),
        }
    }

    fn answer_permission(&mut self, allow: bool) {
        if let Some(p) = self.pending.take() {
            self.activity.push(format!(
                "{} {}",
                if allow { "[allowed]" } else { "[denied]" },
                p.tool
            ));
            self.write_input(&ChatInput::Permission { id: p.id, allow });
        }
    }

    fn write_input(&mut self, input: &ChatInput) {
        if let Ok(line) = input.to_line() {
            let _ = writeln!(self.stdin, "{line}");
            let _ = self.stdin.flush();
        }
    }
}

impl Drop for ChatApp {
    fn drop(&mut self) {
        // Best-effort graceful shutdown, then reap so the harness never orphans.
        self.write_input(&ChatInput::Quit);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain everything the reader thread queued since last frame.
        let events: Vec<ChatEvent> = self.rx.try_iter().collect();
        for ev in events {
            self.apply(ev);
        }

        let mut do_send = false;
        let mut perm: Option<bool> = None;
        let mut run_cmd: Option<&'static str> = None;

        egui::TopBottomPanel::top("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.strong("bwoc-chat");
                ui.separator();
                ui.label(&self.status);
                if self.busy {
                    ui.spinner();
                }
            });
        });

        egui::SidePanel::right("activity")
            .default_width(240.0)
            .show(ctx, |ui| {
                ui.heading("tools");
                ui.separator();
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if self.activity.is_empty() {
                            ui.weak("(no tool activity yet)");
                        }
                        for line in &self.activity {
                            ui.label(line);
                        }
                    });
            });

        egui::TopBottomPanel::bottom("input").show(ctx, |ui| {
            ui.add_space(4.0);
            if let Some(p) = &self.pending {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new(format!("permission: {} ", p.tool))
                            .color(egui::Color32::from_rgb(0xE0, 0xA0, 0x30))
                            .strong(),
                    );
                    ui.label(truncate(&p.detail, 120));
                    if ui.button("Allow").clicked() {
                        perm = Some(true);
                    }
                    if ui.button("Deny").clicked() {
                        perm = Some(false);
                    }
                });
            } else {
                // Slash-command list: when the input starts with `/`, show the
                // matching commands above the input. Click a row to run it.
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
                ui.horizontal(|ui| {
                    let hint = if self.alive {
                        "message…  (/help for commands)"
                    } else {
                        "(session ended)"
                    };
                    let resp = ui.add_enabled(
                        self.alive,
                        egui::TextEdit::singleline(&mut self.input)
                            .desired_width(f32::INFINITY)
                            .hint_text(hint),
                    );
                    let entered =
                        resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if entered {
                        do_send = true;
                        resp.request_focus();
                    }
                    if ui
                        .add_enabled(self.alive, egui::Button::new("Send"))
                        .clicked()
                    {
                        do_send = true;
                    }
                });
            }
            ui.add_space(4.0);
        });

        // Take the markdown cache out of `self` so the conversation loop can
        // borrow `&self.convo` immutably and the viewer `&mut cache` at once.
        let mut md_cache = std::mem::take(&mut self.md_cache);
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for (who, text) in &self.convo {
                        let (tag, color) = match who {
                            Who::User => ("you", egui::Color32::from_rgb(0x6C, 0xB6, 0xFF)),
                            Who::Agent => (
                                self.agent_id.as_str(),
                                egui::Color32::from_rgb(0x9E, 0xE0, 0x93),
                            ),
                            Who::System => ("·", egui::Color32::GRAY),
                        };
                        match who {
                            // Assistant replies render as markdown (code blocks,
                            // lists, emphasis); the tag goes on its own line so a
                            // fenced block gets full width.
                            Who::Agent => {
                                ui.label(
                                    egui::RichText::new(format!("{tag}:")).color(color).strong(),
                                );
                                egui_commonmark::CommonMarkViewer::new().show(
                                    ui,
                                    &mut md_cache,
                                    text,
                                );
                            }
                            _ => {
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(
                                        egui::RichText::new(format!("{tag}: "))
                                            .color(color)
                                            .strong(),
                                    );
                                    ui.label(text);
                                });
                            }
                        }
                        ui.add_space(6.0);
                    }
                });
        });
        self.md_cache = md_cache;

        // A clicked command from the slash list runs immediately.
        if let Some(name) = run_cmd {
            self.input.clear();
            self.run_command(name.trim_start_matches('/'), ctx);
        } else if do_send {
            // A leading `/` is a client-side command, not a message to the agent.
            let text = self.input.trim().to_string();
            if let Some(cmd) = text.strip_prefix('/') {
                self.input.clear();
                self.run_command(cmd, ctx);
            } else {
                self.send_user();
            }
        }
        if let Some(allow) = perm {
            self.answer_permission(allow);
        }

        // Poll the channel a few times a second even without input events.
        ctx.request_repaint_after(Duration::from_millis(80));
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
