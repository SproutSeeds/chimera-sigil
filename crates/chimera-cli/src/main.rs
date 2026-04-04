use chimera_sigil_core::agent::{AgentEvent, ApprovalCallback, ApprovalDecision, EventCallback};
use chimera_sigil_core::{Agent, ApprovalMode, Config, ExitReason, Session};
use chimera_sigil_providers::{
    create_collaborative_provider, create_fallback_provider, create_provider,
};
use chimera_sigil_tools::PermissionLevel;
use clap::{Parser, ValueEnum};
use colored::*;
use std::io::{self, Write};

#[derive(Parser)]
#[command(
    name = "chimera",
    about = "Multi-model AI agent harness for Grok, OpenAI, Anthropic, and Ollama",
    version
)]
struct Cli {
    /// Model to use (e.g., grok-3, gpt-4o, llama3.2:latest)
    #[arg(short, long, default_value = "grok-3")]
    model: String,

    /// Run a single prompt instead of starting the REPL
    #[arg(short, long)]
    prompt: Option<String>,

    /// Temperature for model responses (0.0 - 2.0)
    #[arg(short, long)]
    temperature: Option<f32>,

    /// Fallback models (comma-separated, e.g., "grok-3,gpt-4o,sonnet")
    #[arg(long)]
    fallback: Option<String>,

    /// Collaborating advisor models (comma-separated, e.g., "sonnet,gpt-4o")
    #[arg(long)]
    collab: Option<String>,

    /// Auto-approve all tool executions (shorthand for --approval-mode full)
    #[arg(long, default_value_t = false)]
    auto_approve: bool,

    /// Tool approval policy: prompt, approve (allow writes, deny execute), or full
    #[arg(long, value_enum)]
    approval_mode: Option<CliApprovalMode>,

    /// Resume a previous session by ID
    #[arg(long)]
    resume: Option<String>,

    /// Emit machine-readable JSONL output (one JSON object per event)
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Enable verbose logging
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum CliApprovalMode {
    Prompt,
    Approve,
    Full,
}

fn main() {
    let cli = Cli::parse();

    // Set up tracing
    let filter = if cli.verbose { "debug" } else { "warn" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .with_target(false)
        .init();

    // Run the async runtime
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("{} Failed to start runtime: {e}", "Error:".red().bold());
            std::process::exit(1);
        }
    };

    let code = rt.block_on(run(cli));
    std::process::exit(code);
}

/// Exit codes.
const EXIT_SUCCESS: i32 = 0;
const EXIT_PROVIDER_ERROR: i32 = 1;
const EXIT_TOOL_ERROR: i32 = 2;
const EXIT_MAX_ITERATIONS: i32 = 3;
const EXIT_STREAM_ERROR: i32 = 4;
const EXIT_SESSION_ERROR: i32 = 5;

fn resolve_approval_mode(cli: &Cli) -> anyhow::Result<ApprovalMode> {
    let mode = match cli.approval_mode.unwrap_or(CliApprovalMode::Prompt) {
        CliApprovalMode::Prompt => ApprovalMode::Prompt,
        CliApprovalMode::Approve => ApprovalMode::Approve,
        CliApprovalMode::Full => ApprovalMode::Full,
    };

    if cli.auto_approve {
        if cli.approval_mode.is_some() && mode != ApprovalMode::Full {
            anyhow::bail!("Use either --auto-approve or --approval-mode, not both.");
        }
        Ok(ApprovalMode::Full)
    } else {
        Ok(mode)
    }
}

async fn run(cli: Cli) -> i32 {
    let approval_mode = match resolve_approval_mode(&cli) {
        Ok(mode) => mode,
        Err(e) => {
            eprintln!("{} {e}", "Error:".red().bold());
            return EXIT_PROVIDER_ERROR;
        }
    };

    // Create the provider (with optional fallback chain)
    let (provider, model) = if let Some(ref fallback) = cli.fallback {
        let chain = format!("{},{}", cli.model, fallback);
        match create_fallback_provider(&chain) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{} {e}", "Error:".red().bold());
                std::process::exit(1);
            }
        }
    } else {
        match create_provider(&cli.model) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{} {e}", "Error:".red().bold());
                eprintln!(
                    "\n{} Set the appropriate API key environment variable:",
                    "Hint:".yellow()
                );
                eprintln!("  Grok:      export XAI_API_KEY=your-key");
                eprintln!("  OpenAI:    export OPENAI_API_KEY=your-key");
                eprintln!("  Anthropic: export ANTHROPIC_API_KEY=your-key");
                eprintln!("  Ollama:    No key needed (uses http://localhost:11434)");
                std::process::exit(1);
            }
        }
    };

    let (provider, collaborators) = if let Some(ref collab) = cli.collab {
        match create_collaborative_provider(provider, model.clone(), collab) {
            Ok((provider, collaborators)) => (provider, collaborators),
            Err(e) => {
                eprintln!("{} {e}", "Error:".red().bold());
                std::process::exit(1);
            }
        }
    } else {
        (provider, Vec::new())
    };

    let config = Config {
        model: model.clone(),
        temperature: cli.temperature,
        approval_mode,
        ..Config::default()
    };

    let provider_kind = provider.kind();
    let mut agent = Agent::new(provider, model.clone(), config);

    // Resume a previous session if requested
    if let Some(session_id) = &cli.resume {
        match Session::load(session_id) {
            Ok(session) => {
                eprintln!(
                    "  {} session {} ({} messages)",
                    "Resumed".green().bold(),
                    session_id.dimmed(),
                    session.messages.len()
                );
                *agent.session_mut() = session;
            }
            Err(e) => {
                eprintln!("{} Could not resume session: {e}", "Error:".red().bold());
                std::process::exit(1);
            }
        }
    }

    if approval_mode == ApprovalMode::Prompt {
        agent.set_approval_callback(make_approval_callback());
    }

    let json_mode = cli.json;

    // Single prompt mode
    if let Some(prompt) = cli.prompt {
        let callback = if json_mode {
            make_json_callback()
        } else {
            make_callback()
        };
        emit_event(
            &callback,
            AgentEvent::SessionReady {
                session_id: agent.session().id.clone(),
            },
        );
        match agent.run_turn(&prompt, &callback).await {
            Ok(outcome) => {
                match agent.session().save() {
                    Ok(path) => emit_event(
                        &callback,
                        AgentEvent::SessionSaved {
                            session_id: agent.session().id.clone(),
                            path: path.display().to_string(),
                        },
                    ),
                    Err(e) => {
                        eprintln!("{} {e}", "Error saving session:".red().bold());
                        return EXIT_SESSION_ERROR;
                    }
                }

                if !json_mode {
                    println!();
                }
                return match outcome.exit_reason {
                    ExitReason::Complete => EXIT_SUCCESS,
                    ExitReason::MaxIterations => EXIT_MAX_ITERATIONS,
                    ExitReason::StreamError => EXIT_STREAM_ERROR,
                    ExitReason::ToolError => EXIT_TOOL_ERROR,
                };
            }
            Err(e) => {
                eprintln!("{} {e}", "Error:".red().bold());
                return EXIT_PROVIDER_ERROR;
            }
        }
    }

    // Interactive REPL
    print_banner(&model, provider_kind, &collaborators, json_mode);

    loop {
        // Print prompt
        if json_mode {
            eprint!("\n{} ", "chimera>".cyan().bold());
            let _ = io::stderr().flush();
        } else {
            print!("\n{} ", "chimera>".cyan().bold());
            let _ = io::stdout().flush();
        }

        // Read input
        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input).unwrap_or(0);
        if bytes_read == 0 {
            // EOF
            if json_mode {
                eprintln!("\n{}", "Goodbye!".dimmed());
            } else {
                println!("\n{}", "Goodbye!".dimmed());
            }
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle special commands
        match input {
            "/quit" | "/exit" | "/q" => {
                if json_mode {
                    eprintln!("{}", "Goodbye!".dimmed());
                } else {
                    println!("{}", "Goodbye!".dimmed());
                }
                break;
            }
            "/model" => {
                if json_mode {
                    eprintln!("Current model: {}", model.green());
                } else {
                    println!("Current model: {}", model.green());
                }
                continue;
            }
            "/usage" => {
                let session = agent.session();
                if json_mode {
                    eprintln!(
                        "Tokens — input: {}, output: {}, total: {}",
                        session.total_input_tokens.to_string().yellow(),
                        session.total_output_tokens.to_string().yellow(),
                        session.total_tokens().to_string().green().bold(),
                    );
                } else {
                    println!(
                        "Tokens — input: {}, output: {}, total: {}",
                        session.total_input_tokens.to_string().yellow(),
                        session.total_output_tokens.to_string().yellow(),
                        session.total_tokens().to_string().green().bold(),
                    );
                }
                continue;
            }
            "/save" => {
                match agent.session().save() {
                    Ok(path) => {
                        if json_mode {
                            eprintln!(
                                "{} Session saved to {}",
                                "Saved:".green().bold(),
                                path.display().to_string().dimmed()
                            );
                            eprintln!(
                                "  Resume with: {} {}",
                                "chimera --resume".cyan(),
                                agent.session().id.cyan()
                            );
                        } else {
                            println!(
                                "{} Session saved to {}",
                                "Saved:".green().bold(),
                                path.display().to_string().dimmed()
                            );
                            println!(
                                "  Resume with: {} {}",
                                "chimera --resume".cyan(),
                                agent.session().id.cyan()
                            );
                        }
                    }
                    Err(e) => eprintln!("{} {e}", "Error saving:".red().bold()),
                }
                continue;
            }
            "/sessions" => {
                match Session::list_saved() {
                    Ok(sessions) if sessions.is_empty() => {
                        if json_mode {
                            eprintln!("{}", "No saved sessions.".dimmed());
                        } else {
                            println!("{}", "No saved sessions.".dimmed());
                        }
                    }
                    Ok(sessions) => {
                        if json_mode {
                            eprintln!("{}", "Saved sessions:".bold());
                        } else {
                            println!("{}", "Saved sessions:".bold());
                        }
                        for (id, modified) in sessions.iter().take(20) {
                            let age = modified
                                .elapsed()
                                .map(|d| format!("{}m ago", d.as_secs() / 60))
                                .unwrap_or_else(|_| "unknown".into());
                            if json_mode {
                                eprintln!("  {} ({})", id.cyan(), age.dimmed());
                            } else {
                                println!("  {} ({})", id.cyan(), age.dimmed());
                            }
                        }
                    }
                    Err(e) => eprintln!("{} {e}", "Error:".red().bold()),
                }
                continue;
            }
            "/help" => {
                print_help(json_mode);
                continue;
            }
            _ => {}
        }

        if !json_mode {
            println!();
        } // Visual separator before response

        let callback = if json_mode {
            make_json_callback()
        } else {
            make_callback()
        };
        match agent.run_turn(input, &callback).await {
            Ok(_) => {
                if !json_mode {
                    println!();
                }
            }
            Err(e) => {
                eprintln!("\n{} {e}", "Error:".red().bold());
            }
        }
    }

    EXIT_SUCCESS
}

fn make_json_callback() -> EventCallback {
    Box::new(|event| {
        if let Ok(json) = serde_json::to_string(&event) {
            println!("{json}");
            let _ = io::stdout().flush();
        }
    })
}

fn make_callback() -> EventCallback {
    Box::new(|event| match event {
        AgentEvent::SessionReady { .. } => {}
        AgentEvent::TextDelta { text } => {
            print!("{text}");
            let _ = io::stdout().flush();
        }
        AgentEvent::ToolStart {
            tool_call_id,
            name,
            arguments,
        } => {
            let preview = if arguments.len() > 120 {
                format!("{}...", &arguments[..120])
            } else {
                arguments.clone()
            };
            eprintln!(
                "\n{} {} {} {}",
                "  tool>".dimmed(),
                name.yellow().bold(),
                format!("[{}]", &tool_call_id[..tool_call_id.len().min(8)]).dimmed(),
                preview.dimmed()
            );
        }
        AgentEvent::ToolResult {
            tool_call_id,
            name,
            result,
            is_error,
        } => {
            let status = if is_error {
                "FAIL".red().bold().to_string()
            } else {
                "OK".green().bold().to_string()
            };

            let preview = if result.len() > 200 {
                format!("{}... ({} chars)", &result[..200], result.len())
            } else {
                result.clone()
            };

            eprintln!(
                "{} {} {} [{}] {}",
                "  tool>".dimmed(),
                name.dimmed(),
                format!("[{}]", &tool_call_id[..tool_call_id.len().min(8)]).dimmed(),
                status,
                preview.dimmed()
            );
        }
        AgentEvent::Usage {
            input_tokens,
            output_tokens,
        } => {
            eprintln!(
                "{}",
                format!("  [{input_tokens} in / {output_tokens} out]").dimmed()
            );
        }
        AgentEvent::TurnComplete { iterations, .. } => {
            if iterations > 1 {
                eprintln!("{}", format!("  [{iterations} iterations]").dimmed());
            }
        }
        AgentEvent::ToolDenied {
            tool_call_id,
            name,
            reason,
        } => {
            eprintln!(
                "{} {} {} {}",
                "  tool>".dimmed(),
                name.yellow(),
                format!("[{}]", &tool_call_id[..tool_call_id.len().min(8)]).dimmed(),
                "DENIED".red().bold()
            );
            eprintln!("{}", format!("    {reason}").dimmed());
        }
        AgentEvent::SessionSaved { .. } => {}
        AgentEvent::Error { message } => {
            eprintln!("{} {message}", "  Error:".red().bold());
        }
    })
}

fn emit_event(callback: &EventCallback, event: AgentEvent) {
    callback(event);
}

fn make_approval_callback() -> ApprovalCallback {
    Box::new(|name, _args, permission| {
        let perm_str = match permission {
            PermissionLevel::WorkspaceWrite => "write",
            PermissionLevel::Execute => "execute",
            _ => "read",
        };
        eprint!(
            "\n  {} {} [{}] allow? [y/n/a(ll)] ",
            "approve>".yellow().bold(),
            name.cyan(),
            perm_str.red()
        );
        let _ = io::stderr().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
        match input.trim().to_lowercase().as_str() {
            "y" | "yes" => ApprovalDecision::Allow,
            "a" | "all" | "always" => ApprovalDecision::AllowAll,
            _ => ApprovalDecision::Deny,
        }
    })
}

fn print_banner(
    model: &str,
    provider: chimera_sigil_providers::ProviderKind,
    collaborators: &[String],
    json_mode: bool,
) {
    if json_mode {
        return;
    }

    println!(
        "{}",
        r#"
     _____ _     _
    / ____| |   (_)
   | |    | |__  _ _ __ ___   ___ _ __ __ _
   | |    | '_ \| | '_ ` _ \ / _ \ '__/ _` |
   | |____| | | | | | | | | |  __/ | | (_| |
    \_____|_| |_|_|_| |_| |_|\___|_|  \__,_|
"#
        .cyan()
        .bold()
    );
    println!("  {}", "Chimera Sigil".cyan().bold());

    println!(
        "  {} {} via {}",
        "Model:".dimmed(),
        model.green().bold(),
        provider.to_string().yellow()
    );
    if !collaborators.is_empty() {
        println!(
            "  {} {}",
            "Collaborators:".dimmed(),
            collaborators.join(", ").cyan()
        );
    }
    println!("  {}", "Type /help for commands, /quit to exit".dimmed());
}

fn print_help(json_mode: bool) {
    macro_rules! outln {
        ($($arg:tt)*) => {
            if json_mode {
                eprintln!($($arg)*);
            } else {
                println!($($arg)*);
            }
        };
    }

    outln!("{}", "Commands:".bold());
    outln!("  {}      — Exit the REPL", "/quit".yellow());
    outln!("  {}     — Show current model", "/model".yellow());
    outln!("  {}     — Show token usage", "/usage".yellow());
    outln!("  {}      — Save session to disk", "/save".yellow());
    outln!("  {}  — List saved sessions", "/sessions".yellow());
    outln!("  {}      — Show this help", "/help".yellow());
    outln!();
    outln!("{}", "Supported models:".bold());
    outln!(
        "  {} — grok-3, grok-3-mini, grok-3-fast",
        "Grok (xAI)".yellow()
    );
    outln!("  {} — gpt-4o, gpt-4o-mini, o3, o4-mini", "OpenAI".yellow());
    outln!(
        "  {} — opus, sonnet, haiku (claude-opus-4-6, etc.)",
        "Anthropic".yellow()
    );
    outln!(
        "  {} — Any model via http://localhost:11434",
        "Ollama".yellow()
    );
    outln!();
    outln!("{}", "Multi-Model:".bold());
    outln!(
        "  {} — Add advisor models that contribute parallel perspectives each turn",
        "--collab sonnet,gpt-4o".yellow()
    );
    outln!(
        "  {} — Fallback chain if the primary provider fails",
        "--fallback gpt-4o,sonnet".yellow()
    );
    outln!();
    outln!("{}", "Environment variables:".bold());
    outln!("  XAI_API_KEY       — xAI/Grok API key");
    outln!("  OPENAI_API_KEY    — OpenAI API key");
    outln!("  ANTHROPIC_API_KEY — Anthropic/Claude API key");
    outln!("  OLLAMA_BASE_URL   — Ollama server URL (default: http://localhost:11434/v1)");
}
