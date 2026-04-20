use chimera_sigil_core::agent::{AgentEvent, ApprovalCallback, ApprovalDecision, EventCallback};
use chimera_sigil_core::{Agent, ApprovalMode, Config, ExitReason, Session};
use chimera_sigil_providers::{
    DEFAULT_LOCAL_MODEL, create_collaborative_provider, create_fallback_provider, create_provider,
    ollama_native_api_base_url, ollama_route_for_model, profile_ollama_env_name, resolve_model,
};
use chimera_sigil_tools::PermissionLevel;
use clap::{Parser, ValueEnum};
use colored::*;
use serde_json::json;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "chimera",
    about = "Local-first multi-model AI agent harness for Ollama, Grok, OpenAI, and Anthropic",
    version
)]
struct Cli {
    /// Model to use (e.g., local, local-coder, qwen3:4b, gpt-4o)
    #[arg(short, long, default_value = "local")]
    model: String,

    /// Run a single prompt instead of starting the REPL
    #[arg(short, long)]
    prompt: Option<String>,

    /// Temperature for model responses (0.0 - 2.0)
    #[arg(short, long)]
    temperature: Option<f32>,

    /// Fallback models (comma-separated, e.g., "local-coder,local-small")
    #[arg(long)]
    fallback: Option<String>,

    /// Collaborating advisor models (comma-separated, e.g., "local-tiny,local-coder")
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

    /// Maximum model/tool iterations per user turn
    #[arg(long, default_value_t = 25)]
    max_iterations: usize,

    /// Provider retries after the initial request fails
    #[arg(long, default_value_t = 2)]
    provider_retries: usize,

    /// Require the final response to include report-shaped JSON (`task` and `target`)
    #[arg(long, default_value_t = false)]
    require_report_json: bool,

    /// Repair attempts when --require-report-json is enabled and the final response misses it
    #[arg(long, default_value_t = 1)]
    report_repair_attempts: usize,

    /// Do not save the session automatically after each completed turn
    #[arg(long, default_value_t = false)]
    no_auto_save: bool,

    /// Emit machine-readable JSONL output (one JSON object per event)
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Inspect local hardware/Ollama and recommend a local model profile
    #[arg(long, default_value_t = false)]
    local_doctor: bool,

    /// Benchmark local Ollama model profiles and emit JSONL results
    #[arg(long, default_value_t = false)]
    local_benchmark: bool,

    /// Models/profiles to benchmark (comma-separated). Defaults to doctor recommendation.
    #[arg(long)]
    benchmark_models: Option<String>,

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
    if cli.local_doctor {
        let probe = HardwareProbe::detect();
        let recommendation = recommend_local_profiles(&probe);
        print_local_doctor(&probe, &recommendation, cli.json);
        return EXIT_SUCCESS;
    }

    if cli.local_benchmark {
        let probe = HardwareProbe::detect();
        let recommendation = recommend_local_profiles(&probe);
        let profiles = benchmark_profiles(&cli, &recommendation);
        match run_local_benchmark(&probe, &recommendation, profiles).await {
            Ok(()) => return EXIT_SUCCESS,
            Err(e) => {
                eprintln!("{} {e}", "Error:".red().bold());
                return EXIT_PROVIDER_ERROR;
            }
        }
    }

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
                eprintln!(
                    "  Local:     install Ollama, then run ollama pull {DEFAULT_LOCAL_MODEL}"
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
        max_iterations: cli.max_iterations,
        temperature: cli.temperature,
        approval_mode,
        provider_retries: cli.provider_retries,
        require_report_json: cli.require_report_json,
        report_repair_attempts: cli.report_repair_attempts,
        ..Config::default()
    };

    let provider_kind = provider.kind();
    let local_route = if provider_kind == chimera_sigil_providers::ProviderKind::Ollama {
        Some(ollama_route_for_model(&cli.model))
    } else {
        None
    };
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
                if !cli.no_auto_save
                    && let Err(e) = save_session_and_emit(&agent, &callback)
                {
                    eprintln!("{} {e}", "Error saving session:".red().bold());
                    return EXIT_SESSION_ERROR;
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
                if provider_kind == chimera_sigil_providers::ProviderKind::Ollama {
                    print_local_setup_hint(&model, local_route.as_ref());
                }
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
                if provider_kind == chimera_sigil_providers::ProviderKind::Ollama {
                    print_model_route(&cli.model, &model, local_route.as_ref(), json_mode);
                }
                continue;
            }
            "/route" => {
                print_model_route(&cli.model, &model, local_route.as_ref(), json_mode);
                continue;
            }
            "/doctor" => {
                let probe = HardwareProbe::detect();
                let recommendation = recommend_local_profiles(&probe);
                print_local_doctor(&probe, &recommendation, json_mode);
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
            Ok(outcome) => {
                if !cli.no_auto_save
                    && let Err(e) = save_session_and_emit(&agent, &callback)
                {
                    eprintln!("{} {e}", "Error saving session:".red().bold());
                }

                if !json_mode {
                    println!();
                }

                if outcome.exit_reason != ExitReason::Complete {
                    eprintln!(
                        "{} {}",
                        "Turn stopped:".yellow().bold(),
                        format!("{:?}", outcome.exit_reason).dimmed()
                    );
                }
            }
            Err(e) => {
                eprintln!("\n{} {e}", "Error:".red().bold());
                if provider_kind == chimera_sigil_providers::ProviderKind::Ollama {
                    print_local_setup_hint(&model, local_route.as_ref());
                }
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
        AgentEvent::ProviderRetry {
            attempt,
            max_attempts,
            delay_ms,
            error,
        } => {
            eprintln!(
                "{} {}",
                "  retry>".yellow().bold(),
                format!(
                    "provider attempt {attempt}/{max_attempts} failed; retrying in {delay_ms}ms"
                )
                .dimmed()
            );
            eprintln!("{}", format!("    {error}").dimmed());
        }
        AgentEvent::ContextCompacted {
            removed_messages,
            estimated_tokens,
            context_window,
        } => {
            eprintln!(
                "{}",
                format!(
                    "  [compacted {removed_messages} messages; estimated {estimated_tokens}/{context_window} tokens]"
                )
                .dimmed()
            );
        }
        AgentEvent::ReportContractRepair {
            attempt,
            max_attempts,
            reason,
        } => {
            eprintln!(
                "{} {}",
                "  repair>".yellow().bold(),
                format!("report contract repair {attempt}/{max_attempts}: {reason}").dimmed()
            );
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
        AgentEvent::SessionSaved { path, .. } => {
            eprintln!("{}", format!("  [session saved: {path}]").dimmed());
        }
        AgentEvent::Error { message } => {
            eprintln!("{} {message}", "  Error:".red().bold());
        }
    })
}

fn emit_event(callback: &EventCallback, event: AgentEvent) {
    callback(event);
}

fn save_session_and_emit(agent: &Agent, callback: &EventCallback) -> anyhow::Result<()> {
    let path = agent.session().save()?;
    emit_event(
        callback,
        AgentEvent::SessionSaved {
            session_id: agent.session().id.clone(),
            path: path.display().to_string(),
        },
    );
    Ok(())
}

#[derive(Debug, Clone, Default)]
struct HardwareProbe {
    os: String,
    arch: String,
    total_ram_gb: Option<f32>,
    gpu_name: Option<String>,
    gpu_vram_gb: Option<f32>,
    nvidia_smi_available: bool,
    ollama_available: bool,
    ollama_models: Vec<String>,
    tailscale_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalRecommendation {
    primary: &'static str,
    coding: &'static str,
    fallback: Vec<&'static str>,
    benchmark: Vec<&'static str>,
    tier: &'static str,
    note: &'static str,
}

impl HardwareProbe {
    fn detect() -> Self {
        let (gpu_name, gpu_vram_gb, nvidia_smi_available) = detect_nvidia_gpu();
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            total_ram_gb: detect_total_ram_gb(),
            gpu_name,
            gpu_vram_gb,
            nvidia_smi_available,
            ollama_available: command_available("ollama"),
            ollama_models: detect_ollama_models(),
            tailscale_available: command_available("tailscale"),
        }
    }
}

fn recommend_local_profiles(probe: &HardwareProbe) -> LocalRecommendation {
    let vram = probe.gpu_vram_gb.unwrap_or(0.0);
    let ram = probe.total_ram_gb.unwrap_or(0.0);

    if vram >= 23.0 {
        return LocalRecommendation {
            primary: "local-4090",
            coding: "local-coder-4090",
            fallback: vec!["local-16gb", "local-balanced"],
            benchmark: vec!["local-4090", "local-coder-4090", "local-coder-16gb"],
            tier: "24GB GPU",
            note: "Start with 30B/32B models, but keep context conservative until benchmarks prove VRAM residency.",
        };
    }

    if vram >= 16.0 {
        return LocalRecommendation {
            primary: "local-16gb",
            coding: "local-coder-16gb",
            fallback: vec!["local-balanced", "local-coder"],
            benchmark: vec!["local-16gb", "local-coder-16gb", "local-balanced"],
            tier: "16GB GPU",
            note: "Use the 14B lane first; demote to 7B if context growth spills out of VRAM.",
        };
    }

    if vram >= 12.0 {
        return LocalRecommendation {
            primary: "local-12gb",
            coding: "local-coder-12gb",
            fallback: vec!["local-balanced", "local-coder"],
            benchmark: vec!["local-12gb", "local-coder-12gb", "local-coder"],
            tier: "12GB GPU",
            note: "Try 14B models with modest context; keep 7B coding models ready as the stable daily lane.",
        };
    }

    if vram >= 8.0 || ram >= 24.0 {
        return LocalRecommendation {
            primary: "local-balanced",
            coding: "local-coder",
            fallback: vec!["local", "local-coder-small"],
            benchmark: vec!["local-balanced", "local-coder", "local"],
            tier: "balanced local",
            note: "Use 8B/7B models as the working set; avoid multi-model collaboration until latency is measured.",
        };
    }

    if ram >= 16.0 {
        return LocalRecommendation {
            primary: "local",
            coding: "local-coder-small",
            fallback: vec!["local-tiny"],
            benchmark: vec!["local", "local-coder-small", "local-tiny"],
            tier: "budget laptop",
            note: "This is the default free lane for newer budget laptops; prefer fallback over collaboration.",
        };
    }

    LocalRecommendation {
        primary: "local-tiny",
        coding: "local-coder-small",
        fallback: vec!["local-tiny"],
        benchmark: vec!["local-tiny", "local"],
        tier: "tiny local",
        note: "Keep tasks short. Use this as the minimum viable local path.",
    }
}

fn print_local_doctor(
    probe: &HardwareProbe,
    recommendation: &LocalRecommendation,
    json_mode: bool,
) {
    let (primary_model, _) = resolve_model(recommendation.primary);
    let (coding_model, _) = resolve_model(recommendation.coding);
    let primary_route = ollama_route_for_model(recommendation.primary);
    let coding_route = ollama_route_for_model(recommendation.coding);

    if json_mode {
        println!(
            "{}",
            json!({
                "type": "local_doctor",
                "hardware": {
                    "os": probe.os,
                    "arch": probe.arch,
                    "total_ram_gb": probe.total_ram_gb,
                    "gpu_name": probe.gpu_name,
                    "gpu_vram_gb": probe.gpu_vram_gb,
                    "nvidia_smi_available": probe.nvidia_smi_available,
                    "ollama_available": probe.ollama_available,
                    "ollama_models": probe.ollama_models,
                    "tailscale_available": probe.tailscale_available,
                },
                "recommendation": {
                    "tier": recommendation.tier,
                    "primary": recommendation.primary,
                    "primary_model": primary_model,
                    "primary_base_url": primary_route.base_url,
                    "primary_base_url_source": primary_route.source,
                    "coding": recommendation.coding,
                    "coding_model": coding_model,
                    "coding_base_url": coding_route.base_url,
                    "coding_base_url_source": coding_route.source,
                    "fallback": recommendation.fallback,
                    "benchmark": recommendation.benchmark,
                    "note": recommendation.note,
                }
            })
        );
        return;
    }

    println!("{}", "Local Doctor".cyan().bold());
    println!("  {} {} / {}", "System:".dimmed(), probe.os, probe.arch);
    match probe.total_ram_gb {
        Some(ram) => println!("  {} {:.1} GB", "RAM:".dimmed(), ram),
        None => println!("  {} unknown", "RAM:".dimmed()),
    }
    match (&probe.gpu_name, probe.gpu_vram_gb) {
        (Some(name), Some(vram)) => {
            println!("  {} {} ({:.1} GB VRAM)", "GPU:".dimmed(), name, vram);
        }
        _ if probe.nvidia_smi_available => {
            println!(
                "  {} NVIDIA GPU detected, but VRAM was not readable",
                "GPU:".dimmed()
            );
        }
        _ => println!(
            "  {} no NVIDIA GPU detected via nvidia-smi",
            "GPU:".dimmed()
        ),
    }
    println!(
        "  {} {}",
        "Ollama:".dimmed(),
        if probe.ollama_available {
            "available".green()
        } else {
            "not found".yellow()
        }
    );
    if !probe.ollama_models.is_empty() {
        println!(
            "  {} {}",
            "Pulled models:".dimmed(),
            probe.ollama_models.join(", ")
        );
    }
    println!(
        "  {} {}",
        "Tailscale:".dimmed(),
        if probe.tailscale_available {
            "available".green()
        } else {
            "not found".yellow()
        }
    );

    println!();
    println!(
        "  {} {}",
        "Tier:".dimmed(),
        recommendation.tier.green().bold()
    );
    println!(
        "  {} {} -> {}",
        "Primary:".dimmed(),
        recommendation.primary.cyan(),
        primary_model
    );
    println!(
        "  {} {} ({})",
        "Primary endpoint:".dimmed(),
        primary_route.base_url.cyan(),
        primary_route.source.dimmed()
    );
    println!(
        "  {} {} -> {}",
        "Coding:".dimmed(),
        recommendation.coding.cyan(),
        coding_model
    );
    println!(
        "  {} {} ({})",
        "Coding endpoint:".dimmed(),
        coding_route.base_url.cyan(),
        coding_route.source.dimmed()
    );
    println!(
        "  {} {}",
        "Fallback:".dimmed(),
        recommendation.fallback.join(", ").cyan()
    );
    println!(
        "  {} {}",
        "Benchmark:".dimmed(),
        recommendation.benchmark.join(", ").cyan()
    );
    println!("  {} {}", "Note:".dimmed(), recommendation.note);

    if !probe.ollama_available {
        println!();
        println!(
            "  {} Install/start Ollama, then run {}",
            "Next:".yellow().bold(),
            format!("ollama pull {primary_model}").cyan()
        );
    } else if primary_route.remote_workstation || coding_route.remote_workstation {
        println!();
        println!(
            "  {} Workstation routes stay private by default. Confirm {} is reachable on Tailscale before benchmarking.",
            "Tailnet:".yellow().bold(),
            "umbra-4090.tail649edd.ts.net".cyan()
        );
    } else {
        println!();
        println!(
            "  {} {}",
            "Next:".yellow().bold(),
            format!("ollama pull {primary_model}").cyan()
        );
    }
}

fn detect_total_ram_gb() -> Option<f32> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|value| value.parse::<f32>().ok())?;
                return Some(kb / 1024.0 / 1024.0);
            }
        }
        None
    }

    #[cfg(target_os = "macos")]
    {
        let output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let bytes = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<f32>()
            .ok()?;
        Some(bytes / 1024.0 / 1024.0 / 1024.0)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

fn detect_nvidia_gpu() -> (Option<String>, Option<f32>, bool) {
    let mut output = None;
    for command in ["nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"] {
        match Command::new(command)
            .args([
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            Ok(result) => {
                output = Some(result);
                break;
            }
            Err(_) => continue,
        }
    }

    let Some(output) = output else {
        return (None, None, false);
    };

    if !output.status.success() {
        return (None, None, true);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut best_name = None;
    let mut best_vram_gb = None;

    for line in stdout.lines() {
        let mut parts = line.split(',').map(str::trim);
        let name = match parts.next() {
            Some(name) if !name.is_empty() => name.to_string(),
            _ => continue,
        };
        let vram_gb = parts
            .next()
            .and_then(|mb| mb.parse::<f32>().ok())
            .map(|mb| mb / 1024.0);

        if vram_gb.unwrap_or(0.0) > best_vram_gb.unwrap_or(0.0) {
            best_name = Some(name);
            best_vram_gb = vram_gb;
        }
    }

    (best_name, best_vram_gb, true)
}

fn command_available(command: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };

    std::env::split_paths(&paths).any(|dir| {
        let path = dir.join(command);
        if path.is_file() {
            return true;
        }

        #[cfg(windows)]
        {
            let exe_path = dir.join(format!("{command}.exe"));
            if exe_path.is_file() {
                return true;
            }
        }

        false
    }) || Path::new(command).is_file()
}

fn detect_ollama_models() -> Vec<String> {
    let output = match Command::new("ollama").arg("list").output() {
        Ok(output) if output.status.success() => output,
        _ => return Vec::new(),
    };

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect()
}

#[derive(Debug, Clone)]
struct BenchmarkTask {
    name: &'static str,
    prompt: &'static str,
    max_tokens: u32,
}

const BENCHMARK_TASKS: &[BenchmarkTask] = &[
    BenchmarkTask {
        name: "short_instruction",
        prompt: "Reply with exactly one short sentence explaining why local models are useful.",
        max_tokens: 64,
    },
    BenchmarkTask {
        name: "code_generation",
        prompt: "Return only Rust code for a function named add_one that accepts i32 and returns x + 1.",
        max_tokens: 96,
    },
    BenchmarkTask {
        name: "code_review",
        prompt: r#"Review this Rust snippet for one bug or risk. Keep the answer under 80 words.

fn first(items: Vec<String>) -> String {
    items[0].clone()
}"#,
        max_tokens: 128,
    },
];

fn benchmark_profiles(cli: &Cli, recommendation: &LocalRecommendation) -> Vec<String> {
    let source = cli
        .benchmark_models
        .as_deref()
        .map(str::to_string)
        .unwrap_or_else(|| recommendation.benchmark.join(","));

    source
        .split(',')
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_string)
        .collect()
}

async fn run_local_benchmark(
    probe: &HardwareProbe,
    recommendation: &LocalRecommendation,
    profiles: Vec<String>,
) -> anyhow::Result<()> {
    println!(
        "{}",
        json!({
            "type": "local_benchmark_start",
            "hardware": {
                "os": probe.os,
                "arch": probe.arch,
                "total_ram_gb": probe.total_ram_gb,
                "gpu_name": probe.gpu_name,
                "gpu_vram_gb": probe.gpu_vram_gb,
                "nvidia_smi_available": probe.nvidia_smi_available,
                "ollama_available": probe.ollama_available,
                "ollama_models": probe.ollama_models,
                "tailscale_available": probe.tailscale_available,
            },
            "recommendation": {
                "tier": recommendation.tier,
                "primary": recommendation.primary,
                "coding": recommendation.coding,
                "fallback": recommendation.fallback,
                "benchmark": recommendation.benchmark,
            },
            "profiles": profiles,
            "tasks": BENCHMARK_TASKS.iter().map(|task| task.name).collect::<Vec<_>>(),
        })
    );

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;

    for profile in profiles {
        let (resolved_model, kind) = resolve_model(&profile);
        let route = ollama_route_for_model(&profile);
        let base_url = ollama_native_api_base_url(&route.base_url);
        if kind != chimera_sigil_providers::ProviderKind::Ollama {
            println!(
                "{}",
                json!({
                    "type": "local_benchmark_result",
                    "profile": profile,
                    "model": resolved_model,
                    "ok": false,
                    "error": "profile does not resolve to an Ollama model",
                })
            );
            continue;
        }

        let endpoint_models = fetch_ollama_model_names(&client, &base_url)
            .await
            .unwrap_or_else(|_| {
                if route.remote_workstation {
                    Vec::new()
                } else {
                    probe.ollama_models.clone()
                }
            });
        let installed = endpoint_models.iter().any(|model| model == resolved_model);

        for task in BENCHMARK_TASKS {
            let result = run_ollama_benchmark_task(
                &client,
                &base_url,
                &route,
                &profile,
                resolved_model,
                task,
            )
            .await;
            print_benchmark_result(&profile, resolved_model, &route, installed, task, result);
            let _ = io::stdout().flush();
        }
    }

    println!("{}", json!({ "type": "local_benchmark_complete" }));
    Ok(())
}

async fn run_ollama_benchmark_task(
    client: &reqwest::Client,
    base_url: &str,
    route: &chimera_sigil_providers::OllamaRoute,
    profile: &str,
    model: &str,
    task: &BenchmarkTask,
) -> anyhow::Result<serde_json::Value> {
    let started = Instant::now();
    let response = client
        .post(format!("{base_url}/api/generate"))
        .json(&json!({
            "model": model,
            "prompt": task.prompt,
            "stream": false,
            "think": false,
            "options": {
                "temperature": 0.0,
                "num_predict": task.max_tokens,
            }
        }))
        .send()
        .await
        .map_err(|e| benchmark_request_error(base_url, route, model, e))?;

    let elapsed_ms = started.elapsed().as_millis() as u64;
    let status = response.status();
    let body: serde_json::Value = response.json().await.unwrap_or_else(|_| json!({}));

    if !status.is_success() {
        anyhow::bail!(benchmark_status_error(
            base_url, route, model, status, &body
        ));
    }

    Ok(json!({
        "type": "local_benchmark_result",
        "profile": profile,
        "model": model,
        "base_url": route.base_url.as_str(),
        "base_url_source": route.source.as_str(),
        "remote_workstation": route.remote_workstation,
        "task": task.name,
        "ok": true,
        "wall_ms": elapsed_ms,
        "response_chars": body.get("response").and_then(|value| value.as_str()).map(str::len),
        "thinking_chars": body.get("thinking").and_then(|value| value.as_str()).map(str::len),
        "output_chars": ollama_output_chars(&body),
        "load_ms": duration_ns_to_ms(body.get("load_duration")),
        "prompt_eval_ms": duration_ns_to_ms(body.get("prompt_eval_duration")),
        "eval_ms": duration_ns_to_ms(body.get("eval_duration")),
        "total_ms": duration_ns_to_ms(body.get("total_duration")),
        "prompt_eval_count": body.get("prompt_eval_count").and_then(|value| value.as_u64()),
        "eval_count": body.get("eval_count").and_then(|value| value.as_u64()),
        "eval_tokens_per_second": tokens_per_second(
            body.get("eval_count").and_then(|value| value.as_u64()),
            body.get("eval_duration").and_then(|value| value.as_u64()),
        ),
        "done_reason": body.get("done_reason").and_then(|value| value.as_str()),
    }))
}

fn print_benchmark_result(
    profile: &str,
    model: &str,
    route: &chimera_sigil_providers::OllamaRoute,
    installed: bool,
    task: &BenchmarkTask,
    result: anyhow::Result<serde_json::Value>,
) {
    match result {
        Ok(mut value) => {
            if let Some(object) = value.as_object_mut() {
                object.insert("installed_at_start".to_string(), json!(installed));
            }
            println!("{value}");
        }
        Err(e) => {
            println!(
                "{}",
                json!({
                    "type": "local_benchmark_result",
                    "profile": profile,
                    "model": model,
                    "base_url": route.base_url.as_str(),
                    "base_url_source": route.source.as_str(),
                    "remote_workstation": route.remote_workstation,
                    "task": task.name,
                    "installed_at_start": installed,
                    "ok": false,
                    "error": e.to_string(),
                    "pull_hint": format!("ollama pull {model}"),
                })
            );
        }
    }
}

async fn fetch_ollama_model_names(
    client: &reqwest::Client,
    base_url: &str,
) -> anyhow::Result<Vec<String>> {
    let response = client.get(format!("{base_url}/api/tags")).send().await?;
    if !response.status().is_success() {
        return Ok(Vec::new());
    }

    let body: serde_json::Value = response.json().await?;
    Ok(body
        .get("models")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|model| model.get("name").and_then(|name| name.as_str()))
        .map(str::to_string)
        .collect())
}

fn benchmark_request_error(
    base_url: &str,
    route: &chimera_sigil_providers::OllamaRoute,
    model: &str,
    error: reqwest::Error,
) -> anyhow::Error {
    let mut message = format!(
        "Ollama benchmark endpoint {base_url} is not reachable for model '{model}': {error}."
    );
    if route.remote_workstation {
        message.push_str(" This profile is routed over the private Tailscale workstation lane; check `tailscale status`, DNS for umbra-4090.tail649edd.ts.net, and Ollama on Umbra port 11434.");
    } else {
        message.push_str(" Start Ollama with `ollama serve` or set OLLAMA_BASE_URL.");
    }
    anyhow::anyhow!(message)
}

fn benchmark_status_error(
    base_url: &str,
    route: &chimera_sigil_providers::OllamaRoute,
    model: &str,
    status: reqwest::StatusCode,
    body: &serde_json::Value,
) -> String {
    let rendered = body.to_string();
    let lower = rendered.to_ascii_lowercase();
    if status == reqwest::StatusCode::NOT_FOUND
        || (lower.contains("model") && lower.contains("not found"))
    {
        let mut message = format!(
            "Ollama model '{model}' is not available at {base_url}. Pull it with `ollama pull {model}` on that host."
        );
        if route.remote_workstation {
            message.push_str(" For the 4090 lane, run the pull on Umbra over Tailscale.");
        }
        return message;
    }

    format!("Ollama benchmark API error {status} at {base_url}: {body}")
}

fn duration_ns_to_ms(value: Option<&serde_json::Value>) -> Option<f64> {
    value
        .and_then(|value| value.as_u64())
        .map(|ns| ns as f64 / 1_000_000.0)
}

fn ollama_output_chars(body: &serde_json::Value) -> Option<usize> {
    let response_len = body
        .get("response")
        .and_then(|value| value.as_str())
        .map(str::len)
        .unwrap_or(0);
    let thinking_len = body
        .get("thinking")
        .and_then(|value| value.as_str())
        .map(str::len)
        .unwrap_or(0);
    let total = response_len + thinking_len;
    (total > 0).then_some(total)
}

fn tokens_per_second(count: Option<u64>, duration_ns: Option<u64>) -> Option<f64> {
    let count = count?;
    let duration_ns = duration_ns?;
    if duration_ns == 0 {
        return None;
    }
    Some(count as f64 / (duration_ns as f64 / 1_000_000_000.0))
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

fn print_local_setup_hint(model: &str, route: Option<&chimera_sigil_providers::OllamaRoute>) {
    eprintln!();
    eprintln!("{}", "Local setup:".yellow().bold());
    if let Some(route) = route {
        eprintln!(
            "  Endpoint: {} ({})",
            route.base_url.cyan(),
            route.source.dimmed()
        );
        if route.remote_workstation {
            eprintln!(
                "  1. Confirm Tailscale can reach {}",
                "umbra-4090.tail649edd.ts.net".cyan()
            );
            eprintln!(
                "  2. On Umbra, start Ollama and pull: {}",
                format!("ollama pull {model}").cyan()
            );
        } else {
            eprintln!("  1. Start Ollama: {}", "ollama serve".cyan());
            eprintln!(
                "  2. Pull this model: {}",
                format!("ollama pull {model}").cyan()
            );
        }
    } else {
        eprintln!("  1. Start Ollama: {}", "ollama serve".cyan());
        eprintln!(
            "  2. Pull this model: {}",
            format!("ollama pull {model}").cyan()
        );
    }
    eprintln!(
        "  3. Or try a smaller profile: {}",
        "chimera --model local-tiny".cyan()
    );
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

fn print_model_route(
    requested_model: &str,
    resolved_model: &str,
    route: Option<&chimera_sigil_providers::OllamaRoute>,
    json_mode: bool,
) {
    let Some(route) = route else {
        if json_mode {
            eprintln!("Route: non-local provider");
        } else {
            println!("Route: non-local provider");
        }
        return;
    };

    let line = format!(
        "Route: {requested_model} -> {resolved_model} via {} ({})",
        route.base_url, route.source
    );
    if json_mode {
        eprintln!("{line}");
    } else {
        println!("{line}");
    }
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
    outln!(
        "  {}     — Show provider route for the current model",
        "/route".yellow()
    );
    outln!(
        "  {}    — Run local hardware/Ollama diagnostics",
        "/doctor".yellow()
    );
    outln!("  {}     — Show token usage", "/usage".yellow());
    outln!("  {}      — Save session to disk", "/save".yellow());
    outln!("  {}  — List saved sessions", "/sessions".yellow());
    outln!("  {}      — Show this help", "/help".yellow());
    outln!();
    outln!("{}", "Free local profiles:".bold());
    outln!(
        "  {} -> {} — default laptop profile",
        "local".yellow(),
        DEFAULT_LOCAL_MODEL
    );
    outln!(
        "  {} -> llama3.2:1b — smallest useful profile",
        "local-tiny".yellow()
    );
    outln!(
        "  {} -> gemma3n:e2b — everyday-device profile",
        "local-edge".yellow()
    );
    outln!(
        "  {} -> qwen2.5-coder:3b — lightweight coding",
        "local-coder-small".yellow()
    );
    outln!(
        "  {} -> qwen2.5-coder:7b — coding on 16GB+ machines",
        "local-coder".yellow()
    );
    outln!(
        "  {} -> qwen3:8b — stronger general work",
        "local-balanced".yellow()
    );
    outln!(
        "  {} -> qwen3:14b — 12GB/16GB VRAM profile",
        "local-16gb".yellow()
    );
    outln!(
        "  {} -> qwen2.5-coder:14b — 12GB/16GB coding",
        "local-coder-16gb".yellow()
    );
    outln!(
        "  {} -> qwen3:30b — 24GB GPU generalist",
        "local-4090".yellow()
    );
    outln!(
        "  {} -> qwen2.5-coder:32b — 24GB GPU coding",
        "local-coder-4090".yellow()
    );
    outln!(
        "  {} -> qwen3:30b — local GPU/workstation alias",
        "local-workstation".yellow()
    );
    outln!();
    outln!("{}", "Supported models:".bold());
    outln!(
        "  {} — local, local-tiny, local-coder, or any Ollama model name",
        "Ollama (free local)".yellow()
    );
    outln!(
        "  {} — grok-3, grok-3-mini, grok-3-fast",
        "Grok (xAI)".yellow()
    );
    outln!("  {} — gpt-4o, gpt-4o-mini, o3, o4-mini", "OpenAI".yellow());
    outln!(
        "  {} — opus, sonnet, haiku (claude-opus-4-6, etc.)",
        "Anthropic".yellow()
    );
    outln!();
    outln!("{}", "Multi-Model:".bold());
    outln!(
        "  {} — Add advisor models that contribute parallel perspectives each turn",
        "--collab local-tiny,local-coder".yellow()
    );
    outln!(
        "  {} — Fallback chain if the primary provider fails",
        "--fallback local-small,local-tiny".yellow()
    );
    outln!();
    outln!("{}", "Environment variables:".bold());
    outln!(
        "  OLLAMA_BASE_URL                       — local Ollama URL (default: http://localhost:11434/v1)"
    );
    outln!(
        "  {} — exact profile override",
        profile_ollama_env_name("local-coder-4090")
    );
    outln!("  CLAWDAD_CHIMERA_4090_OLLAMA_BASE_URL — Clawdad/Umbra 4090 route");
    outln!("  CHIMERA_4090_OLLAMA_BASE_URL         — Chimera 4090 route");
    outln!("  XAI_API_KEY                          — xAI/Grok API key");
    outln!("  OPENAI_API_KEY                       — OpenAI API key");
    outln!("  ANTHROPIC_API_KEY                    — Anthropic/Claude API key");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(ram_gb: f32, vram_gb: Option<f32>) -> HardwareProbe {
        HardwareProbe {
            total_ram_gb: Some(ram_gb),
            gpu_vram_gb: vram_gb,
            ..HardwareProbe::default()
        }
    }

    #[test]
    fn recommends_4090_lane_for_24gb_vram() {
        let recommendation = recommend_local_profiles(&probe(64.0, Some(24.0)));
        assert_eq!(recommendation.primary, "local-4090");
        assert_eq!(recommendation.coding, "local-coder-4090");
        assert!(recommendation.fallback.contains(&"local-16gb"));
    }

    #[test]
    fn recommends_midrange_gpu_lane_for_16gb_vram() {
        let recommendation = recommend_local_profiles(&probe(32.0, Some(16.0)));
        assert_eq!(recommendation.primary, "local-16gb");
        assert_eq!(recommendation.coding, "local-coder-16gb");
    }

    #[test]
    fn recommends_budget_laptop_lane_for_16gb_ram_without_gpu() {
        let recommendation = recommend_local_profiles(&probe(16.0, None));
        assert_eq!(recommendation.primary, "local");
        assert_eq!(recommendation.coding, "local-coder-small");
    }

    #[test]
    fn recommends_tiny_lane_for_low_memory_without_gpu() {
        let recommendation = recommend_local_profiles(&probe(8.0, None));
        assert_eq!(recommendation.primary, "local-tiny");
        assert_eq!(recommendation.coding, "local-coder-small");
    }
}
