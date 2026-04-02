use chimera_core::{Agent, Config};
use chimera_core::agent::{AgentEvent, EventCallback};
use chimera_providers::create_provider;
use clap::Parser;
use colored::*;
use std::io::{self, Write};

#[derive(Parser)]
#[command(
    name = "chimera",
    about = "Multi-model AI agent orchestrator — Grok, OpenAI, Ollama, and more",
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

    /// Auto-approve all tool executions (dangerous)
    #[arg(long, default_value_t = false)]
    auto_approve: bool,

    /// Enable verbose logging
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Set up tracing
    let filter = if cli.verbose { "debug" } else { "warn" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| filter.into()),
        )
        .with_target(false)
        .init();

    // Run the async runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(cli))
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    // Create the provider
    let (provider, model) = match create_provider(&cli.model) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{} {e}", "Error:".red().bold());
            eprintln!(
                "\n{} Set the appropriate API key environment variable:",
                "Hint:".yellow()
            );
            eprintln!("  Grok:   export XAI_API_KEY=your-key");
            eprintln!("  OpenAI: export OPENAI_API_KEY=your-key");
            eprintln!("  Ollama: No key needed (uses http://localhost:11434)");
            std::process::exit(1);
        }
    };

    let config = Config {
        model: model.clone(),
        temperature: cli.temperature,
        auto_approve: cli.auto_approve,
        ..Config::default()
    };

    let provider_kind = provider.kind();
    let mut agent = Agent::new(provider, model.clone(), config);

    // Single prompt mode
    if let Some(prompt) = cli.prompt {
        let callback = make_callback();
        agent.run_turn(&prompt, &callback).await?;
        println!();
        return Ok(());
    }

    // Interactive REPL
    print_banner(&model, provider_kind);

    loop {
        // Print prompt
        print!("\n{} ", "chimera>".cyan().bold());
        io::stdout().flush()?;

        // Read input
        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input)?;
        if bytes_read == 0 {
            // EOF
            println!("\n{}", "Goodbye!".dimmed());
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle special commands
        match input {
            "/quit" | "/exit" | "/q" => {
                println!("{}", "Goodbye!".dimmed());
                break;
            }
            "/model" => {
                println!("Current model: {}", model.green());
                continue;
            }
            "/usage" => {
                let session = agent.session();
                println!(
                    "Tokens — input: {}, output: {}, total: {}",
                    session.total_input_tokens.to_string().yellow(),
                    session.total_output_tokens.to_string().yellow(),
                    session.total_tokens().to_string().green().bold(),
                );
                continue;
            }
            "/help" => {
                print_help();
                continue;
            }
            _ => {}
        }

        println!(); // Visual separator before response

        let callback = make_callback();
        match agent.run_turn(input, &callback).await {
            Ok(_) => {
                println!(); // newline after response
            }
            Err(e) => {
                eprintln!("\n{} {e}", "Error:".red().bold());
            }
        }
    }

    Ok(())
}

fn make_callback() -> EventCallback {
    Box::new(|event| match event {
        AgentEvent::TextDelta(text) => {
            print!("{text}");
            let _ = io::stdout().flush();
        }
        AgentEvent::ToolStart { name, arguments } => {
            let preview = if arguments.len() > 120 {
                format!("{}...", &arguments[..120])
            } else {
                arguments.clone()
            };
            eprintln!(
                "\n{} {} {}",
                "  tool>".dimmed(),
                name.yellow().bold(),
                preview.dimmed()
            );
        }
        AgentEvent::ToolResult {
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
                "{} {} [{}] {}",
                "  tool>".dimmed(),
                name.dimmed(),
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
                eprintln!(
                    "{}",
                    format!("  [{iterations} iterations]").dimmed()
                );
            }
        }
        AgentEvent::Error(e) => {
            eprintln!("{} {e}", "  Error:".red().bold());
        }
    })
}

fn print_banner(model: &str, provider: chimera_providers::ProviderKind) {
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

    println!(
        "  {} {} via {}",
        "Model:".dimmed(),
        model.green().bold(),
        provider.to_string().yellow()
    );
    println!(
        "  {}",
        "Type /help for commands, /quit to exit".dimmed()
    );
}

fn print_help() {
    println!("{}", "Commands:".bold());
    println!("  {}  — Exit the REPL", "/quit".yellow());
    println!("  {}  — Show current model", "/model".yellow());
    println!("  {}  — Show token usage", "/usage".yellow());
    println!("  {}  — Show this help", "/help".yellow());
    println!();
    println!("{}", "Supported models:".bold());
    println!("  {} — grok-3, grok-3-mini, grok-3-fast", "Grok (xAI)".yellow());
    println!("  {} — gpt-4o, gpt-4o-mini, o3, o4-mini", "OpenAI".yellow());
    println!("  {} — Any model via http://localhost:11434", "Ollama".yellow());
    println!();
    println!("{}", "Environment variables:".bold());
    println!("  XAI_API_KEY     — xAI/Grok API key");
    println!("  OPENAI_API_KEY  — OpenAI API key");
    println!("  OLLAMA_BASE_URL — Ollama server URL (default: http://localhost:11434/v1)");
}
