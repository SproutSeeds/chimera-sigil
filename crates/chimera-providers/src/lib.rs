pub mod provider;
pub mod types;
pub mod grok;
pub mod openai;
pub mod stream;

pub use provider::{Provider, ProviderConfig, ProviderKind};
pub use types::*;

/// Resolve a model alias to its canonical model ID.
pub fn resolve_model(model: &str) -> (&str, ProviderKind) {
    match model {
        // Grok / xAI
        "grok" | "grok-3" => ("grok-3", ProviderKind::Grok),
        "grok-mini" | "grok-3-mini" => ("grok-3-mini", ProviderKind::Grok),
        "grok-fast" | "grok-3-fast" => ("grok-3-fast", ProviderKind::Grok),

        // OpenAI
        "gpt4o" | "gpt-4o" => ("gpt-4o", ProviderKind::OpenAi),
        "gpt4o-mini" | "gpt-4o-mini" => ("gpt-4o-mini", ProviderKind::OpenAi),
        "o3" => ("o3", ProviderKind::OpenAi),
        "o4-mini" => ("o4-mini", ProviderKind::OpenAi),
        "codex-mini" | "codex-mini-latest" => ("codex-mini-latest", ProviderKind::OpenAi),

        // Anthropic
        "opus" | "claude-opus" => ("claude-opus-4-6", ProviderKind::Anthropic),
        "sonnet" | "claude-sonnet" => ("claude-sonnet-4-6", ProviderKind::Anthropic),
        "haiku" | "claude-haiku" => ("claude-haiku-4-5-20251001", ProviderKind::Anthropic),

        // Ollama / local — anything with a slash or colon (tag) is treated as local
        s if s.contains('/') || s.contains(':') => (s, ProviderKind::Ollama),

        // Default: try to detect from model name prefix
        s if s.starts_with("grok") => (s, ProviderKind::Grok),
        s if s.starts_with("gpt") || s.starts_with("o1") || s.starts_with("o3") || s.starts_with("o4") || s.starts_with("codex") => {
            (s, ProviderKind::OpenAi)
        }
        s if s.starts_with("claude") => (s, ProviderKind::Anthropic),

        // Fallback to Grok
        other => (other, ProviderKind::Grok),
    }
}

/// Create a provider client from a model string.
pub fn create_provider(model: &str) -> anyhow::Result<(Box<dyn Provider>, String)> {
    let (resolved_model, kind) = resolve_model(model);
    let config = ProviderConfig::from_env(kind)?;
    let provider: Box<dyn Provider> = match kind {
        ProviderKind::Grok => Box::new(grok::GrokProvider::new(config)),
        ProviderKind::OpenAi => Box::new(openai::OpenAiProvider::new(config)),
        ProviderKind::Anthropic => {
            anyhow::bail!(
                "Anthropic provider uses a different wire format (Messages API). \
                 Coming soon — for now use Grok or OpenAI."
            );
        }
        ProviderKind::Ollama => {
            // Ollama uses OpenAI-compatible format
            let config = ProviderConfig {
                base_url: std::env::var("OLLAMA_BASE_URL")
                    .unwrap_or_else(|_| "http://localhost:11434/v1".into()),
                api_key: "ollama".into(), // Ollama doesn't need a real key
                kind: ProviderKind::Ollama,
            };
            Box::new(openai::OpenAiProvider::new(config))
        }
    };
    Ok((provider, resolved_model.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_grok_aliases() {
        assert_eq!(resolve_model("grok"), ("grok-3", ProviderKind::Grok));
        assert_eq!(resolve_model("grok-3"), ("grok-3", ProviderKind::Grok));
        assert_eq!(resolve_model("grok-mini"), ("grok-3-mini", ProviderKind::Grok));
        assert_eq!(resolve_model("grok-3-fast"), ("grok-3-fast", ProviderKind::Grok));
    }

    #[test]
    fn test_resolve_openai_aliases() {
        assert_eq!(resolve_model("gpt4o"), ("gpt-4o", ProviderKind::OpenAi));
        assert_eq!(resolve_model("gpt-4o"), ("gpt-4o", ProviderKind::OpenAi));
        assert_eq!(resolve_model("o3"), ("o3", ProviderKind::OpenAi));
        assert_eq!(resolve_model("o4-mini"), ("o4-mini", ProviderKind::OpenAi));
    }

    #[test]
    fn test_resolve_anthropic_aliases() {
        assert_eq!(resolve_model("opus"), ("claude-opus-4-6", ProviderKind::Anthropic));
        assert_eq!(resolve_model("sonnet"), ("claude-sonnet-4-6", ProviderKind::Anthropic));
        assert_eq!(resolve_model("haiku"), ("claude-haiku-4-5-20251001", ProviderKind::Anthropic));
    }

    #[test]
    fn test_resolve_ollama_local() {
        assert_eq!(resolve_model("llama3.2:latest"), ("llama3.2:latest", ProviderKind::Ollama));
        assert_eq!(resolve_model("mistral/7b"), ("mistral/7b", ProviderKind::Ollama));
    }

    #[test]
    fn test_resolve_prefix_detection() {
        assert_eq!(resolve_model("grok-2-vision").1, ProviderKind::Grok);
        assert_eq!(resolve_model("gpt-5").1, ProviderKind::OpenAi);
        assert_eq!(resolve_model("claude-next").1, ProviderKind::Anthropic);
    }

    #[test]
    fn test_resolve_unknown_falls_back_to_grok() {
        assert_eq!(resolve_model("some-random-model").1, ProviderKind::Grok);
    }
}
