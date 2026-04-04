pub mod anthropic;
pub mod collab;
pub mod fallback;
pub mod grok;
pub mod openai;
pub mod provider;
pub mod stream;
pub mod types;

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
        s if s.starts_with("gpt")
            || s.starts_with("o1")
            || s.starts_with("o3")
            || s.starts_with("o4")
            || s.starts_with("codex") =>
        {
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
        ProviderKind::Anthropic => Box::new(anthropic::AnthropicProvider::new(config)),
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

/// Create a fallback provider chain from a comma-separated list of models.
/// The first model is primary; the rest are fallbacks tried in order.
/// Returns the provider and the primary model name.
pub fn create_fallback_provider(models: &str) -> anyhow::Result<(Box<dyn Provider>, String)> {
    let model_list: Vec<&str> = models.split(',').map(|s| s.trim()).collect();

    if model_list.len() < 2 {
        return create_provider(model_list[0]);
    }

    let mut providers: Vec<Box<dyn Provider>> = Vec::new();
    let mut primary_model = String::new();

    for (i, model) in model_list.iter().enumerate() {
        match create_provider(model) {
            Ok((provider, resolved)) => {
                if i == 0 {
                    primary_model = resolved;
                }
                providers.push(provider);
            }
            Err(e) => {
                if i == 0 {
                    return Err(e); // Primary provider must succeed
                }
                tracing::warn!("Fallback provider '{model}' unavailable: {e}");
            }
        }
    }

    Ok((
        Box::new(fallback::FallbackProvider::new(providers)),
        primary_model,
    ))
}

/// Wrap a primary provider with collaborating advisor models.
/// Collaborators do not replace the primary model; they contribute short
/// parallel perspectives that are merged into the primary model's context.
pub fn create_collaborative_provider(
    primary: Box<dyn Provider>,
    primary_model: String,
    collaborators: &str,
) -> anyhow::Result<(Box<dyn Provider>, Vec<String>)> {
    let model_list: Vec<&str> = collaborators
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let mut collaborator_providers = Vec::new();
    let mut collaborator_models = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for model in model_list {
        match create_provider(model) {
            Ok((provider, resolved)) => {
                if resolved == primary_model || !seen.insert(resolved.clone()) {
                    continue;
                }
                collaborator_models.push(resolved.clone());
                collaborator_providers.push((resolved, provider));
            }
            Err(e) => {
                tracing::warn!("Collaborator model '{model}' unavailable: {e}");
            }
        }
    }

    if collaborator_providers.is_empty() {
        anyhow::bail!("No collaborator models were available");
    }

    Ok((
        Box::new(collab::CollaborativeProvider::new(
            primary,
            primary_model,
            collaborator_providers,
        )),
        collaborator_models,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_grok_aliases() {
        assert_eq!(resolve_model("grok"), ("grok-3", ProviderKind::Grok));
        assert_eq!(resolve_model("grok-3"), ("grok-3", ProviderKind::Grok));
        assert_eq!(
            resolve_model("grok-mini"),
            ("grok-3-mini", ProviderKind::Grok)
        );
        assert_eq!(
            resolve_model("grok-3-fast"),
            ("grok-3-fast", ProviderKind::Grok)
        );
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
        assert_eq!(
            resolve_model("opus"),
            ("claude-opus-4-6", ProviderKind::Anthropic)
        );
        assert_eq!(
            resolve_model("sonnet"),
            ("claude-sonnet-4-6", ProviderKind::Anthropic)
        );
        assert_eq!(
            resolve_model("haiku"),
            ("claude-haiku-4-5-20251001", ProviderKind::Anthropic)
        );
    }

    #[test]
    fn test_resolve_ollama_local() {
        assert_eq!(
            resolve_model("llama3.2:latest"),
            ("llama3.2:latest", ProviderKind::Ollama)
        );
        assert_eq!(
            resolve_model("mistral/7b"),
            ("mistral/7b", ProviderKind::Ollama)
        );
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
