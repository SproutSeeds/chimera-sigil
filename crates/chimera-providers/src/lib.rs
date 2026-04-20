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

/// Default no-cost local model used by the CLI.
pub const DEFAULT_LOCAL_MODEL: &str = "qwen3:4b";
/// Default local Ollama OpenAI-compatible endpoint.
pub const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";
/// Private tailnet default for 24GB workstation profiles.
pub const DEFAULT_WORKSTATION_OLLAMA_BASE_URL: &str =
    "http://umbra-4090.tail649edd.ts.net:11434/v1";

/// Resolved Ollama endpoint route for a local profile/model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OllamaRoute {
    pub base_url: String,
    pub source: String,
    pub remote_workstation: bool,
}

/// Resolve a model alias to its canonical model ID.
pub fn resolve_model(model: &str) -> (&str, ProviderKind) {
    match model {
        // Local / Ollama profiles. Keep these aliases small enough that a user
        // on a new budget laptop can get useful work done without cloud keys.
        "local" | "local-default" | "local-small" | "local-laptop" => {
            (DEFAULT_LOCAL_MODEL, ProviderKind::Ollama)
        }
        "local-tiny" => ("llama3.2:1b", ProviderKind::Ollama),
        "local-edge" => ("gemma3n:e2b", ProviderKind::Ollama),
        "local-coder-small" => ("qwen2.5-coder:3b", ProviderKind::Ollama),
        "local-code" | "local-coder" => ("qwen2.5-coder:7b", ProviderKind::Ollama),
        "local-12gb" | "local-16gb" | "local-heavy" => ("qwen3:14b", ProviderKind::Ollama),
        "local-coder-12gb" | "local-coder-16gb" | "local-coder-heavy" => {
            ("qwen2.5-coder:14b", ProviderKind::Ollama)
        }
        "local-balanced" => ("qwen3:8b", ProviderKind::Ollama),
        "local-reasoning" => ("deepseek-r1:8b", ProviderKind::Ollama),
        "local-24gb" | "local-4090" | "local-gpu" | "local-workstation" => {
            ("qwen3:30b", ProviderKind::Ollama)
        }
        "local-coder-24gb" | "local-coder-4090" | "local-coder-gpu" => {
            ("qwen2.5-coder:32b", ProviderKind::Ollama)
        }

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

        // Ollama / local - anything with a slash or colon (tag) is treated as local.
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

        // Local-first fallback: Ollama catalog model names often have no
        // provider prefix (for example, qwen3 or gemma3n).
        other => (other, ProviderKind::Ollama),
    }
}

/// Whether this profile is intended for the private 24GB GPU workstation lane.
pub fn is_workstation_profile(model: &str) -> bool {
    matches!(
        model,
        "local-24gb"
            | "local-4090"
            | "local-gpu"
            | "local-workstation"
            | "local-coder-24gb"
            | "local-coder-4090"
            | "local-coder-gpu"
    )
}

/// Resolve the OpenAI-compatible Ollama endpoint for a requested local model/profile.
pub fn ollama_route_for_model(requested_model: &str) -> OllamaRoute {
    let remote_workstation = is_workstation_profile(requested_model);
    let profile_env = profile_ollama_env_name(requested_model);

    let env_candidates = if remote_workstation {
        vec![
            profile_env,
            "CLAWDAD_CHIMERA_4090_OLLAMA_BASE_URL".to_string(),
            "CHIMERA_4090_OLLAMA_BASE_URL".to_string(),
            "CHIMERA_WORKSTATION_OLLAMA_BASE_URL".to_string(),
            "OLLAMA_BASE_URL".to_string(),
        ]
    } else {
        vec![profile_env, "OLLAMA_BASE_URL".to_string()]
    };

    for env_name in env_candidates {
        if let Ok(value) = std::env::var(&env_name)
            && !value.trim().is_empty()
        {
            return OllamaRoute {
                base_url: normalize_ollama_openai_base_url(&value),
                source: env_name,
                remote_workstation,
            };
        }
    }

    if let Ok(value) = std::env::var("OLLAMA_HOST")
        && !value.trim().is_empty()
    {
        return OllamaRoute {
            base_url: normalize_ollama_openai_base_url(&value),
            source: "OLLAMA_HOST".into(),
            remote_workstation,
        };
    }

    if remote_workstation {
        OllamaRoute {
            base_url: DEFAULT_WORKSTATION_OLLAMA_BASE_URL.into(),
            source: "tailnet default".into(),
            remote_workstation,
        }
    } else {
        OllamaRoute {
            base_url: DEFAULT_OLLAMA_BASE_URL.into(),
            source: "default".into(),
            remote_workstation,
        }
    }
}

/// Convert a profile name into its profile-specific Ollama endpoint variable.
pub fn profile_ollama_env_name(model: &str) -> String {
    let mut env = String::from("CHIMERA_");
    for ch in model.chars() {
        if ch.is_ascii_alphanumeric() {
            env.push(ch.to_ascii_uppercase());
        } else {
            env.push('_');
        }
    }
    env.push_str("_OLLAMA_BASE_URL");
    env
}

/// Normalize an Ollama endpoint into the OpenAI-compatible /v1 base URL.
pub fn normalize_ollama_openai_base_url(raw: &str) -> String {
    let trimmed = raw.trim().trim_end_matches('/');
    let with_scheme = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };

    if with_scheme.ends_with("/v1") {
        with_scheme
    } else {
        format!("{with_scheme}/v1")
    }
}

/// Convert an OpenAI-compatible Ollama base URL to the native Ollama API root.
pub fn ollama_native_api_base_url(openai_base_url: &str) -> String {
    openai_base_url
        .trim()
        .trim_end_matches('/')
        .strip_suffix("/v1")
        .unwrap_or_else(|| openai_base_url.trim().trim_end_matches('/'))
        .to_string()
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
            let route = ollama_route_for_model(model);
            // Ollama uses OpenAI-compatible format
            let config = ProviderConfig {
                base_url: route.base_url,
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
    fn test_resolve_local_aliases() {
        assert_eq!(
            resolve_model("local"),
            (DEFAULT_LOCAL_MODEL, ProviderKind::Ollama)
        );
        assert_eq!(
            resolve_model("local-tiny"),
            ("llama3.2:1b", ProviderKind::Ollama)
        );
        assert_eq!(
            resolve_model("local-coder"),
            ("qwen2.5-coder:7b", ProviderKind::Ollama)
        );
        assert_eq!(
            resolve_model("local-balanced"),
            ("qwen3:8b", ProviderKind::Ollama)
        );
        assert_eq!(
            resolve_model("local-coder-16gb"),
            ("qwen2.5-coder:14b", ProviderKind::Ollama)
        );
        assert_eq!(
            resolve_model("local-4090"),
            ("qwen3:30b", ProviderKind::Ollama)
        );
        assert_eq!(
            resolve_model("local-coder-4090"),
            ("qwen2.5-coder:32b", ProviderKind::Ollama)
        );
    }

    #[test]
    fn test_workstation_profile_routes_to_tailnet_default() {
        if std::env::var("CHIMERA_LOCAL_CODER_4090_OLLAMA_BASE_URL").is_ok()
            || std::env::var("CLAWDAD_CHIMERA_4090_OLLAMA_BASE_URL").is_ok()
            || std::env::var("CHIMERA_4090_OLLAMA_BASE_URL").is_ok()
            || std::env::var("CHIMERA_WORKSTATION_OLLAMA_BASE_URL").is_ok()
            || std::env::var("OLLAMA_BASE_URL").is_ok()
            || std::env::var("OLLAMA_HOST").is_ok()
        {
            return;
        }
        let route = ollama_route_for_model("local-coder-4090");
        assert_eq!(route.base_url, DEFAULT_WORKSTATION_OLLAMA_BASE_URL);
        assert!(route.remote_workstation);
    }

    #[test]
    fn test_local_profile_routes_to_localhost_default() {
        if std::env::var("CHIMERA_LOCAL_OLLAMA_BASE_URL").is_ok()
            || std::env::var("OLLAMA_BASE_URL").is_ok()
            || std::env::var("OLLAMA_HOST").is_ok()
        {
            return;
        }
        let route = ollama_route_for_model("local");
        assert_eq!(route.base_url, DEFAULT_OLLAMA_BASE_URL);
        assert!(!route.remote_workstation);
    }

    #[test]
    fn test_profile_env_name_is_stable() {
        assert_eq!(
            profile_ollama_env_name("local-coder-4090"),
            "CHIMERA_LOCAL_CODER_4090_OLLAMA_BASE_URL"
        );
    }

    #[test]
    fn test_ollama_url_normalization() {
        assert_eq!(
            normalize_ollama_openai_base_url("127.0.0.1:11434"),
            "http://127.0.0.1:11434/v1"
        );
        assert_eq!(
            ollama_native_api_base_url("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434"
        );
    }

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
    fn test_resolve_unknown_falls_back_to_ollama() {
        assert_eq!(resolve_model("some-random-model").1, ProviderKind::Ollama);
    }
}
