use crate::types::*;
use async_trait::async_trait;
use tokio::sync::mpsc;

/// Which provider backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Grok,
    OpenAi,
    Anthropic,
    Ollama,
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Grok => write!(f, "xAI (Grok)"),
            Self::OpenAi => write!(f, "OpenAI"),
            Self::Anthropic => write!(f, "Anthropic"),
            Self::Ollama => write!(f, "Ollama (local)"),
        }
    }
}

/// Configuration for a provider.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key: String,
    pub kind: ProviderKind,
}

impl ProviderConfig {
    /// Load provider config from environment variables.
    pub fn from_env(kind: ProviderKind) -> anyhow::Result<Self> {
        let (base_url, env_key) = match kind {
            ProviderKind::Grok => ("https://api.x.ai/v1", "XAI_API_KEY"),
            ProviderKind::OpenAi => ("https://api.openai.com/v1", "OPENAI_API_KEY"),
            ProviderKind::Anthropic => ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
            ProviderKind::Ollama => {
                return Ok(Self {
                    base_url: std::env::var("OLLAMA_BASE_URL")
                        .unwrap_or_else(|_| "http://localhost:11434/v1".into()),
                    api_key: "ollama".into(),
                    kind,
                });
            }
        };

        let api_key = std::env::var(env_key).map_err(|_| {
            anyhow::anyhow!(
                "Missing {env_key} environment variable. \
                 Set it to use {kind}."
            )
        })?;

        let base_url = std::env::var(format!("{}_BASE_URL", env_key.trim_end_matches("_API_KEY")))
            .unwrap_or_else(|_| base_url.to_string());

        Ok(Self {
            base_url,
            api_key,
            kind,
        })
    }
}

/// Trait for AI model providers. All providers implement streaming chat completions.
#[async_trait]
pub trait Provider: Send + Sync {
    /// The provider kind.
    fn kind(&self) -> ProviderKind;

    /// Send a chat request and stream events back through the channel.
    async fn chat_stream(
        &self,
        request: ChatRequest,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> anyhow::Result<()>;

    /// Send a chat request and wait for the full response (non-streaming convenience).
    async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse>;
}
