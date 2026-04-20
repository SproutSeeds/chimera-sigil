use crate::provider::{Provider, ProviderConfig, ProviderKind};
use crate::stream::parse_sse_stream;
use crate::types::*;
use async_trait::async_trait;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::debug;

/// OpenAI-compatible provider — works for OpenAI, Ollama, and any compatible API.
pub struct OpenAiProvider {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl OpenAiProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { config, client }
    }

    fn endpoint(&self) -> String {
        format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        )
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn kind(&self) -> ProviderKind {
        self.config.kind
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> anyhow::Result<()> {
        debug!("OpenAI-compat streaming request to {}", self.endpoint());

        let mut req = request;
        req.stream = true;

        let response = self
            .client
            .post(self.endpoint())
            .bearer_auth(&self.config.api_key)
            .json(&req)
            .send()
            .await
            .map_err(|e| self.request_error(&req.model, e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let message = self.status_error_message(&req.model, status, &body);
            let _ = tx.send(StreamEvent::Error(message.clone()));
            anyhow::bail!(message);
        }

        parse_sse_stream(response, tx).await?;
        Ok(())
    }

    async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let mut req = request;
        req.stream = false;

        let response = self
            .client
            .post(self.endpoint())
            .bearer_auth(&self.config.api_key)
            .json(&req)
            .send()
            .await
            .map_err(|e| self.request_error(&req.model, e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(self.status_error_message(&req.model, status, &body));
        }

        let body: serde_json::Value = response.json().await?;
        let choice = body["choices"]
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;

        let message = &choice["message"];
        let content = message["content"].as_str().map(String::from);

        let tool_calls = message
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|tc| serde_json::from_value::<ToolCall>(tc.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();

        let usage = body
            .get("usage")
            .and_then(|u| serde_json::from_value::<Usage>(u.clone()).ok());

        let finish_reason = choice["finish_reason"].as_str().map(String::from);

        Ok(ChatResponse {
            content,
            tool_calls,
            usage,
            finish_reason,
        })
    }
}

impl OpenAiProvider {
    fn request_error(&self, model: &str, error: reqwest::Error) -> anyhow::Error {
        if self.config.kind == ProviderKind::Ollama {
            return anyhow::anyhow!(
                "{}",
                ollama_request_error_message(&self.config.base_url, model, &error)
            );
        }

        anyhow::anyhow!(
            "{} request to {} failed: {}",
            self.config.kind,
            self.config.base_url,
            error
        )
    }

    fn status_error_message(&self, model: &str, status: reqwest::StatusCode, body: &str) -> String {
        if self.config.kind == ProviderKind::Ollama {
            return ollama_status_error_message(&self.config.base_url, model, status, body);
        }

        let body = compact_body(body);
        format!("{} API error {status}: {body}", self.config.kind)
    }
}

fn ollama_request_error_message(base_url: &str, model: &str, error: &reqwest::Error) -> String {
    let mut message =
        format!("Ollama is not reachable at {base_url} for model '{model}': {error}.");

    if is_tailnet_url(base_url) {
        message.push_str(
            " This is a private tailnet route; confirm Tailscale is connected, the Umbra host is reachable, and Ollama is listening on port 11434.",
        );
    } else {
        message.push_str(
            " Start Ollama with `ollama serve`, or set OLLAMA_BASE_URL to the reachable /v1 endpoint.",
        );
    }

    message
}

fn ollama_status_error_message(
    base_url: &str,
    model: &str,
    status: reqwest::StatusCode,
    body: &str,
) -> String {
    let body = compact_body(body);
    let lower = body.to_ascii_lowercase();

    if status == reqwest::StatusCode::NOT_FOUND
        || (lower.contains("model") && lower.contains("not found"))
    {
        let mut message = format!(
            "Ollama model '{model}' is not available at {base_url}. Pull it on that Ollama host with `ollama pull {model}`."
        );
        if is_tailnet_url(base_url) {
            message.push_str(
                " Because this is a tailnet workstation route, run the pull on Umbra or SSH into that host first.",
            );
        }
        return message;
    }

    format!("Ollama API error {status} at {base_url} for model '{model}': {body}")
}

fn compact_body(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.len() <= 1_000 {
        return trimmed.to_string();
    }

    let mut compacted = trimmed[..1_000].to_string();
    compacted.push_str("... (truncated)");
    compacted
}

fn is_tailnet_url(url: &str) -> bool {
    url.contains(".ts.net") || url.contains("tailscale") || url.contains("tailnet")
}
