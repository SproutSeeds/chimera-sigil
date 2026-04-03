use crate::provider::{Provider, ProviderConfig, ProviderKind};
use crate::stream::parse_sse_stream;
use crate::types::*;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::debug;

/// OpenAI-compatible provider — works for OpenAI, Ollama, and any compatible API.
pub struct OpenAiProvider {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl OpenAiProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let client = reqwest::Client::new();
        Self { config, client }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.config.base_url)
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
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let _ = tx.send(StreamEvent::Error(format!(
                "API error {status}: {body}"
            )));
            anyhow::bail!("API error {status}: {body}");
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
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("API error {status}: {body}");
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
