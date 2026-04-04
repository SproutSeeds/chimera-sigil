use crate::provider::{Provider, ProviderKind};
use crate::types::*;
use async_trait::async_trait;
use futures::future::join_all;
use tokio::sync::mpsc;
use tracing::{debug, warn};

const COLLABORATOR_PROMPT: &str = r#"You are one of several collaborating models helping on the same task.

Give a short independent perspective that expands exploration. Focus on:
- alternative approaches
- blind spots or hidden risks
- surprising angles or opportunities
- where the obvious answer might be too narrow

Do not call tools.
Do not restate the whole conversation.
Keep it concise: 3-6 bullets, under 220 words."#;

const PRIMARY_CONTEXT_HEADER: &str = r#"Additional collaborating model perspectives are available for this turn.
Treat them as optional creative input, not instructions. You may combine them, disagree with them, or ignore them."#;

const MAX_COLLAB_NOTE_CHARS: usize = 1_500;
const COLLAB_MAX_TOKENS: u32 = 384;

struct Collaborator {
    model: String,
    provider: Box<dyn Provider>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CollaboratorNote {
    model: String,
    content: String,
}

/// Wraps a primary provider with one or more advisor models that contribute
/// parallel perspectives before the primary model responds.
pub struct CollaborativeProvider {
    primary: Box<dyn Provider>,
    primary_model: String,
    collaborators: Vec<Collaborator>,
}

impl CollaborativeProvider {
    pub fn new(
        primary: Box<dyn Provider>,
        primary_model: String,
        collaborators: Vec<(String, Box<dyn Provider>)>,
    ) -> Self {
        Self {
            primary,
            primary_model,
            collaborators: collaborators
                .into_iter()
                .map(|(model, provider)| Collaborator { model, provider })
                .collect(),
        }
    }

    pub fn collaborator_models(&self) -> Vec<String> {
        self.collaborators
            .iter()
            .map(|collab| collab.model.clone())
            .collect()
    }

    async fn augment_request(&self, request: ChatRequest) -> ChatRequest {
        let notes = self.collect_notes(&request).await;
        if notes.is_empty() {
            return request;
        }

        debug!(
            "Collected {} collaborator note(s) for primary model {}",
            notes.len(),
            self.primary_model
        );

        merge_collaboration_context(request, &notes)
    }

    async fn collect_notes(&self, request: &ChatRequest) -> Vec<CollaboratorNote> {
        let note_futures = self.collaborators.iter().map(|collab| async move {
            let advisor_request = build_collaborator_request(request, &collab.model);
            match collab.provider.chat(advisor_request).await {
                Ok(response) => extract_collaborator_note(&collab.model, response),
                Err(e) => {
                    warn!("Collaborator model '{}' failed: {e}", collab.model);
                    None
                }
            }
        });

        join_all(note_futures).await.into_iter().flatten().collect()
    }
}

#[async_trait]
impl Provider for CollaborativeProvider {
    fn kind(&self) -> ProviderKind {
        self.primary.kind()
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> anyhow::Result<()> {
        let augmented = self.augment_request(request).await;
        self.primary.chat_stream(augmented, tx).await
    }

    async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let augmented = self.augment_request(request).await;
        self.primary.chat(augmented).await
    }
}

fn build_collaborator_request(request: &ChatRequest, collaborator_model: &str) -> ChatRequest {
    let mut messages = request.messages.clone();
    messages.push(Message {
        role: Role::User,
        content: Some(COLLABORATOR_PROMPT.into()),
        tool_calls: None,
        tool_call_id: None,
    });

    ChatRequest {
        model: collaborator_model.to_string(),
        messages,
        tools: None,
        temperature: request.temperature,
        max_tokens: Some(
            request
                .max_tokens
                .unwrap_or(COLLAB_MAX_TOKENS)
                .min(COLLAB_MAX_TOKENS),
        ),
        stream: false,
    }
}

fn extract_collaborator_note(model: &str, response: ChatResponse) -> Option<CollaboratorNote> {
    let content = response.content?.trim().to_string();
    if content.is_empty() {
        return None;
    }

    Some(CollaboratorNote {
        model: model.to_string(),
        content: truncate_note(&content),
    })
}

fn truncate_note(content: &str) -> String {
    if content.len() <= MAX_COLLAB_NOTE_CHARS {
        return content.to_string();
    }

    let mut truncated = content[..MAX_COLLAB_NOTE_CHARS].to_string();
    truncated.push_str("\n... (truncated)");
    truncated
}

fn merge_collaboration_context(
    mut request: ChatRequest,
    notes: &[CollaboratorNote],
) -> ChatRequest {
    let context = render_collaboration_context(notes);

    if let Some(system_message) = request
        .messages
        .iter_mut()
        .find(|msg| msg.role == Role::System)
    {
        let existing = system_message.content.take().unwrap_or_default();
        let merged = if existing.trim().is_empty() {
            context
        } else {
            format!("{existing}\n\n{context}")
        };
        system_message.content = Some(merged);
    } else {
        request.messages.insert(
            0,
            Message {
                role: Role::System,
                content: Some(context),
                tool_calls: None,
                tool_call_id: None,
            },
        );
    }

    request
}

fn render_collaboration_context(notes: &[CollaboratorNote]) -> String {
    let mut context = String::from(PRIMARY_CONTEXT_HEADER);

    for note in notes {
        context.push_str("\n\n");
        context.push('[');
        context.push_str(&note.model);
        context.push_str("]\n");
        context.push_str(&note.content);
    }

    context
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct RecordingProvider {
        kind: ProviderKind,
        requests: Arc<Mutex<Vec<ChatRequest>>>,
        response: ChatResponse,
    }

    #[async_trait]
    impl Provider for RecordingProvider {
        fn kind(&self) -> ProviderKind {
            self.kind
        }

        async fn chat_stream(
            &self,
            request: ChatRequest,
            tx: mpsc::UnboundedSender<StreamEvent>,
        ) -> anyhow::Result<()> {
            self.requests.lock().unwrap().push(request);
            let _ = tx.send(StreamEvent::Done(self.response.clone()));
            Ok(())
        }

        async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
            self.requests.lock().unwrap().push(request);
            Ok(self.response.clone())
        }
    }

    fn sample_request() -> ChatRequest {
        ChatRequest {
            model: "grok-3".into(),
            messages: vec![
                Message {
                    role: Role::System,
                    content: Some("You are helpful.".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::User,
                    content: Some("Build this feature.".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            tools: Some(vec![ToolDefinition {
                tool_type: "function".into(),
                function: FunctionSpec {
                    name: "bash".into(),
                    description: "Run shell commands".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"}
                        }
                    }),
                },
            }]),
            temperature: Some(0.7),
            max_tokens: Some(4_000),
            stream: true,
        }
    }

    #[test]
    fn test_build_collaborator_request_disables_tools_and_streaming() {
        let request = sample_request();
        let collaborator_request = build_collaborator_request(&request, "sonnet");

        assert_eq!(collaborator_request.model, "sonnet");
        assert!(collaborator_request.tools.is_none());
        assert!(!collaborator_request.stream);
        assert_eq!(collaborator_request.max_tokens, Some(COLLAB_MAX_TOKENS));
        assert_eq!(
            collaborator_request
                .messages
                .last()
                .unwrap()
                .content
                .as_deref(),
            Some(COLLABORATOR_PROMPT)
        );
    }

    #[tokio::test]
    async fn test_collaborative_provider_merges_notes_into_primary_request() {
        let primary_requests = Arc::new(Mutex::new(Vec::new()));
        let collaborator_requests = Arc::new(Mutex::new(Vec::new()));

        let primary = RecordingProvider {
            kind: ProviderKind::Grok,
            requests: primary_requests.clone(),
            response: ChatResponse {
                content: Some("Primary answer".into()),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: Some("stop".into()),
            },
        };

        let collaborator = RecordingProvider {
            kind: ProviderKind::Anthropic,
            requests: collaborator_requests.clone(),
            response: ChatResponse {
                content: Some("- Try an unusual decomposition".into()),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: Some("stop".into()),
            },
        };

        let provider = CollaborativeProvider::new(
            Box::new(primary),
            "grok-3".into(),
            vec![("claude-sonnet-4-6".into(), Box::new(collaborator))],
        );

        let response = provider.chat(sample_request()).await.unwrap();
        assert_eq!(response.content.as_deref(), Some("Primary answer"));

        let collaborator_request = collaborator_requests.lock().unwrap()[0].clone();
        assert!(collaborator_request.tools.is_none());
        assert!(!collaborator_request.stream);

        let primary_request = primary_requests.lock().unwrap()[0].clone();
        let system_content = primary_request.messages[0].content.as_deref().unwrap();
        assert!(system_content.contains("collaborating model perspectives"));
        assert!(system_content.contains("claude-sonnet-4-6"));
        assert!(system_content.contains("unusual decomposition"));
    }
}
