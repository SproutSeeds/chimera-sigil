use crate::types::*;
use reqwest::Response;
use tokio::sync::mpsc;
use tracing::debug;

/// Parse an SSE stream from an OpenAI-compatible chat completions endpoint.
/// Assembles tool calls across deltas and emits events through the channel.
pub async fn parse_sse_stream(
    response: Response,
    tx: mpsc::UnboundedSender<StreamEvent>,
) -> anyhow::Result<ChatResponse> {
    let mut content = String::new();
    let mut tool_calls: Vec<ToolCallBuilder> = Vec::new();
    let mut usage = None;
    let mut finish_reason = None;

    let body = response.text().await?;

    for line in body.lines() {
        let line = line.trim();

        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        let data = match line.strip_prefix("data: ") {
            Some(d) => d.trim(),
            None => continue,
        };

        if data == "[DONE]" {
            break;
        }

        let chunk: StreamChunk = match serde_json::from_str(data) {
            Ok(c) => c,
            Err(e) => {
                debug!("Failed to parse SSE chunk: {e} — data: {data}");
                continue;
            }
        };

        if let Some(u) = chunk.usage {
            usage = Some(u);
        }

        for choice in &chunk.choices {
            if let Some(reason) = &choice.finish_reason {
                finish_reason = Some(reason.clone());
            }

            let delta = &choice.delta;

            // Content delta
            if let Some(text) = &delta.content {
                content.push_str(text);
                let _ = tx.send(StreamEvent::ContentDelta(text.clone()));
            }

            // Tool call deltas
            if let Some(tc_deltas) = &delta.tool_calls {
                for tc_delta in tc_deltas {
                    let idx = tc_delta.index as usize;

                    // Grow the builder vec if needed
                    while tool_calls.len() <= idx {
                        tool_calls.push(ToolCallBuilder::default());
                    }

                    let builder = &mut tool_calls[idx];

                    if let Some(id) = &tc_delta.id {
                        builder.id = Some(id.clone());
                    }
                    if let Some(func) = &tc_delta.function {
                        if let Some(name) = &func.name {
                            builder.name = Some(name.clone());
                        }
                        if let Some(args) = &func.arguments {
                            builder.arguments.push_str(args);
                        }
                    }

                    let _ = tx.send(StreamEvent::ToolCallDelta {
                        index: tc_delta.index,
                        id: tc_delta.id.clone(),
                        name: tc_delta.function.as_ref().and_then(|f| f.name.clone()),
                        arguments_delta: tc_delta
                            .function
                            .as_ref()
                            .and_then(|f| f.arguments.clone()),
                    });
                }
            }
        }
    }

    let assembled_tool_calls: Vec<ToolCall> = tool_calls
        .into_iter()
        .filter_map(|b| b.build())
        .collect();

    let response = ChatResponse {
        content: if content.is_empty() {
            None
        } else {
            Some(content)
        },
        tool_calls: assembled_tool_calls,
        usage,
        finish_reason,
    };

    let _ = tx.send(StreamEvent::Done(response.clone()));
    Ok(response)
}

/// Incrementally builds a ToolCall from streaming deltas.
#[derive(Default)]
struct ToolCallBuilder {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl ToolCallBuilder {
    fn build(self) -> Option<ToolCall> {
        Some(ToolCall {
            id: self.id?,
            call_type: "function".into(),
            function: FunctionCall {
                name: self.name?,
                arguments: self.arguments,
            },
        })
    }
}
