use crate::provider::{Provider, ProviderConfig, ProviderKind};
use crate::stream::for_each_sse_line;
use crate::types::*;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::debug;

const ANTHROPIC_API_VERSION: &str = "2023-06-01";

/// Anthropic Messages API provider.
pub struct AnthropicProvider {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let client = reqwest::Client::new();
        Self { config, client }
    }

    fn endpoint(&self) -> String {
        format!("{}/messages", self.config.base_url)
    }
}

// ─── Wire format types ───────────────────────────────────────────────────────

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    stream: bool,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Serialize, Clone)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

// ─── SSE response types ─────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct SseEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    message: Option<SseMessage>,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    content_block: Option<SseContentBlock>,
    #[serde(default)]
    delta: Option<SseDelta>,
    #[serde(default)]
    usage: Option<SseUsage>,
}

#[derive(Deserialize, Debug)]
struct SseMessage {
    #[serde(default)]
    usage: Option<SseUsage>,
}

#[derive(Deserialize, Debug, Clone)]
struct SseContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    /// Present on text content blocks; unused here since we get text via deltas.
    #[serde(default)]
    #[allow(dead_code)]
    text: Option<String>,
}

#[derive(Deserialize, Debug)]
struct SseDelta {
    #[serde(rename = "type")]
    delta_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
struct SseUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

// ─── Conversion: our types → Anthropic wire format ──────────────────────────

fn convert_request(request: &ChatRequest) -> AnthropicRequest {
    let mut system = None;
    let mut messages = Vec::new();

    for msg in &request.messages {
        match msg.role {
            Role::System => {
                system = msg.content.clone();
            }
            Role::User => {
                if let Some(text) = &msg.content {
                    messages.push(AnthropicMessage {
                        role: "user".into(),
                        content: AnthropicContent::Text(text.clone()),
                    });
                }
            }
            Role::Assistant => {
                let mut blocks = Vec::new();

                if let Some(text) = &msg.content
                    && !text.is_empty()
                {
                    blocks.push(AnthropicContentBlock::Text { text: text.clone() });
                }

                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        let input: Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(Value::Object(serde_json::Map::new()));
                        blocks.push(AnthropicContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            input,
                        });
                    }
                }

                if !blocks.is_empty() {
                    messages.push(AnthropicMessage {
                        role: "assistant".into(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                }
            }
            Role::Tool => {
                // Anthropic: tool results go into a user message with tool_result blocks.
                // Consecutive tool results are merged into one user message.
                let block = AnthropicContentBlock::ToolResult {
                    tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                    content: msg.content.clone().unwrap_or_default(),
                };

                // If the last message is already a user message with blocks, append
                if let Some(last) = messages.last_mut()
                    && last.role == "user"
                    && let AnthropicContent::Blocks(ref mut blocks) = last.content
                {
                    blocks.push(block);
                    continue;
                }

                messages.push(AnthropicMessage {
                    role: "user".into(),
                    content: AnthropicContent::Blocks(vec![block]),
                });
            }
        }
    }

    let tools = request.tools.as_ref().map(|defs| {
        defs.iter()
            .map(|td| AnthropicTool {
                name: td.function.name.clone(),
                description: td.function.description.clone(),
                input_schema: td.function.parameters.clone(),
            })
            .collect()
    });

    AnthropicRequest {
        model: request.model.clone(),
        max_tokens: request.max_tokens.unwrap_or(4096),
        system,
        messages,
        tools,
        stream: request.stream,
    }
}

// ─── SSE stream parsing ─────────────────────────────────────────────────────

/// State for assembling a tool_use block from streaming deltas.
#[derive(Default)]
struct ToolUseBuilder {
    id: String,
    name: String,
    json_buf: String,
}

async fn parse_anthropic_sse(
    response: reqwest::Response,
    tx: mpsc::UnboundedSender<StreamEvent>,
) -> anyhow::Result<ChatResponse> {
    let mut content = String::new();
    let mut tool_builders: Vec<ToolUseBuilder> = Vec::new();
    let mut input_tokens = 0u32;
    let mut output_tokens = 0u32;
    let mut finish_reason = None;

    // Track which content block index maps to which tool builder
    let mut block_index_to_tool: std::collections::HashMap<usize, usize> = Default::default();

    for_each_sse_line(response, |raw_line| {
        let line = raw_line.trim();

        if line.is_empty() || line.starts_with(':') {
            return Ok(());
        }

        // Anthropic SSE has "event: <type>" lines followed by "data: <json>" lines.
        // We only care about the data lines.
        let data = match line.strip_prefix("data:") {
            Some(d) => d.trim(),
            None => return Ok(()),
        };

        let event: SseEvent = match serde_json::from_str(data) {
            Ok(e) => e,
            Err(e) => {
                debug!("Failed to parse Anthropic SSE event: {e} — data: {data}");
                return Ok(());
            }
        };

        match event.event_type.as_str() {
            "message_start" => {
                if let Some(msg) = &event.message
                    && let Some(u) = &msg.usage
                {
                    input_tokens = u.input_tokens;
                }
            }
            "content_block_start" => {
                if let (Some(idx), Some(block)) = (event.index, &event.content_block)
                    && block.block_type == "tool_use"
                {
                    let builder = ToolUseBuilder {
                        id: block.id.clone().unwrap_or_default(),
                        name: block.name.clone().unwrap_or_default(),
                        json_buf: String::new(),
                    };
                    let tool_idx = tool_builders.len();
                    tool_builders.push(builder);
                    block_index_to_tool.insert(idx, tool_idx);

                    let _ = tx.send(StreamEvent::ToolCallDelta {
                        index: tool_idx as u32,
                        id: block.id.clone(),
                        name: block.name.clone(),
                        arguments_delta: None,
                    });
                }
            }
            "content_block_delta" => {
                if let Some(delta) = &event.delta {
                    match delta.delta_type.as_str() {
                        "text_delta" => {
                            if let Some(text) = &delta.text {
                                content.push_str(text);
                                let _ = tx.send(StreamEvent::ContentDelta(text.clone()));
                            }
                        }
                        "input_json_delta" => {
                            if let Some(idx) = event.index
                                && let Some(&tool_idx) = block_index_to_tool.get(&idx)
                                && let Some(json) = &delta.partial_json
                            {
                                tool_builders[tool_idx].json_buf.push_str(json);
                                let _ = tx.send(StreamEvent::ToolCallDelta {
                                    index: tool_idx as u32,
                                    id: None,
                                    name: None,
                                    arguments_delta: Some(json.clone()),
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
            "message_delta" => {
                if let Some(delta) = &event.delta
                    && let Some(reason) = &delta.stop_reason
                {
                    finish_reason = Some(reason.clone());
                }
                if let Some(u) = &event.usage {
                    output_tokens = u.output_tokens;
                }
            }
            _ => {} // content_block_stop, message_stop, ping
        }
        Ok(())
    })
    .await?;

    let tool_calls: Vec<ToolCall> = tool_builders
        .into_iter()
        .map(|b| ToolCall {
            id: b.id,
            call_type: "function".into(),
            function: FunctionCall {
                name: b.name,
                arguments: b.json_buf,
            },
        })
        .collect();

    let usage = Some(Usage {
        prompt_tokens: input_tokens,
        completion_tokens: output_tokens,
        total_tokens: input_tokens + output_tokens,
    });

    let response = ChatResponse {
        content: if content.is_empty() {
            None
        } else {
            Some(content)
        },
        tool_calls,
        usage,
        finish_reason,
    };

    let _ = tx.send(StreamEvent::Done(response.clone()));
    Ok(response)
}

// ─── Non-streaming response parsing ─────────────────────────────────────────

fn parse_non_streaming(body: &Value) -> anyhow::Result<ChatResponse> {
    let content_blocks = body["content"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("No content array in Anthropic response"))?;

    let mut text = String::new();
    let mut tool_calls = Vec::new();

    for block in content_blocks {
        match block["type"].as_str() {
            Some("text") => {
                if let Some(t) = block["text"].as_str() {
                    text.push_str(t);
                }
            }
            Some("tool_use") => {
                let id = block["id"].as_str().unwrap_or("").to_string();
                let name = block["name"].as_str().unwrap_or("").to_string();
                let input = &block["input"];
                tool_calls.push(ToolCall {
                    id,
                    call_type: "function".into(),
                    function: FunctionCall {
                        name,
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    },
                });
            }
            _ => {}
        }
    }

    let usage = body.get("usage").map(|u| Usage {
        prompt_tokens: u["input_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: u["output_tokens"].as_u64().unwrap_or(0) as u32,
        total_tokens: (u["input_tokens"].as_u64().unwrap_or(0)
            + u["output_tokens"].as_u64().unwrap_or(0)) as u32,
    });

    let finish_reason = body["stop_reason"].as_str().map(String::from);

    Ok(ChatResponse {
        content: if text.is_empty() { None } else { Some(text) },
        tool_calls,
        usage,
        finish_reason,
    })
}

// ─── Provider trait impl ─────────────────────────────────────────────────────

#[async_trait]
impl Provider for AnthropicProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> anyhow::Result<()> {
        debug!("Anthropic streaming request to {}", self.endpoint());

        let mut anthropic_req = convert_request(&request);
        anthropic_req.stream = true;

        let response = self
            .client
            .post(self.endpoint())
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("content-type", "application/json")
            .json(&anthropic_req)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let _ = tx.send(StreamEvent::Error(format!(
                "Anthropic API error {status}: {body}"
            )));
            anyhow::bail!("Anthropic API error {status}: {body}");
        }

        parse_anthropic_sse(response, tx).await?;
        Ok(())
    }

    async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
        let mut anthropic_req = convert_request(&request);
        anthropic_req.stream = false;

        let response = self
            .client
            .post(self.endpoint())
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("content-type", "application/json")
            .json(&anthropic_req)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error {status}: {body}");
        }

        let body: Value = response.json().await?;
        parse_non_streaming(&body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_convert_request_extracts_system() {
        let request = ChatRequest {
            model: "claude-sonnet-4-6".into(),
            messages: vec![
                Message {
                    role: Role::System,
                    content: Some("You are helpful.".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::User,
                    content: Some("Hi".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            tools: None,
            temperature: None,
            max_tokens: Some(1024),
            stream: false,
        };

        let converted = convert_request(&request);
        assert_eq!(converted.system, Some("You are helpful.".into()));
        assert_eq!(converted.messages.len(), 1);
        assert_eq!(converted.messages[0].role, "user");
        assert_eq!(converted.max_tokens, 1024);
    }

    #[test]
    fn test_convert_request_tool_calls_to_content_blocks() {
        let request = ChatRequest {
            model: "claude-sonnet-4-6".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: Some("list files".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::Assistant,
                    content: Some("I'll check.".into()),
                    tool_calls: Some(vec![ToolCall {
                        id: "toolu_01".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "bash".into(),
                            arguments: r#"{"command":"ls"}"#.into(),
                        },
                    }]),
                    tool_call_id: None,
                },
                Message {
                    role: Role::Tool,
                    content: Some("file1.txt\nfile2.txt".into()),
                    tool_calls: None,
                    tool_call_id: Some("toolu_01".into()),
                },
            ],
            tools: None,
            temperature: None,
            max_tokens: None,
            stream: false,
        };

        let converted = convert_request(&request);
        assert_eq!(converted.messages.len(), 3);

        // Assistant message should have content blocks
        let assistant_json = serde_json::to_value(&converted.messages[1].content).unwrap();
        let blocks = assistant_json.as_array().unwrap();
        assert_eq!(blocks.len(), 2); // text + tool_use
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["name"], "bash");

        // Tool result should be in a user message
        assert_eq!(converted.messages[2].role, "user");
    }

    #[test]
    fn test_convert_request_merges_consecutive_tool_results() {
        let request = ChatRequest {
            model: "claude-sonnet-4-6".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: Some("hi".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: Role::Assistant,
                    content: None,
                    tool_calls: Some(vec![
                        ToolCall {
                            id: "t1".into(),
                            call_type: "function".into(),
                            function: FunctionCall {
                                name: "bash".into(),
                                arguments: "{}".into(),
                            },
                        },
                        ToolCall {
                            id: "t2".into(),
                            call_type: "function".into(),
                            function: FunctionCall {
                                name: "read_file".into(),
                                arguments: "{}".into(),
                            },
                        },
                    ]),
                    tool_call_id: None,
                },
                Message {
                    role: Role::Tool,
                    content: Some("result1".into()),
                    tool_calls: None,
                    tool_call_id: Some("t1".into()),
                },
                Message {
                    role: Role::Tool,
                    content: Some("result2".into()),
                    tool_calls: None,
                    tool_call_id: Some("t2".into()),
                },
            ],
            tools: None,
            temperature: None,
            max_tokens: None,
            stream: false,
        };

        let converted = convert_request(&request);
        // user + assistant + merged_tool_results = 3 messages (not 4)
        assert_eq!(converted.messages.len(), 3);

        // The tool results should be merged into one user message with 2 blocks
        let tool_msg = &converted.messages[2];
        assert_eq!(tool_msg.role, "user");
        if let AnthropicContent::Blocks(blocks) = &tool_msg.content {
            assert_eq!(blocks.len(), 2);
        } else {
            panic!("Expected Blocks content");
        }
    }

    #[test]
    fn test_convert_tools_to_anthropic_format() {
        let request = ChatRequest {
            model: "claude-sonnet-4-6".into(),
            messages: vec![Message {
                role: Role::User,
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Some(vec![ToolDefinition {
                tool_type: "function".into(),
                function: FunctionSpec {
                    name: "bash".into(),
                    description: "Run a command".into(),
                    parameters: json!({
                        "type": "object",
                        "required": ["command"],
                        "properties": {
                            "command": {"type": "string"}
                        }
                    }),
                },
            }]),
            temperature: None,
            max_tokens: None,
            stream: false,
        };

        let converted = convert_request(&request);
        let tools = converted.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "bash");
        assert_eq!(tools[0].input_schema["type"], "object");
    }

    #[test]
    fn test_parse_non_streaming_text() {
        let body = json!({
            "content": [
                {"type": "text", "text": "Hello there!"}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 50,
                "output_tokens": 10
            }
        });

        let resp = parse_non_streaming(&body).unwrap();
        assert_eq!(resp.content, Some("Hello there!".into()));
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.usage.unwrap().prompt_tokens, 50);
    }

    #[test]
    fn test_parse_non_streaming_tool_use() {
        let body = json!({
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "bash",
                    "input": {"command": "ls"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 25
            }
        });

        let resp = parse_non_streaming(&body).unwrap();
        assert_eq!(resp.content, Some("Let me check.".into()));
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "toolu_abc");
        assert_eq!(resp.tool_calls[0].function.name, "bash");
        assert!(
            resp.tool_calls[0]
                .function
                .arguments
                .contains("\"command\"")
        );
    }

    #[test]
    fn test_default_max_tokens() {
        let request = ChatRequest {
            model: "claude-sonnet-4-6".into(),
            messages: vec![],
            tools: None,
            temperature: None,
            max_tokens: None,
            stream: false,
        };

        let converted = convert_request(&request);
        assert_eq!(converted.max_tokens, 4096);
    }
}
