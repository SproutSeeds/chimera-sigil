use chimera_sigil_providers::types::{FunctionCall, ToolCall};
use serde_json::Value;

pub(crate) fn extract_textual_tool_calls(content: &str) -> Option<(Option<String>, Vec<ToolCall>)> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }

    if contains_structured_report_json(trimmed) {
        return None;
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        match value {
            Value::Array(items) => {
                let calls = parse_textual_tool_call_values(items)?;
                return Some((None, calls));
            }
            other => {
                if let Some(call) = parse_textual_tool_call_value(other) {
                    return Some((None, vec![call.with_index(0)]));
                }
            }
        }
    }

    let mut calls = Vec::new();
    let mut text_lines = Vec::new();
    for line in content.lines() {
        let trimmed_line = line.trim();
        if trimmed_line.is_empty() {
            if !text_lines.is_empty() {
                text_lines.push(String::new());
            }
            continue;
        }

        match serde_json::from_str::<Value>(trimmed_line)
            .ok()
            .and_then(parse_textual_tool_call_value)
            .or_else(|| parse_textual_tool_call_line(trimmed_line))
        {
            Some(call) => calls.push(call.with_index(calls.len())),
            None => text_lines.push(line.to_string()),
        }
    }

    if calls.is_empty() {
        return extract_textual_tool_calls_from_fragments(content);
    }

    let assistant_content = text_lines.join("\n").trim().to_string();
    Some((
        if assistant_content.is_empty() {
            None
        } else {
            Some(assistant_content)
        },
        calls,
    ))
}

fn extract_textual_tool_calls_from_fragments(
    content: &str,
) -> Option<(Option<String>, Vec<ToolCall>)> {
    let fragments = outer_json_fragments(content);
    if fragments.is_empty() {
        return None;
    }

    let mut calls = Vec::new();
    let mut assistant_content = String::new();
    let mut cursor = 0;

    for (start, end) in fragments {
        if start > cursor {
            append_assistant_fragment(&mut assistant_content, &content[cursor..start]);
        }

        let fragment = &content[start..end];
        match serde_json::from_str::<Value>(fragment)
            .ok()
            .and_then(parse_textual_tool_call_value)
        {
            Some(call) => calls.push(call.with_index(calls.len())),
            None => append_assistant_fragment(&mut assistant_content, fragment),
        }

        cursor = end;
    }

    if cursor < content.len() {
        append_assistant_fragment(&mut assistant_content, &content[cursor..]);
    }

    if calls.is_empty() {
        return None;
    }

    Some((
        if assistant_content.is_empty() {
            None
        } else {
            Some(assistant_content)
        },
        calls,
    ))
}

fn append_assistant_fragment(buffer: &mut String, fragment: &str) {
    let trimmed = fragment.trim();
    if trimmed.is_empty() {
        return;
    }

    if !buffer.is_empty() {
        buffer.push('\n');
    }
    buffer.push_str(trimmed);
}

fn contains_structured_report_json(content: &str) -> bool {
    if parses_as_structured_report_json(content) {
        return true;
    }

    if content
        .lines()
        .any(|line| parses_as_structured_report_json(line.trim()))
    {
        return true;
    }

    contains_structured_report_json_fragment(content)
}

fn parses_as_structured_report_json(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }

    serde_json::from_str::<Value>(content)
        .ok()
        .is_some_and(|value| value_has_report_shape(&value))
}

fn value_has_report_shape(value: &Value) -> bool {
    match value {
        Value::Object(object) => object.contains_key("task") && object.contains_key("target"),
        Value::Array(items) => items.iter().any(value_has_report_shape),
        _ => false,
    }
}

fn contains_structured_report_json_fragment(content: &str) -> bool {
    outer_json_fragments(content)
        .into_iter()
        .any(|(start, end)| parses_as_structured_report_json(&content[start..end]))
}

fn outer_json_fragments(content: &str) -> Vec<(usize, usize)> {
    let mut fragments = Vec::new();
    let mut start = None;
    let mut expected_closers = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for (index, ch) in content.char_indices() {
        if start.is_none() {
            match ch {
                '{' => {
                    start = Some(index);
                    expected_closers.push('}');
                }
                '[' => {
                    start = Some(index);
                    expected_closers.push(']');
                }
                _ => {}
            }
            continue;
        }

        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => expected_closers.push('}'),
            '[' => expected_closers.push(']'),
            '}' | ']' => {
                if expected_closers.last().copied() != Some(ch) {
                    start = None;
                    expected_closers.clear();
                    continue;
                }

                expected_closers.pop();
                if expected_closers.is_empty() {
                    fragments.push((start.unwrap(), index + ch.len_utf8()));
                    start = None;
                }
            }
            _ => {}
        }
    }

    fragments
}

fn parse_textual_tool_call_values(values: Vec<Value>) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    for (index, value) in values.into_iter().enumerate() {
        calls.push(parse_textual_tool_call_value(value)?.with_index(index));
    }

    if calls.is_empty() { None } else { Some(calls) }
}

fn parse_textual_tool_call_value(value: Value) -> Option<TextualToolCall> {
    let Value::Object(mut object) = value else {
        return None;
    };

    let Some(Value::String(name)) = object.remove("name") else {
        return None;
    };
    let Some(arguments) = object.remove("arguments") else {
        return None;
    };

    let arguments = match arguments {
        Value::String(value) => value,
        other => serde_json::to_string(&other).ok()?,
    };

    Some(TextualToolCall { name, arguments })
}

fn parse_textual_tool_call_line(line: &str) -> Option<TextualToolCall> {
    let line = line.trim().trim_start_matches("-").trim();
    let rest = line.strip_prefix("name:")?.trim();
    let (name, arguments) = rest.split_once(", arguments:")?;
    let name = name.trim().trim_matches(['`', '"', '\'']).to_string();
    if name.is_empty() {
        return None;
    }

    let arguments = arguments.trim();
    let arguments = serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| serde_json::to_string(&value).ok())
        .unwrap_or_else(|| arguments.to_string());

    Some(TextualToolCall { name, arguments })
}

struct TextualToolCall {
    name: String,
    arguments: String,
}

impl TextualToolCall {
    fn with_index(self, index: usize) -> ToolCall {
        ToolCall {
            id: format!("text_tool_call_{index}"),
            call_type: "function".into(),
            function: FunctionCall {
                name: self.name,
                arguments: self.arguments,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_reports_block_later_textual_tool_calls() {
        let content = "{\n  \"task\": \"repo-scout-chimera\",\n  \"target\": \"../chimera\",\n  \"summary\": \"done\"\n}\n{\"name\":\"glob_search\",\"arguments\":{\"pattern\":\"*.rs\"}}";

        assert!(contains_structured_report_json(content));
        assert!(extract_textual_tool_calls(content).is_none());
    }

    #[test]
    fn tool_call_arguments_do_not_look_like_structured_reports() {
        let (_, calls) = extract_textual_tool_calls(
            r#"{"name":"structured_output","arguments":{"task":"repo-scout","target":"../chimera"}}"#,
        )
        .expect("textual tool call");

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "structured_output");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({"task": "repo-scout", "target": "../chimera"})
        );
    }

    #[test]
    fn parses_pretty_printed_textual_tool_call_objects() {
        let (_, calls) = extract_textual_tool_calls(
            "{\n  \"name\": \"grep_search\",\n  \"arguments\": {\n    \"include\": \"**/*.rs\",\n    \"pattern\": \"provider\"\n  }\n}",
        )
        .expect("textual tool call");

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "grep_search");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({"include": "**/*.rs", "pattern": "provider"})
        );
    }

    #[test]
    fn parses_multiple_pretty_printed_textual_tool_call_objects() {
        let (_, calls) = extract_textual_tool_calls(
            "{\n  \"name\": \"glob_search\",\n  \"arguments\": {\"pattern\": \"**/*provider*\"}\n}\n\n{\n  \"name\": \"glob_search\",\n  \"arguments\": {\"pattern\": \"**/*session*\"}\n}",
        )
        .expect("textual tool calls");

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "text_tool_call_0");
        assert_eq!(calls[0].function.name, "glob_search");
        assert_eq!(calls[1].id, "text_tool_call_1");
        assert_eq!(calls[1].function.name, "glob_search");
    }

    #[test]
    fn parses_name_arguments_textual_tool_call_lines() {
        let (_, calls) = extract_textual_tool_calls(
            "name: read_file, arguments: {\"path\": \"../chimera/README.md\"}\n\
             name: glob_search, arguments: {\"pattern\": \"**/*.rs\", \"path\": \"../chimera\"}",
        )
        .expect("textual tool calls");

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({"path": "../chimera/README.md"})
        );
        assert_eq!(calls[1].function.name, "glob_search");
    }
}
