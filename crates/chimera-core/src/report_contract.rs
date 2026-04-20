use serde_json::Value;

pub(crate) fn contains_report_json(content: Option<&str>) -> bool {
    let Some(content) = content.map(str::trim).filter(|value| !value.is_empty()) else {
        return false;
    };

    if parses_as_report_json(content) {
        return true;
    }

    if content
        .lines()
        .any(|line| parses_as_report_json(line.trim()))
    {
        return true;
    }

    outer_json_fragments(content)
        .into_iter()
        .any(|(start, end)| parses_as_report_json(&content[start..end]))
}

pub(crate) fn repair_prompt(previous_response: Option<&str>) -> String {
    let excerpt = previous_response
        .unwrap_or("")
        .trim()
        .chars()
        .take(4_000)
        .collect::<String>();

    format!(
        "# Final Report Contract Repair\n\n\
         Your previous response did not satisfy the required final report contract.\n\n\
         Return a fenced JSON block with a single JSON object. The object must include \
         top-level `task` and `target` fields, plus the other fields requested by the \
         original user prompt.\n\n\
         Rules:\n\
         - Do not return only tool calls, a prose summary, or a `tool_response` wrapper.\n\
         - Do not invent files, commands, or artifacts.\n\
         - Use concrete file paths, commands, or observed tool output as evidence.\n\
         - If evidence is not verified, mark that limitation clearly in the JSON.\n\
         - Return the JSON report now.\n\n\
         Previous response excerpt:\n\n```text\n{excerpt}\n```"
    )
}

fn parses_as_report_json(content: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_fenced_report_json() {
        assert!(contains_report_json(Some(
            "done\n```json\n{\"task\":\"repo-scout\",\"target\":\"../chimera\"}\n```"
        )));
    }

    #[test]
    fn rejects_tool_response_wrappers() {
        assert!(!contains_report_json(Some(
            r#"{"tool_response":"summary without task and target"}"#
        )));
    }

    #[test]
    fn repair_prompt_includes_previous_excerpt() {
        let prompt = repair_prompt(Some("not json"));
        assert!(prompt.contains("Final Report Contract Repair"));
        assert!(prompt.contains("not json"));
    }
}
