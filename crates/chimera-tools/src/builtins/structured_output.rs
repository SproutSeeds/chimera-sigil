use serde_json::Value;

/// StructuredOutput tool — the model calls this to return arbitrary typed JSON.
/// Inspired by the claw-code pattern: instead of returning structured data as
/// free-form text, the model packages it as a tool call. The harness captures
/// the structured payload directly.
pub fn run(input: Value) -> anyhow::Result<String> {
    if let Some(obj) = input.as_object()
        && obj.is_empty()
    {
        anyhow::bail!("Structured output payload must not be empty");
    }

    serde_json::to_string_pretty(&serde_json::json!({
        "data": "Structured output provided successfully",
        "structured_output": input,
    }))
    .map_err(|e| anyhow::anyhow!("Failed to serialize structured output: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_structured_output_echoes_payload() {
        let input = json!({"status": "ok", "items": [1, 2, 3]});
        let result = run(input).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["data"], "Structured output provided successfully");
        assert_eq!(parsed["structured_output"]["status"], "ok");
        assert_eq!(parsed["structured_output"]["items"][1], 2);
    }

    #[test]
    fn test_structured_output_rejects_empty() {
        let input = json!({});
        assert!(run(input).is_err());
    }

    #[test]
    fn test_structured_output_accepts_non_object() {
        // Arrays and primitives are valid payloads
        let result = run(json!([1, 2, 3])).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["structured_output"], json!([1, 2, 3]));
    }
}
