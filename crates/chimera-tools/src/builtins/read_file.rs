use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct ReadFileInput {
    pub file_path: String,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

pub fn run(input: ReadFileInput) -> anyhow::Result<String> {
    let content = fs::read_to_string(&input.file_path)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {e}", input.file_path))?;

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();

    let start = input.offset.unwrap_or(1).saturating_sub(1);
    let limit = input.limit.unwrap_or(2000);
    let end = (start + limit).min(total);

    let mut result = String::new();
    for (i, line) in lines[start..end].iter().enumerate() {
        let line_num = start + i + 1;
        result.push_str(&format!("{line_num:>6}\t{line}\n"));
    }

    if end < total {
        result.push_str(&format!(
            "\n... ({} more lines, {} total)\n",
            total - end,
            total
        ));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_file_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "line one\nline two\nline three\n").unwrap();

        let input = ReadFileInput {
            file_path: path.display().to_string(),
            offset: None,
            limit: None,
        };

        let result = run(input).unwrap();
        assert!(result.contains("line one"));
        assert!(result.contains("line three"));
        assert!(result.contains("1\t")); // line numbers
    }

    #[test]
    fn test_read_file_with_offset_and_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "a\nb\nc\nd\ne\n").unwrap();

        let input = ReadFileInput {
            file_path: path.display().to_string(),
            offset: Some(2),
            limit: Some(2),
        };

        let result = run(input).unwrap();
        assert!(result.contains("b"));
        assert!(result.contains("c"));
        assert!(!result.contains("\ta\n")); // line 1 excluded
    }

    #[test]
    fn test_read_nonexistent_file() {
        let input = ReadFileInput {
            file_path: "/tmp/chimera_nonexistent_file_test_12345.txt".into(),
            offset: None,
            limit: None,
        };

        let result = run(input);
        assert!(result.is_err());
    }
}
