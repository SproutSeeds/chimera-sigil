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
