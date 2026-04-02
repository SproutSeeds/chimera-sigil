use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct EditFileInput {
    pub file_path: String,
    pub old_string: String,
    pub new_string: String,
}

pub fn run(input: EditFileInput) -> anyhow::Result<String> {
    let content = fs::read_to_string(&input.file_path)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {e}", input.file_path))?;

    let count = content.matches(&input.old_string).count();

    if count == 0 {
        anyhow::bail!(
            "old_string not found in {}. Make sure it matches exactly.",
            input.file_path
        );
    }

    if count > 1 {
        anyhow::bail!(
            "old_string found {count} times in {}. It must be unique — provide more context.",
            input.file_path
        );
    }

    let new_content = content.replacen(&input.old_string, &input.new_string, 1);
    fs::write(&input.file_path, &new_content)?;

    Ok(format!("Edited {} (replaced 1 occurrence)", input.file_path))
}
