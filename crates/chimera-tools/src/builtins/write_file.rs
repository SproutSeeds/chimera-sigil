use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct WriteFileInput {
    pub file_path: String,
    pub content: String,
}

pub fn run(input: WriteFileInput) -> anyhow::Result<String> {
    let path = Path::new(&input.file_path);

    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    fs::write(path, &input.content)?;

    let lines = input.content.lines().count();
    let bytes = input.content.len();
    Ok(format!(
        "Wrote {bytes} bytes ({lines} lines) to {}",
        input.file_path
    ))
}
