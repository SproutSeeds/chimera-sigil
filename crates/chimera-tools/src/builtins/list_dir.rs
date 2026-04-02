use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct ListDirInput {
    pub path: String,
}

pub fn run(input: ListDirInput) -> anyhow::Result<String> {
    let path = Path::new(&input.path);

    if !path.exists() {
        anyhow::bail!("Directory does not exist: {}", input.path);
    }

    if !path.is_dir() {
        anyhow::bail!("Not a directory: {}", input.path);
    }

    let mut entries: Vec<String> = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let file_type = entry.file_type()?;

        let indicator = if file_type.is_dir() {
            "/"
        } else if file_type.is_symlink() {
            "@"
        } else {
            ""
        };

        entries.push(format!("{name}{indicator}"));
    }

    entries.sort();

    if entries.is_empty() {
        return Ok(format!("{} is empty", input.path));
    }

    Ok(entries.join("\n"))
}
