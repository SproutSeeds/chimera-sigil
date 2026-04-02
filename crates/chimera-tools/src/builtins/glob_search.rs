use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct GlobSearchInput {
    pub pattern: String,
    pub path: Option<String>,
}

pub fn run(input: GlobSearchInput) -> anyhow::Result<String> {
    let base = input.path.as_deref().unwrap_or(".");
    let full_pattern = if input.pattern.starts_with('/') {
        input.pattern.clone()
    } else {
        format!("{base}/{}", input.pattern)
    };

    let entries = glob::glob(&full_pattern)
        .map_err(|e| anyhow::anyhow!("Invalid glob pattern: {e}"))?;

    let mut paths: Vec<String> = Vec::new();
    for entry in entries {
        match entry {
            Ok(path) => paths.push(path.display().to_string()),
            Err(e) => tracing::debug!("Glob entry error: {e}"),
        }
    }

    if paths.is_empty() {
        return Ok(format!("No files matching pattern: {}", input.pattern));
    }

    // Sort by modification time (newest first)
    paths.sort_by(|a, b| {
        let time_a = std::fs::metadata(a)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let time_b = std::fs::metadata(b)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        time_b.cmp(&time_a)
    });

    // Cap results
    let total = paths.len();
    if paths.len() > 200 {
        paths.truncate(200);
    }

    let mut result = paths.join("\n");
    if total > 200 {
        result.push_str(&format!("\n... ({total} total, showing first 200)"));
    }

    Ok(result)
}
