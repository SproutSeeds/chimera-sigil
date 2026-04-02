use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct GrepSearchInput {
    pub pattern: String,
    pub path: Option<String>,
    pub include: Option<String>,
    pub context_lines: Option<usize>,
}

pub fn run(input: GrepSearchInput) -> anyhow::Result<String> {
    let re = regex::Regex::new(&input.pattern)
        .map_err(|e| anyhow::anyhow!("Invalid regex pattern: {e}"))?;

    let search_path = input.path.as_deref().unwrap_or(".");
    let path = Path::new(search_path);
    let context = input.context_lines.unwrap_or(0);

    let mut results = Vec::new();
    let mut files_searched = 0u32;

    if path.is_file() {
        search_file(path, &re, context, &mut results)?;
        files_searched = 1;
    } else if path.is_dir() {
        walk_dir(path, &re, context, &input.include, &mut results, &mut files_searched)?;
    } else {
        anyhow::bail!("Path does not exist: {search_path}");
    }

    if results.is_empty() {
        return Ok(format!(
            "No matches for '{}' in {search_path} ({files_searched} files searched)",
            input.pattern
        ));
    }

    let total_matches: usize = results.iter().map(|r: &FileMatch| r.matches.len()).sum();
    let mut output = format!("{total_matches} match(es) across {} file(s):\n\n", results.len());

    for file_match in &results {
        output.push_str(&format!("{}:\n", file_match.path));
        for m in &file_match.matches {
            output.push_str(&format!("  {}: {}\n", m.line_num, m.line));
        }
        output.push('\n');
    }

    // Truncate if too long
    if output.len() > 100_000 {
        output.truncate(100_000);
        output.push_str("\n... (output truncated)");
    }

    Ok(output)
}

struct FileMatch {
    path: String,
    matches: Vec<LineMatch>,
}

struct LineMatch {
    line_num: usize,
    line: String,
}

fn search_file(
    path: &Path,
    re: &regex::Regex,
    _context: usize,
    results: &mut Vec<FileMatch>,
) -> anyhow::Result<()> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()), // Skip binary/unreadable files
    };

    let mut matches = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if re.is_match(line) {
            matches.push(LineMatch {
                line_num: i + 1,
                line: line.to_string(),
            });
        }
    }

    if !matches.is_empty() {
        results.push(FileMatch {
            path: path.display().to_string(),
            matches,
        });
    }

    Ok(())
}

fn walk_dir(
    dir: &Path,
    re: &regex::Regex,
    context: usize,
    include: &Option<String>,
    results: &mut Vec<FileMatch>,
    files_searched: &mut u32,
) -> anyhow::Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();

        // Skip hidden directories and common noise
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.')
                || name == "node_modules"
                || name == "target"
                || name == "__pycache__"
                || name == ".git"
            {
                continue;
            }
        }

        if path.is_dir() {
            walk_dir(&path, re, context, include, results, files_searched)?;
        } else if path.is_file() {
            // Apply include filter
            if let Some(inc) = include {
                let pattern = glob::Pattern::new(inc).unwrap_or_else(|_| {
                    glob::Pattern::new("*").unwrap()
                });
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if !pattern.matches(name) {
                        continue;
                    }
                }
            }

            *files_searched += 1;
            search_file(&path, re, context, results)?;

            // Cap total results
            if results.len() >= 100 {
                return Ok(());
            }
        }
    }

    Ok(())
}
