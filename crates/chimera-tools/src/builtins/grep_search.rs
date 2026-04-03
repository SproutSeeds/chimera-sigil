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
            if m.is_context {
                output.push_str(&format!("  {}- {}\n", m.line_num, m.line));
            } else {
                output.push_str(&format!("  {}: {}\n", m.line_num, m.line));
            }
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
    is_context: bool,
}

fn search_file(
    path: &Path,
    re: &regex::Regex,
    context: usize,
    results: &mut Vec<FileMatch>,
) -> anyhow::Result<()> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()), // Skip binary/unreadable files
    };

    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    // Find all matching line indices
    let match_indices: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| re.is_match(line))
        .map(|(i, _)| i)
        .collect();

    if match_indices.is_empty() {
        return Ok(());
    }

    // Build output with context lines, deduplicating overlaps
    let mut included: Vec<bool> = vec![false; total_lines];
    let mut is_match_line: Vec<bool> = vec![false; total_lines];

    for &idx in &match_indices {
        is_match_line[idx] = true;
        let start = idx.saturating_sub(context);
        let end = (idx + context + 1).min(total_lines);
        for slot in &mut included[start..end] {
            *slot = true;
        }
    }

    let matches: Vec<LineMatch> = (0..total_lines)
        .filter(|&i| included[i])
        .map(|i| LineMatch {
            line_num: i + 1,
            line: lines[i].to_string(),
            is_context: !is_match_line[i],
        })
        .collect();

    results.push(FileMatch {
        path: path.display().to_string(),
        matches,
    });

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
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && (name.starts_with('.')
                || name == "node_modules"
                || name == "target"
                || name == "__pycache__")
        {
            continue;
        }

        if path.is_dir() {
            walk_dir(&path, re, context, include, results, files_searched)?;
        } else if path.is_file() {
            // Apply include filter
            if let Some(inc) = include {
                let Ok(pattern) = glob::Pattern::new(inc) else {
                    continue;
                };
                if let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && !pattern.matches(name)
                {
                    continue;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_grep_basic_match() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let mut f = fs::File::create(&file_path).unwrap();
        writeln!(f, "hello world").unwrap();
        writeln!(f, "foo bar").unwrap();
        writeln!(f, "hello again").unwrap();

        let input = GrepSearchInput {
            pattern: "hello".to_string(),
            path: Some(file_path.display().to_string()),
            include: None,
            context_lines: None,
        };

        let result = run(input).unwrap();
        assert!(result.contains("2 match(es)"));
        assert!(result.contains("hello world"));
        assert!(result.contains("hello again"));
    }

    #[test]
    fn test_grep_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "nothing here\n").unwrap();

        let input = GrepSearchInput {
            pattern: "xyz123".to_string(),
            path: Some(file_path.display().to_string()),
            include: None,
            context_lines: None,
        };

        let result = run(input).unwrap();
        assert!(result.contains("No matches"));
    }

    #[test]
    fn test_grep_context_lines() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let mut f = fs::File::create(&file_path).unwrap();
        writeln!(f, "line 1").unwrap();
        writeln!(f, "line 2").unwrap();
        writeln!(f, "MATCH here").unwrap();
        writeln!(f, "line 4").unwrap();
        writeln!(f, "line 5").unwrap();

        let input = GrepSearchInput {
            pattern: "MATCH".to_string(),
            path: Some(file_path.display().to_string()),
            include: None,
            context_lines: Some(1),
        };

        let result = run(input).unwrap();
        assert!(result.contains("line 2"));   // context before
        assert!(result.contains("MATCH here")); // match
        assert!(result.contains("line 4"));   // context after
        // line 1 and line 5 should NOT be present (only 1 line of context)
        assert!(!result.contains("line 1"));
        assert!(!result.contains("line 5"));
    }

    #[test]
    fn test_grep_invalid_regex() {
        let input = GrepSearchInput {
            pattern: "[invalid".to_string(),
            path: Some(".".to_string()),
            include: None,
            context_lines: None,
        };

        let result = run(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid regex"));
    }

    #[test]
    fn test_grep_directory_search() {
        let dir = tempfile::tempdir().unwrap();

        let sub = dir.path().join("subdir");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("a.rs"), "fn main() {}\n").unwrap();
        fs::write(sub.join("b.txt"), "fn helper() {}\n").unwrap();

        // Search with include filter
        let input = GrepSearchInput {
            pattern: "fn ".to_string(),
            path: Some(dir.path().display().to_string()),
            include: Some("*.rs".to_string()),
            context_lines: None,
        };

        let result = run(input).unwrap();
        assert!(result.contains("fn main"));
        assert!(!result.contains("fn helper")); // .txt file excluded
    }
}
