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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_file_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edit.txt");
        fs::write(&path, "hello world\n").unwrap();

        let input = EditFileInput {
            file_path: path.display().to_string(),
            old_string: "hello".into(),
            new_string: "goodbye".into(),
        };

        let result = run(input).unwrap();
        assert!(result.contains("replaced 1 occurrence"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "goodbye world\n");
    }

    #[test]
    fn test_edit_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edit.txt");
        fs::write(&path, "hello world\n").unwrap();

        let input = EditFileInput {
            file_path: path.display().to_string(),
            old_string: "xyz_not_here".into(),
            new_string: "replaced".into(),
        };

        let result = run(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_edit_file_duplicate_match() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dup.txt");
        fs::write(&path, "aaa\naaa\n").unwrap();

        let input = EditFileInput {
            file_path: path.display().to_string(),
            old_string: "aaa".into(),
            new_string: "bbb".into(),
        };

        let result = run(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2 times"));
    }
}
