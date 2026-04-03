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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_dir_basic() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "").unwrap();
        fs::write(dir.path().join("b.txt"), "").unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();

        let input = ListDirInput {
            path: dir.path().display().to_string(),
        };

        let result = run(input).unwrap();
        assert!(result.contains("a.txt"));
        assert!(result.contains("b.txt"));
        assert!(result.contains("subdir/"));
    }

    #[test]
    fn test_list_dir_empty() {
        let dir = tempfile::tempdir().unwrap();

        let input = ListDirInput {
            path: dir.path().display().to_string(),
        };

        let result = run(input).unwrap();
        assert!(result.contains("is empty"));
    }

    #[test]
    fn test_list_dir_nonexistent() {
        let input = ListDirInput {
            path: "/tmp/chimera_nonexistent_dir_12345".into(),
        };

        let result = run(input);
        assert!(result.is_err());
    }
}
