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
    if let Some(parent) = path.parent()
        && !parent.exists()
    {
        fs::create_dir_all(parent)?;
    }

    fs::write(path, &input.content)?;

    let lines = input.content.lines().count();
    let bytes = input.content.len();
    Ok(format!(
        "Wrote {bytes} bytes ({lines} lines) to {}",
        input.file_path
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_file_creates_new() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.txt");

        let input = WriteFileInput {
            file_path: path.display().to_string(),
            content: "hello world\n".into(),
        };

        let result = run(input).unwrap();
        assert!(result.contains("12 bytes"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello world\n");
    }

    #[test]
    fn test_write_file_creates_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a/b/c/deep.txt");

        let input = WriteFileInput {
            file_path: path.display().to_string(),
            content: "deep\n".into(),
        };

        let result = run(input).unwrap();
        assert!(result.contains("5 bytes"));
        assert!(path.exists());
    }

    #[test]
    fn test_write_file_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overwrite.txt");
        fs::write(&path, "old content").unwrap();

        let input = WriteFileInput {
            file_path: path.display().to_string(),
            content: "new content".into(),
        };

        run(input).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "new content");
    }
}
