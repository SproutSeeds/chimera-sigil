use serde::Deserialize;
use std::time::Duration;
use tokio::process::Command;

#[derive(Debug, Deserialize)]
pub struct BashInput {
    pub command: String,
    pub timeout_ms: Option<u64>,
}

pub async fn run(input: BashInput) -> anyhow::Result<String> {
    let timeout = Duration::from_millis(input.timeout_ms.unwrap_or(120_000));

    let result = tokio::time::timeout(timeout, async {
        let output = Command::new("bash")
            .arg("-c")
            .arg(&input.command)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("stderr: ");
            result.push_str(&stderr);
        }

        if result.is_empty() {
            if output.status.success() {
                result = "(command completed successfully with no output)".into();
            } else {
                result = format!("(command failed with exit code: {})", output.status);
            }
        }

        // Truncate very long outputs
        if result.len() > 100_000 {
            result.truncate(100_000);
            result.push_str("\n... (output truncated at 100KB)");
        }

        Ok::<String, anyhow::Error>(result)
    })
    .await;

    match result {
        Ok(r) => r,
        Err(_) => Ok(format!(
            "Command timed out after {}ms: {}",
            input.timeout_ms.unwrap_or(120_000),
            input.command
        )),
    }
}
